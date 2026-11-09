"""Live mode: run the four approaches on one item with the local model and stream the events.

The runs use runner.run_one, the same code as the benchmark, so live events are exactly the
trace lines a replay would read (with mode "demo"). While a model call is generating, extra
token events carry the text as it is written; they are never written to a trace file.

The approaches run in a worker thread and put events into a queue; the server reads the queue
through an async generator. Closing that generator (the presenter switched away) sets a stop
flag: the next streamed piece raises LiveCancelled, which closes the Ollama request, so the GPU
stops working on a scene nobody watches.
"""

import queue
import threading
import time
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

from agentic.config import ROOT
from agentic.demo_config import DRAW_SCENE, SCENE_TASKS, DemoConfig, replay_scene_for
from agentic.events import Body, Envelope, to_dict
from agentic.llm.base import LLMResult
from agentic.runner import APPROACHES, ORDER, TASKS, RunCancelled, Setup, run_one
from agentic.tasks.base import Item, Task
from agentic.traces import TraceWriter

LIVE_DIR = ROOT / "traces" / "live"
BATCH_MS = 50                         # token pieces are sent at most this often
BUSY_WAIT_S = 5.0                     # how long a new scene waits for the previous one to stop
READER_WAIT_S = 10.0                  # a scene nobody started reading by then is stopped


class LiveCancelled(RunCancelled):
    """Raised inside a model call when the presenter stopped the live scene."""


class LiveBusy(Exception):
    """A live scene is already running (one GPU, one scene at a time)."""


class LiveFailed(Exception):
    """A live scene failed. details tells the page which replay shows the same puzzle."""

    def __init__(self, message: str, details: dict[str, str]):
        super().__init__(message)
        self.details = details


class StreamingLLM:
    """LLM client for live runs: streams each call and emits batched token events.

    Call ids count from call-0 in every run, like approaches.base.Caller, so a token event
    belongs to the llm_call event with the same call_id. begin_run() starts a new run.
    """

    def __init__(self, client: Any, emit: Callable[[dict[str, Any]], None],
                 stop: threading.Event, batch_ms: int = BATCH_MS):
        self.client = client
        self.model = client.model
        self.emit = emit
        self.stop = stop
        self.batch_ms = batch_ms
        self.approach = ""
        self.count = 0

    def begin_run(self, approach: str) -> None:
        self.approach = approach
        self.count = 0

    def chat(self, messages: list[dict[str, str]], *, seed: int | None = None) -> LLMResult:
        call_id = f"call-{self.count}"
        self.count += 1
        pending: list[str] = []
        last_sent = time.perf_counter()

        def send() -> None:
            if pending:
                self.emit({"type": "token", "approach": self.approach, "call_id": call_id,
                           "text": "".join(pending)})
                pending.clear()

        def on_text(text: str) -> None:
            nonlocal last_sent
            if self.stop.is_set():
                raise LiveCancelled("live scene stopped")
            pending.append(text)
            now = time.perf_counter()
            if (now - last_sent) * 1000 >= self.batch_ms:
                send()
                last_sent = now

        result = self.client.chat_stream(messages, seed=seed, on_text=on_text)
        send()
        return result


def demo_setup(task: Task, demo: DemoConfig, *, model: str, machine_id: str,
               ollama_version: str | None = None, model_digest: str | None = None) -> Setup:
    """Run settings shared by live scenes and their recordings."""
    return Setup(task=task, split="demo", model=model, machine_id=machine_id,
                 base_seed=demo.seed, llm_settings=demo.llm_settings,
                 approach_configs=demo.approach_configs, ollama_version=ollama_version,
                 model_digest=model_digest)


class _QueueSink:
    """Trace writer for run_one: writes each event to the live trace file and the queue."""

    def __init__(self, writer: TraceWriter, events: queue.Queue):
        self.writer = writer
        self.events = events

    def write(self, env: Envelope, body: Body) -> None:
        self.writer.write(env, body)
        self.events.put(to_dict(env, body))


_DONE = object()


class _Failure:
    def __init__(self, error: BaseException):
        self.error = error


class LiveRunner:
    """Runs live scenes, one at a time."""

    def __init__(self, client: Any, demo: DemoConfig, *, model: str, machine_id: str,
                 ollama_version: str | None, model_digest: str | None,
                 live_dir: Path = LIVE_DIR):
        self.client = client
        self.demo = demo
        self.model = model
        self.machine_id = machine_id
        self.ollama_version = ollama_version
        self.model_digest = model_digest
        self.live_dir = Path(live_dir)
        self._start_lock = threading.Lock()
        self._worker: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def resolve(self, scene: str, item_id: str | None) -> Item:
        """The item a live scene runs. Raises KeyError for an unknown scene or an item that
        is not in the draw pool, so a request can only run what the presenter prepared."""
        if scene == DRAW_SCENE:
            if not any(item_id in ids for ids in self.demo.pool.values()):
                raise KeyError(f"{item_id!r} is not in the draw pool")
            task_name = "game24"
        elif scene in SCENE_TASKS:
            item_id, task_name = self.demo.scenes[scene], SCENE_TASKS[scene]
        else:
            raise KeyError(f"unknown scene {scene!r}")
        items = {item.item_id: item for item in TASKS[task_name]().load_items("dev")}
        if item_id not in items:
            raise KeyError(f"{item_id!r} is not in the {task_name} dev split")
        return items[item_id]

    def start(self, scene: str, item_id: str | None) -> AsyncIterator[dict[str, Any]]:
        """Check the request, start the worker, and return the event stream.

        Raises KeyError or LiveBusy at once, before anything is streamed. The worker starts
        here (not when the stream is first read), so busy always means a thread is running.
        """
        item = self.resolve(scene, item_id)
        events: queue.Queue = queue.Queue()
        stop = threading.Event()
        with self._start_lock:
            if self.busy:
                self._worker.join(BUSY_WAIT_S)    # Reset then Start: the old run is stopping
            if self.busy:
                raise LiveBusy("a live scene is already running")
            self._worker = threading.Thread(target=self._work, args=(scene, item, events, stop),
                                            daemon=True)
            self._worker.start()
        # Closing the stream stops the worker, but a stream never read cannot be closed.
        reader_timer = threading.Timer(READER_WAIT_S, stop.set)
        reader_timer.daemon = True
        reader_timer.start()
        details = {"scene": scene, "item_id": item.item_id,
                   "replay_scene": replay_scene_for(scene, item.item_id)}
        return self._stream(events, stop, details, reader_timer)

    def _work(self, scene: str, item: Item, events: queue.Queue, stop: threading.Event) -> None:
        try:
            setup = demo_setup(TASKS[item.task](), self.demo, model=self.model,
                               machine_id=self.machine_id, ollama_version=self.ollama_version,
                               model_digest=self.model_digest)
            llm = StreamingLLM(self.client, events.put, stop)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            with TraceWriter(self.live_dir / f"{stamp}-{scene}.jsonl") as writer:
                sink = _QueueSink(writer, events)
                for name in ORDER:
                    if stop.is_set():
                        break
                    llm.begin_run(name)
                    end = run_one(setup, APPROACHES[name], item, 0, 0, llm, sink, mode="demo")
                    if end.status != "ok" and not stop.is_set():
                        raise RuntimeError(f"{name} failed: {end.error}")
            events.put(_DONE)
        except Exception as exc:  # noqa: BLE001  (every failure reaches the presenter)
            events.put(_Failure(exc))

    async def _stream(self, events: queue.Queue, stop: threading.Event,
                      details: dict[str, str],
                      reader_timer: threading.Timer) -> AsyncIterator[dict[str, Any]]:
        reader_timer.cancel()
        try:
            while True:
                try:
                    message = await run_in_threadpool(events.get, True, 0.1)
                except queue.Empty:
                    continue
                if message is _DONE:
                    return
                if isinstance(message, _Failure):
                    raise LiveFailed(str(message.error), details) from message.error
                yield message
        finally:
            stop.set()
