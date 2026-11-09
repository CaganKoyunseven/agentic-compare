"""Tests for live mode, with a fake streaming client (no Ollama needed)."""

import asyncio
import threading
import time

import pytest

from agentic.demo_config import load_demo_config
from agentic.events import RunStart
from agentic.llm.base import LLMResult
from agentic.llm.ollama import OllamaError
from agentic.runner import APPROACHES, ORDER, run_one, write_csv
from agentic.server.live import LiveBusy, LiveFailed, LiveRunner, StreamingLLM, demo_setup
from agentic.tasks.game24 import Game24Task
from agentic.traces import TraceWriter, group_runs, read_trace

REPLY = "Let me think step by step.\n12*(7-5)/1"


class FakeStreamClient:
    """Streams a canned reply in 3-character pieces."""

    model = "fake:3b"

    def __init__(self, reply=REPLY, fail_on_call=None, delay_s=0.0):
        self.reply, self.fail_on_call, self.delay_s = reply, fail_on_call, delay_s
        self.calls_started = 0

    def chat_stream(self, messages, *, seed=None, on_text):
        if self.calls_started == self.fail_on_call:
            raise OllamaError("Cannot reach Ollama: connection refused")
        self.calls_started += 1
        for i in range(0, len(self.reply), 3):
            if self.delay_s:
                time.sleep(self.delay_s)
            on_text(self.reply[i:i + 3])
        return LLMResult(text=self.reply, tokens_in=20, tokens_out=9, wall_ms=1, load_ms=0,
                         model=self.model, done_reason="stop")


def make_runner(tmp_path, client=None):
    return LiveRunner(client or FakeStreamClient(), load_demo_config(), model="fake:3b",
                      machine_id="test", ollama_version=None, model_digest=None,
                      live_dir=tmp_path)


def collect(stream):
    async def run():
        return [event async for event in stream]
    return asyncio.run(run())


def wait_idle(runner, timeout_s=3.0):
    deadline = time.perf_counter() + timeout_s
    while runner.busy and time.perf_counter() < deadline:
        time.sleep(0.02)
    return not runner.busy


def test_live_yields_four_runs_in_order_with_demo_mode(tmp_path):
    events = collect(make_runner(tmp_path).start("easy24", None))
    starts = [e for e in events if e["type"] == "run_start"]
    assert [e["approach"] for e in starts] == ORDER
    assert all(e["mode"] == "demo" and e["split"] == "demo" for e in starts)
    assert all(e["item_id"] == load_demo_config().scenes["easy24"] for e in starts)
    assert [e["status"] for e in events if e["type"] == "run_end"] == ["ok"] * 4


def test_token_call_ids_match_llm_calls(tmp_path):
    events = collect(make_runner(tmp_path).start("easy24", None))
    for approach in ORDER:
        calls = {e["call_id"]: e["response"] for e in events
                 if e["type"] == "llm_call" and e["approach"] == approach}
        tokens: dict[str, str] = {}
        for e in events:
            if e["type"] == "token" and e["approach"] == approach:
                tokens[e["call_id"]] = tokens.get(e["call_id"], "") + e["text"]
        assert calls and tokens == calls


def test_tokens_come_before_their_llm_call(tmp_path):
    events = collect(make_runner(tmp_path).start("easy24", None))
    first_call = next(i for i, e in enumerate(events) if e["type"] == "llm_call")
    assert any(e["type"] == "token" for e in events[:first_call])


def test_live_trace_file_has_no_tokens_and_is_readable(tmp_path):
    collect(make_runner(tmp_path).start("easy24", None))
    paths = list(tmp_path.glob("*.jsonl"))
    assert len(paths) == 1 and paths[0].name.endswith("-easy24.jsonl")
    records = read_trace(paths[0])
    assert not any(body.type == "token" for _, body in records)
    assert len(group_runs(records)) == 4


def test_ollama_failure_raises_live_failed_with_replay_scene(tmp_path):
    runner = make_runner(tmp_path, FakeStreamClient(fail_on_call=1))
    with pytest.raises(LiveFailed) as info:
        collect(runner.start("draw", "game24/0068"))
    assert info.value.details == {"scene": "draw", "item_id": "game24/0068",
                                  "replay_scene": "pool-game24-0068"}
    assert "Cannot reach Ollama" in str(info.value)
    assert wait_idle(runner)


def test_close_early_stops_the_run_and_frees_the_runner(tmp_path):
    client = FakeStreamClient(delay_s=0.05)
    runner = make_runner(tmp_path, client)

    async def first_then_close():
        stream = runner.start("easy24", None)
        first = await anext(stream)
        await stream.aclose()
        return first

    assert asyncio.run(first_then_close())["type"] == "run_start"
    assert wait_idle(runner)
    calls = client.calls_started
    assert calls <= 2
    time.sleep(0.2)
    assert client.calls_started == calls          # nothing runs after the close
    client.delay_s = 0
    collect(runner.start("hard24", None))         # the runner is free again


def test_start_waits_for_a_run_that_is_stopping(tmp_path):
    # Reset followed at once by Start: the old worker needs a moment to see the stop flag.
    runner = make_runner(tmp_path, FakeStreamClient(delay_s=0.3))

    async def scenario():
        stream = runner.start("easy24", None)
        await anext(stream)
        await stream.aclose()
        assert runner.busy
        again = runner.start("hard24", None)
        first = await anext(again)
        await again.aclose()
        return first

    assert asyncio.run(scenario())["type"] == "run_start"
    assert wait_idle(runner)


def test_second_start_while_running_is_busy(tmp_path, monkeypatch):
    monkeypatch.setattr("agentic.server.live.BUSY_WAIT_S", 0.0)
    runner = make_runner(tmp_path, FakeStreamClient(delay_s=0.05))

    async def scenario():
        stream = runner.start("easy24", None)
        await anext(stream)
        with pytest.raises(LiveBusy):
            runner.start("hard24", None)
        await stream.aclose()

    asyncio.run(scenario())
    assert wait_idle(runner)


def test_unknown_scene_and_off_pool_item(tmp_path):
    client = FakeStreamClient()
    runner = make_runner(tmp_path, client)
    with pytest.raises(KeyError):
        runner.start("nope", None)
    with pytest.raises(KeyError):
        runner.start("draw", "game24/0001")
    with pytest.raises(KeyError):
        runner.start("draw", None)
    assert client.calls_started == 0 and not runner.busy


def test_streaming_llm_batches_and_numbers_calls():
    sent = []
    llm = StreamingLLM(FakeStreamClient(reply="abcdefghij"), sent.append, threading.Event(),
                       batch_ms=10_000)
    llm.begin_run("baseline")
    llm.chat([{"role": "user", "content": "x"}])
    llm.chat([{"role": "user", "content": "y"}])
    assert sent == [
        {"type": "token", "approach": "baseline", "call_id": "call-0", "text": "abcdefghij"},
        {"type": "token", "approach": "baseline", "call_id": "call-1", "text": "abcdefghij"},
    ]
    llm.begin_run("tot")
    llm.chat([{"role": "user", "content": "z"}])
    assert sent[-1]["call_id"] == "call-0" and sent[-1]["approach"] == "tot"


def test_run_one_records_the_given_mode(tmp_path):
    class Sink:
        def __init__(self):
            self.bodies = []

        def write(self, env, body):
            self.bodies.append(body)

    item = Game24Task().load_items("dev")[0]
    setup = demo_setup(Game24Task(), load_demo_config(), model="fake:3b", machine_id="test")
    llm = StreamingLLM(FakeStreamClient(), lambda event: None, threading.Event())
    llm.begin_run("baseline")
    demo_sink, bench_sink = Sink(), Sink()
    run_one(setup, APPROACHES["baseline"], item, 0, 0, llm, demo_sink, mode="demo")
    run_one(setup, APPROACHES["baseline"], item, 0, 0, llm, bench_sink)
    assert isinstance(demo_sink.bodies[0], RunStart) and demo_sink.bodies[0].mode == "demo"
    assert bench_sink.bodies[0].mode == "benchmark"
    assert setup.split == "demo" and setup.base_seed == load_demo_config().seed


def test_csv_only_has_benchmark_runs(tmp_path):
    item = Game24Task().load_items("dev")[0]
    setup = demo_setup(Game24Task(), load_demo_config(), model="fake:3b", machine_id="test")
    llm = StreamingLLM(FakeStreamClient(), lambda event: None, threading.Event())
    llm.begin_run("baseline")
    trace = tmp_path / "trace.jsonl"
    with TraceWriter(trace) as writer:
        run_one(setup, APPROACHES["baseline"], item, 0, 0, llm, writer, mode="demo")
    assert write_csv(trace, tmp_path / "runs.csv") == 0
    with TraceWriter(trace) as writer:
        run_one(setup, APPROACHES["baseline"], item, 0, 0, llm, writer)
    assert write_csv(trace, tmp_path / "runs.csv") == 1
