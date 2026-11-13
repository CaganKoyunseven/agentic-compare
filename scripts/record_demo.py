"""Record the replay scenes of the demo: every fixed scene and every draw-card puzzle.

Runs the four approaches with the demo settings (configs/demo.toml) and writes one trace per
scene into traces/examples/: fixed scenes as <scene>.jsonl, pool puzzles as
pool-<item>.jsonl. These are the replays the page falls back to when live mode fails, so run
this after the demo settings are final, then commit the files.

Usage: python -m scripts.record_demo [--only easy24,pool-game24-0043] [--out traces/examples]
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from agentic.config import ROOT, machine_id, model_name
from agentic.demo_config import (
    DRAW_SCENE,
    SCENE_TASKS,
    DemoConfig,
    load_demo_config,
    replay_scene_for,
)
from agentic.llm.backend import make_client
from agentic.llm.base import LLMClient
from agentic.llm.ollama import OllamaError
from agentic.runner import APPROACHES, ORDER, TASKS, run_one
from agentic.server.live import demo_setup
from agentic.traces import TraceWriter

EXAMPLES_DIR = ROOT / "traces" / "examples"


def scene_list(demo: DemoConfig) -> list[tuple[str, str, str]]:
    """(file name without .jsonl, task name, item id) for every scene to record."""
    scenes = [(scene, SCENE_TASKS[scene], item_id) for scene, item_id in demo.scenes.items()]
    for ids in demo.pool.values():
        scenes += [(replay_scene_for(DRAW_SCENE, item_id), "game24", item_id) for item_id in ids]
    unique: dict[str, tuple[str, str, str]] = {}
    for scene in scenes:
        unique.setdefault(scene[0], scene)
    return list(unique.values())


def record(demo: DemoConfig, llm: LLMClient, out_dir: Path, *, model: str, machine_id: str,
           ollama_version: str | None = None, model_digest: str | None = None,
           only: list[str] | None = None, log: Callable[[str], None] = print) -> list[Path]:
    """Record the scenes (all, or the names in only) and return the written files.
    An existing file is replaced only after its new recording is complete."""
    scenes = scene_list(demo)
    if only is not None:
        unknown = sorted(set(only) - {name for name, _, _ in scenes})
        if unknown:
            raise ValueError(f"unknown scenes {unknown}; known: {[n for n, _, _ in scenes]}")
        scenes = [scene for scene in scenes if scene[0] in only]

    out_dir = Path(out_dir)
    written = []
    for name, task_name, item_id in scenes:
        task = TASKS[task_name]()
        item = {it.item_id: it for it in task.load_items("dev")}[item_id]
        setup = demo_setup(task, demo, model=model, machine_id=machine_id,
                           ollama_version=ollama_version, model_digest=model_digest)
        final = out_dir / f"{name}.jsonl"
        tmp = out_dir / f"{name}.jsonl.tmp"
        tmp.unlink(missing_ok=True)
        with TraceWriter(tmp) as writer:
            for approach in ORDER:
                end = run_one(setup, APPROACHES[approach], item, 0, 0, llm, writer, mode="demo")
                verdict = {True: "correct", False: "wrong"}.get(end.correct, end.status)
                log(f"{name:<22} {approach:<11} {verdict:<9} {end.llm_calls} calls "
                    f"{end.wall_ms / 1000:.1f}s" + (f"  {end.error}" if end.error else ""))
        tmp.replace(final)
        written.append(final)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record the demo replay scenes.")
    parser.add_argument("--only", help="comma list of scene names, e.g. easy24,pool-game24-0043")
    parser.add_argument("--out", type=Path, default=EXAMPLES_DIR)
    args = parser.parse_args(argv)

    demo = load_demo_config()
    model, machine = model_name(), machine_id()
    llm = make_client(model, **demo.llm_settings, keep_alive=demo.keep_alive)
    try:
        version, digest = llm.server_info()
        if digest is None:
            raise SystemExit(f"Model {model} is not installed in Ollama (ollama pull {model})")
        warm = llm.chat([{"role": "user", "content": "Reply with OK."}])
        print(f"Model {model} warm ({warm.wall_ms / 1000:.1f} s), backend {version}")
        only = [n.strip() for n in args.only.split(",") if n.strip()] if args.only else None
        paths = record(demo, llm, args.out, model=model, machine_id=machine,
                       ollama_version=version, model_digest=digest, only=only)
    except OllamaError as exc:
        print(f"Ollama error: {exc}", file=sys.stderr)
        return 1
    finally:
        llm.close()
    for path in paths:
        print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
