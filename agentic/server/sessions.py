"""Live and replay sessions that produce event streams.

Every stream yields flat event dicts, exactly as one trace line. Replay reads a recorded trace
and waits between events; live mode (connected by the owner) yields the same dicts from the
model. The server turns both into the same SSE text, so the page cannot tell them apart.
"""

import asyncio
import random
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from agentic.config import ROOT
from agentic.events import to_dict
from agentic.tasks.base import Item
from agentic.tasks.game24 import Game24Task
from agentic.traces import read_trace

EXAMPLES_DIR = ROOT / "traces" / "examples"
MOCK_DIR = ROOT / "traces" / "mock"
DIFFICULTIES = ("easy", "medium", "hard")


# ---- Scenes ----

def find_scenes(traces_dir: Path) -> dict[str, Path]:
    """Scene id (file stem) -> path, for every *.jsonl in traces_dir, sorted by name.

    Only these ids are accepted by the server, so a request can never read another path.
    """
    traces_dir = Path(traces_dir)
    if not traces_dir.is_dir():
        return {}
    paths = sorted(p for p in traces_dir.glob("*.jsonl") if p.is_file())
    return {p.stem: p for p in paths}


def default_traces_dir() -> Path:
    """traces/examples/ when it holds at least one *.jsonl file, otherwise traces/mock/."""
    if find_scenes(EXAMPLES_DIR):
        return EXAMPLES_DIR
    return MOCK_DIR


def load_scene(path: Path) -> list[dict[str, Any]]:
    """All events of a trace file as flat dicts, in file order.

    read_trace validates every line against the event schema, so a broken file fails loudly.
    """
    return [to_dict(env, body) for env, body in read_trace(path)]


# ---- Replay ----

async def replay_events(
    events: list[dict[str, Any]],
    speed: float = 1.0,
    max_gap_ms: int = 2000,
    run_gap_ms: int = 500,
) -> AsyncIterator[dict[str, Any]]:
    """Yield the events one by one, waiting as recorded.

    Inside one run the wait is the t_ms difference to the previous event, capped at max_gap_ms;
    between two runs it is run_gap_ms. Both are divided by speed; speed=0 means no waiting.
    The events are never changed: the page shows the recorded numbers, not the replay clock.
    """
    if speed < 0:
        raise ValueError("speed must be 0 or positive")
    previous: dict[str, Any] | None = None
    for event in events:
        if previous is not None and speed > 0:
            if event["run_id"] != previous["run_id"]:
                wait_ms = run_gap_ms
            else:
                wait_ms = min(max(event["t_ms"] - previous["t_ms"], 0), max_gap_ms)
            if wait_ms > 0:
                await asyncio.sleep(wait_ms / 1000 / speed)
        yield event
        previous = event


# ---- Live (hook for the owner) ----

def live_events(scene: str, item_id: str | None = None) -> Iterator[dict[str, Any]]:
    """Live hook, connected by the owner.

    The owner replaces this with a generator that yields the same flat dicts as a trace line,
    plus {"type": "token", "call_id": ..., "text": ...} events while a model call is streaming.
    """
    raise NotImplementedError("Live mode is connected by the owner")


# ---- Draw card ----

def draw_pool() -> list[Item]:
    """Puzzles the draw card picks from. Replace this one line to use a curated list."""
    return Game24Task().load_items("dev")


def draw_card(difficulty: str, rng: random.Random | None = None) -> dict[str, Any]:
    """A random puzzle of the given difficulty ("easy", "medium" or "hard") from draw_pool()."""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty {difficulty!r}; use one of {', '.join(DIFFICULTIES)}")
    pool = [item for item in draw_pool() if item.difficulty == difficulty]
    if not pool:
        raise ValueError(f"no puzzles with difficulty {difficulty!r}")
    item = (rng or random.Random()).choice(pool)
    return {"item_id": item.item_id, "numbers": list(item.payload["numbers"]),
            "difficulty": item.difficulty}
