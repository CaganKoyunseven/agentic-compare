"""Write and read JSONL traces.

A trace file holds one event per line. One runner call writes one file; runs inside it follow
each other, each from run_start to run_end. Traces are the single source of truth: analysis,
charts and demo replay read only these files.
"""

import json
from pathlib import Path
from typing import Self

from agentic.events import Body, Envelope, from_dict, to_dict


class TraceWriter:
    """Appends events to a JSONL file. Each line is flushed, so a crash loses at most one."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Kept open on purpose for many writes; closed by close() or the with block.
        self._f = open(self.path, "a", encoding="utf-8", newline="\n")  # noqa: SIM115

    def write(self, env: Envelope, body: Body) -> None:
        if body.type == "token":
            raise ValueError("token events are for live streaming only, not for traces")
        self._f.write(json.dumps(to_dict(env, body), ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_trace(path: Path) -> list[tuple[Envelope, Body]]:
    """All events of a trace file, in file order. Blank lines are skipped."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(from_dict(json.loads(line)))
    return records


def group_runs(
    records: list[tuple[Envelope, Body]],
) -> dict[str, list[tuple[Envelope, Body]]]:
    """Events per run_id, runs in order of first appearance, events in file order."""
    runs: dict[str, list[tuple[Envelope, Body]]] = {}
    for env, body in records:
        runs.setdefault(env.run_id, []).append((env, body))
    return runs
