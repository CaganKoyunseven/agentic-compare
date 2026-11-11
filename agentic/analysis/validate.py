"""Check benchmark runs before they enter the results table and the figures.

Every finished run is checked against what the experiment should be (model, split, frozen
settings, seed) and against its own trace (complete events, token totals, a fresh official
re-score of the answer). A run that fails any check is left out, with the reason. Afterwards
an item is kept only if every approach has a valid run for every repetition; otherwise the
item is dropped from all approaches, so the approaches are always compared on the same items.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from agentic.events import Envelope, FinalAnswer, LLMCall, RunEnd, RunStart
from agentic.runner import config_hash, csv_row, run_seed
from agentic.tasks.base import Item, Task

Run = list[tuple[Envelope, Any]]


@dataclass(frozen=True)
class Expected:
    """What a run of the main results must look like."""
    model: str
    split: str
    base_seed: int
    reps: int
    approaches: tuple[str, ...]
    configs: dict[str, dict[str, Any]]     # approach -> config recorded on run_start


@dataclass(frozen=True)
class Exclusion:
    """A run or an item that is not in the results, and why."""
    source: str           # experiment folder name
    task: str
    approach: str         # "" when a whole item is dropped
    item_id: str
    rep: int | None
    run_id: str
    reason: str


def check_run(run: Run, task: Task, items: dict[str, tuple[int, Item]],
              expected: Expected) -> list[str]:
    """Reasons why this run must not enter the results; empty when it is valid."""
    env, start = run[0]
    end = run[-1][1]
    if not isinstance(start, RunStart):
        return ["the trace has no run_start for this run"]
    if not isinstance(end, RunEnd):
        return ["the run never ended (no run_end, probably interrupted)"]
    if end.status != "ok":
        return [f"the run ended with status {end.status}" + (f": {end.error}" if end.error else "")]

    reasons = []
    if start.mode != "benchmark":
        reasons.append(f"mode is {start.mode}, not benchmark")
    if start.split != expected.split:
        reasons.append(f"split is {start.split}, not {expected.split}")
    if env.model != expected.model:
        reasons.append(f"model is {env.model}, not {expected.model}")
    if env.approach not in expected.approaches:
        reasons.append(f"unknown approach {env.approach}")
    elif start.config_hash != config_hash(expected.configs[env.approach]):
        reasons.append("settings differ from configs/experiment.toml "
                       f"(config_hash {start.config_hash}, expected "
                       f"{config_hash(expected.configs[env.approach])})")
    if env.rep not in range(expected.reps):
        reasons.append(f"rep {env.rep} is outside 0..{expected.reps - 1}")

    if env.item_id not in items:
        reasons.append(f"item {env.item_id} is not in the {expected.split} split")
        return reasons
    index, item = items[env.item_id]
    if start.seed != run_seed(expected.base_seed, env.rep, index):
        reasons.append(f"seed {start.seed}, expected {run_seed(expected.base_seed, env.rep, index)}")

    calls = [body for _, body in run if isinstance(body, LLMCall)]
    if not calls:
        reasons.append("no LLM call in the trace")
    totals = (sum(c.tokens_in for c in calls), sum(c.tokens_out for c in calls), len(calls))
    if totals != (end.tokens_in, end.tokens_out, end.llm_calls):
        reasons.append(f"run_end totals {(end.tokens_in, end.tokens_out, end.llm_calls)} "
                       f"do not match the llm_call events {totals}")

    finals = [body.answer for _, body in run if isinstance(body, FinalAnswer)]
    if len(finals) != 1:
        reasons.append(f"{len(finals)} final_answer events, expected 1")
    elif finals[0] != end.answer:
        reasons.append("final_answer differs from the answer in run_end")
    elif end.correct is None:
        reasons.append("run_end has no official verdict")
    else:
        rescored = task.score(item, end.answer).correct
        if rescored != end.correct:
            reasons.append(f"official re-score says correct={rescored}, "
                           f"the trace says correct={end.correct}")
    return reasons


def validate_runs(runs: Iterable[Run], task: Task, split_items: Sequence[Item],
                  expected: Expected, source: str) -> tuple[list[dict], list[Exclusion]]:
    """Rows (with a "ts" key for de-duplication) of the valid runs, and the excluded runs."""
    items = {item.item_id: (index, item) for index, item in enumerate(split_items)}
    rows, excluded = [], []
    for run in runs:
        env, start = run[0]
        reasons = check_run(run, task, items, expected)
        if reasons:
            excluded.append(Exclusion(source, env.task, env.approach, env.item_id, env.rep,
                                      env.run_id, "; ".join(reasons)))
        else:
            rows.append(csv_row(env, start, run[-1][1]) | {"ts": env.ts, "source": source})
    return rows, excluded


def drop_duplicates(rows: list[dict]) -> tuple[list[dict], list[Exclusion]]:
    """One row per (task, approach, item, rep): the latest run wins, the others are noted."""
    latest: dict[tuple, dict] = {}
    for row in sorted(rows, key=lambda r: r["ts"]):
        key = (row["task"], row["approach"], row["item_id"], row["rep"])
        latest[key] = row
    kept_ids = {row["run_id"] for row in latest.values()}
    excluded = [
        Exclusion(row["source"], row["task"], row["approach"], row["item_id"], row["rep"],
                  row["run_id"], "duplicate: a later valid run of the same approach, item "
                  f"and rep is used ({latest[(row['task'], row['approach'], row['item_id'], row['rep'])]['run_id']})")
        for row in rows if row["run_id"] not in kept_ids
    ]
    return list(latest.values()), excluded


def keep_complete_items(rows: list[dict], approaches: Sequence[str],
                        reps: int) -> tuple[list[dict], list[Exclusion]]:
    """Keep an item only if every approach has a valid run for every rep."""
    have: dict[tuple[str, str], set[tuple[str, int]]] = {}
    for row in rows:
        have.setdefault((row["task"], row["item_id"]), set()).add((row["approach"], row["rep"]))
    needed = {(a, r) for a in approaches for r in range(reps)}
    dropped = {}
    for key, present in have.items():
        missing = sorted(needed - present)
        if missing:
            dropped[key] = ", ".join(f"{a} rep {r}" for a, r in missing)
    kept = [row for row in rows if (row["task"], row["item_id"]) not in dropped]
    excluded = [Exclusion("", task, "", item_id, None, "",
                          f"item dropped from all approaches; no valid run for: {missing}")
                for (task, item_id), missing in sorted(dropped.items())]
    return kept, excluded
