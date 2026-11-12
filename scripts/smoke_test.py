"""Smoke test summary: baseline versus ToT on a few Game of 24 puzzles.

First run both approaches with the runner, then summarize the trace:

    python -m agentic.runner --task game24 --split dev --approach baseline,tot --reps 1 --tag smoke
    python -m scripts.smoke_test traces/runs/<experiment folder>/trace.jsonl

Like all analysis, this reads only the trace. Besides accuracy and cost it checks the ToT
tree after the fact (diagnostics only, never shown to the model):
  - valid steps: the step's arithmetic is exact and its "left" numbers follow from the parent;
  - "sure" precision: how many nodes the scorer called "sure" can really still reach 24.
"""

import argparse
import operator
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from agentic.events import NodeAdded, NodeScored, RunEnd, RunStart
from agentic.tasks.game24 import can_reach_24
from agentic.traces import group_runs, read_trace

OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
_NUM = r"-?\d+(?:\.\d+)?(?:/\d+)?"
_STEP = re.compile(rf"^\s*({_NUM})\s*([-+*/])\s*({_NUM})\s*=\s*({_NUM})\s*$")


def parse_state(state: str) -> list[Fraction] | None:
    """'4 6 3.5' -> fractions, or None when a token is not a number."""
    try:
        return [Fraction(tok) for tok in state.split()]
    except (ValueError, ZeroDivisionError):
        return None


def valid_step(parent_state: str, step: str, child_state: str) -> bool:
    """True if 'a op b = c' uses two numbers of the parent, c is exactly right, and the
    child's numbers are the parent's minus a and b plus c."""
    match = _STEP.match(step)
    parent, child = parse_state(parent_state), parse_state(child_state)
    if not match or parent is None or child is None:
        return False
    a, op, b, c = match.groups()
    a, b, c = Fraction(a), Fraction(b), Fraction(c)
    try:
        if OPS[op](a, b) != c:
            return False
        parent.remove(a)
        parent.remove(b)
    except (ZeroDivisionError, ValueError):     # division by zero, or number not available
        return False
    return sorted(parent + [c]) == sorted(child)


@dataclass
class ApproachStats:
    runs: int = 0
    correct: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0
    wall_ms: int = 0
    solved: set[tuple[str, int]] = field(default_factory=set)


@dataclass
class TreeStats:
    steps: int = 0
    valid_steps: int = 0
    labels: Counter = field(default_factory=Counter)
    sure: int = 0
    sure_reachable: int = 0


def summarize(trace_path: Path) -> tuple[dict[str, ApproachStats], TreeStats, set]:
    """Per-approach totals, ToT tree diagnostics, and all (item_id, rep) pairs seen."""
    stats: dict[str, ApproachStats] = {}
    tree = TreeStats()
    pairs = set()
    for run in group_runs(read_trace(trace_path)).values():
        env, start = run[0]
        end = run[-1][1]
        if not isinstance(start, RunStart) or not isinstance(end, RunEnd) or end.status != "ok":
            continue
        s = stats.setdefault(env.approach, ApproachStats())
        s.runs += 1
        s.correct += bool(end.correct)
        s.tokens_in += end.tokens_in
        s.tokens_out += end.tokens_out
        s.calls += end.llm_calls
        s.wall_ms += end.wall_ms
        pairs.add((env.item_id, env.rep))
        if end.correct:
            s.solved.add((env.item_id, env.rep))

        nodes: dict[str, NodeAdded] = {}
        for _, body in run:
            if isinstance(body, NodeAdded):
                nodes[body.node_id] = body
                if body.parent_id is not None:
                    tree.steps += 1
                    tree.valid_steps += valid_step(nodes[body.parent_id].state, body.step,
                                                   body.state)
            elif isinstance(body, NodeScored) and body.call_id is not None:  # not cached
                for label in body.label.split(", "):        # one label per judgement
                    tree.labels[label] += 1
                    if label == "sure":
                        tree.sure += 1
                        values = parse_state(nodes[body.node_id].state)
                        tree.sure_reachable += bool(values) and can_reach_24(values)
    return stats, tree, pairs


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


def report(trace_path: Path) -> str:
    stats, tree, pairs = summarize(trace_path)
    if not stats:
        return "No finished runs in this trace."
    header = (f"{'approach':<10} {'solved':>9} {'acc':>5} {'calls/run':>10} "
              f"{'tok in/run':>11} {'tok out/run':>12} {'sec/run':>8}")
    lines = [header]
    for name, s in stats.items():
        lines.append(f"{name:<10} {s.correct:>4} / {s.runs:<2} {pct(s.correct, s.runs):>5} "
                     f"{s.calls / s.runs:>10.1f} {s.tokens_in / s.runs:>11.0f} "
                     f"{s.tokens_out / s.runs:>12.0f} {s.wall_ms / s.runs / 1000:>8.1f}")
    if {"baseline", "tot"} <= stats.keys():
        base, tot = stats["baseline"].solved, stats["tot"].solved
        paired = (f"Paired (same item and rep): both {len(base & tot)}, "
                  f"only baseline {len(base - tot)}, only ToT {len(tot - base)}, "
                  f"neither {len(pairs - base - tot)}")
        lines += ["", paired]
    if tree.steps:
        labels = ", ".join(f"{k} {v}" for k, v in tree.labels.most_common())
        valid = (f"  valid steps: {tree.valid_steps} / {tree.steps} "
                 f"({pct(tree.valid_steps, tree.steps)})")
        sure = (f"  'sure' that can really reach 24: {tree.sure_reachable} / {tree.sure} "
                f"({pct(tree.sure_reachable, tree.sure)})")
        lines += ["", "ToT tree (checked after the run, never shown to the model):", valid,
                  f"  scorer labels: {labels}", sure]
    lines += ["", "Small sample: treat differences below about 15 points as noise."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a baseline vs ToT smoke run.")
    parser.add_argument("trace", type=Path, help="trace.jsonl written by the runner")
    args = parser.parse_args(argv)
    print(report(args.trace))
    return 0


if __name__ == "__main__":
    sys.exit(main())
