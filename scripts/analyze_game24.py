"""Game of 24 trace analysis: why answers are wrong, and where ToT loses the solution.

Run: python -m scripts.analyze_game24 [--runs <dir>] [--results <runs.csv>] [--out <dir>]

Reads only the traces (the single source of truth) and only the runs listed in the validated
results table, so it describes exactly the runs behind the main results. No model calls.
Writes results/game24_analysis.md and results/game24_tot_funnel.png. Called by
scripts/update_results after the main results are rebuilt.

  1. Error types: the official check's message for every wrong final answer, per approach.
  2. ToT scorer: for every scored node, can its numbers left (as the model wrote them)
     really still make 24? Checked exactly with can_reach_24.
  3. ToT steps: is the arithmetic of each proposed step right, and are the numbers left
     exactly the parent's minus the two used plus the result?
  4. ToT funnel: per run, the first level where no "sound" node exists (every step on its path
     right and its numbers can still make 24), or where all sound nodes were pruned.
  5. Proposals beyond the first `breadth`: at the root, would more of the proposer's own lines
     have given a sound step? Exploratory: the paper keeps every proposed line.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from agentic.analysis.plots import COLORS, DPI, FIGSIZE, STYLE
from agentic.approaches.tot import ALL_LINES, parse_steps
from agentic.approaches.tot import DEFAULTS as TOT_DEFAULTS
from agentic.config import ROOT
from agentic.runner import TRACES_DIR
from agentic.tasks.base import Item
from agentic.tasks.game24 import Game24Task, can_reach_24, check_expression
from agentic.traces import group_runs, read_trace
from scripts.smoke_test import parse_state, valid_step

RESULTS_DIR = ROOT / "results"
REPORT = "game24_analysis.md"
FIGURE = "game24_tot_funnel.png"
APPROACHES = ("baseline", "reflective", "multi_agent", "tot")
KINDS = ("right numbers, value not 24", "wrong numbers used", "unreadable or not allowed",
         "division by zero", "no answer")
REACHED = "reached a sound 24 leaf"


# ---- 1. Error types ----

def error_kind(numbers: list[int], answer: str) -> str:
    message = check_expression(numbers, answer).message
    if message.startswith("evaluates to"):
        return KINDS[0]
    if message.startswith("uses numbers"):
        return KINDS[1]
    if message == "division by zero":
        return KINDS[3]
    if message == "empty answer":
        return KINDS[4]
    return KINDS[2]           # could not parse, operator or number not allowed, too long


def error_types(runs, items: dict[str, Item]) -> dict[str, dict[str, int]]:
    """approach -> error kind -> number of wrong runs."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for run in runs:
        env, end = run[0][0], run[-1][1]
        if end.type == "run_end" and end.status == "ok" and end.correct is False:
            counts[env.approach][error_kind(items[env.item_id].payload["numbers"],
                                            end.answer or "")] += 1
    return {a: dict(c) for a, c in counts.items()}


# ---- 2-5. ToT ----

def _reachable(state: str) -> bool | None:
    values = parse_state(state)
    return None if values is None else can_reach_24(values)


def tot_analysis(runs) -> dict:
    labels: dict[str, Counter] = defaultdict(Counter)
    steps: dict[int, Counter] = defaultdict(Counter)
    funnel: Counter = Counter()
    root = Counter()
    for run in runs:
        if run[0][0].approach != "tot" or run[-1][1].type != "run_end" \
                or run[-1][1].status != "ok":
            continue
        breadth = (run[0][1].config.get("approach", {}).get("breadth", TOT_DEFAULTS["breadth"])
                   or ALL_LINES)
        nodes, pruned, calls = {}, set(), {}
        for _, body in run:
            if body.type == "llm_call":
                calls[body.call_id] = body
            elif body.type == "node_added":
                nodes[body.node_id] = body
            elif body.type == "node_pruned":
                pruned.add(body.node_id)
            elif body.type == "node_scored" and body.call_id is not None:   # not cached
                truth = _reachable(nodes[body.node_id].state)
                key = {True: "reachable", False: "unreachable", None: "unreadable"}[truth]
                for label in body.label.split(", "):      # one label per judgement
                    labels[label][key] += 1

        sound: dict[str, bool] = {"root": True}
        for node in nodes.values():          # parents come before children in the trace
            if node.parent_id is None:
                continue
            ok = valid_step(nodes[node.parent_id].state, node.step, node.state)
            steps[node.depth][ok] += 1
            sound[node.node_id] = (sound[node.parent_id] and ok
                                   and bool(_reachable(node.state)))
        funnel[(_lost_at(nodes, sound, pruned), run[-1][1].correct)] += 1

        root_call = next((c for c in calls.values() if c.role == "proposer"), None)
        if root_call is not None:
            start = nodes["root"].state
            proposed = [valid_step(start, s, st) and bool(_reachable(st))
                        for s, st in parse_steps(root_call.response, limit=100)]
            root["calls"] += 1
            root["sound_in_first"] += any(proposed[:breadth])
            root["sound_in_all"] += any(proposed)
    return {"labels": {k: dict(v) for k, v in labels.items()},
            "steps": {k: dict(v) for k, v in steps.items()},
            "funnel": dict(funnel), "root_proposals": dict(root)}


def _lost_at(nodes, sound, pruned) -> str:
    for depth in range(1, 4):
        level = [n for n in nodes.values() if n.depth == depth and sound[n.node_id]]
        if not level:
            return f"depth {depth}: no sound step proposed"
        if all(n.node_id in pruned for n in level):
            return f"depth {depth}: all sound nodes pruned by the scorer"
    return REACHED


# ---- Report ----

def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "n/a"


def report_markdown(errors: dict, tot: dict) -> str:
    lines = ["# Game of 24: error types and ToT search analysis", "",
             ("Computed from the traces of the validated test runs (the runs in `runs.csv`) "
              "by `python -m scripts.analyze_game24`. No model calls."), "",
             "## Why final answers are wrong (official check message)", "",
             "| Approach | Wrong runs | " + " | ".join(k.capitalize() for k in KINDS) + " |",
             "|---|---|" + "---|" * len(KINDS)]
    for approach in [a for a in APPROACHES if a in errors]:
        counts = errors[approach]
        total = sum(counts.values())
        cells = [f"{counts.get(k, 0)} ({_pct(counts.get(k, 0), total)})" for k in KINDS]
        lines.append(f"| {approach} | {total} | " + " | ".join(cells) + " |")

    lines += ["", "## Tree of Thoughts", ""]
    if not tot["funnel"]:
        return "\n".join(lines + ["No valid ToT runs."]) + "\n"
    lines += ["### Scorer labels vs. the truth", "",
              ("For every judgement of a scored state (cached repeats not counted again): can "
               "its numbers left, as the model wrote them, still make 24 (checked exactly)?"), "",
              "| Label | Judgements | Really reachable | Unreachable | Unreadable |", "|---|---|---|---|---|"]
    for label in sorted(tot["labels"], key=lambda k: ("sure", "likely", "impossible").index(k)
                        if k in ("sure", "likely", "impossible") else 9):
        c = tot["labels"][label]
        n = sum(c.values())
        lines.append(f"| {label} | {n} | {_pct(c.get('reachable', 0), n)} | "
                     f"{_pct(c.get('unreachable', 0), n)} | {_pct(c.get('unreadable', 0), n)} |")

    lines += ["", "### Proposed steps", "",
              ("A step is right when its arithmetic is exact and the numbers left are the parent's "
               "minus the two used plus the result."), "", "| Depth | Steps | Right |", "|---|---|---|"]
    for depth, c in sorted(tot["steps"].items()):
        n = sum(c.values())
        lines.append(f"| {depth} | {n} | {_pct(c.get(True, 0), n)} |")

    runs = sum(tot["funnel"].values())
    lines += ["", "### Where the search lost the solution", "",
              ("A node is sound when every step on its path is right and its numbers can still "
               "make 24. For each run: the first level with no sound node, or where every sound "
               "node was pruned."), "",
              "| Outcome of the search | Runs | Share | Final answer correct |", "|---|---|---|---|"]
    by_stage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for (stage, correct), n in tot["funnel"].items():
        by_stage[stage][0] += n
        by_stage[stage][1] += n if correct else 0
    for stage in sorted(by_stage, key=_stage_order):
        n, ok = by_stage[stage]
        lines.append(f"| {stage} | {n} | {_pct(n, runs)} | {ok} |")
    root = tot["root_proposals"]
    if root.get("calls"):
        lines += ["", "### More proposals at the root (exploratory)", "",
                  (f"Root proposer calls with at least one sound step among the lines used "
                   f"(the first `breadth`): {root['sound_in_first']} of {root['calls']} "
                   f"({_pct(root['sound_in_first'], root['calls'])}); among all lines the model "
                   f"wrote: {root['sound_in_all']} ({_pct(root['sound_in_all'], root['calls'])}). "
                   "The paper's code keeps every proposed line; ours keeps the first `breadth`.")]
    lines += ["", f"![{FIGURE}]({FIGURE})"]
    return "\n".join(lines) + "\n"


def _stage_order(stage: str) -> tuple:
    if stage == REACHED:
        return (9, "")
    return (int(stage.split()[1].rstrip(":")), stage)


def plot_funnel(tot: dict, out_path: Path) -> Path:
    by_stage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for (stage, correct), n in tot["funnel"].items():
        by_stage[stage][0 if correct else 1] += n
    stages = sorted(by_stage, key=_stage_order)
    total = sum(sum(v) for v in by_stage.values())
    with matplotlib.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
        y = range(len(stages))[::-1]
        right = [by_stage[s][0] for s in stages]
        wrong = [by_stage[s][1] for s in stages]
        ax.barh(list(y), right, color=COLORS["tot"], label="Final answer correct",
                edgecolor="white", linewidth=2)
        ax.barh(list(y), wrong, left=right, color="#D1D5DB", label="Final answer wrong",
                edgecolor="white", linewidth=2)
        for yi, r, w in zip(y, right, wrong, strict=True):
            ax.text(r + w + 1, yi, f"{r + w} ({100 * (r + w) / total:.0f}%)", va="center")
        ax.set_yticks(list(y), [s[0].upper() + s[1:] for s in stages])
        ax.set_xlabel("ToT runs")
        ax.set_xlim(0, max(r + w for r, w in zip(right, wrong, strict=True)) * 1.2)
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        fig.suptitle(f"Game of 24: where the ToT search lost the solution (n = {total} runs)")
        fig.legend(loc="outside lower center", ncols=2, frameon=False)
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
    return out_path


def validated_run_ids(results_csv: Path) -> set[str]:
    with open(results_csv, encoding="utf-8", newline="") as f:
        return {row["run_id"] for row in csv.DictReader(f) if row["task"] == "game24"}


def write_analysis(runs_dir: Path, results_csv: Path, out_dir: Path,
                   items: dict[str, Item] | None = None) -> Path:
    keep = validated_run_ids(results_csv)
    runs = []
    for trace in sorted(runs_dir.glob("*/trace.jsonl")):
        runs += [r for r in group_runs(read_trace(trace)).values() if r[0][0].run_id in keep]
    if items is None:
        items = {i.item_id: i for i in Game24Task().load_items("all")}
    tot = tot_analysis(runs)
    report = out_dir / REPORT
    report.write_text(report_markdown(error_types(runs, items), tot), encoding="utf-8",
                      newline="\n")
    if tot["funnel"]:
        plot_funnel(tot, out_dir / FIGURE)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse Game of 24 traces.")
    parser.add_argument("--runs", type=Path, default=TRACES_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR / "runs.csv")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    print(write_analysis(args.runs, args.results, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
