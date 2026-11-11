"""Validate all benchmark traces, then rebuild results/ from the runs that pass.

Run: python -m scripts.update_results [--runs <dir>] [--out <dir>]

The runner calls this automatically after a test-split experiment. Steps:
  1. Find every traces/runs/*/trace.jsonl. Experiments of another split, mode or model are
     listed as "not used" (for example dev-set tuning runs); they never enter the results.
  2. Check every run of the remaining experiments (agentic.analysis.validate).
  3. Keep one run per (task, approach, item, rep) and only items that are complete for all
     approaches and reps.
  4. Write results/runs.csv, results/validation.md (what was left out and why), and the
     figures plus report.md (scripts.make_report).
  5. Separate studies (self-consistency) never enter steps 2-4: their runs are skipped here
     and get their own folder, results/self_consistency/ (agentic.analysis.self_consistency).
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from agentic.analysis import stats
from agentic.analysis.self_consistency import update_self_consistency
from agentic.analysis.validate import (
    Exclusion,
    Expected,
    drop_duplicates,
    keep_complete_items,
    validate_runs,
)
from agentic.config import ROOT, load_toml, model_name
from agentic.events import RunStart
from agentic.runner import ORDER, SEPARATE, TASKS, TRACES_DIR, write_rows
from agentic.traces import group_runs, read_trace
from scripts.make_report import make_report

RESULTS_DIR = ROOT / "results"
SPLIT = "test"
SUPERSEDED_REASON = "settings differ from configs/experiment.toml"


def expected_setup(model: str) -> Expected:
    """The frozen settings from configs/experiment.toml, as the runner records them."""
    settings = load_toml("experiment")
    approaches = settings.get("approaches", {})
    configs = {name: {"llm": settings["llm"], "approach": approaches.get(name, {})}
               for name in ORDER}
    return Expected(model=model, split=SPLIT, base_seed=settings["seed"],
                    reps=settings["reps"], approaches=tuple(ORDER), configs=configs)


def first_start(records) -> tuple | None:
    return next(((env, body) for env, body in records if isinstance(body, RunStart)), None)


def not_used_reason(env, start: RunStart, expected: Expected) -> str:
    """Why a whole experiment is outside the main results by design ("" if it is inside)."""
    if start.mode != "benchmark":
        return f"mode {start.mode}"
    if start.split != expected.split:
        return f"{start.split} split (tuning only)"
    if env.model != expected.model:
        return f"other model ({env.model})"
    if env.task not in TASKS:
        return f"unknown task {env.task}"
    return ""


def collect(runs_dir: Path, expected: Expected):
    """Valid rows, excluded runs, and experiments that are not used, from all traces."""
    rows, excluded, not_used, used = [], [], [], []
    split_items = {}
    for trace in sorted(runs_dir.glob("*/trace.jsonl")):
        source = trace.parent.name
        records = read_trace(trace)
        first = first_start(records)
        if first is None:
            not_used.append((source, "no runs in the trace"))
            continue
        reason = not_used_reason(*first, expected)
        if reason:
            not_used.append((source, reason))
            continue
        runs = [r for r in group_runs(records).values() if r[0][0].approach not in SEPARATE]
        if not runs:
            not_used.append((source, "separate study (see self_consistency/)"))
            continue
        task_name = first[0].task
        if task_name not in split_items:
            task = TASKS[task_name]()
            split_items[task_name] = (task, task.load_items(expected.split))
        task, items = split_items[task_name]
        ok_rows, bad = validate_runs(runs, task, items, expected, source)
        rows += ok_rows
        excluded += bad
        used.append((source, task_name, len(runs), len(ok_rows)))
    return rows, excluded, not_used, used


def validation_markdown(expected: Expected, used, not_used, excluded: list[Exclusion],
                        dropped: list[Exclusion], kept_rows: list[dict]) -> str:
    lines = [
        "# Result validation",
        "",
        (f"Generated {datetime.now().astimezone():%Y-%m-%d %H:%M %Z} by "
         "`python -m scripts.update_results`."),
        "",
        (f"Expected: model `{expected.model}`, split `{expected.split}`, mode `benchmark`, "
         f"{expected.reps} reps, approaches {', '.join(expected.approaches)}, settings from "
         "`configs/experiment.toml`."),
        "",
        ("Checks on every run: finished with status ok; mode, split and model as expected; "
         "settings hash equals the frozen settings; seed as the runner sets it; item in the "
         "split; token and call totals equal the LLM call events; exactly one final answer, "
         "equal to the scored answer; a fresh official re-score gives the same verdict. "
         "An item is kept only if every approach has a valid run for every rep."),
        "",
        "## Summary",
        "",
    ]
    by_task: dict[str, set] = {}
    for row in kept_rows:
        by_task.setdefault(row["task"], set()).add(row["item_id"])
    lines.append(f"- Runs in the results: {len(kept_rows)}")
    for task, items in sorted(by_task.items()):
        lines.append(f"- {task}: {len(items)} items")
    lines.append(f"- Runs left out: {len(excluded)}")
    lines.append(f"- Items dropped from all approaches: {len(dropped)}")
    lines += ["", "## Experiments used", "",
              "| Experiment | Task | Runs in trace | Runs that passed the checks |",
              "|---|---|---|---|"]
    lines += [f"| {s} | {t} | {n} | {ok} |" for s, t, n, ok in used] or ["| (none) | | | |"]

    # Runs made with earlier settings (for example ToT before the writer fix) are expected to
    # be left out; they get one line per task and approach instead of one line per run.
    superseded: dict[tuple[str, str], int] = {}
    others = []
    for e in excluded:
        if e.reason.startswith(SUPERSEDED_REASON):
            superseded[(e.task, e.approach)] = superseded.get((e.task, e.approach), 0) + 1
        else:
            others.append(e)
    lines += ["", "## Runs left out", ""]
    if superseded:
        lines += ["| Task | Approach | Runs | Reason |", "|---|---|---|---|"]
        lines += [f"| {task} | {approach} | {n} | earlier settings (superseded by "
                  "configs/experiment.toml) |" for (task, approach), n in sorted(superseded.items())]
        lines.append("")
    if others:
        lines += ["| Experiment | Approach | Item | Rep | Run | Reason |", "|---|---|---|---|---|---|"]
        lines += [f"| {e.source} | {e.approach} | {e.item_id} | {e.rep} | {e.run_id} | {e.reason} |"
                  for e in others]
    elif not superseded:
        lines.append("None.")

    lines += ["", "## Items dropped from all approaches", ""]
    if dropped:
        lines += ["| Task | Item | Reason |", "|---|---|---|"]
        lines += [f"| {e.task} | {e.item_id} | {e.reason} |" for e in dropped]
    else:
        lines.append("None.")

    # Counts only: folder names of tuning runs carry machine ids that do not belong in the repo.
    lines += ["", "## Experiments not used (by design)", ""]
    if not_used:
        counts: dict[str, int] = {}
        for _source, reason in not_used:
            counts[reason] = counts.get(reason, 0) + 1
        lines += ["| Reason | Experiments |", "|---|---|"]
        lines += [f"| {reason} | {n} |" for reason, n in sorted(counts.items())]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def update_results(runs_dir: Path = TRACES_DIR, out_dir: Path = RESULTS_DIR,
                   model: str | None = None, log=print) -> int:
    """Rebuild out_dir from the traces, then the separate self-consistency folder (it reads
    the new runs.csv for Baseline and ToT). Returns the number of runs in the main results."""
    model = model or model_name()
    n = update_main(runs_dir, out_dir, model, log)
    update_self_consistency(runs_dir, out_dir / "self_consistency", out_dir / "runs.csv",
                            model, TASKS["game24"](), log)
    return n


def update_main(runs_dir: Path, out_dir: Path, model: str, log=print) -> int:
    """Rebuild the main results in out_dir. Returns the number of runs in them."""
    expected = expected_setup(model)
    rows, excluded, not_used, used = collect(runs_dir, expected)
    rows, duplicates = drop_duplicates(rows)
    rows, dropped = keep_complete_items(rows, expected.approaches, expected.reps)
    excluded += duplicates
    rows.sort(key=lambda r: (r["task"], r["rep"], r["item_id"], ORDER.index(r["approach"])))

    out_dir.mkdir(parents=True, exist_ok=True)
    validation = out_dir / "validation.md"
    validation.write_text(validation_markdown(expected, used, not_used, excluded, dropped, rows),
                          encoding="utf-8", newline="\n")
    log(f"{validation}: {len(rows)} runs in the results, {len(excluded)} runs left out, "
        f"{len(dropped)} items dropped")
    if not rows:
        log("No valid runs: figures and report.md were not changed.")
        return 0

    csv_path = out_dir / "runs.csv"
    write_rows(csv_path, [{k: v for k, v in r.items() if k not in ("ts", "source")}
                          for r in rows])
    report = make_report(stats.load_runs(csv_path), csv_path.name, out_dir)
    with open(report, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"\n\nOnly runs that passed the checks are included: see `validation.md` "
                f"({len(excluded)} runs left out, {len(dropped)} items dropped).\n")
    log(f"{report} and figures updated")
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate traces and rebuild results/.")
    parser.add_argument("--runs", type=Path, default=TRACES_DIR, help="folder of experiments")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="output folder")
    args = parser.parse_args(argv)
    update_results(args.runs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
