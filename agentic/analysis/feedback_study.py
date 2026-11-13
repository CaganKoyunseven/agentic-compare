"""Separate study: feedback on Game of 24 (Self-Refine, Debate, Retry with the verifier).

Added after the main test runs. It asks two questions the main comparison cannot answer:
  1. How much of the Reflective gain comes from the exact verifier, and how much from the loop?
     Self-Refine is the same loop with the model's own check instead of the verifier; Retry is
     plain retries with the verifier and no critique.
  2. How does a paper-like multi-agent setup (Debate, Du et al. 2023) compare with our
     planner / solver / critic Multi-agent approach?

The runs never enter the main results. This module validates them with the same checks as the
main results, then writes into results/feedback_study/:
  runs.csv            one row per valid run (contract 6.3 columns)
  validation.md       what was left out and why
  report.md           accuracy, cost, paired differences with the main approaches, and how
                      often the model's own check in Self-Refine was right
  fs_success.png      accuracy of the main and the study approaches
  fs_cost_accuracy.png  accuracy against tokens per run
"""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from agentic.analysis import stats
from agentic.analysis.plots import COLORS, DPI, FIGSIZE, STYLE
from agentic.analysis.validate import (
    Exclusion,
    Expected,
    Run,
    drop_duplicates,
    keep_complete_items,
    validate_runs,
)
from agentic.config import load_toml
from agentic.events import Attempt, RunStart, TestResult
from agentic.runner import FEEDBACK_STUDY, write_rows
from agentic.tasks.base import Item, Task
from agentic.traces import group_runs, read_trace

TASK = "game24"
SPLIT = "test"
MAIN = ("baseline", "reflective", "multi_agent", "tot")
NAMES = {"baseline": "Baseline", "reflective": "Reflective", "multi_agent": "Multi-agent",
         "tot": "Tree-of-Thoughts", "self_refine": "Self-Refine (own check)",
         "debate": "Debate (3 agents, 2 rounds)", "retry": "Retry with the verifier (max 10)"}
# Study approaches share the colour of the main approach they are compared with, hatched.
STUDY_COLORS = {"self_refine": COLORS["reflective"], "debate": COLORS["multi_agent"],
                "retry": "#A16207"}
PAIRS = (("self_refine", "baseline"), ("reflective", "self_refine"),
         ("debate", "baseline"), ("debate", "multi_agent"),
         ("retry", "reflective"), ("retry", "tot"))
FIGURES = ("fs_success.png", "fs_cost_accuracy.png")


def expected_setup(model: str) -> Expected:
    settings = load_toml("experiment")
    approaches = settings.get("approaches", {})
    configs = {a: {"llm": settings["llm"], "approach": approaches.get(a, {})}
               for a in FEEDBACK_STUDY}
    return Expected(model=model, split=SPLIT, base_seed=settings["seed"], reps=settings["reps"],
                    approaches=tuple(FEEDBACK_STUDY), configs=configs)


# ---- Per-run details ----

def self_checks(run: Run, task: Task, item: Item) -> list[tuple[bool, bool]]:
    """(the model said correct, the answer is really correct) for every Self-Refine check."""
    answers = {b.attempt_no: b.answer for _, b in run if isinstance(b, Attempt)}
    return [(b.ok, task.score(item, answers[b.attempt_no]).correct)
            for _, b in run if isinstance(b, TestResult) and b.source == "self_check"]


def attempts(run: Run) -> int:
    return sum(isinstance(b, Attempt) for _, b in run)


# ---- Collect and validate ----

def collect(runs_dir: Path, task: Task, items: list[Item], expected: Expected):
    """Valid rows, excluded runs, how many runs were not used, and per-run details
    (run_id -> {"checks": [...], "attempts": n})."""
    by_id = {item.item_id: item for item in items}
    rows, excluded, not_used, details = [], [], 0, {}
    for trace in sorted(runs_dir.glob("*/trace.jsonl")):
        runs = [r for r in group_runs(read_trace(trace)).values()
                if r[0][0].approach in FEEDBACK_STUDY]
        main = []
        for run in runs:
            start = run[0][1]
            if (isinstance(start, RunStart) and start.mode == "benchmark"
                    and start.split == SPLIT and run[0][0].task == TASK):
                main.append(run)
            else:
                not_used += 1          # dev-set tries or demo runs: never results
        if not main:
            continue
        ok_rows, bad = validate_runs(main, task, items, expected, trace.parent.name)
        excluded += bad
        records = {run[0][0].run_id: run for run in main}
        for row in ok_rows:
            run = records[row["run_id"]]
            details[row["run_id"]] = {"checks": self_checks(run, task, by_id[row["item_id"]]),
                                      "attempts": attempts(run)}
        rows += ok_rows
    return rows, excluded, not_used, details


def complete_per_approach(rows: list[dict], reps: int):
    """Keep, for each study approach on its own, only items with a valid run for every rep, so
    an approach whose runs are still missing does not remove items from the others."""
    kept, dropped = [], []
    for approach in FEEDBACK_STUDY:
        mine = [r for r in rows if r["approach"] == approach]
        if mine:
            k, d = keep_complete_items(mine, (approach,), reps)
            kept += k
            dropped += [Exclusion(e.source, e.task, approach, e.item_id, e.rep, e.run_id,
                                  e.reason.replace("all approaches", approach))
                        for e in d]
    return kept, dropped


# ---- Tables ----

def comparison_frame(study_rows: list[dict], main_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    """Main approaches (from the main results) and study approaches, only on items that every
    study approach present and the main results share."""
    study = pd.DataFrame(study_rows)
    present = [a for a in FEEDBACK_STUDY if a in set(study["approach"])]
    shared = set.intersection(*(set(study[study["approach"] == a]["item_id"])
                                for a in present))
    frames, approaches = [study], []
    if main_csv.is_file():
        main = stats.load_runs(main_csv)
        main = main[(main["task"] == TASK) & main["approach"].isin(MAIN)]
        shared &= set(main["item_id"])
        frames.append(main)
        approaches = [a for a in MAIN if a in set(main["approach"])]
    columns = list(stats.REQUIRED_COLUMNS)
    df = pd.concat([f[columns] for f in frames], ignore_index=True)
    return df[df["item_id"].isin(shared)], approaches + present


def check_table(details: dict, run_ids: set[str]) -> dict[str, int]:
    """Counts of the Self-Refine checks: said correct / wrong, against the truth."""
    counts = {"accept_right": 0, "accept_wrong": 0, "reject_right": 0, "reject_wrong": 0}
    for run_id in run_ids:
        for said, really in details[run_id]["checks"]:
            key = ("accept_" if said else "reject_") + ("right" if really else "wrong")
            counts[key] += 1
    return counts


# ---- Figures ----

def _bar_style(approach: str) -> dict:
    if approach in STUDY_COLORS:
        return {"color": "white", "edgecolor": STUDY_COLORS[approach], "hatch": "//",
                "linewidth": 1.8}
    return {"color": COLORS.get(approach, "#9CA3AF")}


def plot_success(rates: pd.DataFrame, approaches: list[str], out_path: Path) -> Path:
    rates = rates.set_index("approach").reindex(approaches)
    with matplotlib.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
        for i, approach in enumerate(approaches):
            row = rates.loc[approach]
            mean, low, high = row["mean"] * 100, row["ci_low"] * 100, row["ci_high"] * 100
            ax.bar(i, mean, 0.65, yerr=[[mean - low], [high - mean]], capsize=4,
                   ecolor="#374151", **_bar_style(approach))
            ax.text(i, high + 1.5, f"{mean:.0f}%", ha="center", va="bottom", fontsize=11)
        ax.set_xticks(range(len(approaches)),
                      [NAMES.get(a, a).replace(" (", "\n(") for a in approaches], fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        n = int(rates["n_items"].min())
        ax.set_title(f"Game of 24: feedback study (hatched) next to the main approaches "
                     f"(n = {n} items)")
        fig.supxlabel("Error bars: 95% bootstrap CI over items", fontsize=10)
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
    return out_path


def plot_cost_accuracy(rates: pd.DataFrame, cost: pd.DataFrame, approaches: list[str],
                       out_path: Path) -> Path:
    rates, cost = rates.set_index("approach"), cost.set_index("approach")
    with matplotlib.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
        for approach in approaches:
            r, tokens = rates.loc[approach], cost.loc[approach, "mean_tokens"]
            mean = r["mean"] * 100
            study = approach in STUDY_COLORS
            color = STUDY_COLORS.get(approach, COLORS.get(approach, "#9CA3AF"))
            ax.errorbar(tokens, mean, yerr=[[mean - r["ci_low"] * 100],
                                            [r["ci_high"] * 100 - mean]],
                        fmt="D" if study else "o", markersize=10, capsize=4, color=color,
                        markerfacecolor="white" if study else color, markeredgewidth=2,
                        label=NAMES.get(approach, approach))
        ax.set_xscale("log")
        ax.set_xlabel("Mean tokens per run (log scale)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title("Game of 24: accuracy against cost (open diamonds = feedback study)")
        fig.legend(loc="outside lower center", ncols=4, frameon=False, fontsize=9)
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
    return out_path


# ---- Report ----

def _pct(row) -> str:
    return f"{row['mean'] * 100:.1f}% [{row['ci_low'] * 100:.1f}, {row['ci_high'] * 100:.1f}]"


def report_markdown(df: pd.DataFrame, approaches: list[str], details: dict,
                    expected: Expected) -> str:
    rates = stats.success_rate(df).set_index("approach")
    cost = stats.cost_summary(df).set_index("approach")
    n_items = df["item_id"].nunique()
    lines = [
        "# Feedback study (separate study, Game of 24)",
        "",
        (f"Generated {datetime.now().astimezone():%Y-%m-%d %H:%M %Z} by "
         "`python -m scripts.update_results`."),
        "",
        ("This study is NOT part of the main results. It was added after the main test runs "
         "and asks how much of the gains come from the exact verifier, and how a paper-like "
         "multi-agent setup compares with ours:"),
        "",
        ("- **Self-Refine (own check):** the Reflective loop (at most 3 answers, critique, "
         "revise), but the model checks its own answer instead of seeing the verifier result. "
         "It stops when its own check says the answer is correct."),
        ("- **Debate:** 3 agents answer, then each sees the other agents' answers and gives "
         "an updated answer (2 rounds); majority vote over the last answers; no verifier "
         "(Du et al., 2023)."),
        ("- **Retry with the verifier:** plain baseline answers until the verifier accepts "
         "one, at most 10; no critique."),
        "",
        (f"Model `{expected.model}`, split `{expected.split}`, {expected.reps} reps, "
         f"{n_items} items. Main approaches come from the main `results/runs.csv`, on the "
         "same items."),
        "",
        "## Accuracy and cost",
        "",
        ("| Approach | Sees the verifier | Accuracy [95% CI] | Mean LLM calls | Mean tokens | "
         "Tokens per correct |"),
        "|---|---|---|---|---|---|",
    ]
    sees = {"baseline": "no", "reflective": "yes", "multi_agent": "yes", "tot": "no",
            "self_refine": "no", "debate": "no", "retry": "yes"}
    for approach in approaches:
        c = cost.loc[approach]
        tpc = "n/a" if pd.isna(c["tokens_per_correct"]) else f"{c['tokens_per_correct']:,.0f}"
        lines.append(f"| {NAMES[approach]} | {sees[approach]} | {_pct(rates.loc[approach])} | "
                     f"{c['mean_calls']:.1f} | {c['mean_tokens']:,.0f} | {tpc} |")

    present = [(a, b) for a, b in PAIRS if a in approaches and b in approaches]
    if present:
        lines += ["", "## Paired differences (percentage points, 95% bootstrap CI over items)",
                  ""]
        for a, b in present:
            d = stats.paired_difference(df, TASK, a, b)
            lines.append(f"- {NAMES[a]} minus {NAMES[b]}: {d['diff']:+.1f} "
                         f"[{d['ci_low']:+.1f}, {d['ci_high']:+.1f}], n = {d['n_items']} items")
        lines += ["", ("Differences below about 15 points are treated as noise at this "
                       "sample size (project rule).")]

    study_ids = set(df[df["approach"] == "self_refine"]["run_id"])
    if study_ids:
        c = check_table(details, study_ids)
        said_ok = c["accept_right"] + c["accept_wrong"]
        said_bad = c["reject_right"] + c["reject_wrong"]
        lines += ["", "## How often the model's own check was right (Self-Refine)", "",
                  "| The model said | Answer really correct | Answer really wrong |",
                  "|---|---|---|",
                  f"| correct ({said_ok} checks) | {c['accept_right']} | {c['accept_wrong']} |",
                  f"| wrong ({said_bad} checks) | {c['reject_right']} | {c['reject_wrong']} |"]
        if said_ok:
            lines += ["", (f"When the model accepted its own answer, the answer was really "
                           f"correct in {100 * c['accept_right'] / said_ok:.0f}% of the "
                           "checks.")]

    retry_ids = set(df[df["approach"] == "retry"]["run_id"])
    if retry_ids:
        tries = [details[r]["attempts"] for r in retry_ids]
        lines += ["", "## Retry with the verifier", "",
                  f"Mean answers per run: {sum(tries) / len(tries):.1f} (at most 10)."]

    lines += ["", "## Figures", ""] + [f"![{name}]({name})" for name in FIGURES]
    lines += ["", "Only runs that passed the checks are included: see `validation.md`."]
    return "\n".join(lines) + "\n"


def validation_markdown(expected: Expected, rows: list[dict], excluded: list[Exclusion],
                        dropped: list[Exclusion], not_used: int) -> str:
    lines = ["# Feedback study validation", "",
             (f"Expected: model `{expected.model}`, split `{expected.split}`, mode "
              f"`benchmark`, {expected.reps} reps, settings from `configs/experiment.toml` ("
              + ", ".join(f"`[approaches.{a}]`" for a in FEEDBACK_STUDY)
              + "). Same run checks as the main results. Each approach keeps only items with "
              "a valid run for every rep."), "",
             f"- Runs used: {len(rows)}"]
    for approach in FEEDBACK_STUDY:
        n = sum(r["approach"] == approach for r in rows)
        lines.append(f"  - {approach}: {n}")
    lines += [f"- Runs left out: {len(excluded)}",
              f"- Items dropped (a rep missing): {len(dropped)}",
              f"- Runs not used by design (dev split or demo): {not_used}", ""]
    if excluded:
        lines += ["| Approach | Item | Rep | Run | Reason |", "|---|---|---|---|---|"]
        lines += [f"| {e.approach} | {e.item_id} | {e.rep} | {e.run_id} | {e.reason} |"
                  for e in excluded]
    for e in dropped:
        lines.append(f"- {e.approach}: dropped {e.item_id}: {e.reason}")
    return "\n".join(lines) + "\n"


def update_feedback_study(runs_dir: Path, out_dir: Path, main_csv: Path, model: str,
                          task: Task, log=print) -> int:
    """Rebuild out_dir from the feedback study runs. Returns the number of runs used; with no
    runs at all nothing is written."""
    expected = expected_setup(model)
    items = task.load_items(SPLIT)
    rows, excluded, not_used, details = collect(runs_dir, task, items, expected)
    if not rows and not excluded:
        return 0
    rows, duplicates = drop_duplicates(rows)
    rows, dropped = complete_per_approach(rows, expected.reps)
    excluded += duplicates
    rows.sort(key=lambda r: (r["approach"], r["rep"], r["item_id"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation.md").write_text(
        validation_markdown(expected, rows, excluded, dropped, not_used),
        encoding="utf-8", newline="\n")
    if not rows:
        log(f"{out_dir}: no valid feedback study runs; report not changed")
        return 0

    columns = list(stats.REQUIRED_COLUMNS)
    write_rows(out_dir / "runs.csv", [{c: r[c] for c in columns} for r in rows])
    df, approaches = comparison_frame(rows, main_csv)
    if df.empty:
        log(f"{out_dir}: no items shared by all study approaches yet; report not changed")
        return len(rows)
    rates, cost = stats.success_rate(df), stats.cost_summary(df)
    plot_success(rates, approaches, out_dir / FIGURES[0])
    plot_cost_accuracy(rates, cost, approaches, out_dir / FIGURES[1])
    (out_dir / "report.md").write_text(report_markdown(df, approaches, details, expected),
                                       encoding="utf-8", newline="\n")
    log(f"{out_dir}: {len(rows)} feedback study runs, {len(excluded)} left out")
    return len(rows)
