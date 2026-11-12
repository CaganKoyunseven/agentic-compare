"""Separate study: self-consistency on Game of 24, compared with Baseline and Tree of Thoughts.

Self-consistency runs never enter the main results. This module validates them on their own
(same checks as the main results), then writes into results/self_consistency/:
  runs.csv       one row per valid run (contract 6.3 columns)
  samples.csv    per run: majority-vote verdict and best-of-k verdict for k = 1, 5, 10, 20, 29
  validation.md  what was left out and why
  sc_success.png, sc_k_curve.png, report.md

Everything per k comes from the same recorded samples (the first k answers of each run); no
new model calls. Best-of-k uses the official check on every sample, so it is an oracle upper
bound, not an approach: no approach may see the official check.
"""

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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
from agentic.approaches.self_consistency import vote
from agentic.config import load_toml
from agentic.events import LLMCall, RunStart
from agentic.runner import write_rows
from agentic.tasks.base import Item, Task
from agentic.traces import group_runs, read_trace

SC = "self_consistency"
ORACLE = "sc_oracle"
TASK = "game24"
SPLIT = "test"
KS = (1, 5, 10, 20, 29)
SC_COLOR = "#7C3AED"        # checked against the ToT green for colour-blind separation
NAMES = {"baseline": "Baseline", "tot": "Tree-of-Thoughts", SC: "Self-consistency (vote)",
         ORACLE: "Best-of-29 (oracle)"}
FIGURES = ("sc_success.png", "sc_k_curve.png")


# ---- Per-run sample analysis ----

def sample_answers(run: Run, task: Task, item: Item) -> list[str]:
    """The extracted answer of every sample of one run, in call order."""
    return [task.extract_answer(item, body.response) for _, body in run
            if isinstance(body, LLMCall) and body.role == "sampler"]


def sample_verdicts(answers: list[str], task: Task, item: Item,
                    ks: Iterable[int] = KS) -> dict[str, int]:
    """For each k: majority vote over the first k answers is correct (maj_k), and at least
    one of the first k answers is correct (any_k, oracle)."""
    correct = [task.score(item, a).correct for a in answers]
    row = {}
    for k in ks:
        winner = vote(answers[:k])
        row[f"maj_{k}"] = int(winner is not None and correct[winner])
        row[f"any_{k}"] = int(any(correct[:k]))
    return row


# ---- Collect and validate ----

def expected_setup(model: str) -> Expected:
    settings = load_toml("experiment")
    config = {"llm": settings["llm"], "approach": settings.get("approaches", {}).get(SC, {})}
    return Expected(model=model, split=SPLIT, base_seed=settings["seed"], reps=settings["reps"],
                    approaches=(SC,), configs={SC: config})


def collect(runs_dir: Path, task: Task, items: list[Item], expected: Expected):
    """Valid rows (with samples columns), excluded runs, and how many runs were not used."""
    by_id = {item.item_id: item for item in items}
    rows, excluded, not_used = [], [], 0
    for trace in sorted(runs_dir.glob("*/trace.jsonl")):
        runs = [r for r in group_runs(read_trace(trace)).values() if r[0][0].approach == SC]
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
            run, item = records[row["run_id"]], by_id[row["item_id"]]
            answers = sample_answers(run, task, item)
            samples = expected.configs[SC]["approach"].get("samples", max(KS))
            winner = vote(answers)
            final = run[-1][1].answer
            reason = ""
            if len(answers) != samples:
                reason = f"{len(answers)} samples, expected {samples}"
            elif ("" if winner is None else answers[winner]) != final:
                reason = "the recomputed majority vote differs from the final answer"
            if reason:
                excluded.append(Exclusion(row["source"], row["task"], SC, row["item_id"],
                                          row["rep"], row["run_id"], reason))
                continue
            rows.append(row | sample_verdicts(answers, task, item))
    return rows, excluded, not_used


# ---- Tables ----

def comparison_frame(sc_rows: list[dict], main_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    """Baseline and ToT rows of the main results plus self-consistency and oracle rows, only
    on items present in both. Returns the frame and the approaches it holds."""
    sc = pd.DataFrame(sc_rows)
    frames = [sc.assign(approach=SC), sc.assign(approach=ORACLE, correct=sc[f"any_{max(KS)}"])]
    approaches = [SC, ORACLE]
    if main_csv.is_file():
        main = stats.load_runs(main_csv)
        main = main[(main["task"] == TASK) & main["approach"].isin(["baseline", "tot"])]
        shared = set(main["item_id"]) & set(sc["item_id"])
        if shared:      # no shared items (main results not rebuilt yet): SC alone
            main = main[main["item_id"].isin(shared)]
            frames = [f[f["item_id"].isin(shared)] for f in frames] + [main]
            approaches = [a for a in ("baseline", "tot")
                          if a in set(main["approach"])] + approaches
    columns = list(stats.REQUIRED_COLUMNS)
    return pd.concat([f[columns] for f in frames], ignore_index=True), approaches


def k_frame(df_sc: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (run, k, kind) with correct, for success_rate per k."""
    parts = []
    for k in KS:
        for kind in ("maj", "any"):
            parts.append(df_sc.assign(approach=f"{kind}_{k}", correct=df_sc[f"{kind}_{k}"]))
    return pd.concat(parts, ignore_index=True)


# ---- Figures ----

def _bar_style(approach: str) -> dict:
    if approach == ORACLE:
        return {"color": "white", "edgecolor": SC_COLOR, "hatch": "//", "linewidth": 1.5}
    return {"color": SC_COLOR if approach == SC else COLORS.get(approach, "#9CA3AF")}


def plot_sc_success(rates: pd.DataFrame, approaches: list[str], out_path: Path) -> Path:
    with matplotlib.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
        rates = rates.set_index("approach").reindex(approaches)
        x = np.arange(len(approaches))
        for i, approach in enumerate(approaches):
            row = rates.loc[approach]
            mean, low, high = row["mean"] * 100, row["ci_low"] * 100, row["ci_high"] * 100
            ax.bar(i, mean, 0.6, yerr=[[mean - low], [high - mean]], capsize=4,
                   ecolor="#374151", **_bar_style(approach))
            ax.text(i, high + 1.5, f"{mean:.0f}%", ha="center", va="bottom", fontsize=12)
        ax.set_xticks(x, [NAMES.get(a, a) for a in approaches])
        ax.set_ylim(0, 110)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        n = int(rates["n_items"].min())
        ax.set_title(f"Game of 24: self-consistency vs Baseline and ToT (n = {n} items)")
        fig.supxlabel("Error bars: 95% bootstrap CI over items. Oracle = best of 29 samples "
                      "by the official check (upper bound, not an approach)", fontsize=10)
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
    return out_path


def plot_k_curve(k_rates: pd.DataFrame, refs: dict[str, float], out_path: Path) -> Path:
    with matplotlib.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
        rates = k_rates.set_index("approach")
        rows = rates.loc[[f"maj_{k}" for k in KS]]
        ax.fill_between(KS, rows["ci_low"] * 100, rows["ci_high"] * 100, color=SC_COLOR,
                        alpha=0.15, linewidth=0, label="95% CI (majority vote)")
        ax.plot(KS, rows["mean"] * 100, color=SC_COLOR, linewidth=2, marker="o",
                markersize=8, label="Self-consistency (majority vote)")
        oracle = rates.loc[[f"any_{k}" for k in KS], "mean"] * 100
        ax.plot(KS, oracle, color=SC_COLOR, linewidth=2, linestyle="--", marker="o",
                markersize=8, markerfacecolor="white", label="Best-of-k (oracle, upper bound)")
        for approach, value in refs.items():
            ax.axhline(value * 100, color=COLORS[approach], linewidth=2,
                       linestyle=":" if approach == "baseline" else "-.",
                       label=f"{NAMES[approach]} ({value * 100:.0f}%)")
        ax.set_xticks(KS)
        ax.set_xlabel("Samples per puzzle, k (first k of the same recorded samples)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(-3, 103)       # reference lines at 0% or 100% stay visible
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title("Game of 24: accuracy by number of samples per puzzle")
        fig.legend(loc="outside lower center", ncols=3, frameon=False, fontsize=11)
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)
    return out_path


# ---- Report ----

def _pct(row) -> str:
    return f"{row['mean'] * 100:.1f}% [{row['ci_low'] * 100:.1f}, {row['ci_high'] * 100:.1f}]"


def report_markdown(df: pd.DataFrame, approaches: list[str], k_rates: pd.DataFrame,
                    sc_rows: list[dict], expected: Expected) -> str:
    rates = stats.success_rate(df).set_index("approach")
    real = df[df["approach"] != ORACLE]
    cost = stats.cost_summary(real).set_index("approach")
    n_items = df[df["approach"] == SC]["item_id"].nunique()
    samples = expected.configs[SC]["approach"].get("samples", max(KS))
    lines = [
        "# Self-consistency (separate study, Game of 24)",
        "",
        (f"Generated {datetime.now().astimezone():%Y-%m-%d %H:%M %Z} by "
         "`python -m scripts.update_results`."),
        "",
        ("This study is NOT part of the main results. It asks one question: does ToT's gain "
         "come from its search, or just from making many calls? Self-consistency makes "
         f"{samples} calls, each one the plain baseline prompt with its own seed, and answers "
         "by majority vote over the answers (spaces, extra parentheses and the order of + and "
         "* operands ignored; ties go to the answer seen first). It uses no verifier and no "
         "search. The number of calls matches our first, lighter ToT, not the paper-like ToT "
         "of the main results, so compare cost by tokens per correct answer."),
        "",
        (f"Model `{expected.model}`, split `{expected.split}`, {expected.reps} reps, "
         f"{n_items} items shared with the main results. Baseline and ToT rows come from the "
         "main `results/runs.csv`."),
        "",
        "## Accuracy",
        "",
        "| Approach | Accuracy [95% CI] | Mean LLM calls | Mean tokens | Tokens per correct |",
        "|---|---|---|---|---|",
    ]
    for approach in approaches:
        if approach == ORACLE:
            lines.append(f"| {NAMES[ORACLE]} | {_pct(rates.loc[ORACLE])} | (same samples) | "
                         "(same samples) | n/a |")
            continue
        c = cost.loc[approach]
        tpc = "n/a" if pd.isna(c["tokens_per_correct"]) else f"{c['tokens_per_correct']:,.0f}"
        lines.append(f"| {NAMES[approach]} | {_pct(rates.loc[approach])} | "
                     f"{c['mean_calls']:.1f} | {c['mean_tokens']:,.0f} | {tpc} |")
    lines += ["", ("The oracle row checks every sample with the official check and counts a "
                   "run as correct when any sample is right. No approach can do this; it only "
                   "shows how often a right answer was among the samples at all."), ""]

    pairs = [(SC, "baseline"), ("tot", SC)]
    present = [(a, b) for a, b in pairs if a in approaches and b in approaches]
    if present:
        lines += ["## Paired differences (percentage points, 95% bootstrap CI over items)", ""]
        for a, b in present:
            d = stats.paired_difference(df, TASK, a, b)
            lines.append(f"- {NAMES[a]} minus {NAMES[b]}: {d['diff']:+.1f} "
                         f"[{d['ci_low']:+.1f}, {d['ci_high']:+.1f}], n = {d['n_items']} items")
        lines += ["", ("Differences below about 15 points are treated as noise at this "
                       "sample size (project rule)."), ""]
    else:
        lines += ["No main results found: comparisons with Baseline and ToT are skipped.", ""]

    k = k_rates.set_index("approach")
    lines += ["## Accuracy by number of samples", "",
              "Computed from the first k samples of the same runs (no new runs).", "",
              "| k | Majority vote [95% CI] | Best-of-k oracle [95% CI] |", "|---|---|---|"]
    lines += [f"| {n} | {_pct(k.loc[f'maj_{n}'])} | {_pct(k.loc[f'any_{n}'])} |" for n in KS]
    lines += ["", ("k = 1 is the first sample alone: the baseline prompt with the same seed "
                   "as the baseline run, so it should be close to the baseline accuracy."), "",
              "## Figures", ""]
    lines += [f"![{name}]({name})" for name in FIGURES]
    lines += ["", (f"Only runs that passed the checks are included: see `validation.md` "
                   f"({len(sc_rows)} runs used).")]
    return "\n".join(lines) + "\n"


def validation_markdown(expected: Expected, rows: list[dict], excluded: list[Exclusion],
                        dropped: list[Exclusion], not_used: int) -> str:
    lines = ["# Self-consistency validation", "",
             (f"Expected: model `{expected.model}`, split `{expected.split}`, mode `benchmark`, "
              f"{expected.reps} reps, settings from `configs/experiment.toml` "
              "`[approaches.self_consistency]`. Same run checks as the main results, plus: the "
              "number of samples is as configured and a recomputed majority vote gives the "
              "recorded final answer."), "",
             f"- Runs used: {len(rows)}",
             f"- Items: {len({r['item_id'] for r in rows})}",
             f"- Runs left out: {len(excluded)}",
             f"- Items dropped (a rep missing): {len(dropped)}",
             f"- Runs not used by design (dev split or demo): {not_used}", ""]
    if excluded:
        lines += ["| Item | Rep | Run | Reason |", "|---|---|---|---|"]
        lines += [f"| {e.item_id} | {e.rep} | {e.run_id} | {e.reason} |" for e in excluded]
    for e in dropped:
        lines.append(f"- dropped {e.item_id}: {e.reason}")
    return "\n".join(lines) + "\n"


def update_self_consistency(runs_dir: Path, out_dir: Path, main_csv: Path, model: str,
                            task: Task, log=print) -> int:
    """Rebuild out_dir from the self-consistency runs. Returns the number of runs used; with
    no runs at all nothing is written."""
    expected = expected_setup(model)
    items = task.load_items(SPLIT)
    rows, excluded, not_used = collect(runs_dir, task, items, expected)
    if not rows and not excluded:
        return 0
    rows, duplicates = drop_duplicates(rows)
    rows, dropped = keep_complete_items(rows, (SC,), expected.reps)
    excluded += duplicates
    rows.sort(key=lambda r: (r["rep"], r["item_id"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation.md").write_text(
        validation_markdown(expected, rows, excluded, dropped, not_used),
        encoding="utf-8", newline="\n")
    if not rows:
        log(f"{out_dir}: no valid self-consistency runs; report not changed")
        return 0

    columns = list(stats.REQUIRED_COLUMNS)
    write_rows(out_dir / "runs.csv", [{c: r[c] for c in columns} for r in rows])
    sample_cols = ["run_id", "item_id", "difficulty", "rep"] + [
        f"{kind}_{k}" for k in KS for kind in ("maj", "any")]
    pd.DataFrame(rows)[sample_cols].to_csv(out_dir / "samples.csv", index=False,
                                           lineterminator="\n")

    df, approaches = comparison_frame(rows, main_csv)
    sc_df = pd.DataFrame(rows)
    sc_df = sc_df[sc_df["item_id"].isin(set(df["item_id"]))]
    k_rates = stats.success_rate(k_frame(sc_df))
    rates = stats.success_rate(df)
    plot_sc_success(rates, approaches, out_dir / FIGURES[0])
    refs = {a: float(rates.set_index("approach").loc[a, "mean"])
            for a in ("baseline", "tot") if a in approaches}
    plot_k_curve(k_rates, refs, out_dir / FIGURES[1])
    (out_dir / "report.md").write_text(report_markdown(df, approaches, k_rates, rows, expected),
                                       encoding="utf-8", newline="\n")
    log(f"{out_dir}: {len(rows)} self-consistency runs, {len(excluded)} left out")
    return len(rows)
