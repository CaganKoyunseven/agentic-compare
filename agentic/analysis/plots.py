"""Charts built from the results table.

Every figure is a 16:9 PNG (10 x 5.6 inches, 200 dpi) rendered off-screen with matplotlib's
Agg backend, so no display is needed. All visible text lives in ``LABELS``.

Usage:
    python -m agentic.analysis.plots --input <csv> --out <dir> [--machine <machine_id>]
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from agentic.analysis.stats import (
    cost_summary,
    cumulative_accuracy,
    default_machine,
    load_runs,
    paired_counts,
    paired_difference,
    rep_spread,
    success_rate,
)

APPROACHES = ("baseline", "reflective", "multi_agent", "tot")
BASELINE = "baseline"

COLORS = {
    "baseline": "#6B7280",
    "reflective": "#2563EB",
    "multi_agent": "#D97706",
    "tot": "#059669",
}

# Segment colors of the paired chart, chosen to differ from the approach colors.
PAIRED_COLORS = {
    "only_a": "#CC79A7",
    "only_b": "#6B7280",
    "both": "#56B4E9",
    "neither": "#E5E7EB",
}
PAIRED_TEXT_COLORS = {"only_a": "black", "only_b": "white", "both": "black", "neither": "black"}

TASK_MARKERS = ("o", "s", "^", "D")
REP_MARKERS = ("o", "s", "^", "D", "v", "P")

FIGSIZE = (10, 5.6)
DPI = 200
STYLE = {
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 15,
}

LABELS = {
    "approach": {
        "baseline": "Baseline",
        "reflective": "Reflective",
        "multi_agent": "Multi-agent",
        "tot": "Tree-of-Thoughts",
    },
    "task": {"game24": "Game of 24", "humaneval": "HumanEval"},
    "panel": "{task} (n = {n} items)",
    "tick_task": "{task}\n(n = {n} items)",
    "accuracy_pct": "Accuracy (%)",
    "ci_note": "Error bars: 95% bootstrap CI over items",
    "success_title": "Success rate by task and approach",
    "cumulative_title": "Cumulative accuracy (line: mean, band: 95% over {n_perm} item orders)",
    "items_seen": "Items seen",
    "rep_title": "Accuracy of each repetition",
    "rep_label": "Rep {rep}",
    "paired_title": "Items solved compared with Baseline (solved = correct in at least half the reps)",
    "only_a": "Only this approach",
    "only_b": "Only Baseline",
    "both": "Both",
    "neither": "Neither",
    "items": "Items",
    "paired_note": "n = {n}  |  accuracy diff {diff:+.0f} pp, 95% CI [{low:+.0f}, {high:+.0f}]",
    "cost_title": "Cost vs accuracy (n = {n} items per point)",
    "tokens_axis": "Mean tokens per run (log scale)",
    "time_title": "Time and LLM calls on one machine (n = {n} runs)",
    "wall_time": "Mean wall time per run (s)",
    "llm_calls": "Mean LLM calls per run",
    "saved": "Saved {path}",
    "cli_description": "Draw all result charts from a results CSV.",
    "cli_input": "results CSV (runs table)",
    "cli_out": "output folder for the PNG files",
    "cli_machine": "machine_id for the time chart (default: the one with the most rows)",
}


def _approach_name(approach: str) -> str:
    return LABELS["approach"].get(approach, approach)


def _task_name(task: str) -> str:
    return LABELS["task"].get(task, task)


def _n_text(values: Sequence[int] | pd.Series) -> str:
    """'30' when all counts match, '28-30' otherwise."""
    low, high = int(min(values)), int(max(values))
    return str(low) if low == high else f"{low}-{high}"


def _approaches_in(df: pd.DataFrame) -> list[str]:
    present = set(df["approach"])
    known = [a for a in APPROACHES if a in present]
    return known + sorted(present - set(APPROACHES))


def _tasks_in(df: pd.DataFrame) -> list[str]:
    return sorted(set(df["task"]))


def _styled(func: Callable[..., Path]) -> Callable[..., Path]:
    """Run a plot function with the shared font sizes and make sure the output folder exists."""

    @functools.wraps(func)
    def wrapper(df: pd.DataFrame, out_path: str | Path, *args, **kwargs) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with matplotlib.rc_context(STYLE):
            return func(df, out_path, *args, **kwargs)

    return wrapper


def _subplots(ncols: int = 1, nrows: int = 1, sharex: bool = False, sharey: bool = False):
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=FIGSIZE,
        dpi=DPI,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
        layout="constrained",
    )
    return fig, list(axes.flat)


def _save(fig, out_path: Path) -> Path:
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


def _grouped_bars(
    ax,
    table: pd.DataFrame,
    value: str,
    tasks: list[str],
    approaches: list[str],
    scale: float = 1.0,
    ci: tuple[str, str] | None = None,
) -> None:
    """One group of bars per task, one bar per approach, optional CI error bars."""
    width = 0.8 / len(approaches)
    centers = np.arange(len(tasks))
    for i, approach in enumerate(approaches):
        sub = table[table["approach"] == approach].set_index("task").reindex(tasks)
        values = sub[value].to_numpy(dtype=float) * scale
        yerr = None
        if ci is not None:
            low = sub[ci[0]].to_numpy(dtype=float) * scale
            high = sub[ci[1]].to_numpy(dtype=float) * scale
            yerr = np.nan_to_num(np.vstack([values - low, high - values]))
        ax.bar(
            centers + (i - (len(approaches) - 1) / 2) * width,
            np.nan_to_num(values),
            width,
            color=COLORS.get(approach, "#9CA3AF"),
            label=_approach_name(approach),
            yerr=yerr,
            capsize=4,
        )
    ax.set_xticks(centers)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


@_styled
def plot_success(df: pd.DataFrame, out_path: Path) -> Path:
    """Grouped bars of accuracy per task, with 95% bootstrap CIs."""
    rates = success_rate(df)
    tasks, approaches = _tasks_in(df), _approaches_in(df)
    fig, (ax,) = _subplots()
    _grouped_bars(ax, rates, "mean", tasks, approaches, scale=100, ci=("ci_low", "ci_high"))
    n_by_task = rates.groupby("task")["n_items"]
    ax.set_xticklabels(
        [
            LABELS["tick_task"].format(task=_task_name(t), n=_n_text(n_by_task.get_group(t)))
            for t in tasks
        ]
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel(LABELS["accuracy_pct"])
    ax.set_title(LABELS["success_title"])
    ax.legend(loc="upper left", ncols=len(approaches), frameon=True)
    fig.supxlabel(LABELS["ci_note"], fontsize=11)
    return _save(fig, out_path)


@_styled
def plot_cumulative(df: pd.DataFrame, out_path: Path, n_perm: int = 200) -> Path:
    """One panel per task: running accuracy over shuffled item orders, with a 95% band."""
    tasks, approaches = _tasks_in(df), _approaches_in(df)
    fig, axes = _subplots(len(tasks), sharey=True)
    for ax, task in zip(axes, tasks, strict=True):
        n_items = []
        for approach in approaches:
            if not ((df["task"] == task) & (df["approach"] == approach)).any():
                continue
            curve = cumulative_accuracy(df, task, approach, n_perm=n_perm)
            color = COLORS.get(approach, "#9CA3AF")
            ax.plot(
                curve["n"],
                curve["mean"] * 100,
                color=color,
                linewidth=2.2,
                label=_approach_name(approach),
            )
            ax.fill_between(
                curve["n"], curve["low"] * 100, curve["high"] * 100, color=color, alpha=0.15
            )
            n_items.append(len(curve))
        ax.set_title(LABELS["panel"].format(task=_task_name(task), n=_n_text(n_items)))
        ax.set_xlabel(LABELS["items_seen"])
        ax.set_ylim(0, 100)
        ax.grid(linestyle="--", alpha=0.6)
    axes[0].set_ylabel(LABELS["accuracy_pct"])
    axes[0].legend(loc="upper right", frameon=True)
    fig.suptitle(LABELS["cumulative_title"].format(n_perm=n_perm))
    return _save(fig, out_path)


@_styled
def plot_rep_spread(df: pd.DataFrame, out_path: Path) -> Path:
    """One panel per task: accuracy of each repetition per approach, range shown as a line."""
    spread = rep_spread(df)
    tasks, approaches = _tasks_in(df), _approaches_in(df)
    reps = sorted(spread["rep"].unique())
    fig, axes = _subplots(len(tasks), sharey=True)
    for ax, task in zip(axes, tasks, strict=True):
        sub = spread[spread["task"] == task]
        for x, approach in enumerate(approaches):
            points = sub[sub["approach"] == approach]
            if points.empty:
                continue
            color = COLORS.get(approach, "#9CA3AF")
            acc = points["accuracy"].to_numpy() * 100
            ax.vlines(x, acc.min(), acc.max(), color=color, linewidth=3, alpha=0.5)
            for rep, value in zip(points["rep"], acc, strict=True):
                marker = REP_MARKERS[reps.index(rep) % len(REP_MARKERS)]
                ax.plot(x, value, marker=marker, color=color, markersize=9, linestyle="none")
        n_items = df[df["task"] == task].groupby("approach")["item_id"].nunique()
        ax.set_title(LABELS["panel"].format(task=_task_name(task), n=_n_text(n_items)))
        ax.set_xticks(range(len(approaches)))
        ax.set_xticklabels([_approach_name(a) for a in approaches], rotation=15)
        ax.set_xlim(-0.5, len(approaches) - 0.5)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].set_ylabel(LABELS["accuracy_pct"])
    handles = [
        Line2D(
            [],
            [],
            marker=REP_MARKERS[i % len(REP_MARKERS)],
            color="#374151",
            linestyle="none",
            markersize=9,
            label=LABELS["rep_label"].format(rep=rep),
        )
        for i, rep in enumerate(reps)
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=True)
    fig.suptitle(LABELS["rep_title"])
    return _save(fig, out_path)


@_styled
def plot_paired(df: pd.DataFrame, out_path: Path) -> Path:
    """One panel per task: stacked bars of solved items, each approach versus Baseline."""
    if BASELINE not in set(df["approach"]):
        raise ValueError("plot_paired needs runs of the baseline approach")
    tasks = _tasks_in(df)
    others = [a for a in _approaches_in(df) if a != BASELINE]
    segments = ("only_a", "only_b", "both", "neither")
    results = {}
    for task in tasks:
        for approach in others:
            try:
                results[task, approach] = (
                    paired_counts(df, task, approach, BASELINE),
                    paired_difference(df, task, approach, BASELINE),
                )
            except ValueError:
                continue  # approach or baseline has no runs on this task
    max_total = max((sum(c.values()) for c, _ in results.values()), default=1)
    fig, axes = _subplots(nrows=len(tasks), sharex=True)
    for ax, task in zip(axes, tasks, strict=True):
        totals = []
        for row, approach in enumerate(others):
            if (task, approach) not in results:
                continue
            counts, diff = results[task, approach]
            left = 0
            for key in segments:
                value = counts[key]
                ax.barh(row, value, left=left, color=PAIRED_COLORS[key], edgecolor="white")
                # Only label segments wide enough to hold the number.
                if value >= 0.04 * max_total:
                    ax.text(
                        left + value / 2,
                        row,
                        str(value),
                        ha="center",
                        va="center",
                        color=PAIRED_TEXT_COLORS[key],
                        fontsize=12,
                    )
                left += value
            totals.append(left)
            ax.text(
                left + 0.015 * max_total,
                row,
                LABELS["paired_note"].format(
                    n=left, diff=diff["diff"], low=diff["ci_low"], high=diff["ci_high"]
                ),
                va="center",
                fontsize=12,
            )
        ax.set_yticks(range(len(others)))
        ax.set_yticklabels([_approach_name(a) for a in others])
        ax.set_ylim(len(others) - 0.5, -0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(LABELS["panel"].format(task=_task_name(task), n=_n_text(totals or [0])))
    # Leave room on the right for the notes, but only put ticks under the bars.
    ticks = MaxNLocator(nbins=5, integer=True).tick_values(0, max_total)
    axes[-1].set_xticks([t for t in ticks if 0 <= t <= max_total])
    axes[-1].set_xlim(0, max_total * 2.0)
    axes[-1].set_xlabel(LABELS["items"])
    handles = [Patch(color=PAIRED_COLORS[key], label=LABELS[key]) for key in segments]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(segments), frameon=False)
    fig.suptitle(LABELS["paired_title"])
    return _save(fig, out_path)


@_styled
def plot_cost_vs_accuracy(df: pd.DataFrame, out_path: Path) -> Path:
    """Mean tokens per run (log x) against accuracy, one point per approach and task."""
    merged = cost_summary(df).merge(success_rate(df), on=["task", "approach"])
    tasks, approaches = _tasks_in(df), _approaches_in(df)
    fig, (ax,) = _subplots()
    for t_index, task in enumerate(tasks):
        marker = TASK_MARKERS[t_index % len(TASK_MARKERS)]
        for approach in approaches:
            row = merged[(merged["task"] == task) & (merged["approach"] == approach)]
            if row.empty:
                continue
            row = row.iloc[0]
            mean = row["mean"] * 100
            ax.errorbar(
                row["mean_tokens"],
                mean,
                yerr=[[mean - row["ci_low"] * 100], [row["ci_high"] * 100 - mean]],
                marker=marker,
                markersize=11,
                color=COLORS.get(approach, "#9CA3AF"),
                capsize=4,
                linestyle="none",
            )
    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.set_xlabel(LABELS["tokens_axis"])
    ax.set_ylabel(LABELS["accuracy_pct"])
    ax.set_title(LABELS["cost_title"].format(n=_n_text(merged["n_items"])))
    ax.grid(which="both", linestyle="--", alpha=0.5)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=COLORS.get(a, "#9CA3AF"),
            linestyle="none",
            markersize=10,
            label=_approach_name(a),
        )
        for a in approaches
    ] + [
        Line2D(
            [],
            [],
            marker=TASK_MARKERS[i % len(TASK_MARKERS)],
            color="#374151",
            linestyle="none",
            markersize=10,
            label=_task_name(t),
        )
        for i, t in enumerate(tasks)
    ]
    ax.legend(handles=handles, loc="upper left", ncols=2, frameon=True)
    fig.supxlabel(LABELS["ci_note"], fontsize=11)
    return _save(fig, out_path)


@_styled
def plot_time_and_calls(df: pd.DataFrame, out_path: Path, machine_id: str | None = None) -> Path:
    """Mean wall time and mean LLM calls, using the rows of ONE machine only.

    Wall time is only comparable within a machine. Default: the machine with the most rows.
    """
    if machine_id is None:
        machine_id = default_machine(df)
    data = df[df["machine_id"].astype(str) == str(machine_id)]
    if data.empty:
        raise ValueError(f"No rows for machine_id {machine_id!r}")
    costs = cost_summary(data, machine_id=machine_id)
    costs["mean_wall_s"] = costs["mean_wall_ms"] / 1000.0
    tasks, approaches = _tasks_in(data), _approaches_in(data)
    n_items = data.groupby(["task", "approach"])["item_id"].nunique()
    fig, (ax_time, ax_calls) = _subplots(2)
    for ax, column, label in (
        (ax_time, "mean_wall_s", LABELS["wall_time"]),
        (ax_calls, "mean_calls", LABELS["llm_calls"]),
    ):
        _grouped_bars(ax, costs, column, tasks, approaches)
        ax.set_xticklabels(
            [
                LABELS["tick_task"].format(task=_task_name(t), n=_n_text(n_items.loc[t]))
                for t in tasks
            ]
        )
        ax.set_ylabel(label)
    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(approaches), frameon=False)
    fig.suptitle(LABELS["time_title"].format(n=len(data)))
    return _save(fig, out_path)


def plot_all(df: pd.DataFrame, out_dir: str | Path, machine_id: str | None = None) -> list[Path]:
    """Write all six figures into ``out_dir`` and return their paths."""
    out_dir = Path(out_dir)
    return [
        plot_success(df, out_dir / "success.png"),
        plot_cumulative(df, out_dir / "cumulative.png"),
        plot_rep_spread(df, out_dir / "rep_spread.png"),
        plot_paired(df, out_dir / "paired.png"),
        plot_cost_vs_accuracy(df, out_dir / "cost_vs_accuracy.png"),
        plot_time_and_calls(df, out_dir / "time_and_calls.png", machine_id=machine_id),
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=LABELS["cli_description"])
    parser.add_argument("--input", required=True, type=Path, help=LABELS["cli_input"])
    parser.add_argument("--out", required=True, type=Path, help=LABELS["cli_out"])
    parser.add_argument("--machine", default=None, help=LABELS["cli_machine"])
    args = parser.parse_args(argv)
    df = load_runs(args.input)
    for path in plot_all(df, args.out, machine_id=args.machine):
        print(LABELS["saved"].format(path=path))


if __name__ == "__main__":
    main()
