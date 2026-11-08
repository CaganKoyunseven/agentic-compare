"""Turn a results table into the figures and a Markdown summary for the slides.

Run: python -m scripts.make_report --input <runs.csv> [--out <dir>] [--machine <machine_id>]

Every number in the report comes from agentic.analysis.stats; this script only formats them.
The default output folder is report/ next to the input file, so a fake table never writes into
results/.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from agentic.analysis import stats
from agentic.analysis.plots import APPROACHES, LABELS, plot_all

BASELINE = "baseline"
NOISE_POINTS = 15.0
FAKE_MODEL = "fake-model"
FAKE_MACHINE = "fake"
FAKE_WARNING = "**FAKE DATA, NOT A RESULT**"


def approach_name(approach: str) -> str:
    return LABELS["approach"].get(approach, approach)


def task_name(task: str) -> str:
    return LABELS["task"].get(task, task)


def ordered_approaches(values) -> list[str]:
    present = set(values)
    return [a for a in APPROACHES if a in present] + sorted(present - set(APPROACHES))


def is_fake(df: pd.DataFrame) -> bool:
    return bool((df["model"].astype(str) == FAKE_MODEL).any()
                or (df["machine_id"].astype(str) == FAKE_MACHINE).any())


def signed(value: float) -> str:
    """'+12.0' or '-3.5'; a value that rounds to zero is '+0.0', never '-0.0'."""
    value = round(float(value), 1)
    return f"{value + 0.0:+.1f}"


def pct(fraction: float) -> str:
    return f"{fraction * 100:.1f}"


def maybe(value: float, fmt: str) -> str:
    """Formatted value, or 'n/a' when it is missing (NaN)."""
    return "n/a" if pd.isna(value) else fmt.format(value)


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def paired_sentence(approach: str, diff: dict) -> str:
    sentence = (f"{approach_name(approach)} minus {approach_name(BASELINE)} = "
                f"{signed(diff['diff'])} points (95% CI [{signed(diff['ci_low'])}, "
                f"{signed(diff['ci_high'])}]), n = {diff['n_items']} items")
    if abs(diff["diff"]) < NOISE_POINTS:
        sentence += f" (within noise, below {NOISE_POINTS:.0f} points)"
    return sentence


# ---- Sections ----

def header_section(df: pd.DataFrame, input_name: str, machine: str) -> list[str]:
    lines = [FAKE_WARNING, ""] if is_fake(df) else []
    lines += ["# Results report", ""]
    items = df.groupby("task")["item_id"].nunique()
    reps = df.groupby("task")["rep"].nunique()
    lines += [
        f"- Input file: `{input_name}`",
        f"- Model: {', '.join(sorted(set(df['model'].astype(str))))}",
        f"- Machine used for time: `{machine}`",
        f"- Split: {', '.join(sorted(set(df['split'].astype(str))))}",
        "- Items per task: " + ", ".join(f"{task_name(t)} {n}" for t, n in items.items()),
        "- Repetitions per item: " + ", ".join(f"{task_name(t)} {n}" for t, n in reps.items()),
        "",
    ]
    return lines


def success_section(df: pd.DataFrame) -> list[str]:
    lines = ["## Success rate", "",
             "Mean accuracy over all repetitions; 95% bootstrap CI over items.", ""]
    rates = stats.success_rate(df)
    for task in sorted(set(rates["task"])):
        part = rates[rates["task"] == task].set_index("approach")
        rows = []
        for approach in ordered_approaches(part.index):
            r = part.loc[approach]
            rows.append([approach_name(approach), f"{pct(r['mean'])}%",
                         f"[{pct(r['ci_low'])}, {pct(r['ci_high'])}]", str(int(r["n_items"]))])
        lines += [f"### {task_name(task)}", ""]
        lines += table(["Approach", "Success", "95% CI", "n items"], rows) + [""]
    return lines


def paired_section(df: pd.DataFrame) -> list[str]:
    lines = ["## Paired comparison with Baseline", "",
             ("Accuracy difference in percentage points over the items both approaches ran; "
              "95% paired bootstrap CI. Solved = correct in at least half the repetitions."), ""]
    for task in sorted(set(df["task"])):
        lines += [f"### {task_name(task)}", ""]
        approaches = ordered_approaches(df.loc[df["task"] == task, "approach"])
        others = [a for a in approaches if a != BASELINE]
        if BASELINE not in approaches or not others:
            lines += ["No paired comparison: needs Baseline and at least one other approach.", ""]
            continue
        for approach in others:
            diff = stats.paired_difference(df, task, approach, BASELINE)
            counts = stats.paired_counts(df, task, approach, BASELINE)
            lines.append(f"- {paired_sentence(approach, diff)}")
            lines.append(f"  - Items solved: only {approach_name(approach)} {counts['only_a']}, "
                         f"only {approach_name(BASELINE)} {counts['only_b']}, "
                         f"both {counts['both']}, neither {counts['neither']}")
        lines.append("")
    return lines


def cost_section(df: pd.DataFrame, machine: str) -> list[str]:
    lines = ["## Cost", "",
             (f"Means per run. Wall time uses only machine `{machine}`; tokens per correct "
              "answer is all tokens divided by the number of correct runs."), ""]
    cost = stats.cost_summary(df, machine_id=machine)
    for task in sorted(set(cost["task"])):
        part = cost[cost["task"] == task].set_index("approach")
        rows = []
        for approach in ordered_approaches(part.index):
            r = part.loc[approach]
            wall_s = r["mean_wall_ms"] / 1000
            rows.append([approach_name(approach), f"{r['mean_tokens']:,.0f}",
                         f"{r['mean_calls']:.1f}", maybe(wall_s, "{:.1f}"),
                         maybe(r["tokens_per_correct"], "{:,.0f}")])
        lines += [f"### {task_name(task)}", ""]
        lines += table(["Approach", "Mean tokens", "Mean LLM calls", "Mean wall time (s)",
                        "Tokens per correct"], rows) + [""]
    return lines


def figures_section(figures: Sequence[Path], out_dir: Path) -> list[str]:
    lines = ["## Figures", ""]
    lines += [f"- `{Path(p).relative_to(out_dir).as_posix()}`" for p in figures]
    return lines + [""]


# ---- Main ----

def make_report(df: pd.DataFrame, input_name: str, out_dir: Path,
                machine_id: str | None = None) -> Path:
    """Write the figures and report.md into out_dir; return the path of report.md."""
    out_dir = Path(out_dir)
    machine = machine_id if machine_id is not None else stats.default_machine(df)
    if machine not in set(df["machine_id"].astype(str)):
        raise ValueError(f"machine_id {machine!r} is not in the table")
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = plot_all(df, out_dir, machine_id=machine)   # also refuses mixed models or splits
    lines = (header_section(df, input_name, machine) + success_section(df)
             + paired_section(df) + cost_section(df, machine) + figures_section(figures, out_dir))
    report = out_dir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build figures and a Markdown summary.")
    parser.add_argument("--input", required=True, type=Path, help="results CSV (runs table)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (default: report/ next to the input file)")
    parser.add_argument("--machine", default=None,
                        help="machine_id for wall time (default: the one with the most rows)")
    args = parser.parse_args(argv)

    df = stats.load_runs(args.input)
    out_dir = args.out if args.out is not None else args.input.parent / "report"
    try:
        report = make_report(df, args.input.name, out_dir, machine_id=args.machine)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(report)


if __name__ == "__main__":
    main()
