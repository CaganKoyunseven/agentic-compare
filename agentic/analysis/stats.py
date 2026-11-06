"""Success rates, bootstrap intervals and paired comparisons.

All functions take a DataFrame with the columns of the results table (see ``REQUIRED_COLUMNS``).
Correctness is first averaged over repetitions for each (task, approach, item). Confidence
intervals come from a percentile bootstrap over ITEMS, not rows, because repetitions of the
same item are correlated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "run_id",
    "task",
    "approach",
    "item_id",
    "difficulty",
    "rep",
    "split",
    "correct",
    "tokens_in",
    "tokens_out",
    "llm_calls",
    "wall_ms",
    "model",
    "machine_id",
)

N_BOOT = 10_000
CI_LEVEL = 0.95
SOLVED_THRESHOLD = 0.5


def load_runs(path: str | Path) -> pd.DataFrame:
    """Read a results CSV and check that every required column is present."""
    df = pd.read_csv(
        Path(path),
        encoding="utf-8",
        dtype={"run_id": str, "item_id": str, "difficulty": str, "model": str, "machine_id": str},
    )
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {Path(path).name}: {', '.join(missing)}")
    return df


def per_item_correctness(df: pd.DataFrame) -> pd.DataFrame:
    """Mean correctness over repetitions, one row per (task, approach, item_id)."""
    return (
        df.groupby(["task", "approach", "item_id"], as_index=False)["correct"]
        .mean()
        .astype({"correct": float})
    )


def default_machine(df: pd.DataFrame) -> str:
    """The machine_id with the most rows (ties broken by name)."""
    counts = df["machine_id"].astype(str).value_counts()
    top = counts.max()
    return min(counts[counts == top].index)


def _percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    alpha = (1.0 - CI_LEVEL) / 2.0
    low, high = np.percentile(samples, [100 * alpha, 100 * (1 - alpha)])
    return float(low), float(high)


def _bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values`` (resampling items)."""
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return _percentile_ci(values[idx].mean(axis=1))


def success_rate(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """Accuracy per (task, approach) as a fraction in [0, 1], with a 95% bootstrap CI."""
    items = per_item_correctness(df)
    rng = np.random.default_rng(seed)
    rows = []
    for (task, approach), group in items.groupby(["task", "approach"]):
        values = group["correct"].to_numpy()
        low, high = _bootstrap_mean_ci(values, rng, n_boot)
        rows.append(
            {
                "task": task,
                "approach": approach,
                "mean": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
                "n_items": len(values),
            }
        )
    return pd.DataFrame(rows, columns=["task", "approach", "mean", "ci_low", "ci_high", "n_items"])


def _paired_items(df: pd.DataFrame, task: str, a: str, b: str) -> pd.DataFrame:
    """Per-item correctness of approaches a and b, only for items present for both."""
    items = per_item_correctness(df[df["task"] == task])
    wide = items.pivot(index="item_id", columns="approach", values="correct")
    for name in (a, b):
        if name not in wide.columns:
            raise ValueError(f"No runs for approach {name!r} on task {task!r}")
    paired = wide[[a, b]].dropna()
    if paired.empty:
        raise ValueError(f"No items shared by {a!r} and {b!r} on task {task!r}")
    return paired


def paired_difference(
    df: pd.DataFrame, task: str, a: str, b: str, n_boot: int = N_BOOT, seed: int = 0
) -> dict:
    """Accuracy of a minus b in percentage points, with a 95% paired bootstrap CI over items."""
    paired = _paired_items(df, task, a, b)
    diffs = (paired[a] - paired[b]).to_numpy() * 100.0
    low, high = _bootstrap_mean_ci(diffs, np.random.default_rng(seed), n_boot)
    return {"diff": float(diffs.mean()), "ci_low": low, "ci_high": high, "n_items": len(diffs)}


def paired_counts(df: pd.DataFrame, task: str, a: str, b: str) -> dict:
    """How many shared items only a, only b, both or neither solved (mean correct >= 0.5)."""
    paired = _paired_items(df, task, a, b)
    solved_a = paired[a] >= SOLVED_THRESHOLD
    solved_b = paired[b] >= SOLVED_THRESHOLD
    return {
        "only_a": int((solved_a & ~solved_b).sum()),
        "only_b": int((~solved_a & solved_b).sum()),
        "both": int((solved_a & solved_b).sum()),
        "neither": int((~solved_a & ~solved_b).sum()),
    }


def cumulative_accuracy(
    df: pd.DataFrame, task: str, approach: str, n_perm: int = 200, seed: int = 0
) -> pd.DataFrame:
    """Running mean of per-item correctness over shuffled item orders.

    ``mean`` is the average running mean across shuffles; ``low`` and ``high`` are the 2.5th and
    97.5th percentiles across shuffles.
    """
    items = per_item_correctness(df)
    values = items.loc[
        (items["task"] == task) & (items["approach"] == approach), "correct"
    ].to_numpy()
    if len(values) == 0:
        raise ValueError(f"No runs for approach {approach!r} on task {task!r}")
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((n_perm, len(values))), axis=1)
    counts = np.arange(1, len(values) + 1)
    running = np.cumsum(values[order], axis=1) / counts
    low, high = np.percentile(running, [2.5, 97.5], axis=0)
    return pd.DataFrame({"n": counts, "mean": running.mean(axis=0), "low": low, "high": high})


def rep_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of each repetition per (task, approach)."""
    out = df.groupby(["task", "approach", "rep"], as_index=False)["correct"].mean()
    return out.rename(columns={"correct": "accuracy"}).astype({"accuracy": float})


def cost_summary(df: pd.DataFrame, machine_id: str | None = None) -> pd.DataFrame:
    """Mean tokens, LLM calls and wall time per (task, approach).

    Tokens and calls use all rows. Wall time is only comparable within one machine, so
    ``mean_wall_ms`` uses only rows of ``machine_id`` (default: the machine with the most rows);
    it is NaN for groups with no rows on that machine.
    """
    if machine_id is None:
        machine_id = default_machine(df)
    data = df.assign(tokens=df["tokens_in"] + df["tokens_out"])
    summary = data.groupby(["task", "approach"]).agg(
        mean_tokens=("tokens", "mean"),
        mean_calls=("llm_calls", "mean"),
        total_tokens=("tokens", "sum"),
        n_correct=("correct", "sum"),
    )
    on_machine = data[data["machine_id"].astype(str) == str(machine_id)]
    summary["mean_wall_ms"] = on_machine.groupby(["task", "approach"])["wall_ms"].mean()
    correct = summary["n_correct"].astype(float)
    summary["tokens_per_correct"] = (summary["total_tokens"] / correct).where(correct > 0)
    summary = summary.reset_index()
    return summary[
        ["task", "approach", "mean_tokens", "mean_calls", "mean_wall_ms", "tokens_per_correct"]
    ].astype({"mean_tokens": float, "mean_calls": float, "mean_wall_ms": float})
