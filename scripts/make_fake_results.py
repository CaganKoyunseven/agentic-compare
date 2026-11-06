"""
Generate a fake results table with the schema of results/runs.csv.

All numbers are invented. They exist only so that the stats and plotting code can be
developed and tested before real runs exist. The output is git-ignored and must never
be committed or presented as real data.

Usage: python scripts/make_fake_results.py [--seed 0] [--out PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
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
]
APPROACHES = ["baseline", "reflective", "multi_agent", "tot"]
TASKS = ["game24", "humaneval"]
N_ITEMS = 30
N_REPS = 3

# Invented average success level per (task, approach).
BASE_SUCCESS = {
    ("game24", "baseline"): 0.15,
    ("game24", "reflective"): 0.30,
    ("game24", "multi_agent"): 0.20,
    ("game24", "tot"): 0.45,
    ("humaneval", "baseline"): 0.40,
    ("humaneval", "reflective"): 0.60,
    ("humaneval", "multi_agent"): 0.50,
    ("humaneval", "tot"): 0.42,
}

# Invented number of LLM calls per run: (low, high), both inclusive.
CALL_RANGE = {
    ("game24", "baseline"): (1, 1),
    ("game24", "reflective"): (2, 4),
    ("game24", "multi_agent"): (6, 10),
    ("game24", "tot"): (25, 40),
    ("humaneval", "baseline"): (1, 1),
    ("humaneval", "reflective"): (2, 4),
    ("humaneval", "multi_agent"): (6, 10),
    ("humaneval", "tot"): (15, 30),
}

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "traces" / "runs" / "fake" / "runs.csv"


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def _difficulty_labels(task: str, effects: np.ndarray) -> list[str]:
    """Game of 24 items get easy/medium/hard by item effect; HumanEval has none."""
    if task != "game24":
        return [""] * len(effects)
    labels = [""] * len(effects)
    order = np.argsort(-effects)  # easiest items first
    third = len(effects) // 3
    for rank, idx in enumerate(order):
        if rank < third:
            labels[idx] = "easy"
        elif rank < 2 * third:
            labels[idx] = "medium"
        else:
            labels[idx] = "hard"
    return labels


def make_fake_results(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for task in TASKS:
        # Item difficulty is shared by all approaches, so paired comparisons make sense.
        effects = rng.normal(0.0, 1.2, N_ITEMS)
        difficulty = _difficulty_labels(task, effects)
        for approach in APPROACHES:
            base = BASE_SUCCESS[(task, approach)]
            low, high = CALL_RANGE[(task, approach)]
            for i in range(N_ITEMS):
                p = _sigmoid(_logit(base) + float(effects[i]))
                item_id = f"fake-g24-{i:03d}" if task == "game24" else f"HumanEval/{i}"
                for rep in range(N_REPS):
                    correct = int(rng.random() < p)
                    calls = int(rng.integers(low, high + 1))
                    tokens_in = int(rng.normal(350, 60, calls).clip(100).sum())
                    tokens_out = int(rng.normal(140, 40, calls).clip(20).sum())
                    wall_ms = int(calls * rng.lognormal(np.log(2500), 0.25))
                    rows.append(
                        {
                            "task": task,
                            "approach": approach,
                            "item_id": item_id,
                            "difficulty": difficulty[i],
                            "rep": rep,
                            "split": "test",
                            "correct": correct,
                            "tokens_in": tokens_in,
                            "tokens_out": tokens_out,
                            "llm_calls": calls,
                            "wall_ms": wall_ms,
                            "model": "fake-model",
                            "machine_id": "fake",
                        }
                    )
    df = pd.DataFrame(rows)
    df.insert(0, "run_id", [f"fake-{n:04d}" for n in range(len(df))])
    return df[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a fake results table.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = make_fake_results(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")

    print(f"Wrote {len(df)} fake rows to {args.out}")
    summary = df.groupby(["task", "approach"])["correct"].mean().round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()
