import math
import time

import numpy as np
import pandas as pd
import pytest

from agentic.analysis import stats
from scripts.make_fake_results import make_fake_results


def make_runs(spec: dict[tuple[str, str], list[list[int]]], machine_id: str = "m1") -> pd.DataFrame:
    """Build a results table from {(task, approach): [[correct per rep] per item]}."""
    rows = []
    for (task, approach), items in spec.items():
        for i, reps in enumerate(items):
            for rep, correct in enumerate(reps):
                rows.append(
                    {
                        "run_id": f"r{len(rows)}",
                        "task": task,
                        "approach": approach,
                        "item_id": f"{task}-{i}",
                        "difficulty": "",
                        "rep": rep,
                        "split": "test",
                        "correct": correct,
                        "tokens_in": 100,
                        "tokens_out": 50,
                        "llm_calls": 1,
                        "wall_ms": 1000,
                        "model": "test-model",
                        "machine_id": machine_id,
                    }
                )
    return pd.DataFrame(rows, columns=list(stats.REQUIRED_COLUMNS))


def same_items(n: int = 20) -> list[list[int]]:
    return [[i % 2, i % 2, i % 2] for i in range(n)]


def test_load_runs_reads_valid_csv(tmp_path):
    path = tmp_path / "runs.csv"
    make_runs({("game24", "baseline"): same_items(4)}).to_csv(path, index=False)
    df = stats.load_runs(path)
    assert list(df.columns) == list(stats.REQUIRED_COLUMNS)
    assert len(df) == 12


def test_load_runs_lists_missing_columns(tmp_path):
    path = tmp_path / "runs.csv"
    make_runs({("game24", "baseline"): same_items(2)}).drop(
        columns=["wall_ms", "machine_id"]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="wall_ms, machine_id"):
        stats.load_runs(path)


def test_per_item_correctness_averages_reps():
    df = make_runs({("game24", "baseline"): [[1, 0, 1], [0, 0, 0]]})
    items = stats.per_item_correctness(df)
    assert items["correct"].tolist() == pytest.approx([2 / 3, 0.0])


def test_success_rate_known_mean_and_ci():
    df = make_runs({("game24", "baseline"): same_items(20), ("game24", "tot"): [[1, 1, 1]] * 20})
    out = stats.success_rate(df, n_boot=2000).set_index("approach")
    assert out.loc["baseline", "mean"] == pytest.approx(0.5)
    assert out.loc["baseline", "ci_low"] < 0.5 < out.loc["baseline", "ci_high"]
    assert out.loc["baseline", "n_items"] == 20
    # All items solved: no uncertainty left.
    assert out.loc["tot", "ci_low"] == out.loc["tot", "ci_high"] == 1.0


def test_success_rate_is_seeded():
    df = make_runs({("game24", "baseline"): same_items(15)})
    first = stats.success_rate(df, n_boot=500, seed=3)
    second = stats.success_rate(df, n_boot=500, seed=3)
    pd.testing.assert_frame_equal(first, second)


def test_bootstrap_resamples_items_not_rows():
    # Same per-item correctness, many more reps: the interval must not shrink.
    few = make_runs({("game24", "baseline"): [[i % 2] for i in range(20)]})
    many = make_runs({("game24", "baseline"): [[i % 2] * 10 for i in range(20)]})
    a = stats.success_rate(few, n_boot=2000).iloc[0]
    b = stats.success_rate(many, n_boot=2000).iloc[0]
    assert a["ci_low"] == pytest.approx(b["ci_low"])
    assert a["ci_high"] == pytest.approx(b["ci_high"])


def test_paired_difference_identical_approaches_is_zero():
    df = make_runs({("game24", "baseline"): same_items(), ("game24", "tot"): same_items()})
    out = stats.paired_difference(df, "game24", "tot", "baseline", n_boot=2000)
    assert out["diff"] == 0.0
    assert out["ci_low"] <= 0.0 <= out["ci_high"]
    assert out["n_items"] == 20


def test_paired_difference_clearly_better_is_positive():
    df = make_runs({("game24", "baseline"): [[0, 0, 0]] * 20, ("game24", "tot"): same_items(20)})
    out = stats.paired_difference(df, "game24", "tot", "baseline", n_boot=2000)
    assert out["diff"] == pytest.approx(50.0)  # percentage points
    assert out["ci_low"] > 0


def test_paired_difference_uses_shared_items_only():
    df = make_runs({("game24", "baseline"): [[0, 0, 0]] * 10, ("game24", "tot"): [[1, 1, 1]] * 15})
    out = stats.paired_difference(df, "game24", "tot", "baseline", n_boot=200)
    assert out["n_items"] == 10


def test_paired_difference_unknown_approach_raises():
    df = make_runs({("game24", "baseline"): same_items(5)})
    with pytest.raises(ValueError, match="tot"):
        stats.paired_difference(df, "game24", "tot", "baseline")


def test_paired_counts_known_values_and_threshold():
    a = [[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 1]]
    b = [[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]]
    df = make_runs({("humaneval", "reflective"): a, ("humaneval", "baseline"): b})
    out = stats.paired_counts(df, "humaneval", "reflective", "baseline")
    # Item means a: .67 .33 1 0 .67 -> solved at >= 0.5: T F T F T
    assert out == {"only_a": 2, "only_b": 1, "both": 1, "neither": 1}
    assert sum(out.values()) == 5


def test_cumulative_accuracy_ends_at_overall_mean():
    df = make_runs({("game24", "baseline"): [[1, 0, 1]] * 6 + [[0, 0, 0]] * 6})
    out = stats.cumulative_accuracy(df, "game24", "baseline", n_perm=100)
    assert list(out.columns) == ["n", "mean", "low", "high"]
    assert out["n"].tolist() == list(range(1, 13))
    last = out.iloc[-1]
    expected = (6 * 2 / 3) / 12
    assert last["mean"] == pytest.approx(expected)
    assert last["low"] == pytest.approx(expected)
    assert last["high"] == pytest.approx(expected)
    assert (out["low"] <= out["mean"] + 1e-12).all()
    assert (out["mean"] <= out["high"] + 1e-12).all()


def test_rep_spread_accuracy_per_rep():
    df = make_runs({("game24", "tot"): [[1, 0, 1], [1, 0, 0]]})
    out = stats.rep_spread(df)
    assert list(out.columns) == ["task", "approach", "rep", "accuracy"]
    assert out["accuracy"].tolist() == pytest.approx([1.0, 0.0, 0.5])


def test_cost_summary_values_and_nan_when_nothing_correct():
    df = make_runs({("game24", "baseline"): [[0, 0]] * 3, ("game24", "tot"): [[1, 0]] * 3})
    out = stats.cost_summary(df).set_index("approach")
    assert out.loc["tot", "mean_tokens"] == 150
    assert out.loc["tot", "mean_calls"] == 1
    assert out.loc["tot", "mean_wall_ms"] == 1000
    assert out.loc["tot", "tokens_per_correct"] == pytest.approx(150 * 6 / 3)
    assert math.isnan(out.loc["baseline", "tokens_per_correct"])


def test_cost_summary_wall_time_uses_one_machine():
    big = make_runs({("game24", "baseline"): same_items(4)}, machine_id="big")
    small = make_runs({("game24", "baseline"): same_items(2)}, machine_id="small")
    small["wall_ms"] = 9000
    df = pd.concat([big, small], ignore_index=True)
    assert stats.default_machine(df) == "big"
    assert stats.cost_summary(df)["mean_wall_ms"].iloc[0] == 1000
    assert stats.cost_summary(df, machine_id="small")["mean_wall_ms"].iloc[0] == 9000


def test_all_functions_run_on_fake_table(tmp_path):
    path = tmp_path / "runs.csv"
    make_fake_results(seed=0).to_csv(path, index=False)
    df = stats.load_runs(path)

    start = time.perf_counter()
    rates = stats.success_rate(df)
    assert len(rates) == 8
    assert rates["n_items"].eq(30).all()
    assert ((rates["ci_low"] <= rates["mean"]) & (rates["mean"] <= rates["ci_high"])).all()
    for task in ("game24", "humaneval"):
        for approach in ("reflective", "multi_agent", "tot"):
            diff = stats.paired_difference(df, task, approach, "baseline")
            assert diff["ci_low"] <= diff["diff"] <= diff["ci_high"]
            assert sum(stats.paired_counts(df, task, approach, "baseline").values()) == 30
        curve = stats.cumulative_accuracy(df, task, "tot")
        assert len(curve) == 30
    assert len(stats.rep_spread(df)) == 24
    costs = stats.cost_summary(df)
    assert not np.isnan(costs["mean_wall_ms"]).any()
    assert time.perf_counter() - start < 10
