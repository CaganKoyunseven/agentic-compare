import inspect

import numpy as np
import pandas as pd
import pytest

from agentic.analysis import plots
from agentic.analysis.stats import REQUIRED_COLUMNS

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
FIGURES = (
    "success.png",
    "cumulative.png",
    "rep_spread.png",
    "paired.png",
    "cost_vs_accuracy.png",
    "time_and_calls.png",
)


@pytest.fixture
def small_df() -> pd.DataFrame:
    """2 tasks x 4 approaches x 6 items x 2 reps, on two machines."""
    rng = np.random.default_rng(0)
    level = {"baseline": 0.3, "reflective": 0.5, "multi_agent": 0.45, "tot": 0.6}
    rows = []
    for task in ("game24", "humaneval"):
        for approach in plots.APPROACHES:
            for item in range(6):
                for rep in range(2):
                    rows.append(
                        {
                            "run_id": f"r{len(rows)}",
                            "task": task,
                            "approach": approach,
                            "item_id": f"{task}-{item}",
                            "difficulty": "easy",
                            "rep": rep,
                            "split": "test",
                            "correct": int(rng.random() < level[approach]),
                            "tokens_in": int(rng.integers(100, 400)),
                            "tokens_out": int(rng.integers(50, 200)),
                            "llm_calls": int(rng.integers(1, 6)),
                            "wall_ms": int(rng.integers(500, 5000)),
                            "model": "test-model",
                            "machine_id": "main" if item < 4 else "other",
                        }
                    )
    df = pd.DataFrame(rows)
    assert list(df.columns) == list(REQUIRED_COLUMNS)
    return df


def assert_png(path):
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:8] == PNG_MAGIC


def test_plot_all_writes_six_figures(small_df, tmp_path):
    paths = plots.plot_all(small_df, tmp_path)
    assert [p.name for p in paths] == list(FIGURES)
    for path in paths:
        assert_png(path)


def test_each_plot_returns_its_path(small_df, tmp_path):
    out = plots.plot_time_and_calls(small_df, tmp_path / "sub" / "time.png", machine_id="other")
    assert out == tmp_path / "sub" / "time.png"
    assert_png(out)


def test_time_plot_rejects_unknown_machine(small_df, tmp_path):
    with pytest.raises(ValueError, match="nope"):
        plots.plot_time_and_calls(small_df, tmp_path / "t.png", machine_id="nope")


def test_paired_plot_needs_baseline(small_df, tmp_path):
    no_baseline = small_df[small_df["approach"] != "baseline"]
    with pytest.raises(ValueError, match="baseline"):
        plots.plot_paired(no_baseline, tmp_path / "p.png")


def test_no_language_option():
    for name in (
        "plot_success",
        "plot_cumulative",
        "plot_rep_spread",
        "plot_paired",
        "plot_cost_vs_accuracy",
        "plot_time_and_calls",
    ):
        func = inspect.unwrap(getattr(plots, name))
        assert "lang" not in inspect.signature(func).parameters
    assert "tr" not in plots.LABELS


def test_cli_writes_figures(small_df, tmp_path, capsys):
    csv_path = tmp_path / "runs.csv"
    small_df.to_csv(csv_path, index=False)
    out_dir = tmp_path / "figures"
    plots.main(["--input", str(csv_path), "--out", str(out_dir)])
    for name in FIGURES:
        assert_png(out_dir / name)
    assert capsys.readouterr().out.count("Saved") == 6
