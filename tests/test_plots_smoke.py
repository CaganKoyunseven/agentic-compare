import pytest
from pathlib import Path
import pandas as pd
from agentic.analysis.plots import plot_success, plot_cumulative, plot_rep_spread, plot_cost_vs_accuracy, plot_time_and_calls

@pytest.fixture
def fake_df():
    return pd.read_csv("traces/runs/fake/runs.csv")

def test_plots_generation(fake_df, tmp_path):
    out_dir = tmp_path / "figures"
    
    p1 = plot_success(fake_df, out_dir / "success.png", lang="tr")
    assert p1.exists() and p1.stat().st_size > 0

    p2 = plot_cumulative(fake_df, out_dir / "cumulative.png", lang="en")
    assert p2.exists() and p2.stat().st_size > 0

    p3 = plot_rep_spread(fake_df, out_dir / "rep_spread.png", lang="tr")
    assert p3.exists() and p3.stat().st_size > 0

    p4 = plot_cost_vs_accuracy(fake_df, out_dir / "cost_vs_accuracy.png", lang="en")
    assert p4.exists() and p4.stat().st_size > 0

    p5 = plot_time_and_calls(fake_df, out_dir / "time_and_calls.png", lang="tr")
    assert p5.exists() and p5.stat().st_size > 0
