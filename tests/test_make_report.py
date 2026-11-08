import re

import pandas as pd
import pytest

from scripts.make_fake_results import COLUMNS, make_fake_results
from scripts.make_report import FAKE_WARNING, main, make_report, signed

SENTENCE = re.compile(
    r"^- (Reflective|Multi-agent|Tree-of-Thoughts) minus Baseline = [+-]\d+\.\d points "
    r"\(95% CI \[[+-]\d+\.\d, [+-]\d+\.\d\]\), n = \d+ items"
    r"( \(within noise, below 15 points\))?$",
    re.MULTILINE,
)
FIGURES = ["success.png", "cumulative.png", "rep_spread.png", "paired.png",
           "cost_vs_accuracy.png", "time_and_calls.png"]


def hand_table(solved: dict[str, int], n_items: int = 10, reps: int = 3) -> pd.DataFrame:
    """One task; approach -> how many of the n_items it solves (in every repetition)."""
    rows = []
    for approach, k in solved.items():
        for i in range(n_items):
            for rep in range(reps):
                rows.append({
                    "run_id": f"{approach}-{i}-{rep}", "task": "game24", "approach": approach,
                    "item_id": f"g{i:02d}", "difficulty": "easy", "rep": rep, "split": "test",
                    "correct": int(i < k), "tokens_in": 100, "tokens_out": 20, "llm_calls": 1,
                    "wall_ms": 1000, "model": "test-model", "machine_id": "m1",
                })
    return pd.DataFrame(rows)[COLUMNS]


def test_report_on_fake_table(tmp_path, capsys):
    csv = tmp_path / "runs.csv"
    make_fake_results(0).to_csv(csv, index=False, encoding="utf-8")
    main(["--input", str(csv)])

    report = tmp_path / "report" / "report.md"
    assert str(report) in capsys.readouterr().out
    text = report.read_text(encoding="utf-8")
    assert text.splitlines()[0] == FAKE_WARNING
    sentences = SENTENCE.findall(text)
    assert len(sentences) == 3 * 2       # three approaches against baseline, two tasks
    for name in FIGURES:
        assert (tmp_path / "report" / name).is_file(), name
        assert f"`{name}`" in text
    assert "Machine used for time: `fake`" in text
    assert "Items solved: only Reflective" in text


def test_sign_of_the_difference(tmp_path):
    df = hand_table({"baseline": 5, "reflective": 10, "tot": 0})
    text = make_report(df, "runs.csv", tmp_path).read_text(encoding="utf-8")
    assert FAKE_WARNING not in text
    assert "Reflective minus Baseline = +50.0 points" in text
    assert "Tree-of-Thoughts minus Baseline = -50.0 points" in text
    assert "within noise" not in text
    assert "only Reflective 5, only Baseline 0, both 5, neither 0" in text


def test_small_difference_gets_noise_note(tmp_path):
    df = hand_table({"baseline": 5, "reflective": 6})
    text = make_report(df, "runs.csv", tmp_path).read_text(encoding="utf-8")
    line = next(m.group(0) for m in SENTENCE.finditer(text))
    assert line.startswith("- Reflective minus Baseline = +10.0 points")
    assert line.endswith("(within noise, below 15 points)")


def test_tokens_per_correct_is_na_without_correct_runs(tmp_path):
    df = hand_table({"baseline": 0, "reflective": 3})
    text = make_report(df, "runs.csv", tmp_path).read_text(encoding="utf-8")
    baseline_cost = [ln for ln in text.splitlines() if ln.startswith("| Baseline |")][-1]
    assert baseline_cost.endswith("| n/a |")


def test_baseline_only_has_no_paired_comparison(tmp_path):
    df = hand_table({"baseline": 4})
    text = make_report(df, "runs.csv", tmp_path).read_text(encoding="utf-8")
    assert "No paired comparison" in text
    assert "paired.png" not in text


def test_unknown_machine_is_refused(tmp_path):
    df = hand_table({"baseline": 4, "reflective": 6})
    with pytest.raises(ValueError, match="machine_id"):
        make_report(df, "runs.csv", tmp_path, machine_id="other")


@pytest.mark.parametrize(("value", "text"), [(12.04, "+12.0"), (-3.46, "-3.5"), (-0.04, "+0.0")])
def test_signed(value, text):
    assert signed(value) == text
