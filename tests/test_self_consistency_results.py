"""The separate self-consistency results: validated on their own, never in the main results."""

import csv

from agentic.analysis import self_consistency as sca
from agentic.analysis.validate import Expected
from agentic.approaches.baseline import Baseline
from agentic.approaches.self_consistency import SelfConsistency
from agentic.llm.base import LLMResult
from agentic.runner import Setup, run_experiment, write_rows
from agentic.tasks.game24 import Game24Task
from scripts import update_results as ur
from tests.test_runner import FakeLLM
from tests.test_validate import ITEMS, LLM_SETTINGS, expected

SC_CONFIG = {"samples": 5}
REPLIES = ["4 * 6 * 1 * 1", "1 + 1 + 4 + 6", "1 * 1 * 6 * 4"]


class CyclingLLM:
    model = "fake"

    def __init__(self):
        self.n = 0

    def chat(self, messages, *, seed=None):
        text = REPLIES[self.n % len(REPLIES)]
        self.n += 1
        return LLMResult(text=text, tokens_in=10, tokens_out=5, wall_ms=1, load_ms=0,
                         model="fake", done_reason="stop")


def sc_expected(model="fake"):
    return Expected(model=model, split="test", base_seed=2026, reps=2,
                    approaches=(sca.SC,),
                    configs={sca.SC: {"llm": LLM_SETTINGS, "approach": SC_CONFIG}})


def make_sc_trace(folder):
    folder.mkdir(parents=True)
    setup = Setup(task=Game24Task(), split="test", model="fake", machine_id="test",
                  base_seed=2026, llm_settings=LLM_SETTINGS,
                  approach_configs={sca.SC: SC_CONFIG})
    run_experiment(setup, [SelfConsistency()], ITEMS, 2, CyclingLLM(), folder / "trace.jsonl",
                   log=lambda _l: None)


def main_rows():
    rows = []
    for approach, correct in (("baseline", 0), ("tot", 1)):
        for item in ITEMS:
            for rep in range(2):
                rows.append({"run_id": f"{approach}-{item.item_id}-{rep}", "task": "game24",
                             "approach": approach, "item_id": item.item_id,
                             "difficulty": item.difficulty, "rep": rep, "split": "test",
                             "correct": correct, "tokens_in": 100, "tokens_out": 50,
                             "llm_calls": 29 if approach == "tot" else 1, "wall_ms": 10,
                             "model": "fake", "machine_id": "test"})
    return rows


def test_sample_verdicts_majority_and_oracle():
    task, item = Game24Task(), ITEMS[0]
    answers = ["1 + 1 + 4 + 6", "4 * 6 * 1 * 1", "1 + 1 + 6 + 4"]
    row = sca.sample_verdicts(answers, task, item, ks=(1, 2, 3))
    assert row == {"maj_1": 0, "any_1": 0, "maj_2": 0, "any_2": 1, "maj_3": 0, "any_3": 1}


def test_separate_folder_report_and_main_results_untouched(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    make_sc_trace(runs / "game24-test-fake-test-sc")
    main_dir = runs / "game24-test-fake-test"
    main_dir.mkdir()
    run_experiment(Setup(task=Game24Task(), split="test", model="fake", machine_id="test",
                         base_seed=2026, llm_settings=LLM_SETTINGS),
                   [Baseline()], ITEMS, 2, FakeLLM(), main_dir / "trace.jsonl",
                   log=lambda _l: None)
    monkeypatch.setattr(Game24Task, "load_items", lambda self, split: ITEMS)
    monkeypatch.setattr(ur, "expected_setup", lambda model: expected())
    monkeypatch.setattr(sca, "expected_setup", sc_expected)
    monkeypatch.setattr(ur, "make_report", lambda df, name, out: (out / "report.md").write_text(
        "# Report\n", encoding="utf-8") and out / "report.md")

    out = tmp_path / "results"
    assert ur.update_results(runs, out, model="fake", log=lambda _l: None) == 4

    # Main results: baseline only; the self-consistency folder is listed as a separate study.
    with open(out / "runs.csv", encoding="utf-8") as f:
        assert {row["approach"] for row in csv.DictReader(f)} == {"baseline"}
    notes = (out / "validation.md").read_text(encoding="utf-8")
    assert "separate study" in notes and "self_consistency" not in notes.split("Runs left")[1]

    sc_out = out / "self_consistency"
    with open(sc_out / "runs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and {r["approach"] for r in rows} == {sca.SC}
    assert all(r["llm_calls"] == "5" for r in rows)
    with open(sc_out / "samples.csv", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))
    assert {"maj_1", "any_29"} <= set(samples[0])
    # Item 0001: replies 1 and 3 make 24 and vote together, so the vote is right.
    assert {s["maj_29"] for s in samples if s["item_id"] == "game24/0001"} == {"1"}
    for name in ("validation.md", "report.md", *sca.FIGURES):
        assert (sc_out / name).is_file(), name
    report = (sc_out / "report.md").read_text(encoding="utf-8")
    assert "NOT part of the main results" in report
    assert "Self-consistency (vote) minus Baseline" in report
    assert "Best-of-29 (oracle)" in report


def test_report_compares_with_main_baseline_and_tot(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    make_sc_trace(runs / "sc")
    out = tmp_path / "results"
    out.mkdir()
    write_rows(out / "runs.csv", main_rows())
    monkeypatch.setattr(Game24Task, "load_items", lambda self, split: ITEMS)
    monkeypatch.setattr(sca, "expected_setup", sc_expected)

    n = sca.update_self_consistency(runs, out / "sc", out / "runs.csv", "fake", Game24Task(),
                                    log=lambda _l: None)
    assert n == 4
    report = (out / "sc" / "report.md").read_text(encoding="utf-8")
    assert "Tree-of-Thoughts minus Self-consistency (vote)" in report
    assert "| Baseline | 0.0% [0.0, 0.0] | 1.0 |" in report


def test_wrong_setting_is_left_out(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    make_sc_trace(runs / "sc")
    monkeypatch.setattr(Game24Task, "load_items", lambda self, split: ITEMS)
    monkeypatch.setattr(sca, "expected_setup", lambda model: Expected(
        model="fake", split="test", base_seed=2026, reps=2, approaches=(sca.SC,),
        configs={sca.SC: {"llm": LLM_SETTINGS, "approach": {"samples": 29}}}))
    out = tmp_path / "sc"
    assert sca.update_self_consistency(runs, out, tmp_path / "none.csv", "fake", Game24Task(),
                                       log=lambda _l: None) == 0
    assert "settings differ" in (out / "validation.md").read_text(encoding="utf-8")
    assert not (out / "report.md").exists()


def test_no_runs_writes_nothing(tmp_path):
    out = tmp_path / "sc"
    assert sca.update_self_consistency(tmp_path, out, tmp_path / "x.csv", "fake",
                                       Game24Task(), log=lambda _l: None) == 0
    assert not out.exists()
