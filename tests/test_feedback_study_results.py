"""The separate feedback study results: validated on their own, never in the main results."""

import csv

from agentic.analysis import feedback_study as fs
from agentic.analysis.validate import Expected
from agentic.approaches.debate import Debate
from agentic.approaches.retry import Retry
from agentic.approaches.self_refine import SelfRefine
from agentic.llm.base import LLMResult
from agentic.runner import Setup, run_experiment, write_rows
from agentic.tasks.game24 import Game24Task
from tests.test_validate import ITEMS, LLM_SETTINGS

CONFIGS = {"self_refine": {"max_attempts": 2}, "debate": {"agents": 3, "rounds": 2},
           "retry": {"max_samples": 3}}
WRONG = "1 + 1 + 4 + 6"          # 12, uses the numbers of item 0001
RIGHT = "(1 + 1) * (6 + 6)"      # wrong numbers for 0001, but makes 24


class StudyLLM:
    """Wrong answers for the solvers; the self-check always says CORRECT (overconfident)."""
    model = "fake"

    def chat(self, messages, *, seed=None):
        prompt = messages[-1]["content"]
        text = "It is fine.\nVERDICT: CORRECT" if prompt.startswith("You are checking") else WRONG
        return LLMResult(text=text, tokens_in=10, tokens_out=5, wall_ms=1, load_ms=0,
                         model="fake", done_reason="stop")


def study_expected(model="fake"):
    return Expected(model=model, split="test", base_seed=2026, reps=2,
                    approaches=tuple(CONFIGS),
                    configs={a: {"llm": LLM_SETTINGS, "approach": c}
                             for a, c in CONFIGS.items()})


def make_trace(folder, approaches=None):
    approaches = approaches or (SelfRefine(), Debate(), Retry())
    folder.mkdir(parents=True)
    setup = Setup(task=Game24Task(), split="test", model="fake", machine_id="test",
                  base_seed=2026, llm_settings=LLM_SETTINGS, approach_configs=CONFIGS)
    run_experiment(setup, list(approaches), ITEMS, 2, StudyLLM(), folder / "trace.jsonl",
                   log=lambda _l: None)


def main_rows():
    rows = []
    for approach, correct in (("baseline", 0), ("reflective", 1), ("multi_agent", 0),
                              ("tot", 1)):
        for item in ITEMS:
            for rep in range(2):
                rows.append({"run_id": f"{approach}-{item.item_id}-{rep}", "task": "game24",
                             "approach": approach, "item_id": item.item_id,
                             "difficulty": item.difficulty, "rep": rep, "split": "test",
                             "correct": correct, "tokens_in": 100, "tokens_out": 50,
                             "llm_calls": 1, "wall_ms": 10, "model": "fake",
                             "machine_id": "test"})
    return rows


def update(tmp_path, monkeypatch, with_main=True):
    runs = tmp_path / "runs"
    make_trace(runs / "game24-test-fake-test-fs")
    out = tmp_path / "results"
    out.mkdir()
    if with_main:
        write_rows(out / "runs.csv", main_rows())
    monkeypatch.setattr(Game24Task, "load_items", lambda self, split: ITEMS)
    monkeypatch.setattr(fs, "expected_setup", study_expected)
    n = fs.update_feedback_study(runs, out / "fs", out / "runs.csv", "fake", Game24Task(),
                                 log=lambda _l: None)
    return n, out / "fs"


def test_study_folder_has_all_files_and_only_study_runs(tmp_path, monkeypatch):
    n, folder = update(tmp_path, monkeypatch)
    assert n == 12                                    # 3 approaches x 2 items x 2 reps
    with open(folder / "runs.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["approach"] for r in rows} == set(CONFIGS)
    for name in ("validation.md", "report.md", *fs.FIGURES):
        assert (folder / name).is_file(), name


def test_report_compares_with_main_approaches_and_counts_self_checks(tmp_path, monkeypatch):
    _, folder = update(tmp_path, monkeypatch)
    report = (folder / "report.md").read_text(encoding="utf-8")
    assert "NOT part of the main results" in report
    assert "Reflective minus Self-Refine (own check)" in report
    assert "Debate (3 agents, 2 rounds) minus Multi-agent" in report
    assert "Retry with the verifier (max 10) minus Tree-of-Thoughts" in report
    # The check always says CORRECT on a wrong answer: 4 runs, one check each, all wrong.
    assert "| correct (4 checks) | 0 | 4 |" in report
    assert "really correct in 0% of the checks" in report
    # Retry never passes, so it uses all 3 answers.
    assert "Mean answers per run: 3.0" in report


def test_report_without_main_results_shows_the_study_alone(tmp_path, monkeypatch):
    _, folder = update(tmp_path, monkeypatch, with_main=False)
    report = (folder / "report.md").read_text(encoding="utf-8")
    assert "| Self-Refine (own check) | no |" in report
    assert "| Baseline |" not in report


def test_a_missing_approach_does_not_drop_items_of_the_others(tmp_path):
    rows = [{"task": "game24", "approach": "retry", "item_id": i.item_id, "rep": r}
            for i in ITEMS for r in range(2)]
    rows += [{"task": "game24", "approach": "debate", "item_id": ITEMS[0].item_id, "rep": 0}]
    kept, dropped = fs.complete_per_approach(rows, reps=2)
    assert sum(r["approach"] == "retry" for r in kept) == 4
    assert [(d.approach, d.item_id) for d in dropped] == [("debate", ITEMS[0].item_id)]


def test_wrong_setting_is_left_out(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    make_trace(runs / "fs", approaches=(Retry(),))
    monkeypatch.setattr(Game24Task, "load_items", lambda self, split: ITEMS)
    monkeypatch.setattr(fs, "expected_setup", lambda model: Expected(
        model="fake", split="test", base_seed=2026, reps=2, approaches=tuple(CONFIGS),
        configs={**study_expected().configs,
                 "retry": {"llm": LLM_SETTINGS, "approach": {"max_samples": 10}}}))
    out = tmp_path / "fs"
    assert fs.update_feedback_study(runs, out, tmp_path / "none.csv", "fake", Game24Task(),
                                    log=lambda _l: None) == 0
    assert "settings differ" in (out / "validation.md").read_text(encoding="utf-8")
