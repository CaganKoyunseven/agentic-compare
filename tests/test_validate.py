"""Tests for result validation and the results rebuild, with a fake LLM (no model needed)."""

import csv
import json

import pytest

from agentic.analysis.validate import (
    Expected,
    drop_duplicates,
    keep_complete_items,
    validate_runs,
)
from agentic.approaches.baseline import Baseline
from agentic.runner import Setup, run_experiment, update_results_after_run
from agentic.tasks.base import Item
from agentic.tasks.game24 import Game24Task
from agentic.traces import group_runs, read_trace
from scripts import update_results as ur
from tests.test_runner import FakeLLM

ITEMS = [
    Item("game24/0001", "game24", {"numbers": [1, 1, 4, 6]}, "easy"),
    Item("game24/0002", "game24", {"numbers": [4, 4, 6, 8]}, "medium"),
]
LLM_SETTINGS = {"temperature": 0.7}


def expected(**changes):
    values = {"model": "fake", "split": "test", "base_seed": 2026, "reps": 2,
              "approaches": ("baseline",),
              "configs": {"baseline": {"llm": LLM_SETTINGS, "approach": {}}}}
    values.update(changes)
    return Expected(**values)


def make_trace(path, llm=None, reps=2, items=ITEMS, llm_settings=LLM_SETTINGS):
    setup = Setup(task=Game24Task(), split="test", model="fake", machine_id="test",
                  base_seed=2026, llm_settings=llm_settings)
    try:
        run_experiment(setup, [Baseline()], items, reps, llm or FakeLLM(), path,
                       log=lambda _line: None)
    except KeyboardInterrupt:
        pass
    return path


def runs_of(path):
    return list(group_runs(read_trace(path)).values())


def edit_run_end(path, item_id, rep, **fields):
    """Change fields of one run_end line in place (simulates a broken or edited trace)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        event = json.loads(line)
        if event["type"] == "run_end" and event["item_id"] == item_id and event["rep"] == rep:
            lines[i] = json.dumps(event | fields)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(path, exp=None, items=ITEMS):
    return validate_runs(runs_of(path), Game24Task(), items, exp or expected(), "exp")


def test_valid_runs_all_pass(tmp_path):
    rows, excluded = validate(make_trace(tmp_path / "t.jsonl"))
    assert excluded == []
    assert len(rows) == 4
    assert {(r["item_id"], r["rep"], r["correct"]) for r in rows} == {
        ("game24/0001", 0, 1), ("game24/0001", 1, 1), ("game24/0002", 0, 0), ("game24/0002", 1, 0)}


def test_wrong_verdict_in_trace_is_caught_by_rescoring(tmp_path):
    path = make_trace(tmp_path / "t.jsonl")
    edit_run_end(path, "game24/0002", 0, correct=True)      # the answer is really wrong
    rows, excluded = validate(path)
    assert len(rows) == 3
    assert [(e.item_id, e.rep) for e in excluded] == [("game24/0002", 0)]
    assert "re-score says correct=False" in excluded[0].reason


def test_token_totals_must_match_llm_calls(tmp_path):
    path = make_trace(tmp_path / "t.jsonl")
    edit_run_end(path, "game24/0001", 1, tokens_in=999)
    _, excluded = validate(path)
    assert len(excluded) == 1 and "do not match the llm_call events" in excluded[0].reason


def test_other_settings_model_split_and_seed_are_refused(tmp_path):
    path = make_trace(tmp_path / "t.jsonl", llm_settings={"temperature": 0.2})
    _, excluded = validate(path)
    assert len(excluded) == 4 and all("settings differ" in e.reason for e in excluded)

    path = make_trace(tmp_path / "u.jsonl")
    _, excluded = validate(path, expected(model="other", split="dev"))
    assert all("model is fake" in e.reason and "split is test" in e.reason for e in excluded)

    # The seed follows the item's position in the split, so a reordered split is caught.
    _, excluded = validate(path, items=list(reversed(ITEMS)))
    assert len(excluded) == 4 and all("seed" in e.reason for e in excluded)


def test_cancelled_run_is_left_out_and_its_item_dropped(tmp_path):
    path = make_trace(tmp_path / "t.jsonl", llm=FakeLLM(fail_at=3))   # 4th run cancelled
    rows, excluded = validate(path)
    assert [e.reason for e in excluded] == [
        "the run ended with status cancelled: KeyboardInterrupt: "]
    kept, dropped = keep_complete_items(rows, ("baseline",), reps=2)
    assert {r["item_id"] for r in kept} == {"game24/0001"}
    assert [d.item_id for d in dropped] == ["game24/0002"]
    assert "baseline rep 1" in dropped[0].reason


def test_incomplete_item_is_dropped_from_every_approach():
    def row(approach, item, rep):
        return {"task": "game24", "approach": approach, "item_id": item, "rep": rep}
    rows = [row(a, i, r) for a in ("baseline", "tot") for i in ("A", "B") for r in (0, 1)]
    rows.remove(row("tot", "B", 1))
    kept, dropped = keep_complete_items(rows, ("baseline", "tot"), reps=2)
    assert {r["item_id"] for r in kept} == {"A"} and len(kept) == 4
    assert dropped[0].item_id == "B" and "tot rep 1" in dropped[0].reason


def test_duplicate_runs_keep_the_latest(tmp_path):
    first, _ = validate(make_trace(tmp_path / "a.jsonl"))
    second, _ = validate(make_trace(tmp_path / "b.jsonl"))
    rows, duplicates = drop_duplicates(first + second)
    assert {r["run_id"] for r in rows} == {r["run_id"] for r in second}
    assert len(duplicates) == 4 and all("duplicate" in d.reason for d in duplicates)


def test_update_results_writes_table_notes_and_report(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    test_dir = runs / "game24-test-fake-test"
    test_dir.mkdir(parents=True)
    good = make_trace(test_dir / "trace.jsonl")
    edit_run_end(good, "game24/0002", 1, correct=True)
    dev = runs / "game24-dev-fake-secret-box"      # a tuning run: must stay out, unnamed
    dev.mkdir()
    run_experiment(Setup(task=Game24Task(), split="dev", model="fake", machine_id="box",
                         base_seed=2026, llm_settings=LLM_SETTINGS),
                   [Baseline()], ITEMS, 1, FakeLLM(), dev / "trace.jsonl", log=lambda _l: None)

    monkeypatch.setattr(ur, "expected_setup", lambda model: expected())
    monkeypatch.setattr(Game24Task, "load_items", lambda self, split: ITEMS)
    reports = []

    def fake_report(df, name, out_dir):
        reports.append(len(df))
        path = out_dir / "report.md"
        path.write_text("# Report\n", encoding="utf-8")
        return path
    monkeypatch.setattr(ur, "make_report", fake_report)

    out = tmp_path / "results"
    n = ur.update_results(runs, out, model="fake", log=lambda _line: None)

    # game24/0002 rep 1 fails the re-score, so item 0002 is dropped for every approach.
    assert n == 2 and reports == [2]
    with open(out / "runs.csv", encoding="utf-8") as f:
        assert {row["item_id"] for row in csv.DictReader(f)} == {"game24/0001"}
    notes = (out / "validation.md").read_text(encoding="utf-8")
    assert "re-score says correct=False" in notes
    assert "item dropped from all approaches" in notes
    assert "| dev split (tuning only) | 1 |" in notes
    assert "secret-box" not in notes          # folder names of unused runs are not written
    assert "validation.md" in (out / "report.md").read_text(encoding="utf-8")


def test_no_valid_runs_leaves_figures_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(ur, "make_report", lambda *a: pytest.fail("must not build figures"))
    out = tmp_path / "results"
    assert ur.update_results(tmp_path / "empty", out, model="fake", log=lambda _l: None) == 0
    assert "Runs in the results: 0" in (out / "validation.md").read_text(encoding="utf-8")
    assert not (out / "runs.csv").exists()


def test_runner_update_failure_does_not_crash(monkeypatch, capsys):
    def broken():
        raise RuntimeError("disk full")
    monkeypatch.setattr(ur, "update_results", broken)
    update_results_after_run()
    assert "Results not updated (RuntimeError: disk full)" in capsys.readouterr().out
