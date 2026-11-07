"""Tests for the baseline approach and the runner, with a fake LLM (no Ollama needed)."""

import csv

import pytest

from agentic.approaches.base import Caller
from agentic.approaches.baseline import Baseline
from agentic.events import FinalAnswer, LLMCall, RunEnd, RunStart
from agentic.llm.base import LLMResult
from agentic.llm.ollama import OllamaError
from agentic.runner import (
    CSV_COLUMNS,
    Setup,
    choose_approaches,
    experiment_dir,
    run_experiment,
    run_seed,
)
from agentic.tasks.base import Item
from agentic.tasks.game24 import Game24Task
from agentic.tasks.humaneval import HumanEvalTask
from agentic.traces import group_runs, read_trace

ITEMS = [
    Item("game24/0001", "game24", {"numbers": [1, 1, 4, 6]}, "easy"),
    Item("game24/0002", "game24", {"numbers": [4, 4, 6, 8]}, "medium"),
]
REPLY = "Let me try.\n(1*1)*4*6"         # correct only for the first item


class FakeLLM:
    model = "fake"

    def __init__(self, reply=REPLY, fail_at=None, error=KeyboardInterrupt):
        self.reply, self.fail_at, self.error = reply, fail_at, error
        self.seeds = []

    def chat(self, messages, *, seed=None):
        if len(self.seeds) == self.fail_at:
            raise self.error()
        self.seeds.append(seed)
        return LLMResult(text=self.reply, tokens_in=20, tokens_out=7, wall_ms=3, load_ms=0,
                         model="fake", done_reason="stop")


class Broken:
    name = "broken"

    def run(self, item, task, llm, *, seed, config):
        raise ValueError("bad parse")
        yield  # makes this a generator


def make_setup(**changes):
    values = {"task": Game24Task(), "split": "dev", "model": "fake", "machine_id": "test",
              "base_seed": 2026, "llm_settings": {"temperature": 0.7}}
    values.update(changes)
    return Setup(**values)


def runs_of(path):
    return list(group_runs(read_trace(path)).values())


def test_caller_counts_calls_and_adds_seed():
    llm = FakeLLM()
    caller = Caller(llm, seed=100)
    first = caller.chat("solver", [{"role": "user", "content": "a"}])
    second = caller.chat("critic", [{"role": "user", "content": "b"}])
    assert (first.call_id, first.seed) == ("call-0", 100)
    assert (second.call_id, second.seed, second.role) == ("call-1", 101, "critic")
    assert llm.seeds == [100, 101]


def test_baseline_makes_one_call_and_extracts_answer():
    events = list(Baseline().run(ITEMS[0], Game24Task(), FakeLLM(), seed=5, config={}))
    assert [type(e) for e in events] == [LLMCall, FinalAnswer]
    assert events[0].messages[0]["content"] == Game24Task().build_prompt(ITEMS[0])
    assert events[1].answer == "(1*1)*4*6"


def test_experiment_writes_valid_trace_and_csv(tmp_path):
    path = tmp_path / "trace.jsonl"
    llm = FakeLLM()
    run_experiment(make_setup(), [Baseline()], ITEMS, 2, llm, path, log=lambda s: None)

    runs = runs_of(path)
    assert len(runs) == 4                              # 2 reps x 2 items
    for run in runs:
        assert isinstance(run[0][1], RunStart) and isinstance(run[-1][1], RunEnd)
        assert [env.seq for env, _ in run] == list(range(len(run)))
    # Interleaved order: rep 0 for all items first; seed = base + 1000 * rep + index.
    assert [(r[0][0].rep, r[0][0].item_id) for r in runs] == [
        (0, "game24/0001"), (0, "game24/0002"), (1, "game24/0001"), (1, "game24/0002")]
    assert llm.seeds == [run_seed(2026, rep, i) for rep in (0, 1) for i in (0, 1)]

    with open(tmp_path / "runs.csv", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == CSV_COLUMNS
    assert [r["correct"] for r in rows] == ["1", "0", "1", "0"]
    assert rows[0]["tokens_in"] == "20" and rows[0]["llm_calls"] == "1"
    assert rows[1]["difficulty"] == "medium" and rows[1]["split"] == "dev"


def test_rerun_skips_finished_runs(tmp_path):
    path = tmp_path / "trace.jsonl"
    run_experiment(make_setup(), [Baseline()], ITEMS, 1, FakeLLM(), path, log=lambda s: None)
    size = path.stat().st_size
    llm = FakeLLM()
    run_experiment(make_setup(), [Baseline()], ITEMS, 1, llm, path, log=lambda s: None)
    assert llm.seeds == [] and path.stat().st_size == size


def test_ctrl_c_is_recorded_and_resume_finishes(tmp_path):
    path = tmp_path / "trace.jsonl"
    with pytest.raises(KeyboardInterrupt):
        run_experiment(make_setup(), [Baseline()], ITEMS, 1, FakeLLM(fail_at=1), path,
                       log=lambda s: None)
    ends = [run[-1][1] for run in runs_of(path)]
    assert [e.status for e in ends] == ["ok", "cancelled"]
    with open(tmp_path / "runs.csv", encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 1

    llm = FakeLLM()
    run_experiment(make_setup(), [Baseline()], ITEMS, 1, llm, path, log=lambda s: None)
    assert llm.seeds == [run_seed(2026, 0, 1)]         # only the cancelled run again
    with open(tmp_path / "runs.csv", encoding="utf-8", newline="") as f:
        assert [r["item_id"] for r in csv.DictReader(f)] == ["game24/0001", "game24/0002"]


def test_ollama_error_stops_the_experiment(tmp_path):
    path = tmp_path / "trace.jsonl"
    with pytest.raises(OllamaError):
        run_experiment(make_setup(), [Baseline()], ITEMS, 1,
                       FakeLLM(fail_at=0, error=OllamaError), path, log=lambda s: None)
    assert [run[-1][1].status for run in runs_of(path)] == ["error"]


def test_approach_error_is_recorded_and_experiment_goes_on(tmp_path):
    path = tmp_path / "trace.jsonl"
    run_experiment(make_setup(), [Broken()], ITEMS, 1, FakeLLM(), path, log=lambda s: None)
    ends = [run[-1][1] for run in runs_of(path)]
    assert [e.status for e in ends] == ["error", "error"]
    assert ends[0].error == "ValueError: bad parse" and ends[0].correct is None


def test_changed_settings_are_refused(tmp_path):
    path = tmp_path / "trace.jsonl"
    run_experiment(make_setup(), [Baseline()], ITEMS, 1, FakeLLM(), path, log=lambda s: None)
    with pytest.raises(RuntimeError, match="settings of baseline changed"):
        run_experiment(make_setup(llm_settings={"temperature": 0.2}), [Baseline()], ITEMS, 1,
                       FakeLLM(), path, log=lambda s: None)


def test_experiment_dir_and_approach_choice():
    path = experiment_dir("game24", "dev", "qwen2.5-coder:3b", "my laptop", "t1")
    assert path.name == "game24-dev-qwen2.5-coder-3b-my-laptop-t1"
    assert [a.name for a in choose_approaches("all")] == ["baseline", "reflective",
                                                          "multi_agent", "tot"]
    assert [a.name for a in choose_approaches("tot,baseline")] == ["baseline", "tot"]
    with pytest.raises(SystemExit):
        choose_approaches("baseline,magic")


def test_humaneval_data_and_splits():
    """The committed dataset loads with its fixed splits: dev 15, test 60, no overlap."""
    task = HumanEvalTask()
    dev, test = task.load_items("dev"), task.load_items("test")
    assert (len(task.load_items("all")), len(dev), len(test)) == (164, 15, 60)
    assert not {i.item_id for i in dev} & {i.item_id for i in test}
    assert [i.item_id for i in task.load_items("test")] == [i.item_id for i in test]


def test_runner_scores_humaneval_with_official_tests(tmp_path):
    """Baseline on the first real dev item: a wrong body fails the official tests, the
    canonical solution passes them, and runs.csv gets the rows."""
    task = HumanEvalTask()
    item = task.load_items("dev")[0]
    canonical = task._items_cache[item.item_id]["canonical_solution"]
    for name, reply, expected in (("wrong", "    return None", False),
                                  ("right", canonical, True)):
        path = tmp_path / name / "trace.jsonl"
        path.parent.mkdir()
        run_experiment(make_setup(task=task), [Baseline()], [item], 1, FakeLLM(reply=reply),
                       path, log=lambda s: None)
        [run] = runs_of(path)
        end = run[-1][1]
        assert (end.status, end.correct) == ("ok", expected), end.score_message
        with open(path.parent / "runs.csv", encoding="utf-8") as f:
            [row] = list(csv.DictReader(f))
        assert (row["task"], row["item_id"], row["difficulty"]) == ("humaneval",
                                                                    item.item_id, "")
