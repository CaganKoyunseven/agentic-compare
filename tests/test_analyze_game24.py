"""Tests for the Game of 24 trace analysis (error types and where ToT loses the solution)."""

from agentic.approaches.baseline import Baseline
from agentic.approaches.tot import TreeOfThoughts
from agentic.runner import Setup, run_experiment, write_csv
from agentic.tasks.game24 import Game24Task
from agentic.traces import group_runs, read_trace
from scripts import analyze_game24 as an
from tests import test_tot
from tests.test_runner import FakeLLM
from tests.test_tot import ITEM, PROPOSALS, ScriptedLLM

SETUP = {"split": "test", "model": "fake", "machine_id": "test", "base_seed": 2026,
         "llm_settings": {"temperature": 0.7}}


def make_runs(tmp_path, approaches, llm, config=None):
    path = tmp_path / "trace.jsonl"
    setup = Setup(task=Game24Task(), approach_configs=config or {}, **SETUP)
    run_experiment(setup, approaches, [ITEM], 1, llm, path, log=lambda _l: None)
    return list(group_runs(read_trace(path)).values())


def test_error_kind_from_the_official_check():
    numbers = [4, 9, 10, 13]
    assert an.error_kind(numbers, "4 + 9 + 10 + 13") == "right numbers, value not 24"
    assert an.error_kind(numbers, "4 * 6") == "wrong numbers used"
    assert an.error_kind(numbers, "") == "no answer"
    assert an.error_kind(numbers, "(4 + 9") == "unreadable or not allowed"
    assert an.error_kind(numbers, "4 / (9 - 9) + 10 + 13") == "wrong numbers used"


def test_error_types_count_only_wrong_runs(tmp_path):
    runs = make_runs(tmp_path, [Baseline()], FakeLLM())          # FakeLLM answers 1 1 4 6
    assert an.error_types(runs, {ITEM.item_id: ITEM}) == {
        "baseline": {"wrong numbers used": 1}}


def test_tot_search_reaching_a_sound_24_leaf(tmp_path):
    runs = make_runs(tmp_path, [TreeOfThoughts()], ScriptedLLM(), {"tot": {"keep": 2}})
    tot = an.tot_analysis(runs)
    assert tot["funnel"] == {("reached a sound 24 leaf", True): 1}
    # n1 "4 4 10", n4 "4 6" and n6 "24" are judged sure 3 times each and can all reach 24.
    assert tot["labels"]["sure"] == {"reachable": 9}
    assert tot["labels"]["impossible"] == {"unreachable": 9}     # n3, n4 and n8
    assert tot["steps"][1] == {True: 4}                       # every proposed line is used


def test_tot_search_loses_the_solution_at_depth_2(tmp_path):
    proposals = {**PROPOSALS, "4 4 10": "4 + 4 = 8 (left: 8 10)", "6 9 13": "",
                 "10 13 13": "", "8 10": "8 + 10 = 18 (left: 18)"}
    llm = ScriptedLLM(proposals)
    runs = make_runs(tmp_path, [TreeOfThoughts()], llm, {"tot": {"keep": 3}})
    tot = an.tot_analysis(runs)
    assert tot["funnel"] == {("depth 2: no sound step proposed", False): 1}


def test_more_proposals_beyond_the_first_breadth(tmp_path, monkeypatch):
    # The root reply has a 4th line; with breadth 1 only the first line is used.
    proposals = {**PROPOSALS, "4 9 10 13": "10 + 13 = 23 (left: 4 9 23)\n"
                                            "13 - 9 = 4 (left: 4 4 10)"}
    monkeypatch.setitem(test_tot.VALUES, "4 9 23", "impossible")
    runs = make_runs(tmp_path, [TreeOfThoughts()], ScriptedLLM(proposals),
                     {"tot": {"breadth": 1, "keep": 1}})
    extra = an.tot_analysis(runs)["root_proposals"]
    assert extra == {"calls": 1, "sound_in_first": 0, "sound_in_all": 1}


def test_report_reads_only_validated_runs(tmp_path):
    folder = tmp_path / "runs" / "exp"
    folder.mkdir(parents=True)
    setup = Setup(task=Game24Task(), approach_configs={"tot": {"keep": 2}}, **SETUP)
    run_experiment(setup, [Baseline(), TreeOfThoughts()], [ITEM], 1, _Mixed(), folder /
                   "trace.jsonl", log=lambda _l: None)
    csv_path = tmp_path / "runs.csv"
    write_csv(folder / "trace.jsonl", csv_path)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    csv_path.write_text("\n".join(l for l in lines if not l.startswith("tot-")) + "\n",
                        encoding="utf-8")                   # pretend ToT failed validation
    out = tmp_path / "out"
    out.mkdir()
    an.write_analysis(tmp_path / "runs", csv_path, out, {ITEM.item_id: ITEM})
    text = (out / an.REPORT).read_text(encoding="utf-8")
    assert "| baseline | 1 |" in text
    assert "No valid ToT runs" in text


class _Mixed:
    """Baseline prompt -> FakeLLM reply, ToT prompts -> ScriptedLLM reply."""
    model = "fake"

    def __init__(self):
        self.fake, self.tot = FakeLLM(), ScriptedLLM()

    def chat(self, messages, *, seed=None):
        prompt = messages[-1]["content"]
        target = self.fake if prompt.startswith("Use the numbers") else self.tot
        return target.chat(messages, seed=seed)
