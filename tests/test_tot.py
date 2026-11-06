"""Tests for the Tree of Thoughts approach with a scripted fake LLM (no Ollama needed)."""

import operator
import re
from fractions import Fraction

import pytest

from agentic.approaches.tot import TreeOfThoughts, load_prompt, parse_label, parse_steps
from agentic.events import (
    FinalAnswer,
    LLMCall,
    NodeAdded,
    NodePruned,
    NodeScored,
    PathChosen,
    RunEnd,
)
from agentic.llm.base import LLMResult
from agentic.runner import Setup, run_experiment
from agentic.tasks.base import Item
from agentic.tasks.game24 import Game24Task, check_expression
from agentic.traces import group_runs, read_trace

OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
ITEM = Item("game24/0001", "game24", {"numbers": [4, 9, 10, 13]}, "hard")

# What the fake model proposes for each state ("Input: <state>" in the prompt).
PROPOSALS = {
    "4 9 10 13": "13 - 9 = 4 (left: 4 4 10)\n"
                 "1. 10 - 4 = 6 (left: 6 9 13)\n"
                 "- **4 + 9 = 13 (left: 10 13 13)**\n"
                 "9 * 10 = 90 (left: 4 13 90)",           # 4th line: over breadth 3
    "4 4 10": "10 - 4 = 6 (left: 4 6)\n4 + 4 = 8 (left: 8 10)",
    "6 9 13": "I am not sure what to do.",                 # nothing readable
    "4 6": "4 * 6 = 24 (left: 24)",
    "8 10": "8 + 10 = 18 (left: 18)",
}
VALUES = {"4 4 10": "(10 - 4) * 4 = 24\nsure", "6 9 13": "likely",
          "10 13 13": "too big\nimpossible", "4 6": "4 * 6 = 24\nsure", "8 10": "hmm",
          "24": "sure", "18": "impossible"}
WRITER = "Answer: (13 - 9) * (10 - 4) = 24"


class ScriptedLLM:
    """Answers by prompt kind: propose, value or write."""
    model = "fake"

    def __init__(self, proposals=PROPOSALS):
        self.proposals = proposals
        self.prompts, self.seeds = [], []

    def chat(self, messages, *, seed=None):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        self.seeds.append(seed)
        state = re.findall(r"Numbers: (.*)", prompt)[-1].strip()
        if prompt.rstrip().endswith("Next steps:"):
            text = self.proposals.get(state, "")
        elif prompt.startswith("Judge whether"):
            text = VALUES[state]
        else:
            text = WRITER
        return LLMResult(text=text, tokens_in=10, tokens_out=5, wall_ms=1, load_ms=0,
                         model="fake", done_reason="stop")


def run_tot(llm, config=None):
    return list(TreeOfThoughts().run(ITEM, Game24Task(), llm, seed=50,
                                     config=config or {"breadth": 3, "keep": 2}))


def of_type(events, kind):
    return [e for e in events if isinstance(e, kind)]


def test_parse_steps_reads_format_and_respects_limit():
    steps = parse_steps(PROPOSALS["4 9 10 13"], limit=3)
    assert steps == [("13 - 9 = 4", "4 4 10"), ("10 - 4 = 6", "6 9 13"),
                     ("4 + 9 = 13", "10 13 13")]
    assert parse_steps("no steps here", limit=3) == []


def test_parse_steps_handles_latex_commas_and_prefixes():
    # Real replies of a 3B model; the wrong "left" of the last line is kept as written.
    text = ("1. \\(1 + 5 = 6\\) (left: \\(6, 7, 12\\))\n"
            "2. Multiply 3 by 12: \\(3 \\times 12 = 36\\) (left: 36)\n"
            "- \\(12 - 11 = 1\\) (left: 12 1)")
    assert parse_steps(text, limit=5) == [("1 + 5 = 6", "6 7 12"), ("3 * 12 = 36", "36"),
                                          ("12 - 11 = 1", "12 1")]


def test_prompt_examples_are_correct():
    """Every example step in our prompts is right, and every example answer makes 24."""
    for name in ("game24_tot_propose", "game24_tot_write"):
        numbers = None
        for line in load_prompt(name).splitlines():
            if line.startswith("Numbers: ") and "{" not in line:
                numbers = [Fraction(n) for n in line.split()[1:]]
                left = list(numbers)
            elif line.startswith("Answer: ") and numbers:
                expr = line.removeprefix("Answer: ").split("=")[0]
                assert check_expression([int(n) for n in numbers], expr).ok, line
            elif (match := re.fullmatch(r"(\d+) ([-+*/]) (\d+) = (\d+) \(left: (.*)\)", line)):
                a, op, b, result, rest = match.groups()
                a, b, result = Fraction(a), Fraction(b), Fraction(result)
                assert OPS[op](a, b) == result, line
                state = list(left)
                state.remove(a)
                state.remove(b)
                assert sorted(state + [result]) == sorted(Fraction(n) for n in rest.split()), line
                if name == "game24_tot_write":
                    left = state + [result]     # write steps build on each other
    labels = re.findall(r"^(sure|likely|impossible)$", load_prompt("game24_tot_value"),
                        re.MULTILINE)
    assert labels == ["sure", "sure", "likely", "impossible", "impossible"]


def test_parse_label_takes_last_label():
    assert parse_label("maybe sure, but in the end: Impossible") == "impossible"
    assert parse_label("no idea") == "unknown"


def test_search_builds_scores_prunes_and_writes():
    llm = ScriptedLLM()
    events = run_tot(llm)

    added = of_type(events, NodeAdded)
    assert [(n.node_id, n.parent_id, n.state) for n in added] == [
        ("root", None, "4 9 10 13"),
        ("n1", "root", "4 4 10"), ("n2", "root", "6 9 13"), ("n3", "root", "10 13 13"),
        ("n4", "n1", "4 6"), ("n5", "n1", "8 10"),
        ("n6", "n4", "24"), ("n7", "n5", "18")]
    scored = {s.node_id: (s.label, s.score) for s in of_type(events, NodeScored)}
    assert scored["n1"] == ("sure", 1.0)
    assert scored["n2"] == ("likely", 0.5)
    assert scored["n5"] == ("unknown", 0.0)               # no label in the reply
    assert [p.node_id for p in of_type(events, NodePruned)] == ["n3"]

    chosen = of_type(events, PathChosen)[0]
    assert chosen.node_ids == ["root", "n1", "n4", "n6"]
    assert chosen.answer == "(13 - 9) * (10 - 4)"
    assert isinstance(events[-1], FinalAnswer) and events[-1].answer == chosen.answer

    calls = of_type(events, LLMCall)
    # Level 1: 1 propose + 3 scores; level 2: 2 + 2; level 3: 2 + 2; then 1 writer.
    assert len(calls) == 13
    assert [c.role for c in calls[:4]] == ["proposer", "scorer", "scorer", "scorer"]
    assert calls[-1].role == "writer"
    assert "13 - 9 = 4 (left: 4 4 10)\n10 - 4 = 6 (left: 4 6)\n4 * 6 = 24 (left: 24)" \
        in llm.prompts[-1]
    assert llm.seeds == list(range(50, 63))
    # Every node event points at the call that made or scored it.
    call_ids = {c.call_id for c in calls}
    assert all(n.call_id in call_ids for n in added[1:])


def test_no_readable_proposal_still_writes_an_answer():
    events = run_tot(ScriptedLLM(proposals={}))
    calls = of_type(events, LLMCall)
    assert [c.role for c in calls] == ["proposer", "writer"]
    assert of_type(events, PathChosen)[0].node_ids == ["root"]
    assert isinstance(events[-1], FinalAnswer)


def test_humaneval_not_built_yet():
    item = Item("humaneval/0", "humaneval", {"prompt": "def f(): ..."}, None)

    class FakeTask:
        name = "humaneval"

    with pytest.raises(NotImplementedError):
        list(TreeOfThoughts().run(item, FakeTask(), ScriptedLLM(), seed=0, config={}))


def test_runner_scores_tot_answer(tmp_path):
    setup = Setup(task=Game24Task(), split="dev", model="fake", machine_id="test",
                  base_seed=2026, llm_settings={"temperature": 0.7},
                  approach_configs={"tot": {"breadth": 3, "keep": 2}})
    path = tmp_path / "trace.jsonl"
    run_experiment(setup, [TreeOfThoughts()], [ITEM], 1, ScriptedLLM(), path,
                   log=lambda s: None)
    [run] = group_runs(read_trace(path)).values()
    end = run[-1][1]
    assert isinstance(end, RunEnd)
    assert (end.status, end.correct, end.llm_calls) == ("ok", True, 13)
    assert run[0][1].config["approach"] == {"breadth": 3, "keep": 2}
