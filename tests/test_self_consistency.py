"""Tests for the self-consistency approach with a scripted fake LLM (no model needed)."""

import pytest

from agentic.approaches.self_consistency import SelfConsistency, canonical, vote
from agentic.events import FinalAnswer, LLMCall
from agentic.llm.base import LLMResult
from agentic.runner import choose_approaches
from agentic.tasks.base import Item
from agentic.tasks.game24 import Game24Task

ITEM = Item("game24/0001", "game24", {"numbers": [4, 9, 10, 13]}, "hard")


class SequenceLLM:
    """Gives the replies in order, one per call."""
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts, self.seeds = [], []

    def chat(self, messages, *, seed=None):
        self.prompts.append(messages[-1]["content"])
        self.seeds.append(seed)
        return LLMResult(text=self.replies.pop(0), tokens_in=10, tokens_out=5, wall_ms=1,
                         load_ms=0, model="fake", done_reason="stop")


def test_canonical_ignores_order_of_plus_and_times_and_parentheses():
    assert canonical("(13 - 9) * (10 - 4)") == canonical("((10-4)) * (13-9)")
    assert canonical("1 + (2 + 3)") == canonical("3 + 2 + 1") == canonical("(1 + 3) + 2")
    assert canonical("2 × 3 × 4") == canonical("4 * (3 * 2)")
    assert canonical("13 - 9") != canonical("9 - 13")
    assert canonical("8 / (3 - 8 / 3)") != canonical("8 / 3 - 8 / 3")
    assert canonical("") == ""
    assert canonical("(1 + 2") == "(1+2"          # unreadable: raw text without spaces


def test_vote_majority_ties_first_and_ignores_empty_answers():
    assert vote(["1+2", "2+1", "3*1", "3*1"]) == 0           # 2 vs 2: earliest group wins
    assert vote(["3*1", "1+2", "2+1", "2 + 1"]) == 1         # 3 votes for 1+2
    assert vote(["", "", "5*5"]) == 2
    assert vote(["", ""]) is None
    assert vote([]) is None


def test_runs_all_samples_with_the_baseline_prompt_and_votes():
    replies = ["(13 - 9) * (10 - 4)", "13 + 9 + 10 - 4", "(10 - 4) * (13 - 9) = 24",
               "no idea", "13 + 9 + 10 - 4"]
    llm = SequenceLLM(replies)
    task = Game24Task()
    events = list(SelfConsistency().run(ITEM, task, llm, seed=50, config={"samples": 5}))

    calls = [e for e in events if isinstance(e, LLMCall)]
    assert len(calls) == 5 and {c.role for c in calls} == {"sampler"}
    assert llm.seeds == [50, 51, 52, 53, 54]
    assert set(llm.prompts) == {task.build_prompt(ITEM)}
    # Two votes each for the two expressions; the first sample's group wins the tie.
    assert isinstance(events[-1], FinalAnswer)
    assert events[-1].answer == "(13 - 9) * (10 - 4)"
    assert len(events) == 6


def test_no_readable_answer_gives_empty_answer():
    events = list(SelfConsistency().run(ITEM, Game24Task(), SequenceLLM(["?", "no"]), seed=0,
                                        config={"samples": 2}))
    assert events[-1].answer == ""


def test_only_game24():
    class FakeTask:
        name = "humaneval"

    with pytest.raises(NotImplementedError):
        list(SelfConsistency().run(ITEM, FakeTask(), SequenceLLM([]), seed=0, config={}))


def test_runner_all_keeps_the_four_main_approaches():
    assert [a.name for a in choose_approaches("all")] == ["baseline", "reflective",
                                                         "multi_agent", "tot"]
    assert [a.name for a in choose_approaches("self_consistency")] == ["self_consistency"]
    assert [a.name for a in choose_approaches("self_consistency,baseline")] == [
        "baseline", "self_consistency"]
