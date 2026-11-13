"""Tests for the feedback study approaches (Self-Refine, Debate, Retry) with scripted fake LLMs."""

import pytest

from agentic.approaches.debate import Debate
from agentic.approaches.retry import Retry
from agentic.approaches.self_refine import SelfRefine, verdict
from agentic.events import AgentMessage, Attempt, Critique, FinalAnswer, LLMCall, TestResult
from agentic.llm.base import LLMResult
from agentic.runner import SEPARATE, choose_approaches
from agentic.tasks.base import Item
from agentic.tasks.game24 import Game24Task

ITEM = Item("game24/0001", "game24", {"numbers": [4, 9, 10, 13]}, "hard")
WRONG_1 = "(13 - 9) * 10 - 4"           # 36
WRONG_2 = "13 + 9 + 10 - 4"             # 28
RIGHT = "(13 - 9) * (10 - 4)"           # 24


class Replies:
    """Answers from a list in order, whatever the prompt; records prompts and seeds."""
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[tuple[list[dict[str, str]], int | None]] = []

    def chat(self, messages, *, seed=None):
        self.calls.append((messages, seed))
        return LLMResult(text=self.replies.pop(0), tokens_in=10, tokens_out=5, wall_ms=1,
                         load_ms=0, model="fake", done_reason="stop")


def run(approach, replies, config=None):
    llm = Replies(replies)
    events = list(approach.run(ITEM, Game24Task(), llm, seed=70, config=config or {}))
    return events, llm


def of(events, kind):
    return [e for e in events if isinstance(e, kind)]


# ---- Self-Refine ----

@pytest.mark.parametrize(("text", "expected"), [
    ("Looks fine.\nVERDICT: CORRECT", True),
    ("It makes 36.\nVERDICT: WRONG", False),
    ("verdict: correct", True),
    ("First I thought VERDICT: CORRECT, but no.\nVERDICT: WRONG", False),
    ("No verdict line at all", False),
])
def test_verdict_reads_the_last_verdict_line(text, expected):
    assert verdict(text) is expected


def test_self_refine_stops_when_its_own_check_says_correct():
    # The model wrongly believes its first answer is right: Self-Refine trusts it and stops.
    events, llm = run(SelfRefine(), [f"Answer: {WRONG_1}", "It is 24.\nVERDICT: CORRECT"])
    assert [type(e).__name__ for e in events] == [
        "LLMCall", "Attempt", "LLMCall", "Critique", "TestResult", "FinalAnswer"]
    check = of(events, TestResult)[0]
    assert (check.source, check.ok, check.attempt_no) == ("self_check", True, 1)
    assert of(events, FinalAnswer)[0].answer == WRONG_1
    assert all("Checker result" not in m[-1]["content"] for m, _ in llm.calls)  # no verifier


def test_self_refine_revises_until_the_limit():
    replies = [WRONG_1, "Too big.\nVERDICT: WRONG", WRONG_2, "Too small.\nVERDICT: WRONG",
               RIGHT, "VERDICT: CORRECT"]
    events, llm = run(SelfRefine(), replies, {"max_attempts": 3})
    assert [a.answer for a in of(events, Attempt)] == [WRONG_1, WRONG_2, RIGHT]
    assert [c.attempt_no for c in of(events, Critique)] == [1, 2, 3]
    assert of(events, FinalAnswer)[0].answer == RIGHT
    roles = [c.role for c in of(events, LLMCall)]
    assert roles == ["solver", "critic", "reviser", "critic", "reviser", "critic"]
    reviser_prompt = llm.calls[2][0][-1]["content"]
    assert "Too big." in reviser_prompt and WRONG_1 in reviser_prompt
    assert [seed for _, seed in llm.calls] == [70, 71, 72, 73, 74, 75]


def test_self_refine_keeps_the_last_answer_when_never_satisfied():
    replies = [WRONG_1, "VERDICT: WRONG", WRONG_2, "VERDICT: WRONG"]
    events, _ = run(SelfRefine(), replies, {"max_attempts": 2})
    assert of(events, FinalAnswer)[0].answer == WRONG_2
    assert len(of(events, LLMCall)) == 4          # no critic call after the last answer


# ---- Debate ----

def test_debate_two_rounds_of_three_agents_then_majority_vote():
    round1 = [f"I get {WRONG_1}", f"I get {RIGHT}", f"I get {WRONG_2}"]
    round2 = [f"Now {RIGHT}", f"Still {RIGHT}", f"Now {WRONG_2}"]
    events, llm = run(Debate(), round1 + round2, {"agents": 3, "rounds": 2})
    assert len(of(events, LLMCall)) == 6
    messages = of(events, AgentMessage)
    assert [(m.round, m.role) for m in messages] == [
        (1, "agent 1"), (1, "agent 2"), (1, "agent 3"),
        (2, "agent 1"), (2, "agent 2"), (2, "agent 3")]
    assert of(events, FinalAnswer)[0].answer == RIGHT
    # Round 2: an agent sees its own first answer and the answers of the two others.
    history, _ = llm.calls[3]
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert history[1]["content"] == round1[0]
    assert round1[1] in history[2]["content"] and round1[2] in history[2]["content"]
    assert round1[0] not in history[2]["content"]
    assert len({seed for _, seed in llm.calls}) == 6


def test_debate_tie_goes_to_the_first_agent():
    replies = ["x", "y", "z", f"{WRONG_1}", f"{RIGHT}", "no idea"]
    events, _ = run(Debate(), replies, {"agents": 3, "rounds": 2})
    assert of(events, FinalAnswer)[0].answer == WRONG_1


# ---- Retry with the verifier ----

def test_retry_stops_at_the_first_answer_the_verifier_accepts():
    events, llm = run(Retry(), [WRONG_1, WRONG_2, RIGHT, "unused"], {"max_samples": 10})
    assert [a.answer for a in of(events, Attempt)] == [WRONG_1, WRONG_2, RIGHT]
    assert [(t.source, t.ok) for t in of(events, TestResult)] == [
        ("verifier", False), ("verifier", False), ("verifier", True)]
    assert of(events, FinalAnswer)[0].answer == RIGHT
    prompts = {m[-1]["content"] for m, _ in llm.calls}
    assert len(prompts) == 1                 # every sample is the plain baseline prompt
    assert [seed for _, seed in llm.calls] == [70, 71, 72]


def test_retry_gives_the_last_answer_when_none_passes():
    events, _ = run(Retry(), [WRONG_1, WRONG_2], {"max_samples": 2})
    assert of(events, FinalAnswer)[0].answer == WRONG_2


# ---- Runner wiring ----

def test_study_approaches_run_only_when_named():
    for name in ("self_refine", "debate", "retry"):
        assert name in SEPARATE
        assert name not in [a.name for a in choose_approaches("all")]
        assert [a.name for a in choose_approaches(name)] == [name]


@pytest.mark.parametrize("approach", [SelfRefine(), Debate(), Retry()])
def test_study_approaches_are_game24_only(approach):
    from agentic.tasks.humaneval import HumanEvalTask
    item = Item("HumanEval/0", "humaneval", {"prompt": "def f(): pass", "entry_point": "f"})
    with pytest.raises(NotImplementedError):
        list(approach.run(item, HumanEvalTask(), Replies(["x"]), seed=1, config={}))
