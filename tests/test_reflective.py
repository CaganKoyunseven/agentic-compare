"""Tests for the Reflective approach with a scripted fake LLM (no Ollama needed)."""

from pathlib import Path

from agentic.approaches.reflective import Reflective, extract_asserts
from agentic.events import Attempt, Critique, FinalAnswer, LLMCall, RunEnd, TestResult
from agentic.llm.base import LLMResult
from agentic.runner import Setup, run_experiment
from agentic.tasks.base import Item
from agentic.tasks.game24 import Game24Task
from agentic.tasks.humaneval import HumanEvalTask
from agentic.traces import group_runs, read_trace

FENCE = "`" * 3
ITEM = Item("game24/0001", "game24", {"numbers": [4, 9, 10, 13]}, "hard")
WRONG_1 = "(13 - 9) * 10 - 4"           # 36
WRONG_2 = "13 + 9 + 10 - 4"             # 28
RIGHT = "(13 - 9) * (10 - 4)"           # 24


class ScriptedLLM:
    """Answers each role (told apart by the prompt) from its own list, in order."""
    model = "fake"

    def __init__(self, solver, critic=(), reviser=(), tester=()):
        self.replies = {"solver": list(solver), "critic": list(critic),
                        "reviser": list(reviser), "tester": list(tester)}
        self.calls: list[tuple[str, str, int | None]] = []   # (role, prompt, seed)

    @staticmethod
    def role_of(prompt: str) -> str:
        if prompt.startswith(("You are checking", "You are reviewing")):
            return "critic"
        if "previous answer was wrong" in prompt or "previous solution failed" in prompt:
            return "reviser"
        if prompt.startswith("Read the Python function"):
            return "tester"
        return "solver"

    def chat(self, messages, *, seed=None):
        prompt = messages[-1]["content"]
        role = self.role_of(prompt)
        self.calls.append((role, prompt, seed))
        return LLMResult(text=self.replies[role].pop(0), tokens_in=10, tokens_out=5,
                         wall_ms=1, load_ms=0, model="fake", done_reason="stop")


def run_reflective(llm, item=ITEM, task=None, config=None):
    return list(Reflective().run(item, task or Game24Task(), llm, seed=50,
                                 config=config or {"max_attempts": 3}))


def kinds(events):
    return [type(e).__name__ for e in events]


def test_first_answer_passes_and_stops():
    events = run_reflective(ScriptedLLM(solver=[f"Answer: {RIGHT} = 24"]))
    assert kinds(events) == ["LLMCall", "Attempt", "TestResult", "FinalAnswer"]
    result = events[2]
    assert (result.source, result.ok, result.attempt_no) == ("verifier", True, 1)
    assert events[-1].answer == RIGHT


def test_critique_and_fix_until_the_check_passes():
    llm = ScriptedLLM(solver=[WRONG_1], critic=["Too big.", "Still too big."],
                      reviser=[WRONG_2, f"So: {RIGHT}"])
    events = run_reflective(llm)
    assert kinds(events) == [
        "LLMCall", "Attempt", "TestResult",
        "LLMCall", "Critique", "LLMCall", "Attempt", "TestResult",
        "LLMCall", "Critique", "LLMCall", "Attempt", "TestResult",
        "FinalAnswer"]
    assert [role for role, _, _ in llm.calls] == ["solver", "critic", "reviser",
                                                  "critic", "reviser"]
    assert [seed for _, _, seed in llm.calls] == [50, 51, 52, 53, 54]

    attempts = [e for e in events if isinstance(e, Attempt)]
    assert [(a.attempt_no, a.answer) for a in attempts] == [(1, WRONG_1), (2, WRONG_2),
                                                            (3, RIGHT)]
    calls = [e for e in events if isinstance(e, LLMCall)]
    assert [a.call_id for a in attempts] == [calls[0].call_id, calls[2].call_id,
                                             calls[4].call_id]
    critiques = [e for e in events if isinstance(e, Critique)]
    assert [(c.attempt_no, c.text, c.call_id) for c in critiques] == [
        (1, "Too big.", calls[1].call_id), (2, "Still too big.", calls[3].call_id)]
    results = [e for e in events if isinstance(e, TestResult)]
    assert [(r.ok, r.attempt_no) for r in results] == [(False, 1), (False, 2), (True, 3)]
    assert events[-1].answer == RIGHT


def test_prompts_carry_answer_feedback_and_critique():
    llm = ScriptedLLM(solver=[WRONG_1], critic=["Use 10 - 4."], reviser=[RIGHT])
    run_reflective(llm)
    _, critic_prompt, _ = llm.calls[1]
    assert "Numbers: 4 9 10 13" in critic_prompt
    assert WRONG_1 in critic_prompt
    assert "evaluates to 36, not 24" in critic_prompt
    _, reviser_prompt, _ = llm.calls[2]
    assert WRONG_1 in reviser_prompt and "evaluates to 36" in reviser_prompt
    assert "Review: Use 10 - 4." in reviser_prompt


def test_stops_after_max_attempts_with_last_answer():
    llm = ScriptedLLM(solver=[WRONG_1], critic=["a", "b"], reviser=[WRONG_2, "no idea"])
    events = run_reflective(llm)
    assert len([e for e in events if isinstance(e, LLMCall)]) == 5
    results = [e for e in events if isinstance(e, TestResult)]
    assert [r.ok for r in results] == [False, False, False]
    assert results[-1].message == "empty answer"
    assert events[-1] == FinalAnswer(answer="")


def test_one_attempt_is_like_the_baseline():
    events = run_reflective(ScriptedLLM(solver=[WRONG_1]), config={"max_attempts": 1})
    assert kinds(events) == ["LLMCall", "Attempt", "TestResult", "FinalAnswer"]
    assert events[-1].answer == WRONG_1


def test_extract_asserts():
    text = ("Here are tests:\n```python\n    assert add(2, 3) == 5\nassert add(0, 0) == 0\n"
            "assert add(\n    1, 1) == 2\nprint('x')\n```")
    assert extract_asserts(text) == "assert add(2, 3) == 5\nassert add(0, 0) == 0"
    assert extract_asserts("no tests") is None


def test_humaneval_own_tests_then_fix():
    task = HumanEvalTask(data_path=Path("tests/fixtures/humaneval_mini.jsonl"),
                         dev_size=1, test_size=2)
    item = next(i for i in task.load_items("all") if i.item_id == "HumanEval/0")
    llm = ScriptedLLM(
        tester=["assert add(2, 3) == 5\nassert add(10, -1) == 9"],
        solver=[f"{FENCE}python\ndef add(a, b):\n    return a - b\n{FENCE}"],
        critic=["It subtracts instead of adding."],
        reviser=[f"{FENCE}python\ndef add(a, b):\n    return a + b\n{FENCE}"])
    events = run_reflective(llm, item=item, task=task)

    assert [role for role, _, _ in llm.calls] == ["tester", "solver", "critic", "reviser"]
    assert llm.calls[0][1].rstrip().endswith(item.payload["prompt"].rstrip())   # only the task
    results = [e for e in events if isinstance(e, TestResult)]
    assert [(r.source, r.ok) for r in results] == [("own_tests", False), ("own_tests", True)]
    assert "AssertionError" in llm.calls[2][1]                   # critic sees the failure
    assert "return a + b" in events[-1].answer
    # The official tests never reach any prompt.
    assert all("def check(candidate)" not in prompt for _, prompt, _ in llm.calls)


def test_runner_scores_reflective_answer(tmp_path):
    setup = Setup(task=Game24Task(), split="dev", model="fake", machine_id="test",
                  base_seed=2026, llm_settings={"temperature": 0.7},
                  approach_configs={"reflective": {"max_attempts": 3}})
    path = tmp_path / "trace.jsonl"
    llm = ScriptedLLM(solver=[WRONG_1], critic=["Too big."], reviser=[RIGHT])
    run_experiment(setup, [Reflective()], [ITEM], 1, llm, path, log=lambda s: None)
    [run] = group_runs(read_trace(path)).values()
    end = run[-1][1]
    assert isinstance(end, RunEnd)
    assert (end.status, end.correct, end.llm_calls, end.answer) == ("ok", True, 3, RIGHT)
    assert run[0][1].config["approach"] == {"max_attempts": 3}
