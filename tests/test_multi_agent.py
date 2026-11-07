"""Tests for the multi-agent approach with a scripted fake LLM (no Ollama needed)."""

from pathlib import Path

from agentic.approaches.multi_agent import MultiAgent
from agentic.events import AgentMessage, FinalAnswer, LLMCall, RunEnd, TestResult
from agentic.llm.base import LLMResult
from agentic.runner import Setup, run_experiment
from agentic.tasks.base import Item
from agentic.tasks.game24 import Game24Task
from agentic.tasks.humaneval import HumanEvalTask
from agentic.traces import group_runs, read_trace

FENCE = "`" * 3
ITEM = Item("game24/0001", "game24", {"numbers": [4, 9, 10, 13]}, "hard")
WRONG = "(13 - 9) * 10 - 4"            # 36
RIGHT = "(13 - 9) * (10 - 4)"          # 24


class ScriptedLLM:
    """Answers each role (told apart by the prompt) from its own list, in order."""
    model = "fake"

    def __init__(self, planner=(), solver=(), critic=(), tester=()):
        self.replies = {"planner": list(planner), "solver": list(solver),
                        "critic": list(critic), "tester": list(tester)}
        self.calls: list[tuple[str, str, int | None]] = []   # (role, prompt, seed)

    @staticmethod
    def role_of(prompt: str) -> str:
        if prompt.startswith("You are the planner"):
            return "planner"
        if prompt.startswith("You are the critic"):
            return "critic"
        if prompt.startswith("Read the Python function"):
            return "tester"
        return "solver"

    def chat(self, messages, *, seed=None):
        prompt = messages[-1]["content"]
        role = self.role_of(prompt)
        self.calls.append((role, prompt, seed))
        return LLMResult(text=self.replies[role].pop(0), tokens_in=10, tokens_out=5,
                         wall_ms=1, load_ms=0, model="fake", done_reason="stop")


def run_multi(llm, item=ITEM, task=None, config=None):
    return list(MultiAgent().run(item, task or Game24Task(), llm, seed=70,
                                 config=config or {"max_rounds": 2}))


def kinds(events):
    return [type(e).__name__ for e in events]


def messages(events):
    return [(m.round, m.role, m.text) for m in events if isinstance(m, AgentMessage)]


def test_first_round_passes_and_stops():
    llm = ScriptedLLM(planner=["Make 4 and 6, then multiply."], solver=[f"{RIGHT} = 24"])
    events = run_multi(llm)
    assert kinds(events) == ["LLMCall", "AgentMessage", "LLMCall", "AgentMessage",
                             "TestResult", "FinalAnswer"]
    assert messages(events) == [(1, "planner", "Make 4 and 6, then multiply."),
                                (1, "solver", f"{RIGHT} = 24")]
    result = events[4]
    assert (result.source, result.ok, result.attempt_no) == ("verifier", True, 1)
    assert events[-1].answer == RIGHT


def test_critic_notes_go_to_the_planner_of_round_two():
    llm = ScriptedLLM(planner=["Plan A", "Plan B"], solver=[WRONG, RIGHT],
                      critic=["The solver ignored 10 - 4."])
    events = run_multi(llm)
    assert kinds(events) == [
        "LLMCall", "AgentMessage", "LLMCall", "AgentMessage", "TestResult",
        "LLMCall", "AgentMessage",
        "LLMCall", "AgentMessage", "LLMCall", "AgentMessage", "TestResult",
        "FinalAnswer"]
    assert messages(events) == [(1, "planner", "Plan A"), (1, "solver", WRONG),
                                (1, "critic", "The solver ignored 10 - 4."),
                                (2, "planner", "Plan B"), (2, "solver", RIGHT)]
    assert [seed for _, _, seed in llm.calls] == [70, 71, 72, 73, 74]
    calls = [e for e in events if isinstance(e, LLMCall)]
    msgs = [e for e in events if isinstance(e, AgentMessage)]
    assert [m.call_id for m in msgs] == [c.call_id for c in calls]
    assert [c.role for c in calls] == ["planner", "solver", "critic", "planner", "solver"]

    _, solver_prompt, _ = llm.calls[1]
    assert "Plan A" in solver_prompt and "Use the numbers 4 9 10 13" in solver_prompt
    _, critic_prompt, _ = llm.calls[2]
    assert "Plan A" in critic_prompt and WRONG in critic_prompt
    assert "evaluates to 36, not 24" in critic_prompt
    _, replan_prompt, _ = llm.calls[3]
    assert "Earlier plan: Plan A" in replan_prompt
    assert "Critic's notes: The solver ignored 10 - 4." in replan_prompt
    assert "evaluates to 36" in replan_prompt
    _, solver2_prompt, _ = llm.calls[4]
    assert "Plan B" in solver2_prompt and "Plan A" not in solver2_prompt

    results = [e for e in events if isinstance(e, TestResult)]
    assert [(r.ok, r.attempt_no) for r in results] == [(False, 1), (True, 2)]
    assert events[-1].answer == RIGHT


def test_last_round_ends_without_critic():
    llm = ScriptedLLM(planner=["A", "B"], solver=[WRONG, "I give up"], critic=["bad"])
    events = run_multi(llm)
    calls = [e for e in events if isinstance(e, LLMCall)]
    assert [c.role for c in calls] == ["planner", "solver", "critic", "planner", "solver"]
    assert isinstance(events[-2], TestResult) and not events[-2].ok
    assert events[-1] == FinalAnswer(answer="")


def test_one_round_has_no_critic():
    events = run_multi(ScriptedLLM(planner=["A"], solver=[WRONG]), config={"max_rounds": 1})
    assert [m.role for m in events if isinstance(m, AgentMessage)] == ["planner", "solver"]
    assert events[-1].answer == WRONG


def test_humaneval_tester_then_rounds():
    task = HumanEvalTask(data_path=Path("tests/fixtures/humaneval_mini.jsonl"),
                         dev_size=1, test_size=2)
    item = next(i for i in task.load_items("all") if i.item_id == "HumanEval/0")
    llm = ScriptedLLM(
        tester=["assert add(2, 3) == 5\nassert add(10, -1) == 9"],
        planner=["Subtract b from a.", "Return a plus b."],
        solver=[f"{FENCE}python\ndef add(a, b):\n    return a - b\n{FENCE}",
                f"{FENCE}python\ndef add(a, b):\n    return a + b\n{FENCE}"],
        critic=["The plan is wrong: the task asks for a sum."])
    events = run_multi(llm, item=item, task=task)

    assert [role for role, _, _ in llm.calls] == ["tester", "planner", "solver", "critic",
                                                  "planner", "solver"]
    assert messages(events)[0][:2] == (1, "tester")
    results = [e for e in events if isinstance(e, TestResult)]
    assert [(r.source, r.ok) for r in results] == [("own_tests", False), ("own_tests", True)]
    assert "AssertionError" in llm.calls[3][1]                   # critic sees the failure
    assert "return a + b" in events[-1].answer
    assert all("def check(candidate)" not in prompt for _, prompt, _ in llm.calls)


def test_runner_scores_multi_agent_answer(tmp_path):
    setup = Setup(task=Game24Task(), split="dev", model="fake", machine_id="test",
                  base_seed=2026, llm_settings={"temperature": 0.7},
                  approach_configs={"multi_agent": {"max_rounds": 2}})
    path = tmp_path / "trace.jsonl"
    llm = ScriptedLLM(planner=["A", "B"], solver=[WRONG, RIGHT], critic=["bad"])
    run_experiment(setup, [MultiAgent()], [ITEM], 1, llm, path, log=lambda s: None)
    [run] = group_runs(read_trace(path)).values()
    end = run[-1][1]
    assert isinstance(end, RunEnd)
    assert (end.status, end.correct, end.llm_calls, end.answer) == ("ok", True, 5, RIGHT)
    assert run[0][1].config["approach"] == {"max_rounds": 2}
