"""Multi-agent approach: planner, solver and critic roles.

One round:
  1. Planner: writes a short plan, no final answer.
  2. Solver: follows the plan and writes the answer.
  3. Check: the same feedback check as the reflective approach (Game of 24: verifier;
     HumanEval: docstring examples plus own tests). Never the official tests.
  4. Critic: only if the check failed, explains what went wrong with the plan and the answer.
In the next round the planner sees its old plan, the answer, the check result and the critic's
notes, and makes a new plan. At most `max_rounds` rounds; stops early when the check passes.
The final answer is the passing one, or the last one when none passes.

HumanEval only: before the first round, the tester (the critic's second job) writes assert
lines from the task alone, with the same prompt as the reflective approach, so both approaches
get exactly the same kind of feedback.

Every agent call starts fresh: an agent sees only what its prompt passes on, not a shared chat.
"""

from collections.abc import Iterator
from typing import Any

from agentic.approaches.base import Caller, load_prompt, user
from agentic.approaches.reflective import FEEDBACK_SOURCE, extract_asserts, task_fields
from agentic.events import AgentMessage, Body, FinalAnswer, TestResult
from agentic.llm.base import LLMClient
from agentic.tasks.base import Item, Task

DEFAULTS = {"max_rounds": 2}


class MultiAgent:
    name = "multi_agent"

    def run(self, item: Item, task: Task, llm: LLMClient, *, seed: int,
            config: dict[str, Any]) -> Iterator[Body]:
        settings = {**DEFAULTS, **config}
        values = {**task_fields(item, task), "task_prompt": task.build_prompt(item)}
        caller = Caller(llm, seed)

        def ask(role: str, prompt_name: str, round_no: int) -> Iterator[Body]:
            """One agent turn: the LLM call and the chat message it becomes."""
            call = caller.chat(role, user(load_prompt(prompt_name).format(**values)))
            yield call
            yield AgentMessage(round=round_no, role=role, text=call.response,
                               call_id=call.call_id)
            values[role] = call.response

        tests = None
        if task.name == "humaneval":
            yield from ask("tester", "reflective_tests_humaneval", 1)
            tests = extract_asserts(values["tester"])

        answer = ""
        for round_no in range(1, settings["max_rounds"] + 1):
            planner_prompt = "planner" if round_no == 1 else "replan"
            yield from ask("planner", f"multi_agent_{planner_prompt}_{task.name}", round_no)
            yield from ask("solver", f"multi_agent_solver_{task.name}", round_no)
            answer = task.extract_answer(item, values["solver"])
            feedback = task.feedback(item, answer, tests)
            yield TestResult(source=FEEDBACK_SOURCE[task.name], ok=feedback.ok,
                             message=feedback.message, details=feedback.details,
                             attempt_no=round_no)
            if feedback.ok or round_no == settings["max_rounds"]:
                break
            values.update(answer=answer, feedback=feedback.message)
            yield from ask("critic", f"multi_agent_critic_{task.name}", round_no)

        yield FinalAnswer(answer=answer)
