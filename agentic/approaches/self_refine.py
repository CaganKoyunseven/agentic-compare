"""Self-Refine (Madaan et al., 2023), Game of 24 only: the Reflective loop without the verifier.

Same shape and limit as our Reflective approach, but the check is the model's own:
  1. Solver: one call answers the task, like the baseline.
  2. Self-check: a critic call checks the answer itself (no verifier result is given), explains
     what is wrong and ends with "VERDICT: CORRECT" or "VERDICT: WRONG".
  3. If the model says CORRECT, stop. Otherwise a reviser call writes a new answer from the old
     answer and the review.
At most `max_attempts` answers. The final answer is the one the model accepted, or the last one.
A reply without a verdict line counts as WRONG (keep refining).

Part of the feedback study (separate from the main results): compared with Reflective, it
shows how much of the gain comes from the exact verifier and how much from the loop itself.
"""

import re
from collections.abc import Iterator
from typing import Any

from agentic.approaches.base import Caller, load_prompt, user
from agentic.events import Attempt, Body, Critique, FinalAnswer, TestResult
from agentic.llm.base import LLMClient
from agentic.tasks.base import Item, Task

DEFAULTS = {"max_attempts": 3}
_VERDICT = re.compile(r"VERDICT:\s*(CORRECT|WRONG)", re.IGNORECASE)


def verdict(text: str) -> bool:
    """True when the last verdict in the reply is CORRECT."""
    found = _VERDICT.findall(text)
    return bool(found) and found[-1].upper() == "CORRECT"


class SelfRefine:
    name = "self_refine"

    def run(self, item: Item, task: Task, llm: LLMClient, *, seed: int,
            config: dict[str, Any]) -> Iterator[Body]:
        if task.name != "game24":
            raise NotImplementedError("Self-Refine is built for Game of 24 only")
        settings = {**DEFAULTS, **config}
        numbers = " ".join(str(n) for n in item.payload["numbers"])
        caller = Caller(llm, seed)

        call = caller.chat("solver", user(task.build_prompt(item)))
        yield call
        answer = task.extract_answer(item, call.response)
        attempt_no = 1
        while True:
            yield Attempt(attempt_no=attempt_no, answer=answer, call_id=call.call_id)
            call = caller.chat("critic", user(load_prompt("self_refine_feedback_game24")
                                              .format(numbers=numbers, answer=answer)))
            yield call
            yield Critique(attempt_no=attempt_no, text=call.response, call_id=call.call_id)
            accepted = verdict(call.response)
            yield TestResult(source="self_check", ok=accepted,
                             message="model says correct" if accepted else "model says wrong",
                             attempt_no=attempt_no)
            if accepted or attempt_no == settings["max_attempts"]:
                break
            call = caller.chat("reviser", user(load_prompt("self_refine_refine_game24").format(
                numbers=numbers, answer=answer, critique=call.response)))
            yield call
            answer = task.extract_answer(item, call.response)
            attempt_no += 1

        yield FinalAnswer(answer=answer)
