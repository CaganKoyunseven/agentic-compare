"""Baseline approach: one call, first answer."""

from collections.abc import Iterator
from typing import Any

from agentic.approaches.base import Caller
from agentic.events import Body, FinalAnswer
from agentic.llm.base import LLMClient
from agentic.tasks.base import Item, Task


class Baseline:
    name = "baseline"

    def run(self, item: Item, task: Task, llm: LLMClient, *, seed: int,
            config: dict[str, Any]) -> Iterator[Body]:
        caller = Caller(llm, seed)
        call = caller.chat("solver", [{"role": "user", "content": task.build_prompt(item)}])
        yield call
        yield FinalAnswer(answer=task.extract_answer(item, call.response))
