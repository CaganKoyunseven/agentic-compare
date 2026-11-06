"""Common interface for approaches.

An approach is a generator of events: it calls the LLM, yields what happened, and ends with
FinalAnswer. It never scores its own answer and never sees the official tests; the runner
adds RunStart and RunEnd around it.
"""

from collections.abc import Iterator
from typing import Any, Protocol

from agentic.events import Body, LLMCall
from agentic.llm.base import LLMClient
from agentic.tasks.base import Item, Task


class Approach(Protocol):
    name: str

    def run(self, item: Item, task: Task, llm: LLMClient, *, seed: int,
            config: dict[str, Any]) -> Iterator[Body]: ...


class Caller:
    """Makes the LLM calls of one run. Gives each call an id and the seed seed + k, and
    returns the LLMCall event, which the approach yields so the call is counted."""

    def __init__(self, llm: LLMClient, seed: int):
        self.llm = llm
        self.seed = seed
        self.count = 0

    def chat(self, role: str, messages: list[dict[str, str]]) -> LLMCall:
        seed = self.seed + self.count
        call_id = f"call-{self.count}"
        self.count += 1
        result = self.llm.chat(messages, seed=seed)
        return LLMCall(call_id=call_id, role=role, messages=messages, response=result.text,
                       tokens_in=result.tokens_in, tokens_out=result.tokens_out,
                       wall_ms=result.wall_ms, load_ms=result.load_ms,
                       done_reason=result.done_reason, seed=seed)
