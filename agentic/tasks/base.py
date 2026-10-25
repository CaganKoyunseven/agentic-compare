"""Common interface for tasks.

Approaches see only Item.payload and Feedback. Official tests and reference solutions stay
private inside each task object and are used only by score().
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Item:
    item_id: str                      # stable id, e.g. "HumanEval/12"
    task: str                         # "humaneval" or "game24"
    payload: dict[str, Any]           # JSON-serializable, task specific
    difficulty: str | None = None     # "easy" | "medium" | "hard" | None


@dataclass(frozen=True)
class Feedback:
    ok: bool                          # attempt passed the feedback checks
    message: str                      # text an approach may show to the model
    details: dict[str, Any] = field(default_factory=dict)  # JSON-serializable, for traces and UI


@dataclass(frozen=True)
class Score:
    correct: bool                     # official final verdict
    message: str                      # short: "passed", "failed", "timeout" (never test source)
    details: dict[str, Any] = field(default_factory=dict)


class Task(Protocol):
    name: str
    def load_items(self, split: str) -> list[Item]: ...      # "dev" | "test" | "all"
    def build_prompt(self, item: Item) -> str: ...            # task statement, no hints
    def extract_answer(self, item: Item, model_text: str) -> str: ...
    def feedback(self, item: Item, answer: str, tests: str | None = None) -> Feedback: ...
    def score(self, item: Item, answer: str) -> Score: ...
