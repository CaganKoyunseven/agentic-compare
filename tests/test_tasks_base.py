"""Tests for the task contract in agentic.tasks.base."""

import dataclasses

import pytest

from agentic.tasks.base import Feedback, Item, Score, Task


class EchoTask:
    """Smallest possible task, used to check that the Protocol is usable."""

    name = "echo"

    def load_items(self, split: str) -> list[Item]:
        return [Item(item_id="echo/0", task="echo", payload={"text": "hi"})]

    def build_prompt(self, item: Item) -> str:
        return item.payload["text"]

    def extract_answer(self, item: Item, model_text: str) -> str:
        return model_text.strip()

    def feedback(self, item: Item, answer: str, tests: str | None = None) -> Feedback:
        return Feedback(ok=answer == "hi", message="ok" if answer == "hi" else "wrong")

    def score(self, item: Item, answer: str) -> Score:
        return Score(correct=answer == "hi", message="passed" if answer == "hi" else "failed")


def test_defaults():
    item = Item(item_id="x/1", task="game24", payload={"numbers": [1, 2, 3, 4]})
    assert item.difficulty is None
    assert Feedback(ok=True, message="ok").details == {}
    assert Score(correct=False, message="failed").details == {}


def test_details_are_not_shared():
    a = Feedback(ok=True, message="ok")
    b = Feedback(ok=True, message="ok")
    assert a.details is not b.details


@pytest.mark.parametrize(
    "obj, attr",
    [
        (Item(item_id="x/1", task="game24", payload={}), "item_id"),
        (Feedback(ok=True, message="ok"), "ok"),
        (Score(correct=True, message="passed"), "correct"),
    ],
)
def test_dataclasses_are_frozen(obj, attr):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, attr, None)


def test_protocol_round_trip():
    task: Task = EchoTask()
    item = task.load_items("dev")[0]
    answer = task.extract_answer(item, " " + task.build_prompt(item) + "\n")
    assert task.feedback(item, answer).ok
    assert task.score(item, answer) == Score(correct=True, message="passed")
