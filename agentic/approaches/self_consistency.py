"""Self-consistency (Wang et al., 2023), Game of 24 only.

A control for Tree of Thoughts: many LLM calls (`samples`, 29 = the call budget of our first,
lighter ToT on Game of 24), but no search and no scoring. Each call is the baseline prompt on its own, with
its own seed. The final answer is chosen by majority vote over the answers, compared after
light normalisation (spaces and extra parentheses do not matter, and the operands of + and *
may come in any order). No verifier is used: the vote only counts what the model said most.
Ties go to the answer group that appeared first.

This approach is a separate study: it never enters the main results or the live demo.
"""

import ast
from collections.abc import Iterator
from typing import Any

from agentic.approaches.base import Caller
from agentic.events import Body, FinalAnswer
from agentic.llm.base import LLMClient
from agentic.tasks.base import Item, Task
from agentic.tasks.game24 import normalize

DEFAULTS = {"samples": 29}
_SYMBOL = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


def canonical(expr: str) -> str:
    """A key that is equal for answers that differ only in spaces, extra parentheses or the
    order of the operands of + and *. Text that is not a readable expression is kept as it
    is, without spaces."""
    text = normalize(expr).strip()
    if not text:
        return ""
    try:
        return _key(ast.parse(text, mode="eval").body)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return "".join(text.split())


def _key(node: ast.AST) -> str:
    if isinstance(node, ast.BinOp) and type(node.op) in _SYMBOL:
        symbol = _SYMBOL[type(node.op)]
        if isinstance(node.op, (ast.Add, ast.Mult)):
            parts = sorted(_key(n) for n in _flatten(node, type(node.op)))
            return "(" + symbol.join(parts) + ")"
        return f"({_key(node.left)}{symbol}{_key(node.right)})"
    return ast.unparse(node)


def _flatten(node: ast.AST, op: type) -> list[ast.AST]:
    """Operands of a chain of the same operator: 1 + (2 + 3) -> [1, 2, 3]."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, op):
        return _flatten(node.left, op) + _flatten(node.right, op)
    return [node]


def vote(answers: list[str]) -> int | None:
    """Index of the answer that wins the majority vote (the first answer of the largest
    group; ties go to the group seen first). Empty answers do not vote. None when no
    answer is left."""
    groups: dict[str, list[int]] = {}
    for i, answer in enumerate(answers):
        key = canonical(answer)
        if key:
            groups.setdefault(key, []).append(i)
    if not groups:
        return None
    # dicts keep insertion order and max() keeps the first maximum: ties go to the earliest.
    return max(groups.values(), key=len)[0]


class SelfConsistency:
    name = "self_consistency"

    def run(self, item: Item, task: Task, llm: LLMClient, *, seed: int,
            config: dict[str, Any]) -> Iterator[Body]:
        if task.name != "game24":
            raise NotImplementedError("Self-consistency is built for Game of 24 only")
        settings = {**DEFAULTS, **config}
        caller = Caller(llm, seed)
        prompt = [{"role": "user", "content": task.build_prompt(item)}]
        answers = []
        for _ in range(settings["samples"]):
            call = caller.chat("sampler", prompt)
            yield call
            answers.append(task.extract_answer(item, call.response))
        winner = vote(answers)
        yield FinalAnswer(answer="" if winner is None else answers[winner])
