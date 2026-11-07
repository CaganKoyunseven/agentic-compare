"""Tree of Thoughts (breadth-first search with LLM scoring).

Game of 24, following the ToT paper (Yao et al., 2023):
  1. Proposer: for every kept node, one call lists possible next steps such as
     "4 + 8 = 12 (left: 6 12)". The first `breadth` lines that have this form become children.
  2. Scorer: one call per child judges whether the numbers left can still reach 24
     (sure / likely / impossible).
  3. All children of the level are ranked by score and only the best `keep` survive.
  Three levels (four numbers, three steps). Then a writer call turns the steps of the best
  node into one expression, which becomes the final answer.

Pure ToT: the model's own steps and "left" numbers are used as written. Code only reads the
format; it never checks the arithmetic and never feeds the verifier result back.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from agentic.approaches.base import Caller, load_prompt
from agentic.approaches.base import user as _user
from agentic.events import (
    Body,
    FinalAnswer,
    NodeAdded,
    NodePruned,
    NodeScored,
    PathChosen,
)
from agentic.llm.base import LLMClient
from agentic.tasks.base import Item, Task
from agentic.tasks.game24 import normalize

SCORE_OF_LABEL = {"sure": 1.0, "likely": 0.5, "impossible": 0.0}
DEFAULTS = {"breadth": 3, "keep": 3}

# "4 + 8 = 12 (left: 6 12)" -> step "4 + 8 = 12", state "6 12"
_STEP = re.compile(r"^\s*(?:[-*]\s*|\d+[.)]\s+)?(.+?=\s*[^\s(]+)\s*\(\s*left:\s*([^)]*)\)",
                   re.IGNORECASE)
_LABEL = re.compile(r"\b(sure|likely|impossible)\b", re.IGNORECASE)


def clean_line(line: str) -> str:
    """Remove markdown and LaTeX marks so the step format can be read. Only the format is
    touched; the numbers stay exactly as the model wrote them."""
    line = normalize(line)
    for mark in ("\\(", "\\)", "\\[", "\\]", "**", "`", "$"):
        line = line.replace(mark, "")
    return line


def parse_steps(text: str, limit: int) -> list[tuple[str, str]]:
    """The first `limit` lines of the form 'a op b = c (left: ...)' as (step, state)."""
    steps = []
    for line in text.splitlines():
        match = _STEP.match(clean_line(line))
        if match:
            step = match.group(1).split(":")[-1].strip()      # drop "Add 3 to 12:" prefixes
            state = " ".join(match.group(2).replace(",", " ").split())
            steps.append((step, state))
        if len(steps) == limit:
            break
    return steps


def parse_label(text: str) -> str:
    """Last sure/likely/impossible in the reply, or 'unknown' when there is none."""
    labels = _LABEL.findall(text)
    return labels[-1].lower() if labels else "unknown"


@dataclass
class Node:
    node_id: str
    parent: "Node | None"
    depth: int
    step: str
    state: str
    score: float = 0.0

    def path(self) -> list["Node"]:
        """Nodes from the root down to this one."""
        nodes, node = [], self
        while node is not None:
            nodes.insert(0, node)
            node = node.parent
        return nodes


class TreeOfThoughts:
    name = "tot"

    def run(self, item: Item, task: Task, llm: LLMClient, *, seed: int,
            config: dict[str, Any]) -> Iterator[Body]:
        if task.name != "game24":
            raise NotImplementedError(f"ToT for {task.name} is not built yet")
        settings = {**DEFAULTS, **config}
        caller = Caller(llm, seed)
        numbers = " ".join(str(n) for n in item.payload["numbers"])

        root = Node("root", None, 0, "", numbers)
        yield NodeAdded(node_id="root", parent_id=None, depth=0, step="", state=numbers)
        frontier = [root]
        best = root
        made = 0   # nodes created so far, for the ids n1, n2, ...
        for depth in range(1, len(item.payload["numbers"])):
            children: list[Node] = []
            for parent in frontier:
                call = caller.chat("proposer", _user(load_prompt("game24_tot_propose")
                                                     .format(state=parent.state)))
                yield call
                for step, state in parse_steps(call.response, settings["breadth"]):
                    made += 1
                    child = Node(f"n{made}", parent, depth, step, state)
                    children.append(child)
                    yield NodeAdded(node_id=child.node_id, parent_id=parent.node_id,
                                    depth=depth, step=step, state=state, call_id=call.call_id)
            if not children:
                break   # the model proposed nothing readable; keep the best node so far
            for child in children:
                call = caller.chat("scorer", _user(load_prompt("game24_tot_value")
                                                   .format(state=child.state)))
                yield call
                label = parse_label(call.response)
                child.score = SCORE_OF_LABEL.get(label, 0.0)
                yield NodeScored(node_id=child.node_id, score=child.score, label=label,
                                 call_id=call.call_id)
            ranked = sorted(children, key=lambda n: -n.score)   # stable: ties keep order
            for node in ranked[settings["keep"]:]:
                yield NodePruned(node_id=node.node_id)
            frontier = ranked[:settings["keep"]]
            best = frontier[0]

        path = best.path()
        steps = "\n".join(f"{n.step} (left: {n.state})" for n in path[1:])
        call = caller.chat("writer", _user(load_prompt("game24_tot_write")
                                           .format(numbers=numbers, steps=steps)))
        yield call
        answer = task.extract_answer(item, call.response)
        yield PathChosen(node_ids=[n.node_id for n in path], answer=answer)
        yield FinalAnswer(answer=answer)
