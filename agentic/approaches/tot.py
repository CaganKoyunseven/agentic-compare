"""Tree of Thoughts (breadth-first search with LLM scoring).

Game of 24, following the ToT paper (Yao et al., 2023) and its code
(princeton-nlp/tree-of-thought-llm: scripts/game24/bfs.sh, src/tot/methods/bfs.py):
  1. Proposer: for every kept node, one call lists possible next steps such as
     "4 + 8 = 12 (left: 6 12)". Every line of this form becomes a child (`breadth` = 0, as in
     the paper), or only the first `breadth` lines. A line repeated under the same parent is
     skipped (the paper gives such duplicates the value 0).
  2. Scorer: one call per child asks for `value_samples` judgements (3 in the paper, in one
     request) of whether the numbers left can still reach 24. Like the paper, the labels are
     weighted (sure 20, likely 1, impossible 0.001) and summed. A state already scored in this
     run reuses its score without a new call (the paper caches values too).
  3. All children of the level are ranked by score and only the best `keep` survive (5 in
     the paper).
  Three levels (four numbers, three steps). Then a writer call turns the steps of one path
  into one expression, which becomes the final answer. Which path, set by `game24_writer`:
  - "reached_24" (as in the paper's code): the best-scored kept leaf whose numbers left are
    exactly 24. When no kept leaf reached 24 there is no writer call and no answer (wrong).
  - "best_path" (our first, pre-registered test runs): the best node, even when its numbers
    left are not 24; the writer then often wrote a new answer that ignored the tree.
  The paper also lets the model judge the written answers; we do not (one writer call).

Pure ToT: the model's own steps and "left" numbers are used as written. Code only reads the
format; it never checks the arithmetic and never feeds the verifier result back.

HumanEval, two levels (plans, then code):
  0. Tester: one call writes assert lines from the task alone (same prompt as the reflective
     and multi-agent approaches), so all approaches get the same kind of feedback.
  1. Proposer: `humaneval_breadth` separate calls, one plan each (no code). Scorer: one call
     per plan (sure / likely / impossible). Only the best `humaneval_keep` plans survive.
  2. Coder: the kept plans are turned into code in score order; each code is checked with the
     docstring examples plus the own tests. The first code that passes is chosen and the
     search stops. When none passes, the code of the best plan is chosen. Failed codes that
     were not chosen are pruned. The official tests are never used.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from agentic.approaches.base import Caller, load_prompt
from agentic.approaches.base import user as _user
from agentic.approaches.reflective import extract_asserts
from agentic.events import (
    Body,
    FinalAnswer,
    NodeAdded,
    NodePruned,
    NodeScored,
    PathChosen,
    TestResult,
)
from agentic.llm.base import LLMClient
from agentic.tasks.base import Item, Task
from agentic.tasks.game24 import TARGET, normalize

SCORE_OF_LABEL = {"sure": 1.0, "likely": 0.5, "impossible": 0.0}      # HumanEval plans
PAPER_VALUE = {"sure": 20.0, "likely": 1.0, "impossible": 0.001}       # Game of 24 states
DEFAULTS = {"breadth": 0, "keep": 5, "value_samples": 3, "humaneval_breadth": 3,
            "humaneval_keep": 2, "game24_writer": "reached_24"}
ALL_LINES = 100        # breadth 0: no limit in practice (the model writes about 6 lines)
WRITER_MODES = ("reached_24", "best_path")

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


def reached_24(state: str) -> bool:
    """True when the numbers left, as the model wrote them, are the single number 24."""
    parts = state.split()
    if len(parts) != 1:
        return False
    try:
        return Fraction(parts[0]) == TARGET
    except (ValueError, ZeroDivisionError):
        return False


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
        settings = {**DEFAULTS, **config}
        caller = Caller(llm, seed)
        if task.name == "game24":
            yield from self._game24(item, task, caller, settings)
        elif task.name == "humaneval":
            yield from self._humaneval(item, task, caller, settings)
        else:
            raise NotImplementedError(f"ToT for {task.name} is not built yet")

    def _game24(self, item: Item, task: Task, caller: Caller,
                settings: dict[str, Any]) -> Iterator[Body]:
        numbers = " ".join(str(n) for n in item.payload["numbers"])
        writer = settings["game24_writer"]
        if writer not in WRITER_MODES:
            raise ValueError(f"game24_writer must be one of {WRITER_MODES}, not {writer!r}")
        last_depth = len(item.payload["numbers"]) - 1

        root = Node("root", None, 0, "", numbers)
        yield NodeAdded(node_id="root", parent_id=None, depth=0, step="", state=numbers)
        frontier = [root]
        best = root
        made = 0   # nodes created so far, for the ids n1, n2, ...
        limit = settings["breadth"] or ALL_LINES
        cache: dict[str, tuple[float, str]] = {}     # state -> (score, label) in this run
        for depth in range(1, last_depth + 1):
            children: list[Node] = []
            for parent in frontier:
                call = caller.chat("proposer", _user(load_prompt("game24_tot_propose")
                                                     .format(state=parent.state)))
                yield call
                seen = set()
                for step, state in parse_steps(call.response, limit):
                    if (step, state) in seen:
                        continue
                    seen.add((step, state))
                    made += 1
                    child = Node(f"n{made}", parent, depth, step, state)
                    children.append(child)
                    yield NodeAdded(node_id=child.node_id, parent_id=parent.node_id,
                                    depth=depth, step=step, state=state, call_id=call.call_id)
            if not children:
                break   # the model proposed nothing readable; keep the best node so far
            for child in children:
                if child.state in cache:
                    child.score, label = cache[child.state]
                    yield NodeScored(node_id=child.node_id, score=child.score,
                                     label=f"{label} (cached)")
                    continue
                call = caller.sample("scorer", _user(load_prompt("game24_tot_value")
                                                     .format(state=child.state)),
                                     settings["value_samples"])
                yield call
                labels = [parse_label(text) for text in call.responses or [call.response]]
                child.score = sum(PAPER_VALUE.get(lab, 0.0) for lab in labels)
                label = ", ".join(labels)
                cache[child.state] = (child.score, label)
                yield NodeScored(node_id=child.node_id, score=child.score, label=label,
                                 call_id=call.call_id)
            ranked = sorted(children, key=lambda n: -n.score)   # stable: ties keep order
            for node in ranked[settings["keep"]:]:
                yield NodePruned(node_id=node.node_id)
            frontier = ranked[:settings["keep"]]
            best = frontier[0]

        if writer == "reached_24":
            # frontier is in score order, so the first leaf at 24 is the best-scored one.
            done = [n for n in frontier if n.depth == last_depth and reached_24(n.state)]
            if not done:
                yield PathChosen(node_ids=[], answer="")
                yield FinalAnswer(answer="")
                return
            best = done[0]

        path = best.path()
        steps = "\n".join(f"{n.step} (left: {n.state})" for n in path[1:])
        call = caller.chat("writer", _user(load_prompt("game24_tot_write")
                                           .format(numbers=numbers, steps=steps)))
        yield call
        answer = task.extract_answer(item, call.response)
        yield PathChosen(node_ids=[n.node_id for n in path], answer=answer)
        yield FinalAnswer(answer=answer)

    def _humaneval(self, item: Item, task: Task, caller: Caller,
                   settings: dict[str, Any]) -> Iterator[Body]:
        prompt = item.payload["prompt"]
        yield NodeAdded(node_id="root", parent_id=None, depth=0, step="", state="task")

        call = caller.chat("tester", _user(load_prompt("reflective_tests_humaneval")
                                           .format(prompt=prompt)))
        yield call
        tests = extract_asserts(call.response)

        # Level 1: plans, each from its own call, then one LLM score per plan.
        plans: list[Node] = []
        for i in range(1, settings["humaneval_breadth"] + 1):
            call = caller.chat("proposer", _user(load_prompt("humaneval_tot_plan")
                                                 .format(prompt=prompt)))
            yield call
            plan = Node(f"p{i}", None, 1, call.response.strip(), "plan")
            plans.append(plan)
            yield NodeAdded(node_id=plan.node_id, parent_id="root", depth=1, step=plan.step,
                            state="plan", call_id=call.call_id)
        for plan in plans:
            call = caller.chat("scorer", _user(load_prompt("humaneval_tot_value")
                                               .format(prompt=prompt, plan=plan.step)))
            yield call
            label = parse_label(call.response)
            plan.score = SCORE_OF_LABEL.get(label, 0.0)
            yield NodeScored(node_id=plan.node_id, score=plan.score, label=label,
                             call_id=call.call_id)
        ranked = sorted(plans, key=lambda n: -n.score)   # stable: ties keep order
        for plan in ranked[settings["humaneval_keep"]:]:
            yield NodePruned(node_id=plan.node_id)

        # Level 2: code from the kept plans, best plan first, until one passes the own tests.
        codes: list[tuple[str, str, str, bool]] = []     # (plan id, code id, answer, passed)
        for i, plan in enumerate(ranked[:settings["humaneval_keep"]], start=1):
            call = caller.chat("coder", _user(load_prompt("humaneval_tot_code")
                                              .format(prompt=prompt, plan=plan.step)))
            yield call
            answer = task.extract_answer(item, call.response)
            node_id = f"k{i}"
            code = answer.removeprefix(prompt + "\n")    # show only the new code in the tree
            yield NodeAdded(node_id=node_id, parent_id=plan.node_id, depth=2, step="code",
                            state=code, call_id=call.call_id)
            feedback = task.feedback(item, answer, tests)
            yield TestResult(source="own_tests", ok=feedback.ok, message=feedback.message,
                             details=feedback.details, node_id=node_id)
            yield NodeScored(node_id=node_id, score=1.0 if feedback.ok else 0.0,
                             label="tests passed" if feedback.ok else "tests failed")
            codes.append((plan.node_id, node_id, answer, feedback.ok))
            if feedback.ok:
                break

        # The passing code, or the code of the best plan when none passes.
        plan_id, chosen_id, answer, _ = next((c for c in codes if c[3]), codes[0])
        for _, node_id, _, passed in codes:
            if not passed and node_id != chosen_id:
                yield NodePruned(node_id=node_id)
        yield PathChosen(node_ids=["root", plan_id, chosen_id], answer=answer)
        yield FinalAnswer(answer=answer)
