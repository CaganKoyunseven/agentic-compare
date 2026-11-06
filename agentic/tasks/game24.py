"""Game of 24 task with exact fraction checking.

Rules: use each of the four numbers exactly once with + - * / and parentheses to make 24.
Answers are parsed with Python's ast module and evaluated by our own small walker using
fractions.Fraction. Model output is never passed to eval(); anything other than integers and
the four binary operators is rejected.

Data: data/game24/24.csv from the Tree of Thoughts authors (MIT licence, see
scripts/fetch_data.py). Splits: data/splits/game24.json.
"""

import ast
import csv
import functools
import json
import operator
import re
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

from agentic.config import ROOT
from agentic.tasks.base import Feedback, Item, Score

DATA_FILE = ROOT / "data" / "game24" / "24.csv"
SPLITS_FILE = ROOT / "data" / "splits" / "game24.json"
TARGET = Fraction(24)
MAX_ANSWER_CHARS = 200
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv}
UNICODE_OPS = {"×": "*", "·": "*", "∗": "*", "÷": "/", "−": "-", "–": "-", "\\times": "*",
               "\\cdot": "*", "\\div": "/"}

# Difficulty from the number of solution paths (see count_solutions). The thresholds are the
# tertiles over all 1362 puzzles (33rd percentile 8, 67th percentile 20 paths), so each tier
# holds about a third. Path count correlates with the human solve rate in the CSV
# (Spearman 0.72), a sanity check that it measures difficulty.
EASY_MIN_PATHS = 21
HARD_MAX_PATHS = 8


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    message: str
    value: Fraction | None = None


def normalize(text: str) -> str:
    for old, new in UNICODE_OPS.items():
        text = text.replace(old, new)
    return text


def check_expression(numbers: list[int], expr: str) -> CheckResult:
    """Check one answer exactly. Returns ok, a message for the model, and the value."""
    expr = normalize(expr).strip()
    if not expr:
        return CheckResult(False, "empty answer")
    if len(expr) > MAX_ANSWER_CHARS:
        return CheckResult(False, "answer too long")
    try:
        tree = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return CheckResult(False, "could not parse the expression")

    leaves: list[int] = []
    problem = _collect_leaves(tree, leaves)
    if problem:
        return CheckResult(False, f"not allowed: {problem}; use only integers and + - * /")
    if sorted(leaves) != sorted(numbers):
        return CheckResult(False, f"uses numbers {sorted(leaves)}, must use "
                                  f"{sorted(numbers)} exactly once each")
    try:
        value = _evaluate(tree)
    except ZeroDivisionError:
        return CheckResult(False, "division by zero")
    if value != TARGET:
        return CheckResult(False, f"evaluates to {value}, not 24", value)
    return CheckResult(True, "equals 24", value)


def _collect_leaves(node: ast.AST, leaves: list[int]) -> str | None:
    """Append the integer leaves; return a description of the first forbidden node."""
    if isinstance(node, ast.BinOp):
        if type(node.op) not in OPS:
            return f"operator {type(node.op).__name__}"
        return _collect_leaves(node.left, leaves) or _collect_leaves(node.right, leaves)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        leaves.append(node.value)
        return None
    if isinstance(node, ast.UnaryOp):
        return "unary minus or plus"
    if isinstance(node, ast.Constant):
        return f"number {node.value!r}"
    return type(node).__name__


def _evaluate(node: ast.AST) -> Fraction:
    if isinstance(node, ast.BinOp):
        return OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    return Fraction(node.value)


# ---- Solver (used for difficulty tiers and to pick solvable demo puzzles) ----

def _solution_paths(values: list[tuple[Fraction, str]]) -> Iterator[str]:
    """Yield an expression for every way to reach 24 by the ToT moves: pick two numbers,
    combine them with one operation, repeat until one number is left."""
    if len(values) == 1:
        if values[0][0] == TARGET:
            yield values[0][1]
        return
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            (a, ea), (b, eb) = values[i], values[j]
            rest = [v for k, v in enumerate(values) if k not in (i, j)]
            moves = [(a + b, f"({ea}+{eb})"), (a * b, f"({ea}*{eb})"),
                     (a - b, f"({ea}-{eb})"), (b - a, f"({eb}-{ea})")]
            if b != 0:
                moves.append((a / b, f"({ea}/{eb})"))
            if a != 0:
                moves.append((b / a, f"({eb}/{ea})"))
            for move in moves:
                yield from _solution_paths(rest + [move])


def _start(numbers: list[int]) -> list[tuple[Fraction, str]]:
    return [(Fraction(n), str(n)) for n in numbers]


def count_solutions(numbers: list[int]) -> int:
    """Number of step-by-step solution paths. More paths means an easier puzzle."""
    return _count_cached(tuple(sorted(numbers)))


@functools.cache
def _count_cached(numbers: tuple[int, ...]) -> int:
    # Loading all puzzles counts 1362 puzzles (a few seconds); the cache makes reloads instant.
    return sum(1 for _ in _solution_paths(_start(list(numbers))))


def find_solution(numbers: list[int]) -> str | None:
    expr = next(_solution_paths(_start(numbers)), None)
    return expr[1:-1] if expr else None   # drop the outer parentheses


def difficulty(paths: int) -> str:
    if paths >= EASY_MIN_PATHS:
        return "easy"
    if paths <= HARD_MAX_PATHS:
        return "hard"
    return "medium"


# ---- Task ----

_RUN = re.compile(r"[\d\s+\-*/()]+")


class Game24Task:
    name = "game24"

    def load_items(self, split: str) -> list[Item]:
        if split not in ("all", "dev", "test"):
            raise ValueError(f"unknown split {split!r}; use 'dev', 'test' or 'all'")
        items = []
        with open(DATA_FILE, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                numbers = [int(x) for x in row["Puzzles"].split()]
                items.append(Item(item_id=f"game24/{int(row['Rank']):04d}", task=self.name,
                                  payload={"numbers": numbers},
                                  difficulty=difficulty(count_solutions(numbers))))
        if split == "all":
            return items
        if not SPLITS_FILE.exists():
            raise FileNotFoundError(f"{SPLITS_FILE} is missing; run python -m scripts.make_splits")
        wanted = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))[split]
        by_id = {it.item_id: it for it in items}
        return [by_id[item_id] for item_id in wanted]

    def build_prompt(self, item: Item) -> str:
        numbers = " ".join(str(n) for n in item.payload["numbers"])
        return (f"Use the numbers {numbers} and the operations + - * / to make 24. "
                "Use each number exactly once. You may use parentheses. "
                "Write your final answer on the last line as a single expression.")

    def extract_answer(self, item: Item, model_text: str) -> str:
        """Last expression in the reply, preferring one that uses all the numbers.
        Removes markdown and LaTeX marks and a trailing '= 24'."""
        text = normalize(model_text)
        for mark in ("**", "`", "$"):
            text = text.replace(mark, "")
        want = len(item.payload["numbers"])
        candidates = []
        for line in reversed(text.splitlines()):
            for part in line.split("="):
                for run in _RUN.findall(part):
                    run = run.strip()
                    if re.search(r"\d", run) and re.search(r"[+\-*/]", run):
                        candidates.append(run)
        for cand in candidates:
            if len(re.findall(r"\d+", cand)) == want:
                return cand
        return candidates[0] if candidates else ""

    def feedback(self, item: Item, answer: str, tests: str | None = None) -> Feedback:
        """Verifier result an approach may see: ok and why not."""
        result = check_expression(item.payload["numbers"], answer)
        return Feedback(ok=result.ok, message=result.message,
                        details={"value": str(result.value) if result.value is not None
                                 else None})

    def score(self, item: Item, answer: str) -> Score:
        result = check_expression(item.payload["numbers"], answer)
        return Score(correct=result.ok, message="passed" if result.ok else "failed",
                     details={"reason": result.message})
