"""Tests for the Game of 24 task: safe checking, answer extraction, solver, loading."""

import json
from collections import Counter
from fractions import Fraction

import pytest

from agentic.tasks.base import Item
from agentic.tasks.game24 import (
    SPLITS_FILE,
    Game24Task,
    check_expression,
    count_solutions,
    find_solution,
)


def item(*numbers: int) -> Item:
    return Item(item_id="game24/test", task="game24", payload={"numbers": list(numbers)})


# ---- check_expression ----

@pytest.mark.parametrize("numbers, expr", [
    ((1, 2, 3, 4), "1*2*3*4"),
    ((2, 2, 6, 6), "2*6+2*6"),
    ((4, 9, 10, 13), "(10-4)*(13-9)"),
    ((3, 3, 8, 8), "8/(3-8/3)"),               # needs exact fractions
    ((1, 5, 5, 5), "5*(5-1/5)"),               # also fractions
    ((4, 9, 10, 13), "(10 − 4) × (13 − 9)"),   # unicode operators are normalized
])
def test_correct_expressions_pass(numbers, expr):
    result = check_expression(list(numbers), expr)
    assert result.ok, result.message
    assert result.value == Fraction(24)


@pytest.mark.parametrize("numbers, expr, words", [
    ((4, 9, 10, 13), "(13-9)*10-4", "36"),
    ((4, 9, 10, 13), "(10-4)*(13-9)*1", "numbers"),       # extra number
    ((4, 9, 10, 13), "(10-4)*4", "numbers"),              # missing / reused numbers
    ((1, 2, 3, 4), "4/(2-2)*1*3", "numbers"),
    ((1, 2, 3, 4), "4/(3-3)+1+2", "numbers"),
    ((1, 1, 2, 3), "3/(1-1)+2", "division by zero"),
    ((1, 1, 3, 4), "4*3/(1-1)", "division by zero"),
    ((1, 2, 3, 4), "2**3*1*4", "not allowed"),
    ((1, 2, 3, 4), "-1+2*3*4", "not allowed"),            # unary minus
    ((1, 2, 3, 4), "1.5*2*3*4", "not allowed"),
    ((1, 2, 3, 4), "__import__('os')", "not allowed"),
    ((1, 2, 3, 4), "x*2*3*4", "not allowed"),
    ((1, 2, 3, 4), "(1*2*3*4", "parse"),
    ((1, 2, 3, 4), "", "empty"),
    ((1, 2, 3, 4), "1+" * 200 + "1", "too long"),
])
def test_wrong_or_unsafe_expressions_fail_with_a_clear_message(numbers, expr, words):
    result = check_expression(list(numbers), expr)
    assert not result.ok
    assert words in result.message


# ---- extract_answer ----

@pytest.mark.parametrize("text, expected", [
    ("(10-4)*(13-9)", "(10-4)*(13-9)"),
    ("Answer: (10-4)*(13-9) = 24", "(10-4)*(13-9)"),
    ("Let me think.\n13-9=4\n10-4=6\n**(13-9)*(10-4) = 24**", "(13-9)*(10-4)"),
    ("```\n(13 - 9) * (10 - 4)\n```", "(13 - 9) * (10 - 4)"),
    ("The answer is $(13-9) \\times (10-4) = 24$.", "(13-9) * (10-4)"),
    ("I cannot find a solution.", ""),
])
def test_extract_answer_finds_the_last_expression(text, expected):
    assert Game24Task().extract_answer(item(4, 9, 10, 13), text) == expected


# ---- solver ----

def test_solver_counts_and_finds_solutions():
    assert count_solutions([1, 2, 3, 4]) > 0
    assert count_solutions([3, 3, 8, 8]) > 0
    assert count_solutions([1, 1, 1, 1]) == 0
    assert find_solution([1, 1, 1, 1]) is None
    for numbers in ([1, 2, 3, 4], [4, 9, 10, 13], [3, 3, 8, 8], [2, 2, 6, 6]):
        expr = find_solution(numbers)
        assert check_expression(numbers, expr).ok, (numbers, expr)


def test_harder_puzzles_have_fewer_solutions():
    assert count_solutions([3, 3, 8, 8]) < count_solutions([1, 2, 3, 4])


# ---- Task protocol ----

def test_feedback_and_score_agree_but_feedback_explains():
    task, it = Game24Task(), item(4, 9, 10, 13)
    fb = task.feedback(it, "(13-9)*10-4")
    assert not fb.ok and "36" in fb.message
    assert not task.score(it, "(13-9)*10-4").correct
    assert task.feedback(it, "(10-4)*(13-9)").ok
    score = task.score(it, "(10-4)*(13-9)")
    assert score.correct and score.message == "passed"


def test_prompt_lists_the_numbers():
    assert "4 9 10 13" in Game24Task().build_prompt(item(4, 9, 10, 13))


def test_load_all_items_from_the_csv():
    items = Game24Task().load_items("all")
    assert len(items) == 1362
    first = items[0]
    assert first.item_id == "game24/0001"
    assert first.payload == {"numbers": [1, 1, 4, 6]}
    assert first.difficulty in {"easy", "medium", "hard"}
    assert all(count_solutions(it.payload["numbers"]) > 0 for it in items[:50])


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError):
        Game24Task().load_items("train")


# ---- splits ----

def test_saved_splits_match_the_script_and_follow_the_design():
    from scripts.make_splits import build_splits

    saved = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
    assert saved == build_splits()                    # reproducible from seed and data
    task = Game24Task()
    dev, test = task.load_items("dev"), task.load_items("test")
    assert len(dev) == 12 and len(test) == 60
    assert not {it.item_id for it in dev} & {it.item_id for it in test}
    assert Counter(it.difficulty for it in dev) == {"easy": 4, "medium": 4, "hard": 4}
    assert Counter(it.difficulty for it in test[:30]) == {"easy": 10, "medium": 10, "hard": 10}
