"""Tests for the smoke test summary (fake LLM, no Ollama needed)."""

from fractions import Fraction

from agentic.approaches.baseline import Baseline
from agentic.approaches.tot import TreeOfThoughts
from agentic.llm.base import LLMResult
from agentic.runner import Setup, run_experiment
from agentic.tasks.base import Item
from agentic.tasks.game24 import Game24Task, can_reach_24
from scripts.smoke_test import parse_state, report, summarize, valid_step

ITEMS = [Item("game24/0001", "game24", {"numbers": [4, 9, 10, 13]}, "hard"),
         Item("game24/0002", "game24", {"numbers": [1, 5, 7, 12]}, "easy")]
# One reply for every call: a step line (proposer), a label (scorer) and an answer.
REPLY = "13 - 9 = 4 (left: 4 4 10)\nsure\nAnswer: (13 - 9) * (10 - 4) = 24"


class FakeLLM:
    model = "fake"

    def chat(self, messages, *, seed=None):
        return LLMResult(text=REPLY, tokens_in=10, tokens_out=5, wall_ms=2, load_ms=0,
                         model="fake", done_reason="stop")


def test_valid_step():
    assert valid_step("4 9 10 13", "13 - 9 = 4", "4 4 10")
    assert valid_step("7 1 3.5", "7 / 3.5 = 2", "1 2")
    assert not valid_step("1 5 7 12", "1 + 1 = 2", "2 5 7 12")     # only one 1 available
    assert not valid_step("4 9 10 13", "13 - 9 = 5", "4 5 10")      # wrong arithmetic
    assert not valid_step("4 9 10 13", "13 - 9 = 4", "4 10")        # wrong numbers left
    assert not valid_step("2 5 7 0", "5 / 0 = 0", "2 7 0")          # division by zero
    assert not valid_step("4 6", "4 * 6 = 24", "left 24")           # unreadable state
    assert parse_state("2 x") is None


def test_can_reach_24_with_fractions():
    assert can_reach_24([Fraction(24)])
    assert can_reach_24([Fraction(7), Fraction(168)])
    assert can_reach_24([Fraction(1), Fraction(5), Fraction(5), Fraction(5)])  # 5*(5-1/5)
    assert not can_reach_24([Fraction(2), Fraction(3)])


def test_summary_of_a_baseline_and_tot_run(tmp_path):
    setup = Setup(task=Game24Task(), split="dev", model="fake", machine_id="test",
                  base_seed=2026, llm_settings={"temperature": 0.7},
                  approach_configs={"tot": {"breadth": 3, "keep": 3}})
    path = tmp_path / "trace.jsonl"
    run_experiment(setup, [Baseline(), TreeOfThoughts()], ITEMS, 1, FakeLLM(), path,
                   log=lambda s: None)

    stats, tree, pairs = summarize(path)
    assert (stats["baseline"].runs, stats["baseline"].correct, stats["baseline"].calls) == (2, 1, 2)
    # ToT per item: one child per level -> 3 x (1 propose + 1 score) + 1 writer = 7 calls.
    # Every state is "4 4 10": scored once, then cached; no path reaches "24", so no writer.
    # 3 proposer calls + 1 scorer call per run.
    assert (stats["tot"].runs, stats["tot"].correct, stats["tot"].calls) == (2, 0, 8)
    assert pairs == {("game24/0001", 0), ("game24/0002", 0)}
    # Item 1: only the first step fits its parent. Item 2: no step fits.
    assert (tree.steps, tree.valid_steps) == (6, 1)
    # Every node says "4 4 10", which can reach 24: (10 - 4) * 4.
    assert (tree.sure, tree.sure_reachable) == (6, 6)

    text = report(path)
    assert "both 0, only baseline 1, only ToT 0, neither 1" in text
    assert "valid steps: 1 / 6 (17%)" in text
