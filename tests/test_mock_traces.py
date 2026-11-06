"""Checks that the mock traces are valid and internally consistent."""

import ast
import operator
from fractions import Fraction
from itertools import pairwise

import pytest

from agentic import events as ev
from agentic.traces import group_runs, read_trace
from scripts.make_mock_traces import SCENES, write_all

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv}


def eval_fraction(expr: str) -> Fraction:
    """Tiny exact evaluator for + - * / on integers (test only; the real one lives in game24)."""
    def walk(node):
        if isinstance(node, ast.BinOp):
            return OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return Fraction(node.value)
        raise ValueError(f"unexpected node {node!r}")
    return walk(ast.parse(expr, mode="eval").body)


@pytest.fixture(scope="module")
def traces(tmp_path_factory):
    out = tmp_path_factory.mktemp("mock")
    return {p.stem: group_runs(read_trace(p)) for p in write_all(out)}


def test_every_scene_has_four_runs_in_demo_order(traces):
    assert set(traces) == set(SCENES)
    for runs in traces.values():
        assert [records[0][0].approach for records in runs.values()] == [
            "baseline", "reflective", "multi_agent", "tot"]


def test_runs_are_well_formed(traces):
    for runs in traces.values():
        for records in runs.values():
            types = [body.type for _, body in records]
            assert types[0] == "run_start"
            assert types[-2:] == ["final_answer", "run_end"]
            envs = [env for env, _ in records]
            assert [e.seq for e in envs] == list(range(len(envs)))
            assert all(a.t_ms <= b.t_ms for a, b in pairwise(envs))
            assert all(e.model == "mock" and e.machine_id == "mock" for e in envs)


def test_run_end_totals_match_the_llm_calls(traces):
    for runs in traces.values():
        for records in runs.values():
            calls = [b for _, b in records if b.type == "llm_call"]
            end = records[-1][1]
            assert end.llm_calls == len(calls)
            assert end.tokens_in == sum(c.tokens_in for c in calls)
            assert end.tokens_out == sum(c.tokens_out for c in calls)
            assert end.answer == records[-2][1].answer


def test_game24_verdicts_are_true(traces):
    for scene in ["easy24", "hard24"]:
        for records in traces[scene].values():
            start, end = records[0][1], records[-1][1]
            value = eval_fraction(end.answer)
            assert end.correct == (value == 24), (scene, end.answer, value)
            numbers = sorted(int(t.value) for t in ast.walk(ast.parse(end.answer))
                             if isinstance(t, ast.Constant))
            if end.correct:
                assert numbers == sorted(start.payload["numbers"])


def test_tot_tree_is_consistent(traces):
    for runs in traces.values():
        records = list(runs.values())[-1]
        added = {b.node_id: b for _, b in records if isinstance(b, ev.NodeAdded)}
        pruned = {b.node_id for _, b in records if isinstance(b, ev.NodePruned)}
        chosen = next(b for _, b in records if isinstance(b, ev.PathChosen))
        assert all(b.parent_id is None or b.parent_id in added for b in added.values())
        assert chosen.node_ids[0] == "root"
        assert not pruned & set(chosen.node_ids)
        for parent, child in pairwise(chosen.node_ids):
            assert added[child].parent_id == parent
