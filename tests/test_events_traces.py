"""Tests for the event schema (agentic.events) and JSONL traces (agentic.traces)."""

import json

import pytest
from pydantic import ValidationError

from agentic import events as ev
from agentic.traces import TraceWriter, group_runs, read_trace

ONE_OF_EACH = [
    ev.RunStart(
        split="dev", mode="benchmark", seed=1234, config={"rounds": 3}, config_hash="abc",
        ollama_version="0.12.0", model_digest="sha256:x", payload={"numbers": [4, 9, 10, 13]},
        difficulty="hard",
    ),
    ev.LLMCall(
        call_id="c1", role="solver", messages=[{"role": "user", "content": "4 9 10 13"}],
        response="(10-4)*(13-9)", tokens_in=182, tokens_out=21, wall_ms=2305, load_ms=0,
        done_reason="stop", seed=1234,
    ),
    ev.Token(call_id="c1", text="(10"),
    ev.Attempt(attempt_no=1, answer="(10-4)*(13-9)", call_id="c1"),
    ev.Critique(attempt_no=1, text="Looks fine.", call_id="c2"),
    ev.TestResult(attempt_no=1, source="verifier", ok=True, message="equals 24"),
    ev.AgentMessage(round=1, role="planner", text="Try 6 * 4.", call_id="c3"),
    ev.NodeAdded(node_id="n1", parent_id="root", depth=1, step="13-9=4", state="4 4 10",
                 call_id="c4"),
    ev.NodeScored(node_id="n1", score=0.8, label="likely", call_id="c5"),
    ev.NodePruned(node_id="n1"),
    ev.PathChosen(node_ids=["root", "n1"], answer="(10-4)*(13-9)"),
    ev.FinalAnswer(answer="(10-4)*(13-9)"),
    ev.RunEnd(status="ok", answer="(10-4)*(13-9)", correct=True, score_message="passed",
              tokens_in=182, tokens_out=21, llm_calls=1, wall_ms=2312),
]


def make_stamp(run_id: str = "r1") -> ev.RunStamp:
    return ev.RunStamp(run_id=run_id, approach="baseline", task="game24", item_id="game24/0042",
                       rep=0, model="qwen2.5:3b", machine_id="test")


@pytest.mark.parametrize("body", ONE_OF_EACH, ids=lambda b: b.type)
def test_every_event_round_trips_through_a_flat_dict(body):
    env = make_stamp().next()
    line = json.dumps(ev.to_dict(env, body))
    env2, body2 = ev.from_dict(json.loads(line))
    assert env2 == env
    assert body2 == body


def test_flat_dict_has_envelope_and_body_on_one_level():
    d = ev.to_dict(make_stamp().next(), ev.FinalAnswer(answer="1+1"))
    assert d["type"] == "final_answer"
    assert d["answer"] == "1+1"
    assert d["run_id"] == "r1" and d["seq"] == 0


def test_unknown_type_is_rejected():
    d = ev.to_dict(make_stamp().next(), ev.FinalAnswer(answer="x"))
    d["type"] = "no_such_event"
    with pytest.raises(ValidationError):
        ev.from_dict(d)


def test_misspelled_field_is_rejected():
    with pytest.raises(ValidationError):
        ev.FinalAnswer(answer="x", answr="y")


def test_stamp_counts_seq_and_time_forward():
    stamp = make_stamp()
    envs = [stamp.next() for _ in range(3)]
    assert [e.seq for e in envs] == [0, 1, 2]
    assert envs[0].t_ms <= envs[1].t_ms <= envs[2].t_ms
    assert all(e.item_id == "game24/0042" and e.rep == 0 for e in envs)


def test_trace_write_and_read_round_trip(tmp_path):
    path = tmp_path / "sub" / "batch.jsonl"
    stamp = make_stamp()
    written = [b for b in ONE_OF_EACH if b.type != "token"]
    with TraceWriter(path) as w:
        for body in written:
            w.write(stamp.next(), body)
    records = read_trace(path)
    assert [b for _, b in records] == written
    assert [e.seq for e, _ in records] == list(range(len(written)))


def test_token_events_are_not_written_to_traces(tmp_path):
    with TraceWriter(tmp_path / "t.jsonl") as w, pytest.raises(ValueError):
        w.write(make_stamp().next(), ev.Token(call_id="c1", text="x"))


def test_writer_appends_so_a_resumed_run_keeps_old_lines(tmp_path):
    path = tmp_path / "t.jsonl"
    with TraceWriter(path) as w:
        w.write(make_stamp("r1").next(), ev.FinalAnswer(answer="a"))
    with TraceWriter(path) as w:
        w.write(make_stamp("r2").next(), ev.FinalAnswer(answer="b"))
    assert [e.run_id for e, _ in read_trace(path)] == ["r1", "r2"]


def test_non_ascii_text_survives(tmp_path):
    path = tmp_path / "t.jsonl"
    with TraceWriter(path) as w:
        w.write(make_stamp().next(), ev.Critique(attempt_no=1, text="5 ÷ 3 ≠ 24", call_id="c"))
    assert read_trace(path)[0][1].text == "5 ÷ 3 ≠ 24"


def test_group_runs_keeps_order_inside_each_run(tmp_path):
    path = tmp_path / "t.jsonl"
    a, b = make_stamp("a"), make_stamp("b")
    with TraceWriter(path) as w:
        w.write(a.next(), ev.FinalAnswer(answer="1"))
        w.write(b.next(), ev.FinalAnswer(answer="2"))
        w.write(a.next(), ev.FinalAnswer(answer="3"))
    runs = group_runs(read_trace(path))
    assert list(runs) == ["a", "b"]
    assert [body.answer for _, body in runs["a"]] == ["1", "3"]
