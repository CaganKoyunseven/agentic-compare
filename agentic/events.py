"""Event types shared by approaches, traces, server and UI.

An approach only yields event bodies (what happened). The runner or the live server adds the
envelope (which run, order, time). A trace line is the envelope and the body merged into one
flat JSON object, so every line can be read on its own.

Approaches never score their own answer: they end with FinalAnswer, and the runner adds
RunEnd with the official verdict. Official tests therefore never reach an approach.
"""

import time
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Strict(BaseModel):
    # Frozen: events are records of the past. Forbid extra fields: a typo fails loudly.
    model_config = ConfigDict(frozen=True, extra="forbid")


class Envelope(_Strict):
    run_id: str
    seq: int                          # 0, 1, 2, ... inside one run
    ts: str                           # wall clock, ISO 8601 with time zone
    t_ms: int                         # milliseconds since the run started
    approach: str                     # "baseline" | "reflective" | "multi_agent" | "tot"
    task: str                         # "game24" | "humaneval"
    item_id: str
    rep: int
    model: str
    machine_id: str


ENVELOPE_FIELDS = frozenset(Envelope.model_fields)


class RunStamp:
    """Makes the envelopes of one run, counting seq and time from creation."""

    def __init__(self, *, run_id: str, approach: str, task: str, item_id: str, rep: int,
                 model: str, machine_id: str):
        self._fixed = {"run_id": run_id, "approach": approach, "task": task, "item_id": item_id,
                           "rep": rep, "model": model, "machine_id": machine_id}
        self._start = time.perf_counter()
        self._seq = 0

    def next(self) -> Envelope:
        env = Envelope(
            seq=self._seq,
            ts=datetime.now(UTC).isoformat(timespec="milliseconds"),
            t_ms=int((time.perf_counter() - self._start) * 1000),
            **self._fixed,
        )
        self._seq += 1
        return env


# ---- Written by the runner ----

class RunStart(_Strict):
    type: Literal["run_start"] = "run_start"
    split: str                        # "dev" | "test" | "demo"
    mode: str                         # "benchmark" | "demo"
    seed: int
    config: dict[str, Any]            # approach settings used for this run
    config_hash: str
    ollama_version: str | None = None
    model_digest: str | None = None
    payload: dict[str, Any]           # Item.payload, what the UI shows as the question
    difficulty: str | None = None


class RunEnd(_Strict):
    type: Literal["run_end"] = "run_end"
    status: str                       # "ok" | "error" | "cancelled"
    answer: str | None = None
    correct: bool | None = None       # official verdict, None when the run did not finish
    score_message: str = ""
    tokens_in: int
    tokens_out: int
    llm_calls: int
    wall_ms: int                      # approach time: LLM calls plus feedback checks
    error: str | None = None


# ---- Yielded by every approach ----

class LLMCall(_Strict):
    type: Literal["llm_call"] = "llm_call"
    call_id: str
    role: str                         # "solver", "critic", "planner", "proposer", "scorer", ...
    messages: list[dict[str, str]]
    response: str
    tokens_in: int
    tokens_out: int
    wall_ms: int
    load_ms: int
    done_reason: str
    seed: int | None = None
    system_fingerprint: str | None = None
    responses: list[str] | None = None   # all replies when one call asked for several (n > 1)


class Token(_Strict):
    """Live streaming only. Never written to traces; replay types out LLMCall.response."""
    type: Literal["token"] = "token"
    call_id: str
    text: str


class FinalAnswer(_Strict):
    """Last event of every approach."""
    type: Literal["final_answer"] = "final_answer"
    answer: str


# ---- Reflective (TestResult is also used by multi-agent and ToT) ----

class Attempt(_Strict):
    type: Literal["attempt"] = "attempt"
    attempt_no: int
    answer: str
    call_id: str


class Critique(_Strict):
    type: Literal["critique"] = "critique"
    attempt_no: int
    text: str
    call_id: str


class TestResult(_Strict):
    """Feedback an approach is allowed to see. Never the official tests."""
    __test__ = False                  # not a pytest test class despite the name
    type: Literal["test_result"] = "test_result"
    source: str                       # "verifier" | "own_tests"
    ok: bool
    message: str
    details: dict[str, Any] = {}
    attempt_no: int | None = None     # reflective attempt or multi-agent round
    node_id: str | None = None        # ToT node


# ---- Multi-agent ----

class AgentMessage(_Strict):
    type: Literal["agent_message"] = "agent_message"
    round: int
    role: str                         # "planner" | "solver" | "critic" | "tester" (HumanEval)
    text: str
    call_id: str


# ---- Tree of Thoughts ----

class NodeAdded(_Strict):
    type: Literal["node_added"] = "node_added"
    node_id: str
    parent_id: str | None             # None only for the root
    depth: int
    step: str                         # e.g. "13-9=4", or a plan for HumanEval
    state: str                        # e.g. "4 4 10" (numbers left)
    call_id: str | None = None


class NodeScored(_Strict):
    type: Literal["node_scored"] = "node_scored"
    node_id: str
    score: float
    label: str = ""                   # e.g. "sure" | "likely" | "impossible"
    call_id: str | None = None


class NodePruned(_Strict):
    type: Literal["node_pruned"] = "node_pruned"
    node_id: str


class PathChosen(_Strict):
    type: Literal["path_chosen"] = "path_chosen"
    node_ids: list[str]               # root first
    answer: str


Body = Annotated[
    RunStart | RunEnd | LLMCall | Token | FinalAnswer | Attempt | Critique | TestResult
    | AgentMessage | NodeAdded | NodeScored | NodePruned | PathChosen,
    Field(discriminator="type"),
]
_BODY = TypeAdapter(Body)


def to_dict(env: Envelope, body: Body) -> dict[str, Any]:
    """One flat JSON-ready dict: envelope fields plus body fields."""
    return {**env.model_dump(), **body.model_dump()}


def from_dict(d: dict[str, Any]) -> tuple[Envelope, Body]:
    """Inverse of to_dict. Raises pydantic.ValidationError on unknown types or fields."""
    env = Envelope.model_validate({k: v for k, v in d.items() if k in ENVELOPE_FIELDS})
    body = _BODY.validate_python({k: v for k, v in d.items() if k not in ENVELOPE_FIELDS})
    return env, body
