"""Create fake traces so the UI can be built without a model.

Writes one JSONL file per demo scene to traces/mock/ (not committed). Each file holds four runs,
one per approach, in the demo order. Every line has model="mock" and machine_id="mock", so mock
data can never be mistaken for a real run. Texts, token counts and timings are invented but
shaped like real ones; the output is the same on every call (no randomness).

Usage: python -m scripts.make_mock_traces
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agentic import events as ev
from agentic.config import ROOT
from agentic.traces import TraceWriter

OUT_DIR = ROOT / "traces" / "mock"
START = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)
GAP_MS = 500                          # pause between two runs of a scene
SCORE_OF_LABEL = {"sure": 1.0, "likely": 0.5, "impossible": 0.0}


class MockRun:
    """Builds the event list of one fake run on a fake clock."""

    def __init__(self, scene: str, approach: str, task: str, item_id: str, payload: dict,
                 difficulty: str | None, config: dict, start_ms: int):
        self.fixed = {"run_id": f"mock-{scene}-{approach}-r0", "approach": approach,
                      "task": task, "item_id": item_id, "rep": 0, "model": "mock",
                      "machine_id": "mock"}
        self.start_ms = start_ms
        self.clock = 0
        self.records: list[tuple[ev.Envelope, ev.Body]] = []
        self.calls = self.tokens_in = self.tokens_out = 0
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.emit(ev.RunStart(split="dev", mode="demo", seed=0, config=config,
                              config_hash=config_hash[:12], payload=payload,
                              difficulty=difficulty))

    def emit(self, body: ev.Body) -> None:
        ts = START + timedelta(milliseconds=self.start_ms + self.clock)
        env = ev.Envelope(seq=len(self.records), ts=ts.isoformat(timespec="milliseconds"),
                          t_ms=self.clock, **self.fixed)
        self.records.append((env, body))

    def wait(self, ms: int) -> None:
        """Time spent outside the model, e.g. running a verifier or tests."""
        self.clock += ms

    def call(self, role: str, prompt: str, response: str) -> str:
        """A fake LLM call. Token counts and time grow with text length (about 40 tokens/s)."""
        self.calls += 1
        call_id = f"c{self.calls}"
        tokens_in = len(prompt) // 4 + 30
        tokens_out = len(response) // 4 + 1
        wall_ms = 250 + tokens_in // 2 + tokens_out * 25
        self.clock += wall_ms
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.emit(ev.LLMCall(call_id=call_id, role=role,
                             messages=[{"role": "user", "content": prompt}], response=response,
                             tokens_in=tokens_in, tokens_out=tokens_out, wall_ms=wall_ms,
                             load_ms=0, done_reason="stop", seed=0))
        return call_id

    def finish(self, answer: str, correct: bool) -> None:
        self.emit(ev.FinalAnswer(answer=answer))
        approach_ms = self.clock
        self.wait(5)                  # official scoring, outside the approach time
        self.emit(ev.RunEnd(status="ok", answer=answer, correct=correct,
                            score_message="passed" if correct else "failed",
                            tokens_in=self.tokens_in, tokens_out=self.tokens_out,
                            llm_calls=self.calls, wall_ms=approach_ms))


# ---- Game of 24 ----

def g24_prompt(numbers: str) -> str:
    return (f"Use the numbers {numbers} and + - * / to make 24. Use every number exactly once. "
            "Answer with one expression only.")


def g24_verify(run: MockRun, attempt_no: int | None, value: str) -> bool:
    run.wait(3)
    ok = value == "24"
    message = "equals 24" if ok else f"evaluates to {value}, not 24"
    run.emit(ev.TestResult(source="verifier", ok=ok, message=message, attempt_no=attempt_no))
    return ok


def g24_baseline(run: MockRun, numbers: str, answer: str, correct: bool) -> None:
    run.call("solver", g24_prompt(numbers), answer)
    run.finish(answer, correct)


def g24_reflective(run: MockRun, numbers: str, attempts: list[tuple[str, str]]) -> None:
    """attempts: (expression, its value) per round; the last one must equal 24."""
    prompt = g24_prompt(numbers)
    for no, (expr, value) in enumerate(attempts, start=1):
        call_id = run.call("solver", prompt, expr)
        run.emit(ev.Attempt(attempt_no=no, answer=expr, call_id=call_id))
        if g24_verify(run, no, value):
            run.finish(expr, True)
            return
        text = (f"{expr} gives {value}. Look for two parts whose product is 24, "
                "such as 4 and 6, built from pairs of the numbers.")
        call_id = run.call("critic", f"{prompt}\nAttempt: {expr}\nVerifier: {value}", text)
        run.emit(ev.Critique(attempt_no=no, text=text, call_id=call_id))
        prompt = f"{g24_prompt(numbers)}\nPrevious attempt: {expr}\nCritique: {text}"
    raise ValueError("last reflective attempt must be correct")


def g24_multi_agent(run: MockRun, numbers: str, plan: str,
                    rounds: list[tuple[str, str]]) -> None:
    """rounds: (solver expression, its value) per round; the last one must equal 24."""
    call_id = run.call("planner", f"Plan how to solve: {g24_prompt(numbers)}", plan)
    run.emit(ev.AgentMessage(round=1, role="planner", text=plan, call_id=call_id))
    for no, (expr, value) in enumerate(rounds, start=1):
        call_id = run.call("solver", f"{g24_prompt(numbers)}\nPlan: {plan}", expr)
        run.emit(ev.AgentMessage(round=no, role="solver", text=expr, call_id=call_id))
        if g24_verify(run, no, value):
            run.finish(expr, True)
            return
        text = f"The verifier says {expr} = {value}. Keep the plan but pair the numbers differently."
        call_id = run.call("critic", f"Expression: {expr}\nVerifier: {value}", text)
        run.emit(ev.AgentMessage(round=no, role="critic", text=text, call_id=call_id))
        plan = f"{plan} {text}"
    raise ValueError("last multi-agent round must be correct")


def tot(run: MockRun, root_state: str, levels: list[list[tuple]], keep: int,
        propose_prompt: str, score_prompt: str) -> list[str]:
    """Generic ToT search over a fixed tree.

    levels[d] lists the children made at depth d+1 as (node_id, parent_id, step, state, label).
    Each parent gets one proposer call, each child one scorer call, then all but the best
    `keep` children of the level are pruned. Returns the chosen path (root first).
    """
    run.emit(ev.NodeAdded(node_id="root", parent_id=None, depth=0, step="", state=root_state))
    parent_of = {}
    for depth, level in enumerate(levels, start=1):
        for parent in dict.fromkeys(p for _, p, *_ in level):
            children = [c for c in level if c[1] == parent]
            call_id = run.call("proposer", propose_prompt.format(state=parent),
                               "\n".join(c[2] for c in children))
            for node_id, parent_id, step, state, _ in children:
                parent_of[node_id] = parent_id
                run.emit(ev.NodeAdded(node_id=node_id, parent_id=parent_id, depth=depth,
                                      step=step, state=state, call_id=call_id))
        for node_id, _, step, _, label in level:
            call_id = run.call("scorer", score_prompt.format(step=step), label)
            run.emit(ev.NodeScored(node_id=node_id, score=SCORE_OF_LABEL[label], label=label,
                                   call_id=call_id))
        ranked = sorted(level, key=lambda c: -SCORE_OF_LABEL[c[4]])  # stable: ties keep order
        for node_id, *_ in ranked[keep:]:
            run.emit(ev.NodePruned(node_id=node_id))
    path = [ranked[0][0]]
    while path[0] != "root":
        path.insert(0, parent_of.get(path[0], "root"))
    return path


def g24_tot(run: MockRun, numbers: str, levels: list[list[tuple]], answer: str) -> None:
    path = tot(run, numbers, levels, keep=2,
               propose_prompt="Numbers left: {state}. Propose possible next steps.",
               score_prompt="Can the numbers left after {step} still reach 24? "
                            "Answer sure, likely or impossible.")
    steps = "; ".join(c[2] for level in levels for c in level if c[0] in path)
    run.call("writer", f"Steps: {steps}. Write the full expression for 24.", answer)
    run.emit(ev.PathChosen(node_ids=path, answer=answer))
    run.finish(answer, True)


# ---- HumanEval-style item (invented, not from the real dataset) ----

HE_PROMPT = '''def running_max(nums: list[int]) -> list[int]:
    """Return a list where element i is the largest value in nums[0..i].
    >>> running_max([1, 3, 2, 5])
    [1, 3, 3, 5]
    """
'''
HE_BUGGY = '''def running_max(nums):
    result = []
    m = nums[0]
    for x in nums:
        m = max(m, x)
        result.append(m)
    return result
'''
HE_FIXED = '''def running_max(nums):
    result = []
    for x in nums:
        result.append(x if not result else max(result[-1], x))
    return result
'''
HE_NO_IMPORT = '''def running_max(nums):
    return list(accumulate(nums, max))
'''
HE_OWN_TESTS = '''assert running_max([1, 3, 2, 5]) == [1, 3, 3, 5]
assert running_max([]) == []
assert running_max([-2, -5]) == [-2, -2]
'''
HE_TASK = f"Complete this Python function:\n{HE_PROMPT}"


def he_own_tests(run: MockRun, code: str, ok: bool, message: str, **where) -> bool:
    run.wait(400)                     # running the model's own tests in the sandbox
    run.emit(ev.TestResult(source="own_tests", ok=ok, message=message,
                           details={"code_lines": code.count("\n")}, **where))
    return ok


def he_baseline(run: MockRun) -> None:
    run.call("solver", HE_TASK, HE_BUGGY)
    run.finish(HE_BUGGY, False)       # crashes on an empty list in the official tests


def he_reflective(run: MockRun) -> None:
    call_id = run.call("solver", HE_TASK, HE_BUGGY)
    run.emit(ev.Attempt(attempt_no=1, answer=HE_BUGGY, call_id=call_id))
    run.call("tester", f"Write assert tests for:\n{HE_PROMPT}", HE_OWN_TESTS)
    he_own_tests(run, HE_BUGGY, False, "1 of 3 failed: IndexError on running_max([])",
                 attempt_no=1)
    text = "nums[0] fails for an empty list. Start from the first element inside the loop."
    call_id = run.call("critic", f"{HE_BUGGY}\nTest result: IndexError", text)
    run.emit(ev.Critique(attempt_no=1, text=text, call_id=call_id))
    call_id = run.call("solver", f"{HE_TASK}\nCritique: {text}", HE_FIXED)
    run.emit(ev.Attempt(attempt_no=2, answer=HE_FIXED, call_id=call_id))
    he_own_tests(run, HE_FIXED, True, "3 of 3 passed", attempt_no=2)
    run.finish(HE_FIXED, True)


def he_multi_agent(run: MockRun) -> None:
    plan = "Walk through the list once and keep the largest value seen so far."
    call_id = run.call("planner", f"Plan the solution:\n{HE_PROMPT}", plan)
    run.emit(ev.AgentMessage(round=1, role="planner", text=plan, call_id=call_id))
    call_id = run.call("solver", f"{HE_TASK}\nPlan: {plan}", HE_BUGGY)
    run.emit(ev.AgentMessage(round=1, role="solver", text=HE_BUGGY, call_id=call_id))
    call_id = run.call("critic", f"Write tests and review:\n{HE_BUGGY}", HE_OWN_TESTS)
    he_own_tests(run, HE_BUGGY, False, "1 of 3 failed: IndexError on running_max([])",
                 attempt_no=1)
    text = "Empty input crashes on nums[0]. Handle it before reading the first element."
    run.emit(ev.AgentMessage(round=1, role="critic", text=text, call_id=call_id))
    call_id = run.call("solver", f"{HE_TASK}\nCritic: {text}", HE_FIXED)
    run.emit(ev.AgentMessage(round=2, role="solver", text=HE_FIXED, call_id=call_id))
    call_id = run.call("critic", f"Review:\n{HE_FIXED}", "All my tests pass.")
    he_own_tests(run, HE_FIXED, True, "3 of 3 passed", attempt_no=2)
    run.emit(ev.AgentMessage(round=2, role="critic", text="All my tests pass.", call_id=call_id))
    run.finish(HE_FIXED, True)


def he_tot(run: MockRun) -> None:
    run.emit(ev.NodeAdded(node_id="root", parent_id=None, depth=0, step="", state="task"))
    plans = [("p1", "Loop once, keep the largest value seen so far.", "sure"),
             ("p2", "Use itertools.accumulate with max.", "likely"),
             ("p3", "Sort the list and return it.", "impossible")]
    call_id = run.call("proposer", f"Propose three plans for:\n{HE_PROMPT}",
                       "\n".join(p[1] for p in plans))
    for node_id, step, _ in plans:
        run.emit(ev.NodeAdded(node_id=node_id, parent_id="root", depth=1, step=step,
                              state="plan", call_id=call_id))
    for node_id, step, label in plans:
        call_id = run.call("scorer", f"Will this plan solve the task? {step}", label)
        run.emit(ev.NodeScored(node_id=node_id, score=SCORE_OF_LABEL[label], label=label,
                               call_id=call_id))
    run.emit(ev.NodePruned(node_id="p3"))
    plan_text = {node_id: step for node_id, step, _ in plans}
    run.call("tester", f"Write assert tests for:\n{HE_PROMPT}", HE_OWN_TESTS)
    results = {}
    for node_id, parent, code, ok, message in [
        ("k1", "p1", HE_FIXED, True, "3 of 3 passed"),
        ("k2", "p2", HE_NO_IMPORT, False, "3 of 3 failed: NameError: accumulate"),
    ]:
        call_id = run.call("coder", f"{HE_TASK}\nPlan: {plan_text[parent]}", code)
        run.emit(ev.NodeAdded(node_id=node_id, parent_id=parent, depth=2, step="code",
                              state=code, call_id=call_id))
        results[node_id] = he_own_tests(run, code, ok, message, node_id=node_id)
        run.emit(ev.NodeScored(node_id=node_id, score=1.0 if ok else 0.0,
                               label="tests passed" if ok else "tests failed"))
    run.emit(ev.NodePruned(node_id="k2"))
    run.emit(ev.PathChosen(node_ids=["root", "p1", "k1"], answer=HE_FIXED))
    run.finish(HE_FIXED, True)


# ---- Scenes ----

EASY_TREE = [
    [("a1", "root", "3-2=1", "1 4 6", "likely"),
     ("a2", "root", "6/2=3", "3 3 4", "likely"),
     ("a3", "root", "2+3=5", "4 5 6", "impossible")],
    [("b1", "a1", "4*6=24", "1 24", "sure"),
     ("b2", "a1", "1+4=5", "5 6", "impossible"),
     ("b3", "a2", "3*4=12", "3 12", "likely"),
     ("b4", "a2", "3+3=6", "4 6", "sure")],
    [("c1", "b1", "24*1=24", "24", "sure"),
     ("c2", "b1", "24+1=25", "25", "impossible"),
     ("c3", "b4", "4*6=24", "24", "sure"),
     ("c4", "b4", "6-4=2", "2", "impossible")],
]
HARD_TREE = [
    [("n1", "root", "13-9=4", "4 4 10", "sure"),
     ("n2", "root", "10-4=6", "6 9 13", "likely"),
     ("n3", "root", "9+13=22", "4 10 22", "impossible")],
    [("n4", "n1", "10-4=6", "4 6", "sure"),
     ("n5", "n1", "4*4=16", "10 16", "impossible"),
     ("n6", "n1", "4+4=8", "8 10", "impossible"),
     ("n7", "n2", "13-9=4", "4 6", "sure"),
     ("n8", "n2", "9+6=15", "13 15", "impossible"),
     ("n9", "n2", "13-6=7", "7 9", "impossible")],
    [("n10", "n4", "4*6=24", "24", "sure"),
     ("n11", "n4", "6-4=2", "2", "impossible"),
     ("n12", "n7", "6*4=24", "24", "sure"),
     ("n13", "n7", "6+4=10", "10", "impossible")],
]
CONFIGS = {
    "baseline": {},
    "reflective": {"max_rounds": 3},
    "multi_agent": {"max_rounds": 2},
    "tot": {"breadth": 3, "keep": 2},
}

SCENES = {
    "easy24": ("game24", "game24/mock-easy", {"numbers": [2, 3, 4, 6]}, "easy", {
        "baseline": lambda r: g24_baseline(r, "2 3 4 6", "6*4*(3-2)", True),
        "reflective": lambda r: g24_reflective(r, "2 3 4 6", [("6*4*(3-2)", "24")]),
        "multi_agent": lambda r: g24_multi_agent(
            r, "2 3 4 6", "Make 24 from 4 and 6, and turn 2 and 3 into 1.",
            [("6*4*(3-2)", "24")]),
        "tot": lambda r: g24_tot(r, "2 3 4 6", EASY_TREE, "(3-2)*4*6"),
    }),
    "hard24": ("game24", "game24/mock-hard", {"numbers": [4, 9, 10, 13]}, "hard", {
        "baseline": lambda r: g24_baseline(r, "4 9 10 13", "9+13+4-10", False),
        "reflective": lambda r: g24_reflective(
            r, "4 9 10 13", [("(13-9)*10-4", "36"), ("(13-9)*(10-4)", "24")]),
        "multi_agent": lambda r: g24_multi_agent(
            r, "4 9 10 13", "Build 4 and 6 from two pairs, then multiply them.",
            [("(13-10)*(9-4)", "15"), ("(10-4)*(13-9)", "24")]),
        "tot": lambda r: g24_tot(r, "4 9 10 13", HARD_TREE, "(13-9)*(10-4)"),
    }),
    "humaneval": ("humaneval", "humaneval/mock-running-max",
                  {"prompt": HE_PROMPT, "entry_point": "running_max"}, None, {
        "baseline": he_baseline,
        "reflective": he_reflective,
        "multi_agent": he_multi_agent,
        "tot": he_tot,
    }),
}


def build_scene(scene: str) -> list[tuple[ev.Envelope, ev.Body]]:
    task, item_id, payload, difficulty, runners = SCENES[scene]
    records, start_ms = [], 0
    for approach in ["baseline", "reflective", "multi_agent", "tot"]:
        run = MockRun(scene, approach, task, item_id, payload, difficulty, CONFIGS[approach],
                      start_ms)
        runners[approach](run)
        records += run.records
        start_ms += run.clock + GAP_MS
    return records


def write_all(out_dir: Path = OUT_DIR) -> list[Path]:
    paths = []
    for scene in SCENES:
        path = out_dir / f"{scene}.jsonl"
        path.unlink(missing_ok=True)  # the writer appends; start each file fresh
        with TraceWriter(path) as w:
            for env, body in build_scene(scene):
                w.write(env, body)
        paths.append(path)
    return paths


def main() -> None:
    for path in write_all():
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
