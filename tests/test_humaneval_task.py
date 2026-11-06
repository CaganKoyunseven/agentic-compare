import json
from pathlib import Path

import pytest

from agentic.sandbox import SandboxResult
from agentic.tasks.humaneval import HumanEvalTask, docstring_examples

FENCE = "`" * 3
OFFICIAL_MARK = "OFFICIAL_TEST_SECRET_MARKER"
CANON_MARK = "CANONICAL_SECRET_MARKER"

PROMPT = (
    "from typing import List\n"
    "\n"
    "\n"
    "def helper(x):\n"
    '    """\n'
    "    >>> helper(1)\n"
    "    99\n"
    '    """\n'
    "    return x\n"
    "\n"
    "\n"
    "def add(a, b):\n"
    '    """Add two numbers.\n'
    "    >>> add(1, 2)\n"
    "    3\n"
    '    """\n'
)


@pytest.fixture
def mini_task():
    return HumanEvalTask(
        data_path=Path("tests/fixtures/humaneval_mini.jsonl"), dev_size=1, test_size=2
    )


@pytest.fixture
def task(tmp_path):
    record = {
        "task_id": "Test/0",
        "prompt": PROMPT,
        "entry_point": "add",
        "canonical_solution": f"    # {CANON_MARK}\n    return a + b\n",
        "test": (
            "def check(candidate):\n"
            f"    # {OFFICIAL_MARK}\n"
            "    assert candidate(1, 2) == 3\n"
        ),
    }
    path = tmp_path / "mini.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return HumanEvalTask(data_path=path)


@pytest.fixture
def item(task):
    return task.load_items("all")[0]


def test_splits(mini_task):
    dev_items = mini_task.load_items("dev")
    test_items = mini_task.load_items("test")
    all_items = mini_task.load_items("all")

    assert len(dev_items) == 1
    assert len(test_items) == 2
    assert len(all_items) == 3

    dev_ids = {i.item_id for i in dev_items}
    test_ids = {i.item_id for i in test_items}
    assert dev_ids.isdisjoint(test_ids)


def test_missing_data_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_humaneval"):
        HumanEvalTask(data_path=tmp_path / "nope.jsonl")


def test_extract_answer_single_fence(task, item):
    text = f"Here is the code:\n{FENCE}python\ndef add(a, b):\n    return a + b\n{FENCE}"
    extracted = task.extract_answer(item, text)
    assert "def add" in extracted
    assert "return a + b" in extracted
    assert FENCE not in extracted


def test_extract_answer_last_fence_wins(task, item):
    text = (
        f"{FENCE}python\ndef add(a, b):\n    return a - b\n{FENCE}\n"
        "Fixed:\n"
        f"{FENCE}python\ndef add(a, b):\n    return a + b\n{FENCE}\n"
    )
    extracted = task.extract_answer(item, text)
    assert "return a + b" in extracted
    assert "return a - b" not in extracted


def test_extract_answer_no_fence(task, item):
    extracted = task.extract_answer(item, "def add(a, b):\n    return a + b")
    assert "def add" in extracted
    assert "return a + b" in extracted


def test_extract_answer_body_only(task, item):
    extracted = task.extract_answer(item, "    return a + b")
    assert "def add" in extracted
    assert "return a + b" in extracted
    assert task.feedback(item, extracted).ok is True


def test_extract_answer_adds_missing_import(task, item):
    extracted = task.extract_answer(item, "def add(a, b):\n    return a + b")
    assert extracted.startswith("from typing import List")


def test_extract_answer_with_own_import_still_runs(task, item):
    # The prompt's import comes first; repeating it in the model code is harmless.
    text = "from typing import List\n\ndef add(a, b):\n    return a + b"
    extracted = task.extract_answer(item, text)
    assert extracted.startswith("from typing import List")
    assert task.score(item, extracted).correct is True


def test_extract_answer_keeps_prompt_helpers(tmp_path):
    # Regression: the prompt defines a helper that the entry point calls (like HumanEval/10,
    # 32, 38, 50). A full-function answer must still see the helper.
    prompt = (
        "def double(x):\n"
        '    """Return twice x."""\n'
        "    return 2 * x\n"
        "\n"
        "\n"
        "def quadruple(x):\n"
        '    """Return four times x using double.\n'
        "    >>> quadruple(2)\n"
        "    8\n"
        '    """\n'
    )
    record = {
        "task_id": "Test/1",
        "prompt": prompt,
        "entry_point": "quadruple",
        "canonical_solution": "    return double(double(x))\n",
        "test": "def check(candidate):\n    assert candidate(3) == 12\n",
    }
    path = tmp_path / "helper.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    task = HumanEvalTask(data_path=path)
    item = task.load_items("all")[0]

    answer = f"{FENCE}python\ndef quadruple(x):\n    return double(double(x))\n{FENCE}"
    extracted = task.extract_answer(item, answer)
    assert "def double" in extracted
    assert task.feedback(item, extracted).ok is True
    assert task.score(item, extracted).correct is True


def test_extract_answer_keeps_repeated_lines(task, item):
    code = (
        "def add(a, b):\n"
        "    total = 0\n"
        "    for _ in range(a):\n"
        "        total += 1\n"
        "    for _ in range(b):\n"
        "        total += 1\n"
        "    return total\n"
    )
    extracted = task.extract_answer(item, f"{FENCE}python\n{code}{FENCE}")
    assert extracted.count("total += 1") == 2
    assert task.score(item, extracted).correct is True


def test_extract_answer_keeps_repeated_blank_lines(task, item):
    code = "def add(a, b):\n    x = a\n\n\n    return x + b\n"
    extracted = task.extract_answer(item, f"{FENCE}python\n{code}{FENCE}")
    assert "x = a\n\n\n    return x + b" in extracted


def test_docstring_examples_uses_entry_point(item):
    examples = docstring_examples(item.payload["prompt"], item.payload["entry_point"])
    assert examples == ["assert add(1, 2) == 3"]


def test_feedback_pass(task, item):
    fb = task.feedback(item, PROMPT + "    return a + b\n")
    assert fb.ok is True
    assert fb.message == "ok"


def test_feedback_fail(task, item):
    fb = task.feedback(item, PROMPT + "    return a - b\n")
    assert fb.ok is False
    assert fb.message != "ok"


def test_feedback_timeout(task, item, monkeypatch):
    def fake_run(code, **kwargs):
        return SandboxResult(
            passed=False, timed_out=True, returncode=None, stdout="", stderr="", duration_ms=5000
        )

    monkeypatch.setattr("agentic.tasks.humaneval.run_python", fake_run)
    fb = task.feedback(item, "def add(a, b):\n    while True:\n        pass\n")
    assert fb.ok is False
    assert fb.message == "timeout"


def test_score_pass_and_fail(task, item):
    good = task.score(item, PROMPT + "    return a + b\n")
    assert good.correct is True
    assert good.message == "passed"

    bad = task.score(item, PROMPT + "    return a - b\n")
    assert bad.correct is False
    assert bad.message == "failed"


def test_score_timeout(task, item, monkeypatch):
    def fake_run(code, **kwargs):
        return SandboxResult(
            passed=False, timed_out=True, returncode=None, stdout="", stderr="", duration_ms=10000
        )

    monkeypatch.setattr("agentic.tasks.humaneval.run_python", fake_run)
    sc = task.score(item, "def add(a, b):\n    while True:\n        pass\n")
    assert sc.correct is False
    assert sc.message == "timeout"


def test_official_test_and_solution_never_leak(task, item):
    answers = [PROMPT + "    return a + b\n", PROMPT + "    return a - b\n"]
    outputs = [repr(item.payload)]
    for answer in answers:
        outputs.append(repr(task.feedback(item, answer)))
        outputs.append(repr(task.score(item, answer)))
    for out in outputs:
        assert OFFICIAL_MARK not in out
        assert CANON_MARK not in out
    assert set(item.payload) == {"prompt", "entry_point"}


def test_private_data_hidden(mini_task):
    for item in mini_task.load_items("all"):
        assert "canonical_solution" not in item.payload
        assert "test" not in item.payload
