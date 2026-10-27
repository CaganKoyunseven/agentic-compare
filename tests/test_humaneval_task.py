import pytest
from pathlib import Path
from agentic.tasks.humaneval import HumanEvalTask

@pytest.fixture
def mini_task():
    return HumanEvalTask(data_path=Path("tests/fixtures/humaneval_mini.jsonl"), dev_size=1, test_size=2)

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

def test_extract_answer(mini_task):
    items = mini_task.load_items("all")
    item = next(i for i in items if i.item_id == "HumanEval/0")
    
    model_text = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```"
    extracted = mini_task.extract_answer(item, model_text)
    assert "def add" in extracted
    assert "return a + b" in extracted

    model_text_body = "    return a + b"
    extracted_body = mini_task.extract_answer(item, model_text_body)
    assert "def add" in extracted_body
    assert "return a + b" in extracted_body

def test_feedback_and_score(mini_task):
    items = mini_task.load_items("all")
    item = next(i for i in items if i.item_id == "HumanEval/0")

    correct_answer = "def add(a: int, b: int) -> int:\n    return a + b"
    
    fb = mini_task.feedback(item, correct_answer)
    assert fb.ok is True

    sc = mini_task.score(item, correct_answer)
    assert sc.correct is True
    assert sc.message == "passed"

    incorrect_answer = "def add(a: int, b: int) -> int:\n    return a - b"
    sc_bad = mini_task.score(item, incorrect_answer)
    assert sc_bad.correct is False
    assert sc_bad.message == "failed"
    assert "assert" not in sc_bad.message

def test_private_data_hidden(mini_task):
    items = mini_task.load_items("all")
    for item in items:
        assert "canonical_solution" not in item.payload
        assert "test" not in item.payload
