"""
HumanEval task implementation conforming to the Task protocol in agentic/tasks/base.py.

Official tests are kept private: they are read only inside score() and never
appear in Item.payload, Feedback or Score.
"""

import ast
import gzip
import json
import random
from pathlib import Path
from typing import Any

from agentic.sandbox import run_python
from agentic.tasks.base import Feedback, Item, Score

_LITERAL_ERRORS = (ValueError, TypeError, SyntaxError, MemoryError, RecursionError)


def docstring_examples(prompt: str, entry_point: str) -> list[str]:
    """Return assert statements built from the entry_point function's docstring examples.

    Best effort: only ">>> expr" lines followed by a literal expected value are used.
    The same parser is used by feedback() and by scripts/check_humaneval_examples.py.
    """
    tree = None
    for candidate in (prompt, prompt + "\n    pass\n"):
        try:
            tree = ast.parse(candidate)
            break
        except SyntaxError:
            continue
    if tree is None:
        return []

    doc = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            doc = ast.get_docstring(node)
            break
    if not doc:
        return []

    lines = doc.splitlines()
    examples: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line.startswith(">>>"):
            continue
        expr = line[3:].strip()
        expected_lines: list[str] = []
        while i < len(lines):
            next_line = lines[i].strip()
            if not next_line or next_line.startswith(">>>"):
                break
            expected_lines.append(next_line)
            i += 1
        if not expr or not expected_lines:
            continue
        try:
            expected = ast.literal_eval(" ".join(expected_lines))
        except _LITERAL_ERRORS:
            continue
        examples.append(f"assert {expr} == {expected!r}")
    return examples


class HumanEvalTask:
    name: str = "humaneval"

    def __init__(
        self,
        data_path: Path | None = None,
        seed: int = 1234,
        dev_size: int = 15,
        test_size: int = 60,
    ):
        if data_path is None:
            data_path = Path("data/humaneval/HumanEval.jsonl.gz")
        self.data_path = Path(data_path)
        self.seed = seed
        self.dev_size = dev_size
        self.test_size = test_size

        self._items_cache: dict[str, dict[str, Any]] = {}
        self._load_data()

    def _load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"HumanEval data file not found: {self.data_path}. "
                "Run scripts/fetch_humaneval.py to download it."
            )

        opener = gzip.open if self.data_path.suffix == ".gz" else open

        with opener(self.data_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    self._items_cache[record["task_id"]] = record

    def load_items(self, split: str) -> list[Item]:
        task_ids = sorted(self._items_cache.keys())
        rng = random.Random(self.seed)
        rng.shuffle(task_ids)

        dev_ids = task_ids[: self.dev_size]
        test_ids = task_ids[self.dev_size : self.dev_size + self.test_size]

        if split == "dev":
            selected_ids = dev_ids
        elif split == "test":
            selected_ids = test_ids
        elif split == "all":
            selected_ids = task_ids
        else:
            raise ValueError(f"Unknown split: {split}")

        items = []
        for tid in selected_ids:
            rec = self._items_cache[tid]
            items.append(
                Item(
                    item_id=tid,
                    task=self.name,
                    payload={
                        "prompt": rec["prompt"],
                        "entry_point": rec["entry_point"],
                    },
                    difficulty=None,
                )
            )
        return items

    def build_prompt(self, item: Item) -> str:
        return item.payload["prompt"]

    def extract_answer(self, item: Item, model_text: str) -> str:
        prompt = item.payload["prompt"]

        fenced_blocks: list[str] = []
        in_fence = False
        current: list[str] = []
        for line in model_text.splitlines():
            if line.strip().startswith("```"):
                if in_fence:
                    fenced_blocks.append("\n".join(current))
                    current = []
                    in_fence = False
                else:
                    in_fence = True
                continue
            if in_fence:
                current.append(line)
        if in_fence and current:
            # Unclosed fence (e.g. output cut off by the token limit).
            fenced_blocks.append("\n".join(current))

        code = fenced_blocks[-1] if fenced_blocks else model_text

        # Always put the code under the prompt, like the official HumanEval harness. The prompt
        # brings its imports and any helper functions the entry point calls. A body-only
        # completion continues the prompt's function; a full `def <entry_point>` in the code
        # replaces the prompt's stub (signature plus docstring, which is valid Python).
        return prompt + "\n" + code

    def feedback(self, item: Item, answer: str, tests: str | None = None) -> Feedback:
        examples_code = docstring_examples(item.payload["prompt"], item.payload["entry_point"])

        test_code_parts = [answer]
        if examples_code:
            test_code_parts.append("\n# Docstring examples check")
            test_code_parts.extend(examples_code)

        if tests:
            test_code_parts.append("\n# Additional model tests")
            test_code_parts.append(tests)

        full_test_code = "\n".join(test_code_parts)

        result = run_python(full_test_code, timeout_s=5.0)
        ok = result.passed
        if ok:
            message = "ok"
        elif result.timed_out:
            message = "timeout"
        elif result.stderr:
            message = result.stderr[-800:]
        else:
            message = "failed"

        return Feedback(
            ok=ok,
            message=message,
            details={"duration_ms": result.duration_ms, "timed_out": result.timed_out},
        )

    def score(self, item: Item, answer: str) -> Score:
        rec = self._items_cache.get(item.item_id, {})
        official_test = rec.get("test", "")
        entry_point = item.payload["entry_point"]

        full_code = f"{answer}\n\n{official_test}\ncheck({entry_point})\n"

        result = run_python(full_code, timeout_s=10.0)
        correct = result.passed
        if result.timed_out:
            message = "timeout"
        elif correct:
            message = "passed"
        else:
            message = "failed"

        return Score(
            correct=correct,
            message=message,
            details={"duration_ms": result.duration_ms},
        )
