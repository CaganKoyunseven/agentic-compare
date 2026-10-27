"""
HumanEval task implementation conforming to the Task protocol in agentic/tasks/base.py.
"""

import ast
import gzip
import json
import random
from pathlib import Path
from typing import Any

from agentic.tasks.base import Item, Feedback, Score
from agentic.sandbox import run_python

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
        self.data_path = data_path
        self.seed = seed
        self.dev_size = dev_size
        self.test_size = test_size

        self._items_cache: dict[str, dict[str, Any]] = {}
        self._load_data()

    def _load_data(self):
        if not self.data_path.exists():
            mini_path = Path("tests/fixtures/humaneval_mini.jsonl")
            if mini_path.exists():
                path_to_use = mini_path
            else:
                return
        else:
            path_to_use = self.data_path

        if str(path_to_use).endswith(".gz"):
            opener = gzip.open
        else:
            opener = open

        with opener(path_to_use, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    task_id = record["task_id"]
                    self._items_cache[task_id] = record

    def load_items(self, split: str) -> list[Item]:
        task_ids = sorted(self._items_cache.keys())
        rng = random.Random(self.seed)
        rng.shuffle(task_ids)

        dev_ids = task_ids[:self.dev_size]
        test_ids = task_ids[self.dev_size:self.dev_size + self.test_size]

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
            item = Item(
                item_id=tid,
                task=self.name,
                payload={
                    "prompt": rec["prompt"],
                    "entry_point": rec["entry_point"],
                },
                difficulty=None,
            )
            items.append(item)
        return items

    def build_prompt(self, item: Item) -> str:
        return item.payload["prompt"]

    def extract_answer(self, item: Item, model_text: str) -> str:
        entry_point = item.payload["entry_point"]
        prompt = item.payload["prompt"]

        fenced_blocks = []
        lines = model_text.splitlines()
        in_fence = False
        current_fence_lines = []
        for line in lines:
            if line.strip().startswith("```"):
                if in_fence:
                    fenced_blocks.append("\n".join(current_fence_lines))
                    current_fence_lines = []
                    in_fence = False
                else:
                    in_fence = True
                continue
            if in_fence:
                current_fence_lines.append(line)

        code_to_check = ""
        if fenced_blocks:
            code_to_check = fenced_blocks[-1]
        else:
            code_to_check = model_text

        if f"def {entry_point}" not in code_to_check:
            code_to_check = prompt + "\n" + code_to_check

        import_lines = [
            line for line in prompt.splitlines()
            if line.startswith("import ") or line.startswith("from ")
        ]
        code_lines = code_to_check.splitlines()
        final_lines = list(import_lines)
        for line in code_lines:
            if line not in final_lines:
                final_lines.append(line)

        return "\n".join(final_lines)

    def feedback(self, item: Item, answer: str, tests: str | None = None) -> Feedback:
        prompt = item.payload["prompt"]

        examples_code = []
        docstring_lines = []
        in_docstring = False
        for line in prompt.splitlines():
            if '"""' in line or "'''" in line:
                if in_docstring:
                    in_docstring = False
                    break
                else:
                    in_docstring = True
                    continue
            if in_docstring:
                docstring_lines.append(line)

        i = 0
        while i < len(docstring_lines):
            line = docstring_lines[i].strip()
            if line.startswith(">>>"):
                expr = line[3:].strip()
                expected_lines = []
                i += 1
                while i < len(docstring_lines):
                    next_line = docstring_lines[i]
                    if next_line.strip().startswith(">>>") or not next_line.strip():
                        break
                    expected_lines.append(next_line.strip())
                    i += 1
                if expected_lines:
                    expected_str = " ".join(expected_lines)
                    try:
                        expected_val = ast.literal_eval(expected_str)
                        examples_code.append(f"assert {expr} == {repr(expected_val)}")
                    except Exception:
                        pass
                continue
            i += 1

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
        message = "ok" if ok else (result.stderr[-800:] if result.stderr else "failed")

        return Feedback(
            ok=ok,
            message=message,
            details={"duration_ms": result.duration_ms, "timed_out": result.timed_out}
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
            details={"duration_ms": result.duration_ms}
        )
