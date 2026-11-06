"""
Check how many HumanEval prompts have at least one PARSEABLE docstring example.

Uses the same parser as HumanEvalTask.feedback (docstring_examples).
"""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.tasks.humaneval import docstring_examples


def main():
    path = Path("data/humaneval/HumanEval.jsonl.gz")
    if not path.exists():
        print(f"Dataset not found at {path}. Run scripts/fetch_humaneval.py first.")
        return

    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    without = [
        r["task_id"]
        for r in records
        if not docstring_examples(r["prompt"], r["entry_point"])
    ]
    print(f"Total problems: {len(records)}")
    print(f"Problems with at least one parseable example: {len(records) - len(without)}")
    print(f"Problems without a parseable example: {len(without)}")


if __name__ == "__main__":
    main()
