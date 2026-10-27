"""
Check how many HumanEval prompts have at least one parseable docstring example.
"""

import gzip
import json
from pathlib import Path

def main():
    path = Path("data/humaneval/HumanEval.jsonl.gz")
    if not path.exists():
        print(f"Dataset not found at {path}. Run fetch_humaneval.py first.")
        return

    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    parseable_count = 0
    for r in records:
        prompt = r["prompt"]
        if ">>>" in prompt:
            parseable_count += 1

    print(f"Total problems: {len(records)}")
    print(f"Problems with '>>>' in docstring/prompt: {parseable_count}")

if __name__ == "__main__":
    main()
