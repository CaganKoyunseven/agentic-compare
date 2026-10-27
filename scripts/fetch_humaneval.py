"""
Fetch HumanEval dataset from the official OpenAI GitHub repository.
Saves to data/humaneval/HumanEval.jsonl.gz and verifies record count and schema.
"""

import gzip
import json
from pathlib import Path
import httpx

URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

def main():
    data_dir = Path("data/humaneval")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "HumanEval.jsonl.gz"

    print(f"Downloading HumanEval from {URL}...")
    try:
        response = httpx.get(URL, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please check the URL in your browser or internet connection.")
        return

    out_path.write_bytes(response.content)
    print(f"Saved to {out_path} ({len(response.content)} bytes).")

    # Verify records
    records = []
    with gzip.open(out_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records.")
    assert len(records) == 164, f"Expected 164 records, got {len(records)}"

    required_keys = {"task_id", "prompt", "entry_point", "canonical_solution", "test"}
    for i, r in enumerate(records):
        missing = required_keys - set(r.keys())
        assert not missing, record_error(i, missing)

    print("Verification successful: all 164 records contain expected keys.")
    print("Licence note: MIT License (from openai/human-eval repository).")

def record_error(i, missing):
    return f"Record {i} is missing keys: {missing}"

if __name__ == "__main__":
    main()
