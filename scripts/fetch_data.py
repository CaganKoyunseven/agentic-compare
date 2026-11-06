"""Download the Game of 24 puzzles from the Tree of Thoughts authors' repository.

Source: princeton-nlp/tree-of-thought-llm (MIT licence), pinned to one commit so the file can
never change under us. The SHA-256 of each file is checked; a mismatch stops with an error.
The licence file is saved next to the data, as the MIT licence requires.

Usage: python -m scripts.fetch_data
"""

import hashlib
import sys

import httpx

from agentic.config import ROOT

COMMIT = "8050e67d0e3a0fddc424d7fa5801538722a4c4cc"
BASE = f"https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/{COMMIT}"
OUT_DIR = ROOT / "data" / "game24"
FILES = {
    # target name: (path in the source repo, expected SHA-256)
    "24.csv": ("src/tot/data/24/24.csv",
               "b9f12b3e36d987a3c714c4cef17d89a137d7c59da26532fcfb93b4821d8111b5"),
    "LICENSE": ("LICENSE",
                "cd2d341de1d3781e993e9859b1a36e8bfe12c84507c0a52f21c0220743af475b"),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (src, expected) in FILES.items():
        target = OUT_DIR / name
        if target.exists() and expected and sha256(target.read_bytes()) == expected:
            print(f"ok      {target.relative_to(ROOT)} (already present)")
            continue
        response = httpx.get(f"{BASE}/{src}", timeout=30.0)
        response.raise_for_status()
        digest = sha256(response.content)
        if expected and digest != expected:
            print(f"ERROR   {name}: SHA-256 {digest} does not match {expected}", file=sys.stderr)
            return 1
        target.write_bytes(response.content)
        print(f"saved   {target.relative_to(ROOT)} sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
