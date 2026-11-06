"""Make the fixed dev/test split for Game of 24 and save it to data/splits/game24.json.

Stratified by difficulty tier with a fixed seed, so the result is the same on every machine.
dev (12 = 4 per tier): tuning, pilot and demo examples. test (60 = 20 per tier): reported
results only. The test list alternates tiers (easy, medium, hard, easy, ...), so its first 30
items are balanced too: the main comparison uses those, items 31-60 are an optional extension.
dev and test never share a puzzle.

Usage: python -m scripts.make_splits
"""

import json
import random

from agentic.config import ROOT
from agentic.tasks.game24 import SPLITS_FILE, Game24Task

SEED = 2026
TIERS = ["easy", "medium", "hard"]
DEV_PER_TIER = 4
TEST_PER_TIER = 20
TEST_MAIN_SIZE = 30


def interleave(groups: list[list[str]]) -> list[str]:
    return [item_id for row in zip(*groups, strict=True) for item_id in row]


def build_splits() -> dict:
    items = Game24Task().load_items("all")
    rng = random.Random(SEED)
    dev, test = [], []
    for tier in TIERS:
        ids = sorted(it.item_id for it in items if it.difficulty == tier)
        rng.shuffle(ids)
        dev.append(ids[:DEV_PER_TIER])
        test.append(ids[DEV_PER_TIER:DEV_PER_TIER + TEST_PER_TIER])
    return {"seed": SEED, "test_main_size": TEST_MAIN_SIZE,
            "dev": interleave(dev), "test": interleave(test)}


def main() -> None:
    splits = build_splits()
    SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_FILE.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SPLITS_FILE.relative_to(ROOT)}: "
          f"dev {len(splits['dev'])}, test {len(splits['test'])}")


if __name__ == "__main__":
    main()
