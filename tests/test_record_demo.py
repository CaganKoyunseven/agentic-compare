"""Tests for recording demo replay scenes, with a fake LLM (no Ollama needed)."""

from dataclasses import replace

from agentic.demo_config import load_demo_config
from agentic.runner import ORDER
from agentic.server.sessions import find_scenes, load_scene
from scripts.record_demo import record
from tests.test_runner import FakeLLM


def small_demo():
    demo = load_demo_config()
    return replace(demo, scenes={"easy24": demo.scenes["easy24"]},
                   pool={"easy": ["game24/0068"]})


def record_into(out_dir, only=None):
    return record(small_demo(), FakeLLM(reply="12*(7-5)/1"), out_dir, model="fake:3b",
                  machine_id="test", only=only, log=lambda line: None)


def test_record_writes_one_readable_file_per_scene(tmp_path):
    paths = record_into(tmp_path)
    assert sorted(p.name for p in paths) == ["easy24.jsonl", "pool-game24-0068.jsonl"]
    for path in paths:
        events = load_scene(path)
        starts = [e for e in events if e["type"] == "run_start"]
        assert [e["approach"] for e in starts] == ORDER
        assert all(e["mode"] == "demo" and e["split"] == "demo" for e in starts)
        assert not any(e["type"] == "token" for e in events)
    pool_starts = [e for e in load_scene(tmp_path / "pool-game24-0068.jsonl")
                   if e["type"] == "run_start"]
    assert {e["item_id"] for e in pool_starts} == {"game24/0068"}


def test_record_replaces_existing_file(tmp_path):
    record_into(tmp_path)
    record_into(tmp_path)
    events = load_scene(tmp_path / "easy24.jsonl")
    assert sum(e["type"] == "run_start" for e in events) == 4
    assert not list(tmp_path.glob("*.tmp"))


def test_recorded_files_are_replay_scenes(tmp_path):
    record_into(tmp_path)
    assert set(find_scenes(tmp_path)) == {"easy24", "pool-game24-0068"}


def test_record_only_some_scenes(tmp_path):
    paths = record_into(tmp_path, only=["pool-game24-0068"])
    assert [p.name for p in paths] == ["pool-game24-0068.jsonl"]
