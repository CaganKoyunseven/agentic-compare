from agentic import config
from agentic.demo_config import SCENE_TASKS, load_demo_config, replay_scene_for
from agentic.tasks.game24 import Game24Task
from agentic.tasks.humaneval import HumanEvalTask

EXPERIMENT = """
seed = 1
reps = 3
[llm]
temperature = 0.7
num_predict = 512
[approaches.baseline]
[approaches.tot]
breadth = 3
keep = 3
"""

DEMO = """
seed = 2026
keep_alive = "60m"
[approaches.tot]
breadth = 2
[scenes]
easy24 = "game24/0043"
hard24 = "game24/1215"
humaneval = "HumanEval/55"
[pool]
easy = ["game24/0043"]
medium = ["game24/0286"]
hard = ["game24/1215"]
"""


def test_demo_config_merges_overrides(tmp_path, monkeypatch):
    (tmp_path / "experiment.toml").write_text(EXPERIMENT, encoding="utf-8")
    (tmp_path / "demo.toml").write_text(DEMO, encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    cfg = load_demo_config()

    assert cfg.seed == 2026
    assert cfg.keep_alive == "60m"
    assert cfg.llm_settings == {"temperature": 0.7, "num_predict": 512}
    assert cfg.approach_configs["tot"] == {"breadth": 2, "keep": 3}
    assert cfg.approach_configs["baseline"] == {}
    assert cfg.scenes == {"easy24": "game24/0043", "hard24": "game24/1215",
                          "humaneval": "HumanEval/55"}
    assert cfg.pool["medium"] == ["game24/0286"]


def test_real_demo_config_items_exist_in_dev():
    cfg = load_demo_config()
    dev24 = {item.item_id for item in Game24Task().load_items("dev")}
    dev_he = {item.item_id for item in HumanEvalTask().load_items("dev")}

    assert set(cfg.pool) == {"easy", "medium", "hard"}
    assert all(item_id in dev24 for ids in cfg.pool.values() for item_id in ids)
    assert set(cfg.scenes) == set(SCENE_TASKS)
    assert cfg.scenes["easy24"] in dev24
    assert cfg.scenes["hard24"] in dev24
    assert cfg.scenes["humaneval"] in dev_he
    assert cfg.keep_alive == "60m"


def test_replay_scene_for():
    assert replay_scene_for("easy24", "game24/0043") == "easy24"
    assert replay_scene_for("humaneval", "HumanEval/55") == "humaneval"
    assert replay_scene_for("draw", "game24/0043") == "pool-game24-0043"
