"""Demo settings: configs/demo.toml on top of the benchmark settings in configs/experiment.toml.

The LLM settings are always the benchmark ones (same model behavior); the demo may only change
approach settings (for example a lighter ToT), the seed and how long the model stays loaded.
"""

from dataclasses import dataclass
from typing import Any

from agentic import config
from agentic.runner import slug

# Fixed scenes of the talk and their task. Draw-card puzzles use the scene id "draw".
SCENE_TASKS: dict[str, str] = {"easy24": "game24", "hard24": "game24", "humaneval": "humaneval"}
DRAW_SCENE = "draw"


@dataclass(frozen=True)
class DemoConfig:
    seed: int
    keep_alive: str
    llm_settings: dict[str, Any]
    approach_configs: dict[str, dict[str, Any]]
    scenes: dict[str, str]            # scene id -> item id
    pool: dict[str, list[str]]        # difficulty -> Game of 24 item ids for the draw card


def load_demo_config() -> DemoConfig:
    experiment = config.load_toml("experiment")
    demo = config.load_toml("demo")
    approaches = {name: dict(settings)
                  for name, settings in experiment.get("approaches", {}).items()}
    for name, overrides in demo.get("approaches", {}).items():
        approaches.setdefault(name, {}).update(overrides)
    return DemoConfig(
        seed=demo["seed"],
        keep_alive=demo["keep_alive"],
        llm_settings=dict(experiment["llm"]),
        approach_configs=approaches,
        scenes=dict(demo["scenes"]),
        pool={difficulty: list(ids) for difficulty, ids in demo["pool"].items()},
    )


def replay_scene_for(scene: str, item_id: str) -> str:
    """Name of the recorded replay for a live scene: the scene id itself for a fixed scene,
    "pool-<item>" for a draw-card puzzle."""
    if scene == DRAW_SCENE:
        return f"pool-{slug(item_id)}"
    return scene
