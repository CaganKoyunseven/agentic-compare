"""Load settings from .env and configs/*.toml."""

import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
# 127.0.0.1, not localhost: localhost may resolve to IPv6 ::1 only, and Ollama listens on IPv4.
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

load_dotenv(ROOT / ".env")


def load_toml(name: str) -> dict:
    """Read configs/<name>.toml, for example load_toml("experiment")."""
    with open(CONFIG_DIR / f"{name}.toml", "rb") as f:
        return tomllib.load(f)


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST


def model_name() -> str:
    """Model from .env; raises if it is not set, so a run never uses an unknown model."""
    name = os.environ.get("MODEL_NAME", "").strip()
    if not name:
        raise RuntimeError("MODEL_NAME is not set. Copy .env.example to .env and set it.")
    return name
