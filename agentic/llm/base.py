"""LLM client interface and call result.

The client only answers single calls. Control flow, logging and events live in the approaches
and the runner, so every call can be counted.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResult:
    text: str                         # assistant reply
    tokens_in: int                    # prompt tokens counted by the backend
    tokens_out: int                   # generated tokens counted by the backend
    wall_ms: int                      # time measured around the whole HTTP call
    load_ms: int                      # model load time reported by the backend (0 when warm)
    model: str
    done_reason: str                  # "stop" or "length" (hit max tokens)
    system_fingerprint: str | None = None   # cloud API server configuration, None for Ollama


class LLMClient(Protocol):
    model: str
    def chat(self, messages: list[dict[str, str]], *, seed: int | None = None) -> LLMResult: ...
