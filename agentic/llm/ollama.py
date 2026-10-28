"""LLM client for the local Ollama REST API.

Uses POST /api/chat without streaming, so token counts and durations come from one response.
Settings are fixed per client so all approaches share the same model and options.
"""

import time

import httpx

from agentic.config import ollama_host
from agentic.llm.base import LLMResult

NS_PER_MS = 1_000_000


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        model: str,
        *,
        host: str | None = None,
        temperature: float = 0.7,
        num_ctx: int = 4096,
        num_predict: int = 512,
        keep_alive: str = "10m",
        timeout_s: float = 120.0,
        retries: int = 1,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.retries = retries
        self._http = httpx.Client(
            base_url=host or ollama_host(), timeout=timeout_s, transport=transport
        )

    def chat(self, messages: list[dict[str, str]], *, seed: int | None = None) -> LLMResult:
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        if seed is not None:
            options["seed"] = seed
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }

        start = time.perf_counter()
        data = self._post_with_retry("/api/chat", body)
        wall_ms = int((time.perf_counter() - start) * 1000)

        return LLMResult(
            text=data["message"]["content"],
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            wall_ms=wall_ms,
            load_ms=data.get("load_duration", 0) // NS_PER_MS,
            model=data.get("model", self.model),
            done_reason=data.get("done_reason", ""),
        )

    def _post_with_retry(self, path: str, body: dict) -> dict:
        # Retry only connection problems and server errors; a 4xx (for example an unknown
        # model) will not fix itself, so it fails at once.
        for attempt in range(self.retries + 1):
            try:
                response = self._http.post(path, json=body)
            except httpx.TransportError as e:
                if attempt == self.retries:
                    raise OllamaError(f"Cannot reach Ollama: {e}") from e
                continue
            if response.status_code >= 500 and attempt < self.retries:
                continue
            if response.status_code != 200:
                raise OllamaError(f"Ollama returned {response.status_code}: {response.text}")
            return response.json()
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._http.close()
