"""LLM client for the local Ollama REST API.

Uses POST /api/chat. chat() does not stream, so token counts and durations come from one
response (the benchmark uses this). chat_stream() streams the text for the live demo; its final
line carries the same counts, so both return the same LLMResult.
Settings are fixed per client so all approaches share the same model and options.
"""

import json
import time
from collections.abc import Callable

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
        start = time.perf_counter()
        data = self._post_with_retry("/api/chat", self._body(messages, seed, stream=False))
        return self._result(data, data["message"]["content"], start)

    def chat_stream(self, messages: list[dict[str, str]], *, seed: int | None = None,
                    on_text: Callable[[str], None]) -> LLMResult:
        """Like chat(), but streams: on_text gets each piece of text as it is generated.

        Not retried, because part of the answer may already be shown. An exception raised by
        on_text stops the call; leaving the with block closes the connection, so Ollama stops
        generating.
        """
        start = time.perf_counter()
        pieces: list[str] = []
        try:
            with self._http.stream("POST", "/api/chat",
                                   json=self._body(messages, seed, stream=True)) as response:
                if response.status_code != 200:
                    response.read()
                    raise OllamaError(f"Ollama returned {response.status_code}: {response.text}")
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if "error" in data:       # e.g. the model crashed (out of GPU memory)
                        raise OllamaError(f"Ollama error: {data['error']}")
                    text = (data.get("message") or {}).get("content", "")
                    if text:
                        pieces.append(text)
                        on_text(text)
                    if data.get("done"):
                        return self._result(data, "".join(pieces), start)
        except httpx.TransportError as e:
            raise OllamaError(f"Cannot reach Ollama: {e}") from e
        raise OllamaError("Ollama stream ended without a final line")

    def _body(self, messages: list[dict[str, str]], seed: int | None, *, stream: bool) -> dict:
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        if seed is not None:
            options["seed"] = seed
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }

    def _result(self, data: dict, text: str, start: float) -> LLMResult:
        return LLMResult(
            text=text,
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            wall_ms=int((time.perf_counter() - start) * 1000),
            load_ms=data.get("load_duration", 0) // NS_PER_MS,
            model=data.get("model", self.model),
            done_reason=data.get("done_reason", ""),
        )

    def server_info(self) -> tuple[str | None, str | None]:
        """Ollama version and the digest of this client's model, for the run records.
        Returns None for a value the server does not report."""
        version = self._http.get("/api/version").json().get("version")
        models = self._http.get("/api/tags").json().get("models", [])
        digest = next((m.get("digest") for m in models if m.get("name") == self.model), None)
        return version, digest

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
