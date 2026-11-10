"""LLM client for the OpenAI Chat Completions API.

Same interface as OllamaClient, so the approaches, the runner and the demo do not change.
chat() does not stream (the benchmark uses this); chat_stream() streams the text for the live
demo and asks for the token counts in the last chunk, so both return the same LLMResult.
Only httpx is used, no extra package. The API key comes from .env and is never logged.
"""

import json
import time
from collections.abc import Callable

import httpx

from agentic.config import openai_api_key
from agentic.llm.base import LLMResult
from agentic.llm.ollama import OllamaError

DEFAULT_BASE_URL = "https://api.openai.com/v1"
BACKEND_VERSION = "openai-api"
MAX_BACKOFF_S = 20.0


class OpenAIError(OllamaError):
    """A failed API call. Subclass of OllamaError so every place that stops on a backend
    failure (runner, demo, record script) handles it without any change."""


class OpenAIClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.7,
        num_predict: int = 512,
        num_ctx: int | None = None,       # Ollama only; accepted so the settings stay shared
        keep_alive: str | None = None,    # Ollama only
        timeout_s: float = 120.0,
        retries: int = 4,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.retries = retries
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=base_url, timeout=timeout_s, transport=transport,
            headers={"Authorization": f"Bearer {api_key or openai_api_key()}"},
        )

    def chat(self, messages: list[dict[str, str]], *, seed: int | None = None) -> LLMResult:
        start = time.perf_counter()
        data = self._post_with_retry(self._body(messages, seed, stream=False))
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return self._result(choice["message"].get("content") or "", usage,
                            choice.get("finish_reason"), data, start)

    def chat_stream(self, messages: list[dict[str, str]], *, seed: int | None = None,
                    on_text: Callable[[str], None]) -> LLMResult:
        """Like chat(), but calls on_text with each piece of the reply while it is generated."""
        start = time.perf_counter()
        pieces: list[str] = []
        usage: dict = {}
        finish_reason = None
        last: dict = {}
        try:
            with self._http.stream("POST", "/chat/completions",
                                   json=self._body(messages, seed, stream=True)) as response:
                if response.status_code != 200:
                    response.read()
                    raise self._error(response)
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        return self._result("".join(pieces), usage, finish_reason, last, start)
                    last = json.loads(payload)
                    usage = last.get("usage") or usage
                    for choice in last.get("choices") or []:
                        finish_reason = choice.get("finish_reason") or finish_reason
                        text = (choice.get("delta") or {}).get("content") or ""
                        if text:
                            pieces.append(text)
                            on_text(text)
        except httpx.TransportError as e:
            raise OpenAIError(f"Cannot reach the OpenAI API: {e}") from e
        raise OpenAIError("The OpenAI stream ended without [DONE]")

    def server_info(self) -> tuple[str | None, str | None]:
        """Backend name and the exact model snapshot id, for the run records. Also checks
        the API key and the model name before a run starts."""
        try:
            response = self._http.get(f"/models/{self.model}")
        except httpx.TransportError as e:
            raise OpenAIError(f"Cannot reach the OpenAI API: {e}") from e
        if response.status_code != 200:
            raise self._error(response)
        return BACKEND_VERSION, response.json().get("id")

    def close(self) -> None:
        self._http.close()

    def _body(self, messages: list[dict[str, str]], seed: int | None, *, stream: bool) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.num_predict,
            "stream": stream,
        }
        if seed is not None:
            body["seed"] = seed
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def _result(self, text: str, usage: dict, finish_reason: str | None, data: dict,
                start: float) -> LLMResult:
        return LLMResult(
            text=text,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            wall_ms=int((time.perf_counter() - start) * 1000),
            load_ms=0,
            model=data.get("model", self.model),
            done_reason=finish_reason or "",
            system_fingerprint=data.get("system_fingerprint"),
        )

    def _post_with_retry(self, body: dict) -> dict:
        # Retry connection problems, rate limits and server errors. Other 4xx answers and an
        # exhausted budget will not fix themselves, so they fail at once.
        for attempt in range(self.retries + 1):
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as e:
                if attempt == self.retries:
                    raise OpenAIError(f"Cannot reach the OpenAI API: {e}") from e
                self._sleep(self._backoff(attempt, None))
                continue
            if response.status_code == 200:
                return response.json()
            if attempt < self.retries and self._retryable(response):
                self._sleep(self._backoff(attempt, response))
                continue
            raise self._error(response)
        raise AssertionError("unreachable")

    @staticmethod
    def _error_code(response: httpx.Response) -> str:
        try:
            return (response.json().get("error") or {}).get("code") or ""
        except ValueError:
            return ""

    def _retryable(self, response: httpx.Response) -> bool:
        if response.status_code == 429:
            return self._error_code(response) != "insufficient_quota"
        return response.status_code >= 500

    @staticmethod
    def _backoff(attempt: int, response: httpx.Response | None) -> float:
        try:
            wait = float(response.headers["retry-after"])  # type: ignore[union-attr]
        except (AttributeError, KeyError, ValueError):
            wait = 2.0 ** attempt
        return min(wait, MAX_BACKOFF_S)

    def _error(self, response: httpx.Response) -> OpenAIError:
        if response.status_code == 401:     # the error text can echo part of the key
            return OpenAIError("OpenAI rejected the API key (401). Check OPENAI_API_KEY in .env.")
        if self._error_code(response) == "insufficient_quota":
            return OpenAIError("OpenAI budget is used up (insufficient_quota). Add credit, "
                               "then run the same command again to resume.")
        try:
            message = (response.json().get("error") or {}).get("message") or response.text
        except ValueError:
            message = response.text
        return OpenAIError(f"OpenAI returned {response.status_code}: {message[:300]}")
