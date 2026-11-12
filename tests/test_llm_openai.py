"""Tests for the OpenAI client and the backend choice, using a fake transport (no network)."""

import json

import httpx
import pytest

from agentic.llm.backend import make_client
from agentic.llm.ollama import OllamaClient, OllamaError
from agentic.llm.openai_api import OpenAIClient, OpenAIError

KEY = "sk-test-secret"
MESSAGES = [{"role": "user", "content": "hello"}]
REPLY = {
    "model": "fake-mini-2024",
    "system_fingerprint": "fp_123",
    "choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 35, "completion_tokens": 3},
}


def client_with(handler, **kwargs) -> OpenAIClient:
    kwargs.setdefault("sleep", lambda s: None)
    return OpenAIClient("fake-mini", api_key=KEY, base_url="https://api.test/v1",
                        transport=httpx.MockTransport(handler), **kwargs)


def test_request_body_and_result():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=REPLY)

    result = client_with(handler, temperature=0.7, num_predict=64).chat(MESSAGES, seed=7)

    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["body"] == {"model": "fake-mini", "messages": MESSAGES, "temperature": 0.7,
                            "max_tokens": 64, "stream": False, "seed": 7}
    assert (result.text, result.tokens_in, result.tokens_out) == ("Hi!", 35, 3)
    assert (result.load_ms, result.done_reason, result.model) == (0, "stop", "fake-mini-2024")
    assert result.system_fingerprint == "fp_123"


def test_length_finish_is_reported_like_ollama():
    reply = {**REPLY, "choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]}
    result = client_with(lambda r: httpx.Response(200, json=reply)).chat(MESSAGES)
    assert result.done_reason == "length"


def sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def test_stream_returns_the_same_result():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        base = {"model": "fake-mini-2024", "system_fingerprint": "fp_123"}
        return httpx.Response(200, content=sse(
            {**base, "choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
            {**base, "choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]},
            {**base, "choices": [], "usage": {"prompt_tokens": 35, "completion_tokens": 3}},
        ))

    pieces: list[str] = []
    result = client_with(handler).chat_stream(MESSAGES, seed=1, on_text=pieces.append)

    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert pieces == ["Hi", "!"]
    assert (result.text, result.tokens_in, result.tokens_out) == ("Hi!", 35, 3)
    assert (result.done_reason, result.system_fingerprint) == ("stop", "fp_123")


def test_stream_without_done_is_an_error():
    client = client_with(lambda r: httpx.Response(200, content=b"data: {\"choices\": []}\n\n"))
    with pytest.raises(OpenAIError, match="without"):
        client.chat_stream(MESSAGES, on_text=lambda s: None)


def test_rate_limit_is_retried_with_the_server_delay():
    calls, waits = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "3"},
                                  json={"error": {"code": "rate_limit_exceeded"}})
        return httpx.Response(200, json=REPLY)

    assert client_with(handler, sleep=waits.append).chat(MESSAGES).text == "Hi!"
    assert len(calls) == 2
    assert waits == [3.0]


def test_used_up_budget_stops_at_once():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json={"error": {"code": "insufficient_quota"}})

    with pytest.raises(OpenAIError, match="budget"):
        client_with(handler).chat(MESSAGES)
    assert len(calls) == 1


def test_bad_key_error_never_contains_the_key():
    body = {"error": {"message": f"Incorrect API key provided: {KEY}"}}
    with pytest.raises(OpenAIError) as caught:
        client_with(lambda r: httpx.Response(401, json=body)).chat(MESSAGES)
    assert KEY not in str(caught.value)
    assert "OPENAI_API_KEY" in str(caught.value)


def test_server_errors_give_up_after_the_retries():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, json={"error": {"message": "boom"}})

    with pytest.raises(OpenAIError, match="boom"):
        client_with(handler, retries=2).chat(MESSAGES)
    assert len(calls) == 3


def test_connection_failure_is_a_backend_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    # The runner and the demo stop on OllamaError, so the new error must be one.
    with pytest.raises(OllamaError, match="Cannot reach"):
        client_with(handler, retries=0).chat(MESSAGES)


def test_server_info_returns_the_snapshot_and_checks_the_model():
    ok = client_with(lambda r: httpx.Response(200, json={"id": "fake-mini-2024"}))
    assert ok.server_info() == ("openai-api", "fake-mini-2024")

    missing = client_with(
        lambda r: httpx.Response(404, json={"error": {"message": "no such model"}}))
    with pytest.raises(OpenAIError, match="no such model"):
        missing.server_info()


def test_backend_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    assert isinstance(make_client("fake:3b"), OllamaClient)


def test_backend_openai_takes_the_shared_settings(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    client = make_client("fake-mini", temperature=0.5, num_ctx=4096, num_predict=99,
                         keep_alive="60m")
    assert isinstance(client, OpenAIClient)
    assert (client.temperature, client.num_predict) == (0.5, 99)


def test_openai_backend_without_a_key_explains_itself(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_client("fake-mini")


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "other")
    with pytest.raises(ValueError, match="LLM_BACKEND"):
        make_client("fake-mini")


def test_sample_asks_for_n_choices_in_one_request():
    seen = []
    reply = {**REPLY, "choices": [
        {"message": {"content": "sure"}, "finish_reason": "stop"},
        {"message": {"content": "likely"}, "finish_reason": "length"},
        {"message": {"content": "sure"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 35, "completion_tokens": 9}}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=reply)

    result = client_with(handler).sample(MESSAGES, seed=7, n=3)
    assert len(seen) == 1 and seen[0]["n"] == 3 and seen[0]["seed"] == 7
    assert result.texts == ("sure", "likely", "sure") and result.text == "sure"
    assert (result.tokens_in, result.tokens_out) == (35, 9)    # prompt paid once
    assert result.done_reason == "length"                      # any cut choice is reported
