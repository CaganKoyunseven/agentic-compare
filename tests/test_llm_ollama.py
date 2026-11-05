"""Tests for the Ollama client, using a fake HTTP transport (no Ollama needed)."""

import json

import httpx
import pytest

from agentic.llm.ollama import OllamaClient, OllamaError

REPLY = {
    "model": "fake:3b",
    "message": {"role": "assistant", "content": "Hi!"},
    "done": True,
    "done_reason": "stop",
    "total_duration": 3_625_847_450,
    "load_duration": 2_964_779_997,
    "prompt_eval_count": 35,
    "eval_count": 3,
}


def make_client(handler, **kwargs) -> OllamaClient:
    return OllamaClient(
        "fake:3b", host="http://ollama.test", transport=httpx.MockTransport(handler), **kwargs
    )


def test_request_body_and_result():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=REPLY)

    client = make_client(handler, temperature=0.7, num_predict=64)
    result = client.chat([{"role": "user", "content": "hello"}], seed=7)

    assert seen["path"] == "/api/chat"
    body = seen["body"]
    assert body["model"] == "fake:3b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"] == {"temperature": 0.7, "num_ctx": 4096, "num_predict": 64, "seed": 7}

    assert result.text == "Hi!"
    assert result.tokens_in == 35
    assert result.tokens_out == 3
    assert result.load_ms == 2964
    assert result.done_reason == "stop"
    assert result.wall_ms >= 0


def test_seed_is_optional():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "seed" not in json.loads(request.content)["options"]
        return httpx.Response(200, json=REPLY)

    make_client(handler).chat([{"role": "user", "content": "hello"}])


def test_retries_connection_error_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json=REPLY)

    result = make_client(handler, retries=1).chat([{"role": "user", "content": "hi"}])
    assert result.text == "Hi!"
    assert len(calls) == 2


def test_gives_up_after_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        make_client(handler, retries=2).chat([{"role": "user", "content": "hi"}])


def test_client_error_is_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, json={"error": "model 'fake:3b' not found"})

    with pytest.raises(OllamaError, match="404"):
        make_client(handler, retries=3).chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_server_error_is_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(500, text="busy")
        return httpx.Response(200, json=REPLY)

    assert make_client(handler, retries=1).chat([{"role": "user", "content": "hi"}]).text == "Hi!"
    assert len(calls) == 2


def test_server_info_reads_version_and_digest():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "1.2.3"})
        return httpx.Response(200, json={"models": [
            {"name": "other:1b", "digest": "aaa"}, {"name": "fake:3b", "digest": "bbb"}]})

    assert make_client(handler).server_info() == ("1.2.3", "bbb")
