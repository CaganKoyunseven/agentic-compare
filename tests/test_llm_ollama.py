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


# ---- streaming ----

STREAM_LINES = [
    {"message": {"role": "assistant", "content": "Hel"}, "done": False},
    {"message": {"role": "assistant", "content": ""}, "done": False},
    {"done": False},
    {"message": {"role": "assistant", "content": "lo"}, "done": False},
    {"model": "fake:3b", "message": {"role": "assistant", "content": ""}, "done": True,
     "done_reason": "stop", "prompt_eval_count": 35, "eval_count": 2,
     "load_duration": 5_000_000},
]


def ndjson(lines: list[dict]) -> bytes:
    return "".join(json.dumps(line) + "\n" for line in lines).encode("utf-8")


def test_chat_stream_calls_on_text_and_returns_totals():
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_body.update(json.loads(request.content))
        return httpx.Response(200, content=ndjson(STREAM_LINES))

    pieces = []
    result = make_client(handler).chat_stream(
        [{"role": "user", "content": "hello"}], seed=7, on_text=pieces.append)

    assert seen_body["stream"] is True
    assert seen_body["options"]["seed"] == 7
    assert pieces == ["Hel", "lo"]
    assert result.text == "Hello"
    assert result.tokens_in == 35
    assert result.tokens_out == 2
    assert result.load_ms == 5
    assert result.done_reason == "stop"


def test_chat_stream_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'fake:3b' not found"})

    with pytest.raises(OllamaError, match="404"):
        make_client(handler).chat_stream([{"role": "user", "content": "hi"}], on_text=print)


def test_chat_stream_unreachable_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        make_client(handler).chat_stream([{"role": "user", "content": "hi"}], on_text=print)


def test_chat_stream_callback_exception_propagates():
    class Stop(Exception):
        pass

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ndjson(STREAM_LINES))

    def on_text(_text: str) -> None:
        raise Stop

    with pytest.raises(Stop):
        make_client(handler).chat_stream([{"role": "user", "content": "hi"}], on_text=on_text)
