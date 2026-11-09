"""Tests for the demo launcher's Ollama check, with a fake HTTP transport (no Ollama needed)."""

import httpx
import pytest

from agentic.demo import prepare_live
from agentic.demo_config import load_demo_config
from agentic.server.live import LiveRunner

CHAT_REPLY = {"model": "fake:3b", "message": {"role": "assistant", "content": "OK"},
              "done": True, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 1,
              "load_duration": 2_000_000_000}


def fake_ollama(models=("fake:3b",), seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.9.0"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": name, "digest": "abc"} for name in models]})
        return httpx.Response(200, json=CHAT_REPLY)
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "fake:3b")
    monkeypatch.setenv("MACHINE_ID", "test-box")


def test_prepare_live_ok():
    logs, seen = [], []
    runner, reason = prepare_live(load_demo_config(), transport=fake_ollama(seen=seen),
                                  log=logs.append)
    assert isinstance(runner, LiveRunner)
    assert reason == ""
    assert runner.model == "fake:3b" and runner.machine_id == "test-box"
    assert any("warm" in line for line in logs)
    chat = next(r for r in seen if r.url.path == "/api/chat")
    assert b'"keep_alive":"60m"' in chat.content.replace(b" ", b"")


def test_prepare_live_unreachable():
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    logs = []
    runner, reason = prepare_live(load_demo_config(), transport=httpx.MockTransport(refuse),
                                  log=logs.append)
    assert runner is None
    assert reason
    assert "replay-only" in logs[-1]


def test_prepare_live_model_missing():
    runner, reason = prepare_live(load_demo_config(), transport=fake_ollama(models=("other:1b",)),
                                  log=lambda line: None)
    assert runner is None
    assert "fake:3b is not installed" in reason


def test_prepare_live_env_missing(monkeypatch):
    monkeypatch.delenv("MODEL_NAME")
    runner, reason = prepare_live(load_demo_config(), transport=fake_ollama(),
                                  log=lambda line: None)
    assert runner is None
    assert "MODEL_NAME" in reason


def test_prepare_live_non_json_answer_falls_back():
    def not_ollama(request):
        return httpx.Response(200, text="<html>some other server</html>")

    logs = []
    runner, reason = prepare_live(load_demo_config(), transport=httpx.MockTransport(not_ollama),
                                  log=logs.append)
    assert runner is None
    assert reason and "replay-only" in logs[-1]


def test_live_client_has_short_read_timeout():
    runner, _ = prepare_live(load_demo_config(), transport=fake_ollama(), log=lambda line: None)
    assert runner.client._http.timeout.read == 30
