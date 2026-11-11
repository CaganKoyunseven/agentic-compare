import asyncio
import dataclasses
import json
import random

import pytest
from fastapi.testclient import TestClient

from agentic.demo_config import load_demo_config
from agentic.events import from_dict
from agentic.server import app as app_module
from agentic.server import sessions
from agentic.server.app import create_app, sse_stream
from scripts.make_mock_traces import write_all
from tests.test_live import FakeStreamClient, make_runner, wait_idle


@pytest.fixture
def scene_dir(tmp_path):
    write_all(tmp_path)
    return tmp_path


@pytest.fixture
def client(scene_dir):
    return TestClient(create_app(scene_dir))


def parse_sse(text: str) -> list[tuple[str | None, dict]]:
    """(event name or None, parsed data) per SSE message."""
    messages = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        messages.append((event, data))
    return messages


# ---- sessions ----

def test_find_scenes_sorted_and_only_jsonl(scene_dir):
    (scene_dir / "notes.txt").write_text("not a scene", encoding="utf-8")
    scenes = sessions.find_scenes(scene_dir)
    assert list(scenes) == ["easy24", "hard24", "humaneval"]
    assert all(path.parent == scene_dir for path in scenes.values())


def test_find_scenes_missing_folder(tmp_path):
    assert sessions.find_scenes(tmp_path / "missing") == {}


def test_default_traces_dir_prefers_examples(tmp_path, monkeypatch):
    examples, mock = tmp_path / "examples", tmp_path / "mock"
    examples.mkdir()
    monkeypatch.setattr(sessions, "EXAMPLES_DIR", examples)
    monkeypatch.setattr(sessions, "MOCK_DIR", mock)
    assert sessions.default_traces_dir() == mock
    write_all(examples)
    assert sessions.default_traces_dir() == examples


def test_load_scene_keeps_file_order(scene_dir):
    path = scene_dir / "easy24.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert sessions.load_scene(path) == lines


def test_load_scene_rejects_broken_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"type": "nonsense"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        sessions.load_scene(path)


def test_replay_waits_capped_gaps_and_run_gap(monkeypatch):
    waits = []

    async def fake_sleep(seconds):
        waits.append(round(seconds, 6))

    monkeypatch.setattr(sessions.asyncio, "sleep", fake_sleep)
    events = [
        {"run_id": "a", "t_ms": 0},
        {"run_id": "a", "t_ms": 300},
        {"run_id": "a", "t_ms": 10_300},    # 10 s gap, capped at max_gap_ms
        {"run_id": "b", "t_ms": 0},         # new run: run_gap_ms
        {"run_id": "b", "t_ms": 100},
    ]

    async def collect(speed):
        return [e async for e in sessions.replay_events(events, speed=speed, max_gap_ms=2000,
                                                        run_gap_ms=500)]

    assert asyncio.run(collect(1.0)) == events
    assert waits == [0.3, 2.0, 0.5, 0.1]
    waits.clear()
    asyncio.run(collect(2.0))
    assert waits == [0.15, 1.0, 0.25, 0.05]
    waits.clear()
    asyncio.run(collect(0))
    assert waits == []


def test_draw_card_seeded():
    card = sessions.draw_card("easy", random.Random(1))
    assert card == sessions.draw_card("easy", random.Random(1))
    assert card["difficulty"] == "easy"
    assert len(card["numbers"]) == 4


def test_draw_card_only_from_pool_and_names_replay():
    pool = load_demo_config().pool
    for seed in range(20):
        card = sessions.draw_card("hard", random.Random(seed))
        assert card["item_id"] in pool["hard"]
        assert card["replay_scene"] == "pool-" + card["item_id"].replace("/", "-")


def test_draw_card_errors(monkeypatch):
    with pytest.raises(ValueError):
        sessions.draw_card("impossible")
    monkeypatch.setattr(sessions, "draw_pool", lambda demo=None: {})
    with pytest.raises(ValueError):
        sessions.draw_card("easy")


# ---- SSE ----

def test_sse_stream_reports_errors():
    def broken():
        yield {"type": "run_start"}
        raise RuntimeError("boom")

    async def collect():
        return [chunk async for chunk in sse_stream(broken())]

    messages = parse_sse("".join(asyncio.run(collect())))
    assert messages[0] == (None, {"type": "run_start"})
    assert messages[-1] == ("error", {"message": "RuntimeError: boom"})


# ---- endpoints ----

def test_index_before_page_is_built(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "INDEX_FILE", tmp_path / "missing.html")
    response = client.get("/")
    assert response.status_code == 200
    assert "Demo page not built yet" in response.text


def test_scenes_lists_metadata(client):
    response = client.get("/api/scenes")
    assert response.status_code == 200
    scenes = {s["id"]: s for s in response.json()}
    assert list(scenes) == ["easy24", "hard24", "humaneval"]
    assert scenes["easy24"] == {"id": "easy24", "task": "game24", "difficulty": "easy",
                                "model": "mock", "mode": "demo"}
    assert scenes["hard24"]["difficulty"] == "hard"
    assert scenes["humaneval"]["task"] == "humaneval"


@pytest.mark.parametrize("scene", ["easy24", "hard24", "humaneval"])
def test_replay_streams_every_event_in_order(client, scene_dir, scene):
    response = client.get(f"/api/replay/{scene}?speed=0")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    messages = parse_sse(response.text)
    assert messages[-1] == ("end", {})
    data = [d for event, d in messages[:-1]]
    assert all(event is None for event, _ in messages[:-1])
    assert data == sessions.load_scene(scene_dir / f"{scene}.jsonl")
    for d in data:
        from_dict(d)


def test_replay_broken_scene_sends_error_event(scene_dir):
    (scene_dir / "broken.jsonl").write_text('{"type": "nonsense"}\n', encoding="utf-8")
    response = TestClient(create_app(scene_dir)).get("/api/replay/broken?speed=0")
    messages = parse_sse(response.text)
    assert len(messages) == 1
    assert messages[0][0] == "error"
    assert messages[0][1]["message"]


@pytest.mark.parametrize("scene", ["nope", "..%2Fsecret", "..%2F..%2Fpyproject"])
def test_replay_unknown_scene_404(client, scene):
    response = client.get(f"/api/replay/{scene}")
    assert response.status_code == 404


def test_replay_negative_speed_rejected(client):
    assert client.get("/api/replay/easy24?speed=-1").status_code == 422


def test_status_without_live(client):
    assert client.get("/api/status").json() == {"live": False, "model": None,
                                               "reason": "Live mode was not started"}


def test_live_without_runner_is_503(client):
    response = client.get("/api/live/easy24")
    assert response.status_code == 503
    assert response.json() == {"detail": "Live mode was not started"}


@pytest.fixture
def live_runner(tmp_path):
    return make_runner(tmp_path / "live")


@pytest.fixture
def live_client(scene_dir, live_runner):
    return TestClient(create_app(scene_dir, live=live_runner))


def test_status_with_live(live_client):
    assert live_client.get("/api/status").json() == {"live": True, "model": "fake:3b",
                                                    "reason": ""}


def test_live_streams_events_then_end(live_client):
    messages = parse_sse(live_client.get("/api/live/easy24").text)
    assert messages[-1] == ("end", {})
    assert any(data.get("type") == "token" for _, data in messages[:-1])
    starts = [data for _, data in messages if data.get("type") == "run_start"]
    assert len(starts) == 4 and all(d["mode"] == "demo" for d in starts)


def test_live_unknown_scene_404(live_client):
    assert live_client.get("/api/live/nope").status_code == 404


def test_live_draw_off_pool_404(live_client):
    assert live_client.get("/api/live/draw?item_id=game24/0001").status_code == 404


def test_live_draw_from_pool_streams(live_client):
    messages = parse_sse(live_client.get("/api/live/draw?item_id=game24/0068").text)
    assert messages[-1] == ("end", {})
    assert {d["item_id"] for _, d in messages if d.get("type") == "run_start"} == {"game24/0068"}


def test_draw_uses_the_pool_the_live_runner_has(scene_dir, live_runner):
    # The runner only accepts its own pool, so a card from an edited demo.toml would be a 404.
    live_runner.demo = dataclasses.replace(live_runner.demo, pool={"easy": ["game24/0068"]})
    client = TestClient(create_app(scene_dir, live=live_runner))
    for _ in range(5):
        assert client.get("/api/draw?difficulty=easy").json()["item_id"] == "game24/0068"


def test_live_busy_409(scene_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("agentic.server.live.BUSY_WAIT_S", 0.0)
    runner = make_runner(tmp_path / "live", FakeStreamClient(delay_s=0.05))
    busy_client = TestClient(create_app(scene_dir, live=runner))

    async def scenario():
        stream = runner.start("hard24", None)
        await anext(stream)
        response = busy_client.get("/api/live/easy24")
        await stream.aclose()
        return response.status_code

    assert asyncio.run(scenario()) == 409
    assert wait_idle(runner)


def test_live_failure_error_event_has_replay_scene(scene_dir, tmp_path):
    runner = make_runner(tmp_path / "live", FakeStreamClient(fail_on_call=0))
    messages = parse_sse(TestClient(create_app(scene_dir, live=runner))
                         .get("/api/live/easy24").text)
    event, data = messages[-1]
    assert event == "error"
    assert data["scene"] == "easy24" and data["replay_scene"] == "easy24"
    assert "Cannot reach Ollama" in data["message"]


def test_sse_stream_closes_its_source():
    closed = []

    async def source():
        try:
            yield {"type": "a"}
            yield {"type": "b"}
        finally:
            closed.append(True)

    async def first_then_close():
        stream = sse_stream(source())
        await anext(stream)
        await stream.aclose()
        return list(closed)               # checked at once, not at event loop shutdown

    assert asyncio.run(first_then_close()) == [True]


def test_draw_endpoint(client):
    response = client.get("/api/draw?difficulty=hard")
    assert response.status_code == 200
    card = response.json()
    assert card["difficulty"] == "hard"
    assert len(card["numbers"]) == 4
    assert card["item_id"].startswith("game24/")


def test_draw_bad_difficulty(client):
    assert client.get("/api/draw?difficulty=impossible").status_code == 400


# ---- results ----

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.fixture
def results_dir(tmp_path):
    folder = tmp_path / "results"
    folder.mkdir()
    (folder / "success.png").write_bytes(PNG)
    (folder / "paired.png").write_bytes(PNG)
    (folder / "extra.png").write_bytes(PNG)
    (folder / "report.md").write_text("**FAKE DATA, NOT A RESULT**\n# Results report\n",
                                      encoding="utf-8")
    (folder / "runs.csv").write_text("run_id\n", encoding="utf-8")
    (folder / "notes.txt").write_text("not served", encoding="utf-8")
    (tmp_path / "secret.png").write_bytes(PNG)
    return folder


@pytest.fixture
def results_client(scene_dir, results_dir):
    return TestClient(create_app(scene_dir, results_dir))


def test_results_list(results_client):
    response = results_client.get("/api/results")
    assert response.status_code == 200
    # Report order first, then any other PNG by name; no other file types.
    assert response.json() == {"figures": ["success.png", "paired.png", "extra.png"],
                               "report": True}


def test_results_empty_or_missing_folder(scene_dir, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    for folder in (empty, tmp_path / "missing"):
        client = TestClient(create_app(scene_dir, folder))
        assert client.get("/api/results").json() == {"figures": [], "report": False}


def test_results_serves_png_and_report(results_client):
    png = results_client.get("/results/success.png")
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert png.content == PNG
    report = results_client.get("/results/report.md")
    assert report.status_code == 200
    assert report.headers["content-type"].startswith("text/markdown")
    assert "FAKE DATA" in report.text


@pytest.mark.parametrize("name", ["runs.csv", "notes.txt", "missing.png", ".hidden.png"])
def test_results_refuses_other_files(results_client, name):
    assert results_client.get(f"/results/{name}").status_code == 404


@pytest.mark.parametrize("path", [
    "/results/..%2Fsecret.png",
    "/results/%2E%2E%2Fsecret.png",
    "/results/..%5Csecret.png",
    "/results/../secret.png",
])
def test_results_path_traversal_404(results_client, path):
    assert results_client.get(path).status_code == 404


# ---- separate self-consistency results (own folder, own tab) ----

def test_self_consistency_section_lists_and_serves_its_own_folder(results_client, results_dir):
    sc = results_dir / "self_consistency"
    sc.mkdir()
    (sc / "sc_k_curve.png").write_bytes(PNG)
    (sc / "sc_success.png").write_bytes(PNG)
    (sc / "report.md").write_text("# Self-consistency\n", encoding="utf-8")

    listing = results_client.get("/api/results/self_consistency").json()
    assert listing == {"figures": ["sc_success.png", "sc_k_curve.png"], "report": True}
    assert results_client.get("/results/self_consistency/sc_success.png").content == PNG
    assert "Self-consistency" in results_client.get("/results/self_consistency/report.md").text
    # The main section does not show the separate study's files, and vice versa.
    assert "sc_success.png" not in results_client.get("/api/results").json()["figures"]
    assert results_client.get("/results/self_consistency/success.png").status_code == 404


def test_self_consistency_section_empty_and_unknown_sections(results_client):
    assert results_client.get("/api/results/self_consistency").json() == {"figures": [],
                                                                         "report": False}
    assert results_client.get("/api/results/secret").status_code == 404
    assert results_client.get("/results/secret/success.png").status_code == 404
    assert results_client.get("/results/self_consistency/..%2Fsuccess.png").status_code == 404
