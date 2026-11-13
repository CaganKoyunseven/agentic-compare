"""FastAPI app serving the demo page and event streams.

Run: python -m agentic.server.app [--port 8000] [--traces <dir>] [--results <dir>]  (replay only)
     python -m agentic.demo                                                         (live + replay)
The server listens on 127.0.0.1 only.
"""

import argparse
import contextlib
import json
import re
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import iterate_in_threadpool

from agentic.config import ROOT
from agentic.server.live import LiveBusy, LiveRunner
from agentic.server.sessions import (
    default_traces_dir,
    draw_card,
    find_scenes,
    load_scene,
    replay_events,
)
from agentic.traces import read_trace

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"
HOST = "127.0.0.1"
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
DEFAULT_RESULTS_DIR = ROOT / "results"
REPORT_FILE = "report.md"
# Figures in the order of the report; any other PNG follows by name.
FIGURE_ORDER = ("success.png", "cumulative.png", "rep_spread.png", "paired.png",
                "cost_vs_accuracy.png", "time_and_calls.png")
# Separate studies: own subfolder of the results folder, own tab, never mixed with the main
# results. Section name -> figure order.
RESULT_SECTIONS = {
    "self_consistency": ("sc_success.png", "sc_k_curve.png"),
    "feedback_study": ("fs_success.png", "fs_cost_accuracy.png"),
}
RESULT_TYPES = {".png": "image/png", ".md": "text/markdown; charset=utf-8"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


# ---- SSE ----

def _sse_message(data: dict[str, Any], event: str | None = None) -> str:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def sse_stream(
    events: AsyncIterable[dict[str, Any]] | Iterable[dict[str, Any]],
) -> AsyncIterator[str]:
    """Turn any iterator of event dicts into SSE text. Replay and live both use this.

    A plain (blocking) iterator is read in a worker thread, so a slow model call does not block
    the server. Ends with an "end" event, or with an "error" event if the source fails; the
    error also carries the exception's details (live mode: scene, item_id, replay_scene).
    When the client goes away the source is closed at once, which stops a live scene.
    """
    source = events if isinstance(events, AsyncIterable) else iterate_in_threadpool(iter(events))
    async with contextlib.aclosing(source):
        try:
            async for event in source:
                yield _sse_message(event)
        except Exception as exc:  # noqa: BLE001  (shown to the presenter, not a silent stop)
            payload = {"message": f"{type(exc).__name__}: {exc}", **getattr(exc, "details", {})}
            yield _sse_message(payload, event="error")
            return
    yield _sse_message({}, event="end")


def _sse_response(events: AsyncIterable[dict[str, Any]] | Iterable[dict[str, Any]]) -> Response:
    return StreamingResponse(sse_stream(events), media_type="text/event-stream",
                             headers=SSE_HEADERS)


# ---- Scenes ----

def _scene_info(scene_id: str, path: Path) -> dict[str, Any]:
    """Metadata from the first run_start line of a scene file."""
    for env, body in read_trace(path):
        if body.type == "run_start":
            return {"id": scene_id, "task": env.task, "difficulty": body.difficulty,
                    "model": env.model, "mode": body.mode}
    raise ValueError(f"scene {scene_id!r} has no run_start event")


async def _replay_source(path: Path, speed: float) -> AsyncIterator[dict[str, Any]]:
    # Loading happens inside the stream, so a broken file becomes an "error" event.
    async for event in replay_events(load_scene(path), speed=speed):
        yield event


# ---- Results ----

def list_figures(results_dir: Path, order: tuple[str, ...] = FIGURE_ORDER) -> list[str]:
    """PNG file names directly inside results_dir, report order first."""
    if not results_dir.is_dir():
        return []
    names = {p.name for p in results_dir.glob("*.png") if p.is_file() and SAFE_NAME.match(p.name)}
    known = [n for n in order if n in names]
    return known + sorted(names - set(known))


def result_file(results_dir: Path, name: str) -> Path | None:
    """The file `name` directly inside results_dir if it is a .png or .md, else None."""
    if not SAFE_NAME.match(name) or Path(name).suffix.lower() not in RESULT_TYPES:
        return None
    path = results_dir / name
    if not path.is_file() or path.resolve().parent != results_dir.resolve():
        return None
    return path


def _result_response(path: Path | None) -> Response:
    if path is None:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type=RESULT_TYPES[path.suffix.lower()],
                        headers={"Cache-Control": "no-cache"})


# ---- App ----

def create_app(traces_dir: Path | None = None, results_dir: Path | None = None,
               live: LiveRunner | None = None,
               live_reason: str = "Live mode was not started") -> FastAPI:
    """The demo app, reading scenes from traces_dir (default: default_traces_dir()) and
    report files from results_dir (default: results/).

    live runs live scenes; without it the app is replay-only and live_reason says why.
    """
    traces_dir = Path(traces_dir) if traces_dir is not None else default_traces_dir()
    results_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    app = FastAPI(title="Agentic compare demo", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")

    def scene_path(scene: str) -> Path:
        path = find_scenes(traces_dir).get(scene)
        if path is None:
            raise HTTPException(status_code=404, detail=f"Unknown scene {scene!r}")
        return path

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        if not INDEX_FILE.is_file():
            return PlainTextResponse("Demo page not built yet")
        return FileResponse(INDEX_FILE, media_type="text/html")

    @app.get("/api/scenes")
    def scenes() -> list[dict[str, Any]]:
        result = []
        for scene_id, path in find_scenes(traces_dir).items():
            try:
                result.append(_scene_info(scene_id, path))
            except Exception as exc:
                raise HTTPException(status_code=500,
                                    detail=f"Scene {scene_id!r} cannot be read: {exc}") from exc
        return result

    @app.get("/api/replay/{scene}")
    def replay(scene: str, speed: float = Query(1.0, ge=0, le=100)) -> Response:
        return _sse_response(_replay_source(scene_path(scene), speed))

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        if live is None:
            return {"live": False, "model": None, "reason": live_reason}
        return {"live": True, "model": live.model, "reason": ""}

    @app.get("/api/live/{scene}")
    def live_scene(scene: str, item_id: str | None = None) -> Response:
        if live is None:
            raise HTTPException(status_code=503, detail=live_reason)
        try:
            events = live.start(scene, item_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        except LiveBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _sse_response(events)

    @app.get("/api/draw")
    def draw(difficulty: str = "easy") -> dict[str, Any]:
        try:
            return draw_card(difficulty, demo=live.demo if live else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/results")
    def results() -> dict[str, Any]:
        return {"figures": list_figures(results_dir),
                "report": result_file(results_dir, REPORT_FILE) is not None}

    @app.get("/results/{name}", include_in_schema=False)
    def result(name: str) -> Response:
        return _result_response(result_file(results_dir, name))

    def section_dir(section: str) -> Path:
        if section not in RESULT_SECTIONS:
            raise HTTPException(status_code=404, detail=f"Unknown results section {section!r}")
        return results_dir / section

    @app.get("/api/results/{section}")
    def section_results(section: str) -> dict[str, Any]:
        folder = section_dir(section)
        return {"figures": list_figures(folder, RESULT_SECTIONS[section]),
                "report": result_file(folder, REPORT_FILE) is not None}

    @app.get("/results/{section}/{name}", include_in_schema=False)
    def section_result(section: str, name: str) -> Response:
        return _result_response(result_file(section_dir(section), name))

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the demo server on 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--traces", type=Path, default=None,
                        help="folder with scene *.jsonl files (default: traces/examples/ "
                             "if it has any, otherwise traces/mock/)")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="folder with report.md and the figures (default: results/)")
    args = parser.parse_args()

    import uvicorn

    traces_dir = args.traces if args.traces is not None else default_traces_dir()
    print(f"Scenes from {traces_dir}")
    print(f"Results from {args.results}")
    print(f"Open http://{HOST}:{args.port}")
    uvicorn.run(create_app(traces_dir, args.results), host=HOST, port=args.port)


if __name__ == "__main__":
    main()
