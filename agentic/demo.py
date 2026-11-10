"""One-command demo launcher.

Run: python -m agentic.demo [--replay-only] [--port 8000] [--no-browser]

1. Checks Ollama and the model from .env (MODEL_NAME, MACHINE_ID).
2. Warms the model with one short call; demo calls keep it loaded (keep_alive in demo.toml).
3. Starts the demo server on 127.0.0.1 and opens the browser.
If the check fails, the server starts in replay-only mode instead of stopping.
"""

import argparse
import threading
import webbrowser
from collections.abc import Callable

import httpx

from agentic.config import machine_id, model_name
from agentic.demo_config import DemoConfig, load_demo_config
from agentic.llm.backend import make_client
from agentic.llm.ollama import OllamaError
from agentic.server.app import HOST, create_app
from agentic.server.live import LiveRunner
from agentic.server.sessions import default_traces_dir, draw_pool

# Read timeout of live calls. Streaming sends a piece every few hundred ms, so 30 s without
# one means Ollama hangs; the presenter should see an error, not two silent minutes.
LIVE_TIMEOUT_S = 30.0


def prepare_live(demo: DemoConfig, *, transport: httpx.BaseTransport | None = None,
                 log: Callable[[str], None] = print) -> tuple[LiveRunner | None, str]:
    """A ready LiveRunner and "", or None and the reason live mode is not available."""
    try:
        model, machine = model_name(), machine_id()
        client = make_client(model, **demo.llm_settings, keep_alive=demo.keep_alive,
                              timeout_s=LIVE_TIMEOUT_S, transport=transport)
        version, digest = client.server_info()
        if digest is None:
            raise RuntimeError(f"Model {model} is not installed in Ollama (ollama pull {model})")
        # Loading the model can take longer than LIVE_TIMEOUT_S, so warm with the default.
        warmer = make_client(model, **demo.llm_settings, keep_alive=demo.keep_alive,
                              transport=transport)
        warm = warmer.chat([{"role": "user", "content": "Reply with OK."}])
        warmer.close()
        log(f"Model {model} warm ({warm.wall_ms / 1000:.1f} s), backend {version}")
        draw_pool(demo)     # loads the puzzle set now, so the first card or scene is instant
    except (RuntimeError, ValueError, OllamaError, httpx.HTTPError) as exc:
        # ValueError: an answer that is not JSON (another program on the Ollama port).
        reason = str(exc) or type(exc).__name__
        log(f"Live unavailable: {reason}. Starting in replay-only mode.")
        return None, reason
    runner = LiveRunner(client, demo, model=model, machine_id=machine, ollama_version=version,
                        model_digest=digest)
    return runner, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the demo (live if Ollama is ready).")
    parser.add_argument("--replay-only", action="store_true", help="do not use Ollama")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv)

    if args.replay_only:
        live, reason = None, "Started with --replay-only"
    else:
        live, reason = prepare_live(load_demo_config())

    import uvicorn

    traces_dir = default_traces_dir()
    url = f"http://{HOST}:{args.port}"
    print(f"Scenes from {traces_dir}")
    print(f"Mode: {'live + replay' if live else 'replay only'}")
    print(f"Open {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(traces_dir, live=live, live_reason=reason), host=HOST, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
