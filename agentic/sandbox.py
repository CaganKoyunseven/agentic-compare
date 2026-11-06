"""
Sandbox runner for executing model-written Python code safely in an isolated temporary directory.
Output is read while the child runs and only a bounded tail is kept, so a child that prints in an
endless loop can neither fill the parent's memory nor stretch the timeout.
Note: Network and memory limits are not enforced as they are not portable across platforms (e.g., Windows).
Note: On Windows, only the main process is killed on timeout; grandchild processes may survive the kill.
A surviving grandchild that keeps the output pipe open cannot block the call: the readers are only
waited for a short grace period.
"""

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

TRUNCATION_MARKER = "...[truncated]...\n"
READER_GRACE_S = 1.0     # how long to wait for the output readers after the child has ended


@dataclass(frozen=True)
class SandboxResult:
    passed: bool
    timed_out: bool
    returncode: int | None
    stdout: str
    stderr: str
    duration_ms: int


class _TailReader(threading.Thread):
    """Reads a pipe until it closes and keeps only the last `limit_bytes` bytes."""

    def __init__(self, pipe, limit_bytes: int):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.limit_bytes = limit_bytes
        self.tail = bytearray()
        self.dropped = False

    def run(self) -> None:
        try:
            while chunk := self.pipe.read1(65536):
                self.tail += chunk
                if len(self.tail) > self.limit_bytes:
                    del self.tail[:-self.limit_bytes]
                    self.dropped = True
        except (OSError, ValueError):   # pipe closed under us
            pass


def _tail_text(reader: _TailReader, max_chars: int) -> str:
    """Decode the kept bytes and return at most max_chars characters, marker included."""
    text = bytes(reader.tail).decode("utf-8", errors="replace")
    if not reader.dropped and len(text) <= max_chars:
        return text
    if max_chars <= len(TRUNCATION_MARKER):
        return text[-max_chars:] if max_chars > 0 else ""
    return TRUNCATION_MARKER + text[-(max_chars - len(TRUNCATION_MARKER)):]


def run_python(code: str, *, timeout_s: float = 5.0, max_output_chars: int = 4000) -> SandboxResult:
    # Clean environment: remove secrets
    env = {
        k: v for k, v in os.environ.items()
        if not any(sub in k.upper() for sub in ["KEY", "TOKEN", "SECRET", "PASSWORD"])
    }
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    timed_out = False
    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        script_path = Path(tmpdir) / "main.py"
        script_path.write_text(code, encoding="utf-8")

        # Platform-specific process group handling
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        try:
            process = subprocess.Popen(
                [sys.executable, "-I", str(script_path)],
                cwd=tmpdir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                **kwargs
            )
        except (OSError, ValueError) as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return SandboxResult(
                passed=False,
                timed_out=False,
                returncode=None,
                stdout="",
                stderr=str(e),
                duration_ms=duration_ms
            )

        # A UTF-8 character has at most 4 bytes, so 4 bytes per character always keep enough.
        readers = [_TailReader(process.stdout, 4 * max_output_chars),
                   _TailReader(process.stderr, 4 * max_output_chars)]
        for reader in readers:
            reader.start()

        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            if sys.platform == "win32":
                process.kill()
            else:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    process.kill()
            returncode = process.wait()

        deadline = time.perf_counter() + READER_GRACE_S    # one shared grace period
        for reader in readers:
            reader.join(timeout=max(0.0, deadline - time.perf_counter()))

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        stdout_str = _tail_text(readers[0], max_output_chars)
        stderr_str = _tail_text(readers[1], max_output_chars)

        passed = (returncode == 0) and not timed_out

        return SandboxResult(
            passed=passed,
            timed_out=timed_out,
            returncode=returncode,
            stdout=stdout_str,
            stderr=stderr_str,
            duration_ms=duration_ms
        )
