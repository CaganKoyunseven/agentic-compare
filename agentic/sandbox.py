"""
Sandbox runner for model-written Python code.

The code runs in a separate process inside a bubblewrap (bwrap) box, so it can neither damage nor
read anything outside the box:
- it sees only the system's Python and libraries (read-only), a fresh working folder (/work) and
  a small /tmp; the home folder, the repository and .env do not exist inside the box;
- no network, own process tree, empty environment (no keys, no tokens);
- limits on memory, CPU time, file size, open files and processes; the timeout kills the whole box.
Output is read while the child runs and only a bounded tail is kept, so a child that prints in an
endless loop can neither fill the parent's memory nor stretch the timeout.

bubblewrap exists only on Linux. Elsewhere (or when bwrap is missing or not allowed to create
namespaces) run_python raises SandboxUnavailable instead of running the code unprotected. On
Windows, use WSL.
"""

import functools
import math
import os
import shutil
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

WORK = "/work"                       # the working folder inside the box
TMPFS_BYTES = 64 * 1024 * 1024       # size of /work and /tmp (both live in memory)
MEMORY_BYTES = 1024 * 1024 * 1024    # address space of the child
FILE_BYTES = 16 * 1024 * 1024        # largest file the child may write
MAX_PROCESSES = 64                   # counted inside the box only (own user namespace)
MAX_OPEN_FILES = 256

UNAVAILABLE_HELP = ("Running model-written code needs Linux with bubblewrap (bwrap) installed "
                    "and allowed to create user namespaces. On Windows, use WSL.")


class SandboxUnavailable(RuntimeError):
    """bwrap cannot be used on this computer, so model-written code is not run at all."""


@dataclass(frozen=True)
class SandboxResult:
    passed: bool
    timed_out: bool
    returncode: int | None
    stdout: str
    stderr: str
    duration_ms: int


class _TailReader(threading.Thread):
    """Reads a pipe until it closes, keeps only the last `limit_bytes` bytes, then closes it."""

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
        finally:
            # The reader closes its own pipe: closing it from another thread could block while
            # read1 still waits for a grandchild that keeps the pipe open.
            try:
                self.pipe.close()
            except OSError:
                pass


def _tail_text(reader: _TailReader, max_chars: int) -> str:
    """Decode the kept bytes and return at most max_chars characters, marker included."""
    # Code may write "\r\n" line endings; normalize so results look the same everywhere.
    text = bytes(reader.tail).decode("utf-8", errors="replace").replace("\r\n", "\n")
    if not reader.dropped and len(text) <= max_chars:
        return text
    if max_chars <= len(TRUNCATION_MARKER):
        return text[-max_chars:] if max_chars > 0 else ""
    return TRUNCATION_MARKER + text[-(max_chars - len(TRUNCATION_MARKER)):]


def _launcher(cpu_seconds: int) -> str:
    """Python code that runs first inside the box: set the limits, then start main.py.

    The limits are set inside the box on purpose: the process limit then counts only the box's
    own processes, not every process of the user. Soft and hard limits are equal, so the
    model's code cannot raise them again.
    """
    limits = {"RLIMIT_AS": MEMORY_BYTES, "RLIMIT_CPU": cpu_seconds, "RLIMIT_FSIZE": FILE_BYTES,
              "RLIMIT_NPROC": MAX_PROCESSES, "RLIMIT_NOFILE": MAX_OPEN_FILES, "RLIMIT_CORE": 0}
    return (
        "import os, resource, sys\n"
        f"for name, value in {limits!r}.items():\n"
        "    resource.setrlimit(getattr(resource, name), (value, value))\n"
        "os.execv(sys.executable, [sys.executable, '-I', '-X', 'utf8', 'main.py'])\n"
    )


def _python_dirs() -> list[str]:
    """Folders the box needs, besides /usr, to start this Python (read-only binds)."""
    needed = {sys.base_prefix, str(Path(os.path.realpath(sys.executable)).parent)}
    return sorted(d for d in needed if d != "/usr" and not d.startswith("/usr/"))


@functools.cache
def _tmpfs_size_supported() -> bool:
    """bwrap's --size option (limit a tmpfs) exists only in newer versions."""
    try:
        help_text = subprocess.run(["bwrap", "--help"], check=False, capture_output=True,
                                   text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "--size" in help_text


def _bwrap_command(script: Path, timeout_s: float) -> list[str]:
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--hostname",
           "sandbox", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME",
           WORK, "--ro-bind", "/usr", "/usr"]
    for name in ("bin", "sbin", "lib", "lib32", "lib64"):     # often links into /usr
        path = Path("/") / name
        if path.is_symlink():
            cmd += ["--symlink", os.readlink(path), str(path)]
        elif path.is_dir():
            cmd += ["--ro-bind", str(path), str(path)]
    cmd += ["--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    for folder in _python_dirs():
        cmd += ["--ro-bind", folder, folder]
    cmd += ["--proc", "/proc", "--dev", "/dev"]
    for folder in ("/tmp", WORK):
        if _tmpfs_size_supported():
            cmd += ["--size", str(TMPFS_BYTES)]
        cmd += ["--tmpfs", folder]
    cmd += ["--ro-bind", str(script), f"{WORK}/main.py", "--remount-ro", "/",
            "--chdir", WORK]
    python = os.path.realpath(sys.executable)
    return cmd + [python, "-I", "-X", "utf8", "-c", _launcher(math.ceil(timeout_s) + 1)]


@functools.cache
def _availability() -> str:
    """"" when the box works on this computer, otherwise the reason it does not."""
    if not sys.platform.startswith("linux"):
        return f"this is {sys.platform}, not Linux"
    if shutil.which("bwrap") is None:
        return "bwrap was not found"
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "main.py"
        script.write_text("print('ok')\n", encoding="utf-8")
        try:
            probe = subprocess.run(_bwrap_command(script, 5.0), check=False,
                                   capture_output=True, text=True, timeout=10, env={"PATH": "/usr/bin:/bin"})
        except (OSError, subprocess.SubprocessError) as exc:
            return f"bwrap could not start ({exc})"
    if probe.returncode != 0 or probe.stdout != "ok\n":
        return f"bwrap failed: {probe.stderr.strip()[:300] or probe.returncode}"
    return ""


def check_available() -> None:
    """Raise SandboxUnavailable if model-written code cannot be run safely here."""
    reason = _availability()
    if reason:
        raise SandboxUnavailable(f"{UNAVAILABLE_HELP} ({reason})")


def run_python(code: str, *, timeout_s: float = 5.0, max_output_chars: int = 4000) -> SandboxResult:
    check_available()
    timed_out = False
    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        script_path = Path(tmpdir) / "main.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            process = subprocess.Popen(
                _bwrap_command(script_path, timeout_s),
                cwd=tmpdir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={"PATH": "/usr/bin:/bin"},
                start_new_session=True,
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
            # Killing bwrap ends the box: --die-with-parent kills its process tree.
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
