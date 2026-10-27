"""
Sandbox runner for executing model-written Python code safely in an isolated temporary directory.
Note: Network and memory limits are not enforced as they are not portable across platforms (e.g., Windows).
"""

from dataclasses import dataclass
import os
import subprocess
import sys
import tempfile
import time
import signal

@dataclass(frozen=True)
class SandboxResult:
    passed: bool
    timed_out: bool
    returncode: int | None
    stdout: str
    stderr: str
    duration_ms: int

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
        script_path = os.path.join(tmpdir, "main.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Platform-specific process group handling
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        try:
            process = subprocess.Popen(
                [sys.executable, "-I", script_path],
                cwd=tmpdir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                **kwargs
            )
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return SandboxResult(
                passed=False,
                timed_out=False,
                returncode=None,
                stdout="",
                stderr=str(e),
                duration_ms=duration_ms
            )

        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=timeout_s)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            if sys.platform == "win32":
                process.kill()
            else:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    process.kill()
            stdout_bytes, stderr_bytes = process.communicate()
            returncode = process.returncode

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        stdout_str = stdout_bytes.decode("utf-8", errors="replace")
        stderr_str = stderr_bytes.decode("utf-8", errors="replace")

        if len(stdout_str) > max_output_chars:
            stdout_str = "...[truncated]...\n" + stdout_str[-max_output_chars:]
        if len(stderr_str) > max_output_chars:
            stderr_str = "...[truncated]...\n" + stderr_str[-max_output_chars:]

        passed = (returncode == 0) and not timed_out

        return SandboxResult(
            passed=passed,
            timed_out=timed_out,
            returncode=returncode,
            stdout=stdout_str,
            stderr=stderr_str,
            duration_ms=duration_ms
        )
