import gc
import warnings
from pathlib import Path

import pytest

from agentic import sandbox
from agentic.sandbox import SandboxUnavailable, run_python

needs_box = pytest.mark.skipif(bool(sandbox._availability()),
                               reason="bubblewrap sandbox not available here")


@pytest.fixture(autouse=True)
def _fresh_warning(monkeypatch):
    monkeypatch.setattr(sandbox, "_warned", False)


@pytest.fixture(params=["bubblewrap", "basic"])
def mode(request, monkeypatch):
    """Run a test once in the bubblewrap box (where available) and once in the basic sandbox."""
    if request.param == "bubblewrap":
        if sandbox._availability():
            pytest.skip("bubblewrap sandbox not available here")
    else:
        monkeypatch.setattr(sandbox, "_availability", lambda: "turned off for this test")
        monkeypatch.setenv(sandbox.ALLOW_BASIC_ENV, "1")
    assert sandbox.sandbox_mode() == request.param
    return request.param


def test_sandbox_passing(mode):
    code = "print('hello from sandbox')"
    result = run_python(code)
    assert result.passed is True
    assert result.timed_out is False
    assert result.returncode == 0
    assert "hello from sandbox" in result.stdout


def test_sandbox_failing_assert(mode):
    code = "assert 1 == 2"
    result = run_python(code)
    assert result.passed is False
    assert result.returncode != 0
    assert "AssertionError" in result.stderr


def test_sandbox_exception(mode):
    code = "raise ValueError('custom error')"
    result = run_python(code)
    assert result.passed is False
    assert result.returncode != 0
    assert "ValueError: custom error" in result.stderr


def test_sandbox_timeout(mode):
    code = "import time\nwhile True:\n    time.sleep(0.1)"
    result = run_python(code, timeout_s=1)
    assert result.passed is False
    assert result.timed_out is True
    assert result.duration_ms < 5000


def test_sandbox_output_truncation(mode):
    code = "print('a' * 5000 + 'END')"
    result = run_python(code, max_output_chars=100)
    assert len(result.stdout) <= 100
    assert "[truncated]" in result.stdout
    assert result.stdout.rstrip().endswith("END")      # the tail is kept


def test_sandbox_short_output_is_not_truncated(mode):
    result = run_python("print('x' * 50)", max_output_chars=100)
    assert result.stdout == "x" * 50 + "\n"


def test_sandbox_endless_output_keeps_timeout_and_memory_bounded(mode):
    code = "import sys\nwhile True:\n    print('x' * 1000)\n    print('e' * 1000, file=sys.stderr)"
    result = run_python(code, timeout_s=1, max_output_chars=200)
    assert result.timed_out is True
    assert result.duration_ms < 5000
    assert len(result.stdout) <= 200
    assert len(result.stderr) <= 200


def test_sandbox_grandchild_holding_the_pipe_does_not_block(mode):
    # The child starts a detached grandchild that inherits stdout, then exits at once.
    code = (
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(4)'],\n"
        "                 start_new_session=True)\n"
        "print('child done')\n"
    )
    result = run_python(code, timeout_s=5)
    assert result.passed is True
    assert "child done" in result.stdout
    assert result.duration_ms < 2500


def test_sandbox_non_ascii(mode):
    code = "print('naïve café, 日本語, ✓')"
    result = run_python(code)
    assert result.passed is True
    assert "naïve café, 日本語, ✓" in result.stdout


def test_sandbox_child_runs_in_utf8_mode(mode):
    result = run_python("import sys\nprint(sys.flags.utf8_mode)")
    assert result.stdout == "1\n"


def test_sandbox_normalizes_windows_line_endings(mode):
    result = run_python("import sys\nsys.stdout.buffer.write(b'a\\r\\nb\\r\\n')")
    assert result.stdout == "a\nb\n"


def test_sandbox_clean_environment(mode, monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "secret123")
    code = "import os\nprint(os.environ.get('FAKE_API_KEY', 'not found'))"
    result = run_python(code)
    assert result.passed is True
    assert "not found" in result.stdout


def test_sandbox_cwd_has_no_repo_files(mode):
    code = (
        "import os\n"
        "print('README.md' in os.listdir('.'))\n"
        "print('agentic' in os.listdir('.'))\n"
    )
    result = run_python(code)
    assert result.passed is True
    assert result.stdout.split() == ["False", "False"]


@pytest.mark.parametrize("code", ["print('done')", "while True: pass"])
def test_sandbox_closes_pipes(mode, code):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        run_python(code, timeout_s=0.5)
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]


REPO = Path(__file__).resolve().parents[1]


@needs_box
def test_sandbox_sees_no_home_and_no_repository():
    code = (f"import os\nprint(os.path.exists({str(REPO)!r}), os.path.exists('/home'),\n"
            f"      os.path.exists({str(Path.home())!r}), os.getcwd())")
    result = run_python(code)
    assert result.stdout.split() == ["False", "False", "False", "/work"]


@needs_box
def test_sandbox_has_no_network():
    code = ("import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
            "    print('connected')\n"
            "except OSError:\n"
            "    print('blocked')\n")
    assert run_python(code).stdout == "blocked\n"


@needs_box
def test_sandbox_writes_only_inside_the_box():
    code = ("open('note.txt', 'w').write('fine')\n"
            "open('/tmp/x', 'w').write('fine')\n"
            "for path in ('/usr/evil.txt', '/evil.txt'):\n"
            "    try:\n"
            "        open(path, 'w')\n"
            "        print('wrote', path)\n"
            "    except OSError:\n"
            "        print('denied')\n")
    result = run_python(code)
    assert result.stdout.split() == ["denied", "denied"]


@needs_box
def test_sandbox_environment_is_empty(monkeypatch):
    monkeypatch.setenv("SOME_SETTING", "visible?")
    result = run_python("import os\nprint(' '.join(sorted(os.environ)))")
    # PWD comes from bwrap, LC_CTYPE from Python's UTF-8 mode; nothing from outside.
    assert set(result.stdout.split()) <= {"HOME", "PATH", "PWD", "LC_CTYPE"}


@needs_box
def test_sandbox_limits_memory():
    result = run_python("x = bytearray(2 * 1024 ** 3)")
    assert result.passed is False
    assert "MemoryError" in result.stderr


@needs_box
def test_sandbox_limits_file_size():
    result = run_python("open('big', 'wb').write(b'x' * (32 * 1024 * 1024))")
    assert result.passed is False


@needs_box
def test_sandbox_limits_processes():
    code = ("import os, time\n"
            "n = 0\n"
            "try:\n"
            "    for _ in range(200):\n"
            "        if os.fork() == 0:\n"
            "            time.sleep(2)\n"
            "            os._exit(0)\n"
            "        n += 1\n"
            "except OSError:\n"
            "    pass\n"
            "print(n)\n")
    result = run_python(code)
    assert int(result.stdout) < sandbox.MAX_PROCESSES


@needs_box
def test_sandbox_limits_cannot_be_raised():
    code = ("import resource\n"
            "try:\n"
            "    resource.setrlimit(resource.RLIMIT_AS, (-1, -1))\n"
            "    print('raised')\n"
            "except (ValueError, OSError):\n"
            "    print('refused')\n")
    assert run_python(code).stdout == "refused\n"


def test_sandbox_refuses_to_run_without_bubblewrap(monkeypatch, tmp_path):
    marker = tmp_path / "ran"
    monkeypatch.setattr(sandbox, "_availability", lambda: "bwrap was not found")
    monkeypatch.delenv(sandbox.ALLOW_BASIC_ENV, raising=False)
    with pytest.raises(SandboxUnavailable, match="WSL"):
        run_python(f"open({str(marker)!r}, 'w')")
    assert not marker.exists()


def test_basic_sandbox_warns_once(monkeypatch, capsys):
    monkeypatch.setattr(sandbox, "_availability", lambda: "bwrap was not found")
    monkeypatch.setenv(sandbox.ALLOW_BASIC_ENV, "1")
    assert run_python("print('a')").stdout == "a\n"
    assert run_python("print('b')").stdout == "b\n"
    assert capsys.readouterr().err.count("basic sandbox") == 1
