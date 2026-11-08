import gc
import warnings

import pytest

from agentic.sandbox import run_python


def test_sandbox_passing():
    code = "print('hello from sandbox')"
    result = run_python(code)
    assert result.passed is True
    assert result.timed_out is False
    assert result.returncode == 0
    assert "hello from sandbox" in result.stdout


def test_sandbox_failing_assert():
    code = "assert 1 == 2"
    result = run_python(code)
    assert result.passed is False
    assert result.returncode != 0
    assert "AssertionError" in result.stderr


def test_sandbox_exception():
    code = "raise ValueError('custom error')"
    result = run_python(code)
    assert result.passed is False
    assert result.returncode != 0
    assert "ValueError: custom error" in result.stderr


def test_sandbox_timeout():
    code = "import time\nwhile True:\n    time.sleep(0.1)"
    result = run_python(code, timeout_s=1)
    assert result.passed is False
    assert result.timed_out is True
    assert result.duration_ms < 5000


def test_sandbox_output_truncation():
    code = "print('a' * 5000 + 'END')"
    result = run_python(code, max_output_chars=100)
    assert len(result.stdout) <= 100
    assert "[truncated]" in result.stdout
    assert result.stdout.rstrip().endswith("END")      # the tail is kept


def test_sandbox_short_output_is_not_truncated():
    result = run_python("print('x' * 50)", max_output_chars=100)
    assert result.stdout == "x" * 50 + "\n"


def test_sandbox_endless_output_keeps_timeout_and_memory_bounded():
    code = "import sys\nwhile True:\n    print('x' * 1000)\n    print('e' * 1000, file=sys.stderr)"
    result = run_python(code, timeout_s=1, max_output_chars=200)
    assert result.timed_out is True
    assert result.duration_ms < 5000
    assert len(result.stdout) <= 200
    assert len(result.stderr) <= 200


def test_sandbox_grandchild_holding_the_pipe_does_not_block():
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


def test_sandbox_non_ascii():
    code = "print('naïve café, 日本語, ✓')"
    result = run_python(code)
    assert result.passed is True
    assert "naïve café, 日本語, ✓" in result.stdout


def test_sandbox_child_runs_in_utf8_mode():
    result = run_python("import sys\nprint(sys.flags.utf8_mode)")
    assert result.stdout == "1\n"


def test_sandbox_normalizes_windows_line_endings():
    result = run_python("import sys\nsys.stdout.buffer.write(b'a\\r\\nb\\r\\n')")
    assert result.stdout == "a\nb\n"


def test_sandbox_clean_environment(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "secret123")
    code = "import os\nprint(os.environ.get('FAKE_API_KEY', 'not found'))"
    result = run_python(code)
    assert result.passed is True
    assert "not found" in result.stdout


def test_sandbox_cwd_has_no_repo_files():
    code = (
        "import os\n"
        "print('README.md' in os.listdir('.'))\n"
        "print('agentic' in os.listdir('.'))\n"
    )
    result = run_python(code)
    assert result.passed is True
    assert result.stdout.split() == ["False", "False"]


@pytest.mark.parametrize("code", ["print('done')", "while True: pass"])
def test_sandbox_closes_pipes(code):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        run_python(code, timeout_s=0.5)
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]
