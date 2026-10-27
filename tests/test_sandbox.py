import os
import sys
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
    result = run_python(code, timeout_s=1.0)
    assert result.passed is False
    assert result.timed_out is True

def test_sandbox_output_truncation():
    code = "print('a' * 5000)"
    result = run_python(code, max_output_chars=100)
    assert len(result.stdout) <= 120
    assert "[truncated]" in result.stdout

def test_sandbox_non_ascii():
    code = "print('Türkçe karakterler: çğüöşİ')"
    result = run_python(code)
    assert result.passed is True
    assert "Türkçe karakterler: çğüöşİ" in result.stdout

def test_sandbox_clean_environment():
    os.environ["FAKE_API_KEY"] = "secret123"
    code = "import os\nprint(os.environ.get('FAKE_API_KEY', 'not found'))"
    result = run_python(code)
    assert result.passed is True
    assert "not found" in result.stdout
    if "FAKE_API_KEY" in os.environ:
        del os.environ["FAKE_API_KEY"]
