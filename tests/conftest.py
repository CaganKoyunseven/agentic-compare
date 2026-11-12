"""Shared test setup."""

import os

from agentic.sandbox import ALLOW_BASIC_ENV

# Tests run only our own fixed code, never model output, so on systems without bubblewrap
# (Windows, macOS) they may use the basic sandbox. Where bubblewrap works it is used anyway.
os.environ.setdefault(ALLOW_BASIC_ENV, "1")
