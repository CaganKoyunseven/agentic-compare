"""Chooses the LLM client from LLM_BACKEND in .env. The default is the local Ollama client."""

from typing import Any

from agentic.config import llm_backend
from agentic.llm.ollama import OllamaClient
from agentic.llm.openai_api import OpenAIClient


def make_client(model: str, **settings: Any) -> OllamaClient | OpenAIClient:
    """A client for `model`. The caller passes the same settings for every backend."""
    backend = llm_backend()
    if backend == "ollama":
        return OllamaClient(model, **settings)
    if backend == "openai":
        return OpenAIClient(model, **settings)
    raise ValueError(f"Unknown LLM_BACKEND {backend!r}; use ollama or openai")
