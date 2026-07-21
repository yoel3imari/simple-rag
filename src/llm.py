"""Shared Ollama client — centralises all HTTP calls to the Ollama API.

Usage:
    from src.llm import ollama_chat, ollama_embed

    reply = ollama_chat("gemma3:1b", "Hello!")
    vectors = ollama_embed("nomic-embed-text", ["text1", "text2"])
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import requests

from src.config import OLLAMA_BASE_URL


def _request(
    endpoint: str,
    payload: dict[str, Any],
    timeout: int = 120,
    max_retries: int = 1,
) -> dict[str, Any]:
    """Send a POST request to the Ollama API with retry logic.

    Args:
        endpoint: API path (e.g. ``"/api/chat"``, ``"/api/embed"``).
        payload: JSON body.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on failure (0 = no retry).

    Returns:
        Parsed JSON response dict.

    Raises:
        requests.RequestException: If all retries are exhausted.
    """
    url = f"{OLLAMA_BASE_URL}{endpoint}"
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))  # simple backoff
            else:
                raise

    # This line is never reached but satisfies the type checker
    raise RuntimeError("Unexpected: all retries exhausted") from last_error


def ollama_chat(
    model: str,
    prompt: str,
    timeout: int = 120,
) -> str:
    """Send a chat prompt to Ollama and return the response text.

    Args:
        model: Ollama model name (e.g. ``"gemma3:1b"``).
        prompt: User message content.
        timeout: Request timeout in seconds.

    Returns:
        The model's response text, stripped of leading/trailing whitespace.
    """
    data = _request(
        "/api/chat",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=timeout,
    )
    return data["message"]["content"].strip()


def ollama_embed(
    model: str,
    texts: list[str],
    timeout: int = 300,
) -> np.ndarray:
    """Embed a list of texts using Ollama and return as a float32 array.

    Args:
        model: Ollama model name (e.g. ``"nomic-embed-text"``).
        texts: List of text strings to embed.
        timeout: Request timeout in seconds.

    Returns:
        ``np.ndarray`` of shape ``(len(texts), embedding_dim)`` with dtype
        ``float32``.
    """
    data = _request(
        "/api/embed",
        {"model": model, "input": texts},
        timeout=timeout,
    )
    return np.array(data["embeddings"], dtype=np.float32)
