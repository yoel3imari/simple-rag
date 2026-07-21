"""Pytest fixtures for the simple-rag test suite."""

from unittest.mock import patch

import pytest


@pytest.fixture
def mock_ollama_chat():
    """Mock requests.post for the /api/chat endpoint.

    Returns a context manager that patches requests.post to return a
    realistic chat completion response from Ollama.
    """
    mock_response = {"message": {"content": "mocked response"}}
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None
        yield mock_post


@pytest.fixture
def mock_ollama_embed():
    """Mock requests.post for the /api/embed endpoint.

    Returns a context manager that patches requests.post to return a
    realistic embedding response from Ollama (768-dimensional vector).
    """
    mock_response = {"embeddings": [[0.1] * 768]}
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None
        yield mock_post
