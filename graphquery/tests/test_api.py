"""
tests/test_api.py — Integration tests for the FastAPI backend
Uses FastAPI's TestClient with the RAGAgent mocked out.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Mock agent ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_agent():
    agent = MagicMock()
    agent.qdrant_client.get_collections.return_value = MagicMock(collections=[])
    agent.load_documents.return_value = 5
    agent.stream_chat.return_value = iter(["Hello", " world"])
    agent.clear_documents.return_value = None
    return agent


@pytest.fixture(scope="module")
def client(mock_agent):
    with patch("api._agent", mock_agent):
        from api import app
        with TestClient(app) as c:
            yield c


# ── Tests ──────────────────────────────────────────────────────────

def test_health(client, mock_agent):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "qdrant" in data


def test_upload_no_files(client):
    resp = client.post("/upload", files=[])
    # FastAPI returns 422 for missing required field
    assert resp.status_code == 422


def test_upload_unsupported_type(client, mock_agent, tmp_path):
    mock_agent.load_documents.return_value = 0
    f = tmp_path / "data.csv"
    f.write_bytes(b"col1,col2\n1,2")
    resp = client.post(
        "/upload",
        files={"files": ("data.csv", f.read_bytes(), "text/csv")},
    )
    assert resp.status_code == 422


def test_upload_success(client, mock_agent, tmp_path):
    mock_agent.load_documents.return_value = 7
    f = tmp_path / "doc.txt"
    f.write_text("This is a test document.")
    resp = client.post(
        "/upload",
        files={"files": ("doc.txt", f.read_bytes(), "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunks"] == 7
    assert "doc.txt" in data["files"]


def test_chat_empty_question(client):
    resp = client.post("/chat", json={"question": "   ", "history": []})
    assert resp.status_code == 422


def test_clear_documents(client, mock_agent):
    resp = client.delete("/documents")
    assert resp.status_code == 200
    assert "cleared" in resp.json()["message"].lower()
