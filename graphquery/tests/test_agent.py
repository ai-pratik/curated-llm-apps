"""
tests/test_agent.py — Unit tests for RAGAgent
Mocks Qdrant and OllamaLLM so tests run offline with no external services.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from langchain_core.documents import Document


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def patch_agent_deps():
    """Patch all I/O dependencies so RAGAgent can be instantiated in CI."""
    with (
        patch("agent.OllamaLLM") as mock_llm_cls,
        patch("agent.OllamaEmbeddings") as mock_emb_cls,
        patch("agent.QdrantClient") as mock_qdrant_cls,
        patch("agent.LangchainQdrant") as mock_vs_cls,
        patch("agent.Ranker") as mock_ranker_cls,
    ):
        # Embedding size
        mock_emb_cls.return_value.embed_query.return_value = [0.0] * 768

        # Qdrant collections
        mock_qdrant_cls.return_value.get_collections.return_value = MagicMock(
            collections=[]
        )
        mock_qdrant_cls.return_value.get_collection.return_value = MagicMock(
            vectors_count=0
        )

        # Vector store
        mock_vs_cls.return_value.similarity_search.return_value = [
            Document(page_content="The capital of France is Paris.", metadata={})
        ]

        # LLM
        mock_llm_cls.return_value.invoke.return_value = "Paris"
        mock_llm_cls.return_value.stream.return_value = iter(["Pa", "ris"])

        # Reranker
        mock_ranker_cls.return_value.rerank.return_value = [{"id": 0, "score": 0.9}]

        yield


@pytest.fixture
def agent():
    from agent import RAGAgent
    return RAGAgent()


# ── Tests ──────────────────────────────────────────────────────────

def test_retrieve_no_documents(agent):
    """When no docs are loaded, retrieve returns a fallback context."""
    state = {"messages": [], "context": "", "question": "What is Python?", "answer": ""}
    result = agent.retrieve(state)
    assert "No documents" in result["context"]


def test_retrieve_with_documents(agent):
    """When BM25 docs exist, retrieve calls hybrid search."""
    from langchain_core.documents import Document
    agent._bm25_docs = [Document(page_content="Paris is the capital of France.")]
    agent._bm25 = MagicMock()
    agent._bm25.get_scores.return_value = [0.9]

    state = {"messages": [], "context": "", "question": "capital of France?", "answer": ""}
    result = agent.retrieve(state)
    assert result["context"] != ""


def test_generate_answer(agent):
    """Generate node calls LLM and stores answer in state."""
    state = {
        "messages": [],
        "context": "The capital of France is Paris.",
        "question": "What is the capital of France?",
        "answer": "",
    }
    result = agent.generate(state)
    assert result["answer"] == "Paris"
    assert len(result["messages"]) == 1


def test_stream_chat_yields_tokens(agent):
    """stream_chat yields tokens when context is available."""
    agent._bm25_docs = [Document(page_content="Paris is the capital of France.")]
    agent._bm25 = MagicMock()
    agent._bm25.get_scores.return_value = [0.9]

    tokens = list(agent.stream_chat("What is the capital?", []))
    assert tokens == ["Pa", "ris"]


def test_load_documents_empty(agent, tmp_path):
    """load_documents returns 0 if no valid files are provided."""
    # Provide a file type that's not supported
    f = tmp_path / "data.csv"
    f.write_text("col1,col2\n1,2")
    count = agent.load_documents([str(f)])
    assert count == 0
