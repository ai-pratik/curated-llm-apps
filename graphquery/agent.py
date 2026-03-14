"""
agent.py — Production RAG Agent
LangGraph + Qdrant (embedded) + nomic-embed-text + BM25 hybrid search + FlashRank reranker + streaming
"""

from __future__ import annotations

import os
import uuid
from typing import Annotated, Generator, List, TypedDict

import operator
from loguru import logger

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint

from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

from config import settings

# ──────────────────────────────────────────────────────────────────
# Optional: better PDF extractor. Falls back to PyPDFLoader if not
# available so the app still runs without pymupdf4llm installed.
# ──────────────────────────────────────────────────────────────────
try:
    import pymupdf4llm  # type: ignore
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False
    from langchain_community.document_loaders import PyPDFLoader


# ─────────────────────────── State ────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context:  str
    question: str
    answer:   str


# ─────────────────────────── RAGAgent ─────────────────────────────

class RAGAgent:
    """
    Fully self-contained RAG agent.

    * Qdrant (embedded, on-disk) — persistent without any server
    * nomic-embed-text for fast, accurate embeddings
    * BM25 + dense hybrid retrieval
    * FlashRank cross-encoder reranker
    * Streaming generation
    """

    def __init__(self) -> None:
        logger.info(
            "Initialising RAGAgent | model={} embed={} qdrant_path={}",
            settings.ollama_model,
            settings.embed_model,
            settings.qdrant_path,
        )

        # LLM
        self.llm = OllamaLLM(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
        )

        # Embeddings (dedicated model — NOT the LLM)
        self.embeddings = OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_base_url,
        )

        # Qdrant client — embedded (local on-disk) or remote
        if settings.qdrant_url:
            self.qdrant_client = QdrantClient(url=settings.qdrant_url)
            logger.info("Qdrant mode: remote @ {}", settings.qdrant_url)
        else:
            self.qdrant_client = QdrantClient(path=settings.qdrant_path)
            logger.info("Qdrant mode: embedded → {}", settings.qdrant_path)

        # Get embedding dimension and ensure collection exists
        self._embed_dim = len(self.embeddings.embed_query("ping"))
        self._ensure_collection()

        # BM25 index — rebuilt on each load_documents call
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: List[Document] = []

        # FlashRank reranker
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")

        # LangGraph compiled graph
        self.graph = self._build_graph()

        logger.info("RAGAgent ready")

    # ── Collection bootstrap ───────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't exist."""
        existing = {c.name for c in self.qdrant_client.get_collections().collections}
        if settings.qdrant_collection not in existing:
            self.qdrant_client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=self._embed_dim, distance=Distance.COSINE),
            )
            logger.info(
                "Created Qdrant collection '{}' (dim={})",
                settings.qdrant_collection, self._embed_dim,
            )

    # ── Dense search (direct client, no LC wrapper) ───────────────

    def _dense_search(self, question: str, k: int) -> List[Document]:
        """Embed query and run Qdrant vector search directly."""
        query_vec = self.embeddings.embed_query(question)
        results: List[ScoredPoint] = self.qdrant_client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vec,
            limit=k,
        ).points
        docs = []
        for r in results:
            payload = r.payload or {}
            docs.append(Document(
                page_content=payload.get("page_content", ""),
                metadata=payload.get("metadata", {}),
            ))
        return docs

    # ── Document loading ───────────────────────────────────────────

    def _load_file(self, path: str) -> List[Document]:
        ext = os.path.splitext(path)[1].lower()
        docs: List[Document] = []
        try:
            if ext == ".pdf":
                if _HAS_PYMUPDF:
                    md_text = pymupdf4llm.to_markdown(path)
                    docs = [Document(page_content=md_text, metadata={"source": path})]
                else:
                    docs = PyPDFLoader(path).load()
            elif ext in (".txt", ".md"):
                docs = TextLoader(path, encoding="utf-8").load()
            else:
                logger.warning("Unsupported file type: {}", path)
        except Exception as exc:
            logger.error("Could not load {}: {}", path, exc)
        return docs

    def load_documents(self, file_paths: List[str]) -> int:
        """Chunk, embed and index documents. Returns number of chunks added."""
        all_docs: List[Document] = []
        for path in file_paths:
            docs = self._load_file(path)
            all_docs.extend(docs)
            logger.info("Loaded {} pages/sections from {}", len(docs), os.path.basename(path))

        if not all_docs:
            return 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.split_documents(all_docs)
        logger.info("Split into {} chunks", len(chunks))

        # Embed all chunks
        texts = [c.page_content for c in chunks]
        vectors = self.embeddings.embed_documents(texts)

        # Upsert into Qdrant as PointStructs (works with embedded client)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "page_content": chunks[i].page_content,
                    "metadata": chunks[i].metadata,
                },
            )
            for i, vec in enumerate(vectors)
        ]
        self.qdrant_client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )

        # Rebuild BM25 index
        self._bm25_docs.extend(chunks)
        tokenised = [c.page_content.lower().split() for c in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenised)

        logger.info(
            "Indexed {} new chunks (BM25 rebuilt, total={})",
            len(chunks), len(self._bm25_docs),
        )
        return len(chunks)

    def clear_documents(self) -> None:
        """Wipe all documents and reset BM25."""
        self.qdrant_client.delete_collection(settings.qdrant_collection)
        self._ensure_collection()
        self._bm25 = None
        self._bm25_docs = []
        logger.info("Cleared all documents")

    # ── Hybrid retrieval ───────────────────────────────────────────

    def _hybrid_retrieve(self, question: str) -> List[Document]:
        """
        1. Dense vector search  → top-K candidates
        2. BM25 keyword search  → top-K candidates
        3. Merge unique results (union)
        4. FlashRank reranker   → return top-N
        """
        k = settings.retrieval_top_k
        n = settings.rerank_top_n

        # Dense
        dense_docs = self._dense_search(question, k=k)

        # BM25
        bm25_docs: List[Document] = []
        if self._bm25 and self._bm25_docs:
            scores = self._bm25.get_scores(question.lower().split())
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            bm25_docs = [self._bm25_docs[i] for i in top_idx]

        # Merge (deduplicate by first 120 chars)
        seen: set[str] = set()
        candidates: List[Document] = []
        for doc in dense_docs + bm25_docs:
            key = doc.page_content[:120]
            if key not in seen:
                seen.add(key)
                candidates.append(doc)

        if not candidates:
            return []

        # Rerank
        rerank_input = RerankRequest(
            query=question,
            passages=[{"id": i, "text": d.page_content} for i, d in enumerate(candidates)],
        )
        results = self.reranker.rerank(rerank_input)
        top_ids = [r["id"] for r in results[:n]]

        logger.debug(
            "Retrieval | dense={} bm25={} merged={} reranked={}",
            len(dense_docs), len(bm25_docs), len(candidates), len(top_ids),
        )
        return [candidates[i] for i in top_ids]

    # ── LangGraph nodes ────────────────────────────────────────────

    def _has_documents(self) -> bool:
        try:
            info = self.qdrant_client.get_collection(settings.qdrant_collection)
            return (info.points_count or 0) > 0
        except Exception:
            return False

    def retrieve(self, state: AgentState) -> AgentState:
        if not self._has_documents():
            return {**state, "context": "No documents loaded yet."}
        docs = self._hybrid_retrieve(state["question"])
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        return {**state, "context": context}

    def generate(self, state: AgentState) -> AgentState:
        prompt = (
            "You are a helpful assistant. Use ONLY the context below to answer.\n"
            "If the answer isn't in the context, say "
            "\"I couldn't find that in the uploaded documents.\"\n\n"
            f"Context:\n{state['context']}\n\n"
            f"Question: {state['question']}\n\n"
            "Answer:"
        )
        answer = self.llm.invoke(prompt)
        return {**state, "answer": answer, "messages": [AIMessage(content=answer)]}

    def _should_retrieve(self, state: AgentState) -> str:
        return "retrieve"

    # ── Graph ──────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        g = StateGraph(AgentState)
        g.add_node("retrieve", self.retrieve)
        g.add_node("generate", self.generate)
        g.set_conditional_entry_point(self._should_retrieve, {"retrieve": "retrieve"})
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
        return g.compile()

    # ── Public interface ───────────────────────────────────────────

    def chat(self, question: str, history: List) -> str:
        messages: List[BaseMessage] = []
        for user_msg, ai_msg in history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=ai_msg))
        messages.append(HumanMessage(content=question))
        result = self.graph.invoke({
            "messages": messages,
            "context": "",
            "question": question,
            "answer": "",
        })
        return result["answer"]

    def stream_chat(self, question: str, history: List) -> Generator[str, None, None]:
        """Stream the answer token-by-token."""
        if self._has_documents():
            docs = self._hybrid_retrieve(question)
            context = "\n\n---\n\n".join(d.page_content for d in docs)
        else:
            context = "No documents loaded yet."

        prompt = (
            "You are a helpful assistant. Use ONLY the context below to answer.\n"
            "If the answer isn't in the context, say "
            "\"I couldn't find that in the uploaded documents.\"\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        for chunk in self.llm.stream(prompt):
            yield chunk
