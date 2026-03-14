"""
api.py — FastAPI backend for the RAG Agent
Endpoints: /upload, /chat (SSE streaming), /documents (DELETE), /health
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agent import RAGAgent
from config import settings

# ── Shared agent instance ──────────────────────────────────────────

_agent: RAGAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    logger.info("Starting up — initialising RAGAgent")
    _agent = RAGAgent()
    yield
    logger.info("Shutting down")


# ── App ────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Agent API",
    description="Production RAG backend: Qdrant + Ollama + LangGraph",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_agent() -> RAGAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised yet")
    return _agent


# ── Request / Response models ──────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    history: List[List[str]] = []   # [[user, assistant], ...]


class UploadResponse(BaseModel):
    chunks: int
    files: List[str]
    message: str


class HealthResponse(BaseModel):
    status: str
    model: str
    embed_model: str
    qdrant: str


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness + readiness check."""
    agent = get_agent()
    try:
        agent.qdrant_client.get_collections()
        qdrant_status = "connected"
    except Exception as exc:
        logger.warning("Qdrant health check failed: {}", exc)
        qdrant_status = f"error: {exc}"

    return HealthResponse(
        status="ok",
        model=settings.ollama_model,
        embed_model=settings.embed_model,
        qdrant=qdrant_status,
    )


@app.post("/upload", response_model=UploadResponse, tags=["documents"])
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    """
    Upload and index one or more PDF / TXT / MD files.
    Files are saved to a temp directory, indexed, then cleaned up.
    """
    agent = get_agent()
    saved_paths: List[str] = []
    filenames: List[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for upload in files:
            dest = os.path.join(tmp_dir, upload.filename or "upload")
            content = await upload.read()
            with open(dest, "wb") as f:
                f.write(content)
            saved_paths.append(dest)
            filenames.append(upload.filename or "unknown")
            logger.info("Received file: {} ({} bytes)", upload.filename, len(content))

        # Run indexing in a thread pool so we don't block the event loop
        chunks = await asyncio.get_event_loop().run_in_executor(
            None, agent.load_documents, saved_paths
        )

    if chunks == 0:
        raise HTTPException(
            status_code=422,
            detail="No supported content found. Use .pdf, .txt, or .md files.",
        )

    return UploadResponse(
        chunks=chunks,
        files=filenames,
        message=f"Indexed {chunks} chunks from {len(filenames)} file(s).",
    )


@app.post("/chat", tags=["chat"])
async def chat(req: ChatRequest) -> StreamingResponse:
    """
    Stream an SSE response for the given question.
    Each event is a JSON object: {"token": "..."} or {"done": true}.
    """
    agent = get_agent()
    logger.debug("CHAT REQUEST | question={!r} | history={}", req.question, req.history)

    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    async def event_generator():
        try:
            loop = asyncio.get_event_loop()
            # stream_chat is a sync generator — iterate in thread pool
            gen = agent.stream_chat(req.question, req.history)
            for token in gen:
                yield {
                    "event": "token",
                    "data": json.dumps({"token": token}),
                }
                await asyncio.sleep(0)   # yield control to event loop
            yield {"event": "done", "data": json.dumps({"done": True})}
        except Exception as exc:
            logger.error("Stream error: {}", exc)
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(event_generator())


@app.delete("/documents", tags=["documents"])
async def clear_documents() -> dict:
    """Wipe all indexed documents from Qdrant."""
    agent = get_agent()
    await asyncio.get_event_loop().run_in_executor(None, agent.clear_documents)
    return {"message": "All documents cleared."}


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
