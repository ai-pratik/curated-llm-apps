"""
config.py — centralised settings via pydantic-settings + .env
All tuneable parameters live here. Import `settings` anywhere in the project.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"

    # ── Embeddings ───────────────────────────────────────────────
    embed_model: str = "nomic-embed-text"

    # ── Qdrant ───────────────────────────────────────────────────
    qdrant_url: str = ""                          # leave empty to use embedded mode
    qdrant_path: str = "./qdrant_storage"          # used when qdrant_url is empty
    qdrant_collection: str = "rag_documents"

    # ── Chunking ─────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 100

    # ── Retrieval ────────────────────────────────────────────────
    retrieval_top_k: int = 20        # candidates before reranking
    rerank_top_n: int = 4            # final chunks sent to LLM

    # ── API ──────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Gradio ───────────────────────────────────────────────────
    gradio_host: str = "0.0.0.0"
    gradio_port: int = 7860


settings = Settings()
