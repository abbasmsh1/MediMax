"""
MediMax Configuration
Centralized settings using pydantic-settings.
"""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Mistral ──────────────────────────────────────────────────────────────
    MISTRAL_API_KEY: str = ""
    MISTRAL_LLM_MODEL: str = "mistral-large-latest"
    MISTRAL_EMBED_MODEL: str = "mistral-embed"
    USE_LOCAL_EMBEDDINGS: bool = False

    # ── Storage ──────────────────────────────────────────────────────────────
    VECTOR_STORE_TYPE: str = "chroma"  # "chroma" or "pinecone"
    CHROMA_PERSIST_DIR: str = "./storage/chroma_db"
    CHROMA_COLLECTION_NAME: str = "medimax_docs"

    # ── Pinecone ─────────────────────────────────────────────────────────────
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "medimax"

    # ── Chunking ─────────────────────────────────────────────────────────────
    # 1000 chars ≈ 5-8 medical sentences — enough context without overflow.
    # 150-char overlap ensures sentences cut at a boundary appear in both chunks.
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # ── Retrieval ─────────────────────────────────────────────────────────────
    RETRIEVAL_K: int = 5
    RETRIEVAL_FETCH_K: int = 20
    SIMILARITY_THRESHOLD: float = 0.35

    # ── LLM ──────────────────────────────────────────────────────────────────
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1024

    # ── API ───────────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True

    @property
    def chroma_path(self) -> Path:
        p = Path(self.CHROMA_PERSIST_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def use_mistral_embeddings(self) -> bool:
        return bool(self.MISTRAL_API_KEY) and not self.USE_LOCAL_EMBEDDINGS

    @property
    def is_pinecone(self) -> bool:
        return self.VECTOR_STORE_TYPE.lower() == "pinecone"


# Singleton
settings = Settings()
