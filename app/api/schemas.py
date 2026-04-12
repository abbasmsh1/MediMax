"""
API Request/Response Schemas
Pydantic models for FastAPI endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestFileResponse(BaseModel):
    source: str
    pages_loaded: int
    chunks_created: int
    chunks_indexed: int
    duration_seconds: float
    success: bool
    error: Optional[str] = None


class IngestBatchResponse(BaseModel):
    total_files: int
    successful: int
    failed: int
    results: List[IngestFileResponse]


class DeleteSourceRequest(BaseModel):
    source_name: str = Field(..., description="Filename of the source to delete")


class DeleteSourceResponse(BaseModel):
    source_name: str
    chunks_deleted: int


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The medical question to answer",
        examples=["What is the first-line treatment for hypertension?"],
    )
    k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (default: 5)",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Filter by medical domain (e.g., cardiology, oncology)",
    )
    source: Optional[str] = Field(
        default=None,
        description="Filter to a specific source document",
    )
    year_from: Optional[int] = Field(
        default=None,
        description="Filter to documents from this year onwards",
    )
    year_to: Optional[int] = Field(
        default=None,
        description="Filter to documents up to this year",
    )


class SourceCitation(BaseModel):
    source: str
    page: Any
    chunk: Any
    domain: str
    title: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: str  # High | Medium | Low
    max_similarity_score: float
    retrieval_strategy: str
    low_confidence_fallback: bool
    chunks_retrieved: int
    duration_seconds: float


# ── System ────────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    total_chunks: int
    collection_name: str
    persist_dir: str
    sources: List[str]
    embedding_mode: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    embedding_mode: str
