"""
Ingest API Routes
POST /ingest/file    -> Upload and index a single document
POST /ingest/path    -> Index a server-side directory path
GET  /ingest/sources -> List indexed sources
DELETE /ingest/source -> Remove a source from the index
GET  /ingest/stats   -> Vector store statistics
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.schemas import (
    DeleteSourceRequest,
    DeleteSourceResponse,
    IngestBatchResponse,
    IngestFileResponse,
    StatsResponse,
)
from app.config import settings
from app.pipeline.ingest_pipeline import IngestPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingest"])


@lru_cache(maxsize=1)
def get_ingest_pipeline() -> IngestPipeline:
    """Singleton ingest pipeline."""
    return IngestPipeline()


@router.post("/file", response_model=IngestFileResponse, summary="Upload and index a document")
async def ingest_file(file: UploadFile = File(...)) -> IngestFileResponse:
    """
    Upload a medical document (PDF, TXT, MD, DOCX) and index it into the vector
    store.  Supports incremental indexing — existing chunks are upserted safely.
    The heavy PDF-parsing + embedding work is offloaded to a thread so the event
    loop stays responsive during long ingestion jobs.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    MAX_SIZE = 250 * 1024 * 1024  # 250 MB
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 250 MB)")

    filename = file.filename
    content_type = file.content_type or "application/pdf"

    try:
        pipeline = get_ingest_pipeline()

        # Run blocking ingest in a thread — keeps the event loop free
        result = await asyncio.to_thread(
            pipeline.ingest_bytes,
            content=content,
            filename=filename,
            content_type=content_type,
        )

        if not result.success:
            raise HTTPException(status_code=422, detail=result.error)
        return IngestFileResponse(**result.__dict__)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ingest error for %s: %s", filename, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/path", response_model=IngestBatchResponse, summary="Index documents from a directory path")
async def ingest_directory(path: str = Form(...)) -> IngestBatchResponse:
    """
    Scan and index all supported documents from a server-side directory path.
    Offloaded to a thread to avoid blocking the event loop.
    """
    dir_path = Path(path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

    try:
        pipeline = get_ingest_pipeline()
        results = await asyncio.to_thread(pipeline.ingest_directory, dir_path)
        successful = sum(1 for r in results if r.success)
        return IngestBatchResponse(
            total_files=len(results),
            successful=successful,
            failed=len(results) - successful,
            results=[IngestFileResponse(**r.__dict__) for r in results],
        )
    except Exception as exc:
        logger.error("Batch ingest error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sources", summary="List indexed source documents")
async def list_sources() -> dict:
    """
    Returns a sorted list of unique source documents.
    Runs in a thread with a 90-second timeout so large collections
    (100k+ chunks) don't hang the event loop indefinitely.
    """
    pipeline = get_ingest_pipeline()
    try:
        sources = await asyncio.wait_for(
            asyncio.to_thread(pipeline.list_sources),
            timeout=90.0,
        )
    except asyncio.TimeoutError:
        logger.warning("/sources timed out — returning partial or empty list")
        sources = []
    return {"sources": sources, "count": len(sources)}


@router.delete("/source", response_model=DeleteSourceResponse, summary="Remove a source from the index")
async def delete_source(request: DeleteSourceRequest) -> DeleteSourceResponse:
    pipeline = get_ingest_pipeline()
    deleted = await asyncio.to_thread(pipeline.delete_source, request.source_name)
    return DeleteSourceResponse(
        source_name=request.source_name,
        chunks_deleted=deleted,
    )


def _fast_chroma_stats() -> dict:
    """
    Read chunk count directly from ChromaDB's SQLite file.

    This bypasses the pipeline singleton and embedding model entirely,
    so it never deadlocks against the background auto-ingest thread.
    Opens the DB in read-only (immutable) mode so it doesn't compete
    with ongoing writes.  Falls back to 0 on any error.
    """
    db_path = settings.chroma_path / "chroma.sqlite3"
    total = 0
    try:
        # Use URI mode for read-only access — safe concurrent with writer
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM embeddings")
            row = cur.fetchone()
            total = row[0] if row else 0
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("fast_chroma_stats failed: %s", exc)
    return {
        "total_chunks": total,
        "collection_name": settings.CHROMA_COLLECTION_NAME,
        "persist_dir": str(settings.chroma_path),
    }


@router.get("/stats", response_model=StatsResponse, summary="Vector store statistics")
async def get_stats() -> StatsResponse:
    """
    Returns fast statistics (chunk count) by reading ChromaDB's SQLite
    file directly — no pipeline lock, no embedder, instant response.
    For the full source list use GET /ingest/sources.
    """
    # Run in thread so sqlite I/O doesn't block the event loop
    stats = await asyncio.to_thread(_fast_chroma_stats)
    embedding_mode = "mistral-embed" if settings.use_mistral_embeddings else "local-minilm"
    return StatsResponse(
        total_chunks=stats["total_chunks"],
        collection_name=stats["collection_name"],
        persist_dir=stats["persist_dir"],
        sources=[],          # use GET /ingest/sources for the full list
        embedding_mode=embedding_mode,
    )
