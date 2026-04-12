"""
Query API Routes
POST /query -> Medical Q&A with grounded RAG response
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.api.schemas import QueryRequest, QueryResponse
from app.pipeline.query_pipeline import QueryPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@lru_cache(maxsize=1)
def get_query_pipeline() -> QueryPipeline:
    """Singleton query pipeline (lazy init on first request)."""
    return QueryPipeline()


@router.post("", response_model=QueryResponse, summary="Ask a medical question")
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Answer a medical question using the RAG knowledge base.

    - Retrieves relevant context using MMR search
    - Generates grounded answer using Mistral LLM
    - Returns answer with source citations and confidence score
    - Returns 'I don't know' if context is insufficient (anti-hallucination)

    The LLM/retrieval work is offloaded to a thread so the event loop
    stays responsive while waiting for the model.
    """
    try:
        pipeline = get_query_pipeline()
        result = await asyncio.to_thread(
            pipeline.query,
            question=request.question,
            k=request.k,
            domain=request.domain,
            source=request.source,
            year_from=request.year_from,
            year_to=request.year_to,
        )
        return QueryResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(exc)}")

