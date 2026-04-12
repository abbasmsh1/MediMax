"""
Query Pipeline
End-to-end: Retrieve → Confidence Check → Generate → Format
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.embeddings.embedder import get_embedder
from app.generation.chain import MedicalRAGChain, RAGResponse
from app.retrieval.retriever import MedicalRetriever
from app.vectorstore import get_vector_store

logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    Orchestrates the medical Q&A query flow.
    Retrieves context → checks confidence → generates grounded answer.
    """

    def __init__(self):
        logger.info("Initializing QueryPipeline...")
        embedder = get_embedder()
        store = get_vector_store(embedder)
        self._retriever = MedicalRetriever(store)
        self._chain = MedicalRAGChain()

    def query(
        self,
        question: str,
        k: int | None = None,
        domain: Optional[str] = None,
        source: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Full query pipeline.
        
        Returns a structured dict with answer, sources, confidence, and metadata.
        """
        start = time.time()
        logger.info(f"Query received: '{question[:80]}...' " if len(question) > 80 else f"Query: '{question}'")

        # ── Step 1: Retrieve ──────────────────────────────────────────────
        retrieval = self._retriever.retrieve(
            query=question,
            k=k,
            domain=domain,
            source=source,
            year_from=year_from,
            year_to=year_to,
            use_mmr=True,
        )

        # ── Step 2: Generate ──────────────────────────────────────────────
        response: RAGResponse = self._chain.generate(
            question=question,
            retrieval_result=retrieval,
        )

        duration = round(time.time() - start, 2)
        logger.info(
            f"Query completed in {duration}s | "
            f"confidence={response.confidence} | "
            f"fallback={response.low_confidence_fallback}"
        )

        return {
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence,
            "max_similarity_score": round(response.max_score, 4),
            "retrieval_strategy": response.retrieval_strategy,
            "low_confidence_fallback": response.low_confidence_fallback,
            "chunks_retrieved": len(retrieval.documents),
            "duration_seconds": duration,
        }
