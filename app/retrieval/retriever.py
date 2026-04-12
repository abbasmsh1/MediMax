"""
Medical Retriever
Top-K + MMR retrieval with confidence threshold filtering.
Supports metadata filters (domain, year, source).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from app.config import settings
from app.vectorstore.chroma_store import MedicalVectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval output."""
    documents: List[Document] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    is_confident: bool = True
    max_score: float = 0.0
    strategy: str = "mmr"

    @property
    def context_text(self) -> str:
        """Format retrieved chunks into a single context block for the LLM."""
        if not self.documents:
            return "No relevant context found."
        parts = []
        for i, doc in enumerate(self.documents, 1):
            meta = doc.metadata
            source = meta.get("source", "unknown")
            page = meta.get("page", "?")
            chunk = meta.get("chunk_index", "?")
            domain = meta.get("domain", "general")
            parts.append(
                f"[Source {i}] {source} | Page {page} | Chunk {chunk} | Domain: {domain}\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    @property
    def source_citations(self) -> List[Dict[str, Any]]:
        """Structured list of source citations."""
        # For mapping documents to their scores, we try to match by content,
        # or just fallback to the overall max_score.
        score_map = {r.document.page_content: r.score for r in self.search_results}
        
        citations = []
        for doc in self.documents:
            meta = doc.metadata
            score = score_map.get(doc.page_content, self.max_score)
            
            citations.append({
                "metadata": {
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", "?"),
                    "chunk": meta.get("chunk_index", "?"),
                    "domain": meta.get("domain", "general"),
                    "title": meta.get("title", meta.get("source", "unknown")),
                    "year": meta.get("year", ""),
                },
                "score": score
            })
        return citations


class MedicalRetriever:
    """
    Orchestrates retrieval strategy for medical Q&A.
    
    Strategy:
    1. MMR search (diversity-aware) as primary
    2. Confidence score filtering (below threshold → low confidence flag)
    3. Optional metadata pre-filtering
    """

    def __init__(self, vector_store: MedicalVectorStore):
        self._store = vector_store

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        domain: Optional[str] = None,
        source: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        use_mmr: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a medical query.
        
        Args:
            query: The medical question
            k: Number of results to return
            domain: Filter by medical domain (e.g., 'cardiology')
            source: Filter by source document name
            year_from: Filter to documents from this year onwards
            year_to: Filter to documents up to this year
            use_mmr: Use MMR (diverse) vs pure similarity search
        """
        k = k or settings.RETRIEVAL_K
        filter_dict = self._build_filter(domain=domain, source=source,
                                          year_from=year_from, year_to=year_to)

        # ── MMR Search (primary) ──────────────────────────────────────────
        if use_mmr:
            try:
                docs = self._store.mmr_search(
                    query=query,
                    k=k,
                    fetch_k=settings.RETRIEVAL_FETCH_K,
                    lambda_mult=0.6,  # balance relevance/diversity
                    filter_dict=filter_dict,
                )
                # Also run similarity search to get scores for confidence check
                sim_results = self._store.similarity_search(
                    query=query,
                    k=k,
                    filter_dict=filter_dict,
                )
                max_score = max((r.score for r in sim_results), default=0.0)
                is_confident = max_score >= settings.SIMILARITY_THRESHOLD

                logger.info(
                    f"MMR retrieval: {len(docs)} docs, "
                    f"max_score={max_score:.3f}, "
                    f"confident={is_confident}"
                )

                return RetrievalResult(
                    documents=docs,
                    search_results=sim_results,
                    is_confident=is_confident,
                    max_score=max_score,
                    strategy="mmr",
                )
            except Exception as exc:
                logger.warning(f"MMR search failed, falling back to similarity: {exc}")

        # ── Similarity Search (fallback) ──────────────────────────────────
        sim_results = self._store.similarity_search(
            query=query,
            k=k,
            filter_dict=filter_dict,
        )
        max_score = max((r.score for r in sim_results), default=0.0)
        is_confident = max_score >= settings.SIMILARITY_THRESHOLD
        docs = [r.document for r in sim_results]

        logger.info(
            f"Similarity retrieval: {len(docs)} docs, "
            f"max_score={max_score:.3f}, confident={is_confident}"
        )

        return RetrievalResult(
            documents=docs,
            search_results=sim_results,
            is_confident=is_confident,
            max_score=max_score,
            strategy="similarity",
        )

    def _build_filter(
        self,
        domain: Optional[str] = None,
        source: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a ChromaDB metadata filter dict."""
        conditions = []
        if domain:
            conditions.append({"domain": {"$eq": domain}})
        if source:
            conditions.append({"source": {"$eq": source}})
        if year_from:
            conditions.append({"year": {"$gte": year_from}})
        if year_to:
            conditions.append({"year": {"$lte": year_to}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
