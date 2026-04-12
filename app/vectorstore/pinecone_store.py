"""
Pinecone Vector Store Implementation
Remote cloud storage for stateless environments (Vercel).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_pinecone import PineconeVectorStore as LangchainPinecone
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pinecone import Pinecone

from app.config import settings
from app.vectorstore.chroma_store import SearchResult, _make_doc_id

logger = logging.getLogger(__name__)


class MedicalPineconeStore:
    """
    Pinecone-backed vector store for medical documents.
    Used for production deployment on Vercel.
    """

    def __init__(self, embedder: Embeddings):
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required for Pinecone mode")
        
        self._embedder = embedder
        self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # Check if index exists or create it? 
        # For medical RAG, we usually assume the index is created with proper dimensions.
        # mistral-embed uses 1024 dims.
        self._store = LangchainPinecone(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=embedder,
            pinecone_api_key=settings.PINECONE_API_KEY
        )
        logger.info(f"Pinecone Store initialised: index='{settings.PINECONE_INDEX_NAME}'")

    def add_documents(self, documents: List[Document], batch_size: int = 50) -> int:
        """Upsert documents to Pinecone."""
        if not documents:
            return 0
        
        ids = [_make_doc_id(doc) for doc in documents]
        
        # Pinecone's standard LangChain wrapper handles batching and upserts
        self._store.add_documents(documents=documents, ids=ids)
        
        logger.info(f"Upserted {len(documents)} chunks to Pinecone index '{settings.PINECONE_INDEX_NAME}'")
        return len(documents)

    def delete_by_source(self, source_name: str) -> int:
        """Remove all documents with the given source metadata."""
        # Pinecone allows deletion by metadata filter
        try:
            index = self._pc.Index(settings.PINECONE_INDEX_NAME)
            # We fetch IDs first because free-tier Pinecone doesn't always support delete-by-filter
            # or it's safer to use the standard LangChain interface if possible.
            # However, most modern Pinecone indexes support it.
            index.delete(filter={"source": {"$eq": source_name}})
            logger.info(f"Requested deletion for source: {source_name}")
            return -1  # Pinecone doesn't return count easily on delete-by-filter
        except Exception as exc:
            logger.error(f"Error deleting from Pinecone: {exc}")
            return 0

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Similarity search with relevance scores."""
        k = k or settings.RETRIEVAL_K
        results_with_scores = self._store.similarity_search_with_relevance_scores(
            query, k=k, filter=filter_dict
        )
        return [
            SearchResult(doc, score)
            for doc, score in results_with_scores
            if doc.page_content and doc.page_content.strip()
        ]

    def mmr_search(
        self,
        query: str,
        k: int | None = None,
        fetch_k: int | None = None,
        lambda_mult: float = 0.5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Max Marginal Relevance search."""
        k = k or settings.RETRIEVAL_K
        fetch_k = fetch_k or settings.RETRIEVAL_FETCH_K
        return self._store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter_dict
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            index = self._pc.Index(settings.PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            return {
                "total_chunks": stats.total_vector_count,
                "index_name": settings.PINECONE_INDEX_NAME,
                "namespaces": list(stats.namespaces.keys()),
            }
        except Exception as exc:
            logger.error(f"Error getting Pinecone stats: {exc}")
            return {"error": str(exc)}

    def list_sources(self) -> List[str]:
        """
        List all unique sources. 
        Note: Pinecone doesn't support 'list unique metadata values' natively without 
        scanning or using a separate DB. We'll attempt a metadata-only query if possible.
        """
        # This is a limitation of some Vector DBs. 
        # For now, we return a message or a cached list.
        # In production, sources list should probably be in a SQL DB.
        return ["Source listing not natively supported in stateless Pinecone mode"]
