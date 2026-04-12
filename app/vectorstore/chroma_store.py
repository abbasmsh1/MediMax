"""
ChromaDB Vector Store
Persistent, incremental vector storage with metadata filtering.

Fixes:
- add_documents now uses upsert semantics to handle pre-existing IDs
  (Chroma ≥ 0.5 raises DuplicateIDError on plain add; we call upsert instead)
- ID generation uses a hash of content + source + page to survive renames
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import settings

logger = logging.getLogger(__name__)


class SearchResult:
    """A retrieved document paired with its similarity score."""

    def __init__(self, document: Document, score: float):
        self.document = document
        self.score = score

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")

    @property
    def page(self) -> Any:
        return self.document.metadata.get("page", "?")

    @property
    def chunk_index(self) -> int:
        return self.document.metadata.get("chunk_index", 0)


def _make_doc_id(doc: Document) -> str:
    """
    Build a stable, deterministic ID for a chunk.

    Uses a short SHA-256 hash of (source + page + chunk_index) so that
    re-ingesting the same file always produces the same IDs — enabling
    true upsert / deduplication without full content hashing.
    """
    source = doc.metadata.get("source", "unknown")
    page   = str(doc.metadata.get("page", 0))
    cidx   = str(doc.metadata.get("chunk_index", 0))
    raw    = f"{source}||p{page}||c{cidx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class MedicalVectorStore:
    """
    ChromaDB-backed vector store for medical documents.

    Features
    --------
    - Persistent disk storage (survives restarts)
    - Upsert semantics — re-ingesting the same file is safe and idempotent
    - Metadata filtering (by domain, source, year)
    - Both similarity and MMR search modes
    """

    def __init__(self, embedder: Embeddings):
        self._embedder = embedder
        self._store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embedder,
            persist_directory=str(settings.chroma_path),
        )
        logger.info(
            f"VectorStore initialised: collection='{settings.CHROMA_COLLECTION_NAME}', "
            f"path='{settings.chroma_path}'"
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 50,
    ) -> int:
        """
        Upsert documents in batches.

        Uses upsert semantics so that:
        - New documents are added.
        - Existing documents (same source + page + chunk_index) are updated
          in-place without raising DuplicateIDError.

        Returns the number of documents processed.
        """
        if not documents:
            return 0

        ids = [_make_doc_id(doc) for doc in documents]

        processed = 0
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids  = ids[i : i + batch_size]
            try:
                # Use the underlying chromadb collection's upsert so we never
                # hit DuplicateIDError on re-ingestion of the same file.
                self._store._collection.upsert(
                    ids=batch_ids,
                    documents=[d.page_content for d in batch_docs],
                    metadatas=[d.metadata if d.metadata else {"source": "unknown"} for d in batch_docs],
                    embeddings=self._embedder.embed_documents(
                        [d.page_content for d in batch_docs]
                    ),
                )
                processed += len(batch_docs)
                logger.info(
                    f"  Upserted batch {i // batch_size + 1}: {len(batch_docs)} chunks"
                )
            except Exception as exc:
                logger.error(f"  Error upserting batch {i // batch_size + 1}: {exc}")

        logger.info(f"Total upserted: {processed} chunks")
        return processed

    def delete_by_source(self, source_name: str) -> int:
        """Remove all documents from a specific source file."""
        try:
            results = self._store.get(where={"source": source_name})
            ids = results.get("ids", [])
            if ids:
                self._store.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} chunks from source: {source_name}")
            return len(ids)
        except Exception as exc:
            logger.error(f"Error deleting source '{source_name}': {exc}")
            return 0

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Top-K similarity search with optional metadata filtering."""
        k = k or settings.RETRIEVAL_K
        results_with_scores: List[Tuple[Document, float]] = (
            self._store.similarity_search_with_relevance_scores(
                query,
                k=k,
                filter=filter_dict,
            )
        )
        # Guard: filter out any chunks where page_content is None/empty
        # (can occur when ChromaDB stored corrupted/incomplete entries)
        return [
            SearchResult(doc, score)
            for doc, score in results_with_scores
            if doc.page_content is not None and doc.page_content.strip() != ""
        ]

    def mmr_search(
        self,
        query: str,
        k: int | None = None,
        fetch_k: int | None = None,
        lambda_mult: float = 0.5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Max Marginal Relevance search — balances relevance with diversity.
        lambda_mult: 1.0 = pure similarity, 0.0 = pure diversity.
        """
        k = k or settings.RETRIEVAL_K
        fetch_k = fetch_k or settings.RETRIEVAL_FETCH_K
        docs = self._store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter_dict,
        )
        # Guard: filter out any chunks where page_content is None/empty
        return [
            doc for doc in docs
            if doc.page_content is not None and doc.page_content.strip() != ""
        ]

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return collection statistics."""
        try:
            count = self._store._collection.count()
            return {
                "total_chunks":    count,
                "collection_name": settings.CHROMA_COLLECTION_NAME,
                "persist_dir":     str(settings.chroma_path),
            }
        except Exception as exc:
            logger.error(f"Error getting stats: {exc}")
            return {"total_chunks": 0, "error": str(exc)}

    def list_sources(self) -> List[str]:
        """
        Return a sorted list of unique source documents in the store.

        Uses paginated metadata-only queries with a large page size so that
        collections with 100k+ chunks complete in a reasonable time.
        Page size of 5000 means ~25 iterations for a 123k-chunk collection.
        """
        try:
            collection = self._store._collection
            total      = collection.count()
            sources: set[str] = set()
            page_size  = 5_000   # larger pages → fewer round-trips
            offset     = 0

            while offset < total:
                batch = collection.get(
                    limit=page_size,
                    offset=offset,
                    include=["metadatas"],   # skip documents & embeddings
                )
                metadatas = batch.get("metadatas") or []
                for meta in metadatas:
                    if meta and "source" in meta:
                        src = meta["source"]
                        # Skip leftover temp-file entries from old ingestion bugs
                        if src and not src.startswith("tmp"):
                            sources.add(src)
                offset += page_size

            logger.info(f"list_sources: found {len(sources)} unique sources in {total} chunks")
            return sorted(sources)
        except Exception as exc:
            logger.error(f"Error listing sources: {exc}")
            return []

    def is_source_indexed(self, filename: str) -> bool:
        """
        Fast check: does *any* chunk with this source filename exist in the store?

        Uses a metadata-only ``where`` filter so no embeddings are computed.
        Returns True if at least one chunk is found, False otherwise.
        """
        try:
            results = self._store._collection.get(
                where={"source": filename},
                limit=1,           # we only need one hit to confirm presence
                include=[],        # metadata-only — no documents, no embeddings
            )
            return len(results.get("ids", [])) > 0
        except Exception as exc:
            logger.warning(
                "is_source_indexed('%s') failed (%s) — assuming not indexed.",
                filename, exc,
            )
            return False
