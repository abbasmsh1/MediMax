"""
Ingest Pipeline
End-to-end: Load → Chunk → Embed → Store

Fixes:
- ingest_bytes: tmp_path declared before try-block so finally never raises NameError
- ingest_bytes: returns a proper IngestResult on exception (instead of propagating)
- DOCX content-type added to extension map
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.embeddings.embedder import get_embedder
from app.ingestion.chunker import MedicalChunker
from app.ingestion.loader import load_document
from app.vectorstore import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Summary of an ingest operation."""
    source: str
    pages_loaded: int
    chunks_created: int
    chunks_indexed: int
    duration_seconds: float
    success: bool
    error: Optional[str] = None


class IngestPipeline:
    """
    Orchestrates the full document ingestion pipeline.
    Supports single file, batch directory, or in-memory bytes.
    """

    def __init__(self):
        logger.info("Initializing IngestPipeline...")
        self._embedder = get_embedder()
        self._chunker = MedicalChunker()
        self._store = get_vector_store(self._embedder)

    @property
    def vector_store(self) -> Any:
        return self._store

    def ingest_file(self, path: str | Path) -> IngestResult:
        """Ingest a single document file."""
        path = Path(path)
        start = time.time()
        try:
            docs = load_document(path)
            chunks = self._chunker.chunk_documents(docs)
            indexed = self._store.add_documents(chunks)
            return IngestResult(
                source=path.name,
                pages_loaded=len(docs),
                chunks_created=len(chunks),
                chunks_indexed=indexed,
                duration_seconds=round(time.time() - start, 2),
                success=True,
            )
        except Exception as exc:
            logger.error(f"Ingest failed for {path}: {exc}")
            return IngestResult(
                source=path.name,
                pages_loaded=0,
                chunks_created=0,
                chunks_indexed=0,
                duration_seconds=round(time.time() - start, 2),
                success=False,
                error=str(exc),
            )

    def ingest_directory(self, directory: str | Path) -> List[IngestResult]:
        """Batch-ingest all supported files in a directory."""
        directory = Path(directory)
        results: List[IngestResult] = []
        from app.ingestion.loader import SUPPORTED_EXTENSIONS
        files = [
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        total_files = len(files)
        if total_files == 0:
            logger.info(f"No documents found in '{directory}' to auto-ingest.")
            return []

        logger.info(f"Batch ingesting {total_files} files from {directory}")

        for i, file in enumerate(files):
            logger.info(f"Ingesting file {i + 1}/{total_files}: {file.name}")
            result = self.ingest_file(file)
            results.append(result)
            status = "OK" if result.success else "FAIL"
            logger.info(
                f"  [{i + 1}/{total_files}] {status} {result.source}: "
                f"{result.chunks_indexed} chunks in {result.duration_seconds}s"
            )
        return results

    def ingest_new_docs(self, directory: str | Path) -> List[IngestResult]:
        """
        Incremental ingest: only process files NOT already in ChromaDB.

        For each supported file in *directory*, checks whether its filename
        already has indexed chunks (fast metadata-only query).  Files that
        are already indexed are silently skipped so this is safe to call on
        every server start without any performance penalty.

        Returns IngestResult objects only for files that were actually
        ingested (or attempted) this run.
        """
        directory = Path(directory)
        results: List[IngestResult] = []

        from app.ingestion.loader import SUPPORTED_EXTENSIONS
        files = [
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not files:
            logger.info("ingest_new_docs: no supported files found in '%s'.", directory)
            return []

        # Separate already-indexed from new
        new_files = []
        for f in files:
            if self._store.is_source_indexed(f.name):
                logger.info("  [skip] '%s' already indexed.", f.name)
            else:
                new_files.append(f)

        total_new = len(new_files)
        skipped   = len(files) - total_new

        if total_new == 0:
            logger.info(
                "ingest_new_docs: all %d file(s) already indexed — nothing to do.",
                len(files),
            )
            return []

        logger.info(
            "ingest_new_docs: %d new file(s) to ingest, %d already indexed.",
            total_new, skipped,
        )

        for i, file in enumerate(new_files, start=1):
            logger.info("  [%d/%d] ingesting '%s'…", i, total_new, file.name)
            result = self.ingest_file(file)
            results.append(result)
            tag = "OK  " if result.success else "FAIL"
            logger.info(
                "  [%d/%d] %s '%s': %d chunks in %.1fs",
                i, total_new, tag, result.source,
                result.chunks_indexed, result.duration_seconds,
            )

        failed = sum(1 for r in results if not r.success)
        logger.info(
            "ingest_new_docs done: %d ingested, %d failed.",
            total_new - failed, failed,
        )
        return results

    def ingest_bytes(
        self,
        content: bytes,
        filename: str,
        content_type: str = "application/pdf",
    ) -> IngestResult:
        """Ingest document from in-memory bytes (used by API file upload)."""
        import os
        import tempfile

        # Map MIME types → file extensions
        ext_map: dict[str, str] = {
            "application/pdf":  ".pdf",
            "text/plain":       ".txt",
            "text/markdown":    ".md",
            # Word DOCX
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document": ".docx",
        }

        ext = Path(filename).suffix or ext_map.get(content_type, ".pdf")
        start = time.time()

        # Declare before try so the finally block never raises NameError
        tmp_path: Optional[Path] = None

        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            # Load and chunk
            docs   = load_document(tmp_path)
            chunks = self._chunker.chunk_documents(docs)

            # ── Correct source metadata BEFORE indexing ──────────────────
            # Without this, chroma stores 'tmpXXXX.pdf' instead of the real
            # filename, which breaks is_source_indexed() and list_sources().
            for chunk in chunks:
                chunk.metadata["source"] = filename

            start2 = time.time()
            indexed = self._store.add_documents(chunks)

            return IngestResult(
                source=filename,
                pages_loaded=len(docs),
                chunks_created=len(chunks),
                chunks_indexed=indexed,
                duration_seconds=round(time.time() - start, 2),
                success=True,
            )

        except Exception as exc:
            logger.error(f"ingest_bytes failed for '{filename}': {exc}")
            return IngestResult(
                source=filename,
                pages_loaded=0,
                chunks_created=0,
                chunks_indexed=0,
                duration_seconds=round(time.time() - start, 2),
                success=False,
                error=str(exc),
            )

        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def get_stats(self) -> Dict[str, Any]:
        return self._store.get_stats()

    def list_sources(self) -> List[str]:
        return self._store.list_sources()

    def delete_source(self, source_name: str) -> int:
        return self._store.delete_by_source(source_name)
