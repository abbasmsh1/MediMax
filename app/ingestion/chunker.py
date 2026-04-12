"""
Medical Text Chunker
Uses RecursiveCharacterTextSplitter with medical-aware separators.

Key fixes:
- Removed '. ' separator (breaks Dr., Fig., No., et al., e.g., vs., U.S.A. …)
- Added PDF extraction artifact cleanup (de-hyphenation, page numbers, whitespace)
- chunk_total assigned AFTER empty-chunk filtering so indices are always contiguous
- Minimum chunk length raised to 60 chars to avoid single-word fragments
"""
from __future__ import annotations

import logging
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.ingestion.metadata import enrich_metadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Separator hierarchy
# NOTE: '. ' is intentionally absent — it breaks medical abbreviations such as
#   "Dr. Smith", "Fig. 3", "No. 5", "vs.", "et al.", "e.g.", "i.e.", "U.S.A."
# We rely on paragraph / line breaks to preserve sentence integrity.
# ---------------------------------------------------------------------------
MEDICAL_SEPARATORS = [
    "\n\n\n",  # Section / chapter breaks (highest priority)
    "\n\n",    # Paragraph boundaries
    "\n",      # Single line breaks
    "; ",      # Clause separators (purposely before comma)
    ", ",      # Sub-clause separators
    " ",       # Word boundaries (last resort before character split)
    "",        # Character-level absolute fallback
]

# ---------------------------------------------------------------------------
# PDF extraction artifact patterns
# ---------------------------------------------------------------------------
_RE_HYPHEN_BREAK = re.compile(r"-\n([a-zA-Z])")     # "cardio-\nvascular" → "cardiovascular"
_RE_MULTISPACE   = re.compile(r"[ \t]{2,}")          # collapse runs of spaces/tabs
_RE_PAGE_NUM     = re.compile(r"\n\s{0,4}\d{1,4}\s{0,4}\n")  # isolated bare page numbers
_RE_EXCESS_NL    = re.compile(r"\n{3,}")             # 3+ newlines → exactly 2


def _clean_text(text: str) -> str:
    """
    Normalize text extracted from PDFs / medical documents.

    Fixes common extraction artifacts that degrade chunking quality:
    - Words broken by hyphenation across lines
    - Isolated page-number lines
    - Runs of whitespace / excess blank lines
    """
    text = _RE_HYPHEN_BREAK.sub(r"\1", text)   # re-join hyphenated words
    text = _RE_MULTISPACE.sub(" ", text)        # single space
    text = _RE_PAGE_NUM.sub("\n", text)         # strip lone page numbers
    text = _RE_EXCESS_NL.sub("\n\n", text)      # normalise blank lines
    return text.strip()


class MedicalChunker:
    """
    Text chunker optimised for medical literature.

    Design decisions
    ----------------
    * Does NOT split on '. ' to preserve medical abbreviations.
    * Cleans PDF extraction artifacts before splitting.
    * Assigns chunk_index / chunk_total AFTER filtering so indices are
      always contiguous (0 … N-1) and chunk_total is accurate.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=MEDICAL_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
            add_start_index=True,
        )

        logger.info(
            "MedicalChunker initialised: "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Full chunking pipeline:

        1. Clean PDF / text extraction artifacts
        2. Enrich metadata (domain, year, title)
        3. Split into raw chunks
        4. Filter near-empty / whitespace-only chunks
        5. Assign accurate chunk_index / chunk_total AFTER filtering
        """
        if not documents:
            return []

        # 1. Clean artifacts
        cleaned = [self._clean_doc(doc) for doc in documents]

        # 2. Enrich metadata before splitting (operates on whole-page text)
        enriched = [enrich_metadata(doc) for doc in cleaned]

        # 3. Split
        raw_chunks = self._splitter.split_documents(enriched)

        # 4. Strip whitespace + filter near-empty chunks (>= 60 chars)
        chunks: List[Document] = []
        for chunk in raw_chunks:
            chunk.page_content = chunk.page_content.strip()
            if len(chunk.page_content) >= 60:
                chunks.append(chunk)

        # 5. Assign chunk_index / chunk_total AFTER filtering
        #    (Bug fix: previously done before filter, so totals were wrong)
        source_groups: dict[str, list[Document]] = {}
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            source_groups.setdefault(src, []).append(chunk)

        for src_chunks in source_groups.values():
            total = len(src_chunks)
            for idx, chunk in enumerate(src_chunks):
                chunk.metadata["chunk_index"] = idx
                chunk.metadata["chunk_total"] = total

        avg_len = (
            sum(len(c.page_content) for c in chunks) // len(chunks)
            if chunks else 0
        )
        logger.info(
            f"Chunked {len(documents)} doc(s) -> {len(chunks)} chunks "
            f"(avg {avg_len} chars)"
        )
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean_doc(self, doc: Document) -> Document:
        """Apply artifact cleaning to a single Document in-place."""
        doc.page_content = _clean_text(doc.page_content)
        return doc
