"""
Embedding Module
Supports Mistral embeddings (primary) with sentence-transformers fallback.
Uses batch processing for efficiency.

NOTE: PyTorch 2.x with lazy/meta tensor loading breaks the langchain_community
HuggingFaceEmbeddings wrapper.  We bypass it by using sentence-transformers
directly and wrapping it in a minimal LangChain-compatible Embeddings class.
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.embeddings import Embeddings

from app.config import settings

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ── Local embeddings using sentence-transformers directly ─────────────────────

class _LocalEmbeddings(Embeddings):
    """
    Thin LangChain-compatible wrapper around sentence_transformers.SentenceTransformer.

    Bypasses langchain_community.HuggingFaceEmbeddings which calls `.to(device)`
    in a way incompatible with PyTorch ≥2.x meta-tensor lazy loading.

    Windows fix: PyTorch's OpenMP/BLAS routines trigger an Access Violation
    (0xC0000005) when run in a daemon thread on Windows.  Setting the threading
    environment variables to 1 forces single-threaded execution which is safe.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        import os
        # Must be set BEFORE importing torch / sentence_transformers
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        # Disable torch parallelism for thread-safe inference
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        import sentence_transformers  # local import keeps startup fast
        logger.info("Loading local embedding model: %s", model_name)
        self._model = sentence_transformers.SentenceTransformer(
            model_name,
            device="cpu",
        )
        # ── CRITICAL WINDOWS FIX ──────────────────────────────────────
        # PyTorch OpenMP crashes with 0xC0000005 (Access Violation) if the
        # *first* forward pass happens in a spawned child thread. Since
        # queries run in asyncio.to_thread, we perform a dummy pass here
        # in the main thread to safely initialize OpenMP thread pools.
        try:
            self._model.encode(["warmup"])
        except Exception:
            pass
        # ──────────────────────────────────────────────────────────────
        logger.info("Local embedding model loaded.")


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()


# ── Public factory ─────────────────────────────────────────────────────────────

def get_embedder() -> Embeddings:
    """
    Returns the appropriate embeddings model based on configuration.
    - Primary: Mistral embeddings (if MISTRAL_API_KEY is set)
    - Fallback: sentence-transformers/all-MiniLM-L6-v2 (local, free)
    """
    if settings.use_mistral_embeddings:
        logger.info("Using Mistral embeddings: %s", settings.MISTRAL_EMBED_MODEL)
        return _get_mistral_embeddings()
    else:
        logger.info("Using local sentence-transformers embeddings (fallback)")
        return _LocalEmbeddings()


def _get_mistral_embeddings() -> Embeddings:
    from langchain_mistralai import MistralAIEmbeddings
    return MistralAIEmbeddings(
        model=settings.MISTRAL_EMBED_MODEL,
        mistral_api_key=settings.MISTRAL_API_KEY,
    )
