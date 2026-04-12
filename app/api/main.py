"""
FastAPI Application Entry Point
MediMax Medical RAG System API
"""
from __future__ import annotations

# ── Windows PyTorch crash fix ─────────────────────────────────────────────────
# PyTorch's OpenMP/BLAS multi-threading causes an Access Violation (0xC0000005)
# on Windows when called from a daemon thread inside Uvicorn. Setting these
# env vars to "1" BEFORE any torch/numpy import forces single-threaded mode.
# This must happen here at the top of the entry-point module.
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "None")  # Prevent Chroma telemetry thread crash
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# ─────────────────────────────────────────────────────────────────────────────

import logging
import sys
import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import ingest, query
from app.api.schemas import HealthResponse
from app.config import settings

# ── Logging Setup ─────────────────────────────────────────────────────────────
# Set utf-8 encoding for stdout to avoid UnicodeEncodeError on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Smart startup auto-ingest ────────────────────────────────────────────
_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def _count_indexed_sources_sqlite() -> set[str]:
    """
    Return the set of source filenames already in ChromaDB.
    Uses a direct read-only SQLite query — no pipeline, no embedder.
    """
    import sqlite3 as _sqlite3
    db_path = settings.chroma_path / "chroma.sqlite3"
    sources: set[str] = set()
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = _sqlite3.connect(uri, uri=True, timeout=5)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT string_value FROM embedding_metadata WHERE key='source'"
            )
            sources = {row[0] for row in cur.fetchall()}
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Pre-check SQLite query failed (non-fatal): %s", exc)
    return sources


def _auto_ingest_docs():
    """
    Background task: incrementally ingest documents from the docs/ folder.

    Strategy
    --------
    1.  Fast pre-check via SQLite: build the set of already-indexed filenames.
    2.  Scan docs/ for supported files whose basename is NOT in that set.
    3.  If there are zero new files, exit immediately — pipeline and embedding
        model are never loaded, avoiding the PyTorch segfault on Windows.
    4.  Only if new files exist: initialise the full pipeline (embedder +
        ChromaDB) and ingest only the new files.

    This means:
    - First run:  all docs/ files are ingested (slow, expected).
    - Subsequent restarts / hot-reloads: zero work done (fast, no segfault).
    - Adding a new file to docs/ triggers ingestion of that file only.
    """
    try:
        import time as _time
        _time.sleep(2)  # let uvicorn fully bind before any heavy I/O

        root = Path(__file__).parent.parent.parent
        docs_paths = [
            root / "docs",
            root / "data" / "new_medical_literature",
        ]

        # --- Fast pre-check: which source files are already indexed? --------
        already_indexed = _count_indexed_sources_sqlite()
        logger.info(
            "Auto-ingest pre-check: %d sources already in ChromaDB.",
            len(already_indexed),
        )

        new_files: list[Path] = []
        for docs_path in docs_paths:
            if not docs_path.exists():
                continue
            for f in docs_path.iterdir():
                if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS:
                    if f.name not in already_indexed:
                        new_files.append(f)

        if not new_files:
            logger.info(
                "Auto-ingest: all docs already indexed — skipping pipeline init."
            )
            return

        logger.info(
            "Auto-ingest: %d new file(s) detected. Initialising pipeline...",
            len(new_files),
        )

        # --- Only now load the heavy pipeline (embedder + ChromaDB) ---------
        from app.api.routes.ingest import get_ingest_pipeline
        pipeline = get_ingest_pipeline()
        logger.info("Pipeline ready. Ingesting %d new file(s)...", len(new_files))

        for docs_path in docs_paths:
            if not docs_path.exists():
                continue
            results = pipeline.ingest_new_docs(docs_path)
            if results:
                ok   = sum(1 for r in results if r.success)
                fail = len(results) - ok
                logger.info(
                    "Auto-ingest from '%s': %d ingested, %d failed.",
                    docs_path.name, ok, fail,
                )

        logger.info("Auto-ingest scan complete. System ready.")
    except Exception as exc:
        logger.error("Auto-ingest failed (non-fatal): %s", exc, exc_info=True)

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  MediMax Medical RAG System — Starting Up")
    logger.info("=" * 60)
    logger.info(f"  LLM Model:      {settings.MISTRAL_LLM_MODEL}")
    logger.info(f"  Embedding Mode: {'mistral-embed' if settings.use_mistral_embeddings else 'local-minilm'}")
    logger.info(f"  ChromaDB:       {settings.chroma_path}")
    logger.info(f"  Temperature:    {settings.LLM_TEMPERATURE}")
    logger.info(f"  Chunk Size:     {settings.CHUNK_SIZE} (overlap: {settings.CHUNK_OVERLAP})")
    logger.info(f"  Retrieval K:    {settings.RETRIEVAL_K}")
    logger.info(f"  Conf Threshold: {settings.SIMILARITY_THRESHOLD}")
    logger.info("=" * 60)
    
    # ── Launch incremental auto-ingest ────────────────────────────────────────────
    # In serverless environments (like Vercel), background threads are limited
    # or killed soon after the response. We disable auto-ingest if VERCEL is set.
    if not _os.environ.get("VERCEL"):
        t = threading.Thread(target=_auto_ingest_docs, name="auto-ingest", daemon=True)
        t.start()
        logger.info("Auto-ingest background scanner started.")
    else:
        logger.info("Vercel detected: skipping auto-ingest background scanner.")
    
    yield
    logger.info("MediMax shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MediMax",
    description="Medical RAG System — Grounded answers from medical literature",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model=settings.MISTRAL_LLM_MODEL,
        embedding_mode="mistral-embed" if settings.use_mistral_embeddings else "local-minilm",
    )


# ── Serve Frontend ────────────────────────────────────────────────────────────
frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))
