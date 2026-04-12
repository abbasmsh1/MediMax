"""
Quiet debug script — only prints [OK]/[FAIL] lines.
Run: python debug_index2.py
"""
import sys, traceback, logging
from pathlib import Path

# Silence noisy third-party loggers
for noisy in ("httpx", "httpcore", "sentence_transformers", "chromadb", "urllib3", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

root = Path(__file__).parent

# 1. Config
try:
    from app.config import settings
    print(f"[OK] Config: chroma={settings.chroma_path.absolute()}, collection={settings.CHROMA_COLLECTION_NAME}")
    print(f"     use_mistral_embeddings={settings.use_mistral_embeddings}, API_KEY={'SET' if settings.MISTRAL_API_KEY else 'NOT SET'}")
except Exception:
    print("[FAIL] Config"); traceback.print_exc(); sys.exit(1)

# 2. Embedder
try:
    from app.embeddings.embedder import get_embedder
    embedder = get_embedder()
    dim = len(embedder.embed_query("test"))
    print(f"[OK] Embedder dim={dim}")
except Exception:
    print("[FAIL] Embedder"); traceback.print_exc(); sys.exit(1)

# 3. VectorStore
try:
    from app.vectorstore.chroma_store import MedicalVectorStore
    store = MedicalVectorStore(embedder)
    print("[OK] VectorStore init")
except Exception:
    print("[FAIL] VectorStore"); traceback.print_exc(); sys.exit(1)

# 4. Stats
try:
    stats = store.get_stats()
    print(f"[OK] Stats: {stats}")
except Exception:
    print("[FAIL] get_stats()"); traceback.print_exc()

# 5. Sources
try:
    sources = store.list_sources()
    print(f"[OK] Sources ({len(sources)}): {sources[:5]}")
except Exception:
    print("[FAIL] list_sources()"); traceback.print_exc()

# 6. Docs folder
docs_path = root / "docs"
print(f"\n=== DOCS FOLDER: {docs_path} ===")
if docs_path.exists():
    files = list(docs_path.iterdir())
    print(f"     {len(files)} files:")
    for f in files[:15]:
        print(f"       {f.name}  ({f.stat().st_size} bytes)")
else:
    print("     [WARN] docs/ does NOT exist")

# 7. Ingest pipeline
try:
    from app.api.routes.ingest import get_ingest_pipeline
    pipeline = get_ingest_pipeline()
    p_stats = pipeline.get_stats()
    print(f"\n[OK] Pipeline stats: {p_stats}")
except Exception:
    print("\n[FAIL] Ingest pipeline"); traceback.print_exc()

# 8. Try dry-run of ingest logic
if docs_path.exists():
    try:
        print(f"\n=== ingest_new_docs({docs_path.name}/) ===")
        results = pipeline.ingest_new_docs(docs_path)
        if results:
            for r in results:
                status = "OK" if r.success else "FAIL"
                print(f"  [{status}] {getattr(r, 'filename', r)} chunks={getattr(r, 'chunks_added', '?')} err={getattr(r, 'error', '')}")
        else:
            print("  (no results — all files already indexed OR no supported files found)")
    except Exception:
        print("[FAIL] ingest_new_docs"); traceback.print_exc()

print("\n=== DONE ===")
