"""
Debug script - why are documents not indexed?
Run: python debug_index.py
"""
import sys
import traceback
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                    format="%(levelname)-8s | %(name)s | %(message)s")

root = Path(__file__).parent
print(f"\n=== PROJECT ROOT: {root} ===\n")

# 1. Config
try:
    from app.config import settings
    print(f"[OK] Config loaded")
    print(f"     CHROMA PATH:       {settings.chroma_path.absolute()}")
    print(f"     COLLECTION:        {settings.CHROMA_COLLECTION_NAME}")
    print(f"     use_mistral_emb:   {settings.use_mistral_embeddings}")
    print(f"     MISTRAL_API_KEY:   {'SET' if settings.MISTRAL_API_KEY else 'NOT SET'}")
except Exception:
    print("[FAIL] Config")
    traceback.print_exc()
    sys.exit(1)

# 2. Embedder
try:
    from app.embeddings.embedder import get_embedder
    embedder = get_embedder()
    test_emb = embedder.embed_query("hello")
    print(f"\n[OK] Embedder works (dim={len(test_emb)})")
except Exception:
    print("\n[FAIL] Embedder")
    traceback.print_exc()
    sys.exit(1)

# 3. VectorStore
try:
    from app.vectorstore.chroma_store import MedicalVectorStore
    store = MedicalVectorStore(embedder)
    print("[OK] VectorStore initialised")
except Exception:
    print("[FAIL] VectorStore init")
    traceback.print_exc()
    sys.exit(1)

# 4. Stats
try:
    stats = store.get_stats()
    print(f"[OK] Stats: {stats}")
except Exception:
    print("[FAIL] get_stats()")
    traceback.print_exc()

# 5. List sources
try:
    sources = store.list_sources()
    print(f"[OK] Sources ({len(sources)}): {sources[:5]}")
except Exception:
    print("[FAIL] list_sources()")
    traceback.print_exc()

# 6. Docs folder
docs_path = root / "docs"
print(f"\n=== DOCS FOLDER: {docs_path} ===")
if docs_path.exists():
    files = list(docs_path.iterdir())
    print(f"     {len(files)} files found:")
    for f in files[:10]:
        print(f"       - {f.name}")
else:
    print("     [WARN] docs/ folder does NOT exist!")

# 7. Test the ingest pipeline
try:
    from app.api.routes.ingest import get_ingest_pipeline
    print("\n[OK] get_ingest_pipeline import OK")
    pipeline = get_ingest_pipeline()
    print("[OK] Pipeline instantiated")
    stats2 = pipeline.get_stats()
    print(f"[OK] Pipeline stats: {stats2}")
except Exception:
    print("\n[FAIL] Ingest pipeline")
    traceback.print_exc()

# 8. Try running ingest_new_docs on docs/
if docs_path.exists():
    try:
        print(f"\n=== Running ingest_new_docs on {docs_path} ===")
        results = pipeline.ingest_new_docs(docs_path)
        print(f"Results: {results}")
    except Exception:
        print("[FAIL] ingest_new_docs")
        traceback.print_exc()

print("\n=== DONE ===")
