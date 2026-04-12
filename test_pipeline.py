"""
End-to-end test: upload a document and run a query.
Run from project root:  .venv\Scripts\python.exe test_pipeline.py
"""

import sys
import json
import time
import requests

BASE = "http://127.0.0.1:8080"
DOC  = r"docs\Gynae Ten Teachers 20e.pdf"   # smallest doc ~6 MB


def hr(char="─", width=60):
    print(char * width)


def check_health():
    hr()
    print("1. Health check")
    r = requests.get(f"{BASE}/api/health", timeout=10)
    r.raise_for_status()
    data = r.json()
    print(f"   status  : {data.get('status')}")
    print(f"   version : {data.get('version')}")
    print(f"   embed   : {data.get('embedding_model')}")
    return True


def check_stats(label="before ingest"):
    hr()
    print(f"2. Stats ({label})")
    r = requests.get(f"{BASE}/api/ingest/stats", timeout=10)
    r.raise_for_status()
    data = r.json()
    print(f"   total_documents : {data.get('total_documents', 'n/a')}")
    print(f"   total_chunks    : {data.get('total_chunks', 'n/a')}")
    return data


def upload_document(path):
    hr()
    print(f"3. Uploading: {path}")
    with open(path, "rb") as f:
        files = {"file": (path.split("\\")[-1], f, "application/pdf")}
        t0 = time.time()
        r = requests.post(f"{BASE}/api/ingest/upload", files=files, timeout=300)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    print(f"   HTTP {r.status_code}  ({elapsed:.1f}s)")
    print(f"   success        : {data.get('success')}")
    print(f"   chunks_indexed : {data.get('chunks_indexed')}")
    print(f"   title          : {data.get('title', 'n/a')}")
    print(f"   message        : {data.get('message', '')}")
    if data.get("errors"):
        print(f"   ERRORS: {data['errors']}")
    return data


def run_query(question):
    hr()
    print(f"4. Query: {question!r}")
    payload = {"query": question, "top_k": 5}
    t0 = time.time()
    r = requests.post(f"{BASE}/api/query", json=payload, timeout=120)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    print(f"   HTTP {r.status_code}  ({elapsed:.1f}s)")
    print(f"   answer:\n")
    answer = data.get("answer", data.get("response", "no answer field"))
    # Print wrapped
    for line in str(answer).split("\n"):
        print(f"     {line}")
    sources = data.get("sources", data.get("source_documents", []))
    if sources:
        print(f"\n   sources ({len(sources)}):")
        for i, s in enumerate(sources[:3], 1):
            if isinstance(s, dict):
                meta = s.get("metadata", {})
                print(f"     [{i}] {meta.get('source','?')} | page {meta.get('page','?')}")
            else:
                print(f"     [{i}] {s}")
    return data


def main():
    print("\n" + "═" * 60)
    print("  MediMax RAG — End-to-End Pipeline Test")
    print("═" * 60)

    check_health()
    check_stats("before ingest")

    result = upload_document(DOC)
    if not result.get("success"):
        print("\n❌ Upload failed — aborting query test.")
        sys.exit(1)

    check_stats("after ingest")

    run_query("What are the common causes of abnormal uterine bleeding?")

    hr("═")
    print("✅  Test complete.")
    hr("═")


if __name__ == "__main__":
    main()
