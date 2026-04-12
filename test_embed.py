"""
Test whether the OMP_NUM_THREADS fix prevents the Windows Access Violation
during sentence-transformer embedding in a daemon thread.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import threading
import sys

result = {"ok": False, "error": None}

def run_embed():
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        import sentence_transformers
        print("Loading model...", flush=True)
        model = sentence_transformers.SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        print("Model loaded. Running encode...", flush=True)
        
        texts = [
            "The patient presents with fever and cough.",
            "Myocardial infarction is characterized by chest pain.",
            "Antibiotic resistance is a growing public health concern.",
        ]
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
        print(f"SUCCESS: got {len(embeddings)} embeddings of dim {len(embeddings[0])}", flush=True)
        result["ok"] = True
    except Exception as e:
        result["error"] = str(e)
        print(f"ERROR: {e}", flush=True)

t = threading.Thread(target=run_embed, daemon=True)
t.start()
t.join(timeout=60)

if t.is_alive():
    print("TIMEOUT: thread still running after 60s")
    sys.exit(2)

if result["ok"]:
    print("Test PASSED: embedding works in daemon thread")
    sys.exit(0)
else:
    print(f"Test FAILED: {result['error']}")
    sys.exit(1)
