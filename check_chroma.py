import os
import sys
from pathlib import Path

# Add project root to sys.path
root = Path("e:/Projects/MediMax")
sys.path.append(str(root))

# Set environment variables if needed
os.environ["MISTRAL_API_KEY"] = "dummy" # In case it's needed for initialization

from app.config import settings
from app.embeddings.embedder import get_embedder
from app.vectorstore.chroma_store import MedicalVectorStore

def check():
    print(f"Chroma Path: {settings.chroma_path.absolute()}")
    print(f"Collection: {settings.CHROMA_COLLECTION_NAME}")
    
    try:
        embedder = get_embedder()
        store = MedicalVectorStore(embedder)
        stats = store.get_stats()
        print(f"Stats: {stats}")
        
        sources = store.list_sources()
        print(f"Sources found: {len(sources)}")
        for s in sources[:10]:
            print(f"  - {s}")
            
        # Check if collection actually exists in the client
        coll = store._store._client.get_collection(settings.CHROMA_COLLECTION_NAME)
        print(f"Collection count from client directly: {coll.count()}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check()
