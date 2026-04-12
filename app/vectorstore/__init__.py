from app.config import settings
from app.vectorstore.chroma_store import MedicalVectorStore
from app.vectorstore.pinecone_store import MedicalPineconeStore

def get_vector_store(embedder):
    if settings.is_pinecone:
        return MedicalPineconeStore(embedder)
    return MedicalVectorStore(embedder)
