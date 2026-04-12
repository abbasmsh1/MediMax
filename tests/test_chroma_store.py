import pytest
from langchain_core.documents import Document

def test_add_documents_empty_metadata(test_vector_store):
    """
    Test that adding a document with empty metadata succeeds.
    Previously, this failed silently because Chroma requires metadata to be a non-empty dict.
    """
    docs = [Document(page_content="test content")]
    # This previously failed silently if metadata was {}
    added = test_vector_store.add_documents(docs)
    
    assert added == 1
    stats = test_vector_store.get_stats()
    assert stats["total_chunks"] == 1

def test_add_sample_documents(test_vector_store, sample_documents):
    """
    Test adding properly prepared documents with metadata.
    """
    added = test_vector_store.add_documents(sample_documents)
    assert added == len(sample_documents)
    
    # Check that sources are registered
    sources = test_vector_store.list_sources()
    assert len(sources) > 0

def test_chunking_and_adding(test_chunker, test_vector_store, sample_document):
    """
    Test the full chunking and vector store ingestion flow.
    """
    chunks = test_chunker.chunk_documents([sample_document])
    assert len(chunks) > 0
    
    added = test_vector_store.add_documents(chunks)
    assert added == len(chunks)
    
    stats = test_vector_store.get_stats()
    assert stats["total_chunks"] == len(chunks)
