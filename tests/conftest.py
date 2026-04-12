"""
MediMax Test Fixtures

Shared pytest fixtures for the MediMax test suite.
These fixtures provide reusable test infrastructure for:
- Embedding models (mock and local)
- Vector stores (temporary ChromaDB instances)
- Document chunking
- Sample test documents
- Temporary directories and cleanup
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.ingestion.chunker import MedicalChunker


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test artifacts.

    Provides a fresh temporary directory that is automatically cleaned up
    after the test completes. Use this for any test that needs to write
    files to disk.

    Yields:
        Path: Path to the temporary directory

    Example:
        def test_file_writing(temp_dir):
            test_file = temp_dir / "test.txt"
            test_file.write_text("content")
            assert test_file.exists()
    """
    tmp = Path(tempfile.mkdtemp(prefix="medimax_test_"))
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def temp_chroma_path(temp_dir: Path) -> Path:
    """
    Create a temporary directory for ChromaDB persistence.

    This fixture provides an isolated directory for ChromaDB to store
    its vector data. The directory is automatically cleaned up after
    the test completes, ensuring no persistent state between tests.

    Args:
        temp_dir: Base temporary directory fixture

    Yields:
        Path: Path to the temporary ChromaDB directory

    Example:
        def test_vector_store(temp_chroma_path):
            store = MedicalVectorStore(
                embedder=test_embedder(),
                persist_directory=str(temp_chroma_path)
            )
    """
    chroma_dir = temp_dir / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    yield chroma_dir


# =============================================================================
# Embedder Fixtures
# =============================================================================

class MockEmbeddings(Embeddings):
    """
    Mock embedding model for fast unit tests.

    Generates deterministic pseudo-embeddings based on text hash.
    This avoids API calls and model loading, making tests fast and
    reproducible.

    The embedding dimension is 64 (small for speed).
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate mock embeddings for a list of documents.

        Uses a simple hash-based approach to create deterministic
        embeddings that are consistent for the same input text.
        """
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text content
            hash_val = hash(text)
            embedding = [
                (hash_val % (2**31)) / (2**31) * 0.1
                for _ in range(self.dimension)
            ]
            # Add some variation based on text length
            embedding[0] += len(text) * 0.001
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate mock embedding for a single query."""
        return self.embed_documents([text])[0]


@pytest.fixture
def test_embedder() -> Embeddings:
    """
    Provide a mock embedder for fast unit tests.

    Returns a MockEmbeddings instance that generates deterministic
    pseudo-embeddings without requiring API keys or model loading.

    Use this fixture for unit tests where actual embedding quality
    doesn't matter, only the embedding interface behavior.

    Returns:
        Embeddings: Mock embedding model

    Example:
        def test_embedder_interface(test_embedder):
            texts = ["hello", "world"]
            embeddings = test_embedder.embed_documents(texts)
            assert len(embeddings) == 2
    """
    return MockEmbeddings(dimension=64)


@pytest.fixture
def local_embedder() -> Embeddings:
    """
    Provide a real local embedder for integration tests.

    Returns the sentence-transformers based embedder configured in
    the application. This is slower than test_embedder but provides
    real embeddings for integration tests.

    Use this when you need actual embedding similarity behavior,
    such as testing retrieval quality or MMR search.

    Returns:
        Embeddings: Real local embedding model

    Example:
        def test_embedding_similarity(local_embedder):
            texts = ["medical treatment", "healthcare"]
            embeddings = local_embedder.embed_documents(texts)
            # Test actual similarity computation
    """
    from app.embeddings.embedder import _LocalEmbeddings
    return _LocalEmbeddings()


# =============================================================================
# Vector Store Fixtures
# =============================================================================

@pytest.fixture
def test_vector_store(test_embedder: Embeddings, temp_chroma_path: Path):
    """
    Create a temporary ChromaDB vector store for testing.

    This fixture creates an isolated ChromaDB instance with:
    - A unique collection name to avoid conflicts
    - Temporary persistence directory (auto-cleaned)
    - Mock embedder for fast operation

    The store is empty when yielded and ready for test data.

    Args:
        test_embedder: Mock embedder fixture
        temp_chroma_path: Temporary directory fixture

    Yields:
        MedicalVectorStore: Configured vector store instance

    Example:
        def test_add_documents(test_vector_store):
            docs = [Document(page_content="test content")]
            test_vector_store.add_documents(docs)
            assert test_vector_store.get_stats()["total_chunks"] == 1
    """
    from app.vectorstore.chroma_store import MedicalVectorStore

    # Override collection name for test isolation
    original_collection = settings.CHROMA_COLLECTION_NAME
    original_persist = settings.CHROMA_PERSIST_DIR

    try:
        # Use test-specific settings
        settings.CHROMA_COLLECTION_NAME = f"test_medimax_{os.getpid()}"
        settings.CHROMA_PERSIST_DIR = str(temp_chroma_path)

        store = MedicalVectorStore(embedder=test_embedder)
        yield store
    finally:
        # Restore original settings
        settings.CHROMA_COLLECTION_NAME = original_collection
        settings.CHROMA_PERSIST_DIR = original_persist


@pytest.fixture
def cleanup_vector_store(test_vector_store):
    """
    Ensure vector store is cleaned up after test.

    This fixture wraps the test_vector_store and ensures complete
    cleanup after the test, including:
    - Deleting all documents from the collection
    - Removing the collection if possible

    Use this when tests might fail mid-way and leave orphaned data.

    Args:
        test_vector_store: Vector store fixture

    Yields:
        MedicalVectorStore: Clean vector store instance

    Example:
        def test_with_cleanup(cleanup_vector_store):
            # Even if this test fails, cleanup happens
            cleanup_vector_store.add_documents([...])
    """
    store = test_vector_store
    yield store
    # Cleanup: delete all documents
    try:
        collection = store._store._collection
        results = collection.get(include=[])
        if results.get("ids"):
            collection.delete(ids=results["ids"])
    except Exception:
        pass  # Best-effort cleanup


# =============================================================================
# Chunker Fixtures
# =============================================================================

@pytest.fixture
def test_chunker() -> MedicalChunker:
    """
    Create a MedicalChunker with test-optimized settings.

    Configures the chunker with smaller chunk sizes suitable for
    testing (faster processing, easier to verify results).

    Default test settings:
    - chunk_size: 500 (smaller than production for faster tests)
    - chunk_overlap: 50 (proportional overlap)

    Returns:
        MedicalChunker: Configured chunker instance

    Example:
        def test_chunking(test_chunker):
            docs = [Document(page_content="long text...")]
            chunks = test_chunker.chunk_documents(docs)
            assert len(chunks) > 0
    """
    return MedicalChunker(chunk_size=500, chunk_overlap=50)


@pytest.fixture
def production_chunker() -> MedicalChunker:
    """
    Create a MedicalChunker with production settings.

    Uses the same configuration as the live application.
    Use this for integration tests that need production-accurate
    chunking behavior.

    Returns:
        MedicalChunker: Production-configured chunker
    """
    return MedicalChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )


# =============================================================================
# Sample Document Fixtures
# =============================================================================

@pytest.fixture
def sample_documents() -> List[Document]:
    """
    Provide sample medical documents for testing.

    Returns a list of realistic medical document excerpts suitable
    for testing ingestion, chunking, and retrieval pipelines.

    The documents cover different medical domains to test metadata
    enrichment and domain-based filtering.

    Returns:
        List[Document]: Sample documents with medical content

    Example:
        def test_ingestion(sample_documents, test_chunker):
            chunks = test_chunker.chunk_documents(sample_documents)
            assert len(chunks) >= len(sample_documents)
    """
    return [
        Document(
            page_content="""
            Hypertension (High Blood Pressure)

            Hypertension is a chronic medical condition characterized by elevated
            blood pressure in the arteries. According to the American Heart
            Association, hypertension affects approximately 108 million adults
            in the United States.

            Diagnosis: Blood pressure is measured in millimeters of mercury
            (mm Hg) and recorded as two numbers: systolic pressure (when the
            heart beats) over diastolic pressure (when the heart rests).
            Normal blood pressure is below 120/80 mm Hg.

            Treatment options include lifestyle modifications such as reduced
            sodium intake, regular exercise, and weight management. First-line
            medications include thiazide diuretics, ACE inhibitors, and calcium
            channel blockers.
            """,
            metadata={"source": "hypertension_guide.pdf", "page": 1},
        ),
        Document(
            page_content="""
            Type 2 Diabetes Mellitus

            Type 2 diabetes is a metabolic disorder characterized by insulin
            resistance and relative insulin deficiency. The prevalence has
            increased dramatically worldwide, with an estimated 462 million
            adults affected globally.

            Risk factors include obesity, sedentary lifestyle, family history,
            and certain ethnic backgrounds. Screening is recommended for adults
            aged 35-70 years who have overweight or obesity.

            Management focuses on glycemic control through diet, exercise, and
            pharmacotherapy. Metformin remains the first-line medication, with
            HbA1c target typically set at less than 7% for most adults.
            """,
            metadata={"source": "diabetes_overview.pdf", "page": 1},
        ),
        Document(
            page_content="""
            Myocardial Infarction (Heart Attack)

            Acute myocardial infarction occurs when blood flow to a part of
            the heart is blocked, causing damage to the heart muscle. It is
            a medical emergency requiring immediate treatment.

            Symptoms include chest pain or discomfort, shortness of breath,
            nausea, and cold sweat. However, symptoms can be atypical,
            especially in women and diabetic patients.

            Treatment involves immediate aspirin administration, reperfusion
            therapy (PCI or thrombolytics), and supportive care. Long-term
            management includes antiplatelet therapy, statins, and lifestyle
            modifications.
            """,
            metadata={"source": "cardiac_emergency.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def sample_document() -> Document:
    """
    Provide a single sample document for simple tests.

    Returns a minimal medical document suitable for basic unit tests
    that only need one document.

    Returns:
        Document: Single sample document

    Example:
        def test_single_document(sample_document, test_chunker):
            chunks = test_chunker.chunk_documents([sample_document])
            assert len(chunks) >= 1
    """
    return Document(
        page_content="""
        Medical Test Document

        This is a sample medical document for testing purposes.
        It contains information about various medical conditions
        and treatments. The content is designed to test chunking,
        embedding, and retrieval functionality.

        Key points covered:
        1. Sample medical terminology
        2. Treatment protocols
        3. Diagnostic criteria
        4. Patient management guidelines
        """,
        metadata={"source": "test_document.pdf", "page": 1},
    )


@pytest.fixture
def empty_document() -> Document:
    """
    Provide an empty document for edge case testing.

    Returns a document with empty content to test edge cases
    and error handling.

    Returns:
        Document: Empty document
    """
    return Document(
        page_content="",
        metadata={"source": "empty.pdf", "page": 1},
    )


@pytest.fixture
def whitespace_document() -> Document:
    """
    Provide a whitespace-only document for edge case testing.

    Returns a document containing only whitespace to test
    chunking edge cases and filtering logic.

    Returns:
        Document: Whitespace-only document
    """
    return Document(
        page_content="   \n\n\t\n   ",
        metadata={"source": "whitespace.pdf", "page": 1},
    )


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def no_api_key_env():
    """
    Temporarily clear API key environment variables.

    This fixture ensures tests run without external API dependencies
    by clearing MISTRAL_API_KEY and similar variables. Restores the
    original values after the test completes.

    Use this for unit tests that should not make API calls.

    Example:
        def test_local_fallback(no_api_key_env):
            # Forces use of local embeddings
            embedder = get_embedder()
            assert isinstance(embedder, _LocalEmbeddings)
    """
    original = os.environ.get("MISTRAL_API_KEY")
    os.environ.pop("MISTRAL_API_KEY", None)
    try:
        yield
    finally:
        if original is not None:
            os.environ["MISTRAL_API_KEY"] = original


# =============================================================================
# Test Data Directories
# =============================================================================

@pytest.fixture
def test_data_dir(temp_dir: Path) -> Path:
    """
    Create a test data directory structure.

    Creates a standard test directory structure mimicking the
    production layout:
    - test_data/docs/ (for document files)
    - test_data/storage/ (for database files)

    Args:
        temp_dir: Base temporary directory fixture

    Returns:
        Path: Path to the test data directory

    Example:
        def test_file_loading(test_data_dir):
            docs_dir = test_data_dir / "docs"
            test_file = docs_dir / "test.pdf"
            # Create and test file loading
    """
    docs_dir = temp_dir / "docs"
    storage_dir = temp_dir / "storage"
    docs_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


# =============================================================================
# Helper Utilities
# =============================================================================

@pytest.fixture
def create_test_file(temp_dir: Path):
    """
    Provide a factory function for creating test files.

    Returns a function that creates files with specified content
    in the temporary directory. Useful for testing file loading
    and ingestion.

    Args:
        temp_dir: Base temporary directory fixture

    Returns:
        Callable[[str, str, str], Path]: Function to create test files

    Example:
        def test_file_ingestion(create_test_file):
            pdf_path = create_test_file(
                "test.pdf",
                "Medical content here...",
                "application/pdf"
            )
            # Test ingestion of the file
    """
    def _create(filename: str, content: str, subdir: str = "") -> Path:
        if subdir:
            target_dir = temp_dir / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            file_path = target_dir / filename
        else:
            file_path = temp_dir / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    return _create
