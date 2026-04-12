# Indexing Issues, Testing & Evaluation Plan

## Overview
This plan addresses three interconnected goals:
1. Fix indexing issues in the MediMax RAG system
2. Create comprehensive test coverage
3. Build an evaluation system for query quality

## Current State Analysis

### Strengths
- Medical-aware chunking (preserves Dr., Fig., et al.)
- Upsert semantics for safe re-ingestion
- Confidence threshold filtering
- Clean architecture with separation of concerns

### Gaps Identified
1. **No test coverage** - Zero unit/integration tests
2. **No evaluation system** - Cannot measure retrieval quality or answer accuracy
3. **Unknown indexing health** - No way to verify chunks are being created/stored correctly
4. **No query testing** - No systematic way to test different query types

## Tasks

### Task 1: Create Test Infrastructure
**Goal:** Set up pytest infrastructure with proper fixtures and configuration

**Requirements:**
- Create `tests/` directory structure
- Add `pytest.ini` and `conftest.py` with shared fixtures
- Create fixtures for: test embedder, test vector store, test chunker, test documents
- Configure pytest markers (unit, integration, slow)
- Add test requirements to `requirements.txt` or `requirements-dev.txt`

**Files to create:**
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/pytest.ini` or root-level `pytest.ini`
- `requirements-dev.txt` (if not exists)

---

### Task 2: Unit Tests for Core Components
**Goal:** Achieve 80%+ unit test coverage on core modules

**Test modules to create:**
- `tests/test_chunker.py` - MedicalChunker tests (separator behavior, PDF artifact cleanup, metadata enrichment)
- `tests/test_loader.py` - Document loader tests (PDF, TXT, MD, DOCX, HTML loading, encoding fallbacks)
- `tests/test_embedder.py` - Embedder tests (Mistral vs local embeddings, fallback behavior)
- `tests/test_vector_store.py` - ChromaDB tests (upsert semantics, delete, search, MMR)
- `tests/test_retriever.py` - Retriever tests (MMR search, confidence filtering, metadata filters)

**Key test cases:**
- Chunker: Verify '. ' is NOT a separator (Dr. Smith stays intact)
- Chunker: Verify PDF hyphenation cleanup works
- Vector Store: Verify re-ingesting same file doesn't duplicate
- Retriever: Verify confidence threshold filtering

---

### Task 3: Integration Tests for Pipelines
**Goal:** End-to-end tests for ingest and query pipelines

**Test modules to create:**
- `tests/test_ingest_pipeline.py` - Full ingestion flow tests
- `tests/test_query_pipeline.py` - Full query flow tests

**Key test cases:**
- Ingest: Single file ingestion
- Ingest: Directory batch ingestion
- Ingest: Incremental ingest (skip already-indexed files)
- Ingest: In-memory bytes ingestion (API upload path)
- Query: Basic Q&A with grounded answer
- Query: Low confidence fallback behavior
- Query: Metadata filtering (domain, year, source)

---

### Task 4: Indexing Diagnostics & Health Checks
**Goal:** Tools to verify indexing health and diagnose issues

**Files to create:**
- `app/tools/index_health.py` - Index health diagnostics

**Features:**
- Report total chunks, unique sources, chunk size distribution
- Detect orphaned chunks (source file no longer exists)
- Detect duplicate content (same text, different IDs)
- Verify chunk_index is contiguous per source
- Check for empty/corrupted chunks

**CLI command:**
- Add `python -m app.tools.index_health` script

---

### Task 5: Query Evaluation System
**Goal:** Systematic evaluation of RAG quality

**Files to create:**
- `app/evaluation/evaluator.py` - Core evaluation logic
- `app/evaluation/metrics.py` - Quality metrics (precision, recall, groundedness)
- `app/evaluation/test_queries.py` - Standard query test set

**Metrics to implement:**
- **Retrieval metrics:**
  - Precision@K (are retrieved docs relevant?)
  - Mean Reciprocal Rank (is relevant doc ranked high?)
  - Coverage (what % of corpus is retrievable?)
  
- **Generation metrics:**
  - Groundedness (does answer cite retrieved sources?)
  - Hallucination rate (claims not in context)
  - Answer relevance (does answer address question?)

**Test query dataset:**
- Create 20-30 medical questions with known answers
- Include: factoid questions, comparative questions, "I don't know" questions
- Tag by domain (cardiology, oncology, etc.)

**CLI command:**
- `python -m app.evaluate run --dataset=tests/eval_queries.json`

---

### Task 6: Query Testing Harness
**Goal:** Interactive tool to test queries and inspect results

**Files to create:**
- `app/tools/query_debug.py` - Query inspection tool

**Features:**
- Run a query and display:
  - Retrieved chunks with scores
  - Confidence score breakdown
  - Full LLM prompt sent
  - Raw LLM response
  - Final formatted answer
- Compare MMR vs similarity search results
- Test different K values, filter combinations

**CLI command:**
- `python -m app.tools.query_debug "What is the treatment for hypertension?"`

---

### Task 7: Performance Benchmarks
**Goal:** Measure and track system performance

**Files to create:**
- `tests/benchmarks/test_benchmarks.py` - Performance benchmarks

**Benchmarks:**
- Ingestion throughput (docs/sec, chunks/sec)
- Query latency (p50, p95, p99)
- Embedding computation time
- Retrieval time by K value

**Output:**
- JSON report with benchmark results
- Optional: integration with CI/CD for regression detection

---

## Execution Order

Tasks should be executed in order:
1. **Task 1** first (infrastructure is prerequisite)
2. **Task 2** (unit tests provide foundation)
3. **Task 3** (integration tests build on unit tests)
4. **Task 4** (diagnostics help verify indexing)
5. **Task 5** (evaluation system for quality)
6. **Task 6** (debugging tool for development)
7. **Task 7** (performance tracking)

## Acceptance Criteria

- [ ] All existing code has 80%+ test coverage
- [ ] Index health tool runs and reports clean bill of health
- [ ] Evaluation system can run against test dataset and produce metrics report
- [ ] Query debug tool provides full visibility into RAG pipeline
- [ ] Performance benchmarks establish baseline metrics
- [ ] All tests pass on CI/CD (when integrated)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Mistral API rate limits during tests | Use local embeddings for unit tests, mock API calls |
| ChromaDB state pollution between tests | Use temporary directories, cleanup fixtures |
| Test suite becomes slow | Mark slow tests, run subset in CI |
| Evaluation dataset becomes stale | Version the dataset, update quarterly |
