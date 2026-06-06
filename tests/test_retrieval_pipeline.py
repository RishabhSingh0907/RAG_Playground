"""Tests for the session-scoped retrieval pipeline."""

from types import SimpleNamespace

from src.ingestion.models import DocumentChunk, VectorDBType
from src.retrieval.pipeline import RetrievalPipeline, RetrievalSettings, RetrievedChunk, SearchMode
from src.retrieval.pipeline_v2 import RetrievalPipelineV2, RetrievalSettingsV2, RerankerBackend
from src.vector_db.base import SearchResult


class FakeStorageManager:
    def __init__(self, chunks):
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.cleared = False

    def get_chunk(self, chunk_id):
        return self.chunks.get(chunk_id)

    def get_all_chunks(self, limit=1000):
        return list(self.chunks.values())[:limit]

    def clear_all_chunks(self):
        self.cleared = True
        removed_count = len(self.chunks)
        self.chunks.clear()
        return removed_count


class FakeVectorDB:
    def __init__(self, search_results):
        self.search_results = search_results
        self.cleared = False

    def search(self, query_vector, k=5, filters=None):
        return self.search_results[:k]

    def clear(self):
        self.cleared = True


class FakeEmbeddingManager:
    def embed_single(self, text):
        return [float(len(text)), 1.0, 0.5]


class FakeReranker:
    def rerank(self, query, candidates):
        reranked = list(reversed(candidates))
        for index, candidate in enumerate(reranked, start=1):
            candidate.rank = index
            candidate.score = float(len(reranked) - index + 1)
            candidate.reranker_score = candidate.score
        return reranked


class FakeManager:
    def __init__(self, chunks, vector_results):
        self.storage_manager = FakeStorageManager(chunks)
        self.vector_db = FakeVectorDB(vector_results)
        self.embedding_manager = FakeEmbeddingManager()
        self.config = SimpleNamespace(
            storage=SimpleNamespace(
                vector_db=SimpleNamespace(
                    type=VectorDBType.FAISS,
                    faiss=SimpleNamespace(distance_metric="l2"),
                )
            )
        )

    def get_stats(self):
        return {"total_chunks": len(self.storage_manager.chunks)}

    def get_all_chunks(self, limit=1000):
        return self.storage_manager.get_all_chunks(limit=limit)


def make_chunk(chunk_id, content):
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=content,
        chunk_index=0,
        metadata={"source": "test"},
    )


def test_dense_search_returns_semantic_results_in_rank_order():
    chunks = [
        make_chunk("chunk-1", "machine learning enables semantic retrieval"),
        make_chunk("chunk-2", "unrelated content about cooking"),
    ]
    vector_results = [
        SearchResult(chunk_id="chunk-1", distance=0.2, metadata={"document_id": "doc-1"}),
        SearchResult(chunk_id="chunk-2", distance=0.9, metadata={"document_id": "doc-1"}),
    ]
    manager = FakeManager(chunks, vector_results)
    settings = RetrievalSettings(search_mode=SearchMode.DENSE, top_k=2, reranker_enabled=False)
    pipeline = RetrievalPipeline(manager, settings=settings, reranker=FakeReranker())

    results = pipeline.search("semantic retrieval")

    assert [result.chunk_id for result in results] == ["chunk-1", "chunk-2"]
    assert results[0].semantic_score is not None


def test_lexical_search_prioritizes_matching_terms():
    chunks = [
        make_chunk("chunk-1", "machine learning retrieval systems use tokens"),
        make_chunk("chunk-2", "gardening and cooking tips"),
    ]
    manager = FakeManager(chunks, [])
    settings = RetrievalSettings(search_mode=SearchMode.LEXICAL, top_k=1, reranker_enabled=False)
    pipeline = RetrievalPipeline(manager, settings=settings, reranker=FakeReranker())

    results = pipeline.search("machine learning tokens")

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].lexical_score is not None


def test_hybrid_search_combines_dense_and_lexical_candidates():
    chunks = [
        make_chunk("chunk-1", "machine learning retrieval systems use tokens"),
        make_chunk("chunk-2", "semantic search ranks chunks by distance"),
        make_chunk("chunk-3", "gardening and cooking tips"),
    ]
    vector_results = [
        SearchResult(chunk_id="chunk-1", distance=0.1, metadata={"document_id": "doc-1"}),
        SearchResult(chunk_id="chunk-2", distance=0.3, metadata={"document_id": "doc-1"}),
    ]
    manager = FakeManager(chunks, vector_results)
    settings = RetrievalSettings(
        search_mode=SearchMode.HYBRID,
        top_k=3,
        semantic_candidate_k=2,
        lexical_candidate_k=2,
        reranker_enabled=False,
    )
    pipeline = RetrievalPipeline(manager, settings=settings, reranker=FakeReranker())

    results = pipeline.search("machine learning retrieval")

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-1"
    assert results[0].source == "hybrid"
    assert results[0].score > 0


def test_injected_reranker_reorders_candidates():
    chunks = [
        make_chunk("chunk-1", "first candidate"),
        make_chunk("chunk-2", "second candidate"),
    ]
    vector_results = [
        SearchResult(chunk_id="chunk-1", distance=0.1, metadata={"document_id": "doc-1"}),
        SearchResult(chunk_id="chunk-2", distance=0.2, metadata={"document_id": "doc-1"}),
    ]
    manager = FakeManager(chunks, vector_results)
    settings = RetrievalSettings(search_mode=SearchMode.DENSE, top_k=2, reranker_enabled=True)
    pipeline = RetrievalPipeline(manager, settings=settings, reranker=FakeReranker())

    results = pipeline.search("candidate query")

    assert [result.chunk_id for result in results] == ["chunk-2", "chunk-1"]
    assert all(result.reranker_score is not None for result in results)


def test_clear_session_data_clears_backends():
    chunks = [make_chunk("chunk-1", "machine learning retrieval systems use tokens")]
    manager = FakeManager(chunks, [])
    settings = RetrievalSettings(search_mode=SearchMode.DENSE, reranker_enabled=False)
    pipeline = RetrievalPipeline(manager, settings=settings, reranker=FakeReranker())

    summary = pipeline.clear_session_data()

    assert summary["storage_cleared"] is True
    assert summary["vector_db_cleared"] is True
    assert summary["chunks_removed"] == 1
    assert manager.storage_manager.cleared is True
    assert manager.vector_db.cleared is True


def test_v2_settings_load_backend_configuration():
    config = {
        "retrieval": {
            "search_mode": "hybrid",
            "top_k": 7,
            "reranker": {
                "enabled": True,
                "backend": "flashrank",
                "model": "BAAI/bge-reranker-large",
                "device": "cpu",
                "flashrank": {
                    "model": "ms-marco-MiniLM-L-12-v2",
                    "cache_dir": "./tmp/flashrank",
                },
            },
        }
    }

    settings = RetrievalSettingsV2.from_mapping(config)

    assert settings.search_mode == SearchMode.HYBRID
    assert settings.top_k == 7
    assert settings.reranker_backend == RerankerBackend.FLASHRANK
    assert settings.flashrank_model == "ms-marco-MiniLM-L-12-v2"
    assert settings.flashrank_cache_dir == "./tmp/flashrank"


def test_v2_pipeline_uses_sentence_transformer_backend(monkeypatch):
    chunks = [
        make_chunk("chunk-1", "first candidate"),
        make_chunk("chunk-2", "second candidate"),
    ]
    vector_results = [
        SearchResult(chunk_id="chunk-1", distance=0.1, metadata={"document_id": "doc-1"}),
        SearchResult(chunk_id="chunk-2", distance=0.2, metadata={"document_id": "doc-1"}),
    ]
    manager = FakeManager(chunks, vector_results)

    class FakeSentenceTransformerRerankerV2:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device
            self.available = True

        def rerank(self, query, candidates):
            reranked = list(reversed(candidates))
            for index, candidate in enumerate(reranked, start=1):
                candidate.rank = index
                candidate.score = float(len(reranked) - index + 1)
                candidate.reranker_score = candidate.score
            return reranked

    monkeypatch.setattr("src.retrieval.pipeline_v2.SentenceTransformerRerankerV2", FakeSentenceTransformerRerankerV2)

    settings = RetrievalSettingsV2(
        search_mode=SearchMode.DENSE,
        top_k=2,
        reranker_enabled=True,
        reranker_backend=RerankerBackend.SENTENCE_TRANSFORMER,
    )
    pipeline = RetrievalPipelineV2(manager, settings=settings)

    results = pipeline.search("candidate query")

    assert [result.chunk_id for result in results] == ["chunk-2", "chunk-1"]
    assert all(result.reranker_score is not None for result in results)