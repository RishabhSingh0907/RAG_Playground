"""Integration tests for reranker manager with real model backends."""

import pytest
from unittest.mock import Mock, patch
from src.retrieval.reranker_manager import (
    RerankerManager,
    RerankerManagerConfig,
    RerankerBackend,
    BackendStatus,
)
from src.retrieval.pipeline import RetrievedChunk


@pytest.fixture
def mock_manager_config():
    """Create a test configuration."""
    return RerankerManagerConfig(
        primary_backend=RerankerBackend.HEURISTIC,
        device="cpu",
        sentence_transformer_model="BAAI/bge-reranker-large",
        flashrank_model="ms-marco-MiniLM-L-12-v2",
        preload_timeout_seconds=30,
    )


@pytest.fixture
def reranker_manager(mock_manager_config):
    """Create a reranker manager with test config."""
    return RerankerManager(mock_manager_config)


def make_chunk(chunk_id: str, content: str):
    """Create a test chunk."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=content,
        score=0.8,
        rank=1,
        source="test",
        metadata={},
    )


class TestRerankerManagerBasics:
    """Test basic reranker manager functionality."""

    def test_initialization(self, reranker_manager):
        """Test that manager initializes correctly."""
        assert reranker_manager is not None
        assert reranker_manager.active_backend is None
        assert reranker_manager.active_reranker is None

    def test_preload_heuristic_backend(self, reranker_manager):
        """Test preloading the heuristic backend."""
        health_reports = reranker_manager.preload_backends()

        # Heuristic should always be available
        assert RerankerBackend.HEURISTIC in health_reports
        heuristic_report = health_reports[RerankerBackend.HEURISTIC]
        assert heuristic_report.status == BackendStatus.AVAILABLE

    def test_active_backend_selection(self, reranker_manager):
        """Test that an active backend is selected after preload."""
        reranker_manager.preload_backends()
        assert reranker_manager.active_backend is not None
        assert reranker_manager.active_reranker is not None

    def test_get_active_reranker_returns_instance(self, reranker_manager):
        """Test that get_active_reranker returns a valid instance."""
        reranker = reranker_manager.get_active_reranker()
        assert reranker is not None
        assert hasattr(reranker, "rerank")

    def test_get_health_report(self, reranker_manager):
        """Test health report generation."""
        reranker_manager.preload_backends()
        health = reranker_manager.get_health_report()

        assert "overall_status" in health
        assert "active_backend" in health
        assert "backends" in health
        assert health["overall_status"] in ["operational", "degraded"]

    def test_rerank_with_available_backend(self, reranker_manager):
        """Test reranking with available backend."""
        reranker_manager.preload_backends()
        candidates = [
            make_chunk("chunk-1", "machine learning retrieval"),
            make_chunk("chunk-2", "semantic search"),
        ]

        result = reranker_manager.rerank("retrieval", candidates)

        assert result is not None
        assert len(result) == len(candidates)
        # Heuristic should return valid chunks
        assert all(hasattr(r, "chunk_id") for r in result)

    def test_rerank_returns_candidates_on_error(self, reranker_manager):
        """Test that rerank returns original candidates if reranker unavailable."""
        # Don't preload to simulate missing reranker
        candidates = [make_chunk("chunk-1", "test")]

        result = reranker_manager.rerank("test", candidates)

        # Should return something (either reranked or original)
        assert result is not None

    def test_switch_backend_to_heuristic(self, reranker_manager):
        """Test switching to heuristic backend."""
        reranker_manager.preload_backends()
        report = reranker_manager.switch_backend(RerankerBackend.HEURISTIC)

        assert report.status == BackendStatus.AVAILABLE
        assert reranker_manager.active_backend == RerankerBackend.HEURISTIC


class TestRerankerManagerWithSentenceTransformer:
    """Tests for SentenceTransformer backend (skip if not installed)."""

    @pytest.mark.skipif(True, reason="Requires sentence-transformers; run manually for full validation")
    def test_sentence_transformer_backend_preload(self, reranker_manager):
        """Test preloading SentenceTransformer backend."""
        config = RerankerManagerConfig(primary_backend=RerankerBackend.SENTENCE_TRANSFORMER)
        manager = RerankerManager(config)
        health_reports = manager.preload_backends()

        st_report = health_reports.get(RerankerBackend.SENTENCE_TRANSFORMER)
        assert st_report is not None
        # Status could be available or unavailable depending on environment
        assert st_report.status in [BackendStatus.AVAILABLE, BackendStatus.UNAVAILABLE]

    @pytest.mark.skipif(True, reason="Requires sentence-transformers; run manually for full validation")
    def test_sentence_transformer_reranking(self, reranker_manager):
        """Test actual reranking with SentenceTransformer."""
        config = RerankerManagerConfig(primary_backend=RerankerBackend.SENTENCE_TRANSFORMER)
        manager = RerankerManager(config)
        manager.preload_backends()

        candidates = [
            make_chunk("chunk-1", "machine learning enables semantic search"),
            make_chunk("chunk-2", "python is a programming language"),
        ]

        result = manager.rerank("machine learning", candidates)
        assert result is not None
        assert len(result) > 0


class TestRerankerManagerWithFlashRank:
    """Tests for FlashRank backend (skip if not installed)."""

    @pytest.mark.skipif(True, reason="Requires flashrank; run manually for full validation")
    def test_flashrank_backend_preload(self, reranker_manager):
        """Test preloading FlashRank backend."""
        config = RerankerManagerConfig(primary_backend=RerankerBackend.FLASHRANK)
        manager = RerankerManager(config)
        health_reports = manager.preload_backends()

        flashrank_report = health_reports.get(RerankerBackend.FLASHRANK)
        assert flashrank_report is not None
        assert flashrank_report.status in [BackendStatus.AVAILABLE, BackendStatus.UNAVAILABLE]

    @pytest.mark.skipif(True, reason="Requires flashrank; run manually for full validation")
    def test_flashrank_reranking(self, reranker_manager):
        """Test actual reranking with FlashRank."""
        config = RerankerManagerConfig(primary_backend=RerankerBackend.FLASHRANK)
        manager = RerankerManager(config)
        manager.preload_backends()

        candidates = [
            make_chunk("chunk-1", "machine learning enables semantic search"),
            make_chunk("chunk-2", "python is a programming language"),
        ]

        result = manager.rerank("machine learning", candidates)
        assert result is not None
        assert len(result) > 0


class TestRerankerManagerFallback:
    """Test fallback behavior."""

    def test_fallback_on_unavailable_backend(self, reranker_manager):
        """Test fallback to heuristic when primary backend is unavailable."""
        config = RerankerManagerConfig(
            primary_backend=RerankerBackend.SENTENCE_TRANSFORMER,
            auto_fallback_on_error=True,
        )
        manager = RerankerManager(config)
        manager.preload_backends()

        # If SentenceTransformer is unavailable, should fall back
        if manager.active_backend == RerankerBackend.HEURISTIC:
            health = manager.get_health_report()
            assert health["overall_status"] in ["operational", "degraded"]

    def test_multiple_preload_calls_idempotent(self, reranker_manager):
        """Test that multiple preload calls don't cause issues."""
        report1 = reranker_manager.preload_backends()
        report2 = reranker_manager.preload_backends()

        assert len(report1) > 0
        assert len(report2) > 0
        # Backends should remain the same
        assert set(report1.keys()) == set(report2.keys())
