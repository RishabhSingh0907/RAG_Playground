"""Reranker management system with pre-loading, health checks, and diagnostics.

This module handles:
- Safe model pre-loading during app initialization (not on query)
- Backend health checks and availability validation
- Explicit error handling and fallback with user visibility
- Device selection (CPU/CUDA) with auto-detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

from src.retrieval.pipeline_v2 import (
    RerankerBackend,
    SentenceTransformerRerankerV2,
    FlashRankRerankerV2,
    HeuristicReranker,
)
from src.retrieval.pipeline import RetrievedChunk
from src.utils.logger import get_logger


logger = get_logger(__name__)


class BackendStatus(str, Enum):
    """Health status of a reranker backend."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    ERROR = "error"


@dataclass
class BackendHealthReport:
    """Diagnostic report for a single backend."""

    backend: RerankerBackend
    status: BackendStatus
    message: str = ""
    error: Optional[str] = None
    loaded_at: Optional[float] = None
    model_name: Optional[str] = None
    device: Optional[str] = None


@dataclass
class RerankerManagerConfig:
    """Configuration for the reranker manager."""

    primary_backend: RerankerBackend = RerankerBackend.SENTENCE_TRANSFORMER
    device: Optional[str] = None
    sentence_transformer_model: str = "BAAI/bge-reranker-large"
    flashrank_model: str = "ms-marco-MiniLM-L-12-v2"
    flashrank_cache_dir: str = "./data/flashrank"
    preload_timeout_seconds: int = 60
    auto_fallback_on_error: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for UI display."""
        return {
            "primary_backend": self.primary_backend.value,
            "device": self.device or "auto",
            "sentence_transformer_model": self.sentence_transformer_model,
            "flashrank_model": self.flashrank_model,
            "flashrank_cache_dir": self.flashrank_cache_dir,
            "preload_timeout_seconds": self.preload_timeout_seconds,
            "auto_fallback_on_error": self.auto_fallback_on_error,
        }


class RerankerManager:
    """Manage reranker lifecycle, health, and selection.

    Responsibilities:
    - Pre-load selected backend during initialization (with timeout)
    - Validate backend availability before queries
    - Track health status and errors
    - Provide explicit fallback with visibility
    - Support runtime backend switching
    """

    def __init__(self, config: RerankerManagerConfig):
        self.config = config
        self.rerankers: Dict[RerankerBackend, Any] = {}
        self.health_reports: Dict[RerankerBackend, BackendHealthReport] = {}
        self.active_backend: Optional[RerankerBackend] = None
        self.active_reranker: Optional[Any] = None
        self._lock = threading.Lock()
        self._preload_complete = False

        logger.info(
            "RerankerManager initialized",
            primary_backend=config.primary_backend.value,
            device=config.device or "auto",
        )

    def preload_backends(self) -> Dict[RerankerBackend, BackendHealthReport]:
        """Pre-load backends during app initialization.

        Returns:
            Dict mapping backend to its health report.
        """
        with self._lock:
            if self._preload_complete:
                logger.info("Backends already preloaded")
                return self.health_reports

            # Try primary backend first
            self._preload_backend(self.config.primary_backend)

            # Pre-stage other backends in order
            for backend in RerankerBackend:
                if backend != self.config.primary_backend:
                    self._preload_backend(backend, background=True)

            self._preload_complete = True

            # Select first available backend
            self._select_first_available()

            logger.info(
                "Backend preload complete",
                active_backend=self.active_backend.value if self.active_backend else None,
            )

            return self.health_reports

    def _preload_backend(self, backend: RerankerBackend, background: bool = False) -> None:
        """Pre-load a single backend.

        Args:
            backend: Backend to preload
            background: If True, run in background thread
        """
        if backend in self.rerankers:
            return

        if background:
            thread = threading.Thread(target=self._load_backend_sync, args=(backend,), daemon=True)
            thread.start()
        else:
            self._load_backend_sync(backend)

    def _load_backend_sync(self, backend: RerankerBackend) -> None:
        """Synchronously load a backend with error handling."""
        report = BackendHealthReport(
            backend=backend,
            status=BackendStatus.LOADING,
            model_name=self._get_model_name(backend),
            device=self.config.device,
        )

        try:
            if backend == RerankerBackend.SENTENCE_TRANSFORMER:
                reranker = SentenceTransformerRerankerV2(
                    model_name=self.config.sentence_transformer_model,
                    device=self.config.device,
                )
                if not reranker.available:
                    report.status = BackendStatus.UNAVAILABLE
                    report.error = reranker._load_error
                    report.message = f"SentenceTransformer initialization failed: {reranker._load_error}"
                else:
                    report.status = BackendStatus.AVAILABLE
                    report.message = "SentenceTransformer loaded successfully"
                    self.rerankers[backend] = reranker

            elif backend == RerankerBackend.FLASHRANK:
                reranker = FlashRankRerankerV2(
                    model_name=self.config.flashrank_model,
                    cache_dir=self.config.flashrank_cache_dir,
                )
                if not reranker.available:
                    report.status = BackendStatus.UNAVAILABLE
                    report.error = reranker._load_error
                    report.message = f"FlashRank initialization failed: {reranker._load_error}"
                else:
                    report.status = BackendStatus.AVAILABLE
                    report.message = "FlashRank loaded successfully"
                    self.rerankers[backend] = reranker

            elif backend == RerankerBackend.HEURISTIC:
                reranker = HeuristicReranker(
                    model_name=self.config.sentence_transformer_model,
                    device=self.config.device,
                )
                report.status = BackendStatus.AVAILABLE
                report.message = "Heuristic reranker ready (no model download)"
                self.rerankers[backend] = reranker

        except Exception as exc:
            report.status = BackendStatus.ERROR
            report.error = str(exc)
            report.message = f"Backend initialization error: {str(exc)}"
            logger.error(
                "Backend load failed",
                backend=backend.value,
                error=str(exc),
                exc_info=True,
            )

        self.health_reports[backend] = report
        logger.info(
            f"Backend preload result: {report.message}",
            backend=backend.value,
            status=report.status.value,
            details=report.message,
        )

    def _select_first_available(self) -> None:
        """Select first available backend in priority order."""
        priority_order = [
            self.config.primary_backend,
            RerankerBackend.HEURISTIC,
        ]

        for backend in priority_order:
            report = self.health_reports.get(backend)
            if report and report.status == BackendStatus.AVAILABLE:
                self.active_backend = backend
                self.active_reranker = self.rerankers.get(backend)
                logger.info(
                    "Active backend selected",
                    backend=backend.value,
                )
                return

        logger.warning("No available backend found; heuristic will be lazy-loaded on first query")

    def get_active_reranker(self) -> Optional[Any]:
        """Get the currently active reranker.

        Returns:
            Active reranker instance or None if not available.
        """
        with self._lock:
            if self.active_reranker is not None:
                return self.active_reranker

            if self.active_backend is None:
                self.active_backend = RerankerBackend.HEURISTIC

            if self.active_backend not in self.rerankers:
                self._load_backend_sync(self.active_backend)

            return self.rerankers.get(self.active_backend)

    def switch_backend(self, backend: RerankerBackend) -> BackendHealthReport:
        """Switch to a different reranker backend.

        Args:
            backend: Backend to switch to.

        Returns:
            Health report for the selected backend.
        """
        with self._lock:
            if backend not in self.health_reports:
                self._load_backend_sync(backend)

            report = self.health_reports.get(backend)
            if report and report.status == BackendStatus.AVAILABLE:
                self.active_backend = backend
                self.active_reranker = self.rerankers.get(backend)
                logger.info(
                    "Backend switched",
                    new_backend=backend.value,
                )
                return report

            logger.warning(
                "Cannot switch to unavailable backend; staying on current",
                requested_backend=backend.value,
                current_backend=self.active_backend.value if self.active_backend else None,
            )
            return report or BackendHealthReport(
                backend=backend,
                status=BackendStatus.UNAVAILABLE,
                message="Backend not available",
            )

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report for all backends.

        Returns:
            Dict with overall status and per-backend reports.
        """
        reports = {}
        for backend, report in self.health_reports.items():
            reports[backend.value] = {
                "status": report.status.value,
                "details": report.message,
                "error": report.error,
                "model_name": report.model_name,
                "device": report.device,
            }

        # Overall status: operational if any backend available, degraded otherwise
        has_available_backend = any(
            report.status == BackendStatus.AVAILABLE 
            for report in self.health_reports.values()
        )
        overall_status = "operational" if has_available_backend else "degraded"

        return {
            "overall_status": overall_status,
            "active_backend": self.active_backend.value if self.active_backend else None,
            "backends": reports,
        }

    def rerank(self, query: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Rerank candidates using the active backend.

        Args:
            query: Search query
            candidates: Chunks to rerank

        Returns:
            Reranked chunks or original candidates if reranking unavailable.
        """
        reranker = self.get_active_reranker()
        if reranker is None:
            logger.warning("No reranker available; returning original order")
            return candidates

        try:
            return reranker.rerank(query, candidates)
        except Exception as exc:
            logger.error(
                "Reranking failed",
                backend=self.active_backend.value if self.active_backend else "unknown",
                error=str(exc),
                exc_info=True,
            )
            return candidates

    def _get_model_name(self, backend: RerankerBackend) -> str:
        """Get model name for a backend."""
        if backend == RerankerBackend.SENTENCE_TRANSFORMER:
            return self.config.sentence_transformer_model
        elif backend == RerankerBackend.FLASHRANK:
            return self.config.flashrank_model
        return "heuristic"
