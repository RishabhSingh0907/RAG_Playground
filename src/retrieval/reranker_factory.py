"""Factory and initialization utilities for reranker manager."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from src.retrieval.reranker_manager import RerankerManager, RerankerManagerConfig, RerankerBackend
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_reranker_manager_from_config(config_path: str) -> RerankerManager:
    """Load reranker manager from YAML configuration.

    Args:
        config_path: Path to retrieval_config.yaml

    Returns:
        Initialized RerankerManager instance.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(
            "Retrieval config not found; using reranker defaults",
            config_path=config_path,
        )
        return RerankerManager(RerankerManagerConfig())

    try:
        with open(config_file, "r", encoding="utf-8") as handle:
            config_data = yaml.safe_load(handle) or {}

        retrieval_config = config_data.get("retrieval", {})
        reranker_config = retrieval_config.get("reranker", {})
        flashrank_config = retrieval_config.get("flashrank", {})

        manager_config = RerankerManagerConfig(
            primary_backend=RerankerBackend(reranker_config.get("backend", "sentence_transformer")),
            device=reranker_config.get("device"),
            sentence_transformer_model=str(reranker_config.get("model", "BAAI/bge-reranker-large")),
            flashrank_model=str(flashrank_config.get("model", "ms-marco-MiniLM-L-12-v2")),
            flashrank_cache_dir=str(flashrank_config.get("cache_dir", "./data/flashrank")),
            preload_timeout_seconds=int(reranker_config.get("preload_timeout_seconds", 60)),
            auto_fallback_on_error=bool(reranker_config.get("auto_fallback_on_error", True)),
        )

        logger.info(
            "Reranker manager config loaded",
            primary_backend=manager_config.primary_backend.value,
        )

        return RerankerManager(manager_config)

    except Exception as exc:
        logger.error(
            "Failed to load reranker config; using defaults",
            config_path=config_path,
            error=str(exc),
            exc_info=True,
        )
        return RerankerManager(RerankerManagerConfig())


def preload_reranker_with_fallback(reranker_manager: RerankerManager) -> None:
    """Pre-load reranker backends with error handling.

    Args:
        reranker_manager: RerankerManager instance to preload.
    """
    try:
        logger.info("Starting reranker backend preload...")
        health_reports = reranker_manager.preload_backends()

        for backend, report in health_reports.items():
            logger.info(
                f"Backend {backend.value}: {report.status.value} - {report.message}"
            )

        active_backend = reranker_manager.active_backend
        if active_backend:
            logger.info(f"Active backend: {active_backend.value}")
        else:
            logger.warning("No reranker backend available; will use heuristic on first query")

    except Exception as exc:
        logger.error(
            "Reranker preload failed",
            error=str(exc),
            exc_info=True,
        )
