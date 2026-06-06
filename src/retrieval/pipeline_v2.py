"""V2 retrieval pipeline with pluggable reranker backends.

This module preserves the current retrieval behavior while adding configurable
reranker implementations:
- sentence_transformer (CrossEncoder)
- flashrank (ONNX-based)
- heuristic (stable Python fallback)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.retrieval.pipeline import (
    CrossEncoderReranker as HeuristicReranker,
    RetrievalPipeline,
    RetrievalReranker,
    RetrievalSettings,
    RetrievedChunk,
    SearchMode,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class RerankerBackend(str, Enum):
    """Supported reranker backend options for V2."""

    HEURISTIC = "heuristic"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    FLASHRANK = "flashrank"


@dataclass
class RetrievalSettingsV2(RetrievalSettings):
    """V2 retrieval settings with explicit reranker backend selection."""

    reranker_backend: RerankerBackend = RerankerBackend.HEURISTIC
    flashrank_model: str = "ms-marco-MiniLM-L-12-v2"
    flashrank_cache_dir: str = "./data/flashrank"

    @classmethod
    def from_mapping(cls, config: Dict[str, Any]) -> "RetrievalSettingsV2":
        retrieval_config = config.get("retrieval", config)
        reranker_config = retrieval_config.get("reranker", {})
        hybrid_config = retrieval_config.get("hybrid", {})
        flashrank_config = reranker_config.get("flashrank", {})

        backend_value = reranker_config.get(
            "backend",
            retrieval_config.get("reranker_backend", RerankerBackend.HEURISTIC.value),
        )

        return cls(
            search_mode=SearchMode(retrieval_config.get("search_mode", "dense")),
            top_k=int(retrieval_config.get("top_k", 10)),
            semantic_candidate_k=int(retrieval_config.get("semantic_candidate_k", 25)),
            lexical_candidate_k=int(retrieval_config.get("lexical_candidate_k", 25)),
            fusion_k=int(hybrid_config.get("fusion_k", retrieval_config.get("fusion_k", 60))),
            reranker_enabled=bool(reranker_config.get("enabled", retrieval_config.get("reranker_enabled", True))),
            reranker_model=str(
                reranker_config.get(
                    "model",
                    retrieval_config.get("reranker_model", "BAAI/bge-reranker-large"),
                )
            ),
            reranker_device=reranker_config.get("device", retrieval_config.get("reranker_device")),
            session_only=bool(retrieval_config.get("session_only", True)),
            reranker_backend=RerankerBackend(str(backend_value)),
            flashrank_model=str(
                flashrank_config.get("model", retrieval_config.get("flashrank_model", "ms-marco-MiniLM-L-12-v2"))
            ),
            flashrank_cache_dir=str(
                flashrank_config.get("cache_dir", retrieval_config.get("flashrank_cache_dir", "./data/flashrank"))
            ),
        )


def load_retrieval_settings_v2(config_path: str) -> RetrievalSettingsV2:
    """Load V2 retrieval settings from YAML, or fall back to defaults."""

    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning("Retrieval config not found; using V2 defaults", config_path=config_path)
        return RetrievalSettingsV2()

    try:
        with open(config_file, "r", encoding="utf-8") as config_handle:
            config_data = yaml.safe_load(config_handle) or {}
        return RetrievalSettingsV2.from_mapping(config_data)
    except Exception as exc:
        logger.warning(
            "Failed to load V2 retrieval config; using defaults",
            config_path=config_path,
            error=str(exc),
        )
        return RetrievalSettingsV2()


class SentenceTransformerRerankerV2:
    """SentenceTransformer CrossEncoder-based reranker for V2."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_error: Optional[str] = None

        try:
            import torch  # type: ignore[import-not-found]
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

            selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = CrossEncoder(model_name, device=selected_device)
            self.device = selected_device
            logger.info(
                "SentenceTransformer reranker initialized",
                model_name=model_name,
                device=selected_device,
            )
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning(
                "SentenceTransformer reranker unavailable",
                model_name=model_name,
                error=str(exc),
            )

    @property
    def available(self) -> bool:
        return self.model is not None

    def rerank(self, query: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not self.available or len(candidates) <= 1:
            return candidates

        try:
            pairs = [(query, candidate.content) for candidate in candidates]
            scores = self.model.predict(pairs)

            reranked: List[RetrievedChunk] = []
            for candidate, score in zip(candidates, scores):
                reranked.append(
                    RetrievedChunk(
                        chunk_id=candidate.chunk_id,
                        document_id=candidate.document_id,
                        content=candidate.content,
                        score=float(score),
                        rank=candidate.rank,
                        source=f"{candidate.source}:reranked",
                        metadata=dict(candidate.metadata),
                        semantic_score=candidate.semantic_score,
                        lexical_score=candidate.lexical_score,
                        reranker_score=float(score),
                    )
                )

            reranked.sort(key=lambda item: item.score, reverse=True)
            return reranked
        except Exception as exc:
            logger.warning(
                "SentenceTransformer reranking failed; returning pre-reranked order",
                model_name=self.model_name,
                error=str(exc),
            )
            return candidates


class FlashRankRerankerV2:
    """FlashRank-based reranker for CPU-friendly V2 deployment."""

    def __init__(self, model_name: str, cache_dir: str = "./data/flashrank"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.ranker = None
        self._request_cls = None
        self._load_error: Optional[str] = None

        try:
            from flashrank import Ranker, RerankRequest  # type: ignore[import-not-found]

            self.ranker = Ranker(model_name=model_name, cache_dir=cache_dir)
            self._request_cls = RerankRequest
            logger.info(
                "FlashRank reranker initialized",
                model_name=model_name,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning(
                "FlashRank reranker unavailable",
                model_name=model_name,
                error=str(exc),
            )

    @property
    def available(self) -> bool:
        return self.ranker is not None and self._request_cls is not None

    def rerank(self, query: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not self.available or len(candidates) <= 1:
            return candidates

        try:
            passages = []
            candidate_by_id: Dict[str, RetrievedChunk] = {}
            for index, candidate in enumerate(candidates):
                passage_id = candidate.chunk_id or f"candidate-{index}"
                candidate_by_id[passage_id] = candidate
                passages.append({"id": passage_id, "text": candidate.content, "meta": dict(candidate.metadata)})

            request = self._request_cls(query=query, passages=passages)
            ranked = self.ranker.rerank(request)

            reranked: List[RetrievedChunk] = []
            for item in ranked:
                passage_id = str(item.get("id"))
                base_candidate = candidate_by_id.get(passage_id)
                if base_candidate is None:
                    continue

                score = float(item.get("score", 0.0))
                reranked.append(
                    RetrievedChunk(
                        chunk_id=base_candidate.chunk_id,
                        document_id=base_candidate.document_id,
                        content=base_candidate.content,
                        score=score,
                        rank=base_candidate.rank,
                        source=f"{base_candidate.source}:reranked",
                        metadata=dict(base_candidate.metadata),
                        semantic_score=base_candidate.semantic_score,
                        lexical_score=base_candidate.lexical_score,
                        reranker_score=score,
                    )
                )

            if not reranked:
                return candidates

            reranked.sort(key=lambda item: item.score, reverse=True)
            return reranked
        except Exception as exc:
            logger.warning(
                "FlashRank reranking failed; returning pre-reranked order",
                model_name=self.model_name,
                error=str(exc),
            )
            return candidates


class RetrievalPipelineV2(RetrievalPipeline):
    """V2 retrieval pipeline with backend-selectable reranking.

    Can use an optional RerankerManager for pre-loaded and health-checked backends,
    or fall back to lazy-loading if not provided.
    """

    def __init__(
        self,
        manager: Any,
        settings: Optional[RetrievalSettingsV2] = None,
        reranker: Optional[RetrievalReranker] = None,
        reranker_manager: Optional[Any] = None,  # RerankerManager instance
    ):
        super().__init__(manager=manager, settings=settings or RetrievalSettingsV2(), reranker=reranker)
        self.settings: RetrievalSettingsV2
        self.reranker_manager = reranker_manager

    def _get_reranker(self) -> Optional[RetrievalReranker]:
        if self.reranker is not None:
            return self.reranker

        if not self.settings.reranker_enabled:
            return None

        # If reranker_manager is available, use it (pre-loaded and validated)
        if self.reranker_manager is not None:
            managed_reranker = self.reranker_manager.get_active_reranker()
            if managed_reranker is not None:
                self.reranker = managed_reranker
                return self.reranker

        # Fallback: lazy-load backend (legacy behavior)
        backend = self.settings.reranker_backend

        if backend == RerankerBackend.SENTENCE_TRANSFORMER:
            candidate_reranker = SentenceTransformerRerankerV2(
                model_name=self.settings.reranker_model,
                device=self.settings.reranker_device,
            )
            if candidate_reranker.available:
                self.reranker = candidate_reranker
                return self.reranker

            logger.warning("Falling back to heuristic reranker", backend=backend.value)

        elif backend == RerankerBackend.FLASHRANK:
            candidate_reranker = FlashRankRerankerV2(
                model_name=self.settings.flashrank_model,
                cache_dir=self.settings.flashrank_cache_dir,
            )
            if candidate_reranker.available:
                self.reranker = candidate_reranker
                return self.reranker

            logger.warning("Falling back to heuristic reranker", backend=backend.value)

        self.reranker = HeuristicReranker(
            model_name=self.settings.reranker_model,
            device=self.settings.reranker_device,
        )
        return self.reranker
