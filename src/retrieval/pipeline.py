"""Session-scoped retrieval pipeline for semantic, lexical, and hybrid search."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import math
import re
from typing import Any, Dict, List, Optional, Protocol

import yaml

from src.ingestion.models import DocumentChunk, VectorDBType
from src.utils.logger import get_logger


logger = get_logger(__name__)


class SearchMode(str, Enum):
    """Supported retrieval modes."""

    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


@dataclass
class RetrievalSettings:
    """Configuration for the retrieval pipeline."""

    search_mode: SearchMode = SearchMode.DENSE
    top_k: int = 10
    semantic_candidate_k: int = 25
    lexical_candidate_k: int = 25
    fusion_k: int = 60
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_device: Optional[str] = None
    session_only: bool = True

    @classmethod
    def from_mapping(cls, config: Dict[str, Any]) -> "RetrievalSettings":
        """Create settings from a mapping or a retrieval config file."""

        retrieval_config = config.get("retrieval", config)
        reranker_config = retrieval_config.get("reranker", {})
        hybrid_config = retrieval_config.get("hybrid", {})

        return cls(
            search_mode=SearchMode(retrieval_config.get("search_mode", "dense")),
            top_k=int(retrieval_config.get("top_k", 10)),
            semantic_candidate_k=int(retrieval_config.get("semantic_candidate_k", 25)),
            lexical_candidate_k=int(retrieval_config.get("lexical_candidate_k", 25)),
            fusion_k=int(hybrid_config.get("fusion_k", retrieval_config.get("fusion_k", 60))),
            reranker_enabled=bool(reranker_config.get("enabled", retrieval_config.get("reranker_enabled", True))),
            reranker_model=str(reranker_config.get("model", retrieval_config.get("reranker_model", "BAAI/bge-reranker-large"))),
            reranker_device=reranker_config.get("device", retrieval_config.get("reranker_device")),
            session_only=bool(retrieval_config.get("session_only", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize settings for UI state and debugging."""

        return {
            "search_mode": self.search_mode.value,
            "top_k": self.top_k,
            "semantic_candidate_k": self.semantic_candidate_k,
            "lexical_candidate_k": self.lexical_candidate_k,
            "fusion_k": self.fusion_k,
            "reranker_enabled": self.reranker_enabled,
            "reranker_model": self.reranker_model,
            "reranker_device": self.reranker_device,
            "session_only": self.session_only,
        }


@dataclass
class RetrievedChunk:
    """A chunk returned by the retrieval pipeline."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    semantic_score: Optional[float] = None
    lexical_score: Optional[float] = None
    reranker_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a flat dictionary for UI display."""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "rank": self.rank,
            "score": self.score,
            "source": self.source,
            "semantic_score": self.semantic_score,
            "lexical_score": self.lexical_score,
            "reranker_score": self.reranker_score,
            "preview": self.content[:220] + ("..." if len(self.content) > 220 else ""),
            "metadata": self.metadata,
        }


class RetrievalReranker(Protocol):
    """Protocol for reranking implementations."""

    def rerank(self, query: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Rerank candidate chunks for a query."""


class CrossEncoderReranker:
    """Crash-safe reranker implementation.

    The original model-based reranker could take down the process in some
    Windows/CPU setups. This implementation preserves the interface but uses a
    stable Python scoring heuristic so the app stays responsive.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or "cpu"
        logger.info(
            "Reranker initialized with safe Python scoring",
            model_name=model_name,
            device=self.device,
        )

    @property
    def available(self) -> bool:
        """Return whether the reranker is available."""

        return True

    def rerank(self, query: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Rerank candidates with a stable Python scoring heuristic."""

        if len(candidates) <= 1:
            return candidates

        try:
            query_tokens = set(self._tokenize(query))
            if not query_tokens:
                return candidates

            scored_candidates: List[RetrievedChunk] = []
            for candidate in candidates:
                candidate_tokens = set(self._tokenize(candidate.content))
                overlap = len(query_tokens & candidate_tokens)
                coverage = overlap / max(len(query_tokens), 1)
                density = overlap / max(len(candidate_tokens), 1)
                semantic_bonus = candidate.semantic_score or candidate.score or 0.0
                score = (coverage * 0.6) + (density * 0.3) + (semantic_bonus * 0.1)

                reranked_candidate = RetrievedChunk(
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
                scored_candidates.append(reranked_candidate)

            scored_candidates.sort(key=lambda item: item.score, reverse=True)
            return scored_candidates

        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Reranking failed; returning pre-reranked order",
                error=str(exc),
                model_name=self.model_name,
            )
            return candidates

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())


def load_retrieval_settings(config_path: str) -> RetrievalSettings:
    """Load retrieval settings from YAML, or fall back to defaults."""

    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning("Retrieval config not found; using defaults", config_path=config_path)
        return RetrievalSettings()

    try:
        with open(config_file, "r", encoding="utf-8") as config_handle:
            config_data = yaml.safe_load(config_handle) or {}
        return RetrievalSettings.from_mapping(config_data)
    except Exception as exc:
        logger.warning(
            "Failed to load retrieval config; using defaults",
            config_path=config_path,
            error=str(exc),
        )
        return RetrievalSettings()


class RetrievalPipeline:
    """Session-scoped retrieval orchestration."""

    def __init__(
        self,
        manager: Any,
        settings: Optional[RetrievalSettings] = None,
        reranker: Optional[RetrievalReranker] = None,
    ):
        self.manager = manager
        self.settings = settings or RetrievalSettings()
        self.reranker = reranker
        self._reranker_model = self.settings.reranker_model
        self._reranker_device = self.settings.reranker_device

    def _get_reranker(self) -> Optional[RetrievalReranker]:
        """Lazily construct the reranker only when it is actually needed."""

        if self.reranker is None and self.settings.reranker_enabled:
            self.reranker = CrossEncoderReranker(
                model_name=self._reranker_model,
                device=self._reranker_device,
            )

        return self.reranker

    def search(self, query: str) -> List[RetrievedChunk]:
        """Run the configured retrieval pipeline for a query."""

        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        if self.settings.search_mode == SearchMode.LEXICAL:
            candidates = self._lexical_search(cleaned_query)
        elif self.settings.search_mode == SearchMode.HYBRID:
            candidates = self._hybrid_search(cleaned_query)
        else:
            candidates = self._dense_search(cleaned_query)

        if self.settings.reranker_enabled:
            reranker = self._get_reranker()
            if reranker is not None:
                candidates = reranker.rerank(cleaned_query, candidates)

        for position, candidate in enumerate(candidates[: self.settings.top_k], start=1):
            candidate.rank = position

        return candidates[: self.settings.top_k]

    def clear_session_data(self) -> Dict[str, Any]:
        """Clear SQLite chunks and vector embeddings for the active session."""

        cleared: Dict[str, Any] = {
            "storage_cleared": False,
            "vector_db_cleared": False,
            "chunks_removed": 0,
        }

        if hasattr(self.manager, "storage_manager") and self.manager.storage_manager is not None:
            cleared["chunks_removed"] = self.manager.storage_manager.clear_all_chunks()
            cleared["storage_cleared"] = True

        if hasattr(self.manager, "vector_db") and self.manager.vector_db is not None:
            self.manager.vector_db.clear()
            cleared["vector_db_cleared"] = True

        return cleared

    def _dense_search(self, query: str) -> List[RetrievedChunk]:
        if not hasattr(self.manager, "vector_db") or self.manager.vector_db is None:
            raise RuntimeError("Vector database is not initialized")
        if not hasattr(self.manager, "embedding_manager") or self.manager.embedding_manager is None:
            raise RuntimeError("Embedding manager is not initialized")

        query_vector = self.manager.embedding_manager.embed_single(query)
        search_results = self.manager.vector_db.search(query_vector, k=self.settings.semantic_candidate_k)
        return self._convert_vector_results(search_results)

    def _lexical_search(self, query: str) -> List[RetrievedChunk]:
        chunks = self._load_chunks()
        if not chunks:
            return []

        ranked_chunks = self._bm25_rank(query, chunks)
        return ranked_chunks[: self.settings.lexical_candidate_k]

    def _hybrid_search(self, query: str) -> List[RetrievedChunk]:
        dense_candidates = self._dense_search(query)
        lexical_candidates = self._lexical_search(query)

        fused_candidates = self._rrf_fuse(dense_candidates, lexical_candidates)
        fused_candidates.sort(key=lambda item: item.score, reverse=True)
        return fused_candidates[: max(self.settings.semantic_candidate_k, self.settings.lexical_candidate_k)]

    def _convert_vector_results(self, search_results: List[Any]) -> List[RetrievedChunk]:
        converted_results: List[RetrievedChunk] = []
        for position, search_result in enumerate(search_results, start=1):
            chunk = self._get_chunk(search_result.chunk_id)
            if chunk is None:
                continue

            semantic_score = self._distance_to_score(search_result.distance)
            metadata = dict(search_result.metadata or {})
            metadata.setdefault("retrieval_mode", "dense")

            converted_results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=semantic_score,
                    rank=position,
                    source="dense",
                    metadata=metadata,
                    semantic_score=semantic_score,
                )
            )

        return converted_results

    def _rrf_fuse(
        self,
        dense_candidates: List[RetrievedChunk],
        lexical_candidates: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        fused_by_chunk_id: Dict[str, RetrievedChunk] = {}
        score_by_chunk_id: Dict[str, float] = defaultdict(float)

        for rank, candidate in enumerate(dense_candidates, start=1):
            score_by_chunk_id[candidate.chunk_id] += 1.0 / (self.settings.fusion_k + rank)
            fused_by_chunk_id[candidate.chunk_id] = self._clone_candidate(candidate, source="hybrid")

        for rank, candidate in enumerate(lexical_candidates, start=1):
            score_by_chunk_id[candidate.chunk_id] += 1.0 / (self.settings.fusion_k + rank)
            if candidate.chunk_id in fused_by_chunk_id:
                fused_by_chunk_id[candidate.chunk_id].lexical_score = candidate.score
                fused_by_chunk_id[candidate.chunk_id].metadata.setdefault("lexical_rank", rank)
            else:
                fused_by_chunk_id[candidate.chunk_id] = self._clone_candidate(candidate, source="hybrid")
                fused_by_chunk_id[candidate.chunk_id].lexical_score = candidate.score

        fused_results: List[RetrievedChunk] = []
        for chunk_id, candidate in fused_by_chunk_id.items():
            candidate.score = score_by_chunk_id[chunk_id]
            candidate.metadata["fusion_score"] = candidate.score
            fused_results.append(candidate)

        fused_results.sort(key=lambda item: item.score, reverse=True)
        return fused_results

    def _clone_candidate(self, candidate: RetrievedChunk, source: Optional[str] = None) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            content=candidate.content,
            score=candidate.score,
            rank=candidate.rank,
            source=source or candidate.source,
            metadata=dict(candidate.metadata),
            semantic_score=candidate.semantic_score,
            lexical_score=candidate.lexical_score,
            reranker_score=candidate.reranker_score,
        )

    def _load_chunks(self) -> List[DocumentChunk]:
        total_chunks = 0
        if hasattr(self.manager, "get_stats"):
            try:
                total_chunks = int(self.manager.get_stats().get("total_chunks", 0))
            except Exception:
                total_chunks = 0

        if hasattr(self.manager, "get_all_chunks"):
            return self.manager.get_all_chunks(limit=max(total_chunks, 1_000))

        raise RuntimeError("Storage manager does not expose chunk retrieval")

    def _get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        if hasattr(self.manager, "storage_manager") and self.manager.storage_manager is not None:
            return self.manager.storage_manager.get_chunk(chunk_id)
        return None

    def _distance_to_score(self, distance: float) -> float:
        vector_db_type = None
        distance_metric = None

        if hasattr(self.manager, "config"):
            try:
                vector_db_config = self.manager.config.storage.vector_db
                vector_db_type = vector_db_config.type
                if vector_db_type == VectorDBType.FAISS:
                    distance_metric = vector_db_config.faiss.distance_metric
            except Exception:
                vector_db_type = None

        if vector_db_type == VectorDBType.FAISS and distance_metric == "inner_product":
            return float(distance)

        return 1.0 / (1.0 + max(float(distance), 0.0))

    def _bm25_rank(self, query: str, chunks: List[DocumentChunk]) -> List[RetrievedChunk]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        document_tokens = [self._tokenize(chunk.content) for chunk in chunks]
        scores = self._bm25_scores(query_tokens, document_tokens)

        ranked_candidates: List[RetrievedChunk] = []
        for chunk, score in zip(chunks, scores):
            if score <= 0:
                continue

            ranked_candidates.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=float(score),
                    rank=0,
                    source="lexical",
                    metadata=dict(chunk.metadata),
                    lexical_score=float(score),
                )
            )

        ranked_candidates.sort(key=lambda item: item.score, reverse=True)

        for position, candidate in enumerate(ranked_candidates, start=1):
            candidate.rank = position

        return ranked_candidates

    def _bm25_scores(
        self,
        query_tokens: List[str],
        document_tokens: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[float]:
        document_count = len(document_tokens)
        if document_count == 0:
            return []

        document_frequencies = Counter()
        term_frequencies = [Counter(tokens) for tokens in document_tokens]
        document_lengths = [len(tokens) for tokens in document_tokens]
        average_length = sum(document_lengths) / document_count if document_count else 0.0

        for tokens in document_tokens:
            for token in set(tokens):
                document_frequencies[token] += 1

        scores: List[float] = []
        for doc_index, doc_term_frequencies in enumerate(term_frequencies):
            score = 0.0
            document_length = document_lengths[doc_index]

            for token in query_tokens:
                term_frequency = doc_term_frequencies.get(token, 0)
                if term_frequency == 0:
                    continue

                document_frequency = document_frequencies.get(token, 0)
                idf = math.log(1.0 + ((document_count - document_frequency + 0.5) / (document_frequency + 0.5)))
                denominator = term_frequency + k1 * (1.0 - b + b * (document_length / average_length if average_length else 0.0))
                score += idf * (term_frequency * (k1 + 1.0)) / denominator

            scores.append(score)

        return scores

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())