"""
Ingestion module for the RAG platform.

Provides document parsing, chunking, and orchestration capabilities
for multimodal data ingestion.
"""

from src.ingestion.models import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    IngestionConfig,
    IngestionResult,
    ChunkingFramework,
    LangChainChunkingMethod,
    LlamaIndexChunkingMethod,
    VectorDBType,
    VectorDBConfig,
    FAISSVectorDBConfig,
    ChromaVectorDBConfig,
    VectorDatabaseEmbeddingConfig,
    EmbeddingModel,
)
from src.ingestion.parsers import ParserRegistry, BaseParser

__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    "IngestionConfig",
    "IngestionResult",
    "ChunkingFramework",
    "LangChainChunkingMethod",
    "LlamaIndexChunkingMethod",
    "VectorDBType",
    "VectorDBConfig",
    "FAISSVectorDBConfig",
    "ChromaVectorDBConfig",
    "VectorDatabaseEmbeddingConfig",
    "EmbeddingModel",
    "ParserRegistry",
    "BaseParser",
    "IngestionManager",
]


def __getattr__(name):
    """Lazily expose orchestration classes without creating import cycles."""
    if name == "IngestionManager":
        from src.ingestion.ingestion_manager import IngestionManager

        return IngestionManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
