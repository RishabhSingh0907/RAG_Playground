"""
Embedding module for vector database integration.

Provides unified interface for generating embeddings using various providers.
"""

from src.embedding.embedding_manager import EmbeddingManager

__all__ = ["EmbeddingManager"]
