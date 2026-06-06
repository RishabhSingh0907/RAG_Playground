"""Vector database module for storing and retrieving embeddings."""

from src.vector_db.base import VectorDatabaseManager, SearchResult
from src.vector_db.faiss_store import FAISSVectorStore
from src.vector_db.chroma_store import ChromaVectorStore
from src.vector_db.factory import VectorDatabaseFactory

__all__ = [
    "VectorDatabaseManager",
    "SearchResult",
    "FAISSVectorStore",
    "ChromaVectorStore",
    "VectorDatabaseFactory",
]
