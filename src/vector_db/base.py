"""Abstract base class for vector database implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Result of a vector search query."""
    chunk_id: str
    distance: float
    metadata: Dict[str, Any]


class VectorDatabaseManager(ABC):
    """
    Abstract base class for vector database implementations.

    Defines the interface that all vector database implementations must follow.
    """

    @abstractmethod
    def add_vectors(
        self,
        chunk_ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Add vectors to the database.

        Args:
            chunk_ids: List of unique chunk identifiers
            vectors: List of embedding vectors (each is a list of floats)
            metadata: List of metadata dictionaries corresponding to vectors

        Raises:
            ValueError: If lengths don't match or vectors are invalid
            RuntimeError: If database operation fails
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for nearest neighbors.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters (implementation-specific)

        Returns:
            List of SearchResult objects sorted by distance (closest first)

        Raises:
            ValueError: If query vector has wrong dimension
            RuntimeError: If search operation fails
        """
        pass

    @abstractmethod
    def delete_vectors(self, chunk_ids: List[str]) -> None:
        """
        Delete vectors from the database.

        Args:
            chunk_ids: List of chunk IDs to delete

        Raises:
            RuntimeError: If deletion fails
        """
        pass

    @abstractmethod
    def get_vector_count(self) -> int:
        """Get total number of vectors in database."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of stored vectors."""
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist database to disk."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors from the database."""
        pass
