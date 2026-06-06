"""ChromaDB vector database implementation."""

from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import chromadb
except ImportError:
    raise ImportError("chromadb must be installed: pip install chromadb")

from src.vector_db.base import VectorDatabaseManager, SearchResult
from src.ingestion.models import ChromaVectorDBConfig
from src.utils.logger import get_logger


logger = get_logger(__name__)


class ChromaVectorStore(VectorDatabaseManager):
    """
    ChromaDB vector database implementation.

    ChromaDB is an open-source embedding database with built-in metadata filtering.
    Supports persistent and ephemeral modes.
    """

    def __init__(self, config: ChromaVectorDBConfig):
        """
        Initialize ChromaDB vector store using new PersistentClient API.

        Args:
            config: ChromaDB configuration

        Raises:
            RuntimeError: If ChromaDB initialization fails
        """
        logger.info(
            "Initializing ChromaVectorStore",
            collection=config.collection_name,
            persist_directory=config.persist_directory,
        )

        self.config = config
        self.persist_directory = Path(config.persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client using new PersistentClient API
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )
            logger.info("ChromaDB PersistentClient initialized")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            raise RuntimeError(f"ChromaDB initialization failed: {e}")

        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=config.collection_name,
                metadata={"hnsw:space": "l2"},  # L2 distance metric
            )
            logger.info(
                f"Collection '{config.collection_name}' initialized",
                collection_name=config.collection_name,
            )

        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}", exc_info=True)
            raise RuntimeError(f"Collection initialization failed: {e}")

        self._embedding_dimension: Optional[int] = None

    def add_vectors(
        self,
        chunk_ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Add vectors to ChromaDB collection."""
        if not vectors:
            logger.warning("Empty vectors list provided")
            return

        if len(chunk_ids) != len(vectors) or len(vectors) != len(metadata):
            raise ValueError(
                f"Length mismatch: chunk_ids={len(chunk_ids)}, "
                f"vectors={len(vectors)}, metadata={len(metadata)}"
            )

        try:
            # ChromaDB expects documents as text (for display purposes)
            # We use chunk_id as document, but could also use actual content
            documents = [f"Chunk: {cid}" for cid in chunk_ids]

            self.collection.add(
                ids=chunk_ids,
                embeddings=vectors,
                metadatas=metadata,
                documents=documents,
            )

            # Set embedding dimension from first vector
            if self._embedding_dimension is None and vectors:
                self._embedding_dimension = len(vectors[0])

            logger.info(
                f"Added {len(vectors)} vectors to ChromaDB",
                collection=self.config.collection_name,
                total=self.collection.count(),
            )

        except Exception as e:
            logger.error(
                f"Failed to add vectors to ChromaDB: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to add vectors: {e}")

    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for nearest neighbors in ChromaDB collection."""
        if self.collection.count() == 0:
            logger.warning("Collection is empty, returning empty results")
            return []

        try:
            # ChromaDB query
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                where=filters if filters else None,  # Optional metadata filter
            )

            search_results = []

            if results and results["ids"] and len(results["ids"]) > 0:
                chunk_ids = results["ids"][0]  # First query's results
                distances = results["distances"][0]
                metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(chunk_ids)

                for chunk_id, distance, metadata in zip(chunk_ids, distances, metadatas):
                    search_results.append(
                        SearchResult(
                            chunk_id=chunk_id,
                            distance=float(distance),
                            metadata=metadata or {},
                        )
                    )

            logger.info(
                f"Search returned {len(search_results)} results",
                collection=self.config.collection_name,
            )
            return search_results

        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}", exc_info=True)
            raise RuntimeError(f"Search failed: {e}")

    def delete_vectors(self, chunk_ids: List[str]) -> None:
        """Delete vectors from ChromaDB collection."""
        if not chunk_ids:
            return

        try:
            self.collection.delete(ids=chunk_ids)
            logger.info(
                f"Deleted {len(chunk_ids)} vectors from ChromaDB",
                collection=self.config.collection_name,
            )

        except Exception as e:
            logger.error(
                f"Failed to delete vectors from ChromaDB: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Deletion failed: {e}")

    def get_vector_count(self) -> int:
        """Get total number of vectors in collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get vector count: {e}")
            return 0

    def get_embedding_dimension(self) -> int:
        """Get dimensionality of stored vectors."""
        if self._embedding_dimension is not None:
            return self._embedding_dimension

        # Get first vector's dimension
        try:
            if self.collection.count() > 0:
                data = self.collection.get(limit=1)
                if data["embeddings"]:
                    self._embedding_dimension = len(data["embeddings"][0])
                    return self._embedding_dimension
        except Exception as e:
            logger.error(f"Failed to determine embedding dimension: {e}")

        raise RuntimeError("Cannot determine embedding dimension (collection may be empty)")

    def persist(self) -> None:
        """
        Persist ChromaDB collection to disk.
        
        Note: With PersistentClient, persistence is automatic.
        This method logs the persistence status.
        """
        try:
            logger.info(
                "ChromaDB collection persisted to disk (automatic with PersistentClient)",
                path=str(self.persist_directory),
            )

        except Exception as e:
            logger.error(f"Failed to verify ChromaDB persistence: {e}", exc_info=True)
            raise RuntimeError(f"Persistence failed: {e}")

    def clear(self) -> None:
        """Clear all vectors from collection."""
        try:
            # Delete collection and recreate
            self.client.delete_collection(name=self.config.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "l2"},
            )
            logger.info(f"Cleared ChromaDB collection: {self.config.collection_name}")

        except Exception as e:
            logger.error(f"Failed to clear ChromaDB collection: {e}", exc_info=True)
            raise RuntimeError(f"Clear failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get ChromaDB store status for diagnostics."""
        try:
            collection_count = self.collection.count() if self.collection else 0
            embedding_dim = None
            if collection_count > 0:
                try:
                    embedding_dim = self.get_embedding_dimension()
                except:
                    embedding_dim = None
            
            return {
                "collection_initialized": self.collection is not None,
                "total_vectors": collection_count,
                "embedding_dimension": embedding_dim,
                "collection_name": self.config.collection_name,
                "persist_directory": str(self.persist_directory),
                "persist_directory_exists": self.persist_directory.exists(),
                "client_type": "PersistentClient",
            }
        except Exception as e:
            logger.warning(f"Failed to get ChromaDB status: {e}")
            return {
                "collection_initialized": False,
                "total_vectors": 0,
                "embedding_dimension": None,
                "collection_name": self.config.collection_name,
                "persist_directory": str(self.persist_directory),
                "persist_directory_exists": self.persist_directory.exists(),
                "client_type": "PersistentClient",
                "error": str(e),
            }
