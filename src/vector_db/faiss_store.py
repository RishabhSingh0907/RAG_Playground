"""FAISS vector database implementation."""

from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import json
import pickle
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu or faiss-gpu must be installed: pip install faiss-cpu")

from src.vector_db.base import VectorDatabaseManager, SearchResult
from src.ingestion.models import FAISSVectorDBConfig
from src.utils.logger import get_logger


logger = get_logger(__name__)


class FAISSVectorStore(VectorDatabaseManager):
    """
    FAISS-based vector database implementation.

    Uses Facebook AI Similarity Search for efficient vector indexing and retrieval.
    Supports multiple index types: flat (brute-force), ivfpq (product quantization), hnsw.
    """

    def __init__(self, config: FAISSVectorDBConfig):
        """
        Initialize FAISS vector store.

        Args:
            config: FAISS configuration

        Raises:
            ValueError: If configuration is invalid
        """
        logger.info(
            "Initializing FAISSVectorStore",
            index_type=config.index_type,
            distance_metric=config.distance_metric,
            persist_path=config.persist_path,
        )

        self.config = config
        self.persist_path = Path(config.persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        # Metadata storage (separate from FAISS index)
        self.metadata_path = self.persist_path.with_suffix(".metadata.json")
        self.id_map_path = self.persist_path.with_suffix(".id_map.pkl")

        self.index: Optional[faiss.Index] = None
        self.chunk_id_to_index: Dict[str, int] = {}  # chunk_id -> FAISS index position
        self.index_to_metadata: Dict[int, Dict[str, Any]] = {}

        self._embedding_dimension: Optional[int] = None
        self._total_vectors = 0

        # Try to load existing index
        if self.persist_path.exists():
            self._load_index()
        else:
            logger.info("No existing FAISS index found, will create on first add")

    def _create_index(self, dimension: int) -> faiss.Index:
        """Create FAISS index based on configuration."""
        if self.config.index_type.lower() == "flat":
            if self.config.distance_metric.lower() == "l2":
                index = faiss.IndexFlatL2(dimension)
            elif self.config.distance_metric.lower() == "inner_product":
                index = faiss.IndexFlatIP(dimension)
            else:
                raise ValueError(f"Unsupported distance metric: {self.config.distance_metric}")

        elif self.config.index_type.lower() == "ivfpq":
            # IVF (Inverted File) with Product Quantization
            nlist = max(1, int(np.sqrt(1000)))  # Number of clusters
            m = min(8, dimension // 2)  # Number of subquantizers
            nbits = 8

            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
            index.nprobe = 5  # Number of clusters to search

        elif self.config.index_type.lower() == "hnsw":
            # Hierarchical Navigable Small World
            index = faiss.IndexHNSWFlat(dimension, 32)  # M=32

        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")

        logger.info(f"Created FAISS index: {self.config.index_type}", dimension=dimension)
        return index

    def add_vectors(
        self,
        chunk_ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Add vectors to FAISS index."""
        if not vectors:
            logger.warning("Empty vectors list provided")
            return

        if len(chunk_ids) != len(vectors) or len(vectors) != len(metadata):
            raise ValueError(
                f"Length mismatch: chunk_ids={len(chunk_ids)}, "
                f"vectors={len(vectors)}, metadata={len(metadata)}"
            )

        # Convert to numpy array
        vectors_array = np.array(vectors, dtype="float32")

        if vectors_array.shape[0] == 0:
            return

        dimension = vectors_array.shape[1]

        # Initialize index if needed
        if self.index is None:
            self.index = self._create_index(dimension)
            self._embedding_dimension = dimension

        # Verify dimension matches
        if self.index.d != dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.index.d}, got {dimension}"
            )

        # Train index if needed
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(vectors_array)

        # Add vectors to index
        start_idx = self.index.ntotal
        self.index.add(vectors_array)

        # Store mappings
        for i, (chunk_id, meta) in enumerate(zip(chunk_ids, metadata)):
            faiss_idx = start_idx + i
            self.chunk_id_to_index[chunk_id] = faiss_idx
            self.index_to_metadata[faiss_idx] = meta

        self._total_vectors += len(vectors)
        logger.info(
            f"Added {len(vectors)} vectors to FAISS index",
            total_vectors=self.index.ntotal,
        )

        # Auto-persist after adding
        self.persist()

    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for nearest neighbors in FAISS index."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty, returning empty results")
            return []

        query_array = np.array([query_vector], dtype="float32")

        if query_array.shape[1] != self.index.d:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.index.d}, "
                f"got {query_array.shape[1]}"
            )

        # Search in index
        distances, indices = self.index.search(query_array, min(k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # Invalid result
                continue

            # Reverse lookup from FAISS index to chunk_id
            chunk_id = None
            for cid, fidx in self.chunk_id_to_index.items():
                if fidx == idx:
                    chunk_id = cid
                    break

            if chunk_id and idx in self.index_to_metadata:
                metadata = self.index_to_metadata[idx]

                # Apply filters if provided
                if filters and not self._matches_filters(metadata, filters):
                    continue

                results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        distance=float(dist),
                        metadata=metadata,
                    )
                )

        logger.info(f"Search returned {len(results)} results")
        return results

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches all filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def delete_vectors(self, chunk_ids: List[str]) -> None:
        """
        Delete vectors (FAISS doesn't support direct deletion).

        Instead, rebuild index without deleted vectors.
        """
        if not chunk_ids or self.index is None:
            return

        logger.warning("FAISS doesn't support direct deletion, rebuilding index...")

        # Get remaining vectors
        remaining_ids = [
            (chunk_id, idx, self.index_to_metadata[idx])
            for chunk_id, idx in self.chunk_id_to_index.items()
            if chunk_id not in chunk_ids
        ]

        if not remaining_ids:
            self.clear()
            return

        # Reconstruct vectors
        remaining_vectors = [
            self.index.reconstruct(idx) for _, idx, _ in remaining_ids
        ]
        remaining_chunk_ids = [cid for cid, _, _ in remaining_ids]
        remaining_metadata = [meta for _, _, meta in remaining_ids]

        # Clear and rebuild
        self.clear()
        self.add_vectors(remaining_chunk_ids, remaining_vectors, remaining_metadata)

        logger.info(f"Deleted {len(chunk_ids)} vectors from index")

    def get_vector_count(self) -> int:
        """Get total number of vectors in index."""
        return self.index.ntotal if self.index else 0

    def get_embedding_dimension(self) -> int:
        """Get dimensionality of stored vectors."""
        if self.index is None:
            raise RuntimeError("Index not initialized")
        return self.index.d

    def persist(self) -> None:
        """Save FAISS index and metadata to disk."""
        logger.debug(f"persist() called - index state: index is None: {self.index is None}")
        
        if self.index is None:
            logger.warning("No index to persist")
            return

        try:
            logger.debug(
                "Starting FAISS persistence",
                index_path=str(self.persist_path),
                metadata_path=str(self.metadata_path),
                id_map_path=str(self.id_map_path),
                total_vectors=self.index.ntotal,
            )
            
            # Save FAISS index
            logger.debug("Writing FAISS index file...")
            faiss.write_index(self.index, str(self.persist_path))
            logger.debug("FAISS index file written successfully")

            # Save metadata and ID mappings
            logger.debug(f"Writing metadata file with {len(self.index_to_metadata)} entries...")
            with open(self.metadata_path, "w") as f:
                json.dump(self.index_to_metadata, f, indent=2, default=str)
            logger.debug("Metadata file written successfully")

            logger.debug(f"Writing ID map file with {len(self.chunk_id_to_index)} entries...")
            with open(self.id_map_path, "wb") as f:
                pickle.dump(self.chunk_id_to_index, f)
            logger.debug("ID map file written successfully")

            logger.info("FAISS index persisted to disk", path=str(self.persist_path))

        except Exception as e:
            logger.error(
                f"Failed to persist FAISS index: {e}",
                persist_path=str(self.persist_path),
                exc_info=True
            )
            raise

    def _load_index(self) -> None:
        """Load FAISS index from disk."""
        try:
            self.index = faiss.read_index(str(self.persist_path))

            if self.metadata_path.exists():
                with open(self.metadata_path, "r") as f:
                    metadata_dict = json.load(f)
                    # Convert string keys back to int
                    self.index_to_metadata = {
                        int(k): v for k, v in metadata_dict.items()
                    }

            if self.id_map_path.exists():
                with open(self.id_map_path, "rb") as f:
                    self.chunk_id_to_index = pickle.load(f)

            self._total_vectors = self.index.ntotal
            logger.info(
                "Loaded FAISS index from disk",
                path=str(self.persist_path),
                vectors=self._total_vectors,
            )

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}", exc_info=True)
            raise

    def clear(self) -> None:
        """Clear all vectors from index."""
        if self.index is not None:
            # Create new empty index
            dimension = self.index.d
            self.index = self._create_index(dimension)

        self.chunk_id_to_index.clear()
        self.index_to_metadata.clear()
        self._total_vectors = 0

        logger.info("Cleared FAISS index")
    
    def get_status(self) -> Dict[str, Any]:
        """Get FAISS store status for diagnostics."""
        return {
            "index_initialized": self.index is not None,
            "total_vectors": self.get_vector_count(),
            "embedding_dimension": self.index.d if self.index else None,
            "persist_path": str(self.persist_path),
            "persist_path_exists": self.persist_path.exists(),
            "metadata_path_exists": self.metadata_path.exists(),
            "id_map_path_exists": self.id_map_path.exists(),
            "chunk_id_mappings": len(self.chunk_id_to_index),
            "metadata_entries": len(self.index_to_metadata),
        }
