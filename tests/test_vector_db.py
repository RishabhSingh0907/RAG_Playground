"""Tests for vector database and embedding components."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.embedding.embedding_manager import EmbeddingManager, OllamaEmbeddingProvider
from src.vector_db.faiss_store import FAISSVectorStore
from src.vector_db.chroma_store import ChromaVectorStore
from src.vector_db.factory import VectorDatabaseFactory
from src.ingestion.models import (
    VectorDatabaseEmbeddingConfig,
    EmbeddingModel,
    FAISSVectorDBConfig,
    ChromaVectorDBConfig,
    VectorDBConfig,
    VectorDBType,
)


class TestEmbeddingManager:
    """Tests for embedding manager and providers."""

    @pytest.fixture
    def embedding_config(self):
        """Create test embedding configuration."""
        return VectorDatabaseEmbeddingConfig(
            provider="ollama",
            model=EmbeddingModel.OLLAMA_NOMIC_EMBED,
            base_url="http://localhost:11434",
            embed_batch_size=5,
        )

    def test_embedding_manager_initialization(self, embedding_config):
        """Test that embedding manager initializes correctly."""
        try:
            manager = EmbeddingManager(embedding_config)
            assert manager.config == embedding_config
            assert isinstance(manager.provider, OllamaEmbeddingProvider)
        except ValueError as e:
            pytest.skip(f"Ollama not available: {e}")

    def test_embedding_dimension(self, embedding_config):
        """Test getting embedding dimension."""
        try:
            manager = EmbeddingManager(embedding_config)
            dimension = manager.get_embedding_dimension()
            assert isinstance(dimension, int)
            assert dimension > 0
        except ValueError as e:
            pytest.skip(f"Ollama not available: {e}")

    def test_embed_single_text(self, embedding_config):
        """Test embedding a single text."""
        try:
            manager = EmbeddingManager(embedding_config)
            embedding = manager.embed_single("test text")
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, float) for x in embedding)
        except ValueError as e:
            pytest.skip(f"Ollama not available: {e}")

    def test_embed_multiple_texts(self, embedding_config):
        """Test embedding multiple texts."""
        try:
            manager = EmbeddingManager(embedding_config)
            texts = ["text1", "text2", "text3"]
            embeddings = manager.embed_texts(texts)
            assert len(embeddings) == len(texts)
            assert all(isinstance(e, list) for e in embeddings)
            assert all(len(e) > 0 for e in embeddings)
        except ValueError as e:
            pytest.skip(f"Ollama not available: {e}")

    def test_embed_empty_list(self, embedding_config):
        """Test embedding an empty list."""
        try:
            manager = EmbeddingManager(embedding_config)
            embeddings = manager.embed_texts([])
            assert embeddings == []
        except ValueError as e:
            pytest.skip(f"Ollama not available: {e}")


class TestFAISSVectorStore:
    """Tests for FAISS vector store."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def faiss_config(self, temp_dir):
        """Create test FAISS configuration."""
        return FAISSVectorDBConfig(
            index_type="flat",
            distance_metric="l2",
            persist_path=str(Path(temp_dir) / "faiss_index"),
        )

    @pytest.fixture
    def faiss_store(self, faiss_config):
        """Create FAISS vector store instance."""
        return FAISSVectorStore(faiss_config)

    def test_faiss_initialization(self, faiss_store):
        """Test FAISS store initialization."""
        assert faiss_store.index is None
        assert faiss_store.get_vector_count() == 0

    def test_faiss_add_vectors(self, faiss_store):
        """Test adding vectors to FAISS."""
        chunk_ids = ["chunk1", "chunk2", "chunk3"]
        vectors = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        metadata = [
            {"doc": "doc1"},
            {"doc": "doc2"},
            {"doc": "doc3"},
        ]

        faiss_store.add_vectors(chunk_ids, vectors, metadata)
        assert faiss_store.get_vector_count() == 3

    def test_faiss_search(self, faiss_store):
        """Test searching in FAISS."""
        chunk_ids = ["chunk1", "chunk2", "chunk3"]
        vectors = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        metadata = [
            {"doc": "doc1"},
            {"doc": "doc2"},
            {"doc": "doc3"},
        ]

        faiss_store.add_vectors(chunk_ids, vectors, metadata)

        query = [1.1, 2.1, 3.1]
        results = faiss_store.search(query, k=2)

        assert len(results) <= 2
        assert all(hasattr(r, "chunk_id") for r in results)
        assert all(hasattr(r, "distance") for r in results)

    def test_faiss_persistence(self, faiss_store):
        """Test FAISS index persistence."""
        chunk_ids = ["chunk1"]
        vectors = [[1.0, 2.0, 3.0]]
        metadata = [{"doc": "doc1"}]

        faiss_store.add_vectors(chunk_ids, vectors, metadata)
        faiss_store.persist()

        # Create new store and load
        faiss_store2 = FAISSVectorStore(faiss_store.config)
        assert faiss_store2.get_vector_count() == 1

    def test_faiss_delete_vectors(self, faiss_store):
        """Test deleting vectors from FAISS."""
        chunk_ids = ["chunk1", "chunk2", "chunk3"]
        vectors = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        metadata = [{"doc": f"doc{i}"} for i in range(3)]

        faiss_store.add_vectors(chunk_ids, vectors, metadata)
        assert faiss_store.get_vector_count() == 3

        faiss_store.delete_vectors(["chunk1"])
        assert faiss_store.get_vector_count() == 2

    def test_faiss_clear(self, faiss_store):
        """Test clearing FAISS index."""
        chunk_ids = ["chunk1", "chunk2"]
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        metadata = [{"doc": "doc1"}, {"doc": "doc2"}]

        faiss_store.add_vectors(chunk_ids, vectors, metadata)
        assert faiss_store.get_vector_count() == 2

        faiss_store.clear()
        assert faiss_store.get_vector_count() == 0


class TestChromaVectorStore:
    """Tests for ChromaDB vector store."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def chroma_config(self, temp_dir):
        """Create test ChromaDB configuration."""
        return ChromaVectorDBConfig(
            collection_name="test_collection",
            persist_directory=str(temp_dir),
        )

    @pytest.fixture
    def chroma_store(self, chroma_config):
        """Create ChromaDB vector store instance."""
        try:
            return ChromaVectorStore(chroma_config)
        except RuntimeError as e:
            pytest.skip(f"ChromaDB not available: {e}")

    def test_chroma_initialization(self, chroma_store):
        """Test ChromaDB store initialization."""
        assert chroma_store.get_vector_count() == 0

    def test_chroma_add_vectors(self, chroma_store):
        """Test adding vectors to ChromaDB."""
        chunk_ids = ["chunk1", "chunk2", "chunk3"]
        vectors = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        metadata = [
            {"doc": "doc1"},
            {"doc": "doc2"},
            {"doc": "doc3"},
        ]

        chroma_store.add_vectors(chunk_ids, vectors, metadata)
        assert chroma_store.get_vector_count() == 3

    def test_chroma_search(self, chroma_store):
        """Test searching in ChromaDB."""
        chunk_ids = ["chunk1", "chunk2", "chunk3"]
        vectors = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        metadata = [
            {"doc": "doc1"},
            {"doc": "doc2"},
            {"doc": "doc3"},
        ]

        chroma_store.add_vectors(chunk_ids, vectors, metadata)

        query = [1.1, 2.1, 3.1]
        results = chroma_store.search(query, k=2)

        assert len(results) <= 2
        assert all(hasattr(r, "chunk_id") for r in results)
        assert all(hasattr(r, "distance") for r in results)

    def test_chroma_delete_vectors(self, chroma_store):
        """Test deleting vectors from ChromaDB."""
        chunk_ids = ["chunk1", "chunk2", "chunk3"]
        vectors = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        metadata = [{"doc": f"doc{i}"} for i in range(3)]

        chroma_store.add_vectors(chunk_ids, vectors, metadata)
        assert chroma_store.get_vector_count() == 3

        chroma_store.delete_vectors(["chunk1"])
        assert chroma_store.get_vector_count() == 2

    def test_chroma_clear(self, chroma_store):
        """Test clearing ChromaDB collection."""
        chunk_ids = ["chunk1", "chunk2"]
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        metadata = [{"doc": "doc1"}, {"doc": "doc2"}]

        chroma_store.add_vectors(chunk_ids, vectors, metadata)
        assert chroma_store.get_vector_count() == 2

        chroma_store.clear()
        assert chroma_store.get_vector_count() == 0


class TestVectorDatabaseFactory:
    """Tests for vector database factory."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_factory_create_faiss(self, temp_dir):
        """Test factory creates FAISS store."""
        config = VectorDBConfig(
            enabled=True,
            type=VectorDBType.FAISS,
            faiss=FAISSVectorDBConfig(
                persist_path=str(Path(temp_dir) / "faiss_index")
            ),
        )
        store = VectorDatabaseFactory.create(config)
        assert isinstance(store, FAISSVectorStore)

    def test_factory_create_chroma(self, temp_dir):
        """Test factory creates ChromaDB store."""
        try:
            config = VectorDBConfig(
                enabled=True,
                type=VectorDBType.CHROMA,
                chroma=ChromaVectorDBConfig(persist_directory=str(temp_dir)),
            )
            store = VectorDatabaseFactory.create(config)
            assert isinstance(store, ChromaVectorStore)
        except RuntimeError as e:
            pytest.skip(f"ChromaDB not available: {e}")

    def test_factory_disabled_raises_error(self, temp_dir):
        """Test factory raises error when disabled."""
        config = VectorDBConfig(
            enabled=False,
            type=VectorDBType.FAISS,
            faiss=FAISSVectorDBConfig(
                persist_path=str(Path(temp_dir) / "faiss_index")
            ),
        )
        with pytest.raises(RuntimeError):
            VectorDatabaseFactory.create(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
