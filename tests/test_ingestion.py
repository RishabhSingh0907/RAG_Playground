"""
Unit tests for the data ingestion component.

Tests cover:
- Data models validation (Pydantic)
- Parser registry and individual parsers
- SQLite storage operations
- Ingestion manager orchestration
- Configuration loading

Target: 90%+ test coverage
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
import sqlite3
from unittest.mock import patch
from types import SimpleNamespace

from src.ingestion.models import (
    IngestionConfig,
    Document,
    DocumentChunk,
    DocumentMetadata,
    IngestionResult,
    ChunkingFramework,
    LangChainChunkingMethod,
    LlamaIndexChunkingMethod,
    RecursiveCharacterConfig,
    ParsingConfig,
    VectorDBType,
)
from src.ingestion.parsers import (
    ParserRegistry,
    TextParser,
    DOCXParser,
    CSVParser,
)
from src.storage.sqlite_manager import SQLiteManager
from src.ingestion.ingestion_manager import IngestionManager
from src.utils.logger import StructuredLogger


class TestDataModels:
    """Test Pydantic data models."""
    
    def test_chunking_framework_enum(self):
        """Test ChunkingFramework enum values."""
        assert ChunkingFramework.LANGCHAIN.value == "langchain"
        assert ChunkingFramework.LLAMAINDEX.value == "llamaindex"
    
    def test_langchain_chunking_method_enum(self):
        """Test LangChainChunkingMethod enum values."""
        assert LangChainChunkingMethod.RECURSIVE_CHARACTER.value == "recursive_character"
        assert LangChainChunkingMethod.CHARACTER.value == "character"
        assert LangChainChunkingMethod.TOKEN.value == "token"
    
    def test_llamaindex_chunking_method_enum(self):
        """Test LlamaIndexChunkingMethod enum values."""
        assert LlamaIndexChunkingMethod.SENTENCE_SPLITTER.value == "sentence_splitter"
        assert LlamaIndexChunkingMethod.SEMANTIC_SPLITTER.value == "semantic_splitter"
        assert LlamaIndexChunkingMethod.SENTENCE_WINDOW.value == "sentence_window"
    
    def test_recursive_character_config_validation(self):
        """Test RecursiveCharacterConfig validation."""
        # Valid config
        config = RecursiveCharacterConfig(chunk_size=512, chunk_overlap=50)
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        
        # Invalid chunk_size (must be positive)
        with pytest.raises(ValueError):
            RecursiveCharacterConfig(chunk_size=-1)
        
        # Invalid chunk_overlap (must be non-negative)
        with pytest.raises(ValueError):
            RecursiveCharacterConfig(chunk_overlap=-1)
    
    def test_document_metadata_creation(self):
        """Test DocumentMetadata model."""
        metadata = DocumentMetadata(
            file_path="/path/to/file.txt",
            file_name="file.txt",
            file_format="txt",
            file_size=1024,
        )
        
        assert metadata.file_name == "file.txt"
        assert metadata.file_format == "txt"
        assert metadata.file_size == 1024
        assert isinstance(metadata.created_at, datetime)
    
    def test_document_creation(self):
        """Test Document model."""
        metadata = DocumentMetadata(
            file_path="/path/to/file.txt",
            file_name="file.txt",
            file_format="txt",
            file_size=1024,
        )
        
        doc = Document(
            document_id="doc-123",
            content="Sample content",
            metadata=metadata,
        )
        
        assert doc.document_id == "doc-123"
        assert doc.content == "Sample content"
        assert doc.chunk_count == 0
    
    def test_document_chunk_creation(self):
        """Test DocumentChunk model."""
        chunk = DocumentChunk(
            chunk_id="chunk-123",
            document_id="doc-123",
            content="Chunk content",
            chunk_index=0,
        )
        
        assert chunk.chunk_id == "chunk-123"
        assert chunk.document_id == "doc-123"
        assert chunk.content == "Chunk content"
        assert chunk.chunk_index == 0
        assert isinstance(chunk.created_at, datetime)
    
    def test_ingestion_result_model(self):
        """Test IngestionResult model."""
        result = IngestionResult(
            success=True,
            documents_processed=5,
            chunks_created=50,
            errors=[],
            duration_seconds=10.5,
            message="Success",
        )
        
        assert result.success is True
        assert result.documents_processed == 5
        assert result.chunks_created == 50


class TestParserRegistry:
    """Test ParserRegistry and parsers."""
    
    def test_parser_registry_initialization(self):
        """Test ParserRegistry initialization."""
        registry = ParserRegistry()
        
        assert "txt" in registry.parsers
        assert "pdf" in registry.parsers
        assert "docx" in registry.parsers
        assert "csv" in registry.parsers
        assert "ppt" in registry.parsers
    
    def test_text_parser_can_parse(self):
        """Test TextParser format detection."""
        parser = TextParser()
        
        assert parser.can_parse("file.txt") is True
        assert parser.can_parse("file.pdf") is False
        assert parser.can_parse("file.docx") is False
    
    def test_text_parser_parse_valid_file(self):
        """Test parsing a valid text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Sample text content")
            temp_path = f.name
        
        try:
            parser = TextParser()
            doc = parser.parse(temp_path)
            
            assert doc is not None
            assert "Sample text content" in doc.content
            assert doc.metadata.file_format == "txt"
        finally:
            Path(temp_path).unlink()
    
    def test_text_parser_empty_file(self):
        """Test parsing empty text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name  # Write nothing
        
        try:
            parser = TextParser()
            doc = parser.parse(temp_path)
            
            assert doc is None  # Should return None for empty file
        finally:
            Path(temp_path).unlink()
    
    def test_text_parser_nonexistent_file(self):
        """Test parsing nonexistent file."""
        parser = TextParser()
        doc = parser.parse("/nonexistent/file.txt")
        
        assert doc is None
    
    def test_csv_parser_can_parse(self):
        """Test CSVParser format detection."""
        parser = CSVParser()
        
        assert parser.can_parse("file.csv") is True
        assert parser.can_parse("file.txt") is False
    
    def test_get_parser_for_format(self):
        """Test getting parser for file format."""
        registry = ParserRegistry()
        
        txt_parser = registry.get_parser("file.txt")
        assert txt_parser is not None
        assert isinstance(txt_parser, TextParser)
        
        unsupported = registry.get_parser("file.unknown")
        assert unsupported is None
    
    def test_register_custom_parser(self):
        """Test registering custom parser."""
        registry = ParserRegistry()
        
        custom_parser = TextParser()
        registry.register_parser("custom", custom_parser)
        
        assert "custom" in registry.parsers


class TestSQLiteManager:
    """Test SQLiteManager operations."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = SQLiteManager(str(db_path))
            yield manager
            # Cleanup
            if db_path.exists():
                db_path.unlink()
    
    def test_sqlite_manager_initialization(self, temp_db):
        """Test SQLiteManager initialization."""
        assert Path(temp_db.db_path).exists()
    
    def test_insert_and_retrieve_chunk(self, temp_db):
        """Test inserting and retrieving a chunk."""
        chunk = DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Sample content",
            chunk_index=0,
        )
        
        # Insert
        success = temp_db.insert_chunk(chunk)
        assert success is True
        
        # Retrieve
        retrieved = temp_db.get_chunk("chunk-1")
        assert retrieved is not None
        assert retrieved.content == "Sample content"
        assert retrieved.document_id == "doc-1"
    
    def test_insert_duplicate_chunk(self, temp_db):
        """Test inserting duplicate chunk."""
        chunk = DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Content",
            chunk_index=0,
        )
        
        # Insert first time
        success1 = temp_db.insert_chunk(chunk)
        assert success1 is True
        
        # Insert again - should return False
        success2 = temp_db.insert_chunk(chunk)
        assert success2 is False
    
    def test_get_nonexistent_chunk(self, temp_db):
        """Test retrieving nonexistent chunk."""
        chunk = temp_db.get_chunk("nonexistent")
        assert chunk is None
    
    def test_get_chunks_by_document(self, temp_db):
        """Test retrieving all chunks for a document."""
        chunks = [
            DocumentChunk(chunk_id=f"chunk-{i}", document_id="doc-1", content=f"Content {i}", chunk_index=i)
            for i in range(3)
        ]
        
        # Insert chunks
        for chunk in chunks:
            temp_db.insert_chunk(chunk)
        
        # Retrieve
        retrieved = temp_db.get_chunks_by_document("doc-1")
        assert len(retrieved) == 3
        assert retrieved[0].chunk_index == 0
    
    def test_delete_chunk(self, temp_db):
        """Test deleting a chunk."""
        chunk = DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Content",
            chunk_index=0,
        )
        
        temp_db.insert_chunk(chunk)
        
        # Delete
        success = temp_db.delete_chunk("chunk-1")
        assert success is True
        
        # Verify deletion
        retrieved = temp_db.get_chunk("chunk-1")
        assert retrieved is None
    
    def test_delete_nonexistent_chunk(self, temp_db):
        """Test deleting nonexistent chunk."""
        success = temp_db.delete_chunk("nonexistent")
        assert success is False
    
    def test_delete_document_chunks(self, temp_db):
        """Test deleting all chunks for a document."""
        chunks = [
            DocumentChunk(chunk_id=f"chunk-{i}", document_id="doc-1", content=f"Content {i}", chunk_index=i)
            for i in range(3)
        ]
        
        for chunk in chunks:
            temp_db.insert_chunk(chunk)
        
        # Delete all for document
        count = temp_db.delete_document_chunks("doc-1")
        assert count == 3
        
        # Verify deletion
        remaining = temp_db.get_chunks_by_document("doc-1")
        assert len(remaining) == 0
    
    def test_get_chunk_count(self, temp_db):
        """Test getting chunk count."""
        chunks = [
            DocumentChunk(chunk_id=f"chunk-{i}", document_id="doc-1", content=f"Content {i}", chunk_index=i)
            for i in range(5)
        ]
        
        for chunk in chunks:
            temp_db.insert_chunk(chunk)
        
        # Total count
        total = temp_db.get_chunk_count()
        assert total == 5
        
        # Count by document
        doc_count = temp_db.get_chunk_count("doc-1")
        assert doc_count == 5
    
    def test_insert_chunks_bulk(self, temp_db):
        """Test bulk insertion of chunks."""
        chunks = [
            DocumentChunk(chunk_id=f"chunk-{i}", document_id="doc-1", content=f"Content {i}", chunk_index=i)
            for i in range(10)
        ]
        
        inserted = temp_db.insert_chunks(chunks)
        assert inserted == 10
        
        total = temp_db.get_chunk_count()
        assert total == 10


class TestIngestionManager:
    """Test IngestionManager orchestration."""
    
    @pytest.fixture
    def temp_env(self):
        """Create temporary environment for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create config file
            config_path = tmpdir_path / "config.yaml"
            config_content = f"""
ingestion:
  input_path: {tmpdir_path / "input"}
  output_path: {tmpdir_path / "output"}
  supported_formats:
    - txt
  max_file_size: 100
  parsing:
    enable_ocr: true
  chunking:
    strategy: recursive
    recursive:
      chunk_size: 100
      overlap: 10

storage:
  sqlite:
    db_path: {tmpdir_path / "chunks.db"}
    table_name: document_chunks
  vector_db:
    type: chroma
    path: {tmpdir_path / "vector_store"}
    collection_name: documents
"""
            config_path.write_text(config_content)
            
            # Create input directory
            (tmpdir_path / "input").mkdir()
            
            yield {
                "tmpdir": tmpdir_path,
                "config_path": str(config_path),
            }
    
    def test_ingestion_manager_initialization(self, temp_env):
        """Test IngestionManager initialization."""
        manager = IngestionManager(temp_env["config_path"])
        
        assert manager.config is not None
        assert manager.parser_registry is not None
        assert manager.storage_manager is not None
    
    def test_load_valid_config(self, temp_env):
        """Test loading valid configuration."""
        manager = IngestionManager(temp_env["config_path"])
        
        assert "txt" in manager.config.supported_formats
        assert manager.config.chunking.framework.value == "langchain"
    
    def test_load_nonexistent_config(self):
        """Test loading nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            IngestionManager("/nonexistent/config.yaml")
    
    def test_ingest_directory_empty(self, temp_env):
        """Test ingesting empty directory."""
        manager = IngestionManager(temp_env["config_path"])
        result = manager.ingest_directory(str(temp_env["tmpdir"] / "input"))
        
        assert result.success is True
        assert result.documents_processed == 0
        assert result.chunks_created == 0
    
    def test_ingest_directory_with_text_file(self, temp_env):
        """Test ingesting directory with text file."""
        # Create test file
        input_dir = temp_env["tmpdir"] / "input"
        test_file = input_dir / "test.txt"
        test_file.write_text("Sample content for testing ingestion manager")
        
        manager = IngestionManager(temp_env["config_path"])
        result = manager.ingest_directory(str(input_dir))
        
        assert result.success is True
        assert result.documents_processed == 1
        assert result.chunks_created > 0
    
    def test_chunk_document_recursive(self, temp_env):
        """Test recursive chunking."""
        metadata = DocumentMetadata(
            file_path="/test.txt",
            file_name="test.txt",
            file_format="txt",
            file_size=100,
        )
        
        doc = Document(
            document_id="doc-1",
            content="Sample " * 50,  # Long content
            metadata=metadata,
        )
        
        manager = IngestionManager(temp_env["config_path"])
        chunks = manager._chunk_document(doc)
        
        assert len(chunks) > 0
        assert chunks[0].document_id == "doc-1"
    
    def test_get_document_chunks(self, temp_env):
        """Test retrieving document chunks."""
        # Create and ingest a document
        input_dir = temp_env["tmpdir"] / "input"
        test_file = input_dir / "test.txt"
        test_file.write_text("Content for testing")
        
        manager = IngestionManager(temp_env["config_path"])
        manager.ingest_directory(str(input_dir))
        
        # Get total chunks
        stats = manager.get_stats()
        assert stats["total_chunks"] > 0
    
    def test_get_stats(self, temp_env):
        """Test getting ingestion statistics."""
        manager = IngestionManager(temp_env["config_path"])
        stats = manager.get_stats()
        
        assert "total_chunks" in stats
        assert "db_path" in stats
        assert "chunking_framework" in stats
        assert "chunking_method" in stats
        assert "supported_formats" in stats

    def test_delete_chunk_clears_vector_db_first(self):
        """Test deleting a chunk removes the vector before touching SQLite."""

        class FakeChunk:
            def __init__(self):
                self.chunk_id = "chunk-1"
                self.document_id = "doc-1"
                self.content = "chunk content"
                self.chunk_index = 0
                self.metadata = {"source": "test.txt"}

        class FakeStorageManager:
            def __init__(self):
                self.delete_called = False

            def get_chunk(self, chunk_id):
                return FakeChunk() if chunk_id == "chunk-1" else None

            def delete_chunk(self, chunk_id):
                self.delete_called = True
                return True

        class FakeVectorDB:
            def __init__(self):
                self.deleted = []

            def delete_vectors(self, chunk_ids):
                self.deleted.extend(chunk_ids)

        manager = IngestionManager.__new__(IngestionManager)
        manager.storage_manager = FakeStorageManager()
        manager.vector_db = FakeVectorDB()
        manager.embedding_manager = None

        assert manager.delete_chunk("chunk-1") is True
        assert manager.vector_db.deleted == ["chunk-1"]
        assert manager.storage_manager.delete_called is True

    def test_delete_chunk_aborts_when_vector_delete_fails(self):
        """Test that SQLite deletion is skipped if the vector delete fails."""

        class FakeChunk:
            def __init__(self):
                self.chunk_id = "chunk-1"
                self.document_id = "doc-1"
                self.content = "chunk content"
                self.chunk_index = 0
                self.metadata = {"source": "test.txt"}

        class FakeStorageManager:
            def __init__(self):
                self.delete_called = False

            def get_chunk(self, chunk_id):
                return FakeChunk() if chunk_id == "chunk-1" else None

            def delete_chunk(self, chunk_id):
                self.delete_called = True
                return True

        class FailingVectorDB:
            def delete_vectors(self, chunk_ids):
                raise RuntimeError("vector delete failed")

        manager = IngestionManager.__new__(IngestionManager)
        manager.storage_manager = FakeStorageManager()
        manager.vector_db = FailingVectorDB()
        manager.embedding_manager = None

        assert manager.delete_chunk("chunk-1") is False
        assert manager.storage_manager.delete_called is False

    def test_clear_session_data_clears_both_vector_backends(self):
        """Test reset clears active and inactive vector stores before SQLite."""

        class FakeVectorBackend:
            def __init__(self, label):
                self.label = label
                self.cleared = False

            def clear(self):
                self.cleared = True

        class FakeActiveVectorDB(FakeVectorBackend):
            def __init__(self):
                super().__init__("active")

        class FakeStorageManager:
            def __init__(self):
                self.cleared = False

            def clear_all_chunks(self):
                self.cleared = True
                return 3

        fake_backends = {
            "faiss": FakeVectorBackend("faiss"),
            "chroma": FakeVectorBackend("chroma"),
        }

        def fake_create(config):
            return fake_backends[config.type.value]

        manager = IngestionManager.__new__(IngestionManager)
        manager.storage_manager = FakeStorageManager()
        manager.vector_db = FakeActiveVectorDB()
        manager.embedding_manager = None
        manager.config = SimpleNamespace(
            storage=SimpleNamespace(
                vector_db=SimpleNamespace(
                    enabled=True,
                    type=VectorDBType.FAISS,
                    faiss=SimpleNamespace(),
                    chroma=SimpleNamespace(),
                )
            )
        )

        with patch("src.vector_db.factory.VectorDatabaseFactory.create", side_effect=fake_create):
            summary = manager.clear_session_data()

        assert summary["vector_db_cleared"] is True
        assert summary["storage_cleared"] is True
        assert summary["chunks_removed"] == 3
        assert set(summary["vector_backends_cleared"]) == {"faiss", "chroma"}
        assert manager.vector_db.cleared is True
        assert fake_backends["chroma"].cleared is True
        assert manager.storage_manager.cleared is True


class TestStructuredLogger:
    """Test StructuredLogger functionality."""
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = StructuredLogger(name="test_logger")
        
        assert logger.logger is not None
        assert logger.format_type == "json"
    
    def test_logger_info_without_context(self):
        """Test logging info without context."""
        logger = StructuredLogger(name="test_logger", format_type="text")
        
        # Should not raise exception
        logger.info("Test message")
    
    def test_logger_info_with_context(self):
        """Test logging info with context."""
        logger = StructuredLogger(name="test_logger", format_type="text")
        
        # Should not raise exception
        logger.info("Test message", user_id="123", action="test")
    
    def test_logger_debug(self):
        """Test debug logging."""
        logger = StructuredLogger(name="test_logger", format_type="text")
        logger.debug("Debug message", detail="info")
    
    def test_logger_warning(self):
        """Test warning logging."""
        logger = StructuredLogger(name="test_logger", format_type="text")
        logger.warning("Warning message")
    
    def test_logger_error(self):
        """Test error logging."""
        logger = StructuredLogger(name="test_logger", format_type="text")
        logger.error("Error message")
    
    def test_logger_critical(self):
        """Test critical logging."""
        logger = StructuredLogger(name="test_logger", format_type="text")
        logger.critical("Critical message")
    
    def test_logger_with_file(self):
        """Test logger with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = StructuredLogger(
                name="test_logger",
                log_file=str(log_file),
                format_type="text"
            )
            
            logger.info("Test message")
            
            # Verify file was created
            assert log_file.exists()
            assert "Test message" in log_file.read_text()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/ingestion", "--cov=src/storage", "--cov-report=term-missing"])
