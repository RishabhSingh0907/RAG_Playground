"""
Main ingestion manager orchestrating the complete data ingestion pipeline.

Coordinates configuration loading, file parsing, chunking, storage,
embedding generation, and vector database operations with error handling.
"""

import copy
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import yaml

from src.ingestion.models import (
    IngestionConfig,
    Document,
    DocumentChunk,
    IngestionResult,
    VectorDBType,
)
from src.ingestion.parsers import ParserRegistry
from src.ingestion.chunking_strategies import ChunkingStrategyFactory
from src.storage.sqlite_manager import SQLiteManager
from src.embedding.embedding_manager import EmbeddingManager
from src.vector_db.factory import VectorDatabaseFactory
from src.utils.logger import get_logger


logger = get_logger(__name__)


class IngestionManager:
    """
    Orchestrates the data ingestion pipeline.
    
    Responsibilities:
    1. Load and validate configuration
    2. Parse documents (multimodal support)
    3. Chunk content (configurable strategies)
    4. Embed chunks using configured model
    5. Store chunks in SQLite + vector DB
    6. Handle errors and retries
    7. Provide observability
    """
    
    def __init__(self, config_path: str):
        """
        Initialize ingestion manager.
        
        Args:
            config_path: Path to YAML configuration file
        
        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        logger.info(f"Initializing IngestionManager", config_path=config_path)
        
        self.config = self._load_config(config_path)
        self.parser_registry = ParserRegistry()
        self.storage_manager = SQLiteManager(
            self.config.storage.sqlite.db_path,
            self.config.storage.sqlite.table_name,
        )
        
        # Initialize embedding manager and vector database (if enabled)
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.vector_db = None
        
        if self.config.storage.vector_db.enabled:
            try:
                self.embedding_manager = EmbeddingManager(
                    self.config.storage.vector_db.embedding
                )
                self.vector_db = VectorDatabaseFactory.create(
                    self.config.storage.vector_db
                )
                logger.info(
                    "Vector database and embedding manager initialized",
                    vector_db_type=self.config.storage.vector_db.type,
                    embedding_model=self.config.storage.vector_db.embedding.model,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize vector database: {str(e)}. Proceeding without vector DB.",
                    exc_info=True,
                )
                self.embedding_manager = None
                self.vector_db = None
        else:
            logger.info("Vector database disabled in configuration")
        
        logger.info("IngestionManager initialized successfully")
        self._log_vector_db_status()
    
    def _load_config(self, config_path: str) -> IngestionConfig:
        """
        Load and validate configuration from YAML.
        
        Args:
            config_path: Path to YAML config file
        
        Returns:
            IngestionConfig object
        
        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config is invalid
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                raise ValueError("Configuration file is empty")
            
            # Extract ingestion section
            ingestion_data = config_data.get("ingestion", {})
            storage_data = config_data.get("storage", {})
            
            # Merge storage into ingestion config
            ingestion_data["storage"] = storage_data
            
            # Validate using Pydantic
            config = IngestionConfig(**ingestion_data)
            logger.info(f"Configuration loaded successfully", formats=config.supported_formats)
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file", exc_info=True, error=str(e))
            raise ValueError(f"Invalid YAML configuration: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load configuration", exc_info=True, error=str(e))
            raise
    
    def _log_vector_db_status(self) -> None:
        """Log detailed vector DB initialization status for debugging."""
        config = self.config.storage.vector_db
        logger.debug(
            "Vector DB Status - Configuration",
            enabled=config.enabled,
            type=config.type.value if config.enabled else None,
            embedding_provider=config.embedding.provider if config.enabled else None,
        )
        logger.debug(
            "Vector DB Status - Initialization",
            embedding_manager_exists=self.embedding_manager is not None,
            vector_db_exists=self.vector_db is not None,
            both_initialized=self.embedding_manager is not None and self.vector_db is not None,
        )
    
    def ingest_directory(self, input_path: Optional[str] = None) -> IngestionResult:
        """
        Ingest all supported files from a directory.
        
        Args:
            input_path: Optional override for input path from config
        
        Returns:
            IngestionResult with summary and any errors
        """
        start_time = time.time()
        
        path = Path(input_path or self.config.input_path)
        
        if not path.exists():
            logger.error(f"Input directory not found", path=str(path))
            return IngestionResult(
                success=False,
                documents_processed=0,
                chunks_created=0,
                errors=[{"error": "Input directory not found", "path": str(path)}],
                duration_seconds=time.time() - start_time,
                message="Input directory does not exist",
            )
        
        logger.info(f"Starting directory ingestion", path=str(path))
        
        # Collect all supported files
        files_to_ingest = []
        for supported_fmt in self.config.supported_formats:
            files_to_ingest.extend(path.glob(f"*.{supported_fmt}"))
        
        if not files_to_ingest:
            logger.warning(f"No supported files found in directory", path=str(path))
            return IngestionResult(
                success=True,
                documents_processed=0,
                chunks_created=0,
                errors=[],
                duration_seconds=time.time() - start_time,
                message=f"No files found matching supported formats: {', '.join(self.config.supported_formats)}",
            )
        
        logger.info(f"Found files for ingestion", count=len(files_to_ingest))
        
        # Ingest each file
        total_chunks = 0
        total_vectors_stored = 0
        errors = []
        
        for file_path in files_to_ingest:
            try:
                chunk_count, vectors_stored = self._ingest_file(file_path)
                total_chunks += chunk_count
                total_vectors_stored += vectors_stored
                logger.info(
                    f"File ingested successfully",
                    file=file_path.name,
                    chunks=chunk_count,
                    vectors_stored=vectors_stored,
                )
            except Exception as e:
                logger.error(
                    f"Failed to ingest file",
                    file=str(file_path),
                    exc_info=True,
                    error=str(e),
                )
                errors.append({
                    "file": str(file_path),
                    "error": str(e),
                })
        
        duration = time.time() - start_time
        success = len(errors) == 0
        
        result = IngestionResult(
            success=success,
            documents_processed=len(files_to_ingest),
            chunks_created=total_chunks,
            vectors_stored=total_vectors_stored,
            errors=errors,
            duration_seconds=duration,
            message=f"Ingestion {'completed successfully' if success else 'completed with errors'}",
        )
        
        # Exclude 'message' from dict to avoid conflict with logger.info() signature
        result_dict = {k: v for k, v in result.dict().items() if k != 'message'}
        logger.info(result.message, **result_dict)
        return result
    
    def _ingest_file(self, file_path: Path) -> tuple[int, int]:
        """
        Ingest a single file with embedding and vector DB storage.
        
        Args:
            file_path: Path to file
        
        Returns:
            Tuple of (chunks_created, vectors_stored)
        
        Raises:
            Exception: If ingestion fails
        """
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size:
            raise ValueError(
                f"File size ({file_size_mb:.2f} MB) exceeds limit ({self.config.max_file_size} MB)"
            )
        
        # Parse document
        document = self.parser_registry.parse_file(str(file_path))
        
        if not document:
            raise ValueError(f"Failed to parse file: {file_path.name}")
        
        # Chunk document
        chunks = self._chunk_document(document)
        
        if not chunks:
            logger.warning(f"No chunks created from document", doc_id=document.document_id)
            return 0, 0
        
        # Store chunks in SQLite
        stored_count = self.storage_manager.insert_chunks(chunks)
        vectors_stored = 0
        
        # Embed chunks and store in vector database (if enabled)
        logger.debug(
            "Vector DB check",
            vector_db_enabled=self.config.storage.vector_db.enabled,
            embedding_manager_initialized=self.embedding_manager is not None,
            vector_db_initialized=self.vector_db is not None,
        )
        
        if self.embedding_manager and self.vector_db:
            try:
                logger.debug(
                    "Generating embeddings for chunks",
                    doc_id=document.document_id,
                    chunk_count=len(chunks),
                )
                
                # Extract chunk texts for embedding
                chunk_texts = [chunk.content for chunk in chunks]
                
                # Generate embeddings
                embeddings = self.embedding_manager.embed_texts(chunk_texts)
                
                # Prepare metadata for vector storage
                metadata_list = [
                    {
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "source": document.metadata.file_path,
                        "file_name": document.metadata.file_name,
                        "created_at": chunk.created_at.isoformat() if hasattr(chunk.created_at, 'isoformat') else str(chunk.created_at),
                    }
                    for chunk in chunks
                ]
                
                # Store in vector database
                self.vector_db.add_vectors(
                    chunk_ids=[chunk.chunk_id for chunk in chunks],
                    vectors=embeddings,
                    metadata=metadata_list,
                )
                
                vectors_stored = len(chunks)
                logger.info(
                    "Chunks stored in vector database",
                    doc_id=document.document_id,
                    vectors_stored=vectors_stored,
                )
            except Exception as e:
                logger.error(
                    f"Failed to store embeddings in vector database: {str(e)}",
                    doc_id=document.document_id,
                    exc_info=True,
                )
                # Continue without vector DB - chunks are still in SQLite
        else:
            if not self.embedding_manager:
                logger.warning("Embedding manager not initialized - vector DB storage skipped")
            if not self.vector_db:
                logger.warning("Vector database not initialized - embeddings will not be stored")
        
        logger.info(
            f"File ingestion completed",
            file=file_path.name,
            doc_id=document.document_id,
            chunks=stored_count,
            vectors_stored=vectors_stored,
        )
        return stored_count, vectors_stored
    
    def _chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk a document using configured strategy.
        
        Args:
            document: Document to chunk
        
        Returns:
            List of DocumentChunk objects
        """
        logger.debug(
            f"Chunking document",
            doc_id=document.document_id,
            framework=self.config.chunking.framework,
            method=(
                self.config.chunking.langchain.method
                if self.config.chunking.framework.value == "langchain"
                else self.config.chunking.llamaindex.method
            ),
        )
        
        # Create strategy using factory
        strategy = ChunkingStrategyFactory.create_strategy(
            framework=self.config.chunking.framework,
            langchain_config=self.config.chunking.langchain,
            llamaindex_config=self.config.chunking.llamaindex,
        )
        
        # Execute chunking
        chunks = strategy.chunk(document)
        
        logger.info(
            f"Document chunked successfully",
            doc_id=document.document_id,
            chunk_count=len(chunks),
        )
        return chunks
    
    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Retrieve all chunks for a document."""
        return self.storage_manager.get_chunks_by_document(document_id)
    
    def get_all_chunks(self, limit: int = 100) -> List[DocumentChunk]:
        """
        Retrieve all chunks from storage.
        
        Args:
            limit: Maximum number of chunks to retrieve
        
        Returns:
            List of DocumentChunk objects
        """
        return self.storage_manager.get_all_chunks(limit=limit)
    
    def search_chunks(self, search_text: str, limit: int = 100) -> List[DocumentChunk]:
        """
        Search chunks by content.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
        
        Returns:
            List of matching DocumentChunk objects
        """
        return self.storage_manager.search_chunks(search_text, limit=limit)
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk by ID.
        
        Args:
            chunk_id: Chunk identifier to delete
        
        Returns:
            True if deletion successful, False otherwise
        """
        chunk = self.storage_manager.get_chunk(chunk_id)
        if chunk is None:
            logger.warning("Chunk not found for deletion", chunk_id=chunk_id)
            return False

        vector_deleted = True
        if self.vector_db is not None:
            try:
                self.vector_db.delete_vectors([chunk_id])
            except Exception as exc:
                logger.error(
                    "Failed to delete chunk from vector database",
                    chunk_id=chunk_id,
                    error=str(exc),
                    exc_info=True,
                )
                return False

        storage_deleted = self.storage_manager.delete_chunk(chunk_id)
        if not storage_deleted:
            logger.error("Failed to delete chunk from SQLite storage", chunk_id=chunk_id)
            if self.vector_db is not None and self.embedding_manager is not None:
                try:
                    vector = self.embedding_manager.embed_single(chunk.content)
                    self.vector_db.add_vectors(
                        chunk_ids=[chunk.chunk_id],
                        vectors=[vector],
                        metadata=[{
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.chunk_index,
                            "source": chunk.metadata.get("source") if chunk.metadata else None,
                            "restored": True,
                        }],
                    )
                    logger.info("Restored vector after SQLite delete failure", chunk_id=chunk_id)
                except Exception as restore_exc:
                    logger.error(
                        "Failed to restore vector after SQLite delete failure",
                        chunk_id=chunk_id,
                        error=str(restore_exc),
                        exc_info=True,
                    )
            return False

        return vector_deleted and storage_deleted
    
    def get_vector_db_status(self) -> Dict[str, Any]:
        """
        Get vector database status and configuration.
        
        Returns:
            Dictionary with vector DB status, type, embedding config, etc.
        """
        vector_db_config = self.config.storage.vector_db
        
        # Handle embedding model - could be string or enum
        embedding_model_value = None
        if vector_db_config.enabled:
            model = vector_db_config.embedding.model
            embedding_model_value = model.value if hasattr(model, 'value') else str(model)
        
        status = {
            "enabled": vector_db_config.enabled,
            "type": vector_db_config.type.value if vector_db_config.enabled else None,
            "initialized": self.vector_db is not None and self.embedding_manager is not None,
            "embedding_provider": vector_db_config.embedding.provider if vector_db_config.enabled else None,
            "embedding_model": embedding_model_value,
            "chroma_collection": vector_db_config.chroma.collection_name if vector_db_config.enabled and vector_db_config.type == VectorDBType.CHROMA else None,
            "faiss_index_type": vector_db_config.faiss.index_type if vector_db_config.enabled and vector_db_config.type == VectorDBType.FAISS else None,
        }
        
        # Add FAISS-specific diagnostics if available
        if self.vector_db is not None and hasattr(self.vector_db, 'get_status'):
            try:
                status["faiss_diagnostics"] = self.vector_db.get_status()
            except Exception as e:
                logger.warning(f"Failed to get vector DB diagnostics: {e}")
                status["faiss_diagnostics_error"] = str(e)
        
        return status

    def reinit_vector_db(self) -> bool:
        """Re-initialize embedding manager and vector database from current config.

        Returns True on success, False on failure. Logs detailed info for debugging.
        """
        vector_db_cfg = self.config.storage.vector_db

        # If vector DB is disabled, clear any existing managers
        if not vector_db_cfg.enabled:
            if self.vector_db is not None:
                try:
                    self.vector_db.persist()
                except Exception:
                    logger.warning("Failed to persist existing vector DB during reinit", exc_info=True)
            self.embedding_manager = None
            self.vector_db = None
            logger.info("Vector DB disabled in config; cleared embedding manager and vector DB")
            return True

        # Persist existing DB if present
        if self.vector_db is not None:
            try:
                logger.info("Persisting existing vector DB before reinitialization")
                self.vector_db.persist()
            except Exception as e:
                logger.warning(f"Error persisting existing vector DB before reinit: {e}", exc_info=True)

        # Create new embedding manager and vector DB according to current config
        try:
            self.embedding_manager = EmbeddingManager(vector_db_cfg.embedding)
            self.vector_db = VectorDatabaseFactory.create(vector_db_cfg)
            logger.info(
                "Vector database reinitialized successfully",
                vector_db_type=vector_db_cfg.type,
                embedding_model=vector_db_cfg.embedding.model,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to reinitialize vector DB: {e}", exc_info=True)
            self.embedding_manager = None
            self.vector_db = None
            return False

    def clear_session_data(self) -> Dict[str, Any]:
        """Clear session-scoped chunks and embeddings."""

        result: Dict[str, Any] = {
            "storage_cleared": False,
            "vector_db_cleared": False,
            "vector_backends_cleared": [],
            "chunks_removed": 0,
        }

        vector_clear_failed = False
        vector_db_cfg = self.config.storage.vector_db

        def clear_backend(backend_type: VectorDBType, backend_config: Any) -> None:
            from src.vector_db.factory import VectorDatabaseFactory

            temp_config = copy.deepcopy(vector_db_cfg)
            temp_config.type = backend_type
            if backend_type == VectorDBType.FAISS:
                temp_config.faiss = copy.deepcopy(backend_config)
            else:
                temp_config.chroma = copy.deepcopy(backend_config)

            backend = VectorDatabaseFactory.create(temp_config)
            backend.clear()
            result["vector_backends_cleared"].append(backend_type.value)

        try:
            if self.vector_db is not None:
                self.vector_db.clear()
                result["vector_db_cleared"] = True
                result["vector_backends_cleared"].append(vector_db_cfg.type.value)

            if vector_db_cfg.enabled:
                if vector_db_cfg.type != VectorDBType.FAISS:
                    clear_backend(VectorDBType.FAISS, vector_db_cfg.faiss)
                if vector_db_cfg.type != VectorDBType.CHROMA:
                    clear_backend(VectorDBType.CHROMA, vector_db_cfg.chroma)

        except Exception as exc:
            vector_clear_failed = True
            logger.warning("Failed to clear one or more vector backends", error=str(exc), exc_info=True)

        if not vector_clear_failed:
            try:
                result["chunks_removed"] = self.storage_manager.clear_all_chunks()
                result["storage_cleared"] = True
            except Exception as exc:
                logger.warning("Failed to clear SQLite chunks", error=str(exc), exc_info=True)

        return result
        
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        total_chunks = self.storage_manager.get_chunk_count()
        
        return {
            "total_chunks": total_chunks,
            "db_path": self.config.storage.sqlite.db_path,
            "chunking_framework": self.config.chunking.framework.value,
            "chunking_method": self.config.chunking.langchain.method.value if self.config.chunking.framework.value == "langchain" else self.config.chunking.llamaindex.method.value,
            "supported_formats": self.config.supported_formats,
        }
