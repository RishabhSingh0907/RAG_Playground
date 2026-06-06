"""
SQLite manager for storing document chunks.

Handles persistence of chunks, metadata, and unique ID mappings.
Supports efficient querying and filtering of stored chunks.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from src.ingestion.models import DocumentChunk
from src.utils.logger import get_logger


logger = get_logger(__name__)


class SQLiteManager:
    """
    Manages SQLite database for storing document chunks.
    
    Provides CRUD operations for chunks with proper error handling
    and transaction management.
    """
    
    def __init__(self, db_path: str, table_name: str = "document_chunks"):
        """
        Initialize SQLite manager.
        
        Args:
            db_path: Path to SQLite database file
            table_name: Name of table for storing chunks
        
        Raises:
            IOError: If database directory cannot be created
        """
        self.db_path = db_path
        self.table_name = table_name
        
        # Create directory if needed
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing SQLiteManager", db_path=db_path, table_name=table_name)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Create tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create chunks table
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        metadata TEXT,
                        embedding_vector BLOB,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                # Create index on document_id for faster queries
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_document_id 
                    ON {self.table_name}(document_id)
                """)
                
                conn.commit()
                logger.info(f"Database initialized successfully", table=self.table_name)
                
        except sqlite3.DatabaseError as e:
            logger.error(f"Database initialization failed", exc_info=True, error=str(e))
            raise
    
    def insert_chunk(self, chunk: DocumentChunk) -> bool:
        """
        Insert a chunk into the database.
        
        Args:
            chunk: DocumentChunk to insert
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                metadata_json = json.dumps(chunk.metadata) if chunk.metadata else None
                embedding_blob = None  # Store as BLOB if needed
                
                cursor.execute(f"""
                    INSERT INTO {self.table_name} 
                    (chunk_id, document_id, content, chunk_index, metadata, embedding_vector, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.content,
                    chunk.chunk_index,
                    metadata_json,
                    embedding_blob,
                    chunk.created_at.isoformat(),
                    datetime.utcnow().isoformat(),
                ))
                
                conn.commit()
                logger.debug(f"Inserted chunk", chunk_id=chunk.chunk_id, doc_id=chunk.document_id)
                return True
                
        except sqlite3.IntegrityError as e:
            logger.warning(f"Chunk already exists", chunk_id=chunk.chunk_id, error=str(e))
            return False
        except sqlite3.DatabaseError as e:
            logger.error(f"Failed to insert chunk", chunk_id=chunk.chunk_id, exc_info=True, error=str(e))
            return False
    
    def insert_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Insert multiple chunks into the database.
        
        Args:
            chunks: List of DocumentChunk objects
        
        Returns:
            Number of successfully inserted chunks
        """
        inserted_count = 0
        
        for chunk in chunks:
            if self.insert_chunk(chunk):
                inserted_count += 1
        
        logger.info(f"Bulk insert completed", total=len(chunks), inserted=inserted_count)
        return inserted_count
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
        
        Returns:
            DocumentChunk or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"""
                    SELECT chunk_id, document_id, content, chunk_index, metadata, created_at
                    FROM {self.table_name}
                    WHERE chunk_id = ?
                """, (chunk_id,))
                
                row = cursor.fetchone()
                
                if not row:
                    logger.debug(f"Chunk not found", chunk_id=chunk_id)
                    return None
                
                chunk = self._row_to_chunk(row)
                logger.debug(f"Retrieved chunk", chunk_id=chunk_id)
                return chunk
                
        except sqlite3.DatabaseError as e:
            logger.error(f"Failed to retrieve chunk", chunk_id=chunk_id, exc_info=True, error=str(e))
            return None
    
    def get_chunks_by_document(self, document_id: str) -> List[DocumentChunk]:
        """
        Retrieve all chunks for a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            List of DocumentChunk objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"""
                    SELECT chunk_id, document_id, content, chunk_index, metadata, created_at
                    FROM {self.table_name}
                    WHERE document_id = ?
                    ORDER BY chunk_index ASC
                """, (document_id,))
                
                rows = cursor.fetchall()
                chunks = [self._row_to_chunk(row) for row in rows]
                
                logger.debug(f"Retrieved chunks for document", doc_id=document_id, count=len(chunks))
                return chunks
                
        except sqlite3.DatabaseError as e:
            logger.error(f"Failed to retrieve chunks", doc_id=document_id, exc_info=True, error=str(e))
            return []
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE chunk_id = ?
                """, (chunk_id,))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Deleted chunk", chunk_id=chunk_id)
                    return True
                else:
                    logger.debug(f"Chunk not found for deletion", chunk_id=chunk_id)
                    return False
                    
        except sqlite3.DatabaseError as e:
            logger.error(f"Failed to delete chunk", chunk_id=chunk_id, exc_info=True, error=str(e))
            return False
    
    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            Number of deleted chunks
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE document_id = ?
                """, (document_id,))
                
                conn.commit()
                deleted_count = cursor.rowcount
                
                logger.info(f"Deleted document chunks", doc_id=document_id, count=deleted_count)
                return deleted_count
                
        except sqlite3.DatabaseError as e:
            logger.error(f"Failed to delete document chunks", doc_id=document_id, exc_info=True, error=str(e))
            return 0
    
    def get_chunk_count(self, document_id: Optional[str] = None) -> int:
        """
        Get count of chunks, optionally filtered by document.
        
        Args:
            document_id: Optional document identifier to filter by
        
        Returns:
            Number of chunks
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if document_id:
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM {self.table_name}
                        WHERE document_id = ?
                    """, (document_id,))
                else:
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM {self.table_name}
                    """)
                
                count = cursor.fetchone()[0]
                logger.debug(f"Chunk count retrieved", count=count, doc_id=document_id)
                return count
                
        except sqlite3.DatabaseError as e:
            logger.error(f"Failed to get chunk count", exc_info=True, error=str(e))
            return 0
    
    def _row_to_chunk(self, row: tuple) -> DocumentChunk:
        """Convert database row to DocumentChunk object."""
        chunk_id, document_id, content, chunk_index, metadata_json, created_at_str = row
        
        metadata = json.loads(metadata_json) if metadata_json else {}
        created_at = datetime.fromisoformat(created_at_str)
        
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            metadata=metadata,
            created_at=created_at,
        )
    
    def get_all_chunks(self, limit: int = 100) -> List[DocumentChunk]:
        """
        Retrieve all chunks from database.
        
        Args:
            limit: Maximum number of chunks to retrieve
        
        Returns:
            List of DocumentChunk objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT chunk_id, document_id, content, chunk_index, metadata, created_at
                    FROM {self.table_name}
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                chunks = [self._row_to_chunk(row) for row in rows]
                
                logger.debug(f"Retrieved {len(chunks)} chunks from database")
                return chunks
                
        except sqlite3.DatabaseError as e:
            logger.error(f"Failed to retrieve chunks", exc_info=True, error=str(e))
            return []
    
    def search_chunks(self, search_text: str, limit: int = 100) -> List[DocumentChunk]:
        """
        Search chunks by content (case-insensitive partial match).
        
        Args:
            search_text: Text to search for in chunk content
            limit: Maximum number of results to return
        
        Returns:
            List of matching DocumentChunk objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                search_pattern = f"%{search_text}%"
                cursor.execute(f"""
                    SELECT chunk_id, document_id, content, chunk_index, metadata, created_at
                    FROM {self.table_name}
                    WHERE LOWER(content) LIKE LOWER(?)
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (search_pattern, limit))
                
                rows = cursor.fetchall()
                chunks = [self._row_to_chunk(row) for row in rows]
                
                logger.debug(f"Search found {len(chunks)} matching chunks", search_term=search_text)
                return chunks
                
        except sqlite3.DatabaseError as e:
            logger.error(f"Failed to search chunks", exc_info=True, error=str(e), search_term=search_text)
            return []

    def clear_all_chunks(self) -> int:
        """Delete every chunk from the current SQLite table.

        Returns:
            Number of removed rows.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"DELETE FROM {self.table_name}")
                conn.commit()
                removed_rows = cursor.rowcount if cursor.rowcount is not None else 0
                logger.info("Cleared all chunks from database", table=self.table_name, removed=removed_rows)
                return removed_rows
        except sqlite3.DatabaseError as e:
            logger.error(f"Failed to clear chunks", exc_info=True, error=str(e))
            return 0
    
    def close(self) -> None:
        """Close database connection (if needed for cleanup)."""
        logger.info("SQLiteManager closed")
