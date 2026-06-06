"""
Storage module for the RAG platform.

Provides persistence layer for document chunks with SQLite backend.
"""

from src.storage.sqlite_manager import SQLiteManager

__all__ = ["SQLiteManager"]
