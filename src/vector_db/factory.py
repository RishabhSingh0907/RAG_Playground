"""Factory for creating vector database instances."""

from src.vector_db.base import VectorDatabaseManager
from src.vector_db.faiss_store import FAISSVectorStore
from src.vector_db.chroma_store import ChromaVectorStore
from src.ingestion.models import VectorDBConfig, VectorDBType
from src.utils.logger import get_logger


logger = get_logger(__name__)


class VectorDatabaseFactory:
    """Factory for creating vector database instances."""

    @staticmethod
    def create(config: VectorDBConfig) -> VectorDatabaseManager:
        """
        Create vector database instance based on configuration.

        Args:
            config: Vector database configuration

        Returns:
            VectorDatabaseManager instance

        Raises:
            ValueError: If database type is unsupported
            RuntimeError: If database initialization fails
        """
        if not config.enabled:
            raise RuntimeError("Vector database is disabled in configuration")

        logger.info(f"Creating vector database: {config.type}")

        if config.type == VectorDBType.FAISS:
            return FAISSVectorStore(config.faiss)

        elif config.type == VectorDBType.CHROMA:
            return ChromaVectorStore(config.chroma)

        else:
            raise ValueError(f"Unsupported vector database type: {config.type}")
