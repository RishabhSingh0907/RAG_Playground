"""
Embedding manager for generating embeddings using various providers.

Supports multiple embedding providers: Ollama (local), HuggingFace, OpenAI.
Provides batch processing and caching capabilities.
"""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import requests
import time
from src.ingestion.models import VectorDatabaseEmbeddingConfig, EmbeddingModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)

        Raises:
            ValueError: If embedding generation fails
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider (local, open-source models)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        batch_size: int = 10,
    ):
        """
        Initialize Ollama embedding provider.

        Args:
            model: Model name (e.g., nomic-embed-text)
            base_url: Ollama server URL
            batch_size: Batch size for embedding
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self._dimension: Optional[int] = None

        logger.info(
            "Initialized OllamaEmbeddingProvider",
            model=model,
            base_url=base_url,
            batch_size=batch_size,
        )

        # Verify connection
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify Ollama server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Ollama server connection verified")
        except Exception as e:
            logger.error(
                f"Failed to connect to Ollama at {self.base_url}",
                error=str(e),
            )
            raise ValueError(f"Cannot reach Ollama server at {self.base_url}: {e}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama."""
        if not texts:
            return []

        embeddings = []

        try:
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                for text in batch:
                    # Get embedding for single text
                    embedding = self._embed_single(text)
                    embeddings.append(embedding)

                logger.info(
                    f"Processed batch {i // self.batch_size + 1}",
                    batch_size=len(batch),
                )

            return embeddings

        except Exception as e:
            logger.error(
                "Failed to generate embeddings",
                error=str(e),
                exc_info=True,
            )
            raise ValueError(f"Embedding generation failed: {e}")

    def _embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text with retry logic."""
        max_attempts = 3
        initial_delay = 1.0

        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": text},
                    timeout=30,
                )
                response.raise_for_status()

                result = response.json()
                embedding = result.get("embeddings", [[]])[0]

                if not embedding:
                    raise ValueError("Empty embedding returned")

                return embedding

            except requests.exceptions.Timeout:
                if attempt < max_attempts - 1:
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(
                        f"Embedding timeout, retrying in {delay}s",
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                    )
                    time.sleep(delay)
                else:
                    raise
            except Exception as e:
                if attempt < max_attempts - 1:
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(
                        f"Embedding failed: {str(e)}, retrying in {delay}s",
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                    )
                    time.sleep(delay)
                else:
                    raise

    def get_dimension(self) -> int:
        """Get embedding dimension by embedding a test string."""
        if self._dimension is not None:
            return self._dimension

        try:
            test_embedding = self._embed_single("test")
            self._dimension = len(test_embedding)
            logger.info(f"Embedding dimension: {self._dimension}")
            return self._dimension
        except Exception as e:
            logger.error(f"Failed to determine embedding dimension: {e}")
            raise


class EmbeddingManager:
    """
    Manages embedding generation for vector database storage.

    Responsibilities:
    1. Provide unified interface for multiple embedding providers
    2. Handle batching and optimization
    3. Cache embeddings for reuse
    4. Support retry logic with exponential backoff
    """

    def __init__(self, config: VectorDatabaseEmbeddingConfig):
        """
        Initialize embedding manager.

        Args:
            config: Vector database embedding configuration

        Raises:
            ValueError: If provider is not supported or configuration is invalid
        """
        logger.info(
            "Initializing EmbeddingManager",
            provider=config.provider,
            model=config.model,
        )

        self.config = config
        self.provider = self._initialize_provider(config)

    def _initialize_provider(
        self, config: VectorDatabaseEmbeddingConfig
    ) -> EmbeddingProvider:
        """Initialize embedding provider based on configuration."""
        if config.provider.lower() == "ollama":
            return OllamaEmbeddingProvider(
                model=config.model.value,
                base_url=config.base_url,
                batch_size=config.embed_batch_size,
            )
        elif config.provider.lower() == "huggingface":
            raise NotImplementedError("HuggingFace provider not yet implemented")
        elif config.provider.lower() == "openai":
            raise NotImplementedError("OpenAI provider not yet implemented")
        else:
            raise ValueError(f"Unsupported embedding provider: {config.provider}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If embedding generation fails
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts")

        try:
            embeddings = self.provider.embed(texts)
            logger.info(
                f"Successfully generated {len(embeddings)} embeddings",
                dimension=self.get_embedding_dimension(),
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}", exc_info=True)
            raise

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.provider.get_dimension()
