"""LLM Provider abstraction for Ollama and compatible endpoints.

Handles:
- Connection management to Ollama/compatible LLM servers
- Model inference (batch completion, no streaming for Phase 1)
- Model availability checking
- Error handling and retry logic
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
import time

import requests

from src.utils.logger import get_logger


logger = get_logger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM provider types."""

    OLLAMA = "ollama"
    OPENAI = "openai"  # Future support


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""

    provider: LLMProvider = LLMProvider.OLLAMA
    base_url: str = "http://localhost:11434"
    model: str = "gpt-oss:20b"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    top_k: int = 40
    request_timeout_seconds: int = 120
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config for display."""
        return {
            "provider": self.provider.value,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


class OllamaProvider:
    """Ollama-based LLM provider."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self._available_models: Optional[List[str]] = None
        self._health_check_passed = False

        logger.info(
            "OllamaProvider initialized",
            base_url=self.base_url,
            model=self.model,
        )

    def health_check(self) -> bool:
        """Check if Ollama server is reachable and model is available.

        Returns:
            True if server is up and model is available, False otherwise.
        """
        try:
            # Check if server is running
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            response.raise_for_status()

            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            self._available_models = models

            # Check if requested model is available
            if self.model in models:
                self._health_check_passed = True
                logger.info(
                    "Ollama health check passed",
                    model=self.model,
                    available_models=models,
                )
                return True

            logger.warning(
                "Ollama server running but model not found",
                requested_model=self.model,
                available_models=models,
            )
            return False

        except requests.exceptions.ConnectionError:
            logger.error(
                "Ollama server not reachable",
                base_url=self.base_url,
                error="ConnectionError",
            )
            return False
        except Exception as exc:
            logger.error(
                "Ollama health check failed",
                base_url=self.base_url,
                error=str(exc),
                exc_info=True,
            )
            return False

    def generate(self, prompt: str) -> Optional[str]:
        """Generate completion for a prompt.

        Args:
            prompt: Input prompt for the model

        Returns:
            Generated text or None if failed
        """
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided to LLM")
            return None

        for attempt in range(self.config.max_retries):
            try:
                logger.info(
                    f"LLM generation attempt {attempt + 1}/{self.config.max_retries}",
                    model=self.model,
                    prompt_length=len(prompt),
                )

                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "top_k": self.config.top_k,
                        "num_predict": self.config.max_tokens,
                        "stream": False,
                    },
                    timeout=self.config.request_timeout_seconds,
                )
                response.raise_for_status()

                data = response.json()
                generated_text = data.get("response", "").strip()

                if generated_text:
                    logger.info(
                        "LLM generation successful",
                        model=self.model,
                        response_length=len(generated_text),
                    )
                    return generated_text

                logger.warning(
                    "LLM returned empty response",
                    model=self.model,
                )
                return None

            except requests.exceptions.Timeout:
                logger.warning(
                    f"LLM request timeout (attempt {attempt + 1})",
                    model=self.model,
                    timeout_seconds=self.config.request_timeout_seconds,
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds)
                continue

            except requests.exceptions.RequestException as exc:
                logger.error(
                    f"LLM request failed (attempt {attempt + 1})",
                    model=self.model,
                    error=str(exc),
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay_seconds)
                continue

            except Exception as exc:
                logger.error(
                    "LLM generation error",
                    model=self.model,
                    error=str(exc),
                    exc_info=True,
                )
                return None

        logger.error(
            "LLM generation failed after max retries",
            model=self.model,
            max_retries=self.config.max_retries,
        )
        return None

    def get_available_models(self) -> List[str]:
        """Get list of available models on the Ollama server.

        Returns:
            List of model names
        """
        if self._available_models is None:
            self.health_check()

        return self._available_models or []

    def is_healthy(self) -> bool:
        """Check if provider is healthy (from last health_check call).

        Returns:
            True if last health check passed
        """
        return self._health_check_passed


def create_llm_provider(config: LLMConfig) -> Optional[OllamaProvider]:
    """Factory function to create LLM provider.

    Args:
        config: LLM configuration

    Returns:
        Initialized provider or None if creation fails
    """
    try:
        if config.provider == LLMProvider.OLLAMA:
            provider = OllamaProvider(config)
            if provider.health_check():
                return provider

            logger.warning(
                "LLM provider health check failed; provider created but unhealthy",
                provider=config.provider.value,
            )
            return provider

        logger.error(
            "Unsupported LLM provider",
            provider=config.provider.value,
        )
        return None

    except Exception as exc:
        logger.error(
            "Failed to create LLM provider",
            provider=config.provider.value,
            error=str(exc),
            exc_info=True,
        )
        return None
