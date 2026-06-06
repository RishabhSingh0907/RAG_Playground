"""Generation layer for RAG-augmented LLM responses with citations."""

from src.generation.llm_provider import (
    LLMProvider,
    LLMConfig,
    OllamaProvider,
    create_llm_provider,
)
from src.generation.prompt_builder import (
    PromptBuilder,
    PromptTemplate,
    format_retrieved_chunks,
)
from src.generation.citation_extractor import (
    Citation,
    CitationExtractor,
    build_chunk_dict_from_retrieved,
)
from src.generation.chat_manager import (
    ChatMessage,
    ChatSession,
    ChatManager,
)

__all__ = [
    "LLMProvider",
    "LLMConfig",
    "OllamaProvider",
    "create_llm_provider",
    "PromptBuilder",
    "PromptTemplate",
    "format_retrieved_chunks",
    "Citation",
    "CitationExtractor",
    "build_chunk_dict_from_retrieved",
    "ChatMessage",
    "ChatSession",
    "ChatManager",
]
