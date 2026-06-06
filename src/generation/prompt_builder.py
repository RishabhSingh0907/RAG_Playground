"""Prompt builder for formatting retrieval context into LLM prompts.

Responsibilities:
- Format retrieved chunks with source citations
- Assemble system prompt + context + user query
- Instruct LLM to cite sources using specific format
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from src.retrieval.pipeline import RetrievedChunk
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """Template for prompt assembly."""

    system_prompt: str = """You are a helpful assistant answering questions based on provided documents.

When answering, you MUST cite your sources using this exact format:
[CITE: document_name | chunk_id]

For example: "Machine learning is a subset of AI [CITE: document.pdf | chunk_3]"

Always cite the source for factual claims. If information comes from multiple sources, cite each one.
If you don't know the answer based on the provided context, say so explicitly."""

    context_header: str = "## Retrieved Context\n"
    context_item_template: str = "[SOURCE: {doc_name} | chunk_{chunk_id}]\n{content}\n"
    query_header: str = "## User Question"


def format_retrieved_chunks(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into context string with citations.

    Args:
        chunks: List of retrieved chunks

    Returns:
        Formatted context string
    """
    if not chunks:
        return ""

    context_items = []
    for rank, chunk in enumerate(chunks, start=1):
        # Extract document name from metadata or use document_id
        doc_name = chunk.metadata.get("file_name", chunk.document_id)

        item = f"[SOURCE: {doc_name} | chunk_{chunk.chunk_id}]\n"
        item += f"Rank: {chunk.rank} | Score: {chunk.score:.4f}\n"
        item += f"{chunk.content}\n"
        item += "\n"

        context_items.append(item)

    return "".join(context_items)


class PromptBuilder:
    """Build structured prompts for LLM generation."""

    def __init__(self, template: PromptTemplate = None):
        self.template = template or PromptTemplate()

    def build_prompt(
        self,
        user_query: str,
        retrieved_chunks: List[RetrievedChunk],
        system_prompt: str = None,
    ) -> str:
        """Build complete prompt from query and retrieved context.

        Args:
            user_query: User's question
            retrieved_chunks: Retrieved context chunks
            system_prompt: Override system prompt if provided

        Returns:
            Formatted prompt ready for LLM
        """
        system = system_prompt or self.template.system_prompt
        context = format_retrieved_chunks(retrieved_chunks)

        prompt_parts = [
            system,
            "\n\n",
            self.template.context_header,
            context,
            "\n",
            self.template.query_header,
            "\n",
            user_query,
        ]

        prompt = "".join(prompt_parts)

        logger.info(
            "Prompt built",
            query_length=len(user_query),
            num_chunks=len(retrieved_chunks),
            total_prompt_length=len(prompt),
        )

        return prompt

    def build_minimal_prompt(
        self,
        user_query: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> str:
        """Build minimal prompt with only essential structure.

        Useful for token-constrained scenarios.

        Args:
            user_query: User's question
            retrieved_chunks: Retrieved context chunks

        Returns:
            Minimal formatted prompt
        """
        context = format_retrieved_chunks(retrieved_chunks)

        prompt_parts = [
            "Answer this question using ONLY the provided context.\n",
            "CITE sources using [CITE: doc_name | chunk_id] format.\n\n",
            "Context:\n",
            context,
            "\n",
            "Question: ",
            user_query,
        ]

        prompt = "".join(prompt_parts)

        logger.info(
            "Minimal prompt built",
            query_length=len(user_query),
            num_chunks=len(retrieved_chunks),
            total_prompt_length=len(prompt),
        )

        return prompt

    def get_prompt_stats(self, prompt: str) -> Dict[str, Any]:
        """Get statistics about a prompt.

        Args:
            prompt: Prompt string

        Returns:
            Dict with token count estimates and structure info
        """
        # Rough token estimation (1 token ≈ 4 characters)
        estimated_tokens = len(prompt) // 4

        return {
            "character_count": len(prompt),
            "estimated_tokens": estimated_tokens,
            "lines": len(prompt.split("\n")),
        }
