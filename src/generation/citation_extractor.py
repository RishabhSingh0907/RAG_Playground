"""Citation extraction and processing from LLM responses.

Responsibilities:
- Parse LLM response for citation patterns
- Extract document and chunk references
- Link citations back to original chunk metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import re

from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class Citation:
    """A citation to a source document/chunk."""

    doc_name: str
    chunk_id: str
    position: int  # Character position in response where citation appears
    citation_text: str  # The text that was cited
    chunk_content: Optional[str] = None  # Full chunk content (added later)
    chunk_metadata: Optional[Dict[str, Any]] = None  # Chunk metadata


class CitationExtractor:
    """Extract and process citations from LLM responses."""

    # Pattern to match citations: [CITE: doc_name | chunk_id]
    CITATION_PATTERN = r"\[CITE:\s*([^\|]+?)\s*\|\s*chunk_(\w+)\s*\]"

    # Pattern to match source citations: [SOURCE: doc_name | chunk_id]
    SOURCE_PATTERN = r"\[SOURCE:\s*([^\|]+?)\s*\|\s*chunk_(\w+)\s*\]"

    def __init__(self):
        """Initialize citation extractor."""
        self.compiled_cite_pattern = re.compile(self.CITATION_PATTERN, re.IGNORECASE)
        self.compiled_source_pattern = re.compile(
            self.SOURCE_PATTERN, re.IGNORECASE
        )

    def extract_citations(self, response_text: str) -> List[Citation]:
        """Extract all citations from LLM response.

        Supports both [CITE: doc | chunk_id] and [SOURCE: doc | chunk_id] formats.

        Args:
            response_text: LLM response text

        Returns:
            List of citations found in the response
        """
        citations = []

        # Search for CITE patterns
        for match in self.compiled_cite_pattern.finditer(response_text):
            doc_name = match.group(1).strip()
            chunk_id = match.group(2).strip()
            position = match.start()

            citation = Citation(
                doc_name=doc_name,
                chunk_id=chunk_id,
                position=position,
                citation_text=match.group(0),
            )
            citations.append(citation)

        # Search for SOURCE patterns (if no CITE patterns found)
        if not citations:
            for match in self.compiled_source_pattern.finditer(response_text):
                doc_name = match.group(1).strip()
                chunk_id = match.group(2).strip()
                position = match.start()

                citation = Citation(
                    doc_name=doc_name,
                    chunk_id=chunk_id,
                    position=position,
                    citation_text=match.group(0),
                )
                citations.append(citation)

        # Sort by position in text
        citations.sort(key=lambda c: c.position)

        logger.info(
            "Citations extracted from response",
            num_citations=len(citations),
            response_length=len(response_text),
        )

        return citations

    def clean_response(self, response_text: str) -> str:
        """Remove citation tags from response text for cleaner display.

        Args:
            response_text: Original response text

        Returns:
            Response with citation tags removed
        """
        cleaned = self.compiled_cite_pattern.sub("", response_text)
        cleaned = self.compiled_source_pattern.sub("", cleaned)

        # Clean up extra spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def link_citations_to_chunks(
        self,
        citations: List[Citation],
        chunk_dict: Dict[str, Any],
    ) -> List[Citation]:
        """Link citations to actual chunk content and metadata.

        Args:
            citations: List of extracted citations
            chunk_dict: Dict mapping chunk_id → chunk data

        Returns:
            Citations with chunk_content and chunk_metadata populated
        """
        linked_citations = []

        for citation in citations:
            chunk_id = citation.chunk_id
            chunk_data = chunk_dict.get(chunk_id)

            if chunk_data:
                citation.chunk_content = chunk_data.get("content")
                citation.chunk_metadata = chunk_data.get("metadata", {})
                linked_citations.append(citation)
            else:
                logger.warning(
                    "Citation chunk not found in chunk dictionary",
                    chunk_id=chunk_id,
                    doc_name=citation.doc_name,
                )

        logger.info(
            "Citations linked to chunks",
            total_citations=len(citations),
            linked_citations=len(linked_citations),
        )

        return linked_citations

    def format_response_with_tags(
        self,
        response_text: str,
        citations: List[Citation],
    ) -> str:
        """Format response with HTML-like tags for UI rendering.

        Args:
            response_text: Original response text
            citations: List of citations in response

        Returns:
            Response with citation tags marked (for UI to render)
        """
        if not citations:
            return response_text

        # Build response with citation markers
        formatted = response_text

        # Replace each citation text with a tagged version
        for citation in sorted(citations, key=lambda c: c.position, reverse=True):
            # Create tag: <cite doc="doc_name" chunk="chunk_id">text</cite>
            tag = f'<cite data-doc="{citation.doc_name}" data-chunk="{citation.chunk_id}">{citation.citation_text}</cite>'
            formatted = formatted.replace(citation.citation_text, tag, 1)

        return formatted


def build_chunk_dict_from_retrieved(chunks: List[Any]) -> Dict[str, Any]:
    """Build a dictionary of chunk data for citation linking.

    Args:
        chunks: List of RetrievedChunk objects

    Returns:
        Dict mapping chunk_id → {content, metadata}
    """
    chunk_dict = {}

    for chunk in chunks:
        chunk_dict[chunk.chunk_id] = {
            "content": chunk.content,
            "metadata": {
                "doc_name": chunk.metadata.get("file_name", chunk.document_id),
                "document_id": chunk.document_id,
                "score": chunk.score,
                "rank": chunk.rank,
                "source": chunk.source,
                **chunk.metadata,
            },
        }

    return chunk_dict
