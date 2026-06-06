"""
Chunking strategy implementations for both LangChain and LlamaIndex frameworks.

This module provides a unified interface for different chunking strategies,
allowing users to select and configure their preferred approach.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
from dataclasses import dataclass

from src.ingestion.models import (
    DocumentChunk,
    Document,
    ChunkingFramework,
    LangChainChunkingMethod,
    LlamaIndexChunkingMethod,
    LangChainChunkingConfig,
    LlamaIndexChunkingConfig,
    EmbeddingConfig,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk a document into multiple chunks.

        Args:
            document: Document to chunk

        Returns:
            List of DocumentChunk objects
        """
        pass


# ============================================================================
# LangChain Implementations
# ============================================================================


class RecursiveCharacterSplitter(ChunkingStrategy):
    """
    LangChain RecursiveCharacterTextSplitter implementation.

    Recursively tries multiple separators to find optimal split points.
    """

    def __init__(self, config):
        """Initialize splitter with configuration."""
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.separators = config.separators
        self.is_separator_regex = config.is_separator_regex
        self.keep_separator = config.keep_separator

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document using recursive character splitting."""
        import re
        import uuid

        chunks = []
        chunk_index = 0
        text = document.content

        # Recursively split text
        splits = self._split_text(text, self.separators)

        # Create document chunks
        for split in splits:
            if not split.strip():
                continue

            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document.document_id,
                content=split,
                chunk_index=chunk_index,
                metadata={
                    "strategy": "recursive_character",
                    "source_file": document.metadata.file_name,
                    "method": "langchain",
                },
            )
            chunks.append(chunk)
            chunk_index += 1

        logger.info(
            "Document chunked using RecursiveCharacterSplitter",
            doc_id=document.document_id,
            chunks_created=len(chunks),
        )
        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text by separators."""
        import re

        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, s in enumerate(separators):
            sep_pattern = s if self.is_separator_regex else re.escape(s)
            if re.search(sep_pattern, text):
                separator = s
                new_separators = separators[i + 1 :]
                break

        sep_pattern = separator if self.is_separator_regex else re.escape(separator)
        splits = re.split(sep_pattern, text)

        good_splits = []
        for split in splits:
            if len(split) < self.chunk_size:
                if good_splits and len(good_splits[-1]) + len(split) < self.chunk_size:
                    good_splits[-1] += separator + split
                else:
                    good_splits.append(split)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits)
                    final_chunks.extend(merged)
                    good_splits = []

                if not new_separators:
                    final_chunks.append(split)
                else:
                    other_splits = self._split_text(split, new_separators)
                    final_chunks.extend(other_splits)

        if good_splits:
            merged = self._merge_splits(good_splits)
            final_chunks.extend(merged)

        return [c for c in final_chunks if c.strip()]

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge splits that fit within chunk size."""
        merged = []
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(split) < self.chunk_size:
                current_chunk += split
            else:
                if current_chunk:
                    merged.append(current_chunk)
                current_chunk = split

        if current_chunk:
            merged.append(current_chunk)

        return merged


class CharacterSplitter(ChunkingStrategy):
    """LangChain CharacterTextSplitter implementation."""

    def __init__(self, config):
        """Initialize splitter with configuration."""
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.separator = config.separator
        self.is_separator_regex = config.is_separator_regex
        self.keep_separator = config.keep_separator

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document using character splitting."""
        import re
        import uuid

        chunks = []
        chunk_index = 0
        text = document.content

        # Split by separator
        sep_pattern = self.separator if self.is_separator_regex else re.escape(self.separator)
        splits = re.split(sep_pattern, text)

        # Merge and create chunks
        merged_chunks = self._merge_splits([s for s in splits if s.strip()])

        for chunk_text in merged_chunks:
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document.document_id,
                content=chunk_text,
                chunk_index=chunk_index,
                metadata={
                    "strategy": "character",
                    "source_file": document.metadata.file_name,
                    "method": "langchain",
                },
            )
            chunks.append(chunk)
            chunk_index += 1

        logger.info(
            "Document chunked using CharacterTextSplitter",
            doc_id=document.document_id,
            chunks_created=len(chunks),
        )
        return chunks

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge splits accounting for overlap."""
        merged = []
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(split) < self.chunk_size:
                current_chunk += self.separator + split if current_chunk else split
            else:
                if current_chunk:
                    merged.append(current_chunk)
                current_chunk = split

        if current_chunk:
            merged.append(current_chunk)

        return merged


class TokenSplitter(ChunkingStrategy):
    """LangChain TokenTextSplitter implementation."""

    def __init__(self, config):
        """Initialize splitter with configuration."""
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.encoding_name = config.encoding_name
        self._init_tokenizer()

    def _init_tokenizer(self):
        """Initialize the tokenizer."""
        try:
            import tiktoken

            self.encoding = tiktoken.get_encoding(self.encoding_name)
        except ImportError:
            logger.warning("tiktoken not installed, falling back to simple token counting")
            self.encoding = None

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document using token-based splitting."""
        import uuid

        chunks = []
        chunk_index = 0
        text = document.content

        # Split by tokens
        token_chunks = self._split_by_tokens(text)

        for chunk_text in token_chunks:
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document.document_id,
                content=chunk_text,
                chunk_index=chunk_index,
                metadata={
                    "strategy": "token",
                    "source_file": document.metadata.file_name,
                    "method": "langchain",
                    "encoding": self.encoding_name,
                },
            )
            chunks.append(chunk)
            chunk_index += 1

        logger.info(
            "Document chunked using TokenTextSplitter",
            doc_id=document.document_id,
            chunks_created=len(chunks),
        )
        return chunks

    def _split_by_tokens(self, text: str) -> List[str]:
        """Split text by token count."""
        if not self.encoding:
            # Fallback: approximate tokens as words
            words = text.split()
            chunks = []
            current = []
            current_token_count = 0

            for word in words:
                word_tokens = len(word.split())
                if current_token_count + word_tokens > self.chunk_size:
                    if current:
                        chunks.append(" ".join(current))
                    current = [word]
                    current_token_count = word_tokens
                else:
                    current.append(word)
                    current_token_count += word_tokens

            if current:
                chunks.append(" ".join(current))
            return chunks

        # Use actual tokenizer
        tokens = self.encoding.encode(text)
        chunks = []

        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks


class MarkdownHeaderSplitter(ChunkingStrategy):
    """LangChain MarkdownHeaderTextSplitter implementation."""

    def __init__(self, config):
        """Initialize splitter with configuration."""
        self.headers_to_split_on = config.headers_to_split_on
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document using markdown headers."""
        import re
        import uuid

        chunks = []
        chunk_index = 0
        text = document.content

        # Extract sections by headers
        sections = self._split_by_headers(text)

        for section in sections:
            # Further split sections by chunk size
            sub_chunks = self._split_section(section["content"])

            for sub_chunk in sub_chunks:
                metadata = {
                    "strategy": "markdown_header",
                    "source_file": document.metadata.file_name,
                    "method": "langchain",
                }
                # Add header hierarchy to metadata
                metadata.update(section["headers"])

                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document.document_id,
                    content=sub_chunk,
                    chunk_index=chunk_index,
                    metadata=metadata,
                )
                chunks.append(chunk)
                chunk_index += 1

        logger.info(
            "Document chunked using MarkdownHeaderTextSplitter",
            doc_id=document.document_id,
            chunks_created=len(chunks),
        )
        return chunks

    def _split_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """Split document by markdown headers."""
        import re

        sections = []
        current_section = {"headers": {}, "content": ""}

        lines = text.split("\n")

        for line in lines:
            header_match = False
            for header_marker, header_name in self.headers_to_split_on:
                if line.startswith(header_marker):
                    if current_section["content"].strip():
                        sections.append(current_section)
                    current_section = {"headers": {}, "content": ""}
                    current_section["headers"][header_name] = line.strip("# ").strip()
                    header_match = True
                    break

            if not header_match:
                current_section["content"] += line + "\n"

        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def _split_section(self, text: str) -> List[str]:
        """Split section into chunks by size."""
        chunks = []
        words = text.split()
        current = []

        for word in words:
            current.append(word)
            if len(" ".join(current)) > self.chunk_size:
                chunks.append(" ".join(current[:-1]))
                current = [word]

        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if c.strip()]


# ============================================================================
# LlamaIndex Implementations
# ============================================================================


class SentenceSplitterImpl(ChunkingStrategy):
    """LlamaIndex SentenceSplitter implementation."""

    def __init__(self, config):
        """Initialize splitter with configuration."""
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.separator = getattr(config, "separator", " ")
        self.paragraph_separator = getattr(config, "paragraph_separator", "\n\n")
        self.secondary_chunking_regex = getattr(config, "secondary_chunking_regex", None)

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document using sentence splitting."""
        import uuid
        import re

        chunks = []
        chunk_index = 0
        text = document.content

        # Split into sentences
        sentences = self._split_into_sentences(text)

        # Merge sentences into chunks
        merged_chunks = self._merge_into_chunks(sentences)

        for chunk_text in merged_chunks:
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document.document_id,
                content=chunk_text,
                chunk_index=chunk_index,
                metadata={
                    "strategy": "sentence_splitter",
                    "source_file": document.metadata.file_name,
                    "method": "llamaindex",
                },
            )
            chunks.append(chunk)
            chunk_index += 1

        logger.info(
            "Document chunked using SentenceSplitter",
            doc_id=document.document_id,
            chunks_created=len(chunks),
        )
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re

        # Simple sentence splitter using common separators
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _merge_into_chunks(self, sentences: List[str]) -> List[str]:
        """Merge sentences into chunks respecting chunk_size."""
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


class SemanticSplitterImpl(ChunkingStrategy):
    """LlamaIndex SemanticSplitterNodeParser implementation."""

    def __init__(self, config):
        """Initialize semantic splitter with configuration."""
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.breakpoint_percentile_threshold = config.breakpoint_percentile_threshold
        self.embedding_config = config.embedding_model
        self._embedder = None
        self._init_embedder()

    def _init_embedder(self):
        """Initialize embedding model."""
        try:
            if self.embedding_config.provider == "ollama":
                try:
                    from langchain_community.embeddings import OllamaEmbeddings
                except ImportError:
                    # Fallback to langchain.embeddings if langchain_community not available
                    try:
                        from langchain.embeddings import OllamaEmbeddings
                    except ImportError:
                        logger.warning("OllamaEmbeddings not available")
                        return

                self._embedder = OllamaEmbeddings(
                    model=self.embedding_config.model.value,
                    base_url=self.embedding_config.base_url,
                )
                logger.info(
                    "Initialized Ollama embeddings",
                    model=self.embedding_config.model.value,
                    url=self.embedding_config.base_url,
                )
            else:
                logger.warning(f"Unsupported embedding provider: {self.embedding_config.provider}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize embedder: {str(e)}. Semantic splitting will use simplified approach."
            )
            self._embedder = None

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document using semantic similarity."""
        import uuid

        chunks = []
        chunk_index = 0
        text = document.content

        if not self._embedder:
            logger.warning("Embedder not available, falling back to sentence splitting")
            # Fallback to sentence splitting
            splitter = SentenceSplitterImpl(
                type(
                    "Config",
                    (),
                    {
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "separator": " ",
                        "paragraph_separator": "\n\n",
                        "secondary_chunking_regex": None,
                    },
                )()
            )
            return splitter.chunk(document)

        # Split into sentences first
        sentences = self._split_into_sentences(text)

        # Group sentences by semantic similarity
        grouped = self._group_by_similarity(sentences)

        # Create chunks from groups
        for group in grouped:
            chunk_text = " ".join(group)
            if chunk_text.strip():
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document.document_id,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    metadata={
                        "strategy": "semantic_splitter",
                        "source_file": document.metadata.file_name,
                        "method": "llamaindex",
                        "embedding_model": self.embedding_config.model.value,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

        logger.info(
            "Document chunked using SemanticSplitter",
            doc_id=document.document_id,
            chunks_created=len(chunks),
        )
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _group_by_similarity(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences by semantic similarity."""
        if not sentences or not self._embedder:
            return [[s] for s in sentences]

        try:
            # Get embeddings for sentences
            embeddings = self._embedder.embed_documents(sentences)

            # Simple grouping based on similarity
            groups = [[sentences[0]]]
            current_group_embedding = embeddings[0]

            for i in range(1, len(sentences)):
                similarity = self._cosine_similarity(current_group_embedding, embeddings[i])

                # If similarity above threshold, add to group
                if similarity > (self.breakpoint_percentile_threshold / 100):
                    groups[-1].append(sentences[i])
                    # Update group embedding as average
                    current_group_embedding = self._average_embeddings(
                        [embeddings[j] for j, sent in enumerate(sentences) if sent in groups[-1]]
                    )
                else:
                    # Start new group
                    groups.append([sentences[i]])
                    current_group_embedding = embeddings[i]

            # Merge groups to respect chunk size
            return self._merge_groups(groups)

        except Exception as e:
            logger.warning(f"Error in semantic grouping: {str(e)}, falling back to simple grouping")
            return self._simple_group(sentences)

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _average_embeddings(self, embeddings) -> list:
        """Average multiple embeddings."""
        if not embeddings:
            return []

        dim = len(embeddings[0])
        avg = [0.0] * dim

        for emb in embeddings:
            for i, val in enumerate(emb):
                avg[i] += val

        return [x / len(embeddings) for x in avg]

    def _merge_groups(self, groups: List[List[str]]) -> List[List[str]]:
        """Merge groups to respect chunk size."""
        merged = []
        current = []
        current_size = 0

        for group in groups:
            group_text = " ".join(group)
            group_size = len(group_text)

            if current_size + group_size < self.chunk_size:
                current.extend(group)
                current_size += group_size
            else:
                if current:
                    merged.append(current)
                current = group
                current_size = group_size

        if current:
            merged.append(current)

        return merged

    def _simple_group(self, sentences: List[str]) -> List[List[str]]:
        """Simple grouping by size."""
        groups = []
        current = []
        current_size = 0

        for sent in sentences:
            sent_size = len(sent)
            if current_size + sent_size < self.chunk_size:
                current.append(sent)
                current_size += sent_size
            else:
                if current:
                    groups.append(current)
                current = [sent]
                current_size = sent_size

        if current:
            groups.append(current)

        return groups


class CodeSplitterImpl(ChunkingStrategy):
    """LlamaIndex CodeSplitter implementation."""

    def __init__(self, config):
        """Initialize code splitter with configuration."""
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.language = config.language

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document using code-aware splitting."""
        import uuid

        chunks = []
        chunk_index = 0
        text = document.content

        # Split code by logical boundaries
        code_chunks = self._split_code(text)

        for chunk_text in code_chunks:
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document.document_id,
                content=chunk_text,
                chunk_index=chunk_index,
                metadata={
                    "strategy": "code_splitter",
                    "source_file": document.metadata.file_name,
                    "method": "llamaindex",
                    "language": self.language,
                },
            )
            chunks.append(chunk)
            chunk_index += 1

        logger.info(
            "Document chunked using CodeSplitter",
            doc_id=document.document_id,
            chunks_created=len(chunks),
            language=self.language,
        )
        return chunks

    def _split_code(self, text: str) -> List[str]:
        """Split code by functions, classes, or size."""
        import re

        # Language-specific separators
        separators = self._get_language_separators(self.language)

        chunks = []
        current = ""

        lines = text.split("\n")

        for line in lines:
            is_separator = any(sep in line for sep in separators)

            if is_separator and current and len(current) > self.chunk_size:
                chunks.append(current.strip())
                current = line + "\n"
            else:
                current += line + "\n"

            if len(current) > self.chunk_size:
                chunks.append(current.strip())
                current = ""

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _get_language_separators(self, language: str) -> List[str]:
        """Get language-specific code separators."""
        separators_map = {
            "python": ["^def ", "^class ", "^async def "],
            "javascript": ["^function ", "^class ", "^const .*=.*=>", "^async function "],
            "java": ["^public class ", "^public static ", "^public void ", "^private "],
            "cpp": ["^class ", "^void ", "^int ", "^float "],
            "go": ["^func ", "^type "],
        }

        return separators_map.get(language.lower(), ["^def ", "^class "])


# ============================================================================
# Strategy Factory
# ============================================================================


class ChunkingStrategyFactory:
    """Factory for creating chunking strategy instances."""

    _langchain_strategies = {
        LangChainChunkingMethod.RECURSIVE_CHARACTER: RecursiveCharacterSplitter,
        LangChainChunkingMethod.CHARACTER: CharacterSplitter,
        LangChainChunkingMethod.TOKEN: TokenSplitter,
    }

    _llamaindex_strategies = {
        LlamaIndexChunkingMethod.SENTENCE_SPLITTER: SentenceSplitterImpl,
        LlamaIndexChunkingMethod.TOKEN_TEXT_SPLITTER: SentenceSplitterImpl,  # Simplified
        LlamaIndexChunkingMethod.SEMANTIC_SPLITTER: SemanticSplitterImpl,
        LlamaIndexChunkingMethod.SENTENCE_WINDOW: SentenceSplitterImpl,  # Simplified
    }

    @staticmethod
    def create_strategy(
        framework: ChunkingFramework,
        langchain_config: Optional[LangChainChunkingConfig] = None,
        llamaindex_config: Optional[LlamaIndexChunkingConfig] = None,
    ) -> ChunkingStrategy:
        """
        Create a chunking strategy instance.

        Args:
            framework: Which framework to use
            langchain_config: LangChain configuration (required if framework is LANGCHAIN)
            llamaindex_config: LlamaIndex configuration (required if framework is LLAMAINDEX)

        Returns:
            Configured ChunkingStrategy instance

        Raises:
            ValueError: If configuration is invalid
        """
        if framework == ChunkingFramework.LANGCHAIN:
            if not langchain_config:
                raise ValueError("langchain_config required for LangChain framework")

            strategy_class = ChunkingStrategyFactory._langchain_strategies.get(
                langchain_config.method
            )
            if not strategy_class:
                raise ValueError(f"Unknown LangChain method: {langchain_config.method}")

            # Get the specific config for the method
            if langchain_config.method == LangChainChunkingMethod.RECURSIVE_CHARACTER:
                config = langchain_config.recursive_character
            elif langchain_config.method == LangChainChunkingMethod.CHARACTER:
                config = langchain_config.character
            elif langchain_config.method == LangChainChunkingMethod.TOKEN:
                config = langchain_config.token
            else:
                raise ValueError(f"Unknown LangChain method: {langchain_config.method}")

            logger.info("Created LangChain chunking strategy", method=langchain_config.method)
            return strategy_class(config)

        elif framework == ChunkingFramework.LLAMAINDEX:
            if not llamaindex_config:
                raise ValueError("llamaindex_config required for LlamaIndex framework")

            strategy_class = ChunkingStrategyFactory._llamaindex_strategies.get(
                llamaindex_config.method
            )
            if not strategy_class:
                raise ValueError(f"Unknown LlamaIndex method: {llamaindex_config.method}")

            # Get the specific config for the method
            if llamaindex_config.method == LlamaIndexChunkingMethod.SENTENCE_SPLITTER:
                config = llamaindex_config.sentence_splitter
            elif llamaindex_config.method == LlamaIndexChunkingMethod.TOKEN_TEXT_SPLITTER:
                config = llamaindex_config.token_text_splitter
            elif llamaindex_config.method == LlamaIndexChunkingMethod.SEMANTIC_SPLITTER:
                config = llamaindex_config.semantic_splitter
            elif llamaindex_config.method == LlamaIndexChunkingMethod.SENTENCE_WINDOW:
                config = llamaindex_config.sentence_window
            else:
                raise ValueError(f"Unknown LlamaIndex method: {llamaindex_config.method}")

            logger.info("Created LlamaIndex chunking strategy", method=llamaindex_config.method)
            return strategy_class(config)

        else:
            raise ValueError(f"Unknown framework: {framework}")
