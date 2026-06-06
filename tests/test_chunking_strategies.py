"""
Unit tests for chunking strategies.

Tests both LangChain and LlamaIndex implementations with various configurations.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from src.ingestion.models import (
    Document,
    DocumentMetadata,
    ChunkingFramework,
    LangChainChunkingMethod,
    LlamaIndexChunkingMethod,
    LangChainChunkingConfig,
    LlamaIndexChunkingConfig,
    RecursiveCharacterConfig,
    CharacterTextSplitterConfig,
    TokenTextSplitterConfig,
    SentenceSplitterConfig,
    SemanticSplitterConfig,
    EmbeddingConfig,
)
from src.ingestion.chunking_strategies import (
    RecursiveCharacterSplitter,
    CharacterSplitter,
    TokenSplitter,
    MarkdownHeaderSplitter,
    SentenceSplitterImpl,
    SemanticSplitterImpl,
    CodeSplitterImpl,
    ChunkingStrategyFactory,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    metadata = DocumentMetadata(
        file_path="/data/test.txt",
        file_name="test.txt",
        file_format="txt",
        file_size=1024,
    )
    
    content = """
    This is a test document. It contains multiple sentences.
    This is the second sentence. And this is the third.
    
    Here is a new paragraph. It has its own content.
    And another sentence to complete the paragraph.
    
    Final paragraph with some text. Just for testing purposes.
    This completes our sample document structure.
    """
    
    return Document(
        document_id=str(uuid4()),
        content=content,
        metadata=metadata,
    )


@pytest.fixture
def code_document():
    """Create a sample code document for testing."""
    metadata = DocumentMetadata(
        file_path="/data/test.py",
        file_name="test.py",
        file_format="py",
        file_size=1024,
    )
    
    content = """
def hello_world():
    print("Hello, World!")
    return True

class DataProcessor:
    def __init__(self, name):
        self.name = name
    
    def process(self, data):
        return data.upper()

def main():
    processor = DataProcessor("test")
    result = processor.process("hello")
    print(result)
"""
    
    return Document(
        document_id=str(uuid4()),
        content=content,
        metadata=metadata,
    )


@pytest.fixture
def markdown_document():
    """Create a sample markdown document for testing."""
    metadata = DocumentMetadata(
        file_path="/data/test.md",
        file_name="test.md",
        file_format="md",
        file_size=1024,
    )
    
    content = """
# Main Title

This is the introduction paragraph.

## Section One

Content for section one. This contains useful information.

### Subsection

More detailed content goes here.

## Section Two

Different content for section two.
"""
    
    return Document(
        document_id=str(uuid4()),
        content=content,
        metadata=metadata,
    )


# ============================================================================
# LangChain Tests
# ============================================================================


class TestRecursiveCharacterSplitter:
    """Tests for RecursiveCharacterSplitter."""
    
    def test_basic_splitting(self, sample_document):
        """Test basic recursive character splitting."""
        config = RecursiveCharacterConfig(
            chunk_size=100,
            chunk_overlap=10,
        )
        splitter = RecursiveCharacterSplitter(config)
        chunks = splitter.chunk(sample_document)
        
        assert len(chunks) > 0
        assert all(len(c.content) <= config.chunk_size + 50 for c in chunks)
        assert all(c.document_id == sample_document.document_id for c in chunks)
        assert all(c.metadata["strategy"] == "recursive_character" for c in chunks)
    
    def test_chunk_indexing(self, sample_document):
        """Test that chunks are properly indexed."""
        config = RecursiveCharacterConfig(chunk_size=100)
        splitter = RecursiveCharacterSplitter(config)
        chunks = splitter.chunk(sample_document)
        
        chunk_indices = [c.chunk_index for c in chunks]
        assert chunk_indices == list(range(len(chunks)))
    
    def test_metadata_preservation(self, sample_document):
        """Test that metadata is preserved in chunks."""
        config = RecursiveCharacterConfig(chunk_size=100)
        splitter = RecursiveCharacterSplitter(config)
        chunks = splitter.chunk(sample_document)
        
        for chunk in chunks:
            assert chunk.metadata["source_file"] == "test.txt"
            assert chunk.metadata["method"] == "langchain"


class TestCharacterSplitter:
    """Tests for CharacterSplitter."""
    
    def test_basic_splitting(self, sample_document):
        """Test basic character splitting."""
        config = CharacterTextSplitterConfig(
            chunk_size=150,
            chunk_overlap=20,
        )
        splitter = CharacterSplitter(config)
        chunks = splitter.chunk(sample_document)
        
        assert len(chunks) > 0
        assert all(len(c.content) <= config.chunk_size + 50 for c in chunks)
    
    def test_custom_separator(self, sample_document):
        """Test with custom separator."""
        config = CharacterTextSplitterConfig(
            chunk_size=100,
            separator=". ",
        )
        splitter = CharacterSplitter(config)
        chunks = splitter.chunk(sample_document)
        
        assert len(chunks) > 0


class TestTokenSplitter:
    """Tests for TokenSplitter."""
    
    def test_token_splitting(self, sample_document):
        """Test token-based splitting."""
        config = TokenTextSplitterConfig(
            chunk_size=50,
            chunk_overlap=10,
        )
        splitter = TokenSplitter(config)
        chunks = splitter.chunk(sample_document)
        
        assert len(chunks) > 0
        assert all(c.metadata["encoding"] == "cl100k_base" for c in chunks)


class TestMarkdownHeaderSplitter:
    """Tests for MarkdownHeaderSplitter."""
    
    def test_markdown_splitting(self, markdown_document):
        """Test markdown header-aware splitting."""
        config_list = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        from src.ingestion.models import MarkdownHeaderConfig
        config = MarkdownHeaderConfig(
            headers_to_split_on=config_list,
            chunk_size=200,
        )
        splitter = MarkdownHeaderSplitter(config)
        chunks = splitter.chunk(markdown_document)
        
        assert len(chunks) > 0
        # Check that at least some chunks have header metadata
        headers_found = any("Header" in str(c.metadata.keys()) for c in chunks)
        assert headers_found


# ============================================================================
# LlamaIndex Tests
# ============================================================================


class TestSentenceSplitter:
    """Tests for SentenceSplitterImpl."""
    
    def test_sentence_splitting(self, sample_document):
        """Test sentence-based splitting."""
        config = SentenceSplitterConfig(
            chunk_size=200,
            chunk_overlap=20,
        )
        splitter = SentenceSplitterImpl(config)
        chunks = splitter.chunk(sample_document)
        
        assert len(chunks) > 0
        assert all(c.metadata["strategy"] == "sentence_splitter" for c in chunks)
        assert all(c.metadata["method"] == "llamaindex" for c in chunks)
    
    def test_empty_splits_excluded(self, sample_document):
        """Test that empty splits are excluded."""
        config = SentenceSplitterConfig(chunk_size=50)
        splitter = SentenceSplitterImpl(config)
        chunks = splitter.chunk(sample_document)
        
        assert all(c.content.strip() for c in chunks)


class TestCodeSplitter:
    """Tests for CodeSplitterImpl."""
    
    def test_python_code_splitting(self, code_document):
        """Test Python code splitting."""
        config = type(
            "Config",
            (),
            {
                "chunk_size": 200,
                "chunk_overlap": 20,
                "language": "python",
            },
        )()
        
        from src.ingestion.models import CodeSplitterConfig
        code_config = CodeSplitterConfig(
            chunk_size=200,
            chunk_overlap=20,
            language="python",
        )
        splitter = CodeSplitterImpl(code_config)
        chunks = splitter.chunk(code_document)
        
        assert len(chunks) > 0
        assert all(c.metadata["language"] == "python" for c in chunks)


# ============================================================================
# Factory Tests
# ============================================================================


class TestChunkingStrategyFactory:
    """Tests for ChunkingStrategyFactory."""
    
    def test_create_langchain_recursive(self, sample_document):
        """Test creating LangChain RecursiveCharacterSplitter."""
        config = LangChainChunkingConfig(
            method=LangChainChunkingMethod.RECURSIVE_CHARACTER,
        )
        
        strategy = ChunkingStrategyFactory.create_strategy(
            framework=ChunkingFramework.LANGCHAIN,
            langchain_config=config,
        )
        
        assert isinstance(strategy, RecursiveCharacterSplitter)
        chunks = strategy.chunk(sample_document)
        assert len(chunks) > 0
    
    def test_create_langchain_character(self, sample_document):
        """Test creating LangChain CharacterSplitter."""
        config = LangChainChunkingConfig(
            method=LangChainChunkingMethod.CHARACTER,
        )
        
        strategy = ChunkingStrategyFactory.create_strategy(
            framework=ChunkingFramework.LANGCHAIN,
            langchain_config=config,
        )
        
        assert isinstance(strategy, CharacterSplitter)
    
    def test_create_langchain_token(self, sample_document):
        """Test creating LangChain TokenSplitter."""
        config = LangChainChunkingConfig(
            method=LangChainChunkingMethod.TOKEN,
        )
        
        strategy = ChunkingStrategyFactory.create_strategy(
            framework=ChunkingFramework.LANGCHAIN,
            langchain_config=config,
        )
        
        assert isinstance(strategy, TokenSplitter)
    
    def test_create_llamaindex_sentence(self, sample_document):
        """Test creating LlamaIndex SentenceSplitter."""
        config = LlamaIndexChunkingConfig(
            method=LlamaIndexChunkingMethod.SENTENCE_SPLITTER,
        )
        
        strategy = ChunkingStrategyFactory.create_strategy(
            framework=ChunkingFramework.LLAMAINDEX,
            llamaindex_config=config,
        )
        
        assert isinstance(strategy, SentenceSplitterImpl)
    
    def test_invalid_framework(self):
        """Test that invalid framework raises error."""
        with pytest.raises(ValueError):
            ChunkingStrategyFactory.create_strategy(
                framework=type("InvalidFramework", (), {"value": "invalid"})(),
            )
    
    def test_missing_config(self):
        """Test that missing configuration raises error."""
        with pytest.raises(ValueError):
            ChunkingStrategyFactory.create_strategy(
                framework=ChunkingFramework.LANGCHAIN,
                langchain_config=None,
            )


# ============================================================================
# Integration Tests
# ============================================================================


class TestChunkingIntegration:
    """Integration tests for complete chunking workflows."""
    
    def test_langchain_recursive_workflow(self, sample_document):
        """Test complete LangChain recursive workflow."""
        config = LangChainChunkingConfig(
            method=LangChainChunkingMethod.RECURSIVE_CHARACTER,
            recursive_character=RecursiveCharacterConfig(
                chunk_size=150,
                chunk_overlap=30,
            ),
        )
        
        strategy = ChunkingStrategyFactory.create_strategy(
            framework=ChunkingFramework.LANGCHAIN,
            langchain_config=config,
        )
        
        chunks = strategy.chunk(sample_document)
        
        assert len(chunks) > 0
        assert all(c.document_id == sample_document.document_id for c in chunks)
        assert all(c.content.strip() for c in chunks)
        assert all(c.chunk_id for c in chunks)
    
    def test_llamaindex_semantic_workflow(self, sample_document):
        """Test complete LlamaIndex semantic workflow (without actual embeddings)."""
        config = LlamaIndexChunkingConfig(
            method=LlamaIndexChunkingMethod.SEMANTIC_SPLITTER,
            semantic_splitter=SemanticSplitterConfig(
                chunk_size=150,
                chunk_overlap=30,
            ),
        )
        
        strategy = ChunkingStrategyFactory.create_strategy(
            framework=ChunkingFramework.LLAMAINDEX,
            llamaindex_config=config,
        )
        
        chunks = strategy.chunk(sample_document)
        
        # Should fall back to sentence splitter if embedder not available
        assert len(chunks) > 0
    
    def test_multiple_document_chunking(self, sample_document, code_document):
        """Test chunking multiple documents."""
        config = LangChainChunkingConfig(
            method=LangChainChunkingMethod.RECURSIVE_CHARACTER,
        )
        
        strategy = ChunkingStrategyFactory.create_strategy(
            framework=ChunkingFramework.LANGCHAIN,
            langchain_config=config,
        )
        
        chunks1 = strategy.chunk(sample_document)
        chunks2 = strategy.chunk(code_document)
        
        assert len(chunks1) > 0
        assert len(chunks2) > 0
        
        # Verify documents are properly separated
        assert all(c.document_id == sample_document.document_id for c in chunks1)
        assert all(c.document_id == code_document.document_id for c in chunks2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
