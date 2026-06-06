"""
Stage 0: Define data models for the ingestion pipeline.

Defines Pydantic models for configuration, documents, and chunks.
Ensures type safety and explicit contracts across the platform.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ChunkingFramework(str, Enum):
    """Supported chunking frameworks."""
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"


class LangChainChunkingMethod(str, Enum):
    """LangChain text splitting methods."""
    RECURSIVE_CHARACTER = "recursive_character"
    CHARACTER = "character"
    TOKEN = "token"
    # MARKDOWN_HEADER = "markdown_header"


class LlamaIndexChunkingMethod(str, Enum):
    """LlamaIndex node parsing methods."""
    SENTENCE_SPLITTER = "sentence_splitter"
    TOKEN_TEXT_SPLITTER = "token_text_splitter"
    SEMANTIC_SPLITTER = "semantic_splitter"
    SENTENCE_WINDOW = "sentence_window"


class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    OLLAMA_NOMIC_EMBED = "nomic-embed-text"
    OLLAMA_MXBAI_EMBED = "mxbai-embed-large"
    OLLAMA_SNOWFLAKE_EMBED = "snowflake-arctic-embed"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    provider: str = Field(default="ollama", description="Embedding provider (ollama, huggingface, openai)")
    model: EmbeddingModel = Field(default=EmbeddingModel.OLLAMA_NOMIC_EMBED, description="Embedding model to use")
    base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    embed_batch_size: int = Field(default=10, gt=0, description="Batch size for embedding computation")


class VectorDBType(str, Enum):
    """Supported vector database types."""
    FAISS = "faiss"
    CHROMA = "chroma"


class ErrorHandlingStrategy(str, Enum):
    """Strategies for handling errors during ingestion."""
    FAIL = "fail"
    SKIP = "skip"
    LOG_AND_SKIP = "log_and_skip"
    LOG_AND_CONTINUE = "log_and_continue"


# LangChain Configuration Models
class RecursiveCharacterConfig(BaseModel):
    """Configuration for LangChain RecursiveCharacterTextSplitter."""
    chunk_size: int = Field(default=512, gt=0, description="Max characters per chunk")
    chunk_overlap: int = Field(default=50, ge=0, description="Character overlap between chunks")
    separators: List[str] = Field(
        default=["\n\n", "\n", " ", ""],
        description="Separators to try in order"
    )
    is_separator_regex: bool = Field(default=False, description="Whether separators are regex patterns")
    keep_separator: bool = Field(default=True, description="Keep separator in output")


class CharacterTextSplitterConfig(BaseModel):
    """Configuration for LangChain CharacterTextSplitter."""
    chunk_size: int = Field(default=512, gt=0, description="Max characters per chunk")
    chunk_overlap: int = Field(default=50, ge=0, description="Character overlap between chunks")
    separator: str = Field(default="\n\n", description="Separator character")
    is_separator_regex: bool = Field(default=False, description="Whether separator is regex pattern")
    keep_separator: bool = Field(default=True, description="Keep separator in output")


class TokenTextSplitterConfig(BaseModel):
    """Configuration for LangChain TokenTextSplitter."""
    chunk_size: int = Field(default=512, gt=0, description="Max tokens per chunk")
    chunk_overlap: int = Field(default=50, ge=0, description="Token overlap between chunks")
    encoding_name: str = Field(default="cl100k_base", description="Tokenizer encoding (cl100k_base, o200k_base)")


class MarkdownHeaderConfig(BaseModel):
    """Configuration for LangChain MarkdownHeaderTextSplitter."""
    headers_to_split_on: List[tuple] = Field(
        default=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
        description="Headers to split on"
    )
    chunk_size: int = Field(default=512, gt=0, description="Max characters per chunk")
    chunk_overlap: int = Field(default=50, ge=0, description="Character overlap between chunks")


class LangChainChunkingConfig(BaseModel):
    """LangChain-specific chunking configuration."""
    method: LangChainChunkingMethod = Field(
        default=LangChainChunkingMethod.RECURSIVE_CHARACTER,
        description="Chunking method to use"
    )
    recursive_character: RecursiveCharacterConfig = RecursiveCharacterConfig()
    character: CharacterTextSplitterConfig = CharacterTextSplitterConfig()
    token: TokenTextSplitterConfig = TokenTextSplitterConfig()
    markdown_header: MarkdownHeaderConfig = MarkdownHeaderConfig()


# LlamaIndex Configuration Models
class SentenceSplitterConfig(BaseModel):
    """Configuration for LlamaIndex SentenceSplitter."""
    chunk_size: int = Field(default=512, gt=0, description="Max tokens per chunk")
    chunk_overlap: int = Field(default=20, ge=0, description="Token overlap between chunks")
    separator: str = Field(default=" ", description="Token separator for word splitting")
    paragraph_separator: str = Field(default="\n\n", description="Paragraph boundary marker")
    secondary_chunking_regex: Optional[str] = Field(default=None, description="Optional regex for fallback chunking")


class LlamaIndexTokenSplitterConfig(BaseModel):
    """Configuration for LlamaIndex TokenTextSplitter.
    
    Note: TokenTextSplitter is currently simplified to use SentenceSplitter implementation.
    """
    chunk_size: int = Field(default=512, gt=0, description="Max tokens per chunk")
    chunk_overlap: int = Field(default=20, ge=0, description="Token overlap between chunks")
    separator: str = Field(default=" ", description="Token separator for word splitting")
    paragraph_separator: str = Field(default="\n\n", description="Paragraph boundary marker")
    secondary_chunking_regex: Optional[str] = Field(default=None, description="Optional regex for fallback chunking")


class SemanticSplitterConfig(BaseModel):
    """Configuration for LlamaIndex SemanticSplitterNodeParser."""
    chunk_size: int = Field(default=512, gt=0, description="Max characters per chunk")
    chunk_overlap: int = Field(default=50, ge=0, description="Character overlap between chunks")
    breakpoint_percentile_threshold: int = Field(
        default=95,
        ge=0,
        le=100,
        description="Percentile threshold for breakpoints (higher=fewer breaks)"
    )
    embedding_model: EmbeddingConfig = Field(default_factory=EmbeddingConfig, description="Embedding model config")


class SentenceWindowConfig(BaseModel):
    """Configuration for LlamaIndex SentenceWindowNodeParser."""
    window_size: int = Field(default=3, gt=0, description="Number of sentences around each sentence")
    window_metadata_key: str = Field(default="window", description="Metadata key for window context")
    original_text_metadata_key: str = Field(default="original_text", description="Metadata key for original text")


class CodeSplitterConfig(BaseModel):
    """Configuration for LlamaIndex CodeSplitter."""
    chunk_size: int = Field(default=512, gt=0, description="Max characters per chunk")
    chunk_overlap: int = Field(default=50, ge=0, description="Character overlap between chunks")
    language: str = Field(default="python", description="Programming language (python, javascript, java, etc.)")


class LlamaIndexChunkingConfig(BaseModel):
    """LlamaIndex-specific chunking configuration."""
    method: LlamaIndexChunkingMethod = Field(
        default=LlamaIndexChunkingMethod.SENTENCE_SPLITTER,
        description="Node parsing method to use"
    )
    sentence_splitter: SentenceSplitterConfig = SentenceSplitterConfig()
    token_text_splitter: LlamaIndexTokenSplitterConfig = LlamaIndexTokenSplitterConfig()
    semantic_splitter: SemanticSplitterConfig = SemanticSplitterConfig()
    sentence_window: SentenceWindowConfig = SentenceWindowConfig()
    code_splitter: CodeSplitterConfig = CodeSplitterConfig()


class ChunkingConfig(BaseModel):
    """Main chunking configuration supporting both frameworks."""
    framework: ChunkingFramework = Field(
        default=ChunkingFramework.LANGCHAIN,
        description="Which framework to use for chunking"
    )
    langchain: LangChainChunkingConfig = LangChainChunkingConfig()
    llamaindex: LlamaIndexChunkingConfig = LlamaIndexChunkingConfig()


class ParsingConfig(BaseModel):
    """Configuration for document parsing."""
    enable_ocr: bool = Field(default=True, description="Enable OCR for images")
    preserve_layout: bool = Field(default=True, description="Preserve document layout")
    extract_tables: bool = Field(default=True, description="Extract tables separately")
    extract_metadata: bool = Field(default=True, description="Extract document metadata")


class SQLiteConfig(BaseModel):
    """Configuration for SQLite storage."""
    db_path: str = Field(description="Path to SQLite database file")
    table_name: str = Field(default="document_chunks", description="Table name for chunks")


class FAISSVectorDBConfig(BaseModel):
    """Configuration for FAISS vector database."""
    index_type: str = Field(
        default="flat",
        description="FAISS index type (flat, ivfpq, hnsw)"
    )
    distance_metric: str = Field(
        default="l2",
        description="Distance metric (l2, inner_product)"
    )
    persist_path: str = Field(
        description="Path to persist FAISS index"
    )


class ChromaVectorDBConfig(BaseModel):
    """Configuration for ChromaDB vector database."""
    collection_name: str = Field(
        default="documents",
        description="ChromaDB collection name"
    )
    persist_directory: str = Field(
        description="Path for ChromaDB persistence"
    )


class VectorDatabaseEmbeddingConfig(BaseModel):
    """Configuration for embeddings used in vector database (independent from chunking embeddings)."""
    provider: str = Field(
        default="ollama",
        description="Embedding provider (ollama, huggingface, openai)"
    )
    model: EmbeddingModel = Field(
        default=EmbeddingModel.OLLAMA_NOMIC_EMBED,
        description="Embedding model to use for vector DB"
    )
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama base URL"
    )
    embed_batch_size: int = Field(
        default=10,
        gt=0,
        description="Batch size for embedding computation"
    )


class VectorDBConfig(BaseModel):
    """Configuration for vector database."""
    enabled: bool = Field(
        default=True,
        description="Whether to use vector database for embeddings"
    )
    type: VectorDBType = Field(
        default=VectorDBType.FAISS,
        description="Vector database type (faiss, chroma)"
    )
    embedding: VectorDatabaseEmbeddingConfig = Field(
        default_factory=VectorDatabaseEmbeddingConfig,
        description="Embedding configuration for vector DB"
    )
    faiss: FAISSVectorDBConfig = Field(
        default_factory=lambda: FAISSVectorDBConfig(persist_path="./data/faiss_index"),
        description="FAISS-specific configuration"
    )
    chroma: ChromaVectorDBConfig = Field(
        default_factory=lambda: ChromaVectorDBConfig(persist_directory="./data/chroma"),
        description="ChromaDB-specific configuration"
    )


class StorageConfig(BaseModel):
    """Configuration for storage backends."""
    sqlite: SQLiteConfig
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)


class RetryConfig(BaseModel):
    """Configuration for retry logic."""
    max_attempts: int = Field(default=3, gt=0, description="Maximum retry attempts")
    backoff_factor: float = Field(default=2.0, gt=1.0, description="Exponential backoff multiplier")
    initial_delay: float = Field(default=1.0, gt=0, description="Initial delay in seconds")


class ErrorHandlingConfig(BaseModel):
    """Configuration for error handling."""
    retry: RetryConfig = RetryConfig()
    on_parse_error: ErrorHandlingStrategy = ErrorHandlingStrategy.LOG_AND_SKIP
    on_unsupported_format: ErrorHandlingStrategy = ErrorHandlingStrategy.SKIP


class IngestionConfig(BaseModel):
    """Main ingestion configuration model."""
    input_path: str = Field(description="Input directory path")
    output_path: str = Field(description="Output directory path")
    supported_formats: List[str] = Field(description="Supported file formats")
    max_file_size: int = Field(default=100, gt=0, description="Max file size in MB")
    parsing: ParsingConfig = ParsingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    storage: StorageConfig
    error_handling: ErrorHandlingConfig = ErrorHandlingConfig()
    logging: Dict[str, Any] = Field(default_factory=dict, description="Logging configuration")
    
    @validator("supported_formats", pre=True)
    def lowercase_formats(cls, v: List[str]) -> List[str]:
        """Convert all formats to lowercase."""
        return [fmt.lower() for fmt in v]


class DocumentMetadata(BaseModel):
    """Metadata associated with a document."""
    file_path: str = Field(description="Original file path")
    file_name: str = Field(description="Original file name")
    file_format: str = Field(description="File format (pdf, docx, etc.)")
    file_size: int = Field(description="File size in bytes")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    author: Optional[str] = Field(default=None, description="Document author if available")
    title: Optional[str] = Field(default=None, description="Document title if available")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class Document(BaseModel):
    """Represents a parsed document."""
    document_id: str = Field(description="Unique document identifier")
    content: str = Field(description="Full document content")
    metadata: DocumentMetadata = Field(description="Document metadata")
    chunk_count: int = Field(default=0, ge=0, description="Number of chunks created from document")


class DocumentChunk(BaseModel):
    """Represents a chunk of text from a document."""
    chunk_id: str = Field(description="Unique chunk identifier")
    document_id: str = Field(description="Reference to parent document")
    content: str = Field(description="Chunk text content")
    chunk_index: int = Field(description="Index of chunk within document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk-level metadata")
    embedding_vector: Optional[List[float]] = Field(
        default=None,
        description="Embedding vector (if computed)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class IngestionResult(BaseModel):
    """Result of an ingestion operation."""
    success: bool = Field(description="Whether ingestion succeeded")
    documents_processed: int = Field(default=0, description="Number of documents processed")
    chunks_created: int = Field(default=0, description="Total chunks created")
    vectors_stored: int = Field(default=0, description="Total vectors stored in vector database")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Errors encountered")
    duration_seconds: float = Field(description="Processing duration in seconds")
    message: str = Field(description="Status message")
