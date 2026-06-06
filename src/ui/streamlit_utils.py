"""
Utility functions for the Streamlit UI.

Provides helper functions for:
- Configuration management
- File operations
- Data formatting
- Error handling
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import streamlit as st
from src.ingestion.models import ChunkingFramework

from src.ingestion.models import DocumentChunk, IngestionResult
from src.utils.logger import get_logger


logger = get_logger(__name__, level="INFO")


def format_file_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format duration to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_file_icon(file_extension: str) -> str:
    """Get emoji icon for file type."""
    icons = {
        "pdf": "📄",
        "docx": "📃",
        "doc": "📃",
        "txt": "📝",
        "csv": "📊",
        "ppt": "🎯",
        "pptx": "🎯",
    }
    return icons.get(file_extension.lower(), "📎")


def format_chunk_preview(content: str, max_length: int = 200) -> str:
    """Format chunk content for display."""
    if len(content) > max_length:
        return content[:max_length] + "..."
    return content


def display_ingestion_result(result: IngestionResult) -> None:
    """Display ingestion result in Streamlit."""
    if result.success:
        st.success("✅ Ingestion completed successfully!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents", result.documents_processed)
        with col2:
            st.metric("Chunks Created", result.chunks_created)
        with col3:
            st.metric("Duration", format_duration(result.duration_seconds))
    
    else:
        st.error(f"❌ Ingestion failed: {result.message}")
        
        if result.errors:
            st.subheader("Errors:")
            for error in result.errors:
                st.error(f"- {error.get('file', 'Unknown')}: {error.get('error')}")


def display_chunks_table(chunks: List[DocumentChunk]) -> None:
    """Display chunks as table in Streamlit."""
    if not chunks:
        st.info("No chunks to display")
        return
    
    data = {
        "Chunk ID": [c.chunk_id[:8] + "..." for c in chunks],
        "Document ID": [c.document_id[:8] + "..." for c in chunks],
        "Index": [str(c.chunk_index) for c in chunks],
        "Content Length": [str(len(c.content)) for c in chunks],
        "Preview": [format_chunk_preview(c.content) for c in chunks],
    }
    
    st.dataframe(data, hide_index=True)


def display_configuration_details(config) -> None:
    """Display configuration details in Streamlit."""
    
    # Ingestion settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ingestion Settings")
        st.write(f"**Input Path:** `{config.input_path}`")
        st.write(f"**Output Path:** `{config.output_path}`")
        st.write(f"**Max File Size:** {config.max_file_size} MB")
    
    with col2:
        st.subheader("Supported Formats")
        for fmt in config.supported_formats:
            st.write(f"- {get_file_icon(fmt)} {fmt.upper()}")
    
    # Chunking settings
    st.subheader("Chunking Configuration")
    
    chunking = config.chunking
    col1, col2, col3 = st.columns(3)
    
    # Get current framework and method
    if chunking.framework == ChunkingFramework.LANGCHAIN:
        method = chunking.langchain.method.value
        chunk_size = chunking.langchain.recursive_character.chunk_size
        chunk_overlap = chunking.langchain.recursive_character.chunk_overlap
    else:
        method = chunking.llamaindex.method.value
        chunk_size = chunking.llamaindex.sentence_splitter.chunk_size
        chunk_overlap = chunking.llamaindex.sentence_splitter.chunk_overlap
    
    with col1:
        st.metric("Framework", chunking.framework.value)
    with col2:
        st.metric("Method", method)
    with col3:
        st.metric("Chunk Size", f"{chunk_size} chars")
    
    st.metric("Overlap", f"{chunk_overlap} chars")
    
    # Parsing settings
    st.subheader("Parsing Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("OCR Enabled", "✅" if config.parsing.enable_ocr else "❌")
    with col2:
        st.metric("Preserve Layout", "✅" if config.parsing.preserve_layout else "❌")
    with col3:
        st.metric("Extract Tables", "✅" if config.parsing.extract_tables else "❌")
    with col4:
        st.metric("Extract Metadata", "✅" if config.parsing.extract_metadata else "❌")


def create_sample_documents() -> None:
    """Create sample documents for testing."""
    sample_dir = Path("./sample_documents")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample text file
    txt_file = sample_dir / "sample.txt"
    if not txt_file.exists():
        txt_file.write_text("""
Sample Document for Testing

This is a sample text document created for testing the ingestion pipeline.

Section 1: Introduction
This section introduces the concept of document ingestion. Document ingestion is the 
process of acquiring, parsing, and storing documents in a structured format that can 
be used for retrieval and analysis.

Section 2: Benefits
- Enables full-text search
- Supports multimodal content
- Preserves document structure
- Maintains metadata

Section 3: Use Cases
Document ingestion is useful for:
1. Knowledge base creation
2. Document search and discovery
3. Information extraction
4. Content analysis

Section 4: Conclusion
Document ingestion is a critical component of any document management system.
It ensures that all content is properly formatted and accessible for downstream 
processing and retrieval.

The end of the sample document.
""")
    
    # Sample CSV file
    csv_file = sample_dir / "sample.csv"
    if not csv_file.exists():
        csv_file.write_text("""ID,Name,Category,Value,Date
1,Product A,Electronics,999.99,2024-01-15
2,Product B,Clothing,49.99,2024-01-16
3,Product C,Books,19.99,2024-01-17
4,Product D,Electronics,1299.99,2024-01-18
5,Product E,Clothing,79.99,2024-01-19
6,Product F,Books,34.99,2024-01-20
7,Product G,Electronics,599.99,2024-01-21
8,Product H,Clothing,89.99,2024-01-22
9,Product I,Books,24.99,2024-01-23
10,Product J,Electronics,1999.99,2024-01-24
""")
    
    logger.info(f"Sample documents created in {sample_dir}")
    return sample_dir


def load_json_logs(log_file: Path, limit: int = 50) -> List[Dict[str, Any]]:
    """Load JSON logs from file."""
    if not log_file.exists():
        return []
    
    logs = []
    try:
        for line in log_file.read_text().split('\n')[-limit:]:
            if line.strip():
                try:
                    log_entry = json.loads(line)
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        logger.error(f"Error loading logs: {str(e)}", exc_info=True)
    
    return logs


def display_status_banner(status: str, message: str) -> None:
    """Display a status banner."""
    if status == "success":
        st.success(f"✅ {message}")
    elif status == "error":
        st.error(f"❌ {message}")
    elif status == "warning":
        st.warning(f"⚠️ {message}")
    else:
        st.info(f"ℹ️ {message}")


def export_chunks_csv(chunks: List[DocumentChunk], filename: str = "chunks.csv") -> bytes:
    """Export chunks to CSV format."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["chunk_id", "document_id", "chunk_index", "content_length", "content"])
    
    # Write data
    for chunk in chunks:
        writer.writerow([
            chunk.chunk_id,
            chunk.document_id,
            chunk.chunk_index,
            len(chunk.content),
            chunk.content,
        ])
    
    return output.getvalue().encode()


def export_chunks_json(chunks: List[DocumentChunk]) -> str:
    """Export chunks to JSON format."""
    data = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "chunk_index": c.chunk_index,
            "content": c.content,
            "metadata": c.metadata,
            "created_at": c.created_at.isoformat(),
        }
        for c in chunks
    ]
    
    return json.dumps(data, indent=2)
