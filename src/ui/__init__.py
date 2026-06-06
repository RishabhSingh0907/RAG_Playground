"""
UI module for the RAG platform.

Provides Streamlit utilities and helper functions for the web interface.
"""

from src.ui.streamlit_utils import (
    format_file_size,
    format_duration,
    get_file_icon,
    format_chunk_preview,
    display_ingestion_result,
    display_chunks_table,
    display_configuration_details,
    create_sample_documents,
    load_json_logs,
    display_status_banner,
    export_chunks_csv,
    export_chunks_json,
)

__all__ = [
    "format_file_size",
    "format_duration",
    "get_file_icon",
    "format_chunk_preview",
    "display_ingestion_result",
    "display_chunks_table",
    "display_configuration_details",
    "create_sample_documents",
    "load_json_logs",
    "display_status_banner",
    "export_chunks_csv",
    "export_chunks_json",
]
