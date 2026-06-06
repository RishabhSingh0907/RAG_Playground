"""
Streamlit UI for Reusable Agentic Multimodal RAG Platform.

Provides web interface for:
- File upload and ingestion
- Configuration management
- Chunk browsing and statistics
- Real-time monitoring
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import json
import warnings

# Suppress known deprecation warnings from transformers library
# The __path__ deprecation warning appears for all image processing modules
warnings.filterwarnings("ignore", message=".*Accessing `__path__` from.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.ingestion.ingestion_manager import IngestionManager
from src.ingestion.models import (
    ChunkingFramework,
    LangChainChunkingMethod,
    LlamaIndexChunkingMethod,
    LangChainChunkingConfig,
    LlamaIndexChunkingConfig,
    VectorDBType,
    FAISSVectorDBConfig,
    ChromaVectorDBConfig,
    VectorDBConfig,
    EmbeddingModel,
)
from src.retrieval.pipeline_v2 import (
    RetrievalPipelineV2,
    RetrievalSettingsV2,
    SearchMode,
    load_retrieval_settings_v2,
)
from src.utils.logger import get_logger


# Configure Streamlit page
st.set_page_config(
    page_title="RAG Platform - Data Ingestion",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = get_logger(__name__, level="INFO")

# Apply custom CSS for sidebar styling
st.markdown(
    """
    <style>
    /* Reduce sidebar font size */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
    }
    
    /* Sidebar text - reduce font size */
    [data-testid="stSidebar"] * {
        font-size: 13px !important;
    }
    
    /* Sidebar section headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 15px !important;
        margin-top: 10px !important;
        margin-bottom: 8px !important;
    }
    
    /* Sidebar buttons and selectboxes */
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] select {
        font-size: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "manager" not in st.session_state:
    st.session_state.manager = None
if "ingestion_result" not in st.session_state:
    st.session_state.ingestion_result = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "retrieval_results" not in st.session_state:
    st.session_state.retrieval_results = []
if "retrieval_query" not in st.session_state:
    st.session_state.retrieval_query = ""
if "retrieval_settings" not in st.session_state:
    st.session_state.retrieval_settings = load_retrieval_settings_v2("src/config/retrieval_config.yaml")


def initialize_manager():
    """Initialize IngestionManager from config."""
    try:
        config_path = "src/config/ingestion_config.yaml"
        manager = IngestionManager(config_path)
        st.session_state.manager = manager
        logger.info("IngestionManager initialized successfully")
        return manager
    except Exception as e:
        logger.error(f"Failed to initialize IngestionManager: {str(e)}", exc_info=True)
        st.error(f"❌ Failed to initialize manager: {str(e)}")
        return None


def render_sidebar():
    """Render sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")
    
    manager = st.session_state.manager
    if manager is None:
        st.sidebar.warning("Manager not initialized. Initialize first.")
        return None
    
    # Display current configuration
    with st.sidebar.expander("📋 Current Chunking Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        # Determine current framework and method
        framework = manager.config.chunking.framework
        if framework == ChunkingFramework.LANGCHAIN:
            method = manager.config.chunking.langchain.method.value
            chunk_size = manager.config.chunking.langchain.recursive_character.chunk_size
            chunk_overlap = manager.config.chunking.langchain.recursive_character.chunk_overlap
        else:  # LLAMAINDEX
            method = manager.config.chunking.llamaindex.method.value
            chunk_size = manager.config.chunking.llamaindex.sentence_splitter.chunk_size
            chunk_overlap = manager.config.chunking.llamaindex.sentence_splitter.chunk_overlap
        
        with col1:
            st.metric("Framework", framework.value.upper())
            st.metric("Method", method.replace("_", " ").title())
        
        with col2:
            st.metric("Chunk Size", f"{chunk_size} chars")
            st.metric("Max File Size", f"{manager.config.max_file_size} MB")
    
    # Display database info
    stats = manager.get_stats()
    with st.sidebar.expander("📊 Database Statistics", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Chunks", stats["total_chunks"])
        
        with col2:
            st.metric("Database Path", Path(stats["db_path"]).name)
        
        st.write("**Supported Formats:**")
        st.write(", ".join([f.upper() for f in stats["supported_formats"]]))
    
    # Display vector database configuration
    vector_db_status = manager.get_vector_db_status()
    with st.sidebar.expander("🗄️ Embedding Configurations", expanded=True):
        if vector_db_status["enabled"]:
            status_color = "🟢" if vector_db_status["initialized"] else "🔴"
            st.write(f"**Status:** {status_color} {'Initialized' if vector_db_status['initialized'] else 'Failed to Initialize'}")
            
            if vector_db_status["initialized"]:
                st.write(f"**Vector DB:** {vector_db_status['type'].upper()}")
                st.write(f"**Embedding Provider:** {vector_db_status['embedding_provider']}")
                st.write(f"**Embedding Model:** {vector_db_status['embedding_model']}")
                
                if vector_db_status["type"] == "chroma":
                    st.write(f"**Collection:** {vector_db_status['chroma_collection']}")
                elif vector_db_status["type"] == "faiss":
                    st.write(f"**Index Type:** {vector_db_status['faiss_index_type']}")
                
                st.success("✅ Vector embeddings will be stored during ingestion")
            else:
                st.error("❌ Vector database failed to initialize. Check logs for details.")
                st.warning("Chunks will be stored in SQLite only, without embeddings.")
        else:
            st.warning("⚠️ Vector database is disabled in configuration")
    
    # Vector Database Diagnostics
    with st.sidebar.expander("🔍 Vector DB Diagnostics", expanded=False):
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🔄", key="refresh_diag", help="Refresh diagnostics"):
                st.rerun()
        
        with col1:
            st.write("")
        
        if vector_db_status["enabled"] and vector_db_status["initialized"]:
            # Prepare diagnostics data
            db_type = vector_db_status["type"]
            
            if "faiss_diagnostics" in vector_db_status and db_type == "faiss":
                diag = vector_db_status["faiss_diagnostics"]
                
                # Check if there was an error getting diagnostics
                if "faiss_diagnostics_error" in vector_db_status:
                    st.error(f"Failed to get FAISS diagnostics: {vector_db_status['faiss_diagnostics_error']}")
                elif diag and isinstance(diag, dict) and "index_initialized" in diag:
                    st.write("**FAISS Index Status:**")
                    
                    status_data = {
                        "Index Initialized": "✅" if diag["index_initialized"] else "❌",
                        "Total Vectors": diag["total_vectors"],
                        "Embedding Dimension": diag["embedding_dimension"],
                        "Chunk ID Mappings": diag["chunk_id_mappings"],
                        "Metadata Entries": diag["metadata_entries"],
                    }
                    
                    for key, value in status_data.items():
                        st.write(f"• **{key}:** {value}")
                    
                    st.write("**File Persistence:**")
                    file_status = {
                        "Index File": "✅ Exists" if diag["persist_path_exists"] else "❌ Missing",
                        "Metadata File": "✅ Exists" if diag["metadata_path_exists"] else "❌ Missing",
                        "ID Map File": "✅ Exists" if diag["id_map_path_exists"] else "❌ Missing",
                    }
                    
                    for key, value in file_status.items():
                        st.write(f"• **{key}:** {value}")
                    
                    st.write(f"**Path:** `{diag['persist_path']}`")
                else:
                    st.warning("FAISS diagnostics not available or in unexpected format")
                
            elif db_type == "chroma":
                # Try to get ChromaDB diagnostics
                if manager.vector_db and hasattr(manager.vector_db, 'get_status'):
                    try:
                        diag = manager.vector_db.get_status()
                        
                        if diag and isinstance(diag, dict) and "collection_initialized" in diag:
                            st.write("**ChromaDB Collection Status:**")
                            
                            status_data = {
                                "Collection Initialized": "✅" if diag["collection_initialized"] else "❌",
                                "Total Vectors": diag["total_vectors"],
                                "Embedding Dimension": diag["embedding_dimension"] or "N/A",
                                "Collection Name": diag["collection_name"],
                                "Client Type": diag["client_type"],
                            }
                            
                            for key, value in status_data.items():
                                st.write(f"• **{key}:** {value}")
                            
                            st.write("**Persistence:**")
                            persist_status = "✅ Directory exists" if diag["persist_directory_exists"] else "❌ Directory missing"
                            st.write(f"• **Status:** {persist_status}")
                            st.write(f"• **Path:** `{diag['persist_directory']}`")
                            
                            if "error" in diag:
                                st.error(f"Error: {diag['error']}")
                        else:
                            st.warning("ChromaDB diagnostics not in expected format")
                    except Exception as e:
                        st.error(f"Failed to get ChromaDB diagnostics: {str(e)}")
                else:
                    st.warning("Unable to fetch ChromaDB diagnostics")
        else:
            st.info("Enable and initialize vector database to see diagnostics")
    
    return manager


def render_upload_section():
    """Render file upload interface."""
    # st.header("📤 Upload Files for Ingestion")
    
    manager = st.session_state.manager
    if manager is None:
        st.error("⚠️ Manager not initialized. Click 'Initialize' in Configuration section.")
        return
        
    # ========== VECTOR DATABASE CONFIGURATION ==========
    with st.expander("🗄️ Configure Vector Database", expanded=True):
        col1, col2 = st.columns(2)
        
        # Get current config
        current_vector_db_config = manager.config.storage.vector_db
        
        with col1:
            # Vector DB Type selection
            vector_db_type = st.selectbox(
                "Select Vector Database Type:",
                options=[VectorDBType.CHROMA.value, VectorDBType.FAISS.value],
                index=0 if current_vector_db_config.type == VectorDBType.CHROMA else 1,
                key="vector_db_type_select"
            )
            
            # Embedding provider and model
            embedding_provider = st.selectbox(
                "Embedding Provider:",
                options=["ollama"],  # Currently only ollama is fully supported
                key="embedding_provider_select"
            )
            
            embedding_model = st.selectbox(
                "Embedding Model:",
                options=["nomic-embed-text", "mxbai-embed-large", "snowflake-arctic-embed"],
                index=0,  # Default to nomic-embed-text
                key="embedding_model_select"
            )
        
        with col2:
            # Embedding provider details
            base_url = st.text_input(
                "Ollama Base URL:",
                value=current_vector_db_config.embedding.base_url,
                key="embedding_base_url",
                placeholder="http://localhost:11434"
            )
            
            embed_batch_size = st.number_input(
                "Embedding Batch Size:",
                min_value=1,
                max_value=100,
                value=current_vector_db_config.embedding.embed_batch_size,
                step=1,
                key="embedding_batch_size"
            )
        
        # Database-specific configurations
        st.subheader("Database-Specific Configuration")
        
        if vector_db_type == VectorDBType.CHROMA.value:
            col1, col2 = st.columns(2)
            with col1:
                chroma_collection = st.text_input(
                    "ChromaDB Collection Name:",
                    value=current_vector_db_config.chroma.collection_name,
                    key="chroma_collection_name",
                    placeholder="documents"
                )
            
            with col2:
                chroma_persist_dir = st.text_input(
                    "ChromaDB Persist Directory:",
                    value=current_vector_db_config.chroma.persist_directory,
                    key="chroma_persist_dir",
                    placeholder="./data/chroma"
                )
        
        else:  # FAISS
            # col1, col2, col3 = st.columns(3)
            col1, col2 = st.columns(2)
            with col1:
                faiss_index_type = st.selectbox(
                    "FAISS Index Type:",
                    options=["flat", "ivfpq", "hnsw"],
                    index=0,
                    key="faiss_index_type"
                )
            
            with col2:
                faiss_distance_metric = st.selectbox(
                    "Distance Metric:",
                    options=["l2", "inner_product"],
                    index=0,
                    key="faiss_distance_metric"
                )
            
            # with col3:
            #     faiss_persist_path = st.text_input(
            #         "FAISS Persist Path:",
            #         value=current_vector_db_config.faiss.persist_path,
            #         key="faiss_persist_path",
            #         placeholder="./data/faiss_index"
            #     )
        
        # Apply configuration button
        if st.button("✅ Apply Vector DB Configuration", key="apply_vector_db_config"):
            try:
                # Update vector database configuration
                manager.config.storage.vector_db.type = VectorDBType(vector_db_type)
                manager.config.storage.vector_db.embedding.provider = embedding_provider
                manager.config.storage.vector_db.embedding.model = EmbeddingModel(embedding_model)
                manager.config.storage.vector_db.embedding.base_url = base_url
                manager.config.storage.vector_db.embedding.embed_batch_size = embed_batch_size
                
                # Update database-specific configs
                if vector_db_type == VectorDBType.CHROMA.value:
                    manager.config.storage.vector_db.chroma.collection_name = chroma_collection
                    manager.config.storage.vector_db.chroma.persist_directory = chroma_persist_dir
                else:  # FAISS
                    manager.config.storage.vector_db.faiss.index_type = faiss_index_type
                    manager.config.storage.vector_db.faiss.distance_metric = faiss_distance_metric
                    # manager.config.storage.vector_db.faiss.persist_path = faiss_persist_path
                
                st.success(f"✅ Vector DB Configuration Updated!")
                st.info(f"📊 Type: {vector_db_type.upper()} | Model: {embedding_model}")
                
                # Re-initialize the vector database with new config
                logger.info("Reinitializing vector database with new configuration")
                try:
                    reinit_ok = manager.reinit_vector_db()
                    if reinit_ok:
                        st.success("🔁 Vector database reinitialized successfully")
                        logger.info("Vector database reinitialized via UI")
                    else:
                        st.error("❌ Failed to reinitialize vector database. Check logs for details.")
                        logger.error("Vector DB reinitialization returned failure")
                except Exception as e:
                    st.error(f"❌ Error reinitializing vector DB: {str(e)}")
                    logger.error(f"Error during vector DB reinitialization: {str(e)}", exc_info=True)
                
            except Exception as e:
                st.error(f"❌ Error applying vector database configuration: {str(e)}")
                logger.error(f"Vector DB config error: {str(e)}", exc_info=True)
    
    # ========== VECTOR DATABASE CONFIGURATION ==========
    with st.expander("🗄️ Vector Database Configuration", expanded=False):
        vector_db_status = manager.get_vector_db_status()
        
        if not vector_db_status["enabled"]:
            st.warning("Vector database is currently disabled. Enable it in the configuration file to use embeddings.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Current Settings")
                st.write(f"**Type:** {vector_db_status['type'].upper()}")
                st.write(f"**Status:** {'🟢 Initialized' if vector_db_status['initialized'] else '🔴 Failed'}")
                st.write(f"**Embedding Provider:** {vector_db_status['embedding_provider']}")
                st.write(f"**Model:** {vector_db_status['embedding_model']}")
            
            with col2:
                st.subheader("Details")
                if vector_db_status["type"] == "chroma":
                    st.write(f"**ChromaDB Collection:** {vector_db_status['chroma_collection']}")
                    st.info("Vectors will be stored in ChromaDB for efficient semantic search")
                elif vector_db_status["type"] == "faiss":
                    st.write(f"**FAISS Index Type:** {vector_db_status['faiss_index_type']}")
                    st.info("Vectors will be stored in FAISS for fast similarity search")
            
            if not vector_db_status["initialized"]:
                st.error("⚠️ Vector database failed to initialize. This usually happens if:")
                st.write("- Ollama is not running (for local embeddings)")
                st.write("- The embedding model is not available")
                st.write("- There's a connection issue with the embedding provider")
                st.write("Check the application logs for more details.")

    # ========== CHUNKING STRATEGY CONFIGURATION ==========
    with st.expander("⚙️ Configure Chunking Strategy", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Framework selection
            framework = st.radio(
                "Select Framework:",
                options=[ChunkingFramework.LANGCHAIN.value, ChunkingFramework.LLAMAINDEX.value],
                horizontal=True,
                key="framework_select"
            )
            
            # Method selection based on framework
            if framework == ChunkingFramework.LANGCHAIN.value:
                methods = [m.value for m in LangChainChunkingMethod]
                selected_method = st.selectbox(
                    "Select Method:",
                    options=methods,
                    key="method_select"
                )
            else:  # LlamaIndex
                methods = [m.value for m in LlamaIndexChunkingMethod]
                selected_method = st.selectbox(
                    "Select Method:",
                    options=methods,
                    key="method_select"
                )
        
        with col2:
            # Common parameters
            chunk_size = st.number_input(
                "Chunk Size (characters):",
                min_value=100,
                max_value=2000,
                value=512,
                step=50,
                key="chunk_size_input"
            )
            
            chunk_overlap = st.number_input(
                "Chunk Overlap (characters):",
                min_value=0,
                max_value=500,
                value=50,
                step=10,
                key="chunk_overlap_input"
            )
        
        # Framework-specific parameters
        st.subheader("Framework-Specific Parameters")
        
        if framework == ChunkingFramework.LANGCHAIN.value:
            if selected_method == "recursive_character":
                separators_text = st.text_area(
                    "Separators (one per line):",
                    value="\n\n\n \n",
                    height=100,
                    key="separators_input"
                )
                separators = [s for s in separators_text.split("\n") if s]
                
                keep_separator = st.checkbox(
                    "Keep separator in output",
                    value=True,
                    key="keep_separator"
                )
                
                is_separator_regex = st.checkbox(
                    "Use regex patterns",
                    value=False,
                    key="is_separator_regex"
                )
        else:  # LlamaIndex
            if selected_method == "semantic_splitter":
                breakpoint_threshold = st.slider(
                    "Breakpoint Percentile Threshold:",
                    min_value=0,
                    max_value=100,
                    value=95,
                    step=5,
                    key="breakpoint_threshold"
                )
                
                embedding_model = st.selectbox(
                    "Embedding Model:",
                    options=["nomic-embed-text", "mxbai-embed-large", "snowflake-arctic-embed"],
                    key="embedding_model"
                )
            
            elif selected_method == "sentence_window":
                window_size = st.slider(
                    "Window Size (sentences):",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="window_size"
                )
        
        # Apply configuration button
        if st.button("✅ Apply Configuration", key="apply_config"):
            try:
                # Update the manager's chunking configuration
                if framework == ChunkingFramework.LANGCHAIN.value:
                    config = LangChainChunkingConfig(
                        method=LangChainChunkingMethod(selected_method)
                    )
                    # Update specific method configs
                    if selected_method == "recursive_character":
                        config.recursive_character.chunk_size = chunk_size
                        config.recursive_character.chunk_overlap = chunk_overlap
                        config.recursive_character.separators = separators if 'separators' in locals() else ["\n\n", "\n", " ", ""]
                        config.recursive_character.keep_separator = keep_separator if 'keep_separator' in locals() else True
                        config.recursive_character.is_separator_regex = is_separator_regex if 'is_separator_regex' in locals() else False
                    else:
                        if selected_method == "character":
                            config.character.chunk_size = chunk_size
                            config.character.chunk_overlap = chunk_overlap
                        elif selected_method == "token":
                            config.token.chunk_size = chunk_size
                            config.token.chunk_overlap = chunk_overlap
                    
                    manager.config.chunking.framework = ChunkingFramework.LANGCHAIN
                    manager.config.chunking.langchain = config
                
                else:  # LlamaIndex
                    config = LlamaIndexChunkingConfig(
                        method=LlamaIndexChunkingMethod(selected_method)
                    )
                    # Update specific method configs
                    if selected_method == "sentence_splitter":
                        config.sentence_splitter.chunk_size = chunk_size
                        config.sentence_splitter.chunk_overlap = chunk_overlap
                    elif selected_method == "token_text_splitter":
                        config.token_text_splitter.chunk_size = chunk_size
                        config.token_text_splitter.chunk_overlap = chunk_overlap
                    elif selected_method == "semantic_splitter":
                        config.semantic_splitter.chunk_size = chunk_size
                        config.semantic_splitter.chunk_overlap = chunk_overlap
                        config.semantic_splitter.breakpoint_percentile_threshold = breakpoint_threshold if 'breakpoint_threshold' in locals() else 95
                        config.semantic_splitter.embedding_model.model = embedding_model if 'embedding_model' in locals() else "nomic-embed-text"
                    elif selected_method == "sentence_window":
                        config.sentence_window.window_size = window_size if 'window_size' in locals() else 3
                    
                    manager.config.chunking.framework = ChunkingFramework.LLAMAINDEX
                    manager.config.chunking.llamaindex = config
                
                st.success(f"✅ Configuration updated! Framework: {framework} | Method: {selected_method}")
                st.info(f"📊 Chunk Size: {chunk_size} | Overlap: {chunk_overlap}")
                
            except Exception as e:
                st.error(f"❌ Error applying configuration: {str(e)}")
                logger.error(f"Config error: {str(e)}", exc_info=True)
    
    # ========== FILE UPLOAD SECTION ==========
    st.subheader("📁 Select Files to Ingest")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to ingest (PDF, DOCX, TXT, CSV, PPT)",
        type=list(manager.config.supported_formats),
        accept_multiple_files=True,
        key="file_uploader",
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        # Create temporary directory for uploaded files
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            saved_files.append(file_path)
            logger.info(f"Uploaded file saved: {file_path}")
        
        # Display uploaded files
        st.subheader("📋 Uploaded Files")
        
        col1, col2, col3 = st.columns(3)
        for i, file in enumerate(uploaded_files):
            with col1 if i % 3 == 0 else (col2 if i % 3 == 1 else col3):
                st.write(f"✅ {file.name}")
        
        # Ingest button
        st.write("Click the button below to start ingestion")
        ingest_button = st.button("🚀 Start Ingestion", key="ingest_button")
        
        if ingest_button:
            with st.spinner("⏳ Ingesting files..."):
                start_time = time.time()
                
                try:
                    # Ingest the uploaded directory
                    result = manager.ingest_directory(str(temp_dir))
                    
                    st.session_state.ingestion_result = result
                    duration = time.time() - start_time
                    
                    # Display results
                    if result.success:
                        st.success(f"✅ {result.message}")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Documents Processed", result.documents_processed)
                        
                        with col2:
                            st.metric("Chunks Created", result.chunks_created)
                        
                        with col3:
                            st.metric("Vectors Stored", result.vectors_stored)
                        
                        with col4:
                            st.metric("Duration", f"{duration:.2f}s")
                        
                        with col5:
                            avg_chunks = (
                                result.chunks_created / result.documents_processed
                                if result.documents_processed > 0
                                else 0
                            )
                            st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")
                    
                    else:
                        st.error(f"❌ {result.message}")
                        
                        if result.errors:
                            st.subheader("Errors Encountered:")
                            for error in result.errors:
                                st.warning(f"- {error.get('file', 'Unknown')}: {error.get('error', 'Unknown error')}")
                    
                    # Update stats
                    logger.info(
                        "Ingestion completed",
                        success=result.success,
                        docs=result.documents_processed,
                        chunks=result.chunks_created,
                    )
                    
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {str(e)}")
                    logger.error(f"Ingestion error: {str(e)}", exc_info=True)
                
                finally:
                    # Cleanup temp files
                    import shutil
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                        logger.info("Temporary upload directory cleaned up")


def render_retrieval_section():
    """Render the retrieval interface."""

    st.header("🔎 Retrieval Pipeline")

    manager = st.session_state.manager
    if manager is None:
        st.error("Manager not initialized.")
        return

    stats = manager.get_stats()
    if stats["total_chunks"] == 0:
        st.info("No chunks are available yet. Ingest documents first, then come back here.")
        return

    vector_db_status = manager.get_vector_db_status()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", stats["total_chunks"])
    with col2:
        st.metric("Vector DB", vector_db_status["type"].upper() if vector_db_status["type"] else "N/A")
    with col3:
        st.metric("Embedding Model", vector_db_status["embedding_model"] or "N/A")

    st.info("Dense semantic retrieval is the default. Lexical and hybrid search are available as configurable options.")

    initial_settings = st.session_state.retrieval_settings

    with st.expander("⚙️ Retrieval Settings", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            search_mode = st.radio(
                "Search Algorithm",
                options=[SearchMode.DENSE.value, SearchMode.LEXICAL.value, SearchMode.HYBRID.value],
                index=[SearchMode.DENSE.value, SearchMode.LEXICAL.value, SearchMode.HYBRID.value].index(
                    st.session_state.get("retrieval_mode", initial_settings.search_mode.value)
                ),
                key="retrieval_mode",
            )
            top_k = st.slider(
                "Top K Results",
                min_value=1,
                max_value=50,
                value=int(st.session_state.get("retrieval_top_k", initial_settings.top_k)),
                step=1,
                key="retrieval_top_k",
            )
            semantic_candidate_k = st.slider(
                "Semantic Candidate K",
                min_value=5,
                max_value=100,
                value=int(st.session_state.get("retrieval_semantic_candidate_k", initial_settings.semantic_candidate_k)),
                step=5,
                key="retrieval_semantic_candidate_k",
            )

        with col2:
            lexical_candidate_k = st.slider(
                "Lexical Candidate K",
                min_value=5,
                max_value=100,
                value=int(st.session_state.get("retrieval_lexical_candidate_k", initial_settings.lexical_candidate_k)),
                step=5,
                key="retrieval_lexical_candidate_k",
            )
            reranker_enabled = st.checkbox(
                "Enable CrossEncoder Reranking",
                value=bool(st.session_state.get("retrieval_reranker_enabled", initial_settings.reranker_enabled)),
                key="retrieval_reranker_enabled",
            )
            reranker_model = st.selectbox(
                "Reranker Model",
                options=[
                    "BAAI/bge-reranker-large",
                    "Qwen/Qwen3-VL-Reranker-2B",
                    "mixedbread-ai/mxbai-rerank-base-v1",
                ],
                index=[
                    "BAAI/bge-reranker-large",
                    "Qwen/Qwen3-VL-Reranker-2B",
                    "mixedbread-ai/mxbai-rerank-base-v1",
                ].index(st.session_state.get("retrieval_reranker_model", initial_settings.reranker_model)),
                key="retrieval_reranker_model",
            )

    st.subheader("Run Retrieval")

    query = st.text_area(
        "Search Query",
        value=st.session_state.get("retrieval_query", ""),
        placeholder="Ask a question about the ingested documents...",
        height=120,
        key="retrieval_query",
    )

    run_col, reset_col = st.columns([1, 1])
    with run_col:
        run_retrieval = st.button("🚀 Run Retrieval", key="run_retrieval_button")
    with reset_col:
        reset_session = st.button("🧹 Reset Current Session Data", key="reset_session_button")

    if reset_session:
        try:
            reset_summary = manager.clear_session_data()
            st.session_state.retrieval_results = []
            st.session_state.ingestion_result = None
            st.session_state.uploaded_files = []
            st.success(
                f"Cleared session data: {reset_summary.get('chunks_removed', 0)} chunks removed"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to clear session data: {exc}")
            logger.error(f"Session reset failed: {exc}", exc_info=True)

    if run_retrieval:
        if not query.strip():
            st.warning("Please enter a query before running retrieval.")
        else:
            try:
                settings = RetrievalSettingsV2(
                    search_mode=SearchMode(search_mode),
                    top_k=int(top_k),
                    semantic_candidate_k=int(semantic_candidate_k),
                    lexical_candidate_k=int(lexical_candidate_k),
                    fusion_k=int(initial_settings.fusion_k),
                    reranker_enabled=bool(reranker_enabled),
                    reranker_model=reranker_model,
                    reranker_device=None,
                    session_only=True,
                    reranker_backend=initial_settings.reranker_backend,
                    flashrank_model=initial_settings.flashrank_model,
                    flashrank_cache_dir=initial_settings.flashrank_cache_dir,
                )
                pipeline = RetrievalPipelineV2(manager, settings=settings)
                results = pipeline.search(query)
                st.session_state.retrieval_results = [result.to_dict() for result in results]
                st.success(f"Retrieved {len(results)} chunks")
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")
                logger.error(f"Retrieval failed: {exc}", exc_info=True)

    if st.session_state.retrieval_results:
        st.subheader("Retrieved Chunks")
        results_frame = pd.DataFrame(st.session_state.retrieval_results)
        st.dataframe(results_frame, hide_index=True, use_container_width=True)

        st.subheader("Chunk Details")
        selected_rank = st.selectbox(
            "Select a result to inspect",
            options=list(range(len(st.session_state.retrieval_results))),
            format_func=lambda index: f"Rank {index + 1}: {st.session_state.retrieval_results[index]['chunk_id'][:8]}...",
            key="retrieval_result_selector",
        )

        selected_result = st.session_state.retrieval_results[selected_rank]
        with st.expander("View full retrieved chunk", expanded=True):
            full_chunk = manager.storage_manager.get_chunk(selected_result["chunk_id"])
            st.write(f"**Chunk ID:** {selected_result['chunk_id']}")
            st.write(f"**Document ID:** {selected_result['document_id']}")
            st.write(f"**Rank:** {selected_result['rank']}")
            st.write(f"**Score:** {selected_result['score']:.4f}")
            st.write(f"**Source:** {selected_result['source']}")
            if selected_result.get("metadata"):
                st.write(f"**Metadata:** {selected_result['metadata']}")
            st.text_area(
                "Content",
                value=full_chunk.content if full_chunk else selected_result["preview"],
                height=240,
                disabled=True,
                label_visibility="collapsed",
            )


def render_chunk_browser():
    """Render chunk browser interface."""
    st.header("🔍 Chunk Browser")
    
    manager = st.session_state.manager
    if manager is None:
        st.error("Manager not initialized.")
        return
    
    stats = manager.get_stats()
    total_chunks = stats["total_chunks"]
    
    if total_chunks == 0:
        st.info("📭 No chunks in database yet. Upload and ingest files first.")
        return
    
    st.write(f"**Total chunks in database: {total_chunks}**")
    
    # Display controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Search chunks by content (leave empty to show recent):",
            placeholder="Enter search term..."
        )
    
    with col2:
        limit = st.slider("Limit results:", min_value=5, max_value=100, value=20, step=5)
    
    with col3:
        refresh_button = st.button("🔄 Refresh", key="refresh_chunks")
    
    try:
        # Retrieve chunks based on search
        if search_query.strip():
            chunks = manager.search_chunks(search_query, limit=limit)
            st.write(f"🔍 Found {len(chunks)} matching chunks")
        else:
            chunks = manager.get_all_chunks(limit=limit)
            st.write(f"📋 Showing {len(chunks)} recent chunks")
        
        if chunks:
            # Display chunks in table format and include Vector DB membership/UUID
            # Build a DataFrame with chunk info and vector DB id (if present)
            vector_ids = []
            for c in chunks:
                vid = ""
                try:
                    vdb = manager.vector_db
                    if vdb is not None:
                        # FAISS: chunk_id_to_index mapping
                        if hasattr(vdb, "chunk_id_to_index") and isinstance(getattr(vdb, "chunk_id_to_index"), dict):
                            if c.chunk_id in vdb.chunk_id_to_index:
                                vid = c.chunk_id
                        # Chroma: try fetching by id
                        if not vid and hasattr(vdb, "collection"):
                            try:
                                data = vdb.collection.get(ids=[c.chunk_id])
                                # If collection.get returns the id, it's present
                                if data and data.get("ids") and len(data.get("ids")[0]) > 0:
                                    # Chroma returns nested lists for queries; if id present, mark it
                                    if c.chunk_id in data.get("ids")[0]:
                                        vid = c.chunk_id
                            except Exception:
                                # If any error, assume not present
                                pass
                except Exception:
                    vid = ""

                vector_ids.append(vid)

            df = pd.DataFrame(
                {
                    "Chunk ID": [c.chunk_id for c in chunks],
                    "Document ID": [c.document_id for c in chunks],
                    "Index": [c.chunk_index for c in chunks],
                    "Content Length": [len(c.content) for c in chunks],
                    "Preview": [c.content[:200] + ("..." if len(c.content) > 200 else "") for c in chunks],
                    "Vector DB ID": vector_ids,
                    "In Vector DB": [bool(v) for v in vector_ids],
                }
            )

            st.dataframe(df, hide_index=True, use_container_width=True)
            
            # Display chunk details on expand
            st.subheader("📖 Chunk Details")
            
            with st.expander("View full chunk content", expanded=False):
                selected_idx = st.selectbox(
                    "Select chunk to view:",
                    range(len(chunks)),
                    format_func=lambda i: f"Chunk {i+1}: {chunks[i].content[:50]}..."
                )
                
                if selected_idx is not None:
                    selected_chunk = chunks[selected_idx]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Chunk ID:** `{selected_chunk.chunk_id}`")
                        st.write(f"**Document ID:** `{selected_chunk.document_id}`")
                        st.write(f"**Chunk Index:** {selected_chunk.chunk_index}")
                    
                    with col2:
                        st.write(f"**Content Length:** {len(selected_chunk.content)} characters")
                        st.write(f"**Created At:** {selected_chunk.created_at}")
                        if selected_chunk.metadata:
                            st.write(f"**Metadata:** {selected_chunk.metadata}")
                    
                    st.divider()
                    st.write("**Full Content:**")
                    st.text_area(
                        "Chunk content:",
                        value=selected_chunk.content,
                        height=300,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    
                    st.divider()
                    
                    # Delete chunk button
                    col1, col2, col3 = st.columns([1, 2, 2])
                    with col1:
                        if st.button("🗑️ Delete Chunk", key=f"delete_chunk_{selected_chunk.chunk_id}"):
                            if manager.delete_chunk(selected_chunk.chunk_id):
                                st.success(f"✅ Chunk {selected_chunk.chunk_id[:8]}... deleted successfully!")
                                st.session_state.chunk_deleted = True
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to delete chunk {selected_chunk.chunk_id[:8]}...")
        else:
            st.info("❌ No chunks found matching your search.")
        
    except Exception as e:
        st.error(f"Error retrieving chunks: {str(e)}")
        logger.error(f"Chunk browser error: {str(e)}", exc_info=True)


def render_statistics():
    """Render statistics dashboard."""
    st.header("📊 Statistics Dashboard")
    
    manager = st.session_state.manager
    if manager is None:
        st.error("Manager not initialized.")
        return
    
    stats = manager.get_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📦 Total Chunks", stats["total_chunks"])
    
    with col2:
        st.metric("📋 Framework", stats["chunking_framework"].upper())
    
    with col3:
        st.metric("🔧 Method", stats["chunking_method"].upper())
    
    # Display database info
    st.info(f"📁 Database: {Path(stats['db_path']).name}")
    
    # Display last ingestion result if available
    if st.session_state.ingestion_result:
        st.subheader("Last Ingestion Result")
        
        result = st.session_state.ingestion_result
        
        data = {
            "Metric": [
                "Status",
                "Documents Processed",
                "Chunks Created",
                "Vectors Stored",
                "Duration (seconds)",
                "Errors",
            ],
            "Value": [
                "✅ Success" if result.success else "❌ Failed",
                str(result.documents_processed),
                str(result.chunks_created),
                str(result.vectors_stored),
                f"{result.duration_seconds:.2f}",
                str(len(result.errors)),
            ],
        }
        
        df = pd.DataFrame(data)
        st.dataframe(df, hide_index=True)
        
        if result.errors:
            st.subheader("Errors:")
            for error in result.errors:
                st.write(f"- **{error.get('file', 'Unknown')}**: {error.get('error', 'Unknown error')}")


def render_configuration_manager():
    """Render configuration management interface."""
    st.header("⚙️ Configuration Manager")
    
    manager = st.session_state.manager
    if manager is None:
        st.error("Manager not initialized.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Settings")
        
        config = manager.config
        
        st.write(f"**Input Path:** `{config.input_path}`")
        st.write(f"**Output Path:** `{config.output_path}`")
        st.write(f"**Max File Size:** {config.max_file_size} MB")
        st.write(f"**Supported Formats:** {', '.join([f.upper() for f in config.supported_formats])}")
    
    with col2:
        st.subheader("Chunking Configuration")
        
        chunking = config.chunking
        
        # Display framework and method
        st.write(f"**Framework:** {chunking.framework.value.upper()}")
        if chunking.framework.value == "langchain":
            method = chunking.langchain.method.value
            chunk_size = chunking.langchain.recursive_character.chunk_size
            chunk_overlap = chunking.langchain.recursive_character.chunk_overlap
            separators = chunking.langchain.recursive_character.separators
        else:
            method = chunking.llamaindex.method.value
            chunk_size = chunking.llamaindex.sentence_splitter.chunk_size
            chunk_overlap = chunking.llamaindex.sentence_splitter.chunk_overlap
            separators = None
        
        st.write(f"**Method:** {method}")
        st.write(f"**Chunk Size:** {chunk_size} characters")
        st.write(f"**Overlap:** {chunk_overlap} characters")
        
        if separators:
            st.write(f"**Separators:** {separators}")
    
    # Show parsing configuration
    st.subheader("Parsing Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "OCR Enabled",
            "✅" if config.parsing.enable_ocr else "❌"
        )
    
    with col2:
        st.metric(
            "Preserve Layout",
            "✅" if config.parsing.preserve_layout else "❌"
        )
    
    with col3:
        st.metric(
            "Extract Tables",
            "✅" if config.parsing.extract_tables else "❌"
        )


def render_logs():
    """Render application logs."""
    st.header("📝 Application Logs")
    
    log_file = Path("logs/ingestion.log")
    
    if log_file.exists():
        with st.expander("View Logs", expanded=False):
            try:
                logs = log_file.read_text()
                st.code(logs[-2000:], language="json")  # Last 2000 chars
            except Exception as e:
                st.error(f"Error reading logs: {str(e)}")
    else:
        st.info("📭 No logs yet. Start an ingestion to generate logs.")


def main():
    """Main Streamlit application."""
    # Header
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.title("🚀 RAG Platform - Ingestion & Retrieval")
    
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh", key="refresh_page"):
            st.rerun()
    
    # Sidebar with configuration
    manager = render_sidebar()
    
    # Initialize manager if not already done
    if manager is None and st.button("Initialize IngestionManager", key="init_manager"):
        initialize_manager()
        st.rerun()
    
    st.markdown("---")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📤 Upload & Ingest", "🔍 Browse Chunks", "🔎 Retrieval", "📊 Statistics", "⚙️ Configuration", "📝 Logs"]
    )
    
    with tab1:
        render_upload_section()
    
    with tab2:
        render_chunk_browser()
    
    with tab3:
        render_retrieval_section()
    
    with tab4:
        render_statistics()
    
    with tab5:
        render_configuration_manager()
    
    with tab6:
        render_logs()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("🏗️ Component 1: Data Ingestion & Parsing")

    with col2:
        st.caption("🔎 Component 2: Retrieval Pipeline")

    with col3:
        st.caption(f"⏱️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
