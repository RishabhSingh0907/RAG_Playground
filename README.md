# RAG Playground - Streamlit Application Guide

## Overview

RAG Playground is a configurable Retrieval-Augmented Generation studio for document ingestion, chunking, vector storage, retrieval, chat-based generation, and response evaluation.

The Streamlit application is designed as an operational UI for experimenting with RAG pipeline choices without changing source code for every run. It supports document upload, configurable chunking, SQLite persistence, ChromaDB or FAISS vector storage, dense and hybrid retrieval, optional reranking, chat generation through Ollama, and RAG quality monitoring.

The latest UI uses a workspace-style layout:

- A left configuration viewer/sidebar for database, vector database, and diagnostic status.
- A fixed/sticky tab navigation area for the seven main workflows.
- A scrollable main tab content frame.
- In the Chat Playground tab, separate scrollable panes for conversation and generation settings.

This keeps navigation and configuration visible while longer workflows are being inspected.

## Application Tabs

The app exposes seven primary tabs:

1. Upload & Ingest
2. Browse Chunks
3. Retrieval
4. Chat Playground
5. Statistics
6. Configuration
7. Logs

## Project Components

### 1. Ingestion and Parsing

Location: `src/ingestion/`

Responsibilities:

- Load ingestion configuration from `src/config/ingestion_config.yaml`.
- Parse supported document formats.
- Normalize documents and metadata.
- Create document chunks through configurable chunking strategies.
- Store chunks in SQLite.
- Trigger vector embedding and vector database storage when enabled.

Supported formats:

- PDF
- DOCX
- TXT
- CSV
- PPT

### 2. Chunking Strategies

Location: `src/ingestion/chunking_strategies.py`

The ingestion pipeline supports both LangChain and LlamaIndex chunking frameworks.

LangChain methods:

- `recursive_character`
- `character`
- `token`
- `markdown_header`

LlamaIndex methods:

- `sentence_splitter`
- `token_text_splitter`
- `semantic_splitter`
- `sentence_window`
- `code_splitter`

The active framework and method are configured in `src/config/ingestion_config.yaml` and can also be adjusted through the Streamlit UI.

### 3. Storage

Locations:

- `src/storage/`
- `data/chunks.db`

SQLite is used as the primary chunk store. It preserves chunk content, document IDs, source metadata, chunk indexes, and creation timestamps.

### 4. Embeddings

Location: `src/embedding/`

The embedding manager provides a unified interface for embedding providers. The current implementation supports Ollama embeddings for local execution.

Default embedding model:

- `nomic-embed-text`

Other configured model options include:

- `mxbai-embed-large`
- `snowflake-arctic-embed`

### 5. Vector Database

Location: `src/vector_db/`

The app supports two vector database backends:

- ChromaDB
- FAISS

The vector database can be configured from the Upload & Ingest tab and inspected from the sidebar diagnostics section.

Default vector database:

- ChromaDB

Default persistence path:

- `data/chroma`

FAISS persistence path:

- `data/faiss_index`

### 6. Retrieval Pipeline

Location: `src/retrieval/`

The retrieval pipeline is configured through `src/config/retrieval_config.yaml`.

Supported behavior:

- Dense semantic retrieval
- Lexical candidate retrieval
- Hybrid retrieval with fusion
- Optional reranking
- Session-only retrieval/reset behavior

Reranker backend options:

- `heuristic`
- `sentence_transformer`
- `flashrank`

### 7. Generation and Chat

Location: `src/generation/`

The Chat Playground combines retrieval and generation:

1. User submits a question.
2. The retrieval pipeline fetches relevant chunks.
3. A prompt is assembled with retrieved context.
4. Ollama generates the response.
5. Citations are extracted when available.
6. The response is evaluated.

Generation configuration is read from `src/config/generation_config.yaml` and can be adjusted in the Chat Playground generation settings panel.

Default provider:

- Ollama

Default model in config:

- `mistral`

### 8. Monitoring and Evaluation

Location: `src/monitoring/`

The app includes RAG response evaluation metrics such as:

- Faithfulness
- Answer relevancy
- Context precision

The evaluation output is shown in the Chat Playground after response generation.

## Prerequisites

### System Requirements

- Python 3.8 or later
- 4 GB RAM minimum
- 8 GB RAM recommended
- 2 GB disk space minimum for local data and vector stores
- Windows, macOS, or Linux

### Recommended Local Services

Ollama is recommended for local embeddings and local LLM generation.

Install Ollama from:

```text
https://ollama.com/download
```

Pull the default embedding model:

```bash
ollama pull nomic-embed-text
```

Pull at least one generation model:

```bash
ollama pull mistral
```

Verify Ollama is running:

```bash
ollama list
```

The default Ollama base URL used by the project is:

```text
http://localhost:11434
```

## Installation

### 1. Navigate to the Project

```bash
cd RAG_Playground
```

### 2. Create and Activate a Virtual Environment

Using `venv`:

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

Using Conda:

```bash
conda create -n rag python=3.12
conda activate rag
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Major dependency groups:

- `streamlit` for the web UI
- `pydantic` and `pyyaml` for configuration validation
- `pandas` and `pyarrow` for tabular data handling
- `PyMuPDF`, `langchain-docling`, `python-docx`, and `python-pptx` for parsing
- `chromadb` and `faiss-cpu` for vector database backends
- `requests` and `numpy` for embeddings and vector operations
- `sentence-transformers`, `flashrank`, and `onnxruntime` for optional reranking
- `pytest` and `pytest-cov` for testing

## Running the App

Use the updated Streamlit application:

```bash
streamlit run streamlit_app_updated.py
```

The app usually opens at:

```text
http://localhost:8501
```

The older `streamlit_app.py` file may still exist, but `streamlit_app_updated.py` is the current UI entry point for the latest retrieval, chat, monitoring, and layout updates.

## Quick Start Workflow

### 1. Start Ollama

Ensure Ollama is running and the required models are available:

```bash
ollama list
```

At minimum, the app expects an embedding model such as `nomic-embed-text`.

### 2. Launch Streamlit

```bash
streamlit run streamlit_app_updated.py
```

### 3. Initialize the Manager

Use the initialization button when the app starts. This loads:

- `src/config/ingestion_config.yaml`
- SQLite storage
- Vector database configuration
- Embedding manager, if vector DB is enabled

### 4. Configure Vector Database

In Upload & Ingest, open Configure Vector Database and choose:

- ChromaDB or FAISS
- Embedding provider
- Embedding model
- Ollama base URL
- Embedding batch size
- Backend-specific persistence settings

Apply the configuration and reinitialize the vector database when prompted.

### 5. Upload and Ingest Documents

In Upload & Ingest:

1. Select one or more files.
2. Confirm supported formats.
3. Start ingestion.
4. Review processed document count, chunks created, and vectors stored.

### 6. Browse Chunks

Use Browse Chunks to:

- Search chunk content.
- Inspect chunk metadata.
- Check whether chunks exist in the vector database.
- View vector database IDs when available.

### 7. Run Retrieval

Use Retrieval to:

- Enter a query.
- Configure search mode.
- Set `top_k`.
- Tune semantic and lexical candidate counts.
- Enable or disable reranking.
- Reset session-scoped chunks and embeddings when needed.

### 8. Use Chat Playground

Use Chat Playground to ask document-grounded questions.

The latest UI splits this tab into:

- Conversation panel: scrolls independently.
- Generation Settings panel: scrolls independently.

Generation controls include:

- Ollama model
- Temperature
- Max tokens
- System prompt
- Clear conversation

Generated answers include RAG evaluation and retrieved context details.

## Configuration Files

### Ingestion Configuration

Path:

```text
src/config/ingestion_config.yaml
```

Controls:

- Input and output paths
- Supported formats
- Max file size
- Parsing behavior
- Chunking framework and method
- SQLite storage
- Vector database backend
- Embedding configuration
- Logging
- Error handling

Example:

```yaml
ingestion:
  input_path: "./data/raw"
  output_path: "./data/processed"
  max_file_size: 100
  chunking:
    framework: "langchain"
    langchain:
      method: "recursive_character"

storage:
  sqlite:
    db_path: "./data/chunks.db"
  vector_db:
    enabled: true
    type: "chroma"
    embedding:
      provider: "ollama"
      model: "nomic-embed-text"
      base_url: "http://localhost:11434"
```

### Retrieval Configuration

Path:

```text
src/config/retrieval_config.yaml
```

Controls:

- Search mode
- Top-k results
- Dense and lexical candidate counts
- Hybrid fusion
- Reranker backend
- FlashRank cache path

Example:

```yaml
retrieval:
  search_mode: "dense"
  top_k: 10
  semantic_candidate_k: 25
  lexical_candidate_k: 25
  reranker:
    enabled: false
    backend: "sentence_transformer"
```

### Generation Configuration

Path:

```text
src/config/generation_config.yaml
```

Controls:

- LLM provider
- Ollama base URL
- Model
- Temperature
- Token limit
- Retry behavior
- Prompt context limit
- Citation behavior

Example:

```yaml
generation:
  llm_provider: "ollama"
  ollama:
    base_url: "http://localhost:11434"
    model: "llama3"
    temperature: 0.7
    max_tokens: 1024
```

## Data Flow

```text
Uploaded file
    -> ParserRegistry
    -> Normalized Document
    -> ChunkingStrategyFactory
    -> DocumentChunk list
    -> SQLiteManager
    -> EmbeddingManager
    -> ChromaDB or FAISS
    -> RetrievalPipelineV2
    -> PromptBuilder
    -> Ollama LLM Provider
    -> Chat response
    -> RAG evaluation
```

On macOS/Linux:

```bash
cp ./data/chunks.db ./data/chunks.db.backup
```

### Clear Session Data from the UI

Use the reset/clear controls in the app when possible. This is preferred because it coordinates SQLite and vector backend cleanup through the application logic.

## Testing

Run the full test suite:

```bash
pytest -v
```

Run targeted tests:

```bash
pytest tests/test_ingestion.py -v
pytest tests/test_vector_db.py -v
pytest tests/test_retrieval_pipeline.py -v
pytest tests/test_generation.py -v
pytest tests/test_monitoring.py -v
pytest tests/test_reranker_manager.py -v
pytest tests/test_chunking_strategies.py -v
```

Run with coverage:

```bash
pytest tests -v --cov=src
```

## Troubleshooting

### Manager Not Initialized

Check:

- `src/config/ingestion_config.yaml` exists.
- YAML syntax is valid.
- `data/` directory is writable.
- Ollama is running if vector DB embeddings are enabled.

### Circular Import Error During Startup

The ingestion package now exposes `IngestionManager` lazily to avoid circular imports between ingestion and embedding modules. If this error returns, inspect imports involving:

- `src/ingestion/__init__.py`
- `src/ingestion/ingestion_manager.py`
- `src/embedding/embedding_manager.py`

Avoid eager imports from package `__init__.py` files when modules depend on each other.

### Ollama Connection Fails

Check:

```bash
ollama list
```

Confirm the configured base URL:

```text
http://localhost:11434
```

Pull required models:

```bash
ollama pull nomic-embed-text
ollama pull mistral
```

### Vector Database Fails to Initialize

Check:

- Vector DB is enabled in `src/config/ingestion_config.yaml`.
- ChromaDB or FAISS dependencies are installed.
- Persistence directories under `data/` are writable.
- Embedding model is available through Ollama.

### Ingestion Succeeds but No Vectors Are Stored

Check:

- `storage.vector_db.enabled` is `true`.
- The embedding manager initialized successfully.
- Ollama is reachable.
- The selected embedding model exists locally.
- The sidebar Vector DB Diagnostics section shows a healthy backend.

### Retrieval Returns Weak Results

Try:

- Increase `semantic_candidate_k`.
- Increase `top_k`.
- Switch from dense to hybrid retrieval if available.
- Enable reranking.
- Improve chunking settings.
- Increase chunk overlap for context continuity.

### Chat Generation Fails

Check:

- Ollama is running.
- The selected chat model is installed.
- Retrieved context exists.
- The system prompt is not overly restrictive.
- Generation timeout and max token settings are appropriate.

### Streamlit Layout Looks Wrong

Try:

- Refresh the browser.
- Clear Streamlit cache.
- Restart the Streamlit process.
- Confirm you are running `streamlit_app_updated.py`, not the older app entry point.

## Performance Guidance

### Faster Ingestion

- Use larger chunks.
- Reduce chunk overlap.
- Disable OCR if not required.
- Process fewer files per batch.
- Use ChromaDB for simpler persistent local vector storage.

### Better Retrieval Quality

- Use smaller or more semantically coherent chunks.
- Increase overlap.
- Enable hybrid retrieval.
- Enable reranking.
- Use higher-quality embedding models.
- Inspect retrieved context before tuning generation.

### Better Chat Answers

- Keep the system prompt strict but not contradictory.
- Use a generation model appropriate for instruction following.
- Increase `max_tokens` if answers are truncated.
- Lower temperature for factual answers.
- Review RAG Evaluation and Retrieved Context after each response.

## Development Notes

### Main Entry Point

```text
streamlit_app_updated.py
```

### Important Classes

- `IngestionManager`
- `ChunkingStrategyFactory`
- `EmbeddingManager`
- `VectorDatabaseFactory`
- `RetrievalPipelineV2`
- `ChatManager`
- `PromptBuilder`
- `CitationExtractor`
- `RagasEvaluator`

### Adding a Parser

Implement a parser in `src/ingestion/parsers.py` or a related parser module, then register it through `ParserRegistry`.

### Adding a Chunking Strategy

Add the implementation in `src/ingestion/chunking_strategies.py`, extend the relevant model/config definitions in `src/ingestion/models.py`, and update `ChunkingStrategyFactory`.

### Adding a Vector Backend

Add a backend implementation under `src/vector_db/` and register it through `VectorDatabaseFactory`.

## Current Run Command

```bash
streamlit run streamlit_app_updated.py
```
