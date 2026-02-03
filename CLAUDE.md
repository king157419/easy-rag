# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese medical RAG (Retrieval-Augmented Generation) system for Traditional Chinese Medicine (TCM) knowledge Q&A. The system uses ChromaDB for vector storage, moka-ai/m3e-base for embeddings, and Ollama with qwen3:8b for answer generation.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web application
streamlit run app.py

# Run performance tests
python test_rag_performance.py

# Run RAG evaluation
python evaluate_rag.py

# Run system diagnostics
python diagnostics.py

# Preprocess source data (regenerate processed_data.json)
python preprocess.py
```

**Prerequisites**: Ollama must be running with qwen3:8b model:
```bash
ollama serve
ollama pull qwen3:8b
```

**HuggingFace mirror** (for China): `export HF_ENDPOINT=https://hf-mirror.com`

## Architecture

### Data Flow
```
sources/*.txt → preprocess.py → data/processed_data.json → ChromaDB → Query → Ollama → Answer
```

### Core Modules

- **app.py**: Streamlit web interface, orchestrates all components
- **config.py**: Central configuration (models, parameters, paths)
- **models.py**: Loads embedding model (m3e-base) and generation model (Ollama)
- **chroma_utils.py**: ChromaDB operations (indexing, similarity search)
- **rag_core.py**: Answer generation via Ollama API with prompt engineering
- **retrieval_optimizer.py**: Hybrid search, reranking, query expansion, deduplication
- **data_utils.py**: JSON data loading
- **preprocess.py**: Converts source txt files to chunked JSON
- **chunk_optimizer.py**: Semantic-aware text chunking

### Key Configuration (config.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| EMBEDDING_MODEL_NAME | moka-ai/m3e-base | Chinese embedding model (768 dim) |
| OLLAMA_MODEL | qwen3:8b | Generation model |
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama API endpoint |
| TOP_K | 5 | Number of documents to retrieve |
| MAX_NEW_TOKENS_GEN | 1024 | Max generation length |

### Global State

`config.id_to_doc_map`: Dictionary mapping document IDs to content, populated during indexing and used across modules for document retrieval.

## Data Format

Documents in `processed_data.json`:
```json
{
  "id": "吴银根.txt_0",
  "title": "吴银根",
  "abstract": "...",
  "source_file": "吴银根.txt",
  "chunk_index": 0
}
```

## Vector Database

Uses ChromaDB with cosine similarity. Collection name: `medical_rag_lite`. Database persisted locally. Note: milvus_utils.py exists but is deprecated.
