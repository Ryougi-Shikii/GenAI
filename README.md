# GenAI Learning Journey

A collection of Python experiments exploring LangChain, tool-using agents, embeddings, and retrieval-augmented generation (RAG). The current application is a local document Q&A assistant; the numbered `Day` folders preserve earlier learning examples and prototypes.

## Current RAG application

The maintained application is the `rag` Python package. Its main flow is:

1. Load supported files from `document/`, split their contents into semantic chunks, and index them in Chroma.
2. Keep the index synchronized using file hashes and a document registry, so unchanged files are not re-ingested.
3. Route conversational messages to chat and document questions to retrieval.
4. Rewrite and expand retrieval queries and generate a hypothetical answer passage (HyDE).
5. Search with a weighted BM25/vector ensemble, then rerank the candidates with a cross-encoder.
6. Answer using the selected document context.

The app accepts PDF, DOCX, TXT, Markdown, HTML, and common image files (PNG, JPG/JPEG, TIFF). The bundled `document/` folder contains introductory Python material on data types, operators, collections, loops, and control flow.

### Requirements

- Python 3.10 or newer
- API credentials for the configured Groq and Google GenAI chat models
- Internet access on first use to download the embedding and reranker models, unless they are already cached

### Set up

From the repository root, create and activate a virtual environment, then install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a local `.env` file in the repository root and configure `GROQ_API_KEY` and `GEMINI_API_KEY`. The application loads these values with `python-dotenv`. Optional LangSmith tracing settings can also be configured there. Do not commit `.env` or put credentials in source files.

### Run

```powershell
python -m rag
```

At startup, the app synchronizes `document/` with the local index and opens an interactive prompt. Ask a document question or start a conversation; enter `exit` (or press Ctrl+C) to quit. Add supported files to `document/` and restart the app to index them.

The package also exposes `ask`, `chat`, `classify`, `sync_documents`, and `RetrieverFactory` from `rag` for use by another Python program.

### Evaluate retrieval

```powershell
python rag_eval.py
```

The evaluation writes a small synthetic Markdown corpus into `document/` when needed, synchronizes it, and reports Hit@1, Hit@3, Hit@8, and mean reciprocal rank. It uses the configured language models and may make API calls. Remove the evaluation files and synchronize the index with:

```powershell
python rag_eval.py --cleanup
```

Cleanup deletes files with the evaluation corpus's filenames from `document/`; do not use it if you already have your own files with those names. After cleanup, the index is synchronized to the remaining documents.

## Repository guide

```text
rag/                         Current RAG package
  __main__.py                Interactive entry point (`python -m rag`)
  config.py                  Models, paths, and retrieval settings
  indexing.py                Supported loaders and document synchronization
  rag.py                     Routing, chat, query translation, and retrieval
Day 1/                       Agent setup, basic tools, and conversation history
Day 2/                       Logging and simple tool/agent experiments
Day 3/                       Embedding and vector-store experiments
Day 4/                       Document loading, hybrid retrieval, and RAG prototypes
Day 5/                       Earlier standalone RAG implementation
document/                    Example Python reference PDFs
rag_eval.py                  Retrieval evaluation corpus and metrics
plane.py                     Small model experiment
theGraph.txt                 LangChain concept notes
requirements.txt             Pinned Python dependencies
```

The `Day 1`–`Day 5` scripts are independent learning examples, not required by the current package entry point. Some use their own prompts, model settings, or imports; use `python -m rag` for the current application.

## Local generated data

The application creates or updates its Chroma database, BM25 cache, document registry, context salt, and Hugging Face model cache locally. These files and `.env` are excluded from Git; they are runtime state, not required source files. A fresh checkout can rebuild the index from `document/` when the application starts.
