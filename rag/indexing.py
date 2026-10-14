"""Load files from disk, keep the vector store in sync, maintain the registry."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredImageLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from .config import (
    BM25_CACHE_FILE, CHUNK_SCHEMA, DOC_FOLDER, REGISTRY_FILE, get_vector_store,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- loaders
LOADERS = {
    ".pdf":  lambda p: PyPDFLoader(p),
    ".docx": lambda p: Docx2txtLoader(p),
    ".txt":  lambda p: TextLoader(p, encoding="utf-8", autodetect_encoding=True),
    ".md":   lambda p: UnstructuredMarkdownLoader(p),
    ".html": lambda p: UnstructuredHTMLLoader(p),
    ".htm":  lambda p: UnstructuredHTMLLoader(p),
    ".png":  lambda p: UnstructuredImageLoader(p),
    ".jpg":  lambda p: UnstructuredImageLoader(p),
    ".jpeg": lambda p: UnstructuredImageLoader(p),
    ".tiff": lambda p: UnstructuredImageLoader(p),
}


def load_any(path_str: str) -> list[Document]:
    ext = Path(path_str).suffix.lower()
    loader_fn = LOADERS.get(ext)
    if loader_fn is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader_fn(path_str).load()


# ---------------------------------------------------------------- registry
def _compute_file_hash(file_path: Path, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_registry(path: str = REGISTRY_FILE) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            logger.warning("Registry at %s is corrupt; starting fresh", path)
            return {}
    return {}


def _save_registry(registry: dict, path: str = REGISTRY_FILE) -> None:
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=4))
    tmp.replace(path)


# ---------------------------------------------------------------- sync
def sync_documents(vector_store: Chroma | None = None) -> bool:
    """Sync DOC_FOLDER with the vector store. Returns True if the DB changed.

    Crash-safe: the registry is committed only after a successful add; a failed
    file is retried next run.

    Fast path: the file hashes are compared BEFORE the vector store (and
    therefore the embedding model) is opened. On a no-op run this returns
    without paying the ~20s embedding model load.
    """
    registry = _load_registry()

    current_files = [
        p.resolve()
        for p in Path(DOC_FOLDER).iterdir()
        if p.suffix.lower() in LOADERS
    ]
    current_paths = {str(p): p for p in current_files}
    deleted_files = set(registry.keys()) - set(current_paths)

    # ---- decide whether any real work is needed (no model load yet) ----
    needs_ingest = False
    for path_str, pdf_path in current_paths.items():
        try:
            file_hash = _compute_file_hash(pdf_path)
        except OSError:
            logger.exception("[sync] cannot hash %s", pdf_path)
            needs_ingest = True  # be safe, let the ingest path log it
            continue
        entry = registry.get(path_str)
        if (entry is None
                or entry.get("hash") != file_hash
                or entry.get("schema") != CHUNK_SCHEMA):
            needs_ingest = True
            break

    if not deleted_files and not needs_ingest:
        logger.info("Nothing to sync; skipping vector store open")
        return False

    # ---- real work ahead: open the store (loads embedding model once) ----
    if vector_store is None:
        vector_store = get_vector_store()
    embeddings = vector_store._embedding_function

    changed = False
    if deleted_files:
        logger.info("Removing %d deleted file(s) from index", len(deleted_files))
        for df in deleted_files:
            vector_store.delete(where={"source_file": df})
            del registry[df]
        changed = True

    new_docs: list = []
    dirty_entries: list[tuple[str, str]] = []

    for path_str, pdf_path in sorted(current_paths.items()):
        file_hash = _compute_file_hash(pdf_path)

        if (path_str in registry
            and registry[path_str].get("hash") == file_hash
            and registry[path_str].get("schema") == CHUNK_SCHEMA):
            continue

        try:
            if path_str in registry:
                vector_store.delete(where={"source_file": path_str})

            logger.info("Ingesting %s", pdf_path.name)
            file_docs = load_any(path_str)
            stat = os.stat(path_str)
            for doc in file_docs:
                doc.metadata.update({
                    "source_file": path_str,
                    "file_name": Path(path_str).name,
                    "doc_type": Path(path_str).suffix.lstrip(".").lower(),
                    "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
            new_docs.extend(file_docs)
            dirty_entries.append((path_str, file_hash))
        except Exception:
            logger.exception("[sync] failed to process %s", pdf_path.name)

    if new_docs:
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        )
        chunks = splitter.split_documents(new_docs)
        logger.info("Adding %d chunk(s) to vector store", len(chunks))
        vector_store.add_documents(chunks)

    for path_str, file_hash in dirty_entries:
        registry[path_str] = {"hash": file_hash, "schema": CHUNK_SCHEMA}

    if changed or dirty_entries:
        _save_registry(registry)
        _invalidate_derived_caches()
        return True
    return False


def _invalidate_derived_caches() -> None:
    bm25_path = Path(BM25_CACHE_FILE)
    if bm25_path.exists():
        bm25_path.unlink()

    # Local import: rag imports indexing at module load, so a top-level
    # import here would create a cycle.
    from .rag import clear_retriever_cache
    from .config import clear_vector_store_cache

    clear_retriever_cache()
    clear_vector_store_cache()