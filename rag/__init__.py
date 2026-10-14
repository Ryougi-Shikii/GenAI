"""RAG pipeline for the document Q&A assistant."""
import logging

from .indexing import sync_documents
from .rag import ask, RetrieverFactory

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["ask", "sync_documents", "RetrieverFactory"]