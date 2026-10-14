"""RAG pipeline for the document Q&A assistant."""
import logging

from .indexing import sync_documents
from .rag import ask, chat, classify, RetrieverFactory

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["ask", "chat", "classify", "sync_documents", "RetrieverFactory"]
