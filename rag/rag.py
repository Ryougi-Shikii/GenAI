"""Retrieval chain, query translation, and the top-level Q&A pipeline."""
from __future__ import annotations

import asyncio
import logging
import pickle
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.utils.uuid import uuid7

from .config import (
    BM25_CACHE_FILE, ENSEMBLE_WEIGHTS, RERANK_TOP_N, SCORE_THRESHOLD,
    get_agent, get_reranker, get_vector_store, salt,
)

logger = logging.getLogger(__name__)


# ================================================================ retriever
class RetrieverFactory:
    """BM25 + vector ensemble, cross-encoder reranked. Cached per vector store."""

    _instance_cache: dict = {}

    def __init__(self, vector_store: Chroma | None = None):
        self.vector_store = vector_store if vector_store is not None else get_vector_store()
        self._base_ensemble = None
        self._compressor = None

    @classmethod
    def cached(cls, vector_store: Chroma | None = None) -> "RetrieverFactory":
        key = id(vector_store) if vector_store is not None else "default"
        if key not in cls._instance_cache:
            cls._instance_cache[key] = cls(vector_store)
        return cls._instance_cache[key]

    def get_vector_retriever(self):
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": SCORE_THRESHOLD},
        )

    def get_base_ensemble(self):
        if self._base_ensemble is None:
            vector_retriever = self.get_vector_retriever()
            bm25 = self._get_bm25_retriever()
            if bm25 is not None:
                self._base_ensemble = EnsembleRetriever(
                    retrievers=[bm25, vector_retriever],
                    weights=ENSEMBLE_WEIGHTS,
                )
            else:
                self._base_ensemble = vector_retriever
        return self._base_ensemble

    def get_compressor(self):
        if self._compressor is None:
            self._compressor = CrossEncoderReranker(
                model=get_reranker(), top_n=RERANK_TOP_N,
            )
        return self._compressor

    def _get_bm25_retriever(self):
        if Path(BM25_CACHE_FILE).exists():
            try:
                with open(BM25_CACHE_FILE, "rb") as f:
                    return pickle.load(f)
            except Exception:
                logger.warning("BM25 cache unreadable; rebuilding", exc_info=True)

        stored = self.vector_store.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=d, metadata=m)
            for d, m in zip(stored["documents"], stored["metadatas"])
        ]
        if not docs:
            return None
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = 10
        try:
            with open(BM25_CACHE_FILE, "wb") as f:
                pickle.dump(retriever, f)
        except Exception:
            logger.warning("Could not persist BM25 cache", exc_info=True)
        return retriever


def clear_retriever_cache() -> None:
    RetrieverFactory._instance_cache.clear()


# ================================================================ agent turn
async def _agent_text(agent, prompt: str, thread_id: str) -> str:
    out = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return out["messages"][-1].content_blocks[0]["text"]


# ================================================================ query translation
async def _rewrite_query(agent, query: str) -> str:
    prompt = (
        "You are a search-query rewriter for a technical document retrieval system. "
        "Rewrite the user's question into a single, self-contained search query. "
        "Resolve pronouns and conversational filler using the conversation context "
        "if needed. Output ONLY the rewritten query, nothing else.\n\n"
        f"User question: {query}"
    )
    return (await _agent_text(agent, prompt, thread_id=f"rewrite-{uuid7()}")).strip()


async def _expand_queries(agent, query: str, n: int = 3) -> list[str]:
    prompt = (
        "You are a search-query expander for a technical document retrieval system. "
        f"Given the question below, generate {n} alternative search queries that "
        "together cover different angles of the topic (e.g. definition, usage, "
        "examples, edge cases). Number them 1., 2., 3. ... "
        "Output ONLY the numbered queries, nothing else.\n\n"
        f"Question: {query}"
    )
    raw = await _agent_text(agent, prompt, thread_id=f"expand-{uuid7()}")
    out = []
    for line in raw.splitlines():
        line = line.strip()
        for prefix in ("-", "*"):
            if line.startswith(prefix):
                line = line[1:].strip()
        while line and line[0].isdigit() and len(line) > 2 and line[1] in ". ":
            line = line[2:].strip()
        if line:
            out.append(line)
    return out[:n]


async def _hyde_document(agent, query: str) -> str:
    prompt = (
        "You are a technical writer. Write a short, factual, encyclopedic paragraph "
        "(80-150 words) that would be the PERFECT answer to the question below, as "
        "if it were a passage from a textbook. Do not reference the question. "
        "Output ONLY the paragraph.\n\n"
        f"Question: {query}"
    )
    return (await _agent_text(agent, prompt, thread_id=f"hyde-{uuid7()}")).strip()


async def _translate_query(agent, query: str, expansion_n: int = 3) -> dict:
    """Rewrite first (expansion/HyDE depend on it), then fan out the rest."""
    rewritten = await _rewrite_query(agent, query)
    seed = rewritten or query
    expansions, hyde = await asyncio.gather(
        _expand_queries(agent, seed, n=expansion_n),
        _hyde_document(agent, seed),
    )
    variants = {"original": query, "rewrite": rewritten, "hyde": hyde}
    variants.update({f"expansion_{i+1}": e for i, e in enumerate(expansions)})
    logger.debug("Translated query into %d variants", len(variants))
    return variants


# ================================================================ pipeline
def _dedupe_docs(docs: list[Document]) -> list[Document]:
    seen, out = set(), []
    for d in docs:
        key = d.page_content.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(d)
    return out


async def ask(
    query: str,
    factory: RetrieverFactory | None = None,
    expansion_n: int = 3,
    return_context: bool = False,
    session_id: str = "test",
):
    """translate -> retrieve per variant -> pool -> rerank -> answer."""
    if factory is None:
        factory = RetrieverFactory.cached()
    agent = get_agent()
    variants = await _translate_query(agent, query, expansion_n=expansion_n)

    base = factory.get_base_ensemble()
    compressor = factory.get_compressor()

    pooled: list[Document] = []
    for name, text in variants.items():
        if name == "hyde":
            continue
        pooled.extend(await base.ainvoke(text))

    pooled.extend(await factory.vector_store.asimilarity_search(variants["hyde"], k=10))
    pooled = _dedupe_docs(pooled)
    logger.info("Pooled %d unique document(s) across query variants", len(pooled))

    try:
        reranked = await compressor.acompress_documents(pooled, query)
    except Exception:
        logger.exception(
            "[ask] reranker failed; falling back to pool top-%d", RERANK_TOP_N
        )
        reranked = pooled[:RERANK_TOP_N]
    logger.info("Reranked down to %d document(s)", len(reranked))

    context_parts = []
    for i, res in enumerate(reranked, 1):
        src = Path(res.metadata.get("source_file", "Unknown")).name
        context_parts.append(f"[Document {i} | Source: {src}]\n{res.page_content}")
    context = "\n\n".join(context_parts)

    answer_prompt = f"""You are a technical question-answering assistant.
        Answer the user's question using the retrieved context below.

        Rules:
            Use the context as the primary source of truth.
            Do not invent facts that are not supported by the context.
            If the context does not contain enough information, explicitly say so.
            Give a clear, well-structured answer.
            Do not cite relevant sources using their document names.

        Retrieved Context:
        <<<CONTEXT-{salt}>>>
        {context}
        <<<END-CONTEXT-{salt}>>>

        User Question:
        {query}

        Answer:"""

    try:
        response = await _agent_text(agent, answer_prompt, thread_id=f"chat-{session_id}")
    except Exception:
        logger.exception("[ask] all LLMs failed; returning fallback message")
        response = "I couldn't reach the language model."

    logger.info("Answer produced for session %s", session_id)
    print(response)
    return reranked if return_context else response