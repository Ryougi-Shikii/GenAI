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

from .config import (
    BM25_CACHE_FILE, ENSEMBLE_WEIGHTS, MAX_RERANK_INPUT,
    RERANK_TOP_N, get_agent, get_reranker, get_translation_agent,
    get_vector_store, salt, ENSEMBLE_K, HYDE_K,
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
            search_type="similarity",
            search_kwargs={"k": ENSEMBLE_K},
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
        retriever.k = ENSEMBLE_K
        try:
            with open(BM25_CACHE_FILE, "wb") as f:
                pickle.dump(retriever, f)
        except Exception:
            logger.warning("Could not persist BM25 cache", exc_info=True)
        return retriever


def clear_retriever_cache() -> None:
    RetrieverFactory._instance_cache.clear()


# ================================================================ agent turn
def _reasoning_text(block: dict) -> str:
    """Extract the reasoning payload from a reasoning-type content block.

    Different providers name the field differently (`reasoning`, `summary`,
    `text`), and `summary` can be a list of sub-blocks. Be liberal.
    """
    for key in ("reasoning", "text", "content", "summary"):
        value = block.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    t = item.get("text") or item.get("content")
                    if isinstance(t, str):
                        parts.append(t)
            if parts:
                return "".join(parts)
    return ""


def _extract_blocks(msg) -> tuple[str, str]:
    """Return (reasoning, text) from an assistant message.

    Reasoning models (gpt-oss-120b, deepseek-r1, ...) prepend a
    {"type": "reasoning", ...} block to `content_blocks`; the actual answer
    is a later {"type": "text", ...} block. Non-reasoning models only emit
    the text block.
    """
    reasoning_chunks: list[str] = []
    text = ""

    blocks = getattr(msg, "content_blocks", None) or []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "reasoning":
            chunk = _reasoning_text(block)
            if chunk:
                reasoning_chunks.append(chunk)
        elif btype == "text" and not text:
            text = block.get("text", "")

    # Fallbacks for providers that don't use content_blocks.
    if not text:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )

    return "\n\n".join(reasoning_chunks).strip(), text


async def _agent_text(
        agent,
        prompt: str,
        thread_id: str | None = None,
        show_reasoning: bool = True,
    ) -> str:
    """Run one turn and return the text. Prints reasoning blocks if present."""
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    out = await agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
    )

    reasoning, text = _extract_blocks(out["messages"][-1])

    if show_reasoning and reasoning:
        print("\n--- reasoning ---")
        print(reasoning)
        print("--- /reasoning ---")

    return text


# ================================================================ query translation
async def _rewrite_query(agent, query: str) -> str:
    prompt = (
        "You are a search-query rewriter for a technical document retrieval system. "
        "Rewrite the user's question into a single, self-contained search query. "
        "Resolve pronouns and conversational filler using the conversation context "
        "if needed. Output ONLY the rewritten query, nothing else.\n\n"
        f"User question: {query}"
    )
    return (await _agent_text(agent, prompt)).strip()


async def _expand_queries(agent, query: str, n: int = 3) -> list[str]:
    prompt = (
        "You are a search-query expander for a technical document retrieval system. "
        f"Given the question below, generate {n} alternative search queries that "
        "together cover different angles of the topic (e.g. definition, usage, "
        "examples, edge cases). Number them 1., 2., 3. ... "
        "Output ONLY the numbered queries, nothing else.\n\n"
        f"Question: {query}"
    )
    raw = await _agent_text(agent, prompt)
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
    return (await _agent_text(agent, prompt)).strip()


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


# ================================================================ routing
async def classify(query: str) -> str:
    """Return 'retrieve' or 'chat'. One cheap LLM call, no memory."""
    agent = get_translation_agent()
    prompt = (
        "You are a router inside a document Q&A assistant. Decide whether the "
        "user's latest message needs a lookup in a technical document store, or "
        "whether it is conversational (greeting, smalltalk, meta-question, or a "
        "follow-up answerable from prior turns alone).\n\n"
        "Reply with exactly one word: retrieve or chat.\n\n"
        f"Message: {query}"
    )
    raw = await _agent_text(agent, prompt, show_reasoning=False)
    decision = raw.strip().lower()
    return "retrieve" if decision.startswith("retrieve") else "chat"


# ================================================================ chat
async def chat(query: str, session_id: str = "test") -> str:
    """Conversational reply. No retrieval, but keeps session memory."""
    agent = get_agent()
    try:
        response = await _agent_text(
            agent, query, thread_id=f"chat-{session_id}",
        )
    except Exception:
        logger.exception("[chat] chat turn failed")
        response = "I couldn't reach the language model."

    print("\n=== Answer ===")
    print(response)
    print()
    return response


# ================================================================ ask
async def ask(
        query: str,
        factory: RetrieverFactory | None = None,
        expansion_n: int = 3,
        return_context: bool = False,
        session_id: str = "test",
    ):
    """translate -> retrieve per variant -> pool -> rerank -> answer.

    Query translation (rewrite / expand / HyDE) is done with the dedicated
    translation model (Qwen); the final answer is produced by the chat chain.
    """
    if factory is None:
        factory = RetrieverFactory.cached()

    chat_agent = get_agent()
    translation_agent = get_translation_agent()

    variants = await _translate_query(
        translation_agent, query, expansion_n=expansion_n,
    )
    base = factory.get_base_ensemble()
    compressor = factory.get_compressor()

    pooled: list[Document] = []
    for name, text in variants.items():
        if name == "hyde":
            continue
        pooled.extend(await base.ainvoke(text))

    pooled.extend(await factory.vector_store.asimilarity_search(variants["hyde"], k=HYDE_K))
    
    pooled = _dedupe_docs(pooled)
    if len(pooled) > MAX_RERANK_INPUT:
        pooled = pooled[:MAX_RERANK_INPUT]
        
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
            Use the context below when it is relevant. 
            If the user's question cannot be answered from it, say so clearly rather than guessing.
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

    logger.info("Answering %r via retrieval", query)
    try:
        response = await _agent_text(
            chat_agent, answer_prompt, thread_id=f"chat-{session_id}",
        )
    except Exception:
        logger.exception("[ask] chat turn failed")
        response = "I couldn't reach the language model."
    print("\n=== Answer ===")
    print(response)
    print()
    return reranked if return_context else response