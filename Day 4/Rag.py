import hashlib
import json
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document

from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("rag")

# ---------------------------------------------------------------- config
BASE_DIR = Path(__file__).resolve().parent  
BASE_DIR = Path(BASE_DIR).resolve().parent  

DB_PATH = str(BASE_DIR / "chroma_db")
REGISTRY_FILE = str(BASE_DIR / "doc_registry.json")
BM25_CACHE_FILE = str(BASE_DIR / "bm25_cache.pkl")
DOC_FOLDER = str(BASE_DIR / "document")
HF_CACHE = str(BASE_DIR / "hf_models_cache")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"

SCORE_THRESHOLD = 0.5
ENSEMBLE_WEIGHTS = [0.3, 0.7]
RERANK_TOP_N = 8

# ---------------------------------------------------------------- model cache
_model_cache: dict = {}

def get_embeddings() -> HuggingFaceEmbeddings:
    """Cached singleton — the embedding model loads only once per process."""
    if "embeddings" not in _model_cache:
        log.info("loading embedding model (%s)...", EMBEDDING_MODEL)
        _model_cache["embeddings"] = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, cache_folder=HF_CACHE
        )
    return _model_cache["embeddings"]


def get_reranker() -> HuggingFaceCrossEncoder:
    """Cached singleton — the cross-encoder loads only once per process."""
    if "reranker" not in _model_cache:
        log.info("loading cross-encoder reranker (%s)...", RERANKER_MODEL)
        _model_cache["reranker"] = HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL,
            # cache_folder must go through model_kwargs: the direct kwarg is
            # rejected by pydantic, and kwargs here are forwarded to the
            # underlying sentence-transformers CrossEncoder
            model_kwargs={"cache_folder": HF_CACHE},
        )
    return _model_cache["reranker"]


# ---------------------------------------------------------------- registry
def compute_file_hash(file_path: Path, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_registry(path: str = REGISTRY_FILE) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            log.warning("registry file corrupt, starting fresh")
            return {}
    return {}


def save_registry(registry: dict, path: str = REGISTRY_FILE):
    # atomic write: temp file + rename, so a crash can't corrupt the registry
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=4))
    tmp.replace(path)


# ---------------------------------------------------------------- vector store cache
_vector_store_cache: dict = {}

def get_vector_store() -> Chroma:
    """Cached singleton Chroma store (persisted on disk, reused in memory)."""
    if "store" not in _vector_store_cache:
        _vector_store_cache["store"] = Chroma(
            persist_directory=DB_PATH,
            embedding_function=get_embeddings(),
        )
    return _vector_store_cache["store"]


# ---------------------------------------------------------------- sync
def sync_documents(vector_store: Chroma | None = None) -> bool:
    """Sync PDFs in DOC_FOLDER with the vector store.

    Returns True if the DB changed (so callers know to invalidate caches).
    Crash-safe: the registry is only updated after a successful add;
    a failed file is left out of the registry and retried next run.
    """
    if vector_store is None:
        vector_store = get_vector_store()
    embeddings = vector_store._embedding_function
    registry = load_registry()

    current_files = [p.resolve() for p in Path(DOC_FOLDER).glob("*.pdf")]
    current_paths = {str(p): p for p in current_files}
    deleted_files = set(registry.keys()) - set(current_paths)

    changed = False

    if deleted_files:
        log.info("removing %d deleted file(s) from DB", len(deleted_files))
        for df in deleted_files:
            log.info("  [-] %s", Path(df).name)
            vector_store.delete(where={"source_file": df})
            del registry[df]
        changed = True

    new_docs = []
    dirty_entries = []  # committed to registry only after a successful add

    log.info("checking %d document(s) for changes...", len(current_files))
    for path_str, pdf_path in sorted(current_paths.items()):
        file_hash = compute_file_hash(pdf_path)

        if path_str in registry and registry[path_str]["hash"] == file_hash:
            continue  # unchanged

        log.info("  [+] processing: %s", pdf_path.name)
        try:
            if path_str in registry:
                vector_store.delete(where={"source_file": path_str})

            file_docs = PyPDFLoader(path_str).load()
            for doc in file_docs:
                doc.metadata["source_file"] = path_str
            new_docs.extend(file_docs)
            dirty_entries.append((path_str, file_hash))
        except Exception:
            log.exception("  [x] failed to process %s - will retry next run", pdf_path.name)

    if new_docs:
        log.info("chunking %d page(s)...", len(new_docs))
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        )
        chunks = splitter.split_documents(new_docs)
        log.info("adding %d chunk(s) to vector DB", len(chunks))
        vector_store.add_documents(chunks)

    # commit registry entries only for files that made it this far
    for path_str, file_hash in dirty_entries:
        registry[path_str] = {"hash": file_hash}

    if changed or dirty_entries:
        save_registry(registry)
        _invalidate_derived_caches()
        return True

    log.info("no changes detected")
    return False


def _invalidate_derived_caches():
    """Drop cached objects that depend on DB contents."""
    if Path(BM25_CACHE_FILE).exists():
        Path(BM25_CACHE_FILE).unlink()
    RetrieverFactory._instance_cache.clear()


# ---------------------------------------------------------------- retrieval
class RetrieverFactory:
    """Builds and caches the production retriever chain.

    Caches, in order:
      - the BM25 index (pickled to disk, survives across runs)
      - the ensemble retriever
      - the reranker model (via get_reranker singleton)
      - the final contextual-compression retriever
    """

    _instance_cache: dict = {}

    def __init__(self, vector_store: Chroma | None = None):
        self.vector_store = vector_store if vector_store is not None else get_vector_store()
        self._base_ensemble = None
        self._compressor = None
        self._retriever = None

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
        """BM25 + vector ensemble, no reranking. Used for multi-query pooling."""
        if self._base_ensemble is None:
            vector_retriever = self.get_vector_retriever()

            bm25_retriever = self._get_bm25_retriever()
            if bm25_retriever is not None:
                self._base_ensemble = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=ENSEMBLE_WEIGHTS,
                )
            else:
                log.warning("vector store is empty; falling back to vector retriever only")
                self._base_ensemble = vector_retriever
        return self._base_ensemble

    def get_compressor(self):
        """Cross-encoder reranker as a standalone compressor (for pooling)."""
        if self._compressor is None:
            self._compressor = CrossEncoderReranker(
                model=get_reranker(),  # cached singleton
                top_n=RERANK_TOP_N,
            )
        return self._compressor

    def get(self):
        """Fully assembled chain: ensemble -> cross-encoder reranker."""
        if self._retriever is None:
            self._retriever = ContextualCompressionRetriever(
                base_compressor=self.get_compressor(),
                base_retriever=self.get_base_ensemble(),
            )
        return self._retriever

    def _get_bm25_retriever(self):
        # 1) try the pickled index first (survives across process runs)
        if Path(BM25_CACHE_FILE).exists():
            try:
                with open(BM25_CACHE_FILE, "rb") as f:
                    retriever = pickle.load(f)
                log.info("loaded cached BM25 index from %s", BM25_CACHE_FILE)
                return retriever
            except Exception:
                log.warning("BM25 cache unreadable, rebuilding")

        # 2) rebuild from Chroma and persist
        stored = self.vector_store.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=d, metadata=m)
            for d, m in zip(stored["documents"], stored["metadatas"])
        ]
        if not docs:
            return None
        log.info("rebuilding BM25 index over %d chunk(s)...", len(docs))
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = 10
        try:
            with open(BM25_CACHE_FILE, "wb") as f:
                pickle.dump(retriever, f)
        except Exception:
            log.warning("could not persist BM25 index (will rebuild next run)")
        return retriever
 
# ---------------------------------------------------------------- agent (query translation)
AGENT_SYSTEM_PROMPT = """You are Fang Yuan from the novel Reverend Insanity.
You are helpful, and carry all other traits of Fang Yuan's personality."""

def get_chat_model():
    if "chat" not in _model_cache:
        log.info("initializing chat model for query translation...")
        _model_cache["chat"] = init_chat_model(
            "gemini-3.5-flash-lite",
            model_provider="google-genai",
            temperature=0.9,
            timeout=600,
            max_tokens=3000,
        )
    return _model_cache["chat"]


def get_agent():
    """Cached singleton agent (InMemorySaver checkpointer, same thread reused)."""
    if "agent" not in _model_cache:
        _model_cache["agent"] = create_agent(
            model=get_chat_model(),
            system_prompt=AGENT_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
            middleware=[
                ModelRetryMiddleware(max_retries=3),
                ToolRetryMiddleware(max_retries=2),
            ],
        )
    return _model_cache["agent"]


def _agent_text(agent, prompt: str, thread_id: str) -> str:
    """Run one turn and return the final text (agent is stateful per thread)."""
    out = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    msg = out["messages"][-1].content_blocks[0]["text"]
    return msg


def rewrite_query(agent, query: str) -> str:
    """Step 1 - Rewriting: make the query standalone and explicit."""
    prompt = (
        "You are a search-query rewriter for a technical document retrieval system. "
        "Rewrite the user's question into a single, self-contained search query. "
        "Resolve pronouns and conversational filler using the conversation context "
        "if needed. Output ONLY the rewritten query, nothing else.\n\n"
        f"User question: {query}"
    )
    return _agent_text(agent, prompt, thread_id="rewrite").strip()


def expand_queries(agent, query: str, n: int = 3) -> list[str]:
    """Step 2 - Expansion: generate n sub-queries covering different angles."""
    prompt = (
        "You are a search-query expander for a technical document retrieval system. "
        f"Given the question below, generate {n} alternative search queries that "
        "together cover different angles of the topic (e.g. definition, usage, "
        "examples, edge cases). Number them 1., 2., 3. ... "
        "Output ONLY the numbered queries, nothing else.\n\n"
        f"Question: {query}"
    )
    raw = _agent_text(agent, prompt, thread_id="expand")
    out = []
    for line in raw.splitlines():
        line = line.strip()
        # strip leading "1." / "- " markers
        for prefix in ("-", "*"):
            if line.startswith(prefix):
                line = line[1:].strip()
        while line and line[0].isdigit() and len(line) > 2 and line[1] in ". ":
            line = line[2:].strip()
        if line:
            out.append(line)
    return out[:n]


def hyde_document(agent, query: str) -> str:
    """Step 3 - HyDE: write the hypothetical answer document we'd WANT to find."""
    prompt = (
        "You are a technical writer. Write a short, factual, encyclopedic paragraph "
        "(80-150 words) that would be the PERFECT answer to the question below, as "
        "if it were a passage from a textbook. Do not reference the question. "
        "Output ONLY the paragraph.\n\n"
        f"Question: {query}"
    )
    return _agent_text(agent, prompt, thread_id="hyde").strip()


def translate_query(agent, query: str, expansion_n: int = 3) -> dict:
    """Run all three translations. Returns a dict of variant -> text."""
    log.info("translating query (rewrite / expand / HyDE)...")
    rewritten = rewrite_query(agent, query)
    expansions = expand_queries(agent, rewritten or query, n=expansion_n)
    hyde = hyde_document(agent, rewritten or query)
    variants = {"original": query, "rewrite": rewritten, "hyde": hyde}
    variants.update({f"expansion_{i+1}": e for i, e in enumerate(expansions)})
    for name, text in variants.items():
        log.info("  [%s] %s", name, text[:90])
    return variants


# ---------------------------------------------------------------- multi-query search
def dedupe_docs(docs: list[Document]) -> list[Document]:
    seen, out = set(), []
    for d in docs:
        key = d.page_content.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(d)
    return out


def ask(query: str, factory: RetrieverFactory | None = None,
        expansion_n: int = 3, return_context: bool = False):
    """Full pipeline: translate -> retrieve per variant -> pool -> rerank."""
    if factory is None:
        factory = RetrieverFactory.cached()
    agent = get_agent()
    variants = translate_query(agent, query, expansion_n=expansion_n)

    base = factory.get_base_ensemble()
    compressor = factory.get_compressor()

    # retrieve with every textual variant (original/rewrite/expansions)
    pooled: list[Document] = []
    for name, text in variants.items():
        if name == "hyde":
            continue  # HyDE embeds differently, handled below
        pooled.extend(base.invoke(text))

    # HyDE: embed the hypothetical answer and do a pure vector search with it
    vs = factory.vector_store
    hyde_hits = vs.similarity_search(variants["hyde"], k=10)
    pooled.extend(hyde_hits)

    pooled = dedupe_docs(pooled)
    log.info("pooled %d unique chunks; reranking...", len(pooled))

    reranked = compressor.compress_documents(pooled, query)
    # for i, res in enumerate(reranked, 1):
    #     src = Path(res.metadata.get("source_file", "Unknown")).name
    #     print(f"\n[Result {i}] (Source: {src})\n{res.page_content[:300]}...")

    context_parts = []

    for i, res in enumerate(reranked, 1):
        src = Path(res.metadata.get("source_file", "Unknown")).name

        context_parts.append(
            f"[Document {i} | Source: {src}]\n{res.page_content}"
        )

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
    {context}

    User Question:
    {query}

    Answer:"""

    response = _agent_text(agent, answer_prompt, thread_id="chat")

    print(response)

    if return_context:
        return reranked
    return None
    

# ---------------------------------------------------------------- entrypoint
if __name__ == "__main__":

    sync_documents()
    ask("What are the different set operations?")
