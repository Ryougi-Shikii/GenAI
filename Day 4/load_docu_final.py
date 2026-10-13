import hashlib
import json
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

DB_PATH = "./chroma_db"
REGISTRY_FILE = "./doc_registry.json"

def compute_file_hash(file_path: Path) -> str:
    """Computes a unique SHA-256 hash of a file to check for changes."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def load_registry() -> dict:
    if Path(REGISTRY_FILE).exists():
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_registry(registry: dict):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=4)

print("> loading embedding model")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", 
    cache_folder="./hf_models_cache"
)

registry = load_registry()
folder_path = "document"
current_files = list(Path(folder_path).glob("*.pdf"))
current_paths = {str(p.resolve()) for p in current_files}
deleted_files = set(registry.keys()) - current_paths

vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

if deleted_files:
    print("Detected deleted files")
    print("removing from registry and vector store...")
    for df in deleted_files:
        print(f"    [-] removing: {Path(df).name}")
        vector_store.delete(where={"source_file": df})
        del registry[df]
    save_registry(registry)

new_docs = []

print("> Checking documents for changes...")
for pdf_path in current_files:
    path_str = str(pdf_path.resolve())
    file_hash = compute_file_hash(pdf_path)
    
    # If file is new based on its hash
    if path_str not in registry or registry[path_str]["hash"] != file_hash:
        print(f"    [+] Changes detected in: {pdf_path.name} (Processing...)")
        
        if path_str in registry:
            vector_store.delete(where={"source_file": path_str})
            
        loader = PyPDFLoader(path_str)
        file_docs = loader.load()
        
        # Tag metadata so we know which file chunks belong to
        for doc in file_docs:
            doc.metadata["source_file"] = path_str
            
        new_docs.extend(file_docs)
        registry[path_str] = {"hash": file_hash}
    else:
        print(f"    [*] Unchanged: {pdf_path.name} (Skipping)")

# If there are new/modified docs, chunk them and add them to Chroma
if new_docs:
    print(f"\n> Chunking {len(new_docs)} newly loaded pages...")
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )
    new_chunks = text_splitter.split_documents(new_docs)
    
    print(f"> Adding {len(new_chunks)} new chunks to vector database...")
    vector_store.add_documents(new_chunks)
    save_registry(registry)
    print("> Database updated and saved!")
else:
    print("\n> No changes detected in any documents.")
    print("Loaded existing vector store instantly.")




# 1. Dense Vector Retriever
vector_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.3}
)

# 2. Sparse Keyword Retriever (BM25)
all_stored_docs = vector_store.get(include=["documents", "metadatas"])
bm25_source_docs = [
    Document(page_content=doc, metadata=meta)
    for doc, meta in zip(all_stored_docs["documents"], all_stored_docs["metadatas"])
]

if bm25_source_docs:
    bm25_retriever = BM25Retriever.from_documents(bm25_source_docs)
    bm25_retriever.k = 10

    # 3. Ensemble Hybrid Retriever (BM25 + Vector)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
else:
    ensemble_retriever = vector_retriever

# 4. Cross-Encoder Reranker
print("> Loading Cross-Encoder reranker model...")
rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=rerank_model, top_n=10)

production_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=ensemble_retriever
)

# Test search
query = "What are the different set operations?"
print(f"\n> Running production hybrid test search for: '{query}'")
results = production_retriever.invoke(query)

for i, res in enumerate(results, 1):
    source_name = Path(res.metadata.get('source_file', 'Unknown')).name
    print(f"\n[Production Result {i}] (Source: {source_name})")
    print(res.page_content[:300] + "...")