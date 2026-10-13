import hashlib
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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
for df in deleted_files:
    print(f"    [-] Detected deleted file, removing from registry: {Path(df).name}")
    del registry[df]

vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

new_docs = []

print("> Checking documents for changes...")
for pdf_path in current_files:
    path_str = str(pdf_path.resolve())
    file_hash = compute_file_hash(pdf_path)
    
    # If file is new based on its hash
    if path_str not in registry or registry[path_str]["hash"] != file_hash:
        print(f"    [+] Changes detected in: {pdf_path.name} (Processing...)")
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
    
    print(f"> Adding {len(new_chunks)} new chunks to persistent vector database...")
    vector_store.add_documents(new_chunks)
    save_registry(registry)
    print("> Database updated and saved!")
else:
    print("\n> No changes detected in any documents.")
    print("Loaded existing vector store instantly.")

# Test search
query = "What are the core concepts?"
print(f"\n> Running test search for: '{query}'")
results = vector_store.similarity_search(query, k=2)

for i, res in enumerate(results, 1):
    print(f"\n[Result {i}] (Source: {Path(res.metadata.get('source_file', 'Unknown')).name})")
    print(res.page_content[:300] + "...")