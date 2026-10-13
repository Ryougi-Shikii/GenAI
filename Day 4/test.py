
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  
BASE_DIR = Path(BASE_DIR).resolve().parent  

DB_PATH = str(BASE_DIR / "chroma_db")
REGISTRY_FILE = str(BASE_DIR / "doc_registry.json")
BM25_CACHE_FILE = str(BASE_DIR / "bm25_cache.pkl")
DOC_FOLDER = str(BASE_DIR / "document")
HF_CACHE = str(BASE_DIR / "hf_models_cache")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"


lis = [DB_PATH, REGISTRY_FILE, BM25_CACHE_FILE]
for i in lis:
    print(i)

