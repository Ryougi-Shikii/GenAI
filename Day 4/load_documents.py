from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", cache_folder="./hf_models_cache")

print(">loading documents...")
folder_path = "document"
docs = []
for pdf_path in Path(folder_path).glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_path))
    docs.extend(loader.load())
    print(f"    {pdf_path} done!")
print(f">{len(docs)} documents loaded!")

print(">chuncking started...")
text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90
)
chunks = text_splitter.split_documents(docs)
print(">chunking done!")

print(f"\n>Total documents: {len(docs)}")
print(f">Total semantic chunks: {len(chunks)}")
