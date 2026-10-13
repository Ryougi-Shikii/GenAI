from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

folder_path = "Day 3/document"
docs = []
for pdf_path in Path(folder_path).glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_path))
    docs.extend(loader.load())


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n"]
)

chunks = text_splitter.split_documents(docs)

print(f"Total documents: {len(docs)}")
print(f"Total chunks: {len(chunks)}")

for i in chunks:
    print(i.page_content)

# vectors = embeddings.embed_documents(
#     [
#         "Today is Monday",
#         "Today is Tuesday",
#         "Today is April Fools day",
#     ]
# )
# print(len(vectors), len(vectors[0]))

# text = "LangChain is the framework for building context-aware reasoning applications"
# vectorstore = InMemoryVectorStore.from_texts(
#     [text],
#     embedding=embeddings,
# )
# retriever = vectorstore.as_retriever()
# retrieved_documents = retriever.invoke("What is LangChain?")

