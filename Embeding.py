from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os

# Load all PDFs again
docs = []
folder_path = "sop_docs"
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder_path, filename))
        docs.extend(loader.load())

# Generate Embeddings
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in FAISS Vector DB
vectorstore = FAISS.from_documents(docs, embedding_model)
vectorstore.save_local("vector_store")

print("✅ Documents embedded and vector store created.")
