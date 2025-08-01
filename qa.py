import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

def get_qa_chain():
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Try to load the vector store, if not found, build from sop_docs
    try:
        vectorstore = FAISS.load_local(
            "vector_store", 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
    except Exception:
        # Fallback: build from sop_docs
        pdf_files = glob.glob("sop_docs/*.pdf")
        all_texts = []
        for pdf in pdf_files:
            loader = PyPDFLoader(pdf)
            pages = loader.load()
            all_texts.extend(pages)
        if not all_texts:
            raise RuntimeError("No PDFs found in sop_docs to build the knowledge base.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(all_texts)
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local("vector_store")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.2,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    return RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=vectorstore.as_retriever()
    )
