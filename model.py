import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI



# Load .env
load_dotenv()

# Embedding Model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS with safety flag
vectorstore = FAISS.load_local(
    "vector_store",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                             temperature=0.2,
                             google_api_key=os.getenv("GEMINI_API_KEY"))

# RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Ask a question
query = "How do I restart the web server?"
response = qa_chain.run(query)
print("🤖 Gemini:", response)
