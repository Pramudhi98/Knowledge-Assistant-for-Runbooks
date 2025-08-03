import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
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
    
    # Updated prompt template with clearer instructions for follow-up questions
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    
    Provide a complete answer based on the context first.
    
    Then, on a new line after two line breaks, ask ONE specific follow-up question that will help you provide more targeted assistance. 
    Your follow-up question should:
    - Ask for specific details about their situation or environment
    - Inquire about related procedures they might need
    - Ask about their experience level or specific challenges
    - Offer to explain alternatives or dive deeper into specific steps
    - Be genuinely helpful in gathering information to provide better assistance

    Format your response exactly like this:
    [Your complete answer here]

    [Your follow-up question here ending with a question mark?]

    Context: {context}

    Question: {question}

    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.2,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    return RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )