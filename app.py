import os
import streamlit as st
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qa import get_qa_chain

# Constants
VECTOR_STORE_PATH = "vector_store"
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("🧠 Knowledge Assistant for Runbooks and SOPs")
uploaded_files = st.file_uploader("Upload SOP PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.info("🔄 Processing uploaded PDFs...")
    all_texts = []
    for uploaded_file in uploaded_files:
        path = f"sop_docs/{uploaded_file.name}"
        with open(path, "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader(path)
        all_texts.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(all_texts)
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(VECTOR_STORE_PATH)
    st.success("✅ Uploaded SOPs have been processed!")

def rank_steps(text):
    lines = text.split('\n')
    numbered_line_pattern = re.compile(r'^(\d+)[\).]\s*(.*)')
    steps = []
    for line in lines:
        match = numbered_line_pattern.match(line.strip())
        if match:
            steps.append(match.group(2).strip())
        elif line.strip().startswith(('-', '*', '•', 'Step', 'step', 'STEP')):
            steps.append(line.lstrip('-*• ').strip())
    if not steps:
        split_steps = re.split(r'\n\d+[\).]', text)
        steps = [s.strip() for s in split_steps if s.strip()]
    if steps:
        return '\n'.join([f"{i+1}. {step}" for i, step in enumerate(steps)])
    return text

# Input
query = st.text_input("Ask your question:")

if 'qa_history' not in st.session_state:
    st.session_state['qa_history'] = []

if query:
    if not os.path.exists(VECTOR_STORE_PATH):
        # Try to load PDFs from sop_docs
        sop_files = [f for f in os.listdir('sop_docs') if f.lower().endswith('.pdf')]
        if sop_files:
            all_texts = []
            for filename in sop_files:
                loader = PyPDFLoader(f"sop_docs/{filename}")
                all_texts.extend(loader.load())
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = splitter.split_documents(all_texts)
            vectorstore = FAISS.from_documents(docs, embedding_model)
            vectorstore.save_local(VECTOR_STORE_PATH)
        else:
            st.error("❌ No SOP PDFs found. Please upload some first.")
            st.stop()

    if 'last_question' not in st.session_state or st.session_state['last_question'] != query:
        qa_chain = get_qa_chain()
        result = qa_chain(query)
        answer = rank_steps(result['result'])
        st.session_state['qa_history'].append((query, answer))
        st.session_state['last_question'] = query
    else:
        answer = st.session_state['qa_history'][-1][1]

    st.markdown(f"**🤖 Gemini:**\n\n{answer}")

    # Feedback section
    st.markdown("### 🙋 Was this answer helpful?")
    feedback = st.radio(
        "Feedback:",
        ["👍 Yes", "👎 No"],
        index=None,
        horizontal=True,
        key=f"feedback_{query}"
    )
    if feedback:
        if feedback == "👎 No":
            st.error("Thanks for your feedback. We'll work on improving.")
        else:
            st.success("Thanks for the positive feedback!")

# Q&A History
if st.session_state['qa_history']:
    st.markdown("---")
    st.markdown("### 🕓 Previous Q&A")
    for i, (q, a) in enumerate(reversed(st.session_state['qa_history']), 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
        st.markdown("---")
