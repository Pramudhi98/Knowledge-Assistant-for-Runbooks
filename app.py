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

st.title("🧠 Interactive Knowledge Assistant")
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

def format_response_with_followup(response_text):
    """Separate the main answer from the follow-up question"""
    # More flexible follow-up question patterns - looking for question marks at the end
    followup_patterns = [
        # Look for questions that start after a paragraph break
        r'\n\n([^.]*\?[^?]*?)$',
        r'\n\n(.{0,200}?\?)\s*$',
        # Look for questions after certain trigger words
        r'((?:Would you|Do you|Are you|Can I|Should I|Could you|What|Which|How|When|Where|Why)[^.]*?\?)\s*$',
        # Look for any sentence ending with ? in the last part
        r'([^.!]*\?)\s*$',
        # Broader pattern for questions
        r'\?\s*\n\n(.+\?)$',
    ]
    
    # Try each pattern
    for i, pattern in enumerate(followup_patterns):
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            # Find where the question starts
            question_start = match.start(1) if match.lastindex else match.start()
            main_answer = response_text[:question_start].strip()
            followup_question = match.group(1).strip()
            return main_answer, followup_question
    
    # Alternative approach: split by double newlines and check last parts
    parts = response_text.split('\n\n')
    for i in range(len(parts) - 1, max(0, len(parts) - 3), -1):  # Check last 2 parts
        if '?' in parts[i]:
            main_answer = '\n\n'.join(parts[:i]).strip()
            followup_question = parts[i].strip()
            return main_answer, followup_question
    
    # Final fallback: look for any question mark in the last 200 characters
    if '?' in response_text[-200:]:
        # Find the last sentence with a question mark
        sentences = re.split(r'[.!]\s+', response_text)
        for i in range(len(sentences) - 1, -1, -1):
            if '?' in sentences[i]:
                main_answer = '. '.join(sentences[:i]).strip()
                if main_answer and not main_answer.endswith('.'):
                    main_answer += '.'
                followup_question = sentences[i].strip()
                return main_answer, followup_question
    
    return response_text, None

# Input
query = st.chat_input("Ask your question:")

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

    # Check if this is a follow-up response
    if st.session_state.get('waiting_for_followup', False):
        # Handle follow-up response
        st.session_state['followup_answered'] = True
        st.session_state['followup_response'] = query
        st.session_state['waiting_for_followup'] = False
        
        st.markdown(f"**Your response:** {query}")
        
        # Check if response is positive (indicating satisfaction)
        positive_indicators = ['yes', 'good', 'satisfied', 'fine', 'ok', 'okay', 'great', 'perfect', 'thanks', 'thank you', 'sufficient', 'enough', 'clear', 'understood']
        is_positive_response = any(indicator in query.lower() for indicator in positive_indicators)
        
        if is_positive_response:
            st.success("✅ Great! I'm glad the information was helpful.")
            st.session_state['conversation_ended'] = True
        else:
            # Generate response based on follow-up for negative/neutral responses
            qa_chain = get_qa_chain()
            original_query = st.session_state['qa_history'][-1][0]
            followup_result = qa_chain(f"{original_query} Additional context: {query}")
            followup_answer, _ = format_response_with_followup(followup_result['result'])
            formatted_followup = rank_steps(followup_answer)
            st.markdown(f"**🤖 Updated Response:**\n\n{formatted_followup}")
            st.session_state['conversation_ended'] = True
    else:
        # Handle initial question
        if 'last_question' not in st.session_state or st.session_state['last_question'] != query:
            qa_chain = get_qa_chain()
            result = qa_chain(query)
            raw_answer = result['result']
            
            # Format the response and extract follow-up question
            main_answer, followup_question = format_response_with_followup(raw_answer)
            formatted_answer = rank_steps(main_answer)
            
            st.session_state['qa_history'].append((query, formatted_answer, followup_question))
            st.session_state['last_question'] = query
            st.session_state['followup_answered'] = False
            st.session_state['conversation_ended'] = False
        else:
            formatted_answer = st.session_state['qa_history'][-1][1]
            followup_question = st.session_state['qa_history'][-1][2] if len(st.session_state['qa_history'][-1]) > 2 else None

        st.markdown(f"**🤖 Bot:**\n\n{formatted_answer}")
        
        # Display follow-up question if available and not answered
        if followup_question and not st.session_state.get('followup_answered', False):
            st.markdown("---")
            st.markdown("**🤔 To help you better:**")
            st.info(followup_question)
            st.session_state['waiting_for_followup'] = True
            st.info("💬 Please use the chat input below to respond to the follow-up question.")

# Show satisfaction feedback only after follow-up is answered or conversation ended
if st.session_state.get('conversation_ended', False) and not st.session_state.get('feedback_given', False):
    st.markdown("### 💭 Are you satisfied with the answer?")
    satisfaction_key = f"satisfaction_{len(st.session_state['qa_history'])}"
    satisfaction = st.radio(
        "Please let us know:",
        ["✅ Yes, I'm satisfied", "❌ No, I need more help"],
        index=None,
        horizontal=True,
        key=satisfaction_key
    )
    
    if satisfaction:
        st.session_state['feedback_given'] = True
        if satisfaction == "❌ No, I need more help":
            st.error("😔 Sorry, our knowledge base will be updated soon to provide better assistance.")
        elif satisfaction == "✅ Yes, I'm satisfied":
            st.success("🙏 Thank you! We're glad we could help.")

# Q&A History
if st.session_state['qa_history']:
    st.markdown("---")
    st.markdown("### 🕓 Previous Q&A")
    for i, qa_item in enumerate(reversed(st.session_state['qa_history']), 1):
        q, a = qa_item[0], qa_item[1]
        followup = qa_item[2] if len(qa_item) > 2 else None
        
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
        if followup:
            st.markdown(f"**Follow-up:** {followup}")
        st.markdown("---")
