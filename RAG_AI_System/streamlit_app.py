# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import streamlit as st
import requests
from pathlib import Path
import json
import time
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# - - = PAGE CONFIGURATION = - -
# ---------------------------------------------------------------------------
# page config
st.set_page_config(
    page_title="RAG System @ 2025",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'sources' not in st.session_state:
    st.session_state.sources = []

# ---------------------------------------------------------------------------
# - - = API FUNCTIONS = - -
# ---------------------------------------------------------------------------
def query_rag(question: str, k: int = 5, verbose: bool = False) -> Dict[str, Any]:
    """query the RAG API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "question": question,
                "k": k,
                "return_sources": True,
                "verbose": verbose
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying RAG system: {e}")
        return None

def upload_document(file) -> Dict[str, Any]:
    """upload document to RAG API"""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading document: {e}")
        return None

def clear_history():
    """clear conversation history"""
    try:
        response = requests.post(f"{API_BASE_URL}/clear-history")
        response.raise_for_status()
        st.session_state.messages = []
        st.session_state.sources = []
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error clearing history: {e}")
        return False

# ---------------------------------------------------------------------------
# - - = MAIN UI = - -
# ---------------------------------------------------------------------------
# main UI
st.title("RAG System")
st.markdown("*A state-of-the-art Retrieval-Augmented Generation system with hybrid search and reranking* (2025 Edition)")

# ---------------------------------------------------------------------------
# - - = SIDEBAR CONFIGURATION = - -
# ---------------------------------------------------------------------------
# sidebar
with st.sidebar:
    st.header("Configuration")
    
    # retrieval settings
    st.subheader("Retrieval Settings")
    k_value = st.slider("Number of documents to retrieve", 1, 20, 5)
    verbose_mode = st.checkbox("Verbose mode (show retrieved chunks)", value=False)
    
    # document upload
    st.subheader("Document Upload")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'md', 'docx'],
        help="Upload documents to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("Upload Document", type="primary"):
            with st.spinner("Processing document..."):
                result = upload_document(uploaded_file)
                if result:
                    st.success(f"Processed {result['chunks_created']} chunks from document")
    
    # system info
    st.subheader("System Info")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            info = response.json()
            st.info(f"""
            **Vector Store:** {info['vector_store']}  
            **Embedding Model:** {info['embedding_model']}  
            **LLM Model:** {info['llm_model']}
            """)
    except:
        st.warning("API not available")
    
    # clear history button
    if st.button("Clear Conversation", type="secondary"):
        if clear_history():
            st.success("Conversation history cleared")
            st.rerun()

# ---------------------------------------------------------------------------
# - - = CHAT INTERFACE = - -
# ---------------------------------------------------------------------------
# main chat interface
st.subheader("Chat Interface")

# display chat messages
message_container = st.container()
with message_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- **{source['file']}** (Relevance: {source.get('relevance_score', 0):.2f})")

# chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # query RAG system
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_rag(prompt, k=k_value, verbose=verbose_mode)
            
            if result:
                # display response
                st.markdown(result["answer"])
                
                # store message with sources
                message_data = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", [])
                }
                st.session_state.messages.append(message_data)
                
                # show sources
                if result.get("sources"):
                    with st.expander("Sources"):
                        for source in result["sources"]:
                            st.markdown(f"- **{source['file']}** (Relevance: {source.get('relevance_score', 0):.2f})")
                
                # show retrieved documents in verbose mode
                if verbose_mode and result.get("retrieved_documents"):
                    with st.expander("Retrieved Documents"):
                        for i, doc in enumerate(result["retrieved_documents"], 1):
                            st.markdown(f"**Document {i}:**")
                            st.text(doc["content"])
                            st.json(doc["metadata"])
                            st.divider()
            else:
                st.error("Failed to get response from RAG system")

# ---------------------------------------------------------------------------
# - - = FOOTER = - -
# ---------------------------------------------------------------------------
# footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>@sebieire (2025)</p>    
</div>
""", unsafe_allow_html=True)