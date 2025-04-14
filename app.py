import streamlit as st
import os
from pathlib import Path
from typing import List
import tempfile
from dotenv import load_dotenv

from ingest import DocumentIngester
from query import ChatBot

# Load environment variables
load_dotenv()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chatbot" not in st.session_state:
    try:
        st.session_state.chatbot = ChatBot()
    except ValueError as e:
        st.error("⚠️ LLAMA API key not found. Please create a .env file with your LLAMA_API_KEY.")
        st.stop()

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to data directory and return the file path"""
    try:
        # Create data directory if it doesn't exist
        save_dir = Path("data")
        save_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = save_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def process_documents():
    """Process documents using DocumentIngester"""
    try:
        ingester = DocumentIngester()
        documents = ingester.load_documents()
        index = ingester.create_or_update_index(documents)
        # Reinitialize chatbot with new index
        st.session_state.chatbot = ChatBot()
        return True
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False

def main():
    st.title("📚 RAG Chatbot")
    st.write("Upload documents and chat with an AI that can answer questions based on their content!")

    # Sidebar for file upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    # Save all uploaded files
                    for uploaded_file in uploaded_files:
                        file_path = save_uploaded_file(uploaded_file)
                        if file_path:
                            st.success(f"Saved: {uploaded_file.name}")
                    
                    # Process all documents
                    if process_documents():
                        st.success("✅ All documents processed successfully!")
                    else:
                        st.error("❌ Error processing documents")

    # Chat interface
    st.header("Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Display sources if available
            if "sources" in message:
                with st.expander("🔍 Sources"):
                    for source in message["sources"]:
                        st.write(f"- {Path(source).name}")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, sources = st.session_state.chatbot.get_response(prompt)
                    st.write(response)
                    
                    # Display sources
                    if sources:
                        with st.expander("🔍 Sources"):
                            for source in sources:
                                st.write(f"- {Path(source).name}")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                except Exception as e:
                    error_msg = "❌ Error: Failed to get response. Make sure you've uploaded and processed documents first."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if __name__ == "__main__":
    main() 