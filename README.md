# 📚 RAG Chatbot

A web-based RAG (Retrieval-Augmented Generation) chatbot that can answer questions based on your uploaded documents.

## 🌟 Features

- 📄 Upload and process multiple document types (PDF, DOCX, TXT)
- 💬 Interactive chat interface
- 🔍 Source attribution for answers
- 🧠 Uses GPT-3.5 Turbo for natural responses
- 📊 Vector storage for efficient retrieval
- 🔄 Persistent document index

## 🛠️ Tech Stack

- Streamlit (Web UI)
- OpenAI GPT-3.5/4 (LLM)
- LlamaIndex (Document Processing)
- LangChain (Chat Chain)
- ChromaDB (Vector Storage)

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key

## 🚀 Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## 🎮 Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Upload documents:
   - Use the sidebar to upload your documents (PDF, DOCX, or TXT)
   - Click "Process Documents" to index them

3. Chat with your documents:
   - Type questions in the chat input
   - The bot will respond using information from your documents
   - Click the "Sources" expander to see which documents were used

## 📁 Project Structure

- `app.py`: Streamlit web interface
- `ingest.py`: Document processing using LlamaIndex
- `query.py`: Chat functionality using LangChain
- `data/`: Uploaded documents
- `storage/`: Vector store and index

## ⚠️ Important Notes

- Keep your OpenAI API key secure
- Large documents may take longer to process
- The quality of answers depends on the relevance of uploaded documents 