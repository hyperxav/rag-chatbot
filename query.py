from pathlib import Path
from typing import List, Optional
import os
from dotenv import load_dotenv, find_dotenv
import chromadb

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# GLAMA API configuration
GLAMA_API_BASE = "https://glama.ai/api/gateway/openai/v1"

class ChatBot:
    def __init__(self, storage_dir: str = "storage"):
        # Find and load .env file
        dotenv_path = find_dotenv()
        print(f"Found .env file at: {dotenv_path}")
        load_dotenv(dotenv_path, override=True)
        
        # Get API key and check its format
        api_key = os.getenv("GLAMA_API_KEY")
        if not api_key:
            raise ValueError("GLAMA_API_KEY not found in environment variables")
        print(f"API key loaded, starts with: {api_key[:8]}...")
        print(f"API key length: {len(api_key)}")
        
        # Set API key explicitly
        os.environ["OPENAI_API_KEY"] = api_key.strip()  # Remove any whitespace
        os.environ["OPENAI_API_BASE"] = GLAMA_API_BASE
            
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        try:
            # Initialize embedding model with explicit API key
            self.embed_model = OpenAIEmbedding(
                api_key=api_key,
                api_base=GLAMA_API_BASE
            )
            print("Successfully initialized OpenAIEmbedding")
            
            # Set the embed model as the default
            Settings.embed_model = self.embed_model
            
            # Initialize ChromaDB
            chroma_client = chromadb.PersistentClient(path=str(self.storage_dir / "chroma"))
            chroma_collection = chroma_client.get_or_create_collection("documents")
            
            # Initialize vector store
            vector_store = ChromaVectorStore(
                chroma_collection=chroma_collection,
                embedding_model=self.embed_model
            )

            try:
                # Try to load existing index
                storage_context = StorageContext.from_defaults(
                    docstore=SimpleDocumentStore.from_persist_dir(str(self.storage_dir)),
                    vector_store=vector_store,
                    index_store=SimpleIndexStore.from_persist_dir(str(self.storage_dir)),
                )
                self.index = load_index_from_storage(storage_context)
                print("Successfully loaded existing index")
            except:
                print("No existing index found, creating empty index")
                # Create new storage context and empty index
                storage_context = StorageContext.from_defaults(
                    docstore=SimpleDocumentStore(),
                    vector_store=vector_store,
                    index_store=SimpleIndexStore(),
                )
                self.index = VectorStoreIndex([], storage_context=storage_context)
                # Persist empty index
                self.index.storage_context.persist(persist_dir=str(self.storage_dir))
            
            # Initialize retriever
            self.retriever = self.index.as_retriever(similarity_top_k=3)
            
            # Initialize LLM with explicit API key and base URL
            self.llm = ChatOpenAI(
                temperature=0.1,
                model="gpt-4",
                api_key=api_key,
                base_url=GLAMA_API_BASE
            )
            print("Successfully initialized ChatOpenAI")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise
        
        # Create prompt template
        prompt_template = """You are a helpful AI assistant that answers questions based on the provided context.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer: Let me help you with that."""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create chain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
        )

    def get_response(self, query: str) -> tuple[str, List[str]]:
        """
        Get response for a query and return both the answer and source documents
        """
        # Get response from chain
        response = self.chain({"query": query})
        
        # Extract answer and source documents
        answer = response['result']
        source_docs = [doc.metadata.get('source', 'Unknown') for doc in response['source_documents']]
        
        return answer, source_docs

def main():
    """Test the chat functionality"""
    try:
        chatbot = ChatBot()
        print("ChatBot initialized successfully")
    except Exception as e:
        print(f"Failed to initialize ChatBot: {str(e)}")
        return

if __name__ == "__main__":
    main() 