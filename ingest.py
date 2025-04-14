import os
from pathlib import Path
from typing import List
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.node_parser import SimpleNodeParser
import chromadb
from dotenv import load_dotenv

class DocumentIngester:
    def __init__(self, storage_dir: str = "storage"):
        # Load environment variables
        load_dotenv()
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create data directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(self.storage_dir / "chroma"))
        self.collection = self.chroma_client.get_or_create_collection("documents")
        
        # Initialize embedding model
        embed_model = OpenAIEmbedding()
        
        # Initialize vector store
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.collection,
            embedding_model=embed_model
        )
        
        try:
            # Try to load existing storage context
            self.storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore.from_persist_dir(str(self.storage_dir)),
                vector_store=self.vector_store,
                index_store=SimpleIndexStore.from_persist_dir(str(self.storage_dir)),
            )
        except FileNotFoundError:
            # If files don't exist, create new storage context
            self.storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                vector_store=self.vector_store,
                index_store=SimpleIndexStore(),
            )
            # Persist empty storage context
            self.storage_context.persist(persist_dir=str(self.storage_dir))

    def load_documents(self, data_dir: str = "data") -> List[str]:
        """Load documents from the data directory"""
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")
            
        reader = SimpleDirectoryReader(data_dir)
        documents = reader.load_data()
        return documents

    def create_or_update_index(self, documents=None):
        """Create or update the vector index with new documents"""
        try:
            # Try to load existing index
            index = load_index_from_storage(
                storage_context=self.storage_context,
            )
            
            # If documents provided, update the index
            if documents:
                for doc in documents:
                    index.insert(doc)
                
        except:
            # If no existing index, create new one
            if not documents:
                documents = self.load_documents()
            
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
            )
        
        # Persist the index
        index.storage_context.persist(persist_dir=str(self.storage_dir))
        return index

def main():
    """Main function to test the ingestion process"""
    try:
        ingester = DocumentIngester()
        # Check if there are any files in the data directory
        data_dir = Path("data")
        if not any(data_dir.iterdir()):
            print("No documents found in the data directory. Please add some documents first.")
            return
            
        documents = ingester.load_documents()
        index = ingester.create_or_update_index(documents)
        print(f"Successfully processed {len(documents)} documents")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 