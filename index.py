import os
import chromadb
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama

def create_index(
    chroma_path: str = "./OSMO",
    collection_name: str = "dataset_laymen",
    embedding_model: str = "BAAI/bge-small-en-v1.5",\
    


    
):
    """Tworzy i zwraca query engine dla podanych parametrów."""
    
    # Inicjalizacja ChromaDB
    db = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Model embeddingowy
    embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    
    # Indeks
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        embed_model=embed_model
    )   
    # Query engine
    return index


def create_llm(llm_model: str = "qwen2:7b"):
    """Tworzy wybrany LLM"""
    llm = Ollama(model=llm_model)
    return llm
