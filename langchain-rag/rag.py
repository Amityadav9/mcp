from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import local_search
from langchain_core.documents import Document
import os
import asyncio
from dotenv import load_dotenv
import requests
import time

# Load environment variables
load_dotenv()

# Get Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "llama3:latest")


# Define a fallback embedding class for testing
class DummyEmbeddings:
    def __init__(self):
        print("⚠️ Using dummy embeddings for testing (random vectors)")

    def embed_documents(self, texts):
        """Return random embeddings for testing"""
        import numpy as np

        return [np.random.rand(384) for _ in texts]

    def embed_query(self, text):
        """Return random embedding for testing"""
        import numpy as np

        return np.random.rand(384)


def check_ollama_connection(base_url, retries=1, retry_delay=1):
    """Check if Ollama server is accessible"""
    for attempt in range(retries):
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return True
            time.sleep(retry_delay)
        except Exception:
            if attempt < retries - 1:
                print(
                    f"Retrying connection to Ollama at {base_url}... ({attempt + 1}/{retries})"
                )
                time.sleep(retry_delay)
            else:
                return False
    return False


async def create_rag(file_paths: list[str]) -> FAISS:
    """
    Create a RAG system from a list of file paths

    Args:
        file_paths: List of file paths to load content from

    Returns:
        FAISS: Vector store object
    """
    try:
        # Check Ollama connection
        ollama_available = check_ollama_connection(OLLAMA_BASE_URL, retries=2)

        # Select embeddings based on availability
        if ollama_available:
            print(f"Using Ollama embeddings with model {OLLAMA_EMBED_MODEL}")
            embeddings = OllamaEmbeddings(
                model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL
            )
        else:
            print(f"⚠️ Cannot connect to Ollama at {OLLAMA_BASE_URL}")
            print("Using dummy embeddings for testing (performance will be poor)")
            embeddings = DummyEmbeddings()

        documents = []
        # Use asyncio.gather to process all document requests in parallel
        tasks = [
            local_search.get_document_content(file_path) for file_path in file_paths
        ]
        results = await asyncio.gather(*tasks)
        for result in results:
            documents.extend(result)

        # Text chunking processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunk size for local documents
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        split_documents = text_splitter.split_documents(documents)
        print(f"Created {len(split_documents)} chunks from {len(documents)} documents")

        vectorstore = FAISS.from_documents(
            documents=split_documents, embedding=embeddings
        )
        return vectorstore
    except Exception as e:
        print(f"Error in create_rag: {str(e)}")
        raise


async def create_rag_from_documents(documents: list[Document]) -> FAISS:
    """
    Create a RAG system directly from a list of documents

    Args:
        documents: List of already loaded documents

    Returns:
        FAISS: Vector store object
    """
    try:
        # Check Ollama connection
        ollama_available = check_ollama_connection(OLLAMA_BASE_URL, retries=2)

        # Select embeddings based on availability
        if ollama_available:
            print(f"Using Ollama embeddings with model {OLLAMA_EMBED_MODEL}")
            embeddings = OllamaEmbeddings(
                model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL
            )
        else:
            print(f"⚠️ Cannot connect to Ollama at {OLLAMA_BASE_URL}")
            print("Using dummy embeddings for testing (performance will be poor)")
            embeddings = DummyEmbeddings()

        # Text chunking processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunk size for local documents
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        split_documents = text_splitter.split_documents(documents)
        print(f"Created {len(split_documents)} chunks from {len(documents)} documents")

        vectorstore = FAISS.from_documents(
            documents=split_documents, embedding=embeddings
        )
        return vectorstore
    except Exception as e:
        print(f"Error in create_rag_from_documents: {str(e)}")
        raise


async def search_rag(query: str, vectorstore: FAISS, k: int = 3) -> list[Document]:
    """
    Search the RAG system for documents relevant to the query

    Args:
        query: The search query
        vectorstore: The FAISS vector store
        k: Number of documents to retrieve

    Returns:
        List of relevant documents
    """
    try:
        documents = vectorstore.similarity_search(query, k=k)
        return documents
    except Exception as e:
        print(f"Error in search_rag: {str(e)}")
        return []


async def create_rag_from_directory() -> FAISS:
    """
    Create a RAG system from all documents in the data directory

    Returns:
        FAISS: Vector store object
    """
    try:
        # Load all documents from the data directory
        documents = await local_search.load_documents()

        if not documents:
            print("No documents found in the data directory")
            return None

        # Create RAG from documents
        return await create_rag_from_documents(documents)
    except Exception as e:
        print(f"Error in create_rag_from_directory: {str(e)}")
        return None
