import asyncio
from mcp.server.fastmcp import FastMCP
import rag
import local_search
import logging
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import requests
import time

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "mistral-small3.1:latest")


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
                logger.info(
                    f"Retrying connection to Ollama at {base_url}... ({attempt + 1}/{retries})"
                )
                time.sleep(retry_delay)
            else:
                return False
    return False


# Create FastMCP instance
mcp = FastMCP(
    name="local_rag",
    version="1.0.0",
    description="Local RAG system using Ollama for embeddings and LLM capabilities. Search through local documents and get AI-enhanced answers.",
)

# Initialize the LLM if Ollama is available
ollama_available = check_ollama_connection(OLLAMA_BASE_URL, retries=2)
if ollama_available:
    logger.info(f"Using Ollama LLM with model {OLLAMA_LLM_MODEL}")
    llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=OLLAMA_LLM_MODEL)
else:
    logger.warning(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}")
    llm = None

# In-memory storage for the vector store
vectorstore_cache = None


@mcp.tool()
async def search_local_tool(query: str) -> str:
    """
    Search local documents for the given query and enhance with RAG results

    Args:
        query: The search query string

    Returns:
        Formatted search results enriched with RAG-based content
    """
    global vectorstore_cache

    logger.info(f"Searching local documents for query: {query}")

    # Create or get the vector store
    if vectorstore_cache is None:
        logger.info("Creating new vector store from documents")
        vectorstore_cache = await rag.create_rag_from_directory()

    if vectorstore_cache is None:
        return "No documents found in the data directory. Please add documents to the data directory first."

    # Get formatted document listing
    formatted_results, _ = await local_search.search_local(query)

    # Perform RAG search
    rag_results = await rag.search_rag(query, vectorstore_cache)

    if not rag_results:
        return f"{formatted_results}\n\nNo relevant content found for your query."

    # Format the results
    full_results = f"{formatted_results}\n\n### RAG Results:\n\n"
    full_results += "\n---\n".join(doc.page_content for doc in rag_results)

    return full_results


@mcp.tool()
async def get_document_content_tool(file_path: str) -> str:
    """
    Fetch the content of a specific document

    Args:
        file_path: Path to the document file

    Returns:
        The content of the document
    """
    try:
        documents = await local_search.get_document_content(file_path)
        if documents:
            return "\n\n".join([doc.page_content for doc in documents])
        return "Unable to retrieve document content."
    except Exception as e:
        return f"An error occurred while fetching document content: {str(e)}"


@mcp.tool()
async def ask_rag_question(query: str) -> str:
    """
    Ask a question using RAG and get an AI-generated answer

    Args:
        query: The question to ask

    Returns:
        AI-generated answer based on relevant document content
    """
    global vectorstore_cache

    logger.info(f"Asking RAG question: {query}")

    # Check if Ollama LLM is available
    if not ollama_available or llm is None:
        return "Cannot generate AI answer: Ollama LLM is not available. Please make sure Ollama is running and has the required model installed."

    # Create or get the vector store
    if vectorstore_cache is None:
        logger.info("Creating new vector store from documents")
        vectorstore_cache = await rag.create_rag_from_directory()

    if vectorstore_cache is None:
        return "No documents found in the data directory. Please add documents to the data directory first."

    # Perform RAG search
    rag_results = await rag.search_rag(query, vectorstore_cache, k=5)

    if not rag_results:
        return "I don't have enough information in my knowledge base to answer this question."

    # Combine all retrieved document contents
    context = "\n\n".join([doc.page_content for doc in rag_results])

    # Create a prompt for the LLM
    prompt = f"""
You are a helpful AI assistant. Use the following context to answer the question. 
If you don't know the answer based on the context, say that you don't know.

Context:
{context}

Question: {query}

Answer:
"""

    # Get response from Ollama
    try:
        response = await llm.ainvoke(prompt)
        return response
    except Exception as e:
        logger.error(f"Error invoking Ollama LLM: {e}")
        return f"Error generating answer: {str(e)}"


@mcp.tool()
async def refresh_knowledge_base() -> str:
    """
    Refresh the RAG knowledge base by reloading all documents from the data directory

    Returns:
        Status message
    """
    global vectorstore_cache

    logger.info("Refreshing knowledge base")

    try:
        vectorstore_cache = await rag.create_rag_from_directory()
        if vectorstore_cache is None:
            return (
                "No documents found in the data directory. Please add documents first."
            )
        return "Knowledge base refreshed successfully."
    except Exception as e:
        logger.error(f"Error refreshing knowledge base: {e}")
        return f"Error refreshing knowledge base: {str(e)}"


@mcp.tool()
async def check_ollama_status() -> str:
    """
    Check the status of the Ollama server

    Returns:
        Status message about Ollama connection and available models
    """
    try:
        if check_ollama_connection(OLLAMA_BASE_URL, retries=1):
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    model_list = "\n".join([f"- {model['name']}" for model in models])
                    return f"Ollama is running at {OLLAMA_BASE_URL}\n\nAvailable models:\n{model_list}"
                else:
                    return f"Ollama is running at {OLLAMA_BASE_URL}, but no models are available. You may need to pull some models."
            else:
                return f"Ollama is running at {OLLAMA_BASE_URL}, but there was an error getting model information."
        else:
            return f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Please make sure Ollama is running and accessible."
    except Exception as e:
        return f"Error checking Ollama status: {str(e)}"


def get_tools(retriever=None):
    """
    Get all the tools registered with the MCP server

    Args:
        retriever: Optional retriever to use for RAG-based tools

    Returns:
        List of MCP tools
    """
    tools = [
        search_local_tool,
        get_document_content_tool,
        ask_rag_question,
        refresh_knowledge_base,
        check_ollama_status,
    ]
    return tools


if __name__ == "__main__":
    # Run the MCP server
    print(f"Starting local RAG MCP server...")
    print(f"Ollama base URL: {OLLAMA_BASE_URL}")
    print(f"Ollama model: {OLLAMA_LLM_MODEL}")
    print(f"Data directory: {os.getenv('DATA_DIR', './data')}")

    # Check if Ollama is available
    if not ollama_available:
        print("\n⚠️ WARNING: Cannot connect to Ollama at {OLLAMA_BASE_URL}")
        print("The ask_rag_question tool will not work properly.")
        print("RAG search will use dummy embeddings (random vectors).")
        print("MCP server will still start, but functionality will be limited.")
        print("\nPossible solutions:")
        print("1. Make sure Ollama is running on the target machine")
        print("2. Check if there's a firewall blocking the connection")
        print("3. Verify the IP address and port are correct")
        print("4. Update the .env file with the correct OLLAMA_BASE_URL")
        print("\nStarting MCP server anyway...\n")

    # Run the MCP server
    mcp.serve(
        host="localhost",
        port=8000,
        sse=True,  # Enable Server-Sent Events for real-time updates
    )
