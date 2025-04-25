import asyncio
import os
import sys
from dotenv import load_dotenv
import argparse
from langchain_ollama import OllamaLLM
import requests
import time

# Import local search and RAG modules
import local_search
import rag

# Load environment variables
load_dotenv()

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
                print(
                    f"Retrying connection to Ollama at {base_url}... ({attempt + 1}/{retries})"
                )
                time.sleep(retry_delay)
            else:
                return False
    return False


# Initialize the LLM if Ollama is available
ollama_available = check_ollama_connection(OLLAMA_BASE_URL, retries=2)
if ollama_available:
    print(f"Using Ollama LLM with model {OLLAMA_LLM_MODEL}")
    llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=OLLAMA_LLM_MODEL)
else:
    print(f"⚠️ Cannot connect to Ollama at {OLLAMA_BASE_URL}")
    llm = None


async def query_documents(query, use_llm=False):
    """
    Query local documents and get RAG results

    Args:
        query: The query string
        use_llm: Whether to use the LLM to generate an answer

    Returns:
        None
    """
    print(f"\nQuerying documents for: {query}")
    print("-" * 50)

    try:
        # Create or refresh the vector store
        print("Creating vector store from local documents...")
        vectorstore = await rag.create_rag_from_directory()

        if vectorstore is None:
            print(
                "No documents found in the data directory. Please add some documents first."
            )
            return

        # Get document listing
        formatted_results, documents = await local_search.search_local(query)

        if not documents:
            print("No documents found.")
            return

        print(f"Found {len(documents)} documents")

        # Perform RAG search
        print("Searching for relevant content...")
        rag_results = await rag.search_rag(query, vectorstore)

        # Format results
        print("\n=== Document Listing ===")
        print(formatted_results)

        print("\n=== RAG Results ===")
        if not rag_results:
            print("No relevant content found. Try a different query.")
        else:
            for doc in rag_results:
                source = doc.metadata.get("source", "Unknown source")
                print(f"\n--- From: {os.path.basename(source)} ---")
                print(doc.page_content)

        # Use LLM to generate an answer if requested
        if use_llm and rag_results:
            print("\n=== AI Answer ===")

            if not ollama_available or llm is None:
                print("⚠️ Cannot generate AI answer: Ollama LLM is not available")
                print(f"Please make sure Ollama is running at {OLLAMA_BASE_URL}")
                print(f"and has the model {OLLAMA_LLM_MODEL} installed.")
                return

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
            print("Generating answer using Ollama...")
            try:
                response = await llm.ainvoke(prompt)
                print(response)
            except Exception as e:
                print(f"Error generating answer: {str(e)}")
                print("This might be due to an issue with the Ollama server or model.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


async def add_sample_documents():
    """Add sample documents to the data directory"""
    data_dir = os.getenv("DATA_DIR", "./data")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at {data_dir}")

    # Create sample documents
    samples = [
        (
            "langchain_info.txt",
            """LangChain is a framework for developing applications powered by language models. It enables applications that:
Are context-aware: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
Reason: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)

The main value props of LangChain are:
Components: abstractions for working with language models, along with a collection of implementations for each abstraction. Components are modular and easy-to-use, whether you are using the rest of the LangChain framework or not
Off-the-shelf chains: for common applications, we provide batteries-included chains that are a collection of components assembled for specific use cases. These contain natural points for customization.""",
        ),
        (
            "rag_explanation.txt",
            """Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based and generation-based approaches to produce more accurate and contextually relevant text.

In RAG systems:
1. The input query is used to retrieve relevant documents or passages from a knowledge base
2. These retrieved documents provide context for the language model
3. The language model generates a response conditioned on both the query and the retrieved context

Benefits of RAG:
- More up-to-date information than what the LLM was trained on
- Grounded responses with citations to source material
- Reduced hallucinations by providing factual context
- Customization for specific domains or knowledge areas

RAG architecture typically consists of:
- A document processing pipeline (splitting, embedding)
- A vector database for semantic search
- A retrieval mechanism to find relevant documents
- A language model that uses the retrieved context to generate responses""",
        ),
        (
            "mcp_protocol.txt",
            """The Multimodal Capabilities Protocol (MCP) is an open protocol that standardizes how applications provide context to LLMs. It enables a common interface for LLMs to call external functions, APIs, and tools during generation.

Key features of MCP:
- Structured function calling allowing LLMs to request specific information
- Tool discovery and registration through a standardized interface
- Streaming data and multi-step reasoning
- Lightweight implementation with minimal dependencies

MCP helps solve the connection problem between LLMs and external tools, allowing developers to create consistent interfaces for AI systems to interact with the world.""",
        ),
        (
            "ollama_guide.txt",
            """Ollama is an open-source framework that allows you to run large language models (LLMs) locally on your machine. It simplifies the process of downloading, setting up, and running these models.

Getting Started with Ollama:
1. Install Ollama from ollama.ai
2. Run models with simple commands like: ollama run llama3
3. Models are automatically downloaded on first use

Ollama supports a variety of models including:
- Llama 3 (8B, 70B)
- Mistral (7B)
- Phi-2 (3B)
- Many others

Ollama provides both an interactive CLI and an API for integration with other applications. The API allows you to:
- Generate text completions
- Create embeddings for vector search
- Configure model parameters
- Manage model libraries""",
        ),
    ]

    for filename, content in samples:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Created sample document: {filename}")


async def main():
    """Main function to parse arguments and run the appropriate command"""
    parser = argparse.ArgumentParser(description="Local RAG System with Ollama")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query local documents")
    query_parser.add_argument("query", nargs="*", help="The search query")
    query_parser.add_argument(
        "--ai", "-a", action="store_true", help="Use Ollama to generate an answer"
    )

    # Add samples command
    subparsers.add_parser(
        "add-samples", help="Add sample documents to the data directory"
    )

    # List documents command
    subparsers.add_parser("list", help="List all documents in the data directory")

    # Test connection command
    subparsers.add_parser("test-connection", help="Test connection to Ollama server")

    args = parser.parse_args()

    # If no command is specified, show help
    if not args.command:
        parser.print_help()
        return

    # Execute the appropriate command
    if args.command == "query":
        if not args.query:
            query = input("Enter your query: ")
        else:
            query = " ".join(args.query)
        await query_documents(query, args.ai)

    elif args.command == "add-samples":
        await add_sample_documents()

    elif args.command == "list":
        documents = await local_search.load_documents()
        if not documents:
            print("No documents found in the data directory.")
            return

        print(f"Found {len(documents)} documents:")
        for doc in documents:
            source = doc.metadata.get("source", "Unknown source")
            print(f"- {os.path.basename(source)}")

            # Show a preview of the document
            preview = (
                doc.page_content[:100] + "..."
                if len(doc.page_content) > 100
                else doc.page_content
            )
            print(f"  Preview: {preview}\n")

    elif args.command == "test-connection":
        print(f"Testing connection to Ollama at {OLLAMA_BASE_URL}...")
        if ollama_available:
            print(f"✅ Successfully connected to Ollama at {OLLAMA_BASE_URL}")
            try:
                response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
                models = response.json().get("models", [])
                if models:
                    print("Available models:")
                    for model in models:
                        print(f"  - {model['name']}")
                else:
                    print("No models found. You may need to pull some models:")
                    print(f"  ollama pull {OLLAMA_LLM_MODEL}")
                    print(f"  ollama pull {OLLAMA_EMBED_MODEL}")
            except Exception as e:
                print(f"Error getting model list: {str(e)}")
        else:
            print(f"❌ Failed to connect to Ollama at {OLLAMA_BASE_URL}")
            print("\nPossible solutions:")
            print("1. Make sure Ollama is running on the target machine")
            print("2. Check if there's a firewall blocking the connection")
            print("3. Verify the IP address and port are correct")
            print(
                "4. Try using 'http://localhost:11434' if running Ollama on the same machine"
            )
            print(
                "\nTo use a different Ollama server, update the OLLAMA_BASE_URL in your .env file"
            )


if __name__ == "__main__":
    asyncio.run(main())
