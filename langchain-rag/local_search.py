import os
import asyncio
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PDFMinerLoader,
)
import glob

# Load environment variables
load_dotenv()

# Get data directory from environment variables
DATA_DIR = os.getenv("DATA_DIR", "./data")

# Supported file types and their loaders
LOADERS = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PDFMinerLoader,
    # Add more file types and loaders as needed
}


async def search_local(query: str, num_results: int = 5) -> Tuple[str, List[Document]]:
    """
    Search local documents based on a query.
    This is a simplified search that just returns all documents.
    RAG will handle the actual semantic search.

    Args:
        query: The search query
        num_results: Maximum number of results to return

    Returns:
        Tuple of (formatted results string, list of documents)
    """
    try:
        # Load all documents from the data directory
        documents = await load_documents()

        if not documents:
            return "No documents found in the data directory.", []

        # Format the results
        formatted_results = format_local_results(documents[:num_results])

        return formatted_results, documents
    except Exception as e:
        return f"An error occurred while searching local documents: {e}", []


def format_local_results(documents: List[Document]) -> str:
    """Format the document results as a string."""
    if not documents:
        return "No documents found."

    markdown_results = "### Local Document Results:\n\n"
    for idx, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown source")
        # Get just the filename from the path
        filename = os.path.basename(source)

        markdown_results += f"**{idx}.** [{filename}]({source})\n"
        # Add a snippet of the content
        content_preview = (
            doc.page_content[:200] + "..."
            if len(doc.page_content) > 200
            else doc.page_content
        )
        markdown_results += f"> **Preview:** {content_preview}\n\n"

    return markdown_results


async def load_documents() -> List[Document]:
    """
    Load all documents from the data directory.

    Returns:
        List of Document objects
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}")
        # Create a sample document to demonstrate functionality
        with open(os.path.join(DATA_DIR, "sample.txt"), "w") as f:
            f.write(
                "This is a sample document to demonstrate the local RAG system. "
                "You can add more documents to the data directory to enhance the knowledge base."
            )

    documents = []

    # Process each supported file type
    for ext, loader_cls in LOADERS.items():
        # Find all files with this extension in the data directory
        files = glob.glob(os.path.join(DATA_DIR, f"**/*{ext}"), recursive=True)

        for file_path in files:
            try:
                loader = loader_cls(file_path)
                file_docs = loader.load()
                documents.extend(file_docs)
                print(f"Loaded {len(file_docs)} documents from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return documents


async def get_document_content(file_path: str) -> List[Document]:
    """
    Get the content of a specific document file.

    Args:
        file_path: Path to the document file

    Returns:
        List of Document objects
    """
    if not os.path.exists(file_path):
        return [
            Document(
                page_content=f"File not found: {file_path}",
                metadata={"source": file_path, "error": "File not found"},
            )
        ]

    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in LOADERS:
            loader = LOADERS[ext](file_path)
            documents = loader.load()
            return documents
        else:
            return [
                Document(
                    page_content=f"Unsupported file type: {ext}",
                    metadata={"source": file_path, "error": "Unsupported file type"},
                )
            ]
    except Exception as e:
        return [
            Document(
                page_content=f"Error loading file: {str(e)}",
                metadata={"source": file_path, "error": str(e)},
            )
        ]
