import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from html2text import html2text
import json

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route, Mount

from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS
from mcp.server.sse import SseServerTransport

# Import Ollama functions for AI operations
from ollama import chat, ChatResponse, list, embeddings

# Create an MCP server instance with an identifier
mcp = FastMCP("enhanced-ollama-tools")


@mcp.tool()
def summarize_wikipedia_article(url: str, model: str = "gemma3:4b") -> str:
    """
    Fetch a Wikipedia article at the provided URL and generate a summary using an Ollama model.

    Args:
        url: URL of the Wikipedia article to summarize
        model: Ollama model to use for summarization (default: gemma3:4b)

    Returns:
        A summary of the Wikipedia article

    Example:
        summarize_wikipedia_article("https://en.wikipedia.org/wiki/Python_(programming_language)")
    """
    try:
        # Validate input
        if not url.startswith("http"):
            raise ValueError("URL must start with http or https.")

        if "wikipedia.org" not in url:
            raise ValueError("URL must be a Wikipedia article.")

        # Fetch the article
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve the article. HTTP status code: {response.status_code}",
                )
            )

        # Parse the main content of the article
        soup = BeautifulSoup(response.text, "html.parser")

        # Get the article title
        title = soup.find("h1", {"id": "firstHeading"}).text

        # Get the main content
        content_div = soup.find("div", {"id": "mw-content-text"})
        if not content_div:
            raise McpError(
                ErrorData(
                    INVALID_PARAMS,
                    "Could not find the main content on the provided Wikipedia URL.",
                )
            )

        # Convert the content to Markdown
        markdown_text = html2text(str(content_div))

        # Create the summarization prompt for Ollama
        prompt = f"""Summarize the following Wikipedia article about {title}:

{markdown_text[:10000]}  # Limit the length to avoid context window issues

Please provide a concise summary that covers the key points and important information.
"""

        # Call the Ollama model to generate a summary
        response: ChatResponse = chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        summary = response.message.content.strip()
        return f"# Summary of {title}\n\n{summary}"

    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e


@mcp.tool()
def list_available_models() -> dict:
    """
    List all available Ollama models on the local system.

    Returns:
        Dictionary containing information about available models
    """
    try:
        models = list()
        return models
    except Exception as e:
        raise McpError(
            ErrorData(INTERNAL_ERROR, f"Error listing models: {str(e)}")
        ) from e


@mcp.tool()
def generate_text(prompt: str, model: str = "gemma3:4b") -> str:
    """
    Generate text based on a prompt using the specified Ollama model.

    Args:
        prompt: The text prompt to send to the model
        model: Ollama model to use (default: gemma3:4b)

    Returns:
        Generated text response

    Example:
        generate_text("Write a short poem about artificial intelligence")
    """
    try:
        response: ChatResponse = chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response.message.content.strip()
    except Exception as e:
        raise McpError(
            ErrorData(INTERNAL_ERROR, f"Error generating text: {str(e)}")
        ) from e


@mcp.tool()
def fetch_webpage_content(url: str) -> str:
    """
    Fetch a webpage and return its content in Markdown format.

    Args:
        url: URL of the webpage to fetch

    Returns:
        Markdown formatted content of the webpage

    Example:
        fetch_webpage_content("https://example.com")
    """
    try:
        # Validate input
        if not url.startswith("http"):
            raise ValueError("URL must start with http or https.")

        # Fetch the webpage
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve the webpage. HTTP status code: {response.status_code}",
                )
            )

        # Parse the content
        soup = BeautifulSoup(response.text, "html.parser")

        # Try to get the page title
        title = soup.find("title")
        title_text = title.text if title else "Webpage Content"

        # Get the body content
        body = soup.find("body")
        if not body:
            body = soup

        # Convert to markdown
        markdown_text = html2text(str(body))

        return f"# {title_text}\n\n{markdown_text}"
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e


@mcp.tool()
def compare_webpages(url1: str, url2: str, model: str = "gemma3:4b") -> str:
    """
    Compare two webpages and provide an analysis of their similarities and differences.

    Args:
        url1: URL of the first webpage
        url2: URL of the second webpage
        model: Ollama model to use (default: gemma3:4b)

    Returns:
        Analysis of the similarities and differences between the two webpages

    Example:
        compare_webpages("https://en.wikipedia.org/wiki/Python_(programming_language)",
                         "https://en.wikipedia.org/wiki/JavaScript")
    """
    try:
        # Fetch both webpages
        content1 = fetch_webpage_content(url1)
        content2 = fetch_webpage_content(url2)

        # Create a prompt for comparison
        prompt = f"""I have content from two different webpages. Please compare them and provide an analysis of their similarities and differences.

Webpage 1:
{content1[:5000]}  # Limit length to avoid context window issues

Webpage 2:
{content2[:5000]}  # Limit length to avoid context window issues

Please provide a detailed comparison focusing on:
1. Main topics covered
2. Key similarities
3. Notable differences
4. Overall tone and style
"""

        # Generate comparison using Ollama
        response: ChatResponse = chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response.message.content.strip()
    except Exception as e:
        raise McpError(
            ErrorData(INTERNAL_ERROR, f"Error comparing webpages: {str(e)}")
        ) from e


@mcp.tool()
def get_webpage_embedding(url: str) -> dict:
    """
    Fetch a webpage and return its vector embedding using Ollama.

    Args:
        url: URL of the webpage to embed

    Returns:
        Dictionary containing the embedding vector

    Example:
        get_webpage_embedding("https://example.com")
    """
    try:
        # Fetch the webpage content
        content = fetch_webpage_content(url)

        # Generate embedding
        embedding = embeddings(
            model="llama3", prompt=content[:2000]
        )  # Truncate to avoid context limits

        return {
            "url": url,
            "embedding": embedding,
            "dimensions": len(embedding["embedding"]),
        }
    except Exception as e:
        raise McpError(
            ErrorData(INTERNAL_ERROR, f"Error generating embedding: {str(e)}")
        ) from e


@mcp.tool()
def analyze_sentiment(text: str, model: str = "gemma3:4b") -> dict:
    """
    Analyze the sentiment of provided text using an Ollama model.

    Args:
        text: Text to analyze for sentiment
        model: Ollama model to use for analysis (default: gemma3:4b)

    Returns:
        Dictionary containing sentiment analysis results

    Example:
        analyze_sentiment("I love this product, it's amazing!")
    """
    try:
        prompt = f"""Analyze the sentiment of the following text. Provide a sentiment classification (positive, negative, or neutral), 
a confidence score between 0 and 1, and a brief explanation of your assessment.

Text: {text}

Format your response as JSON with the following structure:
{{
  "sentiment": "positive|negative|neutral",
  "confidence": 0.XX,
  "explanation": "Your explanation here"
}}
"""

        # Generate sentiment analysis using Ollama
        response: ChatResponse = chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        # Parse the JSON response
        result_text = response.message.content.strip()

        # Extract the JSON portion (in case the model adds extra text)
        import re

        json_match = re.search(r"({[\s\S]*})", result_text)
        if json_match:
            result_text = json_match.group(1)

        try:
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError:
            # Fallback if model doesn't provide valid JSON
            return {
                "sentiment": "unknown",
                "confidence": 0,
                "explanation": "Could not parse model response as JSON",
                "raw_response": result_text,
            }

    except Exception as e:
        raise McpError(
            ErrorData(INTERNAL_ERROR, f"Error analyzing sentiment: {str(e)}")
        ) from e


# Set up the SSE transport for MCP communication
sse = SseServerTransport("/messages/")


async def handle_sse(request: Request) -> None:
    _server = mcp._mcp_server
    async with sse.connect_sse(
        request.scope,
        request.receive,
        request._send,
    ) as (reader, writer):
        await _server.run(reader, writer, _server.create_initialization_options())


# Create the Starlette app with proper endpoints
app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)

if __name__ == "__main__":
    print("Starting Enhanced Ollama MCP Server...")
    print("Available tools:")
    # Get the tool names using the registered tools
    tool_names = [
        "summarize_wikipedia_article",
        "list_available_models",
        "generate_text",
        "fetch_webpage_content",
        "compare_webpages",
        "get_webpage_embedding",
        "analyze_sentiment",
    ]
    for tool_name in tool_names:
        print(f" - {tool_name}")
    print("\nServer running at http://localhost:8000/sse")
    uvicorn.run(app, host="localhost", port=8000)
