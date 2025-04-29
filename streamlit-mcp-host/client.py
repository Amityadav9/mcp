import asyncio
import sys
import json
import traceback
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.sse import sse_client


def print_items(name: str, result: any) -> None:
    """Print list of items from MCP result objects"""
    print(f"\nAvailable {name}:")
    items = getattr(result, name)
    if items:
        for item in items:
            print(" *", item)
    else:
        print("No items available")


async def main(server_url: str, tool_name: str, **kwargs):
    """
    Connect to the MCP server and call the specified tool.

    Args:
        server_url: Full URL to SSE endpoint (e.g. http://localhost:8000/sse)
        tool_name: Name of the tool to call
        **kwargs: Arguments to pass to the tool
    """
    if urlparse(server_url).scheme not in ("http", "https"):
        print("Error: Server URL must start with http:// or https://")
        sys.exit(1)

    try:
        async with sse_client(server_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                print("Connected to MCP server at", server_url)
                print_items("tools", await session.list_tools())
                print_items("resources", await session.list_resources())
                print_items("prompts", await session.list_prompts())

                # Check if the requested tool exists
                tools = await session.list_tools()
                if tool_name not in tools.tools:
                    print(
                        f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(tools.tools)}"
                    )
                    sys.exit(1)

                print(f"\nCalling {tool_name} tool with arguments: {kwargs}...")
                response = await session.call_tool(tool_name, arguments=kwargs)

                print("\n=== Tool Response ===\n")

                # Try to pretty print if the response is JSON
                if isinstance(response, dict):
                    print(json.dumps(response, indent=2))
                else:
                    print(response)
    except Exception as e:
        print(f"Error connecting to server: {e}")
        traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)


def print_usage():
    """Print usage instructions"""
    print(
        "Enhanced MCP Client\n"
        "=================\n"
        "Usage: uv run -- enhanced_client.py <server_url> <tool_name> [argument=value ...]\n\n"
        "Examples:\n"
        "  uv run -- enhanced_client.py http://localhost:8000/sse summarize_wikipedia_article url=https://en.wikipedia.org/wiki/India\n"
        '  uv run -- enhanced_client.py http://localhost:8000/sse generate_text prompt="Write a poem about AI" model=gemma3:4b\n'
        "  uv run -- enhanced_client.py http://localhost:8000/sse list_available_models\n"
        "  uv run -- enhanced_client.py http://localhost:8000/sse fetch_webpage_content url=https://example.com\n"
        "  uv run -- enhanced_client.py http://localhost:8000/sse compare_webpages url1=https://example.com url2=https://example.org\n"
        '  uv run -- enhanced_client.py http://localhost:8000/sse analyze_sentiment text="This is amazing!"\n'
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)

    server_url = sys.argv[1]
    tool_name = sys.argv[2]

    # Parse additional arguments in format key=value
    kwargs = {}
    for arg in sys.argv[3:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            kwargs[key] = value
        else:
            print(f"Warning: Ignoring argument '{arg}' (not in key=value format)")

    asyncio.run(main(server_url, tool_name, **kwargs))
