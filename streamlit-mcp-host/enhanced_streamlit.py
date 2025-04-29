import streamlit as st
import asyncio
import json
import traceback
from mcp import ClientSession
from mcp.client.sse import sse_client


async def run_with_mcp_client(server_url, callback):
    """
    Run a callback function with an MCP client session.
    This handles the connection and cleanup properly.
    """
    try:
        # Connect to MCP in a way that works with the specific implementation
        streams = await sse_client(server_url)
        session = ClientSession(streams[0], streams[1])
        await session.initialize()

        # Run the callback
        result = await callback(session)

        # Clean up
        await session.close()
        return result
    except Exception as e:
        error_message = f"Error with MCP client: {e}\n{traceback.format_exc()}"
        return error_message


async def list_tools(session):
    """Get the list of available tools"""
    tools_response = await session.list_tools()
    return tools_response.tools


def render_wikipedia_tool():
    """Render the Wikipedia summarization tool interface"""
    st.header("Wikipedia Article Summarization")

    article_url = st.text_input(
        "Wikipedia Article URL", "https://en.wikipedia.org/wiki/Artificial_intelligence"
    )

    # Default models in case we can't get them from Ollama
    model_names = ["gemma3:4b", "llama3:8b"]

    selected_model = st.selectbox("Select Ollama Model", model_names)

    if st.button("Summarize Article"):
        st.info("Fetching and summarizing article...")

        async def call_tool(session):
            return await session.call_tool(
                "summarize_wikipedia_article",
                arguments={"url": article_url, "model": selected_model},
            )

        # Run the async function
        result = asyncio.run(
            run_with_mcp_client(st.session_state["server_url"], call_tool)
        )
        st.markdown(result)


def render_text_generation_tool():
    """Render the text generation tool interface"""
    st.header("Text Generation")

    prompt = st.text_area(
        "Enter your prompt", "Write a short poem about technology and nature."
    )

    # Default models
    model_names = ["gemma3:4b", "llama3:8b"]

    selected_model = st.selectbox("Select Ollama Model", model_names, key="gen_model")

    if st.button("Generate Text"):
        st.info("Generating text...")

        async def call_tool(session):
            return await session.call_tool(
                "generate_text", arguments={"prompt": prompt, "model": selected_model}
            )

        # Run the async function
        result = asyncio.run(
            run_with_mcp_client(st.session_state["server_url"], call_tool)
        )
        st.markdown(result)


def render_webpage_content_tool():
    """Render the webpage content fetching tool interface"""
    st.header("Fetch Webpage Content")

    url = st.text_input("Webpage URL", "https://example.com")

    if st.button("Fetch Content"):
        st.info("Fetching webpage content...")

        async def call_tool(session):
            return await session.call_tool(
                "fetch_webpage_content", arguments={"url": url}
            )

        # Run the async function
        result = asyncio.run(
            run_with_mcp_client(st.session_state["server_url"], call_tool)
        )
        st.markdown(result)


def render_compare_webpages_tool():
    """Render the webpage comparison tool interface"""
    st.header("Compare Webpages")

    col1, col2 = st.columns(2)

    with col1:
        url1 = st.text_input(
            "First Webpage URL", "https://en.wikipedia.org/wiki/Artificial_intelligence"
        )

    with col2:
        url2 = st.text_input(
            "Second Webpage URL", "https://en.wikipedia.org/wiki/Machine_learning"
        )

    # Default models
    model_names = ["gemma3:4b", "llama3:8b"]

    selected_model = st.selectbox(
        "Select Ollama Model", model_names, key="compare_model"
    )

    if st.button("Compare Webpages"):
        st.info("Comparing webpages...")

        async def call_tool(session):
            return await session.call_tool(
                "compare_webpages",
                arguments={"url1": url1, "url2": url2, "model": selected_model},
            )

        # Run the async function
        result = asyncio.run(
            run_with_mcp_client(st.session_state["server_url"], call_tool)
        )
        st.markdown(result)


def render_sentiment_analysis_tool():
    """Render the sentiment analysis tool interface"""
    st.header("Sentiment Analysis")

    text = st.text_area(
        "Enter text to analyze",
        "I really enjoyed this product. It exceeded my expectations in every way!",
    )

    # Default models
    model_names = ["gemma3:4b", "llama3:8b"]

    selected_model = st.selectbox(
        "Select Ollama Model", model_names, key="sentiment_model"
    )

    if st.button("Analyze Sentiment"):
        st.info("Analyzing sentiment...")

        async def call_tool(session):
            return await session.call_tool(
                "analyze_sentiment", arguments={"text": text, "model": selected_model}
            )

        # Run the async function
        result = asyncio.run(
            run_with_mcp_client(st.session_state["server_url"], call_tool)
        )

        # Display the result in a formatted way
        if isinstance(result, dict):
            sentiment = result.get("sentiment", "unknown")
            confidence = result.get("confidence", 0)
            explanation = result.get("explanation", "No explanation provided")

            # Display with emoji based on sentiment
            emoji = (
                "😊"
                if sentiment == "positive"
                else "😐"
                if sentiment == "neutral"
                else "😟"
            )

            st.subheader(f"Sentiment: {sentiment.title()} {emoji}")
            st.progress(float(confidence))
            st.write(f"**Confidence:** {confidence:.2f}")
            st.write(f"**Explanation:** {explanation}")
        else:
            st.write(result)


async def test_connection(server_url):
    """Test the connection to the MCP server"""
    try:
        # Connect to the server
        streams = await sse_client(server_url)
        session = ClientSession(streams[0], streams[1])
        await session.initialize()

        # Get the available tools
        tools_response = await session.list_tools()
        tools = tools_response.tools

        # Clean up
        await session.close()

        return True, tools
    except Exception as e:
        return False, str(e)


def main():
    st.set_page_config(
        page_title="Enhanced Ollama MCP Tools",
        page_icon="🤖",
        layout="wide",
    )

    st.title("Enhanced Ollama MCP Tools")
    st.write(
        "An advanced interface for interacting with Ollama models through MCP server tools."
    )

    # Server connection settings
    with st.sidebar:
        st.header("Server Settings")
        server_url = st.text_input("MCP Server URL", "http://localhost:8000/sse")

        if st.button("Connect to Server"):
            # Test connection and get available tools
            connected, result = asyncio.run(test_connection(server_url))

            if connected:
                st.session_state["server_connected"] = True
                st.session_state["server_url"] = server_url
                st.session_state["available_tools"] = result
                st.success(f"Connected successfully! Found {len(result)} tools.")
                st.rerun()
            else:
                st.error(f"Failed to connect: {result}")
                st.session_state["server_connected"] = False

    # Initialize session state
    if "server_connected" not in st.session_state:
        st.session_state["server_connected"] = False

    if "current_tool" not in st.session_state:
        st.session_state["current_tool"] = "Wikipedia"

    # Display available tools if connected
    if st.session_state.get("server_connected", False):
        # Show tools in sidebar
        with st.sidebar:
            st.header("Available Tools")
            available_tools = st.session_state.get("available_tools", [])
            st.write(f"Found {len(available_tools)} tools on the server")

            # Tool selector
            tool_options = [
                "Wikipedia",
                "Text Generation",
                "Webpage Content",
                "Compare Webpages",
                "Sentiment Analysis",
            ]
            selected_tool = st.radio("Select Tool", tool_options)
            st.session_state["current_tool"] = selected_tool

        # Render the selected tool interface
        if st.session_state["current_tool"] == "Wikipedia":
            render_wikipedia_tool()
        elif st.session_state["current_tool"] == "Text Generation":
            render_text_generation_tool()
        elif st.session_state["current_tool"] == "Webpage Content":
            render_webpage_content_tool()
        elif st.session_state["current_tool"] == "Compare Webpages":
            render_compare_webpages_tool()
        elif st.session_state["current_tool"] == "Sentiment Analysis":
            render_sentiment_analysis_tool()
    else:
        # Display connection instructions
        st.info("Please connect to an MCP server using the sidebar settings.")

        with st.expander("What is this app?"):
            st.write("""
            This app provides a user-friendly interface for interacting with Ollama models through MCP server tools.
            
            Available features:
            - Wikipedia article summarization
            - Text generation with various models
            - Webpage content fetching and analysis
            - Webpage comparison
            - Sentiment analysis
            
            To get started:
            1. Make sure your MCP server is running (run ollama_server.py)
            2. Enter the server URL in the sidebar (default: http://localhost:8000/sse)
            3. Click "Connect to Server"
            4. Select a tool from the sidebar and start using it
            """)


if __name__ == "__main__":
    main()
