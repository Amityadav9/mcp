# Enhanced Ollama MCP Tools

This project provides a set of tools for interacting with Ollama models through the Model Control Protocol (MCP). It includes a server that exposes various AI tools, a command-line client, and a Streamlit web interface.

## Features

- **Wikipedia Article Summarization**: Fetch and summarize Wikipedia articles using Ollama models
- **Text Generation**: Generate text based on prompts using any available Ollama model
- **Webpage Content Fetching**: Fetch and convert any webpage to Markdown
- **Webpage Comparison**: Compare the content of two webpages and analyze differences
- **Sentiment Analysis**: Analyze the sentiment of text with confidence scores and explanations
- **Available Models**: List all Ollama models available on your system
- **Embeddings**: Generate vector embeddings for text content

## Requirements

- Python 3.8 or higher
- Ollama installed locally with at least one model (recommended: gemma3:4b or llama3)
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mcp.git
cd mcp
```

2. Install the dependencies using uv:
```bash
uv pip install -r requirements.txt
```

## Usage

### Step 1: Start the MCP Server

Run the enhanced_ollama_server.py script:

```bash
python enhanced_ollama_server.py
```

The server will start on http://localhost:8000/sse and list all available tools.

### Step 2: Use the Streamlit Interface (Recommended)

Run the Streamlit app:

```bash
streamlit run enhanced_streamlit.py
```

This will open a browser window with the web interface. Connect to the MCP server and explore all available tools.

### Step 3: (Alternative) Use the Command Line Client

You can also use the command line client to interact with the server:

```bash
# Summarize a Wikipedia article
python client.py http://localhost:8000/sse summarize_wikipedia_article url=https://en.wikipedia.org/wiki/Artificial_intelligence

# Generate text with a prompt
python client.py http://localhost:8000/sse generate_text prompt="Write a short story about robots" model=gemma3:4b

# List available models
python client.py http://localhost:8000/sse list_available_models

# Analyze sentiment of text
python client.py http://localhost:8000/sse analyze_sentiment text="I love this product!"
```

## Workflow

1. Make sure Ollama is running on your system and you have at least one model downloaded
2. Start the MCP server with `python enhanced_ollama_server.py`
3. Start the Streamlit interface with `streamlit run enhanced_streamlit.py`
4. Connect to the server from the Streamlit interface
5. Select a tool from the sidebar and start using it

## Troubleshooting

- **Error connecting to MCP server**: Make sure the server is running and the URL is correct
- **Model not found**: Ensure Ollama is running and the model is downloaded (`ollama pull modelname`)
- **Rate limiting errors**: Some models may have rate limiting, try using a smaller model or waiting between requests

## Extending the Project

You can add more tools to the MCP server by:

1. Adding new functions decorated with `@mcp.tool()` in enhanced_ollama_server.py
2. Adding corresponding interfaces in enhanced_streamlit_app.py

## License

This project is licensed under the MIT License - see the LICENSE file for details.
