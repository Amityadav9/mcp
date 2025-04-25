# LangChain + MCP + RAG with Ollama

A powerful local Retrieval Augmented Generation (RAG) system built with LangChain, Multimodal Capabilities Protocol (MCP), and Ollama for embeddings and language model capabilities.

## 📋 Features

- **100% Local Execution**: All processing happens on your machine, no data sent to external APIs
- **Document Understanding**: Process TXT, Markdown, and PDF files
- **Semantic Search**: Find the most relevant content using embeddings
- **Customizable**: Easy to adjust embedding models, LLMs, and parameters
- **MCP Integration**: Expose RAG capabilities through standardized tools
- **Robust Error Handling**: Works even with partial connectivity to Ollama
- **CLI Interface**: Simple command-line tools for document management and querying

## 🛠️ System Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) (for embeddings and LLM capabilities)
- 4GB RAM minimum (8GB+ recommended for better performance)
- Storage space for documents and models

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Amityadav9/mcp.git
   cd langchain-mcp-rag
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Ollama from [ollama.ai](https://ollama.ai/) and pull the required models:
   ```bash
   ollama pull llama3:latest
   ollama pull mistral-small3.1:latest
   ```

4. Create a `.env` file with your configuration:
   ```
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_EMBED_MODEL=llama3:latest
   OLLAMA_LLM_MODEL=mistral-small3.1:latest
   DATA_DIR=./data
   ```

5. Add sample documents to get started:
   ```bash
   python agent.py add-samples
   ```

## 🚀 Usage

### Command Line Interface

**List all documents in your knowledge base**:
```bash
python agent.py list
```

**Query your documents without AI-generated answers**:
```bash
python agent.py query "What is RAG?"
```

**Query with AI-generated answers**:
```bash
python agent.py query "How does RAG work with LangChain?" --ai
```

**Test your Ollama connection**:
```bash
python agent.py test-connection
```

### MCP Server

Start the MCP server to expose RAG capabilities to other applications:

```bash
python mcp_server.py
```

The server runs on `http://localhost:8000` and provides the following tools:
- `search_local_tool`: Find relevant documents for a query
- `get_document_content_tool`: Get the full content of a specific document
- `ask_rag_question`: Generate AI responses using retrieved context
- `refresh_knowledge_base`: Reload documents when new ones are added
- `check_ollama_status`: Verify Ollama connection and available models

## 🧩 How It Works

1. **Document Processing**: Documents are loaded from your data directory
2. **Chunking**: Large documents are split into manageable chunks
3. **Embedding**: Ollama generates vector embeddings for each chunk
4. **Vector Storage**: Embeddings are stored in a FAISS vector database
5. **Semantic Search**: When you ask a question, the system finds relevant document chunks
6. **Answer Generation**: Ollama generates a coherent answer based on the retrieved context

## 📁 Project Structure

```
langchain-mcp-rag/
├── agent.py            # Command-line interface
├── local_search.py     # Document loading and management
├── rag.py              # Retrieval Augmented Generation core
├── mcp_server.py       # MCP server for tool integration
├── requirements.txt    # Project dependencies
├── .env                # Configuration (not in repo)
├── .env.example        # Configuration template
└── data/               # Document storage
```

## 🔍 Customization

### Adding More Document Types

Extend the `LOADERS` dictionary in `local_search.py` to support additional file formats.

### Changing Models

Update your `.env` file to use different Ollama models:

```
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3:8b
```

### Adjusting Chunking Parameters

Modify the `chunk_size` and `chunk_overlap` in `rag.py` to better suit your documents.

## ⚠️ Troubleshooting

### Connection Issues

If you can't connect to Ollama:
- Verify Ollama is running with `ollama list`
- Check your firewall settings
- Ensure the URL in `.env` is correct

### Out of Memory

If you encounter memory issues:
- Try a smaller model
- Reduce the chunk size in `rag.py`
- Process fewer documents at once

### Poor Search Results

If search results aren't relevant:
- Try a different embedding model
- Adjust the chunk size (smaller chunks can improve relevance)
- Increase the number of retrieved documents (`k` parameter in `search_rag`)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) for providing the RAG framework
- [Ollama](https://ollama.ai/) for local LLM capabilities
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [MCP](https://github.com/lakshmirp/mcp) for the Multimodal Capabilities Protocol implementation
