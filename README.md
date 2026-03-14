# Multi-Agent RAG System for Course Q&A

A robust Retrieval-Augmented Generation (RAG) system built with a multi-agent architecture designed to answer student questions about course materials. The system grounds its answers in local course documents and supplements them with web searches when necessary.

## Architecture & Design
This project implements patterns from recent AI research:

1. **Agentic RAG**:
   - **Orchestrator Agent**: Plans, delegates tasks to sub-agents, and synthesizes the final answer.
   - **Retriever Agent**: Built on ChromaDB, it searches through your local course documents.
   - **Web Search Agent**: Uses DuckDuckGo to search the web for supplementary or current information when local docs fall short.

2. **Orchestral AI**:
   - Provider-agnostic design utilizing local/cloud Ollama models.
   - Tool-based agent architecture ensuring clear separation of concerns.
   - Synchronous execution for deterministic behavior and reliable subagent delegation.

## Features
- **Local Vector Search**: Uses ChromaDB and `all-MiniLM-L6-v2` embeddings to search through local PDFs and text files.
- **Web Fallback**: Automatically searches Wikipedia/DuckDuckGo when the local documents lack sufficient context.
- **Multi-Agent Orchestration**: Intelligent routing and synthesis of information by a primary Orchestrator Agent.
- **Transparent Reasoning**: Verbose CLI output showing the Orchestrator's thought process, tool calls, and retrieved sources.

## Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running locally with the required model.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/manideep8008/Multi-Agent-RAG-System.git
   cd Multi-Agent-RAG-System
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   pip install -r requirement.txt
   ```

3. **Ensure Ollama is running:**
   Start the Ollama server in a separate terminal:
   ```bash
   ollama serve
   ```
   Pull the necessary model (default in script is `gpt-oss:120b-cloud` or `llama3.2` depending on your setup):
   ```bash
   ollama pull llama3.2
   ```

## Usage

### 1. Ingest Course Documents
First, place your course materials (PDFs or .txt files) into the `docs/` folder.
Then, run the ingestion script to chunk, embed, and store them in the local ChromaDB.

```bash
python ingest.py
```
*(If the `docs/` folder is empty, the script will automatically generate 3 sample text documents for demonstration purposes.)*

### 2. Run the Q&A System
Once the documents are ingested, start the Orchestrator agent.

```bash
python main.py
```

### CLI Commands
During the interactive Q&A session, you can use the following commands:
- Type your question and press **Enter**
- `reset` - Clears the current conversation history/context
- `quit`  - Exits the application

## Repository Structure

- `main.py` - The core application featuring the Multi-Agent orchestrator pipeline.
- `ingest.py` - Script to process and embed PDFs/TXT files into ChromaDB.
- `docs/` - Directory containing your local course material.
- `chroma_db/` - Directory containing the local vector database instance. (gitignored)
- `tools/` - Directory containing the individual agent tools.
  - `retriever.py` - Connects to ChromaDB for local document search.
  - `web_search.py` - Connects to DuckDuckGo/Wikipedia for live web searches.
- `requirement.txt` - Python package dependencies.
