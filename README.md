# mnemo: AI Memory API and RAG Memory System for Python LLM Apps

`mnemo` is a lightweight LLM memory backend for developers building chatbots, copilots, and RAG workflows that need persistent recall.
It provides a local-first hybrid search layer (BM25 + vector retrieval) over SQLite, so you can ship memory without running a separate vector database service.
Use it directly in Python, through CLI tools, or via MCP for Cursor and Claude Desktop.

## Problem

LLM applications lose context across sessions, while full data infrastructure can be too heavy for early-stage products and internal tools.
Teams need a fast memory layer that is easy to integrate, works offline, and supports semantic retrieval for better responses.

## Features

- Persistent memory store backed by SQLite (`memory.db`)
- Hybrid retrieval: BM25 keyword search + embeddings-based vector search
- RRF fusion for better ranking quality across lexical and semantic signals
- Namespace isolation for multi-tenant or multi-project memory separation
- Python API for app integration and CLI for quick operations
- MCP server support for Cursor and Claude Desktop tool-based memory access
- Local-first design with no required retrieval API keys

## Use Cases

- Long-term memory for AI assistants and chatbots
- RAG memory system for product docs, support notes, and user preferences
- LLM memory backend for agent frameworks that need per-user recall
- FastAPI AI backends that need a simple memory service without external infra
- Internal copilots that must run offline in secure environments

## Tech Stack

- Python 3.9+
- SQLite (persistent local storage)
- `sentence-transformers` (embedding generation)
- `rank_bm25` (lexical retrieval)
- Hybrid retrieval with Reciprocal Rank Fusion (RRF)
- Optional Anthropic integration for Claude message construction
- Optional MCP integration for tool-driven memory access
- FastAPI-compatible architecture (easy to wrap as REST endpoints)

## Installation

```bash
pip install mnemo
```

Optional extras:

```bash
pip install "mnemo[anthropic]"
pip install "mnemo[mcp]"
```

## Quick Start

### 1) Initialize and write memory

```python
from mnemo import MemoryStore

mem = MemoryStore(path="./memory.db", namespace="default")
mem.add("Customer prefers weekly status emails")
mem.add("Auth stack uses JWT with 24h expiry")
```

### 2) Retrieve relevant memory

```python
results = mem.search("how does authentication work?", top_k=3)
for r in results:
    print(r.text, r.score)
```

### 3) Use the CLI

```bash
mnemo init
mnemo add "The deployment region is us-east-1"
mnemo search "where is the app deployed?"
mnemo stats
```

## API Example (FastAPI Wrapper)

`mnemo` is a Python library, but many teams expose it through FastAPI as an internal AI memory API.

### Request

```http
POST /memory/search
Content-Type: application/json

{
  "query": "what plan did the user choose?",
  "top_k": 2,
  "namespace": "customer-123"
}
```

### Response

```json
{
  "results": [
    {
      "text": "User selected the Pro annual plan",
      "score": 0.912,
      "namespace": "customer-123"
    },
    {
      "text": "Billing cycle starts on the 1st of each month",
      "score": 0.844,
      "namespace": "customer-123"
    }
  ]
}
```

### Minimal FastAPI Integration

```python
from fastapi import FastAPI
from pydantic import BaseModel
from mnemo import MemoryStore

app = FastAPI(title="mnemo-memory-api")
mem = MemoryStore(path="./memory.db")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    namespace: str = "default"

@app.post("/memory/search")
def search_memory(req: SearchRequest):
    scoped = MemoryStore(path="./memory.db", namespace=req.namespace)
    hits = scoped.search(req.query, top_k=req.top_k)
    return {
        "results": [
            {
                "text": h.text,
                "score": h.score,
                "namespace": req.namespace,
            }
            for h in hits
        ]
    }
```

## Claude Integration

```python
import anthropic
from mnemo.integrations.claude import ClaudeMemory

mem = ClaudeMemory(path="./memory.db")
client = anthropic.Anthropic()

mem.add("User prefers concise bullet-point answers")

messages = mem.build_messages(
    query="summarize product status",
    user_message="what did we decide for auth?",
)

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=messages,
)
```

`build_messages` injects only top relevant chunks, helping control prompt size and token cost.

## MCP Setup

```json
{
  "mcpServers": {
    "mnemo": {
      "command": "python",
      "args": ["-m", "mnemo.mcp.server", "--db", "./memory.db", "--namespace", "default"]
    }
  }
}
```

## CLI Reference

| Command | Description |
|---|---|
| `mnemo init` | Create or open local `memory.db` |
| `mnemo add "text"` | Add memory text (chunk + embed) |
| `mnemo search "query"` | Search top memory results |
| `mnemo stats` | Show DB stats and namespaces |
| `mnemo clear --namespace project-x` | Delete one namespace |

## Project Layout

- `mnemo/core/memory.py`: high-level `MemoryStore` API
- `mnemo/core/chunker.py`: fixed/sentence/semantic chunking
- `mnemo/core/embedder.py`: lazy embedding wrapper
- `mnemo/core/retriever.py`: hybrid retrieval + ranking
- `mnemo/core/store.py`: SQLite persistence and setup
- `mnemo/integrations/claude.py`: Claude memory helper
- `mnemo/mcp/server.py`: MCP server for memory tools
- `mnemo/cli.py`: command-line interface

## Contributing

Issues and pull requests are welcome. Open an issue first for significant API changes.

## License

MIT
