# mnemo

> Persistent memory for LLM applications. No infrastructure required.

`mnemo` gives any Python app a local, searchable memory layer backed by a single SQLite file.

## Problem

LLM apps forget useful context between turns, while full database stacks are often too heavy for teams that just need fast, local, reliable memory retrieval.

## Why we picked the name `mnemo`

`mnemo` is short, easy to type and remember, hints at memory without vendor-heavy vibes, and comes from the same root as "mnemonic" (memory cues, recall, context).

## Why mnemo?

- Context windows forget. Databases are overkill. `mnemo` is a single file.
- Hybrid BM25 + vector search in milliseconds on laptop hardware.
- Works offline. No API keys for retrieval. No vendor lock-in.
- Namespace isolation for multi-project or multi-tenant use.
- MCP-compatible: plug into Claude Desktop or Cursor in minutes.

## Installation

```bash
pip install mnemo
```

With Anthropic integration:

```bash
pip install "mnemo[anthropic]"
```

## Quickstart

```python
from mnemo import MemoryStore

mem = MemoryStore(path="./memory.db")
mem.add("The deployment uses Docker on port 8080")
mem.add("Auth is handled via JWT, tokens expire in 24h")

results = mem.search("how does authentication work?", top_k=3)
for r in results:
    print(r.text, r.score)
```

## Claude integration example

```python
from mnemo.integrations.claude import ClaudeMemory
import anthropic

mem = ClaudeMemory(path="./memory.db")
client = anthropic.Anthropic()

mem.add("User prefers responses in bullet points. Timezone is CET.")

messages = mem.build_messages(
    query="summarise our project status",
    user_message="what did we decide about the auth flow?",
)

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=messages,
)
```

`build_messages` injects only the top relevant memory chunks, not the full memory dump, which keeps prompt size lower.

## Token cost reduction

Naive memory approaches often append full chat history or entire docs into every prompt. `mnemo` retrieves only top-k relevant chunks per query.

Illustrative example (adjust with your own usage):
- Without `mnemo`: ~8,000 context tokens/request.
- With `mnemo`: ~400-600 retrieved tokens/request.
- At Claude Sonnet input pricing ($3/MTok), 10,000 requests/day can save roughly ~$225/day.

These numbers are directional; run your own math based on real traffic and context size.

## MCP setup

Install MCP support:

```bash
pip install "mnemo[mcp]"
```

`~/.cursor/mcp.json`:

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

`claude_desktop_config.json`:

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

## Namespace isolation

```python
from mnemo import MemoryStore

a = MemoryStore(path="./memory.db", namespace="project-alpha")
b = MemoryStore(path="./memory.db", namespace="project-beta")

a.add("Alpha secret: auth flow is JWT + refresh tokens")
print(b.search("auth flow", top_k=3))  # does not return alpha memory
```

## CLI reference

| Command | Description |
|---|---|
| `mnemo init` | Create/open local `memory.db` in current directory |
| `mnemo add "text"` | Add memory text (chunked + embedded) |
| `mnemo search "query"` | Search and print top results |
| `mnemo stats` | Show DB path, size, namespaces, chunk counts |
| `mnemo clear --namespace project-x` | Delete all memory in one namespace |

## Project layout

- `mnemo/core/memory.py`: high-level `MemoryStore` API.
- `mnemo/core/chunker.py`: fixed/sentence/semantic chunking.
- `mnemo/core/embedder.py`: lazy sentence-transformer wrapper.
- `mnemo/core/retriever.py`: BM25 + vector retrieval with RRF.
- `mnemo/core/store.py`: SQLite persistence and WAL mode setup.
- `mnemo/integrations/claude.py`: helper for memory-injected Claude messages.
- `mnemo/mcp/server.py`: local MCP server exposing memory tools.
- `mnemo/cli.py`: terminal commands for init/add/search/stats/clear.

## Benchmarks

Run `python benchmarks/run.py` - results vary by hardware and corpus size.

## Contributing

Issues and PRs are welcome. Open a ticket first for bigger API changes.

## License

MIT
