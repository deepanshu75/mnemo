from __future__ import annotations

import argparse

from mnemo import MemoryStore


def run_server(db_path: str, namespace: str) -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install mcp extra: pip install mnemo[mcp]") from exc

    store = MemoryStore(path=db_path, namespace=namespace)
    app = FastMCP("mnemo")

    @app.tool()
    def memory_add(text: str) -> str:
        added = store.add(text)
        return f"added {added} chunks"

    @app.tool()
    def memory_search(query: str, top_k: int = 5) -> list[dict[str, object]]:
        results = store.search(query, top_k=top_k)
        return [{"text": row.text, "score": row.score} for row in results]

    @app.tool()
    def memory_context(query: str, top_k: int = 5) -> str:
        return store.context(query=query, top_k=top_k)

    app.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mnemo MCP server")
    parser.add_argument("--db", default="./memory.db", help="Path to SQLite memory DB")
    parser.add_argument("--namespace", default="default", help="Memory namespace")
    args = parser.parse_args()
    run_server(db_path=args.db, namespace=args.namespace)


if __name__ == "__main__":
    main()
