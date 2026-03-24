from __future__ import annotations

import argparse
import json
from pathlib import Path

from mnemo import MemoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mnemo", description="Local memory for LLM apps")
    parser.add_argument("--db", default="./memory.db")
    parser.add_argument("--namespace", default="default")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init")

    add_parser = sub.add_parser("add")
    add_parser.add_argument("text")
    add_parser.add_argument("--source", default=None)

    search_parser = sub.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)

    sub.add_parser("stats")

    clear_parser = sub.add_parser("clear")
    clear_parser.add_argument("--namespace", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    store = MemoryStore(path=args.db, namespace=args.namespace)

    if args.command == "init":
        print(f"initialized {Path(args.db).resolve()}")
        return
    if args.command == "add":
        added = store.add(args.text, source=args.source)
        print(f"added {added} chunks")
        return
    if args.command == "search":
        rows = store.search(args.query, top_k=args.top_k)
        for idx, row in enumerate(rows, start=1):
            print(f"{idx}. score={row.score:.4f} :: {row.text}")
        return
    if args.command == "stats":
        print(json.dumps(store.stats(), indent=2))
        return
    if args.command == "clear":
        removed = store.clear(namespace=args.namespace)
        print(f"removed {removed} chunks")
        return


if __name__ == "__main__":
    main()
