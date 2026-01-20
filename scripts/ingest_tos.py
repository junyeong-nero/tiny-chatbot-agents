#!/usr/bin/env python3
"""Ingest crawled ToS (Terms of Service) data into ChromaDB vector store.

Usage:
    python scripts/ingest_tos.py                      # Ingest all JSON files in data/raw/tos/
    python scripts/ingest_tos.py --file path/to/file.json  # Ingest specific file
    python scripts/ingest_tos.py --clear              # Clear existing data before ingesting
    python scripts/ingest_tos.py --search "약관"       # Test search after ingestion
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vectorstore import ToSVectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Ingest crawled ToS data into ChromaDB"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Specific JSON file to ingest (default: all files in data/raw/tos/)",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="data/raw/tos",
        help="Directory containing ToS JSON files",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="data/vectordb/tos",
        help="Directory to store ChromaDB data",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Embedding model key from embedding_config.yaml (default: uses config default)",
    )
    parser.add_argument(
        "--clear",
        "-c",
        action="store_true",
        help="Clear existing data before ingesting",
    )
    parser.add_argument(
        "--search",
        "-s",
        type=str,
        help="Test search query after ingestion",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of results for test search",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Filter search results by category",
    )

    args = parser.parse_args()

    print(f"Initializing ToS Vector Store...")
    print(f"  - Model: {args.model}")
    print(f"  - DB Directory: {args.db_dir}")

    store = ToSVectorStore(
        persist_directory=args.db_dir,
        embedding_model=args.model,
    )

    if args.clear:
        print("Clearing existing data...")
        store.clear()

    print(f"Current document count: {store.count()}")

    # Determine files to ingest
    if args.file:
        files = [Path(args.file)]
    else:
        data_dir = Path(args.data_dir)
        files = sorted(data_dir.glob("*.json"))

    if not files:
        print(f"No JSON files found in {args.data_dir}")
        return

    print(f"\nIngesting {len(files)} file(s)...")

    total_added = 0
    for file_path in files:
        print(f"\n  Processing: {file_path.name}")
        ids = store.load_from_json(file_path)
        total_added += len(ids)
        print(f"    Added {len(ids)} chunks")

    print(f"\nIngestion complete!")
    print(f"   Total chunks added: {total_added}")
    print(f"   Total documents in store: {store.count()}")

    # Test search if requested
    if args.search:
        print(f"\nTesting search: '{args.search}'")
        results = store.search(
            args.search,
            n_results=args.top_k,
            category_filter=args.category,
        )

        if results:
            print(f"   Found {len(results)} results:\n")
            for i, r in enumerate(results, 1):
                print(f"   [{i}] Score: {r.score:.4f}")
                print(f"       Document: {r.document_title[:50]}...")
                print(f"       Section: {r.section_title[:50]}..." if r.section_title else "       Section: (none)")
                print(f"       Content: {r.section_content[:80]}...")
                print(f"       Category: {r.category}")
                print()
        else:
            print("   No results found.")


if __name__ == "__main__":
    main()
