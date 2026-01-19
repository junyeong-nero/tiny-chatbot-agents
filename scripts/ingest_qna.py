#!/usr/bin/env python3
"""Ingest crawled QnA data into ChromaDB vector store.

Usage:
    python scripts/ingest_qna.py                      # Ingest all JSON files in data/raw/qna/
    python scripts/ingest_qna.py --file path/to/file.json  # Ingest specific file
    python scripts/ingest_qna.py --clear              # Clear existing data before ingesting
    python scripts/ingest_qna.py --search "질문"       # Test search after ingestion
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vectorstore import QnAVectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Ingest crawled QnA data into ChromaDB"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Specific JSON file to ingest (default: all files in data/raw/qna/)",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="data/raw/qna",
        help="Directory containing QnA JSON files",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="data/vectordb/qna",
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

    args = parser.parse_args()

    print(f"Initializing QnA Vector Store...")
    print(f"  - Model: {args.model}")
    print(f"  - DB Directory: {args.db_dir}")

    store = QnAVectorStore(
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
        print(f"    Added {len(ids)} entries")

    print(f"\n✅ Ingestion complete!")
    print(f"   Total entries added: {total_added}")
    print(f"   Total documents in store: {store.count()}")

    # Test search if requested
    if args.search:
        print(f"\n🔍 Testing search: '{args.search}'")
        results = store.search(args.search, n_results=args.top_k)

        if results:
            print(f"   Found {len(results)} results:\n")
            for i, r in enumerate(results, 1):
                print(f"   [{i}] Score: {r.score:.4f}")
                print(f"       Q: {r.question[:80]}...")
                print(f"       A: {r.answer[:80]}...")
                print(f"       Category: {r.category} / {r.sub_category}")
                print()
        else:
            print("   No results found.")


if __name__ == "__main__":
    main()
