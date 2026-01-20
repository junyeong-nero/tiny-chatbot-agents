#!/usr/bin/env python3
"""Interactive CLI for RAG Pipeline.

Usage:
    python scripts/run_pipeline.py                  # Interactive mode
    python scripts/run_pipeline.py -q "질문"        # Single question
    python scripts/run_pipeline.py --search-qna "키워드"   # Search QnA
    python scripts/run_pipeline.py --search-tos "키워드"   # Search ToS
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import RAGPipeline


def print_response(response):
    """Pretty print pipeline response."""
    print("\n" + "=" * 60)
    print(f"[Source: {response.source.value.upper()}]")
    print(f"[Confidence: {response.confidence:.2f}]")
    print("=" * 60)
    print(f"\n{response.answer}\n")

    if response.citations:
        print(f"참조: {', '.join(response.citations)}")

    if response.metadata.get("tokens_used"):
        print(f"\n(토큰 사용: {response.metadata['tokens_used']})")


def interactive_mode(pipeline: RAGPipeline):
    """Run interactive Q&A session."""
    print("\n" + "=" * 60)
    print("  RAG Pipeline - Interactive Mode")
    print("  Commands: /quit, /search-qna, /search-tos")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("질문> ").strip()

            if not query:
                continue

            if query.lower() in ["/quit", "/q", "/exit"]:
                print("종료합니다.")
                break

            if query.startswith("/search-qna "):
                keyword = query[12:].strip()
                results = pipeline.search_qna(keyword, n_results=5)
                print(f"\n=== QnA 검색 결과 ({len(results)}개) ===")
                for i, r in enumerate(results, 1):
                    print(f"\n[{i}] Score: {r['score']:.3f}")
                    print(f"Q: {r['question']}")
                    print(f"A: {r['answer'][:200]}...")
                print()
                continue

            if query.startswith("/search-tos "):
                keyword = query[12:].strip()
                results = pipeline.search_tos(keyword, n_results=5)
                print(f"\n=== ToS 검색 결과 ({len(results)}개) ===")
                for i, r in enumerate(results, 1):
                    print(f"\n[{i}] Score: {r['score']:.3f}")
                    print(f"Document: {r['document_title']}")
                    print(f"Section: {r['section_title']}")
                    print(f"Content: {r['section_content'][:200]}...")
                print()
                continue

            response = pipeline.query(query)
            print_response(response)

        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument(
        "-q", "--query", type=str, help="Single question to answer"
    )
    parser.add_argument(
        "--search-qna", type=str, help="Search QnA DB"
    )
    parser.add_argument(
        "--search-tos", type=str, help="Search ToS DB"
    )
    parser.add_argument(
        "--qna-db", type=str, default="data/vectordb/qna", help="QnA DB path"
    )
    parser.add_argument(
        "--tos-db", type=str, default="data/vectordb/tos", help="ToS DB path"
    )
    parser.add_argument(
        "--model", "-m", type=str, help="Embedding model key"
    )
    parser.add_argument(
        "--llm-model", type=str, default="gpt-4o-mini", help="OpenAI model"
    )

    args = parser.parse_args()

    print("Initializing RAG Pipeline...")

    # Import here to avoid loading if just showing help
    from src.llm import OpenAIClient

    try:
        llm = OpenAIClient(model=args.llm_model)
    except ValueError as e:
        print(f"Error: {e}")
        print("Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    pipeline = RAGPipeline(
        llm=llm,
        qna_db_path=args.qna_db,
        tos_db_path=args.tos_db,
        embedding_model=args.model,
    )

    print(f"  QnA documents: {pipeline.qna_store.count()}")
    print(f"  ToS documents: {pipeline.tos_store.count()}")
    print(f"  LLM model: {args.llm_model}")

    if args.query:
        response = pipeline.query(args.query)
        print_response(response)
    elif args.search_qna:
        results = pipeline.search_qna(args.search_qna, n_results=5)
        print(f"\n=== QnA 검색 결과 ({len(results)}개) ===")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score: {r['score']:.3f}")
            print(f"Q: {r['question']}")
            print(f"A: {r['answer']}")
    elif args.search_tos:
        results = pipeline.search_tos(args.search_tos, n_results=5)
        print(f"\n=== ToS 검색 결과 ({len(results)}개) ===")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score: {r['score']:.3f}")
            print(f"Document: {r['document_title']}")
            print(f"Section: {r['section_title']}")
            print(f"Content: {r['section_content'][:300]}...")
    else:
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()
