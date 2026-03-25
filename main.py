#!/usr/bin/env python3
"""RAG Pipeline - unified CLI entry point.

Usage:
    python main.py pipeline [-q QUERY] [--search-qna KW] [--search-tos KW] [...]
    python main.py crawl {qna,tos,all} [options]
    python main.py ingest-qna [options]
    python main.py ingest-tos [options]
    python main.py mcp
    python main.py evaluate [options]
    python main.py streamlit [--port PORT] [--host HOST]
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path when running as a script
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ──────────────────────────────────────────────────────────────
# pipeline
# ──────────────────────────────────────────────────────────────

def _print_response(response) -> None:
    print("\n" + "=" * 60)
    print(f"[Source: {response.source.value.upper()}]")
    print(f"[Confidence: {response.confidence:.2f}]")
    if response.verified:
        print(f"[Verified: YES (score: {response.verification_score:.2f})]")
    else:
        print(f"[Verified: NO (score: {response.verification_score:.2f})]")
    print("=" * 60)
    print(f"\n{response.answer}\n")
    if response.citations:
        print(f"참조: {', '.join(response.citations)}")
    if response.verification_issues:
        print("\n[검증 경고]")
        for issue in response.verification_issues:
            print(f"  - {issue}")
    if response.metadata.get("verification_reasoning"):
        print(f"\n[검증 상세]: {response.metadata['verification_reasoning']}")
    if response.metadata.get("tokens_used"):
        print(f"\n(토큰 사용: {response.metadata['tokens_used']})")


def _interactive_mode(pipeline) -> None:
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
            _print_response(response)
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}\n")


def _add_pipeline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-q", "--query", help="Single question to answer")
    p.add_argument("--search-qna", help="Search QnA DB")
    p.add_argument("--search-tos", help="Search ToS DB")
    p.add_argument("--qna-db", default="data/vectordb/qna", help="QnA DB path")
    p.add_argument("--tos-db", default="data/vectordb/tos", help="ToS DB path")
    p.add_argument("--model", "-m", help="Embedding model key")
    p.add_argument("--llm-provider", default=None,
                   help="LLM provider: vllm, sglang, ollama, openai")
    p.add_argument("--llm-model", default=None, help="LLM model name")
    p.add_argument("--verify", action="store_true", help="Enable hallucination verification")
    p.add_argument("--verify-threshold", type=float, default=0.7,
                   help="Verification confidence threshold (default: 0.7)")


def cmd_pipeline(args: argparse.Namespace) -> None:
    from src.llm import create_llm_client
    from src.pipeline import RAGPipeline

    print("Initializing RAG Pipeline...")
    try:
        llm_kwargs = {}
        if args.llm_model:
            llm_kwargs["model"] = args.llm_model
        llm = create_llm_client(provider=args.llm_provider, **llm_kwargs)
    except ValueError as e:
        print(f"Error: {e}")
        print("Set LLM_PROVIDER env var (vllm, sglang, ollama, or openai for testing)")
        sys.exit(1)

    pipeline = RAGPipeline(
        llm=llm,
        qna_db_path=args.qna_db,
        tos_db_path=args.tos_db,
        embedding_model=args.model,
        enable_verification=args.verify,
        verification_threshold=args.verify_threshold,
    )
    print(f"  QnA documents: {pipeline.qna_store.count()}")
    print(f"  ToS documents: {pipeline.tos_store.count()}")
    print(f"  LLM model: {llm.model}")
    print(f"  Verification: {'enabled' if args.verify else 'disabled'}")

    if args.query:
        _print_response(pipeline.query(args.query))
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
        _interactive_mode(pipeline)


# ──────────────────────────────────────────────────────────────
# crawl
# ──────────────────────────────────────────────────────────────

async def _crawl_qna(args: argparse.Namespace):
    from src.crawlers import QnACrawler
    categories = args.categories.split(",") if args.categories else None
    crawler = QnACrawler(output_dir=args.output, headless=not args.visible, categories=categories)
    return await crawler.run()


async def _crawl_tos(args: argparse.Namespace):
    from src.crawlers import ToSCrawler
    crawler = ToSCrawler(output_dir=args.output, headless=not args.visible)
    return await crawler.run()


async def _crawl_all(args: argparse.Namespace):
    qna_output = await _crawl_qna(args)
    print(f"QnA data saved to: {qna_output}")
    from src.crawlers import ToSCrawler
    tos_crawler = ToSCrawler(output_dir="data/raw/tos", headless=not args.visible)
    tos_output = await tos_crawler.run()
    print(f"ToS data saved to: {tos_output}")


def cmd_crawl(args: argparse.Namespace) -> None:
    if args.crawl_target == "qna":
        asyncio.run(_crawl_qna(args))
    elif args.crawl_target == "tos":
        asyncio.run(_crawl_tos(args))
    elif args.crawl_target == "all":
        asyncio.run(_crawl_all(args))


# ──────────────────────────────────────────────────────────────
# ingest-qna
# ──────────────────────────────────────────────────────────────

def cmd_ingest_qna(args: argparse.Namespace) -> None:
    from src.vectorstore import QnAVectorStore

    print("Initializing QnA Vector Store...")
    print(f"  - Model: {args.model}")
    print(f"  - DB Directory: {args.db_dir}")

    store = QnAVectorStore(persist_directory=args.db_dir, embedding_model=args.model)
    if args.clear:
        print("Clearing existing data...")
        store.clear()

    print(f"Current document count: {store.count()}")

    files = [Path(args.file)] if args.file else sorted(Path(args.data_dir).glob("*.json"))
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

    print(f"\nIngestion complete!")
    print(f"   Total entries added: {total_added}")
    print(f"   Total documents in store: {store.count()}")

    if args.search:
        print(f"\nTesting search: '{args.search}'")
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


# ──────────────────────────────────────────────────────────────
# ingest-tos
# ──────────────────────────────────────────────────────────────

def cmd_ingest_tos(args: argparse.Namespace) -> None:
    from src.vectorstore import ToSVectorStore

    print("Initializing ToS Vector Store...")
    print(f"  - Model: {args.model}")
    print(f"  - DB Directory: {args.db_dir}")

    store = ToSVectorStore(persist_directory=args.db_dir, embedding_model=args.model)
    if args.clear:
        print("Clearing existing data...")
        store.clear()

    print(f"Current document count: {store.count()}")

    files = [Path(args.file)] if args.file else sorted(Path(args.data_dir).glob("*.json"))
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

    if args.search:
        print(f"\nTesting search: '{args.search}'")
        results = store.search(args.search, n_results=args.top_k, category_filter=args.category)
        if results:
            print(f"   Found {len(results)} results:\n")
            for i, r in enumerate(results, 1):
                print(f"   [{i}] Score: {r.score:.4f}")
                print(f"       Document: {r.document_title[:50]}...")
                section = r.section_title[:50] if r.section_title else "(none)"
                print(f"       Section: {section}")
                print(f"       Content: {r.section_content[:80]}...")
                print(f"       Category: {r.category}")
                print()
        else:
            print("   No results found.")


# ──────────────────────────────────────────────────────────────
# mcp
# ──────────────────────────────────────────────────────────────

def cmd_mcp(_args: argparse.Namespace) -> None:
    from src.mcp.server import mcp
    print("Starting MCP Server for RAG Pipeline...")
    print("Available tools: ask_question, search_faq, search_terms, get_section, list_documents")
    mcp.run()


# ──────────────────────────────────────────────────────────────
# streamlit
# ──────────────────────────────────────────────────────────────

def cmd_streamlit(args: argparse.Namespace) -> None:
    app_path = project_root / "src" / "streamlit_app.py"
    cmd = ["streamlit", "run", str(app_path)]
    if args.port:
        cmd += ["--server.port", str(args.port)]
    if args.host:
        cmd += ["--server.address", args.host]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: streamlit not found. Install it with: uv pip install streamlit")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────
# evaluate
# ──────────────────────────────────────────────────────────────

def cmd_evaluate(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    from src.evaluation import EvaluationRunner, LLMEvaluator
    from src.evaluation.report import generate_markdown_report, generate_csv_report

    models = [m.strip() for m in args.models.split(",") if m.strip()] or ["default"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_base = Path(args.output).stem if args.output else f"eval_{timestamp}"

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    generator_model = generator_provider = None
    try:
        from src.evaluation.dataset_generator import EvaluationDataset
        dataset_obj = EvaluationDataset.load(dataset_path)
        generator_model = dataset_obj.generator_model
        generator_provider = dataset_obj.generator_provider
        if generator_model and generator_model != "unknown":
            logger.info(f"Dataset generated by: {generator_provider}/{generator_model}")
    except Exception as e:
        logger.warning(f"Could not load dataset metadata: {e}")

    llm_judge = None
    if args.use_llm_judge:
        try:
            from src.evaluation import create_llm_judge
            llm_judge = create_llm_judge(
                provider=args.judge_provider, model=args.judge_model,
                generator_model=generator_model, generator_provider=generator_provider,
                auto_diverse=args.auto_diverse_judge, strict_diversity=args.strict_diversity)
        except Exception as e:
            logger.error(f"Failed to create LLM judge: {e}")

    evaluator = LLMEvaluator(llm_judge=llm_judge, use_llm_judge=args.use_llm_judge and llm_judge is not None)

    all_results = []
    for model in models:
        logger.info(f"\n{'=' * 50}\nEvaluating model: {model}\n{'=' * 50}")

        pipeline = None
        if not args.no_pipeline:
            try:
                from src.pipeline import RAGPipeline
                from src.llm import create_llm_client
                if args.provider:
                    os.environ["LLM_PROVIDER"] = args.provider
                llm_client = create_llm_client(model=model if model != "default" else None)
                health_check = getattr(llm_client, "health_check", None)
                if callable(health_check) and not health_check():
                    logger.warning(f"LLM health check failed for {model}")
                    continue
                pipeline = RAGPipeline(llm=llm_client)
            except Exception as e:
                logger.error(f"Failed to create pipeline: {e}")
                continue

        runner = EvaluationRunner(pipeline=pipeline, evaluator=evaluator, dataset_path=dataset_path)
        runner.load_dataset()
        result = runner.run(limit=args.limit, model_name=model,
                            parallel=args.parallel, max_workers=args.max_workers)

        output_path = output_dir / f"{output_base}_{model.replace(':', '_')}.json"
        runner.save_results(result, output_path)
        all_results.append(result)

        print(f"\n--- {model} Summary ---")
        print(f"  Evaluated: {result.evaluated_cases}/{result.total_cases} cases")
        print(f"  Similarity: {result.mean_similarity:.3f} (±{result.std_similarity:.3f})")
        print(f"  BLEU: {result.mean_bleu:.3f} (±{result.std_bleu:.3f})")
        print(f"  Faithfulness: {result.mean_faithfulness:.3f}")
        print(f"  Latency: {result.mean_latency_ms:.1f}ms (±{result.std_latency_ms:.1f})")
        print(f"  Verification Rate: {result.verification_rate * 100:.1f}%")
        if result.mean_llm_judge_score > 0:
            print(f"\n  --- LLM-as-a-Judge ({result.llm_judge_model}) ---")
            print(f"  Overall Score: {result.mean_llm_judge_score:.2f}/5")
            print(f"  Correctness: {result.mean_llm_correctness:.2f}/5")
            print(f"  Helpfulness: {result.mean_llm_helpfulness:.2f}/5")
            print(f"  Faithfulness: {result.mean_llm_faithfulness:.2f}/5")
            print(f"  Fluency: {result.mean_llm_fluency:.2f}/5")

    if args.report and all_results:
        md_path = output_dir / f"{output_base}_report.md"
        csv_path = output_dir / f"{output_base}_report.csv"
        generate_markdown_report(all_results, md_path)
        generate_csv_report(all_results, csv_path)
        print(f"\nReports saved to:\n  Markdown: {md_path}\n  CSV: {csv_path}")

    print(f"\nEvaluation complete. Results saved to {output_dir}/")


# ──────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG Pipeline - unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- pipeline ---
    p = sub.add_parser("pipeline", help="Run RAG pipeline (interactive / single query)")
    _add_pipeline_args(p)

    # --- crawl ---
    p = sub.add_parser("crawl", help="Crawl QnA / ToS data from web")
    p.add_argument("crawl_target", choices=["qna", "tos", "all"], metavar="{qna,tos,all}")
    p.add_argument("--output", "-o", default="data/raw/qna", help="Output directory")
    p.add_argument("--categories", "-c", default=None, help="Comma-separated category codes (QnA only)")
    p.add_argument("--visible", action="store_true", help="Show browser window")

    # --- ingest-qna ---
    p = sub.add_parser("ingest-qna", help="Ingest QnA data into vector store")
    p.add_argument("--file", "-f", help="Specific JSON file to ingest")
    p.add_argument("--data-dir", "-d", default="data/raw/qna", help="Source directory")
    p.add_argument("--db-dir", default="data/vectordb/qna", help="ChromaDB directory")
    p.add_argument("--model", "-m", default=None, help="Embedding model key")
    p.add_argument("--clear", "-c", action="store_true", help="Clear existing data first")
    p.add_argument("--search", "-s", help="Test search after ingestion")
    p.add_argument("--top-k", "-k", type=int, default=5)

    # --- ingest-tos ---
    p = sub.add_parser("ingest-tos", help="Ingest ToS data into vector store")
    p.add_argument("--file", "-f", help="Specific JSON file to ingest")
    p.add_argument("--data-dir", "-d", default="data/raw/tos", help="Source directory")
    p.add_argument("--db-dir", default="data/vectordb/tos", help="ChromaDB directory")
    p.add_argument("--model", "-m", default=None, help="Embedding model key")
    p.add_argument("--clear", "-c", action="store_true", help="Clear existing data first")
    p.add_argument("--search", "-s", help="Test search after ingestion")
    p.add_argument("--top-k", "-k", type=int, default=5)
    p.add_argument("--category", default=None, help="Filter search results by category")

    # --- mcp ---
    sub.add_parser("mcp", help="Start MCP server")

    # --- streamlit ---
    p = sub.add_parser("streamlit", help="Start Streamlit web UI")
    p.add_argument("--port", type=int, default=None, help="Port to run on (default: 8501)")
    p.add_argument("--host", default=None, help="Host address to bind (default: localhost)")

    # --- evaluate ---
    p = sub.add_parser("evaluate", help="Run LLM evaluation pipeline")
    p.add_argument("--models", default="", help="Comma-separated model names")
    p.add_argument("--dataset", default="data/evaluation/evaluation_dataset.json")
    p.add_argument("--output", default="")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--provider", default="")
    p.add_argument("--no-pipeline", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--use-llm-judge", action="store_true")
    p.add_argument("--judge-model", default="gpt-4o")
    p.add_argument("--judge-provider", default="openai", choices=["openai", "anthropic", "google"])
    p.add_argument("--auto-diverse-judge", action="store_true")
    p.add_argument("--strict-diversity", action="store_true")
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--max-workers", type=int, default=4)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "pipeline": cmd_pipeline,
        "crawl": cmd_crawl,
        "ingest-qna": cmd_ingest_qna,
        "ingest-tos": cmd_ingest_tos,
        "mcp": cmd_mcp,
        "evaluate": cmd_evaluate,
        "streamlit": cmd_streamlit,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
