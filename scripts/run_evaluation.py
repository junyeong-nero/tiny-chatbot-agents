#!/usr/bin/env python3
"""CLI script for running LLM evaluation.

Usage:
    python scripts/run_evaluation.py --models "llama3.1:8b" --limit 10
    python scripts/run_evaluation.py --dataset data/evaluation/evaluation_dataset.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import EvaluationRunner, LLMEvaluator
from src.evaluation.report import generate_markdown_report, generate_csv_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Evaluation Runner")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated list of model names (e.g., 'llama3.1:8b,mistral:7b')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/evaluation/evaluation_dataset.json",
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output path for results JSON (default: results/eval_<timestamp>.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test cases to evaluate",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="",
        help="LLM provider (vllm, sglang, ollama, openai)",
    )
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Run evaluation without pipeline (compare expected answers only)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate markdown and CSV reports",
    )
    return parser.parse_args()


def create_pipeline(model: str, provider: str):
    """Create RAG pipeline with specified model."""
    try:
        from src.pipeline import RAGPipeline
        from src.llm import create_llm_client, LLMProvider

        # Set environment variables
        if provider:
            os.environ["LLM_PROVIDER"] = provider

        llm_client = create_llm_client(model=model if model else None)

        # Check if LLM is available
        if not llm_client.health_check():
            logger.warning(f"LLM health check failed for {model}")
            return None

        pipeline = RAGPipeline(llm_client=llm_client)
        return pipeline

    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        return None


def main():
    args = parse_args()

    # Parse models
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        models = ["default"]

    # Setup output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    if args.output:
        output_base = Path(args.output).stem
    else:
        output_base = f"eval_{timestamp}"

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    # Create evaluator
    evaluator = LLMEvaluator()

    # Run evaluation for each model
    all_results = []
    for model in models:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating model: {model}")
        logger.info(f"{'='*50}")

        # Create pipeline
        pipeline = None
        if not args.no_pipeline:
            pipeline = create_pipeline(model, args.provider)
            if pipeline is None:
                logger.warning(f"Skipping {model} - pipeline creation failed")
                continue

        # Create runner
        runner = EvaluationRunner(
            pipeline=pipeline,
            evaluator=evaluator,
            dataset_path=dataset_path,
        )

        # Load and run
        runner.load_dataset()
        result = runner.run(limit=args.limit, model_name=model)

        # Save individual result
        output_path = output_dir / f"{output_base}_{model.replace(':', '_')}.json"
        runner.save_results(result, output_path)

        all_results.append(result)

        # Print summary
        print(f"\n--- {model} Summary ---")
        print(f"  Evaluated: {result.evaluated_cases}/{result.total_cases} cases")
        print(f"  Similarity: {result.mean_similarity:.3f} (±{result.std_similarity:.3f})")
        print(f"  BLEU: {result.mean_bleu:.3f} (±{result.std_bleu:.3f})")
        print(f"  Faithfulness: {result.mean_faithfulness:.3f}")
        print(f"  Latency: {result.mean_latency_ms:.1f}ms (±{result.std_latency_ms:.1f})")
        print(f"  Verification Rate: {result.verification_rate * 100:.1f}%")

    # Generate reports if requested
    if args.report and all_results:
        md_path = output_dir / f"{output_base}_report.md"
        csv_path = output_dir / f"{output_base}_report.csv"

        md_report = generate_markdown_report(all_results, md_path)
        generate_csv_report(all_results, csv_path)

        print(f"\nReports saved to:")
        print(f"  Markdown: {md_path}")
        print(f"  CSV: {csv_path}")

    print(f"\nEvaluation complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
