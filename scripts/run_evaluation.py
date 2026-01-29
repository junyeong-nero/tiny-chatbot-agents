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

    # LLM-as-a-Judge options
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Enable LLM-as-a-Judge evaluation",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Model to use for LLM-as-a-Judge (default: gpt-4o)",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "google"],
        help="Provider for judge model (default: openai)",
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

        # Check if LLM is available when health_check exists
        health_check = getattr(llm_client, "health_check", None)
        if callable(health_check) and not health_check():
            logger.warning(f"LLM health check failed for {model}")
            return None

        pipeline = RAGPipeline(llm=llm_client)
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

    # Create LLM judge if requested
    llm_judge = None
    if args.use_llm_judge:
        try:
            from src.evaluation import create_llm_judge

            logger.info(f"Creating LLM judge with {args.judge_provider}/{args.judge_model}")
            llm_judge = create_llm_judge(
                provider=args.judge_provider,
                model=args.judge_model,
            )
        except Exception as e:
            logger.error(f"Failed to create LLM judge: {e}")
            logger.info("Continuing without LLM-as-a-Judge evaluation")

    # Create evaluator
    evaluator = LLMEvaluator(
        llm_judge=llm_judge,
        use_llm_judge=args.use_llm_judge and llm_judge is not None,
    )

    # Run evaluation for each model
    all_results = []
    for model in models:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Evaluating model: {model}")
        logger.info(f"{'=' * 50}")

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

        # Print LLM-judge scores if available
        if result.mean_llm_judge_score > 0:
            print(f"\n  --- LLM-as-a-Judge ({result.llm_judge_model}) ---")
            print(f"  Overall Score: {result.mean_llm_judge_score:.2f}/5")
            print(f"  Correctness: {result.mean_llm_correctness:.2f}/5")
            print(f"  Helpfulness: {result.mean_llm_helpfulness:.2f}/5")
            print(f"  Faithfulness: {result.mean_llm_faithfulness:.2f}/5")
            print(f"  Fluency: {result.mean_llm_fluency:.2f}/5")

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
