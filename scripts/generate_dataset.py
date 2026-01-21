#!/usr/bin/env python3
"""CLI script for generating evaluation datasets with golden answers.

Usage:
    # Generate from existing QnA store
    python scripts/generate_dataset.py --from-qna --n-samples 50

    # Generate from questions file
    python scripts/generate_dataset.py --questions questions.json

    # Specify frontier model
    python scripts/generate_dataset.py --model gpt-4o --provider openai --from-qna

    # Use Claude for generation
    python scripts/generate_dataset.py --model claude-sonnet-4-20250514 --provider anthropic --questions questions.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.dataset_generator import (
    DatasetGenerator,
    EvaluationDataset,
    create_dataset_generator,
)
from src.evaluation.frontier_client import create_frontier_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation Dataset Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 50 samples from QnA store using GPT-4o
    python scripts/generate_dataset.py --from-qna --n-samples 50

    # Generate from a questions file using Claude
    python scripts/generate_dataset.py --questions data/questions.json \\
        --provider anthropic --model claude-sonnet-4-20250514

    # Generate with specific output path
    python scripts/generate_dataset.py --from-qna --output data/eval/my_dataset.json
        """,
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--from-qna",
        action="store_true",
        help="Generate golden answers from QnA vector store samples",
    )
    input_group.add_argument(
        "--questions",
        type=str,
        help="Path to JSON file containing questions to generate answers for",
    )

    # Model configuration
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "google"],
        help="Frontier model provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: gpt-4o for openai, claude-sonnet-4-20250514 for anthropic)",
    )

    # Generation options
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples to generate when using --from-qna (default: 50)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated categories to filter (only for --from-qna)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (only for --from-qna)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/generated_dataset.json",
        help="Output path for generated dataset (default: data/evaluation/generated_dataset.json)",
    )

    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without making API calls",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def load_qna_store():
    """Load QnA vector store."""
    try:
        from src.vectorstore import QnAVectorStore

        store = QnAVectorStore()
        logger.info(f"Loaded QnA store with {store.count()} items")
        return store
    except Exception as e:
        logger.error(f"Failed to load QnA store: {e}")
        return None


def load_questions_file(path: str) -> list[dict]:
    """Load questions from JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "questions" in data:
            return data["questions"]
        elif "items" in data:
            return [
                {"question": item.get("question", ""), "category": item.get("category", "")}
                for item in data["items"]
            ]

    raise ValueError(f"Unknown format in {path}")


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate environment
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    import os

    required_env = env_var_map.get(args.provider)
    if required_env and not os.getenv(required_env):
        logger.error(f"Missing environment variable: {required_env}")
        logger.info(f"Please set {required_env} to use {args.provider} provider")
        sys.exit(1)

    # Dry run mode
    if args.dry_run:
        print("\n=== Dry Run Mode ===")
        print(f"Provider: {args.provider}")
        print(f"Model: {args.model or 'default'}")
        print(f"Output: {args.output}")

        if args.from_qna:
            print(f"Source: QnA store (n_samples={args.n_samples})")
            if args.categories:
                print(f"Categories: {args.categories}")
        else:
            questions = load_questions_file(args.questions)
            print(f"Source: {args.questions} ({len(questions)} questions)")

        print("\nNo API calls made (dry run).")
        return

    # Create generator
    logger.info(f"Creating dataset generator with {args.provider} provider")
    qna_store = load_qna_store() if args.from_qna else None

    generator = create_dataset_generator(
        provider=args.provider,
        model=args.model,
        qna_store=qna_store,
    )

    # Generate dataset
    if args.from_qna:
        if qna_store is None:
            logger.error("QnA store is required for --from-qna mode")
            sys.exit(1)

        categories = args.categories.split(",") if args.categories else None

        logger.info(f"Generating {args.n_samples} samples from QnA store...")
        dataset = generator.generate_from_qna_store(
            n_samples=args.n_samples,
            categories=categories,
            random_seed=args.seed,
        )
    else:
        questions = load_questions_file(args.questions)
        logger.info(f"Generating golden answers for {len(questions)} questions...")
        dataset = generator.generate_from_questions(questions)

    # Save dataset
    output_path = Path(args.output)
    dataset.save(output_path)

    # Print summary
    print("\n=== Dataset Generation Complete ===")
    print(f"Model: {dataset.generator_model}")
    print(f"Items generated: {len(dataset.items)}")
    print(f"Output: {output_path}")

    # Show sample
    if dataset.items:
        print("\n--- Sample Item ---")
        sample = dataset.items[0]
        print(f"Q: {sample.question[:100]}...")
        print(f"A: {sample.golden_answer[:200]}...")
        print(f"Category: {sample.category}")


if __name__ == "__main__":
    main()
