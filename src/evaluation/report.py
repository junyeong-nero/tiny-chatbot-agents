"""Report generation for evaluation results.

This module provides functions to generate markdown and CSV reports
from evaluation results.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .runner import EvaluationResult


def generate_markdown_report(
    results: list[EvaluationResult],
    output_path: str | Path | None = None,
) -> str:
    """Generate a markdown comparison report for multiple models.

    Args:
        results: List of EvaluationResult for different models
        output_path: Optional path to save the report

    Returns:
        Markdown report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# LLM Evaluation Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Models Evaluated**: {len(results)}",
        "",
        "## Summary",
        "",
        "| Model | Similarity | BLEU | Faithfulness | Latency (ms) | Verified |",
        "|-------|------------|------|--------------|--------------|----------|",
    ]

    for r in results:
        lines.append(
            f"| {r.model_name} | {r.mean_similarity:.3f} | {r.mean_bleu:.3f} | "
            f"{r.mean_faithfulness:.3f} | {r.mean_latency_ms:.1f} | "
            f"{r.verification_rate * 100:.1f}% |"
        )

    lines.extend([
        "",
        "## Category Breakdown",
        "",
    ])

    # Collect all categories
    all_categories = set()
    for r in results:
        all_categories.update(r.category_scores.keys())

    for category in sorted(all_categories):
        lines.append(f"### {category}")
        lines.append("")
        lines.append("| Model | Similarity | BLEU | Count |")
        lines.append("|-------|------------|------|-------|")

        for r in results:
            if category in r.category_scores:
                cat = r.category_scores[category]
                lines.append(
                    f"| {r.model_name} | {cat['mean_similarity']:.3f} | "
                    f"{cat['mean_bleu']:.3f} | {cat['count']} |"
                )

        lines.append("")

    # Statistics
    lines.extend([
        "## Statistics",
        "",
        "| Model | Std Similarity | Std BLEU | Std Latency |",
        "|-------|----------------|----------|-------------|",
    ])

    for r in results:
        lines.append(
            f"| {r.model_name} | {r.std_similarity:.3f} | {r.std_bleu:.3f} | "
            f"{r.std_latency_ms:.1f} |"
        )

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

    return report


def generate_csv_report(
    results: list[EvaluationResult],
    output_path: str | Path,
) -> Path:
    """Generate a CSV report with per-case results.

    Args:
        results: List of EvaluationResult for different models
        output_path: Path to save the CSV file

    Returns:
        Path to saved CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "question",
        "category",
        "similarity",
        "bleu",
        "faithfulness",
        "latency_ms",
        "verified",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            for case in result.case_results:
                writer.writerow({
                    "model": result.model_name,
                    "question": case.get("question", "")[:100],  # Truncate
                    "category": case.get("category", ""),
                    "similarity": f"{case.get('answer_similarity', 0):.4f}",
                    "bleu": f"{case.get('bleu_score', 0):.4f}",
                    "faithfulness": f"{case.get('faithfulness', 0):.4f}",
                    "latency_ms": f"{case.get('latency_ms', 0):.1f}",
                    "verified": case.get("verified", False),
                })

    return output_path


def load_results(path: str | Path) -> EvaluationResult:
    """Load evaluation results from JSON file.

    Args:
        path: Path to JSON results file

    Returns:
        EvaluationResult instance
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return EvaluationResult(
        model_name=data.get("model_name", "unknown"),
        total_cases=data.get("total_cases", 0),
        evaluated_cases=data.get("evaluated_cases", 0),
        mean_similarity=data.get("metrics", {}).get("mean_similarity", 0),
        mean_bleu=data.get("metrics", {}).get("mean_bleu", 0),
        mean_faithfulness=data.get("metrics", {}).get("mean_faithfulness", 0),
        mean_latency_ms=data.get("metrics", {}).get("mean_latency_ms", 0),
        std_similarity=data.get("metrics", {}).get("std_similarity", 0),
        std_bleu=data.get("metrics", {}).get("std_bleu", 0),
        std_latency_ms=data.get("metrics", {}).get("std_latency_ms", 0),
        verified_count=data.get("verification", {}).get("verified_count", 0),
        verification_rate=data.get("verification", {}).get("verification_rate", 0),
        category_scores=data.get("category_scores", {}),
        case_results=data.get("case_results", []),
    )
