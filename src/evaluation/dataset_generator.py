"""Evaluation dataset generator using frontier models.

This module provides tools for generating evaluation datasets with
golden (reference) answers using frontier models like Claude, GPT, or Gemini.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .frontier_client import FrontierClient

logger = logging.getLogger(__name__)


class Difficulty(Enum):
    """Question difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class EvaluationItem:
    """Single evaluation QA pair with golden answer."""

    id: str
    question: str
    golden_answer: str
    category: str
    difficulty: Difficulty = Difficulty.MEDIUM
    source_context: list[dict[str, Any]] = field(default_factory=list)
    generation_context: list[dict[str, Any]] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "golden_answer": self.golden_answer,
            "expected_answer": self.golden_answer,
            "category": self.category,
            "difficulty": self.difficulty.value,
            "source_context": self.source_context,
            "generation_context": self.generation_context,
            "expected_sources": self.expected_sources,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationItem":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            question=data.get("question", ""),
            golden_answer=data.get("golden_answer", data.get("expected_answer", "")),
            category=data.get("category", ""),
            difficulty=Difficulty(data.get("difficulty", "medium")),
            source_context=data.get("source_context", []),
            generation_context=data.get("generation_context", []),
            expected_sources=data.get("expected_sources", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvaluationDataset:
    """Collection of evaluation items."""

    items: list[EvaluationItem]
    generator_model: str
    generation_timestamp: str
    version: str = "1.0"
    generator_provider: str = ""  # Track provider for judge diversity selection
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generator_model": self.generator_model,
            "generator_provider": self.generator_provider,
            "generation_timestamp": self.generation_timestamp,
            "metadata": self.metadata,
            "items": [item.to_dict() for item in self.items],
        }

    def to_legacy_format(self) -> list[dict[str, Any]]:
        """Convert to legacy format (list of test cases) for backward compatibility."""
        return [
            {
                "question": item.question,
                "expected_answer": item.golden_answer,
                "category": item.category,
            }
            for item in self.items
        ]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationDataset":
        """Create from dictionary."""
        items = [EvaluationItem.from_dict(item) for item in data.get("items", [])]
        return cls(
            items=items,
            generator_model=data.get("generator_model", "unknown"),
            generation_timestamp=data.get("generation_timestamp", ""),
            version=data.get("version", "1.0"),
            generator_provider=data.get("generator_provider", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationDataset":
        """Load dataset from JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle legacy format (list of test cases)
        if isinstance(data, list):
            items = []
            for i, item in enumerate(data):
                items.append(
                    EvaluationItem(
                        id=f"legacy_{i}",
                        question=item.get("question", ""),
                        golden_answer=item.get("expected_answer", ""),
                        category=item.get("category", ""),
                    )
                )
            return cls(
                items=items,
                generator_model="unknown",
                generator_provider="",
                generation_timestamp="",
                version="1.0",
            )

        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Saved dataset with {len(self.items)} items to {path}")
        return path


# Golden answer generation prompt
GOLDEN_ANSWER_PROMPT = """당신은 한국투자증권 고객 서비스 전문가입니다.

다음 고객 질문에 대해 완벽한 모범 답변을 작성하세요.

{context_section}

[고객 질문]
{question}

[모범 답변 작성 규칙]
1. 정확하고 완전한 정보를 제공하세요
2. 친절하고 전문적인 톤을 유지하세요
3. 필요시 단계별 설명을 포함하세요
4. 관련 참조 정보가 있으면 명시하세요 (예: [참조: 제N조])
5. 200자 내외로 간결하게 작성하세요
6. 불확실한 정보는 포함하지 마세요

[모범 답변]"""


class DatasetGenerator:
    """Generate evaluation datasets with golden answers using frontier models."""

    def __init__(
        self,
        frontier_client: FrontierClient,
        qna_store: Any | None = None,
        tos_store: Any | None = None,
        rag_pipeline: Any | None = None,
        use_pipeline_retrieval: bool = False,
    ) -> None:
        """Initialize the dataset generator.

        Args:
            frontier_client: Client for frontier model API
            qna_store: Optional QnAVectorStore for sampling questions
            tos_store: Optional ToSVectorStore for context
            rag_pipeline: Optional RAG pipeline for standardized retrieval
            use_pipeline_retrieval: Use RAG pipeline for context retrieval
        """
        self.client = frontier_client
        self.qna_store = qna_store
        self.tos_store = tos_store
        self.rag_pipeline = rag_pipeline
        self.use_pipeline_retrieval = use_pipeline_retrieval

    def _generate_id(self, question: str) -> str:
        """Generate a unique ID for a question."""
        hash_input = f"{question}_{datetime.now().isoformat()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    def _format_context(self, context: list[dict[str, Any]]) -> str:
        """Format context for the prompt."""
        if not context:
            return ""

        context_parts = []
        for item in context:
            if "content" in item:
                context_parts.append(item["content"])
            elif "answer" in item:
                context_parts.append(f"Q: {item.get('question', '')}\nA: {item['answer']}")

        return "\n\n".join(context_parts)

    def generate_golden_answer(
        self,
        question: str,
        context: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a golden answer for a single question.

        Args:
            question: The question to answer
            context: Optional context information

        Returns:
            Generated golden answer
        """
        context_text = self._format_context(context or [])
        context_section = f"[참고 정보]\n{context_text}" if context_text else "[참고 정보]\n(없음)"

        prompt = GOLDEN_ANSWER_PROMPT.format(
            context_section=context_section,
            question=question,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.generate(messages)
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate golden answer: {e}")
            raise

    def generate_item(
        self,
        question: str,
        category: str = "",
        difficulty: Difficulty = Difficulty.MEDIUM,
        context: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationItem:
        """Generate a single evaluation item.

        Args:
            question: The question
            category: Question category
            difficulty: Difficulty level
            context: Optional context information (overridden if use_pipeline_retrieval)
            metadata: Optional metadata

        Returns:
            EvaluationItem with golden answer
        """
        generation_context: list[dict[str, Any]] = []
        expected_sources: list[str] = []

        if self.use_pipeline_retrieval and self.rag_pipeline:
            retrieved = self._retrieve_context_from_pipeline(question)
            generation_context = retrieved
            expected_sources = self._extract_source_ids(retrieved)
            effective_context = retrieved
        else:
            effective_context = context or []
            generation_context = effective_context
            expected_sources = self._extract_source_ids(effective_context)

        golden_answer = self.generate_golden_answer(question, effective_context)

        return EvaluationItem(
            id=self._generate_id(question),
            question=question,
            golden_answer=golden_answer,
            category=category,
            difficulty=difficulty,
            source_context=context or [],
            generation_context=generation_context,
            expected_sources=expected_sources,
            metadata=metadata or {},
        )

    def _retrieve_context_from_pipeline(self, question: str) -> list[dict[str, Any]]:
        """Retrieve context using the RAG pipeline."""
        if not self.rag_pipeline:
            return []

        try:
            if hasattr(self.rag_pipeline, "retrieve"):
                return self.rag_pipeline.retrieve(question)
            elif hasattr(self.rag_pipeline, "search_qna"):
                qna_results = self.rag_pipeline.search_qna(question, top_k=3)
                tos_results = self.rag_pipeline.search_tos(question, top_k=3)
                return qna_results + tos_results
            else:
                logger.warning("RAG pipeline has no retrieve method")
                return []
        except Exception as e:
            logger.warning(f"Pipeline retrieval failed: {e}")
            return []

    def _extract_source_ids(self, context: list[dict[str, Any]]) -> list[str]:
        """Extract source identifiers from context items."""
        sources = []
        for item in context:
            source_id = item.get("doc_id") or item.get("id") or item.get("source")
            if source_id:
                sources.append(str(source_id))
            elif "section_title" in item:
                sources.append(item["section_title"])
        return sources

    def generate_from_questions(
        self,
        questions: list[dict[str, Any]],
        show_progress: bool = True,
    ) -> EvaluationDataset:
        """Generate golden answers for provided questions.

        Args:
            questions: List of question dicts with 'question', optional 'category', 'context'
            show_progress: Whether to log progress

        Returns:
            EvaluationDataset with golden answers
        """
        items = []
        total = len(questions)

        for i, q in enumerate(questions):
            try:
                item = self.generate_item(
                    question=q.get("question", q.get("q", "")),
                    category=q.get("category", ""),
                    difficulty=Difficulty(q.get("difficulty", "medium")),
                    context=q.get("context"),
                    metadata=q.get("metadata"),
                )
                items.append(item)

                if show_progress:
                    logger.info(f"Generated {i + 1}/{total}: {item.question[:50]}...")

            except Exception as e:
                logger.error(f"Failed to generate item {i}: {e}")
                continue

        return EvaluationDataset(
            items=items,
            generator_model=self.client.model_name,
            generator_provider=self.client.provider_name,
            generation_timestamp=datetime.now().isoformat(),
            metadata={
                "source": "questions_list",
                "total_requested": total,
                "total_generated": len(items),
            },
        )

    def generate_from_qna_store(
        self,
        n_samples: int = 50,
        categories: list[str] | None = None,
        random_seed: int | None = None,
    ) -> EvaluationDataset:
        """Generate evaluation dataset from existing QnA store.

        Args:
            n_samples: Number of samples to generate
            categories: Optional list of categories to filter
            random_seed: Optional random seed for reproducibility

        Returns:
            EvaluationDataset with golden answers
        """
        if self.qna_store is None:
            raise ValueError("QnA store not provided")

        # Sample questions from QnA store
        import random

        if random_seed is not None:
            random.seed(random_seed)

        # Get all QnA pairs
        all_qna = self.qna_store.get_all_qna()
        if categories:
            all_qna = [q for q in all_qna if q.get("category") in categories]

        if len(all_qna) < n_samples:
            logger.warning(f"Requested {n_samples} samples but only {len(all_qna)} available")
            n_samples = len(all_qna)

        sampled = random.sample(all_qna, n_samples)

        # Convert to question format
        questions = [
            {
                "question": q.get("question", ""),
                "category": q.get("category", ""),
                "context": [{"content": q.get("answer", ""), "source": "qna"}],
            }
            for q in sampled
        ]

        return self.generate_from_questions(questions)

    def generate_from_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> EvaluationDataset:
        """Generate golden answers from a questions file.

        Args:
            input_path: Path to JSON file with questions
            output_path: Optional path to save the generated dataset

        Returns:
            EvaluationDataset with golden answers
        """
        input_path = Path(input_path)

        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle different input formats
        if isinstance(data, list):
            questions = data
        elif isinstance(data, dict) and "questions" in data:
            questions = data["questions"]
        elif isinstance(data, dict) and "items" in data:
            questions = [
                {"question": item.get("question", ""), "category": item.get("category", "")}
                for item in data["items"]
            ]
        else:
            raise ValueError(f"Unknown input format in {input_path}")

        dataset = self.generate_from_questions(questions)

        if output_path:
            dataset.save(output_path)

        return dataset


def create_dataset_generator(
    provider: str = "openai",
    model: str | None = None,
    qna_store: Any | None = None,
    tos_store: Any | None = None,
) -> DatasetGenerator:
    """Factory function to create a dataset generator.

    Args:
        provider: Provider name ('openai', 'anthropic', 'google')
        model: Optional model override
        qna_store: Optional QnAVectorStore
        tos_store: Optional ToSVectorStore

    Returns:
        Configured DatasetGenerator
    """
    from .frontier_client import create_frontier_client

    client = create_frontier_client(
        provider=provider,
        model=model,
        temperature=0.3,  # Slightly higher for more natural answers
    )

    return DatasetGenerator(
        frontier_client=client,
        qna_store=qna_store,
        tos_store=tos_store,
    )
