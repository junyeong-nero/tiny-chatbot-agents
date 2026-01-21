"""Tests for evaluation dataset generator."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.evaluation.dataset_generator import (
    DatasetGenerator,
    Difficulty,
    EvaluationDataset,
    EvaluationItem,
    create_dataset_generator,
)


class TestDifficulty:
    """Tests for Difficulty enum."""

    def test_difficulty_values(self):
        """Test difficulty enum values."""
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"


class TestEvaluationItem:
    """Tests for EvaluationItem dataclass."""

    def test_create_item(self):
        """Test basic creation."""
        item = EvaluationItem(
            id="test_123",
            question="계좌 개설 방법은?",
            golden_answer="앱에서 개설 가능합니다",
            category="계좌",
        )

        assert item.id == "test_123"
        assert item.question == "계좌 개설 방법은?"
        assert item.difficulty == Difficulty.MEDIUM  # default

    def test_to_dict(self):
        """Test serialization."""
        item = EvaluationItem(
            id="test_123",
            question="q",
            golden_answer="a",
            category="cat",
            difficulty=Difficulty.HARD,
        )

        d = item.to_dict()
        assert d["id"] == "test_123"
        assert d["golden_answer"] == "a"
        assert d["expected_answer"] == "a"  # backward compatibility
        assert d["difficulty"] == "hard"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "id": "test_123",
            "question": "q",
            "golden_answer": "a",
            "category": "cat",
            "difficulty": "hard",
        }

        item = EvaluationItem.from_dict(data)
        assert item.id == "test_123"
        assert item.golden_answer == "a"
        assert item.difficulty == Difficulty.HARD

    def test_from_dict_with_expected_answer(self):
        """Test deserialization with legacy expected_answer field."""
        data = {
            "id": "test_123",
            "question": "q",
            "expected_answer": "legacy_answer",
            "category": "cat",
        }

        item = EvaluationItem.from_dict(data)
        assert item.golden_answer == "legacy_answer"


class TestEvaluationDataset:
    """Tests for EvaluationDataset dataclass."""

    @pytest.fixture
    def sample_items(self):
        """Create sample items."""
        return [
            EvaluationItem(id="1", question="q1", golden_answer="a1", category="cat1"),
            EvaluationItem(id="2", question="q2", golden_answer="a2", category="cat2"),
        ]

    def test_create_dataset(self, sample_items):
        """Test basic creation."""
        dataset = EvaluationDataset(
            items=sample_items,
            generator_model="gpt-4o",
            generation_timestamp="2024-01-01T00:00:00",
        )

        assert len(dataset) == 2
        assert dataset.generator_model == "gpt-4o"

    def test_iteration(self, sample_items):
        """Test iteration over items."""
        dataset = EvaluationDataset(
            items=sample_items,
            generator_model="gpt-4o",
            generation_timestamp="",
        )

        items_list = list(dataset)
        assert len(items_list) == 2

    def test_to_dict(self, sample_items):
        """Test serialization."""
        dataset = EvaluationDataset(
            items=sample_items,
            generator_model="gpt-4o",
            generation_timestamp="2024-01-01",
        )

        d = dataset.to_dict()
        assert d["version"] == "1.0"
        assert d["generator_model"] == "gpt-4o"
        assert len(d["items"]) == 2

    def test_to_legacy_format(self, sample_items):
        """Test conversion to legacy format."""
        dataset = EvaluationDataset(
            items=sample_items,
            generator_model="gpt-4o",
            generation_timestamp="",
        )

        legacy = dataset.to_legacy_format()
        assert len(legacy) == 2
        assert "question" in legacy[0]
        assert "expected_answer" in legacy[0]
        assert "category" in legacy[0]

    def test_save_and_load(self, sample_items):
        """Test saving and loading dataset."""
        dataset = EvaluationDataset(
            items=sample_items,
            generator_model="gpt-4o",
            generation_timestamp="2024-01-01",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dataset.json"
            dataset.save(path)

            loaded = EvaluationDataset.load(path)

            assert len(loaded) == 2
            assert loaded.generator_model == "gpt-4o"

    def test_load_legacy_format(self):
        """Test loading legacy format (list of test cases)."""
        legacy_data = [
            {"question": "q1", "expected_answer": "a1", "category": "cat1"},
            {"question": "q2", "expected_answer": "a2", "category": "cat2"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.json"
            with open(path, "w") as f:
                json.dump(legacy_data, f)

            dataset = EvaluationDataset.load(path)

            assert len(dataset) == 2
            assert dataset.items[0].golden_answer == "a1"


class TestDatasetGenerator:
    """Tests for DatasetGenerator class."""

    @pytest.fixture
    def mock_client(self):
        """Create mock frontier client."""
        client = Mock()
        client.model_name = "gpt-4o"
        client.generate.return_value = "이것은 모범 답변입니다."
        return client

    def test_generate_golden_answer(self, mock_client):
        """Test generating a single golden answer."""
        generator = DatasetGenerator(frontier_client=mock_client)

        answer = generator.generate_golden_answer("계좌 개설 방법은?")

        assert answer == "이것은 모범 답변입니다."
        mock_client.generate.assert_called_once()

    def test_generate_golden_answer_with_context(self, mock_client):
        """Test generating answer with context."""
        generator = DatasetGenerator(frontier_client=mock_client)

        context = [{"content": "계좌 개설은 앱에서 가능합니다."}]
        answer = generator.generate_golden_answer("계좌 개설 방법은?", context=context)

        assert answer == "이것은 모범 답변입니다."
        # Verify context was included in prompt
        call_args = mock_client.generate.call_args
        prompt = call_args[0][0][0]["content"]
        assert "계좌 개설은 앱에서 가능합니다" in prompt

    def test_generate_item(self, mock_client):
        """Test generating a single evaluation item."""
        generator = DatasetGenerator(frontier_client=mock_client)

        item = generator.generate_item(
            question="계좌 개설 방법은?",
            category="계좌",
            difficulty=Difficulty.EASY,
        )

        assert isinstance(item, EvaluationItem)
        assert item.question == "계좌 개설 방법은?"
        assert item.category == "계좌"
        assert item.difficulty == Difficulty.EASY
        assert item.golden_answer == "이것은 모범 답변입니다."

    def test_generate_from_questions(self, mock_client):
        """Test generating dataset from questions list."""
        generator = DatasetGenerator(frontier_client=mock_client)

        questions = [
            {"question": "q1", "category": "cat1"},
            {"question": "q2", "category": "cat2"},
        ]

        dataset = generator.generate_from_questions(questions, show_progress=False)

        assert isinstance(dataset, EvaluationDataset)
        assert len(dataset) == 2
        assert dataset.generator_model == "gpt-4o"

    def test_generate_from_questions_with_error(self, mock_client):
        """Test that generation continues after individual errors."""
        mock_client.generate.side_effect = [
            "answer1",
            Exception("API error"),
            "answer3",
        ]

        generator = DatasetGenerator(frontier_client=mock_client)

        questions = [
            {"question": "q1"},
            {"question": "q2"},
            {"question": "q3"},
        ]

        dataset = generator.generate_from_questions(questions, show_progress=False)

        # Should have 2 items (one failed)
        assert len(dataset) == 2

    def test_generate_from_file(self, mock_client):
        """Test generating from questions file."""
        generator = DatasetGenerator(frontier_client=mock_client)

        questions = [
            {"question": "q1", "category": "cat1"},
            {"question": "q2", "category": "cat2"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "questions.json"
            output_path = Path(tmpdir) / "output.json"

            with open(input_path, "w") as f:
                json.dump(questions, f)

            dataset = generator.generate_from_file(input_path, output_path)

            assert len(dataset) == 2
            assert output_path.exists()

    def test_generate_id(self, mock_client):
        """Test ID generation is unique."""
        generator = DatasetGenerator(frontier_client=mock_client)

        id1 = generator._generate_id("question1")
        id2 = generator._generate_id("question2")

        assert id1 != id2
        assert len(id1) == 12  # SHA256 truncated to 12 chars


class TestCreateDatasetGenerator:
    """Tests for factory function."""

    @patch("src.evaluation.frontier_client.create_frontier_client")
    def test_create_dataset_generator(self, mock_create_client):
        """Test factory function creates generator with correct config."""
        mock_client = Mock()
        mock_client.model_name = "claude-sonnet-4-20250514"
        mock_create_client.return_value = mock_client

        generator = create_dataset_generator(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
        )

        assert isinstance(generator, DatasetGenerator)
        mock_create_client.assert_called_once_with(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            temperature=0.3,
        )
