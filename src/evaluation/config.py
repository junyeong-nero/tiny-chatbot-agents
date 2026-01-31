"""Evaluation configuration loader.

This module provides configuration loading and management for the evaluation pipeline.
Loads settings from YAML config file and provides typed access to configuration values.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default config file path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "evaluation_config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file and return as dictionary.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary with configuration values

    Raises:
        FileNotFoundError: If config file does not exist
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for config loading. Install with: pip install pyyaml")

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class JudgeConfig:
    """Configuration for LLM-as-a-Judge evaluation."""

    enabled: bool = True
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 2048
    criteria: list[str] = field(
        default_factory=lambda: ["correctness", "helpfulness", "faithfulness", "fluency"]
    )
    use_comprehensive: bool = True
    max_retries: int = 2
    retry_delay: float = 1.0
    retry_backoff_multiplier: float = 2.0
    auto_diverse: bool = True
    strict_diversity: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeConfig:
        """Create JudgeConfig from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            provider=data.get("provider", "openai"),
            model=data.get("model", "gpt-4o"),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 2048),
            criteria=data.get("criteria", ["correctness", "helpfulness", "faithfulness", "fluency"]),
            use_comprehensive=data.get("use_comprehensive", True),
            max_retries=data.get("max_retries", 2),
            retry_delay=data.get("retry_delay", 1.0),
            retry_backoff_multiplier=data.get("retry_backoff_multiplier", 2.0),
            auto_diverse=data.get("auto_diverse", True),
            strict_diversity=data.get("strict_diversity", True),
        )


@dataclass
class TargetConfig:
    """Configuration for target model being evaluated."""

    provider: str = "openai"
    base_url: str = "http://localhost:8000/v1"
    model: str = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 1024

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetConfig:
        """Create TargetConfig from dictionary."""
        return cls(
            provider=data.get("provider", "openai"),
            base_url=data.get("base_url", "http://localhost:8000/v1"),
            model=data.get("model", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"),
            temperature=data.get("temperature", 0.1),
            max_tokens=data.get("max_tokens", 1024),
        )


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""

    model: str = "intfloat/multilingual-e5-small"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingConfig:
        """Create EmbeddingConfig from dictionary."""
        return cls(model=data.get("model", "intfloat/multilingual-e5-small"))


@dataclass
class RunnerConfig:
    """Configuration for evaluation runner."""

    parallel: bool = False
    max_workers: int = 4
    show_progress: bool = True
    progress_interval: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunnerConfig:
        """Create RunnerConfig from dictionary."""
        return cls(
            parallel=data.get("parallel", False),
            max_workers=data.get("max_workers", 4),
            show_progress=data.get("show_progress", True),
            progress_interval=data.get("progress_interval", 10),
        )


@dataclass
class DatasetConfig:
    """Configuration for evaluation dataset."""

    path: str = "data/evaluation/evaluation_dataset_v1.json"
    validate: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        """Create DatasetConfig from dictionary."""
        return cls(
            path=data.get("path", "data/evaluation/evaluation_dataset_v1.json"),
            validate=data.get("validate", True),
        )


@dataclass
class ScoringConfig:
    """Configuration for scoring parameters."""

    min_score: float = 1.0
    max_score: float = 5.0
    neutral_score: float = 3.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoringConfig:
        """Create ScoringConfig from dictionary."""
        return cls(
            min_score=data.get("min_score", 1.0),
            max_score=data.get("max_score", 5.0),
            neutral_score=data.get("neutral_score", 3.0),
        )


@dataclass
class OutputConfig:
    """Configuration for output settings."""

    directory: str = "data/evaluation/results"
    include_case_results: bool = True
    include_category_breakdown: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputConfig:
        """Create OutputConfig from dictionary."""
        return cls(
            directory=data.get("directory", "data/evaluation/results"),
            include_case_results=data.get("include_case_results", True),
            include_category_breakdown=data.get("include_category_breakdown", True),
        )


@dataclass
class ProviderDefault:
    """Default settings for a provider."""

    model: str
    api_key_env: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderDefault:
        """Create ProviderDefault from dictionary."""
        return cls(
            model=data.get("model", ""),
            api_key_env=data.get("api_key_env", ""),
        )


@dataclass
class DiversityPair:
    """Mapping for diverse model selection."""

    provider: str
    model: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiversityPair:
        """Create DiversityPair from dictionary."""
        return cls(
            provider=data.get("provider", "openai"),
            model=data.get("model", "gpt-4o"),
        )


@dataclass
class EvaluationConfig:
    """Complete evaluation configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    provider_defaults: dict[str, ProviderDefault] = field(default_factory=dict)
    diversity_pairs: dict[str, DiversityPair] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationConfig:
        """Create EvaluationConfig from dictionary."""
        # Parse provider defaults
        provider_defaults = {}
        for provider, settings in data.get("provider_defaults", {}).items():
            provider_defaults[provider] = ProviderDefault.from_dict(settings)

        # Parse diversity pairs
        diversity_pairs = {}
        for model, mapping in data.get("diversity_pairs", {}).items():
            diversity_pairs[model] = DiversityPair.from_dict(mapping)

        return cls(
            dataset=DatasetConfig.from_dict(data.get("dataset", {})),
            judge=JudgeConfig.from_dict(data.get("judge", {})),
            target=TargetConfig.from_dict(data.get("target", {})),
            embedding=EmbeddingConfig.from_dict(data.get("embedding", {})),
            runner=RunnerConfig.from_dict(data.get("runner", {})),
            scoring=ScoringConfig.from_dict(data.get("scoring", {})),
            output=OutputConfig.from_dict(data.get("output", {})),
            provider_defaults=provider_defaults,
            diversity_pairs=diversity_pairs,
        )

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> EvaluationConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file. If None, uses default path.
                        Can also be set via EVALUATION_CONFIG_PATH env var.

        Returns:
            EvaluationConfig instance

        Raises:
            FileNotFoundError: If config file does not exist
        """
        if config_path is None:
            config_path = os.getenv("EVALUATION_CONFIG_PATH")

        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        else:
            config_path = Path(config_path)

        logger.info(f"Loading evaluation config from: {config_path}")
        data = _load_yaml(config_path)
        return cls.from_dict(data)

    def get_provider_default(self, provider: str) -> ProviderDefault | None:
        """Get default settings for a provider.

        Args:
            provider: Provider name (openai, anthropic, google)

        Returns:
            ProviderDefault or None if not found
        """
        return self.provider_defaults.get(provider)

    def get_diverse_judge(self, generator_model: str) -> DiversityPair | None:
        """Get diverse judge model for a generator model.

        Args:
            generator_model: Model used for generation

        Returns:
            DiversityPair with recommended judge or None if not found
        """
        return self.diversity_pairs.get(generator_model)


# Global config instance (lazy loaded)
_config: EvaluationConfig | None = None


def get_config(config_path: str | Path | None = None, reload: bool = False) -> EvaluationConfig:
    """Get the evaluation configuration (singleton).

    Args:
        config_path: Optional path to config file
        reload: Force reload of configuration

    Returns:
        EvaluationConfig instance
    """
    global _config
    if _config is None or reload:
        _config = EvaluationConfig.load(config_path)
    return _config


def reset_config() -> None:
    """Reset the global config instance (for testing)."""
    global _config
    _config = None


def create_evaluation_runner_from_config(
    pipeline: Any = None,
    config_path: str | Path | None = None,
) -> Any:
    """Create a fully configured EvaluationRunner from config.

    This is a convenience function that creates an EvaluationRunner with
    LLMJudge pre-configured based on the evaluation_config.yaml settings.

    Args:
        pipeline: Optional RAGPipeline instance for generating answers
        config_path: Optional path to config file

    Returns:
        Configured EvaluationRunner instance

    Example:
        >>> runner = create_evaluation_runner_from_config()
        >>> runner.load_dataset()
        >>> result = runner.run(model_name="my-model")
    """
    # Import here to avoid circular dependency
    from .evaluator import LLMEvaluator
    from .llm_judge import create_llm_judge
    from .runner import EvaluationRunner

    config = get_config(config_path)

    # Create LLM judge if enabled
    llm_judge = None
    if config.judge.enabled:
        llm_judge = create_llm_judge()

    # Create evaluator with judge
    evaluator = LLMEvaluator(
        llm_judge=llm_judge,
        use_llm_judge=config.judge.enabled,
    )

    # Create runner
    runner = EvaluationRunner(
        pipeline=pipeline,
        evaluator=evaluator,
        dataset_path=config.dataset.path,
    )

    return runner
