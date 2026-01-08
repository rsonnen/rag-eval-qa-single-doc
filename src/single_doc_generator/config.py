"""Application configuration for the question generator.

Single configuration file for the entire application. Loaded once at startup.
All LLM calls go through LiteLLM proxy using OpenAI-compatible API.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a single LLM model.

    Attributes:
        name: The model identifier (e.g., gpt-5.2, gemini-2.5-flash).
        kwargs: Additional parameters passed to LiteLLM (e.g., reasoning_effort).
    """

    name: str = Field(description="Model identifier")
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model parameters passed to LiteLLM",
    )


class AppConfig(BaseModel):
    """Application configuration loaded from config.yaml.

    Specifies models for each stage of the pipeline.

    Attributes:
        generator_model: Model for question generation.
        validator_model: Model for validation.
        deduplicator_model: Model for deduplication.
    """

    generator_model: ModelConfig = Field(description="Model for question generation")
    validator_model: ModelConfig = Field(description="Model for validation")
    deduplicator_model: ModelConfig = Field(description="Model for deduplication")


class EvaluationScenario(BaseModel):
    """A specific evaluation scenario for a corpus.

    Different scenarios allow generating different types of questions from
    the same corpus. For example, a legal corpus might have scenarios for
    graduate exams, paralegal training, and undergraduate surveys.

    Attributes:
        name: Short name for the scenario.
        description: Full description of the evaluation context and intent.
    """

    name: str = Field(description="Short name for this scenario")
    description: str = Field(
        description="Full description of evaluation context, audience, and intent"
    )


class CorpusConfig(BaseModel):
    """Configuration for a document corpus, loaded from corpus.yaml.

    Lives with the corpus data (not in the generator code). Provides context
    about what the corpus is and defines evaluation scenarios for question
    generation.

    Attributes:
        name: Human-readable corpus name.
        corpus_context: Description of what this corpus is and where it came from.
        scenarios: Named evaluation scenarios for this corpus.
    """

    name: str = Field(description="Human-readable corpus name")
    corpus_context: str = Field(
        description="What this corpus is, where it came from, what it contains"
    )
    scenarios: dict[str, EvaluationScenario] = Field(
        description="Named evaluation scenarios"
    )

    def get_scenario(self, scenario_name: str) -> EvaluationScenario:
        """Get a specific evaluation scenario by name.

        Args:
            scenario_name: Name of the scenario to retrieve.

        Returns:
            The requested EvaluationScenario.

        Raises:
            KeyError: If scenario doesn't exist.
        """
        if scenario_name not in self.scenarios:
            available = ", ".join(self.scenarios.keys())
            raise KeyError(
                f"Scenario '{scenario_name}' not found. Available: {available}"
            )
        return self.scenarios[scenario_name]


def load_corpus_config(corpus_path: Path) -> CorpusConfig:
    """Load corpus configuration from a corpus.yaml file.

    Args:
        corpus_path: Path to corpus directory (containing corpus.yaml)
                     or direct path to corpus.yaml file.

    Returns:
        Validated CorpusConfig instance.

    Raises:
        FileNotFoundError: If corpus.yaml not found.
        ValueError: If configuration is invalid.
    """
    config_file = corpus_path if corpus_path.is_file() else corpus_path / "corpus.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Corpus config not found: {config_file}")

    with config_file.open() as f:
        data = yaml.safe_load(f)

    return CorpusConfig(**data)


def _find_config_file() -> Path:
    """Find the config file, searching up from the package directory."""
    # Start from this file's directory and search upward
    current = Path(__file__).parent
    for _ in range(5):  # Don't search forever
        config_path = current / "config.yaml"
        if config_path.exists():
            return config_path
        current = current.parent
    raise FileNotFoundError("config.yaml not found")


@lru_cache(maxsize=1)
def load_config(config_path: Path | None = None) -> AppConfig:
    """Load application configuration from YAML file.

    Configuration is cached after first load. The config file is searched
    for starting from the package directory and moving upward.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If config file not found.
        ValueError: If configuration is invalid.
    """
    if config_path is None:
        config_path = _find_config_file()

    with config_path.open() as f:
        data = yaml.safe_load(f)

    return AppConfig(**data)


def get_config() -> AppConfig:
    """Get the application configuration.

    Returns cached configuration, loading from file on first call.
    """
    return load_config()
