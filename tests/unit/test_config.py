"""Unit tests for application configuration."""

import pytest
from pydantic import ValidationError

from single_doc_generator.config import (
    AppConfig,
    CorpusConfig,
    EvaluationScenario,
    ModelConfig,
    load_config,
    load_corpus_config,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_basic_config(self):
        """Test creating a basic model config."""
        config = ModelConfig(name="gpt-5.1")
        assert config.name == "gpt-5.1"
        assert config.kwargs == {}

    def test_config_with_kwargs(self):
        """Test creating a model config with kwargs."""
        config = ModelConfig(
            name="gemini-2.5-flash", kwargs={"reasoning_effort": "none"}
        )
        assert config.name == "gemini-2.5-flash"
        assert config.kwargs == {"reasoning_effort": "none"}


class TestAppConfig:
    """Tests for AppConfig."""

    def test_valid_config(self):
        """Test creating a valid config."""
        config = AppConfig(
            generator_model={"name": "gpt-5.1"},
            validator_model={"name": "claude-sonnet-4.5"},
            deduplicator_model={"name": "gpt-5-mini"},
        )
        assert config.generator_model.name == "gpt-5.1"
        assert config.validator_model.name == "claude-sonnet-4.5"
        assert config.deduplicator_model.name == "gpt-5-mini"

    def test_config_with_kwargs(self):
        """Test config with model kwargs."""
        config = AppConfig(
            generator_model={"name": "gpt-5.1"},
            validator_model={
                "name": "gemini-2.5-flash",
                "kwargs": {"reasoning_effort": "none"},
            },
            deduplicator_model={"name": "gpt-5-mini"},
        )
        assert config.validator_model.kwargs == {"reasoning_effort": "none"}


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
generator_model:
  name: gpt-5.1
validator_model:
  name: claude-sonnet-4.5
deduplicator_model:
  name: gpt-5-mini
""")
        # Clear cache before testing
        load_config.cache_clear()

        config = load_config(config_file)
        assert config.generator_model.name == "gpt-5.1"
        assert config.validator_model.name == "claude-sonnet-4.5"

    def test_loads_kwargs_from_yaml(self, tmp_path):
        """Test loading model kwargs from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
generator_model:
  name: gpt-5.1
validator_model:
  name: gemini-2.5-flash
  kwargs:
    reasoning_effort: none
deduplicator_model:
  name: gpt-5-mini
""")
        load_config.cache_clear()

        config = load_config(config_file)
        assert config.validator_model.kwargs == {"reasoning_effort": "none"}


class TestEvaluationScenario:
    """Tests for EvaluationScenario model."""

    def test_required_fields(self):
        """Test that name and description are required."""
        scenario = EvaluationScenario(
            name="Graduate Exam",
            description="Questions for graduate-level evaluation",
        )
        assert scenario.name == "Graduate Exam"
        assert scenario.description == "Questions for graduate-level evaluation"

    def test_missing_name_raises_error(self):
        """Test that missing name raises ValidationError."""
        with pytest.raises(ValidationError):
            EvaluationScenario(description="Test description")

    def test_missing_description_raises_error(self):
        """Test that missing description raises ValidationError."""
        with pytest.raises(ValidationError):
            EvaluationScenario(name="Test")


class TestCorpusConfig:
    """Tests for CorpusConfig model."""

    def test_required_fields(self):
        """Test that name, corpus_context, and scenarios are required."""
        config = CorpusConfig(
            name="Test Corpus",
            corpus_context="Test context describing the corpus",
            scenarios={
                "graduate_exam": EvaluationScenario(
                    name="Graduate Exam",
                    description="Test for graduate students",
                )
            },
        )
        assert config.name == "Test Corpus"
        assert config.corpus_context == "Test context describing the corpus"
        assert "graduate_exam" in config.scenarios

    def test_get_scenario_returns_scenario(self):
        """Test that get_scenario returns the correct scenario."""
        config = CorpusConfig(
            name="Test Corpus",
            corpus_context="Test context",
            scenarios={
                "rag_eval": EvaluationScenario(
                    name="RAG Evaluation",
                    description="Test RAG system retrieval",
                )
            },
        )
        scenario = config.get_scenario("rag_eval")
        assert scenario.name == "RAG Evaluation"
        assert "RAG system" in scenario.description

    def test_get_scenario_raises_on_unknown(self):
        """Test that get_scenario raises KeyError for unknown scenario."""
        config = CorpusConfig(
            name="Test Corpus",
            corpus_context="Test context",
            scenarios={
                "graduate_exam": EvaluationScenario(
                    name="Graduate Exam",
                    description="Test description",
                )
            },
        )
        with pytest.raises(KeyError) as exc_info:
            config.get_scenario("nonexistent")
        assert "nonexistent" in str(exc_info.value)
        assert "graduate_exam" in str(exc_info.value)  # Shows available scenarios

    def test_multiple_scenarios(self):
        """Test corpus with multiple evaluation scenarios."""
        config = CorpusConfig(
            name="Medical Papers",
            corpus_context="PubMed papers on genetics",
            scenarios={
                "graduate_exam": EvaluationScenario(
                    name="Graduate Genetics Exam",
                    description="PhD qualifying exam questions",
                ),
                "clinical_review": EvaluationScenario(
                    name="Clinical Review",
                    description="Board certification questions",
                ),
                "rag_eval": EvaluationScenario(
                    name="RAG Evaluation",
                    description="Test retrieval accuracy",
                ),
            },
        )
        assert len(config.scenarios) == 3
        assert config.get_scenario("clinical_review").name == "Clinical Review"


class TestLoadCorpusConfig:
    """Tests for load_corpus_config function."""

    def test_loads_from_yaml_file(self, tmp_path):
        """Test loading corpus config from a YAML file."""
        config_file = tmp_path / "corpus.yaml"
        config_file.write_text("""
name: Test Corpus
corpus_context: Test documents for unit testing
scenarios:
  graduate_exam:
    name: Graduate Exam
    description: Questions for graduate students
  rag_eval:
    name: RAG Evaluation
    description: Test retrieval system accuracy
""")
        config = load_corpus_config(config_file)
        assert config.name == "Test Corpus"
        assert "unit testing" in config.corpus_context
        assert len(config.scenarios) == 2

    def test_loads_from_directory(self, tmp_path):
        """Test loading corpus config from directory containing corpus.yaml."""
        config_file = tmp_path / "corpus.yaml"
        config_file.write_text("""
name: Directory Test
corpus_context: Loaded from directory
scenarios:
  test:
    name: Test Scenario
    description: Test description
""")
        # Pass directory instead of file
        config = load_corpus_config(tmp_path)
        assert config.name == "Directory Test"

    def test_raises_on_missing_file(self, tmp_path):
        """Test that FileNotFoundError is raised for missing config."""
        with pytest.raises(FileNotFoundError):
            load_corpus_config(tmp_path / "nonexistent")

    def test_raises_on_invalid_yaml(self, tmp_path):
        """Test that ValidationError is raised for invalid config."""
        config_file = tmp_path / "corpus.yaml"
        config_file.write_text("""
name: Invalid Config
# Missing corpus_context and scenarios
""")
        with pytest.raises(ValidationError):
            load_corpus_config(config_file)
