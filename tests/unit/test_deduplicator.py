"""Unit tests for question deduplicator."""

from unittest.mock import MagicMock, patch

import pytest

from single_doc_generator.config import ModelConfig
from single_doc_generator.deduplicator import (
    DeduplicationResult,
    deduplicate_question,
)


class TestDeduplicationResult:
    """Tests for DeduplicationResult model."""

    def test_unique_result(self):
        """Test creating a unique (not duplicate) result."""
        result = DeduplicationResult(
            is_duplicate=False,
            duplicate_of=None,
            reasoning="Questions ask about different things",
        )
        assert result.is_duplicate is False
        assert result.duplicate_of is None
        assert "different" in result.reasoning

    def test_duplicate_result(self):
        """Test creating a duplicate result."""
        result = DeduplicationResult(
            is_duplicate=True,
            duplicate_of="Who wrote Hamlet?",
            reasoning="Both ask about the author of Hamlet",
        )
        assert result.is_duplicate is True
        assert result.duplicate_of == "Who wrote Hamlet?"
        assert "Hamlet" in result.reasoning

    def test_duplicate_of_defaults_to_none(self):
        """Test that duplicate_of defaults to None."""
        result = DeduplicationResult(
            is_duplicate=False,
            reasoning="Unique question",
        )
        assert result.duplicate_of is None


class TestDeduplicateQuestion:
    """Tests for deduplicate_question function."""

    def test_empty_list_always_unique(self):
        """First question is always unique when no existing questions."""
        result = deduplicate_question(
            question="What is the capital of France?",
            existing_questions=[],
        )

        assert result.is_duplicate is False
        assert result.duplicate_of is None
        assert "First question" in result.reasoning

    def test_empty_list_no_llm_call(self):
        """Empty list should not make any LLM calls."""
        with patch("single_doc_generator.deduplicator.create_chat_model") as mock:
            deduplicate_question(
                question="Any question",
                existing_questions=[],
            )
            mock.assert_not_called()

    @patch("single_doc_generator.deduplicator.create_chat_model")
    @patch("single_doc_generator.deduplicator.get_config")
    @patch("single_doc_generator.deduplicator.load_prompt")
    def test_detects_exact_duplicate(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
    ):
        """Test detection of exact same question."""
        mock_get_config.return_value = MagicMock(
            deduplicator_model=ModelConfig(name="gpt-5-mini", kwargs={})
        )
        mock_load_prompt.return_value = "Test prompt"

        mock_llm = MagicMock()
        mock_create_model.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            DeduplicationResult(
                is_duplicate=True,
                duplicate_of="Who wrote Hamlet?",
                reasoning="Identical question",
            )
        )

        result = deduplicate_question(
            question="Who wrote Hamlet?",
            existing_questions=["Who wrote Hamlet?", "When was it published?"],
        )

        assert result.is_duplicate is True
        assert result.duplicate_of == "Who wrote Hamlet?"

    @patch("single_doc_generator.deduplicator.create_chat_model")
    @patch("single_doc_generator.deduplicator.get_config")
    @patch("single_doc_generator.deduplicator.load_prompt")
    def test_detects_semantic_duplicate(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
    ):
        """Test detection of semantically equivalent questions."""
        mock_get_config.return_value = MagicMock(
            deduplicator_model=ModelConfig(name="gpt-5-mini", kwargs={})
        )
        mock_load_prompt.return_value = "Test prompt"

        mock_llm = MagicMock()
        mock_create_model.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            DeduplicationResult(
                is_duplicate=True,
                duplicate_of="Who wrote Hamlet?",
                reasoning="Both ask about the playwright of Hamlet",
            )
        )

        result = deduplicate_question(
            question="Which playwright authored Hamlet?",
            existing_questions=["Who wrote Hamlet?"],
        )

        assert result.is_duplicate is True
        assert result.duplicate_of == "Who wrote Hamlet?"

    @patch("single_doc_generator.deduplicator.create_chat_model")
    @patch("single_doc_generator.deduplicator.get_config")
    @patch("single_doc_generator.deduplicator.load_prompt")
    def test_passes_distinct_question(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
    ):
        """Test that distinct questions are not marked as duplicates."""
        mock_get_config.return_value = MagicMock(
            deduplicator_model=ModelConfig(name="gpt-5-mini", kwargs={})
        )
        mock_load_prompt.return_value = "Test prompt"

        mock_llm = MagicMock()
        mock_create_model.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            DeduplicationResult(
                is_duplicate=False,
                duplicate_of=None,
                reasoning="Questions ask about different facts",
            )
        )

        result = deduplicate_question(
            question="When was Hamlet written?",
            existing_questions=["Who wrote Hamlet?"],
        )

        assert result.is_duplicate is False
        assert result.duplicate_of is None

    @patch("single_doc_generator.deduplicator.create_chat_model")
    @patch("single_doc_generator.deduplicator.get_config")
    @patch("single_doc_generator.deduplicator.load_prompt")
    def test_uses_model_from_config(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
    ):
        """Test that deduplicator model is read from app config."""
        mock_config = MagicMock()
        mock_config.deduplicator_model = ModelConfig(name="claude-haiku", kwargs={})
        mock_get_config.return_value = mock_config
        mock_load_prompt.return_value = "Test prompt"

        mock_llm = MagicMock()
        mock_create_model.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            DeduplicationResult(
                is_duplicate=False,
                duplicate_of=None,
                reasoning="Unique",
            )
        )

        deduplicate_question(
            question="Any question?",
            existing_questions=["Previous question?"],
        )

        mock_create_model.assert_called_once_with("claude-haiku", {})

    @patch("single_doc_generator.deduplicator.create_chat_model")
    @patch("single_doc_generator.deduplicator.get_config")
    @patch("single_doc_generator.deduplicator.load_prompt")
    def test_passes_model_kwargs_to_create_chat_model(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
    ):
        """Test that model kwargs from config are passed to create_chat_model."""
        mock_config = MagicMock()
        mock_config.deduplicator_model = ModelConfig(
            name="gemini-2.5-flash",
            kwargs={"reasoning_effort": "none"},
        )
        mock_get_config.return_value = mock_config
        mock_load_prompt.return_value = "Test prompt"

        mock_llm = MagicMock()
        mock_create_model.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            DeduplicationResult(
                is_duplicate=False,
                duplicate_of=None,
                reasoning="Unique",
            )
        )

        deduplicate_question(
            question="Any question?",
            existing_questions=["Previous question?"],
        )

        mock_create_model.assert_called_once_with(
            "gemini-2.5-flash", {"reasoning_effort": "none"}
        )

    @patch("single_doc_generator.deduplicator.create_chat_model")
    @patch("single_doc_generator.deduplicator.get_config")
    @patch("single_doc_generator.deduplicator.load_prompt")
    def test_uses_with_structured_output(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
    ):
        """Test that LLM is called with structured output."""
        mock_get_config.return_value = MagicMock(
            deduplicator_model=ModelConfig(name="gpt-5-mini", kwargs={})
        )
        mock_load_prompt.return_value = "Test prompt"

        mock_llm = MagicMock()
        mock_create_model.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = DeduplicationResult(
            is_duplicate=False,
            duplicate_of=None,
            reasoning="Unique",
        )

        deduplicate_question(
            question="Any question?",
            existing_questions=["Previous?"],
        )

        mock_llm.with_structured_output.assert_called_once_with(DeduplicationResult)

    @patch("single_doc_generator.deduplicator.create_chat_model")
    @patch("single_doc_generator.deduplicator.get_config")
    @patch("single_doc_generator.deduplicator.load_prompt")
    def test_handles_many_existing_questions(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
    ):
        """Test with 50+ existing questions to verify prompt handles scale."""
        mock_get_config.return_value = MagicMock(
            deduplicator_model=ModelConfig(name="gpt-5-mini", kwargs={})
        )
        mock_load_prompt.return_value = "Test prompt"

        mock_llm = MagicMock()
        mock_create_model.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            DeduplicationResult(
                is_duplicate=False,
                duplicate_of=None,
                reasoning="Unique among all 50",
            )
        )

        existing = [f"Question number {i}?" for i in range(50)]

        result = deduplicate_question(
            question="A completely different question?",
            existing_questions=existing,
        )

        assert result.is_duplicate is False

        # Verify prompt loader was called with all 50 questions
        mock_load_prompt.assert_called_once()
        call_kwargs = mock_load_prompt.call_args.kwargs
        assert len(call_kwargs["existing_questions"]) == 50

    @patch("single_doc_generator.deduplicator.create_chat_model")
    @patch("single_doc_generator.deduplicator.get_config")
    @patch("single_doc_generator.deduplicator.load_prompt")
    def test_passes_questions_to_prompt(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
    ):
        """Test that questions are correctly passed to prompt loader."""
        mock_get_config.return_value = MagicMock(
            deduplicator_model=ModelConfig(name="gpt-5-mini", kwargs={})
        )
        mock_load_prompt.return_value = "Test prompt"

        mock_llm = MagicMock()
        mock_create_model.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            DeduplicationResult(
                is_duplicate=False,
                duplicate_of=None,
                reasoning="Unique",
            )
        )

        deduplicate_question(
            question="New question here?",
            existing_questions=["First existing?", "Second existing?"],
        )

        mock_load_prompt.assert_called_once_with(
            "deduplicator",
            new_question="New question here?",
            existing_questions=["First existing?", "Second existing?"],
        )

    @patch("single_doc_generator.deduplicator.create_chat_model")
    @patch("single_doc_generator.deduplicator.get_config")
    @patch("single_doc_generator.deduplicator.load_prompt")
    def test_raises_on_unexpected_type(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
    ):
        """Test that TypeError is raised if LLM returns unexpected type."""
        mock_get_config.return_value = MagicMock(
            deduplicator_model=ModelConfig(name="gpt-5-mini", kwargs={})
        )
        mock_load_prompt.return_value = "Test prompt"

        mock_llm = MagicMock()
        mock_create_model.return_value = mock_llm
        # Return wrong type
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            "not a DeduplicationResult"
        )

        with pytest.raises(TypeError, match="Expected DeduplicationResult"):
            deduplicate_question(
                question="Any question?",
                existing_questions=["Existing?"],
            )
