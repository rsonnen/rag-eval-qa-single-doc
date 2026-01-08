"""Unit tests for question validator."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from single_doc_generator.config import ModelConfig
from single_doc_generator.generator import GeneratedQA
from single_doc_generator.models import GenerationMode, RejectionReason
from single_doc_generator.validator import (
    ValidatorAssessment,
    validate_question,
)


def make_agent_result_with_tool_response(assessment: ValidatorAssessment) -> dict:
    """Create mock agent result with submit_assessment tool response."""
    tool_message = ToolMessage(
        content=assessment.model_dump_json(),
        name="submit_assessment",
        tool_call_id="test-id",
    )
    return {"messages": [tool_message]}


class TestValidatorAssessment:
    """Tests for ValidatorAssessment model."""

    def test_passing_assessment(self):
        """Test creating a passing assessment."""
        assessment = ValidatorAssessment(
            answerable=True,
            my_answer="Paris is the capital",
            answer_matches=True,
            reasoning="The document clearly states this",
        )
        assert assessment.answerable is True
        assert assessment.answer_matches is True
        assert assessment.is_ambiguous is False
        assert assessment.is_trivial is False

    def test_failing_assessment_unanswerable(self):
        """Test creating an unanswerable assessment."""
        assessment = ValidatorAssessment(
            answerable=False,
            my_answer="",
            answer_matches=False,
            reasoning="Could not find relevant information",
        )
        assert assessment.answerable is False

    def test_ambiguous_assessment(self):
        """Test creating an ambiguous assessment."""
        assessment = ValidatorAssessment(
            answerable=True,
            my_answer="Could be A or B",
            answer_matches=False,
            reasoning="Multiple interpretations possible",
            is_ambiguous=True,
        )
        assert assessment.is_ambiguous is True

    def test_trivial_assessment(self):
        """Test creating a trivial assessment."""
        assessment = ValidatorAssessment(
            answerable=True,
            my_answer="Yes",
            answer_matches=True,
            reasoning="Too obvious",
            is_trivial=True,
        )
        assert assessment.is_trivial is True


class TestValidateQuestion:
    """Tests for validate_question function."""

    @pytest.fixture
    def sample_candidate(self):
        """Create a sample Q/A candidate."""
        return GeneratedQA(
            question="Who wrote Hamlet?",
            answer="William Shakespeare",
            reasoning="Tests authorship",
        )

    def test_raises_on_missing_file(self, tmp_path, sample_candidate):
        """Test that FileNotFoundError is raised for missing document."""
        with pytest.raises(FileNotFoundError):
            validate_question(
                document_path=tmp_path / "nonexistent.txt",
                candidate=sample_candidate,
            )

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_passes_valid_question(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_candidate,
    ):
        """Test validation passes for a valid question."""
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="claude-sonnet-4.5", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            ValidatorAssessment(
                answerable=True,
                my_answer="William Shakespeare",
                answer_matches=True,
                reasoning="Found authorship confirmed in document",
            )
        )

        result = validate_question(
            document_path=shakespeare_file,
            candidate=sample_candidate,
        )

        assert result.passed is True
        assert result.validator_answer == "William Shakespeare"
        assert result.rejection_reason is None

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_rejects_unanswerable(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
    ):
        """Test validation rejects unanswerable questions."""
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="claude-sonnet-4.5", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            ValidatorAssessment(
                answerable=False,
                my_answer="",
                answer_matches=False,
                reasoning="Could not find this information in document",
            )
        )

        candidate = GeneratedQA(
            question="What color is the dragon?",
            answer="Blue",
            reasoning="Test question",
        )

        result = validate_question(
            document_path=shakespeare_file,
            candidate=candidate,
        )

        assert result.passed is False
        assert result.rejection_reason == RejectionReason.UNANSWERABLE

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_rejects_wrong_answer(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
    ):
        """Test validation rejects when answers don't match."""
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="claude-sonnet-4.5", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            ValidatorAssessment(
                answerable=True,
                my_answer="Christopher Marlowe",
                answer_matches=False,
                reasoning="Found different author mentioned",
            )
        )

        candidate = GeneratedQA(
            question="Who wrote the play?",
            answer="Shakespeare",
            reasoning="Test",
        )

        result = validate_question(
            document_path=shakespeare_file,
            candidate=candidate,
        )

        assert result.passed is False
        assert result.rejection_reason == RejectionReason.WRONG_ANSWER
        assert result.validator_answer == "Christopher Marlowe"

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_rejects_ambiguous(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
    ):
        """Test validation rejects ambiguous questions."""
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="claude-sonnet-4.5", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            ValidatorAssessment(
                answerable=True,
                my_answer="Could be multiple things",
                answer_matches=False,
                reasoning="Question is unclear",
                is_ambiguous=True,
            )
        )

        candidate = GeneratedQA(
            question="What happened?",
            answer="Something",
            reasoning="Test",
        )

        result = validate_question(
            document_path=shakespeare_file,
            candidate=candidate,
        )

        assert result.passed is False
        assert result.rejection_reason == RejectionReason.AMBIGUOUS

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_rejects_trivial(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
    ):
        """Test validation rejects trivial questions."""
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="claude-sonnet-4.5", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            ValidatorAssessment(
                answerable=True,
                my_answer="Yes",
                answer_matches=True,
                reasoning="This is general knowledge",
                is_trivial=True,
            )
        )

        candidate = GeneratedQA(
            question="Is the sky blue?",
            answer="Yes",
            reasoning="Test",
        )

        result = validate_question(
            document_path=shakespeare_file,
            candidate=candidate,
        )

        assert result.passed is False
        assert result.rejection_reason == RejectionReason.TRIVIAL

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_handles_missing_tool_response(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_candidate,
    ):
        """Test validation handles missing submit_assessment tool response."""
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="claude-sonnet-4.5", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        # No submit_assessment tool message
        mock_agent.invoke.return_value = {"messages": []}

        result = validate_question(
            document_path=shakespeare_file,
            candidate=sample_candidate,
        )

        assert result.passed is False
        assert result.rejection_reason == RejectionReason.VALIDATION_FAILED

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_uses_model_from_config(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_candidate,
    ):
        """Test that validator model is read from app config."""
        mock_config = MagicMock()
        mock_config.validator_model = ModelConfig(name="gemini-3-pro", kwargs={})
        mock_get_config.return_value = mock_config

        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            ValidatorAssessment(
                answerable=True,
                my_answer="Test",
                answer_matches=True,
                reasoning="Test",
            )
        )

        validate_question(
            document_path=shakespeare_file,
            candidate=sample_candidate,
        )

        # Verify create_chat_model was called with model from config
        mock_create_model.assert_called_once_with("gemini-3-pro", {})

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_includes_submit_tool_in_agent(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_candidate,
    ):
        """Test that submit_assessment tool is included in agent tools."""
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="claude-sonnet-4.5", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            ValidatorAssessment(
                answerable=True,
                my_answer="Test",
                answer_matches=True,
                reasoning="Test",
            )
        )

        validate_question(
            document_path=shakespeare_file,
            candidate=sample_candidate,
        )

        # Verify create_react_agent was called with tools including submit_assessment
        call_kwargs = mock_create_agent.call_args.kwargs
        tools = call_kwargs["tools"]
        tool_names = [t.name for t in tools]
        assert "submit_assessment" in tool_names

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_passes_mode_to_prompt(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_candidate,
    ):
        """Test that mode parameter is passed to prompt loader."""
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="claude-sonnet-4.5", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            ValidatorAssessment(
                answerable=True,
                my_answer="Test",
                answer_matches=True,
                reasoning="Test",
            )
        )

        validate_question(
            document_path=shakespeare_file,
            candidate=sample_candidate,
            mode=GenerationMode.VISUAL,
        )

        mock_load_prompt.assert_called_once()
        call_kwargs = mock_load_prompt.call_args.kwargs
        assert call_kwargs["mode"] == "visual"

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_defaults_to_textual_mode(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_candidate,
    ):
        """Test that mode defaults to textual when not specified."""
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="claude-sonnet-4.5", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            ValidatorAssessment(
                answerable=True,
                my_answer="Test",
                answer_matches=True,
                reasoning="Test",
            )
        )

        validate_question(
            document_path=shakespeare_file,
            candidate=sample_candidate,
        )

        mock_load_prompt.assert_called_once()
        call_kwargs = mock_load_prompt.call_args.kwargs
        assert call_kwargs["mode"] == "textual"

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_includes_content_refs_in_visual_mode(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
    ):
        """Test content_refs included in validation request for visual mode."""
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="claude-sonnet-4.5", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            ValidatorAssessment(
                answerable=True,
                my_answer="Test",
                answer_matches=True,
                reasoning="Test",
            )
        )

        candidate = GeneratedQA(
            question="What does Figure 1 show?",
            answer="A diagram of the system",
            reasoning="Tests visual understanding",
            content_refs=["Figure 1", "Table 2"],
        )

        validate_question(
            document_path=shakespeare_file,
            candidate=candidate,
            mode=GenerationMode.VISUAL,
        )

        # Check that the invoke was called with a message containing content refs
        call_args = mock_agent.invoke.call_args
        messages = call_args[0][0]["messages"]
        message_content = messages[0].content
        assert "REFERENCED VISUAL CONTENT: Figure 1, Table 2" in message_content

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_passes_corpus_context_to_prompt(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_candidate,
    ):
        """Test that corpus_context and scenario_description are passed to prompt."""
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="claude-sonnet-4.5", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            ValidatorAssessment(
                answerable=True,
                my_answer="Test",
                answer_matches=True,
                reasoning="Test",
            )
        )

        validate_question(
            document_path=shakespeare_file,
            candidate=sample_candidate,
            corpus_context="Scientific illustration corpus",
            scenario_description="Questions testing specimen identification",
        )

        mock_load_prompt.assert_called_once()
        call_kwargs = mock_load_prompt.call_args.kwargs
        assert call_kwargs["corpus_context"] == "Scientific illustration corpus"
        assert (
            call_kwargs["scenario_description"]
            == "Questions testing specimen identification"
        )

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_corpus_context_defaults_to_empty(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_candidate,
    ):
        """Test that corpus_context defaults to empty string when not provided."""
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="claude-sonnet-4.5", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            ValidatorAssessment(
                answerable=True,
                my_answer="Test",
                answer_matches=True,
                reasoning="Test",
            )
        )

        validate_question(
            document_path=shakespeare_file,
            candidate=sample_candidate,
        )

        mock_load_prompt.assert_called_once()
        call_kwargs = mock_load_prompt.call_args.kwargs
        assert call_kwargs["corpus_context"] == ""
        assert call_kwargs["scenario_description"] == ""

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_passes_model_kwargs_to_create_chat_model(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_candidate,
    ):
        """Test that model kwargs from config are passed to create_chat_model."""
        mock_config = MagicMock()
        mock_config.validator_model = ModelConfig(
            name="gemini-2.5-flash",
            kwargs={"reasoning_effort": "none"},
        )
        mock_get_config.return_value = mock_config

        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            ValidatorAssessment(
                answerable=True,
                my_answer="Test",
                answer_matches=True,
                reasoning="Test",
            )
        )

        validate_question(
            document_path=shakespeare_file,
            candidate=sample_candidate,
        )

        # Verify create_chat_model was called with kwargs from config
        mock_create_model.assert_called_once_with(
            "gemini-2.5-flash", {"reasoning_effort": "none"}
        )


class TestInvokeWithRetry:
    """Tests for _invoke_with_retry helper function."""

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_max_empty_choices_retries_controls_attempts(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
    ):
        """Test that max_empty_choices_retries param controls retry behavior.

        With max_empty_choices_retries=1, we try once and fail immediately
        on empty choices error (no retries). This verifies the parameter
        is correctly passed through to _invoke_with_retry.
        """
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="gemini-2.5-flash", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        # Simulate empty choices error from Gemini
        mock_agent.invoke.side_effect = TypeError("null value for `choices`")

        candidate = GeneratedQA(
            question="Test question?",
            answer="Test answer",
            reasoning="Test reasoning",
        )

        # With max_empty_choices_retries=1, try once and fail immediately
        result = validate_question(
            document_path=shakespeare_file,
            candidate=candidate,
            max_empty_choices_retries=1,
        )

        # Should get a validation failed result after single attempt
        assert result.passed is False
        assert result.rejection_reason == RejectionReason.VALIDATION_FAILED
        # With 1 attempt allowed, invoke should be called exactly once
        assert mock_agent.invoke.call_count == 1

    @patch("single_doc_generator.validator.create_agent")
    @patch("single_doc_generator.validator.create_chat_model")
    @patch("single_doc_generator.validator.get_config")
    @patch("single_doc_generator.validator.load_prompt")
    def test_retries_on_empty_choices_error(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
    ):
        """Test that empty choices errors trigger retries up to max."""
        mock_get_config.return_value = MagicMock(
            validator_model=ModelConfig(name="gemini-2.5-flash", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        # Fail twice, succeed third time
        mock_agent.invoke.side_effect = [
            TypeError("null value for `choices`"),
            TypeError("null value for `choices`"),
            make_agent_result_with_tool_response(
                ValidatorAssessment(
                    answerable=True,
                    my_answer="Test",
                    answer_matches=True,
                    reasoning="Success",
                )
            ),
        ]

        candidate = GeneratedQA(
            question="Test question?",
            answer="Test answer",
            reasoning="Test reasoning",
        )

        result = validate_question(
            document_path=shakespeare_file,
            candidate=candidate,
            max_empty_choices_retries=3,
        )

        assert result.passed is True
        assert mock_agent.invoke.call_count == 3
