"""Tests for the orchestrator module.

These tests mock the generator, deduplicator, and validator to test
orchestration logic in isolation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from single_doc_generator.config import CorpusConfig, EvaluationScenario
from single_doc_generator.deduplicator import DeduplicationResult
from single_doc_generator.generator import DocumentExhaustedError, GenerationError
from single_doc_generator.models import (
    GenerationMode,
    QAPair,
    RejectionReason,
    ValidationResult,
)
from single_doc_generator.orchestrator import run_generation


class MockGeneratedQA:
    """Mock for GeneratedQA from generator."""

    def __init__(
        self,
        question: str = "Test question?",
        answer: str = "Test answer",
        reasoning: str = "Test reasoning",
        content_refs: list[str] | None = None,
    ):
        self.question = question
        self.answer = answer
        self.reasoning = reasoning
        self.content_refs = content_refs or []


@pytest.fixture
def corpus_config() -> CorpusConfig:
    """Create a test corpus config."""
    return CorpusConfig(
        name="test_corpus",
        corpus_context="A test corpus",
        scenarios={
            "test_scenario": EvaluationScenario(
                name="Test Scenario",
                description="A test scenario",
            )
        },
    )


@pytest.fixture
def document_path(tmp_path: Path) -> Path:
    """Create a test document."""
    doc = tmp_path / "test.txt"
    doc.write_text("Test content")
    return doc


@pytest.fixture
def corpus_path(tmp_path: Path) -> Path:
    """Return corpus path."""
    return tmp_path


class TestConsecutiveFailureTracking:
    """Tests for consecutive failure tracking and exhaustion."""

    @patch("single_doc_generator.orchestrator.validate_question")
    @patch("single_doc_generator.orchestrator.deduplicate_question")
    @patch("single_doc_generator.orchestrator.generate_question")
    def test_stops_on_consecutive_failures(
        self,
        mock_generate: MagicMock,
        mock_dedupe: MagicMock,
        mock_validate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """Stops when consecutive failures exceed threshold."""
        mock_generate.return_value = (MockGeneratedQA(), [1])
        mock_dedupe.return_value = DeduplicationResult(
            is_duplicate=False,
            reasoning="Unique",
        )
        mock_validate.return_value = ValidationResult(
            passed=False,
            reasoning="Failed validation",
            rejection_reason=RejectionReason.VALIDATION_FAILED,
        )

        result = run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=10,
            max_consecutive_failures=3,
        )

        assert result.stats.exhausted is True
        assert result.stats.exhaustion_reason == "consecutive_failures"
        assert mock_generate.call_count == 3
        assert result.stats.accepted_count == 0
        assert result.stats.rejected_count == 3

    @patch("single_doc_generator.orchestrator.validate_question")
    @patch("single_doc_generator.orchestrator.deduplicate_question")
    @patch("single_doc_generator.orchestrator.generate_question")
    def test_resets_failures_on_success(
        self,
        mock_generate: MagicMock,
        mock_dedupe: MagicMock,
        mock_validate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """Consecutive failure counter resets on acceptance."""
        call_count = [0]

        def generate_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            return (MockGeneratedQA(question=f"Q{call_count[0]}?"), [1])

        mock_generate.side_effect = generate_side_effect
        mock_dedupe.return_value = DeduplicationResult(
            is_duplicate=False,
            reasoning="Unique",
        )

        # Fail twice, pass, fail twice, pass
        validate_results = [
            ValidationResult(
                passed=False,
                reasoning="fail",
                rejection_reason=RejectionReason.WRONG_ANSWER,
            ),
            ValidationResult(
                passed=False,
                reasoning="fail",
                rejection_reason=RejectionReason.WRONG_ANSWER,
            ),
            ValidationResult(passed=True, reasoning="pass"),
            ValidationResult(
                passed=False,
                reasoning="fail",
                rejection_reason=RejectionReason.WRONG_ANSWER,
            ),
            ValidationResult(
                passed=False,
                reasoning="fail",
                rejection_reason=RejectionReason.WRONG_ANSWER,
            ),
            ValidationResult(passed=True, reasoning="pass"),
        ]
        mock_validate.side_effect = validate_results

        result = run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=2,
            max_consecutive_failures=3,
        )

        assert result.stats.exhausted is False
        assert result.stats.accepted_count == 2
        assert result.stats.rejected_count == 4
        assert mock_generate.call_count == 6

    @patch("single_doc_generator.orchestrator.validate_question")
    @patch("single_doc_generator.orchestrator.deduplicate_question")
    @patch("single_doc_generator.orchestrator.generate_question")
    def test_dedup_failures_count_toward_exhaustion(
        self,
        mock_generate: MagicMock,
        mock_dedupe: MagicMock,
        mock_validate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """Duplicate rejections count toward consecutive failures."""
        mock_generate.return_value = (MockGeneratedQA(), [1])
        mock_dedupe.return_value = DeduplicationResult(
            is_duplicate=True,
            reasoning="Duplicate",
            duplicate_of="Previous Q",
        )

        result = run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=5,
            max_consecutive_failures=3,
        )

        assert result.stats.exhausted is True
        assert mock_generate.call_count == 3
        assert mock_validate.call_count == 0  # Never reached validation
        assert result.stats.rejection_breakdown["duplicate"] == 3


class TestExhaustionDetection:
    """Tests for document exhaustion detection."""

    @patch("single_doc_generator.orchestrator.generate_question")
    def test_stops_on_document_exhausted_error(
        self,
        mock_generate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """Stops immediately on DocumentExhaustedError."""
        mock_generate.side_effect = DocumentExhaustedError(
            reason="No more content",
            viewed_pages=[1, 2],
        )

        result = run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=10,
        )

        assert result.stats.exhausted is True
        assert "document_exhausted" in result.stats.exhaustion_reason
        assert mock_generate.call_count == 1

    @patch("single_doc_generator.orchestrator.generate_question")
    def test_generation_errors_count_toward_exhaustion(
        self,
        mock_generate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """GenerationError exceptions count toward consecutive failures."""
        mock_generate.side_effect = GenerationError(
            message="Agent failed",
            viewed_pages=[1],
        )

        result = run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=5,
            max_consecutive_failures=2,
        )

        assert result.stats.exhausted is True
        assert mock_generate.call_count == 2


class TestTargetReached:
    """Tests for stopping when target count is reached."""

    @patch("single_doc_generator.orchestrator.validate_question")
    @patch("single_doc_generator.orchestrator.deduplicate_question")
    @patch("single_doc_generator.orchestrator.generate_question")
    def test_stops_at_target_count(
        self,
        mock_generate: MagicMock,
        mock_dedupe: MagicMock,
        mock_validate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """Stops exactly when target count is reached."""
        call_count = [0]

        def generate_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            return (MockGeneratedQA(question=f"Q{call_count[0]}?"), [1])

        mock_generate.side_effect = generate_side_effect
        mock_dedupe.return_value = DeduplicationResult(
            is_duplicate=False,
            reasoning="Unique",
        )
        mock_validate.return_value = ValidationResult(
            passed=True,
            reasoning="Valid",
        )

        result = run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=3,
        )

        assert result.stats.exhausted is False
        assert result.stats.accepted_count == 3
        assert mock_generate.call_count == 3
        assert len(result.accepted) == 3


class TestStatistics:
    """Tests for statistics tracking."""

    @patch("single_doc_generator.orchestrator.validate_question")
    @patch("single_doc_generator.orchestrator.deduplicate_question")
    @patch("single_doc_generator.orchestrator.generate_question")
    def test_rejection_breakdown_accuracy(
        self,
        mock_generate: MagicMock,
        mock_dedupe: MagicMock,
        mock_validate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """Rejection breakdown correctly counts each reason."""
        call_count = [0]

        def generate_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            return (MockGeneratedQA(question=f"Q{call_count[0]}?"), [1])

        mock_generate.side_effect = generate_side_effect

        dedupe_count = [0]

        def dedupe_side_effect(_question, _existing):
            dedupe_count[0] += 1
            if dedupe_count[0] <= 2:
                return DeduplicationResult(
                    is_duplicate=True,
                    reasoning="Dup",
                    duplicate_of="Prev",
                )
            return DeduplicationResult(is_duplicate=False, reasoning="Unique")

        mock_dedupe.side_effect = dedupe_side_effect

        validate_count = [0]

        def validate_side_effect(*_args, **_kwargs):
            validate_count[0] += 1
            if validate_count[0] == 1:
                return ValidationResult(
                    passed=False,
                    reasoning="Not in doc",
                    rejection_reason=RejectionReason.UNANSWERABLE,
                )
            if validate_count[0] == 2:
                return ValidationResult(
                    passed=False,
                    reasoning="Wrong",
                    rejection_reason=RejectionReason.WRONG_ANSWER,
                )
            return ValidationResult(passed=True, reasoning="Good")

        mock_validate.side_effect = validate_side_effect

        result = run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=1,
            max_consecutive_failures=10,
        )

        assert result.stats.accepted_count == 1
        assert result.stats.rejected_count == 4
        assert result.stats.rejection_breakdown["duplicate"] == 2
        assert result.stats.rejection_breakdown["unanswerable"] == 1
        assert result.stats.rejection_breakdown["wrong_answer"] == 1


class TestCallbacks:
    """Tests for progress callbacks."""

    @patch("single_doc_generator.orchestrator.validate_question")
    @patch("single_doc_generator.orchestrator.deduplicate_question")
    @patch("single_doc_generator.orchestrator.generate_question")
    def test_callbacks_invoked(
        self,
        mock_generate: MagicMock,
        mock_dedupe: MagicMock,
        mock_validate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """Callbacks are invoked at appropriate times."""
        mock_generate.return_value = (MockGeneratedQA(), [1])
        mock_dedupe.return_value = DeduplicationResult(
            is_duplicate=False,
            reasoning="Unique",
        )
        mock_validate.return_value = ValidationResult(passed=True, reasoning="Good")

        on_attempt = MagicMock()
        on_accepted = MagicMock()
        on_rejected = MagicMock()
        on_exhausted = MagicMock()

        run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=2,
            on_attempt=on_attempt,
            on_accepted=on_accepted,
            on_rejected=on_rejected,
            on_exhausted=on_exhausted,
        )

        assert on_attempt.call_count == 2
        assert on_accepted.call_count == 2
        assert on_rejected.call_count == 0
        assert on_exhausted.call_count == 0

    @patch("single_doc_generator.orchestrator.validate_question")
    @patch("single_doc_generator.orchestrator.deduplicate_question")
    @patch("single_doc_generator.orchestrator.generate_question")
    def test_exhaustion_callback(
        self,
        mock_generate: MagicMock,
        mock_dedupe: MagicMock,
        mock_validate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """Exhaustion callback invoked with reason."""
        mock_generate.return_value = (MockGeneratedQA(), [1])
        mock_dedupe.return_value = DeduplicationResult(
            is_duplicate=False,
            reasoning="Unique",
        )
        mock_validate.return_value = ValidationResult(
            passed=False,
            reasoning="Fail",
            rejection_reason=RejectionReason.VALIDATION_FAILED,
        )

        on_exhausted = MagicMock()

        run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=10,
            max_consecutive_failures=2,
            on_exhausted=on_exhausted,
        )

        on_exhausted.assert_called_once_with("consecutive_failures")

    @patch("single_doc_generator.orchestrator.validate_question")
    @patch("single_doc_generator.orchestrator.deduplicate_question")
    @patch("single_doc_generator.orchestrator.generate_question")
    def test_on_rejected_callback_invoked_with_correct_args(
        self,
        mock_generate: MagicMock,
        mock_dedupe: MagicMock,
        mock_validate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """on_rejected callback receives RejectedQA with correct fields."""
        mock_generate.return_value = (
            MockGeneratedQA(question="Test Q?", answer="Test A"),
            [1],
        )
        mock_dedupe.return_value = DeduplicationResult(
            is_duplicate=False,
            reasoning="Unique",
        )
        mock_validate.return_value = ValidationResult(
            passed=False,
            reasoning="Answer is wrong",
            rejection_reason=RejectionReason.WRONG_ANSWER,
        )

        on_rejected = MagicMock()

        run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=10,
            max_consecutive_failures=1,
            on_rejected=on_rejected,
        )

        assert on_rejected.call_count == 1
        rejected_qa = on_rejected.call_args[0][0]
        assert rejected_qa.question == "Test Q?"
        assert rejected_qa.answer == "Test A"
        assert rejected_qa.rejection_reason == RejectionReason.WRONG_ANSWER
        assert rejected_qa.rejection_detail == "Answer is wrong"

    @patch("single_doc_generator.orchestrator.generate_question")
    def test_on_exhausted_callback_on_document_exhausted(
        self,
        mock_generate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """on_exhausted callback invoked when DocumentExhaustedError raised."""
        mock_generate.side_effect = DocumentExhaustedError(
            reason="All pages viewed",
            viewed_pages=[1, 2, 3],
        )

        on_exhausted = MagicMock()

        result = run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=10,
            on_exhausted=on_exhausted,
        )

        on_exhausted.assert_called_once()
        reason = on_exhausted.call_args[0][0]
        assert "document_exhausted" in reason
        assert "All pages viewed" in reason
        assert result.stats.exhausted is True

    @patch("single_doc_generator.orchestrator.generate_question")
    def test_unexpected_exception_counts_toward_exhaustion(
        self,
        mock_generate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """Unexpected exceptions are caught and count toward consecutive failures."""
        mock_generate.side_effect = RuntimeError("Unexpected LLM failure")

        on_rejected = MagicMock()

        result = run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=5,
            max_consecutive_failures=2,
            on_rejected=on_rejected,
        )

        assert result.stats.exhausted is True
        assert result.stats.exhaustion_reason == "consecutive_failures"
        assert mock_generate.call_count == 2
        assert on_rejected.call_count == 2
        # Verify the rejected QA has the error info
        rejected_qa = on_rejected.call_args[0][0]
        assert rejected_qa.question == "[unexpected_error]"
        assert "Unexpected LLM failure" in rejected_qa.rejection_detail


class TestResumeWithinDocument:
    """Tests for resuming generation within a document using existing_accepted."""

    @patch("single_doc_generator.orchestrator.validate_question")
    @patch("single_doc_generator.orchestrator.deduplicate_question")
    @patch("single_doc_generator.orchestrator.generate_question")
    def test_existing_questions_count_toward_target(
        self,
        mock_generate: MagicMock,
        mock_dedupe: MagicMock,
        mock_validate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """Existing questions reduce the number of new questions needed."""
        mock_generate.return_value = (MockGeneratedQA(question="New Q?"), [1])
        mock_dedupe.return_value = DeduplicationResult(
            is_duplicate=False,
            reasoning="Unique",
        )
        mock_validate.return_value = ValidationResult(passed=True, reasoning="Good")

        existing = [
            QAPair(
                question="Existing Q1?",
                answer="A1",
                source_document=str(document_path),
                mode=GenerationMode.TEXTUAL,
            ),
            QAPair(
                question="Existing Q2?",
                answer="A2",
                source_document=str(document_path),
                mode=GenerationMode.TEXTUAL,
            ),
        ]

        result = run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=3,
            existing_accepted=existing,
        )

        # Only 1 new question needed (3 target - 2 existing = 1)
        assert mock_generate.call_count == 1
        # Result contains all 3 questions
        assert len(result.accepted) == 3
        assert result.stats.accepted_count == 3

    @patch("single_doc_generator.orchestrator.validate_question")
    @patch("single_doc_generator.orchestrator.deduplicate_question")
    @patch("single_doc_generator.orchestrator.generate_question")
    def test_existing_questions_used_for_deduplication(
        self,
        mock_generate: MagicMock,
        mock_dedupe: MagicMock,
        mock_validate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """Existing questions are passed to deduplicator for comparison."""
        mock_generate.return_value = (MockGeneratedQA(question="New Q?"), [1])
        mock_dedupe.return_value = DeduplicationResult(
            is_duplicate=False,
            reasoning="Unique",
        )
        mock_validate.return_value = ValidationResult(passed=True, reasoning="Good")

        existing = [
            QAPair(
                question="Existing Q1?",
                answer="A1",
                source_document=str(document_path),
                mode=GenerationMode.TEXTUAL,
            ),
        ]

        run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=2,
            existing_accepted=existing,
        )

        # Deduplicator should receive the existing question text
        dedupe_call = mock_dedupe.call_args
        existing_questions_arg = dedupe_call[0][1]  # Second positional arg
        assert "Existing Q1?" in existing_questions_arg

    @patch("single_doc_generator.orchestrator.validate_question")
    @patch("single_doc_generator.orchestrator.deduplicate_question")
    @patch("single_doc_generator.orchestrator.generate_question")
    def test_on_accepted_only_called_for_new_questions(
        self,
        mock_generate: MagicMock,
        mock_dedupe: MagicMock,
        mock_validate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """on_accepted callback is NOT called for existing questions."""
        call_count = [0]

        def generate_side_effect(*_args, **_kwargs):
            call_count[0] += 1
            return (MockGeneratedQA(question=f"New Q{call_count[0]}?"), [1])

        mock_generate.side_effect = generate_side_effect
        mock_dedupe.return_value = DeduplicationResult(
            is_duplicate=False,
            reasoning="Unique",
        )
        mock_validate.return_value = ValidationResult(passed=True, reasoning="Good")

        existing = [
            QAPair(
                question="Existing Q1?",
                answer="A1",
                source_document=str(document_path),
                mode=GenerationMode.TEXTUAL,
            ),
            QAPair(
                question="Existing Q2?",
                answer="A2",
                source_document=str(document_path),
                mode=GenerationMode.TEXTUAL,
            ),
        ]

        on_accepted = MagicMock()

        result = run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=4,
            existing_accepted=existing,
            on_accepted=on_accepted,
        )

        # 2 new questions generated (4 target - 2 existing = 2)
        assert mock_generate.call_count == 2
        # on_accepted only called for NEW questions
        assert on_accepted.call_count == 2
        # But result contains all 4
        assert len(result.accepted) == 4

        # Verify the callback received the new questions, not existing
        callback_questions = [
            call[0][0].question for call in on_accepted.call_args_list
        ]
        assert "New Q1?" in callback_questions
        assert "New Q2?" in callback_questions
        assert "Existing Q1?" not in callback_questions
        assert "Existing Q2?" not in callback_questions

    @patch("single_doc_generator.orchestrator.validate_question")
    @patch("single_doc_generator.orchestrator.deduplicate_question")
    @patch("single_doc_generator.orchestrator.generate_question")
    def test_no_generation_when_target_already_met(
        self,
        mock_generate: MagicMock,
        mock_dedupe: MagicMock,
        mock_validate: MagicMock,
        corpus_config: CorpusConfig,
        document_path: Path,
        corpus_path: Path,
    ):
        """No generation attempts if existing questions already meet target."""
        existing = [
            QAPair(
                question=f"Existing Q{i}?",
                answer=f"A{i}",
                source_document=str(document_path),
                mode=GenerationMode.TEXTUAL,
            )
            for i in range(5)
        ]

        result = run_generation(
            document_path=document_path,
            corpus_path=corpus_path,
            corpus_config=corpus_config,
            scenario_name="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=5,
            existing_accepted=existing,
        )

        # No generation needed - target already met
        assert mock_generate.call_count == 0
        assert mock_dedupe.call_count == 0
        assert mock_validate.call_count == 0
        # Result contains all existing questions
        assert len(result.accepted) == 5
        assert result.stats.accepted_count == 5
        assert result.stats.total_attempts == 0
