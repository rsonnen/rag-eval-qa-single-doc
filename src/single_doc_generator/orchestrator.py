"""Question generation orchestrator.

Coordinates the generate -> deduplicate -> validate pipeline with proper
consecutive failure tracking for exhaustion detection.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from single_doc_generator.config import CorpusConfig, EvaluationScenario
from single_doc_generator.deduplicator import deduplicate_question
from single_doc_generator.generator import (
    DocumentExhaustedError,
    GeneratedQA,
    GenerationError,
    generate_question,
)
from single_doc_generator.models import (
    GenerationMode,
    GenerationResult,
    GenerationStats,
    QAPair,
    RejectedQA,
    RejectionReason,
)
from single_doc_generator.validator import validate_question

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONSECUTIVE_FAILURES = 5

# Callback type aliases for progress reporting
OnAttemptCallback = Callable[[int, int, int], None]  # (accepted, target, attempt)
OnPhaseCallback = Callable[[str, int, int], None]  # (phase, attempt, target)
OnAcceptedCallback = Callable[[QAPair], None]
OnRejectedCallback = Callable[[RejectedQA], None]
OnExhaustedCallback = Callable[[str], None]  # (reason)


@dataclass
class _GenerationState:
    """Mutable state for a generation run."""

    max_consecutive_failures: int
    on_rejected: OnRejectedCallback | None = None
    on_exhausted: OnExhaustedCallback | None = None

    accepted: list[QAPair] = field(default_factory=list)
    rejected: list[RejectedQA] = field(default_factory=list)
    rejection_breakdown: dict[str, int] = field(default_factory=dict)
    previous_questions: list[str] = field(default_factory=list)
    previous_viewed_pages: list[int] = field(default_factory=list)

    total_attempts: int = 0
    consecutive_failures: int = 0
    exhausted: bool = False
    exhaustion_reason: str | None = None

    def record_rejection(self, rejected_qa: RejectedQA) -> bool:
        """Record a rejection and check for exhaustion.

        Returns True if exhausted (caller should break loop).
        """
        self.rejected.append(rejected_qa)
        reason_key = rejected_qa.rejection_reason.value
        self.rejection_breakdown[reason_key] = (
            self.rejection_breakdown.get(reason_key, 0) + 1
        )

        if self.on_rejected:
            self.on_rejected(rejected_qa)

        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.exhausted = True
            self.exhaustion_reason = "consecutive_failures"
            logger.info("Exhausted: %d consecutive failures", self.consecutive_failures)
            if self.on_exhausted:
                self.on_exhausted(self.exhaustion_reason)
            return True

        return False

    def record_viewed_pages(self, pages: list[int]) -> None:
        """Add pages to viewed list if not already present."""
        for page in pages:
            if page not in self.previous_viewed_pages:
                self.previous_viewed_pages.append(page)

    def set_exhausted(self, reason: str) -> None:
        """Mark generation as exhausted."""
        self.exhausted = True
        self.exhaustion_reason = reason
        if self.on_exhausted:
            self.on_exhausted(reason)


def run_generation(
    document_path: Path,
    corpus_path: Path,
    corpus_config: CorpusConfig,
    scenario_name: str,
    mode: GenerationMode,
    target_count: int,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    existing_accepted: list[QAPair] | None = None,
    on_attempt: OnAttemptCallback | None = None,
    on_phase: OnPhaseCallback | None = None,
    on_accepted: OnAcceptedCallback | None = None,
    on_rejected: OnRejectedCallback | None = None,
    on_exhausted: OnExhaustedCallback | None = None,
) -> GenerationResult:
    """Run the question generation pipeline for a single document.

    Generates questions until either the target count is reached or the
    document is exhausted (either explicitly or via consecutive failures).

    Supports resume within a document: if existing_accepted is provided,
    those questions are used for deduplication and the loop continues
    from where it left off. The on_accepted callback is only called for
    NEW questions (not existing ones).

    Args:
        document_path: Path to the document file.
        corpus_path: Path to the corpus directory.
        corpus_config: Loaded corpus configuration.
        scenario_name: Name of the scenario to use.
        mode: Generation mode (textual or visual).
        target_count: Number of questions to generate.
        max_consecutive_failures: Stop after this many consecutive failures.
        existing_accepted: Previously accepted questions for this document
            (for resume - used for deduplication, not re-reported via callback).
        on_attempt: Called at start of each attempt with (accepted, target, attempt).
        on_phase: Called before each phase with (phase_name, attempt, target).
        on_accepted: Called when a NEW question is accepted.
        on_rejected: Called when a question is rejected.
        on_exhausted: Called when document is exhausted with the reason.

    Returns:
        GenerationResult with all accepted/rejected questions and statistics.
    """
    scenario = corpus_config.get_scenario(scenario_name)

    # Initialize state, optionally seeding with existing questions for resume
    state = _GenerationState(
        max_consecutive_failures=max_consecutive_failures,
        on_rejected=on_rejected,
        on_exhausted=on_exhausted,
    )
    if existing_accepted:
        state.accepted = list(existing_accepted)
        state.previous_questions = [qa.question for qa in existing_accepted]

    starting_count = len(state.accepted)
    logger.info(
        "Starting generation: document=%s, target=%d, existing=%d, max_failures=%d",
        document_path.name,
        target_count,
        starting_count,
        max_consecutive_failures,
    )

    while len(state.accepted) < target_count:
        state.total_attempts += 1

        if on_attempt:
            on_attempt(len(state.accepted), target_count, state.total_attempts)

        should_break = _process_one_attempt(
            state,
            document_path,
            corpus_config,
            scenario_name,
            scenario,
            mode,
            target_count,
            on_phase,
            on_accepted,
        )
        if should_break:
            break

    stats = GenerationStats(
        total_attempts=state.total_attempts,
        accepted_count=len(state.accepted),
        rejected_count=len(state.rejected),
        rejection_breakdown=state.rejection_breakdown,
        exhausted=state.exhausted,
        exhaustion_reason=state.exhaustion_reason,
    )

    logger.info(
        "Generation complete: accepted=%d, rejected=%d, exhausted=%s",
        stats.accepted_count,
        stats.rejected_count,
        stats.exhausted,
    )

    return GenerationResult(
        document=str(document_path),
        corpus=str(corpus_path),
        scenario=scenario_name,
        mode=mode,
        target_count=target_count,
        accepted=state.accepted,
        rejected=state.rejected,
        stats=stats,
        timestamp="",  # Caller sets this
    )


def _process_one_attempt(
    state: _GenerationState,
    document_path: Path,
    corpus_config: CorpusConfig,
    scenario_name: str,
    scenario: EvaluationScenario,
    mode: GenerationMode,
    target_count: int,
    on_phase: OnPhaseCallback | None,
    on_accepted: OnAcceptedCallback | None,
) -> bool:
    """Process one generation attempt. Returns True if loop should break."""
    # Current question number is accepted + 1 (the one we're working on)
    current_question = len(state.accepted) + 1

    try:
        if on_phase:
            on_phase("generating", current_question, target_count)
        qa_result, viewed_pages = generate_question(
            document_path=document_path,
            corpus_config=corpus_config,
            scenario_name=scenario_name,
            mode=mode,
            previous_questions=state.previous_questions,
            previous_viewed_pages=state.previous_viewed_pages,
        )
        state.record_viewed_pages(viewed_pages)
        return _process_candidate(
            state,
            qa_result,
            document_path,
            corpus_config,
            scenario,
            mode,
            target_count,
            on_phase,
            on_accepted,
        )

    except DocumentExhaustedError as e:
        state.record_viewed_pages(e.viewed_pages)
        reason = f"document_exhausted: {e.reason}"
        logger.info("Document exhausted: %s", e.reason)
        state.set_exhausted(reason)
        return True

    except GenerationError as e:
        logger.warning("Generation error: %s", e)
        state.record_viewed_pages(e.viewed_pages)
        rejected_qa = RejectedQA(
            question="[generation_error]",
            answer="[generation_error]",
            rejection_reason=RejectionReason.VALIDATION_FAILED,
            rejection_detail=str(e),
        )
        return state.record_rejection(rejected_qa)

    except Exception as e:
        logger.exception("Unexpected error during generation")
        rejected_qa = RejectedQA(
            question="[unexpected_error]",
            answer="[unexpected_error]",
            rejection_reason=RejectionReason.VALIDATION_FAILED,
            rejection_detail=str(e),
        )
        return state.record_rejection(rejected_qa)


def _process_candidate(
    state: _GenerationState,
    qa_result: GeneratedQA,
    document_path: Path,
    corpus_config: CorpusConfig,
    scenario: EvaluationScenario,
    mode: GenerationMode,
    target_count: int,
    on_phase: OnPhaseCallback | None,
    on_accepted: OnAcceptedCallback | None,
) -> bool:
    """Process a generated candidate. Returns True if loop should break."""
    logger.debug("Generated candidate: %s", qa_result.question[:80])

    # Current question number is accepted + 1 (the one we're working on)
    current_question = len(state.accepted) + 1

    if on_phase:
        on_phase("deduplicating", current_question, target_count)
    dedupe_result = deduplicate_question(qa_result.question, state.previous_questions)

    if dedupe_result.is_duplicate:
        logger.debug("Rejected as duplicate of: %s", dedupe_result.duplicate_of)
        rejected_qa = RejectedQA(
            question=qa_result.question,
            answer=qa_result.answer,
            rejection_reason=RejectionReason.DUPLICATE,
            rejection_detail=dedupe_result.reasoning,
            duplicate_of=dedupe_result.duplicate_of,
        )
        return state.record_rejection(rejected_qa)

    if on_phase:
        on_phase("validating", current_question, target_count)
    validation = validate_question(
        document_path=document_path,
        candidate=qa_result,
        mode=mode,
        corpus_context=corpus_config.corpus_context,
        scenario_description=scenario.description,
    )

    if not validation.passed:
        reason = validation.rejection_reason or RejectionReason.VALIDATION_FAILED
        logger.debug("Rejected (%s): %s", reason.value, validation.reasoning[:60])
        rejected_qa = RejectedQA(
            question=qa_result.question,
            answer=qa_result.answer,
            rejection_reason=reason,
            rejection_detail=validation.reasoning,
        )
        return state.record_rejection(rejected_qa)

    logger.debug("Accepted: %s", qa_result.question[:80])
    qa_pair = QAPair(
        question=qa_result.question,
        answer=qa_result.answer,
        source_document=str(document_path),
        mode=mode,
        content_refs=qa_result.content_refs,
    )
    state.accepted.append(qa_pair)
    state.previous_questions.append(qa_result.question)
    state.consecutive_failures = 0

    if on_accepted:
        on_accepted(qa_pair)

    return False
