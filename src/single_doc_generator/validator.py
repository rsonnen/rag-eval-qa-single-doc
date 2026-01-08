"""Question validator using a separate LLM model.

This module implements the validator that attempts to answer generated questions
using document tools. It compares its answer to the proposed ground truth to
determine if the question is valid and answerable.
"""

import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from single_doc_generator.agent import AgentState, create_agent
from single_doc_generator.config import get_config
from single_doc_generator.generator import GeneratedQA
from single_doc_generator.llm import create_chat_model
from single_doc_generator.models import (
    GenerationMode,
    RejectionReason,
    ValidationResult,
)
from single_doc_generator.prompt_loader import load_prompt
from single_doc_generator.toolkit.langchain_tools import create_document_tools
from single_doc_generator.tracing import AgentTracer

logger = logging.getLogger(__name__)


class ValidatorAssessment(BaseModel):
    """Structured output from the validator's assessment.

    Attributes:
        answerable: Whether the question can be answered from the document.
        my_answer: The validator's answer to the question.
        answer_matches: Whether the validator's answer matches the proposed answer.
        reasoning: Explanation of the assessment.
        is_ambiguous: Whether the question has multiple valid interpretations.
        is_trivial: Whether the question is too easy or doesn't need the document.
    """

    answerable: bool = Field(
        description="Can this question be answered from the document?"
    )
    my_answer: str = Field(description="Your answer to the question")
    answer_matches: bool = Field(
        description="Does your answer match the proposed ground truth?"
    )
    reasoning: str = Field(description="Explanation of your assessment")
    is_ambiguous: bool = Field(
        default=False,
        description="Does the question have multiple valid interpretations?",
    )
    is_trivial: bool = Field(
        default=False,
        description="Is this question too easy or answerable without the document?",
    )


def _create_assessment_submission(
    answerable: bool,
    my_answer: str,
    answer_matches: bool,
    reasoning: str,
    is_ambiguous: bool = False,
    is_trivial: bool = False,
) -> str:
    """Create assessment as JSON for extraction from tool response."""
    assessment = ValidatorAssessment(
        answerable=answerable,
        my_answer=my_answer,
        answer_matches=answer_matches,
        reasoning=reasoning,
        is_ambiguous=is_ambiguous,
        is_trivial=is_trivial,
    )
    return assessment.model_dump_json()


def _assessment_to_result(assessment: ValidatorAssessment) -> ValidationResult:
    """Convert a ValidatorAssessment to a ValidationResult."""
    if not assessment.answerable:
        return ValidationResult(
            passed=False,
            reasoning=assessment.reasoning,
            validator_answer=assessment.my_answer,
            rejection_reason=RejectionReason.UNANSWERABLE,
        )

    if assessment.is_ambiguous:
        return ValidationResult(
            passed=False,
            reasoning=assessment.reasoning,
            validator_answer=assessment.my_answer,
            rejection_reason=RejectionReason.AMBIGUOUS,
        )

    if assessment.is_trivial:
        return ValidationResult(
            passed=False,
            reasoning=assessment.reasoning,
            validator_answer=assessment.my_answer,
            rejection_reason=RejectionReason.TRIVIAL,
        )

    if not assessment.answer_matches:
        return ValidationResult(
            passed=False,
            reasoning=assessment.reasoning,
            validator_answer=assessment.my_answer,
            rejection_reason=RejectionReason.WRONG_ANSWER,
        )

    return ValidationResult(
        passed=True,
        reasoning=assessment.reasoning,
        validator_answer=assessment.my_answer,
    )


def _invoke_with_retry(
    agent: CompiledStateGraph[AgentState, AgentState, AgentState],
    validation_request: str,
    tracer: AgentTracer,
    question_preview: str,
    max_retries: int = 3,
) -> dict[str, Any] | ValidationResult:
    """Invoke agent with retry logic for Gemini empty-choices errors.

    Args:
        agent: The compiled agent graph.
        validation_request: The validation prompt.
        tracer: OTLP tracer for callbacks.
        question_preview: First 50 chars of question for logging.
        max_retries: Maximum retry attempts for empty-choices errors.

    Returns:
        Agent result dict on success, or ValidationResult on unrecoverable failure.
    """
    for attempt in range(max_retries):
        try:
            return agent.invoke(
                {
                    "messages": [HumanMessage(content=validation_request)],
                    "pending_images": [],
                },
                {"callbacks": [tracer], "recursion_limit": 15},
            )
        except GraphRecursionError:
            logger.warning(
                "Validator hit recursion limit for question: %s",
                question_preview,
            )
            return ValidationResult(
                passed=False,
                reasoning="Validator exceeded iteration limit without submitting",
                rejection_reason=RejectionReason.VALIDATION_FAILED,
            )
        except TypeError as e:
            # Gemini sometimes returns 200 OK with empty choices array.
            if "null value for `choices`" not in str(e):
                raise
            if attempt < max_retries - 1:
                logger.warning(
                    "Empty choices from model (attempt %d/%d), retrying: %s",
                    attempt + 1,
                    max_retries,
                    question_preview,
                )
                time.sleep(1.0)
            else:
                logger.warning(
                    "Empty choices from model after %d attempts: %s",
                    max_retries,
                    question_preview,
                )
                return ValidationResult(
                    passed=False,
                    reasoning="Model returned empty response after retries",
                    rejection_reason=RejectionReason.VALIDATION_FAILED,
                )

    # Should not reach here, but handle gracefully
    return ValidationResult(
        passed=False,
        reasoning="No result from validator",
        rejection_reason=RejectionReason.VALIDATION_FAILED,
    )


def validate_question(
    document_path: Path,
    candidate: GeneratedQA,
    mode: GenerationMode = GenerationMode.TEXTUAL,
    corpus_context: str | None = None,
    scenario_description: str | None = None,
    max_empty_choices_retries: int = 3,
) -> ValidationResult:
    """Validate a Q/A candidate by attempting to answer from the document.

    Uses a separate LLM model (configured in app config) to read the document
    and answer the question independently. Compares the validator's answer to
    the proposed ground truth to determine if the question is valid.

    Args:
        document_path: Path to the source document.
        candidate: The Q/A pair to validate.
        mode: Generation mode (textual or visual). Visual mode validates
            that questions require understanding visual content.
        corpus_context: Description of the corpus for relevance checking.
        scenario_description: Evaluation scenario for relevance checking.
        max_empty_choices_retries: Maximum retries for empty-choices errors.

    Returns:
        ValidationResult with pass/fail, reasoning, and validator's answer.

    Raises:
        FileNotFoundError: If document does not exist.
    """
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    config = get_config()
    logger.info(
        "Validating question '%s...' using %s (mode: %s)",
        candidate.question[:50],
        config.validator_model.name,
        mode.value,
    )

    llm = create_chat_model(config.validator_model.name, config.validator_model.kwargs)
    document_tools = create_document_tools(document_path)
    prompt = load_prompt(
        "validator",
        mode=mode.value,
        corpus_context=corpus_context or "",
        scenario_description=scenario_description or "",
    )

    # Create submit tool - agent calls this to submit, returns JSON for extraction
    submit_tool = StructuredTool.from_function(
        func=_create_assessment_submission,
        name="submit_assessment",
        description=(
            "MANDATORY: Submit your final validation assessment. This tool MUST "
            "be called to complete validation - without it, validation fails. "
            "Call this after 2-4 document tool calls with your verdict on the "
            "question's validity."
        ),
        args_schema=ValidatorAssessment,
    )

    agent = create_agent(
        model=llm,
        tools=[*document_tools, submit_tool],
        system_prompt=prompt,
    )

    # Build validation request with content refs for visual mode
    content_refs_section = ""
    if mode == GenerationMode.VISUAL and candidate.content_refs:
        refs_list = ", ".join(candidate.content_refs)
        content_refs_section = f"\nREFERENCED VISUAL CONTENT: {refs_list}\n"

    validation_request = f"""Please validate this question/answer pair:

QUESTION: {candidate.question}

PROPOSED ANSWER: {candidate.answer}
{content_refs_section}
Use the document tools to find relevant information and determine if:
1. The question can be answered from the document
2. The proposed answer is correct
3. The question is neither trivial nor ambiguous
"""

    # Trace tool calls to OTLP
    tracer = AgentTracer(context="validator")

    invoke_result = _invoke_with_retry(
        agent,
        validation_request,
        tracer,
        candidate.question[:50],
        max_empty_choices_retries,
    )

    # If retry helper returned a ValidationResult, propagate it
    if isinstance(invoke_result, ValidationResult):
        return invoke_result

    result = invoke_result

    # Extract structured data from the submit_assessment tool response
    for message in reversed(result["messages"]):
        if isinstance(message, ToolMessage) and message.name == "submit_assessment":
            content = message.content
            if isinstance(content, str):
                assessment = ValidatorAssessment.model_validate_json(content)
                return _assessment_to_result(assessment)
            raise ValueError("submit_assessment tool returned non-string content")

    return ValidationResult(
        passed=False,
        reasoning="Validator did not call submit_assessment tool",
        rejection_reason=RejectionReason.VALIDATION_FAILED,
    )
