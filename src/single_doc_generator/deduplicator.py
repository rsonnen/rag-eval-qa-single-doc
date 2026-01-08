"""Semantic deduplication of generated questions.

This module checks whether a new question is semantically equivalent to any
already-accepted questions. Unlike the generator and validator which use
ReAct agents with tools, the deduplicator is a single structured LLM call
because it only compares question texts - no document access needed.

Runs before expensive validation to filter duplicates cheaply.
"""

import logging

from pydantic import BaseModel, Field

from single_doc_generator.config import get_config
from single_doc_generator.llm import create_chat_model
from single_doc_generator.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class DeduplicationResult(BaseModel):
    """Result of checking a question for semantic duplicates.

    Attributes:
        is_duplicate: True if the question is semantically equivalent to
            an existing question (asks the same thing, different words).
        duplicate_of: The existing question this duplicates, verbatim.
            Only set when is_duplicate is True.
        reasoning: Brief explanation of the decision.
    """

    is_duplicate: bool = Field(
        description="True if this question asks the same thing as an existing one"
    )
    duplicate_of: str | None = Field(
        default=None,
        description="The existing question this duplicates (verbatim), if any",
    )
    reasoning: str = Field(
        description="Brief explanation of why this is or is not a duplicate"
    )


def deduplicate_question(
    question: str,
    existing_questions: list[str],
) -> DeduplicationResult:
    """Check if a question is semantically equivalent to any existing questions.

    Uses a fast, cheap LLM model to compare question texts. The generator
    already tries to avoid duplicates via in-prompt guidance, but this
    catches semantic duplicates that slip through (same meaning, different words).

    Args:
        question: The new question to check.
        existing_questions: List of already-accepted question texts.

    Returns:
        DeduplicationResult indicating unique or duplicate status.
    """
    if not existing_questions:
        logger.debug("No existing questions - first question is automatically unique")
        return DeduplicationResult(
            is_duplicate=False,
            duplicate_of=None,
            reasoning="First question in the set - automatically unique",
        )

    config = get_config()
    logger.info(
        "Checking question for duplicates using %s (comparing against %d existing)",
        config.deduplicator_model.name,
        len(existing_questions),
    )

    llm = create_chat_model(
        config.deduplicator_model.name, config.deduplicator_model.kwargs
    )
    llm_with_output = llm.with_structured_output(DeduplicationResult)

    prompt = load_prompt(
        "deduplicator",
        new_question=question,
        existing_questions=existing_questions,
    )

    result = llm_with_output.invoke(prompt)

    if not isinstance(result, DeduplicationResult):
        logger.error("LLM returned unexpected type: %s", type(result))
        raise TypeError(f"Expected DeduplicationResult, got {type(result).__name__}")

    if result.is_duplicate:
        logger.info(
            "Question is duplicate of: %s",
            result.duplicate_of[:50] if result.duplicate_of else "unknown",
        )
    else:
        logger.debug("Question is unique")

    return result
