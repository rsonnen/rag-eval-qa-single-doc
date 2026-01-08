"""Question generator agent using LangGraph ReAct pattern.

This module implements the question generator that explores documents using tools
and produces Q/A pair candidates. It uses corpus configuration and evaluation
scenarios to adapt questioning style to the document's domain and intended use.
"""

import logging
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from single_doc_generator.agent import create_agent
from single_doc_generator.config import CorpusConfig, get_config
from single_doc_generator.llm import create_chat_model
from single_doc_generator.models import GenerationMode
from single_doc_generator.prompt_loader import load_prompt
from single_doc_generator.toolkit.langchain_tools import create_document_tools
from single_doc_generator.tracing import AgentTracer

logger = logging.getLogger(__name__)

# Sentinel value returned by report_exhausted tool
_EXHAUSTED_SENTINEL = "__DOCUMENT_EXHAUSTED__"


class DocumentExhaustedError(Exception):
    """Raised when the generator cannot produce more unique questions.

    Attributes:
        reason: Explanation of why generation was exhausted.
        viewed_pages: Page numbers viewed during this attempt. Callers should
            accumulate these across attempts to track document coverage.
    """

    def __init__(self, reason: str, viewed_pages: list[int] | None = None) -> None:
        self.reason = reason
        self.viewed_pages = viewed_pages or []
        super().__init__(f"Document exhausted: {reason}")


class GenerationError(Exception):
    """Raised when generation fails (e.g., agent didn't call required tools).

    Attributes:
        viewed_pages: Page numbers viewed before failure. Callers should
            accumulate these across attempts to avoid re-exploring the same pages.
    """

    def __init__(self, message: str, viewed_pages: list[int] | None = None) -> None:
        self.viewed_pages = viewed_pages or []
        super().__init__(message)


class GeneratedQA(BaseModel):
    """Structured output from the generator agent.

    Attributes:
        question: The generated question.
        answer: The ground truth answer.
        content_refs: References to visual content if applicable.
        reasoning: Brief explanation of why this is a good question.
    """

    question: str = Field(description="The question about the document")
    answer: str = Field(description="The correct answer from the document")
    content_refs: list[str] = Field(
        default_factory=list,
        description="References to figures, tables, or images if applicable",
    )
    reasoning: str = Field(
        description="Explanation of why this tests document understanding"
    )


def _create_qa_submission(
    question: str,
    answer: str,
    reasoning: str,
    content_refs: list[str] | None = None,
) -> str:
    """Create Q/A submission as JSON for extraction from tool response."""
    qa = GeneratedQA(
        question=question,
        answer=answer,
        reasoning=reasoning,
        content_refs=content_refs or [],
    )
    return qa.model_dump_json()


def _report_exhausted(reason: str) -> str:
    """Signal that no more unique questions can be generated."""
    return f"{_EXHAUSTED_SENTINEL}:{reason}"


def _extract_viewed_pages(messages: list[object]) -> list[int]:
    """Extract page numbers viewed from agent message history.

    Scans through AIMessage tool_calls to find all view_page invocations
    and returns the unique page numbers in order of viewing.

    Args:
        messages: List of messages from agent execution.

    Returns:
        List of page numbers viewed (no duplicates, in order).
    """
    viewed_pages: list[int] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call["name"] == "view_page":
                    page = tool_call["args"].get("page")
                    if isinstance(page, int) and page not in viewed_pages:
                        viewed_pages.append(page)
    return viewed_pages


def generate_question(
    document_path: Path,
    corpus_config: CorpusConfig,
    scenario_name: str,
    mode: GenerationMode = GenerationMode.TEXTUAL,
    previous_questions: list[str] | None = None,
    previous_viewed_pages: list[int] | None = None,
) -> tuple[GeneratedQA, list[int]]:
    """Generate a Q/A pair using corpus configuration and evaluation scenario.

    Uses a LangGraph ReAct agent to explore the document with tools and
    produce a question appropriate to the specified evaluation scenario.

    Args:
        document_path: Path to the document to generate questions from.
        corpus_config: Corpus configuration with context and scenarios.
        scenario_name: Name of the evaluation scenario to use.
        mode: Generation mode (textual or visual).
        previous_questions: List of previously asked questions to avoid repeating.
        previous_viewed_pages: List of page numbers already viewed in prior attempts.

    Returns:
        Tuple of (GeneratedQA, viewed_pages) where viewed_pages is the list of
        page numbers viewed during this generation attempt.

    Raises:
        FileNotFoundError: If document does not exist.
        KeyError: If scenario_name not found in corpus_config.
        DocumentExhaustedError: If the generator cannot produce more unique questions.
        ValueError: If generation fails without calling either tool.
    """
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    scenario = corpus_config.get_scenario(scenario_name)

    config = get_config()
    logger.info(
        "Generating question from %s using %s (scenario: %s)",
        document_path,
        config.generator_model.name,
        scenario_name,
    )

    llm = create_chat_model(config.generator_model.name, config.generator_model.kwargs)
    document_tools = create_document_tools(document_path)

    submit_tool = StructuredTool.from_function(
        func=_create_qa_submission,
        name="submit_qa",
        description=(
            "Submit your generated question/answer pair. Call this when you have "
            "found content to ask about and formulated your question and answer."
        ),
        args_schema=GeneratedQA,
    )

    exhausted_tool = StructuredTool.from_function(
        func=_report_exhausted,
        name="report_exhausted",
        description=(
            "Call this ONLY if you have thoroughly explored the document and "
            "cannot find any content suitable for a NEW question that differs "
            "from the previously asked questions. Provide a brief reason."
        ),
    )

    prompt = load_prompt(
        "generator",
        corpus_name=corpus_config.name,
        corpus_context=corpus_config.corpus_context,
        scenario_description=scenario.description,
        mode=mode.value,
        previous_questions=previous_questions or [],
        previous_viewed_pages=previous_viewed_pages or [],
    )

    agent = create_agent(
        model=llm,
        tools=[*document_tools, submit_tool, exhausted_tool],
        system_prompt=prompt,
    )

    # Trace tool calls to OTLP
    tracer = AgentTracer(context="generator")

    try:
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Generate one question about this document.")
                ],
                "pending_images": [],
            },
            {"callbacks": [tracer], "recursion_limit": 25},
        )
    except GraphRecursionError:
        logger.warning("Generator hit recursion limit for document: %s", document_path)
        raise GenerationError(
            "Generator exceeded iteration limit without submitting question",
            viewed_pages=[],
        ) from None

    # Always extract viewed pages first - needed for error tracking
    viewed_pages = _extract_viewed_pages(result["messages"])

    for message in reversed(result["messages"]):
        if isinstance(message, ToolMessage):
            content = message.content
            if not isinstance(content, str):
                continue

            if message.name == "submit_qa":
                return GeneratedQA.model_validate_json(content), viewed_pages

            if message.name == "report_exhausted":
                # Parse reason from sentinel format
                reason = content.replace(f"{_EXHAUSTED_SENTINEL}:", "", 1)
                raise DocumentExhaustedError(reason, viewed_pages)

    raise GenerationError(
        "Agent did not call submit_qa or report_exhausted", viewed_pages
    )
