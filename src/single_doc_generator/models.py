"""Pydantic models for document toolkit and question generation.

These models define the structured data types used by document exploration tools
and the question generation pipeline. They provide type safety and validation
for tool inputs/outputs and Q/A pair management.
"""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class GenerationMode(str, Enum):
    """Question generation mode.

    TEXTUAL: Factual recall questions from document text only.
    VISUAL: Questions requiring understanding of visual content.
    """

    TEXTUAL = "textual"
    VISUAL = "visual"


class RejectionReason(str, Enum):
    """Reason why a question was rejected.

    VALIDATION_FAILED: Validator could not confirm the answer.
    DUPLICATE: Question is semantically equivalent to an accepted question.
    UNANSWERABLE: Question cannot be answered from document content.
    WRONG_ANSWER: Validator's answer differs from proposed ground truth.
    AMBIGUOUS: Question has multiple valid interpretations or answers.
    TRIVIAL: Question is too easy or doesn't require document reading.
    """

    VALIDATION_FAILED = "validation_failed"
    DUPLICATE = "duplicate"
    UNANSWERABLE = "unanswerable"
    WRONG_ANSWER = "wrong_answer"
    AMBIGUOUS = "ambiguous"
    TRIVIAL = "trivial"


class QAPair(BaseModel):
    """A validated question/answer pair.

    Attributes:
        question: The question text.
        answer: Ground truth answer.
        source_document: Path to the source document.
        mode: Generation mode (textual/visual).
        content_refs: For visual mode, specific figures/tables referenced.
    """

    question: str = Field(description="Question text")
    answer: str = Field(description="Ground truth answer")
    source_document: str = Field(description="Path to source document")
    mode: GenerationMode = Field(description="Generation mode used")
    content_refs: list[str] = Field(
        default_factory=list,
        description="Visual content references (visual mode)",
    )


class ValidationResult(BaseModel):
    """Result of validating a Q/A candidate.

    Attributes:
        passed: Whether validation passed.
        reasoning: Explanation of the validation decision.
        validator_answer: The answer the validator produced.
        rejection_reason: If failed, the specific reason.
    """

    passed: bool = Field(description="Whether validation passed")
    reasoning: str = Field(description="Explanation of validation decision")
    validator_answer: str | None = Field(
        default=None,
        description="Answer the validator produced",
    )
    rejection_reason: RejectionReason | None = Field(
        default=None,
        description="Reason for rejection if failed",
    )


class RejectedQA(BaseModel):
    """A question that failed validation or was rejected as duplicate.

    Attributes:
        question: The question text.
        answer: Proposed answer (may be incorrect).
        rejection_reason: Why the question was rejected.
        rejection_detail: Specific explanation.
        duplicate_of: If duplicate, which question it duplicates.
    """

    question: str = Field(description="Question text")
    answer: str = Field(description="Proposed answer")
    rejection_reason: RejectionReason = Field(description="Reason for rejection")
    rejection_detail: str = Field(description="Specific explanation")
    duplicate_of: str | None = Field(
        default=None,
        description="Question this duplicates (if duplicate)",
    )


class DocumentFormat(str, Enum):
    """Supported document formats."""

    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"
    XML = "xml"
    HTML = "html"
    ADOC = "adoc"


class ReadLinesResult(BaseModel):
    """Result from read_lines tool.

    Attributes:
        lines: List of tuples containing (line_number, line_content).
        total_lines: Total number of lines in the document.
        start_line: First line number returned (1-indexed).
        end_line: Last line number returned (1-indexed).
    """

    lines: list[tuple[int, str]] = Field(
        description="List of (line_number, content) tuples"
    )
    total_lines: int = Field(description="Total lines in document")
    start_line: int = Field(description="First line number returned (1-indexed)")
    end_line: int = Field(description="Last line number returned (1-indexed)")


class SearchMatch(BaseModel):
    """A single search match with context.

    Attributes:
        line_number: Line number where match occurred (1-indexed).
        line_content: The full line containing the match.
        context_before: Lines before the match.
        context_after: Lines after the match.
    """

    line_number: int = Field(description="Line number of match (1-indexed)")
    line_content: str = Field(description="Full line containing match")
    context_before: list[tuple[int, str]] = Field(
        default_factory=list, description="(line_num, content) before match"
    )
    context_after: list[tuple[int, str]] = Field(
        default_factory=list, description="(line_num, content) after match"
    )


class SearchResult(BaseModel):
    """Result from search tool.

    Attributes:
        pattern: The regex pattern that was searched.
        matches: List of matches found.
        total_matches: Total number of matches.
    """

    pattern: str = Field(description="Regex pattern searched")
    matches: list[SearchMatch] = Field(description="List of matches")
    total_matches: int = Field(description="Total matches found")


class ViewPageResult(BaseModel):
    """Result from view_page tool.

    For text-based formats, this returns not_applicable=True.
    For paginated formats (PDF), this returns base64 image data.

    Attributes:
        not_applicable: True if format doesn't support page viewing.
        page_number: Requested page number (if applicable).
        total_pages: Total pages in document (if applicable).
        image_base64: Base64-encoded image data (if applicable).
        mime_type: MIME type of image (if applicable).
    """

    not_applicable: bool = Field(
        default=False, description="True if format doesn't support pages"
    )
    page_number: int | None = Field(default=None, description="Requested page number")
    total_pages: int | None = Field(default=None, description="Total pages in document")
    image_base64: str | None = Field(default=None, description="Base64 image data")
    mime_type: str | None = Field(default=None, description="Image MIME type")


class VisualContent(BaseModel):
    """A visual element discovered in the document.

    Attributes:
        content_type: Type of visual content (image, figure, table, etc.).
        reference: The reference string (e.g., image path, figure ID).
        alt_text: Alternative text or caption if available.
        location: Location description (line number, page, etc.).
    """

    content_type: str = Field(description="Type: image, figure, table, etc.")
    reference: str = Field(description="Reference path or identifier")
    alt_text: str | None = Field(default=None, description="Alt text or caption")
    location: str | None = Field(default=None, description="Location in document")


class VisualContentResult(BaseModel):
    """Result from list_visual_content tool.

    Attributes:
        items: List of visual content items found.
        total_items: Total count of visual elements.
    """

    items: list[VisualContent] = Field(description="Visual content items found")
    total_items: int = Field(description="Count of visual elements")


def detect_format(file_path: Path) -> DocumentFormat:
    """Detect document format from file extension.

    Args:
        file_path: Path to the document file.

    Returns:
        The detected DocumentFormat.

    Raises:
        ValueError: If file extension is not supported.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return DocumentFormat.TEXT
    if suffix in (".md", ".markdown"):
        return DocumentFormat.MARKDOWN
    if suffix == ".pdf":
        return DocumentFormat.PDF
    if suffix == ".xml":
        return DocumentFormat.XML
    if suffix == ".html":
        return DocumentFormat.HTML
    if suffix in (".adoc", ".asciidoc"):
        return DocumentFormat.ADOC
    raise ValueError(f"Unsupported file extension: {suffix}")


class GenerationStats(BaseModel):
    """Statistics from a question generation run.

    Tracks the outcomes of all generation attempts including acceptance rate,
    rejection breakdown by reason, and exhaustion status.

    Attributes:
        total_attempts: Total number of generation attempts made.
        accepted_count: Number of questions that passed validation.
        rejected_count: Number of questions rejected (dedup, validation, errors).
        rejection_breakdown: Count of rejections by RejectionReason value.
        exhausted: Whether generation stopped due to exhaustion.
        exhaustion_reason: If exhausted, the reason (consecutive failures, etc).
    """

    total_attempts: int = Field(description="Total generation attempts")
    accepted_count: int = Field(description="Questions that passed validation")
    rejected_count: int = Field(description="Questions rejected")
    rejection_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Rejection counts by reason",
    )
    exhausted: bool = Field(description="Whether stopped due to exhaustion")
    exhaustion_reason: str | None = Field(
        default=None,
        description="Reason for exhaustion if applicable",
    )


class GenerationResult(BaseModel):
    """Complete result from a question generation run.

    Contains all accepted and rejected questions along with statistics
    and metadata about the generation run.

    Attributes:
        document: Path to the source document.
        corpus: Path to the corpus directory.
        scenario: Scenario name used for generation.
        mode: Generation mode (textual/visual).
        target_count: Number of questions requested.
        accepted: List of accepted Q/A pairs.
        rejected: List of rejected questions with reasons.
        stats: Generation statistics.
        timestamp: When generation was run.
    """

    document: str = Field(description="Path to source document")
    corpus: str = Field(description="Path to corpus directory")
    scenario: str = Field(description="Scenario name")
    mode: GenerationMode = Field(description="Generation mode")
    target_count: int = Field(description="Requested question count")
    accepted: list[QAPair] = Field(
        default_factory=list,
        description="Accepted Q/A pairs",
    )
    rejected: list[RejectedQA] = Field(
        default_factory=list,
        description="Rejected questions",
    )
    stats: GenerationStats = Field(description="Generation statistics")
    timestamp: str = Field(description="ISO format timestamp")


# Supported file extensions for document discovery
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".txt", ".md", ".markdown", ".pdf", ".xml", ".html", ".adoc", ".asciidoc"}
)


class CorpusGenerationResult(BaseModel):
    """Q/A output for RAGAS evaluation.

    Contains corpus metadata and generated Q/A pairs. Processing state
    (resumption, statistics, per-document results) stays in the database.

    Attributes:
        corpus_name: Human-readable name from corpus.yaml.
        corpus_path: Absolute path to corpus directory.
        scenario: Scenario name used for generation.
        mode: Generation mode (textual/visual).
        questions: All accepted Q/A pairs from all documents.
        timestamp: When corpus processing completed (ISO format).
    """

    corpus_name: str = Field(description="Human-readable corpus name")
    corpus_path: str = Field(description="Absolute path to corpus directory")
    scenario: str = Field(description="Scenario name")
    mode: GenerationMode = Field(description="Generation mode")
    questions: list[QAPair] = Field(
        default_factory=list,
        description="All accepted Q/A pairs from all documents",
    )
    timestamp: str = Field(description="ISO format timestamp")
