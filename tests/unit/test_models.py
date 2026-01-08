"""Tests for single_doc_generator models."""

import pytest

from single_doc_generator.models import (
    DocumentFormat,
    GenerationMode,
    GenerationResult,
    GenerationStats,
    QAPair,
    ReadLinesResult,
    RejectedQA,
    RejectionReason,
    SearchMatch,
    SearchResult,
    ViewPageResult,
    VisualContent,
    VisualContentResult,
    detect_format,
)


class TestDetectFormat:
    """Tests for detect_format function."""

    def test_txt_extension(self, tmp_path):
        """Detects .txt as TEXT format."""
        file = tmp_path / "doc.txt"
        file.touch()
        assert detect_format(file) == DocumentFormat.TEXT

    def test_md_extension(self, tmp_path):
        """Detects .md as MARKDOWN format."""
        file = tmp_path / "doc.md"
        file.touch()
        assert detect_format(file) == DocumentFormat.MARKDOWN

    def test_markdown_extension(self, tmp_path):
        """Detects .markdown as MARKDOWN format."""
        file = tmp_path / "doc.markdown"
        file.touch()
        assert detect_format(file) == DocumentFormat.MARKDOWN

    def test_pdf_extension(self, tmp_path):
        """Detects .pdf as PDF format."""
        file = tmp_path / "doc.pdf"
        file.touch()
        assert detect_format(file) == DocumentFormat.PDF

    def test_xml_extension(self, tmp_path):
        """Detects .xml as XML format."""
        file = tmp_path / "doc.xml"
        file.touch()
        assert detect_format(file) == DocumentFormat.XML

    def test_uppercase_extension(self, tmp_path):
        """Extension detection is case-insensitive."""
        file = tmp_path / "doc.TXT"
        file.touch()
        assert detect_format(file) == DocumentFormat.TEXT

    def test_unsupported_extension(self, tmp_path):
        """Raises ValueError for unsupported extensions."""
        file = tmp_path / "doc.docx"
        file.touch()
        with pytest.raises(ValueError, match="Unsupported file extension"):
            detect_format(file)

    def test_no_extension(self, tmp_path):
        """Raises ValueError for files without extension."""
        file = tmp_path / "doc"
        file.touch()
        with pytest.raises(ValueError, match="Unsupported file extension"):
            detect_format(file)


class TestReadLinesResult:
    """Tests for ReadLinesResult model."""

    def test_basic_construction(self):
        """Can construct with required fields."""
        result = ReadLinesResult(
            lines=[(1, "hello"), (2, "world")],
            total_lines=10,
            start_line=1,
            end_line=2,
        )
        assert len(result.lines) == 2
        assert result.total_lines == 10

    def test_empty_lines(self):
        """Can construct with empty lines list."""
        result = ReadLinesResult(
            lines=[],
            total_lines=0,
            start_line=1,
            end_line=1,
        )
        assert result.lines == []


class TestSearchMatch:
    """Tests for SearchMatch model."""

    def test_basic_construction(self):
        """Can construct with required fields."""
        match = SearchMatch(
            line_number=5,
            line_content="found it here",
        )
        assert match.line_number == 5
        assert match.context_before == []
        assert match.context_after == []

    def test_with_context(self):
        """Can construct with context."""
        match = SearchMatch(
            line_number=5,
            line_content="found it here",
            context_before=[(3, "line 3"), (4, "line 4")],
            context_after=[(6, "line 6")],
        )
        assert len(match.context_before) == 2
        assert len(match.context_after) == 1


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_basic_construction(self):
        """Can construct with required fields."""
        result = SearchResult(
            pattern=r"\btest\b",
            matches=[],
            total_matches=0,
        )
        assert result.pattern == r"\btest\b"
        assert result.matches == []


class TestViewPageResult:
    """Tests for ViewPageResult model."""

    def test_not_applicable(self):
        """Can construct not_applicable result."""
        result = ViewPageResult(not_applicable=True, page_number=1)
        assert result.not_applicable is True
        assert result.image_base64 is None

    def test_with_image(self):
        """Can construct with image data."""
        result = ViewPageResult(
            not_applicable=False,
            page_number=1,
            image_base64="base64data==",
            mime_type="image/png",
        )
        assert result.not_applicable is False
        assert result.image_base64 == "base64data=="


class TestVisualContent:
    """Tests for VisualContent model."""

    def test_basic_construction(self):
        """Can construct with required fields."""
        item = VisualContent(
            content_type="image",
            reference="path/to/img.png",
        )
        assert item.content_type == "image"
        assert item.alt_text is None

    def test_with_optional_fields(self):
        """Can construct with optional fields."""
        item = VisualContent(
            content_type="figure",
            reference="fig1",
            alt_text="A diagram",
            location="page 3",
        )
        assert item.alt_text == "A diagram"
        assert item.location == "page 3"


class TestVisualContentResult:
    """Tests for VisualContentResult model."""

    def test_empty(self):
        """Can construct empty result."""
        result = VisualContentResult(items=[], total_items=0)
        assert result.items == []
        assert result.total_items == 0

    def test_with_items(self):
        """Can construct with items."""
        items = [
            VisualContent(content_type="image", reference="img1.png"),
            VisualContent(content_type="image", reference="img2.png"),
        ]
        result = VisualContentResult(items=items, total_items=2)
        assert len(result.items) == 2


class TestGenerationStats:
    """Tests for GenerationStats model."""

    def test_basic_construction(self):
        """Can construct with required fields."""
        stats = GenerationStats(
            total_attempts=10,
            accepted_count=5,
            rejected_count=5,
            exhausted=False,
        )
        assert stats.total_attempts == 10
        assert stats.accepted_count == 5
        assert stats.rejected_count == 5
        assert stats.exhausted is False
        assert stats.exhaustion_reason is None
        assert stats.rejection_breakdown == {}

    def test_with_rejection_breakdown(self):
        """Can construct with rejection breakdown."""
        stats = GenerationStats(
            total_attempts=10,
            accepted_count=3,
            rejected_count=7,
            rejection_breakdown={
                "duplicate": 3,
                "unanswerable": 2,
                "wrong_answer": 2,
            },
            exhausted=True,
            exhaustion_reason="consecutive_failures",
        )
        assert stats.rejection_breakdown["duplicate"] == 3
        assert stats.exhaustion_reason == "consecutive_failures"

    def test_exhaustion_reason_without_exhausted(self):
        """Can set exhaustion_reason even if exhausted is False."""
        stats = GenerationStats(
            total_attempts=5,
            accepted_count=5,
            rejected_count=0,
            exhausted=False,
            exhaustion_reason=None,
        )
        assert stats.exhaustion_reason is None


class TestGenerationResult:
    """Tests for GenerationResult model."""

    def test_basic_construction(self):
        """Can construct with required fields."""
        stats = GenerationStats(
            total_attempts=1,
            accepted_count=1,
            rejected_count=0,
            exhausted=False,
        )
        result = GenerationResult(
            document="/path/to/doc.pdf",
            corpus="/path/to/corpus",
            scenario="rag_eval",
            mode=GenerationMode.TEXTUAL,
            target_count=5,
            stats=stats,
            timestamp="2025-01-01T00:00:00Z",
        )
        assert result.document == "/path/to/doc.pdf"
        assert result.mode == GenerationMode.TEXTUAL
        assert result.accepted == []
        assert result.rejected == []

    def test_with_accepted_and_rejected(self):
        """Can construct with accepted and rejected lists."""
        stats = GenerationStats(
            total_attempts=3,
            accepted_count=1,
            rejected_count=2,
            rejection_breakdown={"duplicate": 1, "unanswerable": 1},
            exhausted=False,
        )
        accepted = [
            QAPair(
                question="What is X?",
                answer="X is Y",
                source_document="/doc.pdf",
                mode=GenerationMode.TEXTUAL,
            )
        ]
        rejected = [
            RejectedQA(
                question="Duplicate Q?",
                answer="Some answer",
                rejection_reason=RejectionReason.DUPLICATE,
                rejection_detail="Same as Q1",
                duplicate_of="What is X?",
            ),
            RejectedQA(
                question="Unanswerable Q?",
                answer="Unknown",
                rejection_reason=RejectionReason.UNANSWERABLE,
                rejection_detail="Not in document",
            ),
        ]
        result = GenerationResult(
            document="/doc.pdf",
            corpus="/corpus",
            scenario="test",
            mode=GenerationMode.TEXTUAL,
            target_count=3,
            accepted=accepted,
            rejected=rejected,
            stats=stats,
            timestamp="2025-01-01T00:00:00Z",
        )
        assert len(result.accepted) == 1
        assert len(result.rejected) == 2
        assert result.rejected[0].duplicate_of == "What is X?"

    def test_json_serialization(self):
        """Can serialize to JSON-compatible dict."""
        stats = GenerationStats(
            total_attempts=1,
            accepted_count=1,
            rejected_count=0,
            exhausted=False,
        )
        result = GenerationResult(
            document="/doc.pdf",
            corpus="/corpus",
            scenario="test",
            mode=GenerationMode.VISUAL,
            target_count=1,
            accepted=[
                QAPair(
                    question="Q?",
                    answer="A",
                    source_document="/doc.pdf",
                    mode=GenerationMode.VISUAL,
                    content_refs=["fig1"],
                )
            ],
            stats=stats,
            timestamp="2025-01-01T00:00:00Z",
        )
        data = result.model_dump(mode="json")
        assert data["mode"] == "visual"
        assert data["accepted"][0]["content_refs"] == ["fig1"]
        assert data["stats"]["exhausted"] is False
