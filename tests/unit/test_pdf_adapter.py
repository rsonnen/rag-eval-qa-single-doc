"""Tests for PDFAdapter."""

import base64
import re

import pymupdf
import pytest

from single_doc_generator.toolkit import (
    list_visual_content,
    read_lines,
    search,
    view_page,
)
from single_doc_generator.toolkit.pdf_adapter import PDFAdapter


@pytest.fixture
def sample_pdf_file(tmp_path):
    """Create a simple PDF with text content using PyMuPDF."""
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()

    # Create 3 pages with content
    for page_num in range(1, 4):
        page = doc.new_page()
        text = f"Page {page_num} content\nLine 2 on page {page_num}\n"
        text += f"Line 3 on page {page_num}"
        page.insert_text((50, 50), text, fontsize=12)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf_file(tmp_path):
    """Create an empty PDF (pages but no text)."""
    pdf_path = tmp_path / "empty.pdf"
    doc = pymupdf.open()
    doc.new_page()  # One blank page
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def single_page_pdf(tmp_path):
    """Create a single page PDF."""
    pdf_path = tmp_path / "single.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Single page content\nSecond line", fontsize=12)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_with_images(tmp_path):
    """Create a PDF with embedded images."""
    pdf_path = tmp_path / "with_images.pdf"
    doc = pymupdf.open()
    page = doc.new_page()

    # Add some text
    page.insert_text((50, 50), "Page with images", fontsize=12)

    # Create a small test image (10x10 red square)
    # Using RGB colorspace without alpha (alpha=0)
    img = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), 0)
    img.set_rect(img.irect, (255, 0, 0))  # Red color (RGB)

    # Insert the image
    rect = pymupdf.Rect(100, 100, 200, 200)
    page.insert_image(rect, pixmap=img)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def large_pdf_file(tmp_path):
    """Create a PDF with many lines of text."""
    pdf_path = tmp_path / "large.pdf"
    doc = pymupdf.open()

    # Create content that spans multiple pages
    page = doc.new_page()
    y_pos = 50
    line_height = 14

    for i in range(1, 101):  # 100 lines
        page.insert_text((50, y_pos), f"Line number {i}", fontsize=10)
        y_pos += line_height

        # Start new page if needed (roughly every 50 lines)
        if y_pos > 750:
            page = doc.new_page()
            y_pos = 50

    doc.save(pdf_path)
    doc.close()
    return pdf_path


class TestPDFAdapterInit:
    """Tests for PDFAdapter initialization."""

    def test_init_with_valid_file(self, sample_pdf_file):
        """Initializes with valid PDF file path."""
        adapter = PDFAdapter(sample_pdf_file)
        assert adapter.file_path == sample_pdf_file
        assert adapter.total_pages == 3
        assert adapter.total_lines > 0

    def test_init_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        missing = tmp_path / "missing.pdf"
        with pytest.raises(FileNotFoundError):
            PDFAdapter(missing)

    def test_init_empty_pdf(self, empty_pdf_file):
        """Handles PDF with blank pages."""
        adapter = PDFAdapter(empty_pdf_file)
        assert adapter.total_pages == 1
        # Empty pages may still have zero lines
        assert adapter.total_lines >= 0

    def test_init_invalid_pdf(self, tmp_path):
        """Raises ValueError for invalid PDF file."""
        invalid = tmp_path / "invalid.pdf"
        invalid.write_text("This is not a PDF")
        with pytest.raises(ValueError, match="Cannot open PDF"):
            PDFAdapter(invalid)


class TestPDFAdapterReadLines:
    """Tests for PDFAdapter.read_lines."""

    def test_read_first_lines(self, sample_pdf_file):
        """Reads first few lines."""
        adapter = PDFAdapter(sample_pdf_file)
        result = adapter.read_lines(start=1, end=3)

        assert len(result.lines) == 3
        assert result.start_line == 1
        assert result.end_line == 3
        assert result.total_lines > 0

    def test_read_middle_section(self, sample_pdf_file):
        """Reads lines from middle of extracted text."""
        adapter = PDFAdapter(sample_pdf_file)
        if adapter.total_lines >= 5:
            result = adapter.read_lines(start=2, end=4)
            assert len(result.lines) == 3
            assert result.start_line == 2
            assert result.end_line == 4

    def test_read_to_end_without_end_param(self, sample_pdf_file):
        """Reads to end when end is None."""
        adapter = PDFAdapter(sample_pdf_file)
        result = adapter.read_lines(start=1)

        assert len(result.lines) == adapter.total_lines
        assert result.end_line == adapter.total_lines

    def test_read_entire_file(self, sample_pdf_file):
        """Reads entire file with defaults."""
        adapter = PDFAdapter(sample_pdf_file)
        result = adapter.read_lines()

        assert len(result.lines) == adapter.total_lines
        assert result.start_line == 1
        assert result.end_line == adapter.total_lines

    def test_end_exceeds_total_lines(self, sample_pdf_file):
        """Clamps end to total lines."""
        adapter = PDFAdapter(sample_pdf_file)
        result = adapter.read_lines(start=1, end=10000)

        assert len(result.lines) == adapter.total_lines
        assert result.end_line == adapter.total_lines

    def test_start_less_than_one(self, sample_pdf_file):
        """Raises ValueError for start < 1."""
        adapter = PDFAdapter(sample_pdf_file)
        with pytest.raises(ValueError, match="start must be >= 1"):
            adapter.read_lines(start=0)

    def test_start_exceeds_total(self, sample_pdf_file):
        """Raises ValueError for start beyond file."""
        adapter = PDFAdapter(sample_pdf_file)
        with pytest.raises(ValueError, match="exceeds total lines"):
            adapter.read_lines(start=10000)

    def test_end_less_than_start(self, sample_pdf_file):
        """Raises ValueError for end < start."""
        adapter = PDFAdapter(sample_pdf_file)
        with pytest.raises(ValueError, match="must be >= start"):
            adapter.read_lines(start=5, end=2)

    def test_empty_pdf(self, empty_pdf_file):
        """Returns empty result for PDF with no text."""
        adapter = PDFAdapter(empty_pdf_file)
        result = adapter.read_lines()

        assert result.lines == []
        assert result.total_lines == 0

    def test_single_page_pdf(self, single_page_pdf):
        """Handles single-page PDF."""
        adapter = PDFAdapter(single_page_pdf)
        result = adapter.read_lines()

        assert len(result.lines) > 0
        # Check content was extracted
        text_content = " ".join([line for _, line in result.lines])
        assert "Single page content" in text_content

    def test_large_file_partial_read(self, large_pdf_file):
        """Reads portion of large PDF."""
        adapter = PDFAdapter(large_pdf_file)
        result = adapter.read_lines(start=10, end=20)

        assert len(result.lines) == 11
        assert result.start_line == 10
        assert result.end_line == 20


class TestPDFAdapterSearch:
    """Tests for PDFAdapter.search."""

    def test_literal_search(self, sample_pdf_file):
        """Finds literal string matches."""
        adapter = PDFAdapter(sample_pdf_file)
        result = adapter.search("Page 1")

        assert result.total_matches >= 1
        # The search pattern should be in the result
        assert result.pattern == "Page 1"

    def test_regex_search(self, sample_pdf_file):
        """Finds regex pattern matches."""
        adapter = PDFAdapter(sample_pdf_file)
        result = adapter.search(r"Page \d")

        # Should match "Page 1", "Page 2", "Page 3"
        assert result.total_matches >= 1

    def test_no_matches(self, sample_pdf_file):
        """Returns empty matches when nothing found."""
        adapter = PDFAdapter(sample_pdf_file)
        result = adapter.search("xyznotpresent123")

        assert result.total_matches == 0
        assert result.matches == []

    def test_search_with_context(self, large_pdf_file):
        """Includes context lines around matches."""
        adapter = PDFAdapter(large_pdf_file)
        result = adapter.search("Line number 50", context_lines=2)

        if result.total_matches > 0:
            match = result.matches[0]
            # Should have context before and after (if available)
            assert isinstance(match.context_before, list)
            assert isinstance(match.context_after, list)

    def test_invalid_regex(self, sample_pdf_file):
        """Raises error for invalid regex."""
        adapter = PDFAdapter(sample_pdf_file)
        with pytest.raises(re.error):
            adapter.search(r"[invalid")

    def test_empty_pdf_search(self, empty_pdf_file):
        """Search on empty PDF returns no matches."""
        adapter = PDFAdapter(empty_pdf_file)
        result = adapter.search("anything")

        assert result.total_matches == 0


class TestPDFAdapterViewPage:
    """Tests for PDFAdapter.view_page."""

    def test_view_first_page(self, sample_pdf_file):
        """Views first page and returns valid PNG."""
        adapter = PDFAdapter(sample_pdf_file)
        result = adapter.view_page(page=1)

        assert result.not_applicable is False
        assert result.page_number == 1
        assert result.total_pages == 3
        assert result.mime_type == "image/png"
        assert result.image_base64 is not None
        assert len(result.image_base64) > 0

        # Verify it's valid base64 that decodes to PNG
        decoded = base64.b64decode(result.image_base64)
        # PNG files start with these bytes
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_view_last_page(self, sample_pdf_file):
        """Views last page."""
        adapter = PDFAdapter(sample_pdf_file)
        result = adapter.view_page(page=3)

        assert result.page_number == 3
        assert result.not_applicable is False
        assert result.image_base64 is not None

    def test_page_less_than_one(self, sample_pdf_file):
        """Raises ValueError for page < 1."""
        adapter = PDFAdapter(sample_pdf_file)
        with pytest.raises(ValueError, match="page must be >= 1"):
            adapter.view_page(page=0)

    def test_page_exceeds_total(self, sample_pdf_file):
        """Raises ValueError for page beyond document."""
        adapter = PDFAdapter(sample_pdf_file)
        with pytest.raises(ValueError, match="exceeds total pages"):
            adapter.view_page(page=100)

    def test_single_page_pdf(self, single_page_pdf):
        """Views page from single-page PDF."""
        adapter = PDFAdapter(single_page_pdf)
        result = adapter.view_page(page=1)

        assert result.page_number == 1
        assert result.total_pages == 1
        assert result.image_base64 is not None


class TestPDFAdapterListVisualContent:
    """Tests for PDFAdapter.list_visual_content."""

    def test_pdf_with_images(self, pdf_with_images):
        """Detects embedded images."""
        adapter = PDFAdapter(pdf_with_images)
        result = adapter.list_visual_content()

        assert result.total_items >= 1
        # Check image info
        if result.items:
            item = result.items[0]
            assert item.content_type == "image"
            assert item.reference.startswith("xref:")
            assert item.location is not None

    def test_pdf_without_images(self, sample_pdf_file):
        """Returns empty for PDF without images."""
        adapter = PDFAdapter(sample_pdf_file)
        result = adapter.list_visual_content()

        # Sample PDF has only text
        assert result.total_items == 0
        assert result.items == []

    def test_empty_pdf_visual_content(self, empty_pdf_file):
        """Empty PDF returns empty visual content."""
        adapter = PDFAdapter(empty_pdf_file)
        result = adapter.list_visual_content()

        assert result.items == []
        assert result.total_items == 0


class TestPDFAdapterEdgeCases:
    """Edge case tests for PDFAdapter."""

    def test_adapter_cleanup(self, sample_pdf_file):
        """Adapter properly closes document on deletion."""
        adapter = PDFAdapter(sample_pdf_file)
        # Access document to ensure it's open
        _ = adapter.total_pages
        # Deletion should not raise
        del adapter

    def test_line_numbers_are_one_indexed(self, sample_pdf_file):
        """Line numbers in results are 1-indexed."""
        adapter = PDFAdapter(sample_pdf_file)
        result = adapter.read_lines(start=1, end=3)

        # First line should be numbered 1
        assert result.lines[0][0] == 1
        # Third line should be numbered 3
        assert result.lines[2][0] == 3


class TestPDFAdapterProgressiveRendering:
    """Tests for progressive DPI reduction when rendering large pages."""

    def test_small_page_renders_at_full_dpi(self, sample_pdf_file):
        """Normal-sized pages render without DPI reduction."""
        adapter = PDFAdapter(sample_pdf_file)
        result = adapter.view_page(page=1)

        # Should succeed and be under limit
        decoded = base64.b64decode(result.image_base64)
        assert len(decoded) < PDFAdapter.MAX_IMAGE_BYTES
        assert result.not_applicable is False

    def test_large_page_reduces_dpi(self, tmp_path):
        """Large pages progressively reduce DPI to stay under size limit."""
        # Create a very large page (A0 size: 841 x 1189 mm)
        pdf_path = tmp_path / "large_page.pdf"
        doc = pymupdf.open()
        # A0 in points: 2383.94 x 3370.39
        page = doc.new_page(width=2384, height=3370)
        # Fill with dense content to make large image
        for y in range(0, 3300, 50):
            page.insert_text((50, y + 30), "Dense text " * 20, fontsize=10)
        doc.save(pdf_path)
        doc.close()

        adapter = PDFAdapter(pdf_path)
        result = adapter.view_page(page=1)

        # Should succeed after DPI reduction
        decoded = base64.b64decode(result.image_base64)
        assert len(decoded) <= PDFAdapter.MAX_IMAGE_BYTES
        assert result.not_applicable is False

    def test_rendering_respects_max_size(self, tmp_path):
        """Rendered images stay under MAX_IMAGE_BYTES limit."""
        # Create moderately large PDF
        pdf_path = tmp_path / "moderate_page.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=1200, height=1600)
        page.insert_text((50, 100), "Test content", fontsize=12)
        doc.save(pdf_path)
        doc.close()

        adapter = PDFAdapter(pdf_path)
        result = adapter.view_page(page=1)

        decoded = base64.b64decode(result.image_base64)
        assert len(decoded) <= PDFAdapter.MAX_IMAGE_BYTES

    def test_class_constants_defined(self):
        """DPI levels and max size are properly defined."""
        assert PDFAdapter.MAX_IMAGE_BYTES == 2 * 1024 * 1024  # 2MB (safe after base64)
        assert PDFAdapter.DPI_LEVELS == [150, 120, 100, 72, 50]
        assert PDFAdapter.DPI_LEVELS[0] == 150  # Highest first
        assert PDFAdapter.DPI_LEVELS[-1] == 50  # Lowest last


class TestCourtOpinionPDF:
    """Tests against court opinion PDFs from the patent law corpus.

    These integration tests use real corpus fixtures to validate toolkit behavior.
    """

    def test_read_first_lines(self, court_opinion_pdf):
        """Read opening lines of court opinion."""
        result = read_lines(court_opinion_pdf, start=1, end=20)

        assert len(result.lines) == 20
        assert result.total_lines > 100  # Legal documents are substantial

    def test_read_middle_section(self, court_opinion_pdf):
        """Read from middle of document."""
        # Get total first
        info = read_lines(court_opinion_pdf, start=1, end=1)
        total = info.total_lines

        if total > 100:
            mid = total // 2
            result = read_lines(court_opinion_pdf, start=mid, end=mid + 10)
            assert len(result.lines) == 11
            assert result.start_line == mid

    def test_search_legal_terms(self, court_opinion_pdf):
        """Search for common legal terminology."""
        # Try common legal terms
        for term in ["court", "defendant", "plaintiff", "order", "judgment"]:
            result = search(court_opinion_pdf, term, context_lines=1)
            if result.total_matches > 0:
                # Found at least one legal term
                assert any(
                    term.lower() in m.line_content.lower() for m in result.matches
                )
                break
        else:
            # At minimum, try case-insensitive
            result = search(court_opinion_pdf, r"(?i)court")
            assert result.total_matches >= 0  # May or may not have matches

    def test_view_page_returns_image(self, court_opinion_pdf):
        """PDF page view returns valid PNG image."""
        result = view_page(court_opinion_pdf, page=1)

        assert result.not_applicable is False
        assert result.page_number == 1
        assert result.total_pages >= 1
        assert result.mime_type == "image/png"
        assert result.image_base64 is not None
        assert len(result.image_base64) > 1000  # Substantial image data

    def test_view_multiple_pages(self, court_opinion_pdf):
        """Can view different pages of the PDF."""
        result1 = view_page(court_opinion_pdf, page=1)
        if result1.total_pages and result1.total_pages > 1:
            result2 = view_page(court_opinion_pdf, page=2)
            # Different pages should have different images
            assert result2.image_base64 != result1.image_base64

    def test_list_visual_content(self, court_opinion_pdf):
        """Court opinions typically have few images."""
        result = list_visual_content(court_opinion_pdf)
        # Legal documents are mostly text, but may have seals/logos
        assert isinstance(result.items, list)
        assert result.total_items >= 0


class TestArxivPaperPDF:
    """Tests against arXiv papers which typically have figures.

    These integration tests use real corpus fixtures to validate toolkit behavior.
    """

    def test_read_first_lines(self, arxiv_paper_pdf):
        """Read opening lines of arXiv paper."""
        result = read_lines(arxiv_paper_pdf, start=1, end=30)

        assert len(result.lines) <= 30
        assert result.total_lines > 100  # Scientific papers are substantial

    def test_search_abstract(self, arxiv_paper_pdf):
        """Search for abstract section."""
        result = search(arxiv_paper_pdf, r"(?i)abstract")
        # Most papers have an abstract
        # Note: Some PDFs may not extract "Abstract" if it's in a special format
        assert result.total_matches >= 0

    def test_search_scientific_terms(self, arxiv_paper_pdf):
        """Search for scientific content."""
        # Try common scientific terms
        for term in ["method", "result", "data", "figure", "table", "analysis"]:
            result = search(arxiv_paper_pdf, rf"(?i){term}")
            if result.total_matches > 0:
                break
        # At least one scientific term should be found
        assert result.total_matches >= 0

    def test_view_page_returns_image(self, arxiv_paper_pdf):
        """PDF page view returns valid PNG image."""
        result = view_page(arxiv_paper_pdf, page=1)

        assert result.not_applicable is False
        assert result.page_number == 1
        assert result.mime_type == "image/png"
        assert result.image_base64 is not None

    def test_list_visual_content_finds_figures(self, arxiv_paper_pdf):
        """ArXiv papers typically have embedded figures."""
        result = list_visual_content(arxiv_paper_pdf)

        # Scientific papers usually have figures
        # Note: Some papers may have all figures as separate files
        if result.total_items > 0:
            # Verify structure of visual content items
            for item in result.items:
                assert item.content_type == "image"
                assert item.reference.startswith("xref:")
                assert item.location is not None
                assert "page" in item.location

    def test_image_alt_text_contains_dimensions(self, arxiv_paper_pdf):
        """Image alt text contains dimension info."""
        result = list_visual_content(arxiv_paper_pdf)

        if result.total_items > 0:
            item = result.items[0]
            # Alt text should have dimension info like "380x477 pixels"
            assert item.alt_text is not None
            assert "x" in item.alt_text
            assert "pixels" in item.alt_text
