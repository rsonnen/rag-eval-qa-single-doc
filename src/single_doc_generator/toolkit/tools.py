"""Document exploration tools for agent use.

This module provides the four document tools as standalone functions.
Each function takes a file path and delegates to the appropriate format adapter.
These will become @tool decorated functions for LangChain integration later.

Tools:
    read_lines: Read a range of lines with line numbers
    search: Regex search returning matches with context
    view_page: Render page as image (format-dependent)
    list_visual_content: Discover images, figures, tables
"""

import logging
from pathlib import Path

from single_doc_generator.models import (
    DocumentFormat,
    ReadLinesResult,
    SearchResult,
    ViewPageResult,
    VisualContentResult,
    detect_format,
)
from single_doc_generator.toolkit.adoc_adapter import AdocAdapter
from single_doc_generator.toolkit.base import DocumentAdapter
from single_doc_generator.toolkit.html_adapter import HTMLAdapter
from single_doc_generator.toolkit.markdown_adapter import MarkdownAdapter
from single_doc_generator.toolkit.pdf_adapter import PDFAdapter
from single_doc_generator.toolkit.text_adapter import TextAdapter
from single_doc_generator.toolkit.xml_adapter import XMLAdapter

logger = logging.getLogger(__name__)


def _get_adapter(file_path: Path) -> DocumentAdapter:
    """Get the appropriate adapter for a file based on its format.

    Args:
        file_path: Path to the document file.

    Returns:
        DocumentAdapter instance for the file's format.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is not supported.
    """
    doc_format = detect_format(file_path)

    if doc_format == DocumentFormat.TEXT:
        return TextAdapter(file_path)
    if doc_format == DocumentFormat.MARKDOWN:
        return MarkdownAdapter(file_path)
    if doc_format == DocumentFormat.PDF:
        return PDFAdapter(file_path)
    if doc_format == DocumentFormat.XML:
        return XMLAdapter(file_path)
    if doc_format == DocumentFormat.HTML:
        return HTMLAdapter(file_path)
    if doc_format == DocumentFormat.ADOC:
        return AdocAdapter(file_path)

    # This shouldn't happen if detect_format works correctly
    raise ValueError(f"No adapter for format: {doc_format}")


def read_lines(
    file_path: str | Path,
    start: int = 1,
    end: int | None = None,
) -> ReadLinesResult:
    """Read a range of lines from a document.

    Returns lines with their line numbers, along with total document length.
    Line numbers are 1-indexed to match text editor conventions.

    Args:
        file_path: Path to the document file (str or Path).
        start: Starting line number (1-indexed, inclusive). Defaults to 1.
        end: Ending line number (1-indexed, inclusive). If None, reads to EOF.

    Returns:
        ReadLinesResult containing:
            - lines: List of (line_number, content) tuples
            - total_lines: Total lines in document
            - start_line: First line returned
            - end_line: Last line returned

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If start < 1 or start > total_lines, or format unsupported.

    Example:
        >>> result = read_lines("/path/to/doc.txt", start=1, end=10)
        >>> for line_num, content in result.lines:
        ...     print(f"{line_num}: {content}")
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    logger.debug("read_lines: %s, start=%d, end=%s", path, start, end)

    adapter = _get_adapter(path)
    return adapter.read_lines(start=start, end=end)


def search(
    file_path: str | Path,
    pattern: str,
    context_lines: int = 0,
) -> SearchResult:
    r"""Search for a regex pattern in a document.

    Returns all lines matching the pattern, optionally with surrounding context.
    Useful for discovering specific content without reading the entire document.

    Args:
        file_path: Path to the document file (str or Path).
        pattern: Regular expression pattern to search for.
        context_lines: Number of lines to include before/after each match.

    Returns:
        SearchResult containing:
            - pattern: The search pattern used
            - matches: List of SearchMatch objects with line content and context
            - total_matches: Count of matches found

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If format is unsupported.
        re.error: If pattern is invalid regex.

    Example:
        >>> result = search("/path/to/doc.txt", r"\\bfunction\\b", context_lines=2)
        >>> for match in result.matches:
        ...     print(f"Line {match.line_number}: {match.line_content}")
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    logger.debug("search: %s, pattern=%s, context=%d", path, pattern, context_lines)

    adapter = _get_adapter(path)
    return adapter.search(pattern=pattern, context_lines=context_lines)


def view_page(file_path: str | Path, page: int) -> ViewPageResult:
    """View a specific page of a document as an image.

    For paginated formats (PDF), renders the page as a base64-encoded image.
    For text-based formats (txt, markdown), returns not_applicable=True
    since these formats have no inherent page structure.

    Args:
        file_path: Path to the document file (str or Path).
        page: Page number to view (1-indexed).

    Returns:
        ViewPageResult containing:
            - not_applicable: True for text formats without page structure
            - page_number: The requested page number
            - image_base64: Base64 image data (for paginated formats)
            - mime_type: Image MIME type (for paginated formats)

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If format is unsupported or page is out of range.

    Example:
        >>> result = view_page("/path/to/doc.pdf", page=1)
        >>> if result.not_applicable:
        ...     print("This format doesn't support page viewing")
        >>> else:
        ...     # Use result.image_base64 with a vision model
        ...     pass
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    logger.debug("view_page: %s, page=%d", path, page)

    adapter = _get_adapter(path)
    return adapter.view_page(page=page)


def list_visual_content(file_path: str | Path) -> VisualContentResult:
    """List visual content elements in a document.

    Discovers figures, tables, images, and other visual elements.
    What gets detected depends on the document format:
        - Plain text: Always returns empty list (no visual content)
        - Markdown: Parses ![alt](path) image references
        - PDF: Extracts embedded images from PDF structure

    Args:
        file_path: Path to the document file (str or Path).

    Returns:
        VisualContentResult containing:
            - items: List of VisualContent objects with type, reference, alt text
            - total_items: Count of visual elements found

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If format is unsupported.

    Example:
        >>> result = list_visual_content("/path/to/doc.md")
        >>> for item in result.items:
        ...     print(f"{item.content_type}: {item.reference}")
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    logger.debug("list_visual_content: %s", path)

    adapter = _get_adapter(path)
    return adapter.list_visual_content()
