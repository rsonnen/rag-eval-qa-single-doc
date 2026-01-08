"""Abstract interface for document format adapters.

This module defines the protocol that format-specific adapters must implement.
Currently supports text and markdown; PDF and XML adapters can be added later
by implementing this interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from single_doc_generator.models import (
    ReadLinesResult,
    SearchResult,
    ViewPageResult,
    VisualContentResult,
)


class DocumentAdapter(ABC):
    """Abstract base class for document format adapters.

    Each format (text, markdown, PDF, XML) implements this interface to provide
    consistent document exploration tools regardless of underlying format.

    Attributes:
        file_path: Path to the document being adapted.
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize adapter with document path.

        Args:
            file_path: Path to the document file.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file is empty.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        self.file_path = file_path

    @abstractmethod
    def read_lines(self, start: int = 1, end: int | None = None) -> ReadLinesResult:
        """Read a range of lines from the document.

        Args:
            start: Starting line number (1-indexed, inclusive).
            end: Ending line number (1-indexed, inclusive). If None, reads to EOF.

        Returns:
            ReadLinesResult with lines, total count, and range info.
        """

    @abstractmethod
    def search(self, pattern: str, context_lines: int = 0) -> SearchResult:
        """Search for regex pattern in document.

        Args:
            pattern: Regular expression pattern to search for.
            context_lines: Number of lines to include before/after each match.

        Returns:
            SearchResult with matches and context.
        """

    @abstractmethod
    def view_page(self, page: int) -> ViewPageResult:
        """View a specific page as an image.

        For text-based formats without pagination, returns not_applicable=True.
        For paginated formats (PDF), returns base64-encoded page image.

        Args:
            page: Page number to view (1-indexed).

        Returns:
            ViewPageResult with image data or not_applicable flag.
        """

    @abstractmethod
    def list_visual_content(self) -> VisualContentResult:
        """List visual content in the document.

        Discovers figures, tables, images, and other visual elements.
        For plain text, returns empty list.
        For markdown, parses image references.
        For PDF/XML, extracts embedded figures and tables.

        Returns:
            VisualContentResult with list of visual elements.
        """
