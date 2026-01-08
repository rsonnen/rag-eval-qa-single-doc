"""PDF file adapter for document toolkit.

This adapter handles PDF files using PyMuPDF (fitz). It provides line-based text
extraction, regex search, page rendering as images, and visual content discovery.
"""

import base64
import re
from pathlib import Path
from typing import ClassVar

import pymupdf

from single_doc_generator.models import (
    ReadLinesResult,
    SearchMatch,
    SearchResult,
    ViewPageResult,
    VisualContent,
    VisualContentResult,
)
from single_doc_generator.toolkit.base import DocumentAdapter


class PDFAdapter(DocumentAdapter):
    """Adapter for PDF files.

    Provides document exploration tools for PDF files. Text is extracted and
    split into lines for the read_lines and search tools. Visual features
    (view_page, list_visual_content) are fully supported for PDFs.

    Attributes:
        file_path: Path to the PDF file.
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize PDF adapter and extract text content.

        Args:
            file_path: Path to the PDF file.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file cannot be opened as PDF.
        """
        # Initialize attributes before super().__init__ in case it raises
        self._doc: pymupdf.Document | None = None
        self._lines: list[str] = []
        super().__init__(file_path)
        self._load_content()

    def _load_content(self) -> None:
        """Load PDF and extract text content as lines.

        Raises:
            ValueError: If PDF cannot be opened or is encrypted.
        """
        try:
            self._doc = pymupdf.open(self.file_path)
        except Exception as e:
            raise ValueError(f"Cannot open PDF: {e}") from e

        if self._doc.is_encrypted:
            self._doc.close()
            self._doc = None
            raise ValueError("Cannot open encrypted PDF")

        # Extract text from all pages and split into lines
        all_text: list[str] = []
        for page in self._doc:
            page_text = page.get_text()
            all_text.append(page_text)

        # Join all page text and split into lines
        full_text = "\n".join(all_text)
        self._lines = full_text.splitlines()

    def __del__(self) -> None:
        """Close the PDF document on cleanup."""
        if self._doc is not None:
            self._doc.close()

    @property
    def total_lines(self) -> int:
        """Total number of lines in extracted text."""
        return len(self._lines)

    @property
    def total_pages(self) -> int:
        """Total number of pages in the PDF."""
        if self._doc is None:
            return 0
        return len(self._doc)

    def read_lines(self, start: int = 1, end: int | None = None) -> ReadLinesResult:
        """Read a range of lines from the extracted PDF text.

        Args:
            start: Starting line number (1-indexed, inclusive). Defaults to 1.
            end: Ending line number (1-indexed, inclusive). If None, reads to EOF.

        Returns:
            ReadLinesResult containing the requested lines with line numbers.

        Raises:
            ValueError: If start < 1 or start > total_lines.
        """
        if start < 1:
            raise ValueError(f"start must be >= 1, got {start}")
        if end is not None and end < start:
            raise ValueError(f"end ({end}) must be >= start ({start})")
        if start > self.total_lines and self.total_lines > 0:
            raise ValueError(f"start {start} exceeds total lines {self.total_lines}")

        # Handle empty PDF
        if self.total_lines == 0:
            return ReadLinesResult(
                lines=[],
                total_lines=0,
                start_line=start,
                end_line=start,
            )

        # Determine actual end line
        actual_end = end if end is not None else self.total_lines
        actual_end = min(actual_end, self.total_lines)

        # Convert to 0-indexed for slicing
        start_idx = start - 1
        end_idx = actual_end

        lines_with_numbers = [
            (i + 1, self._lines[i]) for i in range(start_idx, end_idx)
        ]

        return ReadLinesResult(
            lines=lines_with_numbers,
            total_lines=self.total_lines,
            start_line=start,
            end_line=actual_end,
        )

    def search(self, pattern: str, context_lines: int = 0) -> SearchResult:
        """Search for regex pattern in extracted PDF text.

        Args:
            pattern: Regular expression pattern to search for.
            context_lines: Number of lines to include before/after each match.

        Returns:
            SearchResult with all matches and their context.

        Raises:
            re.error: If pattern is invalid regex.
        """
        regex = re.compile(pattern)
        matches: list[SearchMatch] = []

        for i, line in enumerate(self._lines):
            if regex.search(line):
                line_num = i + 1  # 1-indexed

                # Gather context before
                context_before: list[tuple[int, str]] = []
                for j in range(max(0, i - context_lines), i):
                    context_before.append((j + 1, self._lines[j]))

                # Gather context after
                context_after: list[tuple[int, str]] = []
                for j in range(i + 1, min(len(self._lines), i + 1 + context_lines)):
                    context_after.append((j + 1, self._lines[j]))

                matches.append(
                    SearchMatch(
                        line_number=line_num,
                        line_content=line,
                        context_before=context_before,
                        context_after=context_after,
                    )
                )

        return SearchResult(
            pattern=pattern,
            matches=matches,
            total_matches=len(matches),
        )

    # Maximum image size in bytes (2MB → ~2.7MB base64, safe under 5MB API limit)
    # Note: base64 encoding adds ~33% overhead, and we need room for prompts/messages
    MAX_IMAGE_BYTES: ClassVar[int] = 2 * 1024 * 1024
    # DPI values to try, from highest quality to lowest acceptable
    DPI_LEVELS: ClassVar[list[int]] = [150, 120, 100, 72, 50]

    def view_page(self, page: int) -> ViewPageResult:
        """Render a PDF page as a PNG image.

        Renders at 150 DPI initially, then progressively reduces DPI if the
        resulting image exceeds 4MB (to stay safely under the 5MB API limit).

        Args:
            page: Page number to view (1-indexed).

        Returns:
            ViewPageResult with base64-encoded PNG image data.

        Raises:
            ValueError: If page number is out of range or image cannot be
                rendered under the size limit.
        """
        if self._doc is None:
            raise ValueError("PDF document not loaded")

        if page < 1:
            raise ValueError(f"page must be >= 1, got {page}")
        if page > self.total_pages:
            raise ValueError(f"page {page} exceeds total pages {self.total_pages}")

        # Get the page (0-indexed in PyMuPDF)
        pdf_page = self._doc[page - 1]

        # Try rendering at progressively lower DPI until under size limit
        for dpi in self.DPI_LEVELS:
            zoom = dpi / 72  # Default matrix is 72 DPI
            matrix = pymupdf.Matrix(zoom, zoom)
            pixmap = pdf_page.get_pixmap(matrix=matrix)

            png_bytes = pixmap.tobytes("png")

            if len(png_bytes) <= self.MAX_IMAGE_BYTES:
                image_base64 = base64.b64encode(png_bytes).decode("ascii")
                return ViewPageResult(
                    not_applicable=False,
                    page_number=page,
                    total_pages=self.total_pages,
                    image_base64=image_base64,
                    mime_type="image/png",
                )

        # If even lowest DPI exceeds limit, raise error
        max_mb = self.MAX_IMAGE_BYTES // (1024 * 1024)
        min_dpi = self.DPI_LEVELS[-1]
        raise ValueError(
            f"Page {page} cannot be rendered under {max_mb}MB "
            f"even at {min_dpi} DPI ({len(png_bytes)} bytes)"
        )

    def list_visual_content(self) -> VisualContentResult:
        """List visual content (images and tables) in the PDF.

        Detects embedded images using PyMuPDF's image extraction.
        Note: Table detection is limited to identifying potential table regions
        based on repeated patterns or explicit table structures.

        Returns:
            VisualContentResult with list of images found.
        """
        items: list[VisualContent] = []

        if self._doc is None:
            return VisualContentResult(items=items, total_items=0)

        for page_num, page in enumerate(self._doc, start=1):
            # Get list of images on this page
            # Returns list of tuples: (xref, smask, width, height, bpc, colorspace, ...)
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list, start=1):
                xref = img_info[0]
                width = img_info[2]
                height = img_info[3]

                items.append(
                    VisualContent(
                        content_type="image",
                        reference=f"xref:{xref}",
                        alt_text=f"{width}x{height} pixels",
                        location=f"page {page_num}, image {img_idx}",
                    )
                )

        return VisualContentResult(items=items, total_items=len(items))
