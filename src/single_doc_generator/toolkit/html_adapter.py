"""HTML file adapter for document toolkit.

This adapter handles HTML (.html) documentation files. It provides line-based
reading and regex search on raw HTML content (preserving line numbers), and
parses HTML to discover visual content like images, figures, and tables.

The adapter is designed for HTML documentation files (like Django docs or Python
stdlib docs) where the HTML structure contains both navigation/boilerplate and
the actual documentation content.
"""

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from single_doc_generator.models import (
    ReadLinesResult,
    SearchMatch,
    SearchResult,
    ViewPageResult,
    VisualContent,
    VisualContentResult,
)
from single_doc_generator.toolkit.base import DocumentAdapter


class HTMLAdapter(DocumentAdapter):
    """Adapter for HTML documentation files.

    Provides document exploration tools for .html files. Line-based operations
    (read_lines, search) work on raw HTML to preserve line numbers. Visual
    content detection parses HTML to find images, figures, and tables.

    This adapter is optimized for documentation HTML files which typically
    contain code blocks, tables, and embedded images alongside text content.
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize HTML adapter and load file content.

        Args:
            file_path: Path to the HTML file.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        super().__init__(file_path)
        self._lines: list[str] = []
        self._soup: BeautifulSoup | None = None
        self._load_content()

    def _load_content(self) -> None:
        """Load and cache file content as lines and parsed HTML."""
        with self.file_path.open(encoding="utf-8") as f:
            content = f.read()
        self._lines = content.splitlines()
        # Parse HTML with html.parser (stdlib, no external deps)
        self._soup = BeautifulSoup(content, "html.parser")

    @property
    def total_lines(self) -> int:
        """Total number of lines in document."""
        return len(self._lines)

    def read_lines(self, start: int = 1, end: int | None = None) -> ReadLinesResult:
        """Read a range of lines from the HTML file.

        Returns raw HTML lines to preserve line numbers for reference.
        This allows the LLM to cite specific line numbers when generating
        questions about the document.

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

        # Handle empty file
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
        """Search for regex pattern in HTML file.

        Searches raw HTML content to allow finding both text content and
        HTML structure. Returns line numbers that can be used with read_lines.

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

    def view_page(self, page: int) -> ViewPageResult:
        """View page is not applicable for HTML files.

        HTML files have no page structure, so this always returns
        a not_applicable result.

        Args:
            page: Page number (ignored for HTML files).

        Returns:
            ViewPageResult with not_applicable=True.
        """
        return ViewPageResult(not_applicable=True, page_number=page)

    def list_visual_content(self) -> VisualContentResult:
        """List visual content by parsing HTML elements.

        Parses the HTML document to find:
        - <img> tags (images with src and alt attributes)
        - <figure> elements (typically containing images with captions)
        - <table> elements (data tables in documentation)

        Returns:
            VisualContentResult with visual elements found.
        """
        if self._soup is None:
            return VisualContentResult(items=[], total_items=0)

        items: list[VisualContent] = []
        items.extend(self._extract_images())
        items.extend(self._extract_figures())
        items.extend(self._extract_tables())

        return VisualContentResult(items=items, total_items=len(items))

    def _extract_images(self) -> list[VisualContent]:
        """Extract image elements from HTML.

        Finds all <img> tags and extracts src and alt attributes.

        Returns:
            List of VisualContent items for images.
        """
        items: list[VisualContent] = []
        if self._soup is None:
            return items

        for img in self._soup.find_all("img"):
            if not isinstance(img, Tag):
                continue
            src = img.get("src", "")
            alt = img.get("alt", "")
            src_str = src if isinstance(src, str) else (src[0] if src else "")
            alt_str = alt if isinstance(alt, str) else (alt[0] if alt else "")

            if src_str:
                location = self._find_element_location(f'src="{src_str}"')
                items.append(
                    VisualContent(
                        content_type="image",
                        reference=src_str,
                        alt_text=alt_str if alt_str else None,
                        location=location,
                    )
                )
        return items

    def _extract_figures(self) -> list[VisualContent]:
        """Extract figure elements from HTML.

        Finds all <figure> tags, extracting figcaption text and
        any nested image src.

        Returns:
            List of VisualContent items for figures.
        """
        items: list[VisualContent] = []
        if self._soup is None:
            return items

        for figure in self._soup.find_all("figure"):
            if not isinstance(figure, Tag):
                continue

            caption = self._get_figcaption_text(figure)
            reference = self._get_figure_image_src(figure)

            if reference or caption:
                location = self._find_element_location("<figure")
                items.append(
                    VisualContent(
                        content_type="figure",
                        reference=reference or "figure",
                        alt_text=caption,
                        location=location,
                    )
                )
        return items

    def _get_figcaption_text(self, figure: Tag) -> str | None:
        """Extract text from a figure's figcaption element."""
        figcaption = figure.find("figcaption")
        if figcaption and isinstance(figcaption, Tag):
            return figcaption.get_text(strip=True)
        return None

    def _get_figure_image_src(self, figure: Tag) -> str:
        """Extract src attribute from a figure's nested img element."""
        fig_img = figure.find("img")
        if fig_img and isinstance(fig_img, Tag):
            src = fig_img.get("src", "")
            return src if isinstance(src, str) else (src[0] if src else "")
        return ""

    def _extract_tables(self) -> list[VisualContent]:
        """Extract table elements from HTML.

        Finds all <table> tags, extracting caption text and id/class
        for reference.

        Returns:
            List of VisualContent items for tables.
        """
        items: list[VisualContent] = []
        if self._soup is None:
            return items

        for table in self._soup.find_all("table"):
            if not isinstance(table, Tag):
                continue

            caption = self._get_table_caption(table)
            reference = self._get_table_reference(table)
            location = self._find_element_location("<table")

            items.append(
                VisualContent(
                    content_type="table",
                    reference=reference,
                    alt_text=caption,
                    location=location,
                )
            )
        return items

    def _get_table_caption(self, table: Tag) -> str | None:
        """Extract text from a table's caption element."""
        caption_elem = table.find("caption")
        if caption_elem and isinstance(caption_elem, Tag):
            return caption_elem.get_text(strip=True)
        return None

    def _get_table_reference(self, table: Tag) -> str:
        """Generate a reference string from table's id or class attributes."""
        table_id = table.get("id", "")
        table_class = table.get("class", [])
        id_str = table_id if isinstance(table_id, str) else ""
        class_str = (
            " ".join(table_class) if isinstance(table_class, list) else table_class
        )
        return id_str or class_str or "table"

    def _find_element_location(self, search_text: str) -> str | None:
        """Find approximate line number of an element by searching raw content.

        Args:
            search_text: Text snippet to search for in raw HTML.

        Returns:
            Location string like "line 42" or None if not found.
        """
        for i, line in enumerate(self._lines):
            if search_text in line:
                return f"line {i + 1}"
        return None
