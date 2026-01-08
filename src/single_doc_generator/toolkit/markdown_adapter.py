"""Markdown file adapter for document toolkit.

This adapter handles markdown (.md, .markdown) files. It provides line-based
reading, regex search, and parses markdown image references for visual content.
"""

import re
from pathlib import Path

from single_doc_generator.models import (
    ReadLinesResult,
    SearchMatch,
    SearchResult,
    ViewPageResult,
    VisualContent,
    VisualContentResult,
)
from single_doc_generator.toolkit.base import DocumentAdapter

# Regex for markdown image syntax: ![alt text](path/to/image.png)
# Captures: group 1 = alt text, group 2 = image path
IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


class MarkdownAdapter(DocumentAdapter):
    """Adapter for markdown files.

    Provides document exploration tools for .md and .markdown files.
    Markdown is treated as line-based text for reading and searching.
    Visual content detection parses markdown image syntax.
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize markdown adapter and load file content.

        Args:
            file_path: Path to the markdown file.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        super().__init__(file_path)
        self._lines: list[str] = []
        self._load_content()

    def _load_content(self) -> None:
        """Load and cache file content as lines."""
        with self.file_path.open(encoding="utf-8") as f:
            self._lines = f.read().splitlines()

    @property
    def total_lines(self) -> int:
        """Total number of lines in document."""
        return len(self._lines)

    def read_lines(self, start: int = 1, end: int | None = None) -> ReadLinesResult:
        """Read a range of lines from the markdown file.

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
        """Search for regex pattern in markdown file.

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
        """View page is not applicable for markdown files.

        Markdown files have no page structure, so this always returns
        a not_applicable result.

        Args:
            page: Page number (ignored for markdown files).

        Returns:
            ViewPageResult with not_applicable=True.
        """
        return ViewPageResult(not_applicable=True, page_number=page)

    def list_visual_content(self) -> VisualContentResult:
        """List visual content by parsing markdown image references.

        Parses markdown image syntax: ![alt text](path/to/image.png)
        Most markdown files don't contain images, so this typically
        returns an empty list.

        Returns:
            VisualContentResult with any image references found.
        """
        items: list[VisualContent] = []

        for i, line in enumerate(self._lines):
            for match in IMAGE_PATTERN.finditer(line):
                alt_text = match.group(1) or None
                image_path = match.group(2)

                items.append(
                    VisualContent(
                        content_type="image",
                        reference=image_path,
                        alt_text=alt_text if alt_text else None,
                        location=f"line {i + 1}",
                    )
                )

        return VisualContentResult(items=items, total_items=len(items))
