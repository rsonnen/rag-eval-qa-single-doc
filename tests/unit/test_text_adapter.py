"""Tests for TextAdapter."""

import re

import pytest

from single_doc_generator.toolkit import (
    list_visual_content,
    read_lines,
    search,
    view_page,
)
from single_doc_generator.toolkit.text_adapter import TextAdapter


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file with 10 lines."""
    content = "\n".join([f"Line {i}" for i in range(1, 11)])
    file = tmp_path / "sample.txt"
    file.write_text(content)
    return file


@pytest.fixture
def empty_text_file(tmp_path):
    """Create an empty text file."""
    file = tmp_path / "empty.txt"
    file.write_text("")
    return file


@pytest.fixture
def single_line_file(tmp_path):
    """Create a file with a single line."""
    file = tmp_path / "single.txt"
    file.write_text("Only one line here")
    return file


@pytest.fixture
def large_text_file(tmp_path):
    """Create a file with 1500 lines."""
    content = "\n".join([f"Line number {i}" for i in range(1, 1501)])
    file = tmp_path / "large.txt"
    file.write_text(content)
    return file


class TestTextAdapterInit:
    """Tests for TextAdapter initialization."""

    def test_init_with_valid_file(self, sample_text_file):
        """Initializes with valid file path."""
        adapter = TextAdapter(sample_text_file)
        assert adapter.file_path == sample_text_file
        assert adapter.total_lines == 10

    def test_init_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        missing = tmp_path / "missing.txt"
        with pytest.raises(FileNotFoundError):
            TextAdapter(missing)

    def test_init_empty_file(self, empty_text_file):
        """Handles empty file."""
        adapter = TextAdapter(empty_text_file)
        assert adapter.total_lines == 0


class TestTextAdapterReadLines:
    """Tests for TextAdapter.read_lines."""

    def test_read_first_5_lines(self, sample_text_file):
        """Reads first 5 lines."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.read_lines(start=1, end=5)

        assert len(result.lines) == 5
        assert result.start_line == 1
        assert result.end_line == 5
        assert result.total_lines == 10
        assert result.lines[0] == (1, "Line 1")
        assert result.lines[4] == (5, "Line 5")

    def test_read_middle_section(self, sample_text_file):
        """Reads lines from middle of file."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.read_lines(start=4, end=7)

        assert len(result.lines) == 4
        assert result.lines[0] == (4, "Line 4")
        assert result.lines[3] == (7, "Line 7")

    def test_read_last_lines(self, sample_text_file):
        """Reads last few lines."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.read_lines(start=8, end=10)

        assert len(result.lines) == 3
        assert result.lines[0] == (8, "Line 8")
        assert result.lines[2] == (10, "Line 10")

    def test_read_to_end_without_end_param(self, sample_text_file):
        """Reads to end when end is None."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.read_lines(start=8)

        assert len(result.lines) == 3
        assert result.end_line == 10

    def test_read_entire_file(self, sample_text_file):
        """Reads entire file with defaults."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.read_lines()

        assert len(result.lines) == 10
        assert result.start_line == 1
        assert result.end_line == 10

    def test_end_exceeds_total_lines(self, sample_text_file):
        """Clamps end to total lines."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.read_lines(start=8, end=100)

        assert len(result.lines) == 3
        assert result.end_line == 10

    def test_start_less_than_one(self, sample_text_file):
        """Raises ValueError for start < 1."""
        adapter = TextAdapter(sample_text_file)
        with pytest.raises(ValueError, match="start must be >= 1"):
            adapter.read_lines(start=0)

    def test_start_exceeds_total(self, sample_text_file):
        """Raises ValueError for start beyond file."""
        adapter = TextAdapter(sample_text_file)
        with pytest.raises(ValueError, match="exceeds total lines"):
            adapter.read_lines(start=20)

    def test_empty_file(self, empty_text_file):
        """Returns empty result for empty file."""
        adapter = TextAdapter(empty_text_file)
        result = adapter.read_lines()

        assert result.lines == []
        assert result.total_lines == 0

    def test_single_line_file(self, single_line_file):
        """Handles single-line file."""
        adapter = TextAdapter(single_line_file)
        result = adapter.read_lines()

        assert len(result.lines) == 1
        assert result.lines[0] == (1, "Only one line here")

    def test_large_file_partial_read(self, large_text_file):
        """Reads portion of large file."""
        adapter = TextAdapter(large_text_file)
        result = adapter.read_lines(start=500, end=510)

        assert len(result.lines) == 11
        assert result.total_lines == 1500
        assert result.lines[0] == (500, "Line number 500")


class TestTextAdapterSearch:
    """Tests for TextAdapter.search."""

    def test_literal_search(self, sample_text_file):
        """Finds literal string matches."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.search("Line 5")

        assert result.total_matches == 1
        assert result.matches[0].line_number == 5
        assert result.matches[0].line_content == "Line 5"

    def test_regex_search(self, sample_text_file):
        """Finds regex pattern matches."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.search(r"Line [1-3]$")

        assert result.total_matches == 3

    def test_no_matches(self, sample_text_file):
        """Returns empty matches when nothing found."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.search("not present")

        assert result.total_matches == 0
        assert result.matches == []

    def test_search_with_context(self, sample_text_file):
        """Includes context lines around matches."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.search("Line 5", context_lines=2)

        match = result.matches[0]
        assert len(match.context_before) == 2
        assert match.context_before[0] == (3, "Line 3")
        assert match.context_before[1] == (4, "Line 4")
        assert len(match.context_after) == 2
        assert match.context_after[0] == (6, "Line 6")
        assert match.context_after[1] == (7, "Line 7")

    def test_context_at_start_of_file(self, sample_text_file):
        """Context is truncated at start of file."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.search("Line 1", context_lines=2)

        match = result.matches[0]
        assert len(match.context_before) == 0  # No lines before line 1
        assert len(match.context_after) == 2

    def test_context_at_end_of_file(self, sample_text_file):
        """Context is truncated at end of file."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.search("Line 10", context_lines=2)

        match = result.matches[0]
        assert len(match.context_before) == 2
        assert len(match.context_after) == 0  # No lines after line 10

    def test_invalid_regex(self, sample_text_file):
        """Raises error for invalid regex."""
        adapter = TextAdapter(sample_text_file)
        with pytest.raises(re.error):  # re.error
            adapter.search(r"[invalid")

    def test_empty_file_search(self, empty_text_file):
        """Search on empty file returns no matches."""
        adapter = TextAdapter(empty_text_file)
        result = adapter.search("anything")

        assert result.total_matches == 0


class TestTextAdapterViewPage:
    """Tests for TextAdapter.view_page."""

    def test_returns_not_applicable(self, sample_text_file):
        """Text files don't support page view."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.view_page(page=1)

        assert result.not_applicable is True
        assert result.page_number == 1
        assert result.image_base64 is None


class TestTextAdapterListVisualContent:
    """Tests for TextAdapter.list_visual_content."""

    def test_returns_empty(self, sample_text_file):
        """Text files have no visual content."""
        adapter = TextAdapter(sample_text_file)
        result = adapter.list_visual_content()

        assert result.items == []
        assert result.total_items == 0

    def test_empty_file_visual_content(self, empty_text_file):
        """Empty file also returns empty visual content."""
        adapter = TextAdapter(empty_text_file)
        result = adapter.list_visual_content()

        assert result.items == []
        assert result.total_items == 0


class TestShakespeareText:
    """Tests against Shakespeare's Complete Works (~196K lines).

    These integration tests use real corpus fixtures to validate toolkit behavior.
    """

    def test_read_first_lines(self, shakespeare_file):
        """Read opening lines of Shakespeare."""
        result = read_lines(shakespeare_file, start=1, end=10)

        assert len(result.lines) == 10
        assert result.total_lines > 100000  # Very large file
        assert "Shakespeare" in result.lines[0][1]

    def test_read_middle_section(self, shakespeare_file):
        """Read from middle of large file."""
        result = read_lines(shakespeare_file, start=50000, end=50010)

        assert len(result.lines) == 11
        assert result.start_line == 50000

    def test_read_end_section(self, shakespeare_file):
        """Read last lines of file."""
        # First get total to know where end is
        info = read_lines(shakespeare_file, start=1, end=1)
        total = info.total_lines

        result = read_lines(shakespeare_file, start=total - 10)
        assert len(result.lines) == 11
        assert result.end_line == total

    def test_search_hamlet(self, shakespeare_file):
        """Search for Hamlet references."""
        result = search(shakespeare_file, r"\bHamlet\b")

        assert result.total_matches > 0
        # Hamlet appears many times in the play
        assert any("Hamlet" in m.line_content for m in result.matches)

    def test_search_with_context(self, shakespeare_file):
        """Search with context lines."""
        result = search(shakespeare_file, r"To be, or not to be", context_lines=2)

        if result.total_matches > 0:
            match = result.matches[0]
            assert len(match.context_before) <= 2
            assert len(match.context_after) <= 2

    def test_search_regex_pattern(self, shakespeare_file):
        """Search using regex pattern."""
        # Find lines starting with stage directions [...]
        result = search(shakespeare_file, r"^\s*\[.*\]")

        # Stage directions should exist in Shakespeare
        assert result.total_matches > 0

    def test_view_page_not_applicable(self, shakespeare_file):
        """Text files don't support page view."""
        result = view_page(shakespeare_file, page=1)
        assert result.not_applicable is True

    def test_no_visual_content(self, shakespeare_file):
        """Plain text has no visual content."""
        result = list_visual_content(shakespeare_file)
        assert result.total_items == 0
