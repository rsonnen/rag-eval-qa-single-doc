"""Tests for AdocAdapter."""

import re

import pytest

from single_doc_generator.toolkit import (
    list_visual_content,
    read_lines,
    search,
    view_page,
)
from single_doc_generator.toolkit.adoc_adapter import AdocAdapter


@pytest.fixture
def sample_adoc_file(tmp_path):
    """Create a sample AsciiDoc file."""
    content = """git-example(1)
==============

NAME
----
git-example - An example command

SYNOPSIS
--------
[verse]
'git example' [<options>]

DESCRIPTION
-----------
This is an example AsciiDoc document.
"""
    file = tmp_path / "sample.adoc"
    file.write_text(content)
    return file  # 17 lines (including trailing newline makes 18 content lines)


@pytest.fixture
def adoc_with_images(tmp_path):
    """Create an AsciiDoc file with image references."""
    content = """= Document with Images

Here is a block image:

image::images/diagram.png[Architecture diagram]

And an inline image: image:icons/tip.png[tip icon] in the text.

Multiple images:

image::figures/chart1.svg[Chart 1]
image::figures/chart2.svg[]
"""
    file = tmp_path / "images.adoc"
    file.write_text(content)
    return file


@pytest.fixture
def empty_adoc_file(tmp_path):
    """Create an empty AsciiDoc file."""
    file = tmp_path / "empty.adoc"
    file.write_text("")
    return file


class TestAdocAdapterInit:
    """Tests for AdocAdapter initialization."""

    def test_init_with_valid_file(self, sample_adoc_file):
        """Initializes with valid file path."""
        adapter = AdocAdapter(sample_adoc_file)
        assert adapter.file_path == sample_adoc_file
        assert adapter.total_lines == 15

    def test_init_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        missing = tmp_path / "missing.adoc"
        with pytest.raises(FileNotFoundError):
            AdocAdapter(missing)

    def test_init_empty_file(self, empty_adoc_file):
        """Handles empty file."""
        adapter = AdocAdapter(empty_adoc_file)
        assert adapter.total_lines == 0


class TestAdocAdapterReadLines:
    """Tests for AdocAdapter.read_lines."""

    def test_read_first_lines(self, sample_adoc_file):
        """Reads first lines."""
        adapter = AdocAdapter(sample_adoc_file)
        result = adapter.read_lines(start=1, end=3)

        assert len(result.lines) == 3
        assert result.lines[0] == (1, "git-example(1)")

    def test_read_entire_file(self, sample_adoc_file):
        """Reads entire file with defaults."""
        adapter = AdocAdapter(sample_adoc_file)
        result = adapter.read_lines()

        assert len(result.lines) == 15
        assert result.total_lines == 15

    def test_start_less_than_one(self, sample_adoc_file):
        """Raises ValueError for start < 1."""
        adapter = AdocAdapter(sample_adoc_file)
        with pytest.raises(ValueError, match="start must be >= 1"):
            adapter.read_lines(start=0)

    def test_empty_file(self, empty_adoc_file):
        """Returns empty result for empty file."""
        adapter = AdocAdapter(empty_adoc_file)
        result = adapter.read_lines()
        assert result.lines == []


class TestAdocAdapterSearch:
    """Tests for AdocAdapter.search."""

    def test_search_section_header(self, sample_adoc_file):
        """Finds section headers using dashes."""
        adapter = AdocAdapter(sample_adoc_file)
        # AsciiDoc uses various lengths of dashes for section underlines
        result = adapter.search(r"^-+$")

        # Sample file has NAME (----), SYNOPSIS (--------), DESCRIPTION (-----------)
        assert result.total_matches >= 3

    def test_search_with_context(self, sample_adoc_file):
        """Includes context lines around matches."""
        adapter = AdocAdapter(sample_adoc_file)
        result = adapter.search("SYNOPSIS", context_lines=1)

        match = result.matches[0]
        assert len(match.context_before) == 1
        assert len(match.context_after) == 1

    def test_no_matches(self, sample_adoc_file):
        """Returns empty matches when nothing found."""
        adapter = AdocAdapter(sample_adoc_file)
        result = adapter.search("not present")
        assert result.total_matches == 0

    def test_invalid_regex(self, sample_adoc_file):
        """Raises error for invalid regex."""
        adapter = AdocAdapter(sample_adoc_file)
        with pytest.raises(re.error):
            adapter.search(r"[invalid")


class TestAdocAdapterViewPage:
    """Tests for AdocAdapter.view_page."""

    def test_returns_not_applicable(self, sample_adoc_file):
        """AsciiDoc files don't support page view."""
        adapter = AdocAdapter(sample_adoc_file)
        result = adapter.view_page(page=1)

        assert result.not_applicable is True
        assert result.page_number == 1


class TestAdocAdapterListVisualContent:
    """Tests for AdocAdapter.list_visual_content."""

    def test_no_images(self, sample_adoc_file):
        """File without images returns empty list."""
        adapter = AdocAdapter(sample_adoc_file)
        result = adapter.list_visual_content()

        assert result.items == []
        assert result.total_items == 0

    def test_finds_block_images(self, adoc_with_images):
        """Finds block image references (image::path[alt])."""
        adapter = AdocAdapter(adoc_with_images)
        result = adapter.list_visual_content()

        # Should find all 4 images
        assert result.total_items == 4
        assert result.items[0].content_type == "image"
        assert result.items[0].reference == "images/diagram.png"
        assert result.items[0].alt_text == "Architecture diagram"

    def test_finds_inline_images(self, adoc_with_images):
        """Finds inline image references (image:path[alt])."""
        adapter = AdocAdapter(adoc_with_images)
        result = adapter.list_visual_content()

        # Second image is inline
        inline_image = result.items[1]
        assert inline_image.reference == "icons/tip.png"
        assert inline_image.alt_text == "tip icon"

    def test_image_without_alt_text(self, adoc_with_images):
        """Handles images without alt text."""
        adapter = AdocAdapter(adoc_with_images)
        result = adapter.list_visual_content()

        # Last image has no alt text
        last_image = result.items[-1]
        assert last_image.reference == "figures/chart2.svg"
        assert last_image.alt_text is None

    def test_empty_file(self, empty_adoc_file):
        """Empty file also returns empty visual content."""
        adapter = AdocAdapter(empty_adoc_file)
        result = adapter.list_visual_content()
        assert result.items == []

    def test_location_tracking(self, adoc_with_images):
        """Visual content includes line location."""
        adapter = AdocAdapter(adoc_with_images)
        result = adapter.list_visual_content()

        for item in result.items:
            assert item.location is not None
            assert "line" in item.location


class TestRealGitDocsAdoc:
    """Tests against real Git documentation AsciiDoc files.

    These integration tests use real corpus fixtures to validate toolkit behavior.
    """

    def test_read_entire_file(self, git_stage_adoc):
        """Read entire git-stage documentation."""
        result = read_lines(git_stage_adoc)

        assert result.total_lines > 0
        # First line should be the command name
        assert "git-stage" in result.lines[0][1]

    def test_search_sections(self, git_stage_adoc):
        """Search for section headers."""
        result = search(git_stage_adoc, r"^[A-Z]+$")

        # Should find NAME, SYNOPSIS, DESCRIPTION, GIT sections
        assert result.total_matches >= 3

    def test_search_linkgit(self, git_stage_adoc):
        """Search for linkgit references (AsciiDoc cross-references)."""
        result = search(git_stage_adoc, r"linkgit:")

        # git-stage references other git commands
        assert result.total_matches >= 1

    def test_view_page_not_applicable(self, git_stage_adoc):
        """AsciiDoc files don't support page view."""
        result = view_page(git_stage_adoc, page=1)
        assert result.not_applicable is True

    def test_no_visual_content(self, git_stage_adoc):
        """Git man pages typically have no images."""
        result = list_visual_content(git_stage_adoc)
        assert result.total_items == 0
