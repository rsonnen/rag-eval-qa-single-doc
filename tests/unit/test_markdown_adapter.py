"""Tests for MarkdownAdapter."""

import re

import pytest

from single_doc_generator.toolkit import (
    list_visual_content,
    read_lines,
    search,
    view_page,
)
from single_doc_generator.toolkit.markdown_adapter import MarkdownAdapter


@pytest.fixture
def sample_md_file(tmp_path):
    """Create a sample markdown file."""
    content = """# Title

Some text here.

## Section 1

More content.

## Section 2

Final section.
"""
    file = tmp_path / "sample.md"
    file.write_text(content)
    return file


@pytest.fixture
def md_with_images(tmp_path):
    """Create a markdown file with image references."""
    content = """# Document with Images

Here is an image: ![Alt text](images/photo.png)

And another: ![](diagrams/chart.svg)

Multiple on one line: ![a](1.png) and ![b](2.png)
"""
    file = tmp_path / "images.md"
    file.write_text(content)
    return file


@pytest.fixture
def empty_md_file(tmp_path):
    """Create an empty markdown file."""
    file = tmp_path / "empty.md"
    file.write_text("")
    return file


class TestMarkdownAdapterInit:
    """Tests for MarkdownAdapter initialization."""

    def test_init_with_valid_file(self, sample_md_file):
        adapter = MarkdownAdapter(sample_md_file)
        assert adapter.file_path == sample_md_file
        assert adapter.total_lines == 11

    def test_init_file_not_found(self, tmp_path):
        missing = tmp_path / "missing.md"
        with pytest.raises(FileNotFoundError):
            MarkdownAdapter(missing)

    def test_init_empty_file(self, empty_md_file):
        adapter = MarkdownAdapter(empty_md_file)
        assert adapter.total_lines == 0


class TestMarkdownAdapterReadLines:
    """Tests for MarkdownAdapter.read_lines."""

    def test_read_first_lines(self, sample_md_file):
        adapter = MarkdownAdapter(sample_md_file)
        result = adapter.read_lines(start=1, end=3)

        assert len(result.lines) == 3
        assert result.lines[0] == (1, "# Title")

    def test_read_entire_file(self, sample_md_file):
        adapter = MarkdownAdapter(sample_md_file)
        result = adapter.read_lines()

        assert len(result.lines) == 11
        assert result.total_lines == 11

    def test_start_less_than_one(self, sample_md_file):
        adapter = MarkdownAdapter(sample_md_file)
        with pytest.raises(ValueError, match="start must be >= 1"):
            adapter.read_lines(start=0)

    def test_empty_file(self, empty_md_file):
        adapter = MarkdownAdapter(empty_md_file)
        result = adapter.read_lines()
        assert result.lines == []


class TestMarkdownAdapterSearch:
    """Tests for MarkdownAdapter.search."""

    def test_search_heading(self, sample_md_file):
        adapter = MarkdownAdapter(sample_md_file)
        result = adapter.search(r"^## ")

        assert result.total_matches == 2

    def test_search_with_context(self, sample_md_file):
        adapter = MarkdownAdapter(sample_md_file)
        result = adapter.search("Section 1", context_lines=1)

        match = result.matches[0]
        assert len(match.context_before) == 1
        assert len(match.context_after) == 1

    def test_no_matches(self, sample_md_file):
        adapter = MarkdownAdapter(sample_md_file)
        result = adapter.search("not present")
        assert result.total_matches == 0

    def test_invalid_regex(self, sample_md_file):
        adapter = MarkdownAdapter(sample_md_file)
        with pytest.raises(re.error):
            adapter.search(r"[invalid")


class TestMarkdownAdapterViewPage:
    """Tests for MarkdownAdapter.view_page."""

    def test_returns_not_applicable(self, sample_md_file):
        adapter = MarkdownAdapter(sample_md_file)
        result = adapter.view_page(page=1)

        assert result.not_applicable is True
        assert result.page_number == 1


class TestMarkdownAdapterListVisualContent:
    """Tests for MarkdownAdapter.list_visual_content."""

    def test_no_images(self, sample_md_file):
        adapter = MarkdownAdapter(sample_md_file)
        result = adapter.list_visual_content()

        assert result.items == []
        assert result.total_items == 0

    def test_finds_images(self, md_with_images):
        adapter = MarkdownAdapter(md_with_images)
        result = adapter.list_visual_content()

        assert result.total_items == 4
        assert result.items[0].content_type == "image"
        assert result.items[0].reference == "images/photo.png"
        assert result.items[0].alt_text == "Alt text"

    def test_image_without_alt_text(self, md_with_images):
        adapter = MarkdownAdapter(md_with_images)
        result = adapter.list_visual_content()

        # Second image has empty alt text
        assert result.items[1].alt_text is None
        assert result.items[1].reference == "diagrams/chart.svg"

    def test_multiple_images_per_line(self, md_with_images):
        adapter = MarkdownAdapter(md_with_images)
        result = adapter.list_visual_content()

        # Last two are on the same line
        assert result.items[2].reference == "1.png"
        assert result.items[3].reference == "2.png"

    def test_empty_file(self, empty_md_file):
        adapter = MarkdownAdapter(empty_md_file)
        result = adapter.list_visual_content()
        assert result.items == []


class TestRPGSpellMarkdown:
    """Tests against RPG spell markdown files.

    These integration tests use real corpus fixtures to validate toolkit behavior.
    """

    def test_read_entire_spell(self, rpg_spell_file):
        """Read entire spell description."""
        result = read_lines(rpg_spell_file)

        assert result.total_lines > 0
        # First line should be the title
        assert result.lines[0][1].startswith("# ")

    def test_search_spell_level(self, rpg_spell_file):
        """Search for spell level info."""
        result = search(rpg_spell_file, r"level")

        assert result.total_matches > 0

    def test_search_casting_time(self, rpg_spell_file):
        """Search for casting time."""
        result = search(rpg_spell_file, r"Casting Time")

        assert result.total_matches == 1

    def test_search_damage_dice(self, rpg_spell_file):
        """Search for damage dice notation."""
        result = search(rpg_spell_file, r"\d+d\d+")

        # Most damage spells have dice notation
        if result.total_matches > 0:
            # Verify dice format found
            assert any("d" in m.line_content for m in result.matches)

    def test_view_page_not_applicable(self, rpg_spell_file):
        """Markdown files don't support page view."""
        result = view_page(rpg_spell_file, page=1)
        assert result.not_applicable is True

    def test_no_image_references(self, rpg_spell_file):
        """RPG spell files typically have no images."""
        result = list_visual_content(rpg_spell_file)
        # Most spell files are text-only
        assert result.total_items == 0


class TestKubernetesDocsWithImages:
    """Tests against Kubernetes docs markdown with image references.

    These integration tests use real corpus fixtures to validate toolkit behavior.
    """

    def test_finds_image_references(self, kubernetes_docs_file):
        """Detect image references in real documentation."""
        result = list_visual_content(kubernetes_docs_file)

        assert result.total_items >= 3  # File has multiple images
        assert all(item.content_type == "image" for item in result.items)

    def test_image_paths_extracted(self, kubernetes_docs_file):
        """Image paths are correctly extracted."""
        result = list_visual_content(kubernetes_docs_file)

        # Check that paths contain expected patterns
        paths = [item.reference for item in result.items]
        assert any("/images/docs/" in p for p in paths)

    def test_alt_text_extracted(self, kubernetes_docs_file):
        """Alt text is extracted from image references."""
        result = list_visual_content(kubernetes_docs_file)

        # Some images should have alt text
        alt_texts = [item.alt_text for item in result.items if item.alt_text]
        assert len(alt_texts) > 0

    def test_image_locations_recorded(self, kubernetes_docs_file):
        """Image locations (line numbers) are recorded."""
        result = list_visual_content(kubernetes_docs_file)

        for item in result.items:
            assert item.location is not None
            assert "line" in item.location
