"""Tests for public tool functions."""

import pytest

from single_doc_generator.toolkit import (
    list_visual_content,
    read_lines,
    search,
    view_page,
)


@pytest.fixture
def text_file(tmp_path):
    """Create a sample text file."""
    content = "\n".join([f"Line {i}" for i in range(1, 11)])
    file = tmp_path / "doc.txt"
    file.write_text(content)
    return file


@pytest.fixture
def md_file(tmp_path):
    """Create a sample markdown file."""
    content = """# Header

Some content here.

![image](path/to/img.png)
"""
    file = tmp_path / "doc.md"
    file.write_text(content)
    return file


@pytest.fixture
def xml_file(tmp_path):
    """Create a sample JATS XML file."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns="https://jats.nlm.nih.gov/ns/archiving/1.4/"
         xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <title-group>
        <article-title>Sample Article</article-title>
      </title-group>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Section One</title>
      <p>Content of section one.</p>
      <fig id="f1">
        <label>Figure 1</label>
        <caption><p>A test figure.</p></caption>
        <graphic xlink:href="figure.png"/>
      </fig>
    </sec>
  </body>
</article>"""
    file = tmp_path / "doc.xml"
    file.write_text(content)
    return file


class TestReadLines:
    """Tests for read_lines tool function."""

    def test_text_file(self, text_file):
        result = read_lines(text_file, start=1, end=5)
        assert len(result.lines) == 5
        assert result.total_lines == 10

    def test_markdown_file(self, md_file):
        result = read_lines(md_file)
        assert result.total_lines == 5

    def test_xml_file(self, xml_file):
        result = read_lines(xml_file)
        assert result.total_lines > 0
        # Should extract title
        lines_content = [line for _, line in result.lines]
        assert any("Sample Article" in line for line in lines_content)

    def test_string_path(self, text_file):
        result = read_lines(str(text_file), start=1, end=3)
        assert len(result.lines) == 3

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_lines(tmp_path / "missing.txt")

    def test_unsupported_format(self, tmp_path):
        docx = tmp_path / "doc.docx"
        docx.write_text("fake docx")
        with pytest.raises(ValueError, match="Unsupported"):
            read_lines(docx)


class TestSearch:
    """Tests for search tool function."""

    def test_text_file(self, text_file):
        result = search(text_file, r"Line [5-7]")
        assert result.total_matches == 3

    def test_markdown_file(self, md_file):
        result = search(md_file, r"^#")
        assert result.total_matches == 1

    def test_xml_file(self, xml_file):
        result = search(xml_file, "Section One")
        assert result.total_matches == 1

    def test_with_context(self, text_file):
        result = search(text_file, "Line 5", context_lines=2)
        assert len(result.matches[0].context_before) == 2

    def test_string_path(self, text_file):
        result = search(str(text_file), "Line 1")
        assert result.total_matches == 2  # Line 1 and Line 10


class TestViewPage:
    """Tests for view_page tool function."""

    def test_text_not_applicable(self, text_file):
        result = view_page(text_file, page=1)
        assert result.not_applicable is True

    def test_markdown_not_applicable(self, md_file):
        result = view_page(md_file, page=1)
        assert result.not_applicable is True

    def test_xml_not_applicable(self, xml_file):
        result = view_page(xml_file, page=1)
        assert result.not_applicable is True

    def test_string_path(self, text_file):
        result = view_page(str(text_file), page=1)
        assert result.not_applicable is True


class TestListVisualContent:
    """Tests for list_visual_content tool function."""

    def test_text_empty(self, text_file):
        result = list_visual_content(text_file)
        assert result.items == []
        assert result.total_items == 0

    def test_markdown_finds_image(self, md_file):
        result = list_visual_content(md_file)
        assert result.total_items == 1
        assert result.items[0].reference == "path/to/img.png"

    def test_xml_finds_figure(self, xml_file):
        result = list_visual_content(xml_file)
        assert result.total_items == 1
        assert result.items[0].content_type == "figure"
        # Reference should be the graphic href or figure id
        assert (
            "figure.png" in result.items[0].reference
            or "f1" in result.items[0].reference
        )

    def test_string_path(self, text_file):
        result = list_visual_content(str(text_file))
        assert result.total_items == 0
