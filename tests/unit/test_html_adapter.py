"""Tests for HTMLAdapter."""

import re

import pytest

from single_doc_generator.toolkit.html_adapter import HTMLAdapter


@pytest.fixture
def sample_html_file(tmp_path):
    """Create a sample HTML file with various elements."""
    content = """<!DOCTYPE html>
<html>
<head>
    <title>Sample Doc</title>
</head>
<body>
    <h1>Welcome</h1>
    <p>This is paragraph 1.</p>
    <p>This is paragraph 2 with <strong>bold</strong> text.</p>
    <img src="images/diagram.png" alt="Architecture diagram">
    <p>More content here.</p>
</body>
</html>"""
    file = tmp_path / "sample.html"
    file.write_text(content)
    return file


@pytest.fixture
def empty_html_file(tmp_path):
    """Create an empty HTML file."""
    file = tmp_path / "empty.html"
    file.write_text("")
    return file


@pytest.fixture
def single_line_html_file(tmp_path):
    """Create a file with a single line."""
    file = tmp_path / "single.html"
    file.write_text("<p>Only one line here</p>")
    return file


@pytest.fixture
def complex_html_file(tmp_path):
    """Create an HTML file with figures, tables, and images."""
    content = """<!DOCTYPE html>
<html>
<head><title>Complex Doc</title></head>
<body>
    <h1>Documentation</h1>

    <img src="logo.png" alt="Logo">

    <figure>
        <img src="diagram.svg" alt="Flow diagram">
        <figcaption>Figure 1: System architecture</figcaption>
    </figure>

    <table id="config-options" class="reference-table">
        <caption>Configuration Options</caption>
        <thead>
            <tr><th>Option</th><th>Description</th></tr>
        </thead>
        <tbody>
            <tr><td>debug</td><td>Enable debug mode</td></tr>
            <tr><td>verbose</td><td>Verbose output</td></tr>
        </tbody>
    </table>

    <figure>
        <img src="screenshot.png" alt="Screenshot">
        <figcaption>Figure 2: User interface</figcaption>
    </figure>

    <img src="footer.gif">
</body>
</html>"""
    file = tmp_path / "complex.html"
    file.write_text(content)
    return file


@pytest.fixture
def large_html_file(tmp_path):
    """Create a file with many lines."""
    lines = ["<!DOCTYPE html>", "<html><body>"]
    for i in range(1, 501):
        lines.append(f"<p>Paragraph number {i}</p>")
    lines.extend(["</body>", "</html>"])
    content = "\n".join(lines)
    file = tmp_path / "large.html"
    file.write_text(content)
    return file


class TestHTMLAdapterInit:
    """Tests for HTMLAdapter initialization."""

    def test_init_with_valid_file(self, sample_html_file):
        """Initializes with valid file path."""
        adapter = HTMLAdapter(sample_html_file)
        assert adapter.file_path == sample_html_file
        assert adapter.total_lines == 13

    def test_init_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        missing = tmp_path / "missing.html"
        with pytest.raises(FileNotFoundError):
            HTMLAdapter(missing)

    def test_init_empty_file(self, empty_html_file):
        """Handles empty file."""
        adapter = HTMLAdapter(empty_html_file)
        assert adapter.total_lines == 0


class TestHTMLAdapterReadLines:
    """Tests for HTMLAdapter.read_lines."""

    def test_read_first_5_lines(self, sample_html_file):
        """Reads first 5 lines."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.read_lines(start=1, end=5)

        assert len(result.lines) == 5
        assert result.start_line == 1
        assert result.end_line == 5
        assert result.total_lines == 13
        assert result.lines[0] == (1, "<!DOCTYPE html>")
        assert result.lines[4] == (5, "</head>")

    def test_read_middle_section(self, sample_html_file):
        """Reads lines from middle of file."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.read_lines(start=7, end=9)

        assert len(result.lines) == 3
        assert result.lines[0] == (7, "    <h1>Welcome</h1>")

    def test_read_last_lines(self, sample_html_file):
        """Reads last few lines."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.read_lines(start=11, end=13)

        assert len(result.lines) == 3
        assert result.lines[2] == (13, "</html>")

    def test_read_to_end_without_end_param(self, sample_html_file):
        """Reads to end when end is None."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.read_lines(start=10)

        assert len(result.lines) == 4
        assert result.end_line == 13

    def test_read_entire_file(self, sample_html_file):
        """Reads entire file with defaults."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.read_lines()

        assert len(result.lines) == 13
        assert result.start_line == 1
        assert result.end_line == 13

    def test_end_exceeds_total_lines(self, sample_html_file):
        """Clamps end to total lines."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.read_lines(start=10, end=100)

        assert len(result.lines) == 4
        assert result.end_line == 13

    def test_start_less_than_one(self, sample_html_file):
        """Raises ValueError for start < 1."""
        adapter = HTMLAdapter(sample_html_file)
        with pytest.raises(ValueError, match="start must be >= 1"):
            adapter.read_lines(start=0)

    def test_start_exceeds_total(self, sample_html_file):
        """Raises ValueError for start beyond file."""
        adapter = HTMLAdapter(sample_html_file)
        with pytest.raises(ValueError, match="exceeds total lines"):
            adapter.read_lines(start=50)

    def test_empty_file(self, empty_html_file):
        """Returns empty result for empty file."""
        adapter = HTMLAdapter(empty_html_file)
        result = adapter.read_lines()

        assert result.lines == []
        assert result.total_lines == 0

    def test_single_line_file(self, single_line_html_file):
        """Handles single-line file."""
        adapter = HTMLAdapter(single_line_html_file)
        result = adapter.read_lines()

        assert len(result.lines) == 1
        assert result.lines[0] == (1, "<p>Only one line here</p>")

    def test_large_file_partial_read(self, large_html_file):
        """Reads portion of large file."""
        adapter = HTMLAdapter(large_html_file)
        result = adapter.read_lines(start=100, end=110)

        assert len(result.lines) == 11
        assert result.total_lines == 504
        assert result.lines[0] == (100, "<p>Paragraph number 98</p>")


class TestHTMLAdapterSearch:
    """Tests for HTMLAdapter.search."""

    def test_literal_search(self, sample_html_file):
        """Finds literal string matches."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.search("paragraph 1")

        assert result.total_matches == 1
        assert result.matches[0].line_number == 8
        assert "paragraph 1" in result.matches[0].line_content

    def test_regex_search(self, sample_html_file):
        """Finds regex pattern matches."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.search(r"<p>.*paragraph.*</p>")

        assert result.total_matches == 2

    def test_search_html_tags(self, sample_html_file):
        """Finds HTML tag patterns."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.search(r"<img\s+src=")

        assert result.total_matches == 1
        assert "diagram.png" in result.matches[0].line_content

    def test_no_matches(self, sample_html_file):
        """Returns empty matches when nothing found."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.search("not present")

        assert result.total_matches == 0
        assert result.matches == []

    def test_search_with_context(self, sample_html_file):
        """Includes context lines around matches."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.search("<h1>", context_lines=2)

        match = result.matches[0]
        assert len(match.context_before) == 2
        assert len(match.context_after) == 2

    def test_context_at_start_of_file(self, sample_html_file):
        """Context is truncated at start of file."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.search("DOCTYPE", context_lines=2)

        match = result.matches[0]
        assert len(match.context_before) == 0  # No lines before line 1
        assert len(match.context_after) == 2

    def test_context_at_end_of_file(self, sample_html_file):
        """Context is truncated at end of file."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.search("</html>", context_lines=2)

        match = result.matches[0]
        assert len(match.context_before) == 2
        assert len(match.context_after) == 0  # No lines after last line

    def test_invalid_regex(self, sample_html_file):
        """Raises error for invalid regex."""
        adapter = HTMLAdapter(sample_html_file)
        with pytest.raises(re.error):
            adapter.search(r"[invalid")

    def test_empty_file_search(self, empty_html_file):
        """Search on empty file returns no matches."""
        adapter = HTMLAdapter(empty_html_file)
        result = adapter.search("anything")

        assert result.total_matches == 0


class TestHTMLAdapterViewPage:
    """Tests for HTMLAdapter.view_page."""

    def test_returns_not_applicable(self, sample_html_file):
        """HTML files don't support page view."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.view_page(page=1)

        assert result.not_applicable is True
        assert result.page_number == 1
        assert result.image_base64 is None


class TestHTMLAdapterListVisualContent:
    """Tests for HTMLAdapter.list_visual_content."""

    def test_finds_images(self, sample_html_file):
        """Finds img tags with src and alt."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.list_visual_content()

        assert result.total_items == 1
        img = result.items[0]
        assert img.content_type == "image"
        assert img.reference == "images/diagram.png"
        assert img.alt_text == "Architecture diagram"

    def test_finds_all_visual_elements(self, complex_html_file):
        """Finds images, figures, and tables."""
        adapter = HTMLAdapter(complex_html_file)
        result = adapter.list_visual_content()

        # Images: finds ALL <img> tags (standalone and inside figures)
        # Figures: 2 <figure> elements with figcaptions
        # Tables: 1 <table> element
        content_types = [item.content_type for item in result.items]
        assert "image" in content_types
        assert "figure" in content_types
        assert "table" in content_types

        # Check we found the expected number of each type
        images = [i for i in result.items if i.content_type == "image"]
        figures = [i for i in result.items if i.content_type == "figure"]
        tables = [i for i in result.items if i.content_type == "table"]

        assert len(images) == 4  # All <img> tags including those in figures
        assert len(figures) == 2  # Two figures with figcaptions
        assert len(tables) == 1  # One table

    def test_table_with_caption(self, complex_html_file):
        """Tables capture caption as alt_text."""
        adapter = HTMLAdapter(complex_html_file)
        result = adapter.list_visual_content()

        tables = [i for i in result.items if i.content_type == "table"]
        assert len(tables) == 1
        assert tables[0].alt_text == "Configuration Options"
        assert tables[0].reference == "config-options"

    def test_figure_with_caption(self, complex_html_file):
        """Figures capture figcaption as alt_text."""
        adapter = HTMLAdapter(complex_html_file)
        result = adapter.list_visual_content()

        figures = [i for i in result.items if i.content_type == "figure"]
        captions = [f.alt_text for f in figures if f.alt_text]
        assert "Figure 1: System architecture" in captions
        assert "Figure 2: User interface" in captions

    def test_image_without_alt(self, complex_html_file):
        """Handles images without alt text."""
        adapter = HTMLAdapter(complex_html_file)
        result = adapter.list_visual_content()

        images = [i for i in result.items if i.content_type == "image"]
        # footer.gif has no alt text
        footer = next((i for i in images if "footer" in i.reference), None)
        assert footer is not None
        assert footer.alt_text is None

    def test_empty_file_visual_content(self, empty_html_file):
        """Empty file also returns empty visual content."""
        adapter = HTMLAdapter(empty_html_file)
        result = adapter.list_visual_content()

        assert result.items == []
        assert result.total_items == 0

    def test_html_without_visual_content(self, tmp_path):
        """HTML with no images/figures/tables returns empty."""
        content = """<!DOCTYPE html>
<html>
<body>
    <h1>Text Only</h1>
    <p>No visual content here.</p>
</body>
</html>"""
        file = tmp_path / "text_only.html"
        file.write_text(content)

        adapter = HTMLAdapter(file)
        result = adapter.list_visual_content()

        assert result.items == []
        assert result.total_items == 0

    def test_location_tracking(self, sample_html_file):
        """Visual content includes approximate line location."""
        adapter = HTMLAdapter(sample_html_file)
        result = adapter.list_visual_content()

        assert result.total_items == 1
        assert result.items[0].location is not None
        assert "line" in result.items[0].location


class TestHTMLAdapterRealDjangoDocs:
    """Tests against real Django documentation HTML."""

    def test_read_first_lines(self, django_docs_html):
        """Read opening lines of Django docs HTML."""
        adapter = HTMLAdapter(django_docs_html)
        result = adapter.read_lines(start=1, end=20)

        assert len(result.lines) == 20
        assert result.total_lines > 50
        assert "CSRF" in result.lines[0][1]

    def test_read_middle_section(self, django_docs_html):
        """Read from middle of HTML document."""
        adapter = HTMLAdapter(django_docs_html)
        total = adapter.total_lines

        mid = total // 2
        result = adapter.read_lines(start=mid, end=mid + 10)
        assert len(result.lines) == 11
        assert result.start_line == mid

    def test_search_html_content(self, django_docs_html):
        """Search for content within HTML."""
        adapter = HTMLAdapter(django_docs_html)
        result = adapter.search(r"csrf_token")

        assert result.total_matches > 0
        assert any("csrf_token" in m.line_content for m in result.matches)

    def test_search_html_structure(self, django_docs_html):
        """Search can find HTML structural elements."""
        adapter = HTMLAdapter(django_docs_html)
        result = adapter.search(r"<h2")

        assert result.total_matches > 0

    def test_search_code_blocks(self, django_docs_html):
        """Search finds content in code blocks."""
        adapter = HTMLAdapter(django_docs_html)
        result = adapter.search(r"getCookie")

        assert result.total_matches >= 1

    def test_search_with_context(self, django_docs_html):
        """Search with context lines around matches."""
        adapter = HTMLAdapter(django_docs_html)
        result = adapter.search(r"AJAX", context_lines=2)

        if result.total_matches > 0:
            match = result.matches[0]
            assert len(match.context_before) <= 2
            assert len(match.context_after) <= 2

    def test_view_page_not_applicable(self, django_docs_html):
        """HTML files don't support page view."""
        adapter = HTMLAdapter(django_docs_html)
        result = adapter.view_page(page=1)

        assert result.not_applicable is True
        assert result.page_number == 1

    def test_no_visual_content(self, django_docs_html):
        """This Django docs page has no images/figures/tables."""
        adapter = HTMLAdapter(django_docs_html)
        result = adapter.list_visual_content()

        assert result.total_items == 0
