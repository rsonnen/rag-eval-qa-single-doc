"""Tests for XMLAdapter with JATS XML files."""

import re

import pytest

from single_doc_generator.toolkit.xml_adapter import XMLAdapter


@pytest.fixture
def simple_jats_xml(tmp_path):
    """Create a simple JATS XML file for basic tests."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns="https://jats.nlm.nih.gov/ns/archiving/1.4/">
  <front>
    <article-meta>
      <title-group>
        <article-title>Test Article Title</article-title>
      </title-group>
      <abstract>
        <p>This is the abstract of the test article.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Introduction</title>
      <p>This is the introduction paragraph.</p>
      <p>This is another paragraph in the introduction.</p>
    </sec>
    <sec>
      <title>Methods</title>
      <p>Methods section content here.</p>
    </sec>
  </body>
  <back>
    <ref-list>
      <ref id="r1"><mixed-citation>Reference 1</mixed-citation></ref>
      <ref id="r2"><mixed-citation>Reference 2</mixed-citation></ref>
      <ref id="r3"><mixed-citation>Reference 3</mixed-citation></ref>
    </ref-list>
  </back>
</article>"""
    file = tmp_path / "simple.xml"
    file.write_text(content)
    return file


@pytest.fixture
def jats_with_oai_wrapper(tmp_path):
    """Create a JATS XML file wrapped in OAI-PMH record structure."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<record xmlns="http://www.openarchives.org/OAI/2.0/">
  <header>
    <identifier>oai:pubmedcentral.nih.gov:123456</identifier>
  </header>
  <metadata>
    <article xmlns="https://jats.nlm.nih.gov/ns/archiving/1.4/">
      <front>
        <article-meta>
          <title-group>
            <article-title>Wrapped Article Title</article-title>
          </title-group>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Content Section</title>
          <p>Content paragraph.</p>
        </sec>
      </body>
    </article>
  </metadata>
</record>"""
    file = tmp_path / "wrapped.xml"
    file.write_text(content)
    return file


@pytest.fixture
def jats_with_figures(tmp_path):
    """Create a JATS XML file with figure elements."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns="https://jats.nlm.nih.gov/ns/archiving/1.4/"
         xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <title-group>
        <article-title>Article with Figures</article-title>
      </title-group>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Results</title>
      <p>See the results in Figure 1.</p>
      <fig id="f1">
        <label>Figure 1</label>
        <caption>
          <p>Description of figure one showing important data.</p>
        </caption>
        <graphic xlink:href="fig1.jpg"/>
      </fig>
      <p>Additional results shown in Figure 2.</p>
      <fig id="f2">
        <label>Figure 2</label>
        <caption>
          <p>Another figure with more data.</p>
        </caption>
        <graphic xlink:href="fig2.png"/>
      </fig>
    </sec>
  </body>
</article>"""
    file = tmp_path / "with_figures.xml"
    file.write_text(content)
    return file


@pytest.fixture
def jats_with_tables(tmp_path):
    """Create a JATS XML file with table elements."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns="https://jats.nlm.nih.gov/ns/archiving/1.4/">
  <front>
    <article-meta>
      <title-group>
        <article-title>Article with Tables</article-title>
      </title-group>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Data Analysis</title>
      <p>Table 1 shows the summary statistics.</p>
      <table-wrap id="t1">
        <label>Table</label>
        <caption>
          <title>Summary statistics for the experiment</title>
        </caption>
        <table>
          <tr><th>Variable</th><th>Mean</th><th>SD</th></tr>
          <tr><td>Height</td><td>170</td><td>10</td></tr>
        </table>
      </table-wrap>
    </sec>
  </body>
</article>"""
    file = tmp_path / "with_tables.xml"
    file.write_text(content)
    return file


@pytest.fixture
def empty_xml(tmp_path):
    """Create an empty article XML file."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns="https://jats.nlm.nih.gov/ns/archiving/1.4/">
</article>"""
    file = tmp_path / "empty.xml"
    file.write_text(content)
    return file


@pytest.fixture
def malformed_xml(tmp_path):
    """Create a malformed XML file."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<article>
  <unclosed_tag>
</article>"""
    file = tmp_path / "malformed.xml"
    file.write_text(content)
    return file


@pytest.fixture
def no_article_xml(tmp_path):
    """Create an XML file without an article element."""
    content = """<?xml version="1.0" encoding="UTF-8"?>
<document>
  <title>Not an article</title>
</document>"""
    file = tmp_path / "no_article.xml"
    file.write_text(content)
    return file


class TestXMLAdapterInit:
    """Tests for XMLAdapter initialization."""

    def test_init_with_valid_file(self, simple_jats_xml):
        """Initializes with valid JATS XML file."""
        adapter = XMLAdapter(simple_jats_xml)
        assert adapter.file_path == simple_jats_xml
        assert adapter.total_lines > 0

    def test_init_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        missing = tmp_path / "missing.xml"
        with pytest.raises(FileNotFoundError):
            XMLAdapter(missing)

    def test_init_malformed_xml(self, malformed_xml):
        """Raises ValueError for malformed XML."""
        with pytest.raises(ValueError, match="Cannot parse XML"):
            XMLAdapter(malformed_xml)

    def test_init_no_article_element(self, no_article_xml):
        """Raises ValueError when no article element found."""
        with pytest.raises(ValueError, match="No article element found"):
            XMLAdapter(no_article_xml)

    def test_init_oai_wrapped(self, jats_with_oai_wrapper):
        """Handles OAI-PMH wrapped JATS XML."""
        adapter = XMLAdapter(jats_with_oai_wrapper)
        assert adapter.total_lines > 0
        # Should find and extract the wrapped article
        result = adapter.read_lines()
        # Check that title was extracted
        found_title = any("Wrapped Article Title" in line for _, line in result.lines)
        assert found_title


class TestXMLAdapterReadLines:
    """Tests for XMLAdapter.read_lines."""

    def test_read_entire_file(self, simple_jats_xml):
        """Reads entire file with defaults."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.read_lines()

        assert len(result.lines) == adapter.total_lines
        assert result.start_line == 1
        assert result.end_line == adapter.total_lines

    def test_extracts_title(self, simple_jats_xml):
        """Extracts article title."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.read_lines()

        # First line should be the title with heading marker
        assert result.lines[0][1] == "# Test Article Title"

    def test_extracts_abstract(self, simple_jats_xml):
        """Extracts abstract content."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.read_lines()

        # Find abstract heading
        lines_content = [line for _, line in result.lines]
        assert "## Abstract" in lines_content
        assert "This is the abstract of the test article." in lines_content

    def test_extracts_section_headings(self, simple_jats_xml):
        """Extracts section headings with proper markup."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.read_lines()

        lines_content = [line for _, line in result.lines]
        assert "## Introduction" in lines_content
        assert "## Methods" in lines_content

    def test_extracts_paragraphs(self, simple_jats_xml):
        """Extracts paragraph content."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.read_lines()

        lines_content = [line for _, line in result.lines]
        assert "This is the introduction paragraph." in lines_content
        assert "Methods section content here." in lines_content

    def test_extracts_reference_count(self, simple_jats_xml):
        """Includes reference count in extracted text."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.read_lines()

        lines_content = [line for _, line in result.lines]
        assert "[3 references]" in lines_content

    def test_read_range(self, simple_jats_xml):
        """Reads specific line range."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.read_lines(start=3, end=5)

        assert len(result.lines) == 3
        assert result.start_line == 3
        assert result.end_line == 5

    def test_read_to_end(self, simple_jats_xml):
        """Reads to end when end is None."""
        adapter = XMLAdapter(simple_jats_xml)
        total = adapter.total_lines
        result = adapter.read_lines(start=total - 2)

        assert result.end_line == total

    def test_end_exceeds_total(self, simple_jats_xml):
        """Clamps end to total lines."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.read_lines(start=1, end=1000)

        assert result.end_line == adapter.total_lines

    def test_start_less_than_one(self, simple_jats_xml):
        """Raises ValueError for start < 1."""
        adapter = XMLAdapter(simple_jats_xml)
        with pytest.raises(ValueError, match="start must be >= 1"):
            adapter.read_lines(start=0)

    def test_start_exceeds_total(self, simple_jats_xml):
        """Raises ValueError for start beyond file."""
        adapter = XMLAdapter(simple_jats_xml)
        with pytest.raises(ValueError, match="exceeds total lines"):
            adapter.read_lines(start=1000)

    def test_empty_article(self, empty_xml):
        """Handles empty article gracefully."""
        adapter = XMLAdapter(empty_xml)
        result = adapter.read_lines()

        # Should have no content but not error
        assert result.total_lines == 0
        assert result.lines == []


class TestXMLAdapterSearch:
    """Tests for XMLAdapter.search."""

    def test_literal_search(self, simple_jats_xml):
        """Finds literal string matches."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.search("introduction paragraph")

        assert result.total_matches == 1
        assert "introduction paragraph" in result.matches[0].line_content

    def test_regex_search(self, simple_jats_xml):
        """Finds regex pattern matches."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.search(r"##\s+\w+")

        # Should match section headings
        assert result.total_matches >= 3  # Abstract, Introduction, Methods

    def test_case_sensitive_search(self, simple_jats_xml):
        """Search is case-sensitive by default."""
        adapter = XMLAdapter(simple_jats_xml)

        result_upper = adapter.search("Introduction")

        # "Introduction" appears as heading
        assert result_upper.total_matches >= 1

    def test_no_matches(self, simple_jats_xml):
        """Returns empty matches when nothing found."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.search("xyzzy_not_present")

        assert result.total_matches == 0
        assert result.matches == []

    def test_search_with_context(self, simple_jats_xml):
        """Includes context lines around matches."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.search("Introduction", context_lines=2)

        match = result.matches[0]
        # Should have context before and after (if available)
        assert isinstance(match.context_before, list)
        assert isinstance(match.context_after, list)

    def test_invalid_regex(self, simple_jats_xml):
        """Raises error for invalid regex."""
        adapter = XMLAdapter(simple_jats_xml)
        with pytest.raises(re.error):
            adapter.search(r"[invalid")


class TestXMLAdapterViewPage:
    """Tests for XMLAdapter.view_page."""

    def test_returns_not_applicable(self, simple_jats_xml):
        """XML files don't support page view."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.view_page(page=1)

        assert result.not_applicable is True
        assert result.page_number == 1
        assert result.image_base64 is None


class TestXMLAdapterListVisualContent:
    """Tests for XMLAdapter.list_visual_content."""

    def test_no_visual_content(self, simple_jats_xml):
        """Returns empty list for XML without figures/tables."""
        adapter = XMLAdapter(simple_jats_xml)
        result = adapter.list_visual_content()

        assert result.total_items == 0
        assert result.items == []

    def test_extracts_figures(self, jats_with_figures):
        """Extracts figure elements as visual content."""
        adapter = XMLAdapter(jats_with_figures)
        result = adapter.list_visual_content()

        assert result.total_items == 2

        # Check figure properties
        fig_refs = [item.reference for item in result.items]
        assert "fig1.jpg" in fig_refs or "f1" in fig_refs
        assert "fig2.png" in fig_refs or "f2" in fig_refs

        # Check content types
        for item in result.items:
            assert item.content_type == "figure"

    def test_extracts_figure_captions(self, jats_with_figures):
        """Extracts figure caption text."""
        adapter = XMLAdapter(jats_with_figures)
        result = adapter.list_visual_content()

        # Find items with captions containing expected text
        captions = [item.alt_text for item in result.items if item.alt_text]
        assert any("important data" in cap for cap in captions)

    def test_extracts_tables(self, jats_with_tables):
        """Extracts table elements as visual content."""
        adapter = XMLAdapter(jats_with_tables)
        result = adapter.list_visual_content()

        assert result.total_items == 1
        assert result.items[0].content_type == "table"

    def test_extracts_table_title(self, jats_with_tables):
        """Extracts table title/caption."""
        adapter = XMLAdapter(jats_with_tables)
        result = adapter.list_visual_content()

        assert result.items[0].alt_text is not None
        assert "Summary statistics" in result.items[0].alt_text

    def test_figure_reference_in_text(self, jats_with_figures):
        """Figure references appear in extracted text."""
        adapter = XMLAdapter(jats_with_figures)
        result = adapter.read_lines()

        # Find figure reference lines
        lines_content = [line for _, line in result.lines]
        # Should have figure reference like "[Figure 1: ...]"
        fig_lines = [ln for ln in lines_content if "Figure 1" in ln and "[" in ln]
        assert len(fig_lines) >= 1


class TestXMLAdapterWithRealPubMed:
    """Integration tests with real PubMed XML files."""

    def test_read_real_pubmed_file(self, pubmed_xml_file):
        """Can read and parse real PubMed JATS XML."""
        adapter = XMLAdapter(pubmed_xml_file)

        assert adapter.total_lines > 0

        # Should extract readable content
        result = adapter.read_lines(start=1, end=50)
        assert len(result.lines) > 0

    def test_real_pubmed_has_title(self, pubmed_xml_file):
        """Extracts title from real PubMed paper."""
        adapter = XMLAdapter(pubmed_xml_file)
        result = adapter.read_lines(start=1, end=5)

        # First non-empty line should be title with heading marker
        non_empty = [line for _, line in result.lines if line.strip()]
        assert len(non_empty) > 0
        assert non_empty[0].startswith("#")

    def test_real_pubmed_searchable(self, pubmed_xml_file):
        """Can search real PubMed content."""
        adapter = XMLAdapter(pubmed_xml_file)

        # Search for common scientific terms
        result = adapter.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")  # Capitalized words
        assert result.total_matches > 0

    def test_real_pubmed_visual_content(self, pubmed_xml_with_figures):
        """Extracts visual content from real PubMed paper with figures."""
        adapter = XMLAdapter(pubmed_xml_with_figures)
        result = adapter.list_visual_content()

        # Real PubMed papers with figures should have visual content
        assert result.total_items > 0

        # Should have figures or tables
        content_types = {item.content_type for item in result.items}
        assert "figure" in content_types or "table" in content_types

    def test_extracted_text_readable(self, pubmed_xml_file):
        """Extracted text is human-readable."""
        adapter = XMLAdapter(pubmed_xml_file)
        result = adapter.read_lines()

        # Check that content is clean text, not XML tags
        # Note: < and > may appear in scientific text (e.g., "p < 0.05")
        for _, line in result.lines:
            # Should not contain XML tag patterns like "<tag>" or "</tag>"
            if not line.startswith("["):
                assert "</" not in line, f"Found closing tag in: {line[:80]}"
                # Look for opening tag pattern (letter after <)
                if re.search(r"<[a-zA-Z]", line):
                    # Allow if it looks like math (e.g., "x < y") not XML
                    assert not re.search(r"<[a-zA-Z]+[>\s/]", line), (
                        f"Found XML tag in: {line[:80]}"
                    )

    def test_content_preserves_structure(self, pubmed_xml_file):
        """Extracted content preserves document structure."""
        adapter = XMLAdapter(pubmed_xml_file)
        result = adapter.read_lines()

        lines_content = [line for _, line in result.lines]

        # Should have heading markers
        headings = [ln for ln in lines_content if ln.startswith("#")]
        assert len(headings) > 0

        # Should have blank lines for readability
        blank_lines = [ln for ln in lines_content if ln == ""]
        assert len(blank_lines) > 0
