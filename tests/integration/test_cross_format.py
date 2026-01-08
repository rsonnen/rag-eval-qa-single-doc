"""Integration tests verifying consistent behavior across document formats.

Cross-format tests that ensure all adapters return consistent data structures.
Individual adapter-specific tests are in their respective test files under tests/unit/.
"""

from single_doc_generator.toolkit import read_lines, search


class TestMultipleFormats:
    """Tests that verify consistent behavior across formats."""

    def test_both_formats_return_line_numbers(self, shakespeare_file, rpg_spell_file):
        """Both formats return proper line numbers."""
        txt_result = read_lines(shakespeare_file, start=5, end=10)
        md_result = read_lines(rpg_spell_file, start=1, end=5)

        # Line numbers should be 1-indexed
        assert txt_result.lines[0][0] == 5
        assert md_result.lines[0][0] == 1

    def test_search_returns_consistent_structure(
        self, shakespeare_file, rpg_spell_file
    ):
        """Search results have same structure regardless of format."""
        txt_result = search(shakespeare_file, "the", context_lines=1)
        md_result = search(rpg_spell_file, "the", context_lines=1)

        # Both should return SearchResult with matches
        assert hasattr(txt_result, "matches")
        assert hasattr(md_result, "matches")
        assert hasattr(txt_result, "total_matches")
        assert hasattr(md_result, "total_matches")

    def test_all_formats_return_consistent_structure(
        self, shakespeare_file, rpg_spell_file, court_opinion_pdf, django_docs_html
    ):
        """All formats return consistent structure for read_lines."""
        txt_result = read_lines(shakespeare_file, start=1, end=5)
        md_result = read_lines(rpg_spell_file, start=1, end=5)
        pdf_result = read_lines(court_opinion_pdf, start=1, end=5)
        html_result = read_lines(django_docs_html, start=1, end=5)

        # All should have same attributes
        for result in [txt_result, md_result, pdf_result, html_result]:
            assert hasattr(result, "lines")
            assert hasattr(result, "total_lines")
            assert hasattr(result, "start_line")
            assert hasattr(result, "end_line")

        # Line numbers should be 1-indexed in all formats
        assert txt_result.lines[0][0] == 1
        assert md_result.lines[0][0] == 1
        assert pdf_result.lines[0][0] == 1
        assert html_result.lines[0][0] == 1
