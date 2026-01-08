"""Unit tests for LangChain tool wrappers."""

from unittest.mock import patch

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from single_doc_generator.agent import PendingImage
from single_doc_generator.models import SearchMatch
from single_doc_generator.toolkit.langchain_tools import (
    _format_search_match,
    create_document_tools,
)


class TestFormatSearchMatch:
    """Tests for _format_search_match helper."""

    def test_basic_match_formatting(self):
        """Test formatting a match without context."""
        match = SearchMatch(
            line_number=42,
            line_content="This is the matching line",
            context_before=[],
            context_after=[],
        )
        result = _format_search_match(match)
        assert "--- Match at line 42 ---" in result
        assert "→ 42: This is the matching line" in result

    def test_match_with_context_before(self):
        """Test formatting a match with context before."""
        match = SearchMatch(
            line_number=42,
            line_content="This is the matching line",
            context_before=[(40, "Context line one"), (41, "Context line two")],
            context_after=[],
        )
        result = _format_search_match(match)
        assert "40: Context line one" in result
        assert "41: Context line two" in result
        assert "→ 42: This is the matching line" in result

    def test_match_with_context_after(self):
        """Test formatting a match with context after."""
        match = SearchMatch(
            line_number=42,
            line_content="This is the matching line",
            context_before=[],
            context_after=[(43, "After line one"), (44, "After line two")],
        )
        result = _format_search_match(match)
        assert "→ 42: This is the matching line" in result
        assert "43: After line one" in result
        assert "44: After line two" in result

    def test_match_with_full_context(self):
        """Test formatting with both before and after context."""
        match = SearchMatch(
            line_number=50,
            line_content="Match line",
            context_before=[(48, "Before 1"), (49, "Before 2")],
            context_after=[(51, "After 1"), (52, "After 2")],
        )
        result = _format_search_match(match)
        # Verify all expected content is present
        assert "--- Match at line 50 ---" in result
        assert "48: Before 1" in result
        assert "49: Before 2" in result
        assert "→ 50: Match line" in result
        assert "51: After 1" in result
        assert "52: After 2" in result


class TestCreateDocumentTools:
    """Tests for create_document_tools factory."""

    def test_creates_four_tools(self, shakespeare_file):
        """Test that factory creates exactly 4 tools."""
        tools = create_document_tools(shakespeare_file)
        assert len(tools) == 4

    def test_tools_have_names(self, shakespeare_file):
        """Test that all tools have proper names."""
        tools = create_document_tools(shakespeare_file)
        names = {t.name for t in tools}
        assert "read_lines" in names
        assert "search" in names
        assert "view_page" in names
        assert "list_visual_content" in names

    def test_tools_have_descriptions(self, shakespeare_file):
        """Test that all tools have descriptions for LLM."""
        tools = create_document_tools(shakespeare_file)
        for tool in tools:
            assert tool.description is not None
            assert len(tool.description) > 10  # Not empty/trivial

    def test_read_lines_tool_works(self, shakespeare_file):
        """Test read_lines tool returns formatted output."""
        tools = create_document_tools(shakespeare_file)
        read_lines = next(t for t in tools if t.name == "read_lines")
        result = read_lines.invoke({"start": 1, "end": 5})
        assert "Lines 1-5" in result
        assert "total:" in result

    def test_search_tool_works(self, shakespeare_file):
        """Test search tool returns formatted output."""
        tools = create_document_tools(shakespeare_file)
        search = next(t for t in tools if t.name == "search")
        result = search.invoke({"pattern": "Hamlet", "context_lines": 1})
        assert "matches" in result.lower() or "match" in result.lower()

    def test_search_tool_no_matches(self, shakespeare_file):
        """Test search tool handles no matches gracefully."""
        tools = create_document_tools(shakespeare_file)
        search = next(t for t in tools if t.name == "search")
        result = search.invoke({"pattern": "xyznonexistent123"})
        assert "No matches found" in result

    def test_search_tool_invalid_regex(self, shakespeare_file):
        """Test search tool returns error message for invalid regex."""
        tools = create_document_tools(shakespeare_file)
        search = next(t for t in tools if t.name == "search")
        # \p is PCRE unicode property syntax, not valid in Python re
        result = search.invoke({"pattern": r"foo\pbar"})
        assert "Invalid regex pattern" in result
        assert "Python regex syntax" in result

    def test_view_page_tool_text_file(self, shakespeare_file):
        """Test view_page returns not applicable for text files."""
        tools = create_document_tools(shakespeare_file)
        view_page = next(t for t in tools if t.name == "view_page")
        # Tool requires injected tool_call_id via full ToolCall structure
        tool_call = {
            "args": {"page": 1},
            "name": "view_page",
            "type": "tool_call",
            "id": "test-id",
        }
        result = view_page.invoke(tool_call)

        # Result is a Command that updates state with a ToolMessage
        assert isinstance(result, Command)
        messages = result.update["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], ToolMessage)
        assert "doesn't have visual pages" in messages[0].content

    def test_view_page_tool_pdf(self, court_opinion_pdf):
        """Test view_page returns Command with pending image data for PDFs.

        The view_page tool returns a Command that updates agent state with
        pending_images. The image_injector node will later inject these as
        a HumanMessage after all tool responses complete.
        """
        tools = create_document_tools(court_opinion_pdf)
        view_page = next(t for t in tools if t.name == "view_page")
        # Tool requires injected tool_call_id via full ToolCall structure
        tool_call = {
            "args": {"page": 1},
            "name": "view_page",
            "type": "tool_call",
            "id": "test-id",
        }
        result = view_page.invoke(tool_call)

        # Result should be a Command with ToolMessage and pending_images
        assert isinstance(result, Command)

        # Check ToolMessage
        messages = result.update["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], ToolMessage)
        assert "Page 1" in messages[0].content

        # Check pending_images for later injection
        pending = result.update["pending_images"]
        assert len(pending) == 1
        assert isinstance(pending[0], PendingImage)
        assert pending[0].page == 1
        assert len(pending[0].image_base64) > 100  # Base64 should have content

    def test_list_visual_content_text_file(self, shakespeare_file):
        """Test list_visual_content returns empty for text files."""
        tools = create_document_tools(shakespeare_file)
        list_visual = next(t for t in tools if t.name == "list_visual_content")
        result = list_visual.invoke({})
        assert "No visual content" in result

    def test_list_visual_content_pdf(self, court_opinion_pdf):
        """Test list_visual_content finds images in PDFs."""
        tools = create_document_tools(court_opinion_pdf)
        list_visual = next(t for t in tools if t.name == "list_visual_content")
        result = list_visual.invoke({})
        # Court opinion PDF should have some visual content
        assert "visual" in result.lower() or "found" in result.lower()

    def test_tools_bound_to_specific_document(self, fixtures_dir):
        """Test that tools are bound to their specific document."""
        # Create tools for two different files
        shakespeare = fixtures_dir / "shakespeare.txt"
        spell = fixtures_dir / "abhorrent-apparition.md"

        tools_shakespeare = create_document_tools(shakespeare)
        tools_spell = create_document_tools(spell)

        read_shakespeare = next(t for t in tools_shakespeare if t.name == "read_lines")
        read_spell = next(t for t in tools_spell if t.name == "read_lines")

        result_shakespeare = read_shakespeare.invoke({"start": 1, "end": 3})
        result_spell = read_spell.invoke({"start": 1, "end": 3})

        # They should return different content
        assert result_shakespeare != result_spell

    def test_read_lines_out_of_range_returns_error_message(self, shakespeare_file):
        """Test read_lines returns helpful error when start exceeds document length.

        When an agent requests lines past the end of a document, the tool should
        return an error message (not raise an exception) so the agent can adjust.
        """
        tools = create_document_tools(shakespeare_file)
        read_lines = next(t for t in tools if t.name == "read_lines")

        # Request lines way past the end of the document (shakespeare has ~196k lines)
        result = read_lines.invoke({"start": 999999, "end": 1000000})

        # Should return an error message, not crash
        assert "Error" in result
        assert "past the end" in result
        # Should tell the agent how many lines actually exist
        assert "only has" in result

    def test_view_page_returns_error_message_on_render_failure(self, court_opinion_pdf):
        """Test view_page returns error message when page can't be rendered.

        When a PDF page is too large to render under the size limit, the tool
        should return an error message (not raise an exception) so the agent
        can try a different page.
        """
        tools = create_document_tools(court_opinion_pdf)
        view_page = next(t for t in tools if t.name == "view_page")

        # Mock the core view_page to raise ValueError (simulates page too large)
        with patch(
            "single_doc_generator.toolkit.langchain_tools.core_tools.view_page"
        ) as mock_view:
            mock_view.side_effect = ValueError(
                "Page 314 cannot be rendered under 2MB even at 50 DPI"
            )

            tool_call = {
                "args": {"page": 314},
                "name": "view_page",
                "type": "tool_call",
                "id": "test-id",
            }
            result = view_page.invoke(tool_call)

            # Should return Command with error ToolMessage, not crash
            assert isinstance(result, Command)
            messages = result.update["messages"]
            assert len(messages) == 1
            assert isinstance(messages[0], ToolMessage)
            assert "ERROR" in messages[0].content
            assert "cannot be rendered" in messages[0].content
            assert "DIFFERENT" in messages[0].content
