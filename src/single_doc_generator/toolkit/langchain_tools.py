"""LangChain tool wrappers for document exploration.

This module provides document exploration tools decorated with LangChain's @tool
decorator for use with LangGraph agents. These wrap the core toolkit functions
and format outputs as strings suitable for LLM consumption.

The tools are created via a factory function that binds them to a specific
document path, since agents typically work with one document at a time.

For vision-capable tools (view_page), the tool returns a Command that updates
the agent state with pending image data. The custom vision agent graph then
injects images as a HumanMessage after all tool responses complete. This
pattern is necessary because OpenAI's API only allows images in user messages
and requires all tool responses to immediately follow the assistant message.
"""

import re
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command

from single_doc_generator.agent import PendingImage
from single_doc_generator.models import SearchMatch
from single_doc_generator.toolkit import tools as core_tools

# Type alias for LangChain tools - the @tool decorator returns BaseTool which
# lacks complete type stubs. Using Any is appropriate here since we're wrapping
# an external library's decorator pattern.
LangChainTool = Any

# Maximum lines per read_lines call to prevent context overflow on large documents
MAX_LINES_PER_READ = 500

# Maximum search matches to return to prevent context overflow
MAX_SEARCH_MATCHES = 50


def _format_search_match(match: SearchMatch) -> str:
    """Format a single search match with context for display.

    Args:
        match: The search match to format.

    Returns:
        Formatted string with line numbers and context.
    """
    lines = [f"\n--- Match at line {match.line_number} ---"]
    for num, content in match.context_before:
        lines.append(f"  {num}: {content}")
    lines.append(f"→ {match.line_number}: {match.line_content}")
    for num, content in match.context_after:
        lines.append(f"  {num}: {content}")
    return "\n".join(lines)


def _create_read_lines_tool(document_path: Path) -> LangChainTool:
    """Create a read_lines tool bound to a document.

    Args:
        document_path: Path to the document.

    Returns:
        LangChain BaseTool instance.
    """

    @tool
    def read_lines(start: int = 1, end: int | None = None) -> str:
        """Read a range of lines from the document.

        Returns lines with their line numbers, like viewing a file in an editor.
        Use this to explore document content by reading specific sections.
        Limited to 500 lines per call to manage context size.

        Args:
            start: Starting line number (1-indexed). Defaults to 1.
            end: Ending line number (1-indexed, inclusive). If not provided,
                 reads up to 500 lines from start.

        Returns:
            Lines with line numbers, plus total document length.
        """
        # Enforce maximum lines per call to prevent context overflow
        effective_end = end
        if effective_end is None or effective_end - start + 1 > MAX_LINES_PER_READ:
            effective_end = start + MAX_LINES_PER_READ - 1

        try:
            result = core_tools.read_lines(
                document_path, start=start, end=effective_end
            )
        except ValueError as e:
            # Return helpful message instead of crashing when out of range
            error_msg = str(e)
            if "exceeds total lines" in error_msg:
                # Get total lines for the helpful message
                probe = core_tools.read_lines(document_path, start=1, end=1)
                return (
                    f"Error: Requested start line {start} is past the end of the "
                    f"document. This document only has {probe.total_lines} lines. "
                    f"Try reading lines 1-{probe.total_lines} instead."
                )
            raise  # Re-raise other ValueErrors

        lines_text = "\n".join(f"{num}: {content}" for num, content in result.lines)

        truncation_note = ""
        if end is None or (end != effective_end):
            truncation_note = (
                f" (limited to {MAX_LINES_PER_READ} lines; use start/end to paginate)"
            )

        return (
            f"Lines {result.start_line}-{result.end_line} "
            f"of {result.total_lines} total{truncation_note}:\n{lines_text}"
        )

    return read_lines


def _create_search_tool(document_path: Path) -> LangChainTool:
    """Create a search tool bound to a document.

    Args:
        document_path: Path to the document.

    Returns:
        LangChain BaseTool instance.
    """

    @tool
    def search(pattern: str, context_lines: int = 2) -> str:
        """Search for a Python regex pattern in the document.

        Matches are per-line: the pattern must match within a single line.
        Use OR patterns like `term1|term2|term3` for multi-term searches.
        Returns matching lines with surrounding context.
        Limited to 50 matches to manage context size.

        Args:
            pattern: Python regex pattern (matched per-line, not across lines).
            context_lines: Number of lines to show before and after each match.

        Returns:
            List of matches with line numbers and context.
        """
        try:
            result = core_tools.search(
                document_path, pattern=pattern, context_lines=context_lines
            )
        except re.error as e:
            return f"Invalid regex pattern '{pattern}': {e}. Use Python regex syntax."
        if result.total_matches == 0:
            return f"No matches found for pattern: {pattern}"

        # Limit matches to prevent context overflow
        matches_to_show = result.matches[:MAX_SEARCH_MATCHES]
        truncated = result.total_matches > MAX_SEARCH_MATCHES

        output_parts = [f"Found {result.total_matches} matches for '{pattern}'"]
        if truncated:
            output_parts[0] += f" (showing first {MAX_SEARCH_MATCHES})"
        output_parts[0] += ":\n"

        for match in matches_to_show:
            output_parts.append(_format_search_match(match))

        return "\n".join(output_parts)

    return search


def _create_view_page_tool(document_path: Path) -> LangChainTool:
    """Create a view_page tool bound to a document.

    Returns a Command that updates agent state with pending image data.
    The image is injected as a HumanMessage after all tool responses complete.

    Args:
        document_path: Path to the document.

    Returns:
        LangChain BaseTool instance.
    """

    @tool
    def view_page(
        page: int,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[str]:
        """View a specific page of the document as an image.

        For paginated formats like PDF, this renders the page visually.
        For text-based formats, returns a message indicating pages not applicable.

        Args:
            page: Page number to view (1-indexed).
            tool_call_id: Injected by LangGraph, used for tool response.

        Returns:
            Command updating state with image data for later injection.
        """
        try:
            result = core_tools.view_page(document_path, page=page)
        except ValueError as e:
            # Page couldn't be rendered (too large, out of range, etc.)
            # Return error as tool message so agent can try a different page
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=(f"ERROR: {e}. Try a DIFFERENT page number."),
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        if result.not_applicable:
            # No image - just return text message via Command
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=(
                                "This document format doesn't have visual pages. "
                                "Use read_lines or search instead."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        # These are guaranteed non-None when not_applicable is False
        page_num = result.page_number
        total = result.total_pages
        image_data = result.image_base64
        if page_num is None or total is None or image_data is None:
            raise ValueError("view_page result missing required fields")

        # Return Command with ToolMessage AND pending image for later injection
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Page {page_num} of {total} "
                        f"rendered. Image will be shown after tool execution.",
                        tool_call_id=tool_call_id,
                    )
                ],
                "pending_images": [
                    PendingImage(
                        page=page_num,
                        total_pages=total,
                        image_base64=image_data,
                    )
                ],
            }
        )

    return view_page


def _create_list_visual_content_tool(document_path: Path) -> LangChainTool:
    """Create a list_visual_content tool bound to a document.

    Args:
        document_path: Path to the document.

    Returns:
        LangChain BaseTool instance.
    """

    @tool
    def list_visual_content() -> str:
        """List visual content elements in the document.

        Discovers figures, tables, images, and diagrams in the document.
        Use this in visual mode to find visual elements to ask questions about.

        Returns:
            List of visual elements with their types, references, and locations.
        """
        result = core_tools.list_visual_content(document_path)
        if result.total_items == 0:
            return "No visual content (figures, tables, images) found."

        output_parts = [f"Found {result.total_items} visual elements:\n"]
        for item in result.items:
            parts = [f"- {item.content_type}: {item.reference}"]
            if item.alt_text:
                parts.append(f"  Caption: {item.alt_text}")
            if item.location:
                parts.append(f"  Location: {item.location}")
            output_parts.append("\n".join(parts))

        return "\n".join(output_parts)

    return list_visual_content


def create_document_tools(document_path: Path) -> list[LangChainTool]:
    """Create document exploration tools bound to a specific document.

    This factory creates LangChain tools that are pre-configured to work with
    a specific document. Each tool returned is a callable that can be bound
    to a LangGraph agent.

    Args:
        document_path: Path to the document to explore.

    Returns:
        List of LangChain tools for document exploration.

    Example:
        >>> tools = create_document_tools(Path("/path/to/doc.txt"))
        >>> agent = create_react_agent(model, tools)
    """
    return [
        _create_read_lines_tool(document_path),
        _create_search_tool(document_path),
        _create_view_page_tool(document_path),
        _create_list_visual_content_tool(document_path),
    ]
