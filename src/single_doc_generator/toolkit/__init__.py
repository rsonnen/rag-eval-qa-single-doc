"""Document toolkit for exploring files via agent tools.

This module provides format-agnostic tools for document exploration:
- read_lines: Read a range of lines with line numbers
- search: Regex search with context
- view_page: Render page as image (format-dependent)
- list_visual_content: Discover figures, tables, images
"""

from single_doc_generator.toolkit.tools import (
    list_visual_content,
    read_lines,
    search,
    view_page,
)

__all__ = ["list_visual_content", "read_lines", "search", "view_page"]
