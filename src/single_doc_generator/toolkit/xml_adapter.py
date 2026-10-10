"""XML/JATS file adapter for document toolkit.

This adapter handles XML files, specifically PubMed JATS (Journal Article Tag Suite)
format. It extracts readable text content while preserving section structure,
supports regex search on extracted content, and extracts visual content metadata
from figure and table elements.

JATS is a standard XML format used by PubMed Central for scientific articles.
It provides semantic markup for article structure including sections, figures,
tables, and references.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import defusedxml.ElementTree as DefusedET

from single_doc_generator.models import (
    ReadLinesResult,
    SearchMatch,
    SearchResult,
    ViewPageResult,
    VisualContent,
    VisualContentResult,
)
from single_doc_generator.toolkit.base import DocumentAdapter

# Common XML namespaces in JATS documents
JATS_NAMESPACES = {
    "jats": "https://jats.nlm.nih.gov/ns/archiving/1.4/",
    "xlink": "http://www.w3.org/1999/xlink",
    "mml": "http://www.w3.org/1998/Math/MathML",
    "oai": "http://www.openarchives.org/OAI/2.0/",
}


class XMLAdapter(DocumentAdapter):
    """Adapter for XML files, specifically JATS format.

    Provides document exploration tools for PubMed JATS XML files. Text is
    extracted from article content (title, abstract, body sections) and split
    into logical lines preserving section structure. Visual content detection
    parses `<fig>` and `<table-wrap>` elements.

    The adapter handles JATS XML which may be wrapped in OAI-PMH record elements.
    It automatically detects and parses the article element regardless of wrapper.

    Attributes:
        file_path: Path to the XML file.
    """

    def __init__(self, file_path: Path) -> None:
        """Initialize XML adapter and parse content.

        Args:
            file_path: Path to the XML file.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If XML cannot be parsed.
        """
        super().__init__(file_path)
        self._lines: list[str] = []
        self._visual_content: list[VisualContent] = []
        self._load_content()

    def _load_content(self) -> None:
        """Load XML and extract text content as lines.

        Parses the JATS XML structure and extracts readable text content,
        preserving section headings and paragraph structure. Also extracts
        metadata about figures and tables for visual content discovery.

        Uses defusedxml for safe parsing of untrusted XML content.

        Raises:
            ValueError: If XML parsing fails.
        """
        try:
            tree = DefusedET.parse(self.file_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Cannot parse XML: {e}") from e

        if root is None:
            raise ValueError("XML document has no root element")

        # Find the article element, handling possible OAI wrapper
        article = self._find_article_element(root)
        if article is None:
            raise ValueError("No article element found in XML")

        # Extract text content from article
        self._lines = self._extract_text_lines(article)

        # Extract visual content metadata
        self._visual_content = self._extract_visual_content(article)

    def _find_article_element(self, root: ET.Element) -> ET.Element | None:
        """Find the JATS article element, handling various wrapper structures.

        JATS XML from PubMed may be wrapped in OAI-PMH record/metadata elements.
        This method searches for the article element regardless of wrapper.

        Args:
            root: Root element of the XML tree.

        Returns:
            The article Element if found, None otherwise.
        """
        # Check if root is the article itself (various namespace possibilities)
        if self._is_article_element(root):
            return root

        # Search for article element in descendants
        # Try without namespace first
        article = root.find(".//article")
        if article is not None:
            return article

        # Try with common JATS namespaces
        for ns_prefix, ns_uri in JATS_NAMESPACES.items():
            if ns_prefix == "jats":
                article = root.find(f".//{{{ns_uri}}}article")
                if article is not None:
                    return article

        # Last resort: search all descendants for any element named 'article'
        for elem in root.iter():
            if self._is_article_element(elem):
                return elem

        return None

    def _is_article_element(self, elem: ET.Element) -> bool:
        """Check if an element is a JATS article element.

        Args:
            elem: Element to check.

        Returns:
            True if element is an article element.
        """
        # Get local name without namespace
        tag = elem.tag
        if "}" in tag:
            tag = tag.split("}")[1]
        return tag == "article"

    def _get_local_name(self, elem: ET.Element) -> str:
        """Get element's local name without namespace prefix.

        Args:
            elem: Element to get name from.

        Returns:
            Local name string.
        """
        tag = elem.tag
        if "}" in tag:
            return tag.split("}")[1]
        return tag

    def _extract_text_lines(self, article: ET.Element) -> list[str]:
        """Extract readable text content from article as lines.

        Processes the article structure to extract text in reading order:
        1. Article title
        2. Abstract
        3. Body sections with headings
        4. Preserves paragraph structure

        Args:
            article: The JATS article element.

        Returns:
            List of text lines representing document content.
        """
        lines: list[str] = []

        # Extract front matter (title, abstract)
        front = self._find_child_by_local_name(article, "front")
        if front is not None:
            lines.extend(self._extract_front_matter(front))

        # Extract body content
        body = self._find_child_by_local_name(article, "body")
        if body is not None:
            lines.extend(self._extract_body_content(body))

        # Extract back matter (references summary)
        back = self._find_child_by_local_name(article, "back")
        if back is not None:
            lines.extend(self._extract_back_matter(back))

        return lines

    def _find_child_by_local_name(
        self, parent: ET.Element, local_name: str
    ) -> ET.Element | None:
        """Find direct child element by local name, ignoring namespace.

        Args:
            parent: Parent element to search in.
            local_name: Local name to match.

        Returns:
            First matching child element, or None if not found.
        """
        for child in parent:
            if self._get_local_name(child) == local_name:
                return child
        return None

    def _find_all_by_local_name(
        self, parent: ET.Element, local_name: str
    ) -> list[ET.Element]:
        """Find all descendant elements by local name, ignoring namespace.

        Args:
            parent: Parent element to search in.
            local_name: Local name to match.

        Returns:
            List of matching elements.
        """
        results: list[ET.Element] = []
        for elem in parent.iter():
            if self._get_local_name(elem) == local_name:
                results.append(elem)
        return results

    def _extract_front_matter(self, front: ET.Element) -> list[str]:
        """Extract title and abstract from front matter.

        Args:
            front: The front element containing article metadata.

        Returns:
            List of text lines from front matter.
        """
        lines: list[str] = []

        # Find article-meta
        article_meta = self._find_child_by_local_name(front, "article-meta")
        if article_meta is None:
            return lines

        # Extract title
        title_group = self._find_child_by_local_name(article_meta, "title-group")
        if title_group is not None:
            article_title = self._find_child_by_local_name(title_group, "article-title")
            if article_title is not None:
                title_text = self._get_element_text(article_title)
                if title_text:
                    lines.append(f"# {title_text}")
                    lines.append("")  # Blank line after title

        # Extract abstract
        abstract = self._find_child_by_local_name(article_meta, "abstract")
        if abstract is not None:
            lines.append("## Abstract")
            lines.append("")
            abstract_text = self._get_element_text(abstract)
            if abstract_text:
                lines.append(abstract_text)
                lines.append("")

        return lines

    def _extract_body_content(self, body: ET.Element) -> list[str]:
        """Extract text content from body sections.

        Processes sections recursively, preserving headings and paragraph structure.

        Args:
            body: The body element containing article content.

        Returns:
            List of text lines from body.
        """
        lines: list[str] = []

        for child in body:
            local_name = self._get_local_name(child)
            if local_name == "sec":
                lines.extend(self._extract_section(child, level=2))
            elif local_name == "p":
                para_text = self._get_element_text(child)
                if para_text:
                    lines.append(para_text)
                    lines.append("")
            elif local_name == "boxed-text":
                # Handle boxed content like key messages
                lines.extend(self._extract_boxed_text(child))

        return lines

    def _extract_section(self, section: ET.Element, level: int = 2) -> list[str]:
        """Extract content from a section element.

        Args:
            section: The sec element to extract from.
            level: Heading level (2 = ##, 3 = ###, etc.).

        Returns:
            List of text lines from section.
        """
        lines: list[str] = []
        lines.extend(self._extract_section_title(section, level))

        # Process section content
        for child in section:
            lines.extend(self._process_section_child(child, level))

        return lines

    def _extract_section_title(self, section: ET.Element, level: int) -> list[str]:
        """Extract section title as heading line.

        Args:
            section: The section element.
            level: Heading level for markdown formatting.

        Returns:
            List with heading line and blank line, or empty list.
        """
        title_elem = self._find_child_by_local_name(section, "title")
        if title_elem is None:
            return []
        title_text = self._get_element_text(title_elem)
        if not title_text:
            return []
        heading_prefix = "#" * min(level, 6)
        return [f"{heading_prefix} {title_text}", ""]

    def _process_section_child(self, child: ET.Element, level: int) -> list[str]:
        """Process a child element within a section.

        Args:
            child: Child element to process.
            level: Current heading level for nested sections.

        Returns:
            List of text lines from this child.
        """
        local_name = self._get_local_name(child)

        if local_name == "sec":
            return self._extract_section(child, level=level + 1)
        if local_name == "p":
            return self._extract_paragraph(child)
        if local_name == "fig":
            return self._extract_figure_ref(child)
        if local_name == "table-wrap":
            return self._extract_table_ref(child)
        if local_name == "list":
            return self._extract_list(child)
        return []

    def _extract_paragraph(self, para: ET.Element) -> list[str]:
        """Extract paragraph content."""
        para_text = self._get_element_text(para)
        return [para_text, ""] if para_text else []

    def _extract_figure_ref(self, fig: ET.Element) -> list[str]:
        """Extract figure reference line."""
        fig_ref = self._format_figure_reference(fig)
        return [fig_ref, ""] if fig_ref else []

    def _extract_table_ref(self, table: ET.Element) -> list[str]:
        """Extract table reference line."""
        table_ref = self._format_table_reference(table)
        return [table_ref, ""] if table_ref else []

    def _extract_boxed_text(self, boxed: ET.Element) -> list[str]:
        """Extract content from boxed-text element.

        Args:
            boxed: The boxed-text element.

        Returns:
            List of text lines from boxed content.
        """
        lines: list[str] = []

        # Get caption/title
        caption = self._find_child_by_local_name(boxed, "caption")
        if caption is not None:
            title = self._find_child_by_local_name(caption, "title")
            if title is not None:
                title_text = self._get_element_text(title)
                if title_text:
                    lines.append(f"### {title_text}")
                    lines.append("")

        # Extract paragraphs
        for child in boxed:
            local_name = self._get_local_name(child)
            if local_name == "p":
                para_text = self._get_element_text(child)
                if para_text:
                    lines.append(para_text)
                    lines.append("")

        return lines

    def _extract_list(self, list_elem: ET.Element) -> list[str]:
        """Extract content from list element.

        Args:
            list_elem: The list element.

        Returns:
            List of text lines representing the list.
        """
        lines: list[str] = []

        for item in list_elem:
            if self._get_local_name(item) == "list-item":
                item_text = self._get_element_text(item)
                if item_text:
                    lines.append(f"• {item_text}")

        if lines:
            lines.append("")  # Blank line after list

        return lines

    def _extract_back_matter(self, back: ET.Element) -> list[str]:
        """Extract summary of back matter (references count, acknowledgments).

        Args:
            back: The back element containing references and acknowledgments.

        Returns:
            List of text lines from back matter.
        """
        lines: list[str] = []

        # Count references
        ref_list = self._find_child_by_local_name(back, "ref-list")
        if ref_list is not None:
            refs = self._find_all_by_local_name(ref_list, "ref")
            if refs:
                lines.append("## References")
                lines.append("")
                lines.append(f"[{len(refs)} references]")
                lines.append("")

        return lines

    def _format_figure_reference(self, fig: ET.Element) -> str:
        """Format a reference to a figure element.

        Args:
            fig: The fig element.

        Returns:
            Formatted string describing the figure.
        """
        label_elem = self._find_child_by_local_name(fig, "label")
        label = self._get_element_text(label_elem) if label_elem is not None else ""

        caption_elem = self._find_child_by_local_name(fig, "caption")
        caption = ""
        if caption_elem is not None:
            # Get first paragraph of caption
            p_elem = self._find_child_by_local_name(caption_elem, "p")
            if p_elem is not None:
                caption = self._get_element_text(p_elem)
            else:
                caption = self._get_element_text(caption_elem)

        if label and caption:
            return f"[{label}: {caption}]"
        if label:
            return f"[{label}]"
        if caption:
            return f"[Figure: {caption}]"
        return ""

    def _format_table_reference(self, table_wrap: ET.Element) -> str:
        """Format a reference to a table element.

        Args:
            table_wrap: The table-wrap element.

        Returns:
            Formatted string describing the table.
        """
        label_elem = self._find_child_by_local_name(table_wrap, "label")
        label = self._get_element_text(label_elem) if label_elem is not None else ""

        caption_elem = self._find_child_by_local_name(table_wrap, "caption")
        caption = ""
        if caption_elem is not None:
            title_elem = self._find_child_by_local_name(caption_elem, "title")
            if title_elem is not None:
                caption = self._get_element_text(title_elem)
            else:
                caption = self._get_element_text(caption_elem)

        if label and caption:
            return f"[{label}: {caption}]"
        if label:
            return f"[{label}]"
        if caption:
            return f"[Table: {caption}]"
        return ""

    def _get_element_text(self, elem: ET.Element) -> str:
        """Get all text content from an element, including nested elements.

        Concatenates text from all descendant text nodes, handling inline
        elements like <italic>, <bold>, <xref>, etc.

        Args:
            elem: Element to extract text from.

        Returns:
            Combined text content, cleaned and normalized.
        """
        # Collect all text parts
        parts: list[str] = []

        def collect_text(node: ET.Element) -> None:
            if node.text:
                parts.append(node.text)
            for child in node:
                collect_text(child)
                if child.tail:
                    parts.append(child.tail)

        collect_text(elem)

        # Join and clean up whitespace
        text = "".join(parts)
        # Normalize whitespace (collapse multiple spaces/newlines)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_visual_content(self, article: ET.Element) -> list[VisualContent]:
        """Extract visual content metadata from article.

        Finds all figure and table elements and creates VisualContent entries.

        Args:
            article: The article element.

        Returns:
            List of VisualContent items.
        """
        items: list[VisualContent] = []

        # Find all figure elements
        for fig in self._find_all_by_local_name(article, "fig"):
            fig_item = self._parse_figure(fig)
            if fig_item:
                items.append(fig_item)

        # Find all table elements
        for table_wrap in self._find_all_by_local_name(article, "table-wrap"):
            table_item = self._parse_table(table_wrap)
            if table_item:
                items.append(table_item)

        # Find supplementary materials
        for supp in self._find_all_by_local_name(article, "supplementary-material"):
            supp_item = self._parse_supplementary(supp)
            if supp_item:
                items.append(supp_item)

        return items

    def _parse_figure(self, fig: ET.Element) -> VisualContent | None:
        """Parse a figure element into VisualContent.

        Args:
            fig: The fig element.

        Returns:
            VisualContent for the figure, or None if invalid.
        """
        fig_id = fig.get("id", "")

        # Get label
        label_elem = self._find_child_by_local_name(fig, "label")
        label = self._get_element_text(label_elem) if label_elem is not None else ""

        # Get caption text
        caption_elem = self._find_child_by_local_name(fig, "caption")
        caption = ""
        if caption_elem is not None:
            p_elem = self._find_child_by_local_name(caption_elem, "p")
            if p_elem is not None:
                caption = self._get_element_text(p_elem)

        # Get graphic reference (the actual image file)
        graphic = self._find_child_by_local_name(fig, "graphic")
        reference = ""
        if graphic is not None:
            # Try xlink:href attribute
            for attr_name, attr_value in graphic.attrib.items():
                if "href" in attr_name:
                    reference = attr_value
                    break

        return VisualContent(
            content_type="figure",
            reference=reference or fig_id or label,
            alt_text=caption if caption else (label if label else None),
            location=f"id:{fig_id}" if fig_id else None,
        )

    def _parse_table(self, table_wrap: ET.Element) -> VisualContent | None:
        """Parse a table-wrap element into VisualContent.

        Args:
            table_wrap: The table-wrap element.

        Returns:
            VisualContent for the table, or None if invalid.
        """
        table_id = table_wrap.get("id", "")

        # Get label
        label_elem = self._find_child_by_local_name(table_wrap, "label")
        label = self._get_element_text(label_elem) if label_elem is not None else ""

        # Get caption/title
        caption_elem = self._find_child_by_local_name(table_wrap, "caption")
        caption = ""
        if caption_elem is not None:
            title_elem = self._find_child_by_local_name(caption_elem, "title")
            if title_elem is not None:
                caption = self._get_element_text(title_elem)

        return VisualContent(
            content_type="table",
            reference=table_id or label,
            alt_text=caption if caption else (label if label else None),
            location=f"id:{table_id}" if table_id else None,
        )

    def _parse_supplementary(self, supp: ET.Element) -> VisualContent | None:
        """Parse a supplementary-material element into VisualContent.

        Args:
            supp: The supplementary-material element.

        Returns:
            VisualContent for the supplementary material, or None if invalid.
        """
        supp_id = supp.get("id", "")

        # Get caption/title
        caption_elem = self._find_child_by_local_name(supp, "caption")
        caption = ""
        if caption_elem is not None:
            title_elem = self._find_child_by_local_name(caption_elem, "title")
            if title_elem is not None:
                caption = self._get_element_text(title_elem)

        # Get media reference
        media = self._find_child_by_local_name(supp, "media")
        reference = ""
        if media is not None:
            for attr_name, attr_value in media.attrib.items():
                if "href" in attr_name:
                    reference = attr_value
                    break

        return VisualContent(
            content_type="supplementary",
            reference=reference or supp_id,
            alt_text=caption if caption else None,
            location=f"id:{supp_id}" if supp_id else None,
        )

    @property
    def total_lines(self) -> int:
        """Total number of lines in extracted text."""
        return len(self._lines)

    def read_lines(self, start: int = 1, end: int | None = None) -> ReadLinesResult:
        """Read a range of lines from the extracted XML text.

        Args:
            start: Starting line number (1-indexed, inclusive). Defaults to 1.
            end: Ending line number (1-indexed, inclusive). If None, reads to EOF.

        Returns:
            ReadLinesResult containing the requested lines with line numbers.

        Raises:
            ValueError: If start < 1 or start > total_lines.
        """
        if start < 1:
            raise ValueError(f"start must be >= 1, got {start}")
        if end is not None and end < start:
            raise ValueError(f"end ({end}) must be >= start ({start})")
        if start > self.total_lines and self.total_lines > 0:
            raise ValueError(f"start {start} exceeds total lines {self.total_lines}")

        # Handle empty content
        if self.total_lines == 0:
            return ReadLinesResult(
                lines=[],
                total_lines=0,
                start_line=start,
                end_line=start,
            )

        # Determine actual end line
        actual_end = end if end is not None else self.total_lines
        actual_end = min(actual_end, self.total_lines)

        # Convert to 0-indexed for slicing
        start_idx = start - 1
        end_idx = actual_end

        lines_with_numbers = [
            (i + 1, self._lines[i]) for i in range(start_idx, end_idx)
        ]

        return ReadLinesResult(
            lines=lines_with_numbers,
            total_lines=self.total_lines,
            start_line=start,
            end_line=actual_end,
        )

    def search(self, pattern: str, context_lines: int = 0) -> SearchResult:
        """Search for regex pattern in extracted XML text.

        Args:
            pattern: Regular expression pattern to search for.
            context_lines: Number of lines to include before/after each match.

        Returns:
            SearchResult with all matches and their context.

        Raises:
            re.error: If pattern is invalid regex.
        """
        regex = re.compile(pattern)
        matches: list[SearchMatch] = []

        for i, line in enumerate(self._lines):
            if regex.search(line):
                line_num = i + 1  # 1-indexed

                # Gather context before
                context_before: list[tuple[int, str]] = []
                for j in range(max(0, i - context_lines), i):
                    context_before.append((j + 1, self._lines[j]))

                # Gather context after
                context_after: list[tuple[int, str]] = []
                for j in range(i + 1, min(len(self._lines), i + 1 + context_lines)):
                    context_after.append((j + 1, self._lines[j]))

                matches.append(
                    SearchMatch(
                        line_number=line_num,
                        line_content=line,
                        context_before=context_before,
                        context_after=context_after,
                    )
                )

        return SearchResult(
            pattern=pattern,
            matches=matches,
            total_matches=len(matches),
        )

    def view_page(self, page: int) -> ViewPageResult:
        """View page is not applicable for XML files.

        XML files have no page structure, so this always returns
        a not_applicable result.

        Args:
            page: Page number (ignored for XML files).

        Returns:
            ViewPageResult with not_applicable=True.
        """
        return ViewPageResult(not_applicable=True, page_number=page)

    def list_visual_content(self) -> VisualContentResult:
        """List visual content (figures, tables, supplements) in the XML.

        Parses JATS figure (`<fig>`), table (`<table-wrap>`), and supplementary
        material elements to build a manifest of visual content.

        Returns:
            VisualContentResult with list of visual elements found.
        """
        return VisualContentResult(
            items=self._visual_content, total_items=len(self._visual_content)
        )
