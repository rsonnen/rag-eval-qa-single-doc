"""Shared test fixtures for single_doc_generator tests."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the fixtures directory path."""
    return FIXTURES_DIR


@pytest.fixture
def shakespeare_file() -> Path:
    """Get the Complete Works of Shakespeare text file."""
    return FIXTURES_DIR / "shakespeare.txt"


@pytest.fixture
def rpg_spell_file() -> Path:
    """Get a sample RPG spell markdown file."""
    return FIXTURES_DIR / "abhorrent-apparition.md"


@pytest.fixture
def kubernetes_docs_file() -> Path:
    """Get Kubernetes docs markdown file with image references."""
    return FIXTURES_DIR / "kubernetes_dashboard.md"


@pytest.fixture
def court_opinion_pdf() -> Path:
    """Get a court opinion PDF (Utah Supreme Court patent case)."""
    return FIXTURES_DIR / "court_opinion.pdf"


@pytest.fixture
def arxiv_paper_pdf() -> Path:
    """Get an arXiv paper PDF (computational biology)."""
    return FIXTURES_DIR / "arxiv_paper.pdf"


@pytest.fixture
def pubmed_xml_file() -> Path:
    """Get a PubMed JATS XML file for testing.

    Uses a real genetics paper (PMC11605797) with figures and tables.
    """
    return FIXTURES_DIR / "pubmed_paper.xml"


@pytest.fixture
def pubmed_xml_with_figures() -> Path:
    """Get a PubMed XML file with multiple figures.

    Uses PMC11393823 which has detailed figure elements.
    """
    return FIXTURES_DIR / "pubmed_paper_with_figures.xml"


@pytest.fixture
def django_docs_html() -> Path:
    """Get a Django documentation HTML file (CSRF protection howto)."""
    return FIXTURES_DIR / "django_csrf.html"


@pytest.fixture
def git_stage_adoc() -> Path:
    """Get a Git documentation AsciiDoc file (git-stage man page)."""
    return FIXTURES_DIR / "git_stage.adoc"
