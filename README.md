# Single Document Question Generator

A tool that generates validated question/answer pairs from documents for evaluating RAG (Retrieval-Augmented Generation) systems.

## Companion Corpora

This tool is designed to work with the RAG evaluation corpora published under [github.com/rsonnen](https://github.com/rsonnen):

- [rag-eval-arxiv-papers](https://github.com/rsonnen/rag-eval-arxiv-papers) - Academic papers across ML/AI domains
- [rag-eval-cookbooks](https://github.com/rsonnen/rag-eval-cookbooks) - Historical cookbooks
- [rag-eval-court-opinions](https://github.com/rsonnen/rag-eval-court-opinions) - Federal court opinions
- [rag-eval-historical-newspapers](https://github.com/rsonnen/rag-eval-historical-newspapers) - Chronicling America archive
- [rag-eval-internet-rfcs](https://github.com/rsonnen/rag-eval-internet-rfcs) - IETF RFC documents
- [rag-eval-public-domain-books](https://github.com/rsonnen/rag-eval-public-domain-books) - Gutenberg/Internet Archive books
- [rag-eval-pubmed-oa](https://github.com/rsonnen/rag-eval-pubmed-oa) - PubMed Open Access medical literature
- [rag-eval-rpg-rulebooks](https://github.com/rsonnen/rag-eval-rpg-rulebooks) - Open-licensed tabletop RPG content
- [rag-eval-technical-docs](https://github.com/rsonnen/rag-eval-technical-docs) - Technical documentation
- [rag-eval-vintage-cocktails](https://github.com/rsonnen/rag-eval-vintage-cocktails) - Pre-Prohibition cocktail manuals

Each corpus includes a `corpus.yaml` configuration and document metadata. Some corpora include pre-generated Q/A sets created with this tool. Clone a corpus, download its documents, and use this generator to create evaluation datasets for your RAG system.

## What It Does

Given a document and corpus configuration, this tool:

1. **Generates** domain-appropriate Q/A pairs using an LLM agent that explores the document with tools
2. **Deduplicates** questions semantically to prevent the same question asked different ways
3. **Validates** each question using a *different* LLM to confirm answerability and correctness

The result is a set of Q/A pairs suitable for evaluating whether a RAG system can correctly retrieve and answer questions about the document.

### Key Design Decisions

- **No semantic search**: The generator explores documents via line reading and regex search, not embeddings. This prevents bias toward content that retrieval systems naturally find.
- **Domain adaptation via meta-prompting**: Instead of custom prompts per corpus, the system uses corpus context and evaluation scenarios to let the LLM adapt its questioning style.
- **Different models for generate vs validate**: Prevents self-confirmation bias.
- **Visual mode**: For documents with figures/tables/diagrams, generates questions requiring understanding of visual content.

## Installation

Requires Python 3.11+. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/rsonnen/rag-eval-qa-single-doc.git
cd rag-eval-qa-single-doc

# Install dependencies
make install-dev

# Or manually with uv
uv sync
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your API credentials
```

All LLM calls go through the OpenAI-compatible API (or LiteLLM proxy if configured).

## Quick Start

```bash
# Generate 5 textual questions from a PDF
uv run python scripts/generate_questions.py \
    --document /path/to/document.pdf \
    --corpus /path/to/corpus/dir \
    --scenario rag_eval \
    --count 5

# Generate visual questions (requires vision-capable models)
uv run python scripts/generate_questions.py \
    --document /path/to/document.pdf \
    --corpus /path/to/corpus/dir \
    --scenario rag_eval \
    --mode visual \
    --count 3

# Save results to JSON file
uv run python scripts/generate_questions.py \
    --document /path/to/document.pdf \
    --corpus /path/to/corpus/dir \
    --scenario rag_eval \
    --output results.json
```

## CLI Reference: generate_questions.py (Single Document)

```
usage: generate_questions.py [-h] --document DOCUMENT --corpus CORPUS
                             --scenario SCENARIO [--mode {textual,visual}]
                             [--count COUNT] [--max-failures MAX_FAILURES]
                             [--output OUTPUT] [--verbose]

Arguments:
  --document PATH       Path to document (PDF, XML, markdown, or text file)
  --corpus PATH         Path to corpus directory containing corpus.yaml
  --scenario NAME       Scenario name from corpus.yaml (e.g., rag_eval)
  --mode {textual,visual}
                        Generation mode (default: textual)
  --count N             Number of questions to generate (default: 5)
  --max-failures N      Consecutive failures before exhaustion (default: 5)
  --output PATH         Output JSON file (prints to stdout if not specified)
  --verbose             Enable debug logging
```

### Exit Conditions

Generation stops when either:
- Target count is reached (success)
- Document is exhausted (consecutive failures exceed threshold)
- Generator explicitly reports exhaustion (e.g., no more visual content)

## CLI Reference: generate_corpus.py (Batch Processing)

Process all documents in a corpus directory with automatic resume capability.

```bash
# Process entire corpus, 5 questions per document
uv run python scripts/generate_corpus.py \
    --corpus /path/to/corpus \
    --scenario rag_eval

# Limit to first 10 documents (for testing)
uv run python scripts/generate_corpus.py \
    --corpus /path/to/corpus \
    --scenario rag_eval \
    --max-docs 10 \
    --target-per-doc 2

# Set total question limit across corpus
uv run python scripts/generate_corpus.py \
    --corpus /path/to/corpus \
    --scenario rag_eval \
    --total-target 100
```

```
usage: generate_corpus.py [-h] --corpus CORPUS --scenario SCENARIO
                          [--mode {textual,visual}] [--target-per-doc N]
                          [--total-target N] [--max-docs N]
                          [--max-failures N] [--db PATH] [--output PATH]
                          [--verbose]

Arguments:
  --corpus PATH         Path to corpus directory containing corpus.yaml
  --scenario NAME       Scenario name from corpus.yaml
  --mode {textual,visual}
                        Generation mode (default: textual)
  --target-per-doc N    Questions per document (default: 5)
  --total-target N      Stop after N total questions across corpus
  --max-docs N          Process at most N documents (for testing)
  --max-failures N      Consecutive failures before document exhaustion (default: 5)
  --db PATH             SQLite database file (default: ~/.cache/single_doc_generator/<hash>.db)
  --output PATH         Output JSON file (default: <corpus>/<scenario>_<mode>_questions.json)
  --verbose             Enable debug logging
```

### Resume Behavior

Resume is automatic and based on the SQLite database:

1. **Run command** - processes documents, commits each to database, writes JSON after each document
2. **Interrupt (Ctrl+C)** - progress saved to database (lose at most current document's work)
3. **Re-run same command** - skips documents already in database, processes remaining

The SQLite database is the single source of truth for processing state. The JSON output file is regenerated after each document for visibility during long runs.

### Network Filesystem Support

The database defaults to `~/.cache/single_doc_generator/<hash>.db` on local disk. This avoids SQLite locking issues when the corpus is on a network filesystem (SMB/CIFS/NFS). The hash is derived from the corpus path, so each corpus gets its own database file.

### Corpus Output Format

The JSON output contains only RAGAS-relevant data for evaluation consumers:

```json
{
  "corpus_name": "D&D 5e SRD Spells",
  "corpus_path": "/path/to/corpus",
  "scenario": "rag_eval",
  "mode": "textual",
  "questions": [
    {
      "question": "What is the casting time of Fireball?",
      "answer": "1 action",
      "source_document": "spells/fireball.md",
      "mode": "textual",
      "content_refs": []
    }
  ],
  "timestamp": "2026-01-08T10:30:00Z"
}
```

Processing statistics (documents processed, exhausted, accepted/rejected counts) are stored in the SQLite database only.

## Corpus Configuration

Each corpus requires a `corpus.yaml` file that provides context and defines evaluation scenarios.

### corpus.yaml Structure

```yaml
name: "Patent Law Court Opinions"

corpus_context: >
  Federal court opinions on patent infringement and intellectual property
  disputes, sourced from CourtListener. Each document contains case name,
  date, judges, and the full opinion text including procedural history,
  factual background, and legal analysis.

scenarios:
  rag_eval:
    name: "RAG System Evaluation"
    description: >
      Questions with exact answers from the opinion text to verify retrieval
      accuracy. "What was the patent number at issue?" (U.S. Patent 7,123,456),
      "Who represented the plaintiff?" (Smith & Jones LLP). Tests accurate
      extraction of case-specific facts.

  law_school:
    name: "Law School Exam"
    description: >
      Questions testing understanding of legal reasoning and precedent
      application. Questions should read like a law school exam, requiring
      analysis of the court's reasoning and application of legal principles.
```

### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Human-readable corpus name |
| `corpus_context` | Yes | Description of what this corpus is and where it came from |
| `scenarios` | Yes | Dictionary of named evaluation scenarios |

Each scenario contains:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Short display name |
| `description` | Yes | Full description of evaluation context, intended audience, and question style |

The `corpus_context` and `scenario.description` are injected into LLM prompts to guide question generation style without requiring per-corpus prompt engineering.

## The Pipeline

### 1. Generate

A LangGraph ReAct agent explores the document using four tools:

- **read_lines**: Read document content by line range (like viewing in an editor)
- **search**: Regex search with context lines (like grep)
- **view_page**: Render a page as an image (PDFs only, for visual mode)
- **list_visual_content**: Discover figures, tables, images

The agent receives corpus context and scenario description to shape its questioning style. It also receives the list of previously accepted questions to avoid duplicates.

When done exploring, the agent calls either:
- `submit_qa`: Submit a question/answer pair
- `report_exhausted`: Signal that no more unique questions can be generated

### 2. Deduplicate

Before expensive validation, a fast/cheap LLM checks if the new question is semantically equivalent to any accepted question. This catches cases where differently-worded questions ask the same thing.

### 3. Validate

A *different* LLM model receives the question and uses the same document tools to:
- Attempt to answer the question from document content only
- Compare its answer to the proposed ground truth
- Check for triviality (answerable without reading the document)
- Check for ambiguity (multiple valid interpretations)
- In visual mode: verify the question requires understanding visual content

Validation rejects questions that are:
- **Unanswerable**: Cannot be answered from the document
- **Wrong answer**: Validator's answer differs from proposed ground truth
- **Ambiguous**: Multiple valid interpretations
- **Trivial**: Answerable without reading the specific document

### Flow Diagram

```
                    +----------+
                    | Generate |
                    +----+-----+
                         |
                         v
                    +----------+     duplicate    +--------+
                    | Dedupe   |----------------->| Reject |--+
                    +----+-----+                  +--------+  |
                         |                                    |
                         | unique                             |
                         v                                    |
                    +----------+       fail       +--------+  |
                    | Validate |----------------->| Reject |--+
                    +----+-----+                  +--------+  |
                         |                                    |
                         | pass                               |
                         v                                    |
                    +----------+                              |
                    | Accept   |                              |
                    +----+-----+                              |
                         |                                    |
                         v                                    |
                  [check exit conditions]<--------------------+
                         |
          +--------------+--------------+
          |                             |
          | continue                    | target reached OR exhausted
          v                             v
      (loop back)                    +-----+
                                     | END |
                                     +-----+
```

## Textual vs Visual Modes

### Textual Mode (default)

- Uses `read_lines` and `search` tools
- Generates factual recall questions from document text
- Works with all supported formats
- Does not require vision-capable models

### Visual Mode

- Also uses `view_page` and `list_visual_content` tools
- Generates questions requiring understanding of figures, tables, diagrams
- First checks if visual content exists; if not, exits gracefully
- Requires vision-capable generator and validator models
- Tracks viewed pages across attempts to avoid revisiting trivial content

Visual mode rejects questions about non-substantive content:
- Library stamps, ownership marks
- Age-related damage (foxing, stains)
- Blank pages, scanning artifacts
- Page numbers, decorative borders

## Configuration

### config.yaml

Application configuration specifying models for each pipeline stage. Model names are LiteLLM model identifiers - configure your LiteLLM proxy to map these to actual provider models.

```yaml
# Question generation - capable model with tool use and vision
generator_model: gpt-5.2

# Validation - different model to prevent self-confirmation
validator_model: claude-sonnet-4.5

# Deduplication - fast/cheap model, no tools needed
deduplicator_model: gpt-5-mini
```

**Requirements**:
- Generator and validator must use different models (configuration validates this at startup)
- Generator and validator models must support tool use
- For visual mode, generator and validator must support vision

### Supported Document Formats

| Format | Extensions | read_lines | search | view_page | list_visual_content |
|--------|-----------|-----------|--------|-----------|---------------------|
| Plain text | `.txt` | Yes | Yes | Not applicable | Empty |
| Markdown | `.md`, `.markdown` | Yes | Yes | Not applicable | Parses image refs |
| PDF | `.pdf` | Extracted text | Yes | Renders page | Extracts from structure |
| XML (JATS) | `.xml` | Parsed to text | Yes | Not applicable | Parses figure elements |
| HTML | `.html` | Parsed to text | Yes | Not applicable | Parses img/table tags |
| AsciiDoc | `.adoc` | Parsed to text | Yes | Not applicable | Parses image macros |

## Output Format

Generation produces a `GenerationResult` with:

```json
{
  "document": "/path/to/document.pdf",
  "corpus": "/path/to/corpus",
  "scenario": "rag_eval",
  "mode": "textual",
  "target_count": 5,
  "accepted": [
    {
      "question": "What was the patent number at issue in this case?",
      "answer": "U.S. Patent 7,123,456",
      "source_document": "/path/to/document.pdf",
      "mode": "textual",
      "content_refs": []
    }
  ],
  "rejected": [
    {
      "question": "What is patent law?",
      "answer": "A form of intellectual property law",
      "rejection_reason": "trivial",
      "rejection_detail": "Can be answered without reading this document",
      "duplicate_of": null
    }
  ],
  "stats": {
    "total_attempts": 8,
    "accepted_count": 5,
    "rejected_count": 3,
    "rejection_breakdown": {
      "trivial": 2,
      "duplicate": 1
    },
    "exhausted": false,
    "exhaustion_reason": null
  },
  "timestamp": "2026-01-08T10:30:00Z"
}
```

### Rejection Reasons

| Reason | Description |
|--------|-------------|
| `duplicate` | Semantically equivalent to an accepted question |
| `unanswerable` | Cannot be answered from document content |
| `wrong_answer` | Validator's answer differs from proposed ground truth |
| `ambiguous` | Multiple valid interpretations |
| `trivial` | Answerable without reading the specific document |
| `validation_failed` | Validator did not complete (error) |

## Development

### Make Commands

```bash
make help           # Show all targets

# Quality checks
make format         # Run ruff format
make lint           # Run ruff check with auto-fix
make typecheck      # Run mypy type checking
make security       # Run bandit security scanning
make test           # Run unit tests
make test-integration  # Run integration tests (requires LLM, costs money)
make coverage       # Run tests with coverage report
make all            # Run format, lint, security, typecheck, test
make full           # Run all + integration tests

# Dependencies
make install-dev    # Install all dependencies
make sync           # Sync from uv.lock
make update-lock    # Update uv.lock
make clean          # Clean build artifacts
```

### Running Single Tests

```bash
# Without coverage (faster)
uv run pytest tests/unit/test_generator.py::test_function -v
```

### Integration Tests

Integration tests make real LLM calls and cost money. They are excluded by default:

```bash
# Run integration tests explicitly
make test-integration

# Or with pytest
uv run pytest -m integration -v --no-cov
```

### Project Structure

```
rag-eval-qa-single-doc/
  config.yaml           # Model configuration
  pyproject.toml        # Project metadata and dependencies
  Makefile              # Development commands

  scripts/
    generate_questions.py   # Single document CLI
    generate_corpus.py      # Corpus batch processing CLI

  src/single_doc_generator/
    config.py           # Configuration loading (corpus + app)
    models.py           # Pydantic models for all data types
    llm.py              # LLM factory (ChatOpenAI via LiteLLM)
    prompt_loader.py    # Jinja2 prompt template loading
    agent.py            # Custom LangGraph agent with vision support
    generator.py        # Question generator agent
    validator.py        # Question validator agent
    deduplicator.py     # Semantic deduplication
    orchestrator.py     # Pipeline coordination (single document)
    corpus_processor.py # Batch processing with resume
    persistence.py      # SQLite state management

    toolkit/
      base.py           # DocumentAdapter abstract interface
      text_adapter.py   # Plain text adapter
      markdown_adapter.py  # Markdown adapter
      pdf_adapter.py    # PDF adapter (PyMuPDF)
      xml_adapter.py    # XML/JATS adapter
      html_adapter.py   # HTML adapter
      adoc_adapter.py   # AsciiDoc adapter
      tools.py          # Tool functions for all formats
      langchain_tools.py  # LangChain @tool wrappers

    prompts/
      generator.yaml    # Generator system prompt template
      validator.yaml    # Validator system prompt template
      deduplicator.yaml # Deduplicator prompt template

  tests/
    unit/               # Unit tests (mocked LLM calls)
    integration/        # Integration tests (real LLM calls)
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed component design and LangGraph integration patterns.

## License

MIT License. See [LICENSE](LICENSE) for details.
