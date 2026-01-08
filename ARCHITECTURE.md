# Single-Document Question Generator Architecture

This document describes the high-level architecture for the single-document question generator. It assumes familiarity with the problem statement and constraints described in DESIGN.md.

---

## Component Inventory

The system comprises seven distinct components:

### 1. Document Toolkit

Provides the agent with tools to explore document content without loading the entire document into context at once. Handles format diversity (PDF, markdown, XML/JATS, plain text) behind a unified interface.

### 2. Question Generator Agent

A LangGraph agent that explores documents using the toolkit and produces candidate Q/A pairs. Uses corpus metadata to adapt its questioning style to the domain.

### 3. Validator

A separate LLM that attempts to answer generated questions using only the document. Determines whether questions are answerable with correct ground truth.

### 4. Deduplicator

Compares new questions against previously accepted questions to detect semantic duplicates. Prevents the same question asked different ways.

### 5. Orchestrator

The LangGraph graph that coordinates the generate-dedupe-validate loop. Manages state, routes between components, and handles termination conditions.

### 6. Corpus Adapter

Loads corpus metadata and constructs the meta-prompt that adapts the generator to the domain. Provides question style guidance without per-corpus prompt engineering.

### 7. Output Collector

Accumulates accepted and rejected Q/A pairs, tracks statistics, and produces the final output structure.

---

## Component Relationships

```
                                    ┌─────────────────┐
                                    │  Corpus Adapter │
                                    │  (metadata →    │
                                    │   meta-prompt)  │
                                    └────────┬────────┘
                                             │
                                             ▼
┌──────────────┐     tools      ┌─────────────────────────┐
│   Document   │◄──────────────►│  Question Generator     │
│   Toolkit    │                │  Agent                  │
│              │                │  (explores document,    │
│  - read      │                │   produces Q/A)         │
│  - search    │                └────────────┬────────────┘
│  - view_page │                             │
│  - list_visual│                             │ candidate Q/A
└──────────────┘                             ▼
       ▲                        ┌─────────────────────────┐
       │                        │     Deduplicator        │
       │                        │  (cheap/fast model)     │
       │                        │                         │
       │                        │  compares against       │
       │                        │  accepted questions     │
       │                        └────────────┬────────────┘
       │                                     │
       │                                     │ unique/duplicate
       │                                     ▼
       │                        ┌─────────────────────────┐
       │         tools          │      Validator          │
       └────────────────────────│  (different model)      │
                                │                         │
                                │  attempts to answer     │
                                │  from document only     │
                                └────────────┬────────────┘
                                             │
                                             │ pass/fail
                                             ▼
                                ┌─────────────────────────┐
                                │   Output Collector      │
                                │                         │
                                │  - accepted Q/A pairs   │
                                │  - rejected Q/A pairs   │
                                │  - statistics           │
                                └─────────────────────────┘
```

All components are coordinated by the Orchestrator (not shown - it's the graph that contains these as nodes or calls them from nodes).

---

## The Agent Loop

The central loop is generate → dedupe → validate, with termination based on success count or exhaustion.

Ordering rationale: Deduplication is cheap (semantic comparison of question text using a fast model, no tool calls needed). Validation is expensive (a capable model using document tools to attempt answering). We filter duplicates first to avoid wasting expensive validation on questions we'd discard anyway.

### Flow

1. **Generate**: The generator agent uses document tools to explore content and proposes one Q/A candidate. The agent receives:
   - Corpus context and evaluation scenario (shapes questioning style)
   - List of previously accepted questions (to avoid generating duplicates)
   - List of previously viewed pages (visual mode only - to avoid revisiting trivial content)

   The generator's prompt instructs it to create questions different from those already asked. This in-prompt feedback prevents most duplicate generation, reducing wasted API calls.

   In visual mode, the prompt also includes guidance to skip trivial visual content (blank pages, library stamps, foxing damage, watermarks, etc.) and focus on content relevant to the corpus context.

   The generator has two terminal tools:
   - `submit_qa`: Submit a new question/answer pair
   - `report_exhausted`: Signal that no more unique questions can be generated

   The generator returns both the Q/A pair and a list of page numbers viewed during generation. The orchestrator accumulates these across iterations so subsequent attempts can avoid revisiting pages that yielded no useful content.

   If the generator calls `report_exhausted`, the function raises `DocumentExhaustedError` with the reason provided. The orchestrator catches this to terminate generation for the document.

2. **Dedupe**: The candidate goes to deduplication. The deduplicator uses a cheap/fast LLM to compare the new question against all previously accepted questions for semantic equivalence - detecting when two differently-worded questions ask the same thing despite the generator's instructions. This catches near-duplicates that slip through the in-prompt guidance.

3. **Validate**: If the question is unique, it passes to validation - a different model that receives the question and has access to the same document tools. It attempts to answer the question using only document content. The validator also receives corpus context and scenario description to check relevance. Validation checks:
   - Can the question be answered from the document?
   - Does the validator's answer match the generator's ground truth?
   - Is the question non-trivial (requires reading the document)?
   - Is the question unambiguous (one clear answer)?
   - Is the question relevant to the corpus context? (visual mode: rejects questions about trivial content like blank pages, library stamps, foxing damage)

4. **Accept or Reject**:
   - If validation passes: add to accepted set, reset failure counter
   - If duplicate or validation failed: add to rejected set (with reason), increment failure counter

5. **Loop or Exit**:
   - If accepted count reaches target: exit with success
   - If consecutive failures exceed threshold: exit with exhaustion
   - Otherwise: loop back to generate

### State Transitions

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
              ┌──────────┐                                    │
      ┌──────►│ GENERATE │                                    │
      │       └────┬─────┘                                    │
      │            │                                          │
      │            ▼                                          │
      │       ┌──────────┐   duplicate   ┌─────────┐         │
      │       │ DEDUPE   │──────────────►│ REJECT  │─────────┤
      │       └────┬─────┘               └─────────┘         │
      │            │                                          │
      │            │ unique                                   │
      │            ▼                                          │
      │       ┌──────────┐     fail      ┌─────────┐         │
      │       │ VALIDATE │──────────────►│ REJECT  │─────────┤
      │       └────┬─────┘               └─────────┘         │
      │            │                                          │
      │            │ pass                                     │
      │            ▼                                          │
      │       ┌──────────┐                                    │
      │       │ ACCEPT   │                                    │
      │       └────┬─────┘                                    │
      │            │                                          │
      │            ▼                                          │
      │     [check exit conditions]                           │
      │            │                                          │
      │            │ continue                                 │
      └────────────┴──────────────────────────────────────────┘
                   │
                   │ target reached OR exhausted
                   ▼
              ┌──────────┐
              │   END    │
              └──────────┘
```

### Exhaustion Detection

The system tracks consecutive failures (validation fail + dedupe fail). When this count exceeds a threshold (configurable, suggested: 5-10), the document is considered exhausted - it cannot yield more good questions.

This prevents infinite loops on documents with limited content while allowing retry on legitimately difficult documents. The threshold should be tuned based on observed behavior across corpus types.

---

## Document Tool Layer

The agent needs to explore documents without loading everything into context. The toolkit provides four tools:

### read_lines

Reads a range of lines from the document with line numbers for reference. Works like viewing a file in an editor - the agent can navigate by requesting specific ranges.

**Input**: start_line, end_line (optional: returns to end if omitted)
**Output**: Text content with line numbers, total line count

For all formats, the document is first converted to a line-based text representation. PDFs become extracted text, XML/JATS is parsed to readable content, etc.

### search

Searches for literal strings or regex patterns in the document. Returns matching lines with context and line numbers.

**Input**: pattern (string or regex), context_lines (how many lines before/after to include)
**Output**: List of matches with line numbers and surrounding context

This is the agent's discovery mechanism - it can search for terms, names, concepts without reading the entire document.

### view_page

Renders a page of the document as an image and returns it as base64. Only meaningful for formats with visual layout (PDF, scanned documents). For text-only formats, returns an indicator that visual rendering is not applicable.

**Input**: page_number
**Output**: Base64-encoded image OR not_applicable indicator OR error message

This tool enables visual mode - questions about tables, figures, diagrams. The agent sees the rendered page as a vision model would.

**Error Handling**: If a page cannot be rendered (too large for API limits, corrupted, etc.), the tool returns an error message instructing the agent to try a different page. This prevents the agent from getting stuck retrying the same unrenderable page. The agent's prompt also includes guidance to try different page numbers when encountering rendering errors.

### list_visual_content

Discovers what visual content exists in the document - figures, tables, images, diagrams. Returns a manifest of visual elements with page numbers and descriptions where available.

**Input**: None
**Output**: List of visual elements with type, page/location, and any caption/label text

This helps the agent in visual mode identify what visual content to examine. For text-only documents, returns an empty list.

### Format Abstraction

The toolkit includes a format adapter that handles conversion:

| Format | read_lines | search | view_page | list_visual_content |
|--------|-----------|--------|-----------|---------------------|
| Plain text | Direct | Direct | Not applicable | Empty |
| Markdown | Direct | Direct | Not applicable | Parse image refs |
| PDF | Extract text | Search extracted | Render page | Extract from structure |
| XML (JATS) | Parse to text | Search content | Not applicable | Parse figure elements |

The adapter is configured per-format, not per-corpus. New formats require implementing the adapter interface.

### Tool Return Handling

For text tools (read_lines, search, list_visual_content), returns are standard text content.

For view_page, the tool returns a `Command` that updates agent state with pending image data. The custom agent graph (see `agent.py`) includes an image injector node that runs after all tool responses complete. This node collects any pending images and injects them as a single `HumanMessage` with multimodal content before the agent continues.

This deferred injection pattern is necessary because OpenAI's API only allows images in user messages (not tool messages) and requires all tool responses to immediately follow the assistant message with tool_calls. By deferring image injection until after all tools complete, we avoid violating message ordering constraints when multiple view_page calls execute in parallel.

The generator model must be vision-capable for visual mode.

---

## LangGraph Integration

The system maps to LangGraph concepts as follows:

### State Definition

```
State:
  document_path: str                    # Path to document being processed
  corpus_config: CorpusConfig           # Name, context, and scenarios
  scenario_name: str                    # Which scenario to use for generation
  mode: "textual" | "visual"            # Question generation mode
  target_count: int                     # How many questions to generate

  accepted_questions: list[QAPair]      # Successfully validated, unique questions
  rejected_questions: list[RejectedQA]  # Failed validation or duplicate

  current_candidate: QAPair | None      # Question currently being evaluated
  consecutive_failures: int             # Tracks exhaustion

  previous_viewed_pages: list[int]      # Pages viewed across all generation attempts (visual mode)

  messages: list[Message]               # Conversation history for generator agent
```

The messages field uses LangGraph's add_messages reducer - each node appends messages rather than replacing the list.

The `previous_viewed_pages` field accumulates page numbers viewed during visual mode generation. After each generation attempt, the pages viewed are extracted from the agent's tool calls and added to this list. This prevents the generator from repeatedly viewing the same trivial pages (blank pages, library stamps, etc.) across multiple generation attempts.

### Nodes

**generate_node**: Calls the generator model with document tools. The model explores the document and produces a Q/A candidate. Uses the ReAct pattern - model can make multiple tool calls before producing output. Returns both the candidate and viewed pages in state. The viewed pages are extracted from the agent's `view_page` tool calls and accumulated across iterations.

**validate_node**: Calls the validator model with the candidate question and document tools. Also receives corpus context and scenario description for relevance checking. Validator attempts to answer, compares to ground truth. Returns pass/fail decision.

**dedupe_node**: Calls the deduplicator model with the candidate question and list of accepted questions. Returns unique/duplicate decision.

**accept_node**: Moves candidate to accepted list, resets failure counter.

**reject_node**: Moves candidate to rejected list (with reason), increments failure counter.

**check_exit_node**: Evaluates termination conditions. Returns routing decision.

### Edges and Routing

The graph uses conditional edges for the deduplication and validation decisions:

```
START → generate_node

generate_node → dedupe_node

dedupe_node → [conditional]
  - if unique: validate_node
  - if duplicate: reject_node

validate_node → [conditional]
  - if pass: accept_node
  - if fail: reject_node

accept_node → check_exit_node
reject_node → check_exit_node

check_exit_node → [conditional]
  - if target_reached OR exhausted: END
  - otherwise: generate_node
```

### Tool Integration

The generator and validator use a custom agent graph (`create_agent` in `agent.py`) rather than LangGraph's built-in `create_react_agent`. This custom graph handles the complexity of vision tool responses while maintaining the standard ReAct pattern.

Tools are defined with LangChain's @tool decorator and bound to the model via bind_tools(). The agent graph implements:

1. **Agent node**: Calls the model with messages, prepending system prompt
2. **Tools node**: Executes tool calls via LangGraph's ToolNode
3. **Image injector node**: After tools complete, if any view_page calls added pending images, injects them as a HumanMessage before returning to the agent
4. **Routing**: Agent continues calling tools until no tool_calls in response

The validator follows the same pattern - it receives the question as input and uses document tools to attempt answering.

### State Updates

Most nodes return simple state dicts that update specific fields. The Command pattern is not needed here since routing is based on discrete decisions (pass/fail, unique/duplicate) that map cleanly to conditional edges.

---

## Model Configuration

Three distinct roles require potentially different models:

### Generator Model

Used by: generate_node
Requirements:
- Strong reasoning and question formulation
- Tool use capability
- Vision capability (for visual mode)
- Domain knowledge for adapting question style

Recommendation: Capable frontier model (GPT-5.1, Claude Opus, etc.)

### Validator Model

Used by: validate_node
Requirements:
- Tool use capability
- Vision capability (for visual mode)
- Different from generator (prevents self-confirmation)
- Reading comprehension focus

Recommendation: Different model family from generator. Could be smaller/cheaper since task is more constrained.

### Deduplicator Model

Used by: dedupe_node
Requirements:
- Semantic comparison capability
- Fast (called on every candidate)
- No tool use needed
- No vision needed

Recommendation: Fast, cheap model (GPT-5.1-mini, Claude Haiku, etc.). The task is simple comparison.

### Configuration Structure

Models are configured at the orchestrator level, not per-component. Configuration includes:

- model identifier (provider + model name)
- temperature (typically 0 for validation/deduplication, higher for generation)
- any provider-specific settings

The orchestrator passes model configuration to each node. Nodes construct their own model clients.

### Enforcement

The validator model MUST differ from the generator model. This is a hard constraint - the configuration should validate this at startup and fail if both point to the same model.

---

## Corpus Configuration

Each corpus provides configuration that shapes question generation without requiring per-corpus prompt engineering. Configuration includes corpus context and named evaluation scenarios.

### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| name | Yes | Human-readable corpus name |
| corpus_context | Yes | Paragraph describing what the corpus is, where it came from, and what it contains |
| scenarios | Yes | Dictionary of named evaluation scenarios |

Each scenario contains:

| Field | Required | Description |
|-------|----------|-------------|
| name | Yes | Short display name for the scenario |
| description | Yes | Full description of the evaluation context, intended audience, and question style |

### Meta-Prompt Construction

The Corpus Adapter transforms configuration into system prompt content. The meta-prompt tells the generator:

1. What kind of documents it's working with (corpus context)
2. The evaluation context and expected question style (scenario description)

Example transformation:

```yaml
# corpus.yaml
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

The generator model's pretraining includes many examples of domain-specific questions (bar exams, medical boards, trivia, etc.). By describing the corpus and evaluation context, we leverage this existing capability rather than engineering custom prompts.

### Configuration File Format

Corpus configuration is a YAML file (`corpus.yaml`) alongside the corpus data:

```yaml
name: "Kobold Press Monster Collection"

corpus_context: >
  1,527 creatures from Kobold Press supplements: Tome of Beasts (tob),
  Tome of Beasts 2 (tob2), Tome of Beasts 3 (tob3), and Creature Codex (cc).
  Full stat blocks with size, type, alignment, AC, HP, abilities, actions,
  and special abilities. Markdown format. OGL license.

scenarios:
  rag_eval:
    name: "RAG System Evaluation"
    description: >
      Specific factual questions with exact answers from stat blocks.
      "What is the A-mi-kuk's Challenge Rating?" (7), "How much fire damage
      does touching the Abominable Beauty deal?" (28/8d6). Verifies accurate
      extraction from structured stat block data.

  encounter_design:
    name: "Encounter Design Reference"
    description: >
      DM seeking monster abilities during encounter preparation. Questions
      like "What happens when you touch the Abominable Beauty?" or "How does
      the A-mi-kuk's Strangle action work?" Tests retrieval of unique
      mechanics not found in standard monster manuals.
```

The system loads this at runtime. Adding a new corpus requires only adding this configuration file. The scenario is specified at generation time, allowing different question types from the same corpus.

---

## Outputs

The system produces three categories of output:

### Accepted Questions

The primary output - validated, unique Q/A pairs ready for evaluation.

```
QAPair:
  question: str                      # The question text
  answer: str                        # Ground truth answer
  source_document: str               # Path to source document
  category: "textual" | "visual"     # Generation mode
  content_refs: list[str]            # For visual: specific figures/tables referenced
  generation_metadata:
    generator_model: str             # Model that generated
    validator_model: str             # Model that validated
    attempt_number: int              # How many attempts before acceptance
```

### Rejected Questions

Questions that failed validation or were duplicates. Useful for debugging and understanding generator behavior.

```
RejectedQA:
  question: str
  answer: str
  rejection_reason: "validation_failed" | "duplicate" | "unanswerable" | "wrong_answer" | "ambiguous" | "trivial"
  rejection_detail: str              # Specific explanation
  duplicate_of: str | None           # If duplicate, which accepted question
```

### Statistics

Aggregate information about the generation run:

```
GenerationStats:
  document_path: str
  mode: str
  target_count: int
  accepted_count: int
  rejected_count: int
  total_attempts: int
  validation_pass_rate: float
  dedup_rejection_rate: float
  exhausted: bool                    # Whether document was exhausted
  rejection_reasons: dict[str, int]  # Count by reason
```

### Output Format

All outputs are structured data (Pydantic models) that can be serialized to JSON. The orchestrator returns:

```
GenerationResult:
  accepted: list[QAPair]
  rejected: list[RejectedQA]
  stats: GenerationStats
```

For corpus-level processing (running generator across all documents in a corpus), the JSON output contains only RAGAS-relevant data:

```
CorpusGenerationResult (JSON output):
  corpus_name: str
  corpus_path: str
  scenario: str
  mode: GenerationMode
  questions: list[QAPair]            # All accepted from all documents
  timestamp: str
```

Processing statistics (documents processed, exhausted, accepted/rejected counts, per-document results) are stored in the SQLite database only and not included in the JSON output.

---

## Appendix: Textual vs Visual Mode

The mode parameter changes generator behavior:

### Textual Mode

- Uses read_lines and search tools
- view_page returns "not applicable"
- Questions focus on textual factual content
- Works for all document types
- Does not require vision-capable model

### Visual Mode

- Uses all tools including view_page and list_visual_content
- First checks if visual content exists (list_visual_content)
- If no visual content: reports this and exits (not an error)
- If visual content exists: generates questions requiring that content
- Requires vision-capable generator and validator models
- Tracks viewed pages across generation attempts to avoid revisiting trivial content
- Prompt includes guidance to skip trivial visual content:
  - Blank or near-blank pages
  - Library stamps, catalog numbers, ownership marks
  - Foxing, staining, or other damage artifacts
  - Generic decorative elements
  - Watermarks or institutional logos
- Both generator and validator check that questions are relevant to the corpus context

The mode is set per-generation run, not per-corpus. A corpus might be processed twice - once in textual mode, once in visual mode - to generate both question types.

---

## References

LangGraph documentation and patterns referenced in this design:

- [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [ReAct Agent from Scratch](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/)
- [Reflection Pattern](https://langchain-ai.github.io/langgraph/tutorials/reflection/reflection/)
- [LangGraph-Reflection Library](https://github.com/langchain-ai/langgraph-reflection)
- [Command Pattern for Control Flow](https://langchain-ai.github.io/langgraphjs/how-tos/command/)
- [LangChain Messages Reference](https://reference.langchain.com/python/langchain/messages/)
- [OpenAI Vision API](https://platform.openai.com/docs/guides/images-vision)
