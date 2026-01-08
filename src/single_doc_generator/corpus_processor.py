"""Corpus-level batch processing for question generation.

Processes all documents in a corpus directory using SQLite for crash-safe
state persistence. Each Q/A pair is committed atomically after validation
passes, so a crash loses at most the current question's work.

Key features:
- Atomic per-question commits (lose at most current question on crash)
- Resume within documents (continue from question N+1 if interrupted)
- Total target limit to cap questions across corpus
- Max documents limit for testing/sampling
- JSON export after each question for visibility during long runs
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from single_doc_generator.config import CorpusConfig, load_corpus_config
from single_doc_generator.models import (
    SUPPORTED_EXTENSIONS,
    CorpusGenerationResult,
    GenerationMode,
    QAPair,
    RejectedQA,
)
from single_doc_generator.orchestrator import (
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    OnPhaseCallback,
    run_generation,
)
from single_doc_generator.persistence import ProcessingStateDB

logger = logging.getLogger(__name__)

# Callback types for progress reporting
OnDocumentStartCallback = Callable[[int, int, str], None]
OnDocumentCompleteCallback = Callable[[int, int, str, int, int], None]


@dataclass
class _DocumentContext:
    """Context for processing a single document with persistence callbacks."""

    db: ProcessingStateDB
    run_id: int
    doc_path: str
    output_path: Path | None
    on_qa_accepted: Callable[[QAPair], None] | None
    on_qa_rejected: Callable[[RejectedQA], None] | None

    # Mutable counters updated by callbacks
    total_accepted: int = 0
    doc_accepted_count: int = 0
    doc_rejected_count: int = 0

    def on_accepted(self, qa: QAPair) -> None:
        """Persist accepted question and update counters."""
        self.db.save_accepted_question(self.run_id, self.doc_path, qa)
        self.total_accepted += 1
        self.doc_accepted_count += 1
        if self.output_path:
            export_results(db=self.db, run_id=self.run_id, output_path=self.output_path)
        if self.on_qa_accepted:
            self.on_qa_accepted(qa)

    def on_rejected(self, rejected: RejectedQA) -> None:
        """Persist rejected question and update counters."""
        self.db.save_rejected_question(
            self.run_id,
            self.doc_path,
            rejected.question,
            rejected.answer,
            rejected.rejection_reason,
        )
        self.doc_rejected_count += 1
        if self.on_qa_rejected:
            self.on_qa_rejected(rejected)

    def on_exhausted(self, reason: str) -> None:
        """Mark document as exhausted."""
        self.db.mark_document_exhausted(self.run_id, self.doc_path, reason)


def _process_single_document(
    ctx: _DocumentContext,
    corpus_path: Path,
    rel_doc_path: Path,
    corpus_config: CorpusConfig,
    scenario_name: str,
    mode: GenerationMode,
    target_per_document: int,
    max_consecutive_failures: int,
    existing_questions: list[QAPair],
    on_phase: OnPhaseCallback | None,
) -> None:
    """Process a single document with persistence via context callbacks."""
    abs_doc_path = corpus_path / rel_doc_path

    run_generation(
        document_path=abs_doc_path,
        corpus_path=corpus_path,
        corpus_config=corpus_config,
        scenario_name=scenario_name,
        mode=mode,
        target_count=target_per_document,
        max_consecutive_failures=max_consecutive_failures,
        existing_accepted=existing_questions,
        on_phase=on_phase,
        on_accepted=ctx.on_accepted,
        on_rejected=ctx.on_rejected,
        on_exhausted=ctx.on_exhausted,
    )


def export_results(
    db: ProcessingStateDB,
    run_id: int,
    output_path: Path,
) -> None:
    """Export Q/A results to JSON for RAGAS evaluation.

    Writes a clean output containing only what evaluation consumers need:
    corpus metadata and the generated Q/A pairs. All processing state
    (resumption, statistics) stays in the database.

    Args:
        db: Database instance.
        run_id: Current processing run ID.
        output_path: Path for output JSON file.
    """
    result = db.export_to_corpus_result(run_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2)
    logger.info("Exported %d questions to %s", len(result.questions), output_path)


def discover_documents(corpus_path: Path) -> list[Path]:
    """Recursively discover all supported documents in a corpus directory.

    Args:
        corpus_path: Path to corpus directory.

    Returns:
        Sorted list of paths relative to corpus_path for all supported documents.
    """
    documents: list[Path] = []
    for file_path in corpus_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(file_path.relative_to(corpus_path))
    documents.sort()
    return documents


def process_corpus(
    corpus_path: Path,
    scenario_name: str,
    mode: GenerationMode,
    target_per_document: int,
    db: ProcessingStateDB,
    output_path: Path | None = None,
    total_target: int | None = None,
    max_documents: int | None = None,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    on_document_start: OnDocumentStartCallback | None = None,
    on_document_complete: OnDocumentCompleteCallback | None = None,
    on_phase: OnPhaseCallback | None = None,
    on_qa_accepted: Callable[[QAPair], None] | None = None,
    on_qa_rejected: Callable[[RejectedQA], None] | None = None,
) -> int:
    """Process documents in a corpus with crash-safe state persistence.

    Uses SQLite database for state management. Each Q/A pair is committed
    atomically after validation passes, so a crash loses at most the current
    question's work. Resume is automatic and works within documents - if a
    document has 3/5 questions, processing continues from question 4.

    Processing stops when one of these conditions is met:
    - All documents processed (reached target or exhausted)
    - total_target questions accepted (if specified)
    - max_documents documents started (if specified)

    Args:
        corpus_path: Path to corpus directory containing corpus.yaml.
        scenario_name: Scenario name from corpus.yaml to use.
        mode: Generation mode (textual or visual).
        target_per_document: Questions to attempt per document.
        db: Database instance for state persistence. Caller owns the connection.
        output_path: If provided, export JSON results here after each question.
        total_target: Stop after this many total questions (None = no limit).
        max_documents: Stop after starting this many documents (None = all).
        max_consecutive_failures: Per-document consecutive failures before exhaustion.
        on_document_start: Called when starting a document.
        on_document_complete: Called when document completes.
        on_phase: Called before each phase with (phase_name, attempt, target).
        on_qa_accepted: Called when a question is accepted.
        on_qa_rejected: Called when a question is rejected.

    Returns:
        Processing run ID (can be used to export results later).

    Raises:
        FileNotFoundError: If corpus.yaml not found.
        KeyError: If scenario not found in corpus config.
    """
    corpus_path = corpus_path.resolve()
    corpus_config = load_corpus_config(corpus_path)
    corpus_config.get_scenario(scenario_name)

    all_documents = discover_documents(corpus_path)
    logger.info("Discovered %d documents in corpus", len(all_documents))

    if max_documents is not None:
        all_documents = all_documents[:max_documents]
        logger.info("Limited to first %d documents", max_documents)

    run_id = db.get_or_create_run(
        corpus_name=corpus_config.name,
        corpus_path=str(corpus_path),
        scenario=scenario_name,
        mode=mode,
        target_per_document=target_per_document,
        total_target=total_target,
        max_documents=max_documents,
    )

    # Get current state from database for resume
    doc_question_counts = db.get_document_question_counts(run_id)
    exhausted_docs = db.get_exhausted_documents(run_id)
    db_stats = db.get_run_stats(run_id)
    total_accepted = db_stats["total_accepted"]

    # Determine which documents need processing:
    # - Skip if already has >= target questions
    # - Skip if in exhausted_documents table
    documents_to_process = [
        d
        for d in all_documents
        if str(d) not in exhausted_docs
        and doc_question_counts.get(str(d), 0) < target_per_document
    ]

    docs_already_done = len(all_documents) - len(documents_to_process)
    if docs_already_done > 0:
        logger.info("Resuming: %d documents already complete", docs_already_done)
    logger.info("Processing %d documents", len(documents_to_process))

    total_docs = len(all_documents)
    current_doc_num = docs_already_done

    for rel_doc_path in documents_to_process:
        if total_target is not None and total_accepted >= total_target:
            logger.info("Stopping: reached total target of %d questions", total_target)
            break

        current_doc_num += 1
        doc_path_str = str(rel_doc_path)

        if on_document_start:
            on_document_start(current_doc_num, total_docs, doc_path_str)

        logger.info(
            "Processing document %d/%d: %s",
            current_doc_num,
            total_docs,
            rel_doc_path,
        )

        # Get existing questions for this document (for resume within document)
        existing_questions = db.get_existing_questions(run_id, doc_path_str)

        # Create context with persistence callbacks
        ctx = _DocumentContext(
            db=db,
            run_id=run_id,
            doc_path=doc_path_str,
            output_path=output_path,
            on_qa_accepted=on_qa_accepted,
            on_qa_rejected=on_qa_rejected,
            total_accepted=total_accepted,
            doc_accepted_count=len(existing_questions),
        )

        try:
            _process_single_document(
                ctx=ctx,
                corpus_path=corpus_path,
                rel_doc_path=rel_doc_path,
                corpus_config=corpus_config,
                scenario_name=scenario_name,
                mode=mode,
                target_per_document=target_per_document,
                max_consecutive_failures=max_consecutive_failures,
                existing_questions=existing_questions,
                on_phase=on_phase,
            )

            # Update total from context (callbacks modify it)
            total_accepted = ctx.total_accepted

            # Document complete - notify UI with NEW questions only
            new_accepted = ctx.doc_accepted_count - len(existing_questions)
            if on_document_complete:
                on_document_complete(
                    current_doc_num,
                    total_docs,
                    doc_path_str,
                    new_accepted,
                    ctx.doc_rejected_count,
                )

        except Exception as e:
            logger.exception("Error processing document %s: %s", rel_doc_path, e)

    # Mark run complete
    db.mark_run_complete(run_id)

    return run_id


def get_corpus_result(
    db_path: Path, run_id: int | None = None
) -> CorpusGenerationResult:
    """Get corpus results from database for RAGAS evaluation.

    Utility function to retrieve results from a completed or in-progress run.

    Args:
        db_path: Path to SQLite database file.
        run_id: Specific run ID to export, or None to get most recent.

    Returns:
        CorpusGenerationResult containing corpus metadata and all accepted Q/A pairs.

    Raises:
        ValueError: If no processing runs found in database.
    """
    db = ProcessingStateDB(db_path)
    try:
        if run_id is None:
            conn = db._get_connection()
            cursor = conn.execute(
                "SELECT id FROM processing_runs ORDER BY started_at DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError("No processing runs found in database")
            run_id = row["id"]

        return db.export_to_corpus_result(run_id)
    finally:
        db.close()
