"""SQLite-based persistence for corpus processing state.

Provides crash-safe state management for long-running corpus batch processing jobs.
Each Q/A pair is committed atomically after validation passes, so a crash loses
at most the current question's work.

The JSON output file is generated from the database on demand, separating
durability concerns from output format.
"""

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from single_doc_generator.models import (
    CorpusGenerationResult,
    GenerationMode,
    QAPair,
    RejectionReason,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

SCHEMA_V2 = """
-- Processing run configuration (one per corpus processing job)
CREATE TABLE IF NOT EXISTS processing_runs (
    id INTEGER PRIMARY KEY,
    corpus_name TEXT NOT NULL,
    corpus_path TEXT NOT NULL,
    scenario TEXT NOT NULL,
    mode TEXT NOT NULL,
    target_per_document INTEGER NOT NULL,
    total_target INTEGER,
    max_documents INTEGER,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    schema_version INTEGER NOT NULL DEFAULT 2
);

-- Individual accepted Q/A pairs (committed atomically after validation passes)
CREATE TABLE IF NOT EXISTS accepted_questions (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES processing_runs(id),
    document_path TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    mode TEXT NOT NULL,
    content_refs TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

-- Index for fast resume queries (count questions per document)
CREATE INDEX IF NOT EXISTS idx_accepted_questions_run_doc
ON accepted_questions(run_id, document_path);

-- Individual rejected Q/A pairs (for statistics)
CREATE TABLE IF NOT EXISTS rejected_questions (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES processing_runs(id),
    document_path TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    rejection_reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Index for stats queries
CREATE INDEX IF NOT EXISTS idx_rejected_questions_run_doc
ON rejected_questions(run_id, document_path);

-- Documents that exhausted (hit max consecutive failures)
CREATE TABLE IF NOT EXISTS exhausted_documents (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES processing_runs(id),
    document_path TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, document_path)
);
"""


class ProcessingStateDB:
    """SQLite-backed state storage for corpus processing.

    Handles database initialization, run management, and result persistence.
    All writes use transactions for atomicity. Each question is persisted
    individually, enabling resume within a document.
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Created if it doesn't exist.
        """
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._conn
        if conn is None:
            return
        conn.executescript(SCHEMA_V2)
        conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def get_or_create_run(
        self,
        corpus_name: str,
        corpus_path: str,
        scenario: str,
        mode: GenerationMode,
        target_per_document: int,
        total_target: int | None,
        max_documents: int | None,
    ) -> int:
        """Get existing run or create a new one.

        Finds a matching incomplete run (same corpus/scenario/mode) to resume,
        or creates a new run if none exists.

        Args:
            corpus_name: Human-readable name for the corpus.
            corpus_path: Filesystem path to the corpus directory.
            scenario: Name of the generation scenario to use.
            mode: Generation mode (text or visual).
            target_per_document: Target number of Q&A pairs per document.
            total_target: Optional cap on total Q&A pairs across all documents.
            max_documents: Optional limit on number of documents to process.

        Returns:
            Run ID for the processing job.
        """
        conn = self._get_connection()

        # Look for existing incomplete run with same parameters
        cursor = conn.execute(
            """
            SELECT id FROM processing_runs
            WHERE corpus_path = ? AND scenario = ? AND mode = ?
            AND completed_at IS NULL
            ORDER BY started_at DESC LIMIT 1
            """,
            (corpus_path, scenario, mode.value),
        )
        row = cursor.fetchone()
        if row:
            run_id: int = row["id"]
            logger.info("Resuming existing run %d", run_id)
            return run_id

        # Create new run
        with self.transaction() as txn:
            cursor = txn.execute(
                """
                INSERT INTO processing_runs
                (corpus_name, corpus_path, scenario, mode, target_per_document,
                 total_target, max_documents, started_at, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    corpus_name,
                    corpus_path,
                    scenario,
                    mode.value,
                    target_per_document,
                    total_target,
                    max_documents,
                    datetime.now(UTC).isoformat(),
                    SCHEMA_VERSION,
                ),
            )
            new_run_id = cursor.lastrowid
            if new_run_id is None:
                raise RuntimeError("Failed to create processing run")
            logger.info("Created new run %d", new_run_id)
            return new_run_id

    def get_document_question_counts(self, run_id: int) -> dict[str, int]:
        """Get count of accepted questions per document.

        Args:
            run_id: Processing run ID.

        Returns:
            Dictionary mapping document paths to their accepted question counts.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT document_path, COUNT(*) as count
            FROM accepted_questions
            WHERE run_id = ?
            GROUP BY document_path
            """,
            (run_id,),
        )
        return {row["document_path"]: row["count"] for row in cursor.fetchall()}

    def get_exhausted_documents(self, run_id: int) -> set[str]:
        """Get documents that hit exhaustion (max consecutive failures).

        Args:
            run_id: Processing run ID.

        Returns:
            Set of document paths that are exhausted.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT document_path FROM exhausted_documents WHERE run_id = ?",
            (run_id,),
        )
        return {row["document_path"] for row in cursor.fetchall()}

    def get_existing_questions(self, run_id: int, document_path: str) -> list[QAPair]:
        """Get already-accepted questions for a document (for deduplication).

        Args:
            run_id: Processing run ID.
            document_path: Relative path to document.

        Returns:
            List of QAPair objects already accepted for this document.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT question, answer, mode, content_refs
            FROM accepted_questions
            WHERE run_id = ? AND document_path = ?
            ORDER BY id
            """,
            (run_id, document_path),
        )
        return [
            QAPair(
                question=row["question"],
                answer=row["answer"],
                source_document=document_path,
                mode=GenerationMode(row["mode"]),
                content_refs=json.loads(row["content_refs"]),
            )
            for row in cursor.fetchall()
        ]

    def get_run_stats(self, run_id: int) -> dict[str, int]:
        """Get aggregated statistics for a run.

        Returns:
            Dictionary with documents_processed, documents_exhausted,
            total_accepted, total_rejected.
        """
        conn = self._get_connection()

        # Get target_per_document from run config
        cursor = conn.execute(
            "SELECT target_per_document FROM processing_runs WHERE id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        target = row["target_per_document"] if row else 5

        # Count documents that are "done": reached target OR exhausted
        # Uses UNION to deduplicate (a doc could theoretically be both)
        cursor = conn.execute(
            """
            SELECT COUNT(*) as count FROM (
                SELECT document_path FROM accepted_questions
                WHERE run_id = ?
                GROUP BY document_path
                HAVING COUNT(*) >= ?
                UNION
                SELECT document_path FROM exhausted_documents WHERE run_id = ?
            )
            """,
            (run_id, target, run_id),
        )
        docs_processed = cursor.fetchone()["count"] or 0

        # Count exhausted documents
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM exhausted_documents WHERE run_id = ?",
            (run_id,),
        )
        docs_exhausted = cursor.fetchone()["count"] or 0

        # Count accepted questions
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM accepted_questions WHERE run_id = ?",
            (run_id,),
        )
        total_accepted = cursor.fetchone()["count"] or 0

        # Count rejected questions
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM rejected_questions WHERE run_id = ?",
            (run_id,),
        )
        total_rejected = cursor.fetchone()["count"] or 0

        return {
            "documents_processed": docs_processed,
            "documents_exhausted": docs_exhausted,
            "total_accepted": total_accepted,
            "total_rejected": total_rejected,
        }

    def save_accepted_question(
        self,
        run_id: int,
        document_path: str,
        qa: QAPair,
    ) -> None:
        """Save an accepted Q/A pair atomically.

        Called immediately after validation passes. This is the atomic unit
        of work - if interrupted, we lose at most this one question.

        Args:
            run_id: Processing run ID.
            document_path: Relative path to document.
            qa: The accepted question-answer pair.
        """
        with self.transaction() as txn:
            txn.execute(
                """
                INSERT INTO accepted_questions
                (run_id, document_path, question, answer, mode,
                 content_refs, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    document_path,
                    qa.question,
                    qa.answer,
                    qa.mode.value,
                    json.dumps(qa.content_refs),
                    datetime.now(UTC).isoformat(),
                ),
            )
        logger.debug("Saved accepted question for document: %s", document_path)

    def save_rejected_question(
        self,
        run_id: int,
        document_path: str,
        question: str,
        answer: str | None,
        reason: RejectionReason,
    ) -> None:
        """Save a rejected Q/A pair for statistics.

        Args:
            run_id: Processing run ID.
            document_path: Relative path to document.
            question: The rejected question text.
            answer: The answer text (may be None if rejected before answer).
            reason: Why the question was rejected.
        """
        with self.transaction() as txn:
            txn.execute(
                """
                INSERT INTO rejected_questions
                (run_id, document_path, question, answer, rejection_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    document_path,
                    question,
                    answer,
                    reason.value,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def mark_document_exhausted(
        self,
        run_id: int,
        document_path: str,
        reason: str,
    ) -> None:
        """Mark a document as exhausted (hit max consecutive failures).

        Args:
            run_id: Processing run ID.
            document_path: Relative path to document.
            reason: Why the document was exhausted.
        """
        with self.transaction() as txn:
            txn.execute(
                """
                INSERT OR IGNORE INTO exhausted_documents
                (run_id, document_path, reason, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    document_path,
                    reason,
                    datetime.now(UTC).isoformat(),
                ),
            )
        logger.info("Document exhausted: %s (%s)", document_path, reason)

    def mark_run_complete(self, run_id: int) -> None:
        """Mark a processing run as complete."""
        with self.transaction() as txn:
            txn.execute(
                "UPDATE processing_runs SET completed_at = ? WHERE id = ?",
                (datetime.now(UTC).isoformat(), run_id),
            )

    def export_to_corpus_result(self, run_id: int) -> CorpusGenerationResult:
        """Export a run's data as a CorpusGenerationResult for RAGAS evaluation.

        Assembles accepted questions into the clean output format containing only
        corpus metadata and Q/A pairs.

        Args:
            run_id: Processing run ID.

        Returns:
            CorpusGenerationResult ready for JSON serialization.
        """
        conn = self._get_connection()

        # Get run metadata
        cursor = conn.execute("SELECT * FROM processing_runs WHERE id = ?", (run_id,))
        run_row = cursor.fetchone()
        if not run_row:
            raise ValueError(f"Run {run_id} not found")

        # Get all accepted questions
        cursor = conn.execute(
            """
            SELECT document_path, question, answer, mode, content_refs
            FROM accepted_questions
            WHERE run_id = ?
            ORDER BY id
            """,
            (run_id,),
        )

        all_questions: list[QAPair] = []
        for row in cursor.fetchall():
            all_questions.append(
                QAPair(
                    question=row["question"],
                    answer=row["answer"],
                    source_document=row["document_path"],
                    mode=GenerationMode(row["mode"]),
                    content_refs=json.loads(row["content_refs"]),
                )
            )

        return CorpusGenerationResult(
            corpus_name=run_row["corpus_name"],
            corpus_path=run_row["corpus_path"],
            scenario=run_row["scenario"],
            mode=GenerationMode(run_row["mode"]),
            questions=all_questions,
            timestamp=datetime.now(UTC).isoformat(),
        )


def export_run_to_json(db: ProcessingStateDB, run_id: int, output_path: Path) -> None:
    """Export a processing run to JSON file.

    Args:
        db: Database instance.
        run_id: Processing run ID.
        output_path: Path for output JSON file.
    """
    result = db.export_to_corpus_result(run_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2)
    logger.info("Exported results to %s", output_path)
