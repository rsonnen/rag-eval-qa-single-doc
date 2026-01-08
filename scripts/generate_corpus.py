#!/usr/bin/env python3
"""Generate questions from all documents in a corpus.

Batch processes an entire corpus directory, generating questions for each
document. Uses SQLite for crash-safe state persistence - each document's
result is committed atomically, so a crash loses at most the current
document's work. Resume is automatic.

Usage:
    # Process all documents, 5 questions each
    uv run python scripts/generate_corpus.py \
        --corpus /path/to/corpus \
        --scenario rag_eval

    # Limit to 100 total questions across corpus
    uv run python scripts/generate_corpus.py \
        --corpus /path/to/corpus \
        --scenario rag_eval \
        --total-target 100

    # Process only first 10 documents (for testing)
    uv run python scripts/generate_corpus.py \
        --corpus /path/to/corpus \
        --scenario rag_eval \
        --max-docs 10

    # Custom output path
    uv run python scripts/generate_corpus.py \
        --corpus /path/to/corpus \
        --scenario rag_eval \
        --output custom_results.json

    # Resume is automatic - just re-run the same command
    # Completed documents are skipped based on database state
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from single_doc_generator.corpus_processor import (
    discover_documents,
    process_corpus,
)
from single_doc_generator.models import GenerationMode, QAPair, RejectedQA
from single_doc_generator.orchestrator import DEFAULT_MAX_CONSECUTIVE_FAILURES
from single_doc_generator.persistence import ProcessingStateDB

load_dotenv()


def get_default_db_path(corpus_path: Path) -> Path:
    """Compute default database path for a corpus.

    Uses a hash of the absolute corpus path to create a unique database file
    in the local cache directory. This avoids SQLite locking issues on network
    filesystems while ensuring resume works (same corpus = same database).

    Args:
        corpus_path: Path to the corpus directory.

    Returns:
        Path to the SQLite database file in ~/.cache/single_doc_generator/.
    """
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    app_cache = cache_dir / "single_doc_generator"
    app_cache.mkdir(parents=True, exist_ok=True)

    # Hash the absolute path to get a deterministic, collision-resistant filename
    corpus_hash = hashlib.sha256(str(corpus_path.resolve()).encode()).hexdigest()[:16]
    return app_cache / f"{corpus_hash}.db"


class RichProgressTracker:
    """Tracks progress and displays status using rich library."""

    def __init__(
        self,
        console: Console,
        total_docs: int,
        total_target: int | None,
        verbose: bool,
    ) -> None:
        """Initialize the progress tracker.

        Args:
            console: Rich console for output.
            total_docs: Total number of documents to process.
            total_target: Optional total question target.
            verbose: Whether to show individual question details.
        """
        self.console = console
        self.total_docs = total_docs
        self.total_target = total_target
        self.verbose = verbose
        self.questions_accepted = 0
        self.questions_rejected = 0
        self.docs_processed = 0
        self.current_document = ""
        self.current_phase = ""

        # Create progress bar with custom columns
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[phase]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[green]{task.fields[questions]} questions"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=4,
        )

        # Add the main task
        self.task_id = self.progress.add_task(
            "Processing",
            total=total_docs,
            phase="Starting...",
            questions=0,
        )

    def set_initial_stats(self, accepted: int, rejected: int, processed: int) -> None:
        """Set initial stats from database for resumed runs.

        Args:
            accepted: Number of previously accepted questions.
            rejected: Number of previously rejected questions.
            processed: Number of previously processed documents.
        """
        self.questions_accepted = accepted
        self.questions_rejected = rejected
        self.docs_processed = processed
        self.progress.update(
            self.task_id,
            completed=processed,
            questions=accepted,
        )

    def on_document_start(
        self,
        current: int,  # noqa: ARG002 - required by callback signature
        total: int,  # noqa: ARG002 - required by callback signature
        rel_path: str,
    ) -> None:
        """Called when starting a document.

        Args:
            current: Current document number.
            total: Total documents.
            rel_path: Relative path to document.
        """
        self.current_document = rel_path
        # Truncate long paths
        display_path = rel_path if len(rel_path) <= 40 else "..." + rel_path[-37:]
        self.progress.update(
            self.task_id,
            phase=f"[cyan]{display_path}[/cyan]",
        )

    def on_document_complete(
        self,
        current: int,
        total: int,  # noqa: ARG002 - required by callback signature
        rel_path: str,  # noqa: ARG002 - required by callback signature
        accepted: int,  # noqa: ARG002 - tracked via on_qa_accepted instead
        rejected: int,
    ) -> None:
        """Called when document completes.

        Args:
            current: Current document number.
            total: Total documents.
            rel_path: Relative path to document.
            accepted: Number of NEW questions accepted (tracked via on_qa_accepted).
            rejected: Number of questions rejected from this document.
        """
        self.docs_processed = current
        # Note: questions_accepted already updated by on_qa_accepted callbacks
        self.questions_rejected += rejected
        self.progress.update(
            self.task_id,
            completed=current,
        )

    def on_phase(self, phase: str, attempt: int, target: int) -> None:
        """Called when entering a new processing phase.

        Args:
            phase: Name of the phase (generating, deduplicating, validating).
            attempt: Current attempt number.
            target: Target questions per document.
        """
        self.current_phase = phase
        phase_display = {
            "generating": "[yellow]Generating",
            "deduplicating": "[blue]Deduplicating",
            "validating": "[magenta]Validating",
        }.get(phase, phase)

        # Truncate document path for display
        doc = self.current_document
        if len(doc) > 25:
            doc = "..." + doc[-22:]

        self.progress.update(
            self.task_id,
            phase=f"{phase_display}[/] [{attempt}/{target}] {doc}",
        )

    def on_qa_accepted(self, qa: QAPair) -> None:
        """Called when a question is accepted.

        Args:
            qa: The accepted question-answer pair.
        """
        self.questions_accepted += 1
        self.progress.update(self.task_id, questions=self.questions_accepted)
        if self.verbose:
            q_preview = (
                qa.question[:55] + "..." if len(qa.question) > 55 else qa.question
            )
            self.progress.console.print(f"    [green]ACCEPTED:[/green] {q_preview}")

    def on_qa_rejected(self, rejected: RejectedQA) -> None:
        """Called when a question is rejected.

        Args:
            rejected: The rejected question with reason.
        """
        if self.verbose:
            q_preview = (
                rejected.question[:45] + "..."
                if len(rejected.question) > 45
                else rejected.question
            )
            self.progress.console.print(
                f"    [red]REJECTED ({rejected.rejection_reason.value}):[/red] "
                f"{q_preview}"
            )

    def start(self) -> None:
        """Start the progress display."""
        self.progress.start()

    def stop(self) -> None:
        """Stop the progress display."""
        self.progress.stop()

    def print_summary(self, docs_processed: int, docs_exhausted: int) -> None:
        """Print final summary.

        Args:
            docs_processed: Total documents processed.
            docs_exhausted: Documents that ran out of unique questions.
        """
        elapsed = self.progress.tasks[0].elapsed or 0.0

        if docs_processed > 0:
            rate = elapsed / docs_processed
            rate_str = f"{rate:.1f}s/doc"
        else:
            rate_str = "N/A"

        hours, remainder = divmod(int(elapsed), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            elapsed_str = f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            elapsed_str = f"{minutes:02d}:{secs:02d}"

        self.console.print()
        self.console.rule("[bold green]COMPLETE")
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Documents processed", str(docs_processed))
        table.add_row("Documents exhausted", str(docs_exhausted))
        table.add_row("Questions accepted", str(self.questions_accepted))
        table.add_row("Questions rejected", str(self.questions_rejected))
        table.add_row("Time elapsed", elapsed_str)
        table.add_row("Average rate", rate_str)
        self.console.print(table)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate questions from all documents in a corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to corpus directory containing corpus.yaml",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario name from corpus.yaml (e.g., rag_eval)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["textual", "visual"],
        default="textual",
        help="Generation mode: textual (text only) or visual (visual content)",
    )

    parser.add_argument(
        "--target-per-doc",
        type=int,
        default=5,
        dest="target_per_doc",
        help="Questions to generate per document (default: 5)",
    )

    parser.add_argument(
        "--total-target",
        type=int,
        default=None,
        dest="total_target",
        help="Stop after this many total questions across corpus (default: unlimited)",
    )

    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        dest="max_docs",
        help="Only process first N documents (default: all)",
    )

    parser.add_argument(
        "--max-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
        dest="max_failures",
        help=(
            f"Consecutive failures per doc before exhaustion "
            f"(default: {DEFAULT_MAX_CONSECUTIVE_FAILURES})"
        ),
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=(
            "SQLite database file for state "
            "(default: ~/.cache/single_doc_generator/<hash>.db)"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Export results to JSON file "
            "(default: <corpus>/<scenario>_<mode>_questions.json)"
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show individual question accept/reject details",
    )

    return parser.parse_args()


def _get_resume_stats(
    db: ProcessingStateDB,
    corpus_path: Path,
    scenario: str,
    mode: GenerationMode,
) -> tuple[int, int, int]:
    """Get resume statistics from database for an incomplete run.

    Args:
        db: Database instance.
        corpus_path: Resolved corpus path.
        scenario: Scenario name.
        mode: Generation mode.

    Returns:
        Tuple of (already_processed, initial_accepted, initial_rejected).
    """
    conn = db._get_connection()
    cursor = conn.execute(
        """
        SELECT id FROM processing_runs
        WHERE corpus_path = ? AND scenario = ? AND mode = ?
          AND completed_at IS NULL
        ORDER BY started_at DESC LIMIT 1
        """,
        (str(corpus_path), scenario, mode.value),
    )
    row = cursor.fetchone()
    if row:
        db_stats = db.get_run_stats(row["id"])
        return (
            db_stats["documents_processed"],
            db_stats["total_accepted"],
            db_stats["total_rejected"],
        )
    return (0, 0, 0)


def _print_config_header(
    console: Console,
    args: argparse.Namespace,
    docs_to_process: int,
    already_processed: int,
    db_path: Path,
    output_path: Path,
) -> None:
    """Print configuration header table."""
    console.rule("[bold blue]CORPUS QUESTION GENERATION")
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Setting", style="bold")
    config_table.add_column("Value")
    config_table.add_row("Corpus", str(args.corpus))
    config_table.add_row("Scenario", args.scenario)
    config_table.add_row("Mode", args.mode)
    config_table.add_row("Target/doc", str(args.target_per_doc))
    if args.total_target:
        config_table.add_row("Total target", str(args.total_target))
    if args.max_docs:
        config_table.add_row("Max documents", str(args.max_docs))
    config_table.add_row(
        "Documents", f"{docs_to_process} total, {already_processed} already done"
    )
    config_table.add_row("Database", str(db_path))
    config_table.add_row("Output", str(output_path))
    console.print(config_table)
    console.print()


def main() -> int:
    """Run corpus-level question generation."""
    args = parse_args()
    console = Console()

    # Configure logging to only show errors (progress shown via tracker)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    if args.verbose:
        logging.getLogger("single_doc_generator").setLevel(logging.INFO)

    # Validate corpus exists
    if not args.corpus.exists():
        console.print(f"[red]Error:[/red] Corpus directory not found: {args.corpus}")
        return 1

    corpus_yaml = args.corpus / "corpus.yaml"
    if not corpus_yaml.exists():
        console.print(f"[red]Error:[/red] corpus.yaml not found in {args.corpus}")
        return 1

    # Default paths - database on local FS to avoid network FS locking issues
    db_path = args.db if args.db else get_default_db_path(args.corpus)
    mode = GenerationMode.VISUAL if args.mode == "visual" else GenerationMode.TEXTUAL
    output_path = (
        args.output
        if args.output
        else args.corpus / f"{args.scenario}_{args.mode}_questions.json"
    )

    # Discover documents
    all_documents = discover_documents(args.corpus)
    docs_to_process = len(all_documents)

    if args.max_docs:
        docs_to_process = min(docs_to_process, args.max_docs)

    # Single database connection for entire run
    db = ProcessingStateDB(db_path)
    try:
        # Get resume stats from database
        already_processed, initial_accepted, initial_rejected = _get_resume_stats(
            db, args.corpus.resolve(), args.scenario, mode
        )

        _print_config_header(
            console, args, docs_to_process, already_processed, db_path, output_path
        )

        # Create progress tracker with resumed stats
        tracker = RichProgressTracker(
            console, docs_to_process, args.total_target, args.verbose
        )
        tracker.set_initial_stats(initial_accepted, initial_rejected, already_processed)
        tracker.start()

        try:
            run_id = process_corpus(
                corpus_path=args.corpus,
                scenario_name=args.scenario,
                mode=mode,
                target_per_document=args.target_per_doc,
                db=db,
                output_path=output_path,
                total_target=args.total_target,
                max_documents=args.max_docs,
                max_consecutive_failures=args.max_failures,
                on_document_start=tracker.on_document_start,
                on_document_complete=tracker.on_document_complete,
                on_phase=tracker.on_phase,
                on_qa_accepted=tracker.on_qa_accepted,
                on_qa_rejected=tracker.on_qa_rejected,
            )
        except FileNotFoundError as e:
            tracker.stop()
            console.print(f"[red]Error:[/red] {e}")
            return 1
        except KeyError as e:
            tracker.stop()
            console.print(f"[red]Error:[/red] {e}")
            return 1
        except KeyboardInterrupt:
            tracker.stop()
            console.print("\n[yellow]Interrupted.[/yellow] Progress saved to database.")
            return 130

        tracker.stop()

        # Get final stats from same db connection
        stats = db.get_run_stats(run_id)
        tracker.questions_accepted = stats["total_accepted"]
        tracker.questions_rejected = stats["total_rejected"]
        tracker.print_summary(
            stats["documents_processed"], stats["documents_exhausted"]
        )

        console.print(f"[dim]Database:[/dim] {db_path}")
        console.print(f"[dim]Output:[/dim]   {output_path}")

        return 0

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
