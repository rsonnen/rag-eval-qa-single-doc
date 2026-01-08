"""Tests for corpus_processor and persistence modules.

Tests document discovery, SQLite state persistence, resume logic, and batch
processing orchestration. Uses mocks for run_generation to avoid actual LLM calls.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from single_doc_generator.config import CorpusConfig, EvaluationScenario
from single_doc_generator.corpus_processor import (
    discover_documents,
    process_corpus,
)
from single_doc_generator.models import (
    GenerationMode,
    GenerationResult,
    GenerationStats,
    QAPair,
    RejectionReason,
)
from single_doc_generator.persistence import ProcessingStateDB, export_run_to_json


@pytest.fixture
def corpus_config() -> CorpusConfig:
    """Create a test corpus config."""
    return CorpusConfig(
        name="Test Corpus",
        corpus_context="A test corpus for unit testing",
        scenarios={
            "test_scenario": EvaluationScenario(
                name="Test Scenario",
                description="A test evaluation scenario",
            )
        },
    )


@pytest.fixture
def mock_corpus_dir(tmp_path: Path) -> Path:
    """Create a mock corpus directory with various document files."""
    # Create corpus.yaml
    corpus_yaml = tmp_path / "corpus.yaml"
    corpus_yaml.write_text(
        """
name: "Test Corpus"
corpus_context: "A test corpus"
scenarios:
  test_scenario:
    name: "Test"
    description: "Test scenario"
"""
    )

    # Create some documents
    (tmp_path / "doc1.txt").write_text("Document 1 content")
    (tmp_path / "doc2.md").write_text("# Document 2")
    (tmp_path / "doc3.pdf").write_bytes(b"%PDF-1.4 fake pdf")

    # Create subdirectory with more documents
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "doc4.txt").write_text("Document 4 in subdir")
    (subdir / "doc5.xml").write_text("<doc>Document 5</doc>")

    # Create unsupported file (should be ignored)
    (tmp_path / "readme.rst").write_text("RST file - should be ignored")
    (tmp_path / "data.json").write_text("{}")

    return tmp_path


class TestDiscoverDocuments:
    """Tests for document discovery."""

    def test_discovers_all_supported_formats(self, mock_corpus_dir: Path) -> None:
        """Discovers txt, md, pdf, xml files."""
        docs = discover_documents(mock_corpus_dir)

        # Should find 5 documents
        assert len(docs) == 5

        # All should be relative paths
        for doc in docs:
            assert not doc.is_absolute()

        # Check specific files found
        doc_names = [str(d) for d in docs]
        assert "doc1.txt" in doc_names
        assert "doc2.md" in doc_names
        assert "doc3.pdf" in doc_names
        assert "subdir/doc4.txt" in doc_names
        assert "subdir/doc5.xml" in doc_names

    def test_excludes_unsupported_formats(self, mock_corpus_dir: Path) -> None:
        """Does not include unsupported file types."""
        docs = discover_documents(mock_corpus_dir)
        doc_names = [str(d) for d in docs]

        assert "readme.rst" not in doc_names
        assert "data.json" not in doc_names

    def test_returns_sorted_paths(self, mock_corpus_dir: Path) -> None:
        """Returns paths in sorted order for determinism."""
        docs = discover_documents(mock_corpus_dir)
        doc_strs = [str(d) for d in docs]

        assert doc_strs == sorted(doc_strs)

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Handles empty directory gracefully."""
        docs = discover_documents(tmp_path)
        assert docs == []

    def test_discovers_markdown_extension(self, tmp_path: Path) -> None:
        """Discovers .markdown extension as well as .md."""
        (tmp_path / "doc.markdown").write_text("# Markdown doc")
        docs = discover_documents(tmp_path)

        assert len(docs) == 1
        assert str(docs[0]) == "doc.markdown"


class TestProcessingStateDB:
    """Tests for SQLite persistence with question-level granularity."""

    def test_creates_database_and_tables(self, tmp_path: Path) -> None:
        """Creates database file and schema."""
        db_path = tmp_path / "test.db"
        db = ProcessingStateDB(db_path)

        # Force connection
        db._get_connection()

        assert db_path.exists()

        # Verify tables exist
        conn = db._get_connection()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in cursor.fetchall()}

        assert "processing_runs" in tables
        assert "accepted_questions" in tables
        assert "rejected_questions" in tables
        assert "exhausted_documents" in tables

        db.close()

    def test_creates_new_run(self, tmp_path: Path) -> None:
        """Creates a new processing run."""
        db_path = tmp_path / "test.db"
        db = ProcessingStateDB(db_path)

        run_id = db.get_or_create_run(
            corpus_name="Test",
            corpus_path="/corpus",
            scenario="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_per_document=5,
            total_target=100,
            max_documents=None,
        )

        assert run_id == 1

        # Verify run exists
        conn = db._get_connection()
        cursor = conn.execute("SELECT * FROM processing_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()

        assert row["corpus_name"] == "Test"
        assert row["scenario"] == "test_scenario"
        assert row["mode"] == "textual"
        assert row["completed_at"] is None

        db.close()

    def test_resumes_existing_run(self, tmp_path: Path) -> None:
        """Resumes an incomplete run with matching parameters."""
        db_path = tmp_path / "test.db"
        db = ProcessingStateDB(db_path)

        # Create first run
        run_id1 = db.get_or_create_run(
            corpus_name="Test",
            corpus_path="/corpus",
            scenario="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_per_document=5,
            total_target=None,
            max_documents=None,
        )

        # Try to create another - should resume
        run_id2 = db.get_or_create_run(
            corpus_name="Test",
            corpus_path="/corpus",
            scenario="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_per_document=5,
            total_target=None,
            max_documents=None,
        )

        assert run_id1 == run_id2

        db.close()

    def test_creates_new_run_when_previous_complete(self, tmp_path: Path) -> None:
        """Creates new run when previous is marked complete."""
        db_path = tmp_path / "test.db"
        db = ProcessingStateDB(db_path)

        run_id1 = db.get_or_create_run(
            corpus_name="Test",
            corpus_path="/corpus",
            scenario="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_per_document=5,
            total_target=None,
            max_documents=None,
        )

        db.mark_run_complete(run_id1)

        run_id2 = db.get_or_create_run(
            corpus_name="Test",
            corpus_path="/corpus",
            scenario="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_per_document=5,
            total_target=None,
            max_documents=None,
        )

        assert run_id2 != run_id1

        db.close()

    def test_saves_and_retrieves_accepted_question(self, tmp_path: Path) -> None:
        """Saves accepted question and retrieves via get_existing_questions."""
        db_path = tmp_path / "test.db"
        db = ProcessingStateDB(db_path)

        run_id = db.get_or_create_run(
            corpus_name="Test",
            corpus_path="/corpus",
            scenario="test",
            mode=GenerationMode.TEXTUAL,
            target_per_document=5,
            total_target=None,
            max_documents=None,
        )

        qa = QAPair(
            question="What is X?",
            answer="X is Y",
            source_document="doc1.txt",
            mode=GenerationMode.TEXTUAL,
        )

        db.save_accepted_question(run_id, "doc1.txt", qa)

        # Retrieve
        existing = db.get_existing_questions(run_id, "doc1.txt")
        assert len(existing) == 1
        assert existing[0].question == "What is X?"
        assert existing[0].answer == "X is Y"

        db.close()

    def test_get_document_question_counts(self, tmp_path: Path) -> None:
        """Counts questions per document."""
        db_path = tmp_path / "test.db"
        db = ProcessingStateDB(db_path)

        run_id = db.get_or_create_run(
            corpus_name="Test",
            corpus_path="/corpus",
            scenario="test",
            mode=GenerationMode.TEXTUAL,
            target_per_document=5,
            total_target=None,
            max_documents=None,
        )

        # Save 3 questions for doc1, 2 for doc2
        for i in range(3):
            qa = QAPair(
                question=f"Q{i}?",
                answer=f"A{i}",
                source_document="doc1.txt",
                mode=GenerationMode.TEXTUAL,
            )
            db.save_accepted_question(run_id, "doc1.txt", qa)

        for i in range(2):
            qa = QAPair(
                question=f"Q{i}?",
                answer=f"A{i}",
                source_document="doc2.txt",
                mode=GenerationMode.TEXTUAL,
            )
            db.save_accepted_question(run_id, "doc2.txt", qa)

        counts = db.get_document_question_counts(run_id)
        assert counts["doc1.txt"] == 3
        assert counts["doc2.txt"] == 2

        db.close()

    def test_marks_document_exhausted(self, tmp_path: Path) -> None:
        """Marks document as exhausted and retrieves."""
        db_path = tmp_path / "test.db"
        db = ProcessingStateDB(db_path)

        run_id = db.get_or_create_run(
            corpus_name="Test",
            corpus_path="/corpus",
            scenario="test",
            mode=GenerationMode.TEXTUAL,
            target_per_document=5,
            total_target=None,
            max_documents=None,
        )

        db.mark_document_exhausted(run_id, "doc1.txt", "consecutive_failures")

        exhausted = db.get_exhausted_documents(run_id)
        assert "doc1.txt" in exhausted

        db.close()

    def test_get_run_stats(self, tmp_path: Path) -> None:
        """Aggregates statistics from question tables."""
        db_path = tmp_path / "test.db"
        db = ProcessingStateDB(db_path)

        run_id = db.get_or_create_run(
            corpus_name="Test",
            corpus_path="/corpus",
            scenario="test",
            mode=GenerationMode.TEXTUAL,
            target_per_document=5,
            total_target=None,
            max_documents=None,
        )

        # Save 5 questions for doc1 (reaches target)
        for i in range(5):
            qa = QAPair(
                question=f"Q{i}?",
                answer=f"A{i}",
                source_document="doc1.txt",
                mode=GenerationMode.TEXTUAL,
            )
            db.save_accepted_question(run_id, "doc1.txt", qa)

        # Save 2 questions for doc2, mark exhausted
        for i in range(2):
            qa = QAPair(
                question=f"Q{i}?",
                answer=f"A{i}",
                source_document="doc2.txt",
                mode=GenerationMode.TEXTUAL,
            )
            db.save_accepted_question(run_id, "doc2.txt", qa)
        db.mark_document_exhausted(run_id, "doc2.txt", "consecutive_failures")

        # Save 3 rejected for doc3
        for i in range(3):
            db.save_rejected_question(
                run_id, "doc3.txt", f"Q{i}?", f"A{i}", RejectionReason.DUPLICATE
            )

        stats = db.get_run_stats(run_id)

        # doc1 reached target (5), doc2 exhausted
        assert stats["documents_processed"] == 2
        assert stats["documents_exhausted"] == 1
        assert stats["total_accepted"] == 7  # 5 + 2
        assert stats["total_rejected"] == 3

        db.close()

    def test_export_to_corpus_result(self, tmp_path: Path) -> None:
        """Exports database to CorpusGenerationResult for RAGAS evaluation."""
        db_path = tmp_path / "test.db"
        db = ProcessingStateDB(db_path)

        run_id = db.get_or_create_run(
            corpus_name="Test Corpus",
            corpus_path="/corpus",
            scenario="test",
            mode=GenerationMode.TEXTUAL,
            target_per_document=5,
            total_target=100,
            max_documents=10,
        )

        qa = QAPair(
            question="Q?",
            answer="A",
            source_document="doc1.txt",
            mode=GenerationMode.TEXTUAL,
        )
        db.save_accepted_question(run_id, "doc1.txt", qa)

        corpus_result = db.export_to_corpus_result(run_id)

        # Verify RAGAS-relevant fields are present
        assert corpus_result.corpus_name == "Test Corpus"
        assert corpus_result.corpus_path == "/corpus"
        assert corpus_result.scenario == "test"
        assert corpus_result.mode == GenerationMode.TEXTUAL
        assert len(corpus_result.questions) == 1
        assert corpus_result.questions[0].question == "Q?"
        assert corpus_result.timestamp  # Should be present

        # Verify removed housekeeping fields are NOT in serialized output
        serialized = corpus_result.model_dump(mode="json")
        removed_fields = [
            "target_per_document",
            "total_target",
            "max_documents",
            "documents_processed",
            "documents_exhausted",
            "total_accepted",
            "total_rejected",
            "per_document_results",
        ]
        for field in removed_fields:
            assert field not in serialized, f"Field '{field}' should not be in output"

        db.close()


class TestExportRunToJson:
    """Tests for JSON export."""

    def test_exports_to_json_file(self, tmp_path: Path) -> None:
        """Exports run to JSON file."""
        db_path = tmp_path / "test.db"
        output_path = tmp_path / "output.json"

        db = ProcessingStateDB(db_path)
        run_id = db.get_or_create_run(
            corpus_name="Test",
            corpus_path="/corpus",
            scenario="test",
            mode=GenerationMode.TEXTUAL,
            target_per_document=5,
            total_target=None,
            max_documents=None,
        )

        export_run_to_json(db, run_id, output_path)

        assert output_path.exists()

        with output_path.open() as f:
            data = json.load(f)

        assert data["corpus_name"] == "Test"
        assert data["scenario"] == "test"

        db.close()


class TestProcessCorpus:
    """Tests for the main process_corpus function."""

    @patch("single_doc_generator.corpus_processor.run_generation")
    def test_processes_all_documents(
        self, mock_run: MagicMock, mock_corpus_dir: Path, tmp_path: Path
    ) -> None:
        """Processes each document in corpus."""
        db_path = tmp_path / "state.db"

        # Track calls and simulate on_accepted callback
        def mock_run_gen(
            document_path: Path,
            on_accepted: MagicMock | None = None,
            **_kwargs: object,
        ) -> GenerationResult:
            # Simulate accepting a question via callback
            if on_accepted:
                qa = QAPair(
                    question="Q?",
                    answer="A",
                    source_document=str(document_path),
                    mode=GenerationMode.TEXTUAL,
                )
                on_accepted(qa)

            return GenerationResult(
                document=str(document_path),
                corpus=str(mock_corpus_dir),
                scenario="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_count=5,
                accepted=[],  # Empty - persistence via callback
                rejected=[],
                stats=GenerationStats(
                    total_attempts=1,
                    accepted_count=1,
                    rejected_count=0,
                    exhausted=False,
                ),
                timestamp="",
            )

        mock_run.side_effect = mock_run_gen

        db = ProcessingStateDB(db_path)
        try:
            run_id = process_corpus(
                corpus_path=mock_corpus_dir,
                scenario_name="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_per_document=5,
                db=db,
            )

            # Should process all 5 documents
            assert mock_run.call_count == 5

            # Verify stats in database
            stats = db.get_run_stats(run_id)
            assert stats["total_accepted"] == 5  # 1 per doc via callback
        finally:
            db.close()

    @patch("single_doc_generator.corpus_processor.run_generation")
    def test_resume_skips_completed_documents(
        self, mock_run: MagicMock, mock_corpus_dir: Path, tmp_path: Path
    ) -> None:
        """Resume mode skips documents that reached target."""
        db_path = tmp_path / "state.db"

        db = ProcessingStateDB(db_path)
        try:
            run_id = db.get_or_create_run(
                corpus_name="Test Corpus",
                corpus_path=str(mock_corpus_dir.resolve()),
                scenario="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_per_document=5,
                total_target=None,
                max_documents=None,
            )

            # Pre-populate: doc1.txt and doc2.md have 5 questions each (target)
            for doc_name in ["doc1.txt", "doc2.md"]:
                for i in range(5):
                    qa = QAPair(
                        question=f"Q{i}?",
                        answer=f"A{i}",
                        source_document=doc_name,
                        mode=GenerationMode.TEXTUAL,
                    )
                    db.save_accepted_question(run_id, doc_name, qa)

            # Mock for remaining docs - no new questions accepted
            def mock_run_gen(
                document_path: Path,
                _on_accepted: MagicMock | None = None,
                **_kwargs: object,
            ) -> GenerationResult:
                return GenerationResult(
                    document=str(document_path),
                    corpus=str(mock_corpus_dir),
                    scenario="test_scenario",
                    mode=GenerationMode.TEXTUAL,
                    target_count=5,
                    accepted=[],
                    rejected=[],
                    stats=GenerationStats(
                        total_attempts=1,
                        accepted_count=0,
                        rejected_count=0,
                        exhausted=False,
                    ),
                    timestamp="",
                )

            mock_run.side_effect = mock_run_gen

            # Resume processing
            run_id = process_corpus(
                corpus_path=mock_corpus_dir,
                scenario_name="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_per_document=5,
                db=db,
            )

            # Should only process the 3 remaining documents
            assert mock_run.call_count == 3
        finally:
            db.close()

    @patch("single_doc_generator.corpus_processor.run_generation")
    def test_resume_within_document(
        self, mock_run: MagicMock, mock_corpus_dir: Path, tmp_path: Path
    ) -> None:
        """Resumes within a partially processed document."""
        db_path = tmp_path / "state.db"

        db = ProcessingStateDB(db_path)
        try:
            run_id = db.get_or_create_run(
                corpus_name="Test Corpus",
                corpus_path=str(mock_corpus_dir.resolve()),
                scenario="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_per_document=5,
                total_target=None,
                max_documents=None,
            )

            # Pre-populate: doc1.txt has 3 questions (partial)
            for i in range(3):
                qa = QAPair(
                    question=f"Existing Q{i}?",
                    answer=f"A{i}",
                    source_document="doc1.txt",
                    mode=GenerationMode.TEXTUAL,
                )
                db.save_accepted_question(run_id, "doc1.txt", qa)

            existing_questions_received: list[QAPair] = []

            def mock_run_gen(
                document_path: Path,
                existing_accepted: list[QAPair] | None = None,
                **_kwargs: object,
            ) -> GenerationResult:
                if existing_accepted:
                    existing_questions_received.extend(existing_accepted)
                return GenerationResult(
                    document=str(document_path),
                    corpus=str(mock_corpus_dir),
                    scenario="test_scenario",
                    mode=GenerationMode.TEXTUAL,
                    target_count=5,
                    accepted=[],
                    rejected=[],
                    stats=GenerationStats(
                        total_attempts=1,
                        accepted_count=0,
                        rejected_count=0,
                        exhausted=False,
                    ),
                    timestamp="",
                )

            mock_run.side_effect = mock_run_gen

            # Resume processing
            process_corpus(
                corpus_path=mock_corpus_dir,
                scenario_name="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_per_document=5,
                db=db,
            )

            # doc1.txt should have received 3 existing questions
            assert len(existing_questions_received) == 3
            assert existing_questions_received[0].question == "Existing Q0?"
        finally:
            db.close()

    @patch("single_doc_generator.corpus_processor.run_generation")
    def test_skips_exhausted_documents(
        self, mock_run: MagicMock, mock_corpus_dir: Path, tmp_path: Path
    ) -> None:
        """Skips documents marked as exhausted."""
        db_path = tmp_path / "state.db"

        db = ProcessingStateDB(db_path)
        try:
            run_id = db.get_or_create_run(
                corpus_name="Test Corpus",
                corpus_path=str(mock_corpus_dir.resolve()),
                scenario="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_per_document=5,
                total_target=None,
                max_documents=None,
            )

            # Mark doc1.txt as exhausted
            db.mark_document_exhausted(run_id, "doc1.txt", "consecutive_failures")

            def mock_run_gen(
                document_path: Path, **_kwargs: object
            ) -> GenerationResult:
                return GenerationResult(
                    document=str(document_path),
                    corpus=str(mock_corpus_dir),
                    scenario="test_scenario",
                    mode=GenerationMode.TEXTUAL,
                    target_count=5,
                    accepted=[],
                    rejected=[],
                    stats=GenerationStats(
                        total_attempts=1,
                        accepted_count=0,
                        rejected_count=0,
                        exhausted=False,
                    ),
                    timestamp="",
                )

            mock_run.side_effect = mock_run_gen

            process_corpus(
                corpus_path=mock_corpus_dir,
                scenario_name="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_per_document=5,
                db=db,
            )

            # Should skip doc1.txt, process only 4 documents
            assert mock_run.call_count == 4
        finally:
            db.close()

    @patch("single_doc_generator.corpus_processor.run_generation")
    def test_exports_to_json(
        self, mock_run: MagicMock, mock_corpus_dir: Path, tmp_path: Path
    ) -> None:
        """Exports results to JSON file when output_path provided."""
        db_path = tmp_path / "state.db"
        output_path = tmp_path / "subdir" / "output.json"

        # Use side_effect to simulate on_accepted callback - exports happen via callback
        def mock_run_gen(
            document_path: Path,
            on_accepted: MagicMock | None = None,
            **_kwargs: object,
        ) -> GenerationResult:
            # Simulate accepting a question via callback - this triggers export
            if on_accepted:
                qa = QAPair(
                    question="Test question?",
                    answer="Test answer",
                    source_document=str(document_path),
                    mode=GenerationMode.TEXTUAL,
                )
                on_accepted(qa)

            return GenerationResult(
                document=str(document_path),
                corpus=str(mock_corpus_dir),
                scenario="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_count=5,
                accepted=[],  # Empty - persistence via callback
                rejected=[],
                stats=GenerationStats(
                    total_attempts=1,
                    accepted_count=1,
                    rejected_count=0,
                    exhausted=False,
                ),
                timestamp="",
            )

        mock_run.side_effect = mock_run_gen

        db = ProcessingStateDB(db_path)
        try:
            process_corpus(
                corpus_path=mock_corpus_dir,
                scenario_name="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_per_document=5,
                db=db,
                output_path=output_path,
            )
        finally:
            db.close()

        assert output_path.exists()

        # Verify it's valid JSON with expected structure
        with output_path.open() as f:
            data = json.load(f)

        assert data["corpus_name"] == "Test Corpus"
        assert data["scenario"] == "test_scenario"
        # Verify questions were exported
        assert len(data["questions"]) == 5  # One per document

    @patch("single_doc_generator.corpus_processor.run_generation")
    def test_invokes_callbacks(
        self, mock_run: MagicMock, mock_corpus_dir: Path, tmp_path: Path
    ) -> None:
        """Invokes progress callbacks at appropriate times."""
        db_path = tmp_path / "state.db"

        mock_run.return_value = GenerationResult(
            document="doc.txt",
            corpus=str(mock_corpus_dir),
            scenario="test_scenario",
            mode=GenerationMode.TEXTUAL,
            target_count=5,
            accepted=[],
            rejected=[],
            stats=GenerationStats(
                total_attempts=1,
                accepted_count=0,
                rejected_count=1,
                exhausted=False,
            ),
            timestamp="",
        )

        on_start = MagicMock()
        on_complete = MagicMock()

        db = ProcessingStateDB(db_path)
        try:
            process_corpus(
                corpus_path=mock_corpus_dir,
                scenario_name="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_per_document=5,
                db=db,
                on_document_start=on_start,
                on_document_complete=on_complete,
            )
        finally:
            db.close()

        # Should be called once per document (5 docs)
        assert on_start.call_count == 5
        assert on_complete.call_count == 5

    @patch("single_doc_generator.corpus_processor.run_generation")
    def test_continues_on_document_error(
        self, mock_run: MagicMock, mock_corpus_dir: Path, tmp_path: Path
    ) -> None:
        """Continues processing when a document fails."""
        db_path = tmp_path / "state.db"

        call_count = [0]

        def mock_run_gen(document_path: Path, **_kwargs: object) -> GenerationResult:
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Document processing failed")
            return GenerationResult(
                document=str(document_path),
                corpus=str(mock_corpus_dir),
                scenario="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_count=5,
                accepted=[],
                rejected=[],
                stats=GenerationStats(
                    total_attempts=0,
                    accepted_count=0,
                    rejected_count=0,
                    exhausted=False,
                ),
                timestamp="",
            )

        mock_run.side_effect = mock_run_gen

        db = ProcessingStateDB(db_path)
        try:
            process_corpus(
                corpus_path=mock_corpus_dir,
                scenario_name="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_per_document=5,
                db=db,
            )

            # Should attempt all 5 docs
            assert mock_run.call_count == 5
        finally:
            db.close()

    def test_raises_on_missing_corpus_yaml(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when corpus.yaml missing."""
        db_path = tmp_path / "state.db"
        db = ProcessingStateDB(db_path)
        try:
            with pytest.raises(FileNotFoundError):
                process_corpus(
                    corpus_path=tmp_path,
                    scenario_name="test",
                    mode=GenerationMode.TEXTUAL,
                    target_per_document=5,
                    db=db,
                )
        finally:
            db.close()

    def test_raises_on_invalid_scenario(
        self, mock_corpus_dir: Path, tmp_path: Path
    ) -> None:
        """Raises KeyError when scenario doesn't exist."""
        db_path = tmp_path / "state.db"
        db = ProcessingStateDB(db_path)
        try:
            with pytest.raises(KeyError):
                process_corpus(
                    corpus_path=mock_corpus_dir,
                    scenario_name="nonexistent_scenario",
                    mode=GenerationMode.TEXTUAL,
                    target_per_document=5,
                    db=db,
                )
        finally:
            db.close()

    @patch("single_doc_generator.corpus_processor.run_generation")
    def test_stops_at_total_target(
        self, mock_run: MagicMock, mock_corpus_dir: Path, tmp_path: Path
    ) -> None:
        """Stops processing when total_target is reached."""
        db_path = tmp_path / "state.db"

        def mock_run_gen(
            document_path: Path,
            on_accepted: MagicMock | None = None,
            **_kwargs: object,
        ) -> GenerationResult:
            # Simulate accepting 3 questions via callback
            if on_accepted:
                for i in range(3):
                    qa = QAPair(
                        question=f"Q{i}?",
                        answer=f"A{i}",
                        source_document=str(document_path),
                        mode=GenerationMode.TEXTUAL,
                    )
                    on_accepted(qa)

            return GenerationResult(
                document=str(document_path),
                corpus=str(mock_corpus_dir),
                scenario="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_count=5,
                accepted=[],
                rejected=[],
                stats=GenerationStats(
                    total_attempts=3,
                    accepted_count=3,
                    rejected_count=0,
                    exhausted=False,
                ),
                timestamp="",
            )

        mock_run.side_effect = mock_run_gen

        db = ProcessingStateDB(db_path)
        try:
            process_corpus(
                corpus_path=mock_corpus_dir,
                scenario_name="test_scenario",
                mode=GenerationMode.TEXTUAL,
                target_per_document=5,
                db=db,
                total_target=5,  # Stop after 5 total questions
            )
        finally:
            db.close()

        # With 3 questions per doc and target of 5, should stop after 2 docs
        # (first doc gives 3, we're under target; second doc gives 6, we're over)
        assert mock_run.call_count == 2
