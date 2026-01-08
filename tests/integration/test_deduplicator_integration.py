"""Integration tests for deduplicator with real LLMs.

These tests require:
- OPENAI_API_KEY set in environment
- OPENAI_BASE_URL pointing to LiteLLM proxy
- Running LiteLLM proxy with configured models

Run with: uv run pytest tests/integration/test_deduplicator_integration.py -v
Skip in CI by using: pytest -m "not integration"
"""

import os

import pytest

from single_doc_generator.config import load_config
from single_doc_generator.deduplicator import deduplicate_question

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def check_llm_available():
    """Skip tests if LLM not available."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping LLM integration tests")

    # Clear config cache to ensure fresh load
    load_config.cache_clear()


class TestDeduplicatorIntegration:
    """Integration tests for semantic deduplication with real LLMs."""

    def test_detects_exact_duplicate(self):
        """Test detection of identical question text."""
        result = deduplicate_question(
            question="Who wrote Hamlet?",
            existing_questions=["Who wrote Hamlet?", "When was Macbeth written?"],
        )

        assert result.is_duplicate is True
        assert result.duplicate_of == "Who wrote Hamlet?"

    def test_detects_semantic_duplicate_rephrased(self):
        """Test detection of same question with different wording."""
        result = deduplicate_question(
            question="Which playwright authored the play Hamlet?",
            existing_questions=[
                "Who wrote Hamlet?",
                "What year was Romeo and Juliet published?",
            ],
        )

        assert result.is_duplicate is True
        assert result.duplicate_of == "Who wrote Hamlet?"

    def test_detects_semantic_duplicate_synonyms(self):
        """Test detection using synonyms and restructured sentence."""
        result = deduplicate_question(
            question="What is the capital city of France?",
            existing_questions=[
                "Name France's capital",
                "Who is the King of England?",
            ],
        )

        assert result.is_duplicate is True
        assert result.duplicate_of == "Name France's capital"

    def test_passes_distinct_question_same_topic(self):
        """Test that related but distinct questions are not duplicates."""
        result = deduplicate_question(
            question="When was Hamlet written?",
            existing_questions=[
                "Who wrote Hamlet?",
                "What is the setting of Hamlet?",
            ],
        )

        # Different facts about the same topic should NOT be duplicates
        assert result.is_duplicate is False
        assert result.duplicate_of is None

    def test_passes_distinct_question_different_entity(self):
        """Test questions about different entities are not duplicates."""
        result = deduplicate_question(
            question="What is the capital of Germany?",
            existing_questions=[
                "What is the capital of France?",
                "What is the capital of Spain?",
            ],
        )

        assert result.is_duplicate is False
        assert result.duplicate_of is None

    def test_handles_many_existing_questions(self):
        """Test deduplication works with 50+ existing questions."""
        existing = [f"What is fact number {i} about the document?" for i in range(50)]

        result = deduplicate_question(
            question="Who is the main character in Chapter 3?",
            existing_questions=existing,
        )

        # Should process without error and find no duplicate
        assert result.is_duplicate is False
        assert result.reasoning

    def test_identifies_correct_duplicate_among_many(self):
        """Test correct duplicate identification in a large list."""
        existing = [
            "What color is the sky?",
            "How many legs does a spider have?",
            "What is the boiling point of water?",
            "Who painted the Mona Lisa?",
            "What is the largest planet?",
        ]

        result = deduplicate_question(
            question="Which artist created the Mona Lisa painting?",
            existing_questions=existing,
        )

        assert result.is_duplicate is True
        assert result.duplicate_of == "Who painted the Mona Lisa?"

    def test_distinguishes_similar_but_different_scope(self):
        """Test questions with similar words but different scope."""
        result = deduplicate_question(
            question="How many pages is the entire document?",
            existing_questions=[
                "How many pages is Chapter 1?",
                "How many chapters are in the document?",
            ],
        )

        # Same "how many" pattern but asking about different things
        assert result.is_duplicate is False

    def test_empty_list_returns_unique(self):
        """Test first question is always unique (no LLM call needed)."""
        result = deduplicate_question(
            question="Any question at all?",
            existing_questions=[],
        )

        assert result.is_duplicate is False
        assert "First question" in result.reasoning
