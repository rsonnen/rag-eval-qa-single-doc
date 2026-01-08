"""Integration tests that call real LLMs via LiteLLM proxy.

These tests require:
- OPENAI_API_KEY set in environment
- OPENAI_BASE_URL pointing to LiteLLM proxy
- Running LiteLLM proxy with configured models

Run with: uv run pytest tests/integration/test_llm_integration.py -v
Skip in CI by using: pytest -m "not integration"
"""

import os

import pytest

from single_doc_generator.config import CorpusConfig, EvaluationScenario, load_config
from single_doc_generator.generator import (
    DocumentExhaustedError,
    GeneratedQA,
    generate_question,
)
from single_doc_generator.models import GenerationMode, RejectionReason
from single_doc_generator.validator import validate_question

pytestmark = pytest.mark.integration


def make_corpus_config(
    name: str,
    corpus_context: str,
    scenario_name: str = "rag_eval",
    scenario_description: str = "Questions to verify retrieval accuracy",
) -> CorpusConfig:
    """Create a corpus config for testing."""
    return CorpusConfig(
        name=name,
        corpus_context=corpus_context,
        scenarios={
            scenario_name: EvaluationScenario(
                name=scenario_name,
                description=scenario_description,
            )
        },
    )


@pytest.fixture(scope="module", autouse=True)
def check_llm_available():
    """Skip tests if LLM not available."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping LLM integration tests")

    # Clear config cache to ensure fresh load
    load_config.cache_clear()


class TestGeneratorIntegration:
    """Integration tests for question generator with real LLMs."""

    @pytest.fixture
    def shakespeare_config(self):
        """Corpus config for Shakespeare text corpus."""
        return make_corpus_config(
            name="Shakespeare Complete Works",
            corpus_context="Complete works of William Shakespeare",
            scenario_name="literature_trivia",
            scenario_description="Literature trivia questions about Shakespeare",
        )

    @pytest.fixture
    def rpg_config(self):
        """Corpus config for RPG spell corpus."""
        return make_corpus_config(
            name="D&D Spell Compendium",
            corpus_context="RPG spell descriptions with mechanics",
            scenario_name="rules_reference",
            scenario_description="Rules quiz testing spell mechanics",
        )

    def test_generates_question_from_shakespeare(
        self, shakespeare_file, shakespeare_config
    ):
        """Test generating a Q/A pair from Shakespeare text."""
        result, viewed_pages = generate_question(
            document_path=shakespeare_file,
            corpus_config=shakespeare_config,
            scenario_name="literature_trivia",
            mode=GenerationMode.TEXTUAL,
        )

        # Verify structure
        assert isinstance(result, GeneratedQA)
        assert result.question
        assert result.answer
        assert result.reasoning

        # Question should be about Shakespeare content
        assert len(result.question) > 10
        # Question may be interrogative (?) or imperative (name, list, identify...)
        # Just verify it's a substantive question about the content

        # Viewed pages returned (may be empty for text files)
        assert isinstance(viewed_pages, list)

    def test_generates_question_from_rpg_spell(self, rpg_spell_file, rpg_config):
        """Test generating a Q/A pair from RPG spell markdown."""
        result, viewed_pages = generate_question(
            document_path=rpg_spell_file,
            corpus_config=rpg_config,
            scenario_name="rules_reference",
            mode=GenerationMode.TEXTUAL,
        )

        assert isinstance(result, GeneratedQA)
        assert result.question
        assert result.answer
        assert result.reasoning
        assert isinstance(viewed_pages, list)


class TestValidatorIntegration:
    """Integration tests for validator with real LLMs."""

    def test_gemini_reasoning_effort_prevents_empty_choices(self, shakespeare_file):
        """Test that reasoning_effort=none prevents empty choices errors.

        This test verifies the fix for Gemini 2.5 Flash's "thinking mode" causing
        empty choices responses during multi-turn tool calling.

        By setting max_empty_choices_retries=1 (single attempt, no retries), we
        ensure that any empty choices error would cause immediate failure. If this
        test passes consistently, it proves the reasoning_effort=none kwarg is
        preventing the issue.

        If this test fails with VALIDATION_FAILED and "empty response", the fix
        is not working and needs investigation.
        """
        # Use a straightforward question that should be easily answerable
        candidate = GeneratedQA(
            question="What play contains the character Hamlet?",
            answer="Hamlet",
            reasoning="Tests knowledge of character-play associations",
        )

        # Call with max_empty_choices_retries=1: try once, no retries
        # If reasoning_effort=none works, validation should succeed
        # If it doesn't work, we'd see VALIDATION_FAILED with "empty response"
        result = validate_question(
            document_path=shakespeare_file,
            candidate=candidate,
            corpus_context="Complete works of William Shakespeare",
            scenario_description="Literature trivia about Shakespeare",
            max_empty_choices_retries=1,
        )

        # The key assertion: we should NOT get a "Model returned empty response"
        # failure. Any other result (pass, reject for other reasons) is acceptable.
        if (
            not result.passed
            and result.rejection_reason == RejectionReason.VALIDATION_FAILED
        ):
            # VALIDATION_FAILED could be empty choices or other issues
            # Check the reasoning to distinguish
            assert "empty response" not in result.reasoning.lower(), (
                f"Got empty choices error despite reasoning_effort=none fix: "
                f"{result.reasoning}"
            )

    def test_validates_good_question(self, shakespeare_file):
        """Test validator passes a good question answerable from document."""
        # Create a question we know is answerable from Shakespeare
        candidate = GeneratedQA(
            question="What play contains the character Hamlet?",
            answer="Hamlet",
            reasoning="Tests knowledge of character-play associations",
        )

        result = validate_question(
            document_path=shakespeare_file,
            candidate=candidate,
            corpus_context="Complete works of William Shakespeare",
            scenario_description="Literature trivia about Shakespeare",
        )

        # Should pass - this is clearly answerable from Shakespeare text
        assert result.passed is True or result.rejection_reason in [
            RejectionReason.TRIVIAL  # Might be considered too easy
        ]

    def test_rejects_unanswerable_question(self, shakespeare_file):
        """Test validator rejects question not answerable from document."""
        # Create a question that cannot be answered from Shakespeare
        candidate = GeneratedQA(
            question="What was Shakespeare's favorite color?",
            answer="Blue",
            reasoning="Personal preference question",
        )

        result = validate_question(
            document_path=shakespeare_file,
            candidate=candidate,
            corpus_context="Complete works of William Shakespeare",
            scenario_description="Literature trivia about Shakespeare",
        )

        # Should fail - this information isn't in Shakespeare's plays
        # LLMs may reject for various reasons, all are acceptable
        assert result.passed is False
        assert result.rejection_reason is not None

    def test_rejects_wrong_answer(self, shakespeare_file):
        """Test validator rejects question with incorrect ground truth."""
        # Create a question with a wrong answer
        candidate = GeneratedQA(
            question="Who is the main character in the play Hamlet?",
            answer="Macbeth",  # Wrong - should be Hamlet
            reasoning="Character identification",
        )

        result = validate_question(
            document_path=shakespeare_file,
            candidate=candidate,
            corpus_context="Complete works of William Shakespeare",
            scenario_description="Literature trivia about Shakespeare",
        )

        # Should fail - answer is wrong
        # LLM may also classify as trivial or ambiguous - all rejections valid
        assert result.passed is False
        assert result.rejection_reason is not None


class TestEndToEndIntegration:
    """End-to-end tests: generate then validate."""

    @pytest.fixture
    def rpg_config(self):
        """Corpus config for RPG spell corpus."""
        return make_corpus_config(
            name="D&D Spell Compendium",
            corpus_context="RPG spell descriptions with mechanics",
            scenario_name="rules_quiz",
            scenario_description="Rules quiz testing spell mechanics",
        )

    def test_generate_and_validate_cycle(self, rpg_spell_file, rpg_config):
        """Test full generate -> validate cycle."""
        # Generate a question
        generated, viewed_pages = generate_question(
            document_path=rpg_spell_file,
            corpus_config=rpg_config,
            scenario_name="rules_quiz",
            mode=GenerationMode.TEXTUAL,
        )

        assert isinstance(generated, GeneratedQA)
        assert generated.question
        assert generated.answer
        assert isinstance(viewed_pages, list)

        # Validate the generated question with corpus context
        validation = validate_question(
            document_path=rpg_spell_file,
            candidate=generated,
            corpus_context=rpg_config.corpus_context,
            scenario_description=rpg_config.get_scenario("rules_quiz").description,
        )

        # A well-generated question should often pass validation
        # But not always - this tests the integration, not perfection
        assert validation.reasoning  # Should always have reasoning

        if validation.passed:
            assert validation.validator_answer  # Should have answer if passed
        else:
            assert validation.rejection_reason  # Should have reason if failed


class TestVisualModeIntegration:
    """Integration tests for visual mode with visual content."""

    @pytest.fixture
    def research_config(self):
        """Corpus config for research papers."""
        return make_corpus_config(
            name="Research Papers",
            corpus_context="Academic research papers with figures and tables",
            scenario_name="visual_comprehension",
            scenario_description="Questions requiring understanding of visuals",
        )

    def test_generates_visual_question_from_pdf_with_visuals(
        self, arxiv_paper_pdf, research_config
    ):
        """Test generating a visual question from PDF with figures/tables."""
        result, viewed_pages = generate_question(
            document_path=arxiv_paper_pdf,
            corpus_config=research_config,
            scenario_name="visual_comprehension",
            mode=GenerationMode.VISUAL,
        )

        # Verify structure
        assert isinstance(result, GeneratedQA)
        assert result.question
        assert result.answer
        assert result.reasoning

        # Visual mode should include content_refs to visual elements
        assert len(result.content_refs) > 0, (
            "Visual mode should reference visual content"
        )

        # Visual mode should return viewed pages
        assert isinstance(viewed_pages, list)
        assert len(viewed_pages) > 0, "Visual mode should view at least one page"

    def test_visual_mode_exhausts_on_text_document(
        self, shakespeare_file, research_config
    ):
        """Test that visual mode reports exhaustion for text-only documents."""
        with pytest.raises(DocumentExhaustedError) as exc_info:
            generate_question(
                document_path=shakespeare_file,
                corpus_config=research_config,
                scenario_name="visual_comprehension",
                mode=GenerationMode.VISUAL,
            )

        # Should indicate no visual content was found
        assert (
            "visual" in exc_info.value.reason.lower()
            or "no" in exc_info.value.reason.lower()
        )

    def test_visual_mode_generate_and_validate_cycle(
        self, arxiv_paper_pdf, research_config
    ):
        """Test full generate -> validate cycle for visual mode."""
        # Generate a visual question
        generated, viewed_pages = generate_question(
            document_path=arxiv_paper_pdf,
            corpus_config=research_config,
            scenario_name="visual_comprehension",
            mode=GenerationMode.VISUAL,
        )

        assert isinstance(generated, GeneratedQA)
        assert generated.question
        assert generated.answer
        assert len(generated.content_refs) > 0
        assert isinstance(viewed_pages, list)
        assert len(viewed_pages) > 0

        # Validate with visual mode and corpus context
        validation = validate_question(
            document_path=arxiv_paper_pdf,
            candidate=generated,
            mode=GenerationMode.VISUAL,
            corpus_context=research_config.corpus_context,
            scenario_description=research_config.get_scenario(
                "visual_comprehension"
            ).description,
        )

        # Check validation completed (pass or fail is ok for integration)
        assert validation.reasoning

        if validation.passed:
            assert validation.validator_answer
        else:
            # Visual questions may fail validation for various reasons
            assert validation.rejection_reason is not None

    def test_previous_viewed_pages_respected(self, arxiv_paper_pdf, research_config):
        """Test that previous_viewed_pages parameter influences generation."""
        # First generation - no previous pages
        result1, pages1 = generate_question(
            document_path=arxiv_paper_pdf,
            corpus_config=research_config,
            scenario_name="visual_comprehension",
            mode=GenerationMode.VISUAL,
        )

        assert len(pages1) > 0

        # Second generation - pass previous pages
        # The generator should try different pages
        result2, _pages2 = generate_question(
            document_path=arxiv_paper_pdf,
            corpus_config=research_config,
            scenario_name="visual_comprehension",
            mode=GenerationMode.VISUAL,
            previous_viewed_pages=pages1,
        )

        # Both should produce valid questions
        assert result1.question
        assert result2.question

        # Questions should be different
        assert result1.question != result2.question
