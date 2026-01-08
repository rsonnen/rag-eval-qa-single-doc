"""Integration tests for agent tracing with real LLM calls.

These tests verify that tracing works correctly with actual agent invocations,
not mocked callbacks.
"""

import io
import os
import sys
from unittest.mock import patch

import pytest

from single_doc_generator.config import CorpusConfig, EvaluationScenario
from single_doc_generator.generator import GeneratedQA, generate_question
from single_doc_generator.models import GenerationMode
from single_doc_generator.validator import validate_question

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def check_llm_available():
    """Skip tests if LLM not available."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping LLM integration tests")


@pytest.fixture
def rpg_config():
    """Corpus config for RPG spell corpus."""
    return CorpusConfig(
        name="D&D Spell Compendium",
        corpus_context="RPG spell descriptions with mechanics",
        scenarios={
            "rules_quiz": EvaluationScenario(
                name="rules_quiz",
                description="Rules quiz testing spell mechanics",
            )
        },
    )


class TestTracerIntegration:
    """Integration tests for AgentTracer with real agent invocations."""

    def test_tracer_does_not_output_to_console(self, rpg_spell_file, rpg_config):
        """Verify tracer logs don't appear on stdout/stderr."""
        # Capture stdout/stderr
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr

            # Run generator which uses AgentTracer
            generate_question(
                document_path=rpg_spell_file,
                corpus_config=rpg_config,
                scenario_name="rules_quiz",
                mode=GenerationMode.TEXTUAL,
            )

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        stdout_output = captured_stdout.getvalue()
        stderr_output = captured_stderr.getvalue()

        # Tracer logs should NOT appear in console output
        assert "tool_call" not in stdout_output
        assert "tool_result" not in stdout_output
        assert "llm_output" not in stdout_output
        assert "tool_call" not in stderr_output
        assert "tool_result" not in stderr_output
        assert "llm_output" not in stderr_output

    def test_tracer_counts_tool_calls(self, rpg_spell_file, rpg_config):
        """Verify tracer correctly counts tool invocations."""
        tool_calls_logged = []

        def capture_info(msg, *args, **_kwargs):
            if "tool_call" in msg:
                tool_calls_logged.append(args)

        with patch("single_doc_generator.tracing._get_otel_logger") as mock_get:
            mock_logger = mock_get.return_value
            mock_logger.info.side_effect = capture_info
            mock_logger.debug.return_value = None

            generate_question(
                document_path=rpg_spell_file,
                corpus_config=rpg_config,
                scenario_name="rules_quiz",
                mode=GenerationMode.TEXTUAL,
            )

        # Generator should make multiple tool calls
        assert len(tool_calls_logged) >= 2, (
            f"Expected at least 2 tool calls, got {len(tool_calls_logged)}"
        )

        # Tool numbers should increment correctly
        tool_numbers = [args[1] for args in tool_calls_logged]
        assert tool_numbers == list(range(1, len(tool_numbers) + 1)), (
            f"Tool numbers should be sequential: {tool_numbers}"
        )

    def test_tracer_records_tool_names(self, rpg_spell_file, rpg_config):
        """Verify tracer captures correct tool names."""
        tool_names_logged = []

        def capture_info(msg, *args, **_kwargs):
            if "tool_call" in msg and len(args) >= 3:
                tool_names_logged.append(args[2])  # tool_name is 3rd arg

        with patch("single_doc_generator.tracing._get_otel_logger") as mock_get:
            mock_logger = mock_get.return_value
            mock_logger.info.side_effect = capture_info
            mock_logger.debug.return_value = None

            generate_question(
                document_path=rpg_spell_file,
                corpus_config=rpg_config,
                scenario_name="rules_quiz",
                mode=GenerationMode.TEXTUAL,
            )

        # Should have captured tool names
        assert len(tool_names_logged) > 0

        # Tool names should be actual document tools
        valid_tools = {
            "read_lines",
            "search",
            "view_page",
            "list_visual_content",
            "submit_qa",
            "report_exhausted",
        }
        for name in tool_names_logged:
            assert name in valid_tools, f"Unknown tool: {name}"

    def test_tracer_correlates_tool_results_with_names(self, shakespeare_file):
        """Verify tool_result logs include the correct tool name via run_id."""
        results_logged = []

        def capture_info(msg, *args, **_kwargs):
            if "tool_result" in msg:
                # Format: "[%s] tool_result #%d %s: %s"
                # args: (context, tool_num, tool_name, preview)
                results_logged.append(
                    {
                        "tool_num": args[1],
                        "tool_name": args[2],
                    }
                )

        with patch("single_doc_generator.tracing._get_otel_logger") as mock_get:
            mock_logger = mock_get.return_value
            mock_logger.info.side_effect = capture_info
            mock_logger.debug.return_value = None

            candidate = GeneratedQA(
                question="What character appears in Hamlet?",
                answer="Hamlet",
                reasoning="Character identification",
            )
            validate_question(
                document_path=shakespeare_file,
                candidate=candidate,
            )

        # Should have results with tool names
        assert len(results_logged) > 0

        for result in results_logged:
            # Each result should have a valid tool name, not "unknown"
            assert result["tool_name"] != "unknown", (
                f"Tool result #{result['tool_num']} has unknown tool name"
            )
            # Tool number should be > 0
            assert result["tool_num"] > 0


class TestTracerWithParallelTools:
    """Test tracer handles parallel tool execution correctly."""

    def test_parallel_tool_results_have_correct_numbers(self, arxiv_paper_pdf):
        """When tools run in parallel, results should have correct numbers."""
        calls = []
        results = []

        def capture_info(msg, *args, **_kwargs):
            if "tool_call" in msg:
                calls.append({"num": args[1], "name": args[2]})
            elif "tool_result" in msg:
                results.append({"num": args[1], "name": args[2]})

        with patch("single_doc_generator.tracing._get_otel_logger") as mock_get:
            mock_logger = mock_get.return_value
            mock_logger.info.side_effect = capture_info
            mock_logger.debug.return_value = None

            candidate = GeneratedQA(
                question="What is this paper about?",
                answer="Machine learning",
                reasoning="Topic identification",
            )
            validate_question(
                document_path=arxiv_paper_pdf,
                candidate=candidate,
            )

        # Build expected mapping from calls
        expected = {c["num"]: c["name"] for c in calls}

        # Verify each result matches expected
        for result in results:
            num = result["num"]
            name = result["name"]
            if num in expected:
                assert name == expected[num], (
                    f"Result #{num} has name '{name}' but call was '{expected[num]}'"
                )
