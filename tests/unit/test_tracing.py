"""Tests for agent tracing module."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from single_doc_generator.tracing import _MAX_PREVIEW_LEN, _OTLP_ENDPOINT, AgentTracer


class TestAgentTracer:
    """Tests for AgentTracer callback handler."""

    @pytest.fixture(autouse=True)
    def mock_logger(self) -> None:
        """Replace the tracer's logger with a mock after instantiation."""
        # We patch at the module level to prevent actual OTLP connections
        with patch("single_doc_generator.tracing._get_otel_logger") as mock_get:
            mock_get.return_value = MagicMock()
            yield

    def test_init_sets_context(self) -> None:
        """Tracer stores context label."""
        tracer = AgentTracer(context="validator")
        assert tracer._context == "validator"

    def test_init_default_context(self) -> None:
        """Tracer uses 'agent' as default context."""
        tracer = AgentTracer()
        assert tracer._context == "agent"

    def test_init_tool_count_zero(self) -> None:
        """Tracer starts with zero tool count."""
        tracer = AgentTracer()
        assert tracer.tool_count == 0

    def test_on_tool_start_increments_count(self) -> None:
        """on_tool_start increments tool count."""
        tracer = AgentTracer()
        tracer.on_tool_start(
            serialized={"name": "search"},
            input_str="test query",
            run_id=uuid4(),
        )
        assert tracer.tool_count == 1

        tracer.on_tool_start(
            serialized={"name": "read_lines"},
            input_str="1-100",
            run_id=uuid4(),
        )
        assert tracer.tool_count == 2

    def test_on_tool_start_logs_info(self) -> None:
        """on_tool_start logs tool call at INFO level."""
        tracer = AgentTracer(context="generator")

        run_id = uuid4()
        tracer.on_tool_start(
            serialized={"name": "search"},
            input_str="test query",
            run_id=run_id,
        )

        tracer._logger.info.assert_called_once()
        # Check format string and args
        call_args = tracer._logger.info.call_args
        format_str = call_args[0][0]
        args = call_args[0][1:]
        assert "tool_call" in format_str
        assert "generator" in args
        assert "search" in args

    def test_on_tool_end_logs_result_with_tool_name(self) -> None:
        """on_tool_end logs tool result with correct tool name from run_id."""
        tracer = AgentTracer(context="validator")

        # First call on_tool_start to register the run_id
        run_id = uuid4()
        tracer.on_tool_start(
            serialized={"name": "read_lines"},
            input_str="1-100",
            run_id=run_id,
        )
        tracer._logger.reset_mock()

        tracer.on_tool_end(
            output="Line 1: Hello\nLine 2: World",
            run_id=run_id,
        )

        tracer._logger.info.assert_called_once()
        call_args = tracer._logger.info.call_args
        format_str = call_args[0][0]
        args = call_args[0][1:]
        assert "tool_result" in format_str
        assert "validator" in args
        assert "read_lines" in args
        assert 1 in args  # tool number

    def test_on_tool_end_correlates_with_correct_tool(self) -> None:
        """on_tool_end uses run_id to correlate with correct tool_start."""
        tracer = AgentTracer()

        # Start multiple tools in parallel
        run_id_1 = uuid4()
        run_id_2 = uuid4()
        run_id_3 = uuid4()

        tracer.on_tool_start(
            serialized={"name": "search"}, input_str="query1", run_id=run_id_1
        )
        tracer.on_tool_start(
            serialized={"name": "read_lines"}, input_str="1-50", run_id=run_id_2
        )
        tracer.on_tool_start(
            serialized={"name": "view_page"}, input_str="page 1", run_id=run_id_3
        )
        tracer._logger.reset_mock()

        # End tools in different order
        tracer.on_tool_end(output="result 2", run_id=run_id_2)

        call_args = tracer._logger.info.call_args
        args = call_args[0][1:]
        # Should be tool #2 (read_lines), not #3
        assert 2 in args
        assert "read_lines" in args

    def test_on_tool_error_logs_with_tool_name(self) -> None:
        """on_tool_error logs error with correct tool name."""
        tracer = AgentTracer(context="generator")

        run_id = uuid4()
        tracer.on_tool_start(
            serialized={"name": "search"},
            input_str="bad query",
            run_id=run_id,
        )
        tracer._logger.reset_mock()

        tracer.on_tool_error(
            error=ValueError("Something went wrong"),
            run_id=run_id,
        )

        tracer._logger.error.assert_called_once()
        call_args = tracer._logger.error.call_args
        format_str = call_args[0][0]
        args = call_args[0][1:]
        assert "tool_error" in format_str
        assert "generator" in args
        assert "search" in args

    def test_truncate_short_text(self) -> None:
        """_truncate returns short text unchanged."""
        text = "short text"
        result = AgentTracer._truncate(text)
        assert result == "short text"

    def test_truncate_long_text(self) -> None:
        """_truncate truncates long text with ellipsis."""
        text = "x" * 500
        result = AgentTracer._truncate(text)
        assert len(result) == _MAX_PREVIEW_LEN + 3  # +3 for "..."
        assert result.endswith("...")

    def test_truncate_replaces_newlines(self) -> None:
        """_truncate replaces newlines with spaces."""
        text = "line1\nline2\nline3"
        result = AgentTracer._truncate(text)
        assert "\n" not in result
        assert result == "line1 line2 line3"

    def test_truncate_strips_whitespace(self) -> None:
        """_truncate strips leading/trailing whitespace."""
        text = "  padded text  "
        result = AgentTracer._truncate(text)
        assert result == "padded text"

    def test_on_tool_start_handles_missing_name(self) -> None:
        """on_tool_start uses 'unknown' for missing tool name."""
        tracer = AgentTracer()

        tracer.on_tool_start(
            serialized={},  # No 'name' key
            input_str="test",
            run_id=uuid4(),
        )

        call_args = tracer._logger.info.call_args
        args = call_args[0][1:]
        assert "unknown" in args

    def test_on_tool_end_unknown_run_id_defaults(self) -> None:
        """on_tool_end uses defaults for unknown run_id."""
        tracer = AgentTracer()

        # Call on_tool_end without prior on_tool_start
        tracer.on_tool_end(output="result", run_id=uuid4())

        call_args = tracer._logger.info.call_args
        args = call_args[0][1:]
        assert 0 in args  # Default tool number
        assert "unknown" in args  # Default tool name

    def test_extra_attributes_include_context(self) -> None:
        """Log calls include agent_context in extra."""
        tracer = AgentTracer(context="validator")

        tracer.on_tool_start(
            serialized={"name": "search"},
            input_str="query",
            run_id=uuid4(),
        )

        call_kwargs = tracer._logger.info.call_args[1]
        assert "extra" in call_kwargs
        assert call_kwargs["extra"]["agent_context"] == "validator"

    def test_extra_attributes_include_tool_name(self) -> None:
        """Log calls include tool_name in extra."""
        tracer = AgentTracer()

        tracer.on_tool_start(
            serialized={"name": "read_lines"},
            input_str="1-50",
            run_id=uuid4(),
        )

        call_kwargs = tracer._logger.info.call_args[1]
        assert call_kwargs["extra"]["tool_name"] == "read_lines"

    def test_extra_attributes_include_run_id(self) -> None:
        """Log calls include run_id in extra."""
        tracer = AgentTracer()

        run_id = uuid4()
        tracer.on_tool_start(
            serialized={"name": "search"},
            input_str="query",
            run_id=run_id,
        )

        call_kwargs = tracer._logger.info.call_args[1]
        assert call_kwargs["extra"]["run_id"] == str(run_id)


class TestLLMCallbacks:
    """Tests for LLM callback methods."""

    @pytest.fixture(autouse=True)
    def mock_logger(self) -> None:
        """Replace the tracer's logger with a mock."""
        with patch("single_doc_generator.tracing._get_otel_logger") as mock_get:
            mock_get.return_value = MagicMock()
            yield

    def test_on_chat_model_start_increments_llm_count(self) -> None:
        """on_chat_model_start increments LLM call count."""
        tracer = AgentTracer()
        assert tracer._llm_count == 0

        tracer.on_chat_model_start(
            serialized={},
            messages=[[]],
            run_id=uuid4(),
        )
        assert tracer._llm_count == 1

        tracer.on_chat_model_start(
            serialized={},
            messages=[[]],
            run_id=uuid4(),
        )
        assert tracer._llm_count == 2

    def test_on_chat_model_start_logs_debug(self) -> None:
        """on_chat_model_start logs at DEBUG level."""
        tracer = AgentTracer(context="validator")

        tracer.on_chat_model_start(
            serialized={},
            messages=[[]],
            run_id=uuid4(),
        )

        tracer._logger.debug.assert_called_once()
        call_args = tracer._logger.debug.call_args
        format_str = call_args[0][0]
        assert "llm_start" in format_str

    def test_on_llm_end_logs_output(self) -> None:
        """on_llm_end logs LLM output at INFO level."""
        tracer = AgentTracer(context="generator")
        tracer._llm_count = 1

        # Create mock LLMResult
        mock_result = MagicMock()
        mock_gen = MagicMock()
        mock_gen.text = "I will search for the answer"
        mock_result.generations = [[mock_gen]]

        tracer.on_llm_end(response=mock_result, run_id=uuid4())

        tracer._logger.info.assert_called_once()
        call_args = tracer._logger.info.call_args
        format_str = call_args[0][0]
        assert "llm_output" in format_str

    def test_on_llm_end_skips_empty_text(self) -> None:
        """on_llm_end does not log empty or whitespace-only output."""
        tracer = AgentTracer()
        tracer._llm_count = 1

        mock_result = MagicMock()
        mock_gen = MagicMock()
        mock_gen.text = "   "  # Whitespace only
        mock_result.generations = [[mock_gen]]

        tracer.on_llm_end(response=mock_result, run_id=uuid4())

        tracer._logger.info.assert_not_called()

    def test_on_llm_end_handles_multiple_generations(self) -> None:
        """on_llm_end logs each non-empty generation."""
        tracer = AgentTracer()
        tracer._llm_count = 1

        mock_result = MagicMock()
        mock_gen1 = MagicMock()
        mock_gen1.text = "First output"
        mock_gen2 = MagicMock()
        mock_gen2.text = ""  # Empty, should be skipped
        mock_gen3 = MagicMock()
        mock_gen3.text = "Third output"
        mock_result.generations = [[mock_gen1, mock_gen2, mock_gen3]]

        tracer.on_llm_end(response=mock_result, run_id=uuid4())

        assert tracer._logger.info.call_count == 2


class TestOtelLoggerConfiguration:
    """Tests for OTLP logger configuration."""

    def test_endpoint_from_env(self) -> None:
        """Endpoint is read from OTEL_EXPORTER_OTLP_ENDPOINT env var."""
        # _OTLP_ENDPOINT is set at module load time from env
        # It should be empty string if not set, or the env value if set
        assert isinstance(_OTLP_ENDPOINT, str)

    def test_max_preview_len_is_reasonable(self) -> None:
        """Max preview length is a reasonable size."""
        assert 100 <= _MAX_PREVIEW_LEN <= 1000
