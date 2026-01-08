"""Agent tracing via OpenTelemetry OTLP logging.

Sends agent events to an OTLP HTTP endpoint for observability.
Logs LLM reasoning and tool calls as they occur.
"""

import atexit
import logging
import os
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

# OTLP endpoint - must be set via environment variable
_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

# Maximum length for input/output previews in logs
_MAX_PREVIEW_LEN = 300

# Module-level logger provider (initialized once)
_logger_provider: LoggerProvider | None = None
_otel_handler: LoggingHandler | None = None


def _get_otel_logger() -> logging.Logger:
    """Get logger configured with OTLP exporter.

    Initializes the OpenTelemetry logging infrastructure on first call.
    Subsequent calls return the same logger instance.

    If OTEL_EXPORTER_OTLP_ENDPOINT is not set, returns a standard logger
    with no OTLP handler (logs go nowhere unless otherwise configured).
    """
    global _logger_provider, _otel_handler

    logger = logging.getLogger("single_doc_generator.agent_trace")

    # Skip OTLP setup if endpoint not configured
    if not _OTLP_ENDPOINT:
        return logger

    if _logger_provider is None:
        resource = Resource.create({"service.name": "single-doc-generator"})
        _logger_provider = LoggerProvider(resource=resource)
        set_logger_provider(_logger_provider)

        exporter = OTLPLogExporter(endpoint=f"{_OTLP_ENDPOINT}/v1/logs")
        _logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

        _otel_handler = LoggingHandler(
            level=logging.DEBUG, logger_provider=_logger_provider
        )

        # Register shutdown to flush logs on exit
        atexit.register(_shutdown_otel)

    if _otel_handler is not None and _otel_handler not in logger.handlers:
        logger.addHandler(_otel_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # Don't send to root logger / console

    return logger


def _shutdown_otel() -> None:
    """Flush and shutdown the OTLP exporter."""
    global _logger_provider
    if _logger_provider is not None:
        # OTel SDK has incomplete type stubs for shutdown()
        _logger_provider.shutdown()  # type: ignore[no-untyped-call]
        _logger_provider = None


class AgentTracer(BaseCallbackHandler):
    r"""Logs agent LLM and tool events to OTLP endpoint.

    Captures:
    - LLM reasoning (model output before/after tool calls)
    - Tool invocations with arguments
    - Tool results with tool name
    - Tool errors

    Example:
        tracer = AgentTracer(context="validator")
        result = agent.invoke({...}, {"callbacks": [tracer]})
    """

    def __init__(self, context: str = "agent") -> None:
        """Initialize tracer with context label.

        Args:
            context: Label for this agent invocation (e.g., "generator", "validator").
        """
        super().__init__()
        self._context = context
        self._tool_count = 0
        self._llm_count = 0
        # Map run_id to (tool_number, tool_name) for correlating results
        self._tool_runs: dict[UUID, tuple[int, str]] = {}
        self._logger = _get_otel_logger()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log when LLM starts processing."""
        self._llm_count += 1
        self._logger.debug(
            "[%s] llm_start #%d",
            self._context,
            self._llm_count,
            extra={
                "agent_context": self._context,
                "llm_call_number": self._llm_count,
                "run_id": str(run_id),
            },
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Log LLM output including reasoning and tool call decisions."""
        # Extract text from generations
        for gen_list in response.generations:
            for gen in gen_list:
                text = gen.text if hasattr(gen, "text") else str(gen)
                if text.strip():
                    preview = self._truncate(text)
                    self._logger.info(
                        "[%s] llm_output #%d: %s",
                        self._context,
                        self._llm_count,
                        preview,
                        extra={
                            "agent_context": self._context,
                            "llm_call_number": self._llm_count,
                            "run_id": str(run_id),
                        },
                    )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Log tool invocation."""
        self._tool_count += 1
        tool_name = serialized.get("name", "unknown")
        # Store for correlation in on_tool_end
        self._tool_runs[run_id] = (self._tool_count, tool_name)

        preview = self._truncate(input_str)
        self._logger.info(
            "[%s] tool_call #%d: %s(%s)",
            self._context,
            self._tool_count,
            tool_name,
            preview,
            extra={
                "agent_context": self._context,
                "tool_name": tool_name,
                "tool_call_number": self._tool_count,
                "run_id": str(run_id),
            },
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Log tool result with tool name."""
        tool_num, tool_name = self._tool_runs.get(run_id, (0, "unknown"))
        output_str = str(output)
        preview = self._truncate(output_str)
        self._logger.info(
            "[%s] tool_result #%d %s: %s",
            self._context,
            tool_num,
            tool_name,
            preview,
            extra={
                "agent_context": self._context,
                "tool_name": tool_name,
                "tool_call_number": tool_num,
                "run_id": str(run_id),
            },
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Log tool error with tool name."""
        tool_num, tool_name = self._tool_runs.get(run_id, (0, "unknown"))
        self._logger.error(
            "[%s] tool_error #%d %s: %s",
            self._context,
            tool_num,
            tool_name,
            str(error),
            extra={
                "agent_context": self._context,
                "tool_name": tool_name,
                "tool_call_number": tool_num,
                "run_id": str(run_id),
                "error": str(error),
            },
        )

    @property
    def tool_count(self) -> int:
        """Number of tool calls made."""
        return self._tool_count

    @staticmethod
    def _truncate(text: str) -> str:
        """Truncate text to preview length."""
        text = text.replace("\n", " ").strip()
        if len(text) > _MAX_PREVIEW_LEN:
            return text[:_MAX_PREVIEW_LEN] + "..."
        return text
