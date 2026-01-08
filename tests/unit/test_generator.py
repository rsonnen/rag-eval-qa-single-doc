"""Unit tests for question generator."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from single_doc_generator.config import CorpusConfig, EvaluationScenario, ModelConfig
from single_doc_generator.generator import (
    _EXHAUSTED_SENTINEL,
    DocumentExhaustedError,
    GeneratedQA,
    GenerationError,
    _extract_viewed_pages,
    generate_question,
)
from single_doc_generator.models import GenerationMode


def make_agent_result_with_tool_response(qa: GeneratedQA) -> dict:
    """Create mock agent result with submit_qa tool response."""
    tool_message = ToolMessage(
        content=qa.model_dump_json(),
        name="submit_qa",
        tool_call_id="test-id",
    )
    return {"messages": [tool_message]}


def make_exhausted_result(reason: str) -> dict:
    """Create mock agent result with report_exhausted tool response."""
    tool_message = ToolMessage(
        content=f"{_EXHAUSTED_SENTINEL}:{reason}",
        name="report_exhausted",
        tool_call_id="test-id",
    )
    return {"messages": [tool_message]}


def make_corpus_config(
    name: str = "Test Corpus",
    corpus_context: str = "Test documents for evaluation",
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


class TestGeneratedQA:
    """Tests for GeneratedQA model."""

    def test_required_fields(self):
        """Test that question, answer, reasoning are required."""
        qa = GeneratedQA(
            question="What is X?",
            answer="X is Y",
            reasoning="Tests understanding of X",
        )
        assert qa.question == "What is X?"
        assert qa.answer == "X is Y"
        assert qa.reasoning == "Tests understanding of X"

    def test_default_content_refs(self):
        """Test that content_refs defaults to empty list."""
        qa = GeneratedQA(
            question="What is X?",
            answer="X is Y",
            reasoning="Tests understanding",
        )
        assert qa.content_refs == []

    def test_with_content_refs(self):
        """Test setting content_refs explicitly."""
        qa = GeneratedQA(
            question="What does Figure 1 show?",
            answer="A diagram of X",
            reasoning="Tests visual understanding",
            content_refs=["Figure 1", "Table 2"],
        )
        assert qa.content_refs == ["Figure 1", "Table 2"]


class TestDocumentExhaustedError:
    """Tests for DocumentExhaustedError exception."""

    def test_basic_exception(self):
        """Test creating exception with just reason."""
        err = DocumentExhaustedError("No more content")
        assert err.reason == "No more content"
        assert err.viewed_pages == []
        assert "No more content" in str(err)

    def test_with_viewed_pages(self):
        """Test creating exception with viewed pages."""
        err = DocumentExhaustedError("Exhausted", viewed_pages=[1, 5, 10])
        assert err.reason == "Exhausted"
        assert err.viewed_pages == [1, 5, 10]


class TestGenerationError:
    """Tests for GenerationError exception."""

    def test_basic_exception(self):
        """Test creating exception with just message."""
        err = GenerationError("Agent failed")
        assert str(err) == "Agent failed"
        assert err.viewed_pages == []

    def test_with_viewed_pages(self):
        """Test creating exception with viewed pages."""
        err = GenerationError("Agent failed", viewed_pages=[3, 7])
        assert str(err) == "Agent failed"
        assert err.viewed_pages == [3, 7]


class TestExtractViewedPages:
    """Tests for _extract_viewed_pages helper function."""

    def test_extracts_view_page_calls(self):
        """Test that view_page tool calls are extracted."""
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "view_page", "args": {"page": 5}, "id": "1"},
                    {"name": "read_lines", "args": {"start": 1}, "id": "2"},
                    {"name": "view_page", "args": {"page": 10}, "id": "3"},
                ],
            ),
        ]
        result = _extract_viewed_pages(messages)
        assert result == [5, 10]

    def test_removes_duplicates(self):
        """Test that duplicate page numbers are removed."""
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "view_page", "args": {"page": 5}, "id": "1"},
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "view_page", "args": {"page": 5}, "id": "2"},
                    {"name": "view_page", "args": {"page": 8}, "id": "3"},
                ],
            ),
        ]
        result = _extract_viewed_pages(messages)
        assert result == [5, 8]

    def test_preserves_order(self):
        """Test that pages are returned in order of first viewing."""
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "view_page", "args": {"page": 3}, "id": "1"},
                    {"name": "view_page", "args": {"page": 1}, "id": "2"},
                    {"name": "view_page", "args": {"page": 2}, "id": "3"},
                ],
            ),
        ]
        result = _extract_viewed_pages(messages)
        assert result == [3, 1, 2]

    def test_empty_messages(self):
        """Test with empty message list."""
        result = _extract_viewed_pages([])
        assert result == []

    def test_no_view_page_calls(self):
        """Test with messages but no view_page calls."""
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_lines", "args": {"start": 1}, "id": "1"},
                ],
            ),
        ]
        result = _extract_viewed_pages(messages)
        assert result == []

    def test_handles_non_ai_messages(self):
        """Test that non-AIMessage types are safely ignored."""
        messages = [
            ToolMessage(content="result", name="test", tool_call_id="1"),
            AIMessage(
                content="",
                tool_calls=[{"name": "view_page", "args": {"page": 5}, "id": "2"}],
            ),
        ]
        result = _extract_viewed_pages(messages)
        assert result == [5]


class TestGenerateQuestion:
    """Tests for generate_question function."""

    @pytest.fixture
    def sample_config(self):
        """Create sample corpus configuration."""
        return make_corpus_config()

    def test_raises_on_missing_file(self, tmp_path, sample_config):
        """Test that FileNotFoundError is raised for missing document."""
        with patch("single_doc_generator.generator.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                generator_model=ModelConfig(name="gpt-5.1", kwargs={})
            )

            with pytest.raises(FileNotFoundError):
                generate_question(
                    document_path=tmp_path / "nonexistent.txt",
                    corpus_config=sample_config,
                    scenario_name="rag_eval",
                )

    def test_raises_on_invalid_scenario(self, shakespeare_file, sample_config):
        """Test that KeyError is raised for unknown scenario."""
        with patch("single_doc_generator.generator.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                generator_model=ModelConfig(name="gpt-5.1", kwargs={})
            )

            with pytest.raises(KeyError, match="not found"):
                generate_question(
                    document_path=shakespeare_file,
                    corpus_config=sample_config,
                    scenario_name="nonexistent_scenario",
                )

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_generates_qa_via_submit_tool(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that generator extracts Q/A from submit_qa tool response."""
        mock_config = MagicMock()
        mock_config.generator_model = ModelConfig(name="gpt-5.1", kwargs={})
        mock_get_config.return_value = mock_config

        mock_llm = MagicMock()
        mock_create_model.return_value = mock_llm

        mock_load_prompt.return_value = "Test prompt"

        expected_qa = GeneratedQA(
            question="Who wrote Hamlet?",
            answer="William Shakespeare",
            reasoning="Tests authorship knowledge",
        )

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            expected_qa
        )

        result, viewed_pages = generate_question(
            document_path=shakespeare_file,
            corpus_config=sample_config,
            scenario_name="rag_eval",
        )

        assert result.question == expected_qa.question
        assert result.answer == expected_qa.answer
        assert result.reasoning == expected_qa.reasoning
        assert isinstance(viewed_pages, list)

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_raises_on_missing_tool_response(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that GenerationError is raised when submit_qa tool not called."""
        mock_get_config.return_value = MagicMock(
            generator_model=ModelConfig(name="gpt-5.1", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = {"messages": []}

        with pytest.raises(GenerationError, match="did not call submit_qa"):
            generate_question(
                document_path=shakespeare_file,
                corpus_config=sample_config,
                scenario_name="rag_eval",
            )

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_generation_error_includes_viewed_pages(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that GenerationError includes viewed pages from agent execution."""
        mock_get_config.return_value = MagicMock(
            generator_model=ModelConfig(name="gpt-5.1", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        # Agent viewed pages but didn't call submit_qa
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "view_page", "args": {"page": 5}, "id": "call1"},
                        {"name": "view_page", "args": {"page": 10}, "id": "call2"},
                    ],
                ),
            ]
        }

        with pytest.raises(GenerationError) as exc_info:
            generate_question(
                document_path=shakespeare_file,
                corpus_config=sample_config,
                scenario_name="rag_eval",
            )

        # Exception should include the viewed pages
        assert exc_info.value.viewed_pages == [5, 10]

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_exhausted_error_includes_viewed_pages(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that DocumentExhaustedError includes viewed pages."""
        mock_get_config.return_value = MagicMock(
            generator_model=ModelConfig(name="gpt-5.1", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "view_page", "args": {"page": 3}, "id": "call1"},
                    ],
                ),
                ToolMessage(
                    content=f"{_EXHAUSTED_SENTINEL}:No more visual content",
                    name="report_exhausted",
                    tool_call_id="call2",
                ),
            ]
        }

        with pytest.raises(DocumentExhaustedError) as exc_info:
            generate_question(
                document_path=shakespeare_file,
                corpus_config=sample_config,
                scenario_name="rag_eval",
            )

        assert exc_info.value.reason == "No more visual content"
        assert exc_info.value.viewed_pages == [3]

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_loads_prompt_with_corpus_config(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
    ):
        """Test that prompt is loaded with corpus config and scenario."""
        mock_get_config.return_value = MagicMock(
            generator_model=ModelConfig(name="gpt-5.1", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            GeneratedQA(question="Q", answer="A", reasoning="R")
        )

        config = make_corpus_config(
            name="Shakespeare Collection",
            corpus_context="Complete works of Shakespeare",
            scenario_name="literature_quiz",
            scenario_description="Questions testing literary knowledge",
        )

        generate_question(
            document_path=shakespeare_file,
            corpus_config=config,
            scenario_name="literature_quiz",
        )

        mock_load_prompt.assert_called_once()
        call_kwargs = mock_load_prompt.call_args.kwargs
        assert call_kwargs["corpus_name"] == "Shakespeare Collection"
        assert call_kwargs["corpus_context"] == "Complete works of Shakespeare"
        assert (
            call_kwargs["scenario_description"]
            == "Questions testing literary knowledge"
        )

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_uses_model_from_config(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that generator model is read from app config."""
        mock_config = MagicMock()
        mock_config.generator_model = ModelConfig(name="claude-opus-4.5", kwargs={})
        mock_get_config.return_value = mock_config

        mock_load_prompt.return_value = "Test prompt"
        mock_create_model.return_value = MagicMock()

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            GeneratedQA(question="Q", answer="A", reasoning="R")
        )

        generate_question(
            document_path=shakespeare_file,
            corpus_config=sample_config,
            scenario_name="rag_eval",
        )

        mock_create_model.assert_called_once_with("claude-opus-4.5", {})

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_passes_model_kwargs_to_create_chat_model(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that model kwargs from config are passed to create_chat_model."""
        mock_config = MagicMock()
        mock_config.generator_model = ModelConfig(
            name="gemini-2.5-flash",
            kwargs={"reasoning_effort": "none"},
        )
        mock_get_config.return_value = mock_config

        mock_load_prompt.return_value = "Test prompt"
        mock_create_model.return_value = MagicMock()

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            GeneratedQA(question="Q", answer="A", reasoning="R")
        )

        generate_question(
            document_path=shakespeare_file,
            corpus_config=sample_config,
            scenario_name="rag_eval",
        )

        mock_create_model.assert_called_once_with(
            "gemini-2.5-flash", {"reasoning_effort": "none"}
        )

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_includes_submit_tool_in_agent(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that submit_qa tool is included in agent tools."""
        mock_get_config.return_value = MagicMock(
            generator_model=ModelConfig(name="gpt-5.1", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            GeneratedQA(question="Q", answer="A", reasoning="R")
        )

        generate_question(
            document_path=shakespeare_file,
            corpus_config=sample_config,
            scenario_name="rag_eval",
        )

        call_kwargs = mock_create_agent.call_args.kwargs
        tools = call_kwargs["tools"]
        tool_names = [t.name for t in tools]
        assert "submit_qa" in tool_names

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_passes_previous_questions_to_prompt(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that previous questions are passed to prompt loader."""
        mock_get_config.return_value = MagicMock(
            generator_model=ModelConfig(name="gpt-5.1", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            GeneratedQA(question="Q", answer="A", reasoning="R")
        )

        previous = ["What is the theme?", "Who is the protagonist?"]
        generate_question(
            document_path=shakespeare_file,
            corpus_config=sample_config,
            scenario_name="rag_eval",
            previous_questions=previous,
        )

        mock_load_prompt.assert_called_once()
        call_kwargs = mock_load_prompt.call_args.kwargs
        assert call_kwargs["previous_questions"] == previous

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_previous_questions_defaults_to_empty(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that previous_questions defaults to empty list."""
        mock_get_config.return_value = MagicMock(
            generator_model=ModelConfig(name="gpt-5.1", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            GeneratedQA(question="Q", answer="A", reasoning="R")
        )

        generate_question(
            document_path=shakespeare_file,
            corpus_config=sample_config,
            scenario_name="rag_eval",
        )

        mock_load_prompt.assert_called_once()
        call_kwargs = mock_load_prompt.call_args.kwargs
        assert call_kwargs["previous_questions"] == []

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_passes_previous_viewed_pages_to_prompt(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that previous_viewed_pages are passed to prompt loader."""
        mock_get_config.return_value = MagicMock(
            generator_model=ModelConfig(name="gpt-5.1", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            GeneratedQA(question="Q", answer="A", reasoning="R")
        )

        previous_pages = [5, 10, 15]
        generate_question(
            document_path=shakespeare_file,
            corpus_config=sample_config,
            scenario_name="rag_eval",
            previous_viewed_pages=previous_pages,
        )

        mock_load_prompt.assert_called_once()
        call_kwargs = mock_load_prompt.call_args.kwargs
        assert call_kwargs["previous_viewed_pages"] == previous_pages

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_previous_viewed_pages_defaults_to_empty(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that previous_viewed_pages defaults to empty list."""
        mock_get_config.return_value = MagicMock(
            generator_model=ModelConfig(name="gpt-5.1", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            GeneratedQA(question="Q", answer="A", reasoning="R")
        )

        generate_question(
            document_path=shakespeare_file,
            corpus_config=sample_config,
            scenario_name="rag_eval",
        )

        mock_load_prompt.assert_called_once()
        call_kwargs = mock_load_prompt.call_args.kwargs
        assert call_kwargs["previous_viewed_pages"] == []

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_raises_document_exhausted(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that DocumentExhaustedError is raised when agent reports exhaustion."""
        mock_get_config.return_value = MagicMock(
            generator_model=ModelConfig(name="gpt-5.1", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_exhausted_result(
            "All suitable content has been covered"
        )

        with pytest.raises(DocumentExhaustedError) as exc_info:
            generate_question(
                document_path=shakespeare_file,
                corpus_config=sample_config,
                scenario_name="rag_eval",
            )

        assert exc_info.value.reason == "All suitable content has been covered"

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_includes_exhausted_tool_in_agent(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that report_exhausted tool is included in agent tools."""
        mock_get_config.return_value = MagicMock(
            generator_model=ModelConfig(name="gpt-5.1", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            GeneratedQA(question="Q", answer="A", reasoning="R")
        )

        generate_question(
            document_path=shakespeare_file,
            corpus_config=sample_config,
            scenario_name="rag_eval",
        )

        call_kwargs = mock_create_agent.call_args.kwargs
        tools = call_kwargs["tools"]
        tool_names = [t.name for t in tools]
        assert "report_exhausted" in tool_names

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_passes_mode_to_prompt(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that mode parameter is passed to prompt loader."""
        mock_get_config.return_value = MagicMock(
            generator_model=ModelConfig(name="gpt-5.1", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            GeneratedQA(question="Q", answer="A", reasoning="R")
        )

        generate_question(
            document_path=shakespeare_file,
            corpus_config=sample_config,
            scenario_name="rag_eval",
            mode=GenerationMode.VISUAL,
        )

        mock_load_prompt.assert_called_once()
        call_kwargs = mock_load_prompt.call_args.kwargs
        assert call_kwargs["mode"] == "visual"

    @patch("single_doc_generator.generator.create_agent")
    @patch("single_doc_generator.generator.create_chat_model")
    @patch("single_doc_generator.generator.get_config")
    @patch("single_doc_generator.generator.load_prompt")
    def test_mode_defaults_to_textual(
        self,
        mock_load_prompt,
        mock_get_config,
        mock_create_model,
        mock_create_agent,
        shakespeare_file,
        sample_config,
    ):
        """Test that mode defaults to textual when not specified."""
        mock_get_config.return_value = MagicMock(
            generator_model=ModelConfig(name="gpt-5.1", kwargs={})
        )
        mock_create_model.return_value = MagicMock()
        mock_load_prompt.return_value = "Test prompt"

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = make_agent_result_with_tool_response(
            GeneratedQA(question="Q", answer="A", reasoning="R")
        )

        generate_question(
            document_path=shakespeare_file,
            corpus_config=sample_config,
            scenario_name="rag_eval",
        )

        mock_load_prompt.assert_called_once()
        call_kwargs = mock_load_prompt.call_args.kwargs
        assert call_kwargs["mode"] == "textual"
