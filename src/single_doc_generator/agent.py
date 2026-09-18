"""ReAct agent graph for document exploration.

Vision tools return their images inside the tool message itself, so each tool
response directly follows the assistant message that requested it. Gemini-backed
models depend on that ordering to carry their reasoning state (thought
signatures) from one tool call to the next.
"""

from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State for the ReAct agent.

    Attributes:
        messages: Conversation history with add_messages reducer.
    """

    messages: Annotated[list[AnyMessage], add_messages]


def _create_agent_node(
    model: Runnable[Any, AIMessage], system_prompt: str
) -> Callable[[AgentState], dict[str, Any]]:
    """Create the agent node that calls the LLM."""

    def agent_node(state: AgentState) -> dict[str, Any]:
        """Call the model with current messages."""
        messages = state["messages"]

        # Prepend system message if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt), *messages]

        response = model.invoke(messages)
        return {"messages": [response]}

    return agent_node


def _should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Route based on whether agent wants to call tools."""
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"


def create_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
) -> CompiledStateGraph[AgentState, AgentState, AgentState]:
    """Create a ReAct agent.

    Args:
        model: Chat model to use (should be vision-capable).
        tools: Tools available to the agent.
        system_prompt: System prompt for the agent.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    # Bind tools to model
    model_with_tools = model.bind_tools(tools)

    # Build graph
    graph: StateGraph[AgentState] = StateGraph(AgentState)

    # Add nodes - type ignores needed due to LangGraph's complex overload signatures
    # that don't fully support plain callables returning dicts
    graph.add_node("agent", _create_agent_node(model_with_tools, system_prompt))  # type: ignore[call-overload]
    graph.add_node("tools", ToolNode(tools))

    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    return graph.compile()  # type: ignore[return-value]
