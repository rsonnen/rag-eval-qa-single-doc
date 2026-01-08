"""Custom LangGraph agent with vision support.

This module builds a ReAct agent that properly handles vision tools by deferring
image injection until after all tool responses complete. This is necessary because
OpenAI's API only allows images in user messages, not tool messages, and requires
all tool responses to immediately follow the assistant message with tool_calls.

The pattern:
1. Agent calls tools (possibly in parallel)
2. Tools execute, view_page stores image data in state via Command
3. After ALL tool responses, image_injector node runs if there are pending images
4. Image injector adds a single HumanMessage with all images
5. Agent continues with images visible
"""

import logging
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class PendingImage(BaseModel):
    """Image data waiting to be injected into the conversation."""

    page: int
    total_pages: int
    image_base64: str


def _add_images(
    existing: list[PendingImage] | None, new: list[PendingImage] | None
) -> list[PendingImage]:
    """Reducer for pending images.

    Supports both accumulation and clearing:
    - new=None: No change, return existing
    - new=[]: Clear all pending images
    - new=[items]: Append items to existing
    """
    if new is None:
        # No update provided, keep existing
        return existing or []
    if len(new) == 0:
        # Explicit empty list clears pending images
        return []
    # Non-empty list: accumulate
    existing = existing or []
    return existing + new


class AgentState(TypedDict):
    """State for ReAct agent with deferred image injection.

    Attributes:
        messages: Conversation history with add_messages reducer.
        pending_images: Images to inject after tool execution completes.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    pending_images: Annotated[list[PendingImage], _add_images]


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


def _create_image_injector() -> Callable[[AgentState], dict[str, Any]]:
    """Create node that injects pending images as HumanMessage."""

    def image_injector(state: AgentState) -> dict[str, Any]:
        """Inject all pending images as a single HumanMessage."""
        pending = state.get("pending_images", [])
        if not pending:
            return {"pending_images": []}

        logger.debug("Injecting %d pending images", len(pending))

        # Build multimodal content with all images
        content: list[dict[str, Any]] = []
        for img in pending:
            content.append(
                {
                    "type": "text",
                    "text": f"Here is page {img.page} of {img.total_pages}:",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img.image_base64}"},
                }
            )

        # Cast to expected type - the dict structure is correct for multimodal content
        multimodal_content = cast(list[str | dict[str, Any]], content)
        return {
            "messages": [HumanMessage(content=multimodal_content)],
            "pending_images": [],  # Clear after injection
        }

    return image_injector


def _should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Route based on whether agent wants to call tools."""
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"


def _route_after_tools(state: AgentState) -> Literal["image_injector", "agent"]:
    """Route to image injector if there are pending images."""
    if state.get("pending_images"):
        return "image_injector"
    return "agent"


def create_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
) -> CompiledStateGraph[AgentState, AgentState, AgentState]:
    """Create a ReAct agent with deferred image injection.

    This builds a custom graph that defers image injection until after all tool
    responses complete, avoiding OpenAI's message ordering constraints.

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
    graph.add_node("image_injector", _create_image_injector())  # type: ignore[call-overload]

    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, ["tools", END])
    graph.add_conditional_edges(
        "tools", _route_after_tools, ["image_injector", "agent"]
    )
    graph.add_edge("image_injector", "agent")

    return graph.compile()  # type: ignore[return-value]
