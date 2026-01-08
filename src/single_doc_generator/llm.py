"""Shared LLM factory for creating chat model instances.

All models are accessed via LiteLLM proxy using OpenAI-compatible API.
Configuration is read from environment variables.
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()


def create_chat_model(
    model_name: str,
    model_kwargs: dict[str, Any] | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance for the specified model.

    All models are accessed via LiteLLM proxy. The proxy URL and API key
    are read from environment variables.

    Args:
        model_name: The model identifier (e.g., gpt-5.1, claude-sonnet-4.5).
        model_kwargs: Additional model parameters passed to LiteLLM.

    Returns:
        Configured ChatOpenAI instance.

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    base_url = os.environ.get("OPENAI_BASE_URL")

    return ChatOpenAI(
        model=model_name,
        api_key=SecretStr(api_key),
        base_url=base_url,
        max_retries=5,
        **(model_kwargs or {}),
    )
