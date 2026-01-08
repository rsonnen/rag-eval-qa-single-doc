"""Prompt loading utilities for question generation.

Prompts are stored as Jinja2 templates in YAML files under the prompts/
directory. This module provides functions to load and render them.
"""

from pathlib import Path

import yaml
from jinja2 import Template

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Type alias for Jinja2 template variables (strings, lists of strings/ints, etc.)
TemplateVar = str | list[str] | list[int]


def load_prompt(name: str, **variables: TemplateVar) -> str:
    """Load and render a prompt template.

    Args:
        name: Prompt file name without extension (e.g., 'generator').
        **variables: Template variables (strings or lists) for Jinja2.

    Returns:
        Rendered prompt string.

    Raises:
        FileNotFoundError: If prompt file doesn't exist.
    """
    prompt_path = PROMPTS_DIR / f"{name}.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with prompt_path.open() as f:
        data = yaml.safe_load(f)

    template = Template(data["template"])
    return template.render(**variables)
