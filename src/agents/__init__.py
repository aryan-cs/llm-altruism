"""
Agent module for repeated-game diagnostics and society simulations.

Provides the Agent class plus prompt and memory utilities for the repo's
precursor micro-level tasks and its main artificial-society environments.
"""

from .base import Agent
from .memory import Memory, MemoryEntry, MemoryMode
from .prompts import (
    build_composite_prompt,
    list_prompts,
    load_and_compose_by_name,
    load_prompt,
    load_prompt_by_name,
    render_prompt_template,
)

__all__ = [
    "Agent",
    "Memory",
    "MemoryEntry",
    "MemoryMode",
    "build_composite_prompt",
    "list_prompts",
    "load_and_compose_by_name",
    "load_prompt",
    "load_prompt_by_name",
    "render_prompt_template",
]
