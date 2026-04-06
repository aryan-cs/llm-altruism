"""
Agent module for game-theoretic simulations.

Provides Agent class with configurable system prompts, framings, personas,
and memory management for tracking interactions.
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
