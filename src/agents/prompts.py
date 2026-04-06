"""
Prompt loading and composition utilities for agents.

Handles loading prompt templates from the prompts/ directory and combining them
into composite system messages.
"""

import re
from pathlib import Path
from typing import Optional


def _get_project_root() -> Path:
    """
    Find the project root by looking for pyproject.toml.

    Returns:
        Path object pointing to the project root directory

    Raises:
        RuntimeError: If pyproject.toml cannot be found
    """
    current = Path(__file__).resolve()

    # Walk up the directory tree looking for pyproject.toml
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError(
        "Could not find project root (pyproject.toml). "
        "Looked in: " + str(current.parents)
    )


def load_prompt(filepath: str) -> str:
    """
    Load a prompt from a relative filepath in the prompts/ folder.

    Args:
        filepath: Path relative to prompts/ folder (e.g., "system/minimal.txt")

    Returns:
        The prompt text, stripped of leading/trailing whitespace

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    project_root = _get_project_root()
    normalized = filepath.removeprefix("prompts/")
    full_path = project_root / "prompts" / normalized

    if not full_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {full_path}")

    return full_path.read_text(encoding="utf-8").strip()


def load_prompt_by_name(category: str, name: str) -> str:
    """
    Load a prompt by category and name.

    Args:
        category: The category (e.g., "system", "framing", "persona")
        name: The prompt name without extension (e.g., "minimal")

    Returns:
        The prompt text, stripped of whitespace

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    return load_prompt(f"{category}/{name}.txt")


def list_prompts(category: str) -> list[str]:
    """
    List all available prompt names in a category.

    Args:
        category: The category to search (e.g., "system", "framing", "persona")

    Returns:
        List of prompt names without extensions, sorted alphabetically

    Raises:
        FileNotFoundError: If the category directory doesn't exist
    """
    project_root = _get_project_root()
    category_dir = project_root / "prompts" / category

    if not category_dir.exists():
        raise FileNotFoundError(f"Prompt category not found: {category_dir}")

    # Get all .txt files and extract names
    names = sorted([f.stem for f in category_dir.glob("*.txt")])
    return names


def build_composite_prompt(
    system: Optional[str] = None,
    framing: Optional[str] = None,
    persona: Optional[str] = None,
) -> str:
    """
    Build a composite prompt from system, framing, and persona components.

    Components are combined in order: system, framing, persona.
    Each component is separated by a blank line.
    None components are skipped.

    Args:
        system: System prompt text or None
        framing: Framing prompt text or None
        persona: Persona prompt text or None

    Returns:
        The combined prompt text
    """
    parts = []

    if system:
        parts.append(system)
    if framing:
        parts.append(framing)
    if persona:
        parts.append(persona)

    return "\n\n".join(parts)


def render_prompt_template(filepath: str, **kwargs: object) -> str:
    """
    Load a prompt template and replace ``{{name}}`` placeholders.

    Args:
        filepath: Path relative to the prompts/ folder
        **kwargs: Values to interpolate into the template

    Returns:
        Rendered prompt text

    Raises:
        KeyError: If the template contains placeholders without a provided value
    """
    template = load_prompt(filepath)

    for key, value in kwargs.items():
        template = template.replace(f"{{{{{key}}}}}", str(value))

    unresolved = sorted(set(re.findall(r"{{\s*([a-zA-Z0-9_]+)\s*}}", template)))
    if unresolved:
        missing = ", ".join(unresolved)
        raise KeyError(f"Missing template values for {filepath}: {missing}")

    return template


# Convenience functions for loading by name and composing
def load_and_compose_by_name(
    system_name: Optional[str] = None,
    framing_name: Optional[str] = None,
    persona_name: Optional[str] = None,
) -> str:
    """
    Load prompts by name from all categories and compose them.

    Args:
        system_name: System prompt name (e.g., "minimal"), or None
        framing_name: Framing prompt name, or None
        persona_name: Persona prompt name, or None

    Returns:
        The composite prompt text

    Raises:
        FileNotFoundError: If any specified file doesn't exist
    """
    system_text = load_prompt_by_name("system", system_name) if system_name else None
    framing_text = load_prompt_by_name("framing", framing_name) if framing_name else None
    persona_text = load_prompt_by_name("persona", persona_name) if persona_name else None

    return build_composite_prompt(system_text, framing_text, persona_text)
