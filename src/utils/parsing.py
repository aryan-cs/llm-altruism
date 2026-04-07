"""
Response parsing utilities for extracting structured information from LLM outputs.

Handles JSON extraction, action parsing, integer parsing, and reasoning extraction.
"""

import json
import re
from typing import Optional


def parse_json_response(text: str) -> Optional[dict]:
    """
    Extract JSON from an LLM response.

    Handles markdown code blocks, nested JSON, and partial JSON responses.
    Tries multiple strategies to extract valid JSON.

    Args:
        text: The LLM response text

    Returns:
        Parsed JSON as dict, or None if no valid JSON found
    """
    if not text:
        return None

    # Strategy 1: Try direct JSON parsing
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find first { and last } and try to parse
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        try:
            potential_json = text[first_brace : last_brace + 1]
            return json.loads(potential_json)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Look for JSON array
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket >= 0 and last_bracket > first_bracket:
        try:
            potential_json = text[first_bracket : last_bracket + 1]
            parsed = json.loads(potential_json)
            # If it's a list, wrap it in a dict
            if isinstance(parsed, list):
                return {"items": parsed}
            return parsed
        except json.JSONDecodeError:
            pass

    return None


def parse_action_from_text(text: str, valid_actions: list[str]) -> Optional[str]:
    """
    Extract an action from text using fuzzy matching against valid actions.

    Tries exact match first, then case-insensitive, then substring matching.

    Args:
        text: The text to search
        valid_actions: List of valid action names

    Returns:
        Matched action name, or None if no clear match found
    """
    if not text or not valid_actions:
        return None

    text = text.strip()

    # Strategy 1: Exact match (case-sensitive)
    if text in valid_actions:
        return text

    # Strategy 2: Case-insensitive exact match
    text_lower = text.lower()
    for action in valid_actions:
        if action.lower() == text_lower:
            return action

    # Strategy 3: Look for action name in text (word boundary)
    for action in valid_actions:
        # Use word boundaries to find the action as a word
        pattern = r"\b" + re.escape(action.lower()) + r"\b"
        if re.search(pattern, text_lower):
            return action

    # Strategy 4: Substring match (if text contains action as substring)
    for action in valid_actions:
        if action.lower() in text_lower:
            return action

    # Strategy 5: Levenshtein-like fuzzy match (very simple)
    # If we have only one action left, return it as fallback
    if len(valid_actions) == 1:
        return valid_actions[0]

    return None


def parse_integer_from_text(text: str, min_val: int, max_val: int) -> Optional[int]:
    """
    Extract an integer from text within a specified range.

    Searches for integers in the text and returns the first one within range.

    Args:
        text: The text to search
        min_val: Minimum valid value
        max_val: Maximum valid value

    Returns:
        Parsed integer within range, or None if none found
    """
    if not text:
        return None

    # Find all integers in the text
    integers = re.findall(r"-?\d+", text)

    for int_str in integers:
        value = int(int_str)
        if min_val <= value <= max_val:
            return value

    return None


def extract_reasoning(text: str) -> str:
    """
    Extract reasoning/explanation from a response.

    Looks for common reasoning markers and extracts the relevant portion.

    Args:
        text: The LLM response text

    Returns:
        The extracted reasoning text, or original text if no markers found
    """
    if not text:
        return ""

    parsed = parse_json_response(text)
    if isinstance(parsed, dict):
        for key in ("reasoning", "explanation", "analysis", "thinking", "reasoning_content"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in parsed.values():
            if isinstance(value, dict):
                for key in ("reasoning", "explanation", "analysis", "thinking", "reasoning_content"):
                    nested = value.get(key)
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()

    # Common reasoning markers
    reasoning_markers = [
        r"(?:^|[\n\r])\s*(?:reasoning|thinking|analysis|explanation)\s*:?\s*(.*?)(?=(?:[\n\r]+\s*(?:action|my action)\s*:?)|$)",
    ]

    for marker in reasoning_markers:
        match = re.search(marker, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no markers found, try to get text before "action"
    action_split = re.search(r"\n\s*(?:action|my action|i (?:will|should))", text, re.IGNORECASE)
    if action_split:
        return text[: action_split.start()].strip()

    # Return first paragraph or first 500 chars as fallback
    lines = text.strip().split("\n")
    if lines:
        # Take up to the first blank line or first few lines
        reasoning_lines = []
        for line in lines:
            if not line.strip():
                break
            reasoning_lines.append(line)
        if reasoning_lines:
            return "\n".join(reasoning_lines)

    return text


def parse_bool_from_text(text: str) -> Optional[bool]:
    """
    Extract a boolean value from text.

    Looks for common yes/no, true/false indicators.

    Args:
        text: The text to search

    Returns:
        True, False, or None if no clear indication found
    """
    if not text:
        return None

    text_lower = text.lower().strip()

    # Positive indicators
    positive = [
        "yes",
        "true",
        "agree",
        "ok",
        "okay",
        "accept",
        "cooperate",
        "cooperate",
        "collaborate",
    ]
    for word in positive:
        if word in text_lower:
            return True

    # Negative indicators
    negative = [
        "no",
        "false",
        "disagree",
        "decline",
        "reject",
        "defect",
        "refuse",
    ]
    for word in negative:
        if word in text_lower:
            return False

    return None


def parse_multiple_integers(text: str, count: int) -> Optional[list[int]]:
    """
    Extract multiple integers from text.

    Args:
        text: The text to search
        count: How many integers to extract

    Returns:
        List of integers found, or None if fewer than requested found
    """
    if not text:
        return None

    integers = re.findall(r"-?\d+", text)
    if len(integers) >= count:
        return [int(i) for i in integers[:count]]

    return None
