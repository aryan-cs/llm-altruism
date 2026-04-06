"""Tests for response parsing helpers."""

from src.utils.parsing import extract_reasoning, parse_json_response


def test_parse_json_response_extracts_embedded_json_block():
    text = 'Here is my answer:\n```json\n{"action":"cooperate","reasoning":"trust first"}\n```'
    parsed = parse_json_response(text)
    assert parsed == {"action": "cooperate", "reasoning": "trust first"}


def test_extract_reasoning_prefers_json_reasoning_field():
    text = '{"action":"defect","reasoning":"I want to avoid exploitation."}'
    assert extract_reasoning(text) == "I want to avoid exploitation."


def test_extract_reasoning_uses_nested_json_reasoning_field():
    text = '{"response":{"action":"cooperate","analysis":"Mutual cooperation is safer long term."}}'
    assert extract_reasoning(text) == "Mutual cooperation is safer long term."


def test_extract_reasoning_falls_back_to_marker_text():
    text = "Reasoning: I will cooperate first to test reciprocity.\nAction: cooperate"
    assert extract_reasoning(text) == "I will cooperate first to test reciprocity."
