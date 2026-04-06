"""Tests for agents, prompt composition, and memory behavior."""

from __future__ import annotations

from pathlib import Path

from src.agents.base import Agent
from src.agents.memory import Memory, MemoryEntry, MemoryMode
from src.agents.prompts import load_prompt, render_prompt_template


def test_agent_builds_composite_system_message():
    agent = Agent(
        agent_id="a",
        model="gpt-4o-mini",
        provider_name="openai",
        system_prompt="system",
        framing="framing",
        persona="persona",
    )
    assert agent.build_system_message() == "system\n\nframing\n\npersona"


def test_build_messages_respects_hidden_payoffs():
    memory = Memory(mode=MemoryMode.FULL)
    memory.add(MemoryEntry(round_num=1, action="cooperate", payoff=3, opponent_action="defect"))
    agent = Agent(
        agent_id="a",
        model="gpt-4o-mini",
        provider_name="openai",
        system_prompt="system",
        memory=memory,
    )

    with_payoffs = agent.build_messages("next move", include_payoffs=True)
    without_payoffs = agent.build_messages("next move", include_payoffs=False)

    assert "you earned 3" in with_payoffs[1]["content"]
    assert "you earned 3" not in without_payoffs[1]["content"]


def test_memory_summary_can_hide_payoffs():
    memory = Memory(mode=MemoryMode.SUMMARIZED)
    memory.add(MemoryEntry(round_num=1, action="cooperate", payoff=5, opponent_action="cooperate"))
    assert "payoff" in memory.get_summary(include_payoffs=True)
    assert "payoff" not in memory.get_summary(include_payoffs=False)


def test_prompt_loader_accepts_prompts_prefix():
    text = load_prompt("prompts/system/minimal.txt")
    assert text


def test_prompt_template_renderer_replaces_placeholders():
    rendered = render_prompt_template(
        "games/conversation/description.txt",
        owner="Player A",
        other="Player B",
        resource_name="water",
    )
    assert "Player A has a valuable resource: water." in rendered


def test_behavior_prompt_templates_live_under_prompts_folder():
    expected = [
        "prompts/games/prisoners_dilemma/round.txt",
        "prompts/games/ultimatum/proposer_round.txt",
        "prompts/games/conversation/round.txt",
        "prompts/simulation/society_decision.txt",
        "prompts/simulation/society_decision_with_reputation.txt",
    ]
    for relative_path in expected:
        assert Path(relative_path).exists(), relative_path
