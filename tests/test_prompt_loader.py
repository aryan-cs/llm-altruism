from agents.agent_1 import Agent1
from agents.agent_2 import Agent2
from experiments.prompt_loader import load_prompt_config


def test_load_prompt_config_returns_part_metadata() -> None:
    config = load_prompt_config("part_1")

    assert config["experiment_name"] == "Part 1: Basic Interactions"
    assert "direct" in config["agent"]["scenarios"]
    assert "indirect" in config["agent"]["scenarios"]


def test_agent_1_prompts_render_from_json_templates() -> None:
    agent = Agent1(id_="A", provider_="openai", model_="gpt-4.1-mini")

    direct_prompt = agent.build_prisoners_dilemma_prompt(
        years_if_only_snitch=9,
        years_if_both_silent=2,
        years_if_both_snitch=4,
    )
    indirect_prompt = agent.build_indirect_competition_prompt(
        advantage_points=7,
        risk_points=3,
    )

    assert "Agent A" in direct_prompt
    assert "9 years" in direct_prompt
    assert "2 year(s)" in direct_prompt
    assert "4 years" in direct_prompt
    assert "7 points" in indirect_prompt
    assert "3 points" in indirect_prompt


def test_agent_2_prompt_renders_context_from_json_templates() -> None:
    agent = Agent2(id_="society_1", provider_="openai", model_="gpt-4.1-mini")

    prompt = agent.build_commons_prompt(
        resource="water",
        selfish_gain=2,
        depletion_units=3,
        community_benefit=5,
        day=4,
        living_agents=12,
        resource_units=40,
        resource_capacity=120,
        previous_overuse_count=6,
    )

    assert "day 4" in prompt
    assert "12 living agents" in prompt
    assert "40 of 120" in prompt
    assert "6 agents chose to OVERUSE" in prompt
