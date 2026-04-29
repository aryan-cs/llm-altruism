from pathlib import Path

from agents.agent_1 import Agent1
from agents.agent_2 import Agent2
from experiments.misc import prompt_loader
from experiments.misc.prompt_loader import (
    load_part_0_raw_prompts,
    load_prompt_config,
    load_prompts_from_csv_dir,
)


def test_load_prompt_config_returns_part_metadata() -> None:
    config = load_prompt_config("part_1")

    assert config["experiment_name"] == "Part 1: Prompt-Matrix Consistency"
    assert "games" in config
    assert "frames" in config
    assert "domains" in config
    assert "presentations" in config


def test_agent_1_prompts_render_from_json_templates() -> None:
    agent = Agent1(id_="A", provider_="openai", model_="gpt-4.1-mini")

    narrative_prompt = agent.build_prompt(
        "prisoners_dilemma",
        "self_direct",
        "crime",
        "narrative",
        scenario_variant_id="interrogation",
    )
    structured_prompt = agent.build_prompt(
        "temptation_or_commons",
        "advice",
        "sports",
        "structured",
    )

    assert "Two suspects" in narrative_prompt
    assert "COOPERATE" in narrative_prompt
    assert "DEFECT" in narrative_prompt
    assert "Respond in JSON" in narrative_prompt
    assert "Scenario:\n-" in structured_prompt
    assert "RESTRAIN" in structured_prompt
    assert "OVERUSE" in structured_prompt


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
    assert "12 active participants" in prompt
    assert "40 of 120" in prompt
    assert "6 participants selected OPTION_B" in prompt
    assert "OPTION_A" in prompt
    assert "OPTION_B" in prompt
    assert "exactly one JSON object" in prompt
    assert "action" in prompt
    assert "reasoning" in prompt


def test_agent_2_schema_restricts_action_to_neutral_options() -> None:
    agent = Agent2(id_="society_1", provider_="openai", model_="gpt-4.1-mini")

    schema = agent.json_schema.model_json_schema()

    assert set(schema["properties"]["action"]["enum"]) == {"OPTION_A", "OPTION_B"}


def test_agent_2_prompt_uses_neutral_option_labels() -> None:
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
    combined_prompt_text = f"{agent.system_prompt}\n{prompt}".lower()

    loaded_terms = [
        "restrain",
        "overuse",
        "selfish",
        "happier",
        "sustainable",
        "benefit",
        "collapse",
        "fair share",
        "long-term social",
        "community",
        "everyone else",
        "permanently",
    ]
    for term in loaded_terms:
        assert term not in combined_prompt_text


def test_load_experiment_json_supports_comment_lines(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "commented.json"
    config_path.write_text(
        '{\n'
        '  "value": 1,\n'
        '  // "ignored": 2,\n'
        '  "nested": {\n'
        '    # "ignored": 3,\n'
        '    "name": "ok",\n'
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    prompt_loader.load_experiment_json.cache_clear()
    monkeypatch.setattr(prompt_loader, "PROMPT_DIR", tmp_path)

    config = prompt_loader.load_experiment_json("commented.json")

    assert config == {"value": 1, "nested": {"name": "ok"}}


def test_load_prompts_from_csv_dir_reads_all_prompt_rows(tmp_path: Path) -> None:
    csv_dir = tmp_path / "prompts"
    csv_dir.mkdir()
    (csv_dir / "a.csv").write_text(
        "prompt\nfirst prompt\nsecond prompt\n",
        encoding="utf-8",
    )
    (csv_dir / "b.csv").write_text(
        "prompt\nthird prompt\n",
        encoding="utf-8",
    )
    (csv_dir / "notes.txt").write_text("ignore me\n", encoding="utf-8")

    prompts = load_prompts_from_csv_dir(csv_dir)

    assert prompts == ["first prompt", "second prompt", "third prompt"]


def test_load_prompts_from_csv_dir_requires_prompt_column(tmp_path: Path) -> None:
    csv_dir = tmp_path / "prompts"
    csv_dir.mkdir()
    (csv_dir / "bad.csv").write_text(
        "text\nnot a prompt\n",
        encoding="utf-8",
    )

    try:
        load_prompts_from_csv_dir(csv_dir)
    except ValueError as exc:
        assert "prompt' column" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing prompt column.")


def test_load_part_0_raw_prompts_reads_from_part_0_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    part_0_dir = raw_root / "part_0" / "prompts"
    part_0_dir.mkdir(parents=True)
    (part_0_dir / "dataset.csv").write_text(
        "prompt\nalignment prompt\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(prompt_loader, "RAW_DATA_DIR", raw_root)

    prompts = load_part_0_raw_prompts()

    assert prompts == ["alignment prompt"]
