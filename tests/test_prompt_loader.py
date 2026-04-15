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
    part_0_dir = raw_root / "part_0"
    part_0_dir.mkdir(parents=True)
    (part_0_dir / "dataset.csv").write_text(
        "prompt\nalignment prompt\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(prompt_loader, "RAW_DATA_DIR", raw_root)

    prompts = load_part_0_raw_prompts()

    assert prompts == ["alignment prompt"]
