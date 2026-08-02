import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "experiments" / "sota_cross_axis_panel.json"


def test_cross_axis_panel_is_fixed_distinct_and_balanced() -> None:
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    subjects = panel["subject_target_ids"]

    assert panel["schema_version"] == 1
    assert len(subjects) == len(set(subjects)) == 24
    assert panel["judge_target_id"] not in subjects
    assert any(target.startswith("openai/gpt-3.5") for target in subjects)
    assert any(target.startswith("openai/gpt-4") for target in subjects)
    assert any(target.startswith("openai/gpt-5") for target in subjects)
    assert any(target.startswith("anthropic/claude-haiku") for target in subjects)
    assert any(target.startswith("anthropic/claude-sonnet") for target in subjects)
    assert any(target.startswith("anthropic/claude-opus") for target in subjects)
    assert any(target.startswith("google/gemini-2.5") for target in subjects)
    assert any(target.startswith("google/gemini-3") for target in subjects)
    assert len({target.split("/", 1)[0] for target in subjects}) >= 9


def test_cross_axis_deadline_design_uses_independent_units() -> None:
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    part0 = panel["part0"]
    part2 = panel["part2"]

    assert part0["deadline_limit_per_language"] <= part0["archived_unique_prompts_per_language"]
    assert set(part0["languages"]) == {"english", "chinese", "russian"}
    assert part0["judge_batch_size"] >= 2
    assert part2["independent_trajectories"] >= 10
    assert part2["common_environment_seeds"] is True
    assert part2["private_gain_for_option_b"] > 0
    assert part2["reserve_cost_for_option_b"] > 0
