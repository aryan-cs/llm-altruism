import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from agents.agent_1 import Agent1, BinaryGameDecision
from experiments.part1 import part_1
from experiments.misc.prompt_loader import load_prompt_config
from experiments.part1.part_1 import (
    _emit_retry_status_line,
    run_part_1,
    run_part_1_until_complete,
)

PART_1_PROMPTS = load_prompt_config("part_1")


def test_query_variant_stops_after_bounded_attempts(monkeypatch) -> None:
    calls = {"count": 0}

    def always_fail(*args, **kwargs):
        del args, kwargs
        calls["count"] += 1
        raise part_1.ResponseParseError("invalid provider response")

    agent = SimpleNamespace(
        id="P1",
        provider="openai",
        model="model",
        query=always_fail,
    )
    variant = SimpleNamespace(
        prompt_text="prompt",
        allowed_actions=("COOPERATE", "DEFECT"),
    )
    monkeypatch.setattr(part_1.time, "sleep", lambda _: None)

    with pytest.raises(part_1.ResponseParseError, match="invalid provider response"):
        part_1._query_variant_until_valid(agent, variant)

    assert calls["count"] == part_1.MAX_AGENT_ATTEMPTS


def _default_matrix() -> dict[str, list[str]]:
    return {
        "games": PART_1_PROMPTS["defaults"]["games"],
        "frames": PART_1_PROMPTS["defaults"]["frames"],
        "domains": PART_1_PROMPTS["defaults"]["domains"],
        "presentations": PART_1_PROMPTS["defaults"]["presentations"],
    }


def _default_prompt_count() -> int:
    scenario_variants_per_domain_game = 4
    defaults = PART_1_PROMPTS["defaults"]
    return (
        len(defaults["games"])
        * len(defaults["frames"])
        * len(defaults["domains"])
        * len(defaults["presentations"])
        * scenario_variants_per_domain_game
    )


def _single_domain_game_prompt_count(*, presentations: int = 2) -> int:
    return 4 * presentations


def test_prompt_order_is_seeded_reproducible_and_cyclically_counterbalanced() -> None:
    variants = [
        part_1.PromptVariant(
            game="game",
            frame="frame",
            domain="domain",
            scenario_variant=f"scenario-{index}",
            presentation="narrative",
            prompt_id=f"prompt-{index}",
            prompt_text=f"Prompt {index}",
            allowed_actions=("A", "B"),
        )
        for index in range(12)
    ]
    first = part_1._order_prompt_variants(
        variants,
        seed=41,
        strategy="counterbalanced",
        counterbalance_index=0,
    )
    repeated = part_1._order_prompt_variants(
        variants,
        seed=41,
        strategy="counterbalanced",
        counterbalance_index=0,
    )
    rotated = part_1._order_prompt_variants(
        variants,
        seed=41,
        strategy="counterbalanced",
        counterbalance_index=1,
    )
    other_seed = part_1._order_prompt_variants(
        variants,
        seed=42,
        strategy="seeded_shuffle",
        counterbalance_index=0,
    )

    assert [item.prompt_id for item in repeated] == [item.prompt_id for item in first]
    assert rotated == first[1:] + first[:1]
    assert [item.prompt_id for item in other_seed] != [item.prompt_id for item in first]
    assert set(first) == set(variants)


class FlushTrackingIO(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def test_emit_retry_status_line_writes_raw_flushed_carriage_update(monkeypatch) -> None:
    stream = FlushTrackingIO()
    monkeypatch.setattr(
        "experiments.part1.part_1.console",
        Console(file=stream, force_terminal=False, color_system=None, width=100),
    )

    _emit_retry_status_line("  [yellow][WARN] retrying[/yellow]")

    assert stream.getvalue() == "\r\x1b[2K  [WARN] retrying"
    assert stream.flush_count == 1


def test_emit_retry_status_line_finalizes_with_cleared_newline(monkeypatch) -> None:
    stream = FlushTrackingIO()
    monkeypatch.setattr(
        "experiments.part1.part_1.console",
        Console(file=stream, force_terminal=False, color_system=None, width=100),
    )

    _emit_retry_status_line("", finalize=True)

    assert stream.getvalue() == "\r\x1b[2K\n"
    assert stream.flush_count == 1


def test_part_1_config_exposes_expected_prompt_matrix_dimensions() -> None:
    defaults = PART_1_PROMPTS["defaults"]

    assert defaults["games"] == ["prisoners_dilemma", "temptation_or_commons"]
    assert defaults["frames"] == [
        "self_direct",
        "advice",
        "observer_evaluation",
        "prediction",
    ]
    assert defaults["domains"] == [
        "crime",
        "sports",
        "workplace",
        "healthcare",
        "education",
        "neighborhood",
    ]
    assert defaults["presentations"] == ["narrative", "structured"]


def test_agent_1_uses_action_and_justification_schema() -> None:
    schema = BinaryGameDecision.model_json_schema()

    assert set(schema["properties"]) == {"action", "justification"}
    assert "reasoning" not in schema["properties"]


def test_agent_1_builds_prompts_for_every_default_matrix_variant() -> None:
    agent = Agent1(id_="P1", provider_="openai", model_="gpt-4.1-mini")
    defaults = _default_matrix()

    for game in defaults["games"]:
        action_labels = tuple(PART_1_PROMPTS["games"][game]["action_descriptions"])
        for frame in defaults["frames"]:
            for domain in defaults["domains"]:
                for presentation in defaults["presentations"]:
                    prompt = agent.build_prompt(game, frame, domain, presentation)

                    assert prompt
                    for action in action_labels:
                        assert action in prompt


def test_agent_1_exposes_multiple_scenario_variants_per_domain_game() -> None:
    agent = Agent1(id_="P1", provider_="openai", model_="gpt-4.1-mini")

    variant_ids = agent.list_scenario_variant_ids("prisoners_dilemma", "crime")

    assert len(variant_ids) == 4
    assert variant_ids[0] == "interrogation"


def test_agent_1_rejects_unknown_matrix_ids() -> None:
    agent = Agent1(id_="P1", provider_="openai", model_="gpt-4.1-mini")

    with pytest.raises(ValueError, match="Unsupported game"):
        agent.build_prompt("unknown", "self_direct", "crime", "narrative")

    with pytest.raises(ValueError, match="Unsupported frame"):
        agent.build_prompt("prisoners_dilemma", "unknown", "crime", "narrative")

    with pytest.raises(ValueError, match="Unsupported domain"):
        agent.build_prompt("prisoners_dilemma", "self_direct", "unknown", "narrative")

    with pytest.raises(ValueError, match="Unsupported presentation"):
        agent.build_prompt("prisoners_dilemma", "self_direct", "crime", "unknown")


def test_agent_1_renders_distinct_narrative_and_structured_presentations() -> None:
    agent = Agent1(id_="P1", provider_="openai", model_="gpt-4.1-mini")

    narrative_prompt = agent.build_prompt(
        "prisoners_dilemma",
        "self_direct",
        "crime",
        "narrative",
    )
    structured_prompt = agent.build_prompt(
        "prisoners_dilemma",
        "self_direct",
        "crime",
        "structured",
    )

    assert "Scenario:\n-" not in narrative_prompt
    assert "Scenario:\n-" in structured_prompt
    assert narrative_prompt != structured_prompt


def test_agent_1_preserves_canonical_actions_across_frames() -> None:
    agent = Agent1(id_="P1", provider_="openai", model_="gpt-4.1-mini")
    actions = tuple(
        PART_1_PROMPTS["games"]["prisoners_dilemma"]["action_descriptions"]
    )

    for frame in PART_1_PROMPTS["defaults"]["frames"]:
        prompt = agent.build_prompt(
            "prisoners_dilemma",
            frame,
            "sports",
            "narrative",
        )
        assert all(action in prompt for action in actions)


def test_agent_1_supports_new_healthcare_domain_prompts() -> None:
    agent = Agent1(id_="P1", provider_="openai", model_="gpt-4.1-mini")

    prompt = agent.build_prompt(
        "temptation_or_commons",
        "advice",
        "healthcare",
        "structured",
    )

    assert "Two clinicians depend on the same limited treatment resources." in prompt
    assert "RESTRAIN" in prompt
    assert "OVERUSE" in prompt


def test_agent_1_supports_explicit_scenario_variant_selection() -> None:
    agent = Agent1(id_="P1", provider_="openai", model_="gpt-4.1-mini")

    prompt = agent.build_prompt(
        "prisoners_dilemma",
        "self_direct",
        "crime",
        "narrative",
        scenario_variant_id="smuggling-bust",
    )

    assert "Two smugglers were caught after the same border run." in prompt


def test_run_part_1_writes_one_row_per_prompt_variant(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part1.part_1.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )

    def fake_query(self, query: str, json_mode: bool = False) -> str:
        del self, json_mode
        if "DEFECT" in query and "COOPERATE" in query:
            action = "DEFECT"
        else:
            action = "OVERUSE"
        return json.dumps({"action": action, "justification": "Brief justification."})

    monkeypatch.setattr("experiments.part1.part_1.Agent1.query", fake_query)

    defaults = _default_matrix()
    csv_path = run_part_1(
        provider="openai",
        model="gpt-4.1-mini",
        games=defaults["games"],
        frames=defaults["frames"],
        domains=defaults["domains"],
        presentations=defaults["presentations"],
    )

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    file_name = Path(csv_path).name
    assert file_name.startswith("part1__openai__gpt-4-1-mini__full__")

    assert len(rows) == _default_prompt_count()
    assert rows[0]["provider"] == "openai"
    assert rows[0]["model"] == "gpt-4.1-mini"
    assert rows[0]["justification"] == "Brief justification."
    assert [int(row["order_position"]) for row in rows] == list(
        range(1, len(rows) + 1)
    )
    assert {row["order_seed"] for row in rows} == {
        str(part_1.DEFAULT_PART_1_ORDER_SEED)
    }
    assert {row["order_strategy"] for row in rows} == {"counterbalanced"}


def test_run_part_1_respects_filters_and_writes_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part1.part_1.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.part1.part_1.Agent1.query",
        lambda self, query, json_mode=False: json.dumps(
            {"action": "DEFECT", "justification": "Use the safer filtered case."}
        ),
    )

    csv_path = run_part_1(
        provider="openai",
        model="gpt-4.1-mini",
        games=["prisoners_dilemma"],
        frames=["advice"],
        domains=["workplace"],
        presentations=["structured"],
    )

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    file_name = Path(csv_path).name
    assert file_name.startswith("part1__openai__gpt-4-1-mini__subset-4prompts__")

    assert len(rows) == _single_domain_game_prompt_count(presentations=1)
    assert {row["game"] for row in rows} == {"prisoners_dilemma"}
    assert {row["frame"] for row in rows} == {"advice"}
    assert {row["domain"] for row in rows} == {"workplace"}
    assert {row["presentation"] for row in rows} == {"structured"}
    assert {row["action"] for row in rows} == {"DEFECT"}
    assert {row["justification"] for row in rows} == {
        "Use the safer filtered case."
    }
    assert [row["order_position"] for row in rows] == ["1", "2", "3", "4"]


def test_run_part_1_limit_truncates_deterministically(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part1.part_1.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )

    def fake_query(self, query: str, json_mode: bool = False) -> str:
        del self, json_mode
        action = "DEFECT" if "DEFECT" in query else "OVERUSE"
        return json.dumps({"action": action, "justification": "Limited run."})

    monkeypatch.setattr("experiments.part1.part_1.Agent1.query", fake_query)

    defaults = _default_matrix()
    csv_path = run_part_1(
        provider="openai",
        model="gpt-4.1-mini",
        games=defaults["games"],
        frames=defaults["frames"],
        domains=defaults["domains"],
        presentations=defaults["presentations"],
        limit=5,
    )

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    file_name = Path(csv_path).name
    assert file_name.startswith("part1__openai__gpt-4-1-mini__smoke-5prompts__")

    assert len({row["prompt_id"] for row in rows}) == 5
    assert [row["order_position"] for row in rows] == ["1", "2", "3", "4", "5"]
    assert {row["order_seed"] for row in rows} == {
        str(part_1.DEFAULT_PART_1_ORDER_SEED)
    }


def test_run_part_1_headless_uses_default_full_matrix_without_prompting(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part1.part_1.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )

    def fail_if_prompted(*args, **kwargs):
        raise AssertionError("Headless run should not prompt for matrix selection.")

    monkeypatch.setattr("experiments.misc.wizard._prompt_for_part_1_dimension", fail_if_prompted)
    monkeypatch.setattr(
        "experiments.part1.part_1.Agent1.query",
        lambda self, query, json_mode=False: json.dumps(
            {"action": "DEFECT" if "DEFECT" in query else "OVERUSE", "justification": "Headless full run."}
        ),
    )

    csv_path = run_part_1(
        provider="openai",
        model="gpt-4.1-mini",
        headless=True,
    )

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == _default_prompt_count()
    assert Path(csv_path).name.startswith("part1__openai__gpt-4-1-mini__full__")


def test_run_part_1_headless_prints_per_model_prompt_progress(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part1.part_1.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.part1.part_1.Agent1.query",
        lambda self, query, json_mode=False: json.dumps(
            {"action": "DEFECT", "justification": "Progress output test."}
        ),
    )

    buffer = io.StringIO()
    test_console = Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
        width=200,
    )
    monkeypatch.setattr("experiments.part1.part_1.console", test_console)

    run_part_1(
        provider="openai",
        model="gpt-4.1-mini",
        games=["prisoners_dilemma"],
        frames=["self_direct"],
        domains=["crime"],
        presentations=["narrative"],
        headless=True,
    )

    output = buffer.getvalue()
    assert "Model openai/gpt-4.1-mini [prompt 1/4] [--------------------] RUNNING" in output
    assert "Model openai/gpt-4.1-mini [prompt 4/4] [###############-----] RUNNING" in output
    assert "[1/4] [#####---------------]" in output
    assert "-> DEFECT" in output


def test_run_part_1_preserves_written_rows_when_interrupted(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part1.part_1.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )

    calls = {"count": 0}

    def fake_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        calls["count"] += 1
        if calls["count"] == 1:
            return json.dumps({"action": "DEFECT", "justification": "First row survives."})
        raise KeyboardInterrupt()

    monkeypatch.setattr("experiments.part1.part_1.Agent1.query", fake_query)

    csv_path = run_part_1(
        provider="openai",
        model="gpt-4.1-mini",
        games=["prisoners_dilemma"],
        frames=["self_direct"],
        domains=["crime"],
        presentations=["narrative", "structured"],
    )

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["order_position"] == "1"
    assert rows[0]["justification"] == "First row survives."

    metadata_files = list((tmp_path / "data" / "raw" / "part_1").glob("*_meta.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["attempt_log"]["total_attempts"] == 2
    attempt_path = Path(metadata["attempt_log"]["path"])
    assert [
        json.loads(line)["outcome"]
        for line in attempt_path.read_text(encoding="utf-8").splitlines()
    ] == ["success", "interrupted"]


def test_run_part_1_resumes_from_partial_csv(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part1.part_1.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )

    first_pass_calls = {"count": 0}

    def interrupted_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        first_pass_calls["count"] += 1
        if first_pass_calls["count"] == 1:
            return json.dumps({"action": "DEFECT", "justification": "Saved before interruption."})
        raise KeyboardInterrupt()

    monkeypatch.setattr("experiments.part1.part_1.Agent1.query", interrupted_query)

    initial_csv_path = run_part_1(
        provider="openai",
        model="gpt-4.1-mini",
        games=["prisoners_dilemma"],
        frames=["self_direct"],
        domains=["crime"],
        presentations=["narrative", "structured"],
    )

    def resumed_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        return json.dumps({"action": "DEFECT", "justification": "Saved after resume."})

    monkeypatch.setattr("experiments.part1.part_1.Agent1.query", resumed_query)

    resumed_csv_path = run_part_1(resume=True)

    assert resumed_csv_path == initial_csv_path

    with Path(resumed_csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == _single_domain_game_prompt_count()
    assert rows[0]["justification"] == "Saved before interruption."
    assert rows[1]["justification"] == "Saved after resume."
    metadata_files = list((tmp_path / "data" / "raw" / "part_1").glob("*_meta.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["completed_rows"] == _single_domain_game_prompt_count()


def test_run_part_1_until_complete_retries_incomplete_runs(
    monkeypatch,
) -> None:
    attempts = {"count": 0}

    def fake_run_part_1(*args, **kwargs):
        del args, kwargs
        attempts["count"] += 1
        if attempts["count"] == 1:
            return "data/raw/part_1/partial.csv"
        return "data/raw/part_1/final.csv"

    def fake_load_rows(path):
        if str(path).endswith("partial.csv"):
            return [{"prompt_id": "one"}]
        return [{"prompt_id": "one"}, {"prompt_id": "two"}]

    monkeypatch.setattr("experiments.part1.part_1.run_part_1", fake_run_part_1)
    monkeypatch.setattr("experiments.part1.part_1._load_part_1_rows", fake_load_rows)
    monkeypatch.setattr(
        "experiments.part1.part_1._metadata_path_for_csv",
        lambda path: Path("data/raw/part_1/partial_meta.json")
        if str(path).endswith("partial.csv")
        else Path("data/raw/part_1/final_meta.json"),
    )
    monkeypatch.setattr(Path, "exists", lambda self: self.name == "partial_meta.json")

    result = run_part_1_until_complete(
        provider="openai",
        model="gpt-4.1-mini",
        games=["prisoners_dilemma"],
        frames=["self_direct"],
        domains=["crime"],
        presentations=["narrative"],
        headless=True,
    )

    assert result == "data/raw/part_1/final.csv"
    assert attempts["count"] == 2


def test_run_part_1_unloads_ollama_model_after_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part1.part_1.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.part1.part_1.Agent1.query",
        lambda self, query, json_mode=False: json.dumps(
            {"action": "DEFECT", "justification": "Complete ollama run."}
        ),
    )

    unloaded_models: list[str] = []
    monkeypatch.setattr(
        "experiments.part1.part_1.unload_ollama_model",
        lambda model: unloaded_models.append(model),
    )

    run_part_1(
        provider="ollama",
        model="gpt-oss:20b",
        games=["prisoners_dilemma"],
        frames=["self_direct"],
        domains=["crime"],
        presentations=["narrative"],
    )

    assert unloaded_models == ["gpt-oss:20b"]


def test_run_part_1_prepares_ollama_model_before_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part1.part_1.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.part1.part_1.Agent1.query",
        lambda self, query, json_mode=False: json.dumps(
            {"action": "DEFECT", "justification": "Prepared cleanly."}
        ),
    )

    cleanup_events: list[str] = []
    monkeypatch.setattr(
        "experiments.part1.part_1.unload_all_ollama_models",
        lambda: cleanup_events.append("stop-all"),
    )
    monkeypatch.setattr(
        "experiments.part1.part_1.delete_other_ollama_models",
        lambda model: cleanup_events.append(f"delete-others:{model}"),
    )
    monkeypatch.setattr(
        "experiments.part1.part_1.unload_ollama_model",
        lambda model: cleanup_events.append(f"unload-current:{model}"),
    )

    run_part_1(
        provider="ollama",
        model="gpt-oss:20b",
        games=["prisoners_dilemma"],
        frames=["self_direct"],
        domains=["crime"],
        presentations=["narrative"],
    )

    assert cleanup_events == [
        "stop-all",
        "delete-others:gpt-oss:20b",
        "unload-current:gpt-oss:20b",
    ]


def test_run_part_1_retries_invalid_responses_until_success(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part1.part_1.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("experiments.part1.part_1.time.sleep", lambda _: None)

    calls = {"count": 0}

    def flaky_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        calls["count"] += 1
        if calls["count"] < 3:
            return json.dumps({"action": "INVALID", "justification": "Malformed output."})
        return json.dumps({"action": "DEFECT", "justification": "Recovered after retries."})

    monkeypatch.setattr("experiments.part1.part_1.Agent1.query", flaky_query)

    csv_path = run_part_1(
        provider="openai",
        model="gpt-4.1-mini",
        games=["prisoners_dilemma"],
        frames=["self_direct"],
        domains=["crime"],
        presentations=["narrative"],
    )

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert calls["count"] == 6
    assert len(rows) == _single_domain_game_prompt_count(presentations=1)
    assert rows[0]["justification"] == "Recovered after retries."
    metadata_files = list((tmp_path / "data" / "raw" / "part_1").glob("*_meta.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["invalid_attempts"] == 2
    assert metadata["retry_attempts"] == 2
    assert metadata["attempt_log"]["total_attempts"] == 6
    assert metadata["attempt_log"]["successful_attempts"] == 4
    attempt_path = Path(metadata["attempt_log"]["path"])
    attempts = [
        json.loads(line)
        for line in attempt_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [attempt["outcome"] for attempt in attempts[:3]] == [
        "invalid_response",
        "invalid_response",
        "success",
    ]
    assert attempts[0]["raw_response"]
    assert attempts[2]["parsed_response"]["action"] == "DEFECT"


def test_run_part_1_pauses_immediately_when_ollama_disconnects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part1.part_1.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.part1.part_1._prepare_ollama_model_for_run",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.part1.part_1.unload_ollama_model",
        lambda model: None,
    )
    monkeypatch.setattr(
        "experiments.part1.part_1.time.sleep",
        lambda seconds: (_ for _ in ()).throw(
            AssertionError("connection errors must not enter retry backoff")
        ),
    )

    calls = {"count": 0}

    def disconnected_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        calls["count"] += 1
        from providers.api_call import OllamaConnectionError

        raise OllamaConnectionError("Could not connect to Ollama at http://localhost:11434.")

    monkeypatch.setattr("experiments.part1.part_1.Agent1.query", disconnected_query)

    csv_path = run_part_1(
        provider="ollama",
        model="gpt-oss:20b",
        games=["prisoners_dilemma"],
        frames=["self_direct"],
        domains=["crime"],
        presentations=["narrative"],
    )

    assert calls["count"] == 1
    metadata_path = Path(csv_path).with_name(f"{Path(csv_path).stem}_meta.json")
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["failure"]["provenance"]["category"] == "transport"
    assert metadata["provider_error_attempts"] == 1
    assert metadata["retry_attempts"] == 0
    attempt_path = Path(metadata["attempt_log"]["path"])
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert attempt["outcome"] == "provider_error"
    assert attempt["raw_response"] is None
