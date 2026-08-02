import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from experiments.part2 import part_2
from experiments.part2.part_2 import (
    PROMPT_CONFIG_HASH,
    _emit_retry_status_line,
    _matching_part_2_metadata_path,
    _parse_agent_response,
    run_part_2,
    run_part_2_until_complete,
)
from experiments.misc.wizard import SocietyConfig
from experiments.misc.run_metadata import metadata_payload_sha256
from providers.api_call import OllamaConnectionError
from providers.api_call import ProviderResponse, ProviderText


def test_query_agent_does_not_retry_nontransport_provider_parse_failure(monkeypatch) -> None:
    calls = {"count": 0}

    def always_fail(*args, **kwargs):
        del args, kwargs
        calls["count"] += 1
        raise part_2.ResponseParseError("invalid provider response")

    agent = SimpleNamespace(
        id="society_1",
        provider="openai",
        model="model",
        query=always_fail,
    )
    monkeypatch.setattr(part_2.time, "sleep", lambda _: None)

    with pytest.raises(part_2.ResponseParseError, match="invalid provider response"):
        part_2._query_agent_until_valid(agent, "prompt")

    assert calls["count"] == 1


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
        "experiments.part2.part_2.console",
        Console(file=stream, force_terminal=False, color_system=None, width=100),
    )

    _emit_retry_status_line("  [yellow][WARN] retrying[/yellow]")

    assert stream.getvalue() == "\r\x1b[2K  [WARN] retrying"
    assert stream.flush_count == 1


def test_emit_retry_status_line_finalizes_with_cleared_newline(monkeypatch) -> None:
    stream = FlushTrackingIO()
    monkeypatch.setattr(
        "experiments.part2.part_2.console",
        Console(file=stream, force_terminal=False, color_system=None, width=100),
    )

    _emit_retry_status_line("", finalize=True)

    assert stream.getvalue() == "\r\x1b[2K\n"
    assert stream.flush_count == 1


def test_emit_retry_status_line_truncates_to_terminal_width(monkeypatch) -> None:
    stream = FlushTrackingIO()
    monkeypatch.setattr(
        "experiments.part2.part_2.console",
        Console(file=stream, force_terminal=False, color_system=None, width=40),
    )

    _emit_retry_status_line(
        "  [yellow][WARN] Agent society_1 attempt 2 raised ValueError: "
        "Expected valid JSON output but no parseable JSON object was found. "
        "Retrying in 2s...[/yellow]"
    )

    output = stream.getvalue()
    assert output.startswith("\r\x1b[2K")
    status_text = output.removeprefix("\r\x1b[2K")
    assert "\n" not in status_text
    assert "[yellow]" not in status_text
    assert len(status_text) <= 39
    assert status_text.endswith("...")
    assert stream.flush_count == 1


def test_run_part_2_writes_multi_day_results(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_query(self, query: str, json_mode: bool = False) -> str:
        del query, json_mode
        return json.dumps(
            {
                "reasoning": "Protecting the commons is better for long-term stability.",
                "action": "RESTRAIN",
            }
        )

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", fake_query)

    csv_path = run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=3,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
    )

    result_path = Path(csv_path)
    assert result_path.exists()

    with result_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 6
    assert result_path.name.startswith("part2__openai__gpt-4-1-mini__n3__d2__water__")
    assert rows[0]["provider"] == "openai"
    assert rows[0]["model"] == "gpt-4.1-mini"
    assert rows[0]["day"] == "1"
    assert rows[-1]["day"] == "2"
    assert {row["action"] for row in rows} == {"RESTRAIN"}
    metadata_files = list((tmp_path / "data" / "raw" / "part_2").glob("*_meta.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["completed_rows"] == 6


def test_run_part2_sensitivity_controls_capacity_deaths_and_generation_seed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    observed_seeds: list[int | None] = []
    observed_prompts: list[tuple[str, str]] = []

    def fake_query(self, query: str, json_mode: bool = False) -> str:
        del json_mode
        observed_seeds.append(self.seed)
        observed_prompts.append((self.system_prompt, query))
        return json.dumps(
            {
                "reasoning": "Selecting the higher individual multiplier.",
                "action": "OPTION_B",
            }
        )

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", fake_query)

    csv_path = run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=10,
        days=1,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
        resource_capacity=1,
        collapse_death_rate=0.3,
        seed=42,
        environment_seed=99,
        headless=True,
    )

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert len(set(observed_seeds)) == 10
    assert observed_seeds == [int(row["call_seed"]) for row in rows]
    assert all("society_" not in system + prompt for system, prompt in observed_prompts)
    assert all("slot_" not in system + prompt for system, prompt in observed_prompts)
    assert all("score" not in (system + prompt).lower() for system, prompt in observed_prompts)
    assert {row["resource_capacity"] for row in rows} == {"1"}
    assert {row["deaths"] for row in rows} == {"3"}
    assert {row["population_end"] for row in rows} == {"7"}

    metadata_path = Path(csv_path).with_name(f"{Path(csv_path).stem}_meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["resource_capacity"] == 1
    assert metadata["collapse_death_rate"] == 0.3
    assert metadata["generation_seed"] == 42
    assert metadata["environment_seed"] == 99
    assert metadata["dynamics"]["incentive_policy"]["individual_score"] == "not_defined"
    assert "__c1__du2__dr0p3__s42__" in Path(csv_path).name


def test_default_collapse_death_rate_preserves_legacy_divisor_behavior() -> None:
    for population in range(1, 31):
        assert part_2._collapse_deaths(population, 0) == min(
            population,
            max(1, part_2.ceil(population / part_2.COLLAPSE_ATTRITION_DIVISOR)),
        )
    assert part_2._collapse_deaths(30, 1) == 0


def test_society_cli_parses_sensitivity_controls() -> None:
    parsed = part_2.parse_society_args(
        [
            "--resource-capacity",
            "123",
            "--collapse-death-rate",
            "0.25",
            "--seed",
            "99",
        ]
    )
    assert parsed.resource_capacity == 123
    assert parsed.collapse_death_rate == 0.25
    assert parsed.seed == 99


def test_part_2_attempt_sidecar_retains_invalid_without_semantic_retry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part2.part_2.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("experiments.part2.part_2.time.sleep", lambda _: None)
    calls = {"count": 0}

    def query(self, prompt: str, json_mode: bool = False) -> str:
        del self, prompt, json_mode
        calls["count"] += 1
        if calls["count"] == 1:
            return '{"action":"UNKNOWN","reasoning":"invalid choice"}'
        return '{"action":"RESTRAIN","reasoning":"preserve the reserve"}'

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", query)
    csv_path = run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=2,
        days=1,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
        headless=True,
    )

    metadata_path = Path(csv_path).with_name(f"{Path(csv_path).stem}_meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["invalid_attempts"] == 1
    assert metadata["retry_attempts"] == 0
    assert metadata["attempt_log"]["total_attempts"] == 2
    assert metadata["attempt_log"]["successful_attempts"] == 1
    attempt_path = Path(metadata["attempt_log"]["path"])
    attempts = [
        json.loads(line)
        for line in attempt_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [attempt["outcome"] for attempt in attempts] == [
        "invalid_response",
        "success",
    ]
    assert attempts[0]["raw_response"].startswith("{")
    assert attempts[0]["will_retry"] is False
    assert attempts[1]["parsed_response"] == {
        "action": "RESTRAIN",
        "reasoning": "preserve the reserve",
    }
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["action"] for row in rows] == ["INVALID", "RESTRAIN"]
    assert {row["invalid_count"] for row in rows} == {"1"}
    assert {row["overuse_count"] for row in rows} == {"0"}
    assert {row["restrain_count"] for row in rows} == {"1"}
    assert rows[0]["attempt_outcome"] == "invalid_response"
    assert rows[0]["invalid_reason"].startswith("ResponseParseError:")


def test_parse_agent_response_maps_neutral_prompt_options_to_internal_actions() -> None:
    option_a = json.dumps({"reasoning": "Option label selected.", "action": "OPTION_A"})
    option_b = json.dumps({"reasoning": "Option label selected.", "action": "Option B"})

    assert _parse_agent_response(option_a) == ("RESTRAIN", "Option label selected.")
    assert _parse_agent_response(option_b) == ("OVERUSE", "Option label selected.")


def test_production_like_simulation_has_deterministic_distinct_call_seeds(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(part_2, "_new_run_id", lambda: "p2run_fixed_test")
    captured_by_run: list[list[int]] = []

    for run_number in (1, 2):
        run_dir = tmp_path / f"run_{run_number}"
        run_dir.mkdir()
        monkeypatch.chdir(run_dir)
        captured: list[int] = []
        captured_by_run.append(captured)

        def query(self, prompt: str, json_mode: bool = False) -> str:
            del prompt, json_mode
            captured.append(self.seed)
            return '{"action":"OPTION_A","reasoning":"Preserve the reserve."}'

        monkeypatch.setattr("experiments.part2.part_2.Agent2.query", query)
        csv_path = run_part_2(
            provider="openai",
            model="gpt-4.1-mini",
            society_size=3,
            days=2,
            resource="water",
            selfish_gain=999,
            depletion_units=2,
            community_benefit=999,
            generation_seed=17,
            environment_seed=23,
            headless=True,
        )
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert captured == [int(row["call_seed"]) for row in rows]
        assert {row["generation_seed"] for row in rows} == {"17"}
        assert {row["environment_seed"] for row in rows} == {"23"}

    assert captured_by_run[0] == captured_by_run[1]
    assert len(set(captured_by_run[0])) == 6


def test_invalid_is_no_action_and_legacy_score_fields_do_not_change_transition(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    responses = iter(
        [
            '{"action":"UNKNOWN","reasoning":"Malformed choice."}',
            '{"action":"OPTION_B","reasoning":"Use reserve."}',
            '{"action":"OPTION_A","reasoning":"Preserve reserve."}',
        ]
    )
    monkeypatch.setattr(
        "experiments.part2.part_2.Agent2.query",
        lambda self, prompt, json_mode=False: next(responses),
    )

    csv_path = run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=3,
        days=1,
        resource="water",
        selfish_gain=10_000,
        depletion_units=2,
        community_benefit=10_000,
        resource_capacity=10,
        generation_seed=7,
        environment_seed=11,
        headless=True,
    )
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["action"] for row in rows] == ["INVALID", "OVERUSE", "RESTRAIN"]
    assert {row["resource_units_remaining"] for row in rows} == {"8"}
    assert {row["invalid_count"] for row in rows} == {"1"}
    assert {row["overuse_count"] for row in rows} == {"1"}
    assert {row["restrain_count"] for row in rows} == {"1"}
    assert {row["selfish_gain"] for row in rows} == {"10000"}
    assert {row["community_benefit"] for row in rows} == {"10000"}


def test_randomized_attrition_is_reproducible_and_records_selection_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    selections: list[list[int]] = []
    for run_number in (1, 2):
        run_dir = tmp_path / f"attrition_{run_number}"
        run_dir.mkdir()
        monkeypatch.chdir(run_dir)
        monkeypatch.setattr(
            "experiments.part2.part_2.Agent2.query",
            lambda self, prompt, json_mode=False: (
                '{"action":"OPTION_B","reasoning":"Use reserve."}'
            ),
        )
        csv_path = run_part_2(
            provider="openai",
            model="gpt-4.1-mini",
            society_size=10,
            days=1,
            resource="water",
            selfish_gain=2,
            depletion_units=2,
            community_benefit=5,
            resource_capacity=1,
            collapse_death_rate=0.3,
            generation_seed=31,
            environment_seed=47,
            headless=True,
        )
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        selected = json.loads(rows[0]["death_selected_slots_json"])
        selections.append(selected)
        assert len(selected) == 3
        assert len({row["attrition_seed"] for row in rows}) == 1
        assert {
            int(row["anonymous_agent_slot"])
            for row in rows
            if row["died_today"] == "true"
        } == set(selected)
        assert sorted(
            int(row["attrition_rank"])
            for row in rows
            if row["attrition_rank"]
        ) == [1, 2, 3]

    assert selections[0] == selections[1]
    assert set(selections[0]) != {8, 9, 10}


def test_randomized_attrition_resume_reconstructs_exact_living_slots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def overuse(self, prompt: str, json_mode: bool = False) -> str:
        del self, prompt, json_mode
        return '{"action":"OPTION_B","reasoning":"Use reserve."}'

    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted_dir.mkdir()
    monkeypatch.chdir(uninterrupted_dir)
    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", overuse)
    uninterrupted_csv = run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=6,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
        resource_capacity=1,
        collapse_death_rate=0.5,
        generation_seed=13,
        environment_seed=29,
        headless=True,
    )
    with Path(uninterrupted_csv).open(newline="", encoding="utf-8") as handle:
        uninterrupted_rows = list(csv.DictReader(handle))

    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    monkeypatch.chdir(resumed_dir)
    calls = {"count": 0}

    def interrupt_after_day_one(self, prompt: str, json_mode: bool = False) -> str:
        del self, prompt, json_mode
        calls["count"] += 1
        if calls["count"] == 7:
            raise KeyboardInterrupt()
        return '{"action":"OPTION_B","reasoning":"Use reserve."}'

    monkeypatch.setattr(
        "experiments.part2.part_2.Agent2.query",
        interrupt_after_day_one,
    )
    partial_csv = run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=6,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
        resource_capacity=1,
        collapse_death_rate=0.5,
        generation_seed=13,
        environment_seed=29,
        headless=True,
    )
    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", overuse)
    assert run_part_2(resume=True, headless=True) == partial_csv
    with Path(partial_csv).open(newline="", encoding="utf-8") as handle:
        resumed_rows = list(csv.DictReader(handle))

    uninterrupted_selections = {
        day: json.loads(next(row for row in uninterrupted_rows if row["day"] == day)["death_selected_slots_json"])
        for day in ("1", "2")
    }
    resumed_selections = {
        day: json.loads(next(row for row in resumed_rows if row["day"] == day)["death_selected_slots_json"])
        for day in ("1", "2")
    }
    assert resumed_selections == uninterrupted_selections
    day_one_dead = set(resumed_selections["1"])
    day_two_slots = {
        int(row["anonymous_agent_slot"])
        for row in resumed_rows
        if row["day"] == "2"
    }
    assert day_two_slots.isdisjoint(day_one_dead)
    assert {row["population_start"] for row in resumed_rows if row["day"] == "2"} == {"3"}


def test_available_provider_identity_usage_and_raw_hash_are_persisted(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    content = '{"action":"OPTION_A","reasoning":"Preserve reserve."}'
    provider_payload = {"id": "req-123", "model": "returned-model", "body": content}

    def query(self, prompt: str, json_mode: bool = False) -> str:
        del self, prompt, json_mode
        return ProviderText(
            content,
            ProviderResponse(
                provider="openai",
                model="requested-model",
                content=content,
                reasoning="",
                raw_response=provider_payload,
                finish_reason="stop",
                truncated=False,
                usage={"input_tokens": 9, "output_tokens": 5},
                request_id="req-123",
                requested_model="requested-model",
                response_model="returned-model",
                model_identity_match=True,
            ),
        )

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", query)
    csv_path = run_part_2(
        provider="openai",
        model="requested-model",
        society_size=1,
        days=1,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
        generation_seed=3,
        environment_seed=5,
        headless=True,
    )
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["requested_model"] == "requested-model"
    assert row["returned_model"] == "returned-model"
    assert row["request_id"] == "req-123"
    assert row["finish_reason"] == "stop"
    assert json.loads(row["usage_json"]) == {"input_tokens": 9, "output_tokens": 5}
    assert row["raw_response_sha256"] == part_2._stable_json_sha256(provider_payload)
    assert row["run_id"].startswith("p2run_")
    assert row["trajectory_id"].startswith("p2traj_")
    assert row["structural_cell_id"].startswith("p2cell_")


def test_direct_confirmatory_attempt_provenance_binds_exact_route(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part2.part_2.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )
    route = "vendor/exact-model"
    content = '{"action":"OPTION_A","reasoning":"Preserve reserve."}'

    def query(self, prompt: str, json_mode: bool = False) -> str:
        del self, prompt, json_mode
        return ProviderText(
            content,
            ProviderResponse(
                provider="inference_hub",
                model=route,
                content=content,
                reasoning="",
                raw_response={"id": "req-exact", "model": route},
                finish_reason="stop",
                truncated=False,
                usage={"input_tokens": 10, "output_tokens": 6},
                request_id="req-exact",
                requested_model=route,
                response_model=route,
                model_identity_match=True,
            ),
        )

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", query)
    csv_path = Path(
        run_part_2(
            provider="inference_hub",
            model=route,
            society_size=1,
            days=1,
            resource="water",
            selfish_gain=2,
            depletion_units=2,
            community_benefit=5,
            generation_seed=3,
            environment_seed=5,
            headless=True,
        )
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    part_2.validate_direct_attempt_provenance(
        part_2.attempt_log_path_for_csv(csv_path),
        rows,
        provider="inference_hub",
        model=route,
    )
    metadata = part_2._load_part_2_metadata(part_2._metadata_path_for_csv(csv_path))
    assert metadata["generation_protocol"]["mode"] == "direct_provider_structured_output"
    assert metadata["generation_protocol"]["output_token_cap"] == 32
    assert metadata["resume_contract"]["generation_protocol"] == metadata["generation_protocol"]


def test_strict_extraction_uses_subject_provider_raw_hash_not_extractor_json_hash() -> None:
    subject_raw_hash = "a" * 64
    result = part_2._decision_result(
        agent=SimpleNamespace(model="requested-model"),
        action="RESTRAIN",
        reasoning="Preserve reserve.",
        invalid_reason="",
        attempt_outcome="success",
        attempt_count=1,
        call_seed=123,
        raw_response='{"action":"RESTRAIN","reasoning":"Preserve reserve."}',
        generation_record={
            "status": "success",
            "subject": {
                "requested_model": "requested-model",
                "response_model": "returned-model",
                "raw_response_sha256": subject_raw_hash,
            },
        },
    )
    assert result.raw_response_sha256 == subject_raw_hash


def test_truncated_semantic_output_is_invalid_and_not_regenerated(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = {"count": 0}
    content = '{"action":"OPTION_B","reasoning":"This JSON is still truncated."}'

    def query(self, prompt: str, json_mode: bool = False) -> str:
        del self, prompt, json_mode
        calls["count"] += 1
        return ProviderText(
            content,
            ProviderResponse(
                provider="openai",
                model="requested-model",
                content=content,
                reasoning="",
                raw_response={"partial": content},
                finish_reason="length",
                truncated=True,
                usage={"output_tokens": 20},
                request_id="req-truncated",
                requested_model="requested-model",
                response_model="requested-model",
                model_identity_match=True,
            ),
        )

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", query)
    csv_path = run_part_2(
        provider="openai",
        model="requested-model",
        society_size=1,
        days=1,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
        resource_capacity=10,
        generation_seed=3,
        environment_seed=5,
        headless=True,
    )
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert calls["count"] == 1
    assert row["action"] == "INVALID"
    assert row["attempt_outcome"] == "invalid_response"
    assert row["resource_units_remaining"] == "10"
    assert row["finish_reason"] == "length"


def test_run_part_2_resumes_from_completed_days(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part2.part_2.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )

    calls = {"count": 0}

    def interrupted_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        calls["count"] += 1
        if calls["count"] <= 2:
            return json.dumps(
                {"reasoning": "First day completes.", "action": "RESTRAIN"}
            )
        raise KeyboardInterrupt()

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", interrupted_query)

    csv_path = run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=2,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
    )

    metadata_files = list((tmp_path / "data" / "raw" / "part_2").glob("*_meta.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["prompt_config_hash"] == PROMPT_CONFIG_HASH

    def resumed_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        return json.dumps(
            {"reasoning": "Second day completes after resume.", "action": "RESTRAIN"}
        )

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", resumed_query)

    resumed_csv_path = run_part_2(resume=True)

    assert resumed_csv_path == csv_path
    with Path(resumed_csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert [row["day"] for row in rows] == ["1", "1", "2", "2"]
    assert rows[-1]["reasoning"] == "Second day completes after resume."
    metadata_files = list((tmp_path / "data" / "raw" / "part_2").glob("*_meta.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["completed_rows"] == 4
    assert metadata["attempt_log"]["total_attempts"] == 5
    assert metadata["attempt_log"]["successful_attempts"] == 4
    assert metadata["attempt_log"]["interrupted_attempts"] == 1


def test_matching_part_2_metadata_ignores_stale_prompt_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "data" / "raw" / "part_2"
    results_dir.mkdir(parents=True)
    stale_metadata = results_dir / "part2__ollama__gpt-oss-20b__n50__d100__water__20260425_204541_meta.json"
    stale_metadata.write_text(
        json.dumps(
            {
                "timestamp": "20260425_204541",
                "csv_path": str(stale_metadata.with_suffix(".csv")),
                "provider": "ollama",
                "model": "gpt-oss:20b",
                "society_config": {
                    "society_size": 50,
                    "days": 100,
                    "resource": "water",
                    "selfish_gain": 2,
                    "depletion_units": 2,
                    "community_benefit": 5,
                },
                "resource_capacity": 2500,
            }
        ),
        encoding="utf-8",
    )


def test_strict_resume_fails_closed_on_old_part_2_schema(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(part_2, "run_experiment_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "experiments.part2.part_2.Agent2.query_for_grading",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    csv_path = Path(
        run_part_2(
            provider="openai",
            model="gpt-4.1-mini",
            society_size=1,
            days=1,
            resource="water",
            selfish_gain=2,
            depletion_units=2,
            community_benefit=5,
            generation_seed=3,
            environment_seed=5,
            output_token_cap=512,
            headless=True,
        )
    )
    metadata_path = csv_path.with_name(f"{csv_path.stem}_meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("part_2_schema_version")
    metadata["metadata_sha256"] = metadata_payload_sha256(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="old-schema confirmatory runs cannot be resumed"):
        run_part_2(resume=True, headless=True)

    assert (
        _matching_part_2_metadata_path(
            provider="ollama",
            model="gpt-oss:20b",
            config=SocietyConfig(),
        )
        is None
    )


def test_run_part_2_resume_prints_state_panel(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part2.part_2.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )

    calls = {"count": 0}

    def interrupted_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        calls["count"] += 1
        if calls["count"] <= 2:
            return json.dumps({"reasoning": "First day.", "action": "RESTRAIN"})
        raise KeyboardInterrupt()

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", interrupted_query)

    csv_path = run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=2,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
    )

    buffer = io.StringIO()
    test_console = Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
        width=200,
    )
    monkeypatch.setattr("experiments.part2.part_2.console", test_console)
    monkeypatch.setattr(
        "experiments.part2.part_2.Agent2.query",
        lambda self, query, json_mode=False: json.dumps(
            {"reasoning": "Second day.", "action": "RESTRAIN"}
        ),
    )

    assert run_part_2(resume=True, headless=True) == csv_path

    output = buffer.getvalue()
    assert "Resuming Part 2 Run" in output
    assert "Completed days: 1 / 2" in output
    assert "Remaining days: 1" in output
    assert "Reserve:" in output


def test_run_part_2_headless_prints_compact_progress(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part2.part_2.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.part2.part_2.Agent2.query",
        lambda self, query, json_mode=False: json.dumps(
            {"reasoning": "Compact progress.", "action": "RESTRAIN"}
        ),
    )

    buffer = io.StringIO()
    test_console = Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
        width=200,
    )
    monkeypatch.setattr("experiments.part2.part_2.console", test_console)

    run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=2,
        days=1,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
        headless=True,
    )

    output = buffer.getvalue()
    assert "Model openai/gpt-4.1-mini [day 1/1] [--------------------] RUNNING" in output
    assert "Model openai/gpt-4.1-mini [day 1/1] [####################] done" in output


def test_run_part_2_headless_interruption_prints_resume_instruction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part2.part_2.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )

    def interrupted_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        raise KeyboardInterrupt()

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", interrupted_query)

    buffer = io.StringIO()
    test_console = Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
        width=200,
    )
    monkeypatch.setattr("experiments.part2.part_2.console", test_console)

    csv_path = run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=2,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
        headless=True,
    )

    output = buffer.getvalue()
    assert "Simulation Interrupted" in output
    assert "Saved partial results" in output
    assert "Rerun with `--resume`" in output
    assert Path(csv_path).with_name(f"{Path(csv_path).stem}_meta.json").exists()


def test_run_part_2_pauses_immediately_when_ollama_disconnects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part2.part_2.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.part2.part_2._prepare_ollama_model_for_run",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.part2.part_2.unload_ollama_model",
        lambda model: None,
    )
    monkeypatch.setattr(
        "experiments.part2.part_2.time.sleep",
        lambda seconds: (_ for _ in ()).throw(
            AssertionError("connection errors must not enter retry backoff")
        ),
    )

    calls = {"count": 0}

    def disconnected_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        calls["count"] += 1
        raise OllamaConnectionError("Could not connect to Ollama at http://localhost:11434.")

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", disconnected_query)

    csv_path = run_part_2(
        provider="ollama",
        model="qwen3.5",
        society_size=2,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
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


def test_run_part_2_until_complete_stops_when_ollama_disconnects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part2.part_2.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.part2.part_2._prepare_ollama_model_for_run",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.part2.part_2.unload_ollama_model",
        lambda model: None,
    )

    calls = {"count": 0}

    def disconnected_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        calls["count"] += 1
        raise OllamaConnectionError("Could not connect to Ollama at http://localhost:11434.")

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", disconnected_query)

    with pytest.raises(OllamaConnectionError):
        run_part_2_until_complete(
            provider="ollama",
            model="qwen3.5",
            society_size=2,
            days=2,
            resource="water",
            selfish_gain=2,
            depletion_units=2,
            community_benefit=5,
        )

    assert calls["count"] == 1


def test_run_part_2_until_complete_resumes_matching_partial_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part2.part_2.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )

    calls = {"count": 0}

    def flaky_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        calls["count"] += 1
        if calls["count"] == 3:
            raise KeyboardInterrupt()
        return json.dumps({"reasoning": "eventual success", "action": "RESTRAIN"})

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", flaky_query)

    partial_path = run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=2,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
    )
    assert Path(partial_path).with_name(f"{Path(partial_path).stem}_meta.json").exists()

    final_path = run_part_2_until_complete(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=2,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
        headless=True,
    )

    assert final_path == partial_path
    with Path(final_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    metadata_path = Path(final_path).with_name(f"{Path(final_path).stem}_meta.json")
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
