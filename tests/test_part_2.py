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
from providers.api_call import OllamaConnectionError


def test_query_agent_stops_after_bounded_attempts(monkeypatch) -> None:
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

    assert calls["count"] == part_2.MAX_AGENT_ATTEMPTS


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

    def fake_query(self, query: str, json_mode: bool = False) -> str:
        del query, json_mode
        observed_seeds.append(self.seed)
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
        headless=True,
    )

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert observed_seeds == [42] * 10
    assert {row["resource_capacity"] for row in rows} == {"1"}
    assert {row["deaths"] for row in rows} == {"3"}
    assert {row["population_end"] for row in rows} == {"7"}

    metadata_path = Path(csv_path).with_name(f"{Path(csv_path).stem}_meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["resource_capacity"] == 1
    assert metadata["collapse_death_rate"] == 0.3
    assert metadata["generation_seed"] == 42
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


def test_part_2_attempt_sidecar_preserves_invalid_retry_and_success(
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
    assert metadata["retry_attempts"] == 1
    assert metadata["attempt_log"]["total_attempts"] == 3
    assert metadata["attempt_log"]["successful_attempts"] == 2
    attempt_path = Path(metadata["attempt_log"]["path"])
    attempts = [
        json.loads(line)
        for line in attempt_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [attempt["outcome"] for attempt in attempts] == [
        "invalid_response",
        "success",
        "success",
    ]
    assert attempts[0]["raw_response"].startswith("{")
    assert attempts[1]["parsed_response"] == {
        "action": "RESTRAIN",
        "reasoning": "preserve the reserve",
    }


def test_parse_agent_response_maps_neutral_prompt_options_to_internal_actions() -> None:
    option_a = json.dumps({"reasoning": "Option label selected.", "action": "OPTION_A"})
    option_b = json.dumps({"reasoning": "Option label selected.", "action": "Option B"})

    assert _parse_agent_response(option_a) == ("RESTRAIN", "Option label selected.")
    assert _parse_agent_response(option_b) == ("OVERUSE", "Option label selected.")


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
