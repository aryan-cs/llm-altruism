import csv
import io
import json
from pathlib import Path

import pytest
from rich.console import Console

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
    assert not list((tmp_path / "data" / "raw" / "part_2").glob("*_meta.json"))


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
    assert not list((tmp_path / "data" / "raw" / "part_2").glob("*_meta.json"))


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
    assert Path(csv_path).with_name(f"{Path(csv_path).stem}_meta.json").exists()


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
    assert not Path(final_path).with_name(f"{Path(final_path).stem}_meta.json").exists()
