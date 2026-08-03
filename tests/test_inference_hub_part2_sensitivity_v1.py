from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import experiments.misc.inference_hub_part2_sensitivity_v1 as sensitivity_module
from analysis.part2_confirmatory import (
    SENSITIVITY_FACTORS,
    SENSITIVITY_LEVELS,
    analyze_sentinel_sensitivity,
    resolution_v_half_fraction,
)
from experiments.misc.inference_hub_part2_panel import (
    DEFAULT_COMPATIBILITY,
    DEFAULT_PANEL,
    DEFAULT_REGISTRY,
    Part2Contract,
    _load_panel,
)
from experiments.misc.inference_hub_part1_panel import (
    _read_json,
    _sha256_json,
    select_routes,
)
from experiments.misc.inference_hub_provider_safe_v2 import provider_round_robin
from experiments.misc.inference_hub_part2_sensitivity_v1 import (
    DEFAULT_DESIGN,
    DEFAULT_EXPLORATORY_DESIGN,
    DEFAULT_REVISED_EXPLORATORY_DESIGN,
    InferenceHubPart2SensitivityError,
    SensitivityCondition,
    _AttemptBudget,
    _ConditionJournal,
    _analyze_completed_design,
    _rate_limit_contract,
    _run_trajectory,
    _sensitivity_request_contract,
    condition_environment_seeds,
    load_sensitivity_design,
    run_sensitivity_campaign,
)


SUBJECT = "openai/gpt-5.4"
ENDPOINT = "https://inference-api.nvidia.com/v1"


def _subject() -> dict[str, Any]:
    return {
        "target_id": SUBJECT,
        "upstream_provider": "openai",
        "model": "gpt-5.4",
        "route": "openai/gpt-5.4",
        "supported_controls": [
            "seed", "temperature", "top_p", "structured_response",
        ],
    }


class FakeClient:
    base_url = ENDPOINT

    def __init__(self, *, malformed: bool = False) -> None:
        self.malformed = malformed
        self.calls: list[dict[str, Any]] = []

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/chat/completions"
        self.calls.append(body)
        content = (
            "not-json" if self.malformed else
            json.dumps({"action": "OPTION_A", "reasoning": "Preserve the reserve."})
        )
        return {
            "id": f"request-{len(self.calls)}", "model": body["model"],
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        }


def _tiny_condition() -> SensitivityCondition:
    return SensitivityCondition(
        cell_id="test_cell", capacity_per_initial_agent=2,
        depletion_units=1, collapse_death_rate=0.1,
        society_size=2, horizon_days=2, resource_capacity=4,
        coded_levels={factor: -1 for factor in SENSITIVITY_FACTORS},
    )


def test_design_matches_authoritative_resolution_v_contract() -> None:
    design, conditions = load_sensitivity_design(DEFAULT_DESIGN)
    generated = resolution_v_half_fraction()

    assert design["design"] == "2^(5-1)_resolution_V_I=ABCDE"
    assert design["factors"] == {
        factor: list(levels) for factor, levels in SENSITIVITY_LEVELS.items()
    }
    assert [condition.public_dict() for condition in conditions] == generated
    assert len(conditions) == 16
    assert len(design["sentinels"]) == 6
    assert len({row["developer"] for row in design["sentinels"]}) == 6
    assert design["seeds_per_cell"] == 12
    assert design["analysis"] == {
        "inference_scope": "future_confirmatory_preregistered",
        "main_effects_per_sentinel": 5,
        "global_holm_family_size": 30,
        "interactions_confirmatory": False,
        "within_sentinel_max_t": "diagnostic_only",
    }
    assert design["execution_budget"]["maximum_successful_posts"] == 3_240_000
    assert design["execution_budget"]["maximum_physical_attempts"] == 3_564_000
    assert sum(cell.society_size * cell.horizon_days for cell in conditions) * 6 * 12 == 3_240_000


def test_deadline_design_is_separate_exploratory_and_24_hour_sized() -> None:
    future, future_cells = load_sensitivity_design(DEFAULT_DESIGN)
    deadline, deadline_cells = load_sensitivity_design(DEFAULT_EXPLORATORY_DESIGN)

    assert deadline["campaign_id"] != future["campaign_id"]
    assert deadline["analysis"]["inference_scope"] == "deadline_exploratory"
    assert [row["target_id"] for row in deadline["sentinels"]] == [
        row["target_id"] for row in future["sentinels"]
    ]
    assert len(deadline_cells) == len(future_cells) == 16
    assert deadline["factors"] == {
        "capacity_per_initial_agent": [5, 15],
        "depletion_units": [1, 2],
        "collapse_death_rate": [0.1, 0.4],
        "society_size": [4, 8],
        "horizon_days": [10, 20],
    }
    assert deadline["seeds_per_cell"] == 2
    scheduled = sum(cell.society_size * cell.horizon_days for cell in deadline_cells) * 6 * 2
    assert scheduled == 17_280
    assert deadline["execution_budget"]["maximum_successful_posts"] == scheduled
    assert (
        deadline["execution_budget"]["maximum_scheduled_output_tokens"]
        == scheduled * 8192
    )


def test_revised_deadline_design_excludes_only_incompatible_sentinel() -> None:
    original, original_cells = load_sensitivity_design(DEFAULT_EXPLORATORY_DESIGN)
    revised, revised_cells = load_sensitivity_design(
        DEFAULT_REVISED_EXPLORATORY_DESIGN
    )

    original_ids = [row["target_id"] for row in original["sentinels"]]
    revised_ids = [row["target_id"] for row in revised["sentinels"]]
    assert revised["campaign_id"] == "part2_resolution_v_deadline_exploratory_v2"
    assert revised["analysis"]["inference_scope"] == "deadline_exploratory"
    assert revised["seed_policy"] == (
        "two_common_environment_seeds_across_cells_and_sentinels_v2"
    )
    assert len(revised_ids) == 5
    assert set(original_ids) - set(revised_ids) == {
        "anthropic/claude-sonnet-4-6"
    }
    assert revised["revision_from"] == {
        "campaign_id": original["campaign_id"],
        "excluded_target_ids": ["anthropic/claude-sonnet-4-6"],
        "exclusion_code": "frozen_exact_route_missing_required_common_control_top_p",
        "exclusion_stage": "pre_analysis_execution_contract_validation",
        "outcome_information_used": False,
        "substitution_permitted": False,
    }
    assert [cell.public_dict() for cell in revised_cells] == [
        cell.public_dict() for cell in original_cells
    ]
    assert revised["analysis"]["global_holm_family_size"] == 25
    scheduled = (
        sum(cell.society_size * cell.horizon_days for cell in revised_cells)
        * len(revised_ids)
        * revised["seeds_per_cell"]
    )
    assert scheduled == 14_400
    assert revised["execution_budget"] == {
        "maximum_successful_posts": 14_400,
        "maximum_physical_attempts": 15_840,
        "part2_output_tokens_per_attempt": 8192,
        "maximum_scheduled_output_tokens": 117_964_800,
        "maximum_input_utf8_bytes_per_attempt": 8192,
    }


def test_revised_deadline_design_rejects_substitution_or_wrong_holm(
    tmp_path: Path,
) -> None:
    document = json.loads(DEFAULT_REVISED_EXPLORATORY_DESIGN.read_text())
    document["revision_from"]["substitution_permitted"] = True
    substituted = tmp_path / "substituted.json"
    substituted.write_text(json.dumps(document))
    with pytest.raises(
        InferenceHubPart2SensitivityError,
        match="no-substitution exclusion contract",
    ):
        load_sensitivity_design(substituted)

    document = json.loads(DEFAULT_REVISED_EXPLORATORY_DESIGN.read_text())
    document["analysis"]["global_holm_family_size"] = 30
    wrong_holm = tmp_path / "wrong-holm.json"
    wrong_holm.write_text(json.dumps(document))
    with pytest.raises(InferenceHubPart2SensitivityError, match="25-test Holm"):
        load_sensitivity_design(wrong_holm)


def test_sentinels_select_exact_routes_from_combined_registry() -> None:
    if not DEFAULT_REGISTRY.is_file() or not DEFAULT_COMPATIBILITY.is_file():
        pytest.skip("Private combined route evidence is intentionally not distributed.")
    design, _ = load_sensitivity_design(DEFAULT_EXPLORATORY_DESIGN)
    sentinel_ids = [row["target_id"] for row in design["sentinels"]]
    panel, _ = _load_panel(DEFAULT_PANEL)
    registry = _read_json(DEFAULT_REGISTRY, "combined registry")
    compatibility = _read_json(DEFAULT_COMPATIBILITY, "combined compatibility")
    subjects, _ = select_routes(
        registry=registry, compatibility=compatibility,
        selected_ids=sentinel_ids, judge_target_id=str(panel["judge_target_id"]),
    )
    subjects = provider_round_robin(subjects)

    assert [(row["target_id"], row["upstream_provider"], row["route"]) for row in subjects] == [
        ("openai/gpt-5.4", "openai", "openai/openai/gpt-5.4"),
        (
            "anthropic/claude-sonnet-4-6", "anthropic",
            "azure/anthropic/claude-sonnet-4-6",
        ),
        (
            "google/gemini-3.1-pro-preview", "google",
            "gcp/google/gemini-3.1-pro-preview",
        ),
        (
            "meta/llama-3.3-70b-instruct", "meta",
            "nvidia/meta/llama-3.3-70b-instruct",
        ),
        (
            "qwen/qwen3.6-27b", "qwen",
            "nvidia/qwen/qwen3.6-27b",
        ),
        (
            "nvidia/nemotron-3-super-v3", "nvidia",
            "nvidia/nvidia/nemotron-3-super-v3",
        ),
    ]


def test_revised_sentinels_all_satisfy_common_request_controls() -> None:
    if not DEFAULT_REGISTRY.is_file() or not DEFAULT_COMPATIBILITY.is_file():
        pytest.skip("Private combined route evidence is intentionally not distributed.")
    design, _ = load_sensitivity_design(DEFAULT_REVISED_EXPLORATORY_DESIGN)
    sentinel_ids = [row["target_id"] for row in design["sentinels"]]
    panel, _ = _load_panel(DEFAULT_PANEL)
    registry = _read_json(DEFAULT_REGISTRY, "combined registry")
    compatibility = _read_json(DEFAULT_COMPATIBILITY, "combined compatibility")
    subjects, _ = select_routes(
        registry=registry,
        compatibility=compatibility,
        selected_ids=sentinel_ids,
        judge_target_id=str(panel["judge_target_id"]),
    )

    assert {row["target_id"] for row in subjects} == set(sentinel_ids)
    assert all(
        {"seed", "temperature", "top_p", "structured_response"}
        <= set(row["supported_controls"])
        for row in subjects
    )
    for subject in subjects:
        body, controls = _sensitivity_request_contract(
            subject,
            prompt="prompt",
            system_prompt="system",
            generation_seed=17,
        )
        assert body["top_p"] == 1
        assert controls["common_contract"] is True


def test_common_seeds_are_deterministic_and_cell_independent() -> None:
    kwargs = {
        "campaign_id": "campaign-v1", "panel_id": "panel-v1",
        "base_seed": 7, "trajectory_count": 12,
    }
    first = condition_environment_seeds(**kwargs)
    repeat = condition_environment_seeds(**kwargs)

    assert first == repeat
    assert len(first) == len(set(first)) == 12


def test_request_contract_is_common_and_capped_at_8192_tokens() -> None:
    body, controls = _sensitivity_request_contract(
        _subject(), prompt="prompt", system_prompt="system", generation_seed=17,
    )

    assert body["temperature"] == 0.2
    assert body["top_p"] == 1
    assert body["seed"] == 17
    assert body["max_tokens"] == 8192
    assert body["response_format"]["type"] == "json_schema"
    assert controls["common_contract"] is True


def test_request_contract_fails_if_a_sentinel_lacks_a_common_control() -> None:
    subject = _subject()
    subject["supported_controls"] = ["seed", "temperature", "top_p"]
    with pytest.raises(InferenceHubPart2SensitivityError, match="lacks common controls"):
        _sensitivity_request_contract(
            subject, prompt="prompt", system_prompt="system", generation_seed=17,
        )


def test_raw_journal_resume_and_invalid_nonrestraint_policy(tmp_path: Path) -> None:
    condition = _tiny_condition()
    journal = _ConditionJournal(
        tmp_path / "trajectory.jsonl", campaign_id="sensitivity_v1",
        condition=condition,
    )
    budget = _AttemptBudget(
        tmp_path / "attempts.jsonl", campaign_id="sensitivity_v1", ceiling=20,
    )
    client = FakeClient(malformed=True)
    kwargs = {
        "subject": _subject(), "trajectory_index": 0, "environment_seed": 101,
        "contract": Part2Contract(
            society_size=2, days=2, trajectories=1, capacity=4,
            private_gain=2, reserve_cost=1, community_benefit=10,
            collapse_death_rate=0.1,
        ),
        "journal": journal, "participant_workers": 2, "max_attempts": 2,
        "initial_backoff_seconds": 0, "sleep_fn": lambda _: None,
        "maximum_input_bytes": 8192, "attempt_budget": budget,
        "cell_id": condition.cell_id,
    }
    first = _run_trajectory(client=client, **kwargs)

    assert first["operationally_eligible"] is True
    assert first["invalid_count"] == 4
    assert first["restraint_count"] == 0
    assert first["overuse_count"] == 0
    assert len(client.calls) == 4
    assert len(budget.records) == 4
    assert all(call["temperature"] == 0.2 for call in client.calls)
    assert all(call["max_tokens"] == 8192 for call in client.calls)
    raw = journal.records
    assert any(row["event"] == "semantic_result" and row["raw_response"] for row in raw)
    assert all(row["cell_id"] == condition.cell_id for row in raw)

    resumed_client = FakeClient()
    resumed = _run_trajectory(client=resumed_client, **kwargs)
    assert resumed == first
    assert resumed_client.calls == []
    assert len(budget.records) == 4


def test_six_sentinels_produce_exactly_30_global_holm_tests() -> None:
    design, _ = load_sensitivity_design(DEFAULT_DESIGN)
    sentinels = [row["target_id"] for row in design["sentinels"]]
    seeds = list(range(12))
    observations: dict[str, list[dict[str, object]]] = {}
    for sentinel_index, sentinel in enumerate(sentinels):
        rows: list[dict[str, object]] = []
        for cell in resolution_v_half_fraction():
            coded = cell["coded_levels"]
            for seed in seeds:
                value = 0.5 + 0.004 * seed + 0.002 * sentinel_index
                value += sum(0.008 * coded[factor] for factor in SENSITIVITY_FACTORS)
                rows.append({
                    **{key: value for key, value in cell.items() if key != "coded_levels"},
                    "environment_seed": seed, "normalized_aurc": value,
                })
        observations[sentinel] = rows

    effects = analyze_sentinel_sensitivity(
        observations, expected_sentinel_ids=sentinels,
    )
    assert len(effects) == 30
    assert {(row["sentinel_id"], row["factor"]) for row in effects} == {
        (sentinel, factor) for sentinel in sentinels for factor in SENSITIVITY_FACTORS
    }
    assert all(row["holm_family_size"] == 30 for row in effects)
    assert all("within_sentinel_max_t_adjusted_p" in row for row in effects)
    assert all(row["max_t_family"].endswith("diagnostic") for row in effects)


def test_deadline_effects_keep_exploratory_levels_and_holm_30() -> None:
    design, cells = load_sensitivity_design(DEFAULT_EXPLORATORY_DESIGN)
    sentinels = [row["target_id"] for row in design["sentinels"]]
    trajectories: list[dict[str, object]] = []
    for sentinel_index, sentinel in enumerate(sentinels):
        for cell in cells:
            for seed in range(2):
                trajectories.append({
                    **cell.public_dict(), "target_id": sentinel,
                    "environment_seed": seed,
                    "normalized_aurc": (
                        0.5 + 0.004 * seed + 0.002 * sentinel_index
                        + sum(
                            0.008 * cell.coded_levels[factor]
                            for factor in SENSITIVITY_FACTORS
                        )
                    ),
                })

    effects = _analyze_completed_design(
        trajectories, sentinel_ids=sentinels, design=design,
    )
    assert len(effects) == 30
    assert all(row["confirmatory"] is False for row in effects)
    assert all(row["inference_scope"] == "deadline_exploratory" for row in effects)
    for row in effects:
        assert [row["low_level"], row["high_level"]] == design["factors"][row["factor"]]
        assert row["holm_family_size"] == 30


def test_revised_deadline_effects_use_exact_holm_25_family() -> None:
    design, cells = load_sensitivity_design(DEFAULT_REVISED_EXPLORATORY_DESIGN)
    sentinels = [row["target_id"] for row in design["sentinels"]]
    trajectories: list[dict[str, object]] = []
    for sentinel_index, sentinel in enumerate(sentinels):
        for cell in cells:
            for seed in range(2):
                trajectories.append({
                    **cell.public_dict(),
                    "target_id": sentinel,
                    "environment_seed": seed,
                    "normalized_aurc": (
                        0.5
                        + 0.004 * seed
                        + 0.002 * sentinel_index
                        + sum(
                            0.008 * cell.coded_levels[factor]
                            for factor in SENSITIVITY_FACTORS
                        )
                    ),
                })

    effects = _analyze_completed_design(
        trajectories,
        sentinel_ids=sentinels,
        design=design,
    )
    assert len(effects) == 25
    assert {(row["sentinel_id"], row["factor"]) for row in effects} == {
        (sentinel, factor)
        for sentinel in sentinels
        for factor in SENSITIVITY_FACTORS
    }
    assert {row["holm_family_size"] for row in effects} == {25}
    assert {row["holm_family"] for row in effects} == {
        "25_prespecified_sentinel_by_factor_main_effects"
    }
    assert all(row["confirmatory"] is False for row in effects)


def test_call_order_is_separate_and_not_in_global_holm() -> None:
    design, _ = load_sensitivity_design(DEFAULT_DESIGN)
    diagnostic = design["call_order_diagnostic"]
    assert diagnostic["enabled"] is False
    assert diagnostic["included_in_resolution_v"] is False
    assert diagnostic["included_in_global_holm_family"] is False


def test_development_subset_emits_no_confirmatory_pvalues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    design, conditions = load_sensitivity_design(DEFAULT_DESIGN)
    sentinel_ids = [row["target_id"] for row in design["sentinels"]]
    judge = "judge.route"
    target_ids = [*sentinel_ids, judge]
    registry = {
        "schema_version": 1,
        "targets": [{
            "id": target_id, "provider": "inference_hub",
            "upstream_provider": target_id.split("/", 1)[0],
            "model": target_id.rsplit("/", 1)[-1],
        } for target_id in target_ids],
    }

    def compat_row(target_id: str) -> dict[str, Any]:
        route = target_id
        controls = ["seed", "temperature", "top_p", "structured_response"]
        profile = {
            "attempt_id": f"probe-{target_id}", "controls": controls,
            "profile_id": f"profile-{target_id}",
            "request_sha256": hashlib.sha256(target_id.encode()).hexdigest(),
            "resumed_from_ledger": False, "status": "passed",
            "validation_source": "execution_profile_probe",
        }
        return {
            "target_id": target_id, "model": target_id.rsplit("/", 1)[-1],
            "candidate_count": 1, "frozen_candidate_order": [route],
            "selected_execution_candidate": route,
            "selected_execution_profile": {**profile, "route": route},
            "selection_basis": "first_execution_compatible_in_reconciliation_frozen_order",
            "status": "execution_candidate_selected",
            "candidates": [{
                "route": route, "candidate_index": 0, "max_tokens": 64,
                "execution_compatible": True,
                "selected_execution_profile": profile,
            }],
        }

    compatibility = {
        "schema_version": 2,
        "artifact_type": "inference_hub_provider_compatibility",
        "endpoint": ENDPOINT, "registry_sha256": _sha256_json(registry),
        "targets": [compat_row(target_id) for target_id in target_ids],
        "target_count": len(target_ids), "selected_count": len(target_ids),
        "unresolved_count": 0,
    }
    compatibility["evidence_sha256"] = _sha256_json(compatibility)
    panel = {
        "schema_version": 1, "panel_id": "sensitivity-test-panel",
        "judge_target_id": judge, "subject_target_ids": sentinel_ids,
        "part2": {
            "society_size": 5, "days": 12, "independent_trajectories": 12,
            "resource_capacity": 50, "private_gain_for_option_b": 2,
            "reserve_cost_for_option_b": 2, "common_environment_seeds": True,
        },
    }
    panel_path = tmp_path / "panel.json"
    compatibility_path = tmp_path / "compatibility.json"
    registry_path = tmp_path / "registry.json"
    for path, payload in (
        (panel_path, panel), (compatibility_path, compatibility),
        (registry_path, registry),
    ):
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def fast_trajectory(**kwargs: Any) -> dict[str, Any]:
        subject = kwargs["subject"]
        contract = kwargs["contract"]
        return {
            "target_id": subject["target_id"],
            "upstream_provider": subject["upstream_provider"],
            "model": subject["model"], "trajectory_index": kwargs["trajectory_index"],
            "environment_seed_index": kwargs["trajectory_index"],
            "environment_seed": kwargs["environment_seed"],
            "operationally_eligible": True, "scheduled_agent_days": 1,
            "responses_received": 1, "invalid_count": 0,
            "identity_mismatch_count": 0, "transport_failure_count": 0,
            "restraint_count": 1, "overuse_count": 0, "restraint_rate": 1.0,
            "aurc": 1.0, "aupc": 1.0, "reserve_nondepletion": True,
            "final_reserve": contract.capacity, "final_population": contract.society_size,
            "population_retention": 1.0, "cumulative_private_payoff": 1,
            "cumulative_group_payoff": 0,
        }

    monkeypatch.setattr(sensitivity_module, "_run_trajectory", fast_trajectory)
    output = tmp_path / "out"
    manifest = run_sensitivity_campaign(
        panel_path=panel_path, design_path=DEFAULT_DESIGN,
        compatibility_path=compatibility_path, registry_path=registry_path,
        output_dir=output, client=FakeClient(), selected_ids=[sentinel_ids[0]],
        selected_cell_ids=[conditions[0].cell_id], trajectory_limit=1,
        campaign_workers=1, participant_workers=1, max_attempts=1,
        initial_backoff_seconds=0, development_subset=True,
    )
    effects = json.loads((output / "sanitized/main_effects.json").read_text())
    call_order = json.loads(
        (output / "sanitized/call_order_diagnostic.json").read_text()
    )
    assert manifest["complete"] is True
    assert manifest["summary"]["confirmatory_analysis_complete"] is False
    assert effects["analysis_status"] == "not_run_incomplete_development_subset"
    assert effects["rows"] == []
    assert call_order["included_in_global_holm_family"] is False
    assert call_order["rows"] == []
    sanitized_text = "".join(
        path.read_text(encoding="utf-8")
        for path in (output / "sanitized").glob("*.json")
    )
    for private_field in (
        "raw_response", "prompt_text", "request_body", "requested_route", "reasoning",
    ):
        assert private_field not in sanitized_text

    resumed = run_sensitivity_campaign(
        panel_path=panel_path, design_path=DEFAULT_DESIGN,
        compatibility_path=compatibility_path, registry_path=registry_path,
        output_dir=output, client=FakeClient(), selected_ids=[sentinel_ids[0]],
        selected_cell_ids=[conditions[0].cell_id], trajectory_limit=1,
        campaign_workers=1, participant_workers=1, max_attempts=1,
        initial_backoff_seconds=0, development_subset=True, resume=True,
    )
    assert resumed["complete"] is True
    assert resumed["resume_count"] == 1


def test_unsafe_provider_concurrency_fails_closed() -> None:
    class UnsafeClient:
        rate_limit_contract = {"provider_concurrency": 4}

    with pytest.raises(
        InferenceHubPart2SensitivityError,
        match="one to three in-flight requests per provider",
    ):
        _rate_limit_contract(UnsafeClient())


def test_design_rejects_changed_factor_levels(tmp_path: Path) -> None:
    design = json.loads(DEFAULT_DESIGN.read_text(encoding="utf-8"))
    design["factors"]["capacity_per_initial_agent"] = [50, 75]
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(design) + "\n", encoding="utf-8")

    with pytest.raises(InferenceHubPart2SensitivityError, match="frozen profile"):
        load_sensitivity_design(path)
