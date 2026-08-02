from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace

import pytest

from experiments.part1.confirmatory_design import (
    COUNTERBALANCES,
    DOMAINS,
    EXPECTED_ROOT_COUNT,
    GAMES,
    PLACEHOLDER_MODEL,
    PLACEHOLDER_PROVIDER,
    PRIVATE_ADVANTAGE,
    ROLE_FRAME_IDS,
    WELFARE_PRESERVING,
    ParseReason,
    ParsedChoice,
    ParsedOutcome,
    ProductionGateError,
    RetryDecision,
    build_demand_cue_control_schedule,
    build_draft_bank,
    build_primary_schedule,
    build_role_schedule,
    freeze_primary_plan,
    parse_exact_final_token,
    rehash_root,
    render_prompt,
    retry_decision,
    select_role_subset,
    validate_bank,
    validate_primary_schedule,
    validate_role_schedule,
    welfare_preserving_choice,
)


@pytest.fixture(scope="module")
def draft_bank():
    return build_draft_bank()


@pytest.fixture(scope="module")
def primary_schedule(draft_bank):
    return build_primary_schedule(draft_bank, base_seed=20260801)


def test_draft_bank_has_384_distinct_balanced_neutral_roots(draft_bank) -> None:
    report = validate_bank(draft_bank, production=False)

    assert report.valid, report.issues
    assert len(draft_bank) == EXPECTED_ROOT_COUNT == 384
    assert len({root.root_id for root in draft_bank}) == 384
    assert len({root.semantic_cluster_id for root in draft_bank}) == 384
    assert Counter((root.game, root.domain) for root in draft_bank) == Counter(
        {(game, domain): 32 for game in GAMES for domain in DOMAINS}
    )
    assert {root.domain for root in draft_bank} == set(DOMAINS)
    assert all(root.mapping_review.moral_confounds_status == "NOT_REVIEWED" for root in draft_bank)
    assert all(root.mapping_review.welfare_mapping_status == "NOT_REVIEWED" for root in draft_bank)


def test_deterministic_draft_is_explicitly_refused_by_production_gates(
    draft_bank,
    primary_schedule,
) -> None:
    report = validate_bank(draft_bank)

    assert not report.valid
    assert {
        "draft_authorship_not_production_eligible",
        "moral_mapping_review_incomplete",
        "welfare_mapping_review_incomplete",
        "missing_reviewer_identity",
        "nonunanimous_content_approval",
        "incomplete_content_approval",
        "stale_content_approval",
        "missing_review_timestamp",
    } <= report.codes
    with pytest.raises(ProductionGateError):
        build_primary_schedule(
            draft_bank,
            base_seed=1,
            requested_provider="provider",
            requested_model="exact/model-route",
            production=True,
        )
    with pytest.raises(ProductionGateError):
        freeze_primary_plan(draft_bank, primary_schedule)


def test_semantic_duplicate_screen_catches_cloned_content_with_different_ids(
    draft_bank,
) -> None:
    source = draft_bank[0]
    target = draft_bank[1]
    cloned = rehash_root(
        replace(
            target,
            context=source.context,
            actions=source.actions,
            objective_outcomes=source.objective_outcomes,
        )
    )
    tampered = list(draft_bank)
    tampered[1] = cloned

    report = validate_bank(tampered, production=False)

    assert "duplicate_semantic_text" in report.codes


def test_semantic_facets_and_template_reuse_are_independent_gates(draft_bank) -> None:
    duplicate_facets = rehash_root(
        replace(
            draft_bank[1],
            semantic_facets=draft_bank[0].semantic_facets,
            semantic_cluster_id=draft_bank[0].semantic_cluster_id,
        )
    )
    tampered = list(draft_bank)
    tampered[1] = duplicate_facets
    report = validate_bank(tampered, production=False)
    assert "duplicate_semantic_facets" in report.codes
    assert "duplicate_semantic_cluster_id" in report.codes

    one_cell = [
        index
        for index, root in enumerate(draft_bank)
        if (root.game, root.domain) == (GAMES[0], DOMAINS[0])
    ]
    templated = list(draft_bank)
    for index in one_cell:
        templated[index] = rehash_root(
            replace(templated[index], wording_family_id="single-reused-template")
        )
    template_report = validate_bank(templated, production=False)
    assert "excessive_wording_template_reuse" in template_report.codes


def test_crime_and_morally_loaded_cooperation_content_fails_closed(draft_bank) -> None:
    contaminated = rehash_root(
        replace(
            draft_bank[0],
            context=draft_bank[0].context
            + " The participants use the allocation to support an illegal fraud scheme.",
        )
    )
    tampered = list(draft_bank)
    tampered[0] = contaminated

    report = validate_bank(tampered, production=False)

    assert "forbidden_moral_or_crime_confound" in report.codes


def test_objective_payoff_reversal_is_detected_even_if_content_is_rehashed(
    draft_bank,
) -> None:
    root = draft_bank[0]
    reversed_order = replace(
        root.payoff_ordering,
        temptation=root.payoff_ordering.mutual_welfare_preserving - 1,
    )
    tampered_root = rehash_root(replace(root, payoff_ordering=reversed_order))
    tampered = list(draft_bank)
    tampered[0] = tampered_root

    report = validate_bank(tampered, production=False)

    assert "invalid_payoff_ordering" in report.codes
    assert "outcome_payoff_mismatch" in report.codes


def test_four_counterbalances_cross_label_and_position_without_changing_semantics(
    draft_bank,
) -> None:
    root = draft_bank[0]
    observed = {
        (
            counterbalance.welfare_preserving_label,
            counterbalance.welfare_preserving_position,
        )
        for counterbalance in COUNTERBALANCES
    }
    assert observed == {("X", 1), ("Y", 1), ("X", 2), ("Y", 2)}

    for counterbalance in COUNTERBALANCES:
        prompt = render_prompt(
            root,
            counterbalance_id=counterbalance.counterbalance_id,
        )
        options = prompt.split("Options (display order is arbitrary):\n", 1)[1].splitlines()
        first_option, second_option = options[:2]
        preserving_description = root.actions[0].description
        if counterbalance.welfare_preserving_position == 1:
            assert preserving_description in first_option
        else:
            assert preserving_description in second_option
        assert WELFARE_PRESERVING not in prompt
        assert PRIVATE_ADVANTAGE not in prompt
    assert root.objective_outcomes == draft_bank[0].objective_outcomes


def test_primary_schedule_is_full_latin_square_and_hash_bound(
    draft_bank,
    primary_schedule,
) -> None:
    report = validate_primary_schedule(
        primary_schedule,
        draft_bank,
        production=False,
    )

    assert report.valid, report.issues
    assert len(primary_schedule) == 384 * 8 == 3072
    by_root = defaultdict(list)
    for trial in primary_schedule:
        by_root[trial.root_id].append(trial)
    for trials in by_root.values():
        assert Counter(item.generation_block for item in trials) == Counter(range(8))
        assert Counter(item.counterbalance_id for item in trials) == Counter(
            {item.counterbalance_id: 2 for item in COUNTERBALANCES}
        )

    by_block_cell = Counter(
        (
            trial.generation_block,
            trial.game,
            trial.domain,
            trial.counterbalance_id,
        )
        for trial in primary_schedule
    )
    assert set(by_block_cell.values()) == {8}
    assert len({trial.generation_settings.generation_seed for trial in primary_schedule}) == len(
        primary_schedule
    )
    assert build_primary_schedule(draft_bank, base_seed=20260801) == primary_schedule


def test_schedule_validator_rejects_prompt_seed_assignment_and_coverage_tampering(
    draft_bank,
    primary_schedule,
) -> None:
    prompt_tampered = list(primary_schedule)
    prompt_tampered[0] = replace(
        prompt_tampered[0],
        prompt_text=prompt_tampered[0].prompt_text + " ",
    )
    assert "prompt_hash_or_render_mismatch" in validate_primary_schedule(
        prompt_tampered,
        draft_bank,
        production=False,
    ).codes

    seed_tampered = list(primary_schedule)
    seed_tampered[0] = replace(
        seed_tampered[0],
        generation_settings=replace(
            seed_tampered[0].generation_settings,
            generation_seed=seed_tampered[0].generation_settings.generation_seed + 1,
        ),
    )
    assert "generation_seed_mismatch" in validate_primary_schedule(
        seed_tampered,
        draft_bank,
        production=False,
    ).codes

    missing = primary_schedule[:-1]
    missing_codes = validate_primary_schedule(
        missing,
        draft_bank,
        production=False,
    ).codes
    assert "wrong_primary_schedule_size" in missing_codes
    assert "invalid_root_block_coverage" in missing_codes


def test_schedule_schema_retains_requested_and_returned_provenance_placeholders(
    draft_bank,
    primary_schedule,
) -> None:
    trial = primary_schedule[0]
    assert trial.provenance.requested_provider == PLACEHOLDER_PROVIDER
    assert trial.provenance.requested_model == PLACEHOLDER_MODEL
    assert trial.provenance.returned_model is None
    assert trial.provenance.request_id is None
    assert trial.provenance.finish_reason is None
    assert trial.prompt_hash
    assert trial.root_content_hash
    assert trial.welfare_mapping == draft_bank[0].welfare_mapping
    assert trial.payoff_ordering == draft_bank[0].payoff_ordering

    production_report = validate_primary_schedule(
        primary_schedule,
        draft_bank,
        production=True,
    )
    assert "requested_model_provenance_unset" in production_report.codes


@pytest.mark.parametrize(
    ("response", "finish_reason", "truncated", "outcome", "reason"),
    (
        ("X", "stop", False, ParsedOutcome.X, ParseReason.EXACT_FINAL_TOKEN),
        ("Reasoning text.\nY\n", "end_turn", False, ParsedOutcome.Y, ParseReason.EXACT_FINAL_TOKEN),
        ("I choose X", "stop", False, ParsedOutcome.INVALID, ParseReason.MALFORMED_FINAL_TOKEN),
        ("X.", "stop", False, ParsedOutcome.INVALID, ParseReason.MALFORMED_FINAL_TOKEN),
        ("```\nX\n```", "stop", False, ParsedOutcome.INVALID, ParseReason.MALFORMED_FINAL_TOKEN),
        ("Reasoning text.\n X", "stop", False, ParsedOutcome.INVALID, ParseReason.MALFORMED_FINAL_TOKEN),
        ("", "stop", False, ParsedOutcome.INVALID, ParseReason.EMPTY),
        ("X", "length", False, ParsedOutcome.INVALID, ParseReason.TRUNCATED),
        ("X", "stop", True, ParsedOutcome.INVALID, ParseReason.TRUNCATED),
        ("X", "stop", None, ParsedOutcome.INVALID, ParseReason.MISSING_TRUNCATION_STATUS),
        ("X", None, False, ParsedOutcome.INVALID, ParseReason.UNRECOGNIZED_FINISH_REASON),
        ("X", "unknown", False, ParsedOutcome.INVALID, ParseReason.UNRECOGNIZED_FINISH_REASON),
    ),
)
def test_exact_final_token_parser_fails_closed(
    response,
    finish_reason,
    truncated,
    outcome,
    reason,
) -> None:
    parsed = parse_exact_final_token(
        response,
        finish_reason=finish_reason,
        truncated=truncated,
    )

    assert parsed.outcome == outcome
    assert parsed.reason == reason


def test_invalid_and_truncated_semantic_responses_are_never_retried() -> None:
    invalid = parse_exact_final_token(
        "I choose X",
        finish_reason="stop",
        truncated=False,
    )
    truncated = parse_exact_final_token(
        "X",
        finish_reason="length",
        truncated=True,
    )
    valid = parse_exact_final_token("Y", finish_reason="stop", truncated=False)

    assert retry_decision(transport_failure=False, parsed_choice=invalid) == (
        RetryDecision.RETAIN_INVALID_NO_RETRY
    )
    assert retry_decision(transport_failure=False, parsed_choice=truncated) == (
        RetryDecision.RETAIN_INVALID_NO_RETRY
    )
    assert retry_decision(transport_failure=False, parsed_choice=valid) == (
        RetryDecision.COMPLETE_NO_RETRY
    )
    assert retry_decision(transport_failure=True, parsed_choice=None) == (
        RetryDecision.RETRY_TRANSPORT
    )
    with pytest.raises(ValueError):
        retry_decision(transport_failure=True, parsed_choice=invalid)


def test_choice_scoring_uses_counterbalance_mapping() -> None:
    parsed_x = ParsedChoice(ParsedOutcome.X, ParseReason.EXACT_FINAL_TOKEN, "X")
    parsed_invalid = ParsedChoice(ParsedOutcome.INVALID, ParseReason.EMPTY, None)

    assert welfare_preserving_choice(parsed_x, counterbalance_id="CB_X_FIRST") is True
    assert welfare_preserving_choice(parsed_x, counterbalance_id="CB_Y_FIRST") is False
    assert welfare_preserving_choice(parsed_invalid, counterbalance_id="CB_X_FIRST") is None


def test_secondary_role_subset_is_96_root_stratified_and_neutrally_worded(
    draft_bank,
) -> None:
    subset = select_role_subset(draft_bank)
    schedule = build_role_schedule(draft_bank, base_seed=41)
    report = validate_role_schedule(
        schedule,
        draft_bank,
        production=False,
    )

    assert len(subset) == 96
    assert Counter((root.game, root.domain) for root in subset) == Counter(
        {(game, domain): 8 for game in GAMES for domain in DOMAINS}
    )
    assert len(schedule) == 96 * 3 * 4 == 1152
    assert report.valid, report.issues
    assert {trial.frame_id for trial in schedule} == set(ROLE_FRAME_IDS)
    assert not any(trial.positive_demand_cue_control for trial in schedule)
    assert not any(
        "maximizing their own immediate personal payoff" in trial.prompt_text
        for trial in schedule
    )


def test_legacy_immediate_payoff_wording_is_separate_labeled_positive_control(
    draft_bank,
) -> None:
    controls = build_demand_cue_control_schedule(draft_bank, base_seed=43)
    report = validate_role_schedule(
        controls,
        draft_bank,
        positive_control=True,
        production=False,
    )

    assert len(controls) == 96 * 4 == 384
    assert report.valid, report.issues
    assert all(trial.positive_demand_cue_control for trial in controls)
    assert all(
        "Positive demand-cue control" in trial.prompt_text for trial in controls
    )
    assert all(
        "maximizing their own immediate personal payoff" in trial.prompt_text
        for trial in controls
    )


def test_role_validator_rejects_unlabeled_legacy_demand_cue(draft_bank) -> None:
    controls = list(build_demand_cue_control_schedule(draft_bank, base_seed=47))
    controls[0] = replace(controls[0], positive_demand_cue_control=False)

    report = validate_role_schedule(
        controls,
        draft_bank,
        positive_control=True,
        production=False,
    )

    assert "demand_cue_label_mismatch" in report.codes
    assert "legacy_cue_outside_positive_control" in report.codes
