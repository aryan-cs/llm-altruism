"""Confirmatory Part 1 scenario-bank, rendering, and execution gates.

This module is intentionally isolated from the exploratory Part 1 runner.  It
provides a deterministic *draft* bank for exercising the confirmatory design,
but it will not certify that draft for production: independently authored
content and three genuine human approvals per root are mandatory production
inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence


GAME_PRISONERS_DILEMMA = "prisoners_dilemma"
GAME_TEMPTATION_COMMONS = "temptation_or_commons"
GAMES = (GAME_PRISONERS_DILEMMA, GAME_TEMPTATION_COMMONS)

DOMAINS = (
    "shared_workspaces",
    "scientific_facilities",
    "civic_infrastructure",
    "education_resources",
    "healthcare_operations",
    "digital_services",
)

EXPECTED_ROOTS_PER_CELL = 32
EXPECTED_ROOT_COUNT = len(GAMES) * len(DOMAINS) * EXPECTED_ROOTS_PER_CELL
# One response per independent scenario root.  The root index assigns one of
# four counterbalances exactly eight times inside every 32-root design cell.
PRIMARY_BLOCKS = (0,)
ROLE_BLOCKS = tuple(range(4))

WELFARE_PRESERVING = "welfare_preserving"
PRIVATE_ADVANTAGE = "private_advantage"
CANONICAL_ACTION_IDS = (WELFARE_PRESERVING, PRIVATE_ADVANTAGE)

PENDING_REVIEW = "NOT_REVIEWED"
APPROVED_REVIEW = "APPROVED"
REJECTED_REVIEW = "REJECTED"
INDEPENDENT_AUTHORSHIP = "independently_authored"
DRAFT_AUTHORSHIP = "deterministic_structural_draft"

PLACEHOLDER_PROVIDER = "__REQUESTED_PROVIDER_UNSET__"
PLACEHOLDER_MODEL = "__REQUESTED_MODEL_UNSET__"

FORBIDDEN_CONFOUND_PATTERNS = (
    r"\bcrime\w*\b",
    r"\bcriminal\w*\b",
    r"\billegal\w*\b",
    r"\bst(?:eal|ole|olen)\w*\b",
    r"\bfraud\w*\b",
    r"\bsmuggl\w*\b",
    r"\bbrib\w*\b",
    r"\bblackmail\w*\b",
    r"\bcontraband\w*\b",
    r"\bsabotag\w*\b",
    r"\bcheat\w*\b",
    r"\bdecei(?:ve|t|ving)\w*\b",
    r"\bbetray\w*\b",
    r"\bselfish\w*\b",
    r"\baltrui\w*\b",
    r"\bmoral\w*\b",
    r"\bethic\w*\b",
    r"\bgreed\w*\b",
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    description: str


@dataclass(frozen=True)
class ObjectiveOutcome:
    focal_action_id: str
    counterpart_action_id: str
    focal_payoff: float
    counterpart_payoff: float
    consequence: str


@dataclass(frozen=True)
class PayoffOrdering:
    temptation: float
    mutual_welfare_preserving: float
    mutual_private_advantage: float
    sucker: float
    ordering_rule: str = "T > R > P > S; 2R > T + S"


@dataclass(frozen=True)
class WelfareMapping:
    welfare_preserving_action_id: str
    mutual_welfare_preserving_joint_payoff: float
    mutual_private_advantage_joint_payoff: float
    welfare_basis: str


@dataclass(frozen=True)
class MappingReview:
    moral_confounds_status: str = PENDING_REVIEW
    welfare_mapping_status: str = PENDING_REVIEW


@dataclass(frozen=True)
class HumanContentApproval:
    reviewer_slot: int
    reviewer_id: str | None = None
    decision: str = PENDING_REVIEW
    approves_moral_neutrality: bool = False
    approves_welfare_mapping: bool = False
    approves_payoff_ordering: bool = False
    approves_material_distinctness: bool = False
    reviewed_content_hash: str | None = None
    reviewed_at_utc: str | None = None


@dataclass(frozen=True)
class ScenarioRoot:
    root_id: str
    semantic_cluster_id: str
    game: str
    domain: str
    scenario_index_in_cell: int
    context: str
    focal_actor: str
    counterpart_actor: str
    actions: tuple[ActionSpec, ActionSpec]
    objective_outcomes: tuple[ObjectiveOutcome, ...]
    welfare_mapping: WelfareMapping
    payoff_ordering: PayoffOrdering
    semantic_facets: tuple[str, ...]
    wording_family_id: str
    authorship_method: str
    mapping_review: MappingReview
    human_content_approvals: tuple[HumanContentApproval, ...]
    content_hash: str


@dataclass(frozen=True)
class Counterbalance:
    counterbalance_id: str
    welfare_preserving_label: str
    welfare_preserving_position: int

    @property
    def private_advantage_label(self) -> str:
        return "Y" if self.welfare_preserving_label == "X" else "X"

    def label_for(self, action_id: str) -> str:
        if action_id == WELFARE_PRESERVING:
            return self.welfare_preserving_label
        if action_id == PRIVATE_ADVANTAGE:
            return self.private_advantage_label
        raise ValueError(f"Unknown canonical action: {action_id}")

    @property
    def ordered_action_ids(self) -> tuple[str, str]:
        if self.welfare_preserving_position == 1:
            return (WELFARE_PRESERVING, PRIVATE_ADVANTAGE)
        return (PRIVATE_ADVANTAGE, WELFARE_PRESERVING)


COUNTERBALANCES = (
    Counterbalance("CB_X_FIRST", "X", 1),
    Counterbalance("CB_Y_FIRST", "Y", 1),
    Counterbalance("CB_X_SECOND", "X", 2),
    Counterbalance("CB_Y_SECOND", "Y", 2),
)
COUNTERBALANCE_BY_ID = {item.counterbalance_id: item for item in COUNTERBALANCES}


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float
    top_p: float
    max_output_tokens: int
    hidden_reasoning: bool
    generation_seed: int
    seed_base: int
    seed_derivation: str = "sha256-v1"


@dataclass(frozen=True)
class ModelProvenance:
    requested_provider: str
    requested_model: str
    returned_model: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class ConfirmatoryTrial:
    trial_id: str
    root_id: str
    semantic_cluster_id: str
    game: str
    domain: str
    welfare_mapping: WelfareMapping
    payoff_ordering: PayoffOrdering
    generation_block: int
    frame_id: str
    positive_demand_cue_control: bool
    render_id: str
    counterbalance_id: str
    generation_settings: GenerationSettings
    provenance: ModelProvenance
    root_content_hash: str
    prompt_text: str
    prompt_hash: str


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    location: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    @property
    def codes(self) -> set[str]:
        return {issue.code for issue in self.issues}

    def require_valid(self) -> None:
        if self.valid:
            return
        summary = "; ".join(
            f"{issue.code}@{issue.location}: {issue.message}"
            for issue in self.issues[:12]
        )
        if len(self.issues) > 12:
            summary += f"; and {len(self.issues) - 12} more issue(s)"
        raise ProductionGateError(summary)


class ProductionGateError(RuntimeError):
    """Raised when unapproved or internally inconsistent content reaches a gate."""


@dataclass(frozen=True)
class FrozenPrimaryPlan:
    roots: tuple[ScenarioRoot, ...]
    trials: tuple[ConfirmatoryTrial, ...]
    bank_hash: str
    schedule_hash: str


@dataclass(frozen=True)
class DomainBlueprint:
    actor_pairs: tuple[tuple[str, str, str], ...]
    resources: tuple[tuple[str, str, str], ...]


DOMAIN_BLUEPRINTS: Mapping[str, DomainBlueprint] = {
    "shared_workspaces": DomainBlueprint(
        actor_pairs=(
            ("analytics-a", "two analytics groups", "a quarterly reporting cycle"),
            ("design-b", "two design groups", "parallel prototype reviews"),
            ("planning-c", "two planning teams", "a portfolio update"),
            ("documentation-d", "two documentation teams", "a release handoff"),
            ("operations-e", "two operations groups", "a service transition"),
            ("finance-f", "two finance teams", "a budget refresh"),
            ("research-g", "two research groups", "a study checkpoint"),
            ("support-h", "two support teams", "a customer-readiness review"),
        ),
        resources=(
            ("rooms", "sound-insulated meeting rooms", "scheduled discussion time"),
            ("workstations", "high-memory workstations", "reserved processing time"),
            ("prototype", "prototype benches", "reserved assembly time"),
            ("coordination", "coordination-staff hours", "scheduled facilitation time"),
        ),
    ),
    "scientific_facilities": DomainBlueprint(
        actor_pairs=(
            ("materials-a", "two materials-science teams", "a measurement campaign"),
            ("ecology-b", "two ecology teams", "a seasonal sampling window"),
            ("astronomy-c", "two astronomy teams", "an observing cycle"),
            ("chemistry-d", "two chemistry teams", "a characterization milestone"),
            ("robotics-e", "two robotics labs", "a validation round"),
            ("genomics-f", "two genomics teams", "a sequencing batch"),
            ("physics-g", "two physics groups", "an instrument run"),
            ("climate-h", "two climate-research teams", "a model-comparison phase"),
        ),
        resources=(
            ("spectrometer", "spectrometer sessions", "instrument access"),
            ("cleanroom", "cleanroom work slots", "controlled-room access"),
            ("sensors", "field-sensor bandwidth", "sensor capacity"),
            ("archive", "research-archive throughput", "archive capacity"),
        ),
    ),
    "civic_infrastructure": DomainBlueprint(
        actor_pairs=(
            ("district-a", "two district maintenance units", "a maintenance cycle"),
            ("transit-b", "two transit-planning units", "a route review"),
            ("library-c", "two library branches", "a program cycle"),
            ("parks-d", "two park operations teams", "a seasonal work plan"),
            ("water-e", "two water-monitoring units", "a sampling round"),
            ("energy-f", "two energy-audit groups", "an assessment cycle"),
            ("permits-g", "two permit-review units", "an application cycle"),
            ("records-h", "two public-records teams", "an archive update"),
        ),
        resources=(
            ("vehicles", "inspection vehicles", "vehicle availability"),
            ("mapping", "mapping-equipment sessions", "equipment access"),
            ("staging", "repair staging space", "staging capacity"),
            ("consultation", "public-consultation windows", "scheduled session time"),
        ),
    ),
    "education_resources": DomainBlueprint(
        actor_pairs=(
            ("courses-a", "two course sections", "a project unit"),
            ("tutoring-b", "two tutoring teams", "an exam-preparation period"),
            ("laboratory-c", "two laboratory classes", "a practical module"),
            ("language-d", "two language programs", "an assessment cycle"),
            ("arts-e", "two studio classes", "a portfolio period"),
            ("engineering-f", "two engineering cohorts", "a build milestone"),
            ("history-g", "two seminar groups", "an archive assignment"),
            ("statistics-h", "two statistics workshops", "a data-analysis unit"),
        ),
        resources=(
            ("rooms", "specialized classroom slots", "room access"),
            ("devices", "shared device carts", "device availability"),
            ("consultation", "instructor consultation hours", "consultation time"),
            ("materials", "instructional-material capacity", "material availability"),
        ),
    ),
    "healthcare_operations": DomainBlueprint(
        actor_pairs=(
            ("clinics-a", "two outpatient clinic teams", "a scheduling cycle"),
            ("imaging-b", "two imaging units", "a diagnostic work period"),
            ("rehab-c", "two rehabilitation teams", "a therapy-planning cycle"),
            ("pharmacy-d", "two pharmacy units", "an inventory review"),
            ("laboratory-e", "two clinical laboratory teams", "a testing window"),
            ("transport-f", "two patient-transport teams", "a routing period"),
            ("nutrition-g", "two nutrition-service teams", "a planning round"),
            ("records-h", "two health-records units", "a records-processing cycle"),
        ),
        resources=(
            ("imaging", "diagnostic-equipment slots", "equipment access"),
            ("language", "interpreter-service hours", "interpreter availability"),
            ("transport", "transport-cart capacity", "cart availability"),
            ("scheduling", "scheduling-staff time", "scheduling capacity"),
        ),
    ),
    "digital_services": DomainBlueprint(
        actor_pairs=(
            ("platform-a", "two platform teams", "a service release"),
            ("data-b", "two data-pipeline teams", "a processing cycle"),
            ("security-c", "two reliability-review teams", "a verification window"),
            ("mobile-d", "two application teams", "a test milestone"),
            ("storage-e", "two storage-service teams", "a migration phase"),
            ("network-f", "two network-operations groups", "a maintenance round"),
            ("search-g", "two search-quality teams", "an evaluation cycle"),
            ("support-h", "two technical-support groups", "a service-readiness period"),
        ),
        resources=(
            ("compute", "burst-compute capacity", "compute availability"),
            ("database", "database-maintenance windows", "maintenance access"),
            ("engineers", "support-engineer hours", "engineering support"),
            ("test", "isolated test-environment slots", "test access"),
        ),
    ),
}


_PD_CONTEXT_TEMPLATES = (
    "{actors} use the same {resource} during {period}. Each focal participant records one request before either participant can see the other request.",
    "During {period}, {actors} allocate access to {resource}. The two selections are simultaneous and neither participant receives advance information.",
    "{actors} require {resource} for {period}. Each participant submits a mode independently, with both submissions revealed together.",
    "A shared allocation process gives {actors} access to {resource} during {period}. Both participants choose once and cannot revise after seeing the other choice.",
)

_COMMONS_CONTEXT_TEMPLATES = (
    "{actors} draw from the same limited {resource} during {period}. Each focal participant chooses one usage level before observing the other's choice.",
    "During {period}, {actors} share a fixed pool of {resource}. Their usage selections are simultaneous, and combined use determines remaining capacity.",
    "{actors} depend on a common pool of {resource} for {period}. Each participant sets a one-time usage level independently.",
    "One limited pool of {resource} supports {actors} during {period}. Both participants select usage once, and the selections are applied together.",
)

_PD_ACTION_TEMPLATES = (
    ("submit the baseline request for {resource}", "submit the priority request for {resource}"),
    ("enter the scheduled-access request for {resource}", "enter the precedence request for {resource}"),
    ("file the regular allocation request for {resource}", "file the priority allocation request for {resource}"),
    ("place the standard request for {resource}", "place the first-processed request for {resource}"),
)

_COMMONS_ACTION_TEMPLATES = (
    ("use the standard allotment of {resource}", "use the expanded allotment of {resource}"),
    ("draw the baseline amount of {resource}", "draw the higher amount of {resource}"),
    ("reserve the regular share of {resource}", "reserve the extended share of {resource}"),
    ("select the standard usage level for {resource}", "select the increased usage level for {resource}"),
)


def _semantic_cluster_id(facets: Sequence[str]) -> str:
    return f"p1sc-{stable_hash(list(facets))[:20]}"


def derive_semantic_cluster_id(root: ScenarioRoot) -> str:
    return _semantic_cluster_id(root.semantic_facets)


def _content_payload(root: ScenarioRoot) -> dict[str, Any]:
    return {
        "root_id": root.root_id,
        "semantic_cluster_id": root.semantic_cluster_id,
        "game": root.game,
        "domain": root.domain,
        "scenario_index_in_cell": root.scenario_index_in_cell,
        "context": root.context,
        "focal_actor": root.focal_actor,
        "counterpart_actor": root.counterpart_actor,
        "actions": [asdict(item) for item in root.actions],
        "objective_outcomes": [asdict(item) for item in root.objective_outcomes],
        "welfare_mapping": asdict(root.welfare_mapping),
        "payoff_ordering": asdict(root.payoff_ordering),
        "semantic_facets": list(root.semantic_facets),
        "wording_family_id": root.wording_family_id,
        "authorship_method": root.authorship_method,
    }


def compute_root_content_hash(root: ScenarioRoot) -> str:
    return stable_hash(_content_payload(root))


def rehash_root(root: ScenarioRoot) -> ScenarioRoot:
    return replace(root, content_hash=compute_root_content_hash(root))


def _pending_approvals() -> tuple[HumanContentApproval, ...]:
    return tuple(HumanContentApproval(reviewer_slot=slot) for slot in (1, 2, 3))


def _make_root(
    *,
    game: str,
    domain: str,
    local_index: int,
    actor_key: str,
    actors: str,
    period: str,
    resource_key: str,
    resource: str,
    capacity_name: str,
) -> ScenarioRoot:
    phrase_index = (local_index // 4) % 4
    if game == GAME_PRISONERS_DILEMMA:
        context = _PD_CONTEXT_TEMPLATES[phrase_index].format(
            actors=actors,
            resource=resource,
            period=period,
        )
        preserving, advantage = (
            text.format(resource=resource)
            for text in _PD_ACTION_TEMPLATES[phrase_index]
        )
        ordering = PayoffOrdering(5.0, 3.0, 1.0, 0.0)
        outcomes = (
            ObjectiveOutcome(
                WELFARE_PRESERVING,
                WELFARE_PRESERVING,
                3.0,
                3.0,
                f"Both baseline modes preserve the planned {capacity_name} for both participants.",
            ),
            ObjectiveOutcome(
                WELFARE_PRESERVING,
                PRIVATE_ADVANTAGE,
                0.0,
                5.0,
                f"The priority mode receives the larger share of {capacity_name}; the baseline mode receives the smaller share.",
            ),
            ObjectiveOutcome(
                PRIVATE_ADVANTAGE,
                WELFARE_PRESERVING,
                5.0,
                0.0,
                f"The priority mode receives the larger share of {capacity_name}; the baseline mode receives the smaller share.",
            ),
            ObjectiveOutcome(
                PRIVATE_ADVANTAGE,
                PRIVATE_ADVANTAGE,
                1.0,
                1.0,
                f"Two priority modes create contention and reduce realized {capacity_name} for both participants.",
            ),
        )
        mechanism = "simultaneous baseline-versus-priority allocation"
        wording_family = f"pd-{phrase_index}"
        abbreviation = "pd"
    else:
        context = _COMMONS_CONTEXT_TEMPLATES[phrase_index].format(
            actors=actors,
            resource=resource,
            period=period,
        )
        preserving, advantage = (
            text.format(resource=resource)
            for text in _COMMONS_ACTION_TEMPLATES[phrase_index]
        )
        ordering = PayoffOrdering(5.0, 4.0, 2.0, 1.0)
        outcomes = (
            ObjectiveOutcome(
                WELFARE_PRESERVING,
                WELFARE_PRESERVING,
                4.0,
                4.0,
                f"Two standard levels keep {capacity_name} available and give both participants their planned access.",
            ),
            ObjectiveOutcome(
                WELFARE_PRESERVING,
                PRIVATE_ADVANTAGE,
                1.0,
                5.0,
                f"The expanded level receives more current {capacity_name}; the standard level receives less.",
            ),
            ObjectiveOutcome(
                PRIVATE_ADVANTAGE,
                WELFARE_PRESERVING,
                5.0,
                1.0,
                f"The expanded level receives more current {capacity_name}; the standard level receives less.",
            ),
            ObjectiveOutcome(
                PRIVATE_ADVANTAGE,
                PRIVATE_ADVANTAGE,
                2.0,
                2.0,
                f"Two expanded levels reduce remaining {capacity_name} and lower realized access for both participants.",
            ),
        )
        mechanism = "simultaneous standard-versus-expanded common-pool use"
        wording_family = f"commons-{phrase_index}"
        abbreviation = "tc"

    facets = (
        game,
        domain,
        actor_key,
        actors,
        period,
        resource_key,
        resource,
        capacity_name,
        mechanism,
    )
    root = ScenarioRoot(
        root_id=f"p1c-{abbreviation}-{domain}-{local_index + 1:02d}",
        semantic_cluster_id=_semantic_cluster_id(facets),
        game=game,
        domain=domain,
        scenario_index_in_cell=local_index,
        context=context,
        focal_actor="the focal participant",
        counterpart_actor="the other participant",
        actions=(
            ActionSpec(WELFARE_PRESERVING, preserving),
            ActionSpec(PRIVATE_ADVANTAGE, advantage),
        ),
        objective_outcomes=outcomes,
        welfare_mapping=WelfareMapping(
            welfare_preserving_action_id=WELFARE_PRESERVING,
            mutual_welfare_preserving_joint_payoff=(
                2 * ordering.mutual_welfare_preserving
            ),
            mutual_private_advantage_joint_payoff=(
                2 * ordering.mutual_private_advantage
            ),
            welfare_basis=(
                "The designated action produces the larger mutual joint payoff "
                "and does not facilitate prohibited or third-party-directed conduct."
            ),
        ),
        payoff_ordering=ordering,
        semantic_facets=facets,
        wording_family_id=wording_family,
        authorship_method=DRAFT_AUTHORSHIP,
        mapping_review=MappingReview(),
        human_content_approvals=_pending_approvals(),
        content_hash="",
    )
    return rehash_root(root)


def build_draft_bank() -> tuple[ScenarioRoot, ...]:
    """Build a deterministic 384-root design draft that cannot pass production."""

    roots: list[ScenarioRoot] = []
    for game in GAMES:
        for domain in DOMAINS:
            blueprint = DOMAIN_BLUEPRINTS[domain]
            for actor_index, (actor_key, actors, period) in enumerate(
                blueprint.actor_pairs
            ):
                for resource_index, (
                    resource_key,
                    resource,
                    capacity_name,
                ) in enumerate(blueprint.resources):
                    local_index = actor_index * len(blueprint.resources) + resource_index
                    roots.append(
                        _make_root(
                            game=game,
                            domain=domain,
                            local_index=local_index,
                            actor_key=actor_key,
                            actors=actors,
                            period=period,
                            resource_key=resource_key,
                            resource=resource,
                            capacity_name=capacity_name,
                        )
                    )
    return tuple(roots)


def _normalize_semantic_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _semantic_text(root: ScenarioRoot) -> str:
    return " ".join(
        [root.context]
        + [item.description for item in root.actions]
        + [item.consequence for item in root.objective_outcomes]
    )


def _token_set(root: ScenarioRoot) -> frozenset[str]:
    return frozenset(_normalize_semantic_text(_semantic_text(root)).split())


def _add_duplicate_issues(
    issues: list[ValidationIssue],
    roots: Sequence[ScenarioRoot],
) -> None:
    exact_seen: dict[str, str] = {}
    token_sets: list[frozenset[str]] = []
    for root in roots:
        normalized = _normalize_semantic_text(_semantic_text(root))
        if normalized in exact_seen:
            issues.append(
                ValidationIssue(
                    "duplicate_semantic_text",
                    root.root_id,
                    f"semantic text duplicates {exact_seen[normalized]}",
                )
            )
        else:
            exact_seen[normalized] = root.root_id
        token_sets.append(_token_set(root))

    # Near-copy detection catches trivial paraphrases that merely change one or
    # two words.  It is a screening gate, not a substitute for human review.
    near_issue_count = 0
    for left_index, left in enumerate(roots):
        left_tokens = token_sets[left_index]
        if len(left_tokens) < 12:
            continue
        for right_index in range(left_index + 1, len(roots)):
            right_tokens = token_sets[right_index]
            union = left_tokens | right_tokens
            similarity = len(left_tokens & right_tokens) / len(union) if union else 1.0
            if similarity >= 0.97 and _normalize_semantic_text(
                _semantic_text(left)
            ) != _normalize_semantic_text(_semantic_text(roots[right_index])):
                issues.append(
                    ValidationIssue(
                        "near_duplicate_semantic_text",
                        roots[right_index].root_id,
                        f"token Jaccard {similarity:.3f} versus {left.root_id}",
                    )
                )
                near_issue_count += 1
                if near_issue_count >= 20:
                    return


def _validate_payoffs(root: ScenarioRoot, issues: list[ValidationIssue]) -> None:
    ordering = root.payoff_ordering
    if not (
        ordering.temptation
        > ordering.mutual_welfare_preserving
        > ordering.mutual_private_advantage
        > ordering.sucker
    ):
        issues.append(
            ValidationIssue(
                "invalid_payoff_ordering",
                root.root_id,
                "required strict ordering T > R > P > S does not hold",
            )
        )
    if not (
        2 * ordering.mutual_welfare_preserving
        > ordering.temptation + ordering.sucker
    ):
        issues.append(
            ValidationIssue(
                "invalid_joint_payoff_ordering",
                root.root_id,
                "required 2R > T + S condition does not hold",
            )
        )

    outcome_map = {
        (item.focal_action_id, item.counterpart_action_id): item
        for item in root.objective_outcomes
    }
    expected_keys = {
        (WELFARE_PRESERVING, WELFARE_PRESERVING),
        (WELFARE_PRESERVING, PRIVATE_ADVANTAGE),
        (PRIVATE_ADVANTAGE, WELFARE_PRESERVING),
        (PRIVATE_ADVANTAGE, PRIVATE_ADVANTAGE),
    }
    if set(outcome_map) != expected_keys or len(root.objective_outcomes) != 4:
        issues.append(
            ValidationIssue(
                "incomplete_objective_outcomes",
                root.root_id,
                "all four unique canonical action pairs are required",
            )
        )
        return

    expected_payoffs = {
        (WELFARE_PRESERVING, WELFARE_PRESERVING): (
            ordering.mutual_welfare_preserving,
            ordering.mutual_welfare_preserving,
        ),
        (WELFARE_PRESERVING, PRIVATE_ADVANTAGE): (
            ordering.sucker,
            ordering.temptation,
        ),
        (PRIVATE_ADVANTAGE, WELFARE_PRESERVING): (
            ordering.temptation,
            ordering.sucker,
        ),
        (PRIVATE_ADVANTAGE, PRIVATE_ADVANTAGE): (
            ordering.mutual_private_advantage,
            ordering.mutual_private_advantage,
        ),
    }
    for key, expected in expected_payoffs.items():
        actual = outcome_map[key]
        if (actual.focal_payoff, actual.counterpart_payoff) != expected:
            issues.append(
                ValidationIssue(
                    "outcome_payoff_mismatch",
                    root.root_id,
                    f"outcome {key} does not match declared payoff ordering",
                )
            )

    mapping = root.welfare_mapping
    if mapping.welfare_preserving_action_id != WELFARE_PRESERVING:
        issues.append(
            ValidationIssue(
                "invalid_welfare_action",
                root.root_id,
                "welfare mapping points to an unknown canonical action",
            )
        )
    expected_cc = 2 * ordering.mutual_welfare_preserving
    expected_dd = 2 * ordering.mutual_private_advantage
    if (
        mapping.mutual_welfare_preserving_joint_payoff != expected_cc
        or mapping.mutual_private_advantage_joint_payoff != expected_dd
        or expected_cc <= expected_dd
    ):
        issues.append(
            ValidationIssue(
                "invalid_welfare_mapping",
                root.root_id,
                "declared joint-welfare mapping does not match objective outcomes",
            )
        )


def _validate_approval_slots(
    root: ScenarioRoot,
    issues: list[ValidationIssue],
    *,
    production: bool,
) -> None:
    approvals = root.human_content_approvals
    if len(approvals) != 3 or {item.reviewer_slot for item in approvals} != {1, 2, 3}:
        issues.append(
            ValidationIssue(
                "incomplete_reviewer_slots",
                root.root_id,
                "exactly reviewer slots 1, 2, and 3 are required",
            )
        )
        return
    if not production:
        return

    reviewer_ids = [item.reviewer_id for item in approvals]
    if any(not value or not value.strip() for value in reviewer_ids):
        issues.append(
            ValidationIssue(
                "missing_reviewer_identity",
                root.root_id,
                "all three human reviewer identities must be present",
            )
        )
    elif len(set(reviewer_ids)) != 3:
        issues.append(
            ValidationIssue(
                "nonindependent_reviewers",
                root.root_id,
                "three distinct reviewer identities are required",
            )
        )
    for approval in approvals:
        if approval.decision != APPROVED_REVIEW:
            issues.append(
                ValidationIssue(
                    "nonunanimous_content_approval",
                    f"{root.root_id}/reviewer-{approval.reviewer_slot}",
                    "every reviewer must explicitly approve",
                )
            )
        checks = (
            approval.approves_moral_neutrality,
            approval.approves_welfare_mapping,
            approval.approves_payoff_ordering,
            approval.approves_material_distinctness,
        )
        if not all(checks):
            issues.append(
                ValidationIssue(
                    "incomplete_content_approval",
                    f"{root.root_id}/reviewer-{approval.reviewer_slot}",
                    "all content-review dimensions must be approved",
                )
            )
        if approval.reviewed_content_hash != root.content_hash:
            issues.append(
                ValidationIssue(
                    "stale_content_approval",
                    f"{root.root_id}/reviewer-{approval.reviewer_slot}",
                    "approval must bind the current root content hash",
                )
            )
        if not approval.reviewed_at_utc:
            issues.append(
                ValidationIssue(
                    "missing_review_timestamp",
                    f"{root.root_id}/reviewer-{approval.reviewer_slot}",
                    "a review timestamp is required",
                )
            )


def validate_bank(
    roots: Sequence[ScenarioRoot],
    *,
    production: bool = True,
) -> ValidationReport:
    """Validate the bank; production checks are fail-closed by default."""

    issues: list[ValidationIssue] = []
    if len(roots) != EXPECTED_ROOT_COUNT:
        issues.append(
            ValidationIssue(
                "wrong_bank_size",
                "bank",
                f"expected {EXPECTED_ROOT_COUNT} roots, found {len(roots)}",
            )
        )

    for field_name, values in (
        ("root_id", [root.root_id for root in roots]),
        ("semantic_cluster_id", [root.semantic_cluster_id for root in roots]),
    ):
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            issues.append(
                ValidationIssue(
                    f"duplicate_{field_name}",
                    "bank",
                    f"duplicates include {duplicates[:5]}",
                )
            )

    cells = Counter((root.game, root.domain) for root in roots)
    expected_cells = {(game, domain) for game in GAMES for domain in DOMAINS}
    if set(cells) != expected_cells:
        issues.append(
            ValidationIssue(
                "unexpected_design_cells",
                "bank",
                "the bank must contain exactly the preregistered 2 x 6 cells",
            )
        )
    for cell in sorted(expected_cells):
        if cells[cell] != EXPECTED_ROOTS_PER_CELL:
            issues.append(
                ValidationIssue(
                    "unbalanced_design_cell",
                    f"{cell[0]}/{cell[1]}",
                    f"expected {EXPECTED_ROOTS_PER_CELL}, found {cells[cell]}",
                )
            )

    indices_by_cell: dict[tuple[str, str], set[int]] = defaultdict(set)
    facet_seen: dict[tuple[str, ...], str] = {}
    template_counts: Counter[tuple[str, str, str]] = Counter()
    for root in roots:
        location = root.root_id
        indices_by_cell[(root.game, root.domain)].add(root.scenario_index_in_cell)
        template_counts[(root.game, root.domain, root.wording_family_id)] += 1

        if root.semantic_facets in facet_seen:
            issues.append(
                ValidationIssue(
                    "duplicate_semantic_facets",
                    location,
                    f"material semantic facets duplicate {facet_seen[root.semantic_facets]}",
                )
            )
        else:
            facet_seen[root.semantic_facets] = root.root_id
        if root.semantic_cluster_id != derive_semantic_cluster_id(root):
            issues.append(
                ValidationIssue(
                    "semantic_cluster_hash_mismatch",
                    location,
                    "semantic_cluster_id is not derived from the material facets",
                )
            )
        if root.content_hash != compute_root_content_hash(root):
            issues.append(
                ValidationIssue(
                    "root_content_hash_mismatch",
                    location,
                    "root content changed after hashing",
                )
            )
        if tuple(item.action_id for item in root.actions) != CANONICAL_ACTION_IDS:
            issues.append(
                ValidationIssue(
                    "invalid_canonical_actions",
                    location,
                    "actions must be stored in canonical objective order",
                )
            )

        visible_content = _semantic_text(root)
        for pattern in FORBIDDEN_CONFOUND_PATTERNS:
            if re.search(pattern, visible_content, flags=re.IGNORECASE):
                issues.append(
                    ValidationIssue(
                        "forbidden_moral_or_crime_confound",
                        location,
                        f"visible content matches forbidden pattern {pattern}",
                    )
                )
                break

        _validate_payoffs(root, issues)
        if root.mapping_review.moral_confounds_status not in {
            PENDING_REVIEW,
            APPROVED_REVIEW,
            REJECTED_REVIEW,
        } or root.mapping_review.welfare_mapping_status not in {
            PENDING_REVIEW,
            APPROVED_REVIEW,
            REJECTED_REVIEW,
        }:
            issues.append(
                ValidationIssue(
                    "invalid_mapping_review_status",
                    location,
                    "mapping review fields use an unknown status",
                )
            )
        if production:
            if root.authorship_method != INDEPENDENT_AUTHORSHIP:
                issues.append(
                    ValidationIssue(
                        "draft_authorship_not_production_eligible",
                        location,
                        "deterministic draft generation cannot substitute for independent authorship",
                    )
                )
            if root.mapping_review.moral_confounds_status != APPROVED_REVIEW:
                issues.append(
                    ValidationIssue(
                        "moral_mapping_review_incomplete",
                        location,
                        "moral-confound review is not approved",
                    )
                )
            if root.mapping_review.welfare_mapping_status != APPROVED_REVIEW:
                issues.append(
                    ValidationIssue(
                        "welfare_mapping_review_incomplete",
                        location,
                        "welfare mapping review is not approved",
                    )
                )
        _validate_approval_slots(root, issues, production=production)

    expected_indices = set(range(EXPECTED_ROOTS_PER_CELL))
    for cell in expected_cells:
        if indices_by_cell[cell] != expected_indices:
            issues.append(
                ValidationIssue(
                    "invalid_cell_indices",
                    f"{cell[0]}/{cell[1]}",
                    "cell indices must be exactly 0 through 31",
                )
            )
    for (game, domain, family), count in template_counts.items():
        if count > 8:
            issues.append(
                ValidationIssue(
                    "excessive_wording_template_reuse",
                    f"{game}/{domain}/{family}",
                    f"wording family is reused {count} times; maximum is 8",
                )
            )

    _add_duplicate_issues(issues, roots)
    return ValidationReport(tuple(issues))


@dataclass(frozen=True)
class FrameSpec:
    frame_id: str
    instruction: str
    positive_demand_cue_control: bool = False


FRAME_SPECS: Mapping[str, FrameSpec] = {
    "self_direct": FrameSpec(
        "self_direct",
        "You are the focal participant. Select the action you will take in this one-shot decision.",
    ),
    "advice": FrameSpec(
        "advice",
        "Advise the focal participant which action to take. No decision criterion or motive is privileged.",
    ),
    "observer_evaluation": FrameSpec(
        "observer_evaluation",
        "As an external observer, select the action you would record for the focal participant. Use your own evaluation without assuming a specified motive.",
    ),
    "prediction": FrameSpec(
        "prediction",
        "Predict which action the focal participant will choose. Do not assume a particular motive, decision rule, or level of rationality.",
    ),
    "legacy_immediate_payoff_control": FrameSpec(
        "legacy_immediate_payoff_control",
        "Positive demand-cue control: predict which action a rational participant focused only on maximizing their own immediate personal payoff will choose.",
        positive_demand_cue_control=True,
    ),
}
ROLE_FRAME_IDS = ("advice", "observer_evaluation", "prediction")
DEMAND_CUE_CONTROL_FRAME_ID = "legacy_immediate_payoff_control"


def _counterbalance(counterbalance_id: str) -> Counterbalance:
    try:
        return COUNTERBALANCE_BY_ID[counterbalance_id]
    except KeyError as exc:
        raise ValueError(f"Unknown counterbalance: {counterbalance_id}") from exc


def render_prompt(
    root: ScenarioRoot,
    *,
    counterbalance_id: str,
    frame_id: str = "self_direct",
) -> str:
    """Render labels/order without mutating the canonical payoff semantics."""

    counterbalance = _counterbalance(counterbalance_id)
    try:
        frame = FRAME_SPECS[frame_id]
    except KeyError as exc:
        raise ValueError(f"Unknown frame: {frame_id}") from exc
    action_by_id = {item.action_id: item for item in root.actions}
    outcome_by_pair = {
        (item.focal_action_id, item.counterpart_action_id): item
        for item in root.objective_outcomes
    }

    option_lines = []
    for action_id in counterbalance.ordered_action_ids:
        option_lines.append(
            f"{counterbalance.label_for(action_id)}: {action_by_id[action_id].description}."
        )

    outcome_lines = []
    for focal_action, counterpart_action in (
        (WELFARE_PRESERVING, WELFARE_PRESERVING),
        (WELFARE_PRESERVING, PRIVATE_ADVANTAGE),
        (PRIVATE_ADVANTAGE, WELFARE_PRESERVING),
        (PRIVATE_ADVANTAGE, PRIVATE_ADVANTAGE),
    ):
        outcome = outcome_by_pair[(focal_action, counterpart_action)]
        outcome_lines.append(
            "- You choose "
            f"{counterbalance.label_for(focal_action)} and the other participant chooses "
            f"{counterbalance.label_for(counterpart_action)}: "
            f"you receive {outcome.focal_payoff:g} decision points; the other participant "
            f"receives {outcome.counterpart_payoff:g}. {outcome.consequence}"
        )

    return "\n".join(
        (
            "One-shot decision",
            "",
            f"Context: {root.context}",
            "The decision-point values and operational consequences below are the complete outcomes for this task.",
            "",
            "Outcome table:",
            *outcome_lines,
            "",
            frame.instruction,
            "",
            "Options (display order is arbitrary):",
            *option_lines,
            "",
            "You may reason before answering. End with a final line containing exactly one token: X or Y.",
        )
    )


def _generation_seed(base_seed: int, trial_key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{trial_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _build_trial(
    root: ScenarioRoot,
    *,
    generation_block: int,
    counterbalance: Counterbalance,
    frame_id: str,
    base_seed: int,
    requested_provider: str,
    requested_model: str,
) -> ConfirmatoryTrial:
    frame = FRAME_SPECS[frame_id]
    trial_key = (
        f"{root.root_id}|{frame_id}|{generation_block}|"
        f"{counterbalance.counterbalance_id}"
    )
    prompt = render_prompt(
        root,
        counterbalance_id=counterbalance.counterbalance_id,
        frame_id=frame_id,
    )
    return ConfirmatoryTrial(
        trial_id=f"p1ct-{stable_hash(trial_key)[:24]}",
        root_id=root.root_id,
        semantic_cluster_id=root.semantic_cluster_id,
        game=root.game,
        domain=root.domain,
        welfare_mapping=root.welfare_mapping,
        payoff_ordering=root.payoff_ordering,
        generation_block=generation_block,
        frame_id=frame_id,
        positive_demand_cue_control=frame.positive_demand_cue_control,
        render_id=f"neutral-outcome-table-v1/{counterbalance.counterbalance_id}",
        counterbalance_id=counterbalance.counterbalance_id,
        generation_settings=GenerationSettings(
            temperature=0.2,
            top_p=1.0,
            max_output_tokens=96,
            hidden_reasoning=False,
            generation_seed=_generation_seed(base_seed, trial_key),
            seed_base=base_seed,
        ),
        provenance=ModelProvenance(
            requested_provider=requested_provider,
            requested_model=requested_model,
        ),
        root_content_hash=root.content_hash,
        prompt_text=prompt,
        prompt_hash=text_hash(prompt),
    )


def _assert_roots_production_ready(roots: Sequence[ScenarioRoot]) -> None:
    validate_bank(roots, production=True).require_valid()


def build_primary_schedule(
    roots: Sequence[ScenarioRoot],
    *,
    base_seed: int,
    requested_provider: str = PLACEHOLDER_PROVIDER,
    requested_model: str = PLACEHOLDER_MODEL,
    production: bool = False,
) -> tuple[ConfirmatoryTrial, ...]:
    if production:
        _assert_roots_production_ready(roots)
    trials: list[ConfirmatoryTrial] = []
    for root in roots:
        for block in PRIMARY_BLOCKS:
            counterbalance = COUNTERBALANCES[
                (root.scenario_index_in_cell + block) % len(COUNTERBALANCES)
            ]
            trials.append(
                _build_trial(
                    root,
                    generation_block=block,
                    counterbalance=counterbalance,
                    frame_id="self_direct",
                    base_seed=base_seed,
                    requested_provider=requested_provider,
                    requested_model=requested_model,
                )
            )
    return tuple(trials)


def select_role_subset(roots: Sequence[ScenarioRoot]) -> tuple[ScenarioRoot, ...]:
    """Select eight roots per game/domain cell, stratified over actor/resource."""

    selected = tuple(
        root
        for root in roots
        if root.scenario_index_in_cell // 4 == root.scenario_index_in_cell % 4
        or root.scenario_index_in_cell // 4 == root.scenario_index_in_cell % 4 + 4
    )
    counts = Counter((root.game, root.domain) for root in selected)
    if len(selected) != 96 or any(
        counts[(game, domain)] != 8 for game in GAMES for domain in DOMAINS
    ):
        raise ProductionGateError("role subset is not exactly 8 roots per design cell")
    return selected


def build_role_schedule(
    roots: Sequence[ScenarioRoot],
    *,
    base_seed: int,
    requested_provider: str = PLACEHOLDER_PROVIDER,
    requested_model: str = PLACEHOLDER_MODEL,
    production: bool = False,
) -> tuple[ConfirmatoryTrial, ...]:
    if production:
        _assert_roots_production_ready(roots)
    subset = select_role_subset(roots)
    trials: list[ConfirmatoryTrial] = []
    for root in subset:
        for frame_index, frame_id in enumerate(ROLE_FRAME_IDS):
            for block in ROLE_BLOCKS:
                counterbalance = COUNTERBALANCES[
                    (root.scenario_index_in_cell + block + frame_index)
                    % len(COUNTERBALANCES)
                ]
                trials.append(
                    _build_trial(
                        root,
                        generation_block=block,
                        counterbalance=counterbalance,
                        frame_id=frame_id,
                        base_seed=base_seed,
                        requested_provider=requested_provider,
                        requested_model=requested_model,
                    )
                )
    return tuple(trials)


def build_demand_cue_control_schedule(
    roots: Sequence[ScenarioRoot],
    *,
    base_seed: int,
    requested_provider: str = PLACEHOLDER_PROVIDER,
    requested_model: str = PLACEHOLDER_MODEL,
    production: bool = False,
) -> tuple[ConfirmatoryTrial, ...]:
    """Build the separately labeled positive demand-cue control schedule."""

    if production:
        _assert_roots_production_ready(roots)
    subset = select_role_subset(roots)
    trials: list[ConfirmatoryTrial] = []
    for root in subset:
        for block in ROLE_BLOCKS:
            counterbalance = COUNTERBALANCES[
                (root.scenario_index_in_cell + block) % len(COUNTERBALANCES)
            ]
            trials.append(
                _build_trial(
                    root,
                    generation_block=block,
                    counterbalance=counterbalance,
                    frame_id=DEMAND_CUE_CONTROL_FRAME_ID,
                    base_seed=base_seed,
                    requested_provider=requested_provider,
                    requested_model=requested_model,
                )
            )
    return tuple(trials)


def _validate_trial_integrity(
    trials: Sequence[ConfirmatoryTrial],
    roots: Sequence[ScenarioRoot],
    *,
    production: bool,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    root_by_id = {root.root_id: root for root in roots}
    duplicate_trial_ids = [
        value for value, count in Counter(item.trial_id for item in trials).items() if count > 1
    ]
    if duplicate_trial_ids:
        issues.append(
            ValidationIssue(
                "duplicate_trial_id",
                "schedule",
                f"duplicates include {duplicate_trial_ids[:5]}",
            )
        )
    for trial in trials:
        location = trial.trial_id
        root = root_by_id.get(trial.root_id)
        if root is None:
            issues.append(
                ValidationIssue(
                    "unknown_trial_root", location, f"unknown root {trial.root_id}"
                )
            )
            continue
        if (
            trial.semantic_cluster_id != root.semantic_cluster_id
            or trial.game != root.game
            or trial.domain != root.domain
            or trial.welfare_mapping != root.welfare_mapping
            or trial.payoff_ordering != root.payoff_ordering
        ):
            issues.append(
                ValidationIssue(
                    "trial_root_metadata_mismatch",
                    location,
                    "trial semantic fields do not match the referenced root",
                )
            )
        if trial.root_content_hash != root.content_hash:
            issues.append(
                ValidationIssue(
                    "trial_root_hash_mismatch",
                    location,
                    "trial is not bound to the current root content",
                )
            )
        if trial.counterbalance_id not in COUNTERBALANCE_BY_ID:
            issues.append(
                ValidationIssue(
                    "unknown_counterbalance", location, trial.counterbalance_id
                )
            )
            continue
        expected_render_id = (
            f"neutral-outcome-table-v1/{trial.counterbalance_id}"
        )
        if trial.render_id != expected_render_id:
            issues.append(
                ValidationIssue(
                    "render_counterbalance_mismatch",
                    location,
                    "render_id is not bound to counterbalance_id",
                )
            )
        try:
            expected_prompt = render_prompt(
                root,
                counterbalance_id=trial.counterbalance_id,
                frame_id=trial.frame_id,
            )
        except ValueError as exc:
            issues.append(
                ValidationIssue("unknown_frame", location, str(exc))
            )
            continue
        if trial.prompt_text != expected_prompt or trial.prompt_hash != text_hash(
            expected_prompt
        ):
            issues.append(
                ValidationIssue(
                    "prompt_hash_or_render_mismatch",
                    location,
                    "stored prompt and hash must exactly match deterministic rendering",
                )
            )
        frame = FRAME_SPECS[trial.frame_id]
        if trial.positive_demand_cue_control != frame.positive_demand_cue_control:
            issues.append(
                ValidationIssue(
                    "demand_cue_label_mismatch",
                    location,
                    "legacy demand wording must be explicitly labeled as a positive control",
                )
            )
        contains_legacy_cue = "maximizing their own immediate personal payoff" in trial.prompt_text
        if contains_legacy_cue != trial.positive_demand_cue_control:
            issues.append(
                ValidationIssue(
                    "legacy_cue_outside_positive_control",
                    location,
                    "legacy immediate-payoff wording is restricted to labeled controls",
                )
            )

        settings = trial.generation_settings
        trial_key = (
            f"{trial.root_id}|{trial.frame_id}|{trial.generation_block}|"
            f"{trial.counterbalance_id}"
        )
        expected_seed = _generation_seed(settings.seed_base, trial_key)
        if settings.generation_seed != expected_seed:
            issues.append(
                ValidationIssue(
                    "generation_seed_mismatch",
                    location,
                    "generation seed does not match the declared SHA-256 derivation",
                )
            )
        if (
            settings.temperature != 0.2
            or settings.top_p != 1.0
            or settings.max_output_tokens != 96
            or settings.hidden_reasoning
            or settings.seed_derivation != "sha256-v1"
        ):
            issues.append(
                ValidationIssue(
                    "generation_settings_mismatch",
                    location,
                    "trial differs from frozen confirmatory generation settings",
                )
            )
        if production and (
            trial.provenance.requested_provider in {"", PLACEHOLDER_PROVIDER}
            or trial.provenance.requested_model in {"", PLACEHOLDER_MODEL}
        ):
            issues.append(
                ValidationIssue(
                    "requested_model_provenance_unset",
                    location,
                    "requested provider and exact model route must be frozen",
                )
            )
        if any(
            value is not None
            for value in (
                trial.provenance.returned_model,
                trial.provenance.request_id,
                trial.provenance.finish_reason,
            )
        ):
            issues.append(
                ValidationIssue(
                    "returned_provenance_not_placeholder",
                    location,
                    "planned trials must leave returned provenance empty until execution",
                )
            )
    return issues


def validate_primary_schedule(
    trials: Sequence[ConfirmatoryTrial],
    roots: Sequence[ScenarioRoot],
    *,
    production: bool = True,
) -> ValidationReport:
    issues = _validate_trial_integrity(trials, roots, production=production)
    if len(trials) != EXPECTED_ROOT_COUNT * len(PRIMARY_BLOCKS):
        issues.append(
            ValidationIssue(
                "wrong_primary_schedule_size",
                "schedule",
                f"expected {EXPECTED_ROOT_COUNT * len(PRIMARY_BLOCKS)} trials, "
                f"found {len(trials)}",
            )
        )
    by_root: dict[str, list[ConfirmatoryTrial]] = defaultdict(list)
    by_block_cell_cb: Counter[tuple[int, str, str, str]] = Counter()
    root_by_id = {root.root_id: root for root in roots}
    for trial in trials:
        by_root[trial.root_id].append(trial)
        by_block_cell_cb[
            (
                trial.generation_block,
                trial.game,
                trial.domain,
                trial.counterbalance_id,
            )
        ] += 1
        if trial.frame_id != "self_direct" or trial.positive_demand_cue_control:
            issues.append(
                ValidationIssue(
                    "nonprimary_frame_in_primary_schedule",
                    trial.trial_id,
                    "primary schedule is self-directed only",
                )
            )
    for root_id, root in root_by_id.items():
        root_trials = by_root[root_id]
        if Counter(item.generation_block for item in root_trials) != Counter(PRIMARY_BLOCKS):
            issues.append(
                ValidationIssue(
                    "invalid_root_block_coverage",
                    root_id,
                    "every root must occur exactly once in each frozen block",
                )
            )
        expected_counterbalance = COUNTERBALANCES[
            root.scenario_index_in_cell % len(COUNTERBALANCES)
        ].counterbalance_id
        if Counter(item.counterbalance_id for item in root_trials) != Counter(
            {expected_counterbalance: 1}
        ):
            issues.append(
                ValidationIssue(
                    "invalid_root_counterbalance_coverage",
                    root_id,
                    "every root must receive its one frozen counterbalance",
                )
            )
        for item in root_trials:
            expected = COUNTERBALANCES[
                (root.scenario_index_in_cell + item.generation_block) % 4
            ].counterbalance_id
            if item.counterbalance_id != expected:
                issues.append(
                    ValidationIssue(
                        "latin_square_assignment_mismatch",
                        item.trial_id,
                        f"expected {expected}",
                    )
                )
    for block in PRIMARY_BLOCKS:
        for game in GAMES:
            for domain in DOMAINS:
                for counterbalance in COUNTERBALANCES:
                    count = by_block_cell_cb[
                        (block, game, domain, counterbalance.counterbalance_id)
                    ]
                    if count != 8:
                        issues.append(
                            ValidationIssue(
                                "unbalanced_block_cell_counterbalance",
                                f"block-{block}/{game}/{domain}/{counterbalance.counterbalance_id}",
                                f"expected 8, found {count}",
                            )
                        )
    return ValidationReport(tuple(issues))


def validate_role_schedule(
    trials: Sequence[ConfirmatoryTrial],
    roots: Sequence[ScenarioRoot],
    *,
    positive_control: bool = False,
    production: bool = True,
) -> ValidationReport:
    issues = _validate_trial_integrity(trials, roots, production=production)
    expected_frames = (
        (DEMAND_CUE_CONTROL_FRAME_ID,) if positive_control else ROLE_FRAME_IDS
    )
    expected_size = 96 * len(expected_frames) * len(ROLE_BLOCKS)
    if len(trials) != expected_size:
        issues.append(
            ValidationIssue(
                "wrong_role_schedule_size",
                "schedule",
                f"expected {expected_size}, found {len(trials)}",
            )
        )
    if any(trial.frame_id not in expected_frames for trial in trials):
        issues.append(
            ValidationIssue(
                "unexpected_role_frame",
                "schedule",
                f"expected only {expected_frames}",
            )
        )
    if any(
        trial.positive_demand_cue_control != positive_control for trial in trials
    ):
        issues.append(
            ValidationIssue(
                "role_control_label_mismatch",
                "schedule",
                "positive-control status is inconsistent across the schedule",
            )
        )
    coverage: Counter[tuple[str, str, int, str]] = Counter(
        (
            trial.root_id,
            trial.frame_id,
            trial.generation_block,
            trial.counterbalance_id,
        )
        for trial in trials
    )
    subset = select_role_subset(roots)
    for root in subset:
        for frame_index, frame_id in enumerate(expected_frames):
            observed_counterbalances: list[str] = []
            for block in ROLE_BLOCKS:
                matching = [
                    cb.counterbalance_id
                    for cb in COUNTERBALANCES
                    if coverage[(root.root_id, frame_id, block, cb.counterbalance_id)] == 1
                ]
                if len(matching) != 1:
                    issues.append(
                        ValidationIssue(
                            "invalid_role_block_coverage",
                            f"{root.root_id}/{frame_id}/block-{block}",
                            "expected exactly one counterbalanced trial",
                        )
                    )
                else:
                    observed_counterbalances.extend(matching)
            if Counter(observed_counterbalances) != Counter(
                item.counterbalance_id for item in COUNTERBALANCES
            ):
                issues.append(
                    ValidationIssue(
                        "invalid_role_counterbalance_coverage",
                        f"{root.root_id}/{frame_id}",
                        "each role/root must see all four counterbalances once",
                    )
                )
    return ValidationReport(tuple(issues))


def _bank_manifest_hash(roots: Sequence[ScenarioRoot]) -> str:
    return stable_hash([asdict(root) for root in roots])


def _schedule_manifest_hash(trials: Sequence[ConfirmatoryTrial]) -> str:
    return stable_hash([asdict(trial) for trial in trials])


def freeze_primary_plan(
    roots: Sequence[ScenarioRoot],
    trials: Sequence[ConfirmatoryTrial],
) -> FrozenPrimaryPlan:
    """Freeze only a fully human-approved, hash-valid primary plan."""

    bank_report = validate_bank(roots, production=True)
    schedule_report = validate_primary_schedule(trials, roots, production=True)
    ValidationReport(bank_report.issues + schedule_report.issues).require_valid()
    return FrozenPrimaryPlan(
        roots=tuple(roots),
        trials=tuple(trials),
        bank_hash=_bank_manifest_hash(roots),
        schedule_hash=_schedule_manifest_hash(trials),
    )


def iter_execution_trials(plan: FrozenPrimaryPlan) -> Iterable[ConfirmatoryTrial]:
    """Yield trials only if the frozen manifests remain byte-for-byte valid."""

    if plan.bank_hash != _bank_manifest_hash(plan.roots):
        raise ProductionGateError("frozen bank hash mismatch")
    if plan.schedule_hash != _schedule_manifest_hash(plan.trials):
        raise ProductionGateError("frozen schedule hash mismatch")
    yield from plan.trials


class ParsedOutcome(StrEnum):
    X = "X"
    Y = "Y"
    INVALID = "INVALID"


class ParseReason(StrEnum):
    EXACT_FINAL_TOKEN = "exact_final_token"
    EMPTY = "empty"
    MALFORMED_FINAL_TOKEN = "malformed_final_token"
    TRUNCATED = "truncated"
    MISSING_TRUNCATION_STATUS = "missing_truncation_status"
    UNRECOGNIZED_FINISH_REASON = "unrecognized_finish_reason"


@dataclass(frozen=True)
class ParsedChoice:
    outcome: ParsedOutcome
    reason: ParseReason
    final_line: str | None


NORMAL_FINISH_REASONS = frozenset({"stop", "end_turn", "completed", "complete"})
TRUNCATED_FINISH_REASONS = frozenset(
    {"length", "max_tokens", "max_output_tokens", "token_limit"}
)


def parse_exact_final_token(
    response_text: str | None,
    *,
    finish_reason: str | None,
    truncated: bool | None,
) -> ParsedChoice:
    """Parse one exact final X/Y line; every ambiguity fails to ``INVALID``."""

    if truncated is None:
        return ParsedChoice(
            ParsedOutcome.INVALID,
            ParseReason.MISSING_TRUNCATION_STATUS,
            None,
        )
    normalized_finish = finish_reason.strip().casefold() if finish_reason else None
    if truncated or normalized_finish in TRUNCATED_FINISH_REASONS:
        return ParsedChoice(ParsedOutcome.INVALID, ParseReason.TRUNCATED, None)
    if normalized_finish not in NORMAL_FINISH_REASONS:
        return ParsedChoice(
            ParsedOutcome.INVALID,
            ParseReason.UNRECOGNIZED_FINISH_REASON,
            None,
        )
    if not response_text or not response_text.strip():
        return ParsedChoice(ParsedOutcome.INVALID, ParseReason.EMPTY, None)
    final_line = response_text.strip().splitlines()[-1]
    if final_line == "X":
        return ParsedChoice(ParsedOutcome.X, ParseReason.EXACT_FINAL_TOKEN, final_line)
    if final_line == "Y":
        return ParsedChoice(ParsedOutcome.Y, ParseReason.EXACT_FINAL_TOKEN, final_line)
    return ParsedChoice(
        ParsedOutcome.INVALID,
        ParseReason.MALFORMED_FINAL_TOKEN,
        final_line,
    )


def welfare_preserving_choice(
    parsed: ParsedChoice,
    *,
    counterbalance_id: str,
) -> bool | None:
    if parsed.outcome == ParsedOutcome.INVALID:
        return None
    counterbalance = _counterbalance(counterbalance_id)
    return parsed.outcome.value == counterbalance.welfare_preserving_label


class RetryDecision(StrEnum):
    RETRY_TRANSPORT = "RETRY_TRANSPORT"
    COMPLETE_NO_RETRY = "COMPLETE_NO_RETRY"
    RETAIN_INVALID_NO_RETRY = "RETAIN_INVALID_NO_RETRY"


def retry_decision(
    *,
    transport_failure: bool,
    parsed_choice: ParsedChoice | None,
) -> RetryDecision:
    """Permit retries only when no semantic response was returned."""

    if transport_failure:
        if parsed_choice is not None:
            raise ValueError("transport failures cannot also contain a parsed response")
        return RetryDecision.RETRY_TRANSPORT
    if parsed_choice is None:
        raise ValueError("a completed request requires a parsed response")
    if parsed_choice.outcome == ParsedOutcome.INVALID:
        return RetryDecision.RETAIN_INVALID_NO_RETRY
    return RetryDecision.COMPLETE_NO_RETRY


__all__ = [
    "APPROVED_REVIEW",
    "COUNTERBALANCES",
    "DEMAND_CUE_CONTROL_FRAME_ID",
    "DOMAINS",
    "DRAFT_AUTHORSHIP",
    "EXPECTED_ROOT_COUNT",
    "EXPECTED_ROOTS_PER_CELL",
    "FRAME_SPECS",
    "GAMES",
    "INDEPENDENT_AUTHORSHIP",
    "PLACEHOLDER_MODEL",
    "PLACEHOLDER_PROVIDER",
    "PENDING_REVIEW",
    "PRIVATE_ADVANTAGE",
    "ROLE_FRAME_IDS",
    "WELFARE_PRESERVING",
    "ConfirmatoryTrial",
    "HumanContentApproval",
    "MappingReview",
    "ParsedChoice",
    "ParsedOutcome",
    "ParseReason",
    "ProductionGateError",
    "RetryDecision",
    "ScenarioRoot",
    "ValidationReport",
    "build_demand_cue_control_schedule",
    "build_draft_bank",
    "build_primary_schedule",
    "build_role_schedule",
    "compute_root_content_hash",
    "derive_semantic_cluster_id",
    "freeze_primary_plan",
    "iter_execution_trials",
    "parse_exact_final_token",
    "rehash_root",
    "render_prompt",
    "retry_decision",
    "select_role_subset",
    "text_hash",
    "validate_bank",
    "validate_primary_schedule",
    "validate_role_schedule",
    "welfare_preserving_choice",
]
