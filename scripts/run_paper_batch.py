#!/usr/bin/env python3
"""Run a paper-oriented batch of llm-altruism experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from itertools import combinations
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments import (  # noqa: E402
    ExperimentSettings,
    HistoryConfig,
    ModelAccessResult,
    ModelExperimentReadinessResult,
    ModelSpec,
    ParameterConfig,
    PopulationSpec,
    PromptVariantConfig,
    ReputationConfigModel,
    SocietyConfig,
    WorldConfigModel,
    known_model_specs,
    parse_model_selectors,
    probe_model_experiment_readiness_results,
    probe_model_access_results,
    run_experiment_config,
    spec_selector,
)

console = Console()

DEFAULT_MAIN_COHORT_SELECTORS = [
    "cerebras:llama3.1-8b",
    "cerebras:qwen-3-235b-a22b-instruct-2507",
    "nvidia:deepseek-ai/deepseek-v3.2",
    "nvidia:z-ai/glm5",
    "ollama:llama3.2:3b",
    "nvidia:z-ai/glm4.7",
    "nvidia:moonshotai/kimi-k2-instruct",
    "nvidia:moonshotai/kimi-k2-instruct-0905",
    "nvidia:moonshotai/kimi-k2-thinking",
    "nvidia:bytedance/seed-oss-36b-instruct",
    "nvidia:nvidia/nemotron-3-nano-30b-a3b",
    "nvidia:nvidia/nemotron-3-super-120b-a12b",
    "nvidia:stepfun-ai/step-3-5-flash",
]

SAFETY_CONTROL_SELECTORS = [
    "nvidia:nvidia/nemotron-content-safety-reasoning-4b",
    "nvidia:nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
]

BASELINE_GAMES = ("prisoners_dilemma", "chicken", "stag_hunt")
TRACK_CHOICES = ("baseline", "benchmark", "susceptibility", "society", "reputation", "all")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run a paper-focused experiment batch.")
    parser.add_argument(
        "--track",
        action="append",
        choices=TRACK_CHOICES,
        help="Repeatable track selector. Defaults to baseline, benchmark, susceptibility, society, reputation.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="PROVIDER:MODEL",
        help="Optional repeatable model selector override.",
    )
    parser.add_argument(
        "--include-safety-controls",
        action="store_true",
        help="Include the NVIDIA safety models in the selected cohort.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=4,
        help="Maximum number of accessible main-cohort models to use.",
    )
    parser.add_argument(
        "--results-dir",
        default="results/paper",
        help="Directory for experiment outputs and batch manifests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the batch with deterministic mock responses instead of live provider calls.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a smaller runtime budget for quick iteration.",
    )
    parser.add_argument(
        "--multiagent-repetitions",
        type=int,
        default=1,
        help="Override repetitions for Part 2/3 society and reputation experiments.",
    )
    parser.add_argument(
        "--multiagent-concurrency",
        type=int,
        default=6,
        help="Override agent-decision concurrency for Part 2/3 experiments.",
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help="Optional suffix appended to generated experiment names.",
    )
    return parser.parse_args()


def resolve_tracks(raw_tracks: list[str] | None) -> list[str]:
    """Normalize requested track names."""
    if not raw_tracks:
        return ["baseline", "benchmark", "susceptibility", "society", "reputation"]
    if "all" in raw_tracks:
        return ["baseline", "benchmark", "susceptibility", "society", "reputation"]
    return list(dict.fromkeys(raw_tracks))


async def resolve_accessible_cohort(
    args: argparse.Namespace,
) -> tuple[list[ModelSpec], dict[str, object], dict[str, ModelExperimentReadinessResult]]:
    """Select the accessible and experiment-ready cohort for this batch."""
    if args.model:
        candidate_specs = parse_model_selectors(args.model)
    else:
        selectors = list(DEFAULT_MAIN_COHORT_SELECTORS)
        if args.include_safety_controls:
            selectors.extend(SAFETY_CONTROL_SELECTORS)
        candidate_specs = parse_model_selectors(selectors)

    if args.dry_run:
        selected_specs = list(candidate_specs)
        if not args.model and args.max_models > 0:
            main_cohort = [
                spec
                for spec in selected_specs
                if spec_selector(spec) not in SAFETY_CONTROL_SELECTORS
            ]
            safety_controls = [
                spec
                for spec in selected_specs
                if spec_selector(spec) in SAFETY_CONTROL_SELECTORS
            ]
            selected_specs = main_cohort[: args.max_models] + safety_controls
        access_results = {
            spec_selector(spec): ModelAccessResult(spec=spec, accessible=True, status="dry run")
            for spec in selected_specs
        }
        readiness_results = {
            spec_selector(spec): ModelExperimentReadinessResult(
                spec=spec,
                ready=True,
                status="dry run",
            )
            for spec in selected_specs
        }
        return selected_specs, access_results, readiness_results

    access_results = await probe_model_access_results(candidate_specs)
    accessible_specs = [
        access_results[selector].spec
        for selector in [spec_selector(spec) for spec in candidate_specs]
        if selector in access_results and access_results[selector].accessible
    ]
    readiness_results = await probe_model_experiment_readiness_results(accessible_specs)
    experiment_ready_specs = [
        readiness_results[selector].spec
        for selector in [spec_selector(spec) for spec in accessible_specs]
        if selector in readiness_results and readiness_results[selector].ready
    ]

    if not args.model and args.max_models > 0:
        main_cohort = [
            spec
            for spec in experiment_ready_specs
            if spec_selector(spec) not in SAFETY_CONTROL_SELECTORS
        ]
        safety_controls = [
            spec
            for spec in experiment_ready_specs
            if spec_selector(spec) in SAFETY_CONTROL_SELECTORS
        ]
        experiment_ready_specs = main_cohort[: args.max_models] + safety_controls

    return experiment_ready_specs, access_results, readiness_results


def render_cohort_summary(
    selected_specs: list[ModelSpec],
    access_results: dict[str, object],
    readiness_results: dict[str, ModelExperimentReadinessResult],
) -> None:
    """Show the experiment-ready cohort for this batch."""
    table = Table(title="Paper Batch Cohort", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Provider", style="bold cyan", no_wrap=True)
    table.add_column("Model", style="white")
    table.add_column("Access", style="green")
    table.add_column("Experiment Probe", style="magenta")

    for spec in selected_specs:
        selector = spec_selector(spec)
        access = access_results[selector]
        readiness = readiness_results[selector]
        table.add_row(
            spec.provider or "unknown",
            spec.model,
            getattr(access, "status", "verified"),
            readiness.status,
        )

    console.print(table)
    hidden = []
    for selector, result in readiness_results.items():
        if result.ready:
            continue
        hidden.append((selector, result.status))
    if hidden:
        hidden_table = Table(
            title="Excluded From Main Paper Cohort",
            box=box.SIMPLE_HEAVY,
            header_style="bold yellow",
        )
        hidden_table.add_column("Model", style="yellow")
        hidden_table.add_column("Reason", style="white")
        for selector, reason in hidden:
            hidden_table.add_row(selector, reason)
        console.print(hidden_table)
    console.print()


def build_pairings(models: list[ModelSpec], *, include_self_play: bool = True) -> list[tuple[ModelSpec, ModelSpec]]:
    """Build self-play plus pairwise matchups."""
    pairings: list[tuple[ModelSpec, ModelSpec]] = []
    if include_self_play:
        pairings.extend((model, model) for model in models)
    pairings.extend(combinations(models, 2))
    return pairings


def apply_name_suffix(name: str, suffix: str) -> str:
    """Append a stable suffix to an experiment name when requested."""
    cleaned = suffix.strip().strip("-")
    if not cleaned:
        return name
    return f"{name}-{cleaned}"


def baseline_prompt_variants() -> list[PromptVariantConfig]:
    """Minimal baseline prompt family with neutral paraphrases."""
    return [
        PromptVariantConfig(
            name="minimal-neutral",
            system_prompt="prompts/system/minimal.txt",
            framing="prompts/framing/neutral.txt",
        ),
        PromptVariantConfig(
            name="minimal-neutral-compact",
            system_prompt="prompts/system/minimal.txt",
            framing="prompts/framing/neutral_compact.txt",
        ),
        PromptVariantConfig(
            name="minimal-neutral-institutional",
            system_prompt="prompts/system/minimal.txt",
            framing="prompts/framing/neutral_institutional.txt",
        ),
    ]


def susceptibility_prompt_variants() -> list[PromptVariantConfig]:
    """Prompt variants for steerability tests."""
    return [
        PromptVariantConfig(
            name="minimal-neutral",
            system_prompt="prompts/system/minimal.txt",
            framing="prompts/framing/neutral.txt",
        ),
        PromptVariantConfig(
            name="cooperative",
            system_prompt="prompts/system/cooperative.txt",
            framing="prompts/framing/community.txt",
            persona="prompts/persona/community_leader.txt",
        ),
        PromptVariantConfig(
            name="competitive",
            system_prompt="prompts/system/competitive.txt",
            framing="prompts/framing/adversarial.txt",
            persona="prompts/persona/strategist.txt",
        ),
    ]


def society_prompt_variants(*, reputation: bool) -> list[PromptVariantConfig]:
    """Prompt variants for society survival comparisons."""
    if reputation:
        return [
            PromptVariantConfig(
                name="task-only",
                system_prompt="prompts/system/survival_with_reputation_task_only.txt",
                framing="prompts/framing/community.txt",
            ),
            PromptVariantConfig(
                name="cooperative",
                system_prompt="prompts/system/survival_with_reputation_cooperative.txt",
                framing="prompts/framing/community.txt",
                persona="prompts/persona/diplomat.txt",
            ),
            PromptVariantConfig(
                name="competitive",
                system_prompt="prompts/system/survival_with_reputation_competitive.txt",
                framing="prompts/framing/adversarial.txt",
                persona="prompts/persona/strategist.txt",
            ),
        ]
    return [
        PromptVariantConfig(
            name="task-only",
            system_prompt="prompts/system/survival_task_only.txt",
            framing="prompts/framing/community.txt",
        ),
        PromptVariantConfig(
            name="cooperative",
            system_prompt="prompts/system/survival_cooperative.txt",
            framing="prompts/framing/community.txt",
            persona="prompts/persona/community_leader.txt",
        ),
        PromptVariantConfig(
            name="competitive",
            system_prompt="prompts/system/survival_competitive.txt",
            framing="prompts/framing/adversarial.txt",
            persona="prompts/persona/strategist.txt",
        ),
    ]


def benchmark_variant_options(game: str, presentation: str) -> dict[str, object]:
    """Return game-options overrides for canonical vs disguised benchmark presentations."""
    if presentation == "canonical":
        return {}

    if game == "prisoners_dilemma" and presentation == "unnamed":
        return {
            "prompt_overrides": {
                "description": "games_variants/prisoners_dilemma/unnamed_description.txt",
                "round": "games_variants/prisoners_dilemma/unnamed_round.txt",
            },
            "action_aliases": {"cooperate": "share", "defect": "keep"},
        }

    if game == "prisoners_dilemma" and presentation == "resource":
        return {
            "prompt_overrides": {
                "description": "games_variants/prisoners_dilemma/resource_description.txt",
                "round": "games_variants/prisoners_dilemma/resource_round.txt",
            },
            "action_aliases": {"cooperate": "pool", "defect": "lock"},
        }

    if game == "chicken" and presentation == "unnamed":
        return {
            "prompt_overrides": {
                "description": "games_variants/chicken/unnamed_description.txt",
                "round": "games_variants/chicken/unnamed_round.txt",
            },
            "action_aliases": {"swerve": "yield", "straight": "press"},
        }

    if game == "stag_hunt" and presentation == "unnamed":
        return {
            "prompt_overrides": {
                "description": "games_variants/stag_hunt/unnamed_description.txt",
                "round": "games_variants/stag_hunt/unnamed_round.txt",
            },
            "action_aliases": {"stag": "team", "hare": "solo"},
        }

    raise ValueError(f"No benchmark presentation {presentation!r} defined for {game!r}")


def build_part1_config(
    *,
    name: str,
    game: str,
    pairings: list[tuple[ModelSpec, ModelSpec]],
    prompt_variants: list[PromptVariantConfig],
    rounds: int,
    repetitions: int,
    temperatures: list[float],
    game_options: dict[str, object] | None = None,
) -> ExperimentSettings:
    """Create a part 1 experiment config."""
    return ExperimentSettings(
        name=name,
        part=1,
        game=game,
        game_options=game_options or {},
        rounds=rounds,
        repetitions=repetitions,
        pairings=pairings,
        history=HistoryConfig(mode="full", window_size=rounds),
        prompt_variants=prompt_variants,
        parameters=ParameterConfig(
            temperature=temperatures,
            payoff_visibility=True,
            max_tokens=300,
            max_transient_retries=6,
            initial_retry_delay_seconds=10,
            max_retry_delay_seconds=300,
            seed=11,
        ),
    )


def build_society_agents(models: list[ModelSpec], *, per_model_count: int) -> list[PopulationSpec]:
    """Create a balanced society population from selected models."""
    return [
        PopulationSpec(model=spec.model, provider=spec.provider, count=per_model_count)
        for spec in models
    ]


def build_society_config(
    *,
    name: str,
    part: int,
    models: list[ModelSpec],
    prompt_variants: list[PromptVariantConfig],
    rounds: int,
    repetitions: int,
    temperatures: list[float],
    concurrency: int,
) -> ExperimentSettings:
    """Create either a part 2 or part 3 society config."""
    reputation = None
    if part == 3:
        reputation = ReputationConfigModel(
            enabled=True,
            decay=0.95,
            anonymous_ratings=False,
            min_rating=1,
            max_rating=5,
        )

    return ExperimentSettings(
        name=name,
        part=part,
        rounds=rounds,
        repetitions=repetitions,
        agents=build_society_agents(models, per_model_count=2),
        history=HistoryConfig(mode="summarized", window_size=5),
        prompt_variants=prompt_variants,
        parameters=ParameterConfig(
            temperature=temperatures,
            payoff_visibility=False,
            max_tokens=400,
            concurrency=concurrency,
            max_transient_retries=6,
            initial_retry_delay_seconds=10,
            max_retry_delay_seconds=300,
            seed=101 if part == 2 else 202,
        ),
        world=WorldConfigModel(
            initial_public_resources=36,
            max_public_resources=48,
            regeneration_rate=0.2,
            initial_agent_resources=6,
            gather_amount=3,
            steal_amount=2,
            survival_cost=1,
            reproduction_threshold=15,
            offspring_start_resources=4,
            max_agents=12,
        ),
        society=SocietyConfig(
            allow_steal=True,
            allow_private_messages=True,
            allow_unmonitored_agents=(part == 2),
            unmonitored_fraction=0.2 if part == 2 else 0.0,
            trade_offer_ttl=3,
        ),
        reputation=reputation,
    )


def build_experiment_plan(
    models: list[ModelSpec],
    *,
    fast: bool,
    multiagent_repetitions: int,
    multiagent_concurrency: int,
    name_suffix: str,
) -> list[tuple[str, ExperimentSettings, dict[str, object]]]:
    """Create the full paper-oriented experiment plan."""
    part1_rounds = 4 if fast else 6
    part1_repetitions = 1
    part1_temperatures = [0.0]
    society_rounds = 6 if fast else 8
    society_temperatures = [0.3]

    pairings = build_pairings(models, include_self_play=True)
    plan: list[tuple[str, ExperimentSettings, dict[str, object]]] = []

    for game in BASELINE_GAMES:
        plan.append(
            (
                "baseline",
                build_part1_config(
                    name=apply_name_suffix(f"paper-baseline-{game}", name_suffix),
                    game=game,
                    pairings=pairings,
                    prompt_variants=baseline_prompt_variants(),
                    rounds=part1_rounds,
                    repetitions=part1_repetitions,
                    temperatures=part1_temperatures,
                ),
                {"track": "baseline", "presentation": "canonical"},
            )
        )

    benchmark_presentations = {
        "prisoners_dilemma": ("canonical", "unnamed", "resource"),
        "chicken": ("canonical", "unnamed"),
        "stag_hunt": ("canonical", "unnamed"),
    }
    for game, presentations in benchmark_presentations.items():
        for presentation in presentations:
            plan.append(
                (
                    "benchmark",
                    build_part1_config(
                        name=apply_name_suffix(f"paper-benchmark-{game}-{presentation}", name_suffix),
                        game=game,
                        pairings=pairings,
                        prompt_variants=[
                            PromptVariantConfig(
                                name="minimal-neutral",
                                system_prompt="prompts/system/minimal.txt",
                                framing="prompts/framing/neutral.txt",
                            )
                        ],
                        rounds=part1_rounds,
                        repetitions=part1_repetitions,
                        temperatures=part1_temperatures,
                        game_options=benchmark_variant_options(game, presentation),
                    ),
                    {"track": "benchmark", "presentation": presentation},
                )
            )

    for game in BASELINE_GAMES:
        plan.append(
            (
                "susceptibility",
                build_part1_config(
                    name=apply_name_suffix(f"paper-susceptibility-{game}", name_suffix),
                    game=game,
                    pairings=pairings,
                    prompt_variants=susceptibility_prompt_variants(),
                    rounds=part1_rounds,
                    repetitions=part1_repetitions,
                    temperatures=part1_temperatures,
                ),
                {"track": "susceptibility", "presentation": "canonical"},
            )
        )

    society_models = models[: min(4, len(models))]
    if society_models:
        plan.append(
            (
                "society",
                build_society_config(
                    name=apply_name_suffix("paper-society-prompts", name_suffix),
                    part=2,
                    models=society_models,
                    prompt_variants=society_prompt_variants(reputation=False),
                    rounds=society_rounds,
                    repetitions=multiagent_repetitions,
                    temperatures=society_temperatures,
                    concurrency=multiagent_concurrency,
                ),
                {"track": "society", "presentation": "prompt_comparison"},
            )
        )
        plan.append(
            (
                "reputation",
                build_society_config(
                    name=apply_name_suffix("paper-reputation-prompts", name_suffix),
                    part=3,
                    models=society_models,
                    prompt_variants=society_prompt_variants(reputation=True),
                    rounds=society_rounds,
                    repetitions=multiagent_repetitions,
                    temperatures=society_temperatures,
                    concurrency=multiagent_concurrency,
                ),
                {"track": "reputation", "presentation": "prompt_comparison"},
            )
        )

    return plan


def filter_plan_by_tracks(
    plan: list[tuple[str, ExperimentSettings, dict[str, object]]],
    tracks: list[str],
) -> list[tuple[str, ExperimentSettings, dict[str, object]]]:
    """Return only the requested tracks from a plan."""
    allowed = set(tracks)
    return [item for item in plan if item[0] in allowed]


def find_existing_result(results_dir: Path, experiment_name: str) -> Path | None:
    """Return the most recent completed JSON result for an experiment name, if any."""
    matches = sorted(results_dir.glob(f"{experiment_name}-*.json"))
    if not matches:
        return None
    return matches[-1]


def manifest_entry_from_result(
    *,
    result_path: Path,
    config: ExperimentSettings,
    track: str,
) -> dict[str, object]:
    """Build a manifest entry from an existing or newly created result JSON."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "experiment_id": result["experiment_id"],
        "name": config.name,
        "track": track,
        "game": config.game,
        "part": config.part,
        "path": str(result_path),
        "aggregate_summary": result.get("aggregate_summary", {}),
        "skipped_models": result.get("skipped_models", []),
        "skipped_trials": result.get("skipped_trials", []),
    }


async def run_batch(args: argparse.Namespace) -> int:
    """Run the selected experiment batch."""
    if args.multiagent_repetitions < 1:
        console.print(Panel("--multiagent-repetitions must be at least 1.", border_style="red"))
        return 1
    if args.multiagent_concurrency < 1:
        console.print(Panel("--multiagent-concurrency must be at least 1.", border_style="red"))
        return 1

    tracks = resolve_tracks(args.track)
    models, access_results, readiness_results = await resolve_accessible_cohort(args)
    if not models:
        console.print(
            Panel(
                "No models passed both the access test and the experiment-action preflight.",
                border_style="red",
            )
        )
        return 1

    render_cohort_summary(models, access_results, readiness_results)
    plan = filter_plan_by_tracks(
        build_experiment_plan(
            models,
            fast=args.fast,
            multiagent_repetitions=args.multiagent_repetitions,
            multiagent_concurrency=args.multiagent_concurrency,
            name_suffix=args.name_suffix,
        ),
        tracks,
    )
    if not plan:
        console.print(Panel("No experiments matched the requested tracks.", border_style="red"))
        return 1

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / "paper_batch_manifest.json"
    manifest: dict[str, object] = {
        "tracks": tracks,
        "dry_run": args.dry_run,
        "name_suffix": args.name_suffix,
        "multiagent_repetitions": args.multiagent_repetitions,
        "multiagent_concurrency": args.multiagent_concurrency,
        "selected_models": [spec_selector(spec) for spec in models],
        "access_results": {
            selector: {
                "accessible": result.accessible,
                "status": result.status,
            }
            for selector, result in access_results.items()
        },
        "experiment_readiness_results": {
            selector: {
                "ready": result.ready,
                "status": result.status,
                "parsed_action": result.parsed_action,
            }
            for selector, result in readiness_results.items()
        },
        "experiments": [],
    }

    for index, (track, config, extra_metadata) in enumerate(plan, start=1):
        existing_result = find_existing_result(results_dir, config.name)
        if existing_result is not None:
            console.print(
                Panel(
                    f"Skipping {index}/{len(plan)}: {config.name}\n"
                    f"Reusing existing result: {existing_result.name}",
                    title="Paper Batch Resume",
                    border_style="yellow",
                    box=box.ROUNDED,
                )
            )
            manifest["experiments"].append(
                manifest_entry_from_result(
                    result_path=existing_result,
                    config=config,
                    track=track,
                )
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            continue

        console.print(
            Panel(
                f"Running {index}/{len(plan)}: {config.name}\nTrack: {track}\n"
                f"Part: {config.part}\nGame: {config.game or 'society'}",
                title="Paper Batch Progress",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )
        run_metadata = {
            "generated_by": "scripts/run_paper_batch.py",
            "track": track,
            "selected_models": [spec_selector(spec) for spec in models],
            **extra_metadata,
        }
        result = await run_experiment_config(
            config,
            dry_run=args.dry_run,
            results_dir=str(results_dir),
            run_metadata=run_metadata,
        )
        manifest["experiments"].append(
            manifest_entry_from_result(
                result_path=results_dir / f"{result['experiment_id']}.json",
                config=config,
                track=track,
            )
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    console.print(
        Panel(
            f"Completed {len(plan)} experiment(s).\nManifest: {manifest_path}",
            title="Paper Batch Complete",
            border_style="green",
            box=box.ROUNDED,
        )
    )
    return 0


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    return asyncio.run(run_batch(args))


if __name__ == "__main__":
    raise SystemExit(main())
