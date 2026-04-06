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
    ModelSpec,
    ParameterConfig,
    PopulationSpec,
    PromptVariantConfig,
    ReputationConfigModel,
    SocietyConfig,
    WorldConfigModel,
    known_model_specs,
    parse_model_selectors,
    probe_model_access_results,
    run_experiment_config,
    spec_selector,
)

console = Console()

DEFAULT_MAIN_COHORT_SELECTORS = [
    "cerebras:llama3.1-8b",
    "cerebras:qwen-3-235b-a22b-instruct-2507",
    "nvidia:deepseek-ai/deepseek-v3.2",
    "nvidia:moonshotai/kimi-k2-thinking",
    "nvidia:z-ai/glm4.7",
    "ollama:llama3.2:3b",
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
    return parser.parse_args()


def resolve_tracks(raw_tracks: list[str] | None) -> list[str]:
    """Normalize requested track names."""
    if not raw_tracks:
        return ["baseline", "benchmark", "susceptibility", "society", "reputation"]
    if "all" in raw_tracks:
        return ["baseline", "benchmark", "susceptibility", "society", "reputation"]
    return list(dict.fromkeys(raw_tracks))


async def resolve_accessible_cohort(args: argparse.Namespace) -> tuple[list[ModelSpec], dict[str, object]]:
    """Select the accessible cohort for this batch."""
    if args.model:
        candidate_specs = parse_model_selectors(args.model)
    else:
        selectors = list(DEFAULT_MAIN_COHORT_SELECTORS)
        if args.include_safety_controls:
            selectors.extend(SAFETY_CONTROL_SELECTORS)
        candidate_specs = parse_model_selectors(selectors)

    access_results = await probe_model_access_results(candidate_specs)
    accessible_specs = [
        access_results[selector].spec
        for selector in [spec_selector(spec) for spec in candidate_specs]
        if selector in access_results and access_results[selector].accessible
    ]

    if not args.model and args.max_models > 0:
        main_cohort = [
            spec for spec in accessible_specs if spec_selector(spec) not in SAFETY_CONTROL_SELECTORS
        ]
        safety_controls = [
            spec for spec in accessible_specs if spec_selector(spec) in SAFETY_CONTROL_SELECTORS
        ]
        accessible_specs = main_cohort[: args.max_models] + safety_controls

    return accessible_specs, access_results


def render_cohort_summary(accessible_specs: list[ModelSpec], access_results: dict[str, object]) -> None:
    """Show the accessible cohort for this batch."""
    table = Table(title="Paper Batch Cohort", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Provider", style="bold cyan", no_wrap=True)
    table.add_column("Model", style="white")
    table.add_column("Status", style="green")

    for spec in accessible_specs:
        selector = spec_selector(spec)
        result = access_results[selector]
        table.add_row(spec.provider or "unknown", spec.model, getattr(result, "status", "verified"))

    console.print(table)
    console.print()


def build_pairings(models: list[ModelSpec], *, include_self_play: bool = True) -> list[tuple[ModelSpec, ModelSpec]]:
    """Build self-play plus pairwise matchups."""
    pairings: list[tuple[ModelSpec, ModelSpec]] = []
    if include_self_play:
        pairings.extend((model, model) for model in models)
    pairings.extend(combinations(models, 2))
    return pairings


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
    temperatures: list[float],
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
        repetitions=1,
        agents=build_society_agents(models, per_model_count=2),
        history=HistoryConfig(mode="summarized", window_size=5),
        prompt_variants=prompt_variants,
        parameters=ParameterConfig(
            temperature=temperatures,
            payoff_visibility=False,
            max_tokens=400,
            concurrency=6,
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


def build_experiment_plan(models: list[ModelSpec], *, fast: bool) -> list[tuple[str, ExperimentSettings, dict[str, object]]]:
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
                    name=f"paper-baseline-{game}",
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
                        name=f"paper-benchmark-{game}-{presentation}",
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
                    name=f"paper-susceptibility-{game}",
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
                    name="paper-society-prompts",
                    part=2,
                    models=society_models,
                    prompt_variants=society_prompt_variants(reputation=False),
                    rounds=society_rounds,
                    temperatures=society_temperatures,
                ),
                {"track": "society", "presentation": "prompt_comparison"},
            )
        )
        plan.append(
            (
                "reputation",
                build_society_config(
                    name="paper-reputation-prompts",
                    part=3,
                    models=society_models,
                    prompt_variants=society_prompt_variants(reputation=True),
                    rounds=society_rounds,
                    temperatures=society_temperatures,
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


async def run_batch(args: argparse.Namespace) -> int:
    """Run the selected experiment batch."""
    tracks = resolve_tracks(args.track)
    models, access_results = await resolve_accessible_cohort(args)
    if not models:
        console.print(Panel("No accessible models matched the requested cohort.", border_style="red"))
        return 1

    render_cohort_summary(models, access_results)
    plan = filter_plan_by_tracks(build_experiment_plan(models, fast=args.fast), tracks)
    if not plan:
        console.print(Panel("No experiments matched the requested tracks.", border_style="red"))
        return 1

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "tracks": tracks,
        "dry_run": args.dry_run,
        "selected_models": [spec_selector(spec) for spec in models],
        "experiments": [],
    }

    for index, (track, config, extra_metadata) in enumerate(plan, start=1):
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
        experiment_id = result["experiment_id"]
        manifest["experiments"].append(
            {
                "experiment_id": experiment_id,
                "name": config.name,
                "track": track,
                "game": config.game,
                "part": config.part,
                "path": str(results_dir / f"{experiment_id}.json"),
                "aggregate_summary": result.get("aggregate_summary", {}),
                "skipped_models": result.get("skipped_models", []),
                "skipped_trials": result.get("skipped_trials", []),
            }
        )

    manifest_path = results_dir / "paper_batch_manifest.json"
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
