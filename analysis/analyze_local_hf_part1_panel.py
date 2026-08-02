"""Fail-closed aggregate-only analysis of a private local HF Part 1 panel.

The local panel retains prompt and response text because those fields are
needed to audit parsing and exact schedule replay.  This analyzer validates
those private fields in place, but emits only identifiers, cryptographic
bindings, and derived aggregates.  The draft scenario bank is not human
approved, so the resulting artifact is permanently exploratory and cannot be
promoted to a confirmatory or paper result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import stat
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.misc.local_hf_part1_panel import (
    SCHEMA_VERSION as PANEL_SCHEMA_VERSION,
    _parse_final_action,
    build_draft_trials,
)
from experiments.misc.local_hf_smoke import LocalHFSmokeError, _read_registry
from experiments.part1.confirmatory_design import (
    COUNTERBALANCE_BY_ID,
    DOMAINS,
    GAMES,
    WELFARE_PRESERVING,
)


SCHEMA_VERSION = 1
DEFAULT_BOOTSTRAP_SEED = 20_260_804
BOOTSTRAP_REPLICATES = 5_000
EXPECTED_ROOTS = 384
EXPECTED_CELLS = 12
_ROW_FIELDS = {
    "schema_version",
    "model_id",
    "upstream_model_id",
    "revision",
    "trial_id",
    "root_id",
    "game",
    "domain",
    "counterbalance_id",
    "prompt_text",
    "prompt_sha256",
    "response_text",
    "response_sha256",
    "parsed_action",
    "format_valid",
    "snapshot_tree_sha256",
    "runtime_sha256",
    "finished_at_utc",
}
_RUNTIME_FIELDS = {"python", "torch", "transformers", "device"}
_SNAPSHOT_FIELDS = {"asset_count", "total_size_bytes", "snapshot_tree_sha256"}


class LocalHFPart1AnalysisError(RuntimeError):
    """Private local panel evidence cannot safely support aggregation."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise LocalHFPart1AnalysisError(f"Cannot hash bound file: {path}.") from error
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _self_hash(payload: Mapping[str, Any]) -> str:
    return _sha256_json(
        {key: value for key, value in payload.items() if key != "evidence_sha256"}
    )


def _seal(payload: dict[str, Any]) -> None:
    payload.pop("evidence_sha256", None)
    payload["evidence_sha256"] = _self_hash(payload)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LocalHFPart1AnalysisError(
            f"{label} is not readable UTF-8 JSON."
        ) from error
    if not isinstance(value, dict):
        raise LocalHFPart1AnalysisError(f"{label} must be a JSON object.")
    return value


def _require_mode(path: Path, mode: int) -> None:
    if os.name != "posix":
        return
    try:
        observed = stat.S_IMODE(path.stat().st_mode)
    except OSError as error:
        raise LocalHFPart1AnalysisError(f"Cannot inspect private path: {path}.") from error
    if observed != mode:
        raise LocalHFPart1AnalysisError(
            f"Unsafe private permissions for {path}: expected {mode:04o}, "
            f"observed {observed:04o}."
        )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            path.chmod(0o600)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_registry(
    registry_path: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    try:
        registry, models = _read_registry(registry_path)
    except LocalHFSmokeError as error:
        raise LocalHFPart1AnalysisError(
            "The bound local-control registry failed validation."
        ) from error
    if (
        manifest.get("registry_sha256") != _sha256_file(registry_path)
        or manifest.get("registry_version") != registry.get("registry_version")
    ):
        raise LocalHFPart1AnalysisError("Registry file/version binding changed.")
    by_id = {row["id"]: row for row in models}
    selected = manifest.get("selected_model_ids")
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(value, str) or not value for value in selected)
        or len(selected) != len(set(selected))
        or any(value not in by_id for value in selected)
    ):
        raise LocalHFPart1AnalysisError("Manifest selected model identities are invalid.")
    return registry, by_id


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> int:
    if (
        manifest.get("schema_version") != PANEL_SCHEMA_VERSION
        or manifest.get("artifact_type")
        != "local_hf_part1_large_n_exploratory_panel"
        or manifest.get("evidence_sha256") != _self_hash(manifest)
    ):
        raise LocalHFPart1AnalysisError("Panel manifest schema/type/self-hash failed.")
    if (
        manifest.get("analysis_role") != "exploratory_local_scale_control_only"
        or manifest.get("draft_bank_human_approved") is not False
        or manifest.get("confirmatory_or_paper_promotion_permitted") is not False
        or manifest.get("frontier_route_substitution_permitted") is not False
    ):
        raise LocalHFPart1AnalysisError("Manifest exploratory nonpromotion flags changed.")
    if manifest.get("complete") is not True:
        raise LocalHFPart1AnalysisError("Analysis requires a completed local panel.")
    contract = manifest.get("generation_contract")
    expected_keys = {
        "mode",
        "base_seed",
        "batch_size",
        "max_new_tokens",
        "local_files_only",
        "trust_remote_code",
        "device",
        "snapshot_binding",
        "runtime_binding",
        "parallelization",
        "max_workers",
    }
    if not isinstance(contract, Mapping) or set(contract) != expected_keys:
        raise LocalHFPart1AnalysisError("Manifest generation contract is incomplete.")
    base_seed = contract.get("base_seed")
    positive_integer_fields = ("batch_size", "max_new_tokens", "max_workers")
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or any(
            isinstance(contract.get(key), bool)
            or not isinstance(contract.get(key), int)
            or int(contract[key]) < 1
            for key in positive_integer_fields
        )
        or contract.get("mode") != "greedy"
        or contract.get("local_files_only") is not True
        or contract.get("trust_remote_code") is not False
        or contract.get("device") not in {"cpu", "mps", "cuda"}
        or contract.get("snapshot_binding")
        != "recursive_sha256_all_snapshot_assets"
        or contract.get("runtime_binding")
        != "python_torch_transformers_device"
        or contract.get("parallelization")
        != "bounded_model_level_thread_pool_with_batched_prompts"
    ):
        raise LocalHFPart1AnalysisError("Manifest generation contract changed.")
    return base_seed


def _schedule(base_seed: int, manifest: Mapping[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    trials = list(build_draft_trials(base_seed=base_seed))
    compact = [
        {
            "trial_id": trial.trial_id,
            "root_id": trial.root_id,
            "game": trial.game,
            "domain": trial.domain,
            "prompt_sha256": trial.prompt_hash,
        }
        for trial in trials
    ]
    detailed = [
        {
            **row,
            "counterbalance_id": trial.counterbalance_id,
        }
        for row, trial in zip(compact, trials, strict=True)
    ]
    cells = Counter((trial.game, trial.domain) for trial in trials)
    counterbalances = Counter(trial.counterbalance_id for trial in trials)
    cell_counterbalances = Counter(
        (trial.game, trial.domain, trial.counterbalance_id) for trial in trials
    )
    expected_cells = {(game, domain) for game in GAMES for domain in DOMAINS}
    if (
        len(trials) != EXPECTED_ROOTS
        or len({trial.trial_id for trial in trials}) != EXPECTED_ROOTS
        or len({trial.root_id for trial in trials}) != EXPECTED_ROOTS
        or set(cells) != expected_cells
        or len(cells) != EXPECTED_CELLS
        or set(cells.values()) != {32}
        or set(counterbalances) != set(COUNTERBALANCE_BY_ID)
        or set(counterbalances.values()) != {96}
        or len(cell_counterbalances) != EXPECTED_CELLS * 4
        or set(cell_counterbalances.values()) != {8}
    ):
        raise LocalHFPart1AnalysisError(
            "Reconstructed schedule is not the frozen balanced 384-root design."
        )
    if (
        manifest.get("trial_count_per_model") != EXPECTED_ROOTS
        or manifest.get("schedule_sha256") != _sha256_json(compact)
    ):
        raise LocalHFPart1AnalysisError("Manifest schedule hash/root binding failed.")
    binding = {
        "base_seed": base_seed,
        "root_count": EXPECTED_ROOTS,
        "game_domain_strata": EXPECTED_CELLS,
        "roots_per_stratum": 32,
        "counterbalance_count": 4,
        "roots_per_counterbalance_per_stratum": 8,
        "runner_schedule_sha256": _sha256_json(compact),
        "analysis_schedule_with_counterbalance_sha256": _sha256_json(detailed),
        "counterbalance_counts": dict(sorted(counterbalances.items())),
    }
    return trials, binding


def _validate_snapshot(value: object, model_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SNAPSHOT_FIELDS:
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} lacks an exact snapshot binding."
        )
    if any(
        isinstance(value.get(key), bool)
        or not isinstance(value.get(key), int)
        or int(value[key]) <= 0
        for key in ("asset_count", "total_size_bytes")
    ) or not _is_sha256(value.get("snapshot_tree_sha256")):
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} has an invalid snapshot binding."
        )
    return dict(value)


def _validate_runtime(value: object, model_id: str) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _RUNTIME_FIELDS
        or any(not isinstance(item, str) or not item for item in value.values())
    ):
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} lacks an exact runtime binding."
        )
    return {str(key): str(item) for key, item in value.items()}


def _read_jsonl(path: Path, model_id: str) -> tuple[list[dict[str, Any]], str]:
    if path.is_symlink():
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} evidence must not be a symbolic link."
        )
    _require_mode(path, 0o600)
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        lines = text.splitlines()
        if not text.endswith("\n") or any(not line for line in lines):
            raise ValueError("non-canonical JSONL framing")
        rows = [json.loads(line) for line in lines]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} evidence is not canonical UTF-8 JSONL."
        ) from error
    if any(not isinstance(row, dict) for row in rows):
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} evidence contains a non-object row."
        )
    return rows, hashlib.sha256(raw).hexdigest()


def _validate_timestamp(value: object, model_id: str, row_number: int) -> None:
    if not isinstance(value, str):
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} row {row_number} has an invalid UTC timestamp."
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} row {row_number} has an invalid UTC timestamp."
        ) from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} row {row_number} timestamp is not UTC."
        )


def _validate_model_rows(
    *,
    panel_dir: Path,
    model_id: str,
    model: Mapping[str, str],
    result: Mapping[str, Any],
    trials: Sequence[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_name = model_id.replace(".", "_") + ".jsonl"
    if (
        Path(expected_name).name != expected_name
        or result.get("status") != "passed"
        or result.get("output_file") != expected_name
    ):
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} does not have a completed bound evidence file."
        )
    path = panel_dir / expected_name
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} evidence file is unavailable."
        ) from error
    if resolved.parent != panel_dir or not resolved.is_file():
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} evidence file escapes the private panel."
        )
    rows, file_sha256 = _read_jsonl(path, model_id)
    snapshot = _validate_snapshot(result.get("snapshot"), model_id)
    runtime = _validate_runtime(result.get("runtime"), model_id)
    runtime_sha256 = _sha256_json(runtime)
    if (
        len(rows) != EXPECTED_ROOTS
        or result.get("completed_trials") != EXPECTED_ROOTS
        or result.get("output_sha256") != file_sha256
    ):
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} lacks exact 384-row file/count/hash coverage."
        )

    derived: list[dict[str, Any]] = []
    for index, (row, trial) in enumerate(zip(rows, trials, strict=True), start=1):
        if set(row) != _ROW_FIELDS:
            raise LocalHFPart1AnalysisError(
                f"Model {model_id} row {index} has an invalid schema."
            )
        response = row.get("response_text")
        prompt = row.get("prompt_text")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise LocalHFPart1AnalysisError(
                f"Model {model_id} row {index} has invalid private text fields."
            )
        action = _parse_final_action(response)
        expected = {
            "schema_version": PANEL_SCHEMA_VERSION,
            "model_id": model_id,
            "upstream_model_id": model["model_id"],
            "revision": model["revision"],
            "trial_id": trial.trial_id,
            "root_id": trial.root_id,
            "game": trial.game,
            "domain": trial.domain,
            "counterbalance_id": trial.counterbalance_id,
            "prompt_text": trial.prompt_text,
            "prompt_sha256": trial.prompt_hash,
            "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            "parsed_action": action,
            "format_valid": action is not None,
            "snapshot_tree_sha256": snapshot["snapshot_tree_sha256"],
            "runtime_sha256": runtime_sha256,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise LocalHFPart1AnalysisError(
                f"Model {model_id} row {index} changed from its frozen "
                "schedule/model/content/runtime binding."
            )
        _validate_timestamp(row.get("finished_at_utc"), model_id, index)
        counterbalance = COUNTERBALANCE_BY_ID.get(trial.counterbalance_id)
        if counterbalance is None:
            raise LocalHFPart1AnalysisError("The reconstructed counterbalance is unknown.")
        cooperation_label = counterbalance.label_for(WELFARE_PRESERVING)
        derived.append(
            {
                "root_id": trial.root_id,
                "game": trial.game,
                "domain": trial.domain,
                "action": action,
                "format_valid": action in {"X", "Y"},
                "action_x": action == "X",
                "cooperation": action == cooperation_label,
            }
        )
    valid = sum(bool(row["format_valid"]) for row in derived)
    if (
        result.get("valid_actions") != valid
        or result.get("invalid_actions") != EXPECTED_ROOTS - valid
    ):
        raise LocalHFPart1AnalysisError(
            f"Model {model_id} manifest valid/invalid counts changed."
        )
    binding = {
        "model_id": model_id,
        "path": str(path),
        "file_sha256": file_sha256,
        "record_count": len(rows),
        "manifest_output_file": result["output_file"],
        "manifest_output_sha256": result["output_sha256"],
        "snapshot": snapshot,
        "runtime": runtime,
        "runtime_sha256": runtime_sha256,
    }
    return derived, binding


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise LocalHFPart1AnalysisError("Bootstrap distribution is empty.")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _interval(point: float, draws: Sequence[float]) -> dict[str, Any]:
    return {
        "estimate": point,
        "lower": _percentile(draws, 0.025),
        "upper": _percentile(draws, 0.975),
        "confidence_level": 0.95,
        "method": "shared_root_within_game_domain_stratified_bootstrap_percentile",
    }


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if not total:
        raise LocalHFPart1AnalysisError("Cannot aggregate an empty model/cell.")
    valid = sum(bool(row["format_valid"]) for row in rows)
    x_count = sum(row["action"] == "X" for row in rows)
    y_count = sum(row["action"] == "Y" for row in rows)
    cooperation = sum(bool(row["cooperation"]) for row in rows)
    return {
        "planned_and_retained_count": total,
        "format_valid_count": valid,
        "format_invalid_count": total - valid,
        "action_x_count": x_count,
        "action_y_count": y_count,
        "format_valid_rate": valid / total,
        "primary_action_x_rate_format_invalid_retained_as_non_x": x_count / total,
        "primary_cooperation_rate_format_invalid_retained_as_noncooperation": cooperation
        / total,
        "valid_only_action_x_rate_descriptive": x_count / valid if valid else None,
        "valid_only_cooperation_rate_descriptive": cooperation / valid if valid else None,
        "valid_only_denominator": valid,
        "valid_only_label": "descriptive_only_conditioned_on_format_valid_response",
    }


def analyze_panel(
    manifest_path: Path,
    registry_path: Path,
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Validate a completed private local panel and return safe aggregates."""

    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise LocalHFPart1AnalysisError("Bootstrap seed must be an integer.")
    if bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise LocalHFPart1AnalysisError(
            "Local panel analysis requires exactly 5,000 bootstrap replicates."
        )
    original_manifest = Path(manifest_path)
    if original_manifest.is_symlink():
        raise LocalHFPart1AnalysisError("Panel manifest must not be a symbolic link.")
    try:
        manifest_path = original_manifest.resolve(strict=True)
        registry_path = Path(registry_path).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise LocalHFPart1AnalysisError("A required bound input is unavailable.") from error
    if manifest_path.name != "manifest.json" or not manifest_path.is_file():
        raise LocalHFPart1AnalysisError("Expected a regular manifest.json panel manifest.")
    panel_dir = manifest_path.parent
    _require_mode(panel_dir, 0o700)
    _require_mode(manifest_path, 0o600)
    manifest = _read_json(manifest_path, "panel manifest")
    base_seed = _validate_manifest_contract(manifest)
    registry, models_by_id = _validate_registry(registry_path, manifest)
    trials, schedule_binding = _schedule(base_seed, manifest)

    selected = list(manifest["selected_model_ids"])
    model_results = manifest.get("models")
    if not isinstance(model_results, Mapping) or set(model_results) != set(selected):
        raise LocalHFPart1AnalysisError(
            "Manifest model results do not exactly match selected model identities."
        )
    if (
        manifest.get("total_planned_generations") != len(selected) * EXPECTED_ROOTS
        or manifest["generation_contract"].get("max_workers")
        != min(int(manifest["generation_contract"]["max_workers"]), len(selected))
    ):
        raise LocalHFPart1AnalysisError("Manifest planned generation/model contract changed.")

    expected_name_list = [model_id.replace(".", "_") + ".jsonl" for model_id in selected]
    if (
        len(expected_name_list) != len(set(expected_name_list))
        or any(Path(name).name != name for name in expected_name_list)
    ):
        raise LocalHFPart1AnalysisError(
            "Selected model ids do not map uniquely to safe evidence filenames."
        )
    expected_jsonl_names = set(expected_name_list)
    observed_jsonl_names = {path.name for path in panel_dir.glob("*.jsonl")}
    if observed_jsonl_names != expected_jsonl_names:
        raise LocalHFPart1AnalysisError(
            "Private panel JSONL files do not exactly match selected models."
        )

    observations: dict[str, list[dict[str, Any]]] = {}
    jsonl_bindings: list[dict[str, Any]] = []
    for model_id in selected:
        result = model_results[model_id]
        if not isinstance(result, Mapping):
            raise LocalHFPart1AnalysisError(f"Model {model_id} result is malformed.")
        rows, binding = _validate_model_rows(
            panel_dir=panel_dir,
            model_id=model_id,
            model=models_by_id[model_id],
            result=result,
            trials=trials,
        )
        observations[model_id] = rows
        jsonl_bindings.append(binding)

    ordered_cells = [(game, domain) for game in GAMES for domain in DOMAINS]
    roots_by_cell = {
        cell: [trial.root_id for trial in trials if (trial.game, trial.domain) == cell]
        for cell in ordered_cells
    }
    rows_by_model_root = {
        model_id: {str(row["root_id"]): row for row in rows}
        for model_id, rows in observations.items()
    }
    action_draws = {model_id: [] for model_id in selected}
    cooperation_draws = {model_id: [] for model_id in selected}
    equal_model_action_draws: list[float] = []
    equal_model_cooperation_draws: list[float] = []
    rng = random.Random(bootstrap_seed)
    for _ in range(bootstrap_replicates):
        action_totals = dict.fromkeys(selected, 0)
        cooperation_totals = dict.fromkeys(selected, 0)
        for cell in ordered_cells:
            roots = roots_by_cell[cell]
            sampled = Counter(rng.choices(roots, k=len(roots)))
            for model_id in selected:
                lookup = rows_by_model_root[model_id]
                action_totals[model_id] += sum(
                    count * int(lookup[root]["action_x"])
                    for root, count in sampled.items()
                )
                cooperation_totals[model_id] += sum(
                    count * int(lookup[root]["cooperation"])
                    for root, count in sampled.items()
                )
        for model_id in selected:
            action_draws[model_id].append(action_totals[model_id] / EXPECTED_ROOTS)
            cooperation_draws[model_id].append(
                cooperation_totals[model_id] / EXPECTED_ROOTS
            )
        equal_model_action_draws.append(
            sum(action_totals.values()) / (EXPECTED_ROOTS * len(selected))
        )
        equal_model_cooperation_draws.append(
            sum(cooperation_totals.values()) / (EXPECTED_ROOTS * len(selected))
        )

    model_summaries: list[dict[str, Any]] = []
    for model_id in selected:
        model = models_by_id[model_id]
        rows = observations[model_id]
        counts = _counts(rows)
        per_cell = [
            {
                "game": game,
                "domain": domain,
                **_counts(
                    [
                        row
                        for row in rows
                        if row["game"] == game and row["domain"] == domain
                    ]
                ),
            }
            for game, domain in ordered_cells
        ]
        model_summaries.append(
            {
                "model_id": model_id,
                "upstream_model_id": model["model_id"],
                "revision": model["revision"],
                "parameter_scale": model["parameter_scale"],
                "counts": counts,
                "primary_action_x_rate_95_ci": _interval(
                    counts["primary_action_x_rate_format_invalid_retained_as_non_x"],
                    action_draws[model_id],
                ),
                "primary_cooperation_rate_95_ci": _interval(
                    counts[
                        "primary_cooperation_rate_format_invalid_retained_as_noncooperation"
                    ],
                    cooperation_draws[model_id],
                ),
                "per_game_domain": per_cell,
                "inference_status": "exploratory_descriptive_fixed_local_model",
            }
        )

    equal_action = sum(
        summary["counts"]["primary_action_x_rate_format_invalid_retained_as_non_x"]
        for summary in model_summaries
    ) / len(model_summaries)
    equal_cooperation = sum(
        summary["counts"][
            "primary_cooperation_rate_format_invalid_retained_as_noncooperation"
        ]
        for summary in model_summaries
    ) / len(model_summaries)
    equal_cells = [
        {
            "game": game,
            "domain": domain,
            "equal_model_mean_format_valid_rate": sum(
                summary["per_game_domain"][index]["format_valid_rate"]
                for summary in model_summaries
            )
            / len(model_summaries),
            "equal_model_mean_action_x_rate_format_invalid_retained_as_non_x": sum(
                summary["per_game_domain"][index][
                    "primary_action_x_rate_format_invalid_retained_as_non_x"
                ]
                for summary in model_summaries
            )
            / len(model_summaries),
            "equal_model_mean_cooperation_rate_format_invalid_retained_as_noncooperation": sum(
                summary["per_game_domain"][index][
                    "primary_cooperation_rate_format_invalid_retained_as_noncooperation"
                ]
                for summary in model_summaries
            )
            / len(model_summaries),
        }
        for index, (game, domain) in enumerate(ordered_cells)
    ]

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "local_hf_part1_exploratory_private_panel_analysis",
        "analysis_role": "exploratory_only_no_confirmatory_or_paper_promotion",
        "confirmatory_or_paper_promotion_permitted": False,
        "draft_bank_human_approved": False,
        "frontier_route_substitution_permitted": False,
        "privacy_contract": {
            "contains_prompt_text": False,
            "contains_raw_response_or_response_text": False,
            "contains_reasoning": False,
            "contains_only_identifiers_hash_bindings_and_derived_aggregates": True,
        },
        "parameters": {
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_method": "one_shared_root_resampling_with_replacement_within_each_of_12_game_by_domain_strata_across_all_models",
            "confidence_interval": "deterministic_nonparametric_percentile_95",
            "invalid_handling": "format_invalid_responses_retained_in_primary_denominator_as_non_x_and_noncooperation",
            "weighting": "equal_model_weight",
        },
        "schedule_binding": schedule_binding,
        "bindings": {
            "panel_manifest_path": str(manifest_path),
            "panel_manifest_file_sha256": _sha256_file(manifest_path),
            "panel_manifest_evidence_sha256": manifest["evidence_sha256"],
            "registry_path": str(registry_path),
            "registry_file_sha256": _sha256_file(registry_path),
            "registry_canonical_sha256": _sha256_json(registry),
            "jsonl_files": jsonl_bindings,
        },
        "coverage": {
            "model_count": len(selected),
            "roots_per_model": EXPECTED_ROOTS,
            "retained_model_trial_count": len(selected) * EXPECTED_ROOTS,
            "game_domain_strata": EXPECTED_CELLS,
            "roots_per_stratum_per_model": 32,
            "all_selected_models_complete": True,
        },
        "models": model_summaries,
        "overall_equal_model": {
            "weighting": "finite_panel_equal_model_weight",
            "model_count": len(selected),
            "primary_action_x_rate_95_ci": _interval(
                equal_action, equal_model_action_draws
            ),
            "primary_cooperation_rate_95_ci": _interval(
                equal_cooperation, equal_model_cooperation_draws
            ),
            "per_game_domain": equal_cells,
            "inference_status": "exploratory_finite_panel_summary_over_completed_pinned_local_models_only",
        },
        "limitations": [
            "Exploratory draft scenarios were not human-approved.",
            "The fixed local model panel is descriptive and does not establish population-level model-family inference.",
            "Invalid outputs remain in every primary denominator as non-X and noncooperation.",
            "Valid-only rates condition on format validity and are descriptive only.",
            "This artifact cannot be promoted to confirmatory or paper evidence.",
        ],
    }
    _seal(artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and safely aggregate a private local HF Part 1 panel."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = analyze_panel(
        args.manifest, args.registry, bootstrap_seed=args.bootstrap_seed
    )
    _atomic_json(args.output, artifact)
    print(
        f"Validated {artifact['coverage']['model_count']} local models with "
        f"{artifact['coverage']['roots_per_model']} roots each."
    )
    print(f"Exploratory aggregate artifact: {args.output}")
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (LocalHFPart1AnalysisError, OSError, ValueError) as error:
        print(f"Local HF Part 1 analysis failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
