"""Repair operationally ineligible corrected Part 2 panel trajectories.

Source journals are immutable.  A trajectory containing an exhausted transport
failure or an identity mismatch is rerun from day one into a separate,
hash-chained overlay.  Genuine semantic INVALID results in otherwise eligible
source trajectories are never selected or regenerated.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping, Sequence

from experiments.misc import inference_hub_part2_panel as runner


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "inference_hub_part2_operational_trajectory_repair_v1"
NUMBERED_BETTER_GOS_KEY = re.compile(r"^BETTER_GOS_NVIDIA_API_KEY(?:_(\d+))?$")


class Part2OperationalRepairError(RuntimeError):
    """The source evidence or repair overlay violates its frozen contract."""


class PooledInferenceHubClient(runner.InferenceHubClient):
    """Round-robin exact-route clients with an independent limiter per account."""

    def __init__(self, clients: Sequence[runner.InferenceHubClient]) -> None:
        if not clients:
            raise ValueError("A pooled Inference Hub client requires at least one account.")
        endpoints = {client.base_url for client in clients}
        contracts = {runner._sha256_json(client.rate_limit_contract) for client in clients}
        if len(endpoints) != 1 or len(contracts) != 1:
            raise ValueError("Pooled clients must share one endpoint and rate-limit contract.")
        self._clients = tuple(clients)
        self.base_url = self._clients[0].base_url
        self._cursor = 0
        self._lock = threading.Lock()

    @property
    def rate_limit_contract(self) -> dict[str, Any]:
        return self._clients[0].rate_limit_contract

    @property
    def account_count(self) -> int:
        return len(self._clients)

    def post(
        self, path: str, body: Mapping[str, Any], *,
        upstream_provider: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            client = self._clients[self._cursor]
            self._cursor = (self._cursor + 1) % len(self._clients)
        return client.post(path, body, upstream_provider=upstream_provider)


def _read_credential_pool(path: Path, expected_count: int) -> tuple[str, ...]:
    if expected_count < 1:
        raise Part2OperationalRepairError("Expected API-key count must be positive.")
    values: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        match = NUMBERED_BETTER_GOS_KEY.fullmatch(name.strip())
        if match:
            ordinal = int(match.group(1) or 1)
            values.append((ordinal, value.strip().strip('"').strip("'")))
    ordered = sorted(values)
    ordinals = tuple(ordinal for ordinal, value in ordered if value)
    keys = tuple(value for _, value in ordered if value)
    if (
        ordinals != tuple(range(1, expected_count + 1))
        or len(keys) != expected_count
        or len(set(keys)) != expected_count
    ):
        raise Part2OperationalRepairError(
            f"Credential pool must contain exactly {expected_count} unique nonempty accounts."
        )
    return keys


def _pooled_runtime_client(
    path: Path, expected_count: int, timeout_seconds: float, rate_profile: str,
) -> PooledInferenceHubClient:
    keys = _read_credential_pool(path, expected_count)
    original_key = os.environ.get("NVIDIA_API_KEY")
    clients = []
    try:
        for key in keys:
            os.environ["NVIDIA_API_KEY"] = key
            clients.append(runner._runtime_client(timeout_seconds, rate_profile))
    finally:
        if original_key is None:
            os.environ.pop("NVIDIA_API_KEY", None)
        else:
            os.environ["NVIDIA_API_KEY"] = original_key
    return PooledInferenceHubClient(clients)


def _bound_json(reference: Mapping[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(str(reference.get("path", ""))).resolve()
    value = runner._read_json(path, label)
    if reference.get("file_sha256") != runner._sha256_file(path):
        raise Part2OperationalRepairError(f"{label} file hash changed.")
    evidence = reference.get("evidence_sha256")
    if evidence is not None and value.get("evidence_sha256") != evidence:
        raise Part2OperationalRepairError(f"{label} evidence hash changed.")
    canonical = reference.get("canonical_sha256")
    if canonical is not None and runner._sha256_json(value) != canonical:
        raise Part2OperationalRepairError(f"{label} canonical hash changed.")
    return path, value


def _bindings(
    source_path: Path, source: Mapping[str, Any], max_rounds: int,
    credential_pool_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bindings = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "source_manifest": {
            "path": str(source_path.resolve()),
            "file_sha256": runner._sha256_file(source_path),
            "evidence_sha256": source["evidence_sha256"],
        },
        "panel_id": source["panel_id"],
        "part2_contract": source["part2_contract"],
        "common_environment_seeds": source["common_environment_seeds"],
        "subject_routes": source["subject_routes"],
        "repair_policy": "whole_trajectory_day_one_exact_route_separate_overlay",
        "maximum_rounds": max_rounds,
    }
    if credential_pool_binding is not None:
        bindings["credential_pool"] = dict(credential_pool_binding)
    return bindings


def run_repair(
    *, source_manifest_path: Path, output_dir: Path, client: Any,
    max_rounds: int = 8, trajectory_workers: int = 4,
    participant_workers: int = 16, max_attempts: int = 8,
    initial_backoff_seconds: float = 1.0, resume: bool = False,
    credential_pool_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if min(max_rounds, trajectory_workers, participant_workers, max_attempts) < 1:
        raise Part2OperationalRepairError("Repair limits must be positive.")
    source_path = source_manifest_path.resolve()
    source = runner._read_json(source_path, "source Part 2 manifest")
    if (
        source.get("artifact_type") != "inference_hub_part2_corrected_matched_panel"
        or source.get("evidence_sha256") != runner._self_hash(source)
        or source.get("summary", {}).get("completed_trajectories")
        != source.get("summary", {}).get("planned_trajectories")
        or source.get("summary", {}).get("identity_mismatch_count", 0) < 0
    ):
        raise Part2OperationalRepairError("Source campaign is not terminal and intact.")

    inputs = source.get("input_artifacts")
    if not isinstance(inputs, Mapping):
        raise Part2OperationalRepairError("Source input bindings are missing.")
    panel_path, _ = _bound_json(inputs["panel"], "source panel")
    compatibility_path, compatibility = _bound_json(
        inputs["compatibility"], "source compatibility"
    )
    registry_path, registry = _bound_json(inputs["registry"], "source registry")
    panel, contract = runner._load_panel(panel_path)
    selected = [str(row["target_id"]) for row in source["subject_routes"]]
    subjects, _judge = runner.select_routes(
        registry=registry, compatibility=compatibility, selected_ids=selected,
        judge_target_id=str(panel["judge_target_id"]),
    )
    if [row["route"] for row in subjects] != [row["route"] for row in source["subject_routes"]]:
        raise Part2OperationalRepairError("Exact source routes changed.")

    _trajectory_path, trajectory_payload = _bound_json(
        source["sanitized_artifacts"]["trajectory_metrics"],
        "source trajectory metrics",
    )
    source_rows = {
        (str(row["target_id"]), int(row["trajectory_index"])): row
        for row in trajectory_payload["rows"]
    }
    eligible = {
        key: row for key, row in source_rows.items()
        if not bool(row["operationally_eligible"])
    }
    if not eligible:
        raise Part2OperationalRepairError("Source has no operational failures to repair.")
    if any(
        int(row["transport_failure_count"]) == 0
        and int(row["identity_mismatch_count"]) == 0
        for row in eligible.values()
    ):
        raise Part2OperationalRepairError("Semantic INVALID was selected for repair.")

    bindings = _bindings(source_path, source, max_rounds, credential_pool_binding)
    if output_dir.exists() and not resume:
        raise Part2OperationalRepairError("Repair output exists; use --resume.")
    private_dir = output_dir / "private"
    journal_dir = private_dir / "trajectories"
    sanitized_dir = output_dir / "sanitized"
    for directory in (output_dir, private_dir, journal_dir, sanitized_dir):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        runner._secure_mode(directory, 0o700)

    repair_lock = runner._acquire_run_lock(private_dir)
    try:
        manifest_path = private_dir / "manifest.json"
        subject_by_id = {str(row["target_id"]): row for row in subjects}
        journals: dict[tuple[str, int, int], runner._ChainedJournal] = {}
        for target_id, trajectory_index in eligible:
            for round_index in range(1, max_rounds + 1):
                path = (
                    journal_dir / runner._safe_file_stem(target_id)
                    / f"seed-{trajectory_index:03d}-round-{round_index:02d}.jsonl"
                )
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                runner._secure_mode(path.parent, 0o700)
                journals[(target_id, trajectory_index, round_index)] = runner._ChainedJournal(path)

        if resume:
            manifest = runner._read_json(manifest_path, "repair manifest")
            if manifest.get("evidence_sha256") != runner._self_hash(manifest):
                raise Part2OperationalRepairError("Repair manifest self-hash failed.")
            mutable = {"created_at_utc", "last_updated_at_utc", "completed_at_utc", "complete", "summary", "journals", "sanitized_artifacts", "evidence_sha256"}
            if {k: v for k, v in manifest.items() if k not in mutable} != bindings:
                raise Part2OperationalRepairError("Repair bindings changed on resume.")
            refs = manifest.get("journals", {})
            for key, journal in journals.items():
                runner._validate_checkpoint_reference(
                    journal, refs[f"{key[0]}::{key[1]}::{key[2]}"],
                    label=f"repair trajectory {key}",
                )
        else:
            manifest = {
                **bindings, "created_at_utc": runner._utc_now(),
                "last_updated_at_utc": runner._utc_now(), "complete": False,
                "summary": {}, "sanitized_artifacts": {},
                "journals": {
                    f"{key[0]}::{key[1]}::{key[2]}": journal.reference()
                    for key, journal in journals.items()
                },
            }
            runner._seal(manifest)
            runner._atomic_json(manifest_path, manifest)

        successful: dict[tuple[str, int], tuple[int, Mapping[str, Any]]] = {}
        env_seeds = list(source["common_environment_seeds"])
        for round_index in range(1, max_rounds + 1):
            work = [key for key in eligible if key not in successful]
            if not work:
                break
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=min(trajectory_workers, len(work))) as pool:
                futures = {}
                for target_id, trajectory_index in work:
                    future = pool.submit(
                        runner._run_trajectory,
                        subject=subject_by_id[target_id],
                        trajectory_index=trajectory_index,
                        environment_seed=env_seeds[trajectory_index], contract=contract,
                        journal=journals[(target_id, trajectory_index, round_index)],
                        client=client, participant_workers=participant_workers,
                        max_attempts=max_attempts,
                        initial_backoff_seconds=initial_backoff_seconds,
                        sleep_fn=runner.time.sleep,
                    )
                    futures[future] = (target_id, trajectory_index)
                for future in as_completed(futures):
                    key = futures[future]
                    row = future.result()
                    if row["operationally_eligible"]:
                        successful[key] = (round_index, row)
            manifest["journals"] = {
                f"{key[0]}::{key[1]}::{key[2]}": journal.reference()
                for key, journal in journals.items()
            }
            manifest["last_updated_at_utc"] = runner._utc_now()
            runner._seal(manifest)
            runner._atomic_json(manifest_path, manifest)

        effective_rows = []
        for key, row in sorted(source_rows.items()):
            if key in successful:
                repaired = dict(successful[key][1])
                repaired["operational_repair_round"] = successful[key][0]
                repaired["source_replaced_for_operational_failure"] = True
                effective_rows.append(repaired)
            else:
                retained = dict(row)
                retained["operational_repair_round"] = None
                retained["source_replaced_for_operational_failure"] = False
                effective_rows.append(retained)
        effective_models = runner._aggregate_models(
            effective_rows, subjects,
            expected_trajectories=int(source["part2_contract"]["independent_trajectories"]),
            capacity=int(source["part2_contract"]["resource_capacity"]),
        )
        trajectory_out = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "inference_hub_part2_operational_repair_effective_trajectory_metrics_v1",
            "panel_id": source["panel_id"], "generated_at_utc": runner._utc_now(),
            "source_manifest_evidence_sha256": source["evidence_sha256"],
            "rows": effective_rows,
        }
        model_out = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "inference_hub_part2_operational_repair_effective_model_metrics_v1",
            "panel_id": source["panel_id"], "generated_at_utc": runner._utc_now(),
            "source_manifest_evidence_sha256": source["evidence_sha256"],
            "rows": effective_models,
        }
        runner._seal(trajectory_out); runner._seal(model_out)
        trajectory_out_path = sanitized_dir / "effective_trajectory_metrics.json"
        model_out_path = sanitized_dir / "effective_model_metrics.json"
        runner._atomic_json(trajectory_out_path, trajectory_out)
        runner._atomic_json(model_out_path, model_out)
        unresolved = len(eligible) - len(successful)
        manifest["summary"] = {
            "source_operational_failure_trajectories": len(eligible),
            "operational_repairs_succeeded": len(successful),
            "operational_repairs_unresolved": unresolved,
        }
        manifest["complete"] = unresolved == 0
        manifest["sanitized_artifacts"] = {
            "effective_trajectory_metrics": {
                "path": str(trajectory_out_path.resolve()),
                "file_sha256": runner._sha256_file(trajectory_out_path),
                "evidence_sha256": trajectory_out["evidence_sha256"],
            },
            "effective_model_metrics": {
                "path": str(model_out_path.resolve()),
                "file_sha256": runner._sha256_file(model_out_path),
                "evidence_sha256": model_out["evidence_sha256"],
            },
        }
        manifest["last_updated_at_utc"] = runner._utc_now()
        if manifest["complete"]:
            manifest["completed_at_utc"] = runner._utc_now()
        runner._seal(manifest)
        runner._atomic_json(manifest_path, manifest)
        return manifest
    finally:
        fcntl.flock(repair_lock.fileno(), fcntl.LOCK_UN)
        repair_lock.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--trajectory-workers", type=int, default=4)
    parser.add_argument("--participant-workers", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--initial-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--rate-profile", default=runner.HIGH_LATENCY_ORIGINAL_SCALE_RATE_PROFILE)
    parser.add_argument("--credential-env-file", type=Path)
    parser.add_argument("--expected-api-key-count", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.credential_env_file is None:
            client = runner._runtime_client(args.timeout_seconds, args.rate_profile)
            credential_pool_binding = None
        else:
            client = _pooled_runtime_client(
                args.credential_env_file, args.expected_api_key_count,
                args.timeout_seconds, args.rate_profile,
            )
            credential_pool_binding = {
                "account_count": client.account_count,
                "selection_policy": "thread_safe_round_robin",
                "rate_limit_scope": "independent_per_account",
                "rate_limit_contract": client.rate_limit_contract,
                "implementation_files": {
                    str(Path(__file__).resolve()): runner._sha256_file(Path(__file__).resolve()),
                    str(Path(runner.__file__).resolve()): runner._sha256_file(Path(runner.__file__).resolve()),
                },
            }
        manifest = run_repair(
            source_manifest_path=args.source_manifest, output_dir=args.output_dir,
            client=client,
            max_rounds=args.max_rounds, trajectory_workers=args.trajectory_workers,
            participant_workers=args.participant_workers,
            max_attempts=args.max_attempts,
            initial_backoff_seconds=args.initial_backoff_seconds,
            resume=args.resume,
            credential_pool_binding=credential_pool_binding,
        )
    except (Part2OperationalRepairError, runner.InferenceHubPart2PanelError, OSError, TypeError, ValueError) as error:
        print(f"Part 2 operational repair failed: {error}")
        return 2
    print(f"Operational trajectories unresolved: {manifest['summary']['operational_repairs_unresolved']}")
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
