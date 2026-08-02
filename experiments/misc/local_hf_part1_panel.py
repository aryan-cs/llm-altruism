"""Run the full 384-root Part 1 draft bank against pinned offline controls.

This is a large-N exploratory scale/control run. The deterministic draft bank is
not human-approved, and the resulting data are therefore barred from confirmatory
or paper-result promotion. Models execute concurrently; prompts within a model
are batched so each revision is loaded exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.misc.local_hf_smoke import (
    LocalHFSmokeError,
    _read_registry,
    _resolve_snapshot,
    fingerprint_snapshot,
)
from experiments.part1.confirmatory_design import build_draft_bank, build_primary_schedule


SCHEMA_VERSION = 2
DEFAULT_BASE_SEED = 20260802
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_WORKERS = 2
DEFAULT_MAX_NEW_TOKENS = 32
_FINAL_ACTION = re.compile(r"(?:^|\n)\s*([XY])\s*$")
_TRANSFORMERS_IMPORT_LOCK = threading.Lock()
_MODEL_LOAD_LOCK = threading.Lock()
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


class LocalHFPanelError(RuntimeError):
    """The offline large-N panel violates its frozen execution contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _secure_mode(path: Path, mode: int) -> None:
    """Apply private POSIX permissions without pretending Windows has chmod ACLs."""

    if os.name == "posix":
        path.chmod(mode)


def _require_private_mode(path: Path, mode: int) -> None:
    if os.name != "posix":
        return
    observed = stat.S_IMODE(path.stat().st_mode)
    if observed != mode:
        raise LocalHFPanelError(
            f"Private artifact permissions are unsafe for {path.name}: "
            f"expected {mode:04o}, found {observed:04o}."
        )


def _self_hash(payload: Mapping[str, Any]) -> str:
    unhashed = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    return _sha256_bytes(_canonical_bytes(unhashed))


def _seal_manifest(payload: dict[str, Any]) -> None:
    payload["evidence_sha256"] = _self_hash(payload)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _secure_mode(path, 0o600)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _secure_mode(path, 0o600)


def _compact_snapshot_fingerprint(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "asset_count": fingerprint.get("asset_count"),
        "total_size_bytes": fingerprint.get("total_size_bytes"),
        "snapshot_tree_sha256": fingerprint.get("snapshot_tree_sha256"),
    }


def _validate_binding_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LocalHFPanelError(f"{label} must be a lowercase SHA-256.")
    return value


def _validate_snapshot_binding(binding: object) -> dict[str, Any]:
    if not isinstance(binding, Mapping) or set(binding) != {
        "asset_count",
        "total_size_bytes",
        "snapshot_tree_sha256",
    }:
        raise LocalHFPanelError("Model result lacks an exact snapshot fingerprint.")
    for field in ("asset_count", "total_size_bytes"):
        value = binding.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise LocalHFPanelError(f"Snapshot fingerprint {field} must be positive.")
    _validate_binding_digest(
        binding.get("snapshot_tree_sha256"), label="snapshot_tree_sha256"
    )
    return dict(binding)


def _validate_runtime_binding(runtime: object) -> dict[str, str]:
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "python",
        "torch",
        "transformers",
        "device",
    }:
        raise LocalHFPanelError("Model result lacks an exact runtime binding.")
    if any(not isinstance(value, str) or not value for value in runtime.values()):
        raise LocalHFPanelError("Runtime binding values must be non-empty strings.")
    return {str(key): str(value) for key, value in runtime.items()}


def _load_and_validate_rows(
    *,
    output_path: Path,
    model: Mapping[str, str],
    trials: Sequence[Any],
    snapshot_binding: Mapping[str, Any],
    runtime_binding: Mapping[str, str],
) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    _require_private_mode(output_path, 0o600)
    try:
        raw_lines = output_path.read_text(encoding="utf-8").splitlines()
        if any(not line.strip() for line in raw_lines):
            raise LocalHFPanelError(
                f"Existing model evidence contains a blank row: {output_path.name}."
            )
        rows = [json.loads(line) for line in raw_lines]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LocalHFPanelError(
            f"Existing model evidence is not valid UTF-8 JSONL: {output_path.name}."
        ) from error
    if len(rows) > len(trials):
        raise LocalHFPanelError(
            f"Existing model evidence exceeds the schedule: {output_path.name}."
        )
    snapshot_sha256 = str(snapshot_binding["snapshot_tree_sha256"])
    runtime_sha256 = _sha256_bytes(_canonical_bytes(runtime_binding))
    for index, (row, trial) in enumerate(zip(rows, trials, strict=False), start=1):
        if not isinstance(row, dict) or set(row) != _ROW_FIELDS:
            raise LocalHFPanelError(
                f"Existing model evidence row {index} has an invalid schema: "
                f"{output_path.name}."
            )
        response = row.get("response_text")
        if row["prompt_sha256"] != _sha256_bytes(
            str(row["prompt_text"]).encode("utf-8")
        ) or not isinstance(response, str) or row["response_sha256"] != _sha256_bytes(
            response.encode("utf-8")
        ):
            raise LocalHFPanelError(
                f"Existing model evidence row {index} has a content hash mismatch: "
                f"{output_path.name}."
            )
        action = _parse_final_action(response) if isinstance(response, str) else None
        expected = {
            "schema_version": SCHEMA_VERSION,
            "model_id": model["id"],
            "upstream_model_id": model["model_id"],
            "revision": model["revision"],
            "trial_id": trial.trial_id,
            "root_id": trial.root_id,
            "game": trial.game,
            "domain": trial.domain,
            "counterbalance_id": trial.counterbalance_id,
            "prompt_text": trial.prompt_text,
            "prompt_sha256": trial.prompt_hash,
            "parsed_action": action,
            "format_valid": action is not None,
            "snapshot_tree_sha256": snapshot_sha256,
            "runtime_sha256": runtime_sha256,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise LocalHFPanelError(
                f"Existing model evidence row {index} differs from its frozen "
                f"schedule/model/runtime binding: {output_path.name}."
            )
        timestamp = row.get("finished_at_utc")
        try:
            parsed_timestamp = datetime.fromisoformat(
                str(timestamp).replace("Z", "+00:00")
            )
        except ValueError as error:
            raise LocalHFPanelError(
                f"Existing model evidence row {index} has an invalid timestamp: "
                f"{output_path.name}."
            ) from error
        if parsed_timestamp.utcoffset() != timezone.utc.utcoffset(parsed_timestamp):
            raise LocalHFPanelError(
                f"Existing model evidence row {index} timestamp is not UTC: "
                f"{output_path.name}."
            )
    return rows


def _validate_completed_result(
    *,
    result: Mapping[str, Any],
    model: Mapping[str, str],
    trials: Sequence[Any],
    output_path: Path,
) -> None:
    if result.get("status") != "passed":
        return
    snapshot = _validate_snapshot_binding(result.get("snapshot"))
    runtime = _validate_runtime_binding(result.get("runtime"))
    rows = _load_and_validate_rows(
        output_path=output_path,
        model=model,
        trials=trials,
        snapshot_binding=snapshot,
        runtime_binding=runtime,
    )
    valid = sum(row["format_valid"] is True for row in rows)
    if (
        result.get("completed_trials") != len(rows)
        or len(rows) != len(trials)
        or result.get("valid_actions") != valid
        or result.get("invalid_actions") != len(rows) - valid
        or result.get("output_file") != output_path.name
        or result.get("output_sha256") != _sha256_file(output_path)
    ):
        raise LocalHFPanelError(
            f"Completed model result does not match retained rows: {output_path.name}."
        )


def _parse_final_action(response: str) -> str | None:
    match = _FINAL_ACTION.search(response.strip())
    return match.group(1) if match else None


def build_draft_trials(*, base_seed: int, limit: int | None = None) -> tuple[Any, ...]:
    trials = build_primary_schedule(
        build_draft_bank(),
        base_seed=base_seed,
        requested_provider="huggingface_transformers_offline",
        requested_model="per-model-pinned-revision",
        production=False,
    )
    if len(trials) != 384:
        raise LocalHFPanelError(f"Expected 384 Part 1 draft trials, found {len(trials)}.")
    if limit is None:
        return trials
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 384:
        raise LocalHFPanelError("limit must be an integer from 1 through 384.")
    return trials[:limit]


def _real_model_runner(
    *,
    model: Mapping[str, str],
    cache_root: Path,
    trials: Sequence[Any],
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        # Transformers uses lazy module initialization. Serialize only the first
        # imports; model loading and every generation batch remain parallel.
        with _TRANSFORMERS_IMPORT_LOCK:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise LocalHFPanelError(
            f"The selected runtime lacks torch/transformers: {error}"
        ) from error
    if device not in {"cpu", "mps", "cuda"}:
        raise LocalHFPanelError("device must be cpu, mps, or cuda.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise LocalHFPanelError("MPS was requested but is unavailable.")
    if device == "cuda" and not torch.cuda.is_available():
        raise LocalHFPanelError("CUDA was requested but is unavailable.")

    snapshot = _resolve_snapshot(cache_root, model)
    snapshot_binding = _compact_snapshot_fingerprint(
        fingerprint_snapshot(cache_root=cache_root, snapshot=snapshot)
    )
    runtime_binding = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "device": device,
    }
    existing_rows = _load_and_validate_rows(
        output_path=output_path,
        model=model,
        trials=trials,
        snapshot_binding=snapshot_binding,
        runtime_binding=runtime_binding,
    )
    pending_trials = trials[len(existing_rows) :]
    if not pending_trials:
        valid = sum(row.get("format_valid") is True for row in existing_rows)
        return {
            "status": "passed",
            "started_at_utc": _utc_now(),
            "finished_at_utc": _utc_now(),
            "completed_trials": len(existing_rows),
            "valid_actions": valid,
            "invalid_actions": len(existing_rows) - valid,
            "output_file": output_path.name,
            "output_sha256": _sha256_file(output_path),
            "resumed_from_trial_count": len(existing_rows),
            "snapshot": snapshot_binding,
            "runtime": runtime_binding,
        }

    with _MODEL_LOAD_LOCK:
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        loaded = AutoModelForCausalLM.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False, dtype="auto"
        )
    loaded.to(device)
    loaded.eval()
    started = _utc_now()
    completed = len(existing_rows)
    valid = sum(row.get("format_valid") is True for row in existing_rows)
    with torch.inference_mode():
        for offset in range(0, len(pending_trials), batch_size):
            batch = pending_trials[offset : offset + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "Follow the response format exactly.",
                        },
                        {"role": "user", "content": trial.prompt_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for trial in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            kwargs: dict[str, Any] = {
                **encoded,
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            }
            if tokenizer.pad_token_id is not None:
                kwargs["pad_token_id"] = tokenizer.pad_token_id
            generated = loaded.generate(**kwargs)
            input_width = int(encoded["input_ids"].shape[1])
            responses = tokenizer.batch_decode(
                generated[:, input_width:], skip_special_tokens=True
            )
            rows = []
            for trial, response in zip(batch, responses, strict=True):
                response = response.strip()
                action = _parse_final_action(response)
                valid += action is not None
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "model_id": model["id"],
                        "upstream_model_id": model["model_id"],
                        "revision": model["revision"],
                        "trial_id": trial.trial_id,
                        "root_id": trial.root_id,
                        "game": trial.game,
                        "domain": trial.domain,
                        "counterbalance_id": trial.counterbalance_id,
                        "prompt_text": trial.prompt_text,
                        "prompt_sha256": trial.prompt_hash,
                        "response_text": response,
                        "response_sha256": _sha256_bytes(response.encode("utf-8")),
                        "parsed_action": action,
                        "format_valid": action is not None,
                        "snapshot_tree_sha256": snapshot_binding[
                            "snapshot_tree_sha256"
                        ],
                        "runtime_sha256": _sha256_bytes(
                            _canonical_bytes(runtime_binding)
                        ),
                        "finished_at_utc": _utc_now(),
                    }
                )
            _append_jsonl(output_path, rows)
            completed += len(rows)
    return {
        "status": "passed",
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "completed_trials": completed,
        "valid_actions": valid,
        "invalid_actions": completed - valid,
        "output_file": output_path.name,
        "output_sha256": _sha256_file(output_path),
        "resumed_from_trial_count": len(existing_rows),
        "snapshot": snapshot_binding,
        "runtime": runtime_binding,
    }


def run_panel(
    *,
    registry_path: Path,
    cache_root: Path,
    output_dir: Path,
    selected_ids: list[str] | None = None,
    base_seed: int = DEFAULT_BASE_SEED,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device: str = "cpu",
    resume: bool = False,
    model_runner: Callable[..., dict[str, Any]] = _real_model_runner,
) -> dict[str, Any]:
    for name, value in (
        ("batch_size", batch_size),
        ("max_workers", max_workers),
        ("max_new_tokens", max_new_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise LocalHFPanelError(f"{name} must be a positive integer.")
    if output_dir.exists() and not resume:
        raise LocalHFPanelError("Output directory already exists; choose a new evidence path.")
    if not output_dir.exists() and resume:
        raise LocalHFPanelError("Cannot resume because the evidence directory does not exist.")
    output_dir.mkdir(parents=True, exist_ok=resume, mode=0o700)
    _secure_mode(output_dir, 0o700)
    _require_private_mode(output_dir, 0o700)
    registry, models = _read_registry(registry_path)
    by_id = {row["id"]: row for row in models}
    requested = selected_ids or list(by_id)
    if len(requested) != len(set(requested)):
        raise LocalHFPanelError("A model was selected more than once.")
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise LocalHFPanelError(f"Unknown model ids: {', '.join(unknown)}.")
    trials = build_draft_trials(base_seed=base_seed, limit=limit)
    schedule_binding = [
        {
            "trial_id": trial.trial_id,
            "root_id": trial.root_id,
            "game": trial.game,
            "domain": trial.domain,
            "prompt_sha256": trial.prompt_hash,
        }
        for trial in trials
    ]
    expected_contract = {
        "mode": "greedy",
        "base_seed": base_seed,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "local_files_only": True,
        "trust_remote_code": False,
        "device": device,
        "snapshot_binding": "recursive_sha256_all_snapshot_assets",
        "runtime_binding": "python_torch_transformers_device",
        "parallelization": "bounded_model_level_thread_pool_with_batched_prompts",
        "max_workers": min(max_workers, len(requested)),
    }
    fresh_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "local_hf_part1_large_n_exploratory_panel",
        "created_at_utc": _utc_now(),
        "analysis_role": "exploratory_local_scale_control_only",
        "draft_bank_human_approved": False,
        "confirmatory_or_paper_promotion_permitted": False,
        "frontier_route_substitution_permitted": False,
        "registry_version": registry.get("registry_version"),
        "registry_sha256": _sha256_file(registry_path),
        "trial_count_per_model": len(trials),
        "total_planned_generations": len(trials) * len(requested),
        "schedule_sha256": _sha256_bytes(_canonical_bytes(schedule_binding)),
        "generation_contract": expected_contract,
        "selected_model_ids": requested,
        "models": {
            model_id: {"status": "reserved_before_dispatch"} for model_id in requested
        },
        "complete": False,
    }
    manifest_path = output_dir / "manifest.json"
    if resume:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LocalHFPanelError("Existing panel manifest is not valid JSON.") from error
        _require_private_mode(manifest_path, 0o600)
        recorded_manifest_hash = manifest.get("evidence_sha256")
        if recorded_manifest_hash != _self_hash(manifest):
            raise LocalHFPanelError("Existing panel manifest self-hash is invalid.")
        exact_bindings = (
            "schema_version",
            "artifact_type",
            "analysis_role",
            "draft_bank_human_approved",
            "confirmatory_or_paper_promotion_permitted",
            "frontier_route_substitution_permitted",
            "registry_version",
            "registry_sha256",
            "trial_count_per_model",
            "total_planned_generations",
            "schedule_sha256",
            "generation_contract",
            "selected_model_ids",
        )
        if any(manifest.get(key) != fresh_manifest[key] for key in exact_bindings):
            raise LocalHFPanelError("Existing panel manifest does not match this run contract.")
        history = manifest.setdefault("model_attempt_history", {})
        for model_id in requested:
            prior = manifest.get("models", {}).get(model_id)
            if isinstance(prior, Mapping) and prior.get("status") == "passed":
                _validate_completed_result(
                    result=prior,
                    model=by_id[model_id],
                    trials=trials,
                    output_path=output_dir / f"{model_id.replace('.', '_')}.jsonl",
                )
            if isinstance(prior, Mapping) and prior.get("status") not in {
                "reserved_before_dispatch",
                "passed",
            }:
                history.setdefault(model_id, []).append(dict(prior))
            if prior is None or prior.get("status") != "passed":
                manifest["models"][model_id] = {"status": "reserved_before_dispatch"}
        manifest["last_resumed_at_utc"] = _utc_now()
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
    else:
        manifest = fresh_manifest
    _seal_manifest(manifest)
    _atomic_json(manifest_path, manifest)

    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(requested)),
        thread_name_prefix="local-hf-part1",
    ) as executor:
        futures = {}
        for model_id in requested:
            output_path = output_dir / f"{model_id.replace('.', '_')}.jsonl"
            futures[
                executor.submit(
                    model_runner,
                    model=by_id[model_id],
                    cache_root=cache_root,
                    trials=trials,
                    output_path=output_path,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
            ] = model_id
        for future in as_completed(futures):
            model_id = futures[future]
            try:
                result = future.result()
            except BaseException as error:
                manifest["models"][model_id] = {
                    "status": "failed",
                    "finished_at_utc": _utc_now(),
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            else:
                output_path = output_dir / f"{model_id.replace('.', '_')}.jsonl"
                try:
                    _validate_completed_result(
                        result=result,
                        model=by_id[model_id],
                        trials=trials,
                        output_path=output_path,
                    )
                except Exception as error:
                    manifest["models"][model_id] = {
                        "status": "failed",
                        "finished_at_utc": _utc_now(),
                        "error": {"type": type(error).__name__, "message": str(error)},
                    }
                else:
                    manifest["models"][model_id] = result
            manifest["complete"] = all(
                row.get("status") == "passed" for row in manifest["models"].values()
            )
            _seal_manifest(manifest)
            _atomic_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 384-root exploratory Part 1 panel on pinned local models."
    )
    parser.add_argument("--registry", type=Path, default=Path("agents/local_control.registry.json"))
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cache_root = args.cache_root
    if cache_root is None:
        value = os.environ.get("HF_LOCAL_CACHE_ROOT", "").strip()
        if not value:
            raise LocalHFSmokeError("Set HF_LOCAL_CACHE_ROOT or pass --cache-root.")
        cache_root = Path(value)
    manifest = run_panel(
        registry_path=args.registry,
        cache_root=cache_root,
        output_dir=args.output_dir,
        selected_ids=args.models,
        base_seed=args.base_seed,
        limit=args.limit,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        resume=args.resume,
    )
    passed = sum(row.get("status") == "passed" for row in manifest["models"].values())
    print(
        f"Completed {passed}/{len(manifest['models'])} models and "
        f"{manifest['total_planned_generations']} planned generations."
    )
    print(f"Evidence: {args.output_dir / 'manifest.json'}")
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
