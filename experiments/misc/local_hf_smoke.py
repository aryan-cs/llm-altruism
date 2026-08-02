"""Offline, revision-pinned Hugging Face smoke verification for local controls.

The runner never downloads assets, never trusts remote code, and never promotes
a local control into the frontier panel. It fingerprints every snapshot asset,
performs one real greedy generation per selected model, and atomically retains
both successful and failed attempts in a private evidence artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


EVIDENCE_SCHEMA_VERSION = 1
SMOKE_MESSAGES = (
    {"role": "system", "content": "Follow the user's response format exactly."},
    {"role": "user", "content": "Reply with exactly READY and nothing else."},
)
SMOKE_MAX_NEW_TOKENS = 16


class LocalHFSmokeError(RuntimeError):
    """Local assets or runtime behavior violate the offline smoke contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_registry(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LocalHFSmokeError("Local-control registry is not readable UTF-8 JSON.") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise LocalHFSmokeError("Local-control registry schema_version must be 1.")
    if payload.get("frontier_route_substitution_permitted") is not False:
        raise LocalHFSmokeError("Local controls must forbid frontier-route substitution.")
    if payload.get("paper_result_promotion_permitted") is not False:
        raise LocalHFSmokeError("Local-control smokes must not be paper results.")
    contract = payload.get("runtime_contract")
    if not isinstance(contract, Mapping):
        raise LocalHFSmokeError("Local-control registry lacks a runtime contract.")
    if (
        contract.get("backend") != "huggingface_transformers_offline"
        or contract.get("local_files_only") is not True
        or contract.get("remote_code_permitted") is not False
        or contract.get("generation_mode") != "greedy"
    ):
        raise LocalHFSmokeError("Local-control runtime contract is not fail-closed.")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise LocalHFSmokeError("Local-control registry must contain models.")
    models: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_models):
        if not isinstance(raw, Mapping):
            raise LocalHFSmokeError(f"Local model row {index} is invalid.")
        required = {
            key: raw.get(key)
            for key in (
                "id",
                "model_id",
                "revision",
                "cache_repository_dir",
                "parameter_scale",
                "verification_status",
            )
        }
        if any(not isinstance(value, str) or not value for value in required.values()):
            raise LocalHFSmokeError(f"Local model row {index} lacks a required string.")
        model_id = required["id"]
        if model_id in seen:
            raise LocalHFSmokeError(f"Duplicate local model id: {model_id}.")
        if required["verification_status"] != "smoke_pending":
            raise LocalHFSmokeError(f"Local model {model_id} is not smoke_pending.")
        revision = required["revision"]
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise LocalHFSmokeError(f"Local model {model_id} lacks a pinned commit revision.")
        repository_dir = required["cache_repository_dir"]
        if Path(repository_dir).name != repository_dir or not repository_dir.startswith("models--"):
            raise LocalHFSmokeError(f"Local model {model_id} has an unsafe cache directory.")
        seen.add(model_id)
        models.append(required)
    return payload, models


def _resolve_snapshot(cache_root: Path, model: Mapping[str, str]) -> Path:
    root = cache_root.resolve(strict=True)
    snapshot = (
        root
        / model["cache_repository_dir"]
        / "snapshots"
        / model["revision"]
    )
    try:
        resolved = snapshot.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise LocalHFSmokeError(f"Pinned snapshot is unavailable for {model['id']}.") from error
    if not resolved.is_dir() or not resolved.is_relative_to(root):
        raise LocalHFSmokeError(f"Pinned snapshot escapes the cache root for {model['id']}.")
    return snapshot


def fingerprint_snapshot(
    *, cache_root: Path, snapshot: Path
) -> dict[str, Any]:
    root = cache_root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for asset in sorted(snapshot.iterdir(), key=lambda path: path.name):
        if asset.name.startswith(".") or asset.is_dir():
            continue
        try:
            resolved = asset.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise LocalHFSmokeError(f"Snapshot asset {asset.name} cannot be resolved.") from error
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise LocalHFSmokeError(f"Snapshot asset {asset.name} escapes the cache root.")
        records.append(
            {
                "name": asset.name,
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
            }
        )
    required = {"config.json", "tokenizer_config.json"}
    names = {record["name"] for record in records}
    if not required.issubset(names):
        raise LocalHFSmokeError("Snapshot lacks required config/tokenizer assets.")
    if not any(name.endswith((".safetensors", ".bin")) for name in names):
        raise LocalHFSmokeError("Snapshot lacks model weights.")
    return {
        "asset_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "assets": records,
        "snapshot_tree_sha256": _sha256_bytes(_canonical_bytes(records)),
    }


def _generate_with_transformers(
    *, snapshot: Path, device: str, seed: int
) -> tuple[str, dict[str, str]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise LocalHFSmokeError(
            "The selected Python runtime lacks torch and transformers."
        ) from error
    if device not in {"cpu", "mps", "cuda"}:
        raise LocalHFSmokeError("Device must be cpu, mps, or cuda.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise LocalHFSmokeError("MPS was requested but is unavailable.")
    if device == "cuda" and not torch.cuda.is_available():
        raise LocalHFSmokeError("CUDA was requested but is unavailable.")
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype="auto",
    )
    model.to(device)
    model.eval()
    prompt = tokenizer.apply_chat_template(
        list(SMOKE_MESSAGES),
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation_kwargs: dict[str, Any] = {
        **encoded,
        "do_sample": False,
        "max_new_tokens": SMOKE_MAX_NEW_TOKENS,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    with torch.inference_mode():
        generated = model.generate(**generation_kwargs)
    input_length = int(encoded["input_ids"].shape[-1])
    response = tokenizer.decode(
        generated[0, input_length:],
        skip_special_tokens=True,
    ).strip()
    if not response:
        raise LocalHFSmokeError("Greedy smoke generation returned no visible text.")
    return response, {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "device": device,
    }


def _artifact_digest(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(payload))


def run_smokes(
    *,
    registry_path: Path,
    cache_root: Path,
    output_path: Path,
    selected_ids: list[str] | None = None,
    device: str = "cpu",
    seed: int = 20260802,
    generator: Callable[..., tuple[str, dict[str, str]]] = _generate_with_transformers,
) -> dict[str, Any]:
    registry, models = _read_registry(registry_path)
    by_id = {model["id"]: model for model in models}
    requested = selected_ids or list(by_id)
    if len(requested) != len(set(requested)):
        raise LocalHFSmokeError("A local model was selected more than once.")
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise LocalHFSmokeError(f"Unknown local model ids: {', '.join(unknown)}.")
    prompt_sha = _sha256_bytes(_canonical_bytes(list(SMOKE_MESSAGES)))
    payload: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "artifact_type": "local_hf_smoke_evidence",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "registry_version": registry.get("registry_version"),
        "registry_sha256": _sha256_file(registry_path),
        "analysis_role": registry.get("analysis_role"),
        "frontier_route_substitution_permitted": False,
        "paper_result_promotion_permitted": False,
        "generation_contract": {
            "mode": "greedy",
            "seed": seed,
            "max_new_tokens": SMOKE_MAX_NEW_TOKENS,
            "prompt_sha256": prompt_sha,
            "local_files_only": True,
            "trust_remote_code": False,
        },
        "selected_model_ids": requested,
        "attempts": [],
        "complete": False,
    }
    _atomic_write_json(output_path, payload)
    any_failure = False
    for model_id in requested:
        model = by_id[model_id]
        started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        attempt: dict[str, Any] = {
            "model_id": model_id,
            "upstream_model_id": model["model_id"],
            "revision": model["revision"],
            "parameter_scale": model["parameter_scale"],
            "started_at_utc": started,
            "status": "failed",
        }
        try:
            snapshot = _resolve_snapshot(cache_root, model)
            attempt["snapshot"] = fingerprint_snapshot(
                cache_root=cache_root,
                snapshot=snapshot,
            )
            response, runtime = generator(snapshot=snapshot, device=device, seed=seed)
            if not isinstance(response, str) or not response.strip():
                raise LocalHFSmokeError("Generator returned no visible response text.")
            attempt.update(
                {
                    "status": "passed",
                    "runtime": runtime,
                    "response_text": response,
                    "response_sha256": _sha256_bytes(response.encode("utf-8")),
                    "format_contract_match": response == "READY",
                }
            )
        except Exception as error:  # retained evidence, then continue the panel
            any_failure = True
            message = str(error).replace(str(cache_root), "<cache-root>")
            attempt["error"] = {
                "type": type(error).__name__,
                "message": message,
            }
        attempt["finished_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        payload["attempts"].append(attempt)
        payload["complete"] = (
            len(payload["attempts"]) == len(requested) and not any_failure
        )
        payload.pop("evidence_sha256", None)
        payload["evidence_sha256"] = _artifact_digest(payload)
        _atomic_write_json(output_path, payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real offline generations for pinned local HF controls."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("agents/local_control.registry.json"),
    )
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260802)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cache_root = args.cache_root
    if cache_root is None:
        value = os.environ.get("HF_LOCAL_CACHE_ROOT", "").strip()
        if not value:
            raise LocalHFSmokeError(
                "Set HF_LOCAL_CACHE_ROOT or pass --cache-root to the offline cache."
            )
        cache_root = Path(value)
    evidence = run_smokes(
        registry_path=args.registry,
        cache_root=cache_root,
        output_path=args.output,
        selected_ids=args.models,
        device=args.device,
        seed=args.seed,
    )
    passed = sum(attempt["status"] == "passed" for attempt in evidence["attempts"])
    print(f"Passed {passed}/{len(evidence['attempts'])} local HF generation smokes.")
    print(f"Evidence: {args.output}")
    return 0 if evidence["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
