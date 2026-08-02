"""Build a provenance-preserving Part 0 stimulus registry.

The legacy runner concatenated stripped one-column CSVs. Confirmatory runs use
the pinned upstream tables instead: exact normalized duplicates are collapsed,
all source memberships are retained, and JailbreakBench harmful/benign pairs
remain linked. The generated registry contains harmful text and should be kept
out of the default anonymous supplement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import ssl
import tempfile
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import certifi

HARM_BENCH_COMMIT = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
JBB_COMMIT = "886acc352a31533ffbcf4ef22c744658688086fc"
SOURCE_SPECS = {
    "harmbench_harmful": {
        "url": (
            "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
            f"{HARM_BENCH_COMMIT}/data/behavior_datasets/"
            "harmbench_behaviors_text_all.csv"
        ),
        "sha256": "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc",
    },
    "jbb_harmful": {
        "url": (
            "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/"
            f"resolve/{JBB_COMMIT}/data/harmful-behaviors.csv"
        ),
        "sha256": "4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1",
    },
    "jbb_benign": {
        "url": (
            "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/"
            f"resolve/{JBB_COMMIT}/data/benign-behaviors.csv"
        ),
        "sha256": "3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468",
    },
}
STIMULUS_REGISTRY_SCHEMA_VERSION = 1
TARGET_LANGUAGES = ("english", "chinese", "russian")


class StimulusRegistryError(RuntimeError):
    """The pinned sources or stimulus registry violate their contract."""


def normalized_prompt_key(text: str) -> str:
    """Normalize Unicode and whitespace for deterministic exact deduplication."""

    if not isinstance(text, str):
        raise TypeError("Prompt text must be a string.")
    normalized = unicodedata.normalize("NFKC", text)
    normalized = " ".join(normalized.split()).strip()
    if not normalized:
        raise StimulusRegistryError("Prompt text is empty after normalization.")
    return normalized.casefold()


def _stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:20]}"


def _draft_review_fields(prompt_english: str) -> dict[str, Any]:
    registry = {
        "semantic_cluster_id": None,
        "semantic_cluster_review": {
            "status": "pending_human_review",
            "reviewer_ids": [],
        },
        "localized_stimuli": {
            "english": {
                "text": prompt_english,
                "sha256": hashlib.sha256(prompt_english.encode("utf-8")).hexdigest(),
                "translation_method": "source_language",
                "review_status": "source_language",
                "reviewer_ids": [],
            }
        },
    }
    return registry


def _draft_response_instructions() -> dict[str, dict[str, Any]]:
    """Return explicit, unapproved placeholders for localized instructions.

    Response instructions are research stimuli too.  The registry builder must
    not invent translations or approvals, so draft registries leave their text
    and hashes unset until qualified reviewers supply and approve them.
    """

    return {
        language: {
            "text": None,
            "sha256": None,
            "review_status": "pending_human_review",
            "reviewer_ids": [],
        }
        for language in TARGET_LANGUAGES
    }


def _parse_csv(payload: bytes, *, source_name: str) -> list[dict[str, str]]:
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise StimulusRegistryError(f"{source_name} is not UTF-8 CSV.") from error
    reader = csv.DictReader(io.StringIO(decoded))
    if reader.fieldnames is None:
        raise StimulusRegistryError(f"{source_name} has no CSV header.")
    rows = [{key: value or "" for key, value in row.items()} for row in reader]
    if not rows:
        raise StimulusRegistryError(f"{source_name} has no rows.")
    return rows


def fetch_pinned_sources(timeout_seconds: float = 60.0) -> dict[str, bytes]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    payloads: dict[str, bytes] = {}
    tls_context = ssl.create_default_context(cafile=certifi.where())
    for source_name, source in SOURCE_SPECS.items():
        request = urllib.request.Request(
            str(source["url"]),
            headers={"Accept": "text/csv", "User-Agent": "llm-altruism/0.1"},
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - pinned HTTPS URLs and hashes
                request,
                timeout=timeout_seconds,
                context=tls_context,
            ) as response:
                payload = response.read()
        except (urllib.error.HTTPError, urllib.error.URLError) as error:
            raise StimulusRegistryError(
                f"Could not fetch pinned source {source_name}."
            ) from error
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != source["sha256"]:
            raise StimulusRegistryError(
                f"Pinned source {source_name} hash mismatch: {actual_hash}."
            )
        payloads[source_name] = payload
    return payloads


def _provenance_row(
    *,
    source_name: str,
    source_row_id: str,
    source_record: Mapping[str, str],
    source_category: str,
    source_behavior: str,
    source_origin: str,
    functional_category: str = "",
    semantic_category: str = "",
) -> dict[str, Any]:
    registry = {
        "source_name": source_name,
        "source_commit": (
            HARM_BENCH_COMMIT if source_name == "harmbench" else JBB_COMMIT
        ),
        "source_file_sha256": SOURCE_SPECS[
            "harmbench_harmful" if source_name == "harmbench" else "jbb_harmful"
        ]["sha256"],
        "source_row_id": source_row_id,
        "source_category": source_category.strip(),
        "source_behavior": " ".join(source_behavior.split()),
        "source_origin": source_origin.strip(),
        "functional_category": functional_category.strip(),
        "semantic_category": semantic_category.strip(),
        "source_record_sha256": hashlib.sha256(
            json.dumps(
                dict(source_record),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    return registry


def _validate_headers(
    rows: list[dict[str, str]],
    required: Iterable[str],
    *,
    source_name: str,
) -> None:
    missing = sorted(set(required) - set(rows[0]))
    if missing:
        raise StimulusRegistryError(
            f"{source_name} is missing columns: {', '.join(missing)}."
        )


def build_stimulus_registry(payloads: Mapping[str, bytes]) -> dict[str, Any]:
    """Build the exact-deduplicated harmful set plus paired JBB controls."""

    required_payloads = set(SOURCE_SPECS)
    if set(payloads) != required_payloads:
        missing = sorted(required_payloads - set(payloads))
        extra = sorted(set(payloads) - required_payloads)
        raise StimulusRegistryError(
            f"Source payload keys differ (missing={missing}, extra={extra})."
        )
    for source_name, payload in payloads.items():
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != SOURCE_SPECS[source_name]["sha256"]:
            raise StimulusRegistryError(
                f"Source {source_name} hash mismatch: {actual_hash}."
            )

    harmbench = _parse_csv(payloads["harmbench_harmful"], source_name="HarmBench")
    jbb_harmful = _parse_csv(payloads["jbb_harmful"], source_name="JBB harmful")
    jbb_benign = _parse_csv(payloads["jbb_benign"], source_name="JBB benign")
    _validate_headers(
        harmbench,
        (
            "Behavior",
            "FunctionalCategory",
            "SemanticCategory",
            "Tags",
            "ContextString",
            "BehaviorID",
        ),
        source_name="HarmBench",
    )
    _validate_headers(
        jbb_harmful,
        ("Index", "Goal", "Target", "Behavior", "Category", "Source"),
        source_name="JBB harmful",
    )
    _validate_headers(
        jbb_benign,
        ("Index", "Goal", "Target", "Behavior", "Category", "Source"),
        source_name="JBB benign",
    )

    harmful_by_key: dict[str, dict[str, Any]] = {}
    for row in harmbench:
        prompt = " ".join(row["Behavior"].split())
        key = normalized_prompt_key(prompt)
        item = harmful_by_key.setdefault(
            key,
            {
                "base_prompt_id": _stable_id("harm", key),
                "arm": "harmful",
                "prompt_english": prompt,
                "normalized_prompt_sha256": hashlib.sha256(
                    key.encode("utf-8")
                ).hexdigest(),
                "pair_id": None,
                "provenance": [],
                **_draft_review_fields(prompt),
            },
        )
        item["provenance"].append(
            _provenance_row(
                source_name="harmbench",
                source_row_id=row["BehaviorID"].strip(),
                source_record=row,
                source_category=row["SemanticCategory"],
                source_behavior=row["BehaviorID"],
                source_origin=row["Tags"],
                functional_category=row["FunctionalCategory"],
                semantic_category=row["SemanticCategory"],
            )
        )

    harmful_indices = {row["Index"].strip() for row in jbb_harmful}
    benign_indices = {row["Index"].strip() for row in jbb_benign}
    if harmful_indices != benign_indices or len(harmful_indices) != len(jbb_harmful):
        raise StimulusRegistryError(
            "JBB harmful and benign tables must contain unique matching indices."
        )
    benign_by_index = {row["Index"].strip(): row for row in jbb_benign}
    controls: list[dict[str, Any]] = []
    for row in jbb_harmful:
        index = row["Index"].strip()
        benign = benign_by_index[index]
        if normalized_prompt_key(row["Category"]) != normalized_prompt_key(
            benign["Category"]
        ):
            raise StimulusRegistryError(f"JBB pair {index} has mismatched categories.")
        if normalized_prompt_key(row["Behavior"]) != normalized_prompt_key(
            benign["Behavior"]
        ):
            raise StimulusRegistryError(f"JBB pair {index} has mismatched behaviors.")
        pair_id = f"jbb_pair_{int(index):03d}"
        prompt = " ".join(row["Goal"].split())
        key = normalized_prompt_key(prompt)
        item = harmful_by_key.setdefault(
            key,
            {
                "base_prompt_id": _stable_id("harm", key),
                "arm": "harmful",
                "prompt_english": prompt,
                "normalized_prompt_sha256": hashlib.sha256(
                    key.encode("utf-8")
                ).hexdigest(),
                "pair_id": pair_id,
                "provenance": [],
                **_draft_review_fields(prompt),
            },
        )
        if item["pair_id"] not in {None, pair_id}:
            raise StimulusRegistryError(
                f"Harmful prompt maps to multiple JBB pairs: {item['base_prompt_id']}."
            )
        item["pair_id"] = pair_id
        item["provenance"].append(
            _provenance_row(
                source_name="jbb",
                source_row_id=index,
                source_record=row,
                source_category=row["Category"],
                source_behavior=row["Behavior"],
                source_origin=row["Source"],
            )
        )

        control_prompt = " ".join(benign["Goal"].split())
        control_key = normalized_prompt_key(control_prompt)
        controls.append(
            {
                "base_prompt_id": _stable_id("control", control_key),
                "arm": "control",
                "prompt_english": control_prompt,
                "normalized_prompt_sha256": hashlib.sha256(
                    control_key.encode("utf-8")
                ).hexdigest(),
                "pair_id": pair_id,
                "provenance": [
                    {
                        **_provenance_row(
                            source_name="jbb",
                            source_row_id=index,
                            source_record=benign,
                            source_category=benign["Category"],
                            source_behavior=benign["Behavior"],
                            source_origin=benign["Source"],
                        ),
                        "source_file_sha256": SOURCE_SPECS["jbb_benign"]["sha256"],
                    }
                ],
                **_draft_review_fields(control_prompt),
            }
        )

    harmful = sorted(harmful_by_key.values(), key=lambda item: item["base_prompt_id"])
    controls.sort(key=lambda item: item["pair_id"])
    all_ids = [item["base_prompt_id"] for item in [*harmful, *controls]]
    if len(all_ids) != len(set(all_ids)):
        raise StimulusRegistryError("Stable base_prompt_id collision detected.")
    control_keys = [item["normalized_prompt_sha256"] for item in controls]
    if len(control_keys) != len(set(control_keys)):
        raise StimulusRegistryError("JBB control prompts contain normalized duplicates.")

    source_rows = len(harmbench) + len(jbb_harmful)
    registry = {
        "schema_version": STIMULUS_REGISTRY_SCHEMA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "deduplication": "Unicode NFKC, whitespace collapse, casefold, exact match",
        "source_specs": SOURCE_SPECS,
        "counts": {
            "harmful_source_rows": source_rows,
            "harmful_unique_prompts": len(harmful),
            "harmful_duplicates_collapsed": source_rows - len(harmful),
            "paired_control_prompts": len(controls),
            "jbb_pairs": len(jbb_harmful),
        },
        "response_instructions": _draft_response_instructions(),
        "stimuli": [*harmful, *controls],
    }
    validate_stimulus_registry(registry)
    return registry


def validate_stimulus_registry(
    registry: Mapping[str, Any],
    *,
    require_production_approval: bool = False,
) -> None:
    """Validate hashes/pairs and optionally require genuine human approvals."""

    if registry.get("schema_version") != STIMULUS_REGISTRY_SCHEMA_VERSION:
        raise StimulusRegistryError("Unsupported Part 0 stimulus registry schema.")
    if registry.get("source_specs") != SOURCE_SPECS:
        raise StimulusRegistryError("Part 0 registry source pins do not match code.")
    response_instructions = registry.get("response_instructions")
    if require_production_approval:
        if not isinstance(response_instructions, Mapping):
            raise StimulusRegistryError(
                "Part 0 registry lacks approved response_instructions."
            )
        for language in TARGET_LANGUAGES:
            instruction = response_instructions.get(language)
            if not isinstance(instruction, Mapping):
                raise StimulusRegistryError(
                    f"Part 0 registry lacks the exact {language} response instruction."
                )
            text = instruction.get("text")
            if not isinstance(text, str) or not text.strip():
                raise StimulusRegistryError(
                    f"Part 0 {language} response instruction is empty."
                )
            if instruction.get("sha256") != hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest():
                raise StimulusRegistryError(
                    f"Part 0 {language} response instruction hash mismatch."
                )
            if instruction.get("review_status") != "approved":
                raise StimulusRegistryError(
                    f"Part 0 {language} response instruction is not approved."
                )
            instruction_reviewers = instruction.get("reviewer_ids")
            if not _valid_reviewer_ids(instruction_reviewers):
                raise StimulusRegistryError(
                    f"Part 0 {language} response instruction lacks reviewer provenance."
                )
    stimuli = registry.get("stimuli")
    if not isinstance(stimuli, list) or not stimuli:
        raise StimulusRegistryError("Part 0 registry has no stimuli.")
    seen_ids: set[str] = set()
    pair_arms: dict[str, set[str]] = {}
    harmful_count = 0
    control_count = 0
    for index, item in enumerate(stimuli):
        if not isinstance(item, Mapping):
            raise StimulusRegistryError(f"stimuli[{index}] is not an object.")
        item_id = item.get("base_prompt_id")
        arm = item.get("arm")
        prompt = item.get("prompt_english")
        if not isinstance(item_id, str) or not item_id:
            raise StimulusRegistryError(f"stimuli[{index}] lacks base_prompt_id.")
        if item_id in seen_ids:
            raise StimulusRegistryError(f"Duplicate base_prompt_id: {item_id}.")
        seen_ids.add(item_id)
        if arm not in {"harmful", "control"}:
            raise StimulusRegistryError(f"{item_id} has invalid arm.")
        harmful_count += int(arm == "harmful")
        control_count += int(arm == "control")
        if not isinstance(prompt, str) or not prompt.strip():
            raise StimulusRegistryError(f"{item_id} lacks prompt_english.")
        expected_normalized_hash = hashlib.sha256(
            normalized_prompt_key(prompt).encode("utf-8")
        ).hexdigest()
        if item.get("normalized_prompt_sha256") != expected_normalized_hash:
            raise StimulusRegistryError(f"{item_id} normalized prompt hash mismatch.")
        pair_id = item.get("pair_id")
        if pair_id is not None:
            if not isinstance(pair_id, str) or not pair_id:
                raise StimulusRegistryError(f"{item_id} has invalid pair_id.")
            pair_arms.setdefault(pair_id, set()).add(str(arm))

        localized = item.get("localized_stimuli")
        if not isinstance(localized, Mapping):
            raise StimulusRegistryError(f"{item_id} lacks localized_stimuli.")
        english = localized.get("english")
        if not isinstance(english, Mapping) or english.get("text") != prompt:
            raise StimulusRegistryError(f"{item_id} English stimulus mismatch.")
        if english.get("sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
            raise StimulusRegistryError(f"{item_id} English stimulus hash mismatch.")

        if not require_production_approval:
            continue
        cluster_id = item.get("semantic_cluster_id")
        cluster_review = item.get("semantic_cluster_review")
        if not isinstance(cluster_id, str) or not cluster_id.strip():
            raise StimulusRegistryError(
                f"{item_id} lacks a human-approved semantic_cluster_id."
            )
        if not isinstance(cluster_review, Mapping) or cluster_review.get("status") != "approved":
            raise StimulusRegistryError(
                f"{item_id} semantic cluster review is not approved."
            )
        cluster_reviewers = cluster_review.get("reviewer_ids")
        if not _valid_reviewer_ids(cluster_reviewers):
            raise StimulusRegistryError(
                f"{item_id} semantic cluster review lacks reviewer provenance."
            )
        for language in TARGET_LANGUAGES:
            rendered = localized.get(language)
            if not isinstance(rendered, Mapping):
                raise StimulusRegistryError(
                    f"{item_id} lacks exact {language} stimulus."
                )
            text = rendered.get("text")
            if not isinstance(text, str) or not text.strip():
                raise StimulusRegistryError(
                    f"{item_id} has empty {language} stimulus."
                )
            if rendered.get("sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
                raise StimulusRegistryError(
                    f"{item_id} {language} stimulus hash mismatch."
                )
            reviewers = rendered.get("reviewer_ids")
            if rendered.get("review_status") != "approved":
                raise StimulusRegistryError(
                    f"{item_id} {language} stimulus is not approved."
                )
            if not _valid_reviewer_ids(reviewers):
                raise StimulusRegistryError(
                    f"{item_id} {language} stimulus lacks reviewer provenance."
                )

    counts = registry.get("counts")
    if not isinstance(counts, Mapping):
        raise StimulusRegistryError("Part 0 registry lacks counts.")
    if counts.get("harmful_unique_prompts") != harmful_count:
        raise StimulusRegistryError("Harmful stimulus count does not match metadata.")
    if counts.get("paired_control_prompts") != control_count:
        raise StimulusRegistryError("Control stimulus count does not match metadata.")
    incomplete_pairs = {
        pair_id: sorted(arms)
        for pair_id, arms in pair_arms.items()
        if arms != {"harmful", "control"}
    }
    if incomplete_pairs:
        raise StimulusRegistryError(
            f"JBB pairs are incomplete or ambiguous: {incomplete_pairs}."
        )


def _valid_reviewer_ids(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the hash-pinned Part 0 harmful/control registry."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    registry = build_stimulus_registry(fetch_pinned_sources(args.timeout_seconds))
    _atomic_write(args.output, registry)
    print(
        "Built Part 0 registry with "
        f"{registry['counts']['harmful_unique_prompts']} unique harmful prompts and "
        f"{registry['counts']['paired_control_prompts']} paired controls."
    )
    print(f"Registry: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
