from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

from analysis.build_provider_safe_v2_croissant_metadata import (
    DefinitiveCroissantError,
    _scan_public_file,
    _self_hash,
    _sha256_file,
    build_metadata as build_definitive_croissant_metadata,
    serialized_metadata as serialize_definitive_croissant_metadata,
    validate_release_sources,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "conference_submission" / "supplement.zip"
MANIFEST_NAME = "SUPPLEMENT_MANIFEST.json"
ANONYMIZATION_POLICY_NAME = ".supplement-anonymization.json"
REPRODUCIBLE_CREATED_UTC = "2026-01-01T00:00:00+00:00"

# Hosted-panel code is admitted by exact path, never by a substring or broad
# data-directory rule.  This makes additions review-visible and prevents a new
# private collector, credential helper, or raw-output utility from silently
# entering the anonymous archive merely because it lives beside these files.
HOSTED_REPRODUCIBILITY_ALLOWLIST = frozenset(
    {
        Path("analysis") / "analyze_availability_retry_panels.py",
        Path("analysis") / "analyze_inference_hub_part1_panel.py",
        Path("analysis") / "analyze_provider_safe_v2_definitive.py",
        Path("analysis") / "analyze_semantic_invalid_repairs.py",
        Path("analysis") / "accelerated_part0_human_validation.py",
        Path("analysis") / "build_provider_safe_v2_paper_assets.py",
        Path("analysis") / "build_provider_safe_v2_croissant_metadata.py",
        Path("analysis") / "build_developer_descriptives.py",
        Path("analysis") / "build_final_results.py",
        Path("analysis") / "build_original_view_figures.py",
        Path("analysis") / "build_paper_headlines.py",
        Path("analysis") / "build_paper_visuals.py",
        Path("analysis") / "finalize_inference_hub_part2_offline.py",
        Path("analysis") / "merge_sota_compatibility_with_judge.py",
        Path("analysis") / "part2_confirmatory.py",
        Path("analysis") / "reconcile_inference_hub_routes.py",
        Path("docs") / "AVAILABILITY_RETRY_ANALYSIS.md",
        Path("experiments") / "sota_cross_axis_panel.json",
        Path("experiments") / "part1" / "role_calibration_panel_v1.json",
        Path("experiments") / "part2" / "part2_sensitivity_v1.json",
        Path("experiments") / "part2" / "part2_sensitivity_deadline_exploratory_v1.json",
        Path("experiments") / "misc" / "inference_hub_compatibility.py",
        Path("experiments") / "misc" / "inference_hub_compatibility_provider_safe.py",
        Path("experiments") / "misc" / "inference_hub_discovery.py",
        Path("experiments") / "misc" / "inference_hub_exploratory_accelerated.py",
        Path("experiments") / "misc" / "inference_hub_main_accelerated.py",
        Path("experiments") / "misc" / "inference_hub_part0_deadline_retry.py",
        Path("experiments") / "misc" / "inference_hub_part0_panel.py",
        Path("experiments") / "misc" / "inference_hub_part1_deadline_accelerated.py",
        Path("experiments") / "misc" / "inference_hub_part1_panel.py",
        Path("experiments") / "misc" / "inference_hub_part1_semantic_invalid_repair.py",
        Path("experiments") / "misc" / "inference_hub_part1_role_calibration_v1.py",
        Path("experiments") / "misc" / "inference_hub_part1_role_semantic_invalid_repair.py",
        Path("experiments") / "misc" / "inference_hub_part1_stratified_panel.py",
        Path("experiments") / "misc" / "inference_hub_part2_panel.py",
        Path("experiments") / "misc" / "inference_hub_part2_sensitivity_v1.py",
        Path("experiments") / "misc" / "inference_hub_provider_safe.py",
        Path("experiments") / "misc" / "inference_hub_provider_safe_v2.py",
        Path("experiments") / "misc" / "inference_hub_rate_limit.py",
        Path("experiments") / "misc" / "inference_hub_sensitivity_deadline_accelerated.py",
        Path("experiments") / "misc" / "inference_hub_retire_target.py",
        Path("experiments") / "misc" / "inference_hub_visible_compatibility.py",
        Path("tests") / "test_accelerated_part0_human_validation.py",
        Path("tests") / "test_analyze_availability_retry_panels.py",
        Path("tests") / "test_analyze_provider_safe_v2_definitive.py",
        Path("tests") / "test_analyze_semantic_invalid_repairs.py",
        Path("tests") / "test_build_provider_safe_v2_paper_assets.py",
        Path("tests") / "test_build_provider_safe_v2_croissant_metadata.py",
        Path("tests") / "test_inference_hub_compatibility.py",
        Path("tests") / "test_inference_hub_compatibility_provider_safe.py",
        Path("tests") / "test_inference_hub_discovery.py",
        Path("tests") / "test_inference_hub_exploratory_accelerated.py",
        Path("tests") / "test_inference_hub_main_accelerated.py",
        Path("tests") / "test_inference_hub_part0_deadline_retry.py",
        Path("tests") / "test_inference_hub_part0_panel.py",
        Path("tests") / "test_inference_hub_part1_deadline_accelerated.py",
        Path("tests") / "test_inference_hub_part1_panel.py",
        Path("tests") / "test_inference_hub_part1_semantic_invalid_repair.py",
        Path("tests") / "test_inference_hub_part1_role_calibration_v1.py",
        Path("tests") / "test_inference_hub_part1_role_semantic_invalid_repair.py",
        Path("tests") / "test_inference_hub_part1_stratified_panel.py",
        Path("tests") / "test_inference_hub_part2_panel.py",
        Path("tests") / "test_inference_hub_part2_sensitivity_v1.py",
        Path("tests") / "test_inference_hub_provider_safe.py",
        Path("tests") / "test_inference_hub_provider_safe_v2.py",
        Path("tests") / "test_inference_hub_rate_limit.py",
        Path("tests") / "test_inference_hub_sensitivity_deadline_accelerated.py",
        Path("tests") / "test_inference_hub_retire_target.py",
        Path("tests") / "test_inference_hub_visible_compatibility.py",
        Path("tests") / "test_build_developer_descriptives.py",
        Path("tests") / "test_build_final_results.py",
        Path("tests") / "test_build_paper_headlines.py",
        Path("tests") / "test_build_paper_visuals.py",
        Path("tests") / "test_finalize_inference_hub_part2_offline.py",
        Path("tests") / "test_reconcile_inference_hub_routes.py",
        Path("tests") / "test_merge_sota_compatibility_with_judge.py",
        Path("tests") / "test_part2_confirmatory_cli.py",
        Path("tests") / "test_part2_confirmatory_statistics.py",
        Path("tests") / "test_sota_cross_axis_panel.py",
    }
)

INCLUDE_PATHS = (
    Path("docs") / "conference_submission" / "SUPPLEMENT_README.md",
    Path("docs") / "conference_submission" / "SUPPLEMENT_MODEL_REGISTRY.md",
    Path("LICENSE"),
    Path("CHECKPOINT.md"),
    Path(".env.example"),
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("agents"),
    Path("analysis"),
    Path("experiments"),
    Path("providers"),
    Path("tests"),
    Path("docs") / "JUDGE_AUDIT.md",
    Path("docs") / "CONFIRMATORY_PROTOCOL.md",
    Path("docs") / "ACCELERATED_PART0_HUMAN_VALIDATION.md",
    Path("docs") / "AVAILABILITY_RETRY_ANALYSIS.md",
    Path("docs") / "PART1_ROLE_CALIBRATION_V1.md",
    Path("docs") / "PART1_SEMANTIC_INVALID_REPAIR.md",
    Path("docs") / "PART1_ROLE_SEMANTIC_INVALID_REPAIR.md",
    Path("docs") / "PART2_SENSITIVITY_V1.md",
    Path("docs") / "PROVIDER_SAFE_V2_DEFINITIVE_ANALYSIS.md",
    Path("docs") / "PROVIDER_SAFE_V2_PAPER_ASSETS.md",
    Path("docs") / "LOCAL_MODEL_CONTROLS.md",
    Path("docs") / "release",
    Path("docs") / "conference_submission" / "README.md",
    Path("docs") / "conference_submission" / "conference_submission.tex",
    Path("docs") / "conference_submission" / "checklist.tex",
    Path("docs") / "conference_submission" / "references.bib",
    Path("docs") / "conference_submission" / "neurips_2026.sty",
    Path("docs") / "conference_submission" / "figures" / "part0_refusal_rate_by_model.pdf",
    Path("docs") / "conference_submission" / "figures" / "part0_refusal_rate_by_model.png",
    Path("docs") / "conference_submission" / "figures" / "part2_restraint_rate_by_model.pdf",
    Path("docs") / "conference_submission" / "figures" / "part2_restraint_rate_by_model.png",
    Path("docs") / "conference_submission" / "figures" / "part2_shared_reserve_over_time.pdf",
    Path("docs") / "conference_submission" / "figures" / "part2_shared_reserve_over_time.png",
    Path("docs") / "conference_submission" / "figures" / "part2_population_over_time.pdf",
    Path("docs") / "conference_submission" / "figures" / "part2_population_over_time.png",
    Path("docs") / "conference_submission" / "figures" / "part2_all_models.png",
    Path("data") / "analysis" / "local_hf_part1_controls.json",
    Path("data") / "graphs" / "part_0_graphs.py",
    Path("data") / "graphs" / "part_1_graphs.py",
    Path("data") / "graphs" / "part_2_graphs.py",
    Path("data") / "graphs" / "cross_part_graphs.py",
    Path("data") / "graphs" / "paper_visuals.py",
)

# These roots are generated only after every live definitive campaign has
# sealed.  Development builds may omit all of them; final builds use
# ``--require-definitive-artifacts`` and fail if any required root is absent.
DEFINITIVE_RELEASE_PATHS = (
    Path("data") / "processed" / "provider-safe-v2-definitive-analysis",
    Path("data") / "processed" / "provider-safe-v2-paper-assets",
    Path("data") / "processed" / "provider-safe-v2-croissant-metadata.json",
)
ISOLATED_SUPPLEMENTAL_RELEASE_PATHS = (
    Path("artifacts") / "availability_retry_analysis_definitive_v1",
    Path("artifacts") / "semantic_invalid_repair_analysis_definitive_v1",
)
OPTIONAL_INCLUDE_PATHS = (
    *DEFINITIVE_RELEASE_PATHS,
    *ISOLATED_SUPPLEMENTAL_RELEASE_PATHS,
)

EXCLUDED_RELATIVE_PATHS = {
    Path("docs") / "release" / "research-proposal-metadata.json",
    Path("docs") / "release" / "MODEL_REGISTRY.md",
    Path("data") / "analysis" / "tables" / "part0_model_summary.csv",
    Path("data") / "analysis" / "tables" / "part0_language_robustness.csv",
    Path("data") / "analysis" / "tables" / "cross_part_model_summary.csv",
    Path("data") / "analysis" / "tables" / "cross_part_correlations.csv",
    Path("data") / "graphs" / "paper_visuals" / "part0_refusal_rate_by_model.png",
    Path("data") / "graphs" / "paper_visuals" / "part0_refusal_by_language_heatmap.png",
    Path("data") / "graphs" / "paper_visuals" / "behavioral_fingerprint_heatmap.png",
    Path("data") / "graphs" / "paper_visuals" / "model_behavior_pca.png",
    Path("data") / "raw" / "part_2" / "legacy_structural_provenance.json",
    Path("analysis") / "build_legacy_part2_provenance.py",
    Path("tests") / "test_legacy_part2_provenance.py",
    Path("tests") / "test_campaign.py",
    Path("tests") / "test_part0_audit_checkpoint.py",
    # Hosted aggregate analyzers are private until their outputs have passed the
    # completed-run sanitization gate below. Raw local-scale runners stay
    # outside the hosted surface; the exact sanitized local-control aggregate
    # used by the paper-assets generator is included explicitly above.
    Path("analysis") / "analyze_joint_inference_hub_part1_panels.py",
    Path("analysis") / "analyze_local_hf_part1_panel.py",
    Path("analysis") / "build_sota_inference_hub_roster.py",
    Path("analysis") / "build_sota_probe_registry.py",
    Path("experiments") / "misc" / "local_hf_part1_panel.py",
    Path("experiments") / "misc" / "local_hf_smoke.py",
    Path("tests") / "test_analyze_joint_inference_hub_part1_panels.py",
    Path("tests") / "test_analyze_local_hf_part1_panel.py",
    Path("tests") / "test_build_sota_inference_hub_roster.py",
    Path("tests") / "test_build_sota_probe_registry.py",
    Path("tests") / "test_local_hf_part1_panel.py",
    Path("tests") / "test_local_hf_smoke.py",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv",
    ".venv",
    "__pycache__",
    "anonymous_supplement",
    "build",
    "dist",
    "htmlcov",
    "venv",
}

EXCLUDED_RELATIVE_PREFIXES = (
    Path("data") / "private",
    Path("data") / "raw" / "part_0",
    Path("data") / "raw" / "part_1",
    Path("data") / "raw" / "part_2",
    Path("data") / "analysis" / "tables",
    Path("data") / "analysis" / "validation",
    Path("data") / "graphs" / "part_0",
    Path("data") / "graphs" / "part_1",
    Path("data") / "graphs" / "part_2",
    Path("data") / "graphs" / "cross_part",
    Path("data") / "graphs" / "paper_visuals",
)
EXCLUDED_PRIVATE_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".supplement-anonymization.json",
}
EXCLUDED_SECRET_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
EXCLUDED_SECRET_NAME_MARKERS = (
    "api_key",
    "apikey",
    "credential",
    "private_key",
    "secret",
)
EXCLUDED_DATA_NAME_MARKERS = (
    "attempt_ledger",
    "incomplete",
    "interrupted",
    "journal",
    "pending",
    "raw_response",
)
EXCLUDED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pdf",
    ".pyc",
    ".pyo",
    ".so",
    ".synctex.gz",
    ".zip",
}
POLICY_EXCLUSIONS = (
    {
        "path": "data/raw/part_0/",
        "reason": "raw harmful requests, source prompt CSVs, and model completions are withheld from the default anonymous supplement",
    },
    {
        "path": "data/raw/{part_1,part_2}/",
        "reason": "legacy prompt text, model justifications, reasoning traces, and superseded Part 2 execution rows are excluded; the package exposes sealed text-free aggregates instead",
    },
    {
        "path": "data/analysis/{tables,validation}/; data/graphs/{part_0,part_1,part_2,cross_part,paper_visuals}/; superseded conference figure files",
        "reason": "legacy and superseded derived outputs are excluded so they cannot be mistaken for the sealed current results or current paper figures",
    },
    {
        "path": "docs/conference_submission/conference_submission.pdf",
        "reason": "submission PDF is uploaded separately from the supplement",
    },
    {
        "path": "docs/conference_submission/supplement.zip",
        "reason": "the supplement archive is not nested inside itself",
    },
    {
        "path": "docs/release/research-proposal-metadata.json",
        "reason": "author-identifying proposal metadata is excluded from the anonymous supplement",
    },
    {
        "path": "docs/release/MODEL_REGISTRY.md",
        "reason": "the development registry includes post-pilot route planning; the supplement substitutes a pilot-only anonymous registry at the same archive path",
    },
    {
        "path": "tests/test_campaign.py",
        "reason": "legacy campaign tests require withheld raw Part 0 prompts and are not needed to replay the released pilot artifacts",
    },
    {
        "path": "tests/test_part0_audit_checkpoint.py",
        "reason": "this forensic test requires a deliberately withheld legacy Part 0 rejudgment checkpoint and is outside the current aggregate-only artifact",
    },
    {
        "path": "data/private/**",
        "reason": "credentials, prompts, raw responses, journals, run-local aggregates, and incomplete artifacts are private; only the explicit self-hashed provider-safe-v2 aggregate release enters the supplement",
    },
    {
        "path": "deprecated Part 1/Part 2 raw evidence, Part 2 execution archive, and structural provenance",
        "reason": "superseded evidence and model-generated text are excluded so they cannot be mistaken for the corrected matched-panel implementation or cross the aggregate-only release boundary",
    },
    {
        "path": "non-allowlisted hosted utilities and local-HF scale controls",
        "reason": "the supplement exposes an exact reviewed hosted reproducibility surface and omits unrelated exploratory tooling",
    },
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".synctex.gz"):
        return ".synctex.gz"
    return path.suffix


def _is_hosted_reproducibility_path(path: Path) -> bool:
    name = path.name.casefold()
    return (
        "inference_hub" in name
        or "availability_retry" in name
        or name in {
            "reconcile_inference_hub_routes.py",
            "test_reconcile_inference_hub_routes.py",
            "sota_cross_axis_panel.json",
            "test_sota_cross_axis_panel.py",
        }
    )


def _is_sensitive_data_artifact(path: Path) -> bool:
    if not path.parts or path.parts[0] != "data":
        return False
    lowered = path.name.casefold()
    return any(marker in lowered for marker in EXCLUDED_DATA_NAME_MARKERS)


def _should_exclude(rel_path: Path, output_rel_path: Path | None = None) -> bool:
    if output_rel_path is not None and rel_path == output_rel_path:
        return True
    lowered_name = rel_path.name.casefold()
    if rel_path.name in EXCLUDED_PRIVATE_FILE_NAMES or (
        lowered_name.startswith(".env") and lowered_name != ".env.example"
    ):
        return True
    if any(marker in lowered_name for marker in EXCLUDED_SECRET_NAME_MARKERS):
        return True
    if rel_path.suffix.casefold() in EXCLUDED_SECRET_SUFFIXES:
        return True
    if any(_is_relative_to(rel_path, prefix) for prefix in EXCLUDED_RELATIVE_PREFIXES):
        return True
    if rel_path in EXCLUDED_RELATIVE_PATHS:
        return True
    if _is_hosted_reproducibility_path(rel_path) and rel_path not in HOSTED_REPRODUCIBILITY_ALLOWLIST:
        return True
    if _is_sensitive_data_artifact(rel_path):
        return True
    if any(part in EXCLUDED_DIR_NAMES for part in rel_path.parts):
        return True
    if _suffix(rel_path) in EXCLUDED_SUFFIXES:
        return True
    return False


def resolve_include_path(project_root: Path, include_path: Path) -> Path | None:
    """Resolve either the development source path or its archive-layout alias."""

    candidate = project_root / include_path
    if candidate.exists():
        return candidate
    archive_fallbacks = {
        Path("docs") / "conference_submission" / "SUPPLEMENT_README.md": Path("README.md"),
        Path("docs") / "conference_submission" / "SUPPLEMENT_MODEL_REGISTRY.md": Path("docs") / "release" / "MODEL_REGISTRY.md",
    }
    fallback = archive_fallbacks.get(include_path)
    fallback_path = project_root / fallback if fallback is not None else None
    return fallback_path if fallback_path is not None and fallback_path.exists() else None


def collect_supplement_files(
    project_root: Path = PROJECT_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
) -> list[Path]:
    project_root = project_root.resolve()
    output_rel_path: Path | None = None
    try:
        output_rel_path = output_path.resolve().relative_to(project_root)
    except ValueError:
        output_rel_path = None

    files: set[Path] = set()
    for include_path in (*INCLUDE_PATHS, *OPTIONAL_INCLUDE_PATHS):
        absolute_path = resolve_include_path(project_root, include_path)
        if absolute_path is None:
            continue
        if absolute_path.is_file():
            candidates = [absolute_path]
        elif absolute_path.is_dir():
            candidates = [path for path in absolute_path.rglob("*") if path.is_file()]
        else:
            continue

        for candidate in candidates:
            rel_path = candidate.relative_to(project_root)
            archive_registry_fallback = (
                include_path
                == Path("docs") / "conference_submission" / "SUPPLEMENT_MODEL_REGISTRY.md"
                and rel_path == Path("docs") / "release" / "MODEL_REGISTRY.md"
            )
            if archive_registry_fallback or not _should_exclude(
                rel_path, output_rel_path
            ):
                files.add(rel_path)

    return sorted(files, key=lambda path: path.as_posix())


def _validate_isolated_supplemental_directory(
    directory: Path, expected_artifact_type: str
) -> None:
    """Validate one separately reported retry/repair aggregate directory."""

    manifest_path = directory / "analysis_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Supplemental release lacks analysis_manifest.json: {directory.name}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Supplemental release manifest is unreadable: {directory.name}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != expected_artifact_type
        or manifest.get("evidence_sha256") != _self_hash(manifest)
    ):
        raise ValueError(f"Supplemental release manifest contract failed: {directory.name}")
    if expected_artifact_type == "inference_hub_availability_retry_analysis_v1":
        if (
            manifest.get("exploratory_only") is not True
            or manifest.get("replaces_primary") is not False
            or manifest.get("merge_with_primary_permitted") is not False
            or manifest.get("cross_axis_permitted") is not False
        ):
            raise ValueError("Availability-retry release changed its isolation contract")
    elif (
        manifest.get("primary_records_mutated") is not False
        or manifest.get("primary_denominators_changed") is not False
        or manifest.get("repaired_estimates_separate_only") is not True
        or manifest.get("promotion_permitted") is not False
    ):
        raise ValueError("Semantic-repair release changed its isolation contract")

    published = manifest.get("published_outputs")
    if not isinstance(published, dict) or not published:
        raise ValueError(f"Supplemental release output inventory is absent: {directory.name}")
    expected: dict[str, str] = {}
    for group in published.values():
        if not isinstance(group, dict):
            raise ValueError(f"Supplemental release output inventory is malformed: {directory.name}")
        for binding in group.values():
            if not isinstance(binding, dict) or "filename" not in binding:
                continue
            filename = binding.get("filename")
            digest = binding.get("file_sha256")
            if (
                not isinstance(filename, str)
                or filename != Path(filename).name
                or "/" in filename
                or "\\" in filename
                or not isinstance(digest, str)
                or len(digest) != 64
                or filename in expected
            ):
                raise ValueError(f"Supplemental release output binding is unsafe: {directory.name}")
            expected[filename] = digest
    actual = {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.name != manifest_path.name
    }
    if set(actual) != set(expected):
        raise ValueError(f"Supplemental release differs from its output inventory: {directory.name}")
    for name, path in actual.items():
        if _sha256_file(path) != expected[name]:
            raise ValueError(f"Supplemental release output hash changed: {name}")
        _scan_public_file(path)
    _scan_public_file(manifest_path)


def validate_definitive_release(
    project_root: Path, *, require_definitive_artifacts: bool = False
) -> str:
    """Fail closed on partial, stale, tampered, or privacy-unsafe release roots."""

    project_root = project_root.resolve()
    required = [project_root / path for path in DEFINITIVE_RELEASE_PATHS]
    present = [path.exists() for path in required]
    if not any(present):
        if require_definitive_artifacts:
            raise ValueError("Definitive aggregate release is not present")
        status = "not_yet_generated"
    elif not all(present):
        missing = [path.name for path, exists in zip(required, present) if not exists]
        raise ValueError(
            "Definitive aggregate release is partial; missing: " + ", ".join(missing)
        )
    else:
        analysis_dir, assets_dir, metadata_path = required
        try:
            validate_release_sources(analysis_dir, assets_dir)
            expected = serialize_definitive_croissant_metadata(
                build_definitive_croissant_metadata(
                    analysis_dir=analysis_dir,
                    paper_assets_dir=assets_dir,
                    output_path=metadata_path,
                )
            )
        except DefinitiveCroissantError as error:
            raise ValueError(str(error)) from error
        if not metadata_path.is_file() or metadata_path.read_text(encoding="utf-8") != expected:
            raise ValueError("Definitive Croissant metadata is absent or stale")
        status = "complete_hash_and_privacy_validated"

    supplemental_types = {
        "availability_retry_analysis_definitive_v1": (
            "inference_hub_availability_retry_analysis_v1"
        ),
        "semantic_invalid_repair_analysis_definitive_v1": (
            "inference_hub_semantic_invalid_repair_analysis_v1"
        ),
    }
    for relative in ISOLATED_SUPPLEMENTAL_RELEASE_PATHS:
        directory = project_root / relative
        if directory.exists():
            if not directory.is_dir():
                raise ValueError(f"Supplemental release root is not a directory: {relative}")
            _validate_isolated_supplemental_directory(
                directory, supplemental_types[directory.name]
            )
    return status


def _writestr(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    info = zipfile.ZipInfo(arcname)
    info.date_time = (2026, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(info, data)


def _git_value(project_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=project_root, check=True,
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _identity_replacements(project_root: Path) -> tuple[tuple[str, str], ...]:
    """Derive private identifiers locally without shipping them in source code."""

    candidates: dict[str, str] = {}
    home = str(Path.home())
    login = getpass.getuser().strip()
    name = _git_value(project_root, "config", "user.name")
    email = _git_value(project_root, "config", "user.email")
    remote = _git_value(project_root, "remote", "get-url", "origin")
    if home and home not in {"/", "/home/anonymous"}:
        candidates[home] = "/home/anonymous"
    if login and login.lower() not in {"root", "anonymous"}:
        candidates[login] = "anonymous"
    if name and name.lower() != "anonymous author":
        candidates[name] = "Anonymous Author"
    if email and email.lower() != "anonymous@example.invalid":
        candidates[email] = "anonymous@example.invalid"
    if remote:
        owner_match = re.search(r"(?:github\.com|gitlab\.com)[:/]([^/]+)/", remote)
        if owner_match and owner_match.group(1).lower() not in {"anonymous", "anonymous-author"}:
            candidates[owner_match.group(1)] = "anonymous-author"
    return tuple(sorted(candidates.items(), key=lambda pair: (-len(pair[0]), pair[0])))


def _affiliation_replacements(project_root: Path) -> tuple[tuple[str, str], ...]:
    """Load private deployment literals from a policy excluded from the ZIP."""

    policy_path = project_root / ANONYMIZATION_POLICY_NAME
    if not policy_path.is_file():
        return ()
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    raw_replacements = payload.get("replacements") if isinstance(payload, dict) else None
    if not isinstance(raw_replacements, list) or not raw_replacements:
        raise ValueError(f"{ANONYMIZATION_POLICY_NAME} must define replacements[].")
    replacements: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_replacements):
        if not isinstance(item, dict):
            raise ValueError(f"replacements[{index}] must be an object.")
        source = item.get("source")
        replacement = item.get("replacement")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"replacements[{index}].source must be non-empty.")
        if not isinstance(replacement, str) or not replacement.strip():
            raise ValueError(f"replacements[{index}].replacement must be non-empty.")
        if source.casefold() == replacement.casefold():
            raise ValueError(f"replacements[{index}] must change its source.")
        if source.casefold() in seen:
            raise ValueError(f"replacements[{index}].source is duplicated.")
        seen.add(source.casefold())
        replacements.append((source, replacement))
    return tuple(sorted(replacements, key=lambda pair: (-len(pair[0]), pair[0])))


def _archive_replacements(project_root: Path) -> tuple[tuple[str, str], ...]:
    replacements = (
        *_identity_replacements(project_root),
        *_affiliation_replacements(project_root),
    )
    return tuple(sorted(replacements, key=lambda pair: (-len(pair[0]), pair[0])))


def _anonymous_archive_path(
    rel_path: Path, replacements: tuple[tuple[str, str], ...]
) -> str:
    if rel_path == Path("docs") / "conference_submission" / "SUPPLEMENT_README.md":
        return "README.md"
    if rel_path == Path("docs") / "conference_submission" / "SUPPLEMENT_MODEL_REGISTRY.md":
        return "docs/release/MODEL_REGISTRY.md"
    return _anonymous_text(rel_path.as_posix(), replacements)


def _anonymous_text(text: str, replacements: tuple[tuple[str, str], ...]) -> str:
    for source, replacement in replacements:
        text = text.replace(source, replacement)
        text = re.sub(re.escape(source), replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"/Users/[A-Za-z0-9._-]+", "/home/anonymous", text)
    return text


def _anonymous_archive_payload(
    payload: bytes, replacements: tuple[tuple[str, str], ...]
) -> bytes:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload
    return _anonymous_text(text, replacements).encode("utf-8")


def audit_anonymous_archive(
    output_path: Path, project_root: Path = PROJECT_ROOT
) -> list[str]:
    """Return entry/marker descriptions for anonymity leaks in a built ZIP."""

    findings: list[str] = []
    markers = tuple(source.lower() for source, _ in _archive_replacements(project_root))
    with zipfile.ZipFile(output_path) as zf:
        for info in zf.infolist():
            name_lower = info.filename.lower()
            for marker in markers:
                if marker in name_lower:
                    findings.append(f"path {info.filename!r} contains {marker!r}")
            payload = zf.read(info)
            try:
                text_lower = payload.decode("utf-8").lower()
            except UnicodeDecodeError:
                continue
            for marker in markers:
                if marker in text_lower:
                    findings.append(f"entry {info.filename!r} contains {marker!r}")
    return findings


def build_supplement(
    project_root: Path = PROJECT_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    require_definitive_artifacts: bool = False,
) -> tuple[Path, list[Path]]:
    project_root = project_root.resolve()
    output_path = output_path.resolve()
    definitive_release_status = validate_definitive_release(
        project_root,
        require_definitive_artifacts=require_definitive_artifacts,
    )
    files = collect_supplement_files(project_root=project_root, output_path=output_path)
    replacements = _archive_replacements(project_root)
    archive_payloads = dict(
        sorted(
            (
                _anonymous_archive_path(rel_path, replacements),
                _anonymous_archive_payload(
                    (project_root / rel_path).read_bytes(), replacements
                ),
            )
            for rel_path in files
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    manifest = {
        # A wall-clock build time makes two builds of identical source produce
        # different manifests and ZIP hashes.  Use the same epoch as the fixed
        # per-entry ZIP metadata so the anonymous archive is byte-reproducible.
        "created_utc": REPRODUCIBLE_CREATED_UTC,
        "package": "anonymous NeurIPS supplement",
        "included_roots": [
            path.as_posix() for path in (*INCLUDE_PATHS, *OPTIONAL_INCLUDE_PATHS)
        ],
        "definitive_release_status": definitive_release_status,
        "policy_exclusions": list(POLICY_EXCLUSIONS),
        "file_count": len(files),
        "files": list(archive_payloads),
        "file_sha256s": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in archive_payloads.items()
        },
        "anonymization": (
            "author identifiers and affiliation-revealing private gateway literals "
            "are deterministically replaced or already anonymous on clean rebuild; "
            "public scientific model/vendor metadata is preserved"
        ),
    }

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _writestr(
            zf,
            MANIFEST_NAME,
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
        for name, payload in archive_payloads.items():
            _writestr(zf, name, payload)

    findings = audit_anonymous_archive(output_path, project_root)
    if findings:
        output_path.unlink()
        raise ValueError("Anonymous supplement audit failed: " + "; ".join(findings))

    return output_path, files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the anonymous conference supplement ZIP.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--require-definitive-artifacts", action="store_true")
    args = parser.parse_args()

    output_path, files = build_supplement(
        project_root=Path(args.project_root),
        output_path=Path(args.output),
        require_definitive_artifacts=args.require_definitive_artifacts,
    )
    size_mib = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} with {len(files)} files ({size_mib:.1f} MiB)")


if __name__ == "__main__":
    main()
