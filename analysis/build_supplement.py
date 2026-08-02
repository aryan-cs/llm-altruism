from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import re
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "conference_submission" / "supplement.zip"
MANIFEST_NAME = "SUPPLEMENT_MANIFEST.json"

INCLUDE_PATHS = (
    Path("README.md"),
    Path("LICENSE"),
    Path(".env.example"),
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("agents"),
    Path("analysis"),
    Path("experiments"),
    Path("providers"),
    Path("tests"),
    Path("docs") / "JUDGE_AUDIT.md",
    Path("docs") / "CONFIRMATORY_CAMPAIGN.md",
    Path("docs") / "CONFIRMATORY_INTEGRITY.md",
    Path("docs") / "CONFIRMATORY_PROTOCOL.md",
    Path("docs") / "PART0_CONFIRMATORY_RUNNER.md",
    Path("docs") / "PART1_CONFIRMATORY_IMPLEMENTATION.md",
    Path("docs") / "release",
    Path("docs") / "conference_submission" / "README.md",
    Path("docs") / "conference_submission" / "conference_submission.tex",
    Path("docs") / "conference_submission" / "checklist.tex",
    Path("docs") / "conference_submission" / "figures" / "part2_restraint_rate_by_model.png",
    Path("docs") / "conference_submission" / "figures" / "part1_cooperation_by_game_heatmap.png",
    Path("docs") / "conference_submission" / "figures" / "part2_shared_reserve_over_time.png",
    Path("docs") / "conference_submission" / "figures" / "part2_population_over_time.png",
    Path("docs") / "conference_submission" / "figures" / "part2_restraint_choice_over_time.png",
    Path("docs") / "conference_submission" / "figures" / "frame_sensitivity_heatmap.png",
    Path("docs") / "conference_submission" / "figures" / "restraint_vs_final_population.png",
    Path("docs") / "conference_submission" / "figures" / "part2_agent_day_raster.png",
    Path("docs") / "conference_submission" / "references.bib",
    Path("docs") / "conference_submission" / "neurips_2026.sty",
    Path("data") / "analysis",
    Path("data") / "graphs",
    Path("data") / "raw" / "part_1",
    Path("data") / "raw" / "part_2",
)

EXCLUDED_RELATIVE_PATHS = {
    Path("docs") / "release" / "research-proposal-metadata.json",
    Path("data") / "analysis" / "tables" / "part0_model_summary.csv",
    Path("data") / "analysis" / "tables" / "part0_language_robustness.csv",
    Path("data") / "analysis" / "tables" / "cross_part_model_summary.csv",
    Path("data") / "analysis" / "tables" / "cross_part_correlations.csv",
    Path("data") / "graphs" / "paper_visuals" / "part0_refusal_rate_by_model.png",
    Path("data") / "graphs" / "paper_visuals" / "part0_refusal_by_language_heatmap.png",
    Path("data") / "graphs" / "paper_visuals" / "behavioral_fingerprint_heatmap.png",
    Path("data") / "graphs" / "paper_visuals" / "model_behavior_pca.png",
    Path("tests") / "test_campaign.py",
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
    Path("data") / "raw" / "part_0",
    Path("data") / "graphs" / "part_0",
    Path("data") / "graphs" / "cross_part",
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
        "path": "data/analysis/tables/{part0_*,cross_part_*}.csv; data/graphs/{part_0,cross_part}/; four Part 0-dependent paper_visuals plots",
        "reason": "legacy Part 0 labels are invalid; model-level rates and every dependent table and plot are withdrawn",
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
        "path": "tests/test_campaign.py",
        "reason": "legacy campaign tests require withheld raw Part 0 prompts; the fixed confirmatory campaign and its tests are included",
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


def _should_exclude(rel_path: Path, output_rel_path: Path | None = None) -> bool:
    if output_rel_path is not None and rel_path == output_rel_path:
        return True
    if any(_is_relative_to(rel_path, prefix) for prefix in EXCLUDED_RELATIVE_PREFIXES):
        return True
    if rel_path in EXCLUDED_RELATIVE_PATHS:
        return True
    if any(part in EXCLUDED_DIR_NAMES for part in rel_path.parts):
        return True
    if _suffix(rel_path) in EXCLUDED_SUFFIXES:
        return True
    return False


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
    for include_path in INCLUDE_PATHS:
        absolute_path = project_root / include_path
        if absolute_path.is_file():
            candidates = [absolute_path]
        elif absolute_path.is_dir():
            candidates = [path for path in absolute_path.rglob("*") if path.is_file()]
        else:
            continue

        for candidate in candidates:
            rel_path = candidate.relative_to(project_root)
            if not _should_exclude(rel_path, output_rel_path):
                files.add(rel_path)

    return sorted(files, key=lambda path: path.as_posix())


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


def _anonymous_archive_path(
    rel_path: Path, replacements: tuple[tuple[str, str], ...]
) -> str:
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
    markers = tuple(source.lower() for source, _ in _identity_replacements(project_root))
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
) -> tuple[Path, list[Path]]:
    project_root = project_root.resolve()
    output_path = output_path.resolve()
    files = collect_supplement_files(project_root=project_root, output_path=output_path)
    replacements = _identity_replacements(project_root)
    archive_payloads = {
        _anonymous_archive_path(rel_path, replacements): _anonymous_archive_payload(
            (project_root / rel_path).read_bytes(), replacements
        )
        for rel_path in files
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "package": "anonymous NeurIPS supplement",
        "included_roots": [path.as_posix() for path in INCLUDE_PATHS],
        "policy_exclusions": list(POLICY_EXCLUSIONS),
        "file_count": len(files),
        "files": list(archive_payloads),
        "file_sha256s": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in archive_payloads.items()
        },
        "anonymization": (
            "author identifiers are deterministically replaced while executable "
            "protocol/API identifiers are preserved"
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
    args = parser.parse_args()

    output_path, files = build_supplement(
        project_root=Path(args.project_root),
        output_path=Path(args.output),
    )
    size_mib = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} with {len(files)} files ({size_mib:.1f} MiB)")


if __name__ == "__main__":
    main()
