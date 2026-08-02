from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "conference_submission" / "supplement.zip"
MANIFEST_NAME = "SUPPLEMENT_MANIFEST.json"

# The experiment repository supports a private OpenAI-compatible gateway, but an
# anonymous conference artifact must neither disclose the operator nor imply an
# author affiliation.  Apply the same deterministic substitutions to code,
# configuration, documentation, tests, and textual data inside the ZIP.  The
# source tree remains unchanged so the private production runner stays exact.
ANONYMOUS_TEXT_REPLACEMENTS = (
    ("https://inference-api.nvidia.com/v1", "https://YOUR_HOSTED_GATEWAY.example/v1"),
    ("https://inference.nvidia.com", "https://YOUR_HOSTED_GATEWAY.example"),
    ("inference-api.nvidia.com", "YOUR_HOSTED_GATEWAY.example"),
    ("inference.nvidia.com", "YOUR_HOSTED_GATEWAY.example"),
    ("INFERENCE_HUB_BASE_URL", "HOSTED_GATEWAY_BASE_URL"),
    ("NVIDIA_API_KEY", "HOSTED_GATEWAY_API_KEY"),
    ("inference_hub_models_api", "hosted_gateway_models_api"),
    ("InferenceHub", "HostedGateway"),
    ("INFERENCE_HUB", "HOSTED_GATEWAY"),
    ("inference_hub", "hosted_gateway"),
    ("inference-hub", "hosted-gateway"),
    ("Inference Hub", "Hosted Gateway"),
    ("internal NVIDIA", "private hosted"),
    ("Internal NVIDIA", "Private hosted"),
    ("aryan.cs.app@gmail.com", "anonymous@example.invalid"),
    ("aryan-cs", "anonymous-author"),
    ("Aryan Gupta", "Anonymous Author"),
    ("/Users/aryagupta", "/home/anonymous"),
    ("aryagupta", "anonymous"),
)

ANONYMITY_FORBIDDEN_MARKERS = (
    "inference-api.nvidia.com",
    "inference.nvidia.com",
    "nvidia_api_key",
    "inference_hub",
    "inferencehub",
    "aryan.cs.app@gmail.com",
    "aryan-cs",
    "aryan gupta",
    "/users/aryagupta",
    "aryagupta",
)

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
    Path("docs") / "release",
    Path("docs") / "conference_submission" / "README.md",
    Path("docs") / "conference_submission" / "conference_submission.tex",
    Path("docs") / "conference_submission" / "references.bib",
    Path("docs") / "conference_submission" / "neurips_2026.sty",
    Path("data") / "analysis",
    Path("data") / "graphs",
    Path("data") / "raw" / "part_1",
    Path("data") / "raw" / "part_2",
)

EXCLUDED_RELATIVE_PATHS = {
    Path("docs") / "release" / "research-proposal-metadata.json",
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
    if _is_relative_to(rel_path, Path("data") / "raw" / "part_0"):
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


def _anonymous_archive_path(rel_path: Path) -> str:
    return _anonymous_text(rel_path.as_posix())


def _anonymous_text(text: str) -> str:
    for source, replacement in ANONYMOUS_TEXT_REPLACEMENTS:
        text = text.replace(source, replacement)
    text = re.sub(r"inference[ _-]?hub", "hosted_gateway", text, flags=re.IGNORECASE)
    text = re.sub(r"nvidia_api_key", "HOSTED_GATEWAY_API_KEY", text, flags=re.IGNORECASE)
    text = re.sub(r"aryan gupta", "Anonymous Author", text, flags=re.IGNORECASE)
    return text


def _anonymous_archive_payload(payload: bytes) -> bytes:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload
    return _anonymous_text(text).encode("utf-8")


def audit_anonymous_archive(output_path: Path) -> list[str]:
    """Return entry/marker descriptions for anonymity leaks in a built ZIP."""

    findings: list[str] = []
    with zipfile.ZipFile(output_path) as zf:
        for info in zf.infolist():
            name_lower = info.filename.lower()
            for marker in ANONYMITY_FORBIDDEN_MARKERS:
                if marker in name_lower:
                    findings.append(f"path {info.filename!r} contains {marker!r}")
            payload = zf.read(info)
            try:
                text_lower = payload.decode("utf-8").lower()
            except UnicodeDecodeError:
                continue
            for marker in ANONYMITY_FORBIDDEN_MARKERS:
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "package": "anonymous NeurIPS supplement",
        "included_roots": [path.as_posix() for path in INCLUDE_PATHS],
        "policy_exclusions": list(POLICY_EXCLUSIONS),
        "file_count": len(files),
        "files": [_anonymous_archive_path(path) for path in files],
        "anonymization": (
            "private hosted-gateway identifiers, endpoints, credential-variable "
            "names, and author identifiers are deterministically replaced"
        ),
    }

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _writestr(
            zf,
            MANIFEST_NAME,
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
        for rel_path in files:
            _writestr(
                zf,
                _anonymous_archive_path(rel_path),
                _anonymous_archive_payload((project_root / rel_path).read_bytes()),
            )

    findings = audit_anonymous_archive(output_path)
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
