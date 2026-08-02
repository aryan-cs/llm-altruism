from __future__ import annotations

import argparse
import json
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
        "files": [path.as_posix() for path in files],
    }

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _writestr(
            zf,
            MANIFEST_NAME,
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
        for rel_path in files:
            _writestr(zf, rel_path.as_posix(), (project_root / rel_path).read_bytes())

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
