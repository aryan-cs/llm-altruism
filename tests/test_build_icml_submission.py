"""Tests for the ICML submission build helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_build_module():
    """Import the ICML build script for direct helper testing."""
    module_path = ROOT / "scripts" / "build_icml_submission.py"
    spec = importlib.util.spec_from_file_location("build_icml_submission_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selected_figures_tex_uses_two_column_readable_layout():
    """Selected figures should use large two-column panels, not a tiny three-across strip."""
    module = _load_build_module()

    tex = module.selected_figures_tex()

    assert tex.count(r"\includegraphics[width=0.48\textwidth]") >= 5
    assert "repaired_pd_replications" in tex


def test_write_main_tex_includes_anonymous_corresponding_author(tmp_path: Path):
    """The ICML builder should avoid AUTHORERR by emitting a corresponding-author line."""
    module = _load_build_module()
    original_build_dir = module.BUILD_DIR
    try:
        module.BUILD_DIR = tmp_path
        module.write_main_tex("Abstract body.")
        main_tex = (tmp_path / "main.tex").read_text(encoding="utf-8")
    finally:
        module.BUILD_DIR = original_build_dir

    assert r"\icmlcorrespondingauthor{Anonymous Author}{anon.email@domain.com}" in main_tex
    assert r"\bibliographystyle{icml2025}" in main_tex
    assert r"\bibliography{../../references}" in main_tex


def test_normalize_markdown_headings_strips_manual_numbering():
    """Markdown headings should not carry manual numbering into LaTeX sections."""
    module = _load_build_module()

    text = "## 1. Introduction\n### 3.2 Model cohort\n## A. Artifact Map\n"

    normalized = module.normalize_markdown_headings(text)

    assert "## Introduction" in normalized
    assert "### Model cohort" in normalized
    assert "## Artifact Map" in normalized
