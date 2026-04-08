"""Tests for live ecology packet refresh orchestration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _load_refresh_module():
    module_path = ROOT / "scripts" / "refresh_live_ecology_packet.py"
    spec = importlib.util.spec_from_file_location("refresh_live_ecology_packet_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_log_path_picks_newest_jsonl(tmp_path: Path):
    module = _load_refresh_module()
    older = tmp_path / "older.jsonl"
    newer = tmp_path / "newer.jsonl"
    older.write_text("", encoding="utf-8")
    newer.write_text("", encoding="utf-8")
    older.touch()
    newer.touch()

    resolved = module.resolve_log_path(tmp_path)

    assert resolved == newer


def test_packet_paths_use_logical_casebook_name(tmp_path: Path):
    module = _load_refresh_module()
    log_path = tmp_path / "society-baseline-20260408T171454Z.jsonl"

    outputs = module.packet_paths(tmp_path, log_path)

    assert outputs["summary_markdown"] == tmp_path / "interim_summary.md"
    assert outputs["summary_csv"] == tmp_path / "interim_summary.csv"
    assert outputs["figures_dir"] == tmp_path / "monitoring_figures"
    assert outputs["casebook_markdown"] == tmp_path / "society-baseline-casebook.md"
    assert outputs["status_json"] == tmp_path / "live_status.json"
