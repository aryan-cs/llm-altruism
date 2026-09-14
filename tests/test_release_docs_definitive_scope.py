from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ACTIVE_RELEASE_DOCS = (
    Path("README.md"),
    Path("CHECKPOINT.md"),
    Path("docs/PROVIDER_SAFE_V2_DEFINITIVE_ANALYSIS.md"),
    Path("docs/PROVIDER_SAFE_V2_PAPER_ASSETS.md"),
    Path("docs/release/README.md"),
    Path("docs/release/DATA_CARD.md"),
    Path("docs/release/MODEL_REGISTRY.md"),
    Path("docs/release/REPRODUCIBILITY.md"),
    Path("docs/release/COMPUTE.md"),
    Path("docs/release/LICENSES_AND_TERMS.md"),
    Path("docs/conference_submission/SUPPLEMENT_README.md"),
    Path("docs/conference_submission/SUPPLEMENT_MODEL_REGISTRY.md"),
    Path("docs/REVIEW_RESPONSE_MATRIX.md"),
)
PAPER_FACING_RELEASE_DOCS = tuple(
    path for path in ACTIVE_RELEASE_DOCS if path != Path("CHECKPOINT.md")
)


def _read(path: Path) -> str:
    candidate = ROOT / path
    if not candidate.is_file() and (ROOT / "SUPPLEMENT_MANIFEST.json").is_file():
        archive_fallbacks = {
            Path("docs/conference_submission/SUPPLEMENT_README.md"): Path("README.md"),
            Path("docs/conference_submission/SUPPLEMENT_MODEL_REGISTRY.md"): (
                Path("docs/release/MODEL_REGISTRY.md")
            ),
        }
        candidate = ROOT / archive_fallbacks.get(path, path)
    return candidate.read_text(encoding="utf-8")


def test_active_release_docs_do_not_reactivate_superseded_deadline_scopes() -> None:
    # CHECKPOINT.md deliberately preserves sealed historical evidence and is
    # tested separately for an explicit non-paper boundary below.
    combined = "\n".join(_read(path) for path in PAPER_FACING_RELEASE_DOCS)
    forbidden = (
        "part0-sota-panel-v2-n24",
        "part2-sota-matched-v1-n8",
        "16 included of 24",
        "75 reportable of 81",
        "22 executed of 24",
        "73 × n=96",
        "73 at balanced n=96",
        "eight independent common-seed trajectories (176 total)",
        "definitive-part2-n12-main19-v3",
        "19 Part 2 routes",
        "19 exact model routes",
        "228 completed trajectories",
        "228 trajectories total",
        "225 trajectories",
        "13,495 scheduled agent-days",
        "five-agent, 12-step",
        "five-agent populations, 12 steps",
    )
    for stale_claim in forbidden:
        assert stale_claim not in combined


def test_checkpoint_records_terminal_100day_evidence_and_historical_boundary() -> None:
    checkpoint = _read(Path("CHECKPOINT.md"))
    normalized = " ".join(checkpoint.split())
    for required in (
        "2026-09-13T20:10:51.961115Z",
        "8b2034df599ec7a6e2abbdf5964127bb0987fd1f95372be7343e06baef715079",
        "three pairs, 23 routes, 276 trajectories, and the same 12 common seeds",
        "57 source operational-failure trajectories resolved across 85 repair rounds",
        "1,206,808 living agent-days",
        "1,180,046 valid actions",
        "26,762 genuine semantic `INVALID` actions",
        "969,640 restraint",
        "210,406 overuse",
        "198/276 trajectories",
        "Exactly `anthropic/claude-opus-4-5` is excluded, without substitution",
        "No API key value was printed",
        "The evidence-collection phase is closed",
        "analysis, paper-asset generation, narrative integration, PDF build and visual inspection, anonymous supplement build, clean extraction, deterministic rebuild, and clean compile are complete",
    ):
        assert required in normalized
    assert checkpoint.count("    --pair ") == 3
    assert "data/analysis/final_results/final_results.json" in checkpoint
    assert "historical provenance only" in normalized
    assert "operational-repair-multikey-v3" in checkpoint
    assert "immutable diagnostic, non-paper evidence" in normalized
    for stale_live_wording in (
        "Recent live progress",
        "evidence ETA",
        "next paper candidate",
        "only active and thus slowest",
    ):
        assert stale_live_wording not in checkpoint


def test_release_reproduction_uses_exact_100day_part2_composition() -> None:
    reproducibility = _read(Path("docs/release/REPRODUCIBILITY.md"))
    for required in (
        "definitive-part0-large-n48-main22-deadline-v6",
        "definitive-part1-large-n384-main75-deadline-v5",
        "definitive-part1-large-n384-main75-deadline-v5-operational-repair-v1",
        "full-part2-n12-n50-d100-main21-v5",
        "full-part2-n12-n50-d100-main21-v5-operational-completion-capability-v4",
        "full-part2-n12-n50-d100-nemotron-3-ultra-recovered-v1",
        "full-part2-n12-n50-d100-nemotron-3-ultra-operational-repair-v1",
        "full-part2-n12-n50-d100-deepseek-v4-flash-recovered-v1",
        "full-part2-n12-n50-d100-deepseek-v4-flash-operational-repair-v1",
        "definitive-part1-role-calibration-v3",
        "definitive-part2-sensitivity-deadline-fast-v9",
        "definitive-part2-sensitivity-deadline-fast-v9-operational-repair-v1",
        "analysis.analyze_provider_safe_v2_definitive",
        "analysis.build_provider_safe_v2_paper_assets",
        "analysis.build_provider_safe_v2_croissant_metadata",
        "--part0-terminal-policy all-scheduled-operational-invalid-v1",
        "--part1-operational-repair",
        "--part2-declared-exclusion anthropic/claude-opus-4-5",
        "--sensitivity-operational-repair",
        "--require-definitive-artifacts",
    ):
        assert required in reproducibility
    assert reproducibility.count("  --part2-source-overlay ") == 3
    main = reproducibility.index(
        "  --part2-source-overlay data/private/inference_hub/full-part2-n12-n50-d100-main21-v5 "
    )
    nemotron = reproducibility.index(
        "  --part2-source-overlay data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-recovered-v1 "
    )
    deepseek = reproducibility.index(
        "  --part2-source-overlay data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-recovered-v1 "
    )
    assert main < nemotron < deepseek
    assert "  --part2 data/" not in reproducibility


def test_release_docs_bind_completed_original_scale_part2_counts() -> None:
    readme = _read(Path("README.md"))
    data_card = _read(Path("docs/release/DATA_CARD.md"))
    supplement = _read(Path("docs/conference_submission/SUPPLEMENT_README.md"))
    for document in (readme, data_card, supplement):
        normalized = " ".join(document.split())
        assert "22 exact" in normalized
        assert "75 exact" in normalized
        assert "23 exact" in normalized
        assert "276" in normalized
        assert "198" in normalized
        assert "78" in normalized
        assert "18" in normalized
        assert "five zero-eligible routes" in normalized
        assert "NE" in normalized
        assert "57 whole-trajectory repairs" in normalized
        assert "1,206,808" in normalized
        assert "1,180,046" in normalized
        assert "26,762" in normalized
        assert "969,640" in normalized
        assert "210,406" in normalized
        assert (
            "anthropic/claude-opus-4-5" in normalized
            or "Opus 4.5" in normalized
        )


def test_release_docs_preserve_part2_inference_and_review_boundaries() -> None:
    combined = "\n".join(_read(path) for path in PAPER_FACING_RELEASE_DOCS)
    normalized = " ".join(combined.split())
    assert "276 independent trajectories" not in normalized
    assert "78 trajectories" in normalized
    assert "18 routes" in normalized or "18 route summaries" in normalized
    assert "five zero-eligible routes" in normalized

    review_matrix = " ".join(
        _read(Path("docs/REVIEW_RESPONSE_MATRIX.md")).split()
    )
    assert "fixed-panel exploratory rank correlations" in review_matrix
    assert "Exploratory descriptive associations only" in review_matrix
    assert "Collection terminal under explicit policy" in review_matrix
    assert "no cross-axis claim in the current paper" not in review_matrix
    assert "Collection running" not in review_matrix
    assert "exact translated strings were not retained" not in review_matrix


def test_release_docs_bind_frozen_part2_environment() -> None:
    combined = "\n".join(
        _read(path)
        for path in (
            Path("README.md"),
            Path("docs/release/COMPUTE.md"),
            Path("docs/release/DATA_CARD.md"),
            Path("docs/release/MODEL_REGISTRY.md"),
            Path("docs/conference_submission/SUPPLEMENT_README.md"),
            Path("docs/conference_submission/SUPPLEMENT_MODEL_REGISTRY.md"),
        )
    )
    for required in (
        "50-agent",
        "100-day",
        "initial capacity 2,500",
        "OPTION_B private gain 2",
        "reserve cost 2",
        "unanimous group benefit/penalty 5",
        "collapse death rate 0.2",
    ):
        assert required in combined
