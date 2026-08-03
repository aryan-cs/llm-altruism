from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ACTIVE_RELEASE_DOCS = (
    Path("README.md"),
    Path("CHECKPOINT.md"),
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


def _read(path: Path) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_active_release_docs_do_not_reactivate_superseded_deadline_scopes() -> None:
    combined = "\n".join(_read(path) for path in ACTIVE_RELEASE_DOCS)
    forbidden = (
        "part0-sota-panel-v2-n24",
        "part2-sota-matched-v1-n8",
        "16 included of 24",
        "75 reportable of 81",
        "22 executed of 24",
        "73 × n=96",
        "73 at balanced n=96",
        "eight independent common-seed trajectories (176 total)",
    )
    for stale_claim in forbidden:
        assert stale_claim not in combined


def test_release_reproduction_uses_only_definitive_provider_safe_v2_inputs() -> None:
    reproducibility = _read(Path("docs/release/REPRODUCIBILITY.md"))
    for required in (
        "definitive-part0-large-n48-main22-deadline-v6",
        "definitive-part1-large-n384-main75-deadline-v5",
        "definitive-part2-n12-main19-v3",
        "definitive-part1-role-calibration-v3",
        "definitive-part2-sensitivity-deadline-fast-v9",
        "analysis.analyze_provider_safe_v2_definitive",
        "analysis.build_provider_safe_v2_paper_assets",
        "analysis.build_provider_safe_v2_croissant_metadata",
        "--require-definitive-artifacts",
    ):
        assert required in reproducibility


def test_release_docs_distinguish_schedule_from_completion() -> None:
    readme = _read(Path("README.md"))
    data_card = _read(Path("docs/release/DATA_CARD.md"))
    supplement = _read(Path("docs/conference_submission/SUPPLEMENT_README.md"))
    for document in (readme, data_card, supplement):
        assert "22 exact" in document
        assert "75 exact" in document
        assert "19 exact" in document
        assert "228" in document
        assert "225" in document
    assert "Collection is still in progress" in " ".join(readme.split())
    assert "source manifest was still running" in " ".join(supplement.split())
    assert "No final definitive self-hash is claimed" in " ".join(data_card.split())
