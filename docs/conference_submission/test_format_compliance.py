"""Submission-format checks that do not modify or rebuild the paper."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEX = ROOT / "conference_submission.tex"
CHECKLIST = ROOT / "checklist.tex"
STYLE = ROOT / "neurips_2026.sty"
PDF = ROOT / "conference_submission.pdf"

OFFICIAL_STYLE_SHA256 = "c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a"
OFFICIAL_QUESTIONS = (
    "Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?",
    "Does the paper discuss the limitations of the work performed by the authors?",
    "For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?",
    "Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?",
    "Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?",
    "Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?",
    "Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?",
    "For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?",
    "Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics \\url{https://neurips.cc/public/EthicsGuidelines}?",
    "Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?",
    "Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?",
    "Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?",
    "Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?",
    "For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?",
    "Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?",
    "Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does \\emph{not} impact the core methodology, scientific rigor, or originality of the research, declaration is not required.",
)


def _find_pdftotext() -> str | None:
    on_path = shutil.which("pdftotext")
    if on_path:
        return on_path
    bundled = (
        Path.home()
        / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/pdftotext"
    )
    return str(bundled) if bundled.is_file() else None


class ConferenceSubmissionFormatTest(unittest.TestCase):
    def test_official_eandd_submission_style(self) -> None:
        source = TEX.read_text(encoding="utf-8")
        self.assertIn(r"\usepackage[eandd]{neurips_2026}", source)
        self.assertNotRegex(source, r"\\usepackage\[[^]]*(?:final|preprint)[^]]*\]\{neurips_2026\}")
        self.assertEqual(hashlib.sha256(STYLE.read_bytes()).hexdigest(), OFFICIAL_STYLE_SHA256)

    def test_full_official_checklist(self) -> None:
        source = CHECKLIST.read_text(encoding="utf-8")
        questions = tuple(
            re.findall(r"^\s*\\item\[\] Question: (.*)$", source, flags=re.MULTILINE)
        )
        self.assertEqual(questions, OFFICIAL_QUESTIONS)
        self.assertEqual(
            len(re.findall(r"^\s*\\item\[\] Answer: \\answer(?:Yes|No|NA)\{\}$", source, flags=re.MULTILINE)),
            16,
        )
        self.assertEqual(source.count(r"\item[] Guidelines:"), 16)
        self.assertNotRegex(source, r"\\(?:answer|justification)T\w+")
        self.assertNotIn("BEGIN INSTRUCTIONS", source)
        self.assertNotIn("END INSTRUCTIONS", source)

    def test_checklist_follows_references_and_appendix(self) -> None:
        source = TEX.read_text(encoding="utf-8")
        self.assertLess(source.index(r"\bibliography{references}"), source.index(r"\appendix"))
        self.assertLess(source.index(r"\appendix"), source.index(r"\input{checklist}"))

    def test_submission_remains_anonymous(self) -> None:
        source = TEX.read_text(encoding="utf-8")
        self.assertRegex(source, r"\\author\{Anonymous Authors\}")
        self.assertNotRegex(source, r"\\author\{[^}]*@[^}]*\}")
        self.assertNotIn(r"\usepackage[final", source)

    def test_fresh_original_benchmark_narrative_is_locked(self) -> None:
        source = TEX.read_text(encoding="utf-8")
        retired_figures = (
            "behavioral_fingerprint_heatmap.png",
            "model_behavior_pca.png",
            "part0_refusal_by_language_heatmap.png",
            "part0_refusal_rate_by_model.png",
            "safety_refusal_vs_restraint.png",
        )
        for figure in retired_figures:
            self.assertNotIn(figure, source)
        self.assertNotIn("9.4--99.3", source)
        self.assertNotIn("$r{=}0.77$", source)
        self.assertNotIn("300 of 1,243 adjudicated rows", source)
        self.assertNotIn("When a Benchmark Fails Its Audit", source)
        self.assertNotIn("negative benchmark", source.lower())
        self.assertNotIn("prompt--engine contract failure", source.lower())
        self.assertIn(r"\title{Safety Beyond Refusal}", source)
        self.assertIn("three separate behavioral probes", source)
        self.assertIn("explicit and identifiable harm to bilateral and then diffuse, cumulative externalities", source)
        self.assertIn(r"\texttt{OPTION\_A} awards one private point", source)
        self.assertIn(r"\texttt{OPTION\_B} awards two private points and removes two reserve units", source)
        self.assertIn("seven systems preserve both the reserve and population perfectly", source)

    def test_definitive_assets_title_and_table_spacing_are_locked(self) -> None:
        source = TEX.read_text(encoding="utf-8")
        self.assertEqual(source.count(r"\title{Safety Beyond Refusal}"), 1)
        self.assertNotIn(r"\PaperPart", source)
        for stale in ("24-root", "96-root", "8 common seeds", "all eight trajectories"):
            self.assertNotIn(stale, source)
        self.assertEqual(source.count(r"\setlength{\floatsep}{15pt}"), 2)
        self.assertEqual(source.count(r"\setlength{\textfloatsep}{15pt}"), 2)
        self.assertEqual(source.count(r"\setlength{\intextsep}{15pt}"), 2)

        asset_root = "../../data/processed/provider-safe-v2-paper-assets/"
        static_figures = (
            "part0_model_language.pdf",
            "part1_role_calibration.pdf",
            "part2_sensitivity_effects.pdf",
        )
        tables = (
            "part0_model_language_table.tex",
            "part1_all_models_table.tex",
            "part2_all_models_table.tex",
            "part1_role_calibration_table.tex",
            "part2_sensitivity_effects_table.tex",
            "part1_local_controls_table.tex",
        )
        for name in (*static_figures, *tables, "all_models_cross_phase_table.tex"):
            self.assertEqual(source.count(asset_root + name), 1, name)
        self.assertEqual(source.count(asset_root + "paper_headlines.tex"), 2)
        self.assertEqual(source.count("figures/part2_all_models.pdf"), 1)
        self.assertIn(asset_root + r"part1_all_models_block\block.pdf", source)
        self.assertIn(
            asset_root + r"all_models_cross_phase_outcome_profile_block\block.pdf",
            source,
        )
        self.assertIn(r"\foreach \block in {1,2,3}", source)
        self.assertIn(r"\foreach \block in {1,2,3,4}", source)
        self.assertNotIn("availability_retry", source)
        self.assertNotIn("semantic_invalid_repair", source)

    def test_rendered_main_text_boundary_when_extractor_is_available(self) -> None:
        extractor = _find_pdftotext()
        if extractor is None or not PDF.is_file():
            self.skipTest("compiled PDF or pdftotext is unavailable")
        if PDF.stat().st_mtime < TEX.stat().st_mtime:
            self.skipTest("compiled PDF is older than the manuscript source")
        result = subprocess.run(
            [extractor, "-layout", str(PDF), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        pages = result.stdout.split("\f")

        def page_matching(pattern: str) -> int:
            for page_number, page in enumerate(pages, start=1):
                if re.search(pattern, page, flags=re.MULTILINE):
                    return page_number
            self.fail(f"PDF heading not found: {pattern}")

        conclusion_page = page_matching(r"^\s*\d+\s+(?:\d+\s+)?Conclusion\s*$")
        references_page = page_matching(r"^\s*\d+\s+(?:\d+\s+)?References\s*$")
        self.assertLessEqual(conclusion_page, 10)
        self.assertLessEqual(references_page, 10)


if __name__ == "__main__":
    unittest.main()
