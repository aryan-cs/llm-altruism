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
HEADLINES = (
    ROOT / "../../data/processed/provider-safe-v2-paper-assets/paper_headlines.tex"
).resolve()

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


def _find_pdfinfo() -> str | None:
    return shutil.which("pdfinfo")


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
        self.assertEqual(
            re.findall(r"\\author\{[^{}]*\}", source),
            [r"\author{Anonymous Authors}"],
        )
        self.assertNotRegex(
            source,
            r"\\usepackage\[[^]]*\b(?:final|preprint|nonanonymous)\b[^]]*\]"
            r"\{neurips_2026\}",
        )
        self.assertNotRegex(
            source, r"\\(?:thanks|affiliation|institute|address|orcidlink)\b"
        )
        self.assertNotRegex(
            source,
            r"\\(?:begin\{(?:ack|acknowledg(?:e)?ments?)\}|acksection\b|"
            r"(?:section|subsection|paragraph)\*?\{Acknowledg(?:e)?ments?\b)",
        )
        self.assertNotRegex(
            source,
            r"(?im)^\s*(?:Department|School|College|University|Institution)\b",
        )
        self.assertNotRegex(
            source,
            r"(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
        )

    def test_fresh_original_benchmark_narrative_is_locked(self) -> None:
        source = TEX.read_text(encoding="utf-8")
        retired_figures = (
            "behavioral_fingerprint_heatmap.png",
            "model_behavior_pca.png",
            "part0_refusal_by_language_heatmap.png",
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
        self.assertIn("Part~2 places 50 same-model agents in a 100-day", source)
        self.assertIn("initialized with 2,500 reserve units", source)
        self.assertIn("all 23 current exact routes", source)
        for macro in (
            "ProviderSafePartTwoEnvironmentallyEstimableTrajectoryCount",
            "ProviderSafePartTwoSemanticInvalidTrajectoryCount",
            "ProviderSafePartTwoNonestimableModelCount",
            "ProviderSafePartTwoInvalidAgentDayCount",
        ):
            self.assertIn(macro, source)
        self.assertIn("18 route summaries", source)
        self.assertIn("five routes are NE", source)
        self.assertIn(
            "The seed-level restraint mean uses a Student-$t$ interval over all 12 "
            "independent trajectories",
            source,
        )
        self.assertIn(
            "Continuous environmental outcomes use Student-$t$ intervals over the "
            "zero-invalid eligible trajectories available to that route, while reserve "
            "nondepletion uses Wilson score intervals",
            source,
        )
        self.assertIn("primary descriptive action proportion pools all scheduled", source)
        self.assertIn("equal-seed-weighted estimate need not equal the pooled", source)
        self.assertIn(r"\mathrm{AUPC}=", source)
        self.assertIn("route with one eligible trajectory has a point estimate", source)
        self.assertIn("Orange cells are genuine semantic-invalid nonactions", source)
        self.assertIn("zero-invalid eligible trajectories", source)
        self.assertIn("five zero-eligible routes are omitted and reported as NE", source)
        self.assertNotIn("independent Part~2 trajectories", source)
        self.assertNotIn("day-wise mean of 12 common-seed trajectories", source)
        self.assertNotIn("a shorter bar means more private-gain overuse", source)
        stale_part2_patterns = (
            r"\b(?:five|5)[-~ ](?:same-model[-~ ]?)?agents?\b",
            r"\b12[-~ ]day\b",
            r"Part~2[^\n]*(?:schedules|observes|contains|evaluates)\s+"
            r"(?:19|22)\s+(?:systems|routes)\b",
            r"Part~2[^\n]*(?:for\s+)?(?:all\s+)?(?:19|22)\s+"
            r"(?:current\s+)?exact(?:\s+model)?\s+routes\b",
            r"Repeated-commons outcomes for\s+(?:19|22)\b",
            r"longitudinal commons views for the\s+(?:19|22)\b",
            r"Part~2 agent-day action raster for all\s+(?:19|22)\b",
            r"\b(?:19|22)\s+Part~2\s+routes\b",
            r"\b228\s+(?:bound\s+)?Part~2\s+trajectories\b",
            r"shared reserve units out of the initial 50(?:[.,;:]|\s*$)",
            r"simulation days 1--12\b",
        )
        for pattern in stale_part2_patterns:
            self.assertNotRegex(source, pattern)
        self.assertIn(
            "Matched profile for the 22 current exact routes scheduled in all three parts",
            source,
        )
        self.assertIn("Across the six role-calibration sentinels", source)
        self.assertIn("five-compatible-sentinel sensitivity panel", source)
        self.assertIn("25 prespecified high-minus-low AURC contrasts", source)
        self.assertNotIn("six-route sensitivity panel", source)
        self.assertNotIn("30 prespecified route--factor tests", source)

    def test_generated_headline_counts_match_full_part2_panel(self) -> None:
        self.assertTrue(HEADLINES.is_file(), "generated paper headlines are absent")
        headlines = HEADLINES.read_text(encoding="utf-8")

        def macro_integer(name: str) -> int:
            match = re.search(
                rf"^\\newcommand\{{\\{re.escape(name)}\}}\{{(\d+)\}}$",
                headlines,
                flags=re.MULTILINE,
            )
            self.assertIsNotNone(match, f"generated macro is absent: {name}")
            return int(match.group(1))

        self.assertEqual(macro_integer("ProviderSafePartTwoModelCount"), 23)
        self.assertEqual(macro_integer("ProviderSafePartTwoTrajectoryCount"), 276)
        self.assertEqual(
            macro_integer("ProviderSafePairwisePartZeroPartTwoMatchedRouteCount"),
            22,
        )
        self.assertEqual(
            macro_integer("ProviderSafePairwisePartOnePartTwoMatchedRouteCount"),
            22,
        )

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
            "part0_model_language",
            "part1_role_calibration",
            "part2_sensitivity_effects",
        )
        tables = ("compact_matched_core_table.tex",)
        for name in (*static_figures, *tables):
            self.assertEqual(source.count(asset_root + name), 1, name)
        for name in (
            "all_models_cross_phase_table.tex",
            "part0_model_language_table.tex",
            "part1_all_models_table.tex",
            "part2_all_models_table.tex",
            "part1_role_calibration_table.tex",
            "part2_sensitivity_effects_table.tex",
            "part1_local_controls_table.tex",
        ):
            self.assertNotIn(asset_root + name, source, name)
        self.assertEqual(source.count(asset_root + "paper_headlines.tex"), 2)
        self.assertEqual(source.count("figures/part2_all_models"), 1)
        for name in (
            "part0_refusal_rate_by_model",
            "part2_restraint_rate_by_model",
            "part2_shared_reserve_over_time",
            "part2_population_over_time",
            "part2_agent_day_raster_current",
        ):
            self.assertEqual(source.count("figures/" + name), 1, name)
        self.assertIn(asset_root + r"part1_all_models_block\block", source)
        self.assertIn(
            asset_root + r"all_models_cross_phase_outcome_profile_block\block",
            source,
        )
        self.assertIn(r"\foreach \block in {1,2,3}", source)
        self.assertIn(r"\foreach \block in {1,2,3,4}", source)
        self.assertNotIn("availability_retry", source)
        self.assertNotIn("semantic_invalid_repair", source)

    def test_every_submission_page_remains_portrait(self) -> None:
        source = TEX.read_text(encoding="utf-8")
        self.assertNotIn(r"\begin{landscape}", source)
        self.assertNotIn(r"\end{landscape}", source)
        self.assertNotIn(r"\usepackage{pdflscape}", source)
        inspector = _find_pdfinfo()
        if inspector is None or not PDF.is_file():
            return
        result = subprocess.run(
            [inspector, "-f", "1", "-l", "9999", str(PDF)],
            check=True,
            capture_output=True,
            text=True,
        )
        sizes = re.findall(r"Page\s+\d+\s+size:\s+([0-9.]+)\s+x\s+([0-9.]+)", result.stdout)
        rotations = re.findall(r"Page\s+\d+\s+rot:\s+(-?\d+)", result.stdout)
        self.assertTrue(sizes, "pdfinfo did not report per-page sizes")
        self.assertEqual(len(sizes), len(rotations))
        for page_number, ((width, height), rotation) in enumerate(zip(sizes, rotations, strict=True), 1):
            self.assertLess(float(width), float(height), f"page {page_number} is not portrait")
            self.assertEqual(int(rotation) % 360, 0, f"page {page_number} is rotated")

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
