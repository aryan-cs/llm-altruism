#!/usr/bin/env python3
"""Build an anonymous ICML-style submission PDF from the paper source bundle."""

from __future__ import annotations

import re
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
ICML_DIR = PAPER_DIR / "icml2025"
STYLE_DIR = ICML_DIR / "style"
BUILD_DIR = ICML_DIR / "build"
TRACKED_PDF = ICML_DIR / "llm_altruism_icml2025_submission.pdf"
BIB_PATH = PAPER_DIR / "references.bib"


LONGTABLE_CAPTIONS = [
    (
        "Payoff matrices for the repeated-game precursor probes.",
        "tab:payoff-matrices",
    ),
    (
        "Scarcity society survival and social-structure outcomes.",
        "tab:scarcity-society",
    ),
    (
        "Public-reputation society survival and social-structure outcomes.",
        "tab:reputation-society",
    ),
    (
        "Cross-game precursor baseline on the stable triplet cohort.",
        "tab:precursor-cross-game",
    ),
    (
        "Pooled Prisoner's Dilemma neutral-family replication across two cohorts.",
        "tab:pd-neutral-family",
    ),
    (
        "Prompt susceptibility across the three core precursor games.",
        "tab:precursor-susceptibility",
    ),
    (
        "Benchmark presentation in Prisoner's Dilemma.",
        "tab:pd-benchmark",
    ),
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def section_between(text: str, start_heading: str, end_heading: str | None = None) -> str:
    start = text.index(start_heading)
    if end_heading is None:
        return text[start:].strip() + "\n"
    end = text.index(end_heading, start)
    return text[start:end].strip() + "\n"


def body_without_heading(text: str, heading: str) -> str:
    start = text.index(heading)
    body = text[start + len(heading) :].lstrip()
    return body


def section_body_between(text: str, start_heading: str, end_heading: str | None = None) -> str:
    section = section_between(text, start_heading, end_heading)
    return body_without_heading(section, start_heading)


def normalize_markdown_headings(markdown: str) -> str:
    """Strip manual section numbering so LaTeX owns numbering consistently."""
    pattern = re.compile(
        r"^(#{2,6})\s+(?:\d+(?:\.\d+)*\.?|[A-Z]\.)\s+",
        re.MULTILINE,
    )
    return pattern.sub(r"\1 ", markdown)


def run_pandoc(markdown: str, *, shift_headings: int = 0, natbib: bool = False) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "source.md"
        target = Path(tmpdir) / "target.tex"
        source.write_text(markdown, encoding="utf-8")
        cmd = [
            "pandoc",
            str(source),
            "-f",
            "markdown+citations",
            "-t",
            "latex",
            "--listings",
            "-o",
            str(target),
        ]
        if shift_headings:
            cmd.append(f"--shift-heading-level-by={shift_headings}")
        if natbib:
            cmd.extend(["--natbib", f"--bibliography={BIB_PATH}"])
        subprocess.run(cmd, check=True, cwd=ROOT)
        return target.read_text(encoding="utf-8")


def clean_pandoc_tex(tex: str) -> str:
    # Pandoc emits empty \noalign{} and passthrough wrappers around lstinline.
    tex = tex.replace(r"\noalign{}", "")
    return tex


def convert_longtables(tex: str) -> str:
    counter = {"value": 0}
    pattern = re.compile(
        r"\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}\n(.*?)\\end\{longtable\}",
        re.DOTALL,
    )

    def repl(match: re.Match[str]) -> str:
        index = counter["value"]
        counter["value"] += 1
        spec = match.group(1)
        content = match.group(2)
        if r"\endhead" not in content or r"\endlastfoot" not in content:
            return match.group(0)

        header, remainder = content.split(r"\endhead", maxsplit=1)
        footer, body = remainder.split(r"\endlastfoot", maxsplit=1)

        def normalize(block: str) -> str:
            return block.replace(r"\noalign{}", "").strip()

        header = normalize(header)
        body = normalize(body)
        footer = normalize(footer)

        caption, label = LONGTABLE_CAPTIONS[min(index, len(LONGTABLE_CAPTIONS) - 1)]
        tabular = "\n".join(part for part in [header, body, footer] if part)
        return (
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\small\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}\n"
            f"{tabular}\n"
            "\\end{tabular}\n"
            "\\end{table*}\n"
        )

    return pattern.sub(repl, tex)


def selected_figures_tex() -> str:
    return r"""
\section{Selected Figures}

\begin{figure*}[t]
\centering
\includegraphics[width=0.48\textwidth]{../../figures/society_reputation_live/society_reputation_final_survival.png}
\includegraphics[width=0.48\textwidth]{../../figures/society_reputation_live/society_reputation_trade_volume.png}
\caption{Institutional outcomes under scarcity and public reputation. Final survival is shown on the left and trade volume on the right. Public reputation equalizes survival while preserving large prompt-conditioned differences in social activity.}
\label{fig:institution-selected-a}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.48\textwidth]{../../figures/society_reputation_live/society_reputation_alliance_count.png}
\caption{Alliance formation under scarcity and public reputation. Cooperative prompting produces visible alliance structure even when final survival converges under public reputation.}
\label{fig:institution-selected-b}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.48\textwidth]{../../figures/repaired_pd_replications/baseline_prompt_variants_cooperation.png}
\includegraphics[width=0.48\textwidth]{../../figures/triplet_live/baseline_prompt_variants_cooperation.png}
\caption{Repeated-game precursor diagnostics. Neutral-baseline robustness in Prisoner's Dilemma is shown on the left, and the cross-game baseline comparison is shown on the right. These figures motivate why society-level interpretation needs prompt-robust micro-level evidence.}
\label{fig:baseline-selected}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.48\textwidth]{../../figures/benchmark_live/benchmark_presentations_cooperation.png}
\includegraphics[width=0.48\textwidth]{../../figures/susceptibility_live/susceptibility_prompt_variants_cooperation.png}
\caption{Additional repeated-game precursor probes. Benchmark-presentation effects are shown on the left and prompt susceptibility on the right. Both are included to document why pairwise behavior should not be treated as a prompt-free macro baseline.}
\label{fig:part1-selected}
\end{figure*}
""".strip() + "\n"


def write_main_tex(abstract_tex: str) -> None:
    main = rf"""
\documentclass{{article}}

\usepackage{{microtype}}
\usepackage{{graphicx}}
\usepackage{{subfigure}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{calc}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{icml2025}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{mathtools}}
\usepackage{{amsthm}}
\usepackage{{listings}}
\usepackage[capitalize,noabbrev]{{cleveref}}

\providecommand{{\theHalgorithm}}{{\arabic{{algorithm}}}}
\providecommand{{\tightlist}}{{}}
\newcommand{{\passthrough}}[1]{{#1}}
\lstset{{basicstyle=\ttfamily\footnotesize,breaklines=true,columns=fullflexible}}

\icmltitlerunning{{Can LLM Agents Sustain a Society?}}

\begin{{document}}

\twocolumn[
\icmltitle{{Can LLM Agents Sustain a Society? \\ Repeated Games as Precursor Probes for Collective Survival in LLM Societies}}

\begin{{icmlauthorlist}}
\icmlauthor{{Anonymous Authors}}{{anon}}
\end{{icmlauthorlist}}

\icmlaffiliation{{anon}}{{Anonymous Institution}}
\icmlcorrespondingauthor{{Anonymous Author}}{{anon.email@domain.com}}
\icmlkeywords{{large language models, artificial societies, multi-agent systems, alignment evaluation, reputation systems}}

\vskip 0.3in
]

\printAffiliationsAndNotice{{}}

\begin{{abstract}}
{abstract_tex}
\end{{abstract}}

\input{{generated_body.tex}}

\clearpage
\appendix
\input{{generated_appendix.tex}}
\input{{selected_figures.tex}}

\clearpage
\bibliographystyle{{icml2025}}
\bibliography{{../../references}}

\end{{document}}
""".strip() + "\n"
    (BUILD_DIR / "main.tex").write_text(main, encoding="utf-8")


def compile_pdf() -> None:
    env = os.environ.copy()
    texinputs = str(STYLE_DIR.resolve())
    if env.get("TEXINPUTS"):
        texinputs = texinputs + os.pathsep + env["TEXINPUTS"]
    env["TEXINPUTS"] = texinputs + os.pathsep
    bstinputs = str(STYLE_DIR.resolve())
    if env.get("BSTINPUTS"):
        bstinputs = bstinputs + os.pathsep + env["BSTINPUTS"]
    env["BSTINPUTS"] = bstinputs + os.pathsep
    bibinputs = str(PAPER_DIR.resolve())
    if env.get("BIBINPUTS"):
        bibinputs = bibinputs + os.pathsep + env["BIBINPUTS"]
    env["BIBINPUTS"] = bibinputs + os.pathsep
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=BUILD_DIR,
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    subprocess.run(
        ["bibtex", "main"],
        cwd=BUILD_DIR,
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=BUILD_DIR,
            check=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    pdf = BUILD_DIR / "main.pdf"
    if pdf.exists():
        pdf.replace(BUILD_DIR / "llm_altruism_icml2025_submission.pdf")
    shutil.copy2(BUILD_DIR / "llm_altruism_icml2025_submission.pdf", TRACKED_PDF)
    for style_name in ["algorithm.sty", "algorithmic.sty", "fancyhdr.sty", "icml2025.bst", "icml2025.sty"]:
        stale = BUILD_DIR / style_name
        if stale.exists():
            stale.unlink()
    for pattern in ["*.aux", "*.blg", "*.log", "*.out", "*.toc"]:
        for path in BUILD_DIR.glob(pattern):
            path.unlink()


def main() -> None:
    manuscript = normalize_markdown_headings(read_text(PAPER_DIR / "MANUSCRIPT.md"))
    appendix = normalize_markdown_headings(read_text(PAPER_DIR / "APPENDIX.md"))

    abstract_md = section_body_between(manuscript, "## Abstract", "## Introduction")
    body_md = section_between(manuscript, "## Introduction", "## References")
    appendix_md = body_without_heading(appendix, "# Appendix")

    abstract_tex = clean_pandoc_tex(run_pandoc(abstract_md, natbib=True)).strip()
    body_tex = clean_pandoc_tex(run_pandoc(body_md, shift_headings=-1, natbib=True))
    body_tex = convert_longtables(body_tex)
    appendix_tex = clean_pandoc_tex(run_pandoc(appendix_md, shift_headings=-1))

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    (BUILD_DIR / "generated_body.tex").write_text(body_tex, encoding="utf-8")
    (BUILD_DIR / "generated_appendix.tex").write_text(appendix_tex, encoding="utf-8")
    (BUILD_DIR / "selected_figures.tex").write_text(selected_figures_tex(), encoding="utf-8")
    stale_references = BUILD_DIR / "generated_references.tex"
    if stale_references.exists():
        stale_references.unlink()
    write_main_tex(abstract_tex)
    compile_pdf()
    print(TRACKED_PDF)


if __name__ == "__main__":
    main()
