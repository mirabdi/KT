# Comprehensive Survey and Benchmark of Deep Knowledge Tracing Paradigms

This directory contains the complete, self-contained, Overleaf-compatible LaTeX project for the paper:
**"Tracking the Student Mind: A Comprehensive Survey and Benchmark of Deep Knowledge Tracing Paradigms, Interpretability, and Deployment Trade-Offs"**

Structured following the architectural and conceptual framework of landmark attentive KT models (e.g., AKT, Ghosh et al., KDD 2020), this work provides an accessible, rigorous survey and benchmark of deep student modeling designed for computer science and engineering researchers.

---

## Directory Layout

```text
papers/kt_survey/
├── main.tex                    # Master LaTeX document (Overleaf main entrypoint)
├── main.pdf                    # Compiled publication-ready PDF (13 pages, ACL format)
├── custom.bib                  # BibTeX bibliography with 26 authoritative citations
├── acl.sty                     # Official *ACL conference style file
├── acl_natbib.bst              # ACL Natbib citation bibliography style file
├── images/                     # 300 DPI publication figures (both vector PDF and raster PNG)
│   ├── fig1_kt_landscape.pdf & .png                   # Evolutionary Taxonomy of KT (1994--2024)
│   ├── fig2_tradeoff_pareto.pdf & .png                # Latency vs. Accuracy & Parameter Scaling
│   ├── fig3_interpretability_deconstructed.pdf & .png # Attention Heads, Rasch IRT & Slot Collision
│   ├── fig4_cross_dataset_radar.pdf & .png            # Cross-Regime Radar Profile
│   └── fig5_synapsekt_architecture.pdf & .png             # SynapseKT Cognitive Dual-Process Architecture
└── README.md                   # This project guide
```

---

## Overleaf Compatibility & How to Import

### 1. Direct Overleaf Import
1. Download or locate `papers/kt_survey.zip`.
2. Log into [Overleaf](https://www.overleaf.com/).
3. Click **New Project** $\to$ **Upload Project** $\to$ select `kt_survey.zip`.
4. Overleaf will automatically detect `main.tex` as the root document and compile cleanly using pdfLaTeX.

### 2. Submission Modes
In `main.tex`, line 4 controls the compilation mode:
* **Preprint Mode (Current default):** `\usepackage[preprint]{acl}` (produces non-anonymous paper with page numbers, suitable for arXiv).
* **Review Mode (Double-blind):** `\usepackage[review]{acl}` (anonymizes authors and inserts line numbers for peer review).
* **Camera-Ready Mode:** `\usepackage{acl}` (final accepted version).

---

## Local Compilation

To compile from terminal using pdfLaTeX and BibTeX:
```bash
cd papers/kt_survey
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## Generated Figures Summary

| Figure | File | Scientific Meaning |
| :--- | :--- | :--- |
| **Figure 1** | `fig1_kt_landscape.pdf` | Evolutionary taxonomy and timeline of deep knowledge tracing (1994--2024), tracing classical IRT/BKT $\to$ Recurrent DKT $\to$ Memory DKVMN $\to$ Transformers SAKT/SAINT $\to$ Context-Aware AKT $\to$ simpleKT $\to$ Dual-Process SynapseKT. |
| **Figure 2(a)** | `fig2_tradeoff_pareto.pdf` | Accuracy vs. single-step inference latency ($L=200$) with $\le 10$ ms ITS SLA boundary, showing why DKVMN fails and simpleKT/CKT dominate. |
| **Figure 2(b)** | `fig2_tradeoff_pareto.pdf` | 97.5% parameter reduction on Algebra2005 (173k items) using Rasch difficulty shrinkage. |
| **Figure 3(a)** | `fig3_interpretability_deconstructed.pdf` | Attention head decay curves ($\theta_h$) mapping to Ebbinghaus forgetting curves (working memory $\theta_1 = 1.115$ vs. retention $\theta_7 = 0.386$). |
| **Figure 3(b)** | `fig3_interpretability_deconstructed.pdf` | Psychometric 1PL difficulty construct validity ($r = -0.778$, $p < 10^{-15}$ correlation with empirical failure rate). |
| **Figure 3(c)** | `fig3_interpretability_deconstructed.pdf` | Concept-to-slot ratio ($C / N_m$), explaining why memory networks collapse on granular concepts (Statics2011). |
| **Figure 4** | `fig4_cross_dataset_radar.pdf` | Radar performance profiles across 4 distinct educational regimes. |
| **Figure 5** | `fig5_synapsekt_architecture.pdf` | Dual-process cognitive architecture (multi-scale causal convolutions + adaptive monotonic attention + 2PL IRT + confidence gating). |
