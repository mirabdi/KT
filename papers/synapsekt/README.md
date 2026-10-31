# SynapseKT: Dual-Process Cognitive Modeling and 2PL Psychometrics for Attentive Knowledge Tracing

This directory contains the complete, self-contained, publication-ready research paper and Overleaf-compatible LaTeX project for **SynapseKT**.

## Project Layout

```text
papers/synapsekt/
├── main.tex                    # Master LaTeX manuscript (Overleaf entrypoint)
├── main.pdf                    # Compiled publication-ready PDF (ACL style)
├── custom.bib                  # BibTeX references for knowledge tracing and cognitive modeling
├── acl.sty                     # Official *ACL conference style file
├── acl_natbib.bst              # ACL Natbib citation bibliography style file
├── images/                     # Vector PDF & raster PNG publication figures
│   ├── fig5_synapsekt_architecture.pdf & .png # Core SynapseKT Dual-Process Architecture
│   ├── fig4_cross_dataset_radar.pdf & .png    # Cross-Regime Generalization Radar Profiles
│   ├── fig2_tradeoff_pareto.pdf & .png        # Latency vs Accuracy & Parameter Scaling
│   └── fig1_kt_landscape.pdf & .png           # Evolutionary Landscape of Knowledge Tracing
└── README.md                   # Project documentation
```

---

## Overleaf Compatibility & Compilation

### Option 1: Direct Overleaf Import
Upload `papers/synapsekt.zip` directly to Overleaf via **New Project** $\to$ **Upload Project**. Overleaf will detect `main.tex` and compile cleanly with pdfLaTeX and BibTeX.

### Option 2: Local Compilation
```bash
cd papers/synapsekt
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## Paper Highlights

1. **Dual-Process Cognitive Modeling:** Concurrent execution of Stream A (working memory via multi-scale causal depthwise 1D convolutions) and Stream B (long-term associative store via monotonic attention with repetition-damped adaptive Ebbinghaus consolidation decay).
2. **2PL-Rasch Item Psychometrics:** Joint parameterization of scalar item difficulty $\mu_q$ and bounded item discrimination $\alpha_q = 2\sigma(\nu_q)$ with empirical Bayes $L_2$ shrinkage.
3. **Dynamic Confidence Gating:** Feature-wise soft multiplexer balancing immediate momentum and curriculum recall.
4. **Empirical Superiority:** Outperforms 19 competitive baselines across all 6 standard educational benchmarks (including a +5.28% AUC leap on ASSISTments2017).
