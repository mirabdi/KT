# *ACL Paper Template

This directory provides the official LaTeX template and formatting directions for papers submitted to *ACL conferences (ACL, EMNLP, NAACL, EACL, etc.).

## Directory Structure

```
papers/template/
├── formatting.md       # Full formatting guidelines and conference specifications
├── anthology.bib.txt   # Instructions for downloading and using the ACL Anthology bib file
├── cf_compare.png      # Sample image for figure demonstrations
├── sm_compare.png      # Sample image for multi-panel figure demonstrations
└── latex/
    ├── acl_latex.tex    # Main LaTeX template file with author directions
    ├── acl_lualatex.tex # Alternative template for LuaLaTeX and XeLaTeX
    ├── acl.sty          # ACL LaTeX style file
    ├── acl_natbib.bst   # ACL Natbib BibTeX style file
    ├── custom.bib       # Example BibTeX file for custom references
    ├── cf_compare.png   # Local copy of sample image
    └── sm_compare.png   # Local copy of sample image
```

## How to Use This Template

1. Open `latex/acl_latex.tex` in your LaTeX editor (or import the `latex/` folder into Overleaf).
2. Configure your submission mode in the preamble:
   - **Review mode (default):** `\usepackage[review]{acl}` (adds line numbers and anonymizes for double-blind review).
   - **Camera-ready version:** `\usepackage{acl}` (removes line numbers, adds author affiliations).
   - **Preprint version:** `\usepackage[preprint]{acl}` (retains page numbers for non-anonymous arXiv posting).
3. Replace the template guide text with your paper's content:
   - Fill in `\title{...}` and author information using one of the commented author blocks.
   - Write your abstract (under 200 words).
   - Add your paper sections, mathematical formulations, tables, and figures.
   - Add custom citations to `latex/custom.bib`.
4. Consult `formatting.md` for complete rules on paper lengths, fonts, margins, anonymity, and sections.

## Compilation

To compile from the terminal using pdfLaTeX:
```bash
cd latex
pdflatex acl_latex.tex
bibtex acl_latex
pdflatex acl_latex.tex
pdflatex acl_latex.tex
```

For non-Latin scripts (e.g., Arabic, Devanagari), use `latex/acl_lualatex.tex` compiled with `lualatex` or `xelatex`.
