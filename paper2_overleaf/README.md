# Paper 2 — Overleaf-ready bundle

Self-contained source for the Communications Earth & Environment
submission. Drop this folder (or its zip) into Overleaf as a new
project; pdflatex + bibtex + pdflatex × 2 will compile both `Main.tex`
and `SI.tex` without further setup.

## Contents

```
paper2_overleaf/
├── Main.tex          main manuscript (15 pages, 6 figures)
├── SI.tex            supplementary methods (7 pages, no figures)
├── references.bib    34 cited refs across Main + SI
├── README.md         this file
└── figures/
    ├── Figure1.pdf            combined PDF (used by default)
    ├── Figure1{a,b,c}.pdf     split panels (used when \splitfigstrue)
    ├── Figure2.pdf            and Figure2{a,b,c}.pdf
    ├── Figure3.pdf            and Figure3{a,b}.pdf
    ├── Figure4.pdf            and Figure4{a,b,c}.pdf
    ├── Figure5.pdf            and Figure5{a,b,c,d}.pdf
    └── Figure6.pdf            and Figure6{a,b}.pdf
```

## Two figure-layout modes

Main.tex has a single-line toggle near the top of the preamble:

```latex
\newif\ifsplitfigs
\splitfigsfalse   % flip to \splitfigstrue for split-panel mode
```

| Mode | What it does | When to use |
|------|--------------|-------------|
| `\splitfigsfalse` (default) | Each `\begin{figure}` includes one combined `FigureN.pdf`. Layout is baked into the matplotlib output. | **Submission.** Cleanest visual result, matches what the journal will print. |
| `\splitfigstrue` | Each `\begin{figure}` glues together `FigureNa.pdf`, `FigureNb.pdf`, … via `\begin{subfigure}` blocks. | **Editing.** Swap a single panel without re-rendering the others; reviewer comments map directly to file names. |

Both modes compile out-of-the-box on Overleaf — all required PDFs ship in the bundle.

## Build

```
pdflatex Main.tex
bibtex   Main
pdflatex Main.tex
pdflatex Main.tex
```

Same four commands for `SI.tex`. Overleaf does this automatically on
recompile if `Main.tex` is set as the main document.

## Regenerate this bundle

This folder is a snapshot of the live source under `paper2/` and
`figures/paper2/` in the parent repo. To rebuild:

```
bash scripts/build_overleaf_bundle.sh
```

That script (1) copies the canonical tex + bib, (2) rewrites
`\graphicspath{{../figures/paper2/}}` to `\graphicspath{{./figures/}}`
so figures resolve relative to the bundle root, and (3) copies just
the six combined Main figures into `figures/`.
