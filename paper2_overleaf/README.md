# Paper 2 — Overleaf-ready bundle

Self-contained source for the Communications Earth & Environment
submission. Drop this folder (or its zip) into Overleaf as a new
project; pdflatex + bibtex + pdflatex × 2 will compile both `Main.tex`
and `SI.tex` without further setup.

## Contents

```
paper2_overleaf/
├── Main.tex          main manuscript (15 pages, 6 figures)
├── SI.tex            supplementary methods (6 pages, no figures)
├── references.bib    34 cited refs across Main + SI
├── README.md         this file
└── figures/
    ├── Figure1.pdf   timeseries + decomposition + depth profiles
    ├── Figure2.pdf   tiebreaker + AMOC trajectories + boxplot
    ├── Figure3.pdf   bistable-only subset
    ├── Figure4.pdf   post-Argo + period sensitivity + S/N
    ├── Figure5.pdf   2x2 zonal Δv / ΔS
    └── Figure6.pdf   lead-lag + emergent regression
```

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
