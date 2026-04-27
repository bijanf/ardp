#!/usr/bin/env bash
# Rebuild the self-contained Overleaf bundle under paper2_overleaf/
# from the canonical sources under paper2/ + figures/paper2/.
#
# Usage:  bash scripts/build_overleaf_bundle.sh
# Result: paper2_overleaf/ ready to drag-and-drop into Overleaf,
#         and a sibling paper2_overleaf.zip for one-click upload.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_TEX_DIR="$REPO_ROOT/paper2"
SRC_FIG_DIR="$REPO_ROOT/figures/paper2"
DST="$REPO_ROOT/paper2_overleaf"
ZIP="$REPO_ROOT/paper2_overleaf.zip"

# Hard-fail early if the canonical sources are missing.
for f in "$SRC_TEX_DIR/Main.tex" "$SRC_TEX_DIR/SI.tex" \
         "$SRC_TEX_DIR/references.bib"; do
    [[ -f "$f" ]] || { echo "missing source: $f"; exit 1; }
done

rm -rf "$DST"
mkdir -p "$DST/figures"

# Copy tex + bib.
cp "$SRC_TEX_DIR/Main.tex"        "$DST/Main.tex"
cp "$SRC_TEX_DIR/SI.tex"          "$DST/SI.tex"
cp "$SRC_TEX_DIR/references.bib"  "$DST/references.bib"

# Adjust the \graphicspath so figures resolve relative to the bundle
# root rather than the parent repo's figures/paper2/ directory.
sed -i 's|{\.\./figures/paper2/}|{./figures/}|' "$DST/Main.tex" "$DST/SI.tex"

# Copy the six combined Main-text figure PDFs.
for n in 1 2 3 4 5 6; do
    src="$SRC_FIG_DIR/Figure${n}.pdf"
    [[ -f "$src" ]] || { echo "missing figure: $src"; exit 1; }
    cp "$src" "$DST/figures/Figure${n}.pdf"
done

# README — full canonical version is written inline so the bundle is
# fully reproducible from this script alone.
cat >"$DST/README.md" <<'EOF'
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
EOF

# Build a zip alongside the folder for one-click Overleaf upload.
rm -f "$ZIP"
( cd "$REPO_ROOT" && zip -qr "$(basename "$ZIP")" "$(basename "$DST")" )

echo "wrote: $DST"
echo "wrote: $ZIP"

# Optional sanity-compile the bundle in a scratch dir so we fail loud
# if Overleaf would also fail. Skip with --no-test.
if [[ "${1:-}" != "--no-test" ]]; then
    TMP="$(mktemp -d)"
    cp -r "$DST" "$TMP/test"
    pushd "$TMP/test" >/dev/null
    pdflatex -interaction=nonstopmode Main.tex >/dev/null
    bibtex Main >/dev/null
    pdflatex -interaction=nonstopmode Main.tex >/dev/null
    pdflatex -interaction=nonstopmode Main.tex >/dev/null
    pdflatex -interaction=nonstopmode SI.tex >/dev/null
    bibtex SI >/dev/null
    pdflatex -interaction=nonstopmode SI.tex >/dev/null
    pdflatex -interaction=nonstopmode SI.tex >/dev/null
    if grep -qE "Warning.*[Uu]ndefined|! " Main.log SI.log; then
        echo "SANITY-CHECK FAILED: undefined refs or LaTeX errors found"
        grep -E "Warning.*[Uu]ndefined|! " Main.log SI.log | head -10
        popd >/dev/null
        exit 2
    fi
    popd >/dev/null
    rm -rf "$TMP"
    echo "sanity-compile OK ($DST/Main.tex and SI.tex compile clean)"
fi
