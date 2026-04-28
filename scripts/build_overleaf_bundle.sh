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

# Copy the six combined Main-text figure PDFs (used when
# \splitfigsfalse — the submission-ready default).
for n in 1 2 3 4 5 6; do
    src="$SRC_FIG_DIR/Figure${n}.pdf"
    [[ -f "$src" ]] || { echo "missing figure: $src"; exit 1; }
    cp "$src" "$DST/figures/Figure${n}.pdf"
done

# Copy the per-panel split PDFs (used when \splitfigstrue — the
# editing-friendly mode where each subfigure is its own file).
# Panel layouts: 1: a/b/c   2: a/b/c   3: a/b   4: a/b/c
#                5: a/b/c/d 6: a/b/c/d   (Fig 6 absorbs the former Fig 7)
declare -A PANEL_LETTERS=( [1]="abc" [2]="abc" [3]="ab"
                            [4]="abc" [5]="abcd" [6]="abcd" )
for n in 1 2 3 4 5 6; do
    for letter in $(echo "${PANEL_LETTERS[$n]}" | grep -o .); do
        src="$SRC_FIG_DIR/Figure${n}${letter}.pdf"
        [[ -f "$src" ]] || { echo "missing split panel: $src"; exit 1; }
        cp "$src" "$DST/figures/Figure${n}${letter}.pdf"
    done
done

# Copy SI honest-disclosure diagnostic figures (A1 timescale,
# A2 within-class regression, A3 continuous correlation,
# A4 joint threshold + |dF| floor sensitivity).
for diag in diagA1_timescale diagA2_within_class diagA3_continuous \
            diagA4_joint_sensitivity diagA5_gap_bootstrap; do
    src="$SRC_FIG_DIR/${diag}.pdf"
    [[ -f "$src" ]] || { echo "missing SI diagnostic: $src"; exit 1; }
    cp "$src" "$DST/figures/${diag}.pdf"
done

# Note: diagSMILE.pdf and diagSMILE_amoc.pdf are NOT shipped any more --
# the SMILE figures were promoted to Main Fig 6(c,d) in v3 and the
# stand-alone SI versions were retired.

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
├── Main.tex          main manuscript (six figures, one per result)
├── SI.tex            supplementary methods + diagnostic figures
├── references.bib    cited refs across Main + SI
├── README.md         this file
└── figures/
    ├── Figure1.pdf            combined PDF (used by default)
    ├── Figure1{a,b,c}.pdf     split panels (used when \splitfigstrue)
    ├── Figure2.pdf            and Figure2{a,b,c}.pdf
    ├── Figure3.pdf            and Figure3{a,b}.pdf
    ├── Figure4.pdf            and Figure4{a,b,c}.pdf
    ├── Figure5.pdf            and Figure5{a,b,c,d}.pdf
    └── Figure6.pdf            and Figure6{a,b,c,d}.pdf
                                 (lead-lag + emergent constraint
                                  + SMILE class + SMILE AMOC)
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
EOF

# Build a flat zip alongside the folder for one-click Overleaf upload.
# `cd` INTO the bundle so the archive entries are Main.tex, SI.tex,
# figures/, etc. at top level — no paper2_overleaf/ parent directory
# inside the archive. That matches Overleaf's "New Project → Upload"
# expectation and avoids post-upload reorganisation.
rm -f "$ZIP"
( cd "$DST" && zip -qr "$ZIP" . )

echo "wrote: $DST"
echo "wrote: $ZIP"

# Optional sanity-compile the bundle in a scratch dir so we fail loud
# if Overleaf would also fail. We test BOTH figure modes — combined
# (\splitfigsfalse) and split (\splitfigstrue) — so a missing
# split-panel PDF cannot slip through unnoticed. Skip with --no-test.
if [[ "${1:-}" != "--no-test" ]]; then
    TMP="$(mktemp -d)"
    cp -r "$DST" "$TMP/test"
    pushd "$TMP/test" >/dev/null
    for mode_flag in "splitfigsfalse" "splitfigstrue"; do
        # Force the mode at the top of Main.tex.
        sed -i "s|\\\\splitfigs[a-z]*|\\\\${mode_flag}|" Main.tex
        rm -f Main.aux Main.bbl Main.log Main.toc Main.out
        pdflatex -interaction=nonstopmode Main.tex >/dev/null
        bibtex Main >/dev/null
        pdflatex -interaction=nonstopmode Main.tex >/dev/null
        pdflatex -interaction=nonstopmode Main.tex >/dev/null
        if grep -qE "Warning.*[Cc]itation.*undefined|Warning.*[Rr]eference.*undefined|!" Main.log; then
            echo "SANITY-CHECK FAILED in mode '$mode_flag':"
            grep -E "Warning.*[Cc]itation.*undefined|Warning.*[Rr]eference.*undefined|!" Main.log | head -10
            popd >/dev/null
            exit 2
        fi
    done
    # Restore default (combined) for the bundled file the user uploads.
    sed -i 's|\\splitfigstrue|\\splitfigsfalse|' Main.tex
    cp Main.tex "$DST/Main.tex"  # reflect the restored default
    pdflatex -interaction=nonstopmode SI.tex >/dev/null
    bibtex SI >/dev/null
    pdflatex -interaction=nonstopmode SI.tex >/dev/null
    pdflatex -interaction=nonstopmode SI.tex >/dev/null
    if grep -qE "Warning.*[Cc]itation.*undefined|Warning.*[Rr]eference.*undefined|!" SI.log; then
        echo "SANITY-CHECK FAILED for SI.tex:"
        grep -E "Warning.*[Cc]itation.*undefined|Warning.*[Rr]eference.*undefined|!" SI.log | head -10
        popd >/dev/null
        exit 2
    fi
    popd >/dev/null
    rm -rf "$TMP"
    echo "sanity-compile OK (Main.tex passes in BOTH combined and split"
    echo "                   modes; SI.tex compiles clean)"
fi
