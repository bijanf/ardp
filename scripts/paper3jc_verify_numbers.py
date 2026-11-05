"""Check the numbers printed in the Paper 3 Journal of Climate manuscript against the stored outputs.

Usage (run from the repository root, the worktree that holds PAPER_3_v2/):
    python3 scripts/paper3jc_verify_numbers.py <path/to/Main.tex> [--manifest PATH] [--analysis DIR]

The manuscript is not stored in this repository (it lives in the paper folder), so its path is always
passed as the first argument. The default manifest is PAPER_3_v2/analysis/number_manifest.json and the
default analysis directory is PAPER_3_v2/analysis; both are relative to the repository root.

Each manifest entry names a regular expression that locates a printed number in the text. The text is
normalised first: comment lines are dropped, the math delimiters and the braces of \\degS{} style tokens
are removed, and the backslashes of \\Fovs, \\Delta, \\mSvyr and \\% are removed, so that a pattern written
with plain characters matches. Group `group` (1-based) of match `occurrence` (0-based) is the printed
value. The value is recomputed from the JSON file and key path, transformed by `op`, `scale` and `sign`,
rounded to `decimals`, and compared with the printed value. Entries print PASS or FAIL with both values.
The script exits with status 1 when any entry fails and with status 2 when the manuscript cannot be read.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPLACEMENTS = [
    ("\\(", ""),
    ("\\)", ""),
    ("$", ""),
    ("{}", ""),
    ("\\,", " "),
    ("~", " "),
    ("\\%", "%"),
    ("\\mSvyr", "mSvyr"),
    ("\\Fovs", "Fovs"),
    ("\\Delta", "Delta"),
    ("\\degS", "degS"),
    ("\\degN", "degN"),
    ("\\Sup", "Sup"),
    ("\\Sdn", "Sdn"),
]


def normalise(tex):
    lines = []
    for line in tex.splitlines():
        # drop LaTeX comments, keeping escaped percent signs
        cleaned = re.sub(r"(?<!\\)%.*$", "", line)
        lines.append(cleaned)
    text = " ".join(lines)
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text)


def resolve(obj, path):
    """Follow a key path. Keys may contain dots (for example 2.5_97.5), so each dict level
    tries its longest matching key first. A list index [i] may follow a key or start a path."""
    if path == "":
        return obj
    if path.startswith("["):
        m = re.match(r"\[(\d+)\]", path)
        return resolve(obj[int(m.group(1))], path[m.end():].lstrip("."))
    if isinstance(obj, dict):
        for key in sorted(obj, key=len, reverse=True):
            if path == key:
                return obj[key]
            if path.startswith(key + "."):
                try:
                    return resolve(obj[key], path[len(key) + 1:])
                except KeyError:
                    continue
            if path.startswith(key + "["):
                try:
                    return resolve(obj[key], path[len(key):])
                except (KeyError, IndexError):
                    continue
    raise KeyError(f"key path '{path}' not found")


def parse_printed(text):
    cleaned = text.replace("−", "-").replace("+", "").replace("%", "").strip()
    return float(cleaned)


class Sources:
    def __init__(self, analysis):
        self.dir = Path(analysis)
        self.cache = {}

    def load(self, name):
        if name not in self.cache:
            self.cache[name] = json.loads((self.dir / name).read_text())
        return self.cache[name]


def compute(entry, src):
    op = entry.get("op", "value")
    scale = float(entry.get("scale", 1.0))
    sign = float(entry.get("sign", 1))
    if op == "value":
        return sign * scale * float(resolve(src.load(entry["file"]), entry["key"]))
    if op == "scale_pct":
        return sign * 100.0 * float(resolve(src.load(entry["file"]), entry["key"]))
    if op == "ratio_pct_more_negative":
        a = float(resolve(src.load(entry["file"]), entry["key"]))
        ref_file, ref_key = entry["key2"].split("::")
        b = float(resolve(src.load(ref_file), ref_key))
        return sign * 100.0 * (a / b - 1.0)
    if op in ("scan_min_pct", "scan_max_pct"):
        rows = resolve(src.load(entry["file"]), entry["key"])
        shares = [100.0 * float(r["share_T"]) for r in rows if r.get("p_total") is not None and r["p_total"] < 0.05]
        return min(shares) if op == "scan_min_pct" else max(shares)
    if op == "median_field":
        rows = resolve(src.load(entry["file"]), entry["key"])
        vals = [float(r[entry["field"]]) for r in rows if r.get(entry["field"]) is not None]
        return sign * scale * statistics.median(vals)
    if op == "median_field_minus_key":
        data = src.load(entry["file"])
        rows = data["per_model"]
        vals = [float(r[entry["field"]]) for r in rows if r.get(entry["field"]) is not None]
        ref = float(resolve(data, entry["key"]))
        return sign * scale * (statistics.median(vals) - ref)
    raise ValueError(f"unknown op {op}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manuscript", help="path to Main.tex (not stored in the repository)")
    ap.add_argument("--manifest", default="PAPER_3_v2/analysis/number_manifest.json")
    ap.add_argument("--analysis", default="PAPER_3_v2/analysis")
    args = ap.parse_args()

    try:
        tex = Path(args.manuscript).read_text()
    except OSError as err:
        print(f"cannot read manuscript: {err}", file=sys.stderr)
        return 2
    text = normalise(tex)
    manifest = json.loads(Path(args.manifest).read_text())
    src = Sources(args.analysis)

    n_fail = 0
    for entry in manifest["entries"]:
        eid = entry["id"]
        matches = list(re.finditer(entry["pattern"], text))
        occ = int(entry.get("occurrence", 0))
        group = int(entry.get("group", 1))
        if len(matches) <= occ:
            n_fail += 1
            print(f"FAIL {eid}: pattern not found in manuscript ({entry['pattern']})")
            continue
        printed_raw = matches[occ].group(group)
        try:
            printed = parse_printed(printed_raw)
        except ValueError:
            n_fail += 1
            print(f"FAIL {eid}: cannot read printed value '{printed_raw}'")
            continue
        try:
            source = compute(entry, src)
        except (KeyError, IndexError, OSError, ValueError, TypeError, ZeroDivisionError) as err:
            n_fail += 1
            print(f"FAIL {eid}: source error {err} ({entry['file']}:{entry.get('key', '')})")
            continue
        d = int(entry["decimals"])
        ok = round(source, d) == round(printed, d)
        tag = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        unit = entry.get("unit", "")
        print(f"{tag} {eid}: text {printed_raw} | source {round(source, d):.{d}f} {unit} "
              f"({entry['file']}:{entry.get('key', '')}{' ' + entry['op'] if entry.get('op', 'value') != 'value' else ''})")

    total = len(manifest["entries"])
    print(f"\n{total - n_fail} of {total} entries pass; {n_fail} FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
