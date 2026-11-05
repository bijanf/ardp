"""Write the generated Supplement tables of the Journal of Climate revision from the analysis JSON files.

usage (from the repository root): python3 scripts/paper3jc_si_tables.py

Reads the JSON files of PAPER_3_v2/analysis and writes one file per table into
PAPER_3_v2/tables, each holding one booktabs table under its fixed label. The paper
folder has its own copy (tools/jc_si_tables.py) that writes into the manuscript folder
jclim/ and reads the order of first citation from Main.tex; here the order is not read.
Every number is read from a JSON key; no value is typed by hand.

Verifier corrections (analyses_2026-11-04.md, section 1) are applied where the
JSON files are superseded; the table notes state them.
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "PAPER_3_v2" / "analysis"
JCLIM = REPO / "PAPER_3_v2" / "tables"
MAIN = JCLIM / "Main.tex"  # not present in the repository; the label order is then the default order
OUTFILES = {
    "si_size": JCLIM / "si_size.tex",
    "si_steps": JCLIM / "si_steps.tex",
    "si_en4_timing": JCLIM / "si_en4_timing.tex",
    "si_obs_compare": JCLIM / "si_obs_compare.tex",
    "si_cmip6_relation": JCLIM / "si_cmip6_relation.tex",
    "si_frozen": JCLIM / "si_frozen.tex",
    "si_misc": JCLIM / "si_misc.tex",
    "si_data": JCLIM / "si_data.tex",
    "si_models": JCLIM / "si_models.tex",
}

MISSING = []


def flatten(obj, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    else:
        out[prefix] = obj
    return out


class Data:
    def __init__(self, name):
        self.name = name
        with open(ANALYSIS / f"{name}.json") as f:
            self.raw = json.load(f)
        self.flat = flatten(self.raw)

    def v(self, key):
        if key in self.flat:
            return self.flat[key]
        MISSING.append(f"{self.name}:{key}")
        return None

    def grab(self, base):
        """Return value, interval, p, n_eff and floor flag of a stored estimate."""
        def pick(names):
            for n in names:
                k = f"{base}.{n}"
                if k in self.flat:
                    return self.flat[k]
            return None

        out = {
            "value": pick(["value", "slope_per_decade", "value_per_decade", "trend_per_decade",
                           "trend_mSv_per_yr", "change_of_slope", "level_shift_PSU"]),
            "ci": pick(["ci", "ci_per_decade", "ci95", "ci95_per_decade", "ci_95_per_decade", "ci_PSU",
                        "ci_bias_corrected", "ci_analytic", "trend_ci_mSv_per_yr", "trend_ci95_per_decade"]),
            "p": pick(["p", "p_value", "p_santer", "p_trend"]),
            "n_eff": pick(["n_eff"]),
            "floor": pick(["n_eff_at_floor"]),
        }
        if out["value"] is None:
            MISSING.append(f"{self.name}:{base} (estimate)")
        return out

    def matches(self, regex):
        pat = re.compile(regex)
        return [(k, v) for k, v in self.flat.items() if pat.search(k)]


def num(x, nd=2):
    if x is None:
        return "n.a."
    return f"${x:.{nd}f}$"


def ci(c, nd=2):
    if c is None or not isinstance(c, (list, tuple)):
        return "n.a."
    return f"$[{c[0]:.{nd}f}, {c[1]:.{nd}f}]$"


def pval(p):
    """p value; 'n.s.' with the value in brackets when p is not below 0.05 (notation rule A94)."""
    if p is None:
        return "n.a."
    if p < 0.001:
        return "$<0.001$"
    if p >= 0.05:
        return f"n.s.\\ (${p:.3f}$)"
    return f"${p:.3f}$"


DAG = "\\(^\\dagger\\)"


def sci(x):
    """Scientific notation for a small p value, as LaTeX math."""
    mant, expo = f"{x:.1e}".split("e")
    return f"${mant}\\times10^{{{int(expo)}}}$"


def dagger_est(d, base, nd=3):
    """Value, interval, p and N_eff cells; a dagger marks an N_eff at its floor of 3 (interval uninformative)."""
    g = d.grab(base)
    if g["value"] is None:
        return ["n.a.", "n.a.", "n.a.", "n.a."]
    floor = g["floor"] is True or (g["n_eff"] is not None and abs(g["n_eff"] - 3.0) < 1e-9)
    ci_txt = ci(g["ci"], nd) if g["ci"] is not None else "n.a."
    neff_txt = neff(g["n_eff"]) + (DAG if floor else "")
    p_txt = "n.e." if floor else pval(g["p"])
    if floor:
        ci_txt = "uninformative" + DAG
    return [num(g["value"], nd), ci_txt, p_txt, neff_txt]


def neff(n):
    if n is None:
        return "n.a."
    return f"{n:.1f}"


def est_row(d, base, nd=3, pnd=None):
    """Value, interval, p and N_eff cells from a stored estimate; n.e. at the floor."""
    g = d.grab(base)
    if g["floor"] is True:
        return [num(g["value"], nd), "n.e.", "n.e.", "n.e."]
    return [num(g["value"], nd), ci(g["ci"], nd), pval(g["p"]), neff(g["n_eff"])]


def tex_row(cells):
    return " & ".join(cells) + " \\\\"


def table_block(label, caption, header, rows, note, colspec, stretch=None):
    lines = ["\\begin{table}[h]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3pt}",
             *([f"\\renewcommand{{\\arraystretch}}{{{stretch}}}"] if stretch else []),
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             header + " \\\\", "\\midrule"]
    ncols = len(colspec)
    for r in rows:
        if isinstance(r, str):
            lines.append(r)
        else:
            lines.append(tex_row(list(r) + [""] * (ncols - len(r))))
    lines += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip",
              f"\\begin{{minipage}}{{0.95\\linewidth}}\\footnotesize {note}\\end{{minipage}}",
              "\\end{table}", ""]
    return "\n".join(lines)


def group(title, ncol):
    return f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{title}}}}} \\\\"


# ---------------------------------------------------------------- EN4 timing
def table_en4_timing():
    d = Data("jc_n01_en4_window")
    rows = []
    start_base = "series.oras5_full_column.dS.start_year_2005_to_2010_end_2024"
    rows.append(group("(a) Start year, end 2024, ORAS5 circulation, full column", 6))
    starts = sorted({int(m.group(1)) for k, _ in d.matches(
        r"^" + re.escape(start_base) + r"\.(\d{4})\.slope_per_decade$") for m in [re.search(r"\.(\d{4})\.slope", k)]})
    for y in starts:
        base = f"{start_base}.{y}"
        rows.append([f"{y}--2024", "full column"] + est_row(d, base))
    end_base = "series.oras5_full_column.dS.end_year_2015_to_2024_start_2005"
    ends = sorted({int(m.group(1)) for k, _ in d.matches(
        r"^" + re.escape(end_base) + r"\.(\d{4})\.slope_per_decade$") for m in [re.search(r"\.(\d{4})\.slope", k)]})
    rows.append(group("(b) End year, start 2005, ORAS5 circulation, full column", 6))
    for y in ends:
        base = f"{end_base}.{y}"
        rows.append([f"2005--{y}", "full column"] + est_row(d, base))
    rows.append(group("(c) Windows, layers, hinge, split and covariate", 6))
    rows.append(["2005--2024", "full column"] + est_row(d, "series.oras5_full_column.dS.full_2005_2024"))
    rows.append(["2005--2024", "upper 2000 m"] + est_row(d, "series.oras5_upper_2000m.dS.full_2005_2024"))
    rows.append(["2005--2017", "full column"] + est_row(d, "series.oras5_full_column.dS.window_2005_2017"))
    rows.append(["2018--2024", "full column"] + est_row(d, "series.oras5_full_column.dS.window_2018_2024"))
    rows.append(["Change of slope at 2017", "full column"] + est_row(
        d, "series.oras5_full_column.dS.hinge_2017.change_of_slope"))
    rows.append(["Slope after 2017", "full column"] + est_row(
        d, "series.oras5_full_column.dS.hinge_2017.slope_post_2017_within_model"))
    rows.append(["Post minus pre split at 2017", "full column"] + est_row(
        d, "series.oras5_full_column.dS.subperiod_split_2017.difference_post_minus_pre"))
    rows.append(["Trend with both limb weights", "full column"] + est_row(
        d, "series.oras5_full_column.dS.covariate.time_plus_both_weights"))
    best = d.v("series.oras5_full_column.dS.step_scan_2012_2020.best_knot")
    best_base = f"series.oras5_full_column.dS.step_scan_2012_2020.per_knot.{best}"
    rows.append([f"Step at {best}", "full column"] + est_row(d, best_base))
    ps = d.v("series.oras5_full_column.dS.step_scan_2012_2020.p_sup")
    tsup = d.v("series.oras5_full_column.dS.step_scan_2012_2020.sup_abs_t_obs")
    coef = d.grab("series.oras5_full_column.dS.covariate.time_plus_both_weights")
    cn = d.v("series.oras5_full_column.dS.covariate.time_plus_both_weights.coef_north_weight")
    cnp = d.v("series.oras5_full_column.dS.covariate.time_plus_both_weights.p_north_weight")
    cs = d.v("series.oras5_full_column.dS.covariate.time_plus_both_weights.coef_south_weight")
    csp = d.v("series.oras5_full_column.dS.covariate.time_plus_both_weights.p_south_weight")
    # supremum over the knots 2012 to 2020: default (bias-corrected) and fixed-phi sensitivity; p values only
    sup_base = "series.oras5_full_column.dS"
    for lab, path in [("Supremum over knots 2012--2020, step (default)", f"{sup_base}.step_scan_2012_2020"),
                      ("Supremum over knots 2012--2020, hinge (default)", f"{sup_base}.hinge_scan_2012_2020"),
                      ("Supremum over knots 2012--2020, step (fixed \\(\\phi\\))",
                       f"{sup_base}.fixed_phi_sensitivity.step_scan_2012_2020"),
                      ("Supremum over knots 2012--2020, hinge (fixed \\(\\phi\\))",
                       f"{sup_base}.fixed_phi_sensitivity.hinge_scan_2012_2020")]:
        st = d.v(f"{path}.sup_abs_t_obs")
        sp = d.v(f"{path}.p_sup")
        rows.append([lab + " (sup-\\(|t|\\), \\(p\\))", "full column",
                     f"{st:.2f}" if st is not None else "n.a.", "n.a.", pval(sp), "n.a."])
    note = ("Trends and steps in PSU per decade (step in PSU); the start-year and end-year blocks are "
            "the least-squares trend of annual values with the lag-one coefficient bias-corrected "
            "(Methods). "
            "Step scan over 2012--2020: the supremum-|t| statistic is "
            f"{num(tsup, 2)} with supremum p {num(ps, 3)} (10{{,}}000 simulations), so no step year is "
            "resolved. The covariate row adds the observation weights of both limbs to the time trend; "
            f"the northward-weight coefficient is {num(cn, 2)} (p {num(cnp, 2)}) and the southward-weight "
            f"coefficient is {num(cs, 3)} (p {num(csp, 2)}). The supremum rows give the supremum of the "
            "absolute \\(t\\) statistic over the knots and its simulated p value, with the default "
            "lag-one coefficient and with the coefficient fixed at the value of the full record (sensitivity).")
    return table_block(
        "tab:si_en4_timing",
        "Trends of the EN4 limb salinity contrast \\(\\Delta S\\) over 2005--2024 by start year, end year and window, with the supplementary tests of the trend.",
        "Window & Layer & Estimate & 95\\% interval & \\(p\\) & \\(N_{\\mathrm{eff}}\\)",
        rows, note, "llrrrr"), d


# --------------------------------------------------------------- steps
def scan_row(data_raw, var, year):
    for entry in data_raw["a_step_scans"]["oras5"]["variables"][var]["scan"]:
        if entry["step_year"] == year:
            return entry
    MISSING.append(f"n03 scan {var} {year}")
    return None


def family_best(data_raw, var, key_part):
    fam = data_raw["a_step_scans"]["oras5"]["variables"][var]["families"]
    for k, v in fam.items():
        if key_part in k and " and " not in k:
            return k, v
    MISSING.append(f"n03 family {var} {key_part}")
    return None, None


def table_steps():
    d = Data("jc_n03_steps")
    n15 = Data("jc_n15_oras5_early")
    n23 = Data("jc_n23_argo_era")
    raw = d.raw
    rows = []
    rows.append(group("(a) ORAS5 single-step scans over 1995--2015", 8))
    fk, fv = family_best(raw, "T", "1995-2015")
    if fv is not None:
        e = scan_row(raw, "T", fv["best_break_year"])
        rows.append(["Northward-limb transport \\(T\\), best break", f"{fv['best_break_year']}",
                     "Sv", num(e["step"]), ci(e["step_ci"]), pval(e["step_p"]), pval(fv["p_sup"]),
                     num(e["trend"], 2) + " Sv/dec"])
    fk, fv = family_best(raw, "F", "1995-2015")
    if fv is not None:
        e = scan_row(raw, "F", fv["best_break_year"])
        rows.append(["\\(\\Fovs\\), best break", f"{fv['best_break_year']}", "mSv/yr",
                     num(e["step"] * 1000.0, 1), ci([x * 1000.0 for x in e["step_ci"]], 1),
                     pval(e["step_p"]), pval(fv["p_sup"]), num(e["trend"], 2) + " mSv/yr"])
    rows.append(group("(b) ORAS5 joint model with steps at 2004 and 2015 (1958--2025)", 8))
    jb = "b_oras5_joint_2004_2015.variables"
    fu = d.grab(f"{jb}.F.unstepped_trend_1958_2025")
    rows.append(["\\(\\Fovs\\) trend, no step", "1958--2025", "mSv/yr", "", "", "",
                 "", f"{num(fu['value'])} {ci(fu['ci'])}"])
    fj = d.grab(f"{jb}.F.joint_trend")
    rows.append(["\\(\\Fovs\\) trend, two steps", "1958--2025", "mSv/yr", "", "", "",
                 "", f"{num(fj['value'])} {ci(fj['ci'])}"])
    for step_key, yr in [("joint_step_2004", 2004), ("joint_step_2015", 2015)]:
        g = d.grab(f"{jb}.F.{step_key}")
        rows.append(["\\(\\Fovs\\) step", str(yr), "Sv", num(g["value"], 3), ci(g["ci"], 3),
                     pval(g["p"]), "", ""])
    tj = d.grab(f"{jb}.T.joint_trend")
    rows.append(["\\(T\\) trend, two steps", "1958--2025", "Sv/dec", "", "", "", "",
                 f"{num(tj['value'])} {ci(tj['ci'])}"])
    for step_key, yr in [("joint_step_2004", 2004), ("joint_step_2015", 2015)]:
        g = d.grab(f"{jb}.T.{step_key}")
        rows.append(["\\(T\\) step", str(yr), "Sv", num(g["value"]), ci(g["ci"]), pval(g["p"]), "", ""])
    g = d.grab(f"{jb}.dS.joint_step_2015")
    rows.append(["\\(\\Delta S\\) step", "2015", "PSU", num(g["value"], 3), ci(g["ci"], 3),
                 pval(g["p"]), "", ""])
    rows.append(group("(c) Break at 1979 in the ORAS5 freshwater transport (two-line fit and scan)", 8))
    g = n15.grab("two_line_jump_1979.jump_at_1979_mSv")
    rows.append(["\\(\\Fovs\\) jump, two-line fit, fixed year", "1979", "mSv", num(g["value"], 1), ci(g["ci"], 1),
                 pval(g["p"]), "n.a.", ""])
    # One run throughout: the 4,000-draw supremum scan stored in break_scan (the run quoted in Text S3).
    hs_year = n15.v("break_scan.sup_F.argmax_break_year")
    hs_stat = n15.v("break_scan.sup_F.statistic")
    hs_p = n15.v("break_scan.sup_F.p_sup")
    rows.append(["\\(\\Fovs\\) sup-\\(F\\), hinge and jump, 1962--1996", f"{hs_year}", "F", num(hs_stat, 2),
                 "n.a.", pval(hs_p), "n.a.", ""])
    rows.append(group("(d) GLORYS12V1 step at 2004 (1993--2025), Holm adjustment", 8))
    gb = "c_glorys12_2004_holm.fits"
    for var, lab, unit, nd in [("F", "\\(\\Fovs\\) step", "Sv", 3), ("T", "\\(T\\) step", "Sv", 2),
                               ("dS", "\\(\\Delta S\\) step", "PSU", 3)]:
        g = d.grab(f"{gb}.{var}.step")
        h = d.v(f"{gb}.{var}.holm.holm_family_A")
        rows.append([lab, "2004", unit, num(g["value"], nd), ci(g["ci"], nd), pval(g["p"]), pval(h), ""])
    rows.append(group("(e) Argo era (2004--2025) and the sensitivity to the lag-one correction", 8))
    for prod, lab in [("ORAS5", "ORAS5"), ("GLORYSNVN", "GLORYS12V1")]:
        pk = prod.replace("NVN", "12")
        base = f"products.{pk if pk != 'GLORYS12' else 'GLORYS12V1'}.F_ov"
        if prod == "GLORYSNVN":
            base = "products.GLORYS12V1.F_ov"
        c = n23.grab(f"{base}.corrected")
        u = n23.grab(f"{base}.uncorrected_lag1")
        rows.append([f"\\(\\Fovs\\) trend, {lab}, bias-corrected", "2004--2025", "mSv/yr",
                     num(c["value"]), ci(c["ci"]), pval(c["p"]), "", ""])
        rows.append([f"\\(\\Fovs\\) trend, {lab}, uncorrected", "2004--2025", "mSv/yr",
                     num(u["value"]), ci(u["ci"]), pval(u["p"]), "", ""])
    sc = d.grab("d_oras5_argo_reconciliation.variables.F.stepped_2015.trend")
    rows.append(["\\(\\Fovs\\) trend, ORAS5, step at 2015 fitted", "2004--2025", "mSv/yr",
                 num(sc["value"]), ci(sc["ci"]), pval(sc["p"]), "", ""])
    # (f) fit register of the specifications of Text S3 (window, steps, estimate, interval, n_eff)
    g1 = Data("jc_r2_G01_steps_oras5")
    r1 = Data("r1_step_adjusted")
    rows.append(group("(f) Fit register of the step specifications (window; estimate and sieve or analytic "
                      "interval; \\(n_{\\mathrm{eff}}\\) in the last column)", 8))

    def reg_row(label, window, unit, base, src, nd=3, shift=None):
        g = src.grab(base)
        ci_v = g["ci"]
        return [label, window, unit, num(g["value"], nd), ci(ci_v, nd), pval(g["p"]), "",
                f"{g['n_eff']:.1f}" + (DAG if g["floor"] else "")]

    rows.append(reg_row("Unstepped: \\(\\Fovs\\) trend", "1958--2025", "mSv/yr",
                        "A11_transport_share.unstepped.F_trend", g1, 2))
    rows.append(reg_row("Single step 2015: \\(\\Fovs\\) trend", "1958--2025", "mSv/yr",
                        "A11_transport_share.2015_only.F_trend", g1, 2))
    rows.append(reg_row("Single step 2015: \\(\\Fovs\\) step", "1958--2025", "Sv",
                        "A11_transport_share.2015_only.F_step_2015_Sv", g1, 3))
    rows.append(reg_row("Joint 2004 and 2015: \\(\\Fovs\\) trend", "1958--2025", "mSv/yr",
                        "A11_transport_share.joint_2004_2015.F_trend", g1, 2))
    rows.append(reg_row("Joint 2004 and 2015: \\(\\Fovs\\) step 2004", "1958--2025", "Sv",
                        "A11_transport_share.joint_2004_2015.F_step_2004_Sv", g1, 3))
    rows.append(reg_row("Joint 2004 and 2015: \\(\\Fovs\\) step 2015", "1958--2025", "Sv",
                        "A11_transport_share.joint_2004_2015.F_step_2015_Sv", g1, 3))
    rows.append(reg_row("Joint 2004 and 2015: \\(T\\) step 2004", "1958--2025", "Sv",
                        "b_oras5_joint_2004_2015.variables.T.joint_step_2004", d, 2))
    rows.append(reg_row("Ramp 2014 to 2016: \\(\\Fovs\\) trend", "1958--2025", "mSv/yr",
                        "oras5.ramp_fit", r1, 2))
    for lab, key in [("Unstepped", "unstepped"), ("Single step 2015", "2015_only"),
                     ("Joint 2004 and 2015", "joint_2004_2015")]:
        pt = g1.v(f"A11_transport_share.{key}.share_percent_point")
        cis = g1.v(f"A11_transport_share.{key}.share_percent_ci")
        rows.append([f"{lab}: transport share of the trend (sieve)", "1958--2025", "\\%",
                     f"${pt:.1f}$", ci(cis, 1), "n.a.", "", "n.a."])
    hs_cv = n15.v("break_scan.sup_F.cv95")
    hs_run = n15.v("hinge_jump_scan_1962_1996.runner_up_year")
    hs_run_f = n15.v("hinge_jump_scan_1962_1996.runner_up_F")
    hs_nsim = n15.v("break_scan.n_sims")
    hs_ncand = n15.v("hinge_jump_scan_1962_1996.n_candidates")
    hs_res = n15.v("break_scan.sup_F.resolved")
    nsim_txt = f"{hs_nsim:,}".replace(",", "{,}")
    note = ("Steps in the units shown; intervals are 95\\% intervals and \\(p\\) values are from the "
            "bias-corrected test. The supremum columns give the p value of the scan over all admissible "
            "years (Methods); in panel (a) they are the supremum p values of the family of scans. "
            "In panel (c) the jump and the slope change of the two-line fit are conditional on the year "
            "1979, and the \\(p\\) value of the jump is a fixed-year value. The scan of panel (c) fits a "
            f"hinge and a jump at each of {hs_ncand} candidate years from 1962 to 1996, with \\(\\Fovs\\) "
            f"sup-\\(F\\) {num(hs_stat, 2)} at {hs_year} and {num(hs_run_f, 2)} at {hs_run}; its 95\\% "
            f"critical value from {nsim_txt} "
            f"simulated AR(1) series with the bias-corrected lag-one coefficient of the full-record trend "
            f"fit is {num(hs_cv, 2)}, so the break is {'resolved' if hs_res else 'not resolved'} at the 5\\% level. "
            "Panel (d) gives the Holm-adjusted p values across the three variables of the first family. "
            "Panel (f) is the fit register of the step specifications: its estimate, interval and p value are "
            "those of the fitted term, and the last column gives the effective sample size of that term; the "
            "transport-share rows give the share of the trend in per cent with its sieve interval. "
            "The Argo-era rows use the section \\(\\Fovs\\) series of ORAS5 and GLORYS12V1.")
    return table_block(
        "tab:si_steps",
        "Step and break tests for the ORAS5 transport, the \\(\\Fovs\\) record, the limb salinity contrast and the GLORYS12V1 step at 2004.",
        "Test & Year & Unit & Estimate & 95\\% interval & \\(p\\) & Adjusted \\(p\\) & Trend",
        rows, note, "lllrrrrr"), d


# --------------------------------------------------------------- size
def table_size():
    d = Data("jc_n04_ar_recalibration")
    s = "summary"
    rows = []

    def rng(key):
        v = d.v(f"{s}.{key}")
        return f"$[{v[0]:.1f}, {v[1]:.1f}]$" if v else "n.a."

    na = "n.a."
    rows.append(group("(a) AR(1) noise with the sample size and lag-one grid of the test; size and power in \\%", 6))
    rows.append(["Lag-one 0 to 0.8, \\(n\\) 22 to 68",
                 rng("ar1_phi_0_to_0p8_n22_to_68_size_corrected"),
                 rng("ar1_phi_0_to_0p8_n22_to_68_size_uncorrected"),
                 rng("ar1_phi_0_to_0p8_n22_to_68_power_0p5xMDT"),
                 rng("ar1_phi_0_to_0p8_n22_to_68_power_1xMDT")])
    rows.append(["Lag-one 0 to 0.8, \\(n\\) 11 to 20",
                 rng("ar1_phi_0_to_0p8_n11_to_20_size_corrected"), na, na, na])
    rows.append(["Lag-one 0 to 0.8, all \\(n\\)",
                 rng("ar1_phi_0_to_0p8_all_n_size_corrected"), na, na, na])
    rows.append(["Lag-one 0.9 to 0.98, \\(n=22\\)",
                 rng("ar1_phi_0p9_to_0p98_n22_size_corrected"),
                 rng("ar1_phi_0p9_to_0p98_n22_size_uncorrected"), na, na])
    rows.append(group("(b) Fitted AR(\\(p\\)) and ARMA noise per series, \\(n\\) 22 to 68 (size range over the models, \\%)", 6))
    fitted = sorted(k for k in d.flat if k.startswith(s + ".fitted_size_n22_to_68_corrected."))
    keep = ("ORAS5", "GLORYS12V1")
    for k in fitted:
        lab = k.split("corrected.", 1)[1]
        if not lab.startswith(keep):
            continue
        rows.append([lab.replace("(AIC-selected)", "(AIC)").replace("F_ovS", "$F_{\\mathrm{ovS}}$"), rng(k.replace(s + ".", "")), na, na, na])
    # (c) coverage of the sieve intervals of the transport share (paired AR(1) sieve, 10,000 resamples)
    cov = Data("jc_r2_G02_calibration_sieve")
    rows.append(group("(c) Coverage of the sieve intervals of the transport share, 95\\% nominal; columns: "
                      "coverage (\\%, MC SE), median of simulated estimates, median width, inflation factor \\(k\\), "
                      "misses below and above (\\%)", 6))
    labels_c = {"unstepped": "Unstepped, 1958--2025 (share, unitless)",
                "2015_only": "Single step 2015, 1958--2025 (share, unitless)",
                "joint_2004_2015": "Joint steps 2004 and 2015, 1958--2025 (share, unitless)"}
    for spec_key, lab in labels_c.items():
        spec =[s for s in cov.raw["specifications"] if s["case"] == spec_key][0]
        rows.append([lab, f"{100 * spec['coverage_share_interval']:.1f} ({100 * spec['coverage_mc_se']:.1f})",
                     f"{spec['median_point_estimate']:.4f}", f"{spec['median_interval_width']:.3f}",
                     f"{spec['inflation_factor_empirical_95pct']:.2f}",
                     f"{100 * spec['miss_below']:.1f} / {100 * spec['miss_above']:.1f}"])
    rows.append(group("Stylised sensitivity (not the fitted noise): innovation correlation of the joint "
                      "bootstrap fixed at 0.5", 6))
    spec = [s for s in cov.raw["specifications"] if s["case"] == "unstepped_rho_0.5_sensitivity"][0]
    rows.append(["Unstepped, correlation fixed at 0.5 (stylised; share, unitless)",
                 f"{100 * spec['coverage_share_interval']:.1f} ({100 * spec['coverage_mc_se']:.1f})",
                 f"{spec['median_point_estimate']:.4f}", f"{spec['median_interval_width']:.3f}",
                 f"{spec['inflation_factor_empirical_95pct']:.2f}",
                 f"{100 * spec['miss_below']:.1f} / {100 * spec['miss_above']:.1f}"])
    # (d) calibrated thresholds of the trend test: fixed and refitted noise (20,000 replicates per case)
    cal = Data("jc_r2_G02_calibration_bootstrap")
    rows.append(group("(d) Calibrated thresholds of the trend test at 5\\%; columns: critical \\(p\\) with noise "
                      "fixed, critical \\(p\\) with noise refitted, size with noise fixed (\\%), size with noise "
                      "refitted (\\%), headline \\(p\\) (unitless)", 6))
    for case in cal.raw["cases"]:
        fx = case["fixed_coefficients"]
        rf = case["refitted_coefficients"]
        name = {"F_ovS": "\\(\\Fovs\\)", "T": "\\(T\\)", "dS": "\\(\\Delta S\\)"}[case["variable"]]
        model = case["noise_model"]
        n_eff_txt = case["headline_n_eff"]
        rows.append([f"{name}, {model}, \\(n_{{\\mathrm{{eff}}}}={n_eff_txt:.1f}\\)",
                     f"{fx['critical_p_5pct']:.4f}", f"{rf['critical_p_5pct']:.4f}",
                     f"{100 * fx['size_at_nominal_5pct']:.1f}", f"{100 * rf['size_at_nominal_5pct']:.1f}",
                     sci(fx["headline_p"])])
    note = ("Size is the rejection rate of the trend test at the 5\\% level under noise with no trend, in "
            "per cent, from 20{,}000 replicates per case. Corrected: the lag-one coefficient is bias-corrected "
            "(the paper's test); uncorrected: the raw lag-one coefficient. Ranges are the minimum and maximum "
            "over the grid or over the models stated. Power is the rejection rate when the trend equals 0.5 or "
            "1 times the minimum detectable trend (MDT). Fitted noise models are chosen per series by the "
            "minimum AIC of the candidates on the trend residuals where marked (AIC). Panel (c): coverage is the "
            "share of the sieve bootstrap intervals (10,000 resamples per dataset, 5,000 datasets) that contain "
            "the true transport share, with its Monte Carlo standard error in brackets; the stylised row fixes the "
            "innovation correlation at 0.5 and is not a fitted noise model. The column of medians gives the median of "
            "the point estimates over the simulated datasets; the data values of the transport share are in "
            "Table~\\ref{tab:si_sensitivity}(b). Panel (d): the critical p value is the "
            "5\\% quantile of the p values under zero trend (20,000 replicates per case, seed 20261104), with the "
            "noise coefficients fixed at the stored fit or refitted in a two-stage parametric bootstrap; size is "
            "the rejection rate at the nominal 5\\% level; every headline p value lies below its calibrated "
            "threshold in both versions.")
    return table_block(
        "tab:si_size",
        "Size and power of the bias-corrected trend test under autoregressive and moving-average noise. "
        "Computed by the size-calibration script \\texttt{analysis\\_paper3jc\\_n04\\_ar\\_recalibration.py} "
        "(output \\texttt{jc\\_n04\\_ar\\_recalibration.json} in the analysis folder of the code repository), "
        "with twenty thousand parametric replicates per case and seed 20261104.",
        "Setting & Size, corrected (\\%) & Size, uncorrected (\\%) & Power, 0.5 MDT (\\%) & Power, 1 MDT (\\%)",
        rows, note, "lrrrrr"), d


# --------------------------------------------------------------- obs compare
def table_obs():
    e = Data("jc_n01_en4_window")
    r = Data("jc_n05_limb_levels_rg09")
    q = Data("jc_n02_ratio")
    m = Data("jc_n10_samba")
    rows = []
    rows.append(group("(a) EN4 salinity with the frozen circulation of each product (see note), 2005--2024 "
                      "(PSU per decade)", 5))
    for lab, base in [
        ("Full column, ORAS5 circulation", "series.oras5_full_column.dS.full_2005_2024"),
        ("Upper 2000 m, ORAS5 circulation", "series.oras5_upper_2000m.dS.full_2005_2024"),
        ("Full column, GLORYS12V1 circulation", "series.glorys12_full_column.dS.full_2005_2024"),
        ("Upper 2000 m, GLORYS12V1 circulation", "series.glorys12_upper_2000m.dS.full_2005_2024")]:
        g = e.grab(base)
        rows.append([lab, num(g["value"], 3), ci(g["ci"], 3), pval(g["p"])])
    own_g = Data("jc_r2_G05_en4_table")
    rows.append(group("(a, continued) EN4 salinity with each reanalysis's own monthly circulation (pair A; "
                      "2005--2024)", 5))
    for prod, pk, plab in [("oras5", "oras5", "ORAS5"), ("glorys12", "glorys12", "GLORYS12V1")]:
        for var, vlab in [("dS", "\\(\\Delta S\\) (PSU/dec)"), ("F_ov_Sv", "\\(\\Fovs\\) (Sv/dec)")]:
            base = f"table.{pk}.EN4 salinity, own circulation.{var}"
            rows.append([f"{plab} circulation, {vlab}"] + dagger_est(own_g, base, 3))
    rows.append(group("(b) RG09, the 2019 release of \\citet{RoemmichGilsonArgoClimatology2019}, Argo climatology against EN4 in the "
                      "upper 2000 m, 2005--2024", 5))
    for lab, base in [
        ("RG09 minus EN4, \\(\\Delta S\\), ORAS5 circulation (PSU/dec)",
         "summary.upper_2000m.ORAS5.RG09_minus_EN4.dS"),
        ("RG09 minus EN4, southern salinity, ORAS5 circulation (PSU/dec)",
         "summary.upper_2000m.ORAS5.RG09_minus_EN4.S_south"),
        ("RG09, \\(\\Delta S\\), ORAS5 circulation (PSU/dec)", "summary.upper_2000m.ORAS5.RG09.dS"),
        ("RG09, northern salinity, ORAS5 circulation (PSU/dec)", "summary.upper_2000m.ORAS5.RG09.S_north"),
        ("RG09, southern salinity, ORAS5 circulation (PSU/dec)", "summary.upper_2000m.ORAS5.RG09.S_south"),
        ("RG09, \\(F_{\\mathrm{ovS}}\\), ORAS5 circulation (Sv/dec)", "summary.upper_2000m.ORAS5.RG09.F_ov_Sv"),
        ("RG09, \\(\\Delta S\\), GLORYS12V1 circulation (PSU/dec)", "summary.upper_2000m.GLORYS12V1.RG09.dS"),
        ("RG09, northern salinity, GLORYS12V1 circulation (PSU/dec)",
         "summary.upper_2000m.GLORYS12V1.RG09.S_north"),
        ("RG09, southern salinity, GLORYS12V1 circulation (PSU/dec)",
         "summary.upper_2000m.GLORYS12V1.RG09.S_south")]:
        rows.append([lab] + dagger_est(r, base, 4))
    rows.append(group("(c) Reanalysis salinity contrast with reanalysis own salinity, and with the reanalysis "
                      "circulation fixed (PSU per decade)", 5))
    for prod, pk in [("ORAS5", "oras5"), ("GLORYS12V1", "glorys12")]:
        rows.append([f"{prod} own salinity and circulation (PSU/dec)"]
                    + dagger_est(q, f"products.{pk}.trends_per_decade.reanalysis_own_dS", 3))
        rows.append([f"{prod} salinity, reanalysis circulation fixed (PSU/dec)"]
                    + dagger_est(q, f"products.{pk}.trends_per_decade.reanalysis_fixed_dS", 3))
    # (d) split of the overturning cells against SAMBA: standard-deviation ratio, correlation, p value
    rows.append(group("(d) SAMBA at 34.5\\(^\\circ\\)S, September 2013 to July 2017; standard-deviation ratio "
                      "(unitless) with interval, correlation (unitless) and \\(p\\) value", 5))
    samba_products = ("ORAS5", "GLORYS12V1", "ECCO-V4r4")
    for lab, base in ([(f"Upper cell, {p} against SAMBA", f"pairs.{p}.up_cell__samba_up") for p in samba_products]
                      + [(f"Abyssal cell, {p} against SAMBA", f"pairs.{p}.ab_cell__samba_ab") for p in samba_products]):
        sr = m.v(base + ".std_ratio_x_over_y")
        sci = m.v(base + ".std_ratio_ci_95")
        rr = m.v(base + ".r")
        pp = m.v(base + ".p")
        rows.append([f"{lab}; correlation \\({rr:.2f}\\)", num(sr, 2),
                     ci(sci, 2) if sci else "n.a.", pval(pp)])
    # (e) row sensitivity of the EN4 section (A103): 34.0 S current row, 33.0 S row, 34.5 S interpolated row
    rs = Data("jc_r2_G06_en4rows_rowsens")
    rows.append(group("(e) EN4 row sensitivity of the limb contrast at 2005--2024 (PSU per decade); rows "
                      "at 33.0\\(^\\circ\\)S and interpolated 34.5\\(^\\circ\\)S against the 34.0\\(^\\circ\\)S "
                      "row", 5))
    for rowkey, rlab in [("34.0", "34.0\\(^\\circ\\)S (current)"), ("33.0", "33.0\\(^\\circ\\)S"),
                         ("34.5_interpolated", "34.5\\(^\\circ\\)S interpolated")]:
        for circ, clab in [("ORAS5", "ORAS5"), ("GLORYS12V1", "GLORYS12V1")]:
            for var, vlab in [("dS", "\\(\\Delta S\\)"), ("S_north", "\\(S_{\\mathrm{north}}\\)")]:
                base = f"rows.{rowkey}.circulations.{circ}.{var}"
                rows.append([f"{rlab}, {clab} circulation, {vlab}"] + dagger_est(rs, base, 4))
    # (f) EN4 over 2008 to 2024 (core-Argo proxy) with the frozen circulation of each product
    fc = Data("jc_r2_G12_obscounts_fixedcirc")
    rows.append(group("(f) EN4 over 2008--2024 with the frozen circulation (PSU per decade; Sv per decade for "
                      "\\(\\Fovs\\))", 5))
    for circ, clab in [("ORAS5", "ORAS5"), ("GLORYS12V1", "GLORYS12V1")]:
        for var, vlab in [("dS", "\\(\\Delta S\\)"), ("F_ov_Sv", "\\(\\Fovs\\) (Sv/dec)")]:
            base = (f"part_b_en4_fixed_circulation_windows.circulations.{circ}.full_column.windows."
                    f"2008-2024.{var}")
            rows.append([f"{clab} circulation, {vlab}"] + dagger_est(fc, base, 4))
    note = ("EN4 values use the salinity of EN4 with the circulation of the stated product, frozen at "
            "its record mean (Methods); pair A (own monthly circulation) and the frozen circulation give the "
            "EN4 rows of panels (a) and (f). Intervals are 95\\% intervals of the bias-corrected trend test. "
            "A dagger marks an effective sample size at its floor of 3, where the interval is uninformative and "
            "the \\(p\\) value reads n.e. (not evaluated). "
            "n.s. marks \\(p\\ge 0.05\\). Panel (c) gives no ratio of the reanalysis to EN4; the fixed-circulation "
            "ratio is not used in the text. The RG09 \\citep{RoemmichGilsonArgoClimatology2019} rows cover 2005--2024 in the "
            "upper 2000 m only; the RG09 and EN4 segments differ in longitude (RG09 70 points, EN4 74 points) "
            "and in row latitude (34.5\\(^\\circ\\)S and 34.0\\(^\\circ\\)S). In panel (d) the standard-deviation "
            "ratio is the standard deviation of the reanalysis anomalies over that of SAMBA, with an approximate F interval; "
            "the \\(p\\) values use the effective sample size of each pair.")
    return table_block(
        "tab:si_obs_compare",
        "Observational and reanalysis comparisons of the limb salinity contrast: EN4 with the own and frozen "
        "circulations (a), RG09 against EN4 (b), reanalysis contrast trends (c), overturning cells at 34.5\\(^\\circ\\)S "
        "against SAMBA for the three products (d), the EN4 row sensitivity (e) and EN4 over 2008 to 2024 (f).",
        "Quantity & Estimate & 95\\% interval & \\(p\\) & \\(N_{\\mathrm{eff}}\\)",
        rows, note, "lrrrr"), e


# --------------------------------------------------------------- cmip6
def table_cmip6():
    rel = Data("jc_n11_cmip6_relation")
    frc = Data("jc_n12_cmip6_forced")
    rows = []
    rows.append(group("(a) Cross-model correlations \\(r\\) of \\(\\Fovs\\) at 34.5\\(^\\circ\\)S with the overturning", 4))

    def claim(key, label, nd=2):
        val = rel.v(f"claims.{key}.value")
        c = rel.v(f"claims.{key}.ci95")
        if isinstance(val, list):
            return [label, f"{val[0]:.{nd}f} to {val[1]:.{nd}f}", "n.a.", "leave-one-out range"]
        return [label, num(val, nd), ci(c, nd) if c else "n.a.", "family bootstrap"]

    rows.append(claim("P3_raw_26N_21_models", "Raw, 26.5\\(^\\circ\\)N change, 21 models"))
    rows.append(claim("P1_partial_26N_positive", "Partial given 1850--1900 strength, 26.5\\(^\\circ\\)N, 21 models"))
    rows.append(claim("P8_partial_26N_stable_leave_one_model_out", "Partial, 26.5\\(^\\circ\\)N, leave one model out"))
    rows.append(claim("P9_partial_26N_on_common_17_models", "Partial, 26.5\\(^\\circ\\)N, 17 common models"))
    rows.append(claim("P4_raw_26N_common_17_models_negative", "Raw, 26.5\\(^\\circ\\)N change, 17 common models"))
    rows.append(claim("P5_raw_34S_19_models_negative", "Raw, 34.5\\(^\\circ\\)S change, 19 models"))
    rows.append(claim("P6_partial_34S_not_distinguishable_from_zero",
                      "Partial given 1850--1900 strength, 34.5\\(^\\circ\\)S, 19 models"))
    rows.append(claim("P7_partial_26N_minus_34S_contrast",
                      "Partial contrast, 26.5\\(^\\circ\\)N minus 34.5\\(^\\circ\\)S, 17 models"))
    pdiff = rel.v("paired_difference_common_17.r.difference_26N_minus_34S")
    pdci = rel.v("paired_difference_common_17.r.ci95")
    rows.append(["Paired difference of raw correlations, 26.5\\(^\\circ\\)N minus 34.5\\(^\\circ\\)S, 17 models",
                 num(pdiff, 3), ci(pdci, 3), "family bootstrap"])
    f1 = Data("jc_r2_F1_cmip6_change")
    nun = f1.v("family_units")
    rows.append(group("(a, continued) Change-on-change relation and median changes over the 17 common models "
                      "(Sv; family-cluster bootstrap)", 4))
    rows.append(["Correlation of the 26.5\\(^\\circ\\)N change with the 34.5\\(^\\circ\\)S change",
                 num(f1.v("point.r"), 2), ci(f1.v("bootstrap_family_cluster.r.ci95"), 2), f"{nun} units"])
    rows.append(["Slope of the 26.5\\(^\\circ\\)N change on the 34.5\\(^\\circ\\)S change (Sv per Sv)",
                 num(f1.v("point.slope_26N_on_34S"), 2),
                 ci(f1.v("bootstrap_family_cluster.slope_26N_on_34S.ci95"), 2), f"{nun} units"])
    rows.append(["Median change of the overturning at 26.5\\(^\\circ\\)N, 2081--2100 minus 1850--1900 (Sv)",
                 num(f1.v("point.median_26N_change"), 2),
                 ci(f1.v("bootstrap_family_cluster.median_26N_change.ci95"), 2), f"{nun} units"])
    rows.append(["Median change of \\(\\Fovs\\) at 34.5\\(^\\circ\\)S, 2081--2100 minus 1850--1900 (Sv)",
                 num(f1.v("point.median_34S_change"), 2),
                 ci(f1.v("bootstrap_family_cluster.median_34S_change.ci95"), 2), f"{nun} units"])
    rows.append(group("(b) Forced change of \\(\\Fovs\\), SSP5-8.5 minus 1993--2014 (Sv), family-bootstrap median", 4))
    dec = "b_decomposition.ssp585_vs_1993-2014"
    for term, lab in [("dF_ovS", "Total change of \\(\\Fovs\\) (median over the 25 models; bootstrap over units)"),
                      ("transport_term", "Transport term"),
                      ("contrast_term", "Contrast term"),
                      ("product_term", "Product term")]:
        blk = f"{dec}.{term}"
        med = frc.v(f"{blk}.median")
        cif = frc.v(f"{blk}.ci_family_bootstrap")
        rows.append([lab, num(med, 3), ci(cif, 3), f"{frc.v(blk + '.n_units')} units"])
    rows.append(group("(c) Offsets of the 1993--2014 CMIP6 median from each reanalysis (Sv or PSU)", 4))
    for q, unit, lab in [("F_ov", "Sv", "\\(\\Fovs\\)"), ("dS", "PSU", "\\(\\Delta S\\)")]:
        for r, rl in [("oras5", "ORAS5"), ("glorys12", "GLORYS12V1"), ("ecco", "ECCO-V4r4")]:
            base = f"c_offsets_from_reanalyses_1993_2014.{q}.reanalysis.{r}"
            rows.append([f"{lab} offset from {rl}, all 25 models ({unit})",
                         num(frc.v(base + ".offset_median_minus_reanalysis"), 3),
                         ci(frc.v(base + ".offset_ci"), 3), "n.a."])
    # (b, continued) family-grouped median of the total change, SSP2-4.5 and the 1993--2014 spread
    g8 = Data("jc_r2_G08_cmip6")
    rows.append(group("(b, continued) Medians over the models and family-grouped medians (Sv or PSU), and the "
                      "SSP2-4.5 change of \\(\\Fovs\\)", 4))
    for key, lab, unit in [("F_ov_median", "\\(\\Fovs\\), 1993--2014, median of 25 models", "Sv"),
                           ("F_ov_family_grouped_median", "\\(\\Fovs\\), 1993--2014, family-grouped median (18 units)",
                            "Sv"),
                           ("dS_median", "\\(\\Delta S\\), 1993--2014, median of 25 models", "PSU"),
                           ("dS_family_grouped_median", "\\(\\Delta S\\), 1993--2014, family-grouped median (18 units)",
                            "PSU")]:
        base = f"A38_family_grouped_contrast_and_offsets.stats.{key}"
        rows.append([f"{lab} ({unit})", num(g8.v(base + ".point"), 3), ci(g8.v(base + ".ci"), 3), ""])
    fg = "a_table_and_medians.dF_ov_ssp585_vs_1993-2014"
    rows.append(["Models with a positive \\(\\Fovs\\) change, 2081--2100 minus 1993--2014 (SSP5-8.5; count of 25)",
                 f"{frc.v('a_table_and_medians.ssp585_2081-2100_F_ov.n_positive')} of 25", "n.a.", "25 models"])
    rows.append(["SSP5-8.5 change, family-grouped median (18 units)",
                 num(frc.v(fg + ".family_grouped_median"), 3),
                 ci(frc.v(fg + ".ci_family_grouped_bootstrap"), 3), "18 units"])
    rows.append(["SSP2-4.5 change, median (13 models)", num(g8.v("A82_spread_and_ssp245.ssp245.stats.dF_ov_ssp245_median.point"), 3),
                 ci(g8.v("A82_spread_and_ssp245.ssp245.stats.dF_ov_ssp245_median.ci"), 3), "13 models"])
    rows.append(["SSP2-4.5 change, family-grouped median (12 units)",
                 num(g8.v("A82_spread_and_ssp245.ssp245.stats.dF_ov_ssp245_family_grouped_median.point"), 3),
                 ci(g8.v("A82_spread_and_ssp245.ssp245.stats.dF_ov_ssp245_family_grouped_median.ci"), 3),
                 "12 units"])
    # (c, continued) family-grouped offsets from each reanalysis (25 models, 18 units)
    rows.append(group("(c, continued) Family-grouped offsets of the 1993--2014 CMIP6 median from each "
                      "reanalysis (median over 18 family units)", 4))
    for qk, unit, lab in [("F_ov", "Sv", "\\(\\Fovs\\)"), ("dS", "PSU", "\\(\\Delta S\\)")]:
        for rk, rl in [("oras5", "ORAS5"), ("glorys12", "GLORYS12V1"), ("ecco", "ECCO-V4r4")]:
            base = f"A38_family_grouped_contrast_and_offsets.stats.{qk}_offset_family_grouped_{rk}"
            if rk == "glorys12":
                base = f"A38_family_grouped_contrast_and_offsets.stats.{qk}_offset_family_grouped_glorys12"
            rows.append([f"{lab} offset from {rl}, family-grouped ({unit})", num(g8.v(base + ".point"), 3),
                         ci(g8.v(base + ".ci"), 3), "18 units"])
    # (d) level-offset decomposition of the 1993--2014 offset, family-grouped medians (Sv)
    rows.append(group("(d) Level-offset decomposition of \\(\\Fovs\\) at 1993--2014 (Sv, family-grouped medians): "
                      "transport, contrast and cross parts", 4))
    for ref, rl in [("oras5", "ORAS5"), ("glorys12", "GLORYS12V1"), ("ecco", "ECCO-V4r4")]:
        for part, plab in [("transport", "transport part"), ("contrast", "contrast part"),
                           ("cross", "cross part")]:
            key = f"A79_level_offset_decomposition_1993_2014.stats.{part}_{ref}_family_grouped_median"
            rows.append([f"{rl}, {plab}", num(g8.v(key + ".point"), 4), ci(g8.v(key + ".ci"), 4), "18 units"])
        key = f"A79_level_offset_decomposition_1993_2014.stats.actual_F_ov_offset_{ref}_family_grouped_median"
        rows.append([f"{rl}, offset of \\(\\Fovs\\) (median of the total)", num(g8.v(key + ".point"), 4),
                     ci(g8.v(key + ".ci"), 4), "18 units"])
    # pooled 25-model medians of the level-offset parts against ORAS5 (not additive)
    for part, plab in [("transport", "transport part"), ("contrast", "contrast part"), ("cross", "cross part")]:
        key = f"A79_level_offset_decomposition_1993_2014.stats.{part}_oras5_median"
        rows.append([f"ORAS5, {plab}, pooled median of 25 models", num(g8.v(key + ".point"), 3),
                     ci(g8.v(key + ".ci"), 3), "25 models"])
    # (e) spread of the 1993--2014 and 2081--2100 values (family-bootstrap standard deviations, Sv or PSU)
    rows.append(group("(e) Spread across models: family-bootstrap standard deviations (SSP5-8.5; 25 models, 18 "
                      "units) and their ratios", 4))
    for key, lab, unit in [("sd_F_ov_1993-2014", "\\(\\Fovs\\), 1993--2014", "Sv"),
                           ("sd_F_ov_2081-2100", "\\(\\Fovs\\), 2081--2100", "Sv"),
                           ("sd_T_1993-2014", "\\(T\\), 1993--2014", "Sv"),
                           ("sd_T_2081-2100", "\\(T\\), 2081--2100", "Sv"),
                           ("sd_dS_1993-2014", "\\(\\Delta S\\), 1993--2014", "PSU"),
                           ("sd_dS_2081-2100", "\\(\\Delta S\\), 2081--2100", "PSU")]:
        base = f"A82_spread_and_ssp245.sd_ssp585.stats.{key}"
        rows.append([f"Standard deviation, {lab} ({unit})", num(g8.v(base + ".point"), 3),
                     ci(g8.v(base + ".ci"), 3), "18 units"])
    for key, lab in [("sd_ratio_2081_over_1993_F_ov", "\\(\\Fovs\\)"), ("sd_ratio_2081_over_1993_T", "\\(T\\)"),
                     ("sd_ratio_2081_over_1993_dS", "\\(\\Delta S\\)")]:
        base = f"A82_spread_and_ssp245.sd_ssp585.stats.{key}"
        rows.append([f"Ratio of standard deviations 2081--2100 over 1993--2014, {lab} (unitless)",
                     num(g8.v(base + ".point"), 2), ci(g8.v(base + ".ci"), 2), "18 units"])
    fam_note = (f"Families: {rel.v('family_clusters.26.5N_21')} of 21 models (26.5\\(^\\circ\\)N), "
                f"{rel.v('family_clusters.34.5S_19')} of 19 models (34.5\\(^\\circ\\)S), "
                f"{rel.v('family_clusters.common_17')} of 17 common models; the 25 models of \\(\\Fovs\\) form "
                f"{frc.v('a_table_and_medians.F_ov_1993-2014.n_units')} independent family units.")
    note = ("Correlations are for the models named, with 95\\% intervals from the family bootstrap "
            "(10{,}000 resamples; seed 20261104), except the leave-one-out row, which gives the range over "
            "all single-model deletions. Partial correlations are given the 1850--1900 overturning strength "
            "at the same latitude. Panel (b) gives the family-bootstrap median and interval of each term "
            "over the models; the medians of the terms do not sum to the median of the total, so each term "
            "is shown separately. In panel (d) each row is a median over the 18 family units or pooled over the "
            "25 models, and the medians of the parts are not additive: they do not sum to the median of the total "
            "offset of the same product. Panel (c) "
            "gives the median over models of the 1993--2014 value minus the "
            "reanalysis value, with the family-bootstrap interval. " + fam_note)
    return table_block(
        "tab:si_cmip6_relation",
        "Cross-model relations and the forced change of \\(\\Fovs\\) in CMIP6, with offsets from the reanalyses.",
        "Quantity & Estimate & 95\\% interval & Note",
        rows, note, "lrrl", stretch="0.88"), rel


# ---------------------------------------------------------------- frozen-field split
def table_frozen():
    """Frozen-field split of the limb salinity trend into salinity, velocity and cross terms (G07)."""
    g7 = Data("jc_r2_G07_frozen_epochs_intervals")
    rows = []
    for prod, plab, rec in [("oras5", "ORAS5", "1958--2025"), ("glorys12", "GLORYS12V1", "1993--2025")]:
        rows.append(group(f"{plab}, {rec}; trends in PSU per decade", 7))
        for term, tlab in [("full", "Full trend"), ("salinity_only", "Salinity term only"),
                           ("velocity_only", "Velocity term only")]:
            base = f"{prod}.{term}"
            floor = g7.v(f"{base}.n_eff_at_floor")
            t_ci = g7.v(f"{base}.ci_95_per_decade")
            t_txt = ('uninformative' + DAG) if floor else ci(t_ci, 4)
            sieve = g7.v(f"{prod}.three_term_split_per_decade.{term}_ci_95_sieve")
            if term == "full":
                sieve = g7.v(f"{prod}.three_term_split_per_decade.full_ci_95_sieve")
            neff_txt = f"{g7.v(base + '.n_eff'):.1f}" + (DAG if floor else "")
            rows.append([tlab, num(g7.v(f"{base}.trend_per_decade"), 4), t_txt,
                         ('uninformative' + DAG) if floor else ci(sieve, 4),
                         num(g7.v(f"{base}.min_detectable_trend_per_decade"), 4), neff_txt,
                         pval(g7.v(f"{base}.p_value"))])
        cross = f"{prod}.three_term_split_per_decade"
        rows.append(["Cross term (residual)", num(g7.v(f"{cross}.cross_term"), 4), "n.a.",
                     ci(g7.v(f"{cross}.cross_term_ci_95_sieve"), 4), "n.a.", "n.a.",
                     f"sieve two-sided {pval(g7.v(cross + '.cross_term_sieve_two_sided_p'))}"])
        rows.append(group(f"{plab}: shares of the full trend (sieve intervals; dimensionless)", 7))
        for term, tlab in [("salinity_only", "Salinity term only"), ("velocity_only", "Velocity term only"),
                           ("cross_term", "Cross term")]:
            sb = f"{prod}.shares_of_full_trend.{term}"
            rows.append([tlab, num(g7.v(sb + ".point"), 3), "n.a.", ci(g7.v(sb + ".ci_95_sieve"), 3),
                         "n.a.", "n.a.", "n.a."])
    note = ("Trends of the limb salinity contrast \\(\\Delta S\\) split by the frozen-field decomposition: the "
            "full trend equals the sum of the salinity-only term (salinity trend with the record-mean velocity), "
            "the velocity-only term (velocity trend with the record-mean salinity) and the cross term "
            "(full minus the two parts). Intervals in the column of the \\(t\\) test are 95\\% intervals with "
            "the bias-corrected effective sample size; those of the sieve bootstrap (10{,}000 resamples, seed "
            "20261104) are given beside them. A dagger marks an effective sample size at its floor of 3, where "
            "both interval columns read uninformative. The minimum detectable trend "
            "is the half-width of the \\(t\\) interval. The cross term has no \\(t\\) interval; its sieve "
            "two-sided \\(p\\) value is given in the last column. Shares are the fractions of the full trend; "
            "their sieve intervals are wide where the full trend is small. For the full trend of GLORYS12V1 over 1993 to 2025 the sieve "
            "interval excludes zero while the \\(t\\) interval includes it: the sieve interval is conditional on "
            "the stored effective sample size, which lies near its floor of 3, so the verdict follows the "
            "bias-corrected \\(t\\) interval.")
    return table_block(
        "tab:si_frozen",
        "Frozen-field split of the limb salinity trend at 34.5\\(^\\circ\\)S into salinity, velocity and cross "
        "terms, ORAS5 and GLORYS12V1.",
        "Term & Trend & 95\\% interval (\\(t\\)) & 95\\% interval (sieve) & Min. detectable trend & \\(N_{\\mathrm{eff}}\\) & \\(p\\)",
        rows, note, "lrrrrrr"), g7


# ---------------------------------------------------------------- misc (inputs of the discussion)
def table_misc():
    """Jensen term, depth share of the southward weight, depth-band trends and own-circulation EN4 means (G11)."""
    g11 = Data("jc_r2_G11_tables_s8")
    rows = []
    rows.append(group("(a) Jensen term of the northward-limb transport (monthly fields against the annual "
                      "fields; Sv and percent of the mean)", 5))
    for prod, plab in [("ORAS5", "ORAS5"), ("GLORYS12V1", "GLORYS12V1"), ("ECCO-V4r4", "ECCO-V4r4")]:
        b = f"jensen_term_by_product.{prod}"
        rows.append([plab, num(g11.v(b + ".mean_J_T_Sv"), 3), "n.a.",
                     f"{g11.v(b + '.rel_J_T_to_T_mean_percent'):.1f}\\%", num(g11.v(b + ".mean_T_monthly_Sv"), 2)])
    rows.append(group("(b) Depth share of the southward and northward limb weights, ORAS5 (Sv and percent)", 5))
    sw = "southward_weight_depth_share"
    rows.append(["Southward limb: total weight (Sv)", num(g11.v(sw + ".southward.total_weight_Sv"), 2),
                 "n.a.", "n.a.", "n.a."])
    rows.append(["Southward limb: weight below 2000 m (Sv)", num(g11.v(sw + ".southward.weight_below_2000m_Sv"), 2),
                 "n.a.", f"{g11.v(sw + '.southward.share_below_2000m_percent'):.1f}\\%", "n.a."])
    rows.append(["Northward limb: total weight (Sv)", num(g11.v(sw + ".northward.total_weight_Sv"), 2),
                 "n.a.", "n.a.", "n.a."])
    rows.append(["Northward limb: weight below 2000 m (Sv)", num(g11.v(sw + ".northward.weight_below_2000m_Sv"), 2),
                 "n.a.", f"{g11.v(sw + '.northward.share_below_2000m_percent'):.1f}\\%", "n.a."])
    rows.append(group("(c) Salinity trends by depth band, EN4 and reanalyses, 2005--2024 (PSU per decade; "
                      "ratios dimensionless)", 5))
    bands = g11.raw["salinity_trends_by_depth_band"]["bands"]
    for band in bands:
        b = f"salinity_trends_by_depth_band.bands.{band}"
        rows.append([f"{band}".replace("-", "--"),
                     num(g11.v(b + ".trend_EN4_PSU_per_decade"), 4),
                     num(g11.v(b + ".trend_ORAS5_PSU_per_decade"), 4) + " (" + num(
                         g11.v(b + ".ratio_ORAS5_to_EN4"), 2) + ")",
                     num(g11.v(b + ".trend_GLORYS12V1_PSU_per_decade"), 4) + " (" + num(
                         g11.v(b + ".ratio_GLORYS12V1_to_EN4"), 2) + ")",
                     num(g11.v(b + ".obs_weight_en4"), 3), "n.a."])
    rows.append(group("(d) EN4 salinity with each reanalysis circulation, 2005--2024 (mean values; Sv)", 5))
    for prod, plab in [("ORAS5", "ORAS5"), ("GLORYS12V1", "GLORYS12V1")]:
        c = f"own_circulation_en4_means_2005_2024.circulation.{prod}"
        f_ci = g11.v(c + ".EN4_mean_ci95.F_ov_Sv")
        rows.append([f"{plab} circulation: \\(\\Fovs\\) with EN4 salinity (Sv)",
                     num(g11.v(c + ".EN4_mean.F_ov_Sv"), 4), ci(f_ci, 4),
                     f"own monthly: {num(g11.v(c + '.own_monthly_mean.F_ov_Sv'), 4)}",
                     f"\\(\\Delta S\\) {num(g11.v(c + '.EN4_mean.dS'), 3)}", "n.a."])
    cells = Data("r1_cells")
    rows.append(group("(e) Overturning cell trends of ORAS5 by window (mSv yr$^{-1}$; 95\\% intervals and "
                      "effective sample sizes as stored)", 5))
    for wkey, wlab in [("own", "1958--2025"), ("common", "1993--2017"), ("argo", "2004--2025")]:
        w = f"oras5.windows.{wkey}"
        items = [
            ("upper cell, total", cells.v(f"{w}.trend_F_up.trend"), cells.v(f"{w}.trend_F_up.ci"),
             cells.v(f"{w}.trend_F_up.n_eff")),
            ("transport term", cells.v(f"{w}.upper_cell_attribution.trend_T_mSv_per_yr"),
             cells.v(f"{w}.upper_cell_attribution.ci_T_mSv_per_yr"),
             cells.v(f"{w}.upper_cell_attribution.n_eff_T")),
            ("contrast term", cells.v(f"{w}.upper_cell_attribution.trend_S_mSv_per_yr"),
             cells.v(f"{w}.upper_cell_attribution.ci_S_mSv_per_yr"),
             cells.v(f"{w}.upper_cell_attribution.n_eff_S")),
            ("abyssal cell", cells.v(f"{w}.trend_F_ab.trend"), cells.v(f"{w}.trend_F_ab.ci"),
             cells.v(f"{w}.trend_F_ab.n_eff")),
        ]
        for lab, tr, cf, ne in items:
            rows.append([f"{wlab}, {lab}", num(tr, 3), ci(cf, 3), "n.a.", neff(ne)])
    rows.append(group("(f) ORAS5 upper-cell residual and total freshwater transport trends (mSv yr$^{-1}$)", 5))
    rsd = Data("jc_r2_G11_tables_residual")
    rb = "upper_cell_residual_ORAS5.own"
    rows.append(["Upper-cell residual, 1958--2025 (actual minus identity)",
                 f"${rsd.v(rb + '.residual_mSv_per_yr'):+.3f}$",
                 ci(rsd.v(rb + ".residual_ci95_bootstrap_mSv_per_yr"), 3), "n.a.", "n.a."])
    g4 = Data("jc_r2_G04_removal_total")
    for wkey, wlab in [("own", "1958--2025"), ("common_1993_2017", "1993--2017")]:
        for tkey, tlab in [("corrected_total", "with net-transport removal"),
                           ("uncorrected_total", "without net-transport removal")]:
            w = f"A101_uncorrected_total.oras5.windows.{wkey}.{tkey}"
            rows.append([f"ORAS5, {wlab}, {tlab}", num(g4.v(w + ".trend_mSv_per_yr"), 2),
                         ci(g4.v(w + ".ci_mSv_per_yr"), 2), "n.a.", neff(g4.v(w + ".n_eff"))])
    note = ("Jensen term: the difference between the northward-limb transport from the monthly fields and from "
            "the annual fields, with its percent share of the mean monthly-field transport. Depth shares are "
            "shares of the absolute transport weight on the levels of the vertical decomposition (41 levels). "
            "Band trends are layer means of the salinity trend profile; ratios are reanalysis to EN4 and are not "
            "interpreted below 2000 m, where the EN4 trends are near zero. EN4 means "
            "use each reanalysis circulation frozen at its record mean (Methods), with the 95\\% interval stored "
            "with each mean. Panel (e) gives the upper-cell total, its transport and contrast terms, and the abyssal "
            "trend of ORAS5 for the windows 1958--2025, 1993--2017 and 2004--2025; the last column is the effective "
            "sample size of each trend. Panel (f) gives the upper-cell residual of the identity (sieve bootstrap "
            "interval) and the total freshwater transport trends with and without the removal of the net section "
            "transport (bias-corrected test intervals).")
    return table_block(
        "tab:si_misc",
        "Inputs of the Discussion: Jensen term, depth share of the limb weights, salinity trends by depth band "
        "and the EN4 means with the reanalysis circulations.",
        "Quantity & Value & 95\\% interval or ratio & Second value & Weight or mean & Note",
        rows, note, "lrrrrr"), g11


# ---------------------------------------------------------------- data inventory (typed rows)
def table_data():
    """Data inventory (input, provider, DOI or URL, version, access date, terms), typed from the data statement."""
    rows = [
        ["ORAS5", "Copernicus Climate Data Store", "dataset reanalysis-oras5 (CDS)",
         "Monthly means, consolidated and operational streams, 1958--2025", "2026-03-06 to 2026-05-08",
         "registration required"],
        ["GLORYS12V1", "Copernicus Marine Service",
         "GLOBAL MULTIYEAR PHY 001 030 (monthly means); dataset cmems\\_mod\\_glo\\_phy\\_my\\_0.083deg\\_P1M-m",
         "GLORYS12V1 system; product page ID GLOBAL\\_MULTIYEAR\\_PHY\\_001\\_030",
         "2026-03-06 to 2026-03-08", "registration required"],
        ["ECCO-V4r4", "NASA PO.DAAC", "ECCO L4 OCEAN VEL 05DEG MONTHLY V4R4; ECCO L4 TEMP SALINITY 05DEG "
         "MONTHLY V4R4", "V4r4, 1992--2017", "2026-04-20", "registration required (NASA Earthdata)"],
        ["SODA 3.15.2", "University of Maryland", "https://dsrs.atmos.umd.edu/DATA/soda3.15.2/",
         "3.15.2", "2026-04-20 (derived)", "open"],
        ["EN4.2.2 with g10 corrections", "Met Office Hadley Centre", "https://www.metoffice.gov.uk/hadobs/en4/",
         "EN4.2.2 g10 (monthly objective analyses), 2005--2024", "2026-05-14", "open"],
        ["Roemmich and Gilson Argo climatology (RG09)", "Scripps Institution of Oceanography",
         "https://sio-argo.ucsd.edu/RG\\_Climatology.html; file RG\\_ArgoClim\\_Salinity\\_2019.nc, extended by "
         "monthly anomaly files to 2024", "2019 release, extension to 2024", "2026-05-14 (file date)", "open"],
        ["RAPID transports", "RAPID-AMOC programme", "https://rapid.ac.uk/rapidmoc",
         "v2024.1a (April 2004 to March 2024); monthly transports", "2026-03-07", "open"],
        ["RAPID-MOCHA-WBTS (dataset)", "Moat et al. (2025)",
         "DOI 10.5285/48d0bf43-0598-ceb2-e063-7086abc062f1 (release v2024.1a, April 2004 to March 2024)",
         "v2024.1a", "2026-03-07", "open"],
        ["SAMBA cell transports", "NOAA AOML; Kersal\\'e et al. (2020)",
         "https://www.aoml.noaa.gov/phod/SAMOC\\_international/; doi:10.1126/sciadv.aba7573",
         "Monthly and daily anomalies, September 2013 to July 2017", "2026-03-07", "open"],
        ["XBT estimates at 34.5\\(^\\circ\\)S and 35\\(^\\circ\\)S", "Arum\\'i-Planas et al. (2024); Dong et al. (2009)",
         "Through the publications (no DOI in the manuscript)", "Estimates as published",
         "not applicable: values as published in the cited papers", "not applicable: values as published"],
        ["CMIP6 model output (25 models)", "ESGF; Pangeo CMIP6 catalog",
         "https://esgf.llnl.gov/; https://storage.googleapis.com/cmip6/pangeo-cmip6.json; variables vo, so, msftmz, msftyz",
         "Variant r1i1p1f1; list of models in Table~\\ref{tab:si_cmip6_models}", "2026-03-11 to 2026-05-14", "open"],
        ["MPI-ESM1-2-LR large ensemble", "Pangeo", "Members r1 to r8; variables vo, so, thkcello (gn grid)",
         "Version v20190710", "2026-11-04 (derived)", "open"],
        ["Analysis code and summaries", "GitHub", "github.com/bijanf/ardp (versioned archive with a DOI at acceptance)",
         "Tag paper3-jclim-2026-11-05", "not applicable", "MIT licence"],
    ]
    lines = ["\\begin{table}[h]", "\\centering",
             "\\caption{Data inventory: the inputs of the study, their providers, identifiers, versions, access dates "
             "and terms.}", "\\label{tab:si_data}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{p{2.2cm}p{2.4cm}p{3.9cm}p{3.0cm}p{1.8cm}p{2.0cm}}", "\\toprule",
             "Input & Provider & DOI or URL & Version or product & Access date & Terms \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\par\\smallskip",
              "\\begin{minipage}{0.95\\linewidth}\\footnotesize Access dates are the file dates of the raw downloads "
              "kept in the analysis repository; where raw files are not kept, the date of the derived file is given "
              "with (derived). The XBT values are those published in the cited papers; no data product was downloaded. "
              "\\end{minipage}", "\\end{table}", ""]
    return "\n".join(lines), None


INSTITUTIONS = {
    # typed from the CMIP6 model metadata (institution of origin); not stored in the analysis JSON
    "ACCESS-CM2": "CSIRO-ARCCSS", "ACCESS-ESM1-5": "CSIRO", "BCC-CSM2-MR": "BCC", "CESM2": "NCAR",
    "CESM2-WACCM": "NCAR", "CIESM": "Tsinghua University", "CMCC-CM2-SR5": "CMCC", "CMCC-ESM2": "CMCC",
    "CNRM-CM6-1": "CNRM-CERFACS", "CanESM5": "CCCma", "EC-Earth3": "EC-Earth consortium",
    "FGOALS-f3-L": "CAS-LASG", "FGOALS-g3": "CAS-LASG", "FIO-ESM-2-0": "FIO-QLNM", "GFDL-CM4": "NOAA-GFDL",
    "GISS-E2-1-G": "NASA-GISS", "HadGEM3-GC31-LL": "MOHC", "IPSL-CM6A-LR": "IPSL", "MIROC6": "MIROC",
    "MPI-ESM1-2-HR": "MPI-M", "MPI-ESM1-2-LR": "MPI-M", "MRI-ESM2-0": "MRI", "NESM3": "NUIST",
    "TaiESM1": "AS-RCEC", "UKESM1-0-LL": "MOHC",
}


def table_models():
    """Model table of the 25 models: institution, family, subsets used and per-model values (1993 to 2014)."""
    g9 = Data("jc_r2_G09_cmip6meta_models")
    n12 = Data("jc_n12_cmip6_forced")
    n11 = Data("jc_n11_cmip6_relation")
    families = n12.raw["meta"]["families"]
    fam_of = {m: fam for fam, members in families.items() for m in members}
    sub = {"R": set(n11.raw["sets"]["26.5N_21"]["members"]),
           "S": set(n11.raw["sets"]["34.5S_19"]["members"]),
           "C": set(n11.raw["sets"]["common_17_26.5N"]["members"])}
    lines = ["\\begin{table}[h]", "\\centering",
             "\\caption{The 25 CMIP6 models: institution, family group (models sharing a family are one unit in "
             "the family-grouped statistics), subsets used (R: 26.5\\(^\\circ\\)N set of 21 models; S: 34.5\\(^\\circ\\)S "
             "set of 19; C: 17 common models) and the per-model section values over 1993--2014.}",
             "\\label{tab:si_cmip6_models}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{llllrrrrrrrr}", "\\toprule",
             "Model & Institution & Family & Subsets & Lat.\\ S (\\(^\\circ\\)) & Lat.\\ N (\\(^\\circ\\)) & "
             "Lon.\\ W (\\(^\\circ\\)) & Lon.\\ E (\\(^\\circ\\)) & Points (levels) & Net removal (Sv) & "
             "Jensen (mSv) & \\(\\Fovs\\) (Sv) \\\\", "\\midrule"]
    for pm in g9.raw["per_model"]:
        name = pm["model"]
        w = pm["windows"]["1993-2014"]
        fam = fam_of.get(name, "none (single)")
        subs = "".join(k for k in ["R", "S", "C"] if name in sub[k]) or "none"
        cells = [name, INSTITUTIONS.get(name, "n.a."), fam.replace("none (single)", "single"),
                 subs, f"{pm['lat_south']:.2f}", f"{pm['lat_north']:.2f}", f"{pm['lon_west']:.1f}",
                 f"{pm['lon_east']:.1f}", f"{pm['n_points']} ({pm['grid_description']['n_vertical_levels']})",
                 f"{w['V_net_removed_mean_Sv']:.4f}", f"{1000 * w['jensen_term_Sv']:.3f}",
                 f"{w['F_ovS_monthly_mean_Sv']:.4f}"]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\par\\smallskip",
              "\\begin{minipage}{0.95\\linewidth}\\footnotesize Lat.\\ and Lon.\\ give the section row and its longitude "
              "range in each model; points are the number of section cells, with the number of vertical levels in "
              "brackets. Net removal is the net section transport removed, in Sv, over 1993 to 2014; the Jensen term "
              "is the monthly-field correction over the same window, in mSv (reanalysis values in Table~\\ref{tab:si_misc}). "
              "Institutions are those of origin, typed from the CMIP6 model metadata. The subsets are the sets of "
              "Table~\\ref{tab:si_cmip6_relation}, and the family groups are those used for the family-grouped "
              "statistics. \\(\\Fovs\\) is the 1993 to 2014 mean in Sv.\\end{minipage}",
              "\\end{table}", ""]
    return "\n".join(lines)


def label_order(labels):
    """Order the labels by first citation in Main.tex; unseen labels go last in the default order."""
    try:
        text = MAIN.read_text()
    except OSError:
        return labels
    pos = {}
    for lab in labels:
        m = re.search(r"\\ref\{SI-tab:" + re.escape(lab) + r"\}", text)
        if m:
            pos[lab] = m.start()
    seen = sorted([l for l in labels if l in pos], key=lambda l: pos[l])
    rest = [l for l in labels if l not in pos]
    return seen + rest


def main():
    builders = {
        "si_size": table_size,
        "si_frozen": table_frozen,
        "si_steps": table_steps,
        "si_en4_timing": table_en4_timing,
        "si_obs_compare": table_obs,
        "si_misc": table_misc,
        "si_cmip6_relation": table_cmip6,
        "si_data": table_data,
    }
    order = label_order(list(builders))
    for lab in builders:
        text, _ = builders[lab]()
        header = ("% Generated by tools/jc_si_tables.py from the analysis JSON files; do not edit by hand.\n"
                  f"% Table with label tab:{lab.replace('si_cmip6_relation', 'si_cmip6_relation')}; placed in "
                  "Supplement.tex in the order of first citation in Main.tex.\n")
        OUTFILES[lab].write_text(header + text)
        print("wrote", OUTFILES[lab])
    # 25-model table (tab:si_cmip6_models) is cited only from the Supplement, so it is its own file, placed after si_data
    header = ("% Generated by tools/jc_si_tables.py from the analysis JSON files; do not edit by hand.\n"
              "% Table with label tab:si_cmip6_models; placed in Supplement.tex after si_data (cited only from the Supplement).\n")
    OUTFILES["si_models"].write_text(header + table_models())
    print("wrote", OUTFILES["si_models"])
    print("first-citation order of the generated tables in Main.tex:", order)
    if MISSING:
        print("MISSING KEYS:")
        for m in sorted(set(MISSING)):
            print("  ", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
