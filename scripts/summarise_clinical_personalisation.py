"""D37: section 3.5's primary analysis, on the clinical metric, across both cohorts.

Reads what scripts/run_clinical_arms.py wrote and applies the two tests fixed in D37
before any of these numbers existed:

  1. PERSONALISATION, the D28 analogue. Per fold, the own-z FDR/h reduction against the
     seven shuffled-z reductions, pre-registered threshold z > 2.0.
  2. SECTION 3.5's own claim. corr(D_p, FDR/h reduction) at n = 7 -- all folds less
     pat09, per D35 rule 1 -- with an exact permutation test over all n! pairings.

Benefit is defined as a REDUCTION: baseline FDR/h minus the condition's FDR/h, so
positive means fewer false detections per hour. Raw alarm counts and exposure hours are
printed beside every rate, because at 0.10-0.76 h of evaluable exposure per patient a
single alarm moves the rate by 1.3-10 FDR/h and a rate without its count is not
interpretable.

    python scripts/summarise_clinical_personalisation.py
"""
import importlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from src.eval.stats import corr_perm_p, sign_flip_p, zscore_against

RUNS = [("8-patient", ""), ("14-patient", "alldata")]
OUT_NAME = "SECTION_35_CLINICAL.md"


def fmt_z(z):
    """A z-score is undefined when the seven controls are identical; that means the
    metric could not resolve the difference, and it is reported as n/a rather than as
    a pass or a failure."""
    return "n/a" if not np.isfinite(z) else f"{z:+.2f}"


def dirs_for(run):
    os.environ["VSVIG_RUN"] = run
    importlib.reload(config)
    return Path(config.RESULTS_DIR) / "clinical", Path(config.RESULTS_DIR) / "adapted"


DIRS = {name: dirs_for(run) for name, run in RUNS}


def load(name):
    clin, adapted = DIRS[name]
    rows = {}
    for f in sorted(config.COHORT):
        p = clin / f"{f}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        c = d["conditions"]
        base = c["baseline"]["fdr_per_hour"]
        shuf = {k: base - v["fdr_per_hour"] for k, v in c.items()
                if k.startswith("shuffled_")}
        rows[f] = {
            "fold": f,
            "dt": d["decision_threshold"],
            "baseline_fdr": base,
            "baseline_fa": d["baseline_false_alarms"],
            "exposure_h": d["exposure_hours"],
            "own_fdr": c["own"]["fdr_per_hour"],
            "reduction_own": base - c["own"]["fdr_per_hour"],
            "reduction_shuffled": shuf,
            "sens_baseline": c["baseline"]["sensitivity"],
            "sens_own": c["own"]["sensitivity"],
            "zero_fa": d["zero_baseline_false_alarms"],
            "sens_drop": d["sensitivity_drop"].get("own", False),
        }
        rows[f]["z"] = zscore_against(rows[f]["reduction_own"], list(shuf.values()))
        # POST HOC (D38), not pre-registered: the same benefit in raw alarms. FDR/h
        # divides an integer count by an exposure as small as 0.097 h, so one alarm is
        # worth 10-20 FDR/h in the short folds and 1.3 in the long ones. The count is
        # the same quantity without that amplification.
        rows[f]["reduction_own_alarms"] = (c["baseline"]["n_false_alarms"]
                                           - c["own"]["n_false_alarms"])
        rows[f]["own_alarms"] = c["own"]["n_false_alarms"]
        rows[f]["shuffled_alarms"] = sorted(v["n_false_alarms"] for k, v in c.items()
                                            if k.startswith("shuffled_"))
        rows[f]["inv_exposure"] = 1.0 / d["exposure_hours"] if d["exposure_hours"] else float("nan")
    # D_p comes from the cohort summary the AUC analysis already wrote
    cs = adapted / "cohort_summary.json"
    if cs.exists():
        for r in json.loads(cs.read_text())["rows"]:
            if r["fold"] in rows:
                rows[r["fold"]]["atypicality"] = r.get("atypicality", float("nan"))
    return rows


def main():
    cohorts = {name: load(name) for name, _ in RUNS}
    if not any(cohorts.values()):
        print("no clinical results yet -- run scripts/run_clinical_arms.py first")
        return 1

    n7 = [f for f in sorted(config.COHORT) if f not in config.SECTION_35_EXCLUDE]
    L = ["# Section 3.5 on the clinical metric, under D37\n",
         "Benefit is the FDR/h REDUCTION against the fold's own unadapted baseline at the",
         "D16 threshold, which is selected on the internal validation patients and applied",
         "unchanged to all nine conditions. Positive means fewer false detections per hour.\n"]
    summary = {}

    for name, _ in RUNS:
        rows = cohorts[name]
        if not rows:
            L.append(f"## {name}: not yet run\n")
            continue
        L.append(f"## {name} cohort\n")
        L.append("| fold | DT | baseline FA / exposure | baseline FDR/h | own FDR/h | "
                 "reduction | z | sens base -> own | |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for f in sorted(rows):
            r = rows[f]
            flags = []
            if r["zero_fa"]:
                flags.append("**no baseline false alarms** -- cannot show a reduction")
            if r["sens_drop"]:
                flags.append("**sensitivity fell** -- not a benefit")
            L.append(
                f"| {f} | {r['dt']:.3f} | {r['baseline_fa']} / {r['exposure_h']:.3f} h | "
                f"{r['baseline_fdr']:.2f} | {r['own_fdr']:.2f} | "
                f"{r['reduction_own']:+.2f} | "
                f"{fmt_z(r['z'])} | "
                f"{r['sens_baseline']:.2f} -> {r['sens_own']:.2f} | "
                f"{'; '.join(flags)} |")

        zs = [rows[f]["z"] for f in rows if np.isfinite(rows[f]["z"])]
        npass = sum(1 for z in zs if z > 2.0)
        L.append(f"\n**Test 1 (personalisation, D28's z > 2.0): {npass} of {len(zs)} "
                 f"folds pass.**")
        if zs:
            L.append(f"z ranges {min(zs):+.2f} to {max(zs):+.2f}.")

        # paired: is the own-z reduction better than the mean shuffled-z reduction?
        d = [rows[f]["reduction_own"] - float(np.mean(list(rows[f]["reduction_shuffled"].values())))
             for f in rows]
        p, n = sign_flip_p(d)
        L.append(f"\nPaired own-minus-shuffled reduction over all {n} folds: "
                 f"{np.mean([x for x in d if np.isfinite(x)]):+.3f} FDR/h, "
                 f"exact sign-flip p = {p:.4f}.")

        L.append("\n**Test 2 (§3.5's correlation).**\n")
        L.append("| analysis set | r | exact p (n! pairings) | n |")
        L.append("|---|---|---|---|")
        stats = {}
        for label, folds in (("n = 7, D35 rule 1 (primary)", n7),
                             ("all folds (secondary)", sorted(config.COHORT))):
            sel = [f for f in folds if f in rows and "atypicality" in rows[f]]
            r, p2, nn = corr_perm_p([rows[f]["atypicality"] for f in sel],
                                    [rows[f]["reduction_own"] for f in sel])
            L.append(f"| {label} | {r:+.3f} | {p2:.4f} | {nn} |")
            stats[label] = {"r": r, "p": p2, "n": nn}
        L.append("\n§3.5 predicts POSITIVE.\n")

        # ------------------------------------------------------- post hoc (D38)
        L.append("**Post hoc, not pre-registered: the same test without the exposure "
                 "division.**\n")
        sel = [f for f in n7 if f in rows and "atypicality" in rows[f]]
        dp = [rows[f]["atypicality"] for f in sel]
        r_rate, p_rate, _ = corr_perm_p(dp, [rows[f]["reduction_own"] for f in sel])
        r_cnt, p_cnt, _ = corr_perm_p(dp, [rows[f]["reduction_own_alarms"] for f in sel])
        r_exp, p_exp, _ = corr_perm_p(dp, [rows[f]["inv_exposure"] for f in sel])
        amp = float(np.corrcoef([rows[f]["inv_exposure"] for f in sel],
                                [abs(rows[f]["reduction_own"]) for f in sel])[0, 1])
        L.append("| quantity (n = 7) | r | exact p |")
        L.append("|---|---|---|")
        L.append(f"| D_p vs FDR/h reduction (pre-registered) | {r_rate:+.3f} | {p_rate:.4f} |")
        L.append(f"| D_p vs **alarm-count** reduction | {r_cnt:+.3f} | {p_cnt:.4f} |")
        L.append(f"| D_p vs 1/exposure | {r_exp:+.3f} | {p_exp:.4f} |")
        L.append(f"| 1/exposure vs \\|FDR/h reduction\\| | {amp:+.3f} | |")
        L.append("")
        L.append("| fold | baseline FA | own FA | shuffled FA | own-z benefit, in alarms |")
        L.append("|---|---|---|---|---|")
        for f in sorted(rows):
            r = rows[f]
            L.append(f"| {f} | {r['baseline_fa']} | {r['own_alarms']} | "
                     f"{r['shuffled_alarms']} | {r['reduction_own_alarms']:+d} |")
        L.append("")
        summary_extra = {"corr_rate": r_rate, "p_rate": p_rate,
                         "corr_alarm_count": r_cnt, "p_alarm_count": p_cnt,
                         "corr_inv_exposure": r_exp,
                         "corr_inv_exposure_vs_abs_reduction": amp}
        stats["post_hoc_exposure_free"] = summary_extra
        summary[name] = {"folds": {f: {k: v for k, v in rows[f].items()
                                       if k != "reduction_shuffled"} for f in rows},
                         "n_passing_D28": npass, "n_defined_z": len(zs),
                         "paired_sign_flip_p": p, "correlations": stats}

    out = DIRS["14-patient"][0].parent / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    (out.parent / "section_35_clinical.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print("\n".join(L))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
