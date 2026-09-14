"""Cohort-level verdict on personalisation, across all folds.

Per fold, `evaluate_adapted.py` produces one held-out AUC with the patient's own
z_behavior and seven with every other patient's. This pools them:

  * per fold, the D28 z-score
  * across folds, a paired test of own-z against the per-fold shuffled-z mean
  * the section 3.5 correlation: benefit against atypicality D_p = ||z_p - c||,
    with c the cohort centroid over the FOLD'S TRAINING patients, as section 3.5
    defines it

The paired test is the cohort-level statement. n = 8 is small, so the sign test is
reported beside the t-test: with eight folds it needs 8/8 in one direction to reach
p < 0.05, which is the honest resolution limit of this cohort.

    python scripts/summarise_personalisation.py [--tag gentle]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config


def t_paired(d):
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float("nan"), float("nan")
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    # two-sided p from the t distribution, series-free: use the normal approximation
    # and say so, rather than pulling in scipy for one number.
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return float(t), float(p)


def sign_test(d):
    d = np.asarray(d, dtype=float)
    pos = int((d > 0).sum()); n = int((d != 0).sum())
    if n == 0:
        return pos, n, float("nan")
    from math import comb
    k = max(pos, n - pos)
    p = 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return pos, n, float(min(1.0, p))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    suf = f"_{args.tag}" if args.tag else ""
    root = Path(config.RESULTS_DIR) / "adapted"

    rows = []
    for f in sorted(config.COHORT):
        p = root / f"{f}{suf}.json"
        if not p.exists():
            print(f"  [missing] {p}")
            continue
        d = json.loads(p.read_text())
        if "z_score" not in d:
            print(f"  [incomplete] {f}: shuffled-z control unfinished")
            continue
        z = np.load(Path(config.SIGNATURES_DIR) / f / "z_behavior.npz")
        meta = json.loads((Path(config.FOLDS_DIR) / f / "fold.json").read_text())
        c = np.stack([z[q] for q in meta["train_patients"]]).mean(0)   # §3.5 centroid
        rows.append({**d, "atypicality": float(np.linalg.norm(z[f] - c))})

    if not rows:
        print("nothing to summarise")
        return 1

    print(f"\n{'fold':7s} {'baseline':>9s} {'own z':>8s} {'shuf mean':>10s} {'shuf sd':>8s} "
          f"{'z-score':>8s} {'D_p':>7s}")
    for r in rows:
        print(f"{r['fold']:7s} {r['baseline_auc']:9.4f} {r['adapted_auc']:8.4f} "
              f"{r['shuffled_mean']:10.4f} {r['shuffled_sd']:8.4f} {r['z_score']:+8.2f} "
              f"{r['atypicality']:7.2f}")

    own = np.array([r["adapted_auc"] for r in rows])
    shuf = np.array([r["shuffled_mean"] for r in rows])
    base = np.array([r["baseline_auc"] for r in rows])
    zsc = np.array([r["z_score"] for r in rows])
    atyp = np.array([r["atypicality"] for r in rows])

    print(f"\n--- is the patient's OWN signature better than a wrong one? (n={len(rows)}) ---")
    d = own - shuf
    t, pt = t_paired(d)
    pos, n, ps = sign_test(d)
    print(f"  own - shuffled: mean {d.mean():+.4f}  sd {d.std(ddof=1):.4f}")
    print(f"  paired t = {t:+.3f}, p ~ {pt:.3f} (normal approximation)")
    print(f"  sign test: {pos}/{n} folds positive, p = {ps:.3f}")
    print(f"  folds meeting the D28 rule (z > 2.0): "
          f"{int((zsc > 2.0).sum())}/{len(zsc)}")

    print(f"\n--- does adapting at all help, regardless of whose z? ---")
    da = own - base
    ta, pa = t_paired(da)
    print(f"  own - baseline:      mean {da.mean():+.4f}  paired t = {ta:+.3f}, p ~ {pa:.3f}")
    ds = shuf - base
    ts, psb = t_paired(ds)
    print(f"  shuffled - baseline: mean {ds.mean():+.4f}  paired t = {ts:+.3f}, p ~ {psb:.3f}")

    print(f"\n--- section 3.5: benefit vs atypicality D_p = ||z_p - c|| ---")
    for nm, ben in (("own - baseline", da), ("own - shuffled (personalisation only)", d)):
        r = np.corrcoef(atyp, ben)[0, 1]
        print(f"  corr(D_p, {nm:38s}) = {r:+.3f}")
    print("  §3.5 predicts POSITIVE for the benefit it defines (own vs unadapted baseline).")

    out = root / f"cohort_summary{suf}.json"
    out.write_text(json.dumps({"rows": rows, "own_minus_shuffled_mean": float(d.mean()),
                               "paired_t": t, "paired_p_normal_approx": pt,
                               "sign_test": {"positive": pos, "n": n, "p": ps},
                               "n_folds_passing_D28": int((zsc > 2.0).sum())}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
