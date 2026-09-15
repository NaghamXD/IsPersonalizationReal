"""D37: score every condition through the CLINICAL pipeline, and keep the scores.

Nine conditions per fold -- the unadapted baseline, the adapted model under the held-out
patient's own z_behavior, and the adapted model under each of the seven other patients'
-- each scored on the fold's Stage 3 sliding-window clips at the fold's D16 threshold,
which is selected on the internal validation patients and applied unchanged to all nine.

Everything here reuses scripts/evaluate.py's own functions rather than reimplementing
the pipeline, so the adapted arm cannot silently drift from the baseline arm. The one
thing it adds is persistence: per-clip scores, labels, sources and clip start times go
to disk for every condition, so any later threshold-dependent analysis is a re-read
rather than another 45 minutes of inference. The Stage 7 run kept only AUCs, and that is
exactly why this run has to exist.

    VSVIG_RUN=alldata python scripts/select_threshold.py --all-folds   # first, once
    python scripts/run_clinical_arms.py --all-folds
    VSVIG_RUN=alldata python scripts/run_clinical_arms.py --all-folds

Interrupted runs resume: a condition whose scores are already on disk is re-scored only
with --restart.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from scripts.evaluate import (build_model, load_fold_threshold, load_onsets,
                              probability_report, run_inference)
from src.eval.metrics import aggregate, evaluate_source
from src.utils.manifest import write_manifest
from src.utils.seeding import seed_everything


def conditions(fold):
    """-> [(name, kind, shuffled_z)] in a fixed order."""
    out = [("baseline", "baseline", None), ("own", "adapted", None)]
    out += [(f"shuffled_{p}", "adapted", p) for p in sorted(config.COHORT) if p != fold]
    return out


def score_condition(fold, name, kind, z_from, onsets, device, out_dir, restart,
                    limit=None):
    npz = out_dir / "scores" / fold / f"{name}.npz"
    if npz.exists() and not restart and not limit:
        d = np.load(npz, allow_pickle=True)
        by = {}
        for src in np.unique(d["source"]):
            m = d["source"] == src
            by[str(src)] = (d["t"][m], d["score"][m], d["label"][m])
        return by, str(d["checkpoint"]), True

    model, activation, ckpt = build_model(kind, fold, device, None, z_from)
    manifest = Path("processed_data/test_sliding") / f"manifest_{fold}.json"
    by = run_inference(model, activation, manifest, device, limit=limit,
                       data_folder="processed_data/test_sliding")
    if limit:                      # a truncated smoke run must never look like a cache
        del model
        return by, str(ckpt), False
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz,
        t=np.concatenate([v[0] for v in by.values()]),
        score=np.concatenate([v[1] for v in by.values()]),
        label=np.concatenate([v[2] for v in by.values()]),
        source=np.concatenate([[k] * len(v[0]) for k, v in by.items()]),
        checkpoint=str(ckpt), condition=name, fold=fold)
    del model
    return by, str(ckpt), False


def clinical(fold, by_source, onsets):
    res, rows = [], []
    for source, (t, s, y) in sorted(by_source.items()):
        eeg_s, clin_s = onsets.get(source, (None, None))
        res.append(evaluate_source(fold, source, t, s, eeg_s=eeg_s, clin_s=clin_s))
        rows += [{"patient": fold, "source": source, "t_start_s": ti,
                  "label": yi, "prob": si} for ti, si, yi in zip(t, s, y)]
    return aggregate(res), probability_report(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default=None)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--restart", action="store_true",
                    help="re-score conditions already on disk")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: cap clips per condition, write nothing to disk")
    args = ap.parse_args()
    if not args.fold and not args.all_folds:
        ap.error("pass --fold patNN or --all-folds")

    seed_everything(config.GLOBAL_SEED)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    onsets = load_onsets()
    out_dir = Path(config.RESULTS_DIR) / "clinical"
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = sorted(config.COHORT) if args.all_folds else [args.fold.lower()]

    print(f"[env] device={device}  run={config.RUN_NAME or '(phase 1)'}  "
          f"backbones={config.BASELINE_CKPT_ROOT}")
    t_all = time.time()
    for i, fold in enumerate(folds, 1):
        # D37: the D16 threshold is required, not optional. An inherited DT makes FDR/h
        # incomparable across folds, and the whole analysis is a difference of rates.
        if not (Path(config.THRESHOLDS_DIR) / f"{fold}.json").exists():
            raise SystemExit(
                f"no selected threshold for {fold} under {config.THRESHOLDS_DIR}. "
                f"Run scripts/select_threshold.py --all-folds for this run first; "
                f"D37 forbids falling back to an inherited DT.")
        dt, rec = load_fold_threshold(fold)
        print(f"\n{'='*70}\n[{i}/{len(folds)}] fold {fold}   DT={dt:.3f}\n{'='*70}")

        rows, t0 = {}, time.time()
        for name, kind, z_from in conditions(fold):
            by, ckpt, cached = score_condition(fold, name, kind, z_from, onsets,
                                               device, out_dir, args.restart,
                                               limit=args.limit)
            agg, probs = clinical(fold, by, onsets)
            rows[name] = {**agg, "checkpoint": ckpt, "z_from": z_from or fold,
                          "decision_threshold": dt, "probabilities": probs}
            print(f"  {name:<16} sens={agg['sensitivity']}  "
                  f"FA={agg['n_false_alarms']:>3} / {agg['exposure_hours']:.3f} h  "
                  f"FDR/h={agg['fdr_per_hour']:.2f}  "
                  f"L_EO={agg['mean_l_eo_s']}" + ("   [cached]" if cached else ""))

        b = rows["baseline"]
        rec_out = {"fold": fold, "decision_threshold": dt,
                   "dt_selected_on": rec["val_patients"] if rec else None,
                   "baseline_false_alarms": b["n_false_alarms"],
                   "exposure_hours": b["exposure_hours"],
                   # D37 flags, computed here so no later analysis has to invent them
                   "zero_baseline_false_alarms": b["n_false_alarms"] == 0,
                   "sensitivity_drop": {
                       n: bool(r["sensitivity"] < b["sensitivity"])
                       for n, r in rows.items() if n != "baseline"},
                   "conditions": rows}
        if args.limit:
            print(f"  [smoke test --limit {args.limit}] nothing written to disk "
                  f"({time.time() - t0:.0f}s)")
        else:
            (out_dir / f"{fold}.json").write_text(
                json.dumps(rec_out, indent=2, default=str))
            print(f"  -> {out_dir / f'{fold}.json'}   ({time.time() - t0:.0f}s)")

    if args.limit:
        print("\nsmoke test only -- no manifest written")
        return 0
    write_manifest(out_dir / "run_manifest.json", seed=config.GLOBAL_SEED,
                   extra={"folds": folds, "conditions_per_fold": 9,
                          "decision": "D37"})
    print(f"\nall {len(folds)} folds in {(time.time() - t_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
