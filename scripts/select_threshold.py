"""D16: choose the decision threshold per fold, on the internal validation patients.

DT must not be inherited from the paper and must not be tuned on the patient it is
scored against. It is selected here on the fold's two internal validation patients --
never on its held-out patient -- and the SAME rule, and the same resulting DT, is then
applied to the baseline and to the adapted model. Section 3.5 is a paired difference;
if the two arms were allowed different thresholds, the difference would partly measure
threshold tuning.

CRITERION. A false-alarm RATE target (the obvious clinical choice, e.g. "lowest DT with
FDR/h <= 1") is not estimable in this corpus and was rejected on the numbers rather
than on taste. Evaluable interictal exposure per patient runs from 0.101 h (pat06) to
0.764 h (pat04), 3.23 h across the whole cohort. A validation PAIR carries ~0.8-1.0 h,
so a single false alarm moves the measured rate by 1.1-1.3 FDR/h. A target of 1/h is
therefore indistinguishable from "zero false alarms anywhere", which drives DT to the
top of the range and destroys sensitivity for reasons of sample size, not of model
quality.

The criterion used instead is Youden's J (TPR - FPR) on the ACCUMULATED score series of
the validation patients -- accumulated, because that is the quantity DT is compared
against at inference. J needs no exposure estimate, is invariant to how many hours each
patient contributes, and is well defined for both arms.

Note on what this costs: the validation patients are, under D21, also reported as
held-out measurements. Selecting DT on them contaminates THEM for threshold-dependent
metrics. So sensitivity, FDR/h and latency are reported only for each fold's held-out
patient (8 patients); the threshold-free AUC continues to use all 24.

    python scripts/select_threshold.py --all-folds
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.data.dataset import VSViGDataset
from src.eval.decision import accumulate, decision_times
from src.model.vsvig import VSViG_base
from src.utils.manifest import write_manifest


def build(fold, device):
    root = Path(config.BASELINE_CKPT_ROOT) / fold
    ck = next((c for c in (root / "final_model.pth", root / "best_model.pth")
               if c.exists()), None)
    if ck is None:
        raise FileNotFoundError(f"no backbone for fold {fold} under {root}")
    m = VSViG_base(kpt_channels=config.KPT_CHANNELS)
    st = torch.load(ck, map_location=device, weights_only=False)
    m.load_state_dict(st.get("model_state_dict", st) if isinstance(st, dict) else st)
    return m.to(device).eval(), ck


@torch.no_grad()
def score_patient(model, patient, device):
    """-> per source: (decision times, accumulated score, label)"""
    ds = VSViGDataset("processed_data/test_sliding",
                      f"processed_data/test_sliding/manifest_{patient}.json",
                      eval_mode=True)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    by = {}
    for b in dl:
        out = model(b[0].to(device), b[1].to(device))
        if out.dim() > 1:
            out = out.squeeze(1)
        for i, p in enumerate(out.float().cpu().numpy()):
            t, s, y = by.setdefault(b[3][i], ([], [], []))
            t.append(float(b[4][i])); s.append(float(p)); y.append(float(b[2][i]))
    out = {}
    for src, (t, s, y) in by.items():
        t, s, y = np.array(t), np.array(s), np.array(y)
        o = np.argsort(t)
        dt, ap = accumulate(decision_times(t[o]), s[o])
        out[src] = (dt, ap, y[o])
    return out


def sweep(series, grid):
    """Youden's J over the accumulated series of every validation source."""
    ap = np.concatenate([a for _, a, _ in series.values()])
    y = np.concatenate([lab for _, _, lab in series.values()])
    pos, neg = ap[y == 1.0], ap[y == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("validation patients carry only one class; cannot select DT")
    rows = []
    for dt in grid:
        tpr = float((pos > dt).mean()); fpr = float((neg > dt).mean())
        rows.append({"dt": float(dt), "tpr": tpr, "fpr": fpr, "youden_j": tpr - fpr})
    best = max(rows, key=lambda r: (r["youden_j"], -r["dt"]))
    return best, rows, int(len(pos)), int(len(neg))


def main():
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--fold", type=str, default=None)
    ap_.add_argument("--all-folds", action="store_true")
    ap_.add_argument("--grid", type=int, default=199)
    args = ap_.parse_args()
    if not args.fold and not args.all_folds:
        ap_.error("pass --fold patNN or --all-folds")

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    grid = np.linspace(0.005, 0.995, args.grid)
    out_root = Path(config.OUTPUTS_DIR) / "thresholds"
    out_root.mkdir(parents=True, exist_ok=True)

    for fold in (sorted(config.COHORT) if args.all_folds else [args.fold.lower()]):
        meta = json.loads((Path(config.FOLDS_DIR) / fold / "fold.json").read_text())
        val = meta["val_patients"]
        assert meta["test_patient"] not in val, "held-out patient must not select DT"
        model, ck = build(fold, device)
        series = {}
        for p in val:
            for src, v in score_patient(model, p, device).items():
                series[src] = v
        best, rows, n_pos, n_neg = sweep(series, grid)
        rec = {"fold": fold, "backbone": str(ck), "val_patients": val,
               "criterion": "max Youden J on accumulated score, internal validation patients",
               "dt": best["dt"], "tpr_at_dt": best["tpr"], "fpr_at_dt": best["fpr"],
               "youden_j": best["youden_j"], "n_ictal_clips": n_pos,
               "n_interictal_clips": n_neg, "grid": rows}
        (out_root / f"{fold}.json").write_text(json.dumps(rec, indent=2))
        print(f"  {fold}: DT={best['dt']:.3f}  J={best['youden_j']:.3f}  "
              f"(TPR {best['tpr']:.2f}, FPR {best['fpr']:.2f}) "
              f"from {val} — {n_pos} ictal / {n_neg} interictal clips")
    write_manifest(out_root / "run_manifest.json", seed=config.GLOBAL_SEED,
                   extra={"stage": "D16_threshold_selection"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
