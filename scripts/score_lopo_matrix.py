"""Score fold models on patients they never trained on.

For each fold, the held-out patient AND the two internal validation patients are
unseen: under D21 nothing was selected on the validation patients, so they are
ordinary held-out measurements. 8 folds x 3 = 24.

usage: score_matrix.py pat01 pat02 ...   (fold names; appends to score_matrix.json)
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path.cwd()))
from torch.utils.data import DataLoader
import config
from src.data.dataset import VSViGDataset
from src.eval.metrics import within_source_auc
from src.model.vsvig import VSViG_base

DEV = torch.device("cpu")
OUT = Path(config.RESULTS_DIR) / "baseline" / "lopo_score_matrix.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

def load(p):
    m = VSViG_base(kpt_channels=config.KPT_CHANNELS)
    st = torch.load(p, map_location=DEV, weights_only=False)
    if isinstance(st, dict): st = st.get("model_state_dict", st)
    m.load_state_dict(st); return m.to(DEV).eval()

@torch.no_grad()
def run(m, clips, folder):
    dl = DataLoader(VSViGDataset(folder, clips, eval_mode=True),
                    batch_size=32, shuffle=False, num_workers=0)
    P, Y, S = [], [], []
    for b in dl:
        o = m(b[0].to(DEV), b[1].to(DEV))
        if o.dim() > 1: o = o.squeeze(1)
        P.append(o.float().cpu().numpy()); Y.append(np.asarray(b[2], dtype=float)); S += list(b[3])
    return np.concatenate(P), np.concatenate(Y), S

res = json.loads(OUT.read_text()) if OUT.exists() else {}
for fold in sys.argv[1:]:
    meta = json.loads((Path(config.FOLDS_DIR) / fold / "fold.json").read_text())
    ck = Path(config.BASELINE_CKPT_ROOT) / fold / "final_model.pth"
    m = load(ck)
    for pat in [meta["test_patient"]] + meta["val_patients"]:
        p, y, s = run(m, f"processed_data/test_sliding/manifest_{pat}.json",
                      "processed_data/test_sliding")
        r = within_source_auc(p, y, s)
        role = "held_out" if pat == meta["test_patient"] else "internal_val"
        res[f"{fold}|{pat}"] = {
            "fold": fold, "patient": pat, "role": role,
            "within_source": r["patient_balanced"], "pooled": r["pooled"],
            "n_ictal": int((y == 1).sum()), "n_interictal": int((y == 0).sum()),
            "mean_prob": float(p.mean()), "frac_above_0.9": float((p > 0.9).mean()),
            "per_source": {k: {"auc": v["auc"], "n_pos": v["n_pos"], "n_neg": v["n_neg"]}
                           for k, v in r["per_source"].items()}}
        print(f"  {fold:6s} -> {pat:6s} [{role:12s}] within-source {r['patient_balanced']:.4f}  "
              f"pooled {r['pooled']:.4f}")
    OUT.write_text(json.dumps(res, indent=2))
