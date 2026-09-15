"""Score ONE model (fold pat01's) on every patient's sliding-window test clips.

pat01 is held out. pat04 and pat07 were used only for checkpoint selection -- the
model never trained on their clips either, so they are also unseen. pat02/03/06/08/09
are training patients and give the memorisation ceiling.
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
OUT = Path.home() / "scratch" / "score_all.json"

def load(p):
    m = VSViG_base(kpt_channels=config.KPT_CHANNELS)
    st = torch.load(p, map_location=DEV, weights_only=False)
    if isinstance(st, dict): st = st.get("model_state_dict", st)
    m.load_state_dict(st); return m.to(DEV).eval()

@torch.no_grad()
def run(m, clips, folder):
    ds = VSViGDataset(folder, clips, eval_mode=True)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    P, Y, S = [], [], []
    for b in dl:
        o = m(b[0].to(DEV), b[1].to(DEV))
        if o.dim() > 1: o = o.squeeze(1)
        P.append(o.float().cpu().numpy()); Y.append(np.asarray(b[2], dtype=float)); S += list(b[3])
    return np.concatenate(P), np.concatenate(Y), S

if __name__ == "__main__":
    m = load(Path(config.BASELINE_CKPT_ROOT) / "pat01" / "final_model.pth")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for pat in sys.argv[1:]:
        p, y, s = run(m, f"processed_data/test_sliding/manifest_{pat}.json",
                      "processed_data/test_sliding")
        r = within_source_auc(p, y, s)
        res[pat] = {"within_source": r["patient_balanced"], "pooled": r["pooled"],
                    "n_clips": int(len(p)), "n_ictal": int((y == 1).sum()),
                    "n_interictal": int((y == 0).sum()),
                    "mean_prob": float(p.mean()), "sd_prob": float(p.std()),
                    "frac_above_0.9": float((p > 0.9).mean()),
                    "per_source": {k: {"auc": v["auc"], "n_pos": v["n_pos"],
                                       "n_neg": v["n_neg"]}
                                   for k, v in r["per_source"].items()}}
        print(f"{pat}: within-source {r['patient_balanced']:.4f}  pooled {r['pooled']:.4f}  "
              f"({len(p)} clips, {int((y==1).sum())} ictal)")
    OUT.write_text(json.dumps(res, indent=2))
