"""Baseline vs adapted vs shuffled-z on a fold's held-out patient.

The comparison that decides the project. Three conditions on identical clips:

  baseline    dW = 0                       -- the unadapted cohort model
  adapted     dW = HN(z_p), p's own z      -- personalisation as intended
  shuffled-z  dW = HN(z_q) for every q != p -- the control

The shuffled-z condition is evaluated for EVERY other cohort patient, not one, because
a single substitute is a sample of size one and the quantity of interest is how far the
own-z result sits from the distribution of wrong-z results.

[PRE-REGISTERED, D28] Personalisation is claimed only if

    z_score = (AUC_own - mean(AUC_shuffled)) / sd(AUC_shuffled)  >  2.0

with AUC_own > AUC_shuffled in sign. A gain over the BASELINE is not evidence of
personalisation: perturbing a frozen network's weights at all can help, and on fold
pat01's specification run it did -- +0.0129 over baseline, of which +0.0122 was
reproduced by another patient's signature.

    python scripts/evaluate_adapted.py --fold pat01 [--tag gentle]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.data.dataset import VSViGDataset
from src.eval.metrics import within_source_auc
from src.model.adapt import AdaptedForward, base_norms, resolve_targets
from src.model.hypernetwork import Hypernetwork


def _trainer():
    from importlib.util import module_from_spec, spec_from_file_location
    sp = spec_from_file_location(
        "th", str(Path(__file__).resolve().parent / "train_hypernetwork.py"))
    m = module_from_spec(sp); sp.loader.exec_module(m)
    return m


@torch.no_grad()
def score(backbone, targets, deltas, manifest, folder, device):
    dl = DataLoader(VSViGDataset(folder, manifest, eval_mode=True),
                    batch_size=32, shuffle=False, num_workers=0)
    P, Y, S = [], [], []
    for b in dl:
        if deltas is None:
            o = backbone(b[0].to(device), b[1].to(device))
        else:
            with AdaptedForward(targets, deltas):
                o = backbone(b[0].to(device), b[1].to(device))
        if o.dim() > 1:
            o = o.squeeze(1)
        P.append(o.float().cpu().numpy()); Y.append(np.asarray(b[2], dtype=float))
        S += list(b[3])
    p, y = np.concatenate(P), np.concatenate(Y)
    return within_source_auc(p, y, S), p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--tag", default=None, help="checkpoint dir suffix, e.g. 'gentle'")
    args = ap.parse_args()
    fold = args.fold.lower()

    th = _trainer()
    device = th.get_device()
    backbone, ck = th.frozen_backbone(fold, device)
    targets = resolve_targets(backbone); norms = base_norms(targets)
    zs = th.load_z(fold, device)

    hn_dir = Path(config.HYPER_CKPT_ROOT) / (f"{fold}_{args.tag}" if args.tag else fold)
    hn = Hypernetwork().to(device)
    hn.load_state_dict(torch.load(hn_dir / "hypernetwork_best.pth",
                                  map_location=device, weights_only=False))
    hn.eval()

    out = Path(config.OUTPUTS_DIR) / "results" / "adapted" / (
        f"{fold}{'_' + args.tag if args.tag else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(out.read_text()) if out.exists() else {}
    done = cache.get("shuffled_auc", {})

    manifest = f"processed_data/test_sliding/manifest_{fold}.json"
    folder = "processed_data/test_sliding"
    print(f"[env] backbone={ck}\n      hypernetwork={hn_dir/'hypernetwork_best.pth'}")
    print(f"=== fold {fold}, held-out patient {fold} ===")

    if "baseline_auc" in cache:
        base_auc = cache["baseline_auc"]
    else:
        base_auc = score(backbone, targets, None, manifest, folder, device)[0]["patient_balanced"]
    if "adapted_auc" in cache:
        own_auc = cache["adapted_auc"]
    else:
        own_auc = score(backbone, targets, hn(zs[fold], base_norms=norms),
                        manifest, folder, device)[0]["patient_balanced"]
    print(f"  baseline               {base_auc:.4f}")
    print(f"  adapted (own z)        {own_auc:.4f}   "
          f"({own_auc - base_auc:+.4f} vs baseline)")

    shuffled = dict(done)
    for q in sorted(zs.files if hasattr(zs, "files") else zs):
        if q == fold or q in shuffled:
            continue
        r, _ = score(backbone, targets, hn(zs[q], base_norms=norms),
                     manifest, folder, device)
        shuffled[q] = r["patient_balanced"]
        print(f"  shuffled-z ({q})    {shuffled[q]:.4f}")
        cache.update({"fold": fold, "tag": args.tag, "baseline_auc": base_auc,
                      "adapted_auc": own_auc, "shuffled_auc": shuffled})
        out.write_text(json.dumps(cache, indent=2))     # resumable: 9 passes is > 1 call

    need = [q for q in (zs.files if hasattr(zs, "files") else zs) if q != fold]
    if len(shuffled) < len(need):
        print(f"  [partial] {len(shuffled)}/{len(need)} shuffled conditions done; "
              f"re-run to continue")
        return 0
    v = np.array([shuffled[q] for q in sorted(shuffled)])
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    z = (own_auc - mu) / sd if sd > 0 else float("nan")
    verdict = "PERSONALISATION" if (z > 2.0 and own_auc > mu) \
        else "no personalisation detected"
    print(f"\n  shuffled-z: mean {mu:.4f}  sd {sd:.4f}  (n={len(v)})")
    print(f"  own z is {own_auc - mu:+.4f} from that mean, "
          f"z-score {z:+.2f}")
    print(f"  [D28 pre-registered rule: z > 2.0] -> {verdict}")

    cache.update({"fold": fold, "tag": args.tag, "backbone": str(ck),
                  "hypernetwork": str(hn_dir), "baseline_auc": base_auc,
                  "adapted_auc": own_auc, "shuffled_auc": shuffled,
                  "shuffled_mean": mu, "shuffled_sd": sd, "z_score": z,
                  "verdict": verdict})
    out.write_text(json.dumps(cache, indent=2))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
