"""What does the 384 -> 128 random projection cost z_behavior?

§3.2.2 makes the projection a deliberate "structural information bottleneck ...
preventing the downstream hypernetwork from memorizing categorical patient IDs". That
is an argument for throwing information away on purpose. This measures how much is
actually thrown away, by running the D23 identification probe on:

  * the raw 384-d [mu || sigma], no projection at all
  * mu alone (192-d) and sigma alone (192-d), to see which half carries identity
  * random projections at 256, 128 (the configured value), 64, 32, 16, 8

If identification barely moves between 384 and 128, the bottleneck is not costing
anything and the design is free. If it drops sharply, the bottleneck is buying its
anti-memorisation property with signal the hypernetwork needs -- and that trade should
be made knowingly.

    python scripts/probe_projection_dim.py --all-folds
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from src.model.signature import (StaticContextProjector, clip_mu_sigma,
                                 stage_cut_features)
from src.utils.seeding import seed_everything
from scripts.build_signatures import get_device, load_backbone, make_loader, temporal_blocks
from scripts.probe_signature_identity import identify

DIMS = [256, 128, 64, 32, 16, 8]


@torch.no_grad()
def raw_parts(backbone, clips, load, device, batch_size=8):
    mus, sigmas = [], []
    for i in range(0, len(clips), batch_size):
        patches, kpts = load(clips[i:i + batch_size], device)
        mu, sigma = clip_mu_sigma(stage_cut_features(backbone, patches, kpts))
        mus.append(mu); sigmas.append(sigma)
    return (torch.cat(mus).mean(0).reshape(-1).cpu().numpy(),
            torch.cat(sigmas).mean(0).reshape(-1).cpu().numpy())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=str, default=None)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--n-perm", type=int, default=300)
    args = ap.parse_args()
    if not args.fold and not args.all_folds:
        ap.error("pass --fold patNN or --all-folds")

    seed_everything(config.GLOBAL_SEED)
    device = get_device()
    load = make_loader(config.PROCESSED_DIR)
    pools = {p: json.loads((Path(config.POOLS_DIR) / f"{p}.json").read_text())
             for p in config.COHORT}

    dest = Path(config.OUTPUTS_DIR) / "signatures" / "projection_dim_probe.json"
    out = json.loads(dest.read_text()) if dest.exists() else {}

    for fold in (sorted(config.COHORT) if args.all_folds else [args.fold.lower()]):
        backbone, _ = load_backbone(fold, device)
        mu_b, sg_b = {}, {}
        for p in config.COHORT:
            mu_b[p], sg_b[p] = [], []
            for b in temporal_blocks(pools[p]["pool_a"], config.STABILITY_N_BLOCKS):
                m, s = raw_parts(backbone, b, load, device)
                mu_b[p].append(m); sg_b[p].append(s)

        variants = {
            "raw_384_mu+sigma": {p: [np.concatenate([m, s]) for m, s in
                                     zip(mu_b[p], sg_b[p])] for p in config.COHORT},
            "mu_only_192": mu_b,
            "sigma_only_192": sg_b,
        }
        base = variants["raw_384_mu+sigma"]
        for d in DIMS:
            proj = StaticContextProjector(in_dim=config.PROJECTOR_IN_DIM, context_dim=d,
                                          seed=config.PROJECTOR_SEED)
            W = proj.proj.weight.numpy()
            variants[f"proj_{d}"] = {p: [W @ v for v in base[p]] for p in config.COHORT}

        res = {}
        for name, blocks in variants.items():
            r = identify(blocks, n_perm=args.n_perm)
            res[name] = {"accuracy": r["accuracy"], "p_value": r["p_value"],
                         "per_patient_recall": r["per_patient_recall"]}
            print(f"  {fold:7s} {name:20s} {r['accuracy']:6.1%}  p={r['p_value']:.4f}")
        out[fold] = res
        dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
