"""Can z_behavior identify the patient it came from?

This is the precondition for everything downstream. The hypernetwork conditions on
z_behavior; section 3.5 correlates benefit against distance in z-space; Stage 8's
shuffled-z control asks whether the RIGHT patient's z matters. If z cannot separate
patients at all, then conditioning on it is conditioning on noise -- and the shuffled-z
control becomes indistinguishable from the real thing by construction, so the
experiment would be unable to detect its own failure.

The test: split each patient's Pool A into contiguous temporal blocks, compute a z per
block, then identify each block by nearest patient centroid, leaving that block out of
its own centroid. Chance is 1/n_patients. Significance by label permutation.

    python scripts/probe_signature_identity.py --all-folds
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from src.model.signature import StaticContextProjector, compute_z_behavior
from src.utils.seeding import seed_everything
from scripts.build_signatures import (get_device, load_backbone, make_loader,
                                      temporal_blocks)


def identify(blocks, rng=None, n_perm=2000):
    """blocks: {patient: [z, ...]}. Leave-one-block-out nearest-centroid accuracy."""
    pats = sorted(blocks)
    X, y = [], []
    for i, p in enumerate(pats):
        for b in blocks[p]:
            X.append(np.asarray(b, dtype=float)); y.append(i)
    X, y = np.stack(X), np.array(y)

    def acc(labels):
        hit = 0
        for i in range(len(X)):
            cents = []
            for c in range(len(pats)):
                m = (labels == c)
                m_loo = m.copy(); m_loo[i] = False       # never use the query itself
                cents.append(X[m_loo].mean(0) if m_loo.any() else np.full(X.shape[1], np.inf))
            d = np.linalg.norm(np.stack(cents) - X[i], axis=1)
            hit += int(d.argmin() == labels[i])
        return hit / len(X)

    # per-patient recall, so it is visible whether the patients the baseline struggles
    # with are also the ones whose signature is recoverable -- that pairing is what
    # decides whether conditioning can help where it needs to.
    per = {}
    for c, pname in enumerate(pats):
        idx = np.where(y == c)[0]
        hit = 0
        for i in idx:
            cents = []
            for cc in range(len(pats)):
                m = (y == cc); m_loo = m.copy(); m_loo[i] = False
                cents.append(X[m_loo].mean(0) if m_loo.any() else np.full(X.shape[1], np.inf))
            hit += int(np.linalg.norm(np.stack(cents) - X[i], axis=1).argmin() == c)
        per[pname] = hit / len(idx)

    observed = acc(y)
    rng = rng or np.random.default_rng(config.GLOBAL_SEED)
    null = np.array([acc(rng.permutation(y)) for _ in range(n_perm)])
    p = float((null >= observed).sum() + 1) / (n_perm + 1)
    return {"accuracy": observed, "per_patient_recall": per,
            "chance": 1.0 / len(pats), "p_value": p,
            "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "n_blocks": int(len(X)), "n_patients": len(pats)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=str, default=None)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()
    if not args.fold and not args.all_folds:
        ap.error("pass --fold patNN or --all-folds")

    seed_everything(config.GLOBAL_SEED)
    device = get_device()
    load = make_loader(config.PROCESSED_DIR)
    projector = StaticContextProjector().to(device)
    pools = {p: json.loads((Path(config.POOLS_DIR) / f"{p}.json").read_text())
             for p in config.COHORT}

    out = {}
    for fold in (sorted(config.COHORT) if args.all_folds else [args.fold.lower()]):
        backbone, ckpt = load_backbone(fold, device)
        blocks = {p: [compute_z_behavior(backbone, projector, b, load, device).cpu().numpy()
                      for b in temporal_blocks(pools[p]["pool_a"],
                                               config.STABILITY_N_BLOCKS)]
                  for p in config.COHORT}
        np.savez(Path(config.OUTPUTS_DIR) / "signatures" / fold / "z_blocks.npz",
                 **{p: np.stack(v) for p, v in blocks.items()})
        r = identify(blocks, n_perm=args.n_perm)
        out[fold] = r
        print(f"  {fold}: {r['accuracy']:.1%} of {r['n_blocks']} blocks "
              f"(chance {r['chance']:.1%}, null mean {r['null_mean']:.1%}, "
              f"95th pct {r['null_p95']:.1%})  p = {r['p_value']:.4f}")

    dest = Path(config.OUTPUTS_DIR) / "signatures" / "identity_probe.json"
    prev = json.loads(dest.read_text()) if dest.exists() else {}
    prev.update(out); dest.write_text(json.dumps(prev, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
