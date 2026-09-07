"""Stage 5 (runs after Stage 6) -- z_behavior for every patient, and the §3.2.3 gate.

Ordering note: z_behavior is computed from a FROZEN, PER-FOLD backbone, so this cannot
run until scripts/train_backbone.py has produced those checkpoints. Each fold's
backbone was trained without its test patient, so the same patient gets a different z
in different folds -- that is correct and load-bearing, not an inconsistency.

[METHOD] §3.2.3 is a GATE. "A Stability Ratio of >= 2.5 guarantees that the spatial
motor signature successfully dominates arbitrary intra-patient activity shifts." If a
patient's signature is not stable across their own day, conditioning on it cannot work,
and learning that here is far cheaper than after eight folds of hypernetwork training.

    python scripts/build_signatures.py --fold pat01
    python scripts/build_signatures.py --all-folds
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from src.data.dataset import MEAN, STD
from src.data.normalize import normalize_skeleton
from src.eval.stability import stability_ratio
from src.model.signature import StaticContextProjector, compute_z_behavior
from src.model.vsvig import VSViG_base
from src.utils.manifest import write_manifest
from src.utils.seeding import seed_everything


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(data_folder):
    """names -> (patches, kpts) on device, using the dataset's exact normalisation.

    Injected into compute_z_behavior so it works for any clip store without the
    signature code knowing which one it is reading.
    """
    p_dir, k_dir = Path(data_folder) / "patches", Path(data_folder) / "kpts"

    def load(names, device):
        d, k = [], []
        for n in names:
            x = torch.load(p_dir / f"{n}.pt", map_location="cpu").float()
            if x.max() > 2.0:
                x = x / 255.0
            d.append((x - MEAN) / STD)
            k.append(normalize_skeleton(torch.load(k_dir / f"{n}.pt", map_location="cpu")))
        return torch.stack(d).to(device), torch.stack(k).to(device)
    return load


def load_backbone(patient, device):
    ckpt = Path(config.BASELINE_CKPT_ROOT) / patient / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No backbone for fold {patient} at {ckpt}. "
            f"Run: python scripts/train_backbone.py --fold {patient}")
    model = VSViG_base(kpt_channels=config.KPT_CHANNELS)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def temporal_blocks(pool_a, n_blocks):
    """Split Pool A into contiguous, NON-OVERLAPPING temporal blocks [METHOD].

    Pool A is already ordered by (source, start time), so contiguous slices are
    temporally coherent -- which is the point: each block should represent a different
    stretch of the patient's day.
    """
    if len(pool_a) < n_blocks:
        return [pool_a]
    edges = np.linspace(0, len(pool_a), n_blocks + 1).round().astype(int)
    return [pool_a[a:b] for a, b in zip(edges, edges[1:]) if b > a]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=str, default=None)
    ap.add_argument("--all-folds", action="store_true")
    args = ap.parse_args()
    if not args.fold and not args.all_folds:
        ap.error("pass --fold patNN or --all-folds")

    seed_everything(config.GLOBAL_SEED)
    device = get_device()
    load = make_loader(config.PROCESSED_DIR)
    projector = StaticContextProjector().to(device)
    print(f"[env] device={device}  cut=stages 0-{config.SIGNATURE_STAGE_CUT} "
          f"(C'={config.SIGNATURE_CHANNELS})  projector "
          f"{config.PROJECTOR_IN_DIM}->{config.CONTEXT_DIM}  "
          f"Pool A N={config.POOL_A_SIZE}")

    pools = {p: json.loads((Path(config.POOLS_DIR) / f"{p}.json").read_text())
             for p in config.COHORT}
    folds = sorted(config.COHORT) if args.all_folds else [args.fold.lower()]
    out_root = Path(config.OUTPUTS_DIR) / "signatures"
    out_root.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        try:
            backbone, ckpt = load_backbone(fold, device)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue
        print(f"\n=== fold {fold} (backbone {ckpt}) ===")

        z_all, blocks_all = {}, {}
        for p in config.COHORT:
            pool_a = pools[p]["pool_a"]
            z, mu, sigma = compute_z_behavior(backbone, projector, pool_a, load,
                                              device, return_parts=True)
            z_all[p] = z.cpu().numpy()
            blocks_all[p] = [
                compute_z_behavior(backbone, projector, b, load, device).cpu().numpy()
                for b in temporal_blocks(pool_a, config.STABILITY_N_BLOCKS)]
            print(f"  {p}: |z|={np.linalg.norm(z_all[p]):.3f}  "
                  f"|mu|={mu.norm():.2f} |sigma|={sigma.norm():.2f}  "
                  f"blocks={len(blocks_all[p])}")

        # ---- §3.2.3 gate -------------------------------------------------
        stats = stability_ratio(blocks_all)
        spans = {p: _span(pools[p]["pool_a"]) for p in config.COHORT}
        print(f"\n  --- stability (threshold {stats['threshold']}) ---")
        print(f"  {'patient':<9}{'ratio':>8}{'intra':>9}{'PoolA span':>13}")
        passed = []
        for p in config.COHORT:
            r = stats["ratio_per_patient"][p]
            ok = r >= stats["threshold"]
            passed.append(ok)
            print(f"  {p:<9}{r:>8.2f}{stats['intra_per_patient'][p]:>9.3f}"
                  f"{spans[p]:>11.0f}s   {'pass' if ok else 'FAIL'}")
        print(f"  pooled ratio {stats['pooled_ratio']:.2f}  "
              f"({sum(passed)}/{len(passed)} patients pass individually)")
        print("  Read per patient, not pooled: Pool A spans differ ~24x across this "
              "cohort, so a short span shrinks intra-patient distance for reasons "
              "unrelated to signature stability (DECISIONS.md D14).")

        out = out_root / fold
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "z_behavior.npz", **z_all)
        (out / "stability.json").write_text(json.dumps({
            "fold": fold, "backbone": str(ckpt),
            "pool_a_span_s": spans,
            "inter_mean": stats["inter_mean"],
            "intra_per_patient": stats["intra_per_patient"],
            "ratio_per_patient": stats["ratio_per_patient"],
            "pooled_ratio": stats["pooled_ratio"],
            "threshold": stats["threshold"],
            "n_passing": int(sum(passed)), "n_patients": len(passed),
        }, indent=2))
        write_manifest(out / "run_manifest.json", seed=config.GLOBAL_SEED,
                       extra={"stage": "5_signatures", "fold": fold})
        print(f"  wrote {out}/")
    return 0


def _span(pool_a):
    from src.utils.naming import parse_clip_name
    by_src = {}
    for n in pool_a:
        _, s, t = parse_clip_name(n)
        by_src.setdefault(s, []).append(t)
    return float(sum(max(v) - min(v) for v in by_src.values()))


if __name__ == "__main__":
    raise SystemExit(main())
