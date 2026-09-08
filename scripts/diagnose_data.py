"""Is there signal in the extracted clips at all?

Run this before tuning anything. A model that cannot fit its own training set has a
problem upstream of the optimiser, and the assumption never verified in this pipeline
is that pose estimation actually found the patient. `preprocess.py` reporting "0
failed" only means the clips decoded -- a clip of fifteen black patches around
unresolved joints decodes perfectly.

    python scripts/diagnose_data.py                 # sample across the cohort
    python scripts/diagnose_data.py --patient pat01 --n 200
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import config
from src.data.normalize import normalize_skeleton
from src.utils.naming import patient_of

JOINTS = ["nose", "L-eye", "R-eye", "R-sho", "R-elb", "R-wri", "L-sho", "L-elb",
          "L-wri", "R-hip", "R-kne", "R-ank", "L-hip", "L-kne", "L-ank"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", type=str, default=None)
    ap.add_argument("--n", type=int, default=60, help="clips sampled per patient")
    ap.add_argument("--data-folder", type=str, default=str(config.PROCESSED_DIR))
    args = ap.parse_args()

    labels = json.loads(Path(config.LABELS_JSON).read_text())
    by_pat = defaultdict(list)
    for name in labels:
        by_pat[patient_of(name)].append(name)
    pats = [args.patient.lower()] if args.patient else config.COHORT

    p_dir = Path(args.data_folder) / "patches"
    k_dir = Path(args.data_folder) / "kpts"
    rng = random.Random(config.GLOBAL_SEED)

    print(f"{'patient':<9}{'clips':>6}{'kpt valid':>11}{'patch!=0':>10}"
          f"{'patch mean':>12}{'patch std':>11}{'dead joints':>13}")
    print("-" * 74)
    joint_valid_all = np.zeros(config.N_JOINTS)
    n_pat = 0
    for p in pats:
        names = by_pat.get(p, [])
        if not names:
            continue
        sample = rng.sample(names, min(args.n, len(names)))
        kv, nz, means, stds = [], [], [], []
        jv = np.zeros(config.N_JOINTS)
        for nm in sample:
            k = torch.load(k_dir / f"{nm}.pt", map_location="cpu").float()
            valid = ((k[:, :, 0] >= 0) & (k[:, :, 1] >= 0)).numpy()   # (T, P)
            kv.append(valid.mean())
            jv += valid.mean(axis=0)

            x = torch.load(p_dir / f"{nm}.pt", map_location="cpu").float()
            if x.max() > 2.0:
                x = x / 255.0
            # A patch is "dead" if it is all zeros -- the joint was unresolved, so
            # preprocess.py left the slot black.
            per_patch = x.reshape(x.shape[0], x.shape[1], -1)
            nz.append(float((per_patch.abs().sum(-1) > 0).float().mean()))
            means.append(float(x.mean()))
            stds.append(float(x.std()))
        jv /= len(sample)
        joint_valid_all += jv
        n_pat += 1
        dead = [JOINTS[i] for i in range(config.N_JOINTS) if jv[i] < 0.25]
        print(f"{p:<9}{len(sample):>6}{np.mean(kv):>10.1%}{np.mean(nz):>10.1%}"
              f"{np.mean(means):>12.4f}{np.mean(stds):>11.4f}   "
              f"{','.join(dead) if dead else '-'}")

    if n_pat:
        jv = joint_valid_all / n_pat
        print(f"\nper-joint tracking rate (cohort mean):")
        for i in range(config.N_JOINTS):
            bar = "#" * int(round(jv[i] * 40))
            print(f"  {JOINTS[i]:<7}{jv[i]:>7.1%}  {bar}")

    # What the model actually receives after normalisation.
    print(f"\nnormalised keypoints ({config.KPT_NORMALISATION}, "
          f"{config.KPT_CHANNELS} channels), one sample clip:")
    nm = rng.choice(by_pat[pats[0]])
    k = torch.load(k_dir / f"{nm}.pt", map_location="cpu")
    kn = normalize_skeleton(k)
    frac_zero = float((kn.abs().sum(-1) == 0).float().mean())
    print(f"  {nm}: shape={tuple(kn.shape)}  range=[{kn.min():.2f},{kn.max():.2f}]  "
          f"mean={kn.mean():.3f}  exactly-zero joints={frac_zero:.1%}")
    if frac_zero > 0.4:
        print("  WARNING: most joints normalise to exactly 0 (unresolved). The "
              "positional embedding is then carrying almost no information.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
