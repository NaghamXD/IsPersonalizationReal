"""Dummy forward pass through the backbone, reporting the true tensor shape at every
stage boundary and the flat index of every Part_3DCNN.

Two things this settles empirically rather than by reading the constructor:
  1. The signature cut. config.SIGNATURE_CHANNELS asserts C'=192 after stages 0-2.
     If this script disagrees, config is wrong and the projector dimension is wrong.
  2. The modulation targets. config.HN_TARGET_SPECS asserts the three Stage-3
     Part_3DCNN.conv2 layers are backbone indices 25/27/29 with weight 192x192x3x3x1.

Run:  python scripts/verify_shapes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import config
from src.model.vsvig import VSViG_base, Part_3DCNN

FAIL = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def main():
    model = VSViG_base().eval()

    B, T, P = 2, config.CLIP_FRAMES, config.N_JOINTS
    patches = torch.randn(B, T, P, 3, config.PATCH_SIZE, config.PATCH_SIZE)
    kpts = torch.rand(B, T, P, config.KPT_CHANNELS)

    print("=== module inventory ===")
    part3d = [(i, m) for i, m in enumerate(model.backbone) if isinstance(m, Part_3DCNN)]
    for i, m in part3d:
        w = m.conv2[0].weight
        print(f"  [{i:>2}] Part_3DCNN  conv2.weight={tuple(w.shape)}  stride={m.conv2[0].stride}")

    print("\n=== forward, shape at every backbone module ===")
    with torch.no_grad():
        x = model.stem(patches)
        print(f"  stem out            {tuple(x.shape)}   (B,T,P,C)")
        x = model.pe(model.pos_emb, x, kpts)
        Bs, Ts, Ps, C = x.shape
        x = x.transpose(2, 3).contiguous().view(Bs, Ts, C, Ps, 1)

        stage_cut_shape = None
        # Stage boundaries, derived the same way STViG builds them.
        num_layer = [2, 2, 6, 2]
        idx = 0
        boundaries = {}
        for stage, n in enumerate(num_layer):
            if stage > 0:
                idx += 2                 # transition Grapher + Part_3DCNN
            idx += 2 * n                 # n x (Grapher + Part_3DCNN)
            boundaries[idx - 1] = stage  # last module index of this stage

        for i, module in enumerate(model.backbone):
            x = module(x)
            if i in boundaries:
                st = boundaries[i]
                Bq, Tq, Cq, Pq, _ = x.shape
                print(f"  end of stage {st}      C={Cq:<4} T={Tq:<3} P={Pq}   (module {i})")
                if st == config.SIGNATURE_STAGE_CUT:
                    stage_cut_shape = (Cq, Tq, Pq)

    print("\n=== assertions ===")
    check(stage_cut_shape is not None, "stage cut reached")
    if stage_cut_shape:
        Cq, Tq, Pq = stage_cut_shape
        check(Cq == config.SIGNATURE_CHANNELS,
              f"C' after stages 0-{config.SIGNATURE_STAGE_CUT} == {config.SIGNATURE_CHANNELS} (got {Cq})")
        check(Pq == config.N_JOINTS, f"P == {config.N_JOINTS} (got {Pq})")
        print(f"         [info] T' at the cut = {Tq}; sigma is taken across this axis")
        check(config.PROJECTOR_IN_DIM == 2 * Cq,
              f"projector in_dim == 2*C' == {2 * Cq} (config says {config.PROJECTOR_IN_DIM})")

    expected = {n: (o, i) for n, o, i, _ in config.HN_TARGET_SPECS
                if n.startswith("stage3")}
    idxs = [i for i, _ in part3d][-3:]
    check(idxs == [25, 27, 29], f"Stage-3 Part_3DCNN indices == [25,27,29] (got {idxs})")
    for i in idxs:
        w = dict(part3d)[i].conv2[0].weight
        flat = w.shape[1] * w.shape[2] * w.shape[3] * w.shape[4]
        check(w.shape[0] == 192 and flat == 1728,
              f"backbone[{i}].conv2 is 192 x 1728 (got {w.shape[0]} x {flat})")

    fc0, fc3 = model.fc[0], model.fc[3]
    check(tuple(fc0.weight.shape) == (256, 384, 1, 1), f"fc[0] is 256x384x1x1 (got {tuple(fc0.weight.shape)})")
    check(tuple(fc3.weight.shape) == (1, 256, 1, 1), f"fc[3] is 1x256x1x1 (got {tuple(fc3.weight.shape)})")

    with torch.no_grad():
        out = model(patches, kpts)
    check(out.shape == (B,), f"model output is scalar per clip (got {tuple(out.shape)})")

    print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
