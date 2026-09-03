"""Parity test for skeleton normalisation.

`reference_normalize` is a pure-numpy transcription of the intended algorithm, written
independently of the torch implementation and validated against hand-computed cases in
`test_reference_hand_cases`. `test_torch_matches_reference` then holds the shipped
torch code to it. If the two drift apart, one of them is wrong and this fails.

Run:  python -m pytest tests/test_normalize.py -v
      (or plain `python tests/test_normalize.py` for the numpy-only checks)
"""
import numpy as np

HIP = (9, 12)
SHO = (3, 6)
SENTINEL = -1.0


def _midpoint_np(xy, valid, a, b):
    pa, pb = xy[:, a, :], xy[:, b, :]
    va, vb = valid[:, a], valid[:, b]
    point = np.zeros_like(pa)
    both = va & vb
    point[both] = 0.5 * (pa[both] + pb[both])
    point[va & ~vb] = pa[va & ~vb]
    point[vb & ~va] = pb[vb & ~va]
    return point, (va | vb)


def reference_normalize(kpts, frame_w=1920, frame_h=1080, torso_floor=1e-3,
                        torso_fallback_frac=0.25, clamp=5.0, channels=2):
    kpts = np.asarray(kpts, dtype=np.float64)
    xy = kpts[:, :, :2]
    valid = (xy[:, :, 0] >= 0) & (xy[:, :, 1] >= 0)

    midhip, hip_ok = _midpoint_np(xy, valid, *HIP)
    midsh, sh_ok = _midpoint_np(xy, valid, *SHO)

    if hip_ok.any():
        fallback_origin = np.median(midhip[hip_ok], axis=0)
    else:
        fallback_origin = np.array([frame_w / 2.0, frame_h / 2.0])
    origin = np.where(hip_ok[:, None], midhip, fallback_origin[None, :])

    torso = np.linalg.norm(midsh - midhip, axis=-1)
    torso_ok = hip_ok & sh_ok & (torso > torso_floor)
    fallback_scale = np.median(torso[torso_ok]) if torso_ok.any() \
        else torso_fallback_frac * frame_h
    scale = np.where(torso_ok, torso, fallback_scale)
    scale = np.maximum(scale, torso_floor)

    out = (xy - origin[:, None, :]) / scale[:, None, None]
    out = np.clip(out, -clamp, clamp) * valid[:, :, None]
    if channels == 3:
        conf = kpts[:, :, 2:3] if kpts.shape[-1] >= 3 else valid[:, :, None].astype(float)
        return np.concatenate([out, np.clip(conf, 0, 1)], axis=-1)
    return out


def _skeleton(hip_y=600.0, sh_y=400.0, hip_dx=40.0):
    """One frame, 15 joints, upright and symmetric about x=960.
    mid-hip = (960, hip_y); mid-shoulder = (960, sh_y); torso = hip_y - sh_y."""
    k = np.zeros((1, 15, 3))
    k[0, :, 2] = 1.0
    k[0, 0] = [960, 300, 1]; k[0, 1] = [950, 290, 1]; k[0, 2] = [970, 290, 1]
    k[0, 3] = [960 - 60, sh_y, 1]; k[0, 4] = [900, 500, 1]; k[0, 5] = [890, 560, 1]
    k[0, 6] = [960 + 60, sh_y, 1]; k[0, 7] = [1020, 500, 1]; k[0, 8] = [1030, 560, 1]
    k[0, 9] = [960 - hip_dx, hip_y, 1]; k[0, 10] = [920, 800, 1]; k[0, 11] = [915, 950, 1]
    k[0, 12] = [960 + hip_dx, hip_y, 1]; k[0, 13] = [1000, 800, 1]; k[0, 14] = [1005, 950, 1]
    return k


def test_reference_hand_cases():
    # Torso = 600-400 = 200 px. Mid-hip sits at the origin by construction.
    out = reference_normalize(_skeleton())
    assert np.allclose(out[0, 9] + out[0, 12], 0.0), "hips must straddle the origin"
    # Nose is 300 px above the hip -> -300/200 = -1.5 torso lengths.
    assert np.isclose(out[0, 0, 1], (300 - 600) / 200.0), out[0, 0, 1]
    # R shoulder is 60 px left and 200 px above -> (-0.3, -1.0)
    assert np.allclose(out[0, 3], [-60 / 200.0, -1.0]), out[0, 3]

    # TRANSLATION INVARIANCE: shifting the whole patient must not change the output.
    shifted = _skeleton(); shifted[:, :, 0] += 300; shifted[:, :, 1] -= 120
    assert np.allclose(reference_normalize(shifted), out, atol=1e-9), \
        "normalisation must be invariant to where the patient is in frame"

    # SCALE INVARIANCE: a patient imaged 1.5x larger must normalise identically.
    # Scale about the mid-hip, not the frame origin: scaling about (0,0) would drive
    # the upper-body coordinates negative, where they are correctly read as
    # unresolved joints rather than as a larger patient.
    centre = np.array([960.0, 600.0])
    big = _skeleton(); big[:, :, :2] = (big[:, :, :2] - centre) * 1.5 + centre
    assert (big[:, :, :2] >= 0).all(), "test fixture drove a coordinate negative"
    assert np.allclose(reference_normalize(big), reference_normalize(_skeleton()),
                       atol=1e-9), "normalisation must be invariant to apparent size"

    # MISSING JOINTS go to exactly the origin, and do not poison their neighbours.
    miss = _skeleton(); miss[0, 5] = [SENTINEL, SENTINEL, 0.0]
    om = reference_normalize(miss)
    assert np.allclose(om[0, 5], 0.0), "unresolved joint must be zeroed"
    assert np.allclose(om[0, 4], out[0, 4]), "a missing joint must not shift others"

    # ONE HIP MISSING: the remaining hip becomes the origin, no NaN.
    onehip = _skeleton(); onehip[0, 9] = [SENTINEL, SENTINEL, 0.0]
    oh = reference_normalize(onehip)
    assert np.isfinite(oh).all()
    assert np.allclose(oh[0, 12], 0.0), "sole valid hip becomes the origin"

    # TOTAL POSE FAILURE: all joints missing -> all zeros, still finite.
    dead = np.full((1, 15, 3), SENTINEL); dead[:, :, 2] = 0.0
    od = reference_normalize(dead)
    assert np.isfinite(od).all() and np.allclose(od, 0.0)

    # DEGENERATE TORSO (shoulders on top of hips) must not divide by ~0.
    flat = _skeleton(sh_y=600.0)
    of = reference_normalize(flat)
    assert np.isfinite(of).all() and np.abs(of).max() <= 5.0
    # OFF-FRAME EXTRAPOLATION: preprocess.py's velocity imputation can predict a
    # joint past the left/top edge. It must be read as unresolved, not as real.
    off = _skeleton(); off[0, 5] = [-12.0, 400.0, 0.4]
    oo = reference_normalize(off)
    assert np.allclose(oo[0, 5], 0.0), "negative coordinate must count as unresolved"

    print("reference hand cases OK")


def test_torch_matches_reference():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import torch
    import config
    from src.data.normalize import normalize_skeleton

    rng = np.random.default_rng(0)
    for trial in range(20):
        k = rng.uniform(0, 1080, size=(config.CLIP_FRAMES, 15, 3))
        k[:, :, 2] = rng.uniform(0, 1, size=(config.CLIP_FRAMES, 15))
        drop = rng.random((config.CLIP_FRAMES, 15)) < 0.15
        k[drop] = [SENTINEL, SENTINEL, 0.0]

        ref = reference_normalize(
            k, frame_w=config.FRAME_WIDTH, frame_h=config.FRAME_HEIGHT,
            torso_floor=config.TORSO_LENGTH_FLOOR,
            torso_fallback_frac=config.TORSO_FALLBACK_FRAC,
            clamp=config.KPT_CLAMP, channels=config.KPT_CHANNELS)
        got = normalize_skeleton(torch.from_numpy(k)).numpy()
        assert got.shape == ref.shape, (got.shape, ref.shape)
        assert np.allclose(got, ref, atol=1e-4), f"trial {trial} max diff {np.abs(got-ref).max()}"
    print("torch/reference parity OK")


if __name__ == "__main__":
    test_reference_hand_cases()
    try:
        test_torch_matches_reference()
    except ImportError as e:
        print(f"(skipped torch parity: {e})")
