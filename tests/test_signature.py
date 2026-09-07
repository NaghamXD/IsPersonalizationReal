"""Tests for the z_behavior reduction.

`reference_mu_sigma` is a pure-numpy statement of the intended reduction, checked
against hand-computed cases; the torch implementation is then held to it. Same
two-level approach as tests/test_normalize.py, and for the same reason: the mistake
that matters here is a plausible-looking reduction in the wrong order, which produces
finite, reasonable-looking numbers that mean something else entirely.

Run:  python tests/test_signature.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def reference_mu_sigma(feats, unbiased=False):
    """feats (B, T, C, P) -> mu (B, C), sigma (B, C).

    sigma: std over TIME first -> (B, C, P), then mean over joints.
    mu:    mean over time and joints.
    """
    feats = np.asarray(feats, dtype=np.float64)
    mu = feats.mean(axis=(1, 3))
    ddof = 1 if unbiased else 0
    sigma_per_joint = feats.std(axis=1, ddof=ddof)      # (B, C, P)
    sigma = sigma_per_joint.mean(axis=2)
    return mu, sigma


def test_hand_computed():
    # One clip, one channel, two joints, four timesteps.
    # joint 0: 0,2,0,2  -> mean 1, biased std 1
    # joint 1: 5,5,5,5  -> mean 5, biased std 0
    feats = np.array([[[[0.0, 5.0]], [[2.0, 5.0]], [[0.0, 5.0]], [[2.0, 5.0]]]])
    assert feats.shape == (1, 4, 1, 2)
    mu, sigma = reference_mu_sigma(feats)
    assert np.isclose(mu[0, 0], 3.0), mu           # (1 + 5) / 2
    assert np.isclose(sigma[0, 0], 0.5), sigma     # (1 + 0) / 2
    print("  OK  hand-computed mu and sigma")


def test_static_features_have_zero_sigma():
    feats = np.tile(np.arange(6.0).reshape(1, 1, 6, 1), (2, 8, 1, 15))
    _, sigma = reference_mu_sigma(feats)
    assert np.allclose(sigma, 0.0), "motionless features must give sigma = 0"
    print("  OK  motionless input gives sigma = 0")


def test_single_timestep_gives_zero_not_nan():
    """[METHOD] biased estimator, so one sample resolves to 0 rather than NaN."""
    feats = np.random.default_rng(0).normal(size=(1, 1, 4, 15))
    _, sigma = reference_mu_sigma(feats, unbiased=False)
    assert np.isfinite(sigma).all() and np.allclose(sigma, 0.0), sigma
    with np.errstate(invalid="ignore", divide="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, sigma_u = reference_mu_sigma(feats, unbiased=True)
    assert not np.isfinite(sigma_u).all(), "unbiased on n=1 is NaN -- why we use biased"
    print("  OK  single timestep gives 0, not NaN")


def test_temporal_before_spatial_is_not_interchangeable():
    """The ordering is the whole point, and antiphase motion proves it.

    Two joints oscillating in ANTIPHASE -- one limb up while the other goes down, which
    is ordinary seizure semiology. Taking the standard deviation over time FIRST sees
    both joints moving. Pooling over joints first cancels them exactly, and the
    signature reports a perfectly still patient.
    """
    T = 8
    swing = np.array([1.0, -1.0] * (T // 2))
    feats = np.zeros((1, T, 1, 2))
    feats[0, :, 0, 0] = swing
    feats[0, :, 0, 1] = -swing                     # antiphase

    _, sigma_correct = reference_mu_sigma(feats)   # temporal, then spatial
    spatial_first = feats.mean(axis=3)             # (B, T, C)
    sigma_wrong = spatial_first.std(axis=1)

    assert np.isclose(sigma_correct[0, 0], 1.0), sigma_correct
    assert np.isclose(sigma_wrong[0, 0], 0.0), sigma_wrong
    print(f"  OK  ordering matters: temporal-first sigma={sigma_correct[0,0]:.2f}, "
          f"spatial-first sigma={sigma_wrong[0,0]:.2f} (antiphase motion erased)")


def test_stage_cut_index():
    from src.model.layout import stage_cut_index
    assert stage_cut_index(stage_cut=0) == 3
    assert stage_cut_index(stage_cut=1) == 9
    assert stage_cut_index(stage_cut=2) == 23, "stage 2 must end at backbone module 23"
    assert stage_cut_index(stage_cut=3) == 29
    assert stage_cut_index(stage_cut=config.SIGNATURE_STAGE_CUT) == 23
    print("  OK  stage cut index derived from the architecture, not hard-coded")


def test_stability_ratio_shape_and_per_patient():
    from src.eval.stability import stability_ratio
    rng = np.random.default_rng(0)
    # Two patients, well separated, each with tight internal blocks.
    bv = {"patA": [rng.normal(0, 0.01, 8) + 0 for _ in range(4)],
          "patB": [rng.normal(0, 0.01, 8) + 10 for _ in range(4)]}
    out = stability_ratio(bv)
    assert out["pooled_ratio"] > config.STABILITY_RATIO_THRESHOLD
    assert set(out["ratio_per_patient"]) == {"patA", "patB"}
    # A patient with a much tighter spread gets a much higher ratio -- which is the
    # comparability problem D14 describes, made visible rather than averaged away.
    bv["patB"] = [rng.normal(0, 1.0, 8) + 10 for _ in range(4)]
    out2 = stability_ratio(bv)
    assert out2["ratio_per_patient"]["patA"] > out2["ratio_per_patient"]["patB"]
    print("  OK  stability ratio reports per patient, not just pooled")


def test_torch_matches_reference():
    import torch
    from src.model.signature import StaticContextProjector, clip_mu_sigma

    rng = np.random.default_rng(1)
    for _ in range(10):
        f = rng.normal(size=(3, config.CLIP_FRAMES // 4, config.SIGNATURE_CHANNELS,
                             config.N_JOINTS))
        rmu, rsig = reference_mu_sigma(f, unbiased=config.SIGMA_UNBIASED)
        tmu, tsig = clip_mu_sigma(torch.from_numpy(f))
        assert np.allclose(tmu.numpy(), rmu, atol=1e-6), np.abs(tmu.numpy() - rmu).max()
        assert np.allclose(tsig.numpy(), rsig, atol=1e-6), np.abs(tsig.numpy() - rsig).max()

    p1, p2 = StaticContextProjector(), StaticContextProjector()
    assert torch.equal(p1.proj.weight, p2.proj.weight), "seeded projector must be stable"
    assert not any(q.requires_grad for q in p1.parameters()), "projector must be frozen"
    mu = torch.randn(config.SIGNATURE_CHANNELS)
    sig = torch.randn(config.SIGNATURE_CHANNELS)
    z = p1(mu, sig)
    assert z.shape == (config.CONTEXT_DIM,), z.shape
    print("  OK  torch/reference parity; projector seeded, frozen, 384 -> 128")


if __name__ == "__main__":
    for fn in [test_hand_computed, test_static_features_have_zero_sigma,
               test_single_timestep_gives_zero_not_nan,
               test_temporal_before_spatial_is_not_interchangeable,
               test_stage_cut_index, test_stability_ratio_shape_and_per_patient]:
        fn()
    try:
        test_torch_matches_reference()
    except ImportError as e:
        print(f"  (skipped torch parity: {e})")
    print("\nALL SIGNATURE TESTS PASSED")
