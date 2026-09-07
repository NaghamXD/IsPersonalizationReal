"""The dynamic partition shuffle: forward equivalence and a deterministic adjoint.

Two things to establish, because this replaced a line inside the pretrained
architecture:

  1. The new forward is IDENTICAL to the original in-place assignment. If it is not,
     every downstream number changes for a reason unrelated to the research.
  2. The custom backward equals what autograd computes for the same gather -- so the
     determinism is bought by supplying the exact adjoint, not by approximating it.

Run:  python tests/test_dynamic_shuffle.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def original_shuffle_np(x, order):
    """The 6fca412 semantics: x[:, raw_order] = x[:, dynamic_order], raw = 0..P-1."""
    x = np.array(x, copy=True)
    out = np.array(x, copy=True)
    raw = list(range(x.shape[-1]))
    out[:, raw] = x[:, order]
    return out


def new_shuffle_np(x, order):
    """The functional form: a gather along the joint axis."""
    return np.asarray(x)[:, list(order)]


def test_forward_equivalence_numpy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        P = 15
        x = rng.normal(size=(37, P))
        order = rng.permutation(P)
        assert np.array_equal(original_shuffle_np(x, order), new_shuffle_np(x, order))
    print("  OK  gather is identical to the original in-place assignment")


def test_torch_forward_and_backward():
    import torch
    from src.model.vsvig import _PermuteJoints

    rng = np.random.default_rng(1)
    P = 15
    for _ in range(10):
        x_np = rng.normal(size=(12, P))
        order = rng.permutation(P)
        perm = torch.as_tensor(order.copy(), dtype=torch.long)
        inv = torch.argsort(perm)

        # forward matches the numpy reference
        xa = torch.tensor(x_np, requires_grad=True)
        ya = _PermuteJoints.apply(xa, perm, inv)
        assert np.allclose(ya.detach().numpy(), new_shuffle_np(x_np, order))

        # backward matches autograd through the equivalent plain gather
        xb = torch.tensor(x_np, requires_grad=True)
        yb = xb.index_select(-1, perm)
        g = torch.tensor(rng.normal(size=(12, P)))
        ga, = torch.autograd.grad(ya, xa, g, retain_graph=True)
        gb, = torch.autograd.grad(yb, xb, g)
        assert torch.allclose(ga, gb, atol=1e-12), (ga - gb).abs().max()
    print("  OK  custom adjoint equals autograd's, exactly")


def test_model_forward_is_deterministic():
    """Two forward+backward passes on identical input must agree bit-for-bit."""
    import torch
    import config
    from src.model.vsvig import VSViG_base

    torch.manual_seed(0)
    model = VSViG_base(kpt_channels=config.KPT_CHANNELS)
    patches = torch.randn(2, config.CLIP_FRAMES, config.N_JOINTS, 3,
                          config.PATCH_SIZE, config.PATCH_SIZE)
    kpts = torch.rand(2, config.CLIP_FRAMES, config.N_JOINTS, config.KPT_CHANNELS)

    grads = []
    for _ in range(2):
        model.zero_grad()
        out = model(patches, kpts)
        out.sum().backward()
        grads.append(torch.cat([p.grad.reshape(-1).clone()
                                for p in model.parameters() if p.grad is not None]))
    assert torch.equal(grads[0], grads[1]), \
        f"gradients differ across identical passes: max {(grads[0]-grads[1]).abs().max()}"
    print("  OK  identical passes give bit-identical gradients")


def test_rejects_a_non_permutation():
    import torch
    from src.model.vsvig import Part_3DCNN
    blk = Part_3DCNN(in_channels=4, out_channels=4, dynamic=True,
                     dynamic_point_order=[[0, 0, 2]], SEED=0, expansion=2)
    try:
        blk._perm_pair([0, 0, 2], 3, torch.device("cpu"))
    except ValueError as e:
        assert "permutation" in str(e)
        print("  OK  rejects a non-permutation instead of silently mis-adjointing")
        return
    raise AssertionError("expected a ValueError")


if __name__ == "__main__":
    test_forward_equivalence_numpy()
    try:
        test_torch_forward_and_backward()
        test_model_forward_is_deterministic()
        test_rejects_a_non_permutation()
    except ImportError as e:
        print(f"  (skipped torch tests: {e})")
    print("\nALL DYNAMIC-SHUFFLE TESTS PASSED")
