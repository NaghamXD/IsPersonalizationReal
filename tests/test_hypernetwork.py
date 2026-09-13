"""Stage 7 invariants. These are the properties the experiment's validity rests on."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import config
from src.model.adapt import (AdaptedForward, base_norms, resolve_targets,
                             stage_block_indices)
from src.model.hypernetwork import Hypernetwork, TargetHead


def test_delta_is_exactly_zero_at_initialisation():
    """The whole comparison depends on this: at step 0 the adapted model IS the
    baseline, so any later difference was learned rather than introduced by init."""
    hn = Hypernetwork()
    for z in (torch.zeros(config.CONTEXT_DIM), torch.randn(config.CONTEXT_DIM) * 5):
        for name, dw in hn(z).items():
            assert float(dw.abs().max()) == 0.0, f"{name} is non-zero at init"


def test_a_base_std_follows_the_variance_reading():
    """D24: N(0, d_in^-1 * 1e-2) with 1e-2 read as a VARIANCE scale, i.e. std =
    0.1/sqrt(d_in) = 0.1x the standard LoRA init -- not 1e-2/d_in, which would be
    ~6e-6 and leave the base+residual decomposition with no base."""
    for _name, out_dim, in_dim, rank in config.HN_TARGET_SPECS:
        h = TargetHead(out_dim, in_dim, rank, config.HN_TRUNK_HIDDEN)
        want = math.sqrt(config.HN_A_BASE_STD_SCALE / in_dim)
        got = float(h.a_base.detach().std())
        assert abs(got - want) / want < 0.35, f"in_dim={in_dim}: {got:.2e} vs {want:.2e}"
        assert abs(want - 0.1 / math.sqrt(in_dim)) < 1e-12      # 0.1x standard LoRA
        assert want > 100 * (config.HN_A_BASE_STD_SCALE / in_dim)  # not the std reading


def test_z_actually_changes_the_delta():
    """If deltas did not depend on z, the shuffled-z control would be vacuous."""
    hn = Hypernetwork()
    torch.nn.init.normal_(hn.heads["fc0"].b_head.weight, std=0.05)   # undo zero-init
    a = hn(torch.randn(config.CONTEXT_DIM))["fc0"]
    b = hn(torch.randn(config.CONTEXT_DIM))["fc0"]
    assert float((a - b).abs().max()) > 0

def test_frobenius_budget_is_respected_and_is_a_no_op_when_under():
    hn = Hypernetwork()
    for h in hn.heads.values():
        torch.nn.init.normal_(h.b_head.weight, std=1.0)     # force oversized deltas
    z = torch.randn(config.CONTEXT_DIM)
    norms = {n: 1.0 for n, _, _, _ in hn.specs}
    for name, dw in hn(z, base_norms=norms).items():
        assert float(dw.flatten().norm()) <= config.HN_DELTA_CLIP_RATIO * 1.0 + 1e-4

    generous = {n: 1e9 for n, _, _, _ in hn.specs}
    unclipped = hn(z, base_norms=None)
    clipped = hn(z, base_norms=generous)
    for name in unclipped:
        assert torch.allclose(unclipped[name], clipped[name], atol=1e-6), name


def test_targets_resolve_to_the_layers_the_spec_names():
    from src.model.vsvig import VSViG_base
    assert stage_block_indices() == [25, 27, 29]
    m = VSViG_base(kpt_channels=config.KPT_CHANNELS)
    t = resolve_targets(m)                    # raises if a spec and the module disagree
    assert set(t) == {n for n, _, _, _ in config.HN_TARGET_SPECS}
    for name, out_dim, flat_in, _ in config.HN_TARGET_SPECS:
        w = t[name].weight
        assert (int(w.shape[0]), int(w[0].numel())) == (out_dim, flat_in)
    assert all(v > 0 for v in base_norms(t).values())


def test_adapted_forward_restores_the_backbone_exactly():
    """The baseline must be untouched after an adapted pass, or the two arms stop
    being comparable and the damage accumulates silently across batches."""
    from src.model.vsvig import VSViG_base
    m = VSViG_base(kpt_channels=config.KPT_CHANNELS).eval()
    t = resolve_targets(m)
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    deltas = {k: torch.randn_like(v.weight) * 0.01 for k, v in t.items()}
    with AdaptedForward(t, deltas):
        for k, v in t.items():
            assert not torch.allclose(v.weight, before[k]), f"{k} was not adapted"
    for k, v in t.items():
        assert torch.equal(v.weight, before[k]), f"{k} was not restored"
        assert isinstance(v.weight, torch.nn.Parameter)


def test_adapted_forward_refuses_a_heterogeneous_batch():
    """One forward pass carries one weight matrix. A batch mixing patients would
    silently apply the first patient's delta to everyone."""
    from src.model.vsvig import VSViG_base
    m = VSViG_base(kpt_channels=config.KPT_CHANNELS).eval()
    t = resolve_targets(m)
    deltas = {k: torch.randn(3, *v.weight.shape[:1], v.weight[0].numel())
              for k, v in t.items()}
    try:
        with AdaptedForward(t, deltas):
            pass
    except ValueError as e:
        assert "homogeneous" in str(e)
    else:
        raise AssertionError("a 3-patient batch of deltas was accepted silently")
