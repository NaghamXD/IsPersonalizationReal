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


# ------------------------------------------------------------- §3.4.1 cyclic sampler
def test_steps_per_patient_follows_the_majority_class():
    from src.data.sampler import steps_for
    assert steps_for(100, 20, 16) == 100 // 8       # S_p = max(|inter|,|ictal|)/(B/2)
    assert steps_for(20, 100, 16) == 100 // 8       # symmetric in which class is larger
    assert steps_for(3, 1, 16) == 1                 # never rounds down to zero
    assert steps_for(100, 0, 16) == 0               # one class only -> no valid batch
    assert steps_for(0, 100, 16) == 0


def test_batches_are_patient_homogeneous_and_exactly_half_ictal():
    import random
    from src.data.sampler import build_epoch
    by = {"patA": list(range(0, 120)), "patB": list(range(120, 160))}
    lab = {"patA": [0.0] * 100 + [1.0] * 20, "patB": [0.0] * 10 + [1.0] * 30}
    flat = {i: ("patA" if i < 120 else "patB") for i in range(160)}
    labof = {}
    for p in by:
        for i, y in zip(by[p], lab[p]):
            labof[i] = y

    batches, skipped = build_epoch(by, lab, batch_size=16, rng=random.Random(0))
    assert not skipped
    assert {p for p, _ in batches} == {"patA", "patB"}
    for p, idx in batches:
        assert len(idx) == 16
        assert {flat[i] for i in idx} == {p}, "batch mixes patients"
        assert sum(1 for i in idx if labof[i] == 1.0) == 8, "not 50/50"
    assert sum(1 for p, _ in batches if p == "patA") == 12
    assert sum(1 for p, _ in batches if p == "patB") == 3


def test_minority_pool_cycles_without_skipping_anyone():
    """The minority pool wraps via itertools.cycle, so over S_p steps every minority
    instance is used -- that is the point of cycling rather than resampling."""
    import random
    from src.data.sampler import build_epoch
    by = {"patA": list(range(0, 104))}
    lab = {"patA": [0.0] * 100 + [1.0] * 4}         # 4 ictal, 12 steps x 8 = 96 draws
    batches, _ = build_epoch(by, lab, batch_size=16, rng=random.Random(0))
    drawn = [i for _, idx in batches for i in idx if i >= 100]
    assert set(drawn) == set(range(100, 104)), "a minority instance was never drawn"


def test_transition_clips_are_excluded_from_hypernetwork_batches():
    import random
    from src.data.sampler import build_epoch
    by = {"patA": list(range(0, 60))}
    lab = {"patA": [0.0] * 25 + [0.5] * 10 + [1.0] * 25}    # ten soft transition labels
    batches, _ = build_epoch(by, lab, batch_size=16, rng=random.Random(0))
    used = {i for _, idx in batches for i in idx}
    assert not (used & set(range(25, 35))), "a soft-labelled transition clip was sampled"


def test_backbone_logits_path_matches_the_probability_path():
    """3.4.2 needs un-sigmoided logits; every existing caller needs probabilities."""
    import torch
    from src.model.vsvig import VSViG_base
    m = VSViG_base(kpt_channels=config.KPT_CHANNELS).eval()
    d = torch.randn(2, 30, 15, 3, 32, 32)
    k = torch.randn(2, 30, 15, config.KPT_CHANNELS)
    with torch.no_grad():
        prob = m(d, k)
        logit = m(d, k, return_logits=True)
    assert torch.allclose(torch.sigmoid(logit), prob, atol=1e-6)
    assert prob.shape == logit.shape


# ------------------------------------------------------- D34 training-only patients
def test_training_only_patients_join_train_and_nothing_else():
    from src.data.splits import make_all_folds, verify_no_leak
    extra = ("patX", "patY")
    folds = make_all_folds(training_only=extra)
    assert verify_no_leak(folds) == []
    for f in folds:
        assert set(f.training_only) == set(extra)
        assert set(extra) <= set(f.all_train_patients)
        assert not (set(extra) & set(f.val_patients)), "reached validation"
        assert f.test_patient not in extra, "became a test patient"
        assert set(f.all_train_patients) == set(f.train_patients) | set(extra)


def test_a_patient_cannot_be_both_cohort_and_training_only():
    from src.data.splits import make_fold
    try:
        make_fold(config.COHORT[0], training_only=(config.COHORT[1],))
    except ValueError as e:
        assert "cannot be both" in str(e)
    else:
        raise AssertionError("a cohort patient was accepted as training-only")


def test_leak_check_catches_a_training_only_patient_in_validation():
    from src.data.splits import Fold, verify_no_leak
    bad = Fold(test_patient="pat01", val_patients=("pat04", "patZ"),
               train_patients=("pat02",), training_only=("patZ",))
    assert any("training-only patient used for validation" in r[1]
               for r in verify_no_leak([bad]))
    worse = Fold(test_patient="patZ", val_patients=("pat04",),
                 train_patients=("pat02",), training_only=("patZ",))
    assert any("training-only" in r[1] for r in verify_no_leak([worse]))
