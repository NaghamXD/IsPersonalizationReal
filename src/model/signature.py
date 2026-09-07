"""z_behavior -- the patient's resting motor signature.

[METHOD] §3.2.2. Pool A clips pass through the early frozen blocks of the backbone.
The empirical mean mu models "typical resting geometry"; the standard deviation sigma
is computed "across the temporal dimension first, isolating joint velocity before
spatial average pooling is applied", so it captures kinematic volatility rather than
posture. [mu || sigma] is mapped by an UNTRAINED, FROZEN random projection into
z_behavior in R^128 -- a structural bottleneck that stops the hypernetwork memorising
patient identity.

THREE THINGS THE BASE REPO DID DIFFERENTLY, all of which change what z means:

  1. It ran the WHOLE backbone, not stages 0-2. (§3.2.2 says stages 0-2 but also
     states C = 384, which exists nowhere: stage 2 emits 192 and T is downsampled
     30 -> 15 -> 8 -> 4. Verified by scripts/verify_shapes.py: the cut gives
     C' = 192, T' = 8, P = 15, hence a 384 -> 128 projector, not 768 -> 128.)
  2. It average-pooled time and joints together, then took the standard deviation
     ACROSS CLIPS. That measures how much a patient's pooled posture drifts between
     clips -- not within-clip joint velocity.
  3. Consequently its sigma had no temporal meaning at all.

Note for the write-up: at the stage-2 cut T' = 8, so sigma is a standard deviation
over eight downsampled timesteps per clip, not thirty raw frames. Coarser than the
draft's text implies. See DECISIONS.md D3.
"""
import torch
import torch.nn as nn
from torch import Tensor

import config
from src.eval.stability import stability_ratio  # noqa: F401  (re-exported)
from src.model.layout import stage_cut_index  # noqa: F401  (re-exported)


@torch.no_grad()
def stage_cut_features(backbone, inputs: Tensor, kpts: Tensor, stage_cut=None) -> Tensor:
    """Run stem + positional embedding + backbone up to the cut.

    Returns (B, T', C', P) -- the trailing singleton axis STViG carries internally is
    dropped here, because nothing downstream needs it.
    """
    cut = stage_cut_index(stage_cut=stage_cut)
    x = backbone.stem(inputs)
    x = backbone.pe(backbone.pos_emb, x, kpts)
    B, T, P, C = x.shape
    x = x.transpose(2, 3).contiguous().view(B, T, C, P, 1)
    for i, module in enumerate(backbone.backbone):
        x = module(x)
        if i == cut:
            break
    return x.squeeze(-1)                      # (B, T', C', P)


def clip_mu_sigma(feats: Tensor, unbiased: bool | None = None):
    """(B, T, C, P) -> mu (B, C), sigma (B, C), one pair per clip.

    sigma: standard deviation over TIME first -> (B, C, P), then mean over joints.
           Temporal-then-spatial, per the methodology. Reversing the order would
           measure how much joints differ from each other, not how much they move.
    mu:    plain spatiotemporal mean.

    Biased estimator by default [METHOD], so a single timestep gives 0 rather than NaN.
    """
    unbiased = config.SIGMA_UNBIASED if unbiased is None else unbiased
    mu = feats.mean(dim=(1, 3))                            # over T and P
    sigma_per_joint = feats.std(dim=1, unbiased=unbiased)  # over T -> (B, C, P)
    sigma = sigma_per_joint.mean(dim=2)                    # then over P
    return mu, sigma


class StaticContextProjector(nn.Module):
    """Frozen, seeded random [mu || sigma] -> z_behavior.

    Never trained: [METHOD] calls it "an untrained, frozen random linear projection
    matrix ... a structural information bottleneck, preventing the downstream
    hypernetwork from memorizing categorical patient IDs". All parameters have
    requires_grad False at construction and it is excluded from optimiser param groups.
    """

    def __init__(self, in_dim: int | None = None, context_dim: int | None = None,
                 seed: int | None = None):
        super().__init__()
        in_dim = config.PROJECTOR_IN_DIM if in_dim is None else in_dim
        context_dim = config.CONTEXT_DIM if context_dim is None else context_dim
        seed = config.PROJECTOR_SEED if seed is None else seed

        self.in_dim, self.context_dim, self.seed = in_dim, context_dim, seed
        self.proj = nn.Linear(in_dim, context_dim)
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.randn(self.proj.weight.shape, generator=gen) * in_dim ** -0.5)
            self.proj.bias.zero_()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, mu: Tensor, sigma: Tensor) -> Tensor:
        return self.proj(torch.cat([mu.reshape(-1), sigma.reshape(-1)], dim=0))


@torch.no_grad()
def compute_z_behavior(backbone, projector, clips, load_batch, device,
                       batch_size: int = 8, return_parts: bool = False):
    """Pool A clip names -> z_behavior (context_dim,).

    [DECISION] mu and sigma are computed PER CLIP and then averaged across Pool A --
    not by concatenating the clips into one time axis. Pool A clips are drawn hours
    apart by design, so concatenating them would put a spurious velocity spike at every
    seam and inflate sigma with sampling artefacts rather than patient behaviour.
    See DECISIONS.md D4.

    `load_batch(names, device) -> (patches, kpts)` is injected so this works for both
    training-clip and test-clip stores without knowing about either.
    """
    if not clips:
        raise ValueError("Pool A is empty -- cannot compute z_behavior")
    backbone.eval()

    mus, sigmas = [], []
    for i in range(0, len(clips), batch_size):
        patches, kpts = load_batch(clips[i:i + batch_size], device)
        feats = stage_cut_features(backbone, patches, kpts)
        mu, sigma = clip_mu_sigma(feats)
        mus.append(mu)
        sigmas.append(sigma)

    mu_bar = torch.cat(mus).mean(dim=0)         # average across Pool A clips
    sigma_bar = torch.cat(sigmas).mean(dim=0)
    z = projector(mu_bar, sigma_bar)
    return (z, mu_bar, sigma_bar) if return_parts else z
