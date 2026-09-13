"""Amortised hypernetwork: z_behavior -> LoRA deltas for the frozen backbone.

    A_p = A_base + A_hyper(z)          B_p = B_hyper(z)          dW = B_p @ A_p

B is zero-initialised, so dW = 0 at step 0 and the adapted model is EXACTLY the
baseline before any gradient is taken. That is what makes the baseline-versus-adapted
comparison a controlled one: at initialisation the two arms are the same function, and
every difference measured afterwards was learned.

[DECISION D24] A_base ~ N(0, d_in^-1 * 1e-2), with the 1e-2 read as a VARIANCE scale.
The draft is ambiguous and the two readings differ 100-fold. Resolved on the numbers:

    variance reading -> std = 0.1 / sqrt(d_in)  = 0.100x the standard LoRA init
    std reading      -> std = 1e-2 / d_in       = 0.000241x

The std reading puts A_base at ~6e-6 for the stage-3 targets, indistinguishable from
zero. Together with a zero-initialised B that leaves the base+residual decomposition
with no base -- A_p would be A_hyper(z) alone, and the shared patient-independent
direction the decomposition exists to provide would not exist. The variance reading
lands on exactly 0.1x the usual N(0, 1/d_in), which reads as a deliberate choice.
(The 6fca412 base repo implements the std reading; by this argument that is a bug.)

Parameter budget is kept down by generating A through a 32-d bottleneck
(config.HN_BOTTLENECK): the trunk emits rank x 32, which a learned 32 x d_in matrix
expands. Emitting rank x d_in directly would cost 128 x 4 x 1728 ~ 885k parameters per
stage-3 target.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

import config


class TargetHead(nn.Module):
    """Generates (A, B) for one adapted layer."""

    def __init__(self, out_dim: int, in_dim: int, rank: int, trunk_dim: int,
                 bottleneck: int | None = None, use_base: bool | None = None,
                 a_base_scale: float | None = None):
        super().__init__()
        bottleneck = config.HN_BOTTLENECK if bottleneck is None else bottleneck
        use_base = config.HN_USE_BASE if use_base is None else use_base
        a_base_scale = config.HN_A_BASE_STD_SCALE if a_base_scale is None else a_base_scale

        self.out_dim, self.in_dim, self.rank = out_dim, in_dim, rank
        self.bottleneck = min(bottleneck, in_dim)

        # A = (A_code @ expand), A_code from z; expand is learned and z-independent.
        self.a_code = nn.Linear(trunk_dim, rank * self.bottleneck)
        self.a_expand = nn.Linear(self.bottleneck, in_dim, bias=False)

        # B from z, zero-initialised so dW = 0 at step 0.
        self.b_head = nn.Linear(trunk_dim, out_dim * rank)
        nn.init.zeros_(self.b_head.weight)
        nn.init.zeros_(self.b_head.bias)

        if use_base:
            # D24: 1e-2 is a VARIANCE scale on the standard N(0, 1/d_in).
            std = (a_base_scale / in_dim) ** 0.5
            self.register_parameter(
                "a_base", nn.Parameter(torch.randn(rank, in_dim) * std))
        else:
            self.a_base = None

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """z: (trunk_dim,) or (batch, trunk_dim) -> A (…, rank, in_dim), B (…, out, rank)"""
        squeeze = z.dim() == 1
        if squeeze:
            z = z.unsqueeze(0)
        b = z.shape[0]
        a = self.a_expand(self.a_code(z).view(b, self.rank, self.bottleneck))
        if self.a_base is not None:
            a = a + self.a_base
        bb = self.b_head(z).view(b, self.out_dim, self.rank)
        return (a[0], bb[0]) if squeeze else (a, bb)


class Hypernetwork(nn.Module):
    """z_behavior -> {layer name: dW}, with a per-layer Frobenius budget."""

    def __init__(self, context_dim: int | None = None, specs=None,
                 trunk_hidden: int | None = None):
        super().__init__()
        context_dim = config.CONTEXT_DIM if context_dim is None else context_dim
        specs = config.HN_TARGET_SPECS if specs is None else specs
        h = config.HN_TRUNK_HIDDEN if trunk_hidden is None else trunk_hidden

        act = (nn.LeakyReLU(config.HN_LEAKY_SLOPE)
               if config.HN_TRUNK_ACTIVATION == "leaky_relu" else nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(context_dim, h), act,
                                   nn.Linear(h, h),
                                   nn.LeakyReLU(config.HN_LEAKY_SLOPE)
                                   if config.HN_TRUNK_ACTIVATION == "leaky_relu"
                                   else nn.ReLU())
        self.specs = [(n, int(o), int(i), int(r)) for n, o, i, r in specs]
        self.heads = nn.ModuleDict(
            {n: TargetHead(o, i, r, h) for n, o, i, r in self.specs})

    def forward(self, z: Tensor, base_norms: dict[str, float] | None = None,
                clip_ratio: float | None = None) -> dict[str, Tensor]:
        clip_ratio = config.HN_DELTA_CLIP_RATIO if clip_ratio is None else clip_ratio
        t = self.trunk(z)
        out = {}
        for name, _, _, _ in self.specs:
            a, b = self.heads[name](t)
            dw = b @ a                                    # (…, out_dim, in_dim)
            if base_norms is not None and name in base_norms and clip_ratio:
                budget = clip_ratio * float(base_norms[name])
                # Scale down only when over budget; a no-op delta must stay a no-op,
                # and the scale must stay differentiable.
                n = dw.flatten(-2).norm(dim=-1).clamp_min(1e-12)
                factor = torch.clamp(budget / n, max=1.0)
                dw = dw * factor.view(*factor.shape, 1, 1)
            out[name] = dw
        return out

    @torch.no_grad()
    def delta_norms(self, z: Tensor, base_norms=None) -> dict[str, float]:
        return {k: float(v.flatten(-2).norm(dim=-1).mean())
                for k, v in self(z, base_norms).items()}
