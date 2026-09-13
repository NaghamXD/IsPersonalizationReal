"""Resolve HN_TARGET_SPECS to real backbone modules, and apply LoRA deltas.

The spec names ("stage3_block0_conv2") are the methodology's vocabulary; the backbone
uses flat indices ("backbone.25.conv2.0"). Mapping one to the other by searching for a
matching weight SHAPE would be ambiguous -- three stage-3 blocks share (192,192,3,3,1)
-- so the mapping is derived from the architecture's construction, the same arithmetic
src.model.layout uses, and then checked against the live module's shape. A silent
mismatch here would adapt the wrong layers and nothing downstream would notice.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

import config


def stage_block_indices(num_layer=(2, 2, 6, 2), stage: int = 3) -> list[int]:
    """Flat indices of the conv2-bearing modules in `stage`.

    Mirrors STViG's construction: each stage after the first opens with a
    (Grapher, Part_3DCNN) downsampling pair, then stacks num_layer[stage] more pairs.
    The Part_3DCNN of each pair sits at the odd index and carries conv2.
    """
    idx, out = 0, []
    for s, n in enumerate(num_layer):
        pairs = (1 if s > 0 else 0) + n
        here = [idx + 2 * k + 1 for k in range(pairs)]
        if s == stage:
            out = here
        idx += 2 * pairs
    return out


def resolve_targets(model: nn.Module, specs=None, num_layer=(2, 2, 6, 2)):
    """-> {spec name: module}, verified against each spec's (out_dim, flat_in)."""
    specs = config.HN_TARGET_SPECS if specs is None else specs
    named = dict(model.named_modules())
    blocks = stage_block_indices(num_layer, stage=len(num_layer) - 1)

    out = {}
    for name, out_dim, flat_in, _rank in specs:
        if name.startswith("stage"):
            b = int(name.split("block")[1].split("_")[0])
            if b >= len(blocks):
                raise KeyError(f"{name}: stage has only {len(blocks)} blocks")
            path = f"backbone.{blocks[b]}.conv2.0"
        elif name.startswith("fc"):
            path = f"fc.{name[2:]}"
        else:
            raise KeyError(f"cannot resolve target name {name!r}")

        mod = named.get(path)
        if mod is None or not hasattr(mod, "weight"):
            raise KeyError(f"{name} -> {path}: no such module with a weight")
        w = mod.weight
        got_out, got_in = int(w.shape[0]), int(w[0].numel())
        if (got_out, got_in) != (int(out_dim), int(flat_in)):
            raise ValueError(
                f"{name} -> {path}: spec says (out={out_dim}, flat_in={flat_in}) "
                f"but the module is (out={got_out}, flat_in={got_in}). The spec and "
                f"the architecture disagree; adapting this layer would be wrong.")
        out[name] = mod
    return out


def base_norms(targets: dict[str, nn.Module]) -> dict[str, float]:
    """||W_base||_F per target, for the HN_DELTA_CLIP_RATIO budget."""
    return {k: float(m.weight.detach().flatten().norm()) for k, m in targets.items()}


class AdaptedForward:
    """Temporarily add dW to each target's weight.

    The backbone stays frozen: deltas are ADDED to the existing weight for the duration
    of the forward pass and removed afterwards, so no baseline parameter is ever
    modified in place across calls. Gradients still reach the hypernetwork, because the
    added tensor carries its graph.
    """

    def __init__(self, targets: dict[str, nn.Module], deltas: dict[str, Tensor]):
        self.targets, self.deltas, self._saved = targets, deltas, {}

    def __enter__(self):
        for name, mod in self.targets.items():
            dw = self.deltas.get(name)
            if dw is None:
                continue
            if dw.dim() == 3:
                if dw.shape[0] != 1:
                    raise ValueError(
                        f"{name}: got a batch of {dw.shape[0]} deltas. One forward pass "
                        f"can carry ONE weight, so a batch must be homogeneous in "
                        f"z_behavior (all clips from one patient). See DECISIONS.md.")
                dw = dw[0]
            self._saved[name] = mod.weight
            mod.weight = nn.Parameter(mod.weight.detach(), requires_grad=False) \
                if not isinstance(mod.weight, nn.Parameter) else mod.weight
            # Swap the attribute for a plain tensor carrying the hypernetwork's graph.
            del mod._parameters["weight"]
            mod.weight = self._saved[name].detach() + dw.view_as(self._saved[name])
        return self

    def __exit__(self, *exc):
        for name, saved in self._saved.items():
            mod = self.targets[name]
            if "weight" in mod.__dict__:
                del mod.__dict__["weight"]
            mod._parameters["weight"] = saved
        self._saved.clear()
        return False
