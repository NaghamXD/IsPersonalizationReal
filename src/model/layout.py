"""Backbone layout arithmetic. Deliberately free of torch.

Which flat module index ends which stage is a property of the architecture's
CONSTRUCTION, not of a live tensor, so working it out should not require a deep
learning framework. Keeping it here means the signature's extraction depth can be
tested anywhere -- and it is the number that decides what z_behavior even measures.
"""
import config


def stage_cut_index(num_layer=(2, 2, 6, 2), stage_cut=None) -> int:
    """Flat index of the LAST backbone module belonging to `stage_cut`.

    Mirrors how STViG builds `self.backbone`: every stage after the first opens with a
    downsampling Grapher + Part_3DCNN pair, then stacks `num_layer[stage]` more pairs.
    Derived rather than hard-coded so a change to the architecture cannot silently
    leave the signature reading from the wrong depth.
    """
    stage_cut = config.SIGNATURE_STAGE_CUT if stage_cut is None else stage_cut
    idx = 0
    for stage, n in enumerate(num_layer):
        if stage > 0:
            idx += 2
        idx += 2 * n
        if stage == stage_cut:
            return idx - 1
    raise ValueError(f"stage_cut {stage_cut} beyond {len(num_layer)} stages")
