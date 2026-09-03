"""Skeleton normalisation.

METHODOLOGY (section 3.1): "raw skeleton coordinates are centered around the mid-hip
joint and normalized by torso length prior to any modeling. [...] To isolate invariant
resting tremor or hyperkinetic baselines from gross camera positioning and patient
posture shifts."

The base repo does not do this. It divides x by 1920 and y by 1080, which leaves the
representation sensitive to where the patient lies in the bed and where the camera is
mounted -- precisely the nuisance variation the methodology asks to remove, and which
would otherwise leak into z_behavior as if it were motor signature.

Joint order (15, after JOINT_INDICES subsetting):
   0 nose        1 L-eye       2 R-eye
   3 R-shoulder  4 R-elbow     5 R-wrist
   6 L-shoulder  7 L-elbow     8 L-wrist
   9 R-hip      10 R-knee     11 R-ankle
  12 L-hip      13 L-knee     14 L-ankle
"""
import torch
from torch import Tensor

import config


def _masked_midpoint(xy: Tensor, valid: Tensor, a: int, b: int):
    """Midpoint of joints a,b per frame, using whichever are valid.

    Returns (point[T,2], ok[T]). Where only one of the pair is valid that one is used;
    where neither is, ok is False and the caller substitutes a clip-level fallback.
    """
    pa, pb = xy[:, a, :], xy[:, b, :]
    va, vb = valid[:, a], valid[:, b]
    both = va & vb
    point = torch.zeros_like(pa)
    point[both] = 0.5 * (pa[both] + pb[both])
    only_a = va & ~vb
    only_b = vb & ~va
    point[only_a] = pa[only_a]
    point[only_b] = pb[only_b]
    return point, (va | vb)


def normalize_skeleton(kpts: Tensor, mode: str | None = None) -> Tensor:
    """(T, P, C>=2) pixel keypoints -> (T, P, KPT_CHANNELS) normalised.

    mode "midhip_torso" [METHOD]: centre on the mid-hip, scale by torso length
        (mid-shoulder to mid-hip), so the result is in torso-length units and is
        invariant to camera placement, patient position and apparent body size.
    mode "frame" [legacy]: divide by frame dimensions. Kept only so the base repo's
        behaviour can be reproduced for comparison.

    Joints the pose stage could not resolve are set to exactly 0 -- the origin, i.e.
    the mid-hip. Note the cost of that convention: with KPT_CHANNELS == 2 there is no
    longer any flag distinguishing "at the hip" from "unknown". Set KPT_CHANNELS = 3
    to retain the confidence channel and let the model tell them apart.
    """
    mode = mode or config.KPT_NORMALISATION
    kpts = kpts.float()
    xy = kpts[:, :, :2]
    conf = kpts[:, :, 2:3] if kpts.shape[-1] >= 3 else None

    # A joint is usable iff both coordinates are non-negative. This matches
    # preprocess.py, which skips patch extraction on `x < 0 or y < 0` -- so a joint
    # the pose stage extrapolated off the top/left edge is treated as unresolved
    # here exactly as it was there, rather than being silently normalised as real.
    valid = (xy[:, :, 0] >= 0) & (xy[:, :, 1] >= 0)

    if mode == "frame":
        out = torch.stack([xy[:, :, 0] / config.FRAME_WIDTH,
                           xy[:, :, 1] / config.FRAME_HEIGHT], dim=-1)
        out = out * valid.unsqueeze(-1)
    elif mode == "midhip_torso":
        hip_a, hip_b = config.MIDHIP_JOINTS
        sh_a, sh_b = config.TORSO_JOINTS
        midhip, hip_ok = _masked_midpoint(xy, valid, hip_a, hip_b)
        midsh, sh_ok = _masked_midpoint(xy, valid, sh_a, sh_b)

        # Clip-level fallback for frames with no usable hips: the median of the
        # frames that do have them. Median, not mean, so one wild pose failure in a
        # 30-frame clip cannot drag the origin across the image.
        if hip_ok.any():
            fallback_origin = midhip[hip_ok].median(dim=0).values
        else:
            fallback_origin = torch.tensor(
                [config.FRAME_WIDTH / 2.0, config.FRAME_HEIGHT / 2.0],
                dtype=xy.dtype, device=xy.device)
        origin = torch.where(hip_ok.unsqueeze(-1), midhip, fallback_origin)

        torso = torch.linalg.norm(midsh - midhip, dim=-1)
        torso_ok = hip_ok & sh_ok & (torso > config.TORSO_LENGTH_FLOOR)
        if torso_ok.any():
            fallback_scale = torso[torso_ok].median()
        else:
            fallback_scale = torch.tensor(
                config.TORSO_FALLBACK_FRAC * config.FRAME_HEIGHT,
                dtype=xy.dtype, device=xy.device)
        scale = torch.where(torso_ok, torso, fallback_scale).clamp_min(
            config.TORSO_LENGTH_FLOOR)

        out = (xy - origin.unsqueeze(1)) / scale.view(-1, 1, 1)
        out = out.clamp(-config.KPT_CLAMP, config.KPT_CLAMP)
        out = out * valid.unsqueeze(-1)
    else:
        raise ValueError(f"unknown KPT_NORMALISATION: {mode!r}")

    if config.KPT_CHANNELS == 3:
        c = conf if conf is not None else valid.unsqueeze(-1).float()
        return torch.cat([out, c.clamp(0.0, 1.0)], dim=-1)
    return out
