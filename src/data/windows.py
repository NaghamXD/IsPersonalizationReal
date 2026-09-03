"""Which 5-second windows to extract from a recording, and why.

Pure arithmetic -- no video, no pose model, no torch -- so the sampling plan can be
tested exhaustively without decoding 24 GB of MP4.

The methodology bifurcates extraction by phase, and that bifurcation is the whole
point of this module:

  TRAIN  ictal and transition clips overlap by 4 s (1 s hop) as shift-invariant
         augmentation for the minority classes; interictal does not overlap.
  TEST   nothing overlaps. A continuous 5 s sliding window with a 5 s hop, so each
         instant of footage is scored exactly once.

Scoring overlapping clips at test time inflates apparent performance twice over: the
accumulation window sees the same movement several times, and a seizure gets several
independent chances to be detected. Latency measured that way is not latency.
"""
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Window:
    t_start_s: float
    label: float
    phase: str          # "interictal" | "transition" | "ictal"


def transition_label(t_end_s: float, eeg_s: float, clin_s: float,
                     k: float | None = None) -> float:
    """Label at a clip's END, per the paper's rule that a clip's probability follows
    which period its end frame falls in.

    [PAPER] gives only "an exponential function ... ranging from 0 to 1".
    [DECISION] k and the normalisation are ours; see config.TRANSITION_RAMP_K.
    """
    import math
    k = config.TRANSITION_RAMP_K if k is None else k
    if t_end_s < eeg_s:
        return 0.0
    if t_end_s > clin_s:
        return 1.0
    span = clin_s - eeg_s
    if span <= 0:
        return 1.0
    x = (t_end_s - eeg_s) / span
    return float((math.exp(k * x) - 1.0) / (math.exp(k) - 1.0))


def phase_of(t_end_s: float, eeg_s, clin_s) -> str:
    if eeg_s is None or clin_s is None:
        return "interictal"
    if t_end_s < eeg_s:
        return "interictal"
    if t_end_s <= clin_s:
        return "transition"
    return "ictal"


def _frange(start: float, stop: float, step: float):
    """Inclusive-of-start, exclusive-past-stop float range without accumulating error."""
    if step <= 0:
        raise ValueError("step must be positive")
    n = 0
    while True:
        t = start + n * step
        if t > stop + 1e-9:
            return
        yield round(t, 6)
        n += 1


def plan_windows(duration_s: float, eeg_s=None, clin_s=None, *, mode: str,
                 clip_seconds=None, lookback_s=None, lookahead_s=None):
    """Window plan for one recording.

    mode "test":  uniform non-overlapping sweep of everything evaluable.
    mode "train": phase-dependent stride, matching the methodology's augmentation.

    A recording with no annotated onsets (free.mp4, no-Sz2P.mp4) is entirely
    interictal and is swept end to end in both modes.
    """
    clip_seconds = config.CLIP_SECONDS if clip_seconds is None else clip_seconds
    lookback_s = config.INTERICTAL_LOOKBACK_S if lookback_s is None else lookback_s
    lookahead_s = config.ICTAL_LOOKAHEAD_S if lookahead_s is None else lookahead_s

    last_start = duration_s - clip_seconds
    if last_start < 0:
        return []

    if eeg_s is None or clin_s is None:
        lo, hi = 0.0, last_start
    else:
        lo = max(0.0, eeg_s - lookback_s)
        hi = min(last_start, clin_s + lookahead_s)
    if hi < lo:
        return []

    out: list[Window] = []
    if mode == "test":
        for t in _frange(lo, hi, config.TEST_STRIDE_S):
            t_end = t + clip_seconds
            out.append(Window(t, transition_label(t_end, eeg_s, clin_s)
                              if eeg_s is not None else 0.0,
                              phase_of(t_end, eeg_s, clin_s)))
        return out

    if mode == "train":
        # Stride depends on the phase the window ENDS in, so the dense stride starts
        # exactly where the transition begins rather than a window early.
        t = lo
        while t <= hi + 1e-9:
            t_end = t + clip_seconds
            ph = phase_of(t_end, eeg_s, clin_s)
            lab = transition_label(t_end, eeg_s, clin_s) if eeg_s is not None else 0.0
            out.append(Window(round(t, 6), lab, ph))
            step = (config.TRAIN_STRIDE_INTERICTAL_S if ph == "interictal"
                    else config.TRAIN_STRIDE_TRANSITION_S if ph == "transition"
                    else config.TRAIN_STRIDE_ICTAL_S)
            t += step
        return out

    raise ValueError(f"unknown mode {mode!r}")


def coverage_seconds(windows, clip_seconds=None) -> dict:
    """Evaluated coverage by phase, in seconds.

    For a non-overlapping plan this is the exposure that goes into the FDR/h
    denominator, which is why it is computed from the plan itself rather than from
    a file's wall-clock duration: footage the model was never run on is not time it
    was at risk of alarming.
    """
    clip_seconds = config.CLIP_SECONDS if clip_seconds is None else clip_seconds
    out = {"interictal": 0.0, "transition": 0.0, "ictal": 0.0}
    for w in windows:
        out[w.phase] += clip_seconds
    out["total"] = sum(out.values())
    return out
