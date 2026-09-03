"""Turning per-clip probabilities into discrete clinical alarms.

Everything here is pure Python/NumPy on (time, score) pairs -- no torch, no model.
That is deliberate: the decision rule is where the previous implementation went wrong
in two separate ways, so it should be testable without a GPU or a checkpoint.

WHAT WENT WRONG BEFORE, and what this fixes:

  * The old evaluator read a clip's filename suffix -- which preprocess.py writes as
    SECONDS -- as a frame index, then divided by an assumed 25 fps. Every timestamp
    was 25x too small, so tau = 3 s actually spanned 75 s of real time. Here, time is
    seconds throughout and is never converted.

  * It counted every thresholded window as a separate false alarm, so one 15 s bout of
    atypical movement could register a dozen. Alarms are now grouped into events under
    a refractory period, which is what makes FDR/h mean "how often is the ward
    interrupted" rather than "how many windows crossed a threshold".
"""
from dataclasses import dataclass

import numpy as np

import config


@dataclass(frozen=True)
class AlarmEvent:
    onset_s: float          # when the alarm opened
    peak_ap: float          # highest accumulated probability while it was open
    n_windows: int          # thresholded windows merged into this event


def decision_times(t_start_s, clip_seconds: float | None = None, ref: str | None = None):
    """Map clip start times to the instant each clip's prediction is attributed to."""
    clip_seconds = config.CLIP_SECONDS if clip_seconds is None else clip_seconds
    ref = ref or config.DETECTION_TIME_REF
    t = np.asarray(t_start_s, dtype=float)
    if ref == "clip_start":
        return t
    if ref == "clip_center":
        return t + clip_seconds / 2.0
    if ref == "clip_end":
        return t + clip_seconds
    raise ValueError(f"unknown DETECTION_TIME_REF: {ref!r}")


def accumulate(times_s, scores, tau_s: float | None = None, rule: str | None = None):
    """Accumulated probability at each clip's decision time.

    AP_t aggregates every clip whose decision time lies in (t - tau, t] -- the current
    clip included, nothing from the future.

    The paper writes AP_t = sum(P_i) with DT = 0.3. Summing ~6 sigmoid outputs against
    0.3 is satisfied almost unconditionally and cannot reproduce the reported
    latencies, so the default here is the mean. That is our reading, not the paper's
    text; see config.ACCUM_RULE.
    """
    tau_s = config.ACCUM_WINDOW_S if tau_s is None else tau_s
    rule = rule or config.ACCUM_RULE

    times_s = np.asarray(times_s, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(times_s, kind="stable")
    t, s = times_s[order], scores[order]

    # Window start index for each t: first clip with time > t - tau.
    lo = np.searchsorted(t, t - tau_s, side="right")
    csum = np.concatenate([[0.0], np.cumsum(s)])
    hi = np.arange(1, len(t) + 1)
    totals = csum[hi] - csum[lo]
    counts = hi - lo

    if rule == "sum":
        ap = totals
    elif rule == "mean":
        ap = np.divide(totals, counts, out=np.zeros_like(totals),
                       where=counts > 0)
    else:
        raise ValueError(f"unknown ACCUM_RULE: {rule!r}")
    return t, ap


def find_alarms(times_s, scores, *, threshold=None, tau_s=None, rule=None,
                refractory_s=None):
    """Group thresholded windows into discrete alarm events.

    An event opens the first time AP exceeds the threshold. It stays open while AP
    remains above it. A NEW event may only open once BOTH conditions hold:
      1. AP has fallen back below the threshold (the network returned to a stable
         normal classification), and
      2. the refractory period has elapsed since the previous event opened.
    Both are required -- a sustained alarm that never relaxes is one event no matter
    how long it lasts, and a flickering alarm inside the refractory window is still
    the same clinical interruption.

    `times_s` are clip START times; returned onsets are DECISION times.

    Note what this is and is not for: refractory grouping answers "how often is the
    ward interrupted", so it governs the FALSE-ALARM count. It must not govern whether
    a seizure counts as detected -- sensitivity should not depend on whether an
    unrelated false alarm happened to fire 45 s earlier. metrics.evaluate_source
    therefore measures detection from the raw AP series, not from these events.
    """
    threshold = config.DECISION_THRESHOLD if threshold is None else threshold
    refractory_s = config.REFRACTORY_S if refractory_s is None else refractory_s

    # `times_s` are clip START times. Convert to decision instants FIRST, so alarm
    # onsets live on the same clock as the exposure mask and the onset annotations.
    # (Accumulating on start times while measuring exposure on end times puts the two
    # halves of the metric 5 s out of step with each other.)
    t, ap = accumulate(decision_times(times_s), scores, tau_s=tau_s, rule=rule)
    events: list[AlarmEvent] = []
    open_event = False
    relaxed_since_last = True
    onset = peak = None
    n_win = 0

    for ti, api in zip(t, ap):
        above = api > threshold
        if above:
            if not open_event:
                can_open = relaxed_since_last and (
                    not events or (ti - events[-1].onset_s) >= refractory_s)
                if can_open:
                    open_event = True
                    onset, peak, n_win = ti, api, 1
                    relaxed_since_last = False
                else:
                    # Suppressed: inside the refractory shadow of the previous event.
                    if events:
                        prev = events[-1]
                        events[-1] = AlarmEvent(prev.onset_s, max(prev.peak_ap, api),
                                                prev.n_windows + 1)
            else:
                peak = max(peak, api)
                n_win += 1
        else:
            if open_event:
                events.append(AlarmEvent(onset, peak, n_win))
                open_event = False
            relaxed_since_last = True

    if open_event:
        events.append(AlarmEvent(onset, peak, n_win))
    return events, t, ap
