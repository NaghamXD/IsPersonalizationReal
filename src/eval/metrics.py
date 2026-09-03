"""Clinical metrics: sensitivity, detection latency, and FDR per hour.

The denominator is the part that needs stating plainly, because the VSViG paper never
defines it. Hours at risk are the interictal seconds ACTUALLY EVALUATED for a patient,
excluding:
  * the pre-ictal transition window  [eeg_onset, clinical_onset)
  * the ictal window                 [clinical_onset, clinical_onset + lookahead)
  * the post-ictal recovery window   [clinical_onset, clinical_onset + 900 s)

Post-ictal thrashing and disorientation are not a resting state, so counting a flag
there as a false alarm would misrepresent the model. (In this corpus that rule almost
never binds -- the seizure files end 41-140 s after clinical onset -- but it is
implemented so the metric stays correct on any future recording.)

Exposure is measured as evaluated clip coverage rather than wall-clock file duration:
a stretch of video the model was never run on is not time it was at risk of alarming.
"""
from dataclasses import dataclass, field

import numpy as np

import config
from src.eval.decision import find_alarms, decision_times


@dataclass
class SourceResult:
    """One continuous recording (one seizure file, or one free-footage file)."""
    patient: str
    source: str
    detected: bool = False
    detection_time_s: float | None = None
    l_eo_s: float | None = None
    l_co_s: float | None = None
    has_seizure: bool = False
    n_false_alarms: int = 0
    exposure_s: float = 0.0
    false_alarm_times: list = field(default_factory=list)


def _interictal_mask(dec_t, eeg_s, clin_s):
    """True where a decision time counts as valid interictal exposure."""
    dec_t = np.asarray(dec_t, dtype=float)
    if eeg_s is None or clin_s is None:
        return np.ones_like(dec_t, dtype=bool)      # free footage: all interictal
    ok = dec_t < eeg_s                              # before EEG onset
    if config.POST_ICTAL_EXCLUSION_S is not None:
        ok |= dec_t >= (clin_s + config.POST_ICTAL_EXCLUSION_S)
    return ok


def evaluate_source(patient, source, t_start_s, scores, eeg_s=None, clin_s=None,
                    clip_seconds=None):
    """Score one recording. Returns a SourceResult."""
    clip_seconds = config.CLIP_SECONDS if clip_seconds is None else clip_seconds
    events, dec_t, ap = find_alarms(t_start_s, scores)
    dec_t_all = decision_times(t_start_s, clip_seconds)

    res = SourceResult(patient=patient, source=source,
                       has_seizure=(eeg_s is not None and clin_s is not None))

    # Exposure: non-overlapping evaluated clips inside valid interictal regions.
    valid = _interictal_mask(dec_t_all, eeg_s, clin_s)
    res.exposure_s = float(valid.sum()) * clip_seconds

    if not res.has_seizure:
        res.n_false_alarms = len(events)
        res.false_alarm_times = [e.onset_s for e in events]
        return res

    det_hi = clin_s + config.DETECTION_WINDOW_AFTER_CLINICAL_S

    # FALSE ALARMS: grouped events opening before EEG onset. An alarm that precedes
    # electrographic onset is a false alarm by the ground truth, not an early
    # detection -- that is the standard convention and it is why L_EO is reported
    # as a non-negative latency.
    for e in events:
        if e.onset_s < eeg_s:
            res.n_false_alarms += 1
            res.false_alarm_times.append(e.onset_s)

    # DETECTION: measured from the raw AP series, deliberately bypassing refractory
    # grouping. Otherwise a false alarm 45 s before onset would suppress the alarm
    # that actually detects the seizure, and sensitivity would silently depend on
    # unrelated interictal noise.
    in_window = (dec_t >= eeg_s) & (dec_t <= det_hi) & (ap > config.DECISION_THRESHOLD)
    if in_window.any():
        t_det = float(dec_t[in_window][0])
        res.detected = True
        res.detection_time_s = t_det
        res.l_eo_s = t_det - eeg_s
        res.l_co_s = t_det - clin_s
    return res


def aggregate(results):
    """Pool SourceResults into the headline numbers.

    False-alarm COUNT and EXPOSURE HOURS are reported alongside the rate, never folded
    away into it. Exposure spans three orders of magnitude across this cohort (pat13
    has 37 s of pre-EEG footage, pat04 has 45.6 min), so a bare FDR/h is dominated by
    recording length. Section 3.5 fits a Poisson rate model with a log-exposure offset
    for exactly this reason; these fields are its inputs.
    """
    n_sz = sum(1 for r in results if r.has_seizure)
    n_det = sum(1 for r in results if r.has_seizure and r.detected)
    fa = sum(r.n_false_alarms for r in results)
    exposure_s = sum(r.exposure_s for r in results)
    hours = exposure_s / 3600.0

    leo = [r.l_eo_s for r in results if r.l_eo_s is not None]
    lco = [r.l_co_s for r in results if r.l_co_s is not None]

    return {
        "n_seizures": n_sz,
        "n_detected": n_det,
        "sensitivity": (n_det / n_sz) if n_sz else None,
        "n_false_alarms": fa,
        "exposure_hours": hours,
        "fdr_per_hour": (fa / hours) if hours > 0 else None,
        "mean_l_eo_s": float(np.mean(leo)) if leo else None,
        "median_l_eo_s": float(np.median(leo)) if leo else None,
        "mean_l_co_s": float(np.mean(lco)) if lco else None,
        "median_l_co_s": float(np.median(lco)) if lco else None,
    }
