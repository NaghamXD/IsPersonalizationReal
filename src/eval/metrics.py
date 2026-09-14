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

def _average_ranks(a):
    """Tie-corrected average ranks, 1-based. NumPy only -- no scipy dependency for a
    function this small, so the metric is importable anywhere the tests run."""
    import numpy as np
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    sorted_a, ranks, n, i = a[order], np.empty(len(a), dtype=float), len(a), 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(scores, positive):
    """Rank-based AUC (Mann-Whitney U), tie-corrected.

    The probability that a randomly chosen positive scores above a randomly chosen
    negative. 0.5 is chance. Threshold-free, so it says whether the model ORDERS clips
    correctly independently of where DT sits -- which is what a detector actually needs
    and what MSE does not measure.

    This distinction is not academic here: the checkpoint selected by best validation
    MSE scored AUC 0.513, i.e. MSE chose a model that cannot discriminate at all.

    `positive` is a boolean/0-1 mask. Clips that are neither (soft transition labels)
    must be excluded by the caller -- they have no unambiguous class.
    """
    import numpy as np

    s = np.asarray(scores, dtype=float)
    y = np.asarray(positive).astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _average_ranks(s)                 # ties share the mean rank -> count 0.5 each
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auc_from_labels(scores, labels):
    """AUC of ictal (1.0) vs interictal (0.0), dropping soft transition labels."""
    import numpy as np
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    keep = (y == 0.0) | (y == 1.0)
    if keep.sum() == 0:
        return float("nan")
    return roc_auc(s[keep], y[keep] == 1.0)


def within_source_auc(scores, labels, sources):
    """Stratified AUC: only (ictal, interictal) pairs from the SAME recording count.

    [DECISION D19] This, not the pooled AUC, is the selection and reporting metric.

    Pooling clips across recordings manufactures ranking signal that has nothing to do
    with seizure detection. Recordings sit at different baseline score levels, and the
    ictal/interictal mix differs per recording, so a model that merely scores recording
    B above recording A earns AUC above 0.5 even when it orders nothing correctly
    inside either one. Measured on fold 1 (held-out pat01) this was worth +0.07 to
    +0.10 everywhere it could be checked:

        pat01 (held out)     pooled 0.625   within-source 0.511, 0.547
        pat04 (internal val) pooled 0.836   within-source 0.762
        pat07 (internal val) pooled 0.942   within-source 0.860, 0.890

    The held-out patient is at chance inside each of its own recordings; the pooled
    number says 0.625. A baseline-versus-adapted comparison run on the pooled metric
    would credit personalisation with between-recording offsets.

    The estimator is the Mann-Whitney statistic restricted to within-source pairs,
    i.e. the pair-count-weighted mean of the per-source AUCs:

        sum_s (n_pos_s * n_neg_s * AUC_s) / sum_s (n_pos_s * n_neg_s)

    Weighting by pair count rather than averaging the per-source AUCs equally is what
    makes it a single Mann-Whitney estimate, and it needs no minimum-clips-per-source
    rule: a recording holding only one class contributes zero pairs and drops out on
    its own, instead of being discarded by an arbitrary threshold. The unweighted mean
    ("macro") is returned alongside, because the two separating is itself informative
    -- it means the usable recordings disagree and the pair-weighted figure is being
    carried by whichever recording happens to be longest.

    Returns a dict:
        pair_weighted   float   PRIMARY -- selection and reporting
        macro           float   unweighted mean over usable sources
        pooled          float   the old metric, kept only for comparison
        per_source      {source: {"auc", "n_pos", "n_neg"}}
        n_sources_used  int     sources contributing at least one pair
        n_sources_seen  int
        n_pairs         int
    """
    import numpy as np

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    g = np.asarray([str(x) for x in sources])
    if not (len(s) == len(y) == len(g)):
        raise ValueError(f"length mismatch: {len(s)} scores, {len(y)} labels, "
                         f"{len(g)} sources")

    keep = (y == 0.0) | (y == 1.0)          # soft transition labels have no class
    s, y, g = s[keep], y[keep], g[keep]

    per, num, den, macro = {}, 0.0, 0.0, []
    by_patient = {}
    for src in dict.fromkeys(g.tolist()):   # insertion order, deterministic, str keys
        m = g == src
        pos, neg = int((y[m] == 1.0).sum()), int((y[m] == 0.0).sum())
        a = roc_auc(s[m], y[m] == 1.0)
        per[src] = {"auc": a, "n_pos": pos, "n_neg": neg}
        if pos and neg:
            w = float(pos * neg)
            num += w * a
            den += w
            macro.append(a)
            by_patient.setdefault(str(src).split("_")[0], []).append((a, w))

    # [D19a] Pair-count weighting across PATIENTS is wrong, and fold 1 showed why:
    # pat04 contributes one seizure recording and pat07 two, so the pair weights made
    # the combined validation metric numerically identical to pat07 alone
    # (corr = +1.000, while corr with pat04 was +0.425). A metric for a study about
    # patient heterogeneity cannot be one patient. Pairs still weight recordings
    # WITHIN a patient; patients themselves are then averaged equally.
    pb = [sum(a * w for a, w in v) / sum(w for _, w in v) for v in by_patient.values()]

    return {
        "patient_balanced": float(np.mean(pb)) if pb else float("nan"),
        "n_patients_used": len(pb),
        "pair_weighted": (num / den) if den > 0 else float("nan"),
        "macro": float(np.mean(macro)) if macro else float("nan"),
        "pooled": auc_from_labels(s, y),
        "per_source": per,
        "n_sources_used": len(macro),
        "n_sources_seen": len(per),
        "n_pairs": int(den),
    }


def eval_pairs(per_source) -> int:
    """Ordered ictal-vs-interictal pairs behind a within-source AUC.

    The AUC's resolution is 1/pairs. D35 excludes a fold from aggregate AUC analyses
    when this falls below config.MIN_EVAL_PAIRS, because below that the metric's step
    size exceeds the effect under test and the number is not evidence either way.
    """
    return int(sum(d["n_pos"] * d["n_neg"] for d in per_source.values()))


def auc_resolvable(per_source, minimum=None) -> bool:
    minimum = config.MIN_EVAL_PAIRS if minimum is None else minimum
    return eval_pairs(per_source) >= minimum
