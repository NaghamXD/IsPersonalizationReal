"""Tests for the decision rule and clinical metrics.

Pure NumPy -- no torch, no checkpoint -- so the part of the pipeline that was
previously wrong in two independent ways can be verified anywhere.

Run:  python tests/test_eval.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.eval.decision import accumulate, decision_times, find_alarms
from src.eval.metrics import aggregate, evaluate_source


def test_decision_times():
    t = [0.0, 5.0, 10.0]
    assert np.allclose(decision_times(t, 5.0, "clip_end"), [5.0, 10.0, 15.0])
    assert np.allclose(decision_times(t, 5.0, "clip_start"), [0.0, 5.0, 10.0])
    assert np.allclose(decision_times(t, 5.0, "clip_center"), [2.5, 7.5, 12.5])
    print("  OK  decision_times")


def test_units_are_seconds_regression():
    """Regression guard for the bug that corrupted every latency in the old evaluator.

    preprocess.py names clips `{pat}_{sz}_{int(round(t_seconds))}`. The old code read
    that suffix as a FRAME INDEX and divided by an assumed 25 fps, so a clip starting
    at 1780 s was placed at 71.2 s. Nothing in this module may divide by a frame rate.
    """
    from src.utils.naming import parse_clip_name
    patient, event, t = parse_clip_name("pat01_Sz1_1780")
    assert (patient, event) == ("pat01", "Sz1")
    assert t == 1780.0, "clip suffix is SECONDS, not frames"
    assert decision_times([t])[0] == 1785.0, "no frame-rate conversion anywhere"
    # The old behaviour would have produced 1780/25 = 71.2 s.
    assert abs(decision_times([t])[0] - 71.2) > 1000
    print("  OK  units are seconds (regression)")


def test_accumulate_mean():
    # Clips every 1 s, tau = 3 s. AP at t is the mean over (t-3, t].
    t = np.arange(0.0, 6.0, 1.0)
    s = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    dt, ap = accumulate(t, s, tau_s=3.0, rule="mean")
    assert np.isclose(ap[0], 0.0)                       # {0}
    assert np.isclose(ap[1], 0.5)                       # {0,1}
    assert np.isclose(ap[2], 1 / 3)                     # {0,1,0}
    assert np.isclose(ap[3], 2 / 3)                     # {1,0,1} -- t=0 has expired
    assert np.isclose(ap[5], 1.0)                       # {1,1,1}
    _, ap_sum = accumulate(t, s, tau_s=3.0, rule="sum")
    assert np.isclose(ap_sum[5], 3.0)
    print("  OK  accumulate (mean and sum)")


def test_accumulate_is_causal_and_order_free():
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 50.0, 5.0)
    s = rng.random(len(t))
    perm = rng.permutation(len(t))
    a = accumulate(t, s)[1]
    b = accumulate(t[perm], s[perm])[1]
    assert np.allclose(a, b), "AP must not depend on input ordering"
    # Causality: a huge score in the future must not change an earlier AP.
    s2 = s.copy(); s2[-1] = 1e3
    assert np.allclose(accumulate(t, s2)[1][:-1], a[:-1]), "AP must not see the future"
    print("  OK  accumulate is causal and order-free")


def _burst(start, n, step=1.0, hi=1.0):
    return [(start + i * step, hi) for i in range(n)]


def test_alarm_grouping():
    thr, tau, refr = 0.3, 3.0, 60.0

    def alarms(pairs):
        pairs = sorted(pairs)
        t = [p[0] for p in pairs]; s = [p[1] for p in pairs]
        return find_alarms(t, s, threshold=thr, tau_s=tau, rule="mean",
                           refractory_s=refr)[0]

    quiet = [(x, 0.0) for x in np.arange(0.0, 200.0, 1.0)]

    # A sustained 12 s bout is ONE event, not twelve windows.
    base = dict(quiet)
    for tt, v in _burst(10.0, 12):
        base[tt] = v
    ev = alarms(list(base.items()))
    assert len(ev) == 1, f"sustained bout must be one event, got {len(ev)}"
    # Onsets are DECISION times, so a bout of clips starting at t=10 opens the alarm
    # at t=10+CLIP_SECONDS -- the first instant the clip could have been scored.
    lo = 10.0 + config.CLIP_SECONDS
    assert lo <= ev[0].onset_s <= lo + 2.0, ev[0].onset_s

    # A second bout 30 s later is inside the refractory shadow -> still one event.
    # (Decision times 15 and 45: 30 s apart, inside the 60 s refractory.)
    b2 = dict(base)
    for tt, v in _burst(40.0, 5):
        b2[tt] = v
    ev = alarms(list(b2.items()))
    assert len(ev) == 1, f"bout inside refractory must merge, got {len(ev)}"

    # A second bout 90 s later is a genuinely separate clinical interruption.
    # (Decision times 15 and 105: 90 s apart, past the 60 s refractory.)
    b3 = dict(base)
    for tt, v in _burst(100.0, 5):
        b3[tt] = v
    ev = alarms(list(b3.items()))
    assert len(ev) == 2, f"bout past refractory must be its own event, got {len(ev)}"
    print("  OK  alarm grouping and refractory")


def test_metrics_detection_and_exposure():
    # 100 s recording, clips every 5 s. EEG onset 60 s, clinical onset 70 s.
    t = np.arange(0.0, 100.0, 5.0)
    eeg, clin = 60.0, 70.0
    scores = np.where(t + config.CLIP_SECONDS >= eeg, 1.0, 0.0)

    r = evaluate_source("pat01", "pat01_Sz1", t, scores, eeg_s=eeg, clin_s=clin)
    assert r.detected, "an all-ones ictal run must be detected"
    assert r.n_false_alarms == 0, f"no pre-onset alarms expected, got {r.n_false_alarms}"
    assert r.l_eo_s is not None and r.l_eo_s >= 0
    assert r.l_co_s is not None and r.l_co_s < r.l_eo_s, "L_CO must lead L_EO by the transition"

    # Exposure counts only clips whose decision time precedes EEG onset:
    # decision times 5,10,...,100 -> 11 of them below 60 -> 55 s.
    assert np.isclose(r.exposure_s, 55.0), r.exposure_s

    # A pre-onset burst is a false alarm and does NOT count as a detection.
    s2 = scores.copy()
    s2[(t >= 10.0) & (t <= 20.0)] = 1.0
    r2 = evaluate_source("pat01", "pat01_Sz1", t, s2, eeg_s=eeg, clin_s=clin)
    assert r2.n_false_alarms == 1, f"expected 1 pre-onset false alarm, got {r2.n_false_alarms}"
    assert r2.detected

    # Free footage: no onsets, so everything is exposure and every alarm is false.
    tf = np.arange(0.0, 60.0, 5.0)
    sf = np.zeros_like(tf); sf[2:5] = 1.0
    rf = evaluate_source("pat03", "pat03_free", tf, sf)
    assert not rf.has_seizure
    assert np.isclose(rf.exposure_s, len(tf) * config.CLIP_SECONDS)
    assert rf.n_false_alarms == 1
    print("  OK  detection, latency and exposure")


def test_post_ictal_exclusion():
    """Footage inside the post-ictal recovery window is not exposure."""
    t = np.arange(0.0, 1400.0, 5.0)
    eeg, clin = 60.0, 70.0
    r = evaluate_source("patX", "patX_Sz1", t, np.zeros_like(t), eeg_s=eeg, clin_s=clin)
    pre = 11 * config.CLIP_SECONDS                       # decision times < 60
    post_start = clin + config.POST_ICTAL_EXCLUSION_S    # 970 s
    dec = t + config.CLIP_SECONDS
    expected = pre + float((dec >= post_start).sum()) * config.CLIP_SECONDS
    assert np.isclose(r.exposure_s, expected), (r.exposure_s, expected)
    assert r.exposure_s < len(t) * config.CLIP_SECONDS, "recovery window must be excluded"
    print("  OK  post-ictal exclusion")


def test_aggregate_rate_arithmetic():
    from src.eval.metrics import SourceResult
    rs = [
        SourceResult("p1", "a", detected=True, l_eo_s=4.0, l_co_s=-6.0,
                     has_seizure=True, n_false_alarms=1, exposure_s=1800.0),
        SourceResult("p1", "b", detected=False, has_seizure=True,
                     n_false_alarms=2, exposure_s=1800.0),
    ]
    agg = aggregate(rs)
    assert agg["n_seizures"] == 2 and agg["n_detected"] == 1
    assert np.isclose(agg["sensitivity"], 0.5)
    assert np.isclose(agg["exposure_hours"], 1.0)
    assert np.isclose(agg["fdr_per_hour"], 3.0), agg["fdr_per_hour"]
    assert agg["n_false_alarms"] == 3, "raw count must survive alongside the rate"
    print("  OK  aggregate rate arithmetic")


def test_exposure_disparity_is_visible():
    """The cohort's exposure spans orders of magnitude; the report must expose that.

    pat13 has 37 s of pre-EEG footage and pat04 has 45.6 min. One identical false alarm
    yields wildly different rates, which is exactly why count and hours are reported
    beside the rate and why section 3.5 uses a log-exposure offset.
    """
    from src.eval.metrics import SourceResult
    scarce = aggregate([SourceResult("pat13", "s", has_seizure=True,
                                     n_false_alarms=1, exposure_s=37.0)])
    rich = aggregate([SourceResult("pat04", "r", has_seizure=True,
                                   n_false_alarms=1, exposure_s=2736.0)])
    assert scarce["fdr_per_hour"] / rich["fdr_per_hour"] > 50
    print(f"  OK  exposure disparity visible "
          f"({scarce['fdr_per_hour']:.1f} vs {rich['fdr_per_hour']:.2f} FDR/h "
          f"for the SAME single false alarm)")


if __name__ == "__main__":
    for fn in [test_decision_times, test_units_are_seconds_regression,
               test_accumulate_mean, test_accumulate_is_causal_and_order_free,
               test_alarm_grouping, test_metrics_detection_and_exposure,
               test_post_ictal_exclusion, test_aggregate_rate_arithmetic,
               test_exposure_disparity_is_visible]:
        fn()
    print("\nALL EVAL TESTS PASSED")
