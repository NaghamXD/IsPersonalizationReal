"""Tests for the extraction plan.

Pure arithmetic, so the sampling protocol can be checked exhaustively without
decoding any video. The property that matters most is the one the methodology is
explicit about and the previous pipeline never implemented: TEST WINDOWS DO NOT
OVERLAP.

Run:  python tests/test_windows.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data.windows import (Window, coverage_seconds, phase_of, plan_windows,
                              transition_label)


def test_transition_ramp():
    eeg, clin = 100.0, 120.0
    assert transition_label(99.0, eeg, clin) == 0.0
    assert transition_label(121.0, eeg, clin) == 1.0
    assert transition_label(100.0, eeg, clin) == 0.0
    assert abs(transition_label(120.0, eeg, clin) - 1.0) < 1e-9
    mid = transition_label(110.0, eeg, clin)
    assert 0.0 < mid < 1.0
    # Exponential, so the midpoint sits BELOW linear: risk climbs late, which is the
    # clinical claim the ramp encodes.
    assert mid < 0.5, mid
    # Monotone increasing across the transition.
    vals = [transition_label(t, eeg, clin) for t in range(100, 121)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    # A zero-length transition degenerates to ictal, not to a divide-by-zero.
    assert transition_label(100.0, 100.0, 100.0) == 1.0
    print("  OK  transition ramp")


def test_test_windows_do_not_overlap():
    w = plan_windows(600.0, eeg_s=300.0, clin_s=320.0, mode="test")
    starts = [x.t_start_s for x in w]
    gaps = {round(b - a, 6) for a, b in zip(starts, starts[1:])}
    assert gaps == {config.TEST_STRIDE_S}, gaps
    assert config.TEST_STRIDE_S == config.CLIP_SECONDS, \
        "a 5 s hop on a 5 s window is what makes coverage exactly-once"
    # Every instant scored at most once: window i ends exactly where i+1 begins.
    for a, b in zip(w, w[1:]):
        assert a.t_start_s + config.CLIP_SECONDS <= b.t_start_s + 1e-9
    print(f"  OK  test windows do not overlap ({len(w)} windows)")


def test_train_windows_do_overlap_in_the_right_places():
    w = plan_windows(600.0, eeg_s=300.0, clin_s=320.0, mode="train")
    by_phase = {}
    for a, b in zip(w, w[1:]):
        by_phase.setdefault(a.phase, set()).add(round(b.t_start_s - a.t_start_s, 6))
    assert by_phase["interictal"] == {config.TRAIN_STRIDE_INTERICTAL_S}, by_phase
    assert by_phase["ictal"] == {config.TRAIN_STRIDE_ICTAL_S}, by_phase
    assert config.TRAIN_STRIDE_ICTAL_S < config.CLIP_SECONDS, "ictal must overlap"
    # Training must produce strictly more clips than testing on the same recording.
    assert len(w) > len(plan_windows(600.0, 300.0, 320.0, mode="test"))
    print("  OK  train windows overlap only on ictal/transition")


def test_free_footage_is_all_interictal():
    w = plan_windows(660.0, mode="test")            # pat03/free.mp4 is 11.0 min
    assert all(x.phase == "interictal" and x.label == 0.0 for x in w)
    assert w[0].t_start_s == 0.0
    assert w[-1].t_start_s + config.CLIP_SECONDS <= 660.0 + 1e-9
    assert len(w) == 132, len(w)                    # starts 0,5,...,655 inclusive
    print(f"  OK  free footage swept end to end ({len(w)} windows)")


def test_windows_stay_inside_the_file():
    for dur in [5.0, 7.3, 100.0, 660.0]:
        for w in plan_windows(dur, mode="test"):
            assert w.t_start_s >= 0.0
            assert w.t_start_s + config.CLIP_SECONDS <= dur + 1e-9, (dur, w)
    assert plan_windows(4.9, mode="test") == [], "a file shorter than one clip yields none"
    assert plan_windows(0.0, mode="test") == []
    print("  OK  windows never run past the end of the file")


def test_lookback_clamps_at_zero():
    # pat03 Sz1: EEG onset at 7 s, so the 30-minute lookback has nothing to reach for.
    w = plan_windows(78.0, eeg_s=7.0, clin_s=13.0, mode="test")
    assert w[0].t_start_s == 0.0
    inter = [x for x in w if x.phase == "interictal"]
    # Exactly one window ends before EEG onset at 7 s: the one starting at 0.
    assert len(inter) == 1, [x.t_start_s for x in inter]
    assert coverage_seconds(w)["interictal"] == config.CLIP_SECONDS
    # Which is the whole reason free.mp4 exists for this patient: 5 s of usable
    # interictal exposure per seizure file is not a baseline.
    print("  OK  lookback clamps at zero (pat03 Sz1 yields 5 s of interictal)")


def test_real_recording_pat13():
    """pat13/Sz1PG.mp4 -- 1.5 min, EEG onset 37 s, clinical onset 43 s.

    Pinned to real corpus values because this patient is the scarcity stress case: it
    is where a Pool A of 6 and the FDR/h denominator are both tightest.
    """
    w = plan_windows(90.0, eeg_s=37.0, clin_s=43.0, mode="test")
    assert len(w) == 18, len(w)
    cov = coverage_seconds(w)
    assert cov["interictal"] == 35.0, cov
    assert cov["transition"] == 5.0, cov
    assert cov["ictal"] == 50.0, cov
    assert cov["total"] == 90.0 - 0.0, cov
    # 35 s of interictal exposure: ONE false alarm here is 103 FDR/h. This is the
    # disparity the Poisson log-exposure offset exists to absorb.
    fdr_for_one = 1.0 / (cov["interictal"] / 3600.0)
    assert fdr_for_one > 100, fdr_for_one
    print(f"  OK  pat13 plan matches the real file "
          f"({len(w)} windows, {cov['interictal']:.0f}s interictal "
          f"-> 1 false alarm = {fdr_for_one:.0f} FDR/h)")


def test_coverage_arithmetic():
    ws = [Window(0.0, 0.0, "interictal"), Window(5.0, 0.0, "interictal"),
          Window(10.0, 0.4, "transition"), Window(15.0, 1.0, "ictal")]
    cov = coverage_seconds(ws)
    assert cov == {"interictal": 10.0, "transition": 5.0, "ictal": 5.0, "total": 20.0}
    print("  OK  coverage arithmetic")


if __name__ == "__main__":
    for fn in [test_transition_ramp, test_test_windows_do_not_overlap,
               test_train_windows_do_overlap_in_the_right_places,
               test_free_footage_is_all_interictal, test_windows_stay_inside_the_file,
               test_lookback_clamps_at_zero, test_real_recording_pat13,
               test_coverage_arithmetic]:
        fn()
    print("\nALL WINDOW TESTS PASSED")
