"""Tests for Pool A / Pool B construction and LOPO fold structure.

Pure logic, no torch, no clips on disk.

Run:  python tests/test_pools.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data.pools import (build_pools, evenly_spaced, largest_remainder,
                            overlaps, verify_guardrail)
from src.data.splits import make_all_folds, make_fold, verify_no_leak
from src.utils.naming import parse_clip_name


# ---------------------------------------------------------------- allocation
def test_largest_remainder():
    assert sum(largest_remainder([10, 10, 10], 20)) == 20
    assert largest_remainder([100, 1, 1], 20) == [19, 1, 0] or \
           sum(largest_remainder([100, 1, 1], 20)) == 20
    # Never allocate more than a stratum holds.
    a = largest_remainder([1, 1, 40], 20)
    assert all(x <= c for x, c in zip(a, [1, 1, 40])), a
    assert sum(a) == 20, a
    # Wanting everything (or more) returns everything.
    assert largest_remainder([3, 4], 7) == [3, 4]
    assert largest_remainder([3, 4], 99) == [3, 4]
    assert largest_remainder([0, 0], 5) == [0, 0]
    print("  OK  largest_remainder")


def test_evenly_spaced_is_not_a_prefix():
    items = list(range(100))
    got = evenly_spaced(items, 5)
    assert got[0] == 0 and got[-1] == 99, got
    assert got != items[:5], "must spread across the range, not take the first k"
    # Roughly uniform gaps.
    gaps = [b - a for a, b in zip(got, got[1:])]
    assert max(gaps) - min(gaps) <= 1, gaps
    assert evenly_spaced(items, 0) == []
    assert evenly_spaced([1, 2], 5) == [1, 2]
    print("  OK  evenly_spaced spreads across the range")


# ---------------------------------------------------------------- pools
def _clips(patient, spec):
    """spec: {source: [(t, label), ...]} -> [(name, label)]"""
    out = []
    for src, entries in spec.items():
        for t, lab in entries:
            out.append((f"{patient}_{src}_{t}", lab))
    return out


def test_pool_a_is_chronological_not_lexicographic():
    """Regression: the base repo sorted filenames as strings, so '_1000' preceded
    '_200' and Pool A was an artifact of string comparison rather than of time."""
    spec = {"free": [(t, 0.0) for t in [0, 5, 200, 1000, 1005]]}
    pools = build_pools("pat03", _clips("pat03", spec), pool_a_size=5)
    times = [parse_clip_name(n)[2] for n in pools.pool_a]
    assert times == sorted(times), times
    assert times == [0, 5, 200, 1000, 1005], times
    # String order would have put 1000 and 1005 before 200.
    assert sorted([str(int(t)) for t in times]) != [str(int(t)) for t in times]
    print("  OK  Pool A ordered by time, not filename string")


def test_pool_a_draws_from_every_source_in_proportion():
    """pat03's real shape: one clip from each seizure file and 40 from free.mp4."""
    spec = {"Sz1": [(0, 0.0)], "Sz2": [(0, 0.0)],
            "free": [(5 * i, 0.0) for i in range(40)]}
    pools = build_pools("pat03", _clips("pat03", spec), pool_a_size=20)
    assert len(pools.pool_a) == 20, len(pools.pool_a)
    by_src = Counter(parse_clip_name(n)[1] for n in pools.pool_a)
    assert by_src["free"] >= 18, by_src
    assert sum(by_src.values()) == 20
    # The free.mp4 picks must span the file, not cluster at its start.
    free_t = sorted(parse_clip_name(n)[2] for n in pools.pool_a
                    if parse_clip_name(n)[1] == "free")
    assert free_t[-1] > 150, f"Pool A clustered at the start of free.mp4: {free_t}"
    print(f"  OK  Pool A stratified across sources {dict(by_src)}")


def test_pools_are_disjoint_and_guardrailed():
    spec = {"Sz1": [(5 * i, 0.0) for i in range(30)] +
                   [(200 + i, 1.0) for i in range(20)]}
    pools = build_pools("pat06", _clips("pat06", spec), pool_a_size=20)
    assert len(pools.pool_a) == 20
    assert set(pools.pool_a).isdisjoint(pools.pool_b_interictal)
    assert set(pools.pool_a).isdisjoint(pools.pool_b_ictal)
    assert len(pools.pool_b_ictal) == 20
    assert verify_guardrail(pools) == [], verify_guardrail(pools)
    print(f"  OK  pools disjoint, guardrail clean "
          f"(A={len(pools.pool_a)} B_inter={len(pools.pool_b_interictal)} "
          f"B_ictal={len(pools.pool_b_ictal)})")


def test_guardrail_catches_an_injected_overlap():
    spec = {"Sz1": [(5 * i, 0.0) for i in range(30)]}
    pools = build_pools("pat06", _clips("pat06", spec), pool_a_size=10)
    a0 = pools.pool_a[0]
    _, src, t = parse_clip_name(a0)
    pools.pool_b_interictal.append(f"pat06_{src}_{int(t + 1)}")   # 1 s away: overlaps
    bad = verify_guardrail(pools)
    assert bad, "an overlapping Pool B clip must be caught"
    print(f"  OK  guardrail catches an injected overlap ({bad[0][0]})")


def test_boundary_clip_cannot_enter_pool_a():
    """Regression: a window ending exactly AT EEG onset has ramp label 0.0.

    The transition ramp is evaluated at the clip end, so x = 0 there and
    (e^0 - 1)/(e^5 - 1) == 0.0 exactly. By label such a clip looks interictal; by
    phase it is the first transition window, and it overlaps the ictal clips
    extracted at 1 s stride around it. Caught in the real pat06 data as
    pat06_Sz1_265 (Pool A) against pat06_Sz1_269 (ictal).
    """
    spec = {"Sz1": [(5 * i, 0.0) for i in range(54)]                    # 0..265
                   + [(269 + i, 1.0) for i in range(20)]}               # dense ictal
    pools = build_pools("pat06", _clips("pat06", spec), pool_a_size=20)
    assert verify_guardrail(pools) == [], verify_guardrail(pools)
    assert "pat06_Sz1_265" not in pools.pool_a, "boundary clip must not seed the signature"
    assert pools.n_boundary_excluded >= 1, pools.n_boundary_excluded
    # It is still a legitimate training negative, just not a signature clip.
    assert "pat06_Sz1_265" in pools.pool_b_interictal
    print(f"  OK  boundary clip barred from Pool A "
          f"({pools.n_boundary_excluded} excluded), kept as a Pool B negative")


def test_transition_clips_are_excluded():
    spec = {"Sz1": [(0, 0.0), (5, 0.0), (10, 0.37), (11, 0.62), (20, 1.0)]}
    pools = build_pools("pat07", _clips("pat07", spec), pool_a_size=2)
    everything = set(pools.pool_a) | set(pools.pool_b_interictal) | set(pools.pool_b_ictal)
    assert pools.n_transition_excluded == 2
    assert not any(parse_clip_name(n)[2] in (10, 11) for n in everything)
    print("  OK  transition clips excluded from both pools")


def test_overlaps_helper():
    assert overlaps("p_Sz1_100", "p_Sz1_102")
    assert not overlaps("p_Sz1_100", "p_Sz1_105")     # adjacent, not overlapping
    assert not overlaps("p_Sz1_100", "p_Sz2_100")     # different recording
    assert not overlaps("p_Sz1_100", "q_Sz1_100")     # different patient
    print("  OK  overlap helper")


# ---------------------------------------------------------------- folds
def test_folds_have_no_leak():
    folds = make_all_folds()
    assert len(folds) == len(config.COHORT) == 8
    assert verify_no_leak(folds) == []
    for f in folds:
        assert len(f.train_patients) == 5, f
        assert len(f.val_patients) == 2, f
        assert set(f.train_patients) | set(f.val_patients) | {f.test_patient} \
            == set(config.COHORT)
    print("  OK  8 folds, 1 test / 2 val / 5 train, no leak")


def test_validation_pair_is_semiology_stratified():
    for f in make_all_folds():
        sem = sorted(config.SEMIOLOGY[p] for p in f.val_patients)
        assert sem == ["P", "PG"], f"{f.test_patient}: val pair is {sem}"
    print("  OK  every fold validates on one focal and one generalised patient")


def test_folds_are_deterministic_and_rotate():
    a = make_all_folds()
    b = make_all_folds()
    assert [f.val_patients for f in a] == [f.val_patients for f in b], "must be seeded"
    used = Counter(p for f in a for p in f.val_patients)
    assert len(used) >= 4, f"validation duty should rotate, got {used}"
    print(f"  OK  deterministic, and validation duty rotates across {len(used)} patients")


def test_stratification_fails_loudly_if_impossible():
    try:
        make_fold("pat01", cohort=["pat01", "pat02", "pat03"],
                  semiology={"pat01": "P", "pat02": "PG", "pat03": "PG"})
    except ValueError as e:
        assert "stratified" in str(e)
        print("  OK  raises when no focal patient remains, rather than silently skewing")
        return
    raise AssertionError("expected a ValueError")


if __name__ == "__main__":
    for fn in [test_largest_remainder, test_evenly_spaced_is_not_a_prefix,
               test_pool_a_is_chronological_not_lexicographic,
               test_pool_a_draws_from_every_source_in_proportion,
               test_pools_are_disjoint_and_guardrailed,
               test_guardrail_catches_an_injected_overlap,
               test_boundary_clip_cannot_enter_pool_a,
               test_transition_clips_are_excluded, test_overlaps_helper,
               test_folds_have_no_leak, test_validation_pair_is_semiology_stratified,
               test_folds_are_deterministic_and_rotate,
               test_stratification_fails_loudly_if_impossible]:
        fn()
    print("\nALL POOL/SPLIT TESTS PASSED")
