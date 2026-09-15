"""Exact small-sample tests, checked against hand-computable cases.

These carry the project's conclusions at n = 5 to 8, where a normal approximation once
turned an exact p of 0.0625 into a reported 0.043.
"""
import sys
from math import factorial
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.stats import corr_perm_p, sign_flip_p, zscore_against


def test_sign_flip_is_the_exact_minimum_at_small_n():
    # all six differences the same sign and magnitude: the most extreme of 2^6 = 64
    # assignments is the observed one and its mirror -> 2/64.
    p, n = sign_flip_p([1.0] * 6)
    assert n == 6 and p == pytest.approx(2 / 64)
    # which is why six folds agreeing cannot reach p < 0.03, and eight are needed
    assert sign_flip_p([1.0] * 8)[0] == pytest.approx(2 / 256)


def test_sign_flip_is_two_sided_and_ignores_overall_sign():
    assert sign_flip_p([-1.0] * 6)[0] == sign_flip_p([1.0] * 6)[0]
    assert sign_flip_p([1.0, -1.0, 1.0, -1.0])[0] == 1.0     # mean 0: nothing extreme


def test_sign_flip_drops_non_finite_and_reports_the_surviving_n():
    p, n = sign_flip_p([1.0, np.nan, 1.0, np.inf])
    assert n == 2 and p == pytest.approx(2 / 4)


def test_corr_permutation_enumerates_every_pairing():
    x = [1, 2, 3, 4, 5, 6, 7]
    r, p, n = corr_perm_p(x, x)
    assert r == pytest.approx(1.0) and n == 7
    # only the identity and the reversal reach |r| = 1 for a monotone sequence
    assert p == pytest.approx(2 / factorial(7))
    r2, p2, _ = corr_perm_p(x, x[::-1])
    assert r2 == pytest.approx(-1.0) and p2 == pytest.approx(p), "two-sided"


def test_corr_permutation_refuses_a_sample_it_cannot_enumerate():
    with pytest.raises(ValueError):
        corr_perm_p(list(range(9)), list(range(9)))


def test_zscore_is_undefined_when_the_controls_cannot_resolve_it():
    """D28: identical controls mean the metric did not move, not an infinite effect."""
    assert np.isnan(zscore_against(1.0, [0.0] * 7))
    assert zscore_against(2.0, [0.0, 1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.0)
