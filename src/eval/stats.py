"""Small-sample tests, exact rather than approximate.

n here is 5 to 8. A normal approximation at that size is not a shortcut, it is wrong:
the paired t on six folds once produced p = 0.043 for a difference whose exact
sign-flip p is 0.0625, and that number reached a draft before it was caught.
"""
from itertools import product

import numpy as np


def sign_flip_p(d):
    """Exact two-sided paired permutation test over all 2^n sign assignments.

    Returns (p, n) with non-finite entries dropped. Under the null the pairing carries
    no information, so every assignment of signs to the observed magnitudes is equally
    likely; p is the fraction whose mean is at least as extreme as the observed one.
    """
    d = np.asarray([x for x in np.asarray(d, dtype=float).ravel() if np.isfinite(x)],
                   dtype=float)
    n = len(d)
    if n == 0:
        return float("nan"), 0
    obs = abs(d.mean())
    hit = sum(1 for s in product([1, -1], repeat=n)
              if abs(float(np.dot(s, d)) / n) >= obs - 1e-15)
    return hit / 2 ** n, n


def zscore_against(own, others):
    """D28's statistic: how far the own-z result sits from the shuffled-z spread.

    Undefined (nan) when the controls are identical, which at this sample size means
    the metric could not resolve the difference rather than that the difference is
    infinite. Callers must report that as 'n/a', never as a pass or a failure.
    """
    o = np.asarray([x for x in np.asarray(others, dtype=float).ravel()
                    if np.isfinite(x)], dtype=float)
    if len(o) < 2 or not np.isfinite(own):
        return float("nan")
    sd = float(o.std(ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return float((own - o.mean()) / sd)


def corr_perm_p(x, y):
    """Exact permutation test for a correlation: all n! pairings, n <= 8.

    A sign-flip test is the exact test for a PAIRED DIFFERENCE and the wrong scheme for
    a correlation, whose null is that the pairing between x and y carries no
    information. With n = 7 that null has 5,040 realisations and n = 8 has 40,320, so
    both are enumerated rather than sampled. Returns (r, p, n).
    """
    from itertools import permutations

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan"), n
    if n > 8:
        raise ValueError(f"n={n} would enumerate {n}! pairings; use a sampled test")
    r = float(np.corrcoef(x, y)[0, 1])
    xc, yc = x - x.mean(), y - y.mean()
    obs = abs(float(xc @ yc))
    hit = sum(1 for p in permutations(range(n))
              if abs(float(xc @ yc[list(p)])) >= obs - 1e-12)
    from math import factorial
    return r, hit / factorial(n), n
