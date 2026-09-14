# Stage 7 across all eight folds: no personalisation, by the pre-registered test

Hypernetwork trained per fold exactly as §3.4 specifies. Each held-out patient scored
under the patient's own z_behavior and under all seven other patients' — 64 evaluations.

| fold | baseline | own z | shuffled-z mean | shuffled sd | z-score | AUC pairs |
|---|---|---|---|---|---|---|
| pat01 | 0.6037 | 0.6166 | 0.6165 | 0.0015 | +0.05 | 7544 |
| pat02 | 0.8638 | 0.8623 | 0.8624 | 0.0004 | -0.44 | 11450 |
| pat03 | 0.8537 | 0.8537 | 0.8537 | 0.0000 | n/a | 41 |
| pat04 | 0.7143 | 0.8571 | 0.8571 | 0.0000 | +0.93 | 14 |
| pat06 | 0.5257 | 0.4905 | 0.4919 | 0.0052 | -0.28 | 1206 |
| pat07 | 0.9667 | 0.9474 | 0.9475 | 0.0003 | -0.59 | 1710 |
| pat08 | 0.7139 | 0.7186 | 0.7189 | 0.0009 | -0.32 | 9251 |
| pat09 | 0.1799 | 0.1882 | 0.1891 | 0.0005 | -1.61 | 1801 |

**0 of 8 folds meet the D28 criterion (z > 2.0).** Six of eight z-scores are negative:
the patient's own signature tends to be slightly *worse* than a stranger's.

## Two folds cannot resolve the question

Within-source AUC counts only recordings holding both classes. For pat04 that is a
single recording with **7 ictal and 2 interictal clips — 14 pairs**, so its AUC moves in
steps of 0.0714. For pat03 it is 41 pairs, steps of 0.0244. The effect under test is
~0.0005. These two folds are not evidence either way, and pat04's headline "+0.1429 from
adaptation" is **two pairs changing order**.

That matters beyond this table: D22's baseline mean of 0.678 across eight held-out
patients also includes these two.

## The verdict, with and without them

Exact paired sign-flip permutation tests (2^n sign assignments; n is too small for a
normal approximation, and the earlier t-based p of 0.043 was an artifact of using one).

| quantity | all 8 folds | 6 resolvable folds |
|---|---|---|
| own − shuffled (personalisation) | −0.00036, p = 0.0625, 2/8 positive | −0.00048, p = 0.0625, 1/6 positive |
| own − baseline (any adaptation) | +0.01409, p = 0.719, 4/8 positive | **−0.00503**, p = 0.594, 3/6 positive |

Two things follow.

**Personalisation is absent.** Not merely non-significant — the point estimate is
negative in both analyses, and 0/8 folds clear a threshold set in advance.

**The apparent benefit of adapting at all was an artifact.** The +0.0141 over baseline
across eight folds is carried entirely by pat04's two-pair flip. Among the six folds
whose metric can resolve anything, adaptation is **−0.005** — slightly harmful.

## §3.5

| correlation | all 8 | 6 resolvable |
|---|---|---|
| corr(D_p, own − baseline) — what §3.5 defines | +0.049 | −0.495 |
| corr(D_p, own − shuffled) — personalisation only | −0.566 | −0.706 |

§3.5 predicts a positive correlation. On the resolvable folds every version is negative.
At n = 6–8 none of this is significant; the honest statement is that the predicted
relationship is not present, and the data lean the other way.

## What the result is

The method as published does not personalise on this cohort. The diagnosis is recorded
and is more useful than the bare negative: the hypernetwork is trained on the same
patients the frozen backbone was fitted on, where its loss is already ~0.007 against
0.726 on unseen patients (D26), so it optimises where there is nothing to gain; and with
five training patients it has five examples of the z → ΔW mapping it is meant to
amortise. The signature itself is not the problem — z identifies its patient at 55.5%
against 12.5% chance (D23) — but it is least recoverable exactly where the model most
needs help (corr +0.726 with baseline AUC).
