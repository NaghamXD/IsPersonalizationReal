# Two cohorts, one answer: the hypernetwork does not personalise

The all-data repeat (D35 rule 3) exists to close one loophole: that Phase 1's
negative result came from amortising over five patients. The backbones were
retrained on 11 training patients per fold instead of 5 (D34), signatures rebuilt
on them, and Stage 7 repeated end to end. Nothing else changed.

## Per fold, both experiments

| fold | pairs | baseline (8p) | z (8p) | baseline (14p) | z (14p) | D35 |
|---|---|---|---|---|---|---|
| pat01 | 7,544 | 0.6037 | +0.05 | 0.7680 | -0.40 |  |
| pat02 | 11,450 | 0.8638 | -0.44 | 0.7024 | -2.01 |  |
| pat03 | 41 | 0.8537 | n/a | 0.8537 | +0.75 | **excluded** (<50 pairs) |
| pat04 | 14 | 0.7143 | +0.93 | 1.0000 | n/a | **excluded** (<50 pairs) |
| pat06 | 1,206 | 0.5257 | -0.28 | 0.6675 | +0.35 |  |
| pat07 | 1,710 | 0.9667 | -0.59 | 0.5520 | -1.48 |  |
| pat08 | 9,251 | 0.7139 | -0.32 | 0.5972 | -0.92 |  |
| pat09 | 1,801 | 0.1799 | -1.61 | 0.5666 | +0.52 |  |

**0 of 14 defined z-scores pass the pre-registered D28
threshold (z > 2.0), across both experiments.** The range is -2.01 to
+0.93. The only |z| > 2 anywhere is pat02 at -2.01 in the 14-patient run -- the patient's own signature significantly *worse* than a stranger's.

A z-score is undefined where every signature gives the identical AUC, which
happens only in the two folds D35 rule 2 excludes: their metric cannot move by
less than 0.024.

## The D35 analysis set (n = 6)

| quantity | 8-patient | 14-patient |
|---|---|---|
| own - shuffled (personalisation) | -0.00048 | -0.00068 |
| exact sign-flip p | 0.0625 | 0.2500 |
| folds positive | 1/6 | 2/6 |
| own - baseline (any adaptation) | -0.00503 | -0.00177 |
| folds passing D28 | 0/6 | 0/6 |

Tripling the training set moved the point estimate slightly further from zero in
the NEGATIVE direction, and made the fold-to-fold sign pattern less consistent,
not more (exact p 0.0625 -> 0.2500).
The loophole is closed: more patients did not produce personalisation, and there
is no trend toward it.

## The same folds, very different backbones

The retrain did not simply make the baseline better. Across the D35 folds the
mean baseline moved 0.642285 -> 0.642298 -- a coincidence at this precision, not an identity --
while individual folds moved by up to 0.415 AUC (pat07: 0.967 -> 0.552), in both directions. Adding six
training patients reshuffles which held-out patients a backbone happens to suit,
without changing how well it does on average.

That is the scale of the nuisance variation any personalisation effect has to be
seen against. The effect under test is ~0.00068.

## Section 3.5 correlations flip sign between the two cohorts

| correlation | 8-patient | 14-patient |
|---|---|---|
| corr(D_p, own - baseline) -- what 3.5 defines | +0.049 | -0.134 |
| corr(D_p, own - shuffled) -- personalisation only | -0.566 | +0.048 |

On the D35 set (n = 6) the same pair is -0.495 -> -0.155 and -0.706 -> +0.185.

Both correlations reverse sign under a change that should not reverse a real
effect. This is the cleanest available evidence that 3.5's correlations are
noise at this n, and it is stronger than any single-cohort non-significance:
an unstable sign is not a small effect measured imprecisely, it is no effect.

These are AUC-based, and therefore **secondary** under D35: 3.5's primary
benefit is the FDR/h reduction at n = 7, which is still outstanding.

## pat09, restated rather than carried over

D35 rule 1 excludes pat09 from 3.5's primary analysis because its Phase 1
model suffered feature collapse -- held-out within-source AUC 0.180,
anti-correlated with the truth. **That collapse does not survive the all-data
run**: the same fold scores 0.567 on the 14-patient backbone. The
rule's rationale is specific to Phase 1 and must be stated that way, not
silently inherited. For the 14-patient cohort pat09 is an ordinary fold, and the
n = 5 subset is reported only for continuity (-0.00097, p = 0.1875).

## What this establishes

Phase 1 showed the method does not personalise on a 5-patient training set. The
repeat shows the same on 11, with the same architecture, the same signatures and
the same pre-registered test. Two of the five diagnosed causes (D26 gradient
starvation, amortisation over too few patients) were addressed directly by the
retrain; the result did not move. The remaining explanation is the one the
signature probes already point at: z carries identity (55.5%, D23) but is least
recoverable exactly where the model most needs help (corr +0.726 with baseline
AUC), so there is no usable signal to condition on where conditioning would pay.

