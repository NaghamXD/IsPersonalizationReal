# More data does not help: 8 vs 14 training patients

Identical protocol, identical eight test patients, identical test clips, identical D21
training rule. The only change is that six training-only patients (D34) join every
fold's training set: +1,223 clips per fold (+49%) and seizures per fold from ~13 to ~28.
17.7 h of training.

| patient | AUC pairs | 8-patient | 14-patient | change | |
|---|---|---|---|---|---|
| pat01 | 7,544 | 0.6037 | 0.7680 | +0.1643 | |
| pat02 | 11,450 | 0.8638 | 0.7024 | -0.1615 | |
| pat03 | 41 | 0.8537 | 0.8537 | +0.0000 | *excluded by D35* |
| pat04 | 14 | 0.7143 | 1.0000 | +0.2857 | *excluded by D35* |
| pat06 | 1,206 | 0.5257 | 0.6675 | +0.1418 | |
| pat07 | 1,710 | 0.9667 | 0.5520 | -0.4146 | |
| pat08 | 9,251 | 0.7139 | 0.5972 | -0.1166 | |
| pat09 | 1,801 | 0.1799 | 0.5669 | +0.3870 | |

## The headline

On the D35 analysis set (n = 6, pat03 and pat04 excluded for resolution):

| | 8-patient | 14-patient |
|---|---|---|
| mean held-out within-source AUC | **0.6423** | **0.6423** |

Change **+0.0001**, exact sign-flip p = 1.000, 3 of 6 folds improved. Across all eight
folds the mean moves 0.6777 -> 0.7135 (+0.0358, p = 0.688, 4/8 improved), and that
difference is carried by pat04 — a 14-pair fold whose AUC moves in steps of 0.071.

**Adding 75% more training clips and 15 more seizures changed nothing on average.**

## What did move: everything, individually

| improved | worsened |
|---|---|
| pat09 **+0.387** (0.180 -> 0.567) | pat07 **−0.415** (0.967 -> 0.552) |
| pat01 +0.164 | pat02 −0.162 |
| pat06 +0.142 | pat08 −0.117 |

Per-fold swings of up to ±0.41, in both directions, cancelling almost exactly. **At this
cohort size, held-out performance is dominated by which patients happen to land in the
training set, not by how many.** Those swings are roughly 800x the personalisation
effect Phase 1 was trying to measure (~0.0005).

## Two consequences

**The "it just needed more data" objection is closed.** The most obvious explanation for
Phase 1's negative result — an 8-patient cohort being too small — predicts that a
14-patient training set would move the baseline. It did not.

**pat09's feature collapse was a property of the training set, not the patient.** Its
anti-correlated 0.180 (D22) becomes 0.567 here. D35's rule 1 excludes pat09 from §3.5's
primary analysis on the grounds of feature collapse; that rationale holds for the Phase 1
results it was written about, and must be restated rather than carried over silently if
§3.5 is recomputed on these backbones.

The internal-validation patients (16 measurements) move 0.6992 -> 0.7165, consistent
with the held-out picture: no systematic gain.
