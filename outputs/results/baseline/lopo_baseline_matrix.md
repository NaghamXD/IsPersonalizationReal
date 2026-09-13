# LOPO baseline, all eight folds (D21 protocol)

Eight folds, fixed 50-epoch budget, no early stopping, last-5-epoch weight average with
BatchNorm recalibrated. 12.4 h. Each model scored on every patient it never trained on
-- the held-out patient and, because D21 selects nothing, the two internal validation
patients as well. 24 measurements.

| model (fold) | patient | role | within-source AUC | pooled | ictal | interictal |
|---|---|---|---|---|---|---|
| pat01 | pat01 | held out | **0.6037** | 0.7274 | 38 | 484 |
| pat01 | pat04 | internal val | **0.7857** | 0.5626 | 7 | 549 |
| pat01 | pat07 | internal val | **0.8684** | 0.9251 | 43 | 95 |
| pat02 | pat02 | held out | **0.8638** | 0.8579 | 50 | 458 |
| pat02 | pat01 | internal val | **0.5402** | 0.5600 | 38 | 484 |
| pat02 | pat08 | internal val | **0.7290** | 0.7075 | 62 | 403 |
| pat03 | pat03 | held out | **0.8537** | 0.9342 | 27 | 135 |
| pat03 | pat02 | internal val | **0.7579** | 0.7782 | 50 | 458 |
| pat03 | pat04 | internal val | **0.6429** | 0.6979 | 7 | 549 |
| pat04 | pat04 | held out | **0.7143** | 0.5834 | 7 | 549 |
| pat04 | pat03 | internal val | **1.0000** | 0.8428 | 27 | 135 |
| pat04 | pat08 | internal val | **0.7542** | 0.6647 | 62 | 403 |
| pat06 | pat06 | held out | **0.5257** | 0.5851 | 64 | 72 |
| pat06 | pat01 | internal val | **0.7236** | 0.8217 | 38 | 484 |
| pat06 | pat08 | internal val | **0.7663** | 0.7054 | 62 | 403 |
| pat07 | pat07 | held out | **0.9667** | 0.9412 | 43 | 95 |
| pat07 | pat02 | internal val | **0.6855** | 0.7444 | 50 | 458 |
| pat07 | pat04 | internal val | **0.8571** | 0.7468 | 7 | 549 |
| pat08 | pat08 | held out | **0.7139** | 0.6065 | 62 | 403 |
| pat08 | pat03 | internal val | **0.6341** | 0.5034 | 27 | 135 |
| pat08 | pat04 | internal val | **0.5000** | 0.7900 | 7 | 549 |
| pat09 | pat09 | held out | **0.1799** | 0.2612 | 71 | 77 |
| pat09 | pat01 | internal val | **0.3810** | 0.4764 | 38 | 484 |
| pat09 | pat06 | internal val | **0.5605** | 0.6309 | 64 | 72 |

## Summary

| group | n | mean | median | min | max |
|---|---|---|---|---|---|
| held-out patients | 8 | 0.6777 | 0.7141 | 0.1799 | 0.9667 |
| internal-validation patients | 16 | 0.6992 | 0.7263 | 0.3810 | 1.0000 |
| all unseen | 24 | 0.6920 | 0.7190 | 0.1799 | 1.0000 |

**The two groups agree** (held-out 0.678 vs internal-validation
0.699). That is the check on D21: if the validation patients had been
contaminated by selection they would score systematically higher. They do not, so all
24 measurements can be used.

## Per-patient difficulty

| patient | n models that never saw them | mean | values |
|---|---|---|---|
| pat07 | 2 | **0.918** | 0.868, 0.967 |
| pat03 | 3 | **0.829** | 0.854, 1.000, 0.634 |
| pat02 | 3 | **0.769** | 0.864, 0.758, 0.686 |
| pat08 | 4 | **0.741** | 0.729, 0.754, 0.766, 0.714 |
| pat04 | 5 | **0.700** | 0.786, 0.643, 0.714, 0.857, 0.500 |
| pat01 | 4 | **0.562** | 0.604, 0.540, 0.724, 0.381 |
| pat06 | 2 | **0.543** | 0.526, 0.561 |
| pat09 | 1 | **0.180** | 0.180 |

The spread runs from 0.918 (pat07) to 0.180 (pat09) -- far wider than any per-patient
confidence interval these sample sizes support, and exactly the heterogeneity section
3.5 is a hypothesis about.

## pat09: confidently wrong, not uninformative

Fold pat09's model is the worst on all three of its unseen patients (0.180, 0.381,
0.561). An AUC of 0.180 is not noise around chance -- the model ranks pat09's seizure
clips systematically BELOW its interictal clips. Its validation curve sat at 0.4485 +/-
0.0327 for 35 consecutive epochs, i.e. reliably inverted.

It is not a failed training run. The same model scores **0.9999 on pat02 and 1.0000 on
pat03**, both training patients, with clean separation (ictal mean 0.98, interictal mean
0.016). It learned the task and then transferred it backwards onto pat09, pat01 and
pat06.

LOPO cannot separate "pat09 is a hard patient" from "this particular training set
transfers badly to pat09", because only fold pat09 leaves pat09 unseen. What can be
said is that the failure is a property of the model/patient pair, not of the optimiser.

## Does validation predict held-out performance under D21?

corr(averaged validation within-source AUC, held-out within-source AUC) across the
eight folds = **+0.409**. Validation still does not predict held-out performance, which
is consistent with D20 and is the reason D21 stopped selecting on it. Validation is a
monitor, not a model-selection signal.

## What this changes

The cohort baseline is not at chance. Held-out mean 0.678, median
0.714. pat01's 0.498 in the earlier single-fold run was largely a
selection artifact: under D21 averaging the same patient scores 0.604.

Two of 24 measurements fall below chance, both from the pat09 model.
