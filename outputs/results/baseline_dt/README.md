# D16 resolved: per-fold decision thresholds, and the first honest clinical numbers

DT selected on each fold's two internal validation patients, never on its held-out
patient, by maximising Youden's J on the accumulated score series. The same rule and
the same per-fold value must be applied to the adapted model in Stage 7.

| fold | DT | Youden J | sens | false alarms | exposure (h) | FDR/h | L_EO (s) |
|---|---|---|---|---|---|---|---|
| pat01 | 0.945 | 0.535 | 1 | 20 | 0.671 | 29.8 | +8.0 |
| pat02 | 0.160 | 0.187 | 1 | 31 | 0.633 | 48.9 | +20.0 |
| pat03 | 0.260 | 0.412 | 1 | 9 | 0.186 | 48.4 | +1.5 |
| pat04 | 0.345 | 0.358 | 1 | 30 | 0.762 | 39.3 | +6.0 |
| pat06 | 0.530 | 0.432 | 1 | 6 | 0.097 | 61.7 | +2.0 |
| pat07 | 0.870 | 0.367 | 1 | 0 | 0.132 | 0.0 | +74.5 |
| pat08 | 0.235 | 0.415 | 1 | 25 | 0.558 | 44.8 | +20.0 |
| pat09 | 0.960 | 0.163 | 1 | 8 | 0.106 | 75.8 | +17.3 |
| **pooled** | — | — | **18/18** | **129** | **3.146** | **41.0** | — |

## Why not a false-alarm rate target

The obvious clinical criterion — lowest DT with FDR/h below some budget — is not
estimable here, and it was rejected on the numbers rather than on preference. Evaluable
interictal exposure per patient runs 0.101 h (pat06) to 0.764 h (pat04); 3.23 h for the
whole cohort. A validation *pair* carries 0.8-1.0 h, so one false alarm moves the
measured rate by 1.1-1.3 FDR/h. A 1/h target is therefore indistinguishable from "zero
false alarms anywhere", which would drive DT to the top of its range for reasons of
sample size rather than model quality. Youden's J needs no exposure estimate and is
invariant to how many hours each patient contributes.

## What the thresholds say

They span **0.160 to 0.960**. The inherited DT = 0.3 was defensible for no fold but
pat03 and pat08, and would have been badly wrong for pat01 (0.945), pat09 (0.960) and
pat07 (0.870). Per-fold selection was not a refinement; a single shared threshold made
FDR/h incomparable across folds.

## The honest verdict on the baseline

Sensitivity is 18/18 seizures, and it is not worth much: the cohort model alarms **41 times
per hour**, roughly once every 1.5 minutes. pat07 is the sole exception at
0 false alarms — and pays for it with a 74.5 s detection latency.

This is not a usable clinical detector. It is, however, a properly thresholded,
per-fold-honest baseline, which is what Stage 7 has to beat and what section 3.5's
paired difference is computed against.

Two cautions for that comparison:

1. **The metric is compressed.** With a 60 s refractory the structural ceiling is about
   60 alarms/h; the baseline sits at 41. There is room to improve downward, but
   not much room to get worse, so an improvement in FDR/h will look larger than the
   same improvement would on a less saturated detector.
2. **FDR/h can legitimately exceed 60.** pat09 reads 75.8 because it has three short
   recordings, each carrying its own alarm train, while exposure counts only evaluated
   clip seconds. Checked per source: alarms are exactly 60 s apart and the refractory is
   respected. Not a bug, but it means the rate must always be read beside its count and
   its exposure, as `aggregate` already reports.

## Contamination note

The validation patients select DT, so they are contaminated for threshold-dependent
metrics. Sensitivity, FDR/h and latency are therefore reported for each fold's held-out
patient only (8 patients). The threshold-free within-source AUC continues to use all 24
model x patient measurements.
