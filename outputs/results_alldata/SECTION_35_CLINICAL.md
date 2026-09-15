# Section 3.5 on the clinical metric, under D37

Benefit is the FDR/h REDUCTION against the fold's own unadapted baseline at the
D16 threshold, which is selected on the internal validation patients and applied
unchanged to all nine conditions. Positive means fewer false detections per hour.

## 8-patient cohort

| fold | DT | baseline FA / exposure | baseline FDR/h | own FDR/h | reduction | z | sens base -> own | |
|---|---|---|---|---|---|---|---|---|
| pat01 | 0.945 | 20 / 0.671 h | 29.81 | 29.81 | +0.00 | n/a | 1.00 -> 1.00 |  |
| pat02 | 0.160 | 31 / 0.633 h | 48.95 | 48.95 | +0.00 | +2.27 | 1.00 -> 1.00 |  |
| pat03 | 0.260 | 9 / 0.186 h | 48.36 | 48.36 | +0.00 | n/a | 1.00 -> 1.00 |  |
| pat04 | 0.345 | 30 / 0.762 h | 39.34 | 39.34 | +0.00 | -0.12 | 1.00 -> 1.00 |  |
| pat06 | 0.530 | 6 / 0.097 h | 61.71 | 41.14 | +20.57 | +0.00 | 1.00 -> 1.00 |  |
| pat07 | 0.870 | 0 / 0.132 h | 0.00 | 7.58 | -7.58 | n/a | 1.00 -> 1.00 | **no baseline false alarms** -- cannot show a reduction |
| pat08 | 0.235 | 25 / 0.558 h | 44.78 | 35.82 | +8.96 | +0.38 | 1.00 -> 0.67 | **sensitivity fell** -- not a benefit |
| pat09 | 0.960 | 8 / 0.106 h | 75.79 | 66.32 | +9.47 | n/a | 1.00 -> 1.00 |  |

**Test 1 (personalisation, D28's z > 2.0): 1 of 4 folds pass.**
z ranges -0.12 to +2.27.

Paired own-minus-shuffled reduction over all 8 folds: +0.178 FDR/h, exact sign-flip p = 0.5000.

**Test 2 (§3.5's correlation).**

| analysis set | r | exact p (n! pairings) | n |
|---|---|---|---|
| n = 7, D35 rule 1 (primary) | +0.650 | 0.1095 | 7 |
| all folds (secondary) | +0.605 | 0.1125 | 8 |

§3.5 predicts POSITIVE.

**Post hoc, not pre-registered: the same test without the exposure division.**

| quantity (n = 7) | r | exact p |
|---|---|---|
| D_p vs FDR/h reduction (pre-registered) | +0.650 | 0.1095 |
| D_p vs **alarm-count** reduction | -0.037 | 0.9048 |
| D_p vs 1/exposure | +0.455 | 0.3315 |
| 1/exposure vs \|FDR/h reduction\| | +0.766 | |

| fold | baseline FA | own FA | shuffled FA | own-z benefit, in alarms |
|---|---|---|---|---|
| pat01 | 20 | 20 | [20, 20, 20, 20, 20, 20, 20] | +0 |
| pat02 | 31 | 31 | [31, 32, 32, 32, 32, 32, 32] | +0 |
| pat03 | 9 | 9 | [9, 9, 9, 9, 9, 9, 9] | +0 |
| pat04 | 30 | 30 | [28, 29, 29, 30, 31, 31, 31] | +0 |
| pat06 | 6 | 4 | [3, 4, 4, 4, 4, 4, 5] | +2 |
| pat07 | 0 | 1 | [1, 1, 1, 1, 1, 1, 1] | -1 |
| pat08 | 25 | 20 | [20, 20, 20, 20, 20, 20, 21] | +5 |
| pat09 | 8 | 7 | [7, 7, 7, 7, 7, 7, 7] | +1 |

## 14-patient cohort

| fold | DT | baseline FA / exposure | baseline FDR/h | own FDR/h | reduction | z | sens base -> own | |
|---|---|---|---|---|---|---|---|---|
| pat01 | 0.490 | 26 / 0.671 h | 38.76 | 38.76 | +0.00 | +0.59 | 1.00 -> 1.00 |  |
| pat02 | 0.690 | 32 / 0.633 h | 50.53 | 47.37 | +3.16 | +1.35 | 1.00 -> 1.00 |  |
| pat03 | 0.930 | 0 / 0.186 h | 0.00 | 10.75 | -10.75 | +1.24 | 1.00 -> 1.00 | **no baseline false alarms** -- cannot show a reduction |
| pat04 | 0.955 | 33 / 0.762 h | 43.28 | 36.72 | +6.56 | n/a | 1.00 -> 1.00 |  |
| pat06 | 0.980 | 6 / 0.097 h | 61.71 | 51.43 | +10.29 | n/a | 1.00 -> 1.00 |  |
| pat07 | 0.590 | 5 / 0.132 h | 37.89 | 30.32 | +7.58 | n/a | 1.00 -> 1.00 |  |
| pat08 | 0.730 | 28 / 0.558 h | 50.15 | 51.94 | -1.79 | n/a | 1.00 -> 1.00 |  |
| pat09 | 0.990 | 3 / 0.106 h | 28.42 | 47.37 | -18.95 | n/a | 1.00 -> 1.00 |  |

**Test 1 (personalisation, D28's z > 2.0): 0 of 3 folds pass.**
z ranges +0.59 to +1.35.

Paired own-minus-shuffled reduction over all 8 folds: +1.193 FDR/h, exact sign-flip p = 0.2500.

**Test 2 (§3.5's correlation).**

| analysis set | r | exact p (n! pairings) | n |
|---|---|---|---|
| n = 7, D35 rule 1 (primary) | +0.617 | 0.1377 | 7 |
| all folds (secondary) | +0.597 | 0.1197 | 8 |

§3.5 predicts POSITIVE.

**Post hoc, not pre-registered: the same test without the exposure division.**

| quantity (n = 7) | r | exact p |
|---|---|---|
| D_p vs FDR/h reduction (pre-registered) | +0.617 | 0.1377 |
| D_p vs **alarm-count** reduction | +0.433 | 0.3437 |
| D_p vs 1/exposure | +0.714 | 0.0790 |
| 1/exposure vs \|FDR/h reduction\| | +0.764 | |

| fold | baseline FA | own FA | shuffled FA | own-z benefit, in alarms |
|---|---|---|---|---|
| pat01 | 26 | 26 | [26, 26, 26, 26, 26, 27, 27] | +0 |
| pat02 | 32 | 30 | [30, 30, 33, 33, 34, 35, 35] | +2 |
| pat03 | 0 | 2 | [2, 2, 3, 3, 3, 3, 4] | -2 |
| pat04 | 33 | 28 | [28, 28, 28, 28, 28, 28, 28] | +5 |
| pat06 | 6 | 5 | [5, 5, 5, 5, 5, 5, 5] | +1 |
| pat07 | 5 | 4 | [4, 4, 4, 4, 4, 4, 4] | +1 |
| pat08 | 28 | 29 | [29, 29, 29, 29, 29, 29, 29] | -1 |
| pat09 | 3 | 5 | [5, 5, 5, 5, 5, 5, 5] | -2 |

