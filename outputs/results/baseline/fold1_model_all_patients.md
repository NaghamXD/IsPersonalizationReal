# One model, every patient — is pat01 representative?

The fold-1 checkpoint (trained on pat02/03/06/08/09, epoch 29, selected on pat04+pat07)
scored against all eight patients' non-overlapping sliding-window test clips.

| patient | role in fold 1 | within-source AUC | pooled | clips | ictal | frac >0.9 |
|---|---|---|---|---|---|---|
| pat01 | HELD OUT | **0.4980** | 0.5909 | 527 | 38 | 57% |
| pat04 | validation (selection) | **1.0000** | 0.6853 | 559 | 7 | 89% |
| pat07 | validation (selection) | **0.9661** | 0.9684 | 156 | 43 | 58% |
| pat02 | TRAINING | **0.9997** | 0.9999 | 523 | 50 | 10% |
| pat03 | TRAINING | **1.0000** | 0.9995 | 163 | 27 | 15% |
| pat06 | TRAINING | **0.9967** | 0.9989 | 137 | 64 | 45% |
| pat08 | TRAINING | **0.9997** | 0.9997 | 473 | 62 | 12% |
| pat09 | TRAINING | **1.0000** | 1.0000 | 151 | 71 | 47% |

**pat01 is the only patient at chance.** Every other patient, including the two the
model never trained on, sits near the ceiling.

Read this with three caveats.

1. **pat04 and pat07 are selection-contaminated.** Epoch 29 is precisely the epoch that
   maximised their combined validation AUC over 59 draws, so 1.000 and 0.966 are the
   argmax, not an estimate. The non-selected epoch-33 checkpoint of the earlier run put
   them at 0.762 and 0.860/0.890 — a fairer indication of their level, and still far
   above pat01.
2. **The five training patients at ~1.000 measure memorisation**, not generalisation.
   They establish the ceiling and nothing else.
3. **pat04's figure rests on 7 ictal clips** in one recording, the only source of its
   that carries both classes. It is the noisiest number in the table.

Even discounted, the spread across unseen patients is large: pat01 ~0.50, pat04
0.76-1.00, pat07 0.86-0.97. **pat01 looks like an outlier among unseen patients, not a
representative one.** Nothing about the cohort baseline should be concluded from fold 1
alone.

This also means the per-patient heterogeneity that section 3.5 predicts is visible in
the very first fold — which is encouraging for the hypothesis and a reason to measure
more patients before drawing the curve.
