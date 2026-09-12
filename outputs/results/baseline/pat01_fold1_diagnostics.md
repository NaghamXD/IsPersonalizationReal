# Fold 1 (held-out pat01) — first honest evaluation

Checkpoint: `outputs/lopo/checkpoints/pat01/best_model.pth` (epoch 33, val AUC 0.9111)
Eval: `processed_data/test_sliding/manifest_pat01.json`, 527 clips, device=cpu
Preprocessing configs for `processed_data/` and `processed_data/test_sliding/`
were diffed key-by-key: **zero differences** (same git commit 4243419, same
CLIP_FRAMES/GAUSSIAN_SIGMA/KPT_NORMALISATION/JOINT_INDICES). No pipeline bug.

## Clinical numbers (DT = 0.3, mean accumulation, 60 s refractory)
| | |
|---|---|
| sensitivity | 1.0 (2/2 seizures) |
| false alarms | 24 over 0.671 h evaluable |
| FDR/h | 35.8 |
| latency vs EEG onset | +0.5 s |
| latency vs clinical onset | -13.0 s |

Sensitivity 1.0 is not meaningful: 62.4% of ALL clips score > 0.9.

## Discrimination — pooled AUC is an artifact
| scope | pooled AUC | within-source AUC |
|---|---|---|
| pat01 (held out) | 0.625 | **0.511 (Sz1), 0.547 (Sz2)** |
| pat04 (internal val) | 0.836 | 0.762 |
| pat07 (internal val) | 0.942 | 0.860, 0.890 |
| pat03 (in training set) | — | 1.000, 1.000 |

Pooling clips across recordings inflates AUC by +0.07 to +0.10 everywhere
(Simpson's paradox: recordings differ in baseline score level, and the
ictal/interictal mix differs per recording). **Within-source AUC must be the
reported and selected metric**, or a baseline-vs-adapted comparison will
attribute between-recording offsets to personalisation.

## Reproducibility of the validation figure
Re-scoring `folds/pat01/val_clips.json` through the evaluation code path gives
AUC 0.9008 (best_model) / 0.8317 (last_checkpoint), reproducing training's
0.9111 / ~0.83. The validation signal is real; it just does not transfer to
pat01.

## Calibration collapse
Ranking survives where the margin does not: pat07_Sz1 has AUC 0.860 with a mean
ictal-minus-interictal separation of **+0.002**; pat04_Sz1 has AUC 0.762 at
+0.010. Scores saturate near 1. A fixed DECISION_THRESHOLD of 0.3 cannot sit
anywhere sensible on such a distribution — this makes D16 (per-fold DT chosen on
internal validation patients) a prerequisite for any FDR/h number, not a
refinement.

## Not a checkpoint-selection artifact
best_model (epoch 33) and last_checkpoint (epoch 63) agree on pat01:
pooled AUC 0.625 vs 0.631. The gap is patient-level, not epoch-level.


---

## Appendix — fold-1 training curve, transcribed

The per-epoch `training_log.json` for this run was destroyed on 2026-09-12 by a
`--restart` smoke test before it had been committed (`outputs/**/*.json` was
gitignored at the time). The figures below are transcribed from the analysis run
against it while it existed; the raw curve is not recoverable without repeating the
1.98 h run. Both causes are now fixed: `--restart` archives rather than overwrites,
and run logs are tracked.

63 epochs, 1.98 h, mean 113 s/epoch. Early stop on patience 30; best at epoch 33.

| epochs | mean AUC (pooled) | max | min | sd |
|---|---|---|---|---|
| 1-10 | 0.570 | 0.782 | 0.373 | 0.135 |
| 11-20 | 0.732 | 0.836 | 0.562 | 0.090 |
| 21-30 | 0.774 | 0.863 | 0.633 | 0.065 |
| 31-40 | 0.828 | 0.911 | 0.687 | 0.056 |
| 41-50 | 0.841 | 0.876 | 0.797 | 0.027 |
| 51-63 | 0.846 | 0.892 | 0.783 | 0.035 |

- best AUC 0.9111 @ epoch 33; its MSE 0.29027
- best MSE 0.18534 @ epoch 10; its AUC 0.7817
- constant-predictor MSE on this validation set: 0.23066 — the best-AUC checkpoint
  scores 1.26x that, i.e. worse than predicting the mean, while ranking well
- corr(AUC, MSE) across all epochs: -0.166
- post-plateau (epochs 21-63) pooled AUC: 0.824 +/- 0.055; epoch-to-epoch |dAUC|
  after epoch 20: 0.051 mean, 0.192 max
- train huber 0.11218 (ep1) -> 0.00119 (min, ep57) while validation AUC stayed ~0.82

Every AUC in this appendix is the POOLED figure, which D19 retired. The within-source
equivalents were never computed for this run and now cannot be.
