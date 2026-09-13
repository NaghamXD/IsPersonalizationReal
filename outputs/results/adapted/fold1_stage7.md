# Stage 7, fold pat01: the adapted arm, measured against its own controls

Hypernetwork trained exactly as §3.4 specifies — cyclic sampler, patient-homogeneous
50/50 batches, frozen backbone and BatchNorm, AdamW with 300-step warmup to 1e-3 then
cosine, BCEWithLogitsLoss. Early-stopped at epoch 7 on patience 5.

## It is worse than doing nothing

Identical 36 validation batches, three conditions:

| condition | validation BCE |
|---|---|
| baseline — dW = 0, no adaptation | **0.726** |
| adapted — best hypernetwork checkpoint | 1.017 (**+40%**) |
| shuffled-z — another patient's signature | 1.091 |

Every epoch of training scored worse than the untouched baseline. The "best" checkpoint
is the least-bad of seven, not an improvement.

## On the held-out patient, the gain is real but not personalisation

| condition | within-source AUC | change vs baseline |
|---|---|---|
| baseline | 0.6037 | — |
| adapted, pat01's own z | 0.6166 | **+0.0129** |
| shuffled-z, pat06's z | 0.6159 | **+0.0122** |

**The correct patient's signature is worth +0.0007.** Essentially the whole gain is
reproduced by injecting a different patient's z. Perturbing the weights at all helps a
little; conditioning on *this* patient does not.

This is the shuffled-z control doing exactly the job it was added for. D23 established
the control is a real manipulation — z identifies its patient at 55.5% against 12.5%
chance — so its failure here cannot be dismissed as a vacuous comparison.

## Why: the training signal is not there

| patients | baseline BCE |
|---|---|
| the 5 the hypernetwork trains on | 0.0066 |
| the 2 unseen validation patients | 0.726 |

A 112x gap (D26). The objective is already satisfied on every example Stage 7 may see.
With a peak learning rate of 1e-3 applied to that near-zero gradient, ‖dW‖ reached 1.17
after one epoch and oscillated to 1.98, while validation BCE swung 1.36 → 1.02 → 2.76 →
1.13 → 1.51 → 1.78 → 1.51. That is noise driving large weight deltas, not learning.

## What this does and does not establish

Does: on this fold, the method as published does not personalise. The gain is
indistinguishable from a random signature, and the adapted model is substantially worse
on the loss it is selected by.

Does not: generalise to the cohort — this is **one fold**. Nor does it separate the
method from its optimisation: a 1e-3 peak LR on a starved gradient is a plausible
mundane cause that has not been ruled out.
