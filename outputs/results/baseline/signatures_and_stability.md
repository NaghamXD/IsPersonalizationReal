# Stage 5: z_behavior, the §3.2.3 gate, and a first look at §3.5

Signatures built from each fold's own frozen backbone (`final_model.pth`, D21), so a
patient gets a different z in each fold — correct, since each backbone was trained
without its own test patient.

## The §3.2.3 stability gate fails

§3.2.3 states that a stability ratio ≥ 2.5 "guarantees that the spatial motor signature
successfully dominates arbitrary intra-patient activity shifts". Measured across all
eight backbones:

| backbone fold | pat01 | pat02 | pat03 | pat04 | pat06 | pat07 | pat08 | pat09 | pooled | passing |
|---|---|---|---|---|---|---|---|---|---|---|
| pat01 | 1.38 | 2.24 | 2.18 | 2.94 | 1.49 | 2.07 | 1.14 | 1.18 | 1.65 | 1/8 |
| pat02 | 1.17 | 3.09 | 4.83 | 5.57 | 1.28 | 3.14 | 0.93 | 1.14 | 1.73 | 4/8 |
| pat03 | 1.79 | 1.92 | 2.27 | 1.76 | 1.16 | 2.14 | 0.97 | 0.94 | 1.45 | 0/8 |
| pat04 | 2.21 | 3.29 | 1.80 | 2.44 | 1.82 | 1.35 | 0.50 | 1.18 | 1.36 | 1/8 |
| pat06 | 2.24 | 2.31 | 1.64 | 2.94 | 1.16 | 2.52 | 0.79 | 1.06 | 1.51 | 2/8 |
| pat07 | 1.43 | 2.44 | 1.74 | 3.07 | 1.30 | 1.31 | 1.26 | 1.66 | 1.62 | 1/8 |
| pat08 | 1.69 | 3.45 | 0.68 | 1.43 | 1.85 | 2.17 | 2.33 | 2.24 | 1.63 | 1/8 |
| pat09 | 1.31 | 2.94 | 3.67 | 2.62 | 1.02 | 2.39 | 1.20 | 1.34 | 1.69 | 3/8 |
| **mean** | **1.65** | **2.71** | **2.35** | **2.85** | **1.38** | **2.14** | **1.14** | **1.34** | | |

**13 of 64 patient × backbone ratios reach 2.5.** Mean 1.94, median
1.78, max 5.57. Every fold's pooled ratio (1.36–1.73) is far
below threshold. Only pat02 and pat04 average above 2.5; pat08 (1.14), pat09 (1.34) and
pat06 (1.38) are barely above 1.0, meaning a patient's signature moves almost as much
across their own day as it does between patients.

The ratio is also strongly backbone-dependent — pat03 ranges 0.68 to 4.83 across the
eight — so it is not a stable property of a patient at all under this definition.

### D14 is resolved, and not in the direction feared

D14 worried that a short Pool A span would shrink intra-patient distance and inflate the
ratio. The data says the opposite: corr(span, intra-patient distance) = −0.244 and
corr(span, ratio) = **+0.384**. The two shortest-span patients (pat06 335 s, pat09 365 s)
have the *largest* intra-patient distances and the *worst* ratios. Short spans are not
manufacturing passes, so the gate's failure cannot be explained away by heterogeneous
Pool A spans.

## §3.5's precondition does not hold yet

§3.5 predicts personalization benefit correlates with behavioural atypicality. That
mechanism presupposes the cohort baseline does worse on atypical patients. Testing
atypicality (‖z_p − mean z of the other seven‖) against baseline within-source AUC:

- all 24 model × patient pairs: Pearson **−0.003**, Spearman **+0.000**
- the 8 held-out patients: Pearson **−0.143**, Spearman **−0.286**

| patient | atypicality | baseline AUC |
|---|---|---|
| pat08 | 3.18 | 0.714 |
| pat07 | 3.45 | 0.967 |
| pat09 | 3.56 | 0.180 |
| pat02 | 4.11 | 0.864 |
| pat03 | 4.50 | 0.854 |
| pat01 | 4.86 | 0.604 |
| pat04 | 6.55 | 0.714 |
| pat06 | 7.76 | 0.526 |

Directionally negative on the held-out eight, but at n = 8 a Spearman of −0.286 is
indistinguishable from nothing (p ≈ 0.49). And the two extremes contradict the story:
**pat09**, the catastrophic 0.180, is among the *least* atypical patients (3.56), while
**pat06**, the most atypical (7.76), is unremarkable at 0.526. pat07 is nearly as typical
as pat09 (3.45) and scores 0.967.

This is not a refutation of §3.5 — benefit is not the same quantity as baseline
difficulty, and it cannot be measured until the hypernetwork exists. It does mean the
mechanism the draft describes has no support in the data so far.

## The common suspect

Both results are consistent with one cause: **z_behavior as currently defined does not
capture patient-specific motor behaviour.** It is a frozen *random* 384→128 projection
of [μ ‖ σ] over stage-2 features of 20 Pool A clips. Nothing in that pipeline was
trained or selected to be patient-discriminative, and the stability numbers say it is
barely more consistent within a patient than between patients.

Before Stage 7 commits compute to conditioning on this vector, the direct test is cheap:
**can z_behavior identify the patient it came from?** A nearest-centroid or logistic probe
on the 8 × 8 z matrix answers it in minutes. If z cannot separate patients, conditioning
on it is conditioning on noise, and the hypernetwork's shuffled-z control (Stage 8) will
be indistinguishable from the real thing by construction.
