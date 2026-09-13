# Decision log

Every choice that is not dictated by the VSViG paper or the methodology draft, with
the reasoning and the evidence behind it. `config.py` carries the same decisions as
inline tags; this file is the narrative.

Provenance vocabulary, used consistently in both places:

| tag | meaning |
|---|---|
| `[PAPER]` | stated in Xu et al., VSViG, ECCV 2024 |
| `[METHOD]` | stated in `Methodology_draft_updated_3.sep.docx` |
| `[INFERRED]` | not stated anywhere; derived because the source is silent or self-contradictory |
| `[DECISION]` | our deliberate choice |

---

## D1. Starting point: commit `6fca412`, reimplemented

**Decided:** start from the specified commit and rebuild, treating the later
`feature/hypernetwork-lopo-adaptation` branch as reference only.

`6fca412` predates all hypernetwork work — six later commits add 1,686 lines that
implement most of the methodology. Starting here is a deliberate clean-slate choice,
not an oversight. It also means inheriting two defects that the later branch had
already fixed, both since corrected here: model selection on the held-out patient,
and a keypoint channel mismatch that made the commit unable to run at all.

## D2. Cohort: 8 patients, 18 seizures

**Decided:** exclude `pat05, pat10, pat12` (round 1) and `pat11, pat13, pat14`
(round 2). Retained: `pat01, pat02, pat03, pat04, pat06, pat07, pat08, pat09`.

**Round 1** applies the methodology's own rule — under one minute of pre-EEG
baseline, summed across a patient's seizures — with a rescue clause for patients whose
supplementary seizure-free footage can still fill Pool A. That saves `pat03` (17 s of
pre-EEG video but an 11-minute `free.mp4`) and `pat04` (14 s, plus 45.6 min across
`free.mp4` and `no-Sz2P.mp4`).

**Round 2 revises the methodology.** The draft retains `pat11/13/14` to reach 11
patients and 24 seizures. Measurement showed they cannot support the method:

| | pat11 | pat13 | pat14 |
|---|---|---|---|
| pre-EEG footage | 61 s | 37 s | 32 s |
| interictal clips (non-overlapping) | 10 | 7 | **5** |
| evaluable test exposure | 50 s | 35 s | 25 s |
| one false alarm = | 72 /h | 103 /h | 144 /h |

`pat14` cannot fill a Pool A of 6 at all, and the largest Pool A the full 11-patient
cohort could support is **5** — against the methodology's own N ≤ 20.

The decisive argument is not about these three patients; it is about the other eight.
Keeping them forces `POOL_A_SIZE` down to 5 for the entire cohort, estimating every
patient's μ and σ from a quarter of the specified samples, in order to include three
subjects whose FDR/h could not be measured anyway. Dropping them makes the smallest
interictal pool `pat03`'s 42, so **`POOL_A_SIZE = 20` becomes feasible for everyone**,
with at least 22 clips left for Pool B.

**Cost, stated plainly:** 8 patients / 18 seizures instead of 11 / 24, and the §3.5
hypothesis test runs at n = 8. This is a trade of cohort size for signature quality
and must be reported as such — never presented as the methodology's original cohort.

**Unplanned benefit:** the retained cohort is 4 PG and 4 P, so a semiology-stratified
validation pair is drawable in every fold.

**Consequence for fold structure:** the methodology's 1 test / 2 val / 8 train becomes
1 / 2 / **5**. Five conditioning points per fold makes z-jitter load-bearing rather
than merely prudent — without it the trunk can memorise five points outright.

## D3. Signature: μ and σ from stages 0–2

**Decided:** extract at the stage-2 cut, `C′ = 192`, so the projector is 384 → 128
(not 768 → 128).

§3.2.2 says "Stages 0–2" but also states `X ∈ ℝ^{15×30×384}`, which exists nowhere in
the network: stage 2 emits 192 channels and T is downsampled 30 → 15 → 8 → 4. Confirmed
empirically by `scripts/verify_shapes.py`: after stages 0–2, `C′ = 192, T′ = 8, P = 15`.

**Consequence to report:** σ is a standard deviation over **8 downsampled timesteps**
per clip, not 30 raw frames. Coarser than the draft's text implies.

## D4. σ definition and Pool A aggregation

**Decided:** σ is the temporal standard deviation taken *before* spatial pooling, per
the methodology — capturing joint velocity, not inter-clip posture drift. Across the
Pool A clips, μ and σ are computed **per clip and then averaged**, not pooled into one
time axis: concatenating clips drawn hours apart would inject spurious velocity spikes
at the seams. Biased estimator, so a single clip yields σ = 0 rather than NaN.

The base repo computed something different — std *across clips* of a fully
spatiotemporally pooled embedding — which is a measure of posture change between
clips, not of kinematic volatility.

## D5. Skeleton normalisation: mid-hip centred, torso-scaled

**Decided:** implement §3.1's mid-hip centering and torso-length scaling.

The base repo divides x by 1920 and y by 1080 and does no centering, leaving the
representation sensitive to where the patient lies in the bed and where the camera is
mounted — exactly the nuisance variation that would otherwise enter z_behavior as if
it were motor signature. Verified translation- and scale-invariant, and graceful on
missing hips, degenerate torso and total pose failure.

Keypoints carry **2 channels (x, y)**, matching the paper's `Stem(x_it, y_it)`. The
3-channel variant retaining pose confidence is available as a one-line ablation
(`config.KPT_CHANNELS`); confidence is informative about imputed joints but is a
detector artifact rather than a spatial coordinate.

## D6. FDR per hour

**Decided:** a false alarm is a discrete **event**, not a thresholded window.
Consecutive windows above threshold merge; a 60 s refractory period runs from each
event's onset; a new event requires both that the accumulated probability has fallen
back below threshold and that the refractory has expired.

Denominator: interictal seconds **actually evaluated**, excluding the pre-ictal
transition, the ictal window, and 15 minutes of post-ictal recovery. Footage the model
was never run on is not time it was at risk of alarming, so exposure is counted from
clips written, not from file duration.

**Detection is measured separately**, from the raw accumulated-probability series
rather than the grouped events — otherwise a false alarm 45 s before onset would
suppress the alarm that actually detects the seizure, making sensitivity depend on
unrelated interictal noise.

Alarms *before* EEG onset count as false alarms, not early detections. This is the
standard convention and is why L_EO is reported as a non-negative latency.

## D7. Accumulation rule: mean

**Decided:** `AP_t` is the **mean** of clip probabilities over τ = 3 s.

The paper writes a sum. Summing ~6 sigmoid outputs against `DT = 0.3` is satisfied
almost unconditionally and cannot reproduce the paper's reported latencies. Recorded as
our reading, not as the paper's method. Configurable via `config.ACCUM_RULE`.

## D8. Detection timestamp: clip end

**Decided:** a clip's prediction is attributed to `t_start + 5 s`, the earliest instant
a real-time system could have emitted it. The paper never says which edge it used, and
the choice shifts every latency by up to 5 s.

## D9. Hypothesis test: Poisson rate model

**Decided:** §3.5 is fitted as a Poisson rate model with a **log-exposure offset**,
not a Pearson correlation on raw FDR/h ratios. Exposure spans 0.10 h to 0.76 h across
the retained cohort, so a bare ratio is dominated by recording length. Pearson,
Spearman and a bootstrap CI are reported alongside as secondary.

False-alarm **count** and **exposure hours** are reported beside every rate, per fold,
never folded away into it.

## D10. Test-time extraction does not overlap

**Decided:** training extraction overlaps ictal and transition clips by 4 s as
augmentation; test extraction is a continuous 5 s sliding window with a 5 s hop, in a
separate directory.

Scoring overlapping clips at test time inflates results twice: the accumulation window
sees one movement repeatedly, and a seizure gets several independent chances to be
detected. The base pipeline implemented only the training half.

## D11. Supplementary footage sampled uniformly, capped at 40

**Decided:** `free.mp4` / `no-Sz2P.mp4` windows are subsampled **uniformly across the
whole file** to the methodology's cap of 40, not taken as a contiguous prefix. A prefix
of a 30-minute recording is ~200 s from one moment of one activity — the transient
behaviour the uniform sampling exists to avoid.

## D12. Ablation micro-cohort, pre-registered

**Decided:** ablations and hyperparameter sweeps run on three folds — `pat09` (high
seizure count), `pat04` (low count, focal), `pat02` (median count, generalised) —
chosen on seizure count and semiology **only**.

Selecting them by D_p, as first proposed, is circular: D_p depends on the signature
still being built, and tuning on a high-D_p fold optimises for exactly the patients
§3.5 predicts should benefit most, leaking into the headline result. Fixed here before
any z is computed. `pat03` gets a smoke check on every pipeline change as the tightest
remaining Pool A / Pool B split.

Epoch ceilings: 50 for the backbone, 100 for the hypernetwork, validation every epoch
so the patience counter is meaningful inside the cap.

## D14. Pool A time span is not equalised — open, decide when building §3.2.3

**Observed after Stage 4, not yet decided.**

Every patient gets 20 clips, so the estimation *precision* of μ and σ is now equal —
that was the point of the cohort revision (D2) and it worked. But the **timeline those
20 clips span** still varies 24-fold:

| | pat01 | pat02 | pat04 | pat08 | pat03 | pat07 | pat06 | pat09 |
|---|---|---|---|---|---|---|---|---|
| Pool A span | 3570 s | 2816 s | 1775 s | 1190 s | 655 s | 470 s | 260 s | **150 s** |
| Pool A sources | 2 | 2 | 2 | 3 | 2 | **1** | 3 | 3 |

pat09's signature samples 2.5 minutes of the patient's stay; pat01's samples an hour.

This matters specifically for the §3.2.3 stability ratio, whose denominator is
intra-patient variance across temporal blocks. A patient whose Pool A spans 150 s will
look artificially **stable** — its four blocks are minutes apart, not hours — so the
ratio is not comparable across patients, and a pooled ≥ 2.5 threshold would be
measuring recording length as much as signature stability.

**Proposed:** report the stability ratio **per patient alongside its Pool A span**
rather than pooling to a single number, and treat the ≥ 2.5 gate per patient. To be
settled when §3.2.3 is implemented.

Two further consequences of the same heterogeneity, worth carrying into the write-up:

- **pat07's Pool A is single-source.** Its Sz1 recording contains zero interictal
  windows (EEG onset at 4 s), so stratification across sources degenerates to one
  stratum and the signature comes entirely from Sz2.
- **pat04's Pool A comes entirely from supplementary footage** (`free.mp4`,
  `no-Sz2P.mp4`); its seizure recording contributed none of the 20.


## D13. Environment

**Decided:** conda supplies the interpreter only; every library comes from PyPI via
`requirements.txt`.

Splitting the numeric stack across conda-forge and pip loads two OpenMP runtimes on
macOS and aborts at `import torch`. `KMP_DUPLICATE_LIB_OK` is deliberately not used
anywhere: its own documentation says it can "silently produce incorrect results", which
is not a trade worth making in a pipeline whose outputs feed a clinical metric.

`torchvision` was dropped (dead import) and `timm` made optional (used only for a
registry decorator this project never queries).

**Noted:** the previous `vsvig` conda env carries a CPU-only torch build, so all
earlier LOPO training ran on CPU.

---

## D15. Deterministic dynamic partition shuffle

**Decided:** express the joint shuffle as a functional gather with a custom adjoint,
replacing the original in-place advanced-index assignment.

`Part_3DCNN.dynamic_trans` shuffled the 15 joints with
`x[:, raw_order] = x[:, dynamic_order]`. That is a permutation, but its backward is
`index_put_` with accumulation, and the MPS kernel for that
(`index_put_with_accumulate_mps`) has no deterministic implementation. With
`use_deterministic_algorithms(True, warn_only=True)` set in `src/utils/seeding.py`, it
warned rather than raised — so training ran, but **seeded runs were not bit-identical**
and the float drift compounded across epochs.

That matters more here than it usually would: the headline result is a rate model over
eight points, so per-fold run-to-run variance of the same order as the effect would be
indistinguishable from the effect.

Because the index set is a permutation — every joint used once, nothing accumulated —
the adjoint is exactly the inverse permutation. Supplying it directly is mathematically
exact rather than an approximation, and cheaper than the general scatter. Forward output
is unchanged; `state_dict` is unchanged (the permutation is cached, deliberately not
registered as a buffer, so checkpoints stay compatible); and an in-place write inside
the autograd graph is removed as a side effect.

Guarded: the code raises if the configured order is not a permutation, since the
adjoint would then be wrong rather than merely non-deterministic.

**Timing:** fixed after the first pat01 fold started and before any backbone was
finished. Once eight backbones exist, changing anything inside the model means
retraining all eight for consistency — this was the cheapest possible moment.

## D16. The decision threshold must be selected per fold, not inherited — RESOLVED 2026-09-13

**Raised by the first real evaluation, not yet decided.**

`DT = 0.3` comes from the VSViG paper `[PAPER]`. It was tuned there for a model trained
on that paper's distribution and evaluated on a within-patient random split. We have
inherited the number and applied it to a differently-trained model on a
patient-independent split, which is not obviously valid.

The concrete problem: Step 1 trains under a 45/45/10 rebalanced sampler, so the model's
implicit prior is near 0.5, while deployment is overwhelmingly interictal. A model whose
outputs sit near its training prior clears `DT = 0.3` almost everywhere. The first pat01
evaluation showed exactly that shape — sensitivity 1.0, L_EO ≈ 0, and 31 alarm events in
40 minutes — which is a detector that never stops detecting.

**Proposed:** treat DT as a fold-level hyperparameter chosen on the two INTERNAL
VALIDATION patients, never on the held-out test patient. Selecting it on test would be
tuning the operating point on the patient the result is about — the same class of leak
the fold structure exists to prevent, one level up.

An operating criterion has to be chosen with it, since sensitivity and FDR/h trade off:
maximise sensitivity subject to an FDR/h ceiling, or minimise FDR/h subject to a
sensitivity floor. That is a clinical judgement, not a statistical one.

Whatever is chosen, DT must be **identical for the baseline and the adapted model**
within a fold, or the section 3.5 comparison measures threshold placement rather than
personalisation.

## D17. The backbone was trained for ~3% of the reference budget

**Established by diagnosis, fix not yet chosen.**

The first pat01 runs produced AUC(ictal vs interictal) = 0.513 on the held-out patient
— chance — with all three classes sharing one output distribution. Three candidate
causes were eliminated in turn rather than guessed at:

- **Data.** `scripts/diagnose_data.py`: 67–97% keypoint validity, patch means 0.17–0.26
  with real variance, no dead joints, normalised keypoints spanning [−1.29, 1.97] torso
  units with 6.7% zeroed. Pose estimation found the patient.
- **Weight decay.** Ruled out analytically, without spending compute: AdamW's decoupled
  decay shrinks weights by `(1 − lr·wd) = 1 − 5e−6` per step, so over the ~960 steps of
  the real run the total factor is 0.995.
- **Structure.** `--overfit 32` drove MSE to 0.00005 (RMSE 0.71%). The model memorises
  what it is shown, so gradients, architecture and the data path are sound.

**What remains is budget.** The same probe showed the model learns slowly at lr 1e-4:
memorising 32 clips took ~400 gradient steps to reach MSE 0.01 and ~800 to reach 0.0001.
The real run did **960 steps in total**. The paper's 200 epochs at 160 batches/epoch is
roughly **32,000**. Training ran for about 3% of the reference schedule and then
early-stopped on a validation signal still contaminated by BatchNorm warm-up.

Naively raising the budget costs 127 s/epoch × 200 epochs × 8 folds ≈ **56 h**, which is
not acceptable. So the budget question is really a throughput question first:
`num_workers=0` means every batch stalls the main thread on 16 synchronous reads of
~5.5 MB clips before any compute begins. `scripts/benchmark_throughput.py` separates
compute from I/O and reports the achievable epoch time.

**RESOLVED.** Throughput was measured, not assumed, and there is no speedup to be had:

| lever | result |
|---|---|
| DataLoader workers | within noise — the job is compute-bound (519 ms/batch compute vs 28–74 ms of overlappable I/O) |
| fp16 autocast | 1.01× |
| batch 32 / 64 | **worse** per clip: 32.5 → 35.5 → 38.0 ms. Batch 16 is already optimal |

So ~82 s/epoch is the hardware floor, and 8 folds cost 9.2 h / 18.4 h / 36.9 h at
50 / 100 / 200 epochs.

**Settled:**
- `S1_MAX_EPOCHS = 300` as a **cap, not a target**.
- `S1_PATIENCE = 30`, not counted before epoch 15 (`S1_PATIENCE_WARMUP_EPOCHS`).
  Patience 5 fired at epoch 6 on a signal that had not stabilised — convergence here is
  slow, and BatchNorm running statistics need many updates before an eval-mode metric
  means anything (the overfit probe measured a +0.81 MSE gap between batch and running
  statistics at epoch 1).
- Fold 1 (`pat01`) runs against the cap with per-epoch validation AUC logged. It is a
  real fold, so nothing is wasted; the budget for folds 2–8 is then set from where that
  curve actually plateaus. Worst case for fold 1 is 6.8 h.
- Learning rate left at 1e-4. One change at a time: the schedule, the selection metric
  and the budget have all just moved, and an LR change now would be unattributable.

**What the curve decides.** If AUC climbs and flattens, the plateau sets the budget. If
it peaks and declines while training loss keeps falling, more epochs will not help — a
five-patient training cohort cannot support generalisation at this capacity, which is a
finding about the problem rather than a training failure. Either way it must be read
before spending on the remaining seven folds.

## D18. Checkpoint selection on validation AUC

**Decided:** select checkpoints and early-stop on validation AUC over the two internal
validation patients; keep MSE in the log.

The checkpoint chosen by best validation MSE scored **AUC 0.513** — chance. MSE is
dominated by the label distribution, while detection consumes a *ranking*: the
accumulation rule asks whether ictal clips score above interictal ones. Selecting on
MSE was choosing models that cannot discriminate at all.

The two metrics are not close to interchangeable. On synthetic data, a constant 0.44
predictor and a weakly discriminative one score MSE 0.2416 vs 0.2232 — 8% apart — while
their AUCs are 0.500 and 0.691.

AUC is computed on the internal validation patients only, so it stays leak-free, and it
is a direct proxy for the cross-patient transfer the project is actually about. MSE
remains logged so results stay comparable to the paper's reported RMSE. Soft transition
labels are dropped from the AUC rather than bucketed into a class.

## D19. AUC is computed within recording, never pooled across recordings

**Date:** 2026-09-12 — after the first honest held-out evaluation (fold 1, pat01).

Fold 1 finished at validation AUC 0.9111 and was evaluated on pat01's 527
non-overlapping sliding-window test clips. Pooled across the patient's recordings the
AUC read 0.625. Broken down by recording it read **0.511 and 0.547** — chance, twice.

The pooled figure is an artifact, and the mechanism is Simpson's paradox. Recordings
sit at different baseline score levels, and the ictal/interictal mix differs between
them. A model that merely scores recording B above recording A therefore earns AUC
above 0.5 without ordering a single clip correctly inside either one. Measured on
every scope available at the time:

| scope | pooled | within recording |
|---|---|---|
| pat01 — held out | 0.625 | 0.511, 0.547 |
| pat04 — internal validation | 0.836 | 0.762 |
| pat07 — internal validation | 0.942 | 0.860, 0.890 |
| pat03 — in the training set | — | 1.000, 1.000 |

The inflation is +0.07 to +0.10 everywhere it can be checked, and it is not uniform
across patients — which is precisely what makes it dangerous here. **The central claim
of this project is a comparison**: adapted model minus baseline, correlated against
behavioural atypicality. A metric whose bias varies by patient would let
between-recording score offsets enter that correlation as if they were personalisation
benefit. The §3.5 result could come out significant for reasons having nothing to do
with the hypernetwork.

**Decision.** The selection and reporting metric is the Mann-Whitney statistic
restricted to within-recording pairs:

    AUC_ws  =  Σ_s (n_pos_s · n_neg_s · AUC_s)  /  Σ_s (n_pos_s · n_neg_s)

Weighting by pair count, rather than averaging per-recording AUCs equally, is what
keeps this a single Mann-Whitney estimate. It also removes the need for a minimum
clips-per-recording rule: a recording holding only one class contributes zero pairs and
drops out on its own. Free-footage files, which have no ictal clips by construction, are
handled by that rule rather than by a special case.

The unweighted mean over usable recordings (`macro`) and the old pooled figure are both
logged beside it. Pooled is retained **only** so the size of the artifact stays visible
in the record; it is never the headline and never selects.

Implemented in `src/eval/metrics.within_source_auc`, wired into
`scripts/train_backbone.py` (selection) and `scripts/evaluate.py` (reporting).
`config.S1_SELECTION_METRIC = "auc_within_source"`.

**Consequence, stated plainly.** Fold 1 was selected under the pooled metric and must
be retrained; its checkpoint is not comparable to anything produced from here on. The
trainer now refuses to resume a checkpoint whose `selection_metric` differs from the
configured one, rather than silently comparing two incomparable high-water marks.

**What this does not decide.** It does not fix the calibration collapse observed in the
same run (pat07_Sz1: AUC 0.860 at a mean ictal-minus-interictal separation of +0.002).
Ranking survives where margin does not, and a fixed DECISION_THRESHOLD of 0.3 cannot sit
sensibly on such a distribution. That is D16, now blocking rather than optional.


## D19a. Patients are weighted equally; pairs weight recordings only within a patient

**Date:** 2026-09-12, same day as D19, after the fold-1 retrain.

D19 weighted every within-recording pair equally. Decomposing the fold-1 validation
curve by patient showed what that does in this cohort: pat04 contributes one seizure
recording and pat07 two, so the combined metric was numerically **identical to pat07
alone** — corr(combined, pat07) = +1.000, corr(combined, pat04) = +0.425. A selection
metric for a study about patient heterogeneity had quietly become a single-patient
metric.

Pairs still weight recordings within a patient. Patients are then averaged equally.
`within_source_auc` returns `patient_balanced` (selects and reports), with
`pair_weighted`, `macro` and `pooled` retained for comparison.

## D20. Checkpoint selection on internal validation does not transfer — OPEN

**Date:** 2026-09-12, after the fold-1 retrain under D19.

Fold 1 was retrained selecting on within-source AUC. It reached validation 0.9606 at
epoch 29 and early-stopped at 59. Three checkpoints from that single run were then
scored on the held-out patient:

| checkpoint | validation (within-source) | held-out pat01 (within-source) |
|---|---|---|
| epoch 29 — best by within-source | 0.9606 | **0.498** |
| epoch 33 — best by pooled | 0.9359 | 0.525 |
| epoch 59 — final | 0.8528 | 0.542 |

All three sit at chance on pat01, and they are ordered *inversely* to their validation
scores. Fixing the metric changed which epoch was chosen and did not change the
outcome, because **the choice does not matter**: nothing in this validation set
predicts held-out performance.

The mechanism is visible in the per-epoch validation scores (`val_scores_by_epoch.npz`,
the artifact added after D19 precisely so this kind of question would not cost another
training run). Decomposed by validation patient, the two curves are **uncorrelated
after warm-up: corr(pat04, pat07) = -0.185** over epochs 16-59. The validation signal
does not generalise from one validation patient to the other, so there is no reason to
expect it to generalise to a third. Per-epoch noise is correspondingly large: mean
epoch-to-epoch |ΔAUC| is 0.084 combined and 0.186 for pat04, whose estimate rests on a
single recording's worth of pairs.

This is not a metric bug. It is the internal validation set being too small to select
on: LOPO over 8 patients leaves 2 validation patients, and after D19a's balancing,
2 noisy estimates.

**Not yet decided.** The options, none free:

1. **Remove selection from the protocol.** Pre-register a fixed epoch budget, no early
   stopping, take the final weights (or an average of the last k). Selection noise
   becomes zero by construction, and baseline and adapted are then trained under an
   identical rule, which is what a paired comparison needs. Costs the possibility that
   a fixed budget is wrong for some folds.
2. **Enlarge internal validation** to 3 patients, leaving 4 to train on. Almost
   certainly the wrong trade at this cohort size.
3. **Smooth the selection curve** (k-epoch mean). Reduces variance, does not create
   signal that is not there — the cross-patient correlation says there is none.

Whichever is chosen, it must be **identical for the baseline and the adapted model**.
The §3.5 claim is a paired difference; if the two arms select checkpoints under
different amounts of noise, the difference measures the selection rule.

**pat01 is not representative.** The fold-1 checkpoint was scored against all eight
patients (`outputs/results/baseline/fold1_model_all_patients.md`). It is at chance on
pat01 alone: the two patients it never trained on score 0.966 and 1.000 under the
selected checkpoint and 0.860-0.890 and 0.762 under a non-selected one, while the five
training patients sit at 0.997-1.000. So the spread across *unseen* patients is roughly
0.50 to 0.90 — large, and pat01 is its floor. No conclusion about the cohort baseline
should rest on fold 1.

**A consequence that changes the arithmetic of option 1.** Under a fixed pre-registered
budget with no early stopping, the internal validation patients are not used to choose
anything. They stop being selection-contaminated and become ordinary unseen patients.
Each fold then yields **three** honest held-out measurements instead of one, at no extra
compute.

**What this does not say.** It does not say the backbone is untrainable — it reaches
1.000 within-recording AUC on patients it has seen, and 0.76-0.89 on pat04 and pat07.
It says that for pat01 the cohort model transfers no usable ranking, and that we cannot
currently pick a checkpoint that would change this.


## D21. Fixed pre-registered budget: 50 epochs, no early stopping, no selection

**Date:** 2026-09-12, resolving D20.

D20 established that the internal validation set cannot select a checkpoint. D21 stops
trying. Every fold runs a fixed 50 epochs; the evaluated model is the average of the
last 5 epochs' weights with BatchNorm running statistics recomputed afterwards.

**Why 50.** Both pat01 runs plateaued by epoch 30-40. 50 leaves margin and, more
importantly, lets the cosine schedule actually anneal: with `T_max = S1_MAX_EPOCHS =
300` the learning rate had moved only 1e-4 -> 9.1e-5 by epoch 59, so the schedule was
doing nothing at all. At T_max = 50 it reaches `eta_min`.

**Why averaging, and why the BatchNorm pass.** Averaged weights carry averaged
BatchNorm running statistics, which correspond to no forward pass the network ever
made. Recomputing them over training batches is the standard SWA step and skipping it
is the usual reason weight averaging looks ineffective. Capped at 100 batches
(1600 clips) -- far more than running averages need.

**Why this makes the section 3.5 claim stronger, not weaker.** The claim is a paired
difference, baseline minus adapted, per patient. Under per-epoch selection each arm
draws its checkpoint from a signal uncorrelated with held-out performance, so the
difference carries two independent noise terms and the correlation against atypicality
inherits both. A fixed rule applied identically to both arms removes them by
construction. **The rule is now pre-registered and must not differ between arms.**

**Consequence worth stating.** With no selection, the internal validation patients are
not used to choose anything. They stop being contaminated and become ordinary unseen
patients. Each fold therefore yields three held-out measurements, and across eight
folds every patient is measured roughly three times under different training sets -- so
per-patient difficulty gets a variance, not a point estimate.

**Insurance against another retrain.** Weights are written every 10 epochs, and
`val_scores_by_epoch.npz` holds every validation clip's score at every epoch. A later
change to the metric, the selection rule or the budget (within 50) is answerable from
disk. What would still force a rerun is a change to the learning rate, the sampler, the
architecture or the data.

**Cost.** 12.1 h for all eight folds, measured from the one timed fold and scaled by
each fold's clip counts (pat02 1.14 h to pat04 1.71 h). 1.6 GB of checkpoints.


## D22. The LOPO baseline exists: held-out mean 0.678, and it is wildly heterogeneous

**Date:** 2026-09-13. All eight folds under D21, 12.4 h.

24 measurements (8 held-out patients + 16 internal-validation patients, the latter
usable because D21 selects nothing). Full table in
`outputs/results/baseline/lopo_baseline_matrix.md`.

- held-out patients: mean 0.678, median 0.714, range 0.180-0.967
- internal-validation patients: mean 0.699, median 0.726
- the two agree, which is the check that D21's validation patients are genuinely
  uncontaminated; had selection leaked they would score systematically higher

**D21 is vindicated on the one fold we can compare.** pat01 scored 0.498 / 0.525 / 0.542
under three selected checkpoints and **0.604** under the averaged model. The earlier
"held-out is at chance" reading was substantially a selection artifact.

**Validation still does not predict held-out performance**: corr across the eight folds
is +0.409, weak and on n = 8. D20 stands; validation is a monitor.

**Per-patient difficulty spans 0.180 to 0.918.** pat07 0.918, pat03 0.829, pat02 0.769,
pat08 0.741, pat04 0.700, pat01 0.562, pat06 0.543, pat09 0.180. This is the
heterogeneity section 3.5 is a hypothesis about, and it is now measured rather than
assumed -- with 2 to 5 independent measurements per patient for six of the eight.

**pat09 is confidently wrong, not uninformative — OPEN.** Its model ranks pat09's
seizure clips systematically *below* its interictal clips (AUC 0.180), and sat inverted
at 0.4485 +/- 0.0327 on validation for 35 consecutive epochs. It is not a broken run:
the same weights score 0.9999 on pat02 and 1.0000 on pat03 with ictal mean 0.98 against
interictal mean 0.016. This is negative transfer.

LOPO cannot separate "pat09 is hard" from "this training set transfers backwards to
pat09", because only fold pat09 leaves pat09 unseen. Two consequences to settle before
Stage 7:

1. An anti-correlated baseline gives a hypernetwork enormous and cheap headroom on
   pat09. Any section 3.5 correlation will be dominated by that single point unless the
   analysis is robust to it. The Poisson model of D9 must be checked with and without
   pat09, and the sensitivity reported either way.
2. pat09 has 71 ictal and 77 interictal test clips and is the only fold whose training
   set held more interictal than ictal (1081 vs 947). Whether that is causal is not
   established.


## D24. A_base is initialised with 1e-2 read as a variance scale

**Date:** 2026-09-13, resolving the open item.

The draft writes `A_base ~ N(0, d_in^-1 x 10^-2)` and the two readings differ 100-fold.
Settled on the numbers rather than on preference:

| target | d_in | std if VARIANCE | std if STD | standard LoRA N(0, 1/d_in) |
|---|---|---|---|---|
| stage3_*_conv2 | 1728 | 0.002406 | 5.79e-06 | 0.02406 |
| fc0 | 384 | 0.005103 | 2.60e-05 | 0.05103 |
| fc3 | 256 | 0.006250 | 3.91e-05 | 0.06250 |

The variance reading is **exactly 0.100x** the standard LoRA initialisation across every
target — a round number that reads as deliberate. The std reading is 0.000241x, putting
A_base at ~6e-6 on the stage-3 layers. Because B is zero-initialised, that would leave
`A_p = A_base + A_hyper(z)` with no base: the shared, patient-independent direction the
decomposition exists to provide would be numerically absent at initialisation, and the
whole base+residual structure would be decorative.

**Decision: the variance reading.** `std = sqrt(HN_A_BASE_STD_SCALE / d_in)`. The
6fca412 base repo implements the std reading; by this argument that is a bug, not a
choice. Asserted in `tests/test_hypernetwork.py`.

## D25. A training batch must be homogeneous in z_behavior — OPEN

One forward pass carries one weight matrix, but the hypernetwork emits a different dW
per patient. A batch mixing patients cannot be adapted correctly without per-sample
weights (grouped convolution), so `AdaptedForward` refuses a heterogeneous batch loudly
rather than silently applying the first patient's delta to everyone.

The straightforward fix is patient-homogeneous batching: draw each batch from a single
training patient. That is standard for amortised hypernetworks, but it changes the
Stage 6 sampler, and it interacts with BatchNorm — batch statistics would then be
computed within one patient, which is itself a mild form of adaptation and would
contaminate the baseline-versus-adapted comparison unless the baseline is retrained
under the same batching.

Not yet decided. The options and their costs belong in front of a human before ~12 h of
training is spent on either.


## D25. Patient-homogeneous batching and frozen BatchNorm — RESOLVED

**Date:** 2026-09-13, per the methodology's own step 2-3 and the user's instruction.

Stage 7 uses the Max-Pool Dynamic Cyclic Sampler of §3.4.1 (patient-homogeneous
batches, `S_p = max(|D_inter,p|, |D_ictal,p|) / (BatchSize/2)`, majority drawn
sequentially without replacement, minority wrapped with `itertools.cycle`, invariant
50/50 per step), and freezes the backbone **including its BatchNorm buffers** in eval
mode. Freezing BN is what stops patient-homogeneous batches from computing statistics
within a single patient, which would be a second, unlabelled channel of adaptation.

Accepted cost, explicitly: the Stage 6 baseline was trained with multi-patient shuffled
batches and live BN, so it is not a like-for-like control for the adapted arm. A
baseline retrained under frozen BN may be needed before the final comparison.

[INFERRED] Transition clips carry soft labels in (0,1) and belong to neither pool; the
50/50 rule is stated over interictal and ictal only, so they are excluded from Stage 7
training. They are still scored at evaluation.

[DECISION] Batch ORDER is shuffled across patients within an epoch. The draft fixes
batch contents but not sequence; emitting all of a patient's S_p batches consecutively
would make the update direction strongly autocorrelated.

## D26. Stage 7 has almost no gradient signal — OPEN, and it is structural

**Date:** 2026-09-13, measured during the first Stage 7 smoke run.

The methodology's sequence is: train the backbone on 5 patients (step 1), freeze it
(step 2), then train the hypernetwork on **those same 5 patients** (step 3). That
assumes the frozen backbone leaves useful residual error on its own training patients.
Measured on fold pat01, it does not:

| patients | baseline BCE (no hypernetwork) |
|---|---|
| training patients (backbone was fitted on them) | **0.0066** |
| internal validation patients (never seen) | **0.738** (pat04 1.153, pat07 0.530) |

A **112x gap**. The loss Stage 7 is asked to minimise is already ~0 on every example it
is allowed to see. This is the same memorisation D22 measured from the other side —
within-source AUC 0.997-1.000 on training patients — expressed as a loss.

The consequence is that the hypernetwork is optimised where there is nothing to gain,
while the quantity that matters — generalisation to an unseen patient — never enters
the objective. The first smoke run is consistent with that: train BCE fell 0.0069 →
0.0049 over two epochs while validation BCE **rose** 1.197 → 1.313.

Options, none free:

1. **Run it anyway.** Early stopping is on validation loss with patience 5, so a
   hypernetwork that only hurts stops within ~6 epochs. Cheap to find out, and a
   negative result is a finding about the method as published.
2. **Train the hypernetwork on the internal validation patients** instead — real
   gradient, but it consumes the only clean unseen patients and leaves nothing for
   early stopping.
3. **Weaken the backbone deliberately** (shorter budget, stronger regularisation) so
   training patients retain non-trivial loss. Fixes the root cause, keeps the design
   intact, costs a full baseline retrain.
4. **Cross-fitting** — train several backbones per fold so every hypernetwork training
   patient is scored by a backbone that did not see it. Correct and expensive.

## D27. The patient a batch belongs to must travel with the batch

A first version of the Stage 7 trainer paired batches with patients via
`zip(loader, sampler.epoch_patients)`. Python binds that attribute before `__iter__`
fills it: epoch 1 zipped against an empty list and ran **zero** batches (a training loss
of exactly 0.00000 was the only symptom), and every later epoch paired batches against
the **previous** epoch's patient order — injecting the wrong patient's z_behavior
throughout, silently, in the one place the entire experiment's meaning depends on.

The patient is now carried inside the batch (`WithPatient`), and homogeneity is asserted
per batch. Any recurrence raises instead of mis-training.


## D28. The personalisation criterion, pre-registered — and the specification run fails it

**Date:** 2026-09-13, registered before the option-A diagnostic is run.

**The criterion.** A gain over the BASELINE is not evidence of personalisation:
perturbing a frozen network's weights at all can help. Personalisation is claimed only
if the held-out AUC obtained with the patient's OWN z stands apart from the distribution
obtained with every other cohort patient's z:

    z_score = (AUC_own - mean(AUC_shuffled)) / sd(AUC_shuffled) > 2.0,  AUC_own > mean

The control runs over all seven substitutes, not one, because a single substitute is a
sample of size one. Implemented in `scripts/evaluate_adapted.py`.

**The specification run (fold pat01) fails it, decisively.**

| condition | held-out within-source AUC |
|---|---|
| baseline (dW = 0) | 0.6037 |
| adapted, pat01's own z | **0.6166** (+0.0129 vs baseline) |
| shuffled-z (pat02) | 0.6175 |
| shuffled-z (pat03) | 0.6133 |
| shuffled-z (pat04) | 0.6179 |
| shuffled-z (pat06) | 0.6159 |
| shuffled-z (pat07) | 0.6174 |
| shuffled-z (pat08) | 0.6166 |
| shuffled-z (pat09) | 0.6168 |

shuffled-z mean **0.6165**, sd **0.0015** over n=7.
Own z sits **+0.0001** from that mean —
**z = +0.05** against a threshold of 2.0.

The sd of 0.0015 is the striking part: every wrong patient's signature produces very
nearly the same effect as the right one. The +0.0129 over baseline is real and is
entirely attributable to perturbing the weights, not to which patient they were
perturbed for.

**This is not yet a verdict on the method.** It is one fold, under an optimiser that
took a near-zero gradient (D26: 0.0066 training BCE against 0.726 on unseen patients)
and a 1e-3 peak learning rate and produced ‖dW‖ of 1.17 after a single epoch. Option A
— peak LR 1e-4, delta budget 0.1, patience 10 — tests whether gentler optimisation
changes the answer. Registered in advance: if the option-A run also returns z < 2.0, the
project proceeds to option E and reports the negative result across all eight folds,
with this criterion as the pre-registered test rather than one chosen afterwards.

Specification runs and diagnostic runs are kept apart: any deviation from the
methodology's optimisation values requires `--tag`, is printed at startup, and is
recorded in the run manifest under `specification_deviations`.


## Open

- **Nothing extracted yet.** `preprocess.py` and `extract_test_clips.py` have both been
  dry-run only.
- **Old `processed_data/` and `outputs/` are not trusted** and are being rebuilt (Q7).
- ~~Decision threshold selection (D16)~~ — RESOLVED. Per-fold DT by Youden's J on the
  internal validation patients' accumulated scores; range 0.160-0.960. A rate-target
  criterion was rejected because exposure (0.101-0.764 h per patient) cannot resolve
  1 FDR/h. See outputs/results/baseline_dt/README.md. Original entry below.
- **(superseded) Decision threshold selection —** Fold 1 showed ranking without
  margin (AUC 0.860 at +0.002 separation; 62% of all clips above 0.9). No fixed DT can
  sit sensibly on that distribution, so every FDR/h number is meaningless until DT is
  selected per fold on the internal validation patients, under a stated operating
  criterion, held identical between baseline and adapted.
- **Pool A time span heterogeneity (D14)** — whether the §3.2.3 stability gate is
  applied per patient or pooled. Blocking for Stage 5's gate.
- ~~Checkpoint selection (D20)~~ — resolved by D21 (fixed 50-epoch budget, last-5
  weight averaging, no selection). The rule is pre-registered and must be identical
  for the baseline and the adapted model.
- **pat09's negative transfer (D22)** — whether section 3.5 is reported with and
  without it, and whether the training-set class balance is causal.
- **§3.5 at n = 8** — whether to report an additional sensitivity analysis, and against
  what exposure floor, once real FDR/h numbers exist.
- ~~A_base initialisation~~ — RESOLVED as D24 (variance reading; 0.100x standard LoRA).
- ~~Batch homogeneity (D25)~~ — RESOLVED: cyclic sampler + frozen BN.
- **Stage 7 gradient starvation (D26)** — blocking a meaningful adapted arm.
