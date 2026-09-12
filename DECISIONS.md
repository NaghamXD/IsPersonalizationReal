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

## D16. The decision threshold must be selected per fold, not inherited — OPEN

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

## Open

- **Nothing extracted yet.** `preprocess.py` and `extract_test_clips.py` have both been
  dry-run only.
- **Old `processed_data/` and `outputs/` are not trusted** and are being rebuilt (Q7).
- **Decision threshold selection (D16)** — inherit 0.3, or select per fold on
  the internal validation patients under a stated operating criterion.
- **Pool A time span heterogeneity (D14)** — whether the §3.2.3 stability gate is
  applied per patient or pooled. Blocking for Stage 5's gate.
- **§3.5 at n = 8** — whether to report an additional sensitivity analysis, and against
  what exposure floor, once real FDR/h numbers exist.
- **A_base initialisation** — the draft's `N(0, d_in⁻¹ × 10⁻²)` is ambiguous between a
  variance and a standard deviation; the code implements the latter. A 100× difference
  either way. Unresolved.
