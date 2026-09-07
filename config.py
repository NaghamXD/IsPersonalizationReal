"""Single source of truth for every constant this project's methodology specifies.

Nothing downstream should hard-code a number that appears here. If the methodology
draft and this file disagree, one of them is wrong and that is a bug worth raising.

PROVENANCE TAGS -- every value carries one. Do not silently promote an inference
to a fact; that distinction is the whole point of this file.

  [PAPER]    Stated explicitly in Xu et al., "VSViG", ECCV 2024.
  [METHOD]   Stated in Methodology_draft_updated_3.sep.docx.
  [INFERRED] Not stated in either; derived because the source is silent or
             self-contradictory. The reasoning is given inline.
  [DECISION] Our deliberate choice, recorded here so it can never be mistaken
             for something we are citing.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent

# The raw corpus is READ-ONLY. Nothing in this project may write beneath it.
# [DECISION] This project lives in its own directory, separate from the corpus, so a
#   bare relative path resolves against the wrong place. Candidates are tried in
#   order; set VSVIG_DATA_ROOT to override.
def _resolve_data_root() -> Path:
    candidates = [
        os.environ.get("VSVIG_DATA_ROOT"),
        "WU-SAHZU-EMU-Video/dataset",
        str(Path.home() / "Projects/VSViG/WU-SAHZU-EMU-Video/dataset"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    # Unresolved: return the conventional path so the failure surfaces at first use
    # with a real filename in the message, rather than as an obscure import error.
    return Path("WU-SAHZU-EMU-Video/dataset")


DATA_ROOT = _resolve_data_root()
LABEL_XLSX = DATA_ROOT / "Label.xlsx"

PROCESSED_DIR = Path("processed_data")
PATCHES_DIR = PROCESSED_DIR / "patches"
KPTS_DIR = PROCESSED_DIR / "kpts"
LABELS_JSON = PROCESSED_DIR / "labels.json"
FOLDS_DIR = PROCESSED_DIR / "folds"
POOLS_DIR = PROCESSED_DIR / "pools"

# Test-time clips live apart from training clips on purpose: they are extracted
# with a different stride and must never be mixed into a training manifest.
TEST_CLIPS_DIR = PROCESSED_DIR / "test_sliding"

OUTPUTS_DIR = Path("outputs")
BASELINE_CKPT_ROOT = OUTPUTS_DIR / "lopo" / "checkpoints"
HYPER_CKPT_ROOT = OUTPUTS_DIR / "lopo_hypernetwork" / "checkpoints"
RESULTS_DIR = OUTPUTS_DIR / "results"

POSE_WEIGHTS = Path("pose.pth")
DYNAMIC_PARTITION_FILE = Path("dy_point_order.pt")

# =============================================================================
# COHORT
# =============================================================================
# [PAPER] The corpus holds 14 patients and 33 annotated seizures.
ALL_PATIENTS = [f"pat{i:02d}" for i in range(1, 15)]

# [METHOD] Subjects with under one minute of pre-EEG baseline are excluded.
# [DECISION] "Under one minute" is measured as the SUM of `EEG onset` offsets across
#   a patient's annotated seizures -- total interictal video available before any
#   seizure begins. A patient below that threshold is RESCUED if separate seizure-free
#   footage supplies enough interictal video to build Pool A.
#
# Measured pre-EEG footage (seconds), all 14:
#   pat01 4204  pat02 3316  pat03   17  pat04   14  pat05   21  pat06  364
#   pat07  481  pat08 2020  pat09  388  pat10    8  pat11   61  pat12   27
#   pat13   37  pat14   32
#
# EXCLUSIONS, IN TWO ROUNDS -- the second one revises the methodology draft.
#
# Round 1 (pat05, pat10, pat12): fail the <60 s rule outright with no supplementary
#   footage. 21 s, 8 s and 27 s of pre-EEG video respectively.
#
# Round 2 (pat11, pat13, pat14): the methodology retains these to reach 11 patients /
#   24 seizures, but measurement showed they cannot support the method, and that
#   including them degrades every other patient. Three independent signals agreed:
#
#     a. Two of the three already fail the <60 s rule (pat13 37 s, pat14 32 s) and
#        have no supplementary footage; pat11 clears it by one second.
#     b. Non-overlapping interictal clips available: pat11 10, pat13 7, pat14 5.
#        pat14 cannot fill a Pool A of 6 AT ALL, and the largest Pool A the full
#        11-patient cohort could support is 5 -- against the methodology's N <= 20.
#     c. Evaluable test-time interictal exposure: pat11 50 s, pat13 35 s, pat14 25 s.
#        FDR/h is then quantised in steps of 72, 103 and 144 per hour, so a single
#        false alarm swings the metric further than any plausible treatment effect.
#
#   The decisive argument is (b), and it is about the OTHER patients: keeping these
#   three forces POOL_A_SIZE down to 5 for the whole cohort, estimating every
#   patient's mu and sigma from a quarter of the samples the methodology specifies,
#   in order to include three subjects whose FDR/h could not be measured anyway.
#   Dropping them makes the smallest interictal pool pat03's 42, so POOL_A_SIZE = 20
#   -- the methodology's own value -- becomes feasible for everyone, with at least
#   22 clips left for Pool B.
#
#   Cost, stated plainly: 8 patients and 18 seizures instead of 11 and 24, and the
#   section 3.5 hypothesis test runs at n = 8. This is a deliberate trade of cohort
#   size for signature quality and it must be reported as such, not presented as the
#   methodology's original cohort. See DECISIONS.md.
EXCLUDED_PATIENTS = ["pat05", "pat10", "pat12",   # round 1: <60 s pre-EEG baseline
                     "pat11", "pat13", "pat14"]   # round 2: cannot support Pool A
COHORT = [p for p in ALL_PATIENTS if p not in EXCLUDED_PATIENTS]  # 8 patients
assert len(COHORT) == 8, f"cohort must be 8 patients, got {len(COHORT)}"

# Seizure counts of the retained cohort: pat01 2, pat02 2, pat03 2, pat04 1, pat06 3,
# pat08 3, pat09 3, pat07 2  ->  18 seizures.
N_SEIZURES_EXPECTED = 18

# [PAPER] Table 6 semiology. P = partial (focal). PG = partial -> generalized
#         tonic-clonic. Used for stratified validation-pair selection.
SEMIOLOGY = {
    "pat01": "PG", "pat02": "PG", "pat03": "PG", "pat04": "P",  "pat05": "PG",
    "pat06": "P",  "pat07": "PG", "pat08": "P",  "pat09": "P",  "pat10": "P",
    "pat11": "P",  "pat12": "P",  "pat13": "PG", "pat14": "PG",
}

# =============================================================================
# CLIP GEOMETRY
# =============================================================================
CLIP_SECONDS = 5.0        # [PAPER] "a duration of 5 s"
CLIP_FRAMES = 30          # [PAPER] 150 raw frames subsampled to 30 -> 6 Hz
N_JOINTS = 15             # [PAPER] 18 OpenPose joints minus l_ear, r_ear, neck
PATCH_SIZE = 32           # [PAPER] "a size of 32x32 (HxW) for extracted patches"
FUSION_SIZE = 128         # [INFERRED] crop taken at 128x128 then resized to 32.
                          #   The paper gives no intermediate size; this is the
                          #   base repo's choice, preserved for continuity.
GAUSSIAN_SIGMA_SCALE = 0.3  # [PAPER] "sigma of gaussian kernel is 0.3", clarified
                            #   as 0.3 *relative to the patch size*.
GAUSSIAN_SIGMA = FUSION_SIZE * GAUSSIAN_SIGMA_SCALE

# [PAPER] OpenPose index order, regrouped into VSViG's 5 partitions of 3 joints:
#   head(nose,eyes) | right arm | left arm | right leg | left leg
JOINT_INDICES = [0, 14, 15, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# [DECISION] Frame rate is READ PER VIDEO, never assumed. Measured values in this
#   corpus are 29.97 for 37 files and 30.00 for 4 -- the 25.0 previously hard-coded
#   in the evaluator was wrong and silently corrupted every latency figure.
ASSUME_FPS = None

# =============================================================================
# KEYPOINT NORMALISATION
# =============================================================================
# [PAPER] Positional embedding is Stem(x_it, y_it) -- coordinates only, 2 channels.
# [METHOD] "raw skeleton coordinates are centered around the mid-hip joint and
#          normalized by torso length prior to any modeling."
#
# These disagree with the base repo, which divides x by 1920 and y by 1080 and
# does no centering at all. Frame-relative normalisation leaves the signature
# sensitive to where the patient lies in the bed and where the camera sits --
# exactly the nuisance variation the methodology's centering is meant to remove.
KPT_NORMALISATION = "midhip_torso"   # "midhip_torso" | "frame"  [METHOD]
FRAME_WIDTH = 1920                   # [PAPER] Bosch NDP-4502-Z12, 1920x1080
FRAME_HEIGHT = 1080
MIDHIP_JOINTS = (9, 12)              # indices into the 15-joint order: R hip, L hip
TORSO_JOINTS = (3, 6)                # R shoulder, L shoulder (mid-shoulder to mid-hip)
TORSO_LENGTH_FLOOR = 1e-3            # [DECISION] guard against a degenerate skeleton
# [DECISION] Fallbacks when a frame has no usable hips/shoulders. A pose failure must
#   degrade gracefully, never emit inf/NaN into the backbone.
TORSO_FALLBACK_FRAC = 0.25   # fallback torso length as a fraction of frame height
KPT_CLAMP = 5.0              # clamp normalised coords to +/- this many torso lengths
MISSING_JOINT_SENTINEL = -1.0  # preprocess writes -1 for an unresolved joint

# [DECISION] Stem_pe consumes 2 channels (x, y), matching the paper. The base repo
#   instantiates Stem_pe(input_dim=3) while its dataset yields 2 channels, which is
#   why 6fca412 crashes on its first forward pass. Confidence is a pose-detector
#   artifact, not a spatial coordinate, so it does not belong in a positional
#   embedding -- but it IS informative about imputed joints, so the 3-channel
#   variant stays available as a one-line ablation rather than being discarded.
KPT_CHANNELS = 2                     # 2 = (x,y) [PAPER] | 3 = (x,y,confidence)

# =============================================================================
# LABELLING
# =============================================================================
# [PAPER] Interictal 0, ictal 1, transition rising "in an exponential function".
# [DECISION] The paper never writes the function. k=5 and the (e^kx - 1)/(e^k - 1)
#   normalisation are this project's choice, inherited from the base repo.
TRANSITION_RAMP_K = 5.0
# [PAPER] "probabilities of video clips depend on the end frame of video clips
#          lying in which period" -- the label is evaluated at the clip's END.
LABEL_AT_CLIP_END = True

# =============================================================================
# EXTRACTION WINDOWS AND STRIDES
# =============================================================================
INTERICTAL_LOOKBACK_S = 1800.0   # [PAPER] "<30 min before EEG onset"
ICTAL_LOOKAHEAD_S = 120.0        # [PAPER] "<2 min after clinical onset"

# [METHOD] Extraction is bifurcated by phase. Training folds use overlapping
#   ictal/transition clips as augmentation; test evaluation forbids overlap.
TRAIN_STRIDE_ICTAL_S = 1.0        # [PAPER] "4 s overlappings" on a 5 s window
TRAIN_STRIDE_TRANSITION_S = 1.0
TRAIN_STRIDE_INTERICTAL_S = 5.0   # [PAPER] interictal extracted "without overlapping"
TEST_STRIDE_S = 5.0               # [METHOD] continuous sliding window, no overlap

# [METHOD] "Data Abundance Equalization via Algorithmic Expansion": supplementary
#   seizure-free footage is sliced into non-overlapping 5 s windows "sampled uniformly
#   across all available diurnal footage up to a strict cap of 40 clips per subject,
#   preventing high-volume patients from dominating the training gradients."
# [DECISION] UNIFORMLY across the whole file, not the first 40 windows. Taking a
#   contiguous prefix would sample ~200 s from the start of a 30-minute recording,
#   which is exactly the "isolated, transient activity" the methodology says to avoid.
EXTRA_FOOTAGE_CAP = 40
EXTRA_FOOTAGE_SAMPLING = "uniform"

# =============================================================================
# POOL A / POOL B
# =============================================================================
# [METHOD] Pool A holds N <= 20 interictal clips, stratified-uniformly sampled
#   across the patient's whole interictal timeline.
# [DECISION] N is FIXED AT 6 for every patient, not capped at 20.
#
#   Why: available interictal volume spans 6 to 548 non-overlapping clips across
#   the retained cohort (pat14 has 6, pat04 has 548). A variable N makes sigma far
#   noisier for scarce patients than for abundant ones, so D_p -- the distance from
#   the cohort centroid that the whole of section 3.5 correlates against -- would
#   partly measure estimation noise instead of behavioural atypicality. Equalising N
#   costs precision uniformly and buys comparability, which is what the hypothesis
#   test actually requires. 6 is the largest N every retained patient can supply.
POOL_A_SIZE = 20   # [METHOD] N <= 20, now feasible for every retained patient
POOL_A_SAMPLING = "stratified_uniform"   # [METHOD] NOT a sorted prefix; the base
                                         #   repo's interictal[:20] was a lexicographic
                                         #   artifact ("_1000" sorts before "_200").
POOL_A_MIN_CONFIDENCE = 20               # every retained patient meets this; a
                                         # patient below it would be flagged

# [METHOD] "A strict temporal guardrail guarantees that no overlapping windows
#          exist between Pool A and Pool B."
POOL_GUARDRAIL_S = CLIP_SECONDS          # windows within this of a Pool A clip are
                                         # excluded from Pool B, per source video

# =============================================================================
# SIGNATURE (z_behavior)
# =============================================================================
# [METHOD] Features come from "the early frozen blocks (Stages 0-2)".
# [INFERRED] The draft also states X in R^{15 x 30 x 384}, which exists nowhere in
#   the network: stage 2 emits 192 channels and T is downsampled 30->15->8->4.
#   Reading the constructor, the output after stages 0,1,2 is C'=192, T'=8, P=15.
#   Confirmed empirically by scripts/verify_shapes.py.
SIGNATURE_STAGE_CUT = 2          # inclusive, 0-indexed: stages 0,1,2
SIGNATURE_CHANNELS = 192         # C' -- verified, not assumed
CONTEXT_DIM = 128                # [METHOD] z_behavior in R^128
PROJECTOR_IN_DIM = 2 * SIGNATURE_CHANNELS   # [mu || sigma] = 384, NOT 768
PROJECTOR_SEED = 42              # [METHOD] untrained, frozen random projection

# [METHOD] sigma is the temporal standard deviation, taken BEFORE spatial pooling,
#   so it captures joint velocity rather than inter-clip posture drift.
# [DECISION] Across the N Pool A clips: compute mu and sigma per clip, then average
#   both across clips. Concatenating clips into one time axis would inject spurious
#   velocity spikes at the seams between clips drawn hours apart.
SIGMA_AXIS = "temporal_before_spatial_pool"
POOL_A_AGGREGATION = "mean_of_per_clip"
SIGMA_UNBIASED = False           # [METHOD] biased estimator, so N=1 gives 0 not NaN

# [METHOD] Signature stability gate, section 3.2.3.
STABILITY_RATIO_THRESHOLD = 2.5
STABILITY_N_BLOCKS = 4           # [DECISION] Pool A of 20 splits into 4 blocks of 5,
                                 # enough per block for a usable intra-patient variance

# =============================================================================
# HYPERNETWORK
# =============================================================================
HN_TRUNK_HIDDEN = 128            # [METHOD] Linear(128->128) x2
HN_TRUNK_ACTIVATION = "leaky_relu"
HN_LEAKY_SLOPE = 0.1             # [METHOD] LeakyReLU(alpha=0.1)
HN_BOTTLENECK = 32               # [METHOD] 32-d inner compression for A

# [METHOD] (name, out_channels, flattened_in, rank)
HN_TARGET_SPECS = [
    ("stage3_block0_conv2", 192, 1728, 4),
    ("stage3_block1_conv2", 192, 1728, 4),
    ("stage3_block2_conv2", 192, 1728, 4),
    ("fc0", 256, 384, 4),
    ("fc3", 1, 256, 1),
]
HN_USE_BASE = True               # [METHOD] A_p = A_base + A_hyper(z)
HN_Z_JITTER_SIGMA = 0.15         # [METHOD] training only
HN_DELTA_CLIP_RATIO = 0.5        # [METHOD] rho: ||dW||_F <= rho * ||W_base||_F

# [INFERRED] The draft writes N(0, d_in^-1 * 1e-2) for A_base. Read as a variance
#   that is 1e-2/d_in; read as a std it is 1e-4/d_in. The base repo implements the
#   latter. Kept, and flagged: a 100x difference either way.
HN_A_BASE_STD_SCALE = 1e-2

# =============================================================================
# STEP 1 -- BACKBONE OPTIMISATION
# =============================================================================
S1_LOSS = "huber"                # [METHOD] Huber for training
S1_HUBER_DELTA = 1.0
S1_SELECTION_METRIC = "mse"      # [DECISION] MSE for checkpoint selection, so the
                                 #   number stays comparable to the paper's RMSE
S1_OPTIMIZER = "adamw"
S1_LR = 1e-4                     # [PAPER] 1e-4
S1_WEIGHT_DECAY = 0.05
S1_BATCH_SIZE = 16
S1_MAX_EPOCHS = 50               # [DECISION] compute ceiling

# [DECISION] A single smooth cosine decay over the whole budget, NOT warm restarts.
#
#   The first pat01 run used CosineAnnealingWarmRestarts(T_0=10) with patience 5 and
#   validation every epoch. The LR annealed to 3.4e-06 by epoch 9, the model froze,
#   validation stopped improving after epoch 5, and patience expired at epoch 10 --
#   the exact epoch the first warm restart fired. The schedule's entire purpose is the
#   restarts, and early stopping guaranteed we never survived to see one. With T_0=10
#   and patience 5 that is systematic, not luck: it would have happened on all eight
#   folds.
#
#   The base repo hit the same wall and papered over it by loosening patience "because
#   CosineAnnealingWarmRestarts causes periodic val-loss bumps at each restart". The
#   cleaner fix is to drop the restarts: with one smooth decay, early stopping means
#   "stopped improving" rather than "the learning rate reached zero". The methodology
#   specifies early stopping for Step 1 and says nothing about restarts, so this is
#   also closer to the spec.
S1_SCHEDULER = "cosine"          # "cosine" | "cosine_warm_restarts"
S1_COSINE_ETA_MIN = 1e-6
S1_VAL_EVERY = 1                 # [DECISION] every epoch, so patience is meaningful
                                 #   within the 50-epoch cap
S1_PATIENCE = 5
S1_GRAD_CLIP = 1.0
S1_SAMPLER_FRACTIONS = {"interictal": 0.45, "ictal": 0.45, "transition": 0.10}

# =============================================================================
# STEP 3 -- HYPERNETWORK OPTIMISATION
# =============================================================================
S3_LOSS = "bce_with_logits"      # [METHOD] on un-sigmoided logits
S3_OPTIMIZER = "adamw"
S3_WARMUP_STEPS = 300            # [METHOD] linear 1e-5 -> 1e-3
S3_WARMUP_START_LR = 1e-5
S3_TARGET_LR = 1e-3
S3_MIN_LR = 1e-6                 # [METHOD] cosine decay floor
S3_WEIGHT_DECAY = 1e-5
S3_BATCH_SIZE = 8                # must be even: batches are exactly 50/50
S3_MAX_EPOCHS = 100              # [DECISION] compute ceiling
S3_PATIENCE = 5                  # [METHOD] 5 epochs without improvement
S3_GRAD_CLIP = 1.0
S3_EXCLUDE_TRANSITION = True     # [METHOD] transition clips excluded from step 3

# =============================================================================
# FOLD STRUCTURE
# =============================================================================
N_INTERNAL_VAL_PATIENTS = 2      # [METHOD] specifies 1 test / 2 val / 8 train.
                                 # [DECISION] With an 8-patient cohort this becomes
                                 # 1 test / 2 val / 5 train. Five conditioning points
                                 # per fold makes z-jitter (HN_Z_JITTER_SIGMA) load-
                                 # bearing rather than merely prudent: without it the
                                 # trunk can memorise five points outright.
INTERNAL_VAL_STRATIFY = "semiology"   # [METHOD] one P and one PG per fold
FOLD_SEED = 42

# =============================================================================
# EVALUATION
# =============================================================================
ACCUM_WINDOW_S = 3.0             # [PAPER] tau = 3 s
DECISION_THRESHOLD = 0.3         # [PAPER] DT = 0.3
# [DECISION] The paper writes AP_t = sum(P_i). Summing ~6 sigmoid outputs against a
#   threshold of 0.3 is near-trivially satisfied and cannot reproduce the reported
#   latencies, so we use the mean. Recorded as ours, not as the paper's method.
ACCUM_RULE = "mean"              # "mean" | "sum"

# [DECISION] A clip covers [t, t+5]. Its prediction is attributed to the clip's END,
#   because that is the earliest instant a real-time system could have emitted it.
#   The paper never says which edge it used, and the choice shifts every reported
#   latency by up to 5 s -- so it is recorded here rather than buried in the code.
DETECTION_TIME_REF = "clip_end"   # "clip_end" | "clip_start" | "clip_center"

# [METHOD] A seizure counts as detected if an alarm opens between EEG onset and the
#   end of the evaluated ictal window. Alarms before EEG onset are false alarms.
DETECTION_WINDOW_AFTER_CLINICAL_S = ICTAL_LOOKAHEAD_S

REFRACTORY_S = 60.0              # [METHOD] merge alarms within 60 s into one event
POST_ICTAL_EXCLUSION_S = 900.0   # [METHOD] 15 min after clinical onset excluded
                                 #   from the FDR/h denominator. Note: this corpus
                                 #   holds almost no post-ictal footage, so the rule
                                 #   rarely binds -- kept for correctness.

# [METHOD] Hours at risk exclude the pre-ictal transition and the ictal period.
FDR_EXCLUDE_TRANSITION = True
FDR_EXCLUDE_ICTAL = True

# [METHOD] section 3.5 is fitted as a Poisson rate model with a log-exposure
#   offset, not a Pearson correlation on raw ratios -- exposure spans three orders
#   of magnitude across this cohort (pat13 has 37 s at risk, pat04 has 45.6 min),
#   so a raw ratio would be dominated by recording length.
HYPOTHESIS_MODEL = "poisson_log_exposure"
HYPOTHESIS_ALPHA = 0.05
HYPOTHESIS_REPORT_ALSO = ["pearson", "spearman", "bootstrap_ci"]

# =============================================================================
# ABLATION MICRO-COHORT (pre-registered, hypothesis-independent)
# =============================================================================
# [DECISION] Three folds, chosen on seizure count and semiology ONLY -- never on
#   D_p, which depends on the signature we are still building and is the axis the
#   hypothesis is measured along. Fixed here BEFORE any z is computed.
#
#   Caveat: no PG patient in the retained cohort has more than 2 seizures, so the
#   high-count arm is necessarily focal.
ABLATION_FOLDS = ["pat09", "pat04", "pat02"]
#   pat09  P,  3 seizures, 76 interictal clips  -- high seizure count
#   pat04  P,  1 seizure,  82 interictal clips  -- low seizure count, focal
#   pat02  PG, 2 seizures, 456 interictal clips -- median count, generalised
#
# [DECISION] pat03 is the tightest remaining patient (42 interictal clips -> Pool A 20,
#   Pool B 22). Not in the ablation triple, but every pipeline change gets a smoke
#   check on it, because it is where Pool A/Pool B disjointness binds first.
SCARCITY_STRESS_FOLDS = ["pat03"]

# =============================================================================
# SEEDS
# =============================================================================
GLOBAL_SEED = 1337
