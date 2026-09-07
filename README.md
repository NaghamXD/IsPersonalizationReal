# IsPersonalizationReal

Zero-shot, patient-conditioned weight modulation for video-based epileptic seizure
detection.

An amortized hypernetwork generates LoRA weight deltas for a frozen VSViG backbone,
conditioned on a **behavioural signature computed from a patient's unlabeled interictal
video alone**. The claim under test is that this personalizes the detector for a patient
the system has never seen labeled data from — and, specifically, that the benefit is
largest for the patients who move least like the cohort.

## Status

| stage | | |
|---|---|---|
| 1 | Reproducibility floor | done |
| 2 | Evaluation harness | done |
| 3 | Test-time sliding-window extraction | done |
| 4 | LOPO folds and Pool A / Pool B | done |
| 5 | Signature (`z_behavior`) + §3.2.3 stability gate | next |
| 6 | Per-fold backbones | |
| 7 | Hypernetwork training and adapted-model evaluation | |
| 8 | Controls (shuffled-z, cohort LoRA, oracle per-patient LoRA) | |
| 9 | §3.5 hypothesis test | |
| 10–12 | Ablations, robustness, write-up assets | |

## Provenance

Built from `NaghamXD/VSViG_` at commit `6fca412`, re-initialised as an independent
repository with no remote. That commit predates all hypernetwork work; later branches
of the original are reference material only. See `DECISIONS.md` D1.

The backbone is a port of VSViG (Xu et al., ECCV 2024). The corpus is the accompanying
WU-SAHZU-EMU-Video release, held **read-only** outside this project — nothing here
writes beneath it.

## Read these first

- **`config.py`** — the single source of truth. Every constant is tagged
  `[PAPER]` / `[METHOD]` / `[INFERRED]` / `[DECISION]`, so a citation is never mistaken
  for an inference or a choice of ours.
- **`DECISIONS.md`** — D1–D14, each with its reasoning and evidence, plus open items.
- **`SETUP.md`** — environment, data resolution, pipeline commands, troubleshooting.

## Cohort

8 patients, 18 seizures: `pat01 pat02 pat03 pat04 pat06 pat07 pat08 pat09` — four
focal, four focal-to-bilateral tonic-clonic, so every fold can draw a
semiology-stratified validation pair.

Six of the original 14 are excluded for insufficient interictal baseline. This departs
from the methodology draft's 11 patients / 24 seizures and is a deliberate trade of
cohort size for signature quality; the reasoning and its cost are set out in
`DECISIONS.md` D2 and must be reported as such rather than presented as the draft's
original cohort.

## Pipeline

```bash
python scripts/smoke_test.py            # run this first on any new machine
python scripts/preprocess.py --all      # training clips      (~39 min)
python scripts/extract_test_clips.py --all   # test clips     (~28 min)
python scripts/build_splits.py          # folds + Pool A / Pool B
```

Every stage takes `--dry-run` to print its plan and cost without decoding video, and
every stage is idempotent, so an interrupted run resumes.

## Testing

`python scripts/smoke_test.py` runs the whole check suite: config self-consistency,
that every module imports, the torch-free unit tests, the model shapes against the real
network, and one real clip through the dataset.

The unit tests deliberately avoid torch where the logic allows, so the parts that were
previously wrong — clip timing, the decision rule, the sampling plan, pool
construction — can be verified anywhere, without a GPU or a checkpoint.
