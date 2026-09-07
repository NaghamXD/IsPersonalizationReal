# Setup

## Environment

```bash
conda env create -f environment.yml
conda activate ispersonalizationreal
```

Conda supplies the **interpreter only**. Every library comes from PyPI via
`requirements.txt`, so there is one dependency list and one binary source. See
*Troubleshooting* below for why that matters.

Verify, in this order:

```bash
python scripts/check_openmp.py    # duplicate-runtime check; run this FIRST
python scripts/smoke_test.py      # config, unit tests, model shapes, one real clip
python scripts/capture_env.py     # writes environment_lock.{txt,yml}
```

`smoke_test.py` must reach **model shapes vs config: PASS**. That step runs
`verify_shapes.py`, which is what actually confirms `SIGNATURE_CHANNELS = 192` and
the `384 -> 128` projector against the real network rather than against a reading of
the constructor. If it fails, `config.py` is wrong and nothing downstream is
trustworthy.

## Data

The corpus is **read-only**. Nothing in this project writes beneath it.

`config.DATA_ROOT` resolves in order:
1. `$VSVIG_DATA_ROOT`
2. `./WU-SAHZU-EMU-Video/dataset`
3. `~/Projects/VSViG/WU-SAHZU-EMU-Video/dataset`

The third is the working default on this machine, so no symlink is needed.

## Pipeline

```bash
# Stage 3 -- test-time clips (non-overlapping sliding window)
python scripts/extract_test_clips.py --all --dry-run    # plan + cost, decodes nothing
python scripts/extract_test_clips.py --all              # ~21 min

# Evaluation
python scripts/evaluate.py --fold Pat01 --model baseline \
    --data-folder processed_data/test_sliding \
    --clips processed_data/test_sliding/manifest_pat01.json
```

Passing `--data-folder` matters. Without it the evaluator scores training-strided
clips, where ictal and transition windows overlap by 4 s, and it will warn you that
the resulting latency and FDR/h are optimistic.

## Reproducibility

- `config.py` is the single source of truth. Every constant is tagged
  `[PAPER]` / `[METHOD]` / `[INFERRED]` / `[DECISION]` so a citation is never
  confused with an inference or a choice of ours.
- `environment.yml` and `requirements.txt` build an environment;
  `environment_lock.{txt,yml}` record one. Only the latter belongs beside results.
- Every run writes a manifest with git commit, seed, device and a full config
  snapshot next to its outputs.

## Troubleshooting

### `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`

Two OpenMP runtimes have been loaded into one process. On macOS this happens when the
numeric stack (numpy, scipy, opencv) comes from conda-forge, which pulls in
`llvm-openmp`, while torch comes from pip and carries its own `libomp.dylib` inside
the wheel.

**Do not set `KMP_DUPLICATE_LIB_OK=TRUE`.** Its own documentation calls it unsafe,
unsupported and able to "silently produce incorrect results". A pipeline whose outputs
are probabilities feeding a clinical detection metric cannot absorb silently wrong
numerics -- you would have no way to notice it had happened.

Fix it by rebuilding so everything comes from one source:

```bash
conda deactivate
conda env remove -n ispersonalizationreal
conda env create -f environment.yml
conda activate ispersonalizationreal
python scripts/check_openmp.py
```

`check_openmp.py` imports numpy, scipy, cv2 and torch one at a time and prints the
OpenMP images loaded after each. If the process aborts, the last line printed names
the import that introduced the second copy.

### MPS reports unavailable

`capture_env.py` and `check_openmp.py` both report `mps_available` and `mps_built`.
If MPS is missing, training silently falls back to CPU and a full LOPO run becomes
impractically slow, so treat it as a blocker rather than a warning. Confirm you are on
an arm64 interpreter:

```bash
python -c "import platform; print(platform.machine())"   # expect arm64, not x86_64
```

An `x86_64` result means the env is running under Rosetta and will never see MPS.
