# Setup

## Environment

```bash
conda env create -f environment.yml
conda activate ispersonalizationreal
pip install "torch>=2.2"          # MPS build comes from PyPI on Apple Silicon
```

Torch is installed separately on purpose. If the MPS build fails it should fail
loudly and on its own, not disappear into a conda solve and leave you training on
CPU without noticing.

Verify, in this order:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
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
