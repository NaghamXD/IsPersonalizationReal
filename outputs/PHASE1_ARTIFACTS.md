# Phase 1 artifacts — frozen inventory

Frozen 2026-09-14 at commit `03c26481`. 268 files, 2.50 GB.

Verify at any time with `python scripts/freeze_artifacts.py --verify`. Every md5 is recorded in `PHASE1_ARTIFACTS.json`.

## The files the report depends on

| file | size | md5 (first 12) | what it is |
|---|---|---|---|
| `outputs/lopo/checkpoints/pat01/final_model.pth` | 22.6 MB | `64c1ad3121fe` | THE Phase 1 baseline. |
| `outputs/lopo/checkpoints/pat02/final_model.pth` | 22.6 MB | `3013f4bef4cd` | THE Phase 1 baseline. |
| `outputs/lopo/checkpoints/pat03/final_model.pth` | 22.6 MB | `6b66d81b7127` | THE Phase 1 baseline. |
| `outputs/lopo/checkpoints/pat04/final_model.pth` | 22.6 MB | `cdd01d8e5619` | THE Phase 1 baseline. |
| `outputs/lopo/checkpoints/pat06/final_model.pth` | 22.6 MB | `8387e465d2c0` | THE Phase 1 baseline. |
| `outputs/lopo/checkpoints/pat07/final_model.pth` | 22.6 MB | `9af1b606e84a` | THE Phase 1 baseline. |
| `outputs/lopo/checkpoints/pat08/final_model.pth` | 22.6 MB | `fd968a75bcbf` | THE Phase 1 baseline. |
| `outputs/lopo/checkpoints/pat09/final_model.pth` | 22.6 MB | `11bfdd4941cb` | THE Phase 1 baseline. |
| `outputs/lopo_hypernetwork/checkpoints/pat01/hypernetwork_best.pth` | 3.0 MB | `090e88934877` | Stage 7 hypernetwork, lowest validation BCE. |
| `outputs/lopo_hypernetwork/checkpoints/pat01_gentle/hypernetwork_best.pth` | 3.0 MB | `6019d92cf64f` | Stage 7 hypernetwork, lowest validation BCE. |
| `outputs/lopo_hypernetwork/checkpoints/pat02/hypernetwork_best.pth` | 3.0 MB | `f20486d6ff72` | Stage 7 hypernetwork, lowest validation BCE. |
| `outputs/lopo_hypernetwork/checkpoints/pat03/hypernetwork_best.pth` | 3.0 MB | `e3dd82d7c1d0` | Stage 7 hypernetwork, lowest validation BCE. |
| `outputs/lopo_hypernetwork/checkpoints/pat04/hypernetwork_best.pth` | 3.0 MB | `3efa05d728cf` | Stage 7 hypernetwork, lowest validation BCE. |
| `outputs/lopo_hypernetwork/checkpoints/pat06/hypernetwork_best.pth` | 3.0 MB | `f0bd88241691` | Stage 7 hypernetwork, lowest validation BCE. |
| `outputs/lopo_hypernetwork/checkpoints/pat07/hypernetwork_best.pth` | 3.0 MB | `b0ead14542dd` | Stage 7 hypernetwork, lowest validation BCE. |
| `outputs/lopo_hypernetwork/checkpoints/pat08/hypernetwork_best.pth` | 3.0 MB | `cc8ffab883a9` | Stage 7 hypernetwork, lowest validation BCE. |
| `outputs/lopo_hypernetwork/checkpoints/pat09/hypernetwork_best.pth` | 3.0 MB | `80a5bacc1077` | Stage 7 hypernetwork, lowest validation BCE. |

## Role of every weight file kept

| filename | role |
|---|---|
| `final_model.pth` | THE Phase 1 baseline. D21 average of the last 5 epochs with BatchNorm recalibrated. Every baseline number in the report comes from this file. |
| `best_model.pth` | Byte-identical copy of final_model.pth, kept because evaluate.py's older resolution order looks for this name. |
| `best_by_val.pth` | The single epoch with the highest validation AUC. Kept for comparison with the pre-D21 protocol only. NOT evaluated anywhere -- D20 showed validation does not predict held-out performance. |
| `last_checkpoint.pth` | Full training state at epoch 50 (model, optimiser, scheduler, history). Needed to resume, not to evaluate. |
| `hypernetwork_best.pth` | Stage 7 hypernetwork, lowest validation BCE. The adapted arm of the report. |
| `hypernetwork_last.pth` | Stage 7 hypernetwork at the final epoch. |
| `epoch_0N0.pth` | Periodic weights every 10 epochs (D21 insurance). |

## Superseded and quarantined

Runs that `--restart` archived rather than overwrote, plus smoke-test artefacts that were deliberately quarantined so `--skip-done` could not mistake them for real runs:

- `outputs/lopo/checkpoints/pat01/archive/20260913-021221/best_model.pth` — Superseded run, archived automatically by --restart rather than overwritten. Kept for provenance; not used in the report.
- `outputs/lopo/checkpoints/pat01/archive/20260913-021221/last_checkpoint.pth` — Superseded run, archived automatically by --restart rather than overwritten. Kept for provenance; not used in the report.
- `outputs/lopo/checkpoints/pat01/archive/20260913-021221/training_log.json` — Superseded run, archived automatically by --restart rather than overwritten. Kept for provenance; not used in the report.
- `outputs/lopo/checkpoints/pat01/archive/20260913-021221/val_scores_by_epoch.npz` — Superseded run, archived automatically by --restart rather than overwritten. Kept for provenance; not used in the report.
- `outputs/lopo/checkpoints/pat01/archive/smoke-run-do-not-use/best_model.pth` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
- `outputs/lopo/checkpoints/pat01/archive/smoke-run-do-not-use/last_checkpoint.pth` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
- `outputs/lopo/checkpoints/pat01/archive/smoke-run-do-not-use/training_log.json` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
- `outputs/lopo/checkpoints/pat01/archive/smoke-run-do-not-use/val_scores_by_epoch.npz` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
- `outputs/lopo/checkpoints/pat08/archive/20260912-230622/best_model.pth` — Superseded run, archived automatically by --restart rather than overwritten. Kept for provenance; not used in the report.
- `outputs/lopo/checkpoints/pat08/archive/20260912-230622/last_checkpoint.pth` — Superseded run, archived automatically by --restart rather than overwritten. Kept for provenance; not used in the report.
- `outputs/lopo/checkpoints/pat08/archive/20260912-230622/training_log.json` — Superseded run, archived automatically by --restart rather than overwritten. Kept for provenance; not used in the report.
- `outputs/lopo/checkpoints/pat08/archive/20260912-230622/val_scores_by_epoch.npz` — Superseded run, archived automatically by --restart rather than overwritten. Kept for provenance; not used in the report.
- `outputs/lopo/checkpoints/pat08/archive/smoke-run-do-not-use/best_by_val.pth` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
- `outputs/lopo/checkpoints/pat08/archive/smoke-run-do-not-use/best_model.pth` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
- `outputs/lopo/checkpoints/pat08/archive/smoke-run-do-not-use/epoch_002.pth` — Periodic weights at epoch 2. D21 insurance: lets a later change of mind about the training budget be tested without retraining.
- `outputs/lopo/checkpoints/pat08/archive/smoke-run-do-not-use/final_model.pth` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
- `outputs/lopo/checkpoints/pat08/archive/smoke-run-do-not-use/last_checkpoint.pth` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
- `outputs/lopo/checkpoints/pat08/archive/smoke-run-do-not-use/training_log.json` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
- `outputs/lopo/checkpoints/pat08/archive/smoke-run-do-not-use/val_scores_by_epoch.npz` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
- `outputs/lopo_hypernetwork/checkpoints/pat02/archive/smoke-run-do-not-use/hypernetwork_best.pth` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
- `outputs/lopo_hypernetwork/checkpoints/pat02/archive/smoke-run-do-not-use/hypernetwork_last.pth` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
- `outputs/lopo_hypernetwork/checkpoints/pat02/archive/smoke-run-do-not-use/run_manifest.json` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
- `outputs/lopo_hypernetwork/checkpoints/pat02/archive/smoke-run-do-not-use/training_log.json` — ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately.
