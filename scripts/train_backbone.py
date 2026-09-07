"""Stage 6 -- Step 1: the per-fold VSViG backbone.

[METHOD] "Step 1: Train base VSViG backbone (Huber Loss, Multi-Patient Shuffled
Batches) -> Early Stop via Stratified Val Set. Step 2: Freeze Base Backbone +
Batch Normalization Buffers."

This is the model the hypernetwork will be frozen on top of, and the unadapted
population baseline every later comparison is measured against. It has to be clean.

WHAT THE 6fca412 TRAINER DID WRONG, and what is done here instead:

  * It trained on `train_{patient}.json` and selected the checkpoint on
    `val_{patient}.json` -- which held the HELD-OUT patient's own clips. The model was
    selected on the patient it was supposed to generalise to. Here, training and
    selection both come from the fold's own 5 + 2 patients, and `test_clips.json` is
    never opened. There is an explicit assertion to that effect, because "we are
    careful" is not a guarantee.
  * MSE for training. [METHOD] specifies Huber, which is less dominated by the
    handful of clips the model gets badly wrong early on.
  * No class rebalancing, on pools that run roughly 45% interictal / 47% ictal.

    python scripts/train_backbone.py --fold pat01
    python scripts/train_backbone.py --all-folds
    python scripts/train_backbone.py --fold pat01 --verify-only    # one batch
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
from src.data.dataset import VSViGDataset
from src.model.vsvig import VSViG_base
from src.utils.manifest import write_manifest
from src.utils.naming import patient_of
from src.utils.seeding import fold_seed, seed_everything


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def class_of(label: float) -> str:
    return "interictal" if label == 0.0 else "ictal" if label == 1.0 else "transition"


def sample_weights(dataset):
    """Per-sample weights giving the epoch the composition config asks for.

    [DECISION] 45 / 45 / 10. Interictal and ictal are pulled to parity because the raw
    pools sit near 45/47 and the imbalance shifts fold to fold. Transition clips get a
    smaller fixed share deliberately: they carry soft ramp labels, so upweighting them
    to parity would amplify the noisiest targets in the set and trade one bias for
    another.
    """
    labels = [float(l) for _, l in dataset._labels]
    counts = {k: sum(1 for l in labels if class_of(l) == k)
              for k in ("interictal", "ictal", "transition")}
    per_class = {k: (config.S1_SAMPLER_FRACTIONS[k] / counts[k] if counts[k] else 0.0)
                 for k in counts}
    return [per_class[class_of(l)] for l in labels], counts


def load_fold(patient):
    d = Path(config.FOLDS_DIR) / patient
    fold = json.loads((d / "fold.json").read_text())
    train_file, val_file = d / "train_clips.json", d / "val_clips.json"
    if not train_file.exists():
        raise FileNotFoundError(f"{train_file} missing -- run scripts/build_splits.py")

    # Data-separation guardrail. The failure this prevents is silent: a leaked fold
    # trains and converges perfectly well, it just reports a number that means nothing.
    for f in (train_file, val_file):
        pats = {patient_of(n) for n, _ in json.loads(f.read_text())}
        assert patient not in pats, f"LEAK: held-out {patient} appears in {f.name}"
    return fold, train_file, val_file


@torch.no_grad()
def evaluate(model, loader, device):
    """Pooled MSE on the validation patients.

    [DECISION] MSE for selection even though training uses Huber, so the number stays
    comparable to the paper's reported RMSE. Recorded in config as S1_SELECTION_METRIC.
    """
    model.eval()
    total, n = 0.0, 0
    for sample, labels in loader:
        out = model(sample["data"].to(device), sample["kpts"].to(device))
        if out.dim() > 1:
            out = out.squeeze(1)
        labels = labels.float().to(device)
        total += torch.nn.functional.mse_loss(out, labels, reduction="sum").item()
        n += labels.numel()
    model.train()
    return total / max(n, 1)


def train_fold(patient, args, device):
    fold, train_file, val_file = load_fold(patient)
    idx = sorted(config.COHORT).index(patient)
    seed = fold_seed(config.GLOBAL_SEED, idx)
    seed_everything(seed)

    ckpt_dir = Path(config.BASELINE_CKPT_ROOT) / patient
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path_best = ckpt_dir / "best_model.pth"
    path_last = ckpt_dir / "last_checkpoint.pth"
    path_log = ckpt_dir / "training_log.json"

    train_ds = VSViGDataset(config.PROCESSED_DIR, train_file)
    val_ds = VSViGDataset(config.PROCESSED_DIR, val_file)
    weights, counts = sample_weights(train_ds)

    print(f"\n=== fold {patient} "
          f"(val={fold['val_patients']}, train={fold['train_patients']}) ===")
    print(f"  train {len(train_ds)} clips  "
          f"{counts}  -> target "
          f"{'/'.join(f'{int(v*100)}%' for v in config.S1_SAMPLER_FRACTIONS.values())}")
    print(f"  val   {len(val_ds)} clips   seed={seed}")

    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    # drop_last: BatchNorm in training mode needs more than one sample, and a dataset
    # size leaving exactly one leftover would crash on the last batch of every epoch.
    train_loader = DataLoader(train_ds, batch_size=config.S1_BATCH_SIZE,
                              sampler=sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.S1_BATCH_SIZE,
                            shuffle=False, num_workers=0)

    model = VSViG_base(kpt_channels=config.KPT_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.S1_LR,
                                  weight_decay=config.S1_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = nn.HuberLoss(delta=config.S1_HUBER_DELTA)

    start_epoch, best_val, trigger = 0, float("inf"), 0
    history = {"train_loss": [], "val_mse": [], "lr": [], "epoch_s": []}
    if path_last.exists() and not args.restart:
        ck = torch.load(path_last, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val, trigger = ck["best_val"], ck.get("trigger", 0)
        history = ck.get("history", history)
        print(f"  resuming at epoch {start_epoch} (best val MSE {best_val:.5f})")

    if args.verify_only:
        sample, labels = next(iter(train_loader))
        out = model(sample["data"].to(device), sample["kpts"].to(device))
        if out.dim() > 1:
            out = out.squeeze(1)
        loss = criterion(out, labels.float().to(device))
        print(f"  [verify] data={tuple(sample['data'].shape)} "
              f"kpts={tuple(sample['kpts'].shape)} out={tuple(out.shape)} "
              f"range=[{out.min():.3f},{out.max():.3f}] huber={loss.item():.4f}")
        return

    model.train()
    for epoch in range(start_epoch, config.S1_MAX_EPOCHS):
        t0, running, nb = time.time(), 0.0, 0
        for bi, (sample, labels) in enumerate(train_loader):
            if args.limit_batches and bi >= args.limit_batches:
                break
            out = model(sample["data"].to(device), sample["kpts"].to(device))
            if out.dim() > 1:
                out = out.squeeze(1)
            loss = criterion(out, labels.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.S1_GRAD_CLIP)
            optimizer.step()
            running += loss.item()
            nb += 1
        scheduler.step()

        train_loss = running / max(nb, 1)
        val_mse = evaluate(model, val_loader, device)
        dt = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_s"].append(round(dt, 1))

        improved = val_mse < best_val
        if improved:
            best_val, trigger = val_mse, 0
            torch.save(model.state_dict(), path_best)
        else:
            trigger += 1
        print(f"  epoch {epoch+1:>3}/{config.S1_MAX_EPOCHS}  "
              f"huber={train_loss:.5f}  val_mse={val_mse:.5f} "
              f"(rmse={100*val_mse**0.5:.2f}%)  lr={history['lr'][-1]:.2e}  "
              f"{dt:.0f}s  {'** best' if improved else f'no improve {trigger}/{config.S1_PATIENCE}'}")

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val": best_val, "trigger": trigger, "history": history},
                   path_last)
        path_log.write_text(json.dumps(history, indent=2))

        if trigger >= config.S1_PATIENCE:
            print(f"  early stop at epoch {epoch+1} "
                  f"({config.S1_PATIENCE} epochs without improvement)")
            break

    write_manifest(ckpt_dir / "run_manifest.json", seed=seed,
                   extra={"stage": "6_backbone", "fold": patient,
                          "val_patients": fold["val_patients"],
                          "train_patients": fold["train_patients"],
                          "best_val_mse": best_val,
                          "epochs_run": len(history["val_mse"])})
    print(f"  best val MSE {best_val:.5f} (RMSE {100*best_val**0.5:.2f}%) -> {path_best}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=str, default=None)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="one forward pass, to check shapes and timing before committing")
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="cap batches per epoch (timing probe only, not a real run)")
    ap.add_argument("--restart", action="store_true",
                    help="ignore an existing checkpoint and start over")
    args = ap.parse_args()
    if not args.fold and not args.all_folds:
        ap.error("pass --fold patNN or --all-folds")

    device = get_device()
    print(f"[env] device={device}  loss={config.S1_LOSS}(delta={config.S1_HUBER_DELTA})  "
          f"select_on={config.S1_SELECTION_METRIC}  batch={config.S1_BATCH_SIZE}  "
          f"max_epochs={config.S1_MAX_EPOCHS}  patience={config.S1_PATIENCE}")
    if device.type == "cpu":
        print("[warn] running on CPU -- a full 8-fold run will be impractically slow.")

    folds = sorted(config.COHORT) if args.all_folds else [args.fold.lower()]
    t0 = time.time()
    for p in folds:
        if p not in config.COHORT:
            print(f"[skip] {p} not in cohort")
            continue
        train_fold(p, args, device)
    print(f"\ntotal {time.time()-t0:.0f}s")
    if not args.verify_only:
        print("Next: python scripts/build_signatures.py --all-folds   "
              "(z_behavior + the section 3.2.3 stability gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
