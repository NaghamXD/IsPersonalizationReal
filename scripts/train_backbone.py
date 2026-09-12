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
from src.eval.metrics import auc_from_labels
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
def evaluate_batch_stats(model, loader, device):
    """Same data, but BatchNorm uses BATCH statistics instead of running ones.

    Comparing this against evaluate() isolates a BatchNorm train/eval gap from a
    genuine failure to fit. They answer different questions and need opposite fixes:
    a large gap means the running statistics have not converged (train longer, or more
    updates per epoch); no gap means the model really cannot fit the data.

    Running-stat buffers are saved and restored, so this measurement cannot itself
    perturb the statistics it is measuring.
    """
    bns = [m for m in model.modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    saved = [(m.running_mean.clone(), m.running_var.clone(),
              int(m.num_batches_tracked.item())) for m in bns]
    was_training = model.training
    model.train()
    total, n = 0.0, 0
    for sample, labels in loader:
        out = model(sample["data"].to(device), sample["kpts"].to(device))
        if out.dim() > 1:
            out = out.squeeze(1)
        labels = labels.float().to(device)
        total += torch.nn.functional.mse_loss(out, labels, reduction="sum").item()
        n += labels.numel()
    for m, (rm, rv, nb) in zip(bns, saved):
        m.running_mean.copy_(rm)
        m.running_var.copy_(rv)
        m.num_batches_tracked.fill_(nb)
    model.train(was_training)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """MSE and AUC on the validation patients. Returns (mse, auc).

    [DECISION] Selection is on AUC, with MSE reported alongside.

    MSE is dominated by the label distribution; AUC measures whether ictal clips are
    RANKED above interictal ones, which is what the accumulation-and-threshold decision
    rule actually consumes. The two came apart badly in practice: the checkpoint chosen
    by best validation MSE scored AUC 0.513 -- chance. Selecting on MSE was picking
    models that could not discriminate.

    MSE stays in the log so the number remains comparable to the paper's RMSE.
    """
    model.eval()
    total, n = 0.0, 0
    all_s, all_y = [], []
    for sample, labels in loader:
        out = model(sample["data"].to(device), sample["kpts"].to(device))
        if out.dim() > 1:
            out = out.squeeze(1)
        labels = labels.float().to(device)
        total += torch.nn.functional.mse_loss(out, labels, reduction="sum").item()
        n += labels.numel()
        all_s.append(out.float().cpu())
        all_y.append(labels.float().cpu())
    model.train()
    mse = total / max(n, 1)
    auc = auc_from_labels(torch.cat(all_s).numpy(), torch.cat(all_y).numpy())
    return mse, auc


def train_fold(patient, args, device):
    fold, train_file, val_file = load_fold(patient)
    idx = sorted(config.COHORT).index(patient)
    seed = fold_seed(config.GLOBAL_SEED, idx)
    seed_everything(seed)

    ckpt_dir = Path(config.BASELINE_CKPT_ROOT) / (
        f"{patient}_overfit" if args.overfit else patient)
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

    if args.overfit:
        # Deliberately no rebalancing, no shuffling, no held-out set: the only
        # question is whether the model can memorise a handful of examples. If it
        # cannot, nothing about the optimiser is worth tuning.
        from torch.utils.data import Subset
        keep = list(range(min(args.overfit, len(train_ds))))
        train_ds = Subset(train_ds, keep)
        val_ds = train_ds
        train_loader = DataLoader(train_ds, batch_size=min(config.S1_BATCH_SIZE,
                                                           len(keep)),
                                  shuffle=True, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=min(config.S1_BATCH_SIZE, len(keep)),
                                shuffle=False, num_workers=0)
        print(f"  OVERFIT CHECK on {len(keep)} clips (train == val). "
              f"Expect the loss to approach 0; if it plateaus, the pipeline cannot "
              f"learn and hyperparameters are not the issue.")

    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    # drop_last: BatchNorm in training mode needs more than one sample, and a dataset
    # size leaving exactly one leftover would crash on the last batch of every epoch.
    if not args.overfit:
        nw = config.S1_NUM_WORKERS
        extra = dict(persistent_workers=True, prefetch_factor=4) if nw > 0 else {}
        train_loader = DataLoader(train_ds, batch_size=config.S1_BATCH_SIZE,
                                  sampler=sampler, num_workers=nw, drop_last=True,
                                  **extra)
        val_loader = DataLoader(val_ds, batch_size=config.S1_BATCH_SIZE,
                                shuffle=False, num_workers=nw, **extra)

    model = VSViG_base(kpt_channels=config.KPT_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.S1_LR,
                                  weight_decay=config.S1_WEIGHT_DECAY)
    if config.S1_SCHEDULER == "cosine":
        # One smooth decay across the whole budget, so the patience counter measures
        # "stopped improving" rather than "the learning rate reached zero". See the
        # note in config.py: with warm restarts at T_0=10 and patience 5, every fold
        # would early-stop on precisely the epoch the first restart fired.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.S1_MAX_EPOCHS, eta_min=config.S1_COSINE_ETA_MIN)
    elif config.S1_SCHEDULER == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=config.S1_COSINE_ETA_MIN)
    else:
        raise ValueError(f"unknown S1_SCHEDULER {config.S1_SCHEDULER!r}")
    criterion = nn.HuberLoss(delta=config.S1_HUBER_DELTA)

    # Selection is on AUC, where HIGHER is better -- hence -inf, not +inf.
    start_epoch, best_val, trigger = 0, float("-inf"), 0
    history = {"train_loss": [], "val_mse": [], "val_auc": [], "lr": [], "epoch_s": []}
    if path_last.exists() and not args.restart:
        ck = torch.load(path_last, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val, trigger = ck["best_val"], ck.get("trigger", 0)
        history = ck.get("history", history)
        print(f"  resuming at epoch {start_epoch} (best val AUC {best_val:.4f})")

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
    max_epochs = args.overfit_epochs if args.overfit else config.S1_MAX_EPOCHS
    for epoch in range(start_epoch, max_epochs):
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
        val_mse, val_auc = evaluate(model, val_loader, device)
        val_bn = evaluate_batch_stats(model, val_loader, device) if args.overfit else None
        dt = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_mse)
        history["val_auc"].append(val_auc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_s"].append(round(dt, 1))

        # AUC: higher is better. nan (a validation split with only one class)
        # never counts as an improvement.
        improved = (val_auc == val_auc) and val_auc > best_val
        if improved:
            best_val, trigger = val_auc, 0
            torch.save(model.state_dict(), path_best)
        else:
            trigger += 1
        if args.overfit:
            if (epoch + 1) % 20 == 0 or epoch < 3:
                print(f"  epoch {epoch+1:>4}/{max_epochs}  huber={train_loss:.5f}  "
                      f"mse[running_stats]={val_mse:.5f}  mse[batch_stats]={val_bn:.5f}"
                      f"  gap={val_mse - val_bn:+.5f}")
        else:
            print(f"  epoch {epoch+1:>4}/{max_epochs}  "
                  f"huber={train_loss:.5f}  val_auc={val_auc:.4f}  "
                  f"val_mse={val_mse:.5f} (rmse={100*val_mse**0.5:.2f}%)  "
                  f"lr={history['lr'][-1]:.2e}  {dt:.0f}s  "
                  f"{'** best' if improved else f'no improve {trigger}/{config.S1_PATIENCE}'}")

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val": best_val, "trigger": trigger, "history": history},
                   path_last)
        path_log.write_text(json.dumps(history, indent=2))

        if args.overfit:
            continue
        if trigger >= config.S1_PATIENCE:
            print(f"  early stop at epoch {epoch+1} "
                  f"({config.S1_PATIENCE} epochs without improvement)")
            break

    if args.overfit:
        final_run = evaluate(model, val_loader, device)
        final_bn = evaluate_batch_stats(model, val_loader, device)
        n_steps = len(history["val_mse"]) * max(nb, 1)
        print(f"\n  --- overfit verdict ({args.overfit} clips, ~{n_steps} gradient steps) ---")
        print(f"  final MSE, running stats : {final_run:.5f}  (RMSE {100*final_run**0.5:.2f}%)")
        print(f"  final MSE, batch stats   : {final_bn:.5f}  (RMSE {100*final_bn**0.5:.2f}%)")
        if final_bn < 0.01:
            verdict = ("CAN fit. The model memorises what it is shown, so gradients, "
                       "data and architecture are sound.")
        elif final_bn < 0.05:
            verdict = "PARTIALLY fits -- capacity reaches the data but slowly."
        else:
            verdict = ("CANNOT fit even a handful of clips. The fault is structural "
                       "(data, gradient path or architecture), not hyperparameters.")
        print(f"  verdict: {verdict}")
        if final_run - final_bn > 0.02:
            print(f"  NOTE: a {final_run - final_bn:.3f} gap between the two means "
                  f"BatchNorm running statistics have not converged. With {args.overfit} "
                  f"clips there are only a couple of updates per epoch; this gap is an "
                  f"artefact of the probe, not of the real training runs.")

    write_manifest(ckpt_dir / "run_manifest.json", seed=seed,
                   extra={"stage": "6_backbone", "fold": patient,
                          "val_patients": fold["val_patients"],
                          "train_patients": fold["train_patients"],
                          "best_val_auc": best_val,
                          "best_val_mse": min(history["val_mse"]) if history["val_mse"] else None,
                          "epochs_run": len(history["val_mse"])})
    if history["val_auc"]:
        i = int(max(range(len(history["val_auc"])),
                    key=lambda k: (history["val_auc"][k] == history["val_auc"][k],
                                   history["val_auc"][k])))
        print(f"  best val AUC {history['val_auc'][i]:.4f} at epoch {i+1} "
              f"(its MSE {history['val_mse'][i]:.5f}, "
              f"RMSE {100*history['val_mse'][i]**0.5:.2f}%) -> {path_best}")


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
    ap.add_argument("--overfit-epochs", type=int, default=400,
                    help="epoch budget for --overfit. 50 epochs on 32 clips is only "
                         "100 gradient steps, which tests nothing.")
    ap.add_argument("--overfit", type=int, default=None, metavar="N",
                    help="SANITY CHECK: train on N fixed clips and evaluate on those "
                         "same N. A working pipeline drives this loss to ~0. If it "
                         "cannot, the problem is data, gradients or architecture -- "
                         "not the learning rate.")
    args = ap.parse_args()
    if not args.fold and not args.all_folds:
        ap.error("pass --fold patNN or --all-folds")

    device = get_device()
    print(f"[env] device={device}  loss={config.S1_LOSS}(delta={config.S1_HUBER_DELTA})  "
          f"select_on=auc  batch={config.S1_BATCH_SIZE}  workers={config.S1_NUM_WORKERS}  "
          f"sched={config.S1_SCHEDULER}  max_epochs={config.S1_MAX_EPOCHS}  "
          f"patience={config.S1_PATIENCE}")
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
