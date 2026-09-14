"""Stage 7: train the amortised hypernetwork on a frozen backbone.

Methodology step 2-3: freeze the base backbone AND its BatchNorm buffers, then train
the hypernetwork with BCEWithLogitsLoss on patient-homogeneous batches drawn by the
Max-Pool Dynamic Cyclic Sampler (section 3.4.1), AdamW with a 300-step linear warmup
1e-5 -> 1e-3 followed by cosine decay to 1e-6 (section 3.4.2).

Two invariants this script exists to protect:

  * dW = 0 at step 0 (B is zero-initialised), so the adapted arm STARTS as the
    baseline. Checked explicitly before the first update.
  * The held-out patient's z is never used in training, and its clips never appear.
    Asserted, not assumed.

    python scripts/train_hypernetwork.py --fold pat01
"""
import argparse, json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from src.data.dataset import VSViGDataset
from src.data.sampler import CyclicBatchSampler
from src.model.adapt import AdaptedForward, base_norms, resolve_targets
from src.model.hypernetwork import Hypernetwork
from src.model.vsvig import VSViG_base
from src.utils.manifest import write_manifest
from src.utils.naming import patient_of
from src.utils.seeding import seed_everything, fold_seed


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def frozen_backbone(fold, device):
    root = Path(config.BASELINE_CKPT_ROOT) / fold
    ck = next((c for c in (root / "final_model.pth", root / "best_model.pth")
               if c.exists()), None)
    if ck is None:
        raise FileNotFoundError(f"no Stage 6 backbone for {fold} under {root}")
    m = VSViG_base(kpt_channels=config.KPT_CHANNELS)
    st = torch.load(ck, map_location=device, weights_only=False)
    m.load_state_dict(st.get("model_state_dict", st) if isinstance(st, dict) else st)
    m.to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()                       # D25: BatchNorm frozen, buffers never updated
    return m, ck


class WithPatient(torch.utils.data.Dataset):
    """Carry each clip's patient through the DataLoader.

    The batch's patient decides which z_behavior is injected, so it must be derived
    from the batch ITSELF and never from a parallel list kept on the sampler. An
    earlier version zipped the loader against `sampler.epoch_patients`, which Python
    binds before `__iter__` fills it: the first epoch paired against an empty list and
    ran zero batches, and every later epoch paired batches against the PREVIOUS
    epoch's patient order -- injecting the wrong patient's signature throughout, with
    no error. The homogeneity assertion below now makes that class of mistake loud.
    """

    def __init__(self, ds):
        self.ds = ds
        self.patients = [patient_of(str(n)) for n, _ in ds._labels]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        sample, target = self.ds[i]
        return sample, target, self.patients[i]


def group_by_patient(ds):
    by, lab = {}, {}
    for i, (name, y) in enumerate(ds._labels):
        p = patient_of(str(name))
        by.setdefault(p, []).append(i)
        lab.setdefault(p, []).append(float(y))
    return by, lab


def load_z(fold, device):
    f = Path(config.SIGNATURES_DIR) / fold / "z_behavior.npz"
    if not f.exists():
        raise FileNotFoundError(
            f"no signatures for {fold} at {f}. Run scripts/build_signatures.py first.")
    z = np.load(f)
    return {p: torch.tensor(z[p], dtype=torch.float32, device=device) for p in z.files}


def lr_at(step, total_steps):
    if step < config.HN_WARMUP_STEPS:
        t = step / max(1, config.HN_WARMUP_STEPS)
        return config.HN_LR_START + t * (config.HN_LR_PEAK - config.HN_LR_START)
    t = (step - config.HN_WARMUP_STEPS) / max(1, total_steps - config.HN_WARMUP_STEPS)
    t = min(1.0, max(0.0, t))
    return config.HN_LR_MIN + 0.5 * (config.HN_LR_PEAK - config.HN_LR_MIN) * \
        (1 + math.cos(math.pi * t))


def run_batches(hn, backbone, targets, norms, loader, sampler, zs, device,
                criterion, optimizer=None, jitter=0.0, step0=0, total_steps=1,
                limit=None):
    """One pass. Training when optimizer is given, else evaluation."""
    train = optimizer is not None
    hn.train(train)
    total, n, step = 0.0, 0, step0
    for bi, batch in enumerate(loader):
        if limit and bi >= limit:
            break
        sample, labels, pats = batch
        uniq = set(pats)
        if len(uniq) != 1:
            raise RuntimeError(
                f"batch {bi} mixes patients {sorted(uniq)}. One forward pass carries "
                f"one z_behavior; a mixed batch would inject the wrong signature.")
        patient = next(iter(uniq))
        z = zs[patient]
        if train and jitter:
            z = z + torch.randn_like(z) * jitter        # [METHOD] z-jitter, train only
        deltas = hn(z, base_norms=norms)
        with torch.set_grad_enabled(train):
            with AdaptedForward(targets, deltas):
                logits = backbone(sample["data"].to(device), sample["kpts"].to(device),
                                  return_logits=True)
            if logits.dim() > 1:
                logits = logits.squeeze(1)
            loss = criterion(logits, labels.float().to(device))
        if train:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, total_steps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hn.parameters(), config.HN_GRAD_CLIP)
            optimizer.step()
            step += 1
        total += float(loss) * labels.numel()
        n += labels.numel()
    return total / max(n, 1), step


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=str, default=None)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip folds whose hypernetwork_best.pth already exists")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="cap batches per pass (smoke test only, not a real run)")
    # Overrides for the D28 optimisation diagnostic. The defaults in config.py are the
    # methodology's values and are what an unflagged run uses; anything passed here is
    # a deliberate deviation and is recorded in the run manifest and the output path so
    # a diagnostic run can never be mistaken for a spec run.
    ap.add_argument("--lr-peak", type=float, default=None)
    ap.add_argument("--delta-clip", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None,
                    help="suffix for the checkpoint directory, e.g. 'gentle'")
    args = ap.parse_args()

    if not args.fold and not args.all_folds:
        ap.error("pass --fold patNN or --all-folds")
    folds = sorted(config.COHORT) if args.all_folds else [args.fold.lower()]
    t_all = time.time()
    for i, f in enumerate(folds):
        if len(folds) > 1:
            print(f"\n{'='*70}\n[{i+1}/{len(folds)}] fold {f}\n{'='*70}")
        train_one(f, args)
    if len(folds) > 1:
        print(f"\nall {len(folds)} folds in {(time.time()-t_all)/3600:.2f} h")
    return 0


def train_one(fold, args):
    meta = json.loads((Path(config.FOLDS_DIR) / fold / "fold.json").read_text())
    _tag = f"_{args.tag}" if args.tag else ""
    if args.skip_done and (Path(config.HYPER_CKPT_ROOT) / f"{fold}{_tag}" /
                           "hypernetwork_best.pth").exists():
        print(f"  [skip] {fold}{_tag} already trained")
        return
    seed = fold_seed(config.GLOBAL_SEED, sorted(config.COHORT).index(fold))
    seed_everything(seed)
    device = get_device()
    max_epochs = args.max_epochs or config.HN_MAX_EPOCHS

    overrides = {}
    for flag, key in (("lr_peak", "HN_LR_PEAK"), ("delta_clip", "HN_DELTA_CLIP_RATIO"),
                      ("patience", "HN_PATIENCE")):
        v = getattr(args, flag)
        if v is not None:
            overrides[key] = {"spec": getattr(config, key), "used": v}
            setattr(config, key, v)
    if overrides:
        print("[deviation] running with non-specification optimisation:")
        for k, d in overrides.items():
            print(f"    {k}: {d['spec']} -> {d['used']}")
        if not args.tag:
            raise SystemExit("  pass --tag to keep this run out of the specification "
                             "checkpoint directory")

    backbone, ck = frozen_backbone(fold, device)
    targets = resolve_targets(backbone)
    norms = base_norms(targets)
    zs = load_z(fold, device)
    hn = Hypernetwork().to(device)

    train_ds = VSViGDataset(config.PROCESSED_DIR, Path(config.FOLDS_DIR)/fold/"train_clips.json")
    val_ds = VSViGDataset(config.PROCESSED_DIR, Path(config.FOLDS_DIR)/fold/"val_clips.json")
    tr_by, tr_lab = group_by_patient(train_ds)
    va_by, va_lab = group_by_patient(val_ds)

    assert fold not in tr_by and fold not in va_by, f"LEAK: {fold} appears in training"
    assert set(tr_by) <= set(meta["train_patients"]), "unexpected training patients"

    tr_s = CyclicBatchSampler(tr_by, tr_lab, seed=seed)
    va_s = CyclicBatchSampler(va_by, va_lab, seed=seed + 1, shuffle_batches=False)
    tr_dl = DataLoader(WithPatient(train_ds), batch_sampler=tr_s,
                       num_workers=config.S1_NUM_WORKERS)
    va_dl = DataLoader(WithPatient(val_ds), batch_sampler=va_s,
                       num_workers=config.S1_NUM_WORKERS)

    print(f"[env] device={device}  backbone={ck}  BN frozen={config.HN_FREEZE_BN}")
    print(f"=== fold {fold}: hypernetwork ({sum(p.numel() for p in hn.parameters()):,} params) ===")
    print(f"  train {len(tr_s)} batches/epoch over {sorted(tr_by)}")
    print(f"  val   {len(va_s)} batches/epoch over {sorted(va_by)}")
    for tag, sk in (("train", tr_s.skipped), ("val", va_s.skipped)):
        if sk:
            print(f"  [note] {tag}: patients with only one class, no batches: {sk}")

    # The invariant the whole comparison rests on.
    with torch.no_grad():
        d0 = hn(next(iter(zs.values())), base_norms=norms)
    mx = max(float(v.abs().max()) for v in d0.values())
    assert mx == 0.0, f"dW is not zero at init (max {mx}); the arms do not start equal"
    print(f"  dW = 0 at init: adapted == baseline before the first update")

    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(hn.parameters(), lr=config.HN_LR_START,
                            weight_decay=config.HN_WEIGHT_DECAY)
    total_steps = max_epochs * max(1, len(tr_s))

    ck_dir = Path(config.HYPER_CKPT_ROOT) / (f"{fold}_{args.tag}" if args.tag else fold)
    ck_dir.mkdir(parents=True, exist_ok=True)
    hist = {"train_loss": [], "val_loss": [], "lr": [], "epoch_s": [], "delta_norm": []}
    best, trigger, step = float("inf"), 0, 0

    for epoch in range(max_epochs):
        t0 = time.time()
        tr_loss, step = run_batches(hn, backbone, targets, norms, tr_dl, tr_s, zs,
                                    device, criterion, opt,
                                    jitter=config.HN_Z_JITTER_SIGMA,
                                    step0=step, total_steps=total_steps,
                                    limit=args.limit_batches)
        va_loss, _ = run_batches(hn, backbone, targets, norms, va_dl, va_s, zs,
                                 device, criterion, limit=args.limit_batches)
        with torch.no_grad():
            dn = float(np.mean([float(v.flatten().norm())
                                for v in hn(zs[meta["train_patients"][0]],
                                            base_norms=norms).values()]))
        hist["train_loss"].append(tr_loss); hist["val_loss"].append(va_loss)
        hist["lr"].append(opt.param_groups[0]["lr"]); hist["epoch_s"].append(round(time.time()-t0, 1))
        hist["delta_norm"].append(dn)

        improved = va_loss < best - 1e-6
        if improved:
            best, trigger = va_loss, 0
            torch.save(hn.state_dict(), ck_dir / "hypernetwork_best.pth")
        else:
            trigger += 1
        print(f"  epoch {epoch+1:>3}/{max_epochs}  bce={tr_loss:.5f}  val_bce={va_loss:.5f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}  |dW|={dn:.4f}  {time.time()-t0:.0f}s  "
              f"{'** best' if improved else f'no improve {trigger}/{config.HN_PATIENCE}'}")
        (ck_dir / "training_log.json").write_text(json.dumps(hist, indent=2))
        torch.save(hn.state_dict(), ck_dir / "hypernetwork_last.pth")
        if trigger >= config.HN_PATIENCE:
            print(f"  early stop at epoch {epoch+1} "
                  f"({config.HN_PATIENCE} epochs without improvement)")
            break

    write_manifest(ck_dir / "run_manifest.json", seed=seed,
                   extra={"stage": "7_hypernetwork", "fold": fold, "backbone": str(ck),
                          "best_val_bce": best, "epochs_run": len(hist["val_loss"]),
                          "specification_deviations": overrides or None})
    print(f"  best val BCE {best:.5f} -> {ck_dir/'hypernetwork_best.pth'}")


if __name__ == "__main__":
    raise SystemExit(main())
