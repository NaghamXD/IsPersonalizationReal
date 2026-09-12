"""Where does an epoch actually go -- disk or compute?

The overfit probe established that the model CAN learn but needs far more gradient
steps than it was given. Raising the epoch budget is therefore the fix, and the
question becomes whether that is affordable. It is only affordable if the bottleneck
is not what it currently looks like: num_workers=0 means every batch loads 16 clips of
roughly 5.5 MB each from disk, synchronously, before any compute begins.

This measures the three numbers that decide it:
  1. pure compute      -- forward+backward on one cached batch, repeated
  2. pure data loading -- iterate the loader, touch no model
  3. combined          -- at several worker counts

    python scripts/benchmark_throughput.py
"""
import argparse
import contextlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from src.data.dataset import VSViGDataset
from src.model.vsvig import VSViG_base


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=str, default="pat01")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--workers", type=int, nargs="*", default=[0, 2, 4, 8])
    ap.add_argument("--batch-size", type=int, default=config.S1_BATCH_SIZE)
    ap.add_argument("--sweep", action="store_true",
                    help="also sweep batch size x autocast, the two levers that can "
                         "actually move a compute-bound workload")
    ap.add_argument("--sweep-batches", type=int, nargs="*", default=[16, 32, 64])
    args = ap.parse_args()

    device = get_device()
    train_file = Path(config.FOLDS_DIR) / args.fold / "train_clips.json"
    ds = VSViGDataset(config.PROCESSED_DIR, train_file)
    print(f"[env] device={device}  dataset={len(ds)} clips  batch={args.batch_size}  "
          f"timing {args.batches} batches per configuration\n")

    model = VSViG_base(kpt_channels=config.KPT_CHANNELS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.S1_LR)
    crit = nn.HuberLoss(delta=config.S1_HUBER_DELTA)

    # ---- 1. pure compute, one batch held in memory and reused -------------
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    sample, labels = next(iter(loader))
    data, kpts = sample["data"].to(device), sample["kpts"].to(device)
    labels = labels.float().to(device)
    for _ in range(3):                                   # warm up kernels
        out = model(data, kpts)
        crit(out.squeeze() if out.dim() > 1 else out, labels).backward()
        opt.zero_grad()
    sync(device)
    t0 = time.time()
    for _ in range(args.batches):
        out = model(data, kpts)
        loss = crit(out.squeeze() if out.dim() > 1 else out, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    sync(device)
    compute_s = (time.time() - t0) / args.batches
    print(f"  1. compute only (cached batch)      {compute_s*1000:>8.0f} ms/batch")

    # ---- 2. data loading only, no model ----------------------------------
    for w in args.workers:
        kw = dict(num_workers=w)
        if w > 0:
            kw.update(persistent_workers=True, prefetch_factor=4)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, **kw)
        it = iter(dl)
        for _ in range(3):
            next(it)
        t0 = time.time()
        for _ in range(args.batches):
            next(it)
        load_s = (time.time() - t0) / args.batches
        del it, dl
        print(f"  2. data only, num_workers={w:<2}          {load_s*1000:>8.0f} ms/batch"
              f"   -> epoch would cost {load_s*len(ds)/args.batch_size:>6.0f} s in I/O alone")

    # ---- 3. combined -----------------------------------------------------
    print()
    best = None
    for w in args.workers:
        kw = dict(num_workers=w)
        if w > 0:
            kw.update(persistent_workers=True, prefetch_factor=4)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **kw)
        it = iter(dl)
        next(it)
        sync(device)
        t0 = time.time()
        for _ in range(args.batches):
            sample, labels = next(it)
            out = model(sample["data"].to(device), sample["kpts"].to(device))
            loss = crit(out.squeeze() if out.dim() > 1 else out,
                        labels.float().to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sync(device)
        per_batch = (time.time() - t0) / args.batches
        epoch_s = per_batch * len(ds) / args.batch_size
        if best is None or per_batch < best[1]:
            best = (w, per_batch, epoch_s)
        print(f"  3. train step, num_workers={w:<2}        {per_batch*1000:>8.0f} ms/batch"
              f"   -> {epoch_s:>6.0f} s/epoch")
        del it, dl

    # ---- 4. the two levers that matter when compute-bound --------------------
    if args.sweep:
        print("\n  4. batch size x autocast (compute only, one cached batch reused)")
        print(f"     {'batch':>6}{'dtype':>10}{'ms/batch':>11}{'ms/clip':>10}"
              f"{'s/epoch':>10}{'vs base':>10}")
        base_ms_per_clip = compute_s * 1000 / args.batch_size
        results = []
        for bs in args.sweep_batches:
            dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
            try:
                sample, labels = next(iter(dl))
            except Exception as e:
                print(f"     batch {bs}: cannot build a batch ({e})")
                continue
            d, k = sample["data"].to(device), sample["kpts"].to(device)
            lab = labels.float().to(device)
            for use_amp in (False, True):
                ctx = (torch.autocast(device_type=device.type, dtype=torch.float16)
                       if use_amp else contextlib.nullcontext())
                try:
                    for _ in range(3):
                        with ctx:
                            o = model(d, k)
                            l = crit(o.squeeze() if o.dim() > 1 else o, lab)
                        opt.zero_grad(); l.backward(); opt.step()
                    sync(device)
                    t0 = time.time()
                    for _ in range(max(5, args.batches // 2)):
                        with ctx:
                            o = model(d, k)
                            l = crit(o.squeeze() if o.dim() > 1 else o, lab)
                        opt.zero_grad(); l.backward(); opt.step()
                    sync(device)
                    n_it = max(5, args.batches // 2)
                    ms = (time.time() - t0) / n_it * 1000
                except Exception as e:
                    print(f"     {bs:>6}{'fp16' if use_amp else 'fp32':>10}"
                          f"   FAILED: {type(e).__name__}: {str(e)[:60]}")
                    continue
                ms_clip = ms / bs
                ep = ms_clip * len(ds) / 1000
                speed = base_ms_per_clip / ms_clip
                print(f"     {bs:>6}{'fp16' if use_amp else 'fp32':>10}{ms:>11.0f}"
                      f"{ms_clip:>10.1f}{ep:>10.0f}{speed:>9.2f}x")
                results.append((bs, use_amp, ep, speed))
            del d, k, lab, dl
        if results:
            bs, amp, ep, speed = min(results, key=lambda r: r[2])
            print(f"\n     fastest: batch={bs} "
                  f"{'fp16 autocast' if amp else 'fp32'} at {ep:.0f} s/epoch "
                  f"({speed:.2f}x the current configuration)")
            for n in (50, 100, 200, 300):
                print(f"       8 folds x {n:>3} epochs -> {8*n*ep/3600:>6.1f} h")
            print("\n     NOTE: a larger batch means FEWER gradient steps per epoch. "
                  "If steps\n     are what this model is short of, epochs are not "
                  "interchangeable across\n     batch sizes -- compare at equal step "
                  "counts, not equal epochs.")

    w, per_batch, epoch_s = best
    overhead = max(0.0, per_batch - compute_s)
    print(f"\n  --- verdict ---")
    print(f"  best: num_workers={w} at {epoch_s:.0f} s/epoch "
          f"(compute floor {compute_s*len(ds)/args.batch_size:.0f} s/epoch)")
    print(f"  non-compute overhead: {overhead*1000:.0f} ms/batch "
          f"({100*overhead/per_batch:.0f}% of a step)")
    for ep in (50, 100, 200):
        print(f"  8 folds x {ep:>3} epochs -> {8*ep*epoch_s/3600:>6.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
