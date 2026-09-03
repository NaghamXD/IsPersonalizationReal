"""Stage 3 -- test-time clip extraction.

A continuous, NON-OVERLAPPING 5 s sliding window over every held-out patient's
evaluable footage, written to a directory of its own so it can never be confused with
the training-strided clips.

Why this exists. Training extraction deliberately overlaps ictal and transition clips
by 4 s as augmentation. Scoring those same clips at test time inflates results twice:
the accumulation window sees one movement several times, and a seizure gets several
independent chances to be detected. Latency measured on overlapping clips is not
latency. The methodology is explicit that test evaluation forbids overlap; nothing in
the base pipeline implemented that.

    python scripts/extract_test_clips.py --all --dry-run     # plan + cost, no decoding
    python scripts/extract_test_clips.py --patient pat13
    python scripts/extract_test_clips.py --all

Idempotent: clips already on disk are skipped, so an interrupted run resumes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from src.data.windows import coverage_seconds, plan_windows
from src.utils.manifest import write_manifest
from src.utils.seeding import seed_everything

# Seizure-free footage that Label.xlsx does not describe, so preprocess.py never opens
# it. Discovered by inspecting the corpus directly. For pat03 and pat04 this is the
# only interictal video they have -- their seizure files start seconds before onset.
EXTRA_FOOTAGE = {
    "pat03": [("free.mp4", "free")],
    "pat04": [("free.mp4", "free"), ("no-Sz2P.mp4", "free2")],
}

OUT_DIR = Path(config.TEST_CLIPS_DIR)
SECONDS_PER_CLIP_ESTIMATE = 0.45   # rough M3 Max cost, for the dry-run ETA only


def sources_for(patient: str, label_df: pd.DataFrame):
    """[(video_path, event_tag, eeg_s, clin_s)] for one patient."""
    out = []
    rows = label_df[label_df["PatID"] == patient]
    for _, row in rows.iterrows():
        sz = str(row["#Seizure"]).strip()
        sz_type = str(row["Seizure Type"]).strip()
        path = Path(config.DATA_ROOT) / patient / f"{sz}{sz_type}.mp4"
        out.append((path, sz, row["eeg_s"], row["clin_s"]))
    for fname, tag in EXTRA_FOOTAGE.get(patient, []):
        out.append((Path(config.DATA_ROOT) / patient / fname, tag, None, None))
    return out


def load_label_table():
    df = pd.read_excel(config.LABEL_XLSX)
    df.columns = [c.strip() for c in df.columns]
    df[["PatID", "Seizure Type"]] = df[["PatID", "Seizure Type"]].ffill()
    df["PatID"] = df["PatID"].astype(str).str.strip().str.lower()
    df["Seizure Type"] = df["Seizure Type"].astype(str).str.strip()
    from src.data.extract import label_seconds
    df["eeg_s"] = df["EEG onset"].apply(label_seconds)
    df["clin_s"] = df["Clinical Onset"].apply(label_seconds)
    return df


def plan_patient(patient, label_df, probe):
    """Window plan for every source of one patient, without decoding anything."""
    plan = []
    for path, tag, eeg_s, clin_s in sources_for(patient, label_df):
        if not path.exists():
            print(f"    [missing] {path}")
            continue
        info = probe(path)
        if info is None:
            print(f"    [unreadable] {path}")
            continue
        windows = plan_windows(info.duration_s, eeg_s, clin_s, mode="test")
        plan.append({"path": path, "tag": tag, "eeg_s": eeg_s, "clin_s": clin_s,
                     "info": info, "windows": windows,
                     "coverage": coverage_seconds(windows)})
    return plan


def report_plan(patient, plan):
    tot_w = sum(len(p["windows"]) for p in plan)
    inter = sum(p["coverage"]["interictal"] for p in plan)
    print(f"  {patient}: {tot_w} windows across {len(plan)} source(s)")
    for p in plan:
        c = p["coverage"]
        print(f"      {p['path'].name:<14} fps={p['info'].fps:>6.2f} "
              f"dur={p['info'].duration_s/60:>6.1f}m  "
              f"windows={len(p['windows']):>5}  "
              f"interictal={c['interictal']:>7.0f}s  "
              f"transition={c['transition']:>5.0f}s  ictal={c['ictal']:>6.0f}s")
    hours = inter / 3600.0
    rate = (1.0 / hours) if hours > 0 else float("inf")
    print(f"      -> evaluable interictal exposure: {inter:.0f}s = {hours:.4f} h  "
          f"(one false alarm = {rate:.1f} FDR/h)")
    return tot_w, inter


def extract_patient(patient, plan, net, g_filter, device, force=False):
    import cv2
    import numpy as np
    import torch
    from src.data.extract import extract_clip

    patches_dir = OUT_DIR / "patches"
    kpts_dir = OUT_DIR / "kpts"
    patches_dir.mkdir(parents=True, exist_ok=True)
    kpts_dir.mkdir(parents=True, exist_ok=True)

    manifest, coverage, n_new, n_skip, n_fail = [], {}, 0, 0, 0
    for p in plan:
        cap = cv2.VideoCapture(str(p["path"]))
        if not cap.isOpened():
            print(f"    [cannot open] {p['path']}")
            continue
        kept = 0
        for w in p["windows"]:
            name = f"{patient}_{p['tag']}_{int(round(w.t_start_s))}"
            dst = patches_dir / f"{name}.pt"
            if dst.exists() and not force:
                manifest.append([name, w.label]); n_skip += 1; kept += 1
                continue
            patches, kpts = extract_clip(cap, p["info"].fps, w.t_start_s,
                                         net, g_filter, device)
            if patches is None:
                n_fail += 1
                continue
            torch.save(torch.from_numpy(patches).float(), dst)
            torch.save(torch.from_numpy(kpts).float(), kpts_dir / f"{name}.pt")
            manifest.append([name, w.label]); n_new += 1; kept += 1
            if (n_new + n_skip) % 100 == 0:
                print(f"      {patient}: {n_new} new / {n_skip} existing / {n_fail} failed")
        cap.release()

        # Coverage is recorded from what was actually WRITTEN, not from the plan --
        # a clip that failed to decode was never scored and must not inflate the
        # FDR/h denominator.
        cov = coverage_seconds([w for w in p["windows"]][:kept])
        coverage[f"{patient}_{p['tag']}"] = {
            "source": p["path"].name, "fps": p["info"].fps,
            "duration_s": p["info"].duration_s,
            "eeg_s": p["eeg_s"], "clin_s": p["clin_s"],
            "planned_windows": len(p["windows"]), "written_windows": kept,
            **cov,
        }
    return manifest, coverage, n_new, n_skip, n_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", type=str, default=None)
    ap.add_argument("--all", action="store_true", help="every patient in the cohort")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and cost only; no video decoded, nothing written")
    ap.add_argument("--force", action="store_true", help="re-extract existing clips")
    args = ap.parse_args()
    if not args.patient and not args.all:
        ap.error("pass --patient patNN or --all")

    seed_everything(config.GLOBAL_SEED)
    patients = config.COHORT if args.all else [args.patient.lower()]
    for p in patients:
        if p not in config.COHORT:
            print(f"[warn] {p} is not in the configured cohort "
                  f"(excluded: {config.EXCLUDED_PATIENTS})")

    label_df = load_label_table()
    from src.data.extract import probe_video
    print(f"=== test-time sliding window: stride={config.TEST_STRIDE_S}s on a "
          f"{config.CLIP_SECONDS}s clip (no overlap) ===\n")

    plans = {}
    grand_windows = grand_interictal = 0
    for p in patients:
        plan = plan_patient(p, label_df, probe_video)
        plans[p] = plan
        w, i = report_plan(p, plan)
        grand_windows += w
        grand_interictal += i
        print()

    print(f"TOTAL: {grand_windows} windows, "
          f"{grand_interictal/3600.0:.3f} h of evaluable interictal exposure")
    if args.dry_run:
        eta = grand_windows * SECONDS_PER_CLIP_ESTIMATE / 60.0
        print(f"(dry run: nothing decoded or written. Rough extraction cost "
              f"~{eta:.0f} min at {SECONDS_PER_CLIP_ESTIMATE}s/clip.)")
        return 0

    from src.data.extract import build_gaussian_filter, get_device, load_pose_model
    device = get_device()
    print(f"\n[env] device={device}; loading pose model...")
    net = load_pose_model(device)
    g_filter = build_gaussian_filter()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for p in patients:
        print(f"\n--- extracting {p} ---")
        manifest, coverage, n_new, n_skip, n_fail = extract_patient(
            p, plans[p], net, g_filter, device, force=args.force)
        (OUT_DIR / f"manifest_{p}.json").write_text(json.dumps(manifest))
        (OUT_DIR / f"coverage_{p}.json").write_text(json.dumps(coverage, indent=2))
        print(f"    {p}: {n_new} new, {n_skip} existing, {n_fail} failed "
              f"-> {len(manifest)} clips in manifest_{p}.json")

    write_manifest(OUT_DIR / "run_manifest.json", seed=config.GLOBAL_SEED,
                   extra={"stage": "3_test_extraction", "patients": patients,
                          "elapsed_s": round(time.time() - t0, 1)})
    print(f"\nwrote {OUT_DIR}/  ({time.time()-t0:.0f}s)")
    print("Score them with:\n"
          f"  python scripts/evaluate.py --fold PatNN --model baseline \\\n"
          f"      --data-folder {OUT_DIR} --clips {OUT_DIR}/manifest_patNN.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
