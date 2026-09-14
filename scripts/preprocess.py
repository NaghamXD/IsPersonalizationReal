"""Training-clip extraction.

Rewritten onto the shared core in src/data/extract.py and the window planner in
src/data/windows.py, so training and test clips are built by the same code. The
6fca412 version carried its own copy of the pose/patch pipeline and ran the whole
thing as an import-time side effect, which is why nothing could reuse it.

What changed, beyond the refactor:
  * Restricted to config.COHORT. The excluded patients are not preprocessed at all,
    rather than being extracted and filtered out later.
  * Per-video frame rate, instead of whatever cv2 happened to report being ignored.
  * Supplementary seizure-free footage (free.mp4, no-Sz2P.mp4) is ingested. The old
    script only opened files matching Label.xlsx's naming pattern, so for pat03 and
    pat04 -- whose seizure recordings begin 7 s and 14 s before onset -- it collected
    almost no interictal video at all.
  * That footage is sampled UNIFORMLY across each file up to the methodology's cap of
    40 clips, not taken as a contiguous prefix.
  * --dry-run, so the plan and its cost can be inspected before committing hours.

    python scripts/preprocess.py --all --dry-run
    python scripts/preprocess.py --all

Idempotent: existing clips are skipped, so an interrupted run resumes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from src.data.windows import coverage_seconds, plan_windows
from src.utils.manifest import write_manifest
from src.utils.seeding import seed_everything

# Seizure-free footage Label.xlsx does not describe. Kept identical to
# scripts/extract_test_clips.py so both stages see the same sources.
EXTRA_FOOTAGE = {
    "pat03": [("free.mp4", "free")],
    "pat04": [("free.mp4", "free"), ("no-Sz2P.mp4", "free2")],
}

SECONDS_PER_CLIP_ESTIMATE = 0.45


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


def subsample_uniform(windows, cap):
    """Keep `cap` windows spread evenly across the whole recording.

    np.linspace over the index range, not the first `cap` entries: a prefix of a
    30-minute file is ~200 s from one moment of one activity, which is the transient
    behaviour the methodology's uniform sampling exists to avoid.
    """
    if cap is None or len(windows) <= cap:
        return windows
    idx = np.unique(np.linspace(0, len(windows) - 1, cap).round().astype(int))
    return [windows[i] for i in idx]


def plan_patient(patient, label_df, probe):
    plan = []
    for _, row in label_df[label_df["PatID"] == patient].iterrows():
        sz, sz_type = str(row["#Seizure"]).strip(), str(row["Seizure Type"]).strip()
        path = Path(config.DATA_ROOT) / patient / f"{sz}{sz_type}.mp4"
        if not path.exists():
            print(f"    [missing] {path}"); continue
        info = probe(path)
        if info is None:
            print(f"    [unreadable] {path}"); continue
        w = plan_windows(info.duration_s, row["eeg_s"], row["clin_s"], mode="train")
        plan.append({"path": path, "tag": sz, "info": info, "windows": w,
                     "eeg_s": row["eeg_s"], "clin_s": row["clin_s"], "capped": False})

    for fname, tag in EXTRA_FOOTAGE.get(patient, []):
        path = Path(config.DATA_ROOT) / patient / fname
        if not path.exists():
            print(f"    [missing] {path}"); continue
        info = probe(path)
        if info is None:
            print(f"    [unreadable] {path}"); continue
        w = plan_windows(info.duration_s, None, None, mode="train")
        n_before = len(w)
        w = subsample_uniform(w, config.EXTRA_FOOTAGE_CAP)
        plan.append({"path": path, "tag": tag, "info": info, "windows": w,
                     "eeg_s": None, "clin_s": None,
                     "capped": len(w) < n_before, "n_before_cap": n_before})
    return plan


def report_plan(patient, plan):
    total = sum(len(p["windows"]) for p in plan)
    n_i = n_t = n_c = 0
    print(f"  {patient}: {total} clips across {len(plan)} source(s)")
    for p in plan:
        cov = coverage_seconds(p["windows"])
        counts = {k: int(cov[k] / config.CLIP_SECONDS) for k in
                  ("interictal", "transition", "ictal")}
        n_i += counts["interictal"]; n_t += counts["transition"]; n_c += counts["ictal"]
        cap = (f"  [capped {p['n_before_cap']}->{len(p['windows'])}, uniform]"
               if p.get("capped") else "")
        print(f"      {p['path'].name:<14} dur={p['info'].duration_s/60:>6.1f}m  "
              f"clips={len(p['windows']):>5}  "
              f"inter={counts['interictal']:>4} trans={counts['transition']:>4} "
              f"ictal={counts['ictal']:>4}{cap}")
    print(f"      -> interictal={n_i}  transition={n_t}  ictal={n_c}")
    return total, n_i, n_t, n_c


def extract_patient(patient, plan, net, g_filter, device, force=False):
    import cv2
    import torch
    from src.data.extract import extract_clip

    p_dir, k_dir = Path(config.PATCHES_DIR), Path(config.KPTS_DIR)
    p_dir.mkdir(parents=True, exist_ok=True)
    k_dir.mkdir(parents=True, exist_ok=True)

    labels, n_new, n_skip, n_fail = {}, 0, 0, 0
    for p in plan:
        cap = cv2.VideoCapture(str(p["path"]))
        if not cap.isOpened():
            print(f"    [cannot open] {p['path']}"); continue
        for w in p["windows"]:
            name = f"{patient}_{p['tag']}_{int(round(w.t_start_s))}"
            dst = p_dir / f"{name}.pt"
            if dst.exists() and not force:
                labels[name] = w.label; n_skip += 1; continue
            patches, kpts = extract_clip(cap, p["info"].fps, w.t_start_s,
                                         net, g_filter, device)
            if patches is None:
                n_fail += 1; continue
            _dt = getattr(torch, getattr(config, "PATCH_STORE_DTYPE", "float32"))
            torch.save(torch.from_numpy(patches).to(_dt), dst)      # D33
            torch.save(torch.from_numpy(kpts).float(), k_dir / f"{name}.pt")
            labels[name] = w.label; n_new += 1
            if (n_new + n_skip) % 200 == 0:
                print(f"      {patient}: {n_new} new / {n_skip} existing / {n_fail} failed")
        cap.release()
    return labels, n_new, n_skip, n_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", type=str, default=None)
    ap.add_argument("--all", action="store_true", help="every patient in the cohort")
    ap.add_argument("--patients", type=str, default=None,
                    help="comma-separated list, e.g. the excluded patients for the "
                         "Phase 2 universal backbone. Patients outside config.COHORT "
                         "are allowed here but warned about, because they are usable as "
                         "backbone training data while being unusable for evaluation.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not (args.patient or args.all or args.patients):
        ap.error("pass --patient patNN, --patients a,b,c or --all")

    seed_everything(config.GLOBAL_SEED)
    if args.all:
        patients = list(config.COHORT)
    elif args.patients:
        patients = [x.strip().lower() for x in args.patients.split(",") if x.strip()]
    else:
        patients = [args.patient.lower()]
    for p in patients:
        if p not in config.COHORT:
            print(f"[warn] {p} is not in the cohort (excluded: {config.EXCLUDED_PATIENTS})")

    label_df = load_label_table()
    from src.data.extract import probe_video
    print(f"=== training extraction: interictal stride {config.TRAIN_STRIDE_INTERICTAL_S}s, "
          f"ictal/transition stride {config.TRAIN_STRIDE_ICTAL_S}s (4 s overlap) ===")
    print(f"    cohort: {len(patients)} patients\n")

    plans, tot, ti, tt, tc = {}, 0, 0, 0, 0
    for p in patients:
        plans[p] = plan_patient(p, label_df, probe_video)
        a, b, c, d = report_plan(p, plans[p])
        tot += a; ti += b; tt += c; tc += d
        print()

    print(f"TOTAL: {tot} clips  (interictal={ti}  transition={tt}  ictal={tc})")
    if tot:
        print(f"       class balance: {ti/tot:.0%} / {tt/tot:.0%} / {tc/tot:.0%}")
    if args.dry_run:
        print(f"(dry run: nothing decoded or written. "
              f"~{tot*SECONDS_PER_CLIP_ESTIMATE/60:.0f} min at "
              f"{SECONDS_PER_CLIP_ESTIMATE}s/clip.)")
        return 0

    from src.data.extract import build_gaussian_filter, get_device, load_pose_model
    device = get_device()
    print(f"\n[env] device={device}; loading pose model...")
    net = load_pose_model(device)
    g_filter = build_gaussian_filter()

    labels_path = Path(config.LABELS_JSON)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    all_labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}

    t0 = time.time()
    for p in patients:
        print(f"\n--- extracting {p} ---")
        labels, n_new, n_skip, n_fail = extract_patient(
            p, plans[p], net, g_filter, device, force=args.force)
        all_labels.update(labels)
        labels_path.write_text(json.dumps(all_labels))   # checkpoint after each patient
        print(f"    {p}: {n_new} new, {n_skip} existing, {n_fail} failed "
              f"({len(all_labels)} clips in labels.json)")

    write_manifest(Path(config.PROCESSED_DIR) / "run_manifest_preprocess.json",
                   seed=config.GLOBAL_SEED,
                   extra={"stage": "preprocess_train", "patients": patients,
                          "n_clips": len(all_labels),
                          "elapsed_s": round(time.time() - t0, 1)})
    print(f"\nwrote {labels_path}  ({len(all_labels)} clips, {time.time()-t0:.0f}s)")
    print("Next: python scripts/extract_test_clips.py --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
