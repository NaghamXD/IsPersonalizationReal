"""Clinical evaluation, model-agnostic.

Replaces scripts/evaluate_lopo.py, which had three defects that made its output
unusable: it read clip start times (seconds) as frame indices and divided by an
assumed 25 fps; it looked for a checkpoint filename and directory case that training
never writes, so it silently skipped every fold; and it never computed FDR per hour at
all, which is the metric the project's central hypothesis is stated in.

The same harness scores the unadapted backbone and the patient-conditioned model, so
the two are directly comparable -- that comparison IS the experiment.

    python scripts/evaluate.py --fold Pat01 --model baseline
    python scripts/evaluate.py --fold Pat01 --model adapted
    python scripts/evaluate.py --all-folds --model baseline
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from src.data.dataset import VSViGDataset
from src.eval.metrics import aggregate, evaluate_source, within_source_auc
from src.model.vsvig import VSViG_base
from src.utils.manifest import write_manifest
from src.utils.seeding import seed_everything


# ---------------------------------------------------------------- ground truth
def load_onsets(xlsx_path=None):
    """{'pat01_Sz1': (eeg_s, clinical_s)} from Label.xlsx."""
    xlsx_path = xlsx_path or config.LABEL_XLSX
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns]
    df[["PatID", "Seizure Type"]] = df[["PatID", "Seizure Type"]].ffill()

    def secs(v):
        if pd.isna(v):
            return None
        if hasattr(v, "hour"):
            return float(v.hour * 3600 + v.minute * 60 + v.second)
        parts = str(v).strip().split(":")
        try:
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        except ValueError:
            return None
        return None

    out = {}
    for _, row in df.iterrows():
        key = f"{str(row['PatID']).strip().lower()}_{str(row['#Seizure']).strip()}"
        out[key] = (secs(row.get("EEG onset")), secs(row.get("Clinical Onset")))
    return out


# ---------------------------------------------------------------- checkpoints
def resolve_checkpoint(root: Path, patient: str) -> Path:
    """Find a fold's checkpoint without caring about filename or directory casing.

    The old evaluator hard-coded `{lowercase}/best.pth` while training writes
    `{Capitalised}/best_model.pth`, so it found nothing and reported an empty run as
    though it had succeeded. Search the plausible spellings and fail loudly.
    """
    # final_model.pth is the D21 last-k average and takes precedence: under the fixed
    # budget there is no "best" epoch, and best_model.pth is written as a copy of it.
    stems = ["final_model.pth", "best_model.pth", "best.pth", "hypernetwork_best.pth"]
    dirs = [patient, patient.capitalize(), patient.lower(), patient.upper()]
    tried = []
    for d in dict.fromkeys(dirs):
        for stem in stems:
            cand = Path(root) / d / stem
            tried.append(str(cand))
            if cand.exists():
                return cand
    raise FileNotFoundError(
        f"No checkpoint for {patient} under {root}.\nTried:\n  " + "\n  ".join(tried))


def build_model(kind: str, patient: str, device, tag=None, shuffled_z=None):
    if kind == "baseline":
        ckpt = resolve_checkpoint(config.BASELINE_CKPT_ROOT, patient)
        model = VSViG_base(kpt_channels=config.KPT_CHANNELS)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state)
                              if isinstance(state, dict) else state)
        model.to(device).eval()
        print(f"[model] baseline from {ckpt}")
        # VSViG_base.forward already ends in sigmoid.
        return model, (lambda logits: logits), ckpt

    if kind == "adapted":
        # D35: the adapted arm needs the SAME clinical pipeline as the baseline --
        # accumulation, refractory, FDR/h -- because section 3.5's benefit is defined as
        # a reduction in false detections per hour, not as an AUC difference.
        import numpy as _np
        from src.model.adapt import AdaptedModel, base_norms, resolve_targets
        from src.model.hypernetwork import Hypernetwork

        root = Path(config.BASELINE_CKPT_ROOT) / patient
        bck = next((c for c in (root / "final_model.pth", root / "best_model.pth")
                    if c.exists()), None)
        if bck is None:
            raise FileNotFoundError(f"no backbone for {patient} under {root}")
        backbone = VSViG_base(kpt_channels=config.KPT_CHANNELS)
        st = torch.load(bck, map_location=device, weights_only=False)
        backbone.load_state_dict(st.get("model_state_dict", st)
                                 if isinstance(st, dict) else st)
        backbone.to(device).eval()
        for prm in backbone.parameters():
            prm.requires_grad_(False)

        hn_dir = Path(config.HYPER_CKPT_ROOT) / (
            f"{patient}_{tag}" if tag else patient)
        hck = hn_dir / "hypernetwork_best.pth"
        if not hck.exists():
            raise FileNotFoundError(
                f"no hypernetwork for {patient} at {hck}. "
                f"Run scripts/train_hypernetwork.py --fold {patient} first.")
        hn = Hypernetwork().to(device)
        hn.load_state_dict(torch.load(hck, map_location=device, weights_only=False))
        hn.eval()

        zf = Path(config.SIGNATURES_DIR) / patient / "z_behavior.npz"
        if not zf.exists():
            raise FileNotFoundError(f"no signatures at {zf}")
        zs = _np.load(zf)
        z_from = shuffled_z or patient
        if z_from not in zs.files:
            raise KeyError(f"{z_from} has no signature in {zf}")
        if shuffled_z and shuffled_z == patient:
            raise ValueError("--shuffled-z must name a DIFFERENT patient; using the "
                             "held-out patient's own z is the adapted condition")

        targets = resolve_targets(backbone)
        with torch.no_grad():
            deltas = hn(torch.tensor(zs[z_from], dtype=torch.float32, device=device),
                        base_norms=base_norms(targets))
        model = AdaptedModel(backbone, targets, deltas, z_source=z_from).to(device).eval()
        label = "adapted" if not shuffled_z else f"shuffled-z({z_from})"
        print(f"[model] {label}: backbone {bck}, hypernetwork {hck}, z from {z_from}")
        return model, (lambda logits: logits), f"{bck} + {hck} [z={z_from}]"

    raise ValueError(f"unknown --model {kind!r}")


# ---------------------------------------------------------------- inference
@torch.no_grad()
def run_inference(model, activation, manifest_path, device, batch_size=32, limit=None,
                  data_folder=None):
    """-> {source_id: (t_start_s[], score[], label[])}"""
    ds = VSViGDataset(data_folder or config.PROCESSED_DIR, manifest_path,
                      eval_mode=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    buckets = defaultdict(lambda: ([], [], []))
    seen = 0
    for data, kpts, labels, source_ids, t_start in loader:
        out = model(data.to(device), kpts.to(device))
        if out.dim() > 1:
            out = out.squeeze(1)
        probs = activation(out).float().cpu().numpy()
        for i in range(len(probs)):
            t_list, s_list, y_list = buckets[source_ids[i]]
            t_list.append(float(t_start[i]))
            s_list.append(float(probs[i]))
            y_list.append(float(labels[i]))
        seen += len(probs)
        if limit and seen >= limit:
            break
    return {k: (np.array(a), np.array(b), np.array(c)) for k, (a, b, c) in buckets.items()}


# ---------------------------------------------------------------- per fold
def load_fold_threshold(patient):
    """D16: the DT selected on THIS fold's internal validation patients.

    Set on the config module so src.eval.decision and src.eval.metrics -- which read
    config.DECISION_THRESHOLD at call time -- both see it. The same value is applied to
    the baseline and to the adapted model; section 3.5 is a paired difference and must
    not also measure threshold tuning.
    """
    f = Path(config.THRESHOLDS_DIR) / f"{patient}.json"
    if not f.exists():
        print(f"[warn] no selected threshold for {patient}; falling back to the "
              f"inherited DT={config.DECISION_THRESHOLD}. Run scripts/select_threshold.py "
              f"-- an inherited threshold makes FDR/h incomparable across folds.")
        return config.DECISION_THRESHOLD, None
    rec = json.loads(f.read_text())
    config.DECISION_THRESHOLD = rec["dt"]
    print(f"[D16] DT={rec['dt']:.3f} selected on {rec['val_patients']} "
          f"(Youden J={rec['youden_j']:.3f})")
    return rec["dt"], rec


def evaluate_fold(patient, kind, onsets, device, manifest_path=None, limit=None,
                  data_folder=None, tag=None, shuffled_z=None):
    patient = patient.lower()
    dt_used, dt_rec = load_fold_threshold(patient)
    model, activation, ckpt = build_model(kind, patient, device, tag, shuffled_z)

    manifest_path = Path(manifest_path) if manifest_path else \
        Path(config.FOLDS_DIR) / f"val_{patient}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No clip manifest at {manifest_path}. Test-time clips come from Stage 3 "
            f"(continuous non-overlapping sliding window); until that exists you are "
            f"scoring TRAINING-strided clips, which overstates latency performance.")

    if data_folder is None:
        # Test clips live beside their own manifest; training clips live in the shared
        # processed_data root. Infer from where the manifest sits.
        parent = Path(manifest_path).parent
        data_folder = parent if (parent / "patches").exists() else config.PROCESSED_DIR
    if Path(data_folder) == Path(config.PROCESSED_DIR):
        print("[warn] scoring TRAINING-strided clips (ictal/transition overlap by 4 s). "
              "Latency and FDR/h from these are optimistic -- run "
              "scripts/extract_test_clips.py and pass --data-folder for real numbers.")
    by_source = run_inference(model, activation, manifest_path, device, limit=limit,
                              data_folder=data_folder)

    results, rows = [], []
    for source, (t, s, y) in sorted(by_source.items()):
        eeg_s, clin_s = onsets.get(source, (None, None))
        results.append(evaluate_source(patient, source, t, s,
                                       eeg_s=eeg_s, clin_s=clin_s))
        for ti, si, yi in zip(t, s, y):
            rows.append({"patient": patient, "source": source,
                         "t_start_s": ti, "label": yi, "prob": si})
    return results, str(ckpt), str(manifest_path), rows, dt_used


def probability_report(rows):
    """What the model actually outputs, by true class.

    Sensitivity, FDR/h and latency describe the DECISIONS. This describes the
    PROBABILITIES underneath them, and it is what separates two very different
    failures that produce similar-looking metrics: a model that has learned nothing
    and emits one value everywhere, versus a model that discriminates but whose
    operating threshold is set wrong. The first needs more training; the second needs
    a different DT. You cannot tell them apart from FDR/h alone.
    """
    import numpy as np
    out = {}
    for name, keep in (("interictal", lambda y: y == 0.0),
                       ("transition", lambda y: 0.0 < y < 1.0),
                       ("ictal", lambda y: y == 1.0)):
        p = np.array([r["prob"] for r in rows if keep(r["label"])])
        if len(p) == 0:
            continue
        out[name] = {"n": int(len(p)), "mean": float(p.mean()),
                     "p10": float(np.percentile(p, 10)),
                     "p50": float(np.percentile(p, 50)),
                     "p90": float(np.percentile(p, 90)),
                     "frac_above_DT": float((p > config.DECISION_THRESHOLD).mean())}
    if "interictal" in out and "ictal" in out:
        # Rank separation: the probability that a random ictal clip scores above a
        # random interictal one. 0.5 is chance. Threshold-free, so it says whether the
        # model discriminates at all, independent of where DT sits.
        #
        # [D19] Reported WITHIN recording. The pooled figure is kept beside it, but it
        # is not the headline: pooling rewards scoring one recording above another,
        # which is not detection. On fold 1 the pooled number read 0.625 for a model
        # sitting at 0.511 / 0.547 inside pat01's own two recordings.
        a = within_source_auc([r["prob"] for r in rows],
                              [r["label"] for r in rows],
                              [r["source"] for r in rows])
        out["auc_within_source"] = a["patient_balanced"]
        out["auc_within_source_pairweighted"] = a["pair_weighted"]
        out["auc_within_source_macro"] = a["macro"]
        out["auc_pooled"] = a["pooled"]
        out["auc_per_source"] = a["per_source"]
        out["n_sources_used"] = a["n_sources_used"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=str, default=None)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--model", choices=["baseline", "adapted"], default="baseline")
    ap.add_argument("--clips", type=str, default=None,
                    help="Clip manifest to score. Defaults to the fold's val_ file; "
                         "point at the Stage 3 sliding-window manifest for real "
                         "latency and FDR/h numbers.")
    ap.add_argument("--data-folder", type=str, default=None,
                    help="Root holding patches/ and kpts/. Defaults to the manifest's "
                         "own directory when that looks like a clip store, else "
                         "config.PROCESSED_DIR.")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test clip cap")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--tag", type=str, default=None,
                    help="hypernetwork checkpoint dir suffix, e.g. 'gentle'")
    ap.add_argument("--shuffled-z", type=str, default=None, metavar="patNN",
                    help="D28 control: adapt using ANOTHER patient's signature, so the "
                         "clinical gain from adapting at all can be separated from the "
                         "gain attributable to this patient's signature")
    args = ap.parse_args()

    if not args.fold and not args.all_folds:
        ap.error("pass --fold PatNN or --all-folds")

    seed_everything(config.GLOBAL_SEED)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}  cohort={len(config.COHORT)} patients  "
          f"accum={config.ACCUM_RULE}  DT={config.DECISION_THRESHOLD}  "
          f"refractory={config.REFRACTORY_S}s")

    onsets = load_onsets()
    folds = config.COHORT if args.all_folds else [args.fold.lower()]

    all_results, per_fold, skipped, all_rows = [], {}, [], []
    for p in folds:
        if p not in config.COHORT:
            print(f"[skip] {p} is not in the configured cohort "
                  f"(excluded: {config.EXCLUDED_PATIENTS})")
            continue
        try:
            res, ckpt, manifest, rows, dt_used = evaluate_fold(
                p, args.model, onsets, device, args.clips, args.limit,
                args.data_folder, tag=args.tag,
                shuffled_z=(args.shuffled_z or "").lower() or None)
        except (FileNotFoundError, NotImplementedError, KeyError, ValueError) as e:
            print(f"[skip] {p}: {e}")
            skipped.append({"patient": p, "reason": str(e)})
            continue
        agg = aggregate(res)
        probs = probability_report(rows)
        per_fold[p] = {**agg, "checkpoint": ckpt, "manifest": manifest,
                       "decision_threshold": dt_used, "probabilities": probs}
        all_results.extend(res)
        all_rows.extend(rows)
        print(f"[{p}] sens={agg['sensitivity']} "
              f"FA={agg['n_false_alarms']} over {agg['exposure_hours']:.3f} h "
              f"-> FDR/h={agg['fdr_per_hour']}  "
              f"L_EO={agg['mean_l_eo_s']}  L_CO={agg['mean_l_co_s']}")
        if probs:
            print(f"       predicted probability by true class "
                  f"(DT={config.DECISION_THRESHOLD}):")
            for k in ("interictal", "transition", "ictal"):
                if k in probs:
                    d = probs[k]
                    print(f"         {k:<11} n={d['n']:>4}  mean={d['mean']:.3f}  "
                          f"p10/50/90={d['p10']:.2f}/{d['p50']:.2f}/{d['p90']:.2f}  "
                          f"above DT={d['frac_above_DT']:.0%}")
            if "auc_within_source" in probs:
                a = probs["auc_within_source"]
                verdict = ("no discrimination -- more training, not a new threshold"
                           if a < 0.6 else
                           "discriminates; DT may simply be misplaced" if a > 0.75 else
                           "weak discrimination")
                print(f"         AUC within-source = {a:.3f}   <- {verdict}")
                print(f"         AUC pooled        = {probs['auc_pooled']:.3f}   "
                      f"(inflated by {probs['auc_pooled'] - a:+.3f}; not the headline)")
                for src, d in probs["auc_per_source"].items():
                    mark = "" if d["n_pos"] and d["n_neg"] else "   (one class only -- no pairs)"
                    auc_s = f"{d['auc']:.3f}" if d["auc"] == d["auc"] else "  n/a"
                    print(f"           {src:<20} AUC={auc_s}  "
                          f"ictal={d['n_pos']:<4} interictal={d['n_neg']:<4}{mark}")

    if not all_results:
        print("\nNo fold produced results. Nothing was evaluated -- this is a failure, "
              "not an empty result set.")
        return 1

    pooled = aggregate(all_results)
    print("\n=== POOLED ===")
    for k, v in pooled.items():
        print(f"  {k:>18}: {v}")

    out_dir = Path(args.out) if args.out else Path(config.RESULTS_DIR) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_fold.json").write_text(json.dumps(per_fold, indent=2, default=str))
    (out_dir / "pooled.json").write_text(json.dumps(pooled, indent=2, default=str))
    pd.DataFrame([r.__dict__ for r in all_results]).to_csv(
        out_dir / "per_source.csv", index=False)
    # Raw per-clip probabilities, so any later diagnostic (threshold sweeps,
    # calibration curves) can be run without re-doing inference.
    pd.DataFrame(all_rows).to_csv(out_dir / "per_clip_scores.csv", index=False)
    write_manifest(out_dir / "run_manifest.json", seed=config.GLOBAL_SEED,
                   extra={"model": args.model, "folds": list(per_fold),
                          "skipped": skipped})
    print(f"\nwrote {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
