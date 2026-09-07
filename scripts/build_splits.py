"""Stage 4 -- LOPO folds and Pool A / Pool B manifests.

Replaces scripts/split_lopo.py, whose filenames actively misled: it wrote the HELD-OUT
patient's clips to `val_{patient}.json`, and train_lopo.py then used that file for
early stopping and best-checkpoint selection -- selecting the model on the very patient
it was meant to generalise to. Nothing about the name warned you.

The layout here says what each file is:

    processed_data/folds/{patient}/fold.json         who is in which group, and why
    processed_data/folds/{patient}/train_clips.json  the 5 training patients
    processed_data/folds/{patient}/val_clips.json    the 2 validation patients
    processed_data/folds/{patient}/test_clips.json   HELD OUT -- final reporting only
    processed_data/pools/{patient}.json              Pool A / Pool B
    processed_data/pools/summary.md

`test_clips.json` is the only file naming the held-out patient, and nothing in
training may open it.

    python scripts/build_splits.py
    python scripts/build_splits.py --dry-run
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data.pools import build_pools, verify_guardrail
from src.data.splits import make_all_folds, verify_no_leak
from src.utils.manifest import write_manifest
from src.utils.naming import patient_of
from src.utils.seeding import seed_everything


def load_clips_by_patient():
    labels = json.loads(Path(config.LABELS_JSON).read_text())
    by_patient = defaultdict(list)
    skipped = defaultdict(int)
    for name, label in labels.items():
        p = patient_of(name)
        if p not in config.COHORT:
            skipped[p] += 1
            continue
        by_patient[p].append([name, float(label)])
    for p in by_patient:
        by_patient[p].sort()
    return by_patient, dict(skipped)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seed_everything(config.GLOBAL_SEED)
    if not Path(config.LABELS_JSON).exists():
        print(f"No {config.LABELS_JSON}. Run scripts/preprocess.py first.")
        return 1

    by_patient, skipped = load_clips_by_patient()
    print(f"Loaded clips for {len(by_patient)} patients from {config.LABELS_JSON}")
    if skipped:
        print(f"  ignored (not in cohort): "
              f"{', '.join(f'{k}={v}' for k, v in sorted(skipped.items()))}")
    missing = [p for p in config.COHORT if p not in by_patient]
    if missing:
        print(f"  ERROR: cohort patients with no clips: {missing}")
        return 1

    # ---------------------------------------------------------------- folds
    folds = make_all_folds()
    leaks = verify_no_leak(folds)
    if leaks:
        print(f"  ERROR: data-separation violated: {leaks}")
        return 1
    print(f"\n=== {len(folds)} LOPO folds "
          f"(1 test / {config.N_INTERNAL_VAL_PATIENTS} val / "
          f"{len(config.COHORT) - 1 - config.N_INTERNAL_VAL_PATIENTS} train) ===")
    for f in folds:
        print(f"  {f.describe(config.SEMIOLOGY)}")

    # ---------------------------------------------------------------- pools
    print(f"\n=== Pool A (N={config.POOL_A_SIZE}, {config.POOL_A_SAMPLING}) / Pool B ===")
    print(f"  {'patient':<9}{'PoolA':>7}{'B_inter':>9}{'B_ictal':>9}{'trans':>7}"
          f"   sources (available -> picked)")
    all_pools, any_bad = {}, False
    for p in config.COHORT:
        pools = build_pools(p, by_patient[p])
        bad = verify_guardrail(pools)
        if bad:
            any_bad = True
            print(f"  ERROR {p}: {len(bad)} guardrail violation(s), e.g. {bad[0]}")
        src = "  ".join(f"{s}:{d['available']}->{d['picked']}"
                        for s, d in sorted(pools.per_source.items()))
        flag = "  [LOW-CONFIDENCE]" if pools.low_confidence else ""
        print(f"  {p:<9}{len(pools.pool_a):>7}{len(pools.pool_b_interictal):>9}"
              f"{len(pools.pool_b_ictal):>9}{pools.n_transition_excluded:>7}"
              f"   {src}{flag}")
        all_pools[p] = pools
    if any_bad:
        print("\nGuardrail violated -- refusing to write manifests.")
        return 1

    empty_b = [p for p, x in all_pools.items() if not x.pool_b_interictal]
    if empty_b:
        print(f"\n  WARNING: empty interictal Pool B for {empty_b} -- the cyclic "
              f"sampler cannot build a balanced batch for these patients, so they "
              f"contribute no hypernetwork gradient.")

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return 0

    # ---------------------------------------------------------------- write
    folds_dir, pools_dir = Path(config.FOLDS_DIR), Path(config.POOLS_DIR)
    for f in folds:
        d = folds_dir / f.test_patient
        d.mkdir(parents=True, exist_ok=True)
        train = [c for p in f.train_patients for c in by_patient[p]]
        val = [c for p in f.val_patients for c in by_patient[p]]
        test = list(by_patient[f.test_patient])
        (d / "fold.json").write_text(json.dumps({
            "test_patient": f.test_patient,
            "val_patients": list(f.val_patients),
            "train_patients": list(f.train_patients),
            "semiology": {p: config.SEMIOLOGY[p]
                          for p in (f.test_patient, *f.val_patients, *f.train_patients)},
            "n_train_clips": len(train), "n_val_clips": len(val),
            "n_test_clips": len(test),
            "note": "test_clips.json is held out for final reporting only and must "
                    "never be opened during training or model selection.",
        }, indent=2))
        (d / "train_clips.json").write_text(json.dumps(train))
        (d / "val_clips.json").write_text(json.dumps(val))
        (d / "test_clips.json").write_text(json.dumps(test))

    pools_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p, pools in all_pools.items():
        (pools_dir / f"{p}.json").write_text(json.dumps({
            "patient": p,
            "pool_a": pools.pool_a,
            "pool_b_interictal": pools.pool_b_interictal,
            "pool_b_ictal": pools.pool_b_ictal,
            "per_source": pools.per_source,
            "n_transition_excluded": pools.n_transition_excluded,
            "low_confidence": pools.low_confidence,
        }, indent=2))
        rows.append((p, len(pools.pool_a), len(pools.pool_b_interictal),
                     len(pools.pool_b_ictal), pools.n_transition_excluded))

    with open(pools_dir / "summary.md", "w") as fh:
        fh.write(f"# Pool A / Pool B — {len(config.COHORT)} patients, "
                 f"N={config.POOL_A_SIZE}\n\n")
        fh.write("| patient | semiology | Pool A | Pool B interictal | Pool B ictal | "
                 "transition excluded |\n|---|---|---|---|---|---|\n")
        for p, a, bi, bc, t in rows:
            fh.write(f"| {p} | {config.SEMIOLOGY[p]} | {a} | {bi} | {bc} | {t} |\n")

    write_manifest(Path(config.PROCESSED_DIR) / "run_manifest_splits.json",
                   seed=config.GLOBAL_SEED,
                   extra={"stage": "4_splits_and_pools",
                          "folds": [f.test_patient for f in folds],
                          "pool_a_size": config.POOL_A_SIZE})
    print(f"\nwrote {folds_dir}/ and {pools_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
