"""Record and verify the Phase 1 artifacts, so nothing can be silently overwritten.

Written after a --restart smoke test destroyed fold pat01's 63-epoch training log
(D27's sibling mistake). Checksums alone are not enough: a weight file is only useful
if you also know WHICH run produced it and what role it played, so this records both.

    python scripts/freeze_artifacts.py            # write the manifest
    python scripts/freeze_artifacts.py --verify   # re-check every file against it
"""
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

MANIFEST = Path(config.OUTPUTS_DIR) / "PHASE1_ARTIFACTS.json"
DOC = Path(config.OUTPUTS_DIR) / "PHASE1_ARTIFACTS.md"

ROLES = {
    "final_model.pth":
        "THE Phase 1 baseline. D21 average of the last 5 epochs with BatchNorm "
        "recalibrated. Every baseline number in the report comes from this file.",
    "best_model.pth":
        "Byte-identical copy of final_model.pth, kept because evaluate.py's older "
        "resolution order looks for this name.",
    "best_by_val.pth":
        "The single epoch with the highest validation AUC. Kept for comparison with "
        "the pre-D21 protocol only. NOT evaluated anywhere -- D20 showed validation "
        "does not predict held-out performance.",
    "last_checkpoint.pth":
        "Full training state at epoch 50 (model, optimiser, scheduler, history). "
        "Needed to resume, not to evaluate.",
    "val_scores_by_epoch.npz":
        "Every validation clip's score at every epoch, with names, labels and source "
        "ids. The insurance added after D19 cost a retrain: a future change to the "
        "metric or the selection rule can be answered from this file.",
    "training_log.json": "Per-epoch curves.",
    "run_manifest.json": "Seed, git commit, platform, configuration at run time.",
    "hypernetwork_best.pth":
        "Stage 7 hypernetwork, lowest validation BCE. The adapted arm of the report.",
    "hypernetwork_last.pth": "Stage 7 hypernetwork at the final epoch.",
    "z_behavior.npz": "z_behavior for all 8 patients under that fold's backbone.",
    "z_blocks.npz": "Per-temporal-block z, used by the D23 identification probe.",
    "stability.json": "Section 3.2.3 stability ratios for that fold's backbone.",
}


def describe(p: Path) -> str:
    n = p.name
    if n.startswith("epoch_"):
        return (f"Periodic weights at epoch {int(n[6:9])}. D21 insurance: lets a later "
                f"change of mind about the training budget be tested without retraining.")
    if "smoke-run-do-not-use" in str(p):
        return "ARTEFACT OF A SMOKE TEST. Not a real run. Quarantined deliberately."
    if "/archive/" in str(p):
        return ("Superseded run, archived automatically by --restart rather than "
                "overwritten. Kept for provenance; not used in the report.")
    return ROLES.get(n, "")


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect():
    roots = ["outputs/lopo", "outputs/lopo_hypernetwork", "outputs/signatures",
             "outputs/thresholds", "outputs/results"]
    pats = ("*.pth", "*.npz", "*.json", "*.csv", "*.md", "*.pdf")
    files = sorted({f for r in roots for g in pats for f in Path(r).rglob(g)
                    if f.is_file()} | set(Path(config.OUTPUTS_DIR).glob("*.pdf")))
    out = []
    for f in files:
        st = f.stat()
        out.append({"path": str(f), "bytes": st.st_size, "md5": md5(f),
                    "role": describe(f),
                    "superseded": "/archive/" in str(f)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.verify:
        if not MANIFEST.exists():
            print(f"no manifest at {MANIFEST}"); return 1
        recorded = {e["path"]: e for e in json.loads(MANIFEST.read_text())["files"]}
        bad = missing = 0
        for path, e in recorded.items():
            p = Path(path)
            if not p.exists():
                print(f"  MISSING  {path}"); missing += 1; continue
            if md5(p) != e["md5"]:
                print(f"  CHANGED  {path}"); bad += 1
        print(f"\nverified {len(recorded)} files: {bad} changed, {missing} missing")
        if bad or missing:
            print("Phase 1 artifacts are NOT intact.")
            return 1
        print("all Phase 1 artifacts intact.")
        return 0

    files = collect()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = "?"
    MANIFEST.write_text(json.dumps(
        {"frozen_at": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
         "n_files": len(files),
         "total_bytes": sum(f["bytes"] for f in files), "files": files}, indent=2))

    live = [f for f in files if not f["superseded"]
            and "smoke-run" not in f["path"] and f["path"].endswith(".pth")]
    by_fold = {}
    for f in live:
        parts = Path(f["path"]).parts
        by_fold.setdefault(parts[-2], []).append(f)

    L = ["# Phase 1 artifacts — frozen inventory", "",
         f"Frozen {datetime.now(timezone.utc).date().isoformat()} at commit "
         f"`{commit[:8]}`. {len(files)} files, "
         f"{sum(f['bytes'] for f in files)/1e9:.2f} GB.", "",
         "Verify at any time with `python scripts/freeze_artifacts.py --verify`. "
         "Every md5 is recorded in `PHASE1_ARTIFACTS.json`.", "",
         "## The files the report depends on", "",
         "| file | size | md5 (first 12) | what it is |", "|---|---|---|---|"]
    seen = set()
    for name in ["final_model.pth", "hypernetwork_best.pth"]:
        for fold, fs in sorted(by_fold.items()):
            for f in fs:
                if Path(f["path"]).name == name:
                    L.append(f"| `{f['path']}` | {f['bytes']/1e6:.1f} MB | "
                             f"`{f['md5'][:12]}` | {ROLES[name].split('.')[0]}. |")
                    seen.add(f["path"])
    L += ["", "## Role of every weight file kept", "",
          "| filename | role |", "|---|---|"]
    for k, v in ROLES.items():
        if k.endswith(".pth"):
            L.append(f"| `{k}` | {v} |")
    L.append("| `epoch_0N0.pth` | Periodic weights every 10 epochs (D21 insurance). |")
    L += ["", "## Superseded and quarantined", "",
          "Runs that `--restart` archived rather than overwrote, plus smoke-test "
          "artefacts that were deliberately quarantined so `--skip-done` could not "
          "mistake them for real runs:", ""]
    for f in files:
        if f["superseded"] or "smoke-run" in f["path"]:
            L.append(f"- `{f['path']}` — {f['role']}")
    DOC.write_text("\n".join(L) + "\n")
    print(f"wrote {MANIFEST} ({len(files)} files, "
          f"{sum(f['bytes'] for f in files)/1e9:.2f} GB)")
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
