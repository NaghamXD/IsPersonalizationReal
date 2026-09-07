"""End-to-end sanity check. Run this first on any new machine.

    python scripts/smoke_test.py

Checks, in order: config loads and is self-consistent; the torch-free logic passes its
tests; the model builds and its true tensor shapes match what config asserts; and one
real clip loads through the dataset if preprocessed data is present.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED = []
SKIPPED = []


class Skip(Exception):
    """A check that could not run here, as distinct from one that ran and failed."""


def step(name, fn):
    print(f"\n--- {name} ---")
    try:
        fn()
        print(f"  PASS  {name}")
    except Skip as e:
        print(f"  SKIP  {name}: {e}")
        SKIPPED.append(name)
    except ModuleNotFoundError as e:
        # A missing dependency is an environment gap, not a broken pipeline. Saying
        # "FAIL" here would train you to ignore this script's output.
        print(f"  SKIP  {name}: {e} (install requirements.txt)")
        SKIPPED.append(name)
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        FAILED.append(name)


def check_all_modules_import():
    """Import every module under src/ so a missing dependency surfaces here, as a
    named module, rather than as a traceback from whichever script happened to run
    first. scripts/ is excluded: preprocess.py executes its whole pipeline at import.
    """
    import importlib
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as e:
        raise Skip(f"{e} -- run this on the training machine") from None
    failures = []
    for path in sorted((ROOT / "src").glob("**/*.py")):
        if path.name == "__init__.py":
            continue
        mod = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:
            failures.append(f"{mod}: {e}")
        except Exception as e:
            failures.append(f"{mod}: {type(e).__name__}: {e}")
    if failures:
        raise AssertionError("modules failed to import:\n    " + "\n    ".join(failures))
    print(f"  all {len(list((ROOT/'src').glob('**/*.py')))} modules under src/ import cleanly")


def check_config():
    import config
    assert len(config.COHORT) == 8, config.COHORT
    assert config.PROJECTOR_IN_DIM == 2 * config.SIGNATURE_CHANNELS
    assert config.S3_BATCH_SIZE % 2 == 0, "step-3 batches must split 50/50"
    assert config.POOL_A_SIZE >= 1
    assert set(config.ABLATION_FOLDS) <= set(config.COHORT)
    assert set(config.EXCLUDED_PATIENTS).isdisjoint(config.COHORT)
    print(f"  cohort={config.COHORT}")


def _run(script, tail=8):
    """Run a child script, echoing BOTH streams. Capturing stdout only -- as this did
    originally -- hides the traceback that says what actually broke, which turns a
    precise failure into 'something went wrong'."""
    r = subprocess.run([sys.executable, script], cwd=ROOT, capture_output=True, text=True)
    for line in (r.stdout or "").strip().splitlines()[-tail:]:
        print("  " + line)
    if r.returncode != 0:
        err = (r.stderr or "").strip()
        if err:
            print("  --- stderr ---")
            for line in err.splitlines()[-25:]:
                print("  " + line)
    return r


def run_pytests():
    for t in ["tests/test_normalize.py", "tests/test_eval.py", "tests/test_windows.py",
                     "tests/test_pools.py", "tests/test_signature.py"]:
        r = _run(t, tail=3)
        if r.returncode != 0:
            raise AssertionError(f"{t} failed (see stderr above)")


def run_verify_shapes():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as e:
        raise Skip(f"{e} -- run this on the training machine") from None
    r = _run("scripts/verify_shapes.py", tail=30)
    if r.returncode != 0:
        raise AssertionError("verify_shapes.py failed -- see output above. Either a "
                             "config constant disagrees with the real model, or the "
                             "forward pass raised.")


def load_one_clip():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as e:
        raise Skip(f"{e} -- run this on the training machine") from None
    import config
    from src.data.dataset import VSViGDataset
    if not Path(config.LABELS_JSON).exists():
        print(f"  (skipped: no {config.LABELS_JSON} yet -- preprocessing not run here)")
        return
    ds = VSViGDataset(config.PROCESSED_DIR, config.LABELS_JSON)
    sample, target = ds[0]
    d, k = sample["data"], sample["kpts"]
    print(f"  patches={tuple(d.shape)} kpts={tuple(k.shape)} label={target}")
    assert k.shape[-1] == config.KPT_CHANNELS
    assert d.shape[0] == config.CLIP_FRAMES and d.shape[1] == config.N_JOINTS
    import torch
    assert torch.isfinite(k).all(), "normalised keypoints contain inf/NaN"


if __name__ == "__main__":
    step("config self-consistency", check_config)
    step("all src/ modules import", check_all_modules_import)
    step("torch-free unit tests", run_pytests)
    step("model shapes vs config", run_verify_shapes)
    step("dataset loads one real clip", load_one_clip)
    print()
    if FAILED:
        print(f"SMOKE TEST FAILED: {', '.join(FAILED)}")
    elif SKIPPED:
        print(f"SMOKE TEST PASSED (skipped here: {', '.join(SKIPPED)})")
    else:
        print("SMOKE TEST PASSED")
    sys.exit(1 if FAILED else 0)
