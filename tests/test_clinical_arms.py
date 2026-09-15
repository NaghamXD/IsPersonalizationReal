"""D37's clinical arm: the parts that can be checked without a GPU or the dataset."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config


def test_nine_conditions_per_fold_and_never_the_folds_own_z_as_a_control():
    from scripts.run_clinical_arms import conditions
    for fold in config.COHORT:
        c = conditions(fold)
        assert len(c) == 9, "baseline + own + 7 shuffled"
        names = [n for n, _, _ in c]
        assert names[:2] == ["baseline", "own"]
        zs = [z for _, _, z in c if z is not None]
        assert len(zs) == 7 and fold not in zs, \
            "the held-out patient's own z is the adapted condition, not a control"
        assert sorted(zs) == sorted(set(config.COHORT) - {fold})


def test_evaluate_defaults_to_the_sliding_window_manifest():
    """D37: the old default was the training-strided val_ manifest, whose ictal clips
    overlap by 4 s. FDR/h and latency computed from it are optimistic."""
    src = (ROOT / "scripts" / "evaluate.py").read_text()
    i = src.index("def evaluate_fold(")
    body = src[i:i + 2000]
    assert "test_sliding" in body
    assert body.index("test_sliding") < body.index('f"val_{patient}.json"'), \
        "the sliding manifest must be preferred, not the fallback"


def test_a_truncated_smoke_run_cannot_be_mistaken_for_a_cache():
    """--limit writes nothing: a 64-clip file left on disk would silently become the
    cached 'result' of a later full run."""
    src = (ROOT / "scripts" / "run_clinical_arms.py").read_text()
    i = src.index("def score_condition(")
    body = src[i:src.index("def clinical(")]
    assert "not restart and not limit" in body
    assert "if limit:" in body and "return by, str(ckpt), False" in body


def test_the_selected_threshold_is_required_not_inherited():
    """D16/D37: an inherited DT makes FDR/h incomparable across folds, and the whole
    analysis is a difference of rates."""
    src = (ROOT / "scripts" / "run_clinical_arms.py").read_text()
    assert "SystemExit" in src and "select_threshold.py" in src
    assert "THRESHOLDS_DIR" in src
