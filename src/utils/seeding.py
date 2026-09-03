"""Global determinism. Import and call seed_everything() at the top of every entry point.

The base repo seeded random.Random(0) in one place and torch/numpy nowhere, so two
runs of the same fold did not agree. That matters here more than usual: the headline
result is a rate model fitted over 11 points, and per-fold noise of the same order as
the effect would make it unfalsifiable.
"""
import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # MPS has no deterministic-algorithms guarantee; this is best-effort and
        # intentionally does not raise on ops that lack a deterministic kernel.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fold_seed(base_seed: int, fold_index: int) -> int:
    """Per-fold seed. Distinct across folds, reproducible given base_seed."""
    return base_seed + 1000 * (fold_index + 1)
