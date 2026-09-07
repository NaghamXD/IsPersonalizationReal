"""§3.2.3 signature stability. NumPy only -- no torch, no model.

[METHOD] "A Stability Ratio of >= 2.5 guarantees that the spatial motor signature
successfully dominates arbitrary intra-patient activity shifts, providing a reliable
baseline for personalized weight adaptation."

This is a GATE, not a diagnostic: if a patient's signature is not stable across their
own day, conditioning on it cannot work, and that is far cheaper to learn here than
after eight folds of hypernetwork training.
"""
import itertools

import numpy as np

import config


def stability_ratio(block_vectors: dict):
    """[METHOD] §3.2.3 signature stability.

    `block_vectors`: {patient: [z per temporal block]}. Returns the ratio of mean
    inter-patient distance to mean intra-patient distance, plus the per-patient
    intra-patient means -- because the pooled number alone is not interpretable here.

    WHY PER PATIENT MATTERS (DECISIONS.md D14): Pool A time spans differ 24-fold
    across this cohort (pat09 150 s, pat01 3570 s). A patient whose Pool A spans two
    minutes has blocks minutes apart, so its intra-patient variance is small for a
    reason that has nothing to do with signature stability. Pooling to one ratio and
    testing it against 2.5 would partly be testing recording length.
    """
    import itertools

    import numpy as np

    intra = {}
    for p, blocks in block_vectors.items():
        d = [float(np.linalg.norm(a - b)) for a, b in itertools.combinations(blocks, 2)]
        intra[p] = float(np.mean(d)) if d else float("nan")

    inter = []
    for p, q in itertools.combinations(sorted(block_vectors), 2):
        for a in block_vectors[p]:
            for b in block_vectors[q]:
                inter.append(float(np.linalg.norm(a - b)))
    inter_mean = float(np.mean(inter)) if inter else float("nan")

    per_patient = {p: (inter_mean / v if v > 0 else float("inf"))
                   for p, v in intra.items()}
    pooled_intra = float(np.mean([v for v in intra.values() if v == v]))
    return {
        "inter_mean": inter_mean,
        "intra_per_patient": intra,
        "ratio_per_patient": per_patient,
        "pooled_ratio": inter_mean / pooled_intra if pooled_intra > 0 else float("inf"),
        "threshold": config.STABILITY_RATIO_THRESHOLD,
    }
