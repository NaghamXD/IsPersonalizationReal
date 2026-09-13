"""Max-Pool Dynamic Cyclic Sampler -- methodology section 3.4.1.

Weight injection needs one z_p per forward pass, so minibatches must be
patient-homogeneous. Within a patient the classes are severely imbalanced, so:

  * Dynamic step allocation. Patient p gets
        S_p = max(|D_interictal,p|, |D_ictal,p|) / (BatchSize / 2)
    batch iterations per epoch -- proportional to their MAJORITY class volume, so a
    patient with a long interictal record contributes more steps than one with a short
    one.
  * Minority cycling. The majority pool is drawn sequentially without replacement; the
    minority pool wraps continuously (itertools.cycle). Every forward step is therefore
    exactly 50/50, and no minority instance is skipped.

[INFERRED] Transition clips carry a soft ramp label in (0, 1) and belong to neither
pool. The 50/50 rule is stated over interictal and ictal only, so transition clips are
excluded from hypernetwork training. They are still scored at evaluation time.

[DECISION] Batch ORDER is shuffled across patients within an epoch. The draft fixes
what each batch contains but not the sequence. Emitting all of patient p's S_p batches
consecutively would hand the optimiser a long run of one patient and make the update
direction strongly autocorrelated; interleaving costs nothing and avoids it.
"""
from __future__ import annotations

import itertools

import config


def _split_by_class(labels):
    """-> (interictal indices, ictal indices). Transition clips are dropped."""
    inter = [i for i, y in enumerate(labels) if float(y) == 0.0]
    ictal = [i for i, y in enumerate(labels) if float(y) == 1.0]
    return inter, ictal


def steps_for(n_interictal: int, n_ictal: int, batch_size: int) -> int:
    """S_p, per section 3.4.1. At least one step whenever both classes are present."""
    half = max(1, batch_size // 2)
    if n_interictal == 0 or n_ictal == 0:
        return 0
    return max(1, max(n_interictal, n_ictal) // half)


def build_epoch(by_patient: dict[str, list], labels_by_patient: dict[str, list],
                batch_size: int | None = None, rng=None, shuffle_batches: bool = True):
    """-> [(patient, [dataset index, ...]), ...] for one epoch.

    `by_patient[p]` holds that patient's DATASET indices; `labels_by_patient[p]` the
    matching labels, in the same order.
    """
    batch_size = config.S1_BATCH_SIZE if batch_size is None else batch_size
    half = max(1, batch_size // 2)
    batches, skipped = [], {}

    for p in sorted(by_patient):
        idx, lab = by_patient[p], labels_by_patient[p]
        if len(idx) != len(lab):
            raise ValueError(f"{p}: {len(idx)} indices but {len(lab)} labels")
        inter_pos, ictal_pos = _split_by_class(lab)
        inter = [idx[i] for i in inter_pos]
        ictal = [idx[i] for i in ictal_pos]
        s_p = steps_for(len(inter), len(ictal), batch_size)
        if s_p == 0:
            skipped[p] = (len(inter), len(ictal))
            continue

        maj, mino = (inter, ictal) if len(inter) >= len(ictal) else (ictal, inter)
        maj = list(maj)
        if rng is not None:
            rng.shuffle(maj)
            mino = list(mino); rng.shuffle(mino)
        maj_it = iter(maj)                        # sequential, without replacement
        min_it = itertools.cycle(mino)            # wraps; no minority instance skipped

        for _ in range(s_p):
            a = list(itertools.islice(maj_it, half))
            if len(a) < half:                     # majority exhausted: restart it
                maj_it = iter(maj)
                a += list(itertools.islice(maj_it, half - len(a)))
            b = [next(min_it) for _ in range(half)]
            batches.append((p, a + b))

    if shuffle_batches and rng is not None:
        rng.shuffle(batches)
    return batches, skipped


class CyclicBatchSampler:
    """torch BatchSampler protocol: yields lists of dataset indices."""

    def __init__(self, by_patient, labels_by_patient, batch_size=None, seed=0,
                 shuffle_batches=True):
        import random
        self.by_patient = by_patient
        self.labels_by_patient = labels_by_patient
        self.batch_size = config.S1_BATCH_SIZE if batch_size is None else batch_size
        self.shuffle_batches = shuffle_batches
        self._rng = random.Random(seed)
        self.epoch_patients = []
        b, self.skipped = build_epoch(by_patient, labels_by_patient, self.batch_size,
                                      rng=random.Random(seed),
                                      shuffle_batches=shuffle_batches)
        self._len = len(b)

    def __len__(self):
        return self._len

    def __iter__(self):
        batches, _ = build_epoch(self.by_patient, self.labels_by_patient,
                                 self.batch_size, rng=self._rng,
                                 shuffle_batches=self.shuffle_batches)
        self.epoch_patients = [p for p, _ in batches]
        for _p, idx in batches:
            yield idx
