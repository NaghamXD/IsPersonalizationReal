"""Leave-one-patient-out folds with semiology-stratified internal validation.

Two properties this has to guarantee, and the base repo guaranteed neither:

1. NO LEAK. The held-out patient's clips must never reach training or model
   selection. At 6fca412, `split_lopo.py` wrote the test patient's clips to
   `val_{patient}.json` and `train_lopo.py` used that file for early stopping and
   best-checkpoint selection -- selecting the model on the patient it was supposed to
   generalise to.

2. SEMIOLOGY STRATIFICATION. [METHOD] the validation pair is "one patient presenting
   with focal seizures and one patient presenting with focal-to-bilateral tonic-clonic",
   so the pooled validation loss sees both subtle asymmetric kinematics and whole-body
   convulsion. The retained cohort is 4 PG and 4 P, so this is drawable in every fold.
"""
import random
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Fold:
    test_patient: str
    val_patients: tuple
    train_patients: tuple

    def describe(self, semiology):
        def tag(p):
            return f"{p}({semiology[p]})"
        return (f"test={tag(self.test_patient)}  "
                f"val=[{', '.join(tag(p) for p in self.val_patients)}]  "
                f"train=[{', '.join(tag(p) for p in self.train_patients)}]")


def make_fold(test_patient, cohort=None, semiology=None, seed=None, fold_index=0):
    """One LOPO fold with a stratified validation pair."""
    cohort = cohort or config.COHORT
    semiology = semiology or config.SEMIOLOGY
    seed = config.FOLD_SEED if seed is None else seed

    assert test_patient in cohort, f"{test_patient} not in cohort"
    others = [p for p in cohort if p != test_patient]
    pg = sorted(p for p in others if semiology[p] == "PG")
    focal = sorted(p for p in others if semiology[p] == "P")

    if not pg or not focal:
        raise ValueError(
            f"fold {test_patient}: cannot draw a stratified validation pair -- "
            f"PG available {pg}, P available {focal}. The cohort must retain at least "
            f"one of each semiology after the test patient is removed.")

    # Seeded per fold, so which patients sit out rotates across folds rather than the
    # same two always being excluded from training -- but is reproducible.
    rng = random.Random(seed + fold_index)
    val = (rng.choice(pg), rng.choice(focal))
    train = tuple(p for p in others if p not in val)
    return Fold(test_patient=test_patient, val_patients=tuple(sorted(val)),
                train_patients=tuple(sorted(train)))


def make_all_folds(cohort=None, semiology=None, seed=None):
    cohort = cohort or config.COHORT
    return [make_fold(p, cohort, semiology, seed, fold_index=i)
            for i, p in enumerate(sorted(cohort))]


def verify_no_leak(folds):
    """The test patient must appear in neither the training nor validation group."""
    bad = []
    for f in folds:
        if f.test_patient in f.train_patients:
            bad.append((f.test_patient, "in train"))
        if f.test_patient in f.val_patients:
            bad.append((f.test_patient, "in val"))
        if set(f.train_patients) & set(f.val_patients):
            bad.append((f.test_patient, "train/val overlap"))
    return bad
