"""Pool A / Pool B construction.

[METHOD] Pool A holds N <= 20 interictal clips per patient, "sampled using a
stratified uniform distribution across the patient's entire available interictal
timeline" so the signature captures an invariant resting state rather than one
transient activity. Pool B holds everything else and supplies the training gradient.
"A strict temporal guardrail guarantees that no overlapping windows exist between
Pool A and Pool B, eliminating intra-patient data leakage."

The base repo took `interictal[:20]` from a lexicographic filename sort. That is
neither stratified nor uniform nor even chronological -- `pat01_Sz1_1000` sorts before
`pat01_Sz1_200` -- so Pool A was an artifact of string comparison.

STRATIFICATION IS BY SOURCE RECORDING. Each source is a different stretch of the
patient's stay, so a patient with 1 clip from a seizure file and 40 from an 11-minute
`free.mp4` should draw from both in proportion, and evenly within each, rather than
concentrating on whichever happens to sort first.
"""
from dataclasses import dataclass, field

import numpy as np

import config
from src.utils.naming import parse_clip_name


@dataclass
class PatientPools:
    patient: str
    pool_a: list = field(default_factory=list)
    pool_b_interictal: list = field(default_factory=list)
    pool_b_ictal: list = field(default_factory=list)
    n_transition_excluded: int = 0
    n_boundary_excluded: int = 0     # interictal clips barred from Pool A for
                                     # overlapping the transition/ictal boundary
    low_confidence: bool = False
    per_source: dict = field(default_factory=dict)


def largest_remainder(counts, total_wanted):
    """Allocate `total_wanted` across strata proportionally to `counts`.

    Largest-remainder rather than naive rounding, so the allocation sums exactly to
    the target and small strata are not silently rounded out of existence.
    """
    total = sum(counts)
    if total == 0:
        return [0] * len(counts)
    if total_wanted >= total:
        return list(counts)
    raw = [c / total * total_wanted for c in counts]
    base = [int(x) for x in raw]
    remainder = total_wanted - sum(base)
    order = sorted(range(len(counts)), key=lambda i: (raw[i] - base[i]), reverse=True)
    for i in order[:remainder]:
        base[i] += 1
    return [min(b, c) for b, c in zip(base, counts)]


def evenly_spaced(items, k):
    """k items spread across `items`, endpoints included. Not the first k."""
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, k).round().astype(int))
    return [items[i] for i in idx]


def overlaps(a, b, clip_seconds=None):
    """Do two clips share any footage? Same source and closer than one clip length."""
    clip_seconds = config.CLIP_SECONDS if clip_seconds is None else clip_seconds
    pa, sa, ta = parse_clip_name(a)
    pb, sb, tb = parse_clip_name(b)
    return pa == pb and sa == sb and abs(ta - tb) < clip_seconds


def build_pools(patient, clips, pool_a_size=None, guardrail_s=None):
    """clips: [(name, label)] for ONE patient. -> PatientPools.

    Transition clips are dropped entirely: [METHOD] the hypernetwork optimises a
    binary personalisation delta, and soft ambiguous labels would put structural
    noise into it.
    """
    pool_a_size = config.POOL_A_SIZE if pool_a_size is None else pool_a_size
    guardrail_s = config.POOL_GUARDRAIL_S if guardrail_s is None else guardrail_s

    interictal, ictal, n_trans = [], [], 0
    for name, label in clips:
        if label == 0.0:
            interictal.append(name)
        elif label == 1.0:
            ictal.append(name)
        else:
            n_trans += 1

    # POOL A CANDIDACY. A clip is eligible for the signature only if it does not
    # share footage with any ictal or transition clip.
    #
    # Why this is not the same as "label == 0.0": the transition ramp is evaluated at
    # a clip's END, so a window ending exactly AT EEG onset gets x = 0 and therefore a
    # label of exactly 0.0. By label it looks interictal; by phase it is the first
    # transition window. It also sits where ictal clips are extracted at 1 s stride,
    # so it overlaps them -- which is how the guardrail caught it.
    #
    # Semantically this is the right filter regardless of the label edge case: the
    # last five seconds before electrographic onset are not a resting motor state, and
    # the signature is supposed to characterise resting behaviour.
    non_interictal = [n for n, lab in clips if lab != 0.0]
    excluded_boundary = [n for n in interictal
                         if any(overlaps(n, o, guardrail_s) for o in non_interictal)]
    boundary = set(excluded_boundary)
    candidates = [n for n in interictal if n not in boundary]

    # Group by source recording and order each group chronologically -- by parsed
    # start time, never by filename string.
    by_source = {}
    for name in candidates:
        _, source, t = parse_clip_name(name)
        by_source.setdefault(source, []).append((t, name))
    for s in by_source:
        by_source[s].sort()

    sources = sorted(by_source)
    counts = [len(by_source[s]) for s in sources]
    alloc = largest_remainder(counts, pool_a_size)

    pool_a, per_source = [], {}
    for s, k in zip(sources, alloc):
        names = [n for _t, n in by_source[s]]
        picked = evenly_spaced(names, k)
        pool_a.extend(picked)
        per_source[s] = {"available": len(names), "allocated": k, "picked": len(picked)}

    chosen = set(pool_a)
    pool_b_inter = []
    for name in interictal:
        if name in chosen:
            continue
        # Temporal guardrail: drop any Pool B candidate sharing footage with Pool A.
        # With non-overlapping interictal extraction this rarely fires, but it is the
        # methodology's stated requirement and must not depend on a stride choice
        # made elsewhere in the pipeline.
        if any(abs(parse_clip_name(name)[2] - parse_clip_name(a)[2]) < guardrail_s
               and parse_clip_name(name)[1] == parse_clip_name(a)[1]
               for a in pool_a):
            continue
        pool_b_inter.append(name)

    return PatientPools(
        patient=patient,
        pool_a=sorted(pool_a, key=lambda n: (parse_clip_name(n)[1], parse_clip_name(n)[2])),
        pool_b_interictal=sorted(pool_b_inter,
                                 key=lambda n: (parse_clip_name(n)[1], parse_clip_name(n)[2])),
        pool_b_ictal=sorted(ictal,
                            key=lambda n: (parse_clip_name(n)[1], parse_clip_name(n)[2])),
        n_transition_excluded=n_trans,
        n_boundary_excluded=len(excluded_boundary),
        low_confidence=len(pool_a) < config.POOL_A_MIN_CONFIDENCE,
        per_source=per_source,
    )


def verify_guardrail(pools: PatientPools):
    """Assert no Pool B clip shares footage with any Pool A clip. Returns violations."""
    bad = []
    a_by_source = {}
    for a in pools.pool_a:
        _, s, t = parse_clip_name(a)
        a_by_source.setdefault(s, []).append(t)
    for b in pools.pool_b_interictal + pools.pool_b_ictal:
        _, s, t = parse_clip_name(b)
        for ta in a_by_source.get(s, []):
            if abs(t - ta) < config.CLIP_SECONDS:
                bad.append((b, s, t, ta))
                break
    return bad
