# Can z_behavior identify its own patient? Yes — and that is the problem.

Each patient's Pool A is split into 4 contiguous temporal blocks, a z computed per
block, and every block identified by nearest patient centroid with that block left out
of its own centroid. Chance is 1/8 = 12.5%. Significance by label permutation.

| backbone fold | identification | permutation null | p |
|---|---|---|---|
| pat01 | 59.4% | 13.2% | 0.0020 |
| pat02 | 50.0% | 13.5% | 0.0020 |
| pat03 | 46.9% | 13.2% | 0.0020 |
| pat04 | 50.0% | 11.7% | 0.0020 |
| pat06 | 59.4% | 12.4% | 0.0020 |
| pat07 | 62.5% | 11.7% | 0.0020 |
| pat08 | 53.1% | 12.5% | 0.0020 |
| pat09 | 62.5% | 12.3% | 0.0020 |

Mean **55.5%**, range 46.9%–62.5%, every fold significant.
z_behavior carries patient identity at roughly 4.4x chance.

## This overturns the pessimistic reading of §3.2.3

The stability gate said the signature was barely usable (13 of 64 ratios ≥ 2.5, mean
1.94). The identification probe says it is strongly patient-discriminative. Both are
correct: the ratio compares mean intra- to mean inter-patient Euclidean distance, and in
128 dimensions those concentrate, so ratios near 1 are unremarkable even when the
structure is clean. Identification only requires the right centroid to be *nearest*.

**The §3.2.3 threshold of 2.5 is therefore the wrong instrument for the question it is
asked to answer.** It would have rejected a signature that works. Recommend
pre-registering the identification probe as the gate instead, with its permutation test.

## The finding that matters

| patient | identification recall | stability ratio | baseline AUC |
|---|---|---|---|
| pat03 | 81% | 2.35 | 0.829 |
| pat04 | 72% | 2.85 | 0.700 |
| pat02 | 69% | 2.71 | 0.769 |
| pat08 | 59% | 1.14 | 0.741 |
| pat07 | 56% | 2.14 | 0.918 |
| pat06 | 53% | 1.38 | 0.543 |
| pat09 | 28% | 1.34 | 0.180 |
| pat01 | 25% | 1.65 | 0.562 |

- corr(recall, stability ratio) = +0.644
- **corr(recall, baseline AUC) = +0.726**
- corr(keypoint validity, recall) = +0.014; corr(keypoint validity, baseline AUC) = +0.097
- partial corr(recall, AUC | keypoint validity) = **+0.729**

**Signature recoverability and baseline performance are strongly coupled, and pose
tracking does not explain it.** pat06 has the worst keypoint validity in the cohort
(63.1%) and middling recall; pat01 and pat09 have among the best validity (96.3%, 96.1%)
and the worst recall (25%, 28%). Controlling for validity leaves the coupling untouched.

The natural reading: both quantities are downstream of how well the frozen backbone
represents that patient at all. Where its features do not separate that patient's
seizures, they also do not pin down that patient's identity.

## What this predicts about Stage 7, and what to pre-register

§3.5 predicts benefit ∝ atypicality. That is already unsupported (Pearson −0.003 over 24
pairs; −0.143 over the 8 held-out). This probe supplies a **competing prediction that
should be registered before Stage 7 runs**:

> **benefit ∝ signature recoverability** — personalization helps most where the baseline
> is already strongest, because that is where z is informative. The patients who need it
> (pat09 0.180, pat01 0.562, pat06 0.543) are the ones whose z is least recoverable
> (28%, 25%, 53%).

If that is what comes out, the method works but not for the reason the draft gives, and
the honest headline is narrower than "personalization helps atypical patients".

One thing this does settle in the method's favour: **the shuffled-z control is a real
manipulation.** z carries identity at 55%, so swapping in the wrong
patient's z changes something genuine. Had the probe come back at chance, that control
would have been vacuous and Stage 8 could not have detected its own failure.
