# What does the 384 → 128 projection cost z_behavior?

Identification probe (D23) re-run on the pre-projection features and on random
projections at seven widths. Chance 12.5%; every variant significant (p ≤ 0.023).

| variant | pat01 | pat02 | pat03 | pat04 | pat06 | pat07 | pat08 | pat09 | mean | sd |
|---|---|---|---|---|---|---|---|---|---|---|
| `raw_384_mu+sigma` | 65.6% | 56.2% | 50.0% | 50.0% | 53.1% | 59.4% | 53.1% | 68.8% | **57.0%** | 6.6% |
| `mu_only_192` | 59.4% | 59.4% | 62.5% | 46.9% | 50.0% | 50.0% | 43.8% | 71.9% | **55.5%** | 8.8% |
| `sigma_only_192` | 50.0% | 34.4% | 34.4% | 31.2% | 34.4% | 50.0% | 56.2% | 28.1% | **39.8%** | 9.9% |
| `proj_256` | 62.5% | 53.1% | 53.1% | 46.9% | 56.2% | 62.5% | 53.1% | 68.8% | **57.0%** | 6.6% |
| `proj_128` | 59.4% | 50.0% | 46.9% | 50.0% | 59.4% | 62.5% | 53.1% | 62.5% | **55.5%** | 5.8% |
| `proj_64` | 53.1% | 56.2% | 50.0% | 50.0% | 56.2% | 65.6% | 53.1% | 65.6% | **56.2%** | 5.8% |
| `proj_32` | 50.0% | 53.1% | 43.8% | 46.9% | 59.4% | 59.4% | 56.2% | 71.9% | **55.1%** | 8.3% |
| `proj_16` | 50.0% | 59.4% | 43.8% | 46.9% | 53.1% | 53.1% | 43.8% | 65.6% | **52.0%** | 7.1% |
| `proj_8` | 50.0% | 56.2% | 37.5% | 46.9% | 62.5% | 53.1% | 40.6% | 65.6% | **51.6%** | 9.2% |

## 1. The projection costs essentially nothing

384 → 128 loses **1.5 points** (57.0% → 55.5%), against a fold-to-fold sd of ~6%. Even
**8 dimensions retains 51.6%**. This is textbook Johnson–Lindenstrauss: only 32 points
(8 patients × 4 blocks) have to stay separated, and that needs O(log n / ε²) ≈ tens of
dimensions. At 128 the projection is not compressing anything that matters.

**Practical answer: keeping the raw 384-d vector would buy nothing.** Don't change it.

## 2. But that refutes the reason §3.2.2 gives for having it

The methodology calls the projection "a structural information bottleneck, preventing
the downstream hypernetwork from memorizing categorical patient IDs". The measurement
says patient identity survives the bottleneck **intact** — 55.5% at 128 versus 57.0% at
384. A hypernetwork that wanted to recover patient ID from z could do so about as well
as from the raw features.

Nor is there a width that would fix this: identity is still 51.6% at 8 dimensions, four
times chance. The anti-memorisation property cannot be obtained by narrowing the
projection, because identity is not encoded in a few high-variance directions — it is
spread across the representation.

The projection should therefore be justified as what it is — a cheap fixed-width
interface to the hypernetwork — and the memorisation claim dropped, or defended by some
mechanism that actually constrains identity (an adversarial or invariance penalty, which
this design does not have).

## 3. σ contributes little; the signature is mostly μ

- μ alone (192-d): **55.5%** — identical to the full 384-d projection
- σ alone (192-d): **39.8%**, and weak on several folds (28.1% pat09, 31.2% pat04)

The variability half of [μ ‖ σ], the part meant to capture *how a patient moves*, carries
markedly less patient identity than the mean half. z_behavior is close to a mean-feature
descriptor.

**This matters for what the signature is called.** If identity rides mostly on the mean
of stage-2 features, z may be encoding static appearance — body habitus, bed and camera
geometry, lighting — rather than motor behaviour. "Behavioural signature" would then be
a misnomer, and §3.5's atypicality axis would not be about movement at all.

That is a confound, not a finding: this probe cannot separate the two. The clean test
would be whether z stays stable for one patient across a change of position or camera
while differing between patients recorded under matched conditions. Worth stating as a
limitation before Stage 7 either way.
