# Protocol

## Target and circuit

The target is `p(x) ∝ exp(beta * score(x))` on even-parity bitstrings. Bits use
big-endian ordering. The score is the longest internal, linear zero run bracketed
by ones; strings with fewer than two ones score zero. Each dataset contains
200 IID observations with replacement.

The IQP circuit is `H^n [product RZZ(theta_ij)] H^n`, where
`RZZ(theta) = exp(-i theta ZiZj / 2)`. Pairwise couplings are cyclic and each
edge has one trainable angle. The circuit preserves even output parity.

## Losses and evaluation

The parity objective is
`mean((Walsh(q)[mask] - Walsh(empirical)[mask])**2)`.
Mask bits are sampled independently with probability
`(1-exp(-1/(2*sigma^2)))/2`. All-zero masks are resampled; duplicates keep
their weights. Probability MSE averages squared errors over the valid support;
the rescaled MSE variant sums them.

Exact forward KL uses full model normalization and positive target mass.
Missing model mass gives infinite KL. Conditional KL evaluates distributions
after normalization on even-parity support.

The unseen elite contains valid states at or above the 90th-percentile score,
including ties, that do not occur in training. Expected discoveries at Q draws
are `sum(1 - (1-q[x])**Q)` over this elite. Recovery divides by elite size;
coverage divides by Q.

## Main benchmarks

| Benchmark | Architectures | Beta | Confirmation datasets | Selection |
|---|---|---:|---:|---|
| Architecture comparison | 12, 24, 36, 48, 60, 66 parameters | 0.9 | 120 | Independent validation likelihood for each architecture and loss |
| Classical comparison | 36-parameter IQP and classical models | 0.9 | 120 | Validation-selected IQP; fixed main classical configurations |

Both benchmarks use 200 training observations per dataset and 600 updates for
the main models. They use separate confirmation cohorts. Paired losses share
training observations and initial angles. Adam for IQP uses beta1 = 0.9,
beta2 = 0.99 and epsilon = 1e-8 before second-moment bias correction.

[Architecture settings](../protocols/parameter-ladder-confirmation.json) ·
[Classical settings](../protocols/loss-advantage-confirmation.json)

## Fixed-band preset

`iqp-repro run --preset paper` uses the 24-parameter circuit, beta = 0.1–2.0,
seeds 111–120, sigma = 1, K = 512 and 600 updates. The default `matched`
protocol shares IQP initial angles and uses the valid-support MSE denominator.
The `source` protocol uses separate IQP-MSE initialization, a full-cube MSE
denominator and a sorted top-10% elite with truncated ties.

For seed r, data use r+7, masks use r+222, and IQP angles use
`0.01 * default_rng(r+10000+7*K).standard_normal(number_of_edges)`.
The source IQP-MSE initialization uses r+20000+7*512.
Across beta values, paired intervals first average within each seed.

## Limitations

Training enumerates the full probability distribution and uses exponential
time and memory in qubit count. The fixed-band preset reuses historical datasets;
it is separate from the main 120-dataset confirmation benchmarks.

[Reproduction](reproduce.md) · [Corrections](corrections.md)
