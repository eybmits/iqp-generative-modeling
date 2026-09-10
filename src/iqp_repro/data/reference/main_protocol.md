# Benchmark Reporting Protocol

This document defines the active benchmark-side disclosure standard for the
curated analysis reruns in this repository. It mirrors
`STANDARD_TRAINING_PROTOCOL.md` and the saved `RUN_CONFIG.json` files under
`plots/`.

## 1. Matched-Instance Definition

The active final-reporting standard uses `10` matched seeds:

- matched seed IDs: `111, 112, ..., 120`
- beta grid: `0.1, 0.2, ..., 2.0`
- matched-instance index: `(beta, seed)`
- total matched instances for the wide beta sweep: `20 betas x 10 seeds = 200`

Within each matched instance, all models receive the same `D_train`.
Parity-based models additionally receive the same parity band `Omega`.
The canonical seed schedule is `docs/benchmark_seed_schedule_10seeds.csv`.

## 2. Randomness Stack

For each matched instance `(beta, seed)`:

1. Build `p^*` on the even-parity support.
2. Sample the training multiset `D_train` with seed `seed + 7`.
3. Construct the empirical distribution from `D_train`.
4. Sample the parity band `Omega` with seed `seed + 222`.
5. Initialize each model once with its model-specific seed.
6. Train exactly one run per model with no restart selection stage.

Derived seeds used by the current analysis drivers:

- IQP initialization: `seed + 10000 + 7*K`
- Ising+fields (NN+NNN) initialization: `seed + 30001`
- Dense Ising+fields initialization: `seed + 30004`
- Transformer initialization: `seed + 35501`
- Transformer dataloader shuffle: `seed + 35512`
- MaxEnt-parity initialization: `seed + 36001`

## 3. Training Budgets

The comparison is fixed-update-budget based and not wall-clock normalized.

| Model | Optimizer | LR | Budget | Batch size | Early stopping |
| --- | --- | --- | --- | --- | --- |
| IQP-parity QCBM | `qml.AdamOptimizer` | `0.05` | `600` steps | full objective | none |
| Ising+fields (NN+NNN) | `qml.AdamOptimizer` | `0.05` | `600` steps | full objective | none |
| Dense Ising+fields | `qml.AdamOptimizer` | `0.05` | `600` steps | full objective | none |
| AR Transformer | `torch.optim.Adam` | `1e-3` | `600` epochs | `256` | none |
| MaxEnt-parity | `torch.optim.Adam` | `0.05` | `600` steps | full objective | none |

Because `m=200 < 256`, each Transformer epoch corresponds to one full-batch
optimizer update.

## 4. Model Hyperparameters

| Model | Parameterization / feature set | Capacity for `n=12` | Objective |
| --- | --- | --- | --- |
| IQP-parity QCBM | one-layer `ZZ` IQP circuit on cyclic NN+NNN pairs | `24` trainable angles | parity-MSE |
| Ising+fields (NN+NNN) | cyclic NN+NNN spin products plus local fields | `24 + 12 = 36` parameters | parity-MSE |
| Dense Ising+fields | all pairwise spin products plus local fields | `66 + 12 = 78` parameters | empirical cross entropy |
| AR Transformer | autoregressive next-bit model, big-endian bit ordering | `9057` parameters | maximum likelihood |
| MaxEnt-parity | exponential family over the sampled parity band | `512` parameters | log-partition / moment matching |

No model uses regularization, early stopping, validation-based selection, or a
restart sweep in the active final-reporting runs.

## 5. Transformer Transparency

The AR Transformer baseline is intentionally over-documented because it is the
easiest baseline to challenge as under-tuned.

- bit ordering: most-significant to least-significant bit from `int2bits`
- vocabulary: `{0, 1, BOS}`
- `d_model = 32`
- layers: `1`
- heads: `4`
- feed-forward width: `64`
- activation: `GELU`
- dropout: `0.0`
- weight decay: `0.0`
- optimizer: `Adam`
- learning rate: `1e-3`
- epochs: `600`
- batch size: `256`
- early stopping: none
- restart policy: single run only

## 6. Aggregation and Statistics

Metrics are computed per matched instance first and only then aggregated across
instances. There is no sample pooling before metric evaluation.

For the full sweep table:

- `mean KL` is the mean over the `200` matched instances.
- `median KL` is the median over the same instances.
- `95% CI` is the normal-approximation confidence interval for the mean.
- `KL wins` counts matched instances on which a model obtains the lowest
  forward KL among all compared models.
- `mean C_q(1000)` is aggregated over the same matched instances.

## 7. Benchmark Constants

| Quantity | Value |
| --- | --- |
| `n` | `12` |
| support | even parity only |
| support size `|S|` | `2048` |
| wide beta sweep | `0.1..2.0` in steps of `0.1` |
| matched seeds | `111..120` |
| train sample count `m` | `200` |
| parity-band parameters | `sigma = 1.0`, `K = 512` |
| IQP layers | `1` |
| quality coverage budgets | `Q in {1000, 2000, 5000}` |

## 8. Reproducibility Package Contents

The publication package exposes:

- raw per-instance metrics in CSV form
- the active seed schedule in `docs/benchmark_seed_schedule_10seeds.csv`
- the training protocol in `STANDARD_TRAINING_PROTOCOL.md`
- experiment-local `RUN_CONFIG.json`, `RERENDER_CONFIG.json`, and
  `TRAINING_PROTOCOL.md` files where applicable
- scripts at repository root for regenerating final figures from cached
  artifacts
- rendered PDF/PNG figures and retained `.npz`/`.csv`/`.json` intermediates

Historical artifacts with `5`, `20`, or other seed counts are retained only as
legacy snapshots. They do not override this active 10-seed final-reporting
standard.
