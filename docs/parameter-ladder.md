# Architecture comparison

Parity supervision achieves lower mean KL than independently tuned MSE in the
tested 36-, 48-, 60- and 66-parameter circuits. The comparison uses 120 paired
datasets per architecture at beta = 0.9.

## Architectures

Each circuit has 12 qubits and one Hadamard–RZZ–Hadamard block. Increasing the
ring radius adds pairwise RZZ connections, with one trainable angle per edge.
Setting the added angles to zero recovers the smaller circuit. Every circuit
has even-parity output support.

| Ring radius | Parameters | Connections |
|---:|---:|---|
| 1 | 12 | Nearest neighbours |
| 2 | 24 | Up to second neighbours |
| 3 | 36 | Up to third neighbours |
| 4 | 48 | Up to fourth neighbours |
| 5 | 60 | Up to fifth neighbours |
| 6 | 66 | Every pair |

## Results

| Parameters | Parity KL | Tuned MSE KL | Mean KL reduction | Adjusted comparison |
|---:|---:|---:|---:|---|
| 12 | 0.778380 | 0.763016 | -2.0% | MSE better |
| 24 | 0.467443 | 0.437073 | -6.9% | MSE better |
| 36 | 0.353250 | 0.379048 | +6.8% | Parity better |
| 48 | 0.372160 | 0.464213 | +19.8% | Parity better |
| 60 | 0.412423 | 0.571549 | +27.8% | Parity better |
| 66 | 0.429218 | 0.798880 | +46.3% | Parity better |

Percentages compare mean exact KL values. The 36-parameter parity model has the
lowest mean KL. At 66 parameters, the MSE distribution has median KL 0.504264
and mean 0.798880; parity wins 97 of 120 pairs.

| Parameters | Comparison | Mean KL difference | Adjusted interval | Parity wins |
|---:|---|---:|---|---:|
| 12 | Tuned MSE | 0.015364 | [0.000992, 0.029736] | 50/120 |
| 12 | Matched MSE | 0.005540 | [-0.008609, 0.019689] | 57/120 |
| 24 | Tuned MSE | 0.030370 | [0.011683, 0.049058] | 39/120 |
| 24 | Matched MSE | 0.035233 | [0.017023, 0.053442] | 36/120 |
| 36 | Tuned MSE | -0.025798 | [-0.041882, -0.009714] | 81/120 |
| 36 | Matched MSE | -0.045832 | [-0.063984, -0.027680] | 93/120 |
| 48 | Tuned MSE | -0.092053 | [-0.113485, -0.070621] | 105/120 |
| 48 | Matched MSE | -0.440703 | [-0.620849, -0.260557] | 105/120 |
| 60 | Tuned MSE | -0.159126 | [-0.184541, -0.133711] | 117/120 |
| 60 | Matched MSE | -0.222617 | [-0.358047, -0.087186] | 99/120 |
| 66 | Tuned MSE | -0.369662 | [-0.575856, -0.163467] | 97/120 |
| 66 | Matched MSE | -0.369662 | [-0.575856, -0.163467] | 97/120 |

Differences are parity minus MSE. Intervals use Bonferroni correction across
all 12 comparisons for 95% family confidence. Win counts are descriptive.
Matched MSE uses the parity learning rate; tuned MSE is selected independently.

## Training and selection

Each fit uses 200 IID observations, 600 Adam updates, float64 arithmetic and
the final iterate. Parity training uses 512 sampled nonzero masks. Adam uses
beta1 = 0.9, beta2 = 0.99 and epsilon = 1e-8 before bias correction.

Both loss families receive 12 candidates and the same ten development datasets
(seeds 111–120). Selection minimizes mean negative log-likelihood on 2,000
independent validation observations per dataset.

- Parity: sigma in {0.75, 1, 1.5, 2}, learning rate in {0.02, 0.05, 0.1}.
- MSE: mean or summed squared error over valid states, learning rate in
  {0.005, 0.01, 0.02, 0.05, 0.1, 0.2}.

| Parameters | Parity sigma | Parity learning rate | Tuned MSE | MSE learning rate |
|---:|---:|---:|---|---:|
| 12 | 0.75 | 0.02 | mse | 0.01 |
| 24 | 0.75 | 0.1 | mse | 0.2 |
| 36 | 0.75 | 0.1 | mse | 0.2 |
| 48 | 1 | 0.1 | mse | 0.05 |
| 60 | 1 | 0.05 | mse | 0.02 |
| 66 | 1 | 0.05 | mse | 0.05 |

Confirmation uses seeds 300001–300120. Architectures share training observations;
paired losses within each architecture also share initial angles. Exact target
KL is used for evaluation. Nonfinite validation scores are ineligible for selection.

## Reproduction

```bash
python -m iqp_repro.parameter_ladder confirm --protocol protocols/parameter-ladder-confirmation.json --out runs/ladder/confirmation --jobs 3
```

[Setting selection and audit](reproduce.md#architecture-comparison) ·
[Protocol](../protocols/parameter-ladder-confirmation.json) ·
[Per-seed results](../results/parameter-ladder/confirmation/metrics.csv) ·
[Statistics](../results/parameter-ladder/confirmation/summary.json)

## Limitations

The advantage depends on architecture and training settings: tuned MSE has lower
mean KL at 12 and 24 parameters. This comparison evaluates one target and
architecture family by exact simulation; parameter count alone does not establish
a general threshold or predict the best achievable accuracy.
