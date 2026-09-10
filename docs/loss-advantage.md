# Classical comparison

The 36-parameter IQP-parity model achieves lower KL than the four classical
baselines that model all bitstrings on each of the 120 paired datasets.

## Models and training

The target is proportional to `exp(beta * score)` on even-parity 12-bit strings,
with beta = 0.9 and 200 IID training draws. The score is the longest internal zero
run bracketed by ones.

The IQP circuit has first-, second- and third-neighbour RZZ connections, giving
36 trainable angles. It uses 512 sampled parity masks at sigma = 0.75, 600 Adam
updates and float64 arithmetic. Parity and matched MSE use learning rate 0.10;
tuned MSE uses 0.20. Initial angles have standard deviation 0.01 and are shared
between paired losses.

Sparse Ising trains on parity moments; dense Ising trains on likelihood; MaxEnt
matches sampled parity moments; the Transformer predicts bits autoregressively.
All four train for 600 updates. The Transformer has one layer, width 32, four
heads, feedforward width 64, GELU and no dropout. MaxEnt and Transformer use
float32 arithmetic. Full settings are in the
[protocol](../protocols/loss-advantage-confirmation.json).

IQP settings are selected using likelihood on 2,000 independent validation draws
per development dataset. The four main classical configurations are fixed.
Confirmation uses seeds 5001–5120, shared training observations, and final
iterates. Exact target KL is the evaluation metric.

## Results

| Model | Parameters | Mean exact KL | Mean unseen-elite recovery, Q = 1,000 |
|---|---:|---:|---:|
| IQP-parity | 36 | **0.360405** | **0.425516** |
| IQP-MSE, matched learning rate | 36 | 0.394327 | 0.409947 |
| IQP-MSE, tuned learning rate | 36 | 0.386771 | 0.412756 |
| Sparse Ising, parity loss | 36 | 1.095551 | 0.283863 |
| Dense Ising, likelihood | 78 | 1.100497 | 0.281891 |
| Transformer | 9,057 | 0.890017 | 0.267116 |
| MaxEnt | 512 | 2.493116 | 0.139230 |

| Control | Mean KL difference, parity minus control | Adjusted paired interval | Parity wins |
|---|---:|---|---:|
| Matched IQP-MSE | −0.033922 | [−0.053952, −0.013892] | 84/120 |
| Tuned IQP-MSE | −0.026366 | [−0.042865, −0.009868] | 75/120 |
| Sparse Ising | −0.735145 | [−0.765443, −0.704848] | 120/120 |
| Dense Ising | −0.740091 | [−0.758127, −0.722056] | 120/120 |
| Transformer | −0.529612 | [−0.560122, −0.499102] | 120/120 |
| MaxEnt | −2.132711 | [−2.314129, −1.951292] | 120/120 |

Paired t intervals use 99.5% confidence each, giving 95% Bonferroni family
confidence across ten comparisons, including the
[models with explicit parity support](supplementary-controls.md).
Recovery and win counts are descriptive.

## Reproduction

```bash
python scripts/confirm_loss_advantage.py --protocol protocols/loss-advantage-confirmation.json --out runs/classical-comparison --jobs 3
```

[Setting selection](reproduce.md#classical-comparison) ·
[Selected settings](../results/loss-advantage/selection.json) ·
[Per-seed results](../results/loss-advantage/confirmation/metrics.csv) ·
[Statistics](../results/loss-advantage/confirmation/summary.json)

## Limitations

IQP enforces even parity, while these four classical baselines model all
bitstrings. With explicit parity support, dense Ising and the prefix Transformer
achieve lower KL than IQP-parity. Model capacities, selection budgets and
computational costs differ across classes; the comparison measures accuracy
under the specified configurations.
