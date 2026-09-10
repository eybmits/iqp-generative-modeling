# Supplementary models

These models use even-parity support explicitly. Ising and MaxEnt train with
normalization over even states. The Transformer learns the first eleven bits
and appends their parity as bit twelve. Each model's configuration and global
update count were selected on development validation data, then fixed for all
120 confirmation datasets. All models receive the same training observations.

| Model | Mean exact KL | Adjusted interval, IQP-parity minus model |
|---|---:|---|
| IQP-parity | 0.360405 | — |
| Range-3 Ising, parity loss | 0.397851 | [−0.063028, −0.011863] |
| Dense Ising, likelihood | 0.330254 | [+0.014338, +0.045964] |
| MaxEnt | 0.393006 | [−0.050424, −0.014778] |
| Prefix Transformer | 0.228093 | [+0.117276, +0.147348] |

Lower KL is better. IQP-parity outperforms the parity-trained Ising and MaxEnt
variants; likelihood-trained dense Ising and the prefix Transformer have lower
KL than IQP-parity. Intervals belong to the same ten-comparison Bonferroni family
as the [main results](loss-advantage.md).

[Configuration](../protocols/loss-advantage-confirmation.json) ·
[Complete statistics](../results/loss-advantage/confirmation/summary.json).
