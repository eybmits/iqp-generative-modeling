# Support ablation: the IQP architecture's built-in constraint

The IQP architecture used here has a concrete structural advantage for this
target: it enforces even-parity output by construction, for every choice of
trainable angles. Conjugating each RZZ gate by the Hadamard layers gives an XX
rotation, which flips bits in pairs and preserves the even parity of the initial
all-zero state. The model therefore assigns no probability to invalid odd-parity
strings, without learning the constraint from the 200 training observations or
requiring rejection or post-selection. This is an architectural inductive bias
aligned with the target's support, specific to the pairwise H–RZZ–H circuits
studied here.

The support controls ask what happens when classical models are explicitly
given that same constraint. They make the value of the IQP model's built-in
support restriction visible: the full-cube comparison includes this structural
advantage. Supplying the constraint to a classical model changes the comparison
to fitting probabilities within the valid support. These controls also use
selected configurations and update counts, so differences from the full-cube
baselines cannot be attributed quantitatively to support alone.

## Models with explicit support

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

## Interpretation

Built-in support validity is a practical advantage of this IQP architecture over
the unrestricted baselines. It is not exclusive to quantum models: the dense
Ising and prefix Transformer results show that explicitly encoding the same
constraint classically can remove or reverse the observed KL advantage. The
supported conclusion is an advantage from matching the architecture to the
target's structure, rather than superiority over all classical models supplied
with the same support knowledge.

This architectural effect is distinct from the benefit of parity supervision
over MSE: both IQP losses use the same even-parity-preserving architecture.
Their difference in KL therefore concerns the learned distribution within the
valid support, not avoidance of odd-parity outputs. The historical
[conditional evaluation](results.md#conditional-evaluation) separately measures
the effect of removing invalid-support mass after training.

[Configuration](../protocols/loss-advantage-confirmation.json) ·
[Complete statistics](../results/loss-advantage/confirmation/summary.json).
