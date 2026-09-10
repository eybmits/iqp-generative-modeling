# iqp-generative-modeling

Code and experiments for **[Parity Supervision as a Driver of
Generalization in Quantum Generative Modeling](https://arxiv.org/abs/2605.10258v2)**.

## Central finding

**The generalization benefit of parity supervision depends on the combination of
the target distribution (β), IQP architecture and expressivity (e.g. ring radii
2, 3, 4, …), and training objective (parity-moment loss versus probability MSE).**
Within suitable tested regimes, parity supervision improves generalization while
keeping the IQP architecture fixed.

See [Post-publication refinement](docs/corrections.md#post-publication-refinement)
for the relationship to the paper.

## Installation

Use Python 3.13:

```bash
git clone https://github.com/eybmits/iqp-generative-modeling.git
cd iqp-generative-modeling
python3.13 -m venv .venv
source .venv/bin/activate
```

On Linux, install CPU-only PyTorch before installing the package:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

Install the package and test dependencies on all platforms:

```bash
python -m pip install '.[test]'
```

## Reproduction

Run the architecture comparison and the classical comparison from the repository root:

```bash
python -m iqp_repro.parameter_ladder confirm --protocol protocols/parameter-ladder-confirmation.json --out runs/ladder/confirmation --jobs 3
python scripts/confirm_loss_advantage.py --protocol protocols/loss-advantage-confirmation.json --out runs/classical-comparison --jobs 3
```

Each command uses a separate cohort of 120 datasets and saves per-model metrics,
summary statistics and resumable checkpoints. Both benchmarks use 12 qubits,
beta = 0.9 and 200 training observations per dataset. The main models train for
600 Adam updates. Run the tests with `python -m pytest -q`.

See [Reproduction](docs/reproduce.md) for setting selection, audits and optional plots.

## Key results

### Parity supervision versus MSE

Lower mean exact KL is better. Parity and MSE settings are selected independently
using validation likelihood; each paired comparison shares training data and
initial angles.

| Parameters | Parity KL | Tuned MSE KL | Mean KL reduction |
|---:|---:|---:|---:|
| 36 | 0.353250 | 0.379048 | 6.8% |
| 48 | 0.372160 | 0.464213 | 19.8% |
| 60 | 0.412423 | 0.571549 | 27.8% |
| 66 | 0.429218 | 0.798880 | 46.3% |

All four advantages have multiplicity-adjusted paired confidence intervals below
zero for parity minus MSE. The 36-parameter parity model has the lowest mean KL.
[Full architecture results](docs/parameter-ladder.md) include the 12- and
24-parameter circuits and controls with matched learning rates.

### Comparison with classical baselines

The 36-parameter IQP-parity model has lower KL than each of these four baselines
on all 120 paired datasets.

| Model | Mean exact KL |
|---|---:|
| IQP-parity | **0.360405** |
| Sparse Ising, parity loss | 1.095551 |
| Dense Ising, likelihood | 1.100497 |
| Transformer | 0.890017 |
| MaxEnt | 2.493116 |

[Classical comparison](docs/loss-advantage.md) contains model settings and paired intervals.

## Limitations

- The results cover this synthetic target and the tested 12-qubit architecture family.
  Tuned MSE performs better in the 12- and 24-parameter comparisons.
- IQP circuits enforce even parity, while the four classical baselines above model
  all bitstrings. When classical models also use this support constraint, dense
  Ising and the prefix Transformer achieve lower KL than IQP-parity; see
  [models with explicit parity support](docs/supplementary-controls.md).
- Training uses exact classical simulation. These results do not establish a
  computational speedup, finite-shot training performance or performance on real-world data.

[Corrections](docs/corrections.md) · [Citation](CITATION.cff) · [MIT license](LICENSE)
