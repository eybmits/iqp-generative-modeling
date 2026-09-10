# Reproduction

Follow [Installation](../README.md#installation), then run commands from the repository root.

## Architecture comparison

```bash
python -m iqp_repro.parameter_ladder confirm --protocol protocols/parameter-ladder-confirmation.json --out runs/ladder/confirmation --jobs 3
python scripts/audit_parameter_ladder.py --protocol protocols/parameter-ladder-confirmation.json --runs runs/ladder/confirmation --out runs/ladder/audit.json
```

The experiment trains parity, tuned MSE and MSE with the parity learning rate
on 120 datasets for each of the six architectures. The audit recalculates saved
probabilities, objectives, metrics and paired statistics.

To repeat setting selection:

```bash
python -m iqp_repro.parameter_ladder develop --protocol protocols/parameter-ladder-development.json --out runs/ladder/development --jobs 3
python -m iqp_repro.parameter_ladder select --protocol protocols/parameter-ladder-development.json --development runs/ladder/development --out runs/ladder/selected.json
```

## Classical comparison

```bash
python scripts/confirm_loss_advantage.py --protocol protocols/loss-advantage-confirmation.json --out runs/classical-comparison --jobs 3
```

To repeat setting selection:

```bash
python -m iqp_repro.loss_search --out runs/development/iqp --jobs 2
python scripts/check_loss_baselines.py --out runs/development/classical --jobs 2
python scripts/select_loss_advantage.py --iqp runs/development/iqp --classical runs/development/classical --out runs/selected-protocol.json --selection-out runs/selection.json
```

## Outputs and resuming

Experiments save `metrics.csv`, `summary.json`, configuration and environment
metadata, and one checkpoint per fit. Repeat the same command to resume.
Use a new output directory when changing the configuration, implementation,
environment or worker count.

## Additional presets

```bash
iqp-repro run --preset smoke --no-plots --out runs/smoke
iqp-repro run --preset paper --no-plots --out runs/paper
iqp-repro summarize --out runs/paper --plots
```

`smoke` runs a small example. `paper` runs the fixed-band 24-parameter benchmark;
`summarize --plots` generates figures from its checkpoints. See
[Protocol](protocol.md) for settings and [Additional results](results.md) for tables.

Run the tests with `python -m pytest -q`.
