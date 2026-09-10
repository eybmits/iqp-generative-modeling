# Reference data

Saved numerical inputs for the optional reference figures. [SOURCES.json](SOURCES.json)
records file checksums and extraction metadata.

| File | Contents |
| --- | --- |
| `beta_metrics.csv` | 200 matched instances x five models, beta 0.1-2.0, seeds 111-120, KL and coverage at 1000/2000/5000 samples |
| `size_metrics.csv` | n 10-20 x ten seeds x five models, beta 0.9 |
| `legacy_fixed.npz` | Figure 4 source: seed 42 heatmap; full grids and MSE for seeds 42-51 |
| `score_marginal_seed42.csv` | Figure 4(a), with original score offset +1; display subtracts 1 |
| `fixed.npz` | Separate fixed-beta grids, trained weights and data indices for seeds 111-120 |
| `recovery_seed118.npz` | Figure 6(a-c) expected recovery curves,12-band family, seed 118 |
| `hardware.npz` | Raw counts, ideal distributions, train/elite sets and chosen parity bands for 20 historical jobs |
| `hardware_curves.npz` | Figure 6(d) means and sample standard deviations; verified from raw counts |
| `hardware_jobs.json`, `hardware_circuits.json` | IBM job metadata and transpiled circuit QASM; no live credentials |
| `main_protocol.md` | Original frozen source protocol, retained as historical documentation |

NPZ files load with `allow_pickle=False`. Hardware files contain 20 circuits
sampled on `ibm_marrakesh`, with 10,000 shots per circuit. The stored hardware
curves use occupancy estimates from observed frequencies and between-seed
sample standard deviations.

Historical coverage uses a sorted top-10% elite with truncated ties. The current
default metric includes ties at the 90th percentile; these elite definitions
can produce different coverage and recovery values.
