# Additional results

## Fixed-band 24-parameter circuit

The fixed-band preset uses sigma = 1, K = 512, 200 training observations and
600 updates. Parity and MSE share data and initial angles.

| Fresh experiment | Parity KL | MSE KL | Paired difference (parity − MSE), 95% t interval | Parity wins |
|---|---:|---:|---|---:|
| Reference band, 20 betas × 10 seeds | .385868 | .381594 | +.004275 [−.008190,.016739] | 89/200 |
| Fixed beta=.9, same reference band | .451550 | .476895 | −.025345 [−.073495,.022804] | 7/10 |

Intervals average within each of the ten seeds before aggregation across beta.
Both intervals include zero.

## Conditional evaluation

Conditional KL normalizes each trained model on the known even-parity support.

| Model | Mean true KL | Mean KL after explicitly conditioning on valid support | Mean C(1000), all-ties elite |
|---|---:|---:|---:|
| IQP parity | .385868 | .385868 | .086484 |
| IQP MSE | .381594 | .381594 | .086444 |
| Sparse Ising + fields | .922920 | .405834 | .059919 |
| Dense Ising + fields | .952686 | .442777 | .056394 |
| AR Transformer | .743757 | .367345 | .057350 |
| MaxEnt parity | 1.926522 | 1.558755 | .031283 |

## Recovery at beta = 0.9

Mean unseen-elite recovery at Q = 1,000 over ten seeds:

| Model / diagnostic | Recovery |
|---|---:|
| IQP parity, fixed sigma1/K512 | .409154 |
| IQP MSE, matched initialization | .398075 |
| Weighted spectral, full cube | .419868 |
| Weighted spectral, valid support | .603261 |
| Unique-mask spectral, valid support | .460006 |
| Uniform valid support | .386393 |

## Reproduction

```bash
iqp-repro run --preset paper --no-plots --out runs/paper
```

[Protocol](protocol.md#fixed-band-preset) ·
[Per-instance results](../results/paper/metrics.csv) ·
[Summary](../results/paper/summary.json)

## Limitations

These results use historical datasets and fixed settings. Conditional evaluation
supplies support knowledge after training; spectral proxies use a separate
reconstruction procedure. The main architecture and classical comparisons use
separate validation and confirmation protocols.
