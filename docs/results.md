# Additional results

## When parity supervision improves generalization

The generalization benefit of parity supervision depends on the target
distribution, IQP architecture and training settings. With settings selected
by validation likelihood, the advantage over independently tuned MSE is
statistically supported at beta = 0.9 in the tested 36-, 48-, 60- and
66-parameter circuits. The 36-parameter parity model achieves the lowest mean
exact KL; the relative advantage over MSE grows in the larger tested circuits.
Tuned MSE performs better at 12 and 24 parameters. These results identify
favorable tested regimes, rather than a universal parameter-count threshold
or a globally optimal parity setting. See the
[architecture comparison](parameter-ladder.md).

The [seven-objective ablation](../README.md#seven-objective-ablation) finds that
parity beats MSE in all four tested circuits and all six alternatives at 36, 48
and 66 parameters, under correction for 24 comparisons. At 60 parameters,
spherical has the lowest mean KL and parity shows several high-KL outcomes.
[Per-seed results](../results/objective-ladder/confirm/metrics.csv) ·
[Paired statistics and variability](../results/objective-ladder/confirm/summary.json)

The earlier 36-parameter cohort is retained separately in its
[results](../results/objective-comparison/confirm/summary.json). In the historical
20-seed rerun, parity and MSE reproduced the earlier rounded means; the previously
reported NLL value did not reproduce (0.3666 at learning rate 0.10, versus the
reported approximately 0.421). See the [historical results](../results/objective-comparison/historical/summary.json).

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

For the valid even-parity set `S`,
`KL(p || q) = KL(p || q(. | S)) - log q(S)` whenever the terms are finite.
Conditioning removes the penalty for probability outside `S`. Both IQP losses
already have `q(S) = 1`, so this architectural property does not explain the
parity-versus-MSE difference. The loss advantage is evaluated separately in the
architecture and objective comparisons above. This table conditions models
after training; the [support ablation](supplementary-controls.md) instead gives
classical models the constraint during training, and some then outperform IQP.

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
