"""Protect selection rules, matched controls, and uncertainty in setting checks."""
import csv
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t


spec = importlib.util.spec_from_file_location(
    'check_parity_settings', Path(__file__).parents[1] / 'scripts/check_parity_settings.py')
settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings)


@pytest.fixture
def cohort():
    protocol = dict(n=12, m=200, steps=600, betas=[.8, .9], seeds=[111, 112],
                    sigma=[.5, 1.], k=[128, 256], models=['iqp-parity', 'iqp-mse'],
                    reference=dict(sigma=1., k=128))
    rows = []
    for beta, seed, sigma, k, model in itertools.product(
            protocol['betas'], protocol['seeds'], protocol['sigma'],
            protocol['k'], protocol['models']):
        if model == 'iqp-parity':
            # Exact KL favors K=256; validation favors K=128.
            kl = (.2 if k == 256 else .4) + (.1 if sigma == 1. else 0.)
            validation = (3. if k == 256 else 1.) + (1. if sigma == 1. else 0.)
        else:
            # MSE's preferred K reverses the parity preference under each rule.
            kl = .4 if k == 128 else .8
            validation = 2. if k == 128 else 1.
        rows.append(dict(n=12, m=200, steps=600, beta=beta, seed=seed, sigma=sigma,
                         k=k, model=model, protocol='matched', kl=kl,
                         validation_nll=validation, coverage_1000=.1, recovery_1000=.2,
                         samples_sha256=f'samples-{beta}-{seed}', init_seed=seed+10000+7*k))
    return protocol, rows


def write_metrics(directory, rows):
    with (directory / 'metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_validation_selection_follows_validation_when_exact_kl_disagrees():
    best_exact = dict(sigma=.5, k=128, kl=.1, validation_nll=2.)
    best_validation = dict(sigma=1., k=256, kl=.9, validation_nll=1.)
    candidates = [best_exact, best_validation]
    assert settings.select(candidates, 'kl') is best_exact
    assert settings.select(candidates, 'validation_nll') is best_validation


def test_matched_mse_and_independently_selected_mse_use_their_own_candidates(cohort):
    protocol, rows = cohort
    indexed = {settings.key(row): row for row in rows}
    report, choices = settings.analyze(indexed, protocol['betas'], protocol)
    assert len(choices) == 8
    for choice in choices:
        if choice['rule'] == 'oracle':
            assert choice['parity_k'] == 256
            assert choice['mse_k'] == 128
            assert choice['matched_mse_kl'] == .8
            assert choice['selected_mse_kl'] == .4
        else:
            assert choice['parity_k'] == 128
            assert choice['mse_k'] == 256
            assert choice['matched_mse_kl'] == .4
            assert choice['selected_mse_kl'] == .8
    comparisons = report['comparisons']
    assert comparisons['oracle_parity_vs_matched_mse']['mse_mean_kl'] == .8
    assert comparisons['oracle_each']['mse_mean_kl'] == .4
    assert comparisons['validation_parity_vs_matched_mse']['mse_mean_kl'] == .4
    assert comparisons['validation_each']['mse_mean_kl'] == .8


def test_contrast_clusters_repeated_betas_within_seed():
    pairs = []
    for seed, differences in [(111, [-2., 0.]), (112, [0., 2.])]:
        for beta, difference in zip([.8, .9], differences):
            common = dict(seed=seed, beta=beta, recovery_1000=.2)
            pairs.append((dict(common, kl=3.+difference), dict(common, kl=3.)))
    result = settings.contrast(pairs)
    # The independent observations for the CI are seed means [-1, 1].
    halfwidth = t.ppf(.975, 1)
    assert result['instances'] == 4
    assert result['seed_clusters'] == 2
    assert result['mean_difference'] == 0.
    assert result['descriptive_ci95'] == pytest.approx([-halfwidth, halfwidth])
    assert not np.isclose(halfwidth, t.ppf(.975, 3)*np.std([-2., 0., 0., 2.], ddof=1)/2)


def test_load_rows_rejects_incomplete_grid(tmp_path, cohort):
    protocol, rows = cohort
    write_metrics(tmp_path, rows)
    assert len(settings.load_rows(tmp_path, protocol)) == len(rows)
    write_metrics(tmp_path, rows[:-1])
    with pytest.raises(ValueError, match='Incomplete, duplicated or unexpected parameter grid'):
        settings.load_rows(tmp_path, protocol)


def test_load_rows_rejects_candidate_sampling_mismatch(tmp_path, cohort):
    protocol, rows = cohort
    rows[-1]['samples_sha256'] = 'different-training-sample'
    write_metrics(tmp_path, rows)
    with pytest.raises(ValueError, match='Candidate data are not shared'):
        settings.load_rows(tmp_path, protocol)


def test_nonfinite_outcomes_are_retained_and_intervals_are_undefined():
    pairs = [
        (dict(seed=111, kl=float('inf'), recovery_1000=.1),
         dict(seed=111, kl=.4, recovery_1000=.1)),
        (dict(seed=112, kl=.2, recovery_1000=.1),
         dict(seed=112, kl=.4, recovery_1000=.1)),
    ]
    with np.errstate(invalid='ignore'):
        result = settings.safe(settings.contrast(pairs))
    assert result['instances'] == 2
    assert result['parity_mean_kl'] == 'Infinity'
    assert result['nonfinite_pairs'] == 1
    assert result['interval_status'] == 'undefined_nonfinite'
    assert result['descriptive_ci95'] == [None, None]
