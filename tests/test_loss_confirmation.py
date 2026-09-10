"""Frozen selection, pairing, multiplicity and retained-failure checks."""

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import t

from iqp_repro import loss_search

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location('confirmation_test_driver', SCRIPTS / 'confirm_loss_advantage.py')
    confirmation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(confirmation)
finally:
    sys.path.pop(0)


def protocol():
    settings = loss_search.development_protocol()
    parity = dict(beta=.9, architecture='ring', objective='parity', sigma=1., lr=.05)
    return dict(n=4, m=20, k=8, steps=2, validation_samples=30,
                optimizer=settings['optimizer'], confirmation_seeds=[700001, 700002, 700003], beta=.9,
                configurations={'parity': dict(kind='iqp', config=parity),
                    'mse': dict(kind='iqp', config=dict(parity, objective='mse')),
                    'paper': dict(kind='classical', config=dict(beta=.9, model='ising-nll',
                        architecture='dense', sigma=1., lr=.05, steps=2, support_aware=False, l2=0.))},
                comparisons=[dict(control='mse', group='loss'), dict(control='paper', group='paper'),
                             dict(control='paper', group='support')], source_hashes={})


def synthetic_rows(p):
    rows = []
    for i, seed in enumerate(p['confirmation_seeds']):
        for role, values in [('parity', [.2, .3, .4]), ('mse', [.5, .7, .9]), ('paper', [.8, .9, 1.1])]:
            rows.append(dict(role=role, seed=seed, status='ok', kl=values[i], recovery_1000=.3,
                             validation_nll=999. if role == 'parity' else 0.,
                             sample_sha256=f'sample-{seed}', validation_sha256=f'validation-{seed}',
                             initial_sha256=f'initial-{seed}' if role in ('parity', 'mse') else None))
    return rows


def test_paired_intervals_count_unique_controls_and_ignore_validation_scores():
    p = protocol()
    confirmation.validate_protocol(p)
    rows = synthetic_rows(p)
    report = confirmation.summarize(rows, p)
    assert report['multiplicity']['unique_control_roles'] == 2  # paper appears in two groups
    result = report['comparisons']['mse']
    d = np.array([-.3, -.4, -.5])
    half = t.ppf(.9875, 2) * d.std(ddof=1)/np.sqrt(3)
    np.testing.assert_allclose(result['bonferroni_ci95'], [d.mean()-half, d.mean()+half])
    assert result['adjusted_confidence'] == .975
    assert result['parity_wins'] == result['n'] == 3
    assert result['nominal_ci95'][1] < result['bonferroni_ci95'][1]
    # Confirmation cannot choose different models using diagnostic validation NLL.
    for row in rows:
        row['validation_nll'] = -row['validation_nll']
    assert confirmation.summarize(rows, p) == report


@pytest.mark.parametrize('field', ['sample_sha256', 'validation_sha256', 'initial_sha256'])
def test_reject_mismatched_data_or_same_architecture_initialization(field):
    p = protocol()
    rows = synthetic_rows(p)
    next(row for row in rows if row['role'] == 'mse')[field] = 'wrong'
    with pytest.raises(ValueError, match='[Uu]nmatched|initialization'):
        confirmation.summarize(rows, p)


def test_reject_missing_or_duplicate_seed_and_wrong_beta():
    p = protocol()
    rows = synthetic_rows(p)
    with pytest.raises(ValueError, match='grid'):
        confirmation.summarize(rows[:-1], p)
    with pytest.raises(ValueError, match='grid'):
        confirmation.summarize(rows[:-1] + [rows[0]], p)
    p['configurations']['mse']['config']['beta'] = 1.2
    with pytest.raises(ValueError, match='beta'):
        confirmation.validate_protocol(p)


@pytest.mark.parametrize('failed_role', ['parity', 'mse'])
def test_failures_and_true_infinities_remain_and_cannot_pass(failed_role):
    p = protocol()
    rows = synthetic_rows(p)
    bad = next(row for row in rows if row['role'] == failed_role)
    bad.update(kl='Infinity', status='failed', recovery_1000=None)
    report = confirmation.summarize(rows, p)
    assert report['rows'] == 9 and report['roles'][failed_role]['n'] == 3
    assert report['roles'][failed_role]['mean_kl'] == 'Infinity'
    result = report['comparisons']['mse']
    assert result['failed_pairs'] == result['nonfinite_pairs'] == 1
    assert result['bonferroni_ci95'] is None and result['passes'] is False
    assert result['parity_wins'] == 2  # failure/infinity does not manufacture a win
    assert report['group_verdicts']['loss'] is False
    assert not report['close_paper_success'] and not report['support_aware_success']
    bad['status'] = 'ok'  # Valid distribution with true infinite KL is also retained.
    result = confirmation.summarize(rows, p)['comparisons']['mse']
    assert result['failed_pairs'] == 0 and result['bonferroni_ci95'] is None


def test_unplanned_groups_do_not_vacuously_pass():
    p = protocol()
    p['comparisons'] = p['comparisons'][:2]
    report = confirmation.summarize(synthetic_rows(p), p)
    assert report['group_verdicts']['support'] is None
    assert report['support_aware_success'] is False


def test_atomic_checkpoint_resume_and_failure_no_retraining(tmp_path, monkeypatch):
    p = protocol()
    provenance = {'source_hashes': {'synthetic': 'hash'}}
    row = confirmation.run_one('parity', 700001, p, tmp_path, provenance)
    assert row['status'] == 'ok'
    path = tmp_path / 'parity_seed700001.npz'
    with np.load(path, allow_pickle=False) as saved:
        assert {'q', 'logq', 'p', 'samples', 'validation', 'masks', 'initial',
                'metrics', 'specification', 'code_hashes'}.issubset(saved.files)
        assert len(saved['loss_history']) == p['steps'] + 1
    def forbidden(*args):
        raise AssertionError('cached fit must not be retrained')
    monkeypatch.setattr(confirmation.loss_search, 'train_configuration', forbidden)
    assert confirmation.run_one('parity', 700001, p, tmp_path, provenance) == row
    changed = copy.deepcopy(p)
    changed['steps'] += 1
    with pytest.raises(ValueError, match='provenance'):
        confirmation.run_one('parity', 700001, changed, tmp_path, provenance)
    failed = confirmation.run_one('mse', 700001, p, tmp_path, provenance)
    assert failed['status'] == 'failed' and failed['kl'] == 'Infinity'
    assert (tmp_path / 'mse_seed700001.failure.json').exists()
    assert not (tmp_path / 'mse_seed700001.npz').exists()
    assert confirmation.run_one('mse', 700001, p, tmp_path, provenance) == failed
    assert not list(tmp_path.glob('*.tmp*'))


def test_classical_wrapper_uses_same_observations_and_saves_final_iterate(tmp_path):
    p = protocol()
    fingerprint = {'source_hashes': {'synthetic': 'hash'}}
    quantum = confirmation.run_one('parity', 700001, p, tmp_path, fingerprint)
    paper = confirmation.run_one('paper', 700001, p, tmp_path, fingerprint)
    assert quantum['status'] == paper['status'] == 'ok'
    assert quantum['sample_sha256'] == paper['sample_sha256']
    assert quantum['validation_sha256'] == paper['validation_sha256']
    with np.load(tmp_path/'paper_seed700001.npz', allow_pickle=False) as saved:
        assert len(saved['loss_history']) == p['steps'] + 1
        positive = saved['p'] > 0
        expected = np.sum(saved['p'][positive] * (np.log(saved['p'][positive]) - saved['logq'][positive]))
        assert paper['kl'] == pytest.approx(expected)
        assert paper['validation_nll'] == pytest.approx(-saved['logq'][saved['validation']].mean())


def test_source_verification_and_lock_reject_changes(tmp_path):
    p = protocol()
    for name in confirmation.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    p['source_hashes'] = {name: confirmation.study.digest(tmp_path/name) for name in confirmation.SOURCE_FILES}
    confirmation.verify_source_hashes(p, tmp_path)
    (tmp_path / confirmation.SOURCE_FILES[0]).write_text('changed')
    with pytest.raises(ValueError, match='source hash'):
        confirmation.verify_source_hashes(p, tmp_path)
    out = tmp_path/'run'
    confirmation.seal_output(out, p, {'workers': 1})
    confirmation.seal_output(out, p, {'workers': 1})
    with pytest.raises(ValueError, match='environment'):
        confirmation.seal_output(out, p, {'workers': 2})
