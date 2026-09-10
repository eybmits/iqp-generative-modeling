#!/usr/bin/env python3
"""Audit every ladder checkpoint and independently recompute selection and inference.

No project modules are imported. This reads saved outcomes and reconstructs
their inputs and laws; it never trains or selects using confirmation outcomes.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {f'src/iqp_repro/{name}.py' for name in ('parameter_ladder', 'core', 'study')}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def equal(actual, expected, label):
    """Compare complete independently computed structures, retaining infinities."""
    expected = json_safe(expected)
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), label + ' keys'
        for key, value in expected.items():
            equal(actual[key], value, label + '/' + str(key))
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), label + ' length'
        for index, value in enumerate(expected):
            equal(actual[index], value, label + '/' + str(index))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert np.isclose(float(actual), expected, atol=2e-12, rtol=1e-11), label
    else:
        assert actual == expected, label


def walsh(values):
    result = np.array(values, copy=True)
    width = 1
    while width < len(result):
        blocks = result.reshape(-1, 2 * width)
        left, right = blocks[:, :width].copy(), blocks[:, width:].copy()
        blocks[:, :width], blocks[:, width:] = left + right, left - right
        width *= 2
    return result


def config(row):
    return dict(radius=int(row['radius']), objective=row['objective'],
                sigma=float(row['sigma']), lr=float(row['lr']))


def config_key(row):
    return f'r{row["radius"]}_{row["objective"]}_sigma{row["sigma"]:g}_lr{row["lr"]:g}'


def identity(row):
    c = config(row)
    return (c['radius'], c['objective'], c['sigma'], c['lr'],
            int(row['seed']), row.get('role') or None)


def candidates(protocol):
    result = []
    for r in protocol['radii']:
        result += [dict(radius=r, objective='parity', sigma=s, lr=lr)
                   for s, lr in itertools.product(protocol['parity_sigmas'], protocol['parity_learning_rates'])]
        result += [dict(radius=r, objective=o, sigma=protocol['mse_sigma'], lr=lr)
                   for o, lr in itertools.product(protocol['mse_objectives'], protocol['mse_learning_rates'])]
    return result


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def check_protocol(protocol):
    n = protocol['n']
    assert isinstance(n, int) and n >= 4
    assert protocol['radii'] == list(range(1, n // 2 + 1)), 'complete natural radius ladder'
    for name in ('development_seeds', 'confirmation_seeds'):
        assert protocol[name] and len(set(protocol[name])) == len(protocol[name]), name
        assert all(isinstance(s, int) for s in protocol[name]), name + ' integer seeds'
    assert not set(protocol['development_seeds']) & set(protocol['confirmation_seeds']), 'disjoint cohorts'
    assert all(protocol[k] > 0 for k in ('m', 'k', 'validation_samples'))
    assert isinstance(protocol['steps'], int) and protocol['steps'] >= 0
    assert protocol['mse_objectives'] == ['mse', 'scaled-mse']
    assert len(candidates(protocol)) == len({config_key(c) for c in candidates(protocol)}), 'unique candidates'


def check_sources(hashes):
    assert set(hashes) == SOURCES, 'complete frozen source map'
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, 'source changed: ' + name


def check_grid(rows, configurations, seeds, protocol, confirmation=False):
    expected = {identity(dict(c, seed=s, **({'role': role} if confirmation else {})))
                for role, c in configurations.items() for s in seeds}
    indexed = {identity(row): row for row in rows}
    assert len(rows) == len(indexed) == len(expected) and set(indexed) == expected, 'complete unique metrics grid'
    for row in rows:
        assert float(row['beta']) == protocol['beta'], 'beta'
        assert row['key'] == config_key(config(row)), 'configuration key'
        assert row['status'] in ('ok', 'failed'), 'status'
        if row['status'] == 'failed':
            assert float(row['kl']) == float(row['validation_nll']) == np.inf, 'failed metrics retained as infinity'
    return indexed


def development_statistics(rows, protocol):
    """Independent candidate averages and validation-only deterministic ranking."""
    configs = {config_key(c): c for c in candidates(protocol)}
    check_grid(rows, configs, protocol['development_seeds'], protocol)
    ranking = []
    n = protocol['n']
    for key, c in configs.items():
        group = [row for row in rows if row['key'] == key]
        values = np.array([float(row['validation_nll']) for row in group])
        edges = sum(min(j-i, n-(j-i)) <= c['radius'] for i in range(n) for j in range(i+1, n))
        ranking.append(dict(c, key=key, fits=len(group), parameters=edges,
                            failures=sum(row['status'] != 'ok' for row in group),
                            nonfinite_validation_nll=int(np.sum(~np.isfinite(values))),
                            mean_validation_nll=float(values.mean()),
                            mean_kl=float(np.mean([float(row['kl']) for row in group]))))
    # Match the published deterministic configuration ordering, not result order.
    ranking.sort(key=lambda row: list(configs).index(row['key']))
    def score(row):
        value = row['mean_validation_nll']
        return value if np.isfinite(value) and row['failures'] == row['nonfinite_validation_nll'] == 0 else np.inf
    def choose(items):
        return min(items, key=lambda row: (score(row), row['key']))
    selections = []
    for radius in protocol['radii']:
        family = [row for row in ranking if row['radius'] == radius]
        parity = choose([row for row in family if row['objective'] == 'parity'])
        tuned = choose([row for row in family if row['objective'] in ('mse', 'scaled-mse')])
        options = [row for row in family if row['objective'] == 'mse' and row['lr'] == parity['lr']]
        matched = choose(options) if options else None
        eligible = matched is not None and all(np.isfinite(score(row)) for row in (parity, tuned, matched))
        selections.append(dict(radius=radius, parity=parity, mse_tuned=tuned,
                               mse_matched=matched, eligible=bool(eligible)))
    return dict(stage='development', fits=len(rows), failed_fits=sum(row['status'] != 'ok' for row in rows),
                ranking=ranking, selections=selections,
                selection_metric='mean held-out validation NLL; key breaks ties')


def check_frozen(document, protocol, development=None):
    assert document['stage'] == 'confirmation'
    assert document['development_seeds'] == protocol['development_seeds']
    assert document['confirmation_seeds'] == protocol['confirmation_seeds']
    expected_roles = {f'r{r}_{role}' for r in protocol['radii']
                      for role in ('parity', 'mse_tuned', 'mse_matched')}
    assert set(document['configurations']) == expected_roles, 'all three roles on every radius'
    expected_comparisons = [dict(radius=r, kind=kind, parity=f'r{r}_parity', control=f'r{r}_mse_{kind}')
                           for r in protocol['radii'] for kind in ('tuned', 'matched')]
    equal(document['comparisons'], expected_comparisons, 'all planned comparisons')
    for r in protocol['radii']:
        parity, tuned, matched = [document['configurations'][f'r{r}_{role}']
                                  for role in ('parity', 'mse_tuned', 'mse_matched')]
        assert all(c['radius'] == r for c in (parity, tuned, matched)), 'role/radius matching'
        assert parity['objective'] == 'parity' and tuned['objective'] in ('mse', 'scaled-mse')
        assert matched['objective'] == 'mse' and matched['lr'] == parity['lr'], 'matched ordinary MSE'
        assert parity['sigma'] == tuned['sigma'] == matched['sigma'], 'shared confirmation masks'
    files = {}
    for name, expected_hash in document['development_evidence'].items():
        path = Path(name)
        choices = [path, ROOT/path]
        if development is not None:
            choices.insert(0, Path(development)/path.name)
        found = next((p for p in choices if p.is_file() and digest(p) == expected_hash), None)
        assert found is not None, 'missing or changed frozen development evidence: ' + name
        assert path.name not in files, 'duplicate development evidence type'
        files[path.name] = found
    assert set(files) == {'lock.json', 'metrics.csv', 'summary.json'}, 'complete frozen development evidence'
    lock = json.loads(files['lock.json'].read_text())
    assert lock['protocol'] == protocol and lock['provenance']['stage'] == 'development'
    assert lock['provenance']['source_hashes'] == document['source_hashes'], 'same development sources'
    expected = development_statistics(read_rows(files['metrics.csv']), protocol)
    equal(json.loads(files['summary.json'].read_text()), expected, 'frozen development summary')
    equal(document['selections'], expected['selections'], 'frozen validation selections')
    for selected in expected['selections']:
        assert selected['eligible'], 'all radii eligible before confirmation'
        for role in ('parity', 'mse_tuned', 'mse_matched'):
            c = config(selected[role])
            c['sigma'] = selected['parity']['sigma']
            equal(document['configurations'][f'r{selected["radius"]}_{role}'], c, 'frozen selected configuration')


def confirmation_statistics(rows, document):
    protocol, seeds = document['protocol'], document['confirmation_seeds']
    by_role = {(row['role'], int(row['seed'])): row for row in rows}
    results = []
    family_size = 2 * len(protocol['radii'])
    for comparison in document['comparisons']:
        pairs = [(by_role[comparison['parity'], s], by_role[comparison['control'], s]) for s in seeds]
        a = np.array([float(pair[0]['kl']) for pair in pairs])
        b = np.array([float(pair[1]['kl']) for pair in pairs])
        valid = np.isfinite(a) & np.isfinite(b) & np.array([p['status'] == m['status'] == 'ok' for p, m in pairs])
        with np.errstate(invalid='ignore', divide='ignore'):
            delta = a-b
            mean = float(delta.mean())
            relative = (float(1-a.mean()/b.mean()) if valid.all() and
                        np.isfinite(a.mean()) and np.isfinite(b.mean()) and b.mean() > 0 else None)
        se = float(np.std(delta, ddof=1)/np.sqrt(len(delta))) if valid.all() and len(delta) > 1 else None
        def interval(probability):
            if se is None:
                return None
            margin = float(t.ppf(probability, len(delta)-1)*se)
            return [mean-margin, mean+margin]
        nominal, adjusted = interval(.975), interval(1-.05/(2*family_size))
        results.append(dict(comparison, n=len(pairs), finite_pairs=int(valid.sum()),
                            mean_parity_kl=float(a.mean()), mean_mse_kl=float(b.mean()),
                            mean_difference=mean, relative_improvement=relative,
                            wins=int(np.sum(valid & (delta < 0))), se=se, ci95=nominal,
                            adjusted_ci95=adjusted,
                            confirmed_win=bool(adjusted is not None and adjusted[1] < 0),
                            confirmed_loss=bool(adjusted is not None and adjusted[0] > 0)))
    primary = [row for row in results if row['kind'] == 'tuned']
    wins = sum(row['confirmed_win'] for row in primary)
    required = len(protocol['radii'])//2 + 1
    return dict(stage='confirmation', fits=len(rows), planned_comparisons=family_size, comparisons=results,
                primary_majority=dict(confirmed_wins=wins, architectures=protocol['radii'],
                                      total_count=len(primary), required_wins=required,
                                      confirmed_majority=wins >= required),
                matched_confirmed_wins=sum(row['confirmed_win'] for row in results if row['kind'] == 'matched'),
                scope='Only the tested cyclic pair-gate architecture family; no quantum computational advantage claim.')


def audit(runs, document, development=None):
    runs = Path(runs)
    protocol = document.get('protocol', document)
    confirmation = 'configurations' in document
    check_protocol(protocol)
    lock = json.loads((runs/'lock.json').read_text())
    assert lock['protocol'] == document, 'locked protocol'
    provenance = lock['provenance']
    expected_sources = document['source_hashes'] if confirmation else provenance['source_hashes']
    check_sources(expected_sources)
    assert provenance['source_hashes'] == expected_sources, 'locked source hashes'
    assert provenance['stage'] == ('confirmation' if confirmation else 'development')
    if confirmation:
        assert provenance['frozen_protocol'] == document, 'locked frozen protocol'
        check_frozen(document, protocol, development)
    elif 'source_hashes' in protocol:
        assert protocol['source_hashes'] == expected_sources
    n, size = protocol['n'], 2**protocol['n']
    support = np.array([x.bit_count() % 2 == 0 for x in range(size)])
    scores = np.array([max(map(len, format(x, f'0{n}b').split('1')[1:-1]), default=0)
                       for x in range(size)])
    p = np.zeros(size)
    logits = protocol['beta']*scores[support]
    p[support] = np.exp(logits-max(logits)); p /= p.sum()
    spins = 1-2*((np.arange(size)[:, None] >> np.arange(n-1, -1, -1)) & 1)
    eigenvalues = {}
    for radius in protocol['radii']:
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if min(j-i, n-(j-i)) <= radius]
        eigenvalues[radius] = np.column_stack([spins[:, i]*spins[:, j] for i, j in edges])
    configurations = document['configurations'] if confirmation else {config_key(c): c for c in candidates(protocol)}
    seeds = document['confirmation_seeds'] if confirmation else protocol['development_seeds']
    rows = read_rows(runs/'metrics.csv')
    index = check_grid(rows, configurations, seeds, protocol, confirmation)
    checked, issues, checkpoint_ids = [], [], set()
    max_errors = dict(probabilities=0., objective=0., kl=0.)
    for path in sorted((runs/'checkpoints').glob('*.npz')):
        try:
            with np.load(path, allow_pickle=False) as z:
                row = json.loads(str(z['metrics']))
                key = identity(row)
                assert key not in checkpoint_ids and key in index, 'checkpoint identity'
                checkpoint_ids.add(key)
                reference = index[key]
                c, seed, role = config(row), int(row['seed']), row.get('role')
                expected_spec = dict(config=c, seed=seed, protocol=protocol, provenance=provenance, role=role)
                assert json.loads(str(z['specification'])) == expected_spec, 'checkpoint specification'
                assert row['status'] == reference['status'] == 'ok', 'successful status'
                assert path.name == f'{role or config_key(c)}_seed{seed}.npz', 'checkpoint filename'
                for name in ('radius', 'objective', 'sigma', 'lr', 'beta', 'seed', 'parameters', 'key'):
                    if name in ('objective', 'key'):
                        assert row[name] == reference[name], name
                    else:
                        assert float(row[name]) == float(reference[name]), name
                samples = np.random.default_rng(seed+7).choice(size, protocol['m'], p=p)
                validation = np.random.default_rng(seed+50000).choice(size, protocol['validation_samples'], p=p)
                elite = support & (scores >= np.quantile(scores[support], .9))
                elite[np.unique(samples)] = False
                for name, value in [('samples', samples), ('validation', validation), ('support', support),
                                    ('scores', scores), ('elite', elite)]:
                    assert np.array_equal(z[name], value), name
                assert np.allclose(z['p'], p, rtol=1e-13, atol=1e-15), 'target'
                rng = np.random.default_rng(seed+222)
                probability = -.5*np.expm1(-1/(2*c['sigma']**2))
                masks = rng.binomial(1, probability, (protocol['k'], n)).astype(np.int8)
                zero = np.flatnonzero(masks.sum(1) == 0)
                while len(zero):
                    masks[zero] = rng.binomial(1, probability, (len(zero), n))
                    zero = np.flatnonzero(masks.sum(1) == 0)
                assert np.array_equal(z['masks'], masks), 'independent masks'
                matrix = eigenvalues[c['radius']]
                initial = .01*np.random.default_rng(seed+10000+7*protocol['k']).standard_normal(matrix.shape[1])
                assert np.array_equal(z['initial'], initial) and len(z['theta']) == matrix.shape[1]
                assert int(reference['parameters']) == matrix.shape[1], 'parameter count'
                for name, field in [('samples', 'sample_sha256'), ('validation', 'validation_sha256'),
                                    ('masks', 'mask_sha256'), ('initial', 'initial_sha256')]:
                    expected = hashlib.sha256(z[name].tobytes()).hexdigest()
                    assert row[field] == reference[field] == expected, field
                assert np.isfinite(z['theta']).all() and np.isfinite(z['q']).all()
                diagonal = np.exp(-.5j*(matrix @ z['theta']))/np.sqrt(size)
                q = np.abs(walsh(diagonal)/np.sqrt(size))**2; q /= q.sum()
                error = float(np.max(np.abs(q-z['q'])))
                max_errors['probabilities'] = max(max_errors['probabilities'], error)
                assert error < 2e-12 and np.max(z['q'][~support]) < 1e-25, 'Born probabilities'
                assert np.isclose(z['q'].sum(), 1., atol=1e-12) and np.all(z['q'] >= 0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    logq = np.log(z['q'])
                    kl = float(np.sum(p[support]*(np.log(p[support])-logq[support])))
                    nll = float(-np.mean(logq[validation]))
                    discoveries = float(np.sum(-np.expm1(1000*np.log1p(-z['q'][elite]))))
                assert np.array_equal(z['logq'], logq), 'log probabilities'
                recovery = discoveries/int(elite.sum()) if elite.any() else np.nan
                for field, actual in [('kl', kl), ('validation_nll', nll),
                                      ('recovery_1000', recovery), ('coverage_1000', discoveries/1000)]:
                    for saved in (row[field], reference[field]):
                        if np.isnan(actual):
                            assert saved is None or saved in ('', 'NaN', 'nan'), field
                        else:
                            assert np.isclose(float(saved), actual, atol=1e-12, rtol=1e-10), field
                if np.isfinite(kl):
                    max_errors['kl'] = max(max_errors['kl'], abs(float(reference['kl'])-kl))
                empirical = np.bincount(samples, minlength=size)/len(samples)
                for theta, history_index in [(initial, 0), (z['theta'], -1)]:
                    law = np.abs(walsh(np.exp(-.5j*(matrix @ theta))/np.sqrt(size))/np.sqrt(size))**2
                    law /= law.sum()
                    difference = law-empirical
                    if c['objective'] == 'parity':
                        ids = masks.astype(int) @ (1 << np.arange(n-1, -1, -1))
                        loss = float(np.mean(walsh(difference)[ids]**2))
                    else:
                        loss = float(np.dot(difference, difference)/(size//2 if c['objective'] == 'mse' else 1))
                    loss_error = abs(loss-float(z['loss_history'][history_index]))
                    max_errors['objective'] = max(max_errors['objective'], loss_error)
                    assert loss_error < 1e-11, 'endpoint objective'
                assert np.array_equal(z['history'], z['loss_history'])
                assert len(z['loss_history']) == len(z['gradient_history']) == protocol['steps']+1
                assert np.isfinite(z['loss_history']).all() and np.isfinite(z['gradient_history']).all()
                checked.append(dict(path=path.name, bytes=path.stat().st_size, sha256=digest(path)))
        except Exception as error:
            issues.append(dict(path=path.name, error=f'{type(error).__name__}: {error}'))
    failures = sorted((runs/'checkpoints').glob('*.failure.json'))
    for path in failures:
        record = json.loads(path.read_text())
        row = record['metrics']; key = identity(row)
        assert key in index and key not in checkpoint_ids, 'failure identity'
        reference = index[key]; c, seed, role = config(row), int(row['seed']), row.get('role')
        assert record['specification'] == dict(config=c, seed=seed, protocol=protocol, provenance=provenance, role=role)
        assert path.name == f'{role or config_key(c)}_seed{seed}.failure.json'
        assert row['status'] == reference['status'] == 'failed'
        assert float(row['kl']) == float(row['validation_nll']) == np.inf
        assert row['key'] == reference['key'] == config_key(c)
        assert float(row['beta']) == float(reference['beta']) == protocol['beta']
        assert int(row['parameters']) == int(reference['parameters']) == eigenvalues[c['radius']].shape[1]
        for field in ('error', 'sample_sha256', 'validation_sha256', 'mask_sha256', 'initial_sha256'):
            assert row[field] == reference[field], 'failure ' + field
        assert row['recovery_1000'] is row['coverage_1000'] is None
        assert reference['recovery_1000'] == reference['coverage_1000'] == ''
        checkpoint_ids.add(key)
        checked.append(dict(path=path.name, bytes=path.stat().st_size, sha256=digest(path)))
    assert checkpoint_ids == set(index), 'checkpoint coverage'
    expected_summary = confirmation_statistics(rows, document) if confirmation else development_statistics(rows, protocol)
    equal(json.loads((runs/'summary.json').read_text()), expected_summary, 'independent summary')
    return dict(status='pass' if not issues else 'fail', fits=len(rows), checkpoint_files=len(checked),
                failures=len(failures), nonfinite_KL=sum(not np.isfinite(float(row['kl'])) for row in rows),
                input_sha256={name: digest(runs/name) for name in ('lock.json', 'metrics.csv', 'summary.json')},
                source_hashes=expected_sources, max_absolute_errors=max_errors, summary=expected_summary,
                issues=issues, checkpoints=checked)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--development', type=Path, help='Relocated development directory; frozen hashes must still match')
    args = parser.parse_args()
    try:
        result = json_safe(audit(args.runs, json.loads(args.protocol.read_text()), args.development))
    except Exception as error:
        result = dict(status='fail', issues=[dict(error=f'{type(error).__name__}: {error}')])
    result['protocol_sha256'] = digest(args.protocol)
    result['auditor_sha256'] = digest(__file__)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checkpoints'}, indent=2, allow_nan=False))
    raise SystemExit(0 if result['status'] == 'pass' else 1)
