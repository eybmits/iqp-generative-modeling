"""Compare fixed, target-selected and validation-selected original-ring settings."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def key(row):
    return (float(row['beta']), int(row['seed']), float(row['sigma']),
            int(row['k']), row['model'])


def load_rows(run_dir, protocol):
    rows = list(csv.DictReader((run_dir / 'metrics.csv').open()))
    expected = set(itertools.product(protocol['betas'], protocol['seeds'],
                   protocol['sigma'], protocol['k'], protocol['models']))
    indexed = {key(row): row for row in rows}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError('Incomplete, duplicated or unexpected parameter grid')
    for row in rows:
        for field in ('n', 'm', 'steps'):
            if int(row[field]) != protocol[field]:
                raise ValueError(f'Unexpected {field}')
        if row['protocol'] != 'matched':
            raise ValueError('This analysis requires the matched ring protocol')
        for field in ('kl', 'coverage_1000', 'recovery_1000'):
            row[field] = float(row[field])
    for beta, seed in itertools.product(protocol['betas'], protocol['seeds']):
        instance = [r for r in rows if float(r['beta']) == beta and int(r['seed']) == seed]
        if len({r['samples_sha256'] for r in instance}) != 1:
            raise ValueError('Candidate data are not shared')
        for k in protocol['k']:
            mse = [r for r in instance if int(r['k']) == k and r['model'] == 'iqp-mse']
            for field in ('kl', 'coverage_1000', 'recovery_1000', 'init_seed'):
                if len({r[field] for r in mse}) != 1:
                    raise ValueError('MSE changed with sigma')
        for sigma, k in itertools.product(protocol['sigma'], protocol['k']):
            pair = [indexed[(beta, seed, sigma, k, model)] for model in protocol['models']]
            if pair[0]['init_seed'] != pair[1]['init_seed']:
                raise ValueError('Loss pair does not share initialization')
    return indexed


def add_validation(indexed, run_dir, protocol):
    """Use only held-out samples, not exact target KL, to score candidates."""
    scores = []
    for beta, seed in itertools.product(protocol['betas'], protocol['seeds']):
        validation = None
        for sigma, k in itertools.product(protocol['sigma'], protocol['k']):
            path = run_dir / f'n12_b{beta:g}_s{seed}_sigma{sigma:g}_k{k}.npz'
            with np.load(path, allow_pickle=False) as checkpoint:
                task = json.loads(str(checkpoint['task']))
                if (task['beta'], task['seed'], task['sigma'], task['k']) != (beta, seed, sigma, k):
                    raise ValueError(f'Checkpoint identity mismatch: {path}')
                saved_rows = {row['model']: row for row in json.loads(str(checkpoint['rows']))}
                if validation is None:
                    validation = np.random.default_rng(seed + protocol['validation_seed_offset']).choice(
                        len(checkpoint['p']), protocol['validation_samples'], p=checkpoint['p'])
                for model in protocol['models']:
                    row = indexed[(beta, seed, sigma, k, model)]
                    saved = saved_rows[model]
                    for field in ('kl', 'coverage_1000', 'recovery_1000'):
                        if float(saved[field]) != row[field]:
                            raise ValueError(f'Checkpoint/CSV {field} mismatch: {path}')
                    for field in ('samples_sha256', 'masks_sha256', 'init_seed'):
                        if str(saved[field]) != str(row[field]):
                            raise ValueError(f'Checkpoint/CSV {field} mismatch: {path}')
                    with np.errstate(divide='ignore'):
                        nll = float(-np.mean(np.log(checkpoint[model + '_q'][validation])))
                    row['validation_nll'] = nll
                    scores.append(dict(beta=beta, seed=seed, sigma=sigma, k=k, model=model,
                                       validation_nll=nll,
                                       validation_sha256=hashlib.sha256(validation.tobytes()).hexdigest()))
    return scores


def contrast(pairs):
    a = np.array([p['kl'] for p, _ in pairs])
    b = np.array([m['kl'] for _, m in pairs])
    delta = a - b
    by_seed = {}
    for (p, _), value in zip(pairs, delta):
        by_seed.setdefault(int(p['seed']), []).append(value)
    clustered = np.array([np.mean(values) for values in by_seed.values()])
    finite = bool(np.all(np.isfinite(clustered)))
    enough = len(clustered) > 1
    half = (t.ppf(.975, len(clustered) - 1) * clustered.std(ddof=1) / np.sqrt(len(clustered))
            if finite and enough else float('nan'))
    return dict(instances=len(pairs), seed_clusters=len(clustered), parity_mean_kl=float(a.mean()),
                mse_mean_kl=float(b.mean()), mean_difference=float(delta.mean()),
                relative_mean_kl_reduction=float(1 - a.mean() / b.mean()),
                descriptive_ci95=[float(clustered.mean() - half), float(clustered.mean() + half)],
                interval_status='descriptive_finite' if finite and enough else
                                ('undefined_nonfinite' if not finite else 'requires_multiple_seeds'),
                nonfinite_pairs=int(np.sum(~np.isfinite(delta))),
                parity_wins=int(np.sum(delta < 0)),
                parity_recovery_1000=float(np.mean([p['recovery_1000'] for p, _ in pairs])),
                mse_recovery_1000=float(np.mean([m['recovery_1000'] for _, m in pairs])))


def select(candidates, field):
    """Deterministic ties; validation selection reads no test metric."""
    return min(candidates, key=lambda row: (row[field], float(row['sigma']), int(row['k'])))


def analyze(indexed, betas, protocol):
    instances = list(itertools.product(betas, protocol['seeds']))
    settings = list(itertools.product(protocol['sigma'], protocol['k']))
    ref = (protocol['reference']['sigma'], protocol['reference']['k'])

    def get(instance, setting, model):
        return indexed[(*instance, *setting, model)]

    table = []
    for setting in settings:
        pairs = [(get(i, setting, 'iqp-parity'), get(i, setting, 'iqp-mse')) for i in instances]
        table.append(dict(sigma=setting[0], k=setting[1], **contrast(pairs)))
    best = min(table, key=lambda r: (r['parity_mean_kl'], r['sigma'], r['k']))
    best_mse = min(table, key=lambda r: (r['mse_mean_kl'], r['sigma'], r['k']))
    best_setting = (best['sigma'], best['k'])
    mse_setting = (best_mse['sigma'], best_mse['k'])
    comparisons = {name: [] for name in (
        'reference', 'best_fixed_parity_vs_matched_mse', 'best_fixed_each',
        'oracle_parity_vs_matched_mse', 'oracle_parity_vs_reference_mse', 'oracle_each',
        'validation_parity_vs_matched_mse', 'validation_each')}
    selected_rows = []
    for instance in instances:
        ps = [get(instance, setting, 'iqp-parity') for setting in settings]
        ms = [get(instance, (ref[0], k), 'iqp-mse') for k in protocol['k']]
        p_fixed = get(instance, best_setting, 'iqp-parity')
        comparisons['reference'].append((get(instance, ref, 'iqp-parity'), get(instance, ref, 'iqp-mse')))
        comparisons['best_fixed_parity_vs_matched_mse'].append((p_fixed, get(instance, best_setting, 'iqp-mse')))
        comparisons['best_fixed_each'].append((p_fixed, get(instance, mse_setting, 'iqp-mse')))
        for rule, field in [('oracle', 'kl'), ('validation', 'validation_nll')]:
            p, m = select(ps, field), select(ms, field)
            matched = get(instance, (float(p['sigma']), int(p['k'])), 'iqp-mse')
            comparisons[rule + '_parity_vs_matched_mse'].append((p, matched))
            comparisons[rule + '_each'].append((p, m))
            if rule == 'oracle':
                comparisons['oracle_parity_vs_reference_mse'].append((p, get(instance, ref, 'iqp-mse')))
            selected_rows.append(dict(beta=instance[0], seed=instance[1], rule=rule,
                parity_sigma=float(p['sigma']), parity_k=int(p['k']), mse_k=int(m['k']),
                parity_kl=p['kl'], matched_mse_kl=matched['kl'], selected_mse_kl=m['kl'],
                parity_validation_nll=p['validation_nll'], mse_validation_nll=m['validation_nll']))
    return dict(best_fixed_parity=dict(sigma=best['sigma'], k=best['k']),
                best_fixed_mse_init_k=best_mse['k'], settings=table,
                comparisons={name: contrast(pairs) for name, pairs in comparisons.items()}), selected_rows


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'protocols/parity-setting-check.json')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    indexed = load_rows(args.run_dir, protocol)
    validation = add_validation(indexed, args.run_dir, protocol)
    full, choices = analyze(indexed, protocol['betas'], protocol)
    fixed, _ = analyze(indexed, [.9], protocol)
    summary = dict(scope=protocol['scope'], uncertainty=protocol['uncertainty'],
                   resource_accounting=protocol['resource_accounting'],
                   source_sha256={name: digest(path) for name, path in {
                       'metrics': args.run_dir / 'metrics.csv', 'protocol': args.protocol,
                       'analysis': Path(__file__)}.items()}, full_sweep=full, beta_09=fixed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(safe(summary), indent=2, allow_nan=False) + '\n')
    write_csv(args.out / 'selected-settings.csv', choices)
    write_csv(args.out / 'validation-scores.csv', validation)
    print(json.dumps(safe({name: data['comparisons'] for name, data in
                          [('full_sweep', full), ('beta_09', fixed)]}), indent=2))


if __name__ == '__main__':
    main()
