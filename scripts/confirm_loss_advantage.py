#!/usr/bin/env python3
"""Run one frozen confirmation; validation likelihood is diagnostic only."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import re
import sys
import traceback

import numpy as np
import scipy
from scipy.stats import t
import torch

from iqp_repro import classical, cli, core, loss_search, study, support_baselines
import check_loss_baselines as baselines

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    'scripts/confirm_loss_advantage.py', 'scripts/check_loss_baselines.py',
    'src/iqp_repro/core.py', 'src/iqp_repro/study.py',
    'src/iqp_repro/loss_search.py', 'src/iqp_repro/classical.py',
    'src/iqp_repro/cli.py', 'src/iqp_repro/support_baselines.py',
)


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def canonical(value):
    return json.dumps(clean(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_protocol(protocol):
    seeds = protocol['confirmation_seeds']
    if (len(seeds) < 2 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int for seed in seeds)):
        raise ValueError('At least two distinct integer confirmation seeds required')
    for name in ('n', 'm', 'k', 'steps', 'validation_samples'):
        if type(protocol[name]) is not int or protocol[name] < 1:
            raise ValueError(f'Positive integer {name} required')
    configurations = protocol['configurations']
    if 'parity' not in configurations:
        raise ValueError('The fixed candidate must have role parity')
    for role, entry in configurations.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', role):
            raise ValueError('Roles must be simple filename-safe identifiers')
        if entry['kind'] not in ('iqp', 'classical'):
            raise ValueError(f'Unknown model kind: {role}')
        if entry['config']['beta'] != protocol['beta']:
            raise ValueError('All configurations must use the frozen beta')
    parity = configurations['parity']
    if parity['kind'] != 'iqp' or parity['config']['objective'] != 'parity':
        raise ValueError('Role parity must use IQP sampled parity')
    seen = set()
    for comparison in protocol['comparisons']:
        control, group = comparison['control'], comparison['group']
        if control == 'parity' or control not in configurations:
            raise ValueError('Every control must name a configured non-parity role')
        if group not in ('loss', 'paper', 'support') or (control, group) in seen:
            raise ValueError('Invalid or duplicated comparison group')
        seen.add((control, group))
    controls = {control for control, _ in seen}
    if not controls or controls != set(configurations) - {'parity'}:
        raise ValueError('Every configured control must have a planned comparison')


def verify_source_hashes(protocol, root=ROOT):
    expected = protocol['source_hashes']
    if not set(SOURCE_FILES).issubset(expected):
        raise ValueError('Freeze hashes for the confirmation script and every training dependency')
    root = Path(root).resolve()
    for name, wanted in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or Path(name).is_absolute():
            raise ValueError('Source hashes must use repository-relative paths')
        if study.digest(path) != wanted:
            raise ValueError(f'Frozen source hash mismatch: {name}')


def provenance(protocol, jobs):
    # Verify the actual imported code, including installs outside the checkout.
    modules = (core, study, loss_search, classical, cli, support_baselines)
    for module in modules:
        name = f'src/iqp_repro/{Path(module.__file__).name}'
        if study.digest(module.__file__) != protocol['source_hashes'][name]:
            raise ValueError(f'Imported module differs from frozen source: {name}')
    if study.digest(baselines.__file__) != protocol['source_hashes']['scripts/check_loss_baselines.py']:
        raise ValueError('Imported classical runner differs from frozen source')
    return dict(source_hashes=protocol['source_hashes'], python=sys.version,
                numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                platform=platform.platform(), workers=jobs, classical_torch_threads=1,
                thread_environment={name: os.environ.get(name) for name in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS')})


def seal_output(out, protocol, fingerprint):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seal = dict(protocol=protocol, provenance=fingerprint)
    lock = out / 'lock.json'
    if lock.exists() and json.loads(lock.read_text()) != seal:
        raise ValueError('Frozen protocol, code or environment changed; use a new output directory')
    save_json(lock, seal)


def failed_row(role, seed, protocol, error):
    return dict(role=role, seed=seed, kind=protocol['configurations'][role]['kind'],
                status='failed', kl='Infinity', validation_nll='Infinity',
                recovery_1000=None, coverage_1000=None, sample_sha256=None,
                validation_sha256=None, initial_sha256=None, error=str(error))


def validate_arrays(result, entry, seed, protocol):
    """Check common observations and initialization independently of saved metrics."""
    p, _, _ = core.target(protocol['n'], protocol['beta'])
    samples = np.random.default_rng(seed + 7).choice(len(p), protocol['m'], p=p)
    validation = np.random.default_rng(seed + 50000).choice(len(p), protocol['validation_samples'], p=p)
    for name, expected in [('p', p), ('samples', samples), ('validation', validation)]:
        if not np.array_equal(result[name], expected):
            raise ValueError(f'Incorrect shared {name}')
    q, logq = result['q'], result['logq']
    if (q.shape != p.shape or logq.shape != p.shape or np.any(q < 0)
            or not np.all(np.isfinite(q)) or not np.isclose(q.sum(), 1, atol=1e-12, rtol=0)
            or np.any(np.isnan(logq)) or np.any(np.isposinf(logq))):
        raise ValueError('Invalid probability or log-probability table')
    if not np.allclose(np.exp(logq), q, atol=1e-14, rtol=1e-10):
        raise ValueError('Probability and log-probability tables disagree')
    masks = core.sample_masks(protocol['n'], entry['config']['sigma'], protocol['k'], seed + 222)
    if not np.array_equal(result['masks'], masks):
        raise ValueError('Incorrect sampled masks')
    if entry['kind'] == 'iqp':
        count = len(loss_search.Circuit(protocol['n'], entry['config']['architecture']).edges)
        initial = .01 * np.random.default_rng(seed + 10000 + 7*protocol['k']).standard_normal(count)
        if not np.array_equal(result['initial'], initial):
            raise ValueError('Incorrect IQP initialization')


def run_one(role, seed, protocol, folder, fingerprint):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{role}_seed{seed}.npz'
    failure = path.with_suffix('.failure.json')
    entry = protocol['configurations'][role]
    specification = dict(role=role, seed=seed, protocol=protocol, provenance=fingerprint)
    if path.exists() and failure.exists():
        raise ValueError('Both success and failure artifacts exist for one fit')
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved['specification'])) != specification:
                raise ValueError('Checkpoint provenance mismatch')
            validate_arrays(saved, entry, seed, protocol)
            return json.loads(str(saved['metrics']))
    if failure.exists():
        saved = json.loads(failure.read_text())
        if saved['specification'] != specification:
            raise ValueError('Failure provenance mismatch')
        return saved['metrics']
    try:
        trainer = loss_search.train_configuration if entry['kind'] == 'iqp' else baselines.train_configuration
        result = trainer(entry['config'], seed, protocol)
        validate_arrays(result, entry, seed, protocol)
        row = dict(result.pop('metrics'), role=role, seed=seed, kind=entry['kind'], status='ok', error='')
        # Use the stable exact-KL implementation for both model families.
        positive = result['p'] > 0
        row['kl'] = float(np.sum(result['p'][positive] *
                         (np.log(result['p'][positive]) - result['logq'][positive])))
        row['validation_nll'] = float(-np.mean(result['logq'][result['validation']]))
        for name, field in [('samples', 'sample_sha256'), ('validation', 'validation_sha256'),
                            ('masks', 'mask_sha256'), ('initial', 'initial_sha256')]:
            row[field] = loss_search.array_digest(result[name]) if name in result else None
        row['source_code_sha256'] = hashlib.sha256(canonical(fingerprint['source_hashes']).encode()).hexdigest()
        row = clean(row)
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **result, metrics=canonical(row),
                            specification=canonical(specification),
                            code_hashes=canonical(fingerprint['source_hashes']))
        temporary.replace(path)
        return row
    except Exception as error:
        row = failed_row(role, seed, protocol, f'{type(error).__name__}: {error}')
        save_json(failure, dict(specification=specification, metrics=row, traceback=traceback.format_exc()))
        return row


def interval(values, confidence):
    if len(values) < 2 or not np.isfinite(values).all():
        return None
    mean = float(np.mean(values))
    half = float(t.ppf((1 + confidence)/2, len(values)-1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return [mean-half, mean+half]


def summarize(rows, protocol):
    """Every planned seed participates; failures/nonfinite pairs cannot pass gates."""
    seeds, configurations = protocol['confirmation_seeds'], protocol['configurations']
    indexed = {(row['role'], int(row['seed'])): row for row in rows}
    expected = set(itertools.product(configurations, seeds))
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError('Incomplete, duplicated or unexpected confirmation grid')
    for seed in seeds:
        successful = [indexed[role, seed] for role in configurations if indexed[role, seed]['status'] == 'ok']
        for field in ('sample_sha256', 'validation_sha256'):
            if any(not row.get(field) for row in successful) or len({row[field] for row in successful}) > 1:
                raise ValueError(f'Unmatched confirmation {field}')
        by_architecture = {}
        for row in successful:
            entry = configurations[row['role']]
            if entry['kind'] == 'iqp':
                if not row.get('initial_sha256'):
                    raise ValueError('Missing IQP initialization hash')
                by_architecture.setdefault(entry['config']['architecture'], set()).add(row['initial_sha256'])
        if any(len(values) > 1 for values in by_architecture.values()):
            raise ValueError('IQP initialization differs within the same architecture')
    roles = {}
    for role in configurations:
        group = [indexed[role, seed] for seed in seeds]
        roles[role] = dict(n=len(group), failures=sum(row['status'] != 'ok' for row in group),
            mean_kl=float(np.mean([float(row['kl']) for row in group])),
            nonfinite_kl=sum(not np.isfinite(float(row['kl'])) for row in group),
            mean_recovery_1000=float(np.mean([np.nan if row.get('recovery_1000') is None
                                             else float(row['recovery_1000']) for row in group])))
    controls = sorted({item['control'] for item in protocol['comparisons']})
    confidence = 1 - .05 / len(controls)
    comparisons = {}
    for control in controls:
        pairs = [(indexed['parity', seed], indexed[control, seed]) for seed in seeds]
        with np.errstate(invalid='ignore'):
            delta = np.array([float(p['kl']) - float(c['kl']) for p, c in pairs])
        failed = sum(p['status'] != 'ok' or c['status'] != 'ok' for p, c in pairs)
        nominal = interval(delta, .95) if not failed else None
        adjusted = interval(delta, confidence) if not failed else None
        finite_success = [p['status'] == c['status'] == 'ok' and np.isfinite(value)
                          for (p, c), value in zip(pairs, delta)]
        comparisons[control] = dict(n=len(pairs), failed_pairs=failed,
            nonfinite_pairs=int(np.sum(~np.isfinite(delta))), finite_successful_pairs=sum(finite_success),
            parity_mean_kl=roles['parity']['mean_kl'], control_mean_kl=roles[control]['mean_kl'],
            mean_difference=float(np.mean(delta)),
            parity_wins=sum(valid and value < 0 for valid, value in zip(finite_success, delta)),
            nominal_ci95=nominal, adjusted_confidence=confidence, bonferroni_ci95=adjusted,
            interval_status='finite' if adjusted is not None else 'undefined_nonfinite_or_failed',
            passes=bool(adjusted is not None and adjusted[1] < 0))
    groups = {}
    for group in ('loss', 'paper', 'support'):
        planned = {item['control'] for item in protocol['comparisons'] if item['group'] == group}
        groups[group] = all(comparisons[role]['passes'] for role in planned) if planned else None
    return clean(dict(stage='confirmation', rows=len(rows), seed_count=len(seeds), roles=roles,
        multiplicity=dict(method='Bonferroni simultaneous two-sided 95% family',
                          unique_control_roles=len(controls), individual_confidence=confidence),
        comparisons=comparisons, group_verdicts=groups,
        close_paper_success=groups['loss'] is True and groups['paper'] is True,
        support_aware_success=groups['loss'] is True and groups['support'] is True,
        scope='Frozen configurations, final iterates, all seeds retained; validation NLL is diagnostic only',
        win_count_definition='Parity wins among finite successful pairs; failed/infinite pairs are retained but not wins',
        recovery='Descriptive only; undefined for a role with a failed fit'))


def run_confirmation(protocol_path, out, jobs=4):
    if jobs < 1:
        raise ValueError('jobs must be positive')
    protocol_path, out = Path(protocol_path).resolve(), Path(out)
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    if protocol_path.is_relative_to(ROOT) and protocol_path.relative_to(ROOT).as_posix() in protocol['source_hashes']:
        raise ValueError('The protocol cannot include its own source hash')
    verify_source_hashes(protocol)
    fingerprint = provenance(protocol, jobs)
    fingerprint['protocol_sha256'] = study.digest(protocol_path)
    seal_output(out, protocol, fingerprint)
    folder = out / 'checkpoints'
    folder.mkdir(exist_ok=True)
    tasks = list(itertools.product(protocol['configurations'], protocol['confirmation_seeds']))
    # Preflight resume integrity before submitting any new training.
    for role, seed in tasks:
        for path in (folder / f'{role}_seed{seed}.npz', folder / f'{role}_seed{seed}.failure.json'):
            if path.exists():
                specification = dict(role=role, seed=seed, protocol=protocol, provenance=fingerprint)
                if path.suffix == '.npz':
                    with np.load(path, allow_pickle=False) as data:
                        saved = json.loads(str(data['specification']))
                else:
                    saved = json.loads(path.read_text())['specification']
                if saved != specification:
                    raise ValueError(f'Resume provenance mismatch: {path}')
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run_one, role, seed, protocol, folder, fingerprint): (role, seed)
                   for role, seed in tasks}
        for future in as_completed(futures):
            # Training exceptions are retained by run_one. Infrastructure errors
            # stop the run; rerunning resumes its already completed checkpoints.
            rows.append(future.result())
            if len(rows) % 20 == 0 or len(rows) == len(tasks):
                print(f'Confirmation: {len(rows)}/{len(tasks)} fits retained', flush=True)
    verify_source_hashes(protocol)
    rows.sort(key=lambda row: (row['role'], row['seed']))
    temporary = out / 'metrics.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(out / 'metrics.csv')
    summary = summarize(rows, protocol)
    save_json(out / 'summary.json', summary)
    print(json.dumps(summary['group_verdicts'], indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    run_confirmation(args.protocol, args.out, args.jobs)


if __name__ == '__main__':
    main()
