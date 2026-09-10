#!/usr/bin/env python3
"""Select once on development likelihood and freeze the confirmation protocol.

Target KL is never consulted by this selection procedure. The two-tier rule
first seeks a benefit over tuned, support-aware controls; otherwise it selects
a narrower comparison against MSE and the original paper's classical defaults.
All stronger classical controls remain in the confirmation protocol either way.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

from iqp_repro import loss_search


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(value, context):
    """Read preserved Infinity explicitly, and reject undefined evidence."""
    number = float(value)
    if math.isnan(number) or number == -math.inf:
        raise ValueError(f'Undefined or negative-infinite metric: {context}')
    return number


def selection_score(row):
    value = numeric(row['mean_validation_nll'], row['key'])
    if row.get('failures', 0) or row.get('nonfinite_validation_nll', 0):
        return math.inf
    return value


def _baseline_driver():
    path = Path(__file__).with_name('check_loss_baselines.py')
    spec = importlib.util.spec_from_file_location('_loss_baseline_grid', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _boolean(value):
    if value in (True, 'True', 'true'):
        return True
    if value in (False, 'False', 'false'):
        return False
    raise ValueError(f'Invalid support_aware flag: {value}')


def _configuration(row, kind):
    result = dict(beta=float(row['beta']), architecture=row['architecture'],
                  sigma=float(row['sigma']), lr=float(row['lr']))
    if kind == 'iqp':
        result['objective'] = row['objective']
    else:
        steps = int(row['steps'])
        if float(row['steps']) != steps:
            raise ValueError('Noninteger classical step count')
        result.update(model=row['model'], steps=steps,
                      support_aware=_boolean(row['support_aware']), l2=float(row['l2']))
    return result


def _identity(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False)


def expected_configurations(development):
    protocol = dict(development, parity_sigmas=development['sigmas'], mse_sigma=1.)
    iq = loss_search.configurations(protocol)
    classical = [dict(config, beta=beta) for beta in development['betas']
                 for config in _baseline_driver().configuration_grid(development)]
    return {'iqp': iq, 'classical': classical}


def _assert_metric(actual, expected, context):
    a, b = numeric(actual, context), numeric(expected, context)
    if a != b and (not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > 1e-12):
        raise ValueError(f'Ranking disagrees with CSV: {context}')


def _mean(values):
    return math.inf if any(value == math.inf for value in values) else math.fsum(values)/len(values)


def validate_rankings(iq_summary, classical_ranking, iq_rows, classical_rows, development):
    """Require the entire declared grid and independently rebuild its ranking."""
    expected = expected_configurations(development)
    seeds = development['development_seeds']
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Development seeds must be nonempty and unique')
    hashes = {'sample_sha256': {}, 'validation_sha256': {}, 'initial_sha256': {}, 'mask_sha256': {}}
    recomputed = {}
    counts = {}
    for kind, rows, ranking in [('iqp', iq_rows, iq_summary['ranking']),
                                ('classical', classical_rows, classical_ranking)]:
        allowed = {_identity(config): config for config in expected[kind]}
        if len(allowed) != len(expected[kind]):
            raise ValueError('Duplicated configuration in development protocol')
        expected_pairs = {(identity, seed) for identity in allowed for seed in seeds}
        seen = set()
        groups = {}
        driver = _baseline_driver() if kind == 'classical' else loss_search
        for row in rows:
            config = _configuration(row, kind)
            identity = _identity(config)
            seed = int(row['seed'])
            if float(row['seed']) != seed or (identity, seed) not in expected_pairs or (identity, seed) in seen:
                raise ValueError(f'Incomplete, duplicate, or unexpected {kind} grid row')
            seen.add((identity, seed))
            expected_key = driver.configuration_key(config)
            if row['key'] != expected_key:
                raise ValueError(f'Configuration key mismatch: {row["key"]}')
            status = row.get('status', 'ok')
            if status not in {'ok', 'failed'}:
                raise ValueError(f'Unknown fit status: {status}')
            failed = status == 'failed'
            kl = numeric(row['kl'], f'{kind}/{expected_key}/kl')
            nll = numeric(row['validation_nll'], f'{kind}/{expected_key}/validation_nll')
            if failed and (kl != math.inf or nll != math.inf):
                raise ValueError('Failed fit must retain infinite KL and validation NLL')
            groups.setdefault(identity, []).append(dict(seed=seed, kl=kl, nll=nll, failed=failed,
                                                        recovery=row.get('recovery_1000')))
            if not failed:
                for field in ('sample_sha256', 'validation_sha256') + (('initial_sha256', 'mask_sha256') if kind == 'iqp' else ()):
                    value = row.get(field, '')
                    if not re.fullmatch('[0-9a-f]{64}', value):
                        raise ValueError(f'Missing or invalid data hash: {field}')
                    group = (config['beta'], seed)
                    if field == 'initial_sha256':
                        group += (config['architecture'],)
                    elif field == 'mask_sha256':
                        group += (config['sigma'],)
                    if group in hashes[field] and hashes[field][group] != value:
                        raise ValueError(f'Unshared data or initialization: {field}')
                    hashes[field][group] = value
        if seen != expected_pairs:
            raise ValueError(f'Incomplete {kind} development grid')
        ranked = {}
        for item in ranking:
            raw_config = item if kind == 'iqp' else dict(item['config'], beta=item['beta'])
            identity = _identity(_configuration(raw_config, kind))
            if identity not in allowed or identity in ranked:
                raise ValueError(f'Duplicate or unexpected {kind} ranking configuration')
            group = groups[identity]
            if item['key'] != driver.configuration_key(allowed[identity]):
                raise ValueError('Ranking key mismatch')
            count_field = 'fits' if kind == 'iqp' else 'runs'
            if item[count_field] != len(seeds):
                raise ValueError('Ranking has incomplete seed count')
            failures = sum(value['failed'] for value in group)
            nonfinite = sum(not math.isfinite(value['nll']) for value in group)
            if kind == 'iqp' and (item['failures'] != failures or item['nonfinite_validation_nll'] != nonfinite):
                raise ValueError('Ranking failure counts disagree with CSV')
            mean_kl = _mean([value['kl'] for value in group])
            mean_nll = _mean([value['nll'] for value in group])
            _assert_metric(item['mean_kl'], mean_kl, item['key']+'/KL')
            _assert_metric(item['mean_validation_nll'], mean_nll, item['key']+'/NLL')
            if kind == 'iqp' and not failures:
                expected_parameters = len(loss_search.Circuit(development['n'], allowed[identity]['architecture']).edges)
                if item['parameters'] != expected_parameters:
                    raise ValueError('IQP ranking parameter count does not match architecture')
                recovery = _mean([numeric(value['recovery'], 'recovery') for value in group])
                _assert_metric(item['mean_recovery_1000'], recovery, item['key']+'/recovery')
            ranked[identity] = dict(item, mean_kl=mean_kl, mean_validation_nll=mean_nll,
                                    failures=failures, nonfinite_validation_nll=nonfinite)
        if set(ranked) != set(allowed):
            raise ValueError(f'Incomplete {kind} development ranking')
        recomputed[kind] = list(ranked.values())
        counts[kind] = dict(configurations=len(allowed), rows=len(rows),
                            failures=sum(value['failed'] for group in groups.values() for value in group))
    if iq_summary['fits'] != counts['iqp']['rows'] or iq_summary['failed_fits'] != counts['iqp']['failures']:
        raise ValueError('IQP summary total disagrees with complete CSV grid')
    return recomputed['iqp'], recomputed['classical'], dict(
        grids=counts, seed_count=len(seeds), rankings_recomputed=True,
        shared_hash_groups={field: len(values) for field, values in hashes.items()})


def validate_inputs(iqp, classical, development_path, development, root):
    """Verify run locks against current sources before consuming their evidence."""
    iq_lock = json.loads((iqp/'lock.json').read_text())
    cl_lock = json.loads((classical/'lock.json').read_text())
    if iq_lock.get('canonical_protocol') != development:
        raise ValueError('IQP run lock does not contain this canonical development protocol')
    iq_protocol = iq_lock['protocol']
    for field in ('n', 'm', 'k', 'steps', 'betas', 'development_seeds', 'validation_samples',
                  'architectures', 'parity_learning_rates', 'mse_learning_rates', 'mse_objectives'):
        if iq_protocol[field] != development[field]:
            raise ValueError(f'IQP protocol mismatch: {field}')
    if iq_protocol['parity_sigmas'] != development['sigmas'] or iq_protocol['mse_sigma'] != 1.:
        raise ValueError('IQP protocol mask settings differ')
    for field in ('beta1', 'beta2', 'epsilon'):
        if iq_protocol['optimizer'][field] != development['optimizer'][field]:
            raise ValueError('IQP optimizer differs from development protocol')
    paths = {
        'iqp': (iq_lock['fingerprint'], {
            'loss_search_sha256': root/'src/iqp_repro/loss_search.py',
            'core_sha256': root/'src/iqp_repro/core.py', 'study_sha256': root/'src/iqp_repro/study.py',
            'canonical_protocol_sha256': development_path}),
        'classical': (cl_lock, {
            'script_sha256': root/'scripts/check_loss_baselines.py',
            'support_baselines_sha256': root/'src/iqp_repro/support_baselines.py',
            'classical_sha256': root/'src/iqp_repro/classical.py',
            'metrics_source_sha256': root/'src/iqp_repro/cli.py',
            'core_sha256': root/'src/iqp_repro/core.py', 'protocol_sha256': development_path})}
    for kind,(lock, files) in paths.items():
        for field,path in files.items():
            if lock.get(field) != sha(path):
                raise ValueError(f'{kind} run source/protocol hash mismatch: {field}')
    with (iqp/'metrics.csv').open() as file:
        iq_rows = list(csv.DictReader(file))
    with (classical/'metrics.csv').open() as file:
        cl_rows = list(csv.DictReader(file))
    result = validate_rankings(json.loads((iqp/'summary.json').read_text()),
        json.loads((classical/'ranking.json').read_text()), iq_rows, cl_rows, development)
    result[2]['source_locks_verified'] = True
    result[2]['run_environments'] = {
        kind: {field: lock.get(field) for field in ('python', 'numpy', 'torch', 'scipy', 'platform')}
        for kind,lock in [('iqp',iq_lock['fingerprint']),('classical',cl_lock)]}
    return result


def choose(iqp_ranking, classical_ranking, development):
    candidates = []
    for beta in development["betas"]:
        iq = [r for r in iqp_ranking if r["beta"] == beta]
        cl = [r for r in classical_ranking if r["beta"] == beta]
        order = lambda r: (selection_score(r), r["key"])
        parity = min((r for r in iq if r["objective"] == "parity"), key=order)
        mse_global = min((r for r in iq if "mse" in r["objective"]), key=order)
        mse_same = min((r for r in iq if "mse" in r["objective"] and
                        r["architecture"] == parity["architecture"]), key=order)
        mse_matched = next(r for r in iq if r["objective"] == "mse" and
                          r["architecture"] == parity["architecture"] and r["lr"] == parity["lr"])
        classical = {}
        for model in development["classical"]:
            for aware in (False, True):
                classical[f'{model}_{"support" if aware else "paper"}'] = min(
                    (r for r in cl if r["config"]["model"] == model and
                     r["config"]["support_aware"] == aware), key=order)
        mse = dict(mse_matched=mse_matched, mse_same=mse_same, mse_global=mse_global)
        p_nll = selection_score(parity)
        gaps = {role: p_nll-selection_score(r) if math.isfinite(p_nll) and math.isfinite(selection_score(r))
                else math.inf for role, r in {**mse, **classical}.items()}
        candidates.append(dict(beta=beta, parity=parity, mse=mse, classical=classical,
            validation_gaps=gaps,
            support_worst_gap=max(v for k,v in gaps.items() if k.startswith("mse_") or k.endswith("_support")),
            paper_worst_gap=max(v for k,v in gaps.items() if k.startswith("mse_") or k.endswith("_paper"))))
    tier = "support" if any(r["support_worst_gap"] < 0 for r in candidates) else "paper"
    eligible = [r for r in candidates if r[f"{tier}_worst_gap"] < 0]
    if not eligible:
        raise ValueError("No development regime beats the declared MSE and paper controls; do not draw confirmation data")
    selected = min(eligible, key=lambda r: (r[f"{tier}_worst_gap"], abs(r["beta"]-.9), r["beta"]))
    return dict(selection_tier=tier, selected=selected, candidates=candidates)


def freeze(selection, development, sources, evidence):
    picked = selection["selected"]
    beta, parity = picked["beta"], picked["parity"]
    configurations = {}
    aliases = {}
    comparisons = []

    def add(role, kind, config, group=None):
        item = dict(kind=kind, config=config)
        canonical_role = next((name for name,value in configurations.items() if value == item), None)
        if canonical_role is None:
            canonical_role = role
            configurations[role] = item
        aliases[role] = canonical_role
        if group is not None and not any(r["control"] == canonical_role for r in comparisons):
            comparisons.append(dict(control=canonical_role, group=group))

    iqfields = ("architecture", "objective", "sigma", "lr")
    add("parity", "iqp", dict(beta=beta, **{k:parity[k] for k in iqfields}))
    for role, row in picked["mse"].items():
        # Sigma never enters an MSE objective. Supply identical masks within
        # this cohort; fixed K keeps its initial angles identical to development.
        config = dict(beta=beta, **{k:row[k] for k in iqfields})
        config["sigma"] = parity["sigma"]
        add(role, "iqp", config, "loss")
    for role, row in picked["classical"].items():
        add(role, "classical", dict(beta=beta, **row["config"]),
            "support" if row["config"]["support_aware"] else "paper")
    return dict(study="Selected-regime IQP loss advantage, one fresh confirmation",
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        **{k:development[k] for k in ("n", "m", "k", "steps", "validation_samples", "optimizer")},
        beta=beta, confirmation_seeds=list(range(5001, 5121)),
        seed_reservation_note="Development initially reserved seeds4001-4120. Before confirmation these were replaced with5001-5120 because4001 was used in a6-qubit unit smoke. No paper-cohort results from either range were used for selection.",
        configurations=configurations, role_aliases=aliases, comparisons=comparisons,
        source_hashes=sources, development_evidence=evidence,
        selection_tier=selection["selection_tier"],
        selection_rule="Minimum mean validation NLL per family within beta; choose beta with smallest worst parity-minus-control NLL gap. Prefer the support-aware tier if any beta has all negative gaps, otherwise use the MSE plus paper-default tier. Target KL never selects. All support-aware controls are retained in either tier.",
        primary_metric="Mean exact forward KL, no probability floor or omitted seed",
        inference="Two-sided paired t intervals; Bonferroni familywise alpha=.05 over all distinct control roles. Loss, paper and support gates each require every corresponding upper bound below zero. Infinite or failed outcomes cannot pass. Recovery and win counts descriptive.",
        stopping_rule="Run these 120 datasets once. No per-dataset selection, restarts, retuning, target-KL choice, or additional confirmation cohorts. Retain every result. Confirmation validation likelihood is diagnostic only.",
        resources="Each model sees the same 200 training draws per seed. Development additionally used 2000 shared validation draws on each of40 previously examined datasets. Classical final update counts may be below600 because a global development search selected them. This is not equal search count or equal compute across families.",
        scope="A modified-protocol simulation extension of the paper. Success against original unrestricted classical defaults is distinct from success against support-aware trained classical models. It is not the printed18% result, an exact paper replication, or quantum computational advantage.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iqp", type=Path, required=True)
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--development-protocol", type=Path, default=Path("protocols/loss-advantage-development.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection-out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.selection_out.exists():
        raise ValueError("Refusing to overwrite a frozen selection; preserve existing study outcomes")
    development = json.loads(args.development_protocol.read_text())
    root = Path(__file__).resolve().parents[1]
    iq, cl, validation = validate_inputs(args.iqp, args.classical,
                                        args.development_protocol, development, root)
    selection = choose(iq, cl, development)
    selection['development_validation'] = validation
    paths = ["src/iqp_repro/"+name+".py" for name in
             ("core", "study", "loss_search", "classical", "cli", "support_baselines")]
    paths += ["scripts/"+name+".py" for name in
              ("check_loss_baselines", "select_loss_advantage", "confirm_loss_advantage")]
    sources = {name: sha(root/name) for name in paths}
    evidence = {str(path): sha(path) for path in [args.development_protocol,
        args.iqp/"metrics.csv", args.iqp/"summary.json", args.iqp/"lock.json",
        args.classical/"metrics.csv", args.classical/"ranking.json", args.classical/"lock.json"]}
    protocol = freeze(selection, development, sources, evidence)
    loss_search.save_json(args.selection_out, selection)
    loss_search.save_json(args.out, protocol)
    print(json.dumps(dict(beta=protocol["beta"], tier=protocol["selection_tier"],
                         configurations=protocol["configurations"], comparisons=protocol["comparisons"]), indent=2))


if __name__ == "__main__":
    main()
