"""Paired IQP loss comparisons along a nested cyclic pair-gate parameter ladder.

Development selects by shared held-out likelihood. Selection writes a frozen
protocol; confirmation is a separate explicit command and never runs implicitly.
The canonical n=12 radii 1..6 have 12,24,36,48,60,66 parameters.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.stats import t

from . import core, study


class Circuit(study.Circuit):
    def __init__(self, n, radius):
        if (not isinstance(n, (int, np.integer)) or n < 4 or
                not isinstance(radius, (int, np.integer)) or not 1 <= radius <= n//2):
            raise ValueError('require integer n >= 4 and 1 <= radius <= floor(n/2)')
        self.n, self.radius, self.size = n, radius, 2**n
        self.edges = sorted({tuple(sorted((i, (i+d) % n)))
                             for i in range(n) for d in range(1, radius+1)})
        self.indices = np.array([(1 << (n-i-1)) | (1 << (n-j-1)) for i,j in self.edges])


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def canonical(value):
    return json.dumps(safe(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def save_json(path, value):
    study.save_json(path, safe(value))


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def source_hashes():
    return {f'src/iqp_repro/{name}': study.digest(path) for name,path in
            [('parameter_ladder.py',__file__),('core.py',core.__file__),('study.py',study.__file__)]}


def _verify_sources(expected):
    if expected != source_hashes():
        raise ValueError('Source provenance changed; use a new development study')


def _read(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value


def _validate(protocol):
    for field in ('radii', 'development_seeds', 'confirmation_seeds'):
        if not protocol[field] or len(set(protocol[field])) != len(protocol[field]):
            raise ValueError(f'{field} must be nonempty and unique')
        if any(not isinstance(v,int) or isinstance(v,bool) or v < 0 for v in protocol[field]):
            raise ValueError(f'{field} must contain nonnegative integers')
    if len(protocol['development_seeds']) < 2 or len(protocol['confirmation_seeds']) < 2:
        raise ValueError('Both seed cohorts require at least two datasets')
    if set(protocol['development_seeds']) & set(protocol['confirmation_seeds']):
        raise ValueError('Development and confirmation seeds overlap')
    if set(protocol.get('previously_used_seeds', [])) & set(protocol['confirmation_seeds']):
        raise ValueError('Confirmation seed was previously used')
    for radius in protocol['radii']:
        Circuit(protocol['n'], radius)
    if protocol['m'] < 1 or protocol['k'] < 1 or protocol['steps'] < 0 or protocol['validation_samples'] < 1:
        raise ValueError('Invalid sample, mask, or update budget')
    if any(not isinstance(protocol[k],int) for k in ('m','k','steps','validation_samples')):
        raise ValueError('Sample, mask, and update budgets must be integers')
    configs = configurations(protocol)
    if not configs or len({configuration_key(c) for c in configs}) != len(configs):
        raise ValueError('Empty or duplicate parameter grid')
    if any(c['objective'] not in {'parity','mse','scaled-mse'} or not np.isfinite(c['lr']) or c['lr'] <= 0
           or not np.isfinite(c['sigma']) or c['sigma'] <= 0 for c in configs):
        raise ValueError('Invalid loss, learning rate, or bandwidth in grid')
    if 'source_hashes' in protocol:
        _verify_sources(protocol['source_hashes'])


def configurations(protocol):
    result = []
    for radius in protocol['radii']:
        result.extend(dict(radius=radius, objective='parity', sigma=sigma, lr=lr)
                      for sigma,lr in itertools.product(protocol['parity_sigmas'], protocol['parity_learning_rates']))
        result.extend(dict(radius=radius, objective=objective, sigma=protocol.get('mse_sigma', 1.), lr=lr)
                      for objective,lr in itertools.product(protocol['mse_objectives'], protocol['mse_learning_rates']))
    return result


def configuration_key(config):
    return f'r{config["radius"]}_{config["objective"]}_sigma{config["sigma"]:g}_lr{config["lr"]:g}'


def _config(row):
    return dict(radius=int(row['radius']), objective=row['objective'], sigma=float(row['sigma']), lr=float(row['lr']))


def train_configuration(config, seed, protocol):
    started = time.perf_counter()
    if config['objective'] not in {'parity', 'mse', 'scaled-mse'} or config['lr'] <= 0:
        raise ValueError('Unknown objective or invalid learning rate')
    p, support, scores = core.target(protocol['n'], protocol['beta'])
    samples = np.random.default_rng(seed+7).choice(len(p), protocol['m'], p=p)
    validation = np.random.default_rng(seed+50000).choice(len(p), protocol['validation_samples'], p=p)
    empirical = core.empirical(samples, protocol['n'])
    masks = core.sample_masks(protocol['n'], config['sigma'], protocol['k'], seed+222)
    circuit = Circuit(protocol['n'], config['radius'])
    initial = .01*np.random.default_rng(seed+10000+7*protocol['k']).standard_normal(len(circuit.edges))
    theta = initial.copy()
    weights = study.parity_weights(protocol['n'], 'parity', masks) if config['objective']=='parity' else None
    options = protocol['optimizer']
    optimizer = core.Adam(config['lr'], options['beta1'], options['beta2'], options['epsilon'])
    history, gradients = [], []
    for _ in range(protocol['steps']):
        value, gradient = circuit.loss_gradient(theta, empirical, config['objective'], weights)
        history.append(value); gradients.append(float(np.linalg.norm(gradient)))
        theta = optimizer.update(theta, gradient)
    value, gradient = circuit.loss_gradient(theta, empirical, config['objective'], weights)
    history.append(value); gradients.append(float(np.linalg.norm(gradient)))
    q = circuit.probabilities(theta)
    if not all(np.isfinite(v).all() for v in (q, theta, history, gradients)) or q.sum() <= 0:
        raise FloatingPointError('Invalid optimization state')
    q /= q.sum()
    with np.errstate(divide='ignore'):
        logq = np.log(q)
    elite = core.elite(scores, support, samples)
    discovery = core.coverage(q, elite, [1000])
    metrics = dict(config, beta=protocol['beta'], seed=int(seed), key=configuration_key(config),
        parameters=len(theta), status='ok', error='', kl=core.forward_kl(p,q),
        validation_nll=float(-np.mean(logq[validation])),
        recovery_1000=float(discovery['recovery'][0]), coverage_1000=float(discovery['yield'][0]),
        sample_sha256=digest(samples), validation_sha256=digest(validation), mask_sha256=digest(masks),
        initial_sha256=digest(initial), seconds=time.perf_counter()-started)
    return dict(q=q, logq=logq, theta=theta, initial=initial, history=np.array(history),
        loss_history=np.array(history), gradient_history=np.array(gradients), samples=samples,
        validation=validation, masks=masks, p=p, support=support, scores=scores, elite=elite, metrics=metrics)


def run_one(config, seed, protocol, folder, provenance, role=None):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    label = role or configuration_key(config)
    if not all(c.isalnum() or c in '_.-' for c in label):
        raise ValueError('Unsafe checkpoint role')
    path = folder/f'{label}_seed{seed}.npz'
    failure = path.with_suffix('.failure.json')
    specification = dict(config=config, seed=seed, protocol=protocol, provenance=provenance, role=role)
    if path.exists() and failure.exists():
        raise ValueError('Both success and failure checkpoints exist')
    if path.exists():
        with np.load(path, allow_pickle=False) as arrays:
            if json.loads(str(arrays['specification'])) != specification:
                raise ValueError('Checkpoint provenance changed')
            return json.loads(str(arrays['metrics']))
    if failure.exists():
        saved = json.loads(failure.read_text())
        if saved['specification'] != specification:
            raise ValueError('Failure provenance changed')
        return saved['metrics']
    try:
        result = train_configuration(config, seed, protocol)
    except Exception as error:
        row = dict(config, beta=protocol['beta'], seed=int(seed), key=configuration_key(config),
            parameters=len(Circuit(protocol['n'],config['radius']).edges), status='failed', error=repr(error),
            kl=float('inf'), validation_nll=float('inf'), recovery_1000=None, coverage_1000=None,
            sample_sha256='unavailable', validation_sha256='unavailable', mask_sha256='unavailable',
            initial_sha256='unavailable', seconds=None)
        if role is not None:
            row['role'] = role
        save_json(failure, dict(specification=specification, metrics=row))
        return safe(row)
    row = result.pop('metrics')
    if role is not None:
        row['role'] = role
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **result, metrics=canonical(row), specification=canonical(specification))
    temporary.replace(path)
    return safe(row)


def _rows_complete(rows, configs, seeds, protocol, roles=False):
    expected = {(key,seed) for key in configs for seed in seeds}
    seen, hashes = set(), {}
    for row in rows:
        key = row.get('role') if roles else row['key']
        identity = (key,int(row['seed']))
        if identity not in expected or identity in seen or _config(row) != configs[key]:
            raise ValueError('Incomplete, duplicate, or unexpected configuration/seed grid')
        if row['key'] != configuration_key(configs[key]) or float(row['seed']) != int(row['seed']):
            raise ValueError('Checkpoint configuration key or seed is inconsistent')
        seen.add(identity)
        if float(row['beta']) != protocol['beta'] or row['status'] not in {'ok','failed'}:
            raise ValueError('Unexpected beta or fit status')
        if int(row['parameters']) != len(Circuit(protocol['n'],int(row['radius'])).edges):
            raise ValueError('Parameter count disagrees with radius')
        if row['status']=='ok':
            for field,group in [('sample_sha256',()),('validation_sha256',()),
                ('mask_sha256',(float(row['sigma']),)),('initial_sha256',(int(row['radius']),))]:
                identity = (field,int(row['seed']),*group)
                value = row[field]
                if not value or value=='unavailable' or identity in hashes and hashes[identity]!=value:
                    raise ValueError('Data, masks, or within-radius initialization are not shared')
                hashes[identity] = value
    if seen != expected:
        raise ValueError('Incomplete configuration/seed grid')


def _score(row):
    value = float(row['mean_validation_nll']) if row['mean_validation_nll'] is not None else float('nan')
    return value if np.isfinite(value) and row['failures']==0 and row['nonfinite_validation_nll']==0 else float('inf')


def summarize_development(rows, protocol):
    configs = {configuration_key(c):c for c in configurations(protocol)}
    if len(configs) != len(configurations(protocol)):
        raise ValueError('Duplicate configuration in protocol')
    _rows_complete(rows, configs, protocol['development_seeds'], protocol)
    ranking = []
    for key,config in configs.items():
        group = [r for r in rows if r['key']==key]
        nll = np.array([float(r['validation_nll']) for r in group])
        kl = np.array([float(r['kl']) for r in group])
        ranking.append(dict(config, key=key, fits=len(group),
            parameters=len(Circuit(protocol['n'],config['radius']).edges),
            failures=sum(r['status']!='ok' for r in group), nonfinite_validation_nll=int((~np.isfinite(nll)).sum()),
            mean_validation_nll=float(nll.mean()), mean_kl=float(kl.mean())))
    selections = []
    for radius in protocol['radii']:
        family = [r for r in ranking if r['radius']==radius]
        choose = lambda candidates: min(candidates,key=lambda r:(_score(r),r['key']))
        parity = choose([r for r in family if r['objective']=='parity'])
        tuned = choose([r for r in family if r['objective'] in {'mse','scaled-mse'}])
        matched = [r for r in family if r['objective']=='mse' and r['lr']==parity['lr']]
        matched = choose(matched) if matched else None
        eligible = matched is not None and all(np.isfinite(_score(r)) for r in [parity,tuned,matched])
        selections.append(dict(radius=radius,parity=parity,mse_tuned=tuned,mse_matched=matched,eligible=bool(eligible)))
    return dict(stage='development',fits=len(rows),failed_fits=sum(r['status']!='ok' for r in rows),
                ranking=ranking,selections=selections,selection_metric='mean held-out validation NLL; key breaks ties')


def make_confirmation_protocol(summary, protocol, source_hashes, evidence_hashes):
    _validate(protocol)
    expected_count = len(configurations(protocol))*len(protocol['development_seeds'])
    if (summary['fits'] != expected_count or len(summary['selections']) != len(protocol['radii']) or
            {r['radius'] for r in summary['selections']} != set(protocol['radii'])):
        raise ValueError('Incomplete development evidence')
    configs, comparisons = {}, []
    for selected in summary['selections']:
        radius = selected['radius']
        if not selected['eligible'] or any(not np.isfinite(_score(selected[role])) for role in ['parity','mse_tuned','mse_matched']):
            raise ValueError('Ineligible radius: failure or nonfinite validation candidate; do not confirm')
        for role in ('parity','mse_tuned','mse_matched'):
            config = _config(selected[role])
            config['sigma'] = selected['parity']['sigma']
            configs[f'r{radius}_{role}'] = config
        for kind in ('tuned','matched'):
            comparisons.append(dict(radius=radius,kind=kind,parity=f'r{radius}_parity',control=f'r{radius}_mse_{kind}'))
    frozen = dict(stage='confirmation', frozen_at_utc=datetime.now(timezone.utc).isoformat(),protocol=protocol,
        development_seeds=protocol['development_seeds'],confirmation_seeds=protocol['confirmation_seeds'],
        known_cohorts=protocol.get('known_cohorts',{}),configurations=configs,comparisons=comparisons,
        selections=summary['selections'],source_hashes=source_hashes,development_evidence=evidence_hashes,
        primary='Independently validation-tuned MSE versus parity, separately at every radius',
        inference='Paired t intervals; simultaneous Bonferroni alpha=.05 over all radius/control comparisons. Any failed or nonfinite pair blocks a confirmed win or loss.',
        majority_rule='At least four of six tuned-MSE comparisons must have adjusted upper bounds below zero; scope is this tested cyclic pair-gate family.',
        mse_mask_override='MSE sigma is replaced by selected parity sigma solely to share masks. Neither MSE objective uses masks; fixed K preserves initialization and the selected MSE training trajectory.',
        stopping_rule='One frozen cohort, all radii and controls retained, no per-seed selection or additional cohorts.')
    validate_frozen(frozen)
    return frozen


def validate_frozen(frozen):
    """Validate every radius/control assignment before any confirmation draw."""
    if frozen.get('stage') != 'confirmation':
        raise ValueError('Expected a frozen confirmation protocol')
    protocol = frozen['protocol']; _validate(protocol)
    if (frozen['development_seeds'] != protocol['development_seeds'] or
            frozen['confirmation_seeds'] != protocol['confirmation_seeds']):
        raise ValueError('Frozen seed list differs from declared protocol')
    radii = protocol['radii']
    selections = frozen['selections']
    if len(selections) != len(radii) or {r['radius'] for r in selections} != set(radii):
        raise ValueError('Frozen selections must contain every radius exactly once')
    allowed = {configuration_key(c):c for c in configurations(protocol)}
    expected, comparisons = {}, []
    for item in selections:
        radius = item['radius']; parity = item['parity']
        if not item['eligible']:
            raise ValueError('Ineligible development selection')
        for role in ('parity','mse_tuned','mse_matched'):
            row = item[role]
            config = _config(row)
            if config != allowed.get(row['key']) or row['radius'] != radius or not np.isfinite(_score(row)):
                raise ValueError('Selected configuration is not an eligible declared candidate for this radius')
            if role=='parity' and row['objective']!='parity' or role!='parity' and row['objective'] not in {'mse','scaled-mse'}:
                raise ValueError('Role does not match selected objective family')
            if role=='mse_matched' and (row['objective']!='mse' or row['lr']!=parity['lr']):
                raise ValueError('Matched MSE must use ordinary MSE and parity learning rate')
            config['sigma'] = parity['sigma']
            expected[f'r{radius}_{role}'] = config
        comparisons.extend(dict(radius=radius,kind=kind,parity=f'r{radius}_parity',control=f'r{radius}_mse_{kind}')
                           for kind in ('tuned','matched'))
    if frozen['configurations'] != expected:
        raise ValueError('Frozen roles do not match selected radius configurations')
    if sorted(map(canonical,frozen['comparisons'])) != sorted(map(canonical,comparisons)):
        raise ValueError('Frozen comparisons must retain every correct radius/control mapping')
    return frozen


def summarize_confirmation(rows, frozen):
    validate_frozen(frozen)
    protocol = frozen['protocol']
    seeds, configs = frozen['confirmation_seeds'], frozen['configurations']
    _rows_complete(rows,configs,seeds,protocol,roles=True)
    expected_comparisons = {(radius,kind) for radius in protocol['radii'] for kind in ('tuned','matched')}
    comparisons = frozen['comparisons']
    if len(comparisons)!=len(expected_comparisons) or {(r['radius'],r['kind']) for r in comparisons}!=expected_comparisons:
        raise ValueError('Confirmation must retain every declared comparison')
    indexed = {(r['role'],int(r['seed'])):r for r in rows}
    output = []
    for comparison in comparisons:
        pairs = [(indexed[(comparison['parity'],s)],indexed[(comparison['control'],s)]) for s in seeds]
        a,b = (np.array([float(pair[i]['kl']) for pair in pairs]) for i in (0,1))
        valid = np.isfinite(a)&np.isfinite(b)&np.array([p['status']=='ok' and m['status']=='ok' for p,m in pairs])
        with np.errstate(invalid='ignore',divide='ignore'):
            delta = a-b; mean = float(delta.mean())
        relative = float(1-a.mean()/b.mean()) if valid.all() and np.isfinite(b.mean()) and b.mean()>0 else None
        se = float(delta.std(ddof=1)/np.sqrt(len(delta))) if valid.all() and len(delta)>1 else None
        nominal = [mean-t.ppf(.975,len(delta)-1)*se,mean+t.ppf(.975,len(delta)-1)*se] if se is not None else None
        critical = t.ppf(1-.05/(2*len(comparisons)),len(delta)-1) if se is not None else None
        adjusted = [mean-critical*se,mean+critical*se] if se is not None else None
        output.append(dict(comparison,n=len(pairs),finite_pairs=int(valid.sum()),mean_parity_kl=float(a.mean()),
            mean_mse_kl=float(b.mean()),mean_difference=mean,relative_improvement=relative,wins=int(np.sum(valid&(delta<0))),
            se=se,ci95=nominal,adjusted_ci95=adjusted,confirmed_win=bool(adjusted is not None and adjusted[1]<0),
            confirmed_loss=bool(adjusted is not None and adjusted[0]>0)))
    primary = [r for r in output if r['kind']=='tuned']
    required = len(protocol['radii'])//2+1
    wins = sum(r['confirmed_win'] for r in primary)
    return dict(stage='confirmation',fits=len(rows),planned_comparisons=len(comparisons),comparisons=output,
        primary_majority=dict(confirmed_wins=wins,architectures=protocol['radii'],total_count=len(primary),
                             required_wins=required,confirmed_majority=wins>=required),
        matched_confirmed_wins=sum(r['confirmed_win'] for r in output if r['kind']=='matched'),
        scope='Only the tested cyclic pair-gate architecture family; no quantum computational advantage claim.')


def _runtime():
    return dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
                thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})


def _lock(out, specification):
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    path = out/'lock.json'
    if path.exists() and json.loads(path.read_text()) != specification:
        raise ValueError('Output protocol, source, or environment changed; use a new output')
    if not path.exists():
        save_json(path,specification)


def _execute(tasks, out, jobs):
    if jobs < 1:
        raise ValueError('jobs must be positive')
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_one,*task) for task in tasks]
        for i,future in enumerate(as_completed(futures),1):
            rows.append(future.result())
            if i%40==0 or i==len(futures):
                print(f'Parameter ladder: {i}/{len(futures)}',flush=True)
    rows.sort(key=lambda r:(r.get('role',r['key']),int(r['seed'])))
    with (Path(out)/'metrics.csv').open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


def run_development(out, protocol, jobs=3):
    protocol = _read(protocol); _validate(protocol)
    provenance = dict(stage='development',source_hashes=source_hashes(),runtime=_runtime())
    _lock(out,dict(protocol=protocol,provenance=provenance))
    tasks = [(c,s,protocol,Path(out)/'checkpoints',provenance)
             for c,s in itertools.product(configurations(protocol),protocol['development_seeds'])]
    summary = summarize_development(_execute(tasks,out,jobs),protocol)
    save_json(Path(out)/'summary.json',summary)
    return summary


def select_confirmation(development_dir, protocol, out):
    protocol = _read(protocol); _validate(protocol)
    if Path(out).exists():
        raise ValueError('Refusing to overwrite a frozen selection')
    development_dir = Path(development_dir)
    lock = json.loads((development_dir/'lock.json').read_text())
    if lock['protocol'] != protocol:
        raise ValueError('Development protocol changed')
    _verify_sources(lock['provenance']['source_hashes'])
    with (development_dir/'metrics.csv').open() as file:
        rows = list(csv.DictReader(file))
    summary = summarize_development(rows,protocol)
    if canonical(summary) != canonical(json.loads((development_dir/'summary.json').read_text())):
        raise ValueError('Development summary disagrees with complete CSV evidence')
    evidence = {str(development_dir/name):study.digest(development_dir/name)
                for name in ('lock.json','metrics.csv','summary.json')}
    frozen = make_confirmation_protocol(summary,protocol,source_hashes(),evidence)
    save_json(out,frozen)
    return frozen


def run_confirmation(out, frozen_protocol, jobs=3):
    frozen = _read(frozen_protocol)
    if frozen.get('stage')!='confirmation' or not frozen.get('source_hashes'):
        raise ValueError('A frozen confirmation protocol with source hashes is required')
    _verify_sources(frozen['source_hashes']); validate_frozen(frozen)
    provenance = dict(stage='confirmation',source_hashes=source_hashes(),runtime=_runtime(),frozen_protocol=frozen)
    _lock(out,dict(protocol=frozen,provenance=provenance))
    tasks = [(config,s,frozen['protocol'],Path(out)/'checkpoints',provenance,role)
             for role,config in frozen['configurations'].items() for s in frozen['confirmation_seeds']]
    summary = summarize_confirmation(_execute(tasks,out,jobs),frozen)
    save_json(Path(out)/'summary.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    for name in ('develop','select','confirm'):
        sub = commands.add_parser(name); sub.add_argument('--protocol',required=True,type=Path)
        sub.add_argument('--out',required=True,type=Path)
        if name=='select':
            sub.add_argument('--development',required=True,type=Path)
        else:
            sub.add_argument('--jobs',default=3,type=int)
    args = parser.parse_args()
    if args.command=='develop':
        run_development(args.out,args.protocol,args.jobs)
    elif args.command=='select':
        select_confirmation(args.development,args.protocol,args.out)
    else:
        run_confirmation(args.out,args.protocol,args.jobs)


if __name__=='__main__':
    main()
