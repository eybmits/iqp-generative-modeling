"""One interface: train matched instances, summarize, or render archived results."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from scipy.stats import t
from . import core

MODELS = ('iqp-parity', 'iqp-mse', 'ising-parity', 'ising-nll', 'transformer', 'maxent')
BUDGETS = np.array([1000, 2000, 5000])


def json_safe(value):
    """JSON has no infinity or NaN; preserve infinity and label undefined values."""
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float,np.floating)) and not np.isfinite(value):
        return None if np.isnan(value) else ('Infinity' if value>0 else '-Infinity')
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n')


def protocol_fingerprint():
    """Cover RNG, optimization and metrics, excluding plotting/summary-only edits."""
    module = ast.parse(Path(__file__).read_text())
    selected = [node for node in module.body if
        isinstance(node,ast.FunctionDef) and node.name in ('metrics','train_instance') or
        isinstance(node,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='BUDGETS' for v in node.targets)]
    content = ast.dump(ast.Module(body=selected,type_ignores=[]),include_attributes=False).encode()
    content += b''.join((Path(__file__).parent/name).read_bytes() for name in ('core.py','classical.py'))
    return hashlib.sha256(content).hexdigest()


def task_identity(task):
    fields = ('n','beta','seed','sigma','k','m','steps','models','protocol','preset','threads','implementation_sha256')
    return json_safe({key:task[key] for key in fields})


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def interval(values):
    x = np.asarray(values, float)
    finite = bool(np.all(np.isfinite(x)))
    with np.errstate(invalid='ignore'):
        mean = float(x.mean())
        median = float(np.median(x))
    half = float(t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))) if finite and len(x)>1 else None
    return dict(n=len(x), finite_n=int(np.isfinite(x).sum()), mean=mean, median=median,
                ci95_halfwidth=half, interval_status='finite' if half is not None else
                ('undefined_nonfinite_values' if not finite else 'requires_multiple_seeds'))


def metrics(p, q, support, unseen, logq=None):
    """True forward KL plus support/region attribution; no probability floor."""
    pos = p > 0
    if logq is None:
        with np.errstate(divide='ignore'):
            logq = np.log(q)
    kl = float(np.sum(p[pos] * (np.log(p[pos]) - logq[pos])))
    mass = float(q[support].sum())
    leakage = -float(np.log(mass)) if mass > 0 else float('inf')
    # Conditional forward KL is diagnostic: support knowledge is supplied at evaluation.
    conditional = kl - leakage if np.isfinite(kl) else float('inf')
    region_mass_kl, shape_kl = 0., 0.
    for region in (unseen, support & ~unseen):
        p_mass, q_mass = p[region].sum(), q[region].sum()
        if p_mass > 0 and q_mass > 0:
            region_mass_kl += p_mass * np.log(p_mass / (q_mass / mass))
            valid = region & pos
            shape_kl += np.sum(p[valid] * (np.log(p[valid]/p_mass) - logq[valid] + np.log(q_mass)))
        elif p_mass > 0:
            region_mass_kl = float('inf')
            shape_kl = float('inf')
    return dict(kl=kl, support_mass=mass, support_kl=leakage,
                conditional_kl=conditional, unseen_mass=float(q[unseen].sum()),
                region_mass_kl=float(region_mass_kl), within_region_kl=float(shape_kl))


def train_instance(task):
    """Each instance owns its data, masks, model initializations and checkpoint."""
    import torch
    from . import classical
    torch.set_num_threads(task['threads'])
    n, beta, seed, sigma, k = task['n'], task['beta'], task['seed'], task['sigma'], task['k']
    path = Path(task['out']) / f'n{n}_b{beta:g}_s{seed}_sigma{sigma:g}_k{k}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if 'task' not in z or json.loads(str(z['task'])) != task_identity(task):
                raise ValueError(f'Checkpoint protocol mismatch: {path}; use a new --out')
            rows = json.loads(str(z['rows']))
            for row in rows:
                if any(row[key] != task[key] for key in ('n','beta','seed','sigma','k','m','steps','protocol')):
                    raise ValueError(f'Checkpoint row metadata mismatch: {path}')
            return rows, True
    start = time.monotonic()
    p, support, scores = core.target(n, beta)
    sample_seed = seed if task['preset']=='legacy' else seed+7
    samples = np.random.default_rng(sample_seed).choice(len(p), task['m'], p=p)
    empirical = core.empirical(samples, n)
    masks = core.sample_masks(n, sigma, k, seed+222)
    indices = core.mask_indices(masks)
    moments = core.fwht(empirical)[indices]
    elite = core.elite(scores, support, samples, method='threshold' if task['protocol']=='matched' else 'topk')
    unseen = support & (empirical == 0)
    bits = core.bits_table(n)
    arrays = dict(samples=samples, masks=masks, p=p, elite=elite, support=support, scores=scores)
    rows = []
    for model in task['models']:
        tick = time.monotonic()
        init = seed + 10000 + 7*k
        if model in ('iqp-parity','iqp-mse'):
            if model=='iqp-mse' and task['protocol']=='source':
                init = seed+20000+7*512
            result = core.train_iqp(n, empirical, masks, steps=task['steps'], lr=.05,
                seed_init=init, loss='parity' if model=='iqp-parity' else 'mse',
                mse_domain='support' if task['protocol']=='matched' else 'cube')
        elif model=='ising-parity':
            init = seed+30001
            result = classical.train_ising(bits, indices, moments, empirical,
                topology='nn_nnn', loss='parity', seed=init, steps=task['steps'], lr=.05)
        elif model=='ising-nll':
            init = seed+30004
            result = classical.train_ising(bits, indices, moments, empirical,
                topology='dense', loss='nll', seed=init, steps=task['steps'], lr=.05)
        elif model=='maxent':
            init = seed+36001
            result = classical.train_maxent(indices, moments, n=n, seed=init,
                steps=task['steps'], lr=.05)
        else:
            init = seed+35501
            result = classical.train_transformer(bits, samples, seed=init,
                epochs=task['steps'], lr=.001, batch_size=256)
        q = result['q']
        row = {key:task[key] for key in ('n','beta','seed','sigma','k','m','steps','protocol')}
        row.update(model=model, init_seed=init, sample_seed=sample_seed, mask_seed=seed+222,
            samples_sha256=digest(samples), masks_sha256=digest(masks),
            unique_masks=len(np.unique(indices)), elite_size=int(elite.sum()),
            elapsed_seconds=time.monotonic()-tick)
        row.update(metrics(p, q, support, unseen, result.get('logq')))
        cov = core.coverage(q, elite, BUDGETS)
        conditioned = np.where(support,q,0) / q[support].sum()
        conditional_cov = core.coverage(conditioned, elite, BUDGETS)
        for i,budget in enumerate(BUDGETS):
            row[f'coverage_{budget}'] = float(cov['yield'][i])
            row[f'recovery_{budget}'] = float(cov['recovery'][i])
            row[f'conditional_coverage_{budget}'] = float(conditional_cov['yield'][i])
        for name in ('q','theta','loss_history','logq'):
            if name in result:
                arrays[f'{model}_{name}'] = result[name]
        rows.append(row)
    for domain in ('cube','support'):
        arrays[f'spectral_{domain}'] = core.spectral(empirical, masks, support if domain=='support' else None)
    arrays['spectral_unique_support'] = core.spectral(empirical, np.unique(masks,axis=0), support)
    # Infinity is retained in the numeric CSV; JSON distinguishes it explicitly.
    safe_rows = json_safe(rows)
    arrays['rows'] = np.array(json.dumps(safe_rows))
    arrays['task'] = np.array(json.dumps(task_identity(task)))
    arrays['seconds'] = np.array(time.monotonic()-start)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return safe_rows, False


def summarize(out):
    out = Path(out)
    rows = []
    for path in sorted(out.glob('n*_b*_s*_sigma*_k*.npz')):
        with np.load(path, allow_pickle=False) as z:
            rows.extend(json.loads(str(z['rows'])))
    if not rows:
        raise ValueError(f'No checkpoints in {out}')
    with (out/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Aggregate repeated beta instances within a seed first, preserving pairing.
    groups = {}
    for row in rows:
        key = (row['n'],row['sigma'],row['k'],row['model'])
        groups.setdefault(key,{}).setdefault(row['seed'],[]).append(row)
    model_stats = []
    for key,by_seed in sorted(groups.items()):
        values = [float(r['kl']) for runs in by_seed.values() for r in runs]
        if not np.all(np.isfinite(values)):
            model_stats.append(dict(n=key[0],sigma=key[1],k=key[2],model=key[3],kl='Infinity'))
            continue
        clusters = [np.mean([float(r['kl']) for r in runs]) for runs in by_seed.values()]
        ci = interval(clusters)
        model_stats.append(dict(n=key[0],sigma=key[1],k=key[2],model=key[3],instances=len(values),
            mean_kl=float(np.mean(values)),median_kl=float(np.median(values)),seed_cluster_ci95=ci,
            **{f'mean_{metric}':float(np.mean([float(r[metric]) for runs in by_seed.values() for r in runs]))
               for metric in ('support_mass','conditional_kl','coverage_1000','conditional_coverage_1000')
               if all(metric in r and r[metric] is not None for runs in by_seed.values() for r in runs)}))
    paired = {}
    for row in rows:
        if row['model'] in ('iqp-parity','iqp-mse'):
            key = tuple(row[k] for k in ('n','beta','seed','sigma','k'))
            paired.setdefault(key,{})[row['model']] = row
    differences = []
    for key,pair in sorted(paired.items()):
        if len(pair)!=2:
            continue
        a,b = pair['iqp-parity'],pair['iqp-mse']
        assert a['samples_sha256']==b['samples_sha256']
        differences.append(dict(n=key[0],beta=key[1],seed=key[2],sigma=key[3],k=key[4],
            delta_kl=float(a['kl'])-float(b['kl']),
            delta_coverage_1000=a['coverage_1000']-b['coverage_1000']))
    contrasts = []
    for key in sorted(set((r['n'],r['sigma'],r['k']) for r in differences)):
        selected = [r for r in differences if (r['n'],r['sigma'],r['k'])==key]
        clusters = {}
        for row in selected:
            clusters.setdefault(row['seed'],[]).append(row['delta_kl'])
        contrasts.append(dict(n=key[0],sigma=key[1],k=key[2],instances=len(selected),
            parity_wins=sum(r['delta_kl']<0 for r in selected),
            undefined_pairs=int(sum(np.isnan(r['delta_kl']) for r in selected)),
            paired_delta_kl=interval([np.mean(v) for v in clusters.values()]),
            mean_delta_coverage_1000=float(np.mean([r['delta_coverage_1000'] for r in selected]))))
    crossclass = {}
    for row in rows:
        if row['model']!='iqp-mse':
            key=tuple(row[k] for k in ('n','beta','seed','sigma','k'))
            crossclass.setdefault(key,[]).append(row)
    wins={}
    for metric in ('kl','conditional_kl'):
        counts={model:0 for model in MODELS if model!='iqp-mse'}
        complete=0
        for values in crossclass.values():
            if {v['model'] for v in values}!=set(counts) or any(metric not in v for v in values):
                continue
            complete+=1
            best=min(float(v[metric]) for v in values)
            for v in values:
                counts[v['model']]+=int(float(v[metric])==best)
        wins[metric]=dict(complete_five_class_instances=complete,wins=counts)
    summary = dict(rows=len(rows),crossclass=wins,uncertainty='Student t, 95%; beta repetitions clustered within seed; descriptive fixed seed cohort',
                   models=model_stats,paired_contrasts=contrasts,paired_instances=differences)
    write_json(out/'summary.json', summary)
    return summary


def render_runs(out, *, plots=False):
    """Write numerical summaries, with figures only when explicitly requested."""
    summary = summarize(out)
    if plots:
        from .run_figures import render
        render(out)
    return summary


def check_environment(path, runtime):
    path=Path(path)
    if path.exists() and json.loads(path.read_text()) != runtime:
        raise ValueError('Resume environment differs (Python, platform, dependencies, threads or workers); use a new --out')
    if not path.exists():
        write_json(path,runtime)


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    presets = {
        'smoke': dict(n=[6],betas=[.9],seeds=[111],m=50,steps=10,sigma=[1.],k=[32],models=MODELS),
        'paired': dict(n=[12],betas=[.9],seeds=list(range(111,121)),m=200,steps=600,sigma=[1.],k=[512],models=MODELS[:2]),
        'paper': dict(n=[12],betas=[i/10 for i in range(1,21)],seeds=list(range(111,121)),m=200,steps=600,sigma=[1.],k=[512],models=MODELS),
        'ablation': dict(n=[12],betas=[.9],seeds=list(range(111,121)),m=200,steps=600,sigma=[.5,1.,2.,3.],k=[128,256,512],models=MODELS[:2]),
        'legacy': dict(n=[12],betas=[.9],seeds=list(range(42,52)),m=200,steps=600,sigma=[.5,1.,2.,3.],k=[128,256,512],models=MODELS[:2]),
        'sizes': dict(n=list(range(10,21)),betas=[.9],seeds=list(range(111,121)),m=200,steps=600,sigma=[1.],k=[512],models=MODELS),
    }
    config = presets[args.preset]
    for key in ('n','betas','seeds','m','steps','sigma','k','models'):
        value = getattr(args,key)
        if value is not None:
            config[key] = value
    if not all(4<=n<=20 for n in config['n']) or config['steps']<1 or config['m']<1:
        raise ValueError('Require 4 <= n <= 20, positive steps and sample count')
    if args.workers<1 or args.threads<1:
        raise ValueError('workers and threads must be positive')
    config.update(protocol=args.protocol,preset=args.preset,threads=args.threads)
    config['implementation_sha256'] = protocol_fingerprint()
    config_path = out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError('Output already contains a different protocol; choose a new --out')
    write_json(config_path,config)
    runtime = dict(python=platform.python_version(),platform=platform.platform(),
                   versions={p:importlib.metadata.version(p) for p in ('iqp-repro','numpy','scipy','torch','matplotlib')},
                   threads=args.threads,workers=args.workers)
    # Never relabel cached checkpoints with a new environment.
    check_environment(out/'environment.json', runtime)
    tasks = []
    for n,beta,seed,sigma,k in itertools.product(config['n'],config['betas'],config['seeds'],config['sigma'],config['k']):
        tasks.append(dict(config,n=n,beta=beta,seed=seed,sigma=sigma,k=k,out=str(out.resolve())))
    print(f'{len(tasks)} instances, {len(config["models"])} models, {config["steps"]} updates; protocol={args.protocol}',flush=True)
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(rows,cached) in enumerate(pool.map(train_instance,tasks),1):
            row = rows[0]
            print(f'{i}/{len(tasks)} n={row["n"]} beta={row["beta"]} seed={row["seed"]} sigma={row["sigma"]} K={row["k"]}'
                  f' {"cached" if cached else "trained"}; elapsed {time.monotonic()-start:.1f}s',flush=True)
    render_runs(out, plots=args.plots)
    print(f'Complete: {out}/summary.json',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('run',help='Train and checkpoint matched instances (resumable)')
    p.add_argument('--preset',choices=['smoke','paired','paper','ablation','legacy','sizes'],default='smoke')
    p.add_argument('--protocol',choices=['matched','source'],default='matched',help='matched: shared IQP initialization and support MSE; source: original separate initializations and cube MSE')
    p.add_argument('--out',required=True)
    plotting = p.add_mutually_exclusive_group()
    plotting.add_argument('--plots',action='store_true',help='Also create optional diagnostic figures')
    plotting.add_argument('--no-plots',dest='plots',action='store_false',help='Write numerical results only (default)')
    p.set_defaults(plots=False)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--threads',type=int,default=4,help='PyTorch CPU threads: 4 matches archived Transformer arithmetic; changes can alter float32 trajectories')
    for key,typ in (('n',int),('betas',float),('seeds',int),('sigma',float),('k',int)):
        p.add_argument('--'+key,nargs='+',type=typ)
    p.add_argument('--m',type=int)
    p.add_argument('--steps',type=int)
    p.add_argument('--models',nargs='+',choices=MODELS)
    p = sub.add_parser('figures',help='Rerender archived reference data; does not train')
    p.add_argument('--out',default='runs/reference-figures')
    p = sub.add_parser('summarize',help='Recalculate numerical summaries from trained checkpoints')
    p.add_argument('--out',required=True)
    plotting = p.add_mutually_exclusive_group()
    plotting.add_argument('--plots',action='store_true',help='Also create optional diagnostic figures')
    plotting.add_argument('--no-plots',dest='plots',action='store_false',help='Write numerical results only (default)')
    p.set_defaults(plots=False)
    args = parser.parse_args()
    if args.command=='run':
        run(args)
    elif args.command=='figures':
        from .figures import render_reference
        render_reference(Path(args.out))
    else:
        render_runs(args.out, plots=args.plots)


if __name__=='__main__':
    main()
