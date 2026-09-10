"""Small, source-backed paper figures. Cached rerendering never trains a model."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

REFERENCE = Path(__file__).parent / 'data' / 'reference'
T95_9 = float(student_t.ppf(.975, 9))
MODELS = {
    'iqp_parity_mse': ('IQP parity', '#d64545'),
    'classical_nnn_fields_parity': ('Sparse Ising', '#4c78a8'),
    'classical_dense_fields_xent': ('Dense Ising', '#926953'),
    'classical_transformer_mle': ('AR Transformer', '#17a3ad'),
    'classical_maxent_parity': ('MaxEnt parity', '#9b70bd'),
}


def _rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def _mean_ci(a, critical=1.96):
    a = np.asarray(a, dtype=float)
    return {'mean': float(a.mean()), 'ci95_halfwidth': float(critical * a.std(ddof=1) / np.sqrt(a.size)),
            'median': float(np.median(a)), 'n': int(a.size)}


def _fixed_summary(grid, mse):
    parity = grid[:, 1, 2]  # predeclared sigma=1, K=512
    return {'reference_parity': _mean_ci(parity, T95_9),
            'mse': _mean_ci(mse, T95_9),
            'paired_delta_parity_minus_mse': _mean_ci(parity - mse, T95_9),
            'reference_wins': int(np.sum(parity < mse)),
            'oracle_min_over_12_bands': _mean_ci(grid.min(axis=(1, 2)), T95_9),
            'interval': 'Student t, df=9; oracle is selected using exact target KL'}


def _recovery(q, elite, budgets):
    return (-np.expm1(np.asarray(budgets)[:, None] * np.log1p(-np.minimum(q[elite], 1-1e-15)))).mean(axis=1)


def validate_reference(data_dir=None):
    """Validate hashes, paired row sets, and hardware curves against raw counts."""
    data = Path(data_dir) if data_dir else REFERENCE
    sources = json.loads((data / 'SOURCES.json').read_text())
    for name, info in sources['files'].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Reference checksum mismatch: {name}')
    beta, sizes = _rows(data / 'beta_metrics.csv'), _rows(data / 'size_metrics.csv')
    for rows, field, coordinates in [(beta, 'beta', np.arange(1, 21)/10), (sizes, 'n', np.arange(10, 21))]:
        expected = {(float(v), s, m) for v in coordinates for s in range(111, 121) for m in MODELS}
        observed = {(round(float(r[field]), 8), int(r['seed']), r['model_key']) for r in rows}
        if observed != expected or len(rows) != len(expected):
            raise ValueError(f'Incomplete or duplicated reference rows: {field}')
    old, fixed = np.load(data / 'legacy_fixed.npz'), np.load(data / 'fixed.npz')
    report = {'status': 'cached historical measurements; no new training or physical device execution',
              'legacy_seeds42_51': _fixed_summary(old['kl_grid_by_seed'], old['mse_kl_by_seed']),
              'fixed_seeds111_120': _fixed_summary(fixed['parity_kl_grid'], fixed['mse_kl']),
              'paper_printed_parity_mean': .402, 'paper_printed_mean_supported_by_raw_arrays': False}
    report['main_table'] = {}
    for key in MODELS:
        rows = [r for r in beta if r['model_key'] == key]
        report['main_table'][key] = {'KL': _mean_ci([float(r['KL_pstar_to_q']) for r in rows]),
                                     'C1000': _mean_ci([float(r['quality_coverage_Q1000']) for r in rows])}
    for key in MODELS:
        lookup = {(r['beta'], r['seed']): float(r['KL_pstar_to_q']) for r in beta if r['model_key'] == key}
        report['main_table'][key]['KL_wins'] = sum(
            value == min(float(r['KL_pstar_to_q']) for r in beta if (r['beta'], r['seed']) == instance)
            for instance, value in lookup.items())
    report['size_median_KL'] = {str(n): {key: float(np.median([float(r['KL_pstar_to_q'])
        for r in sizes if int(r['n']) == n and r['model_key'] == key])) for key in MODELS}
        for n in [10, 15, 20]}
    report['main_table_interval'] = 'Normal 95% interval over 200 instance metrics, as historical protocol; beta grid is fixed and data seeds repeat across beta.'
    raw, curves = np.load(data / 'hardware.npz'), np.load(data / 'hardware_curves.npz')
    if raw['counts_cube'].shape != (2, 10, 4096) or not np.all(raw['counts_cube'].sum(axis=-1) == 10000):
        raise ValueError('Expected 20 historical 10000-shot observations')
    errors, hardware_seed_curves = [], {}
    for m, prefix in enumerate(['parity', 'mse']):
        for mode in ['hw', 'sim']:
            distributions = raw['counts_cube'][m] / 10000 if mode == 'hw' else raw['qideal_cube'][m]
            if not np.allclose(distributions.sum(axis=1), 1, atol=1e-10):
                raise ValueError('Unnormalized hardware reference distribution')
            values = np.array([_recovery(q, mask.astype(bool), curves['q_grid'])
                               for q, mask in zip(distributions, raw['elite_unseen_masks'])])
            if mode == 'hw':
                hardware_seed_curves[prefix] = values
            errors += [float(np.max(np.abs(values.mean(axis=0) - curves[f'{prefix}_{mode}_mean']))),
                       float(np.max(np.abs(values.std(axis=0, ddof=1) - curves[f'{prefix}_{mode}_std'])))]
    report['hardware_curve_max_absolute_reconstruction_error'] = max(errors)
    if max(errors) > 1e-10:
        raise ValueError(f'Hardware curve reconstruction differs: {max(errors):.3g}')
    report['hardware'] = {'jobs': 20, 'shots_per_job': 10000, 'backend': 'ibm_marrakesh',
                          'date': '2026-04-08', 'parity_selection': 'best true-KL band separately per seed',
                          'curve_estimator': 'expected occupancy under observed frequencies; not fresh physical samples'}
    paired = {}
    for budget in [1000, 2000]:
        index = int(np.flatnonzero(curves['q_grid'] == budget)[0])
        delta = hardware_seed_curves['parity'][:, index] - hardware_seed_curves['mse'][:, index]
        stats = _mean_ci(delta, T95_9)
        stats.update(ci95_low=stats['mean'] - stats['ci95_halfwidth'],
                     ci95_high=stats['mean'] + stats['ci95_halfwidth'],
                     parity_wins=int(np.sum(delta > 0)),
                     direction='parity minus MSE recovery; higher is better')
        paired[str(budget)] = stats
    report['hardware']['paired_recovery'] = paired
    report['hardware']['paired_interval'] = 'Student t, df=9, across 10 matched seed differences'
    low = [values.mean(axis=0) - values.std(axis=0, ddof=1)
           for values in hardware_seed_curves.values()]
    high = [values.mean(axis=0) + values.std(axis=0, ddof=1)
            for values in hardware_seed_curves.values()]
    positive = curves['q_grid'] > 0
    overlap = np.maximum(*low) <= np.minimum(*high)
    report['hardware']['mean_plus_minus_sd_overlap'] = {
        'positive_saved_budget_points': int(np.sum(positive)),
        'overlapping_points': int(np.sum(overlap[positive])),
        'overlap_fraction': float(np.mean(overlap[positive])),
        'nonoverlapping_budgets': curves['q_grid'][positive & ~overlap].tolist(),
        'definition': 'Overlap of separate mean +/- sample-SD intervals on the saved positive Q grid; not a paired significance test',
    }
    scores = np.array([float(r['score_level']) for r in _rows(data / 'score_marginal_seed42.csv')])
    if not np.array_equal(scores - scores.min(), np.arange(11)):
        raise ValueError('Legacy score display must be the integer levels 0 through 10')
    report['legacy_score_display'] = (scores - scores.min()).astype(int).tolist()
    report['historical_elite_rule'] = 'Top floor(0.1 * |S|) states, truncating score ties; differs from the tie-inclusive Eq.9 quantile rule used in fresh runs'
    return report


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180, 'pdf.fonttype': 42})
    return plt


def _save(plt, fig, out, name):
    for suffix in ['pdf', 'png']:
        fig.savefig(out / f'{name}.{suffix}', bbox_inches='tight')
    plt.close(fig)


def _metric_panel(ax, rows, xfield, metric):
    for key, (label, color) in MODELS.items():
        selected = [r for r in rows if r['model_key'] == key]
        xs = sorted({float(r[xfield]) for r in selected})
        values = [np.array([float(r[metric]) for r in selected if float(r[xfield]) == x]) for x in xs]
        center = [np.median(v) if metric == 'KL_pstar_to_q' else np.mean(v) for v in values]
        low, high = [np.quantile(v, .25) for v in values], [np.quantile(v, .75) for v in values]
        ax.plot(xs, center, '.-', color=color, label=label, lw=1.6)
        ax.fill_between(xs, low, high, color=color, alpha=.12)
    ax.set_xlabel('System size n' if xfield == 'n' else r'Sharpness $\beta$')
    ax.grid(alpha=.15)


def render_reference(outdir, data_dir=None):
    """Render Figures 1-7 and a separate current-seed fixed-band comparison."""
    data, out = Path(data_dir) if data_dir else REFERENCE, Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    report = validate_reference(data)
    plt = _pyplot()
    # Figure 1: exact conceptual parity example.
    fig, ax = plt.subplots(figsize=(6.6, 2.1), layout='constrained')
    ax.axis('off')
    table = ax.table(cellText=[['1100 (seen)', '-1', '-1', '+1'], ['1001 (unseen)', '-1', '-1', '+1']],
                     colLabels=['State', 'Mask 1010', 'Mask 0111', 'Mask 1111'], loc='center', cellLoc='center')
    table.scale(1, 1.7)
    ax.set_title('Figure 1 · Shared parity fingerprint', loc='left', fontweight='bold')
    ax.text(.5, .05, 'Observed samples constrain shared moments; this schematic is not a training result.',
            ha='center', transform=ax.transAxes, fontsize=8)
    _save(plt, fig, out, 'figure1_parity_example')
    # Figure 2: architecture, with the exact gate convention visible.
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(7.2, 2.5), layout='constrained')
    ax.set(xlim=(0, 10), ylim=(-.8, 3.8)); ax.axis('off')
    for y, label in zip([3, 2, 1, 0], ['0', '1', '2', '11']):
        ax.plot([1, 9], [y, y], color='.6', zorder=0)
        ax.text(.7, y, f'q{label}: |0⟩', va='center', ha='right')
        for x in [1.5, 7.5]:
            ax.add_patch(Rectangle((x-.25, y-.25), .5, .5, facecolor='white', edgecolor='.3'))
            ax.text(x, y, 'H', ha='center', va='center')
        ax.text(9.2, y, 'measure', va='center', fontsize=8)
    ax.add_patch(Rectangle((3, -.4), 3, 3.8, facecolor='#e9f0f6', edgecolor='#4c78a8'))
    ax.text(4.5, 1.5, 'Commuting ZZ block\nNN + NNN ring\n24 angles; n = 12', ha='center', va='center')
    ax.text(5, -.75, r'$U_{ZZ}(\theta)=\prod_{(i,j)\in E}e^{-i\theta_{ij}Z_iZ_j/2}$', ha='center')
    ax.set_title('Figure 2 · One-layer IQP Born machine (intermediate wires omitted)', loc='left', fontweight='bold')
    _save(plt, fig, out, 'figure2_iqp_circuit')
    # Figure 3: deterministic target enumeration (source +1 score shift cancels).
    scores = []
    for x in range(4096):
        if x.bit_count() % 2 == 0:
            ones = [i for i, b in enumerate(f'{x:012b}') if b == '1']
            scores.append(max((b-a-1 for a, b in zip(ones, ones[1:])), default=0))
    scores = np.array(scores)
    fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')
    for b in np.arange(1, 21)/10:
        p = np.exp(b * (scores-scores.max())); p /= p.sum()
        highlight = round(b, 1) in [.6, .8, 1., 1.2, 1.4]
        ax.plot(np.arange(11), np.bincount(scores, weights=p, minlength=11),
                color=None if highlight else '.82', lw=1.7 if highlight else .7,
                label=f'β = {b:g}' if highlight else None)
    ax.set(xlabel='Longest internal zero gap', ylabel='Target probability mass', title='Figure 3 · n = 12 target family')
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.15)
    _save(plt, fig, out, 'figure3_target')
    # Figure 4: each panel derives from the explicitly legacy seed42..51 data.
    old, rows = np.load(data / 'legacy_fixed.npz'), _rows(data / 'score_marginal_seed42.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), layout='constrained')
    x = np.arange(len(rows))
    for i, (col, label, color) in enumerate([('target_mass', 'Target', '.3'), ('iqp_parity_mass', 'Oracle parity', '#d64545'), ('iqp_mse_mass', 'MSE', '#4c78a8')]):
        axes[0].bar(x+(i-1)*.27, [float(r[col]) for r in rows], width=.27, label=label, color=color)
    axes[0].set(xlabel='Score', ylabel='Probability mass', title='(a) Representative seed 42'); axes[0].legend(fontsize=7)
    grid = old['panel_ab_grid']; im = axes[1].imshow(grid, cmap='RdBu', vmin=.3, vmax=.7, aspect='auto')
    for i in range(4):
        for j in range(3):
            axes[1].text(j, i, f'{grid[i,j]:.3f}', ha='center', va='center', fontsize=8)
    axes[1].set(xticks=range(3), xticklabels=[128,256,512], yticks=range(4), yticklabels=[.5,1,2,3], xlabel='K', ylabel='σ', title='(b) Exact KL, seed 42')
    values = [old['kl_grid_by_seed'].min(axis=(1,2)), old['kl_grid_by_seed'][:,1,2], old['mse_kl_by_seed']]
    for i, v in enumerate(values):
        s=_mean_ci(v,T95_9)
        axes[2].scatter(v, np.full(10,i), color='.6', s=12, alpha=.65)
        axes[2].errorbar(s['mean'], i, xerr=s['ci95_halfwidth'], color=['#d64545','#a87070','#4c78a8'][i], fmt='o', capsize=3)
    axes[2].set(yticks=range(3), yticklabels=['Oracle of 12', 'Fixed σ1, K512', 'MSE'], xlabel='Forward KL (mean ± 95% t CI)', title='(c) Raw legacy seeds 42-51')
    fig.suptitle('Figure 4 · Historical measurements; oracle selection uses target KL; published hardcoded mean excluded', fontsize=10)
    _save(plt, fig, out, 'figure4_legacy_fixed')
    # Separate contemporary-seed reference-band paired view; do not combine eras.
    fixed=np.load(data/'fixed.npz'); parity=fixed['parity_kl_grid'][:,1,2]; mse=fixed['mse_kl']; delta=parity-mse
    fig, axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    for p,m in zip(parity,mse): axes[0].plot([0,1],[p,m],'.-',color='.6',alpha=.7)
    axes[0].set(xticks=[0,1],xticklabels=['Parity σ1,K512','MSE'],ylabel='Forward KL',title='Seeds 111-120; historical initializations differ')
    s=_mean_ci(delta,T95_9); axes[1].scatter(delta,np.arange(10),color='.5',s=20)
    axes[1].axvline(0,color='.7',lw=1); axes[1].errorbar(s['mean'],-1.5,xerr=s['ci95_halfwidth'],fmt='o',capsize=4,color='#d64545')
    axes[1].set(xlabel='Paired KL difference: parity - MSE',yticks=[-1.5],yticklabels=['Mean, 95% t CI'],title=f'Reference wins {int((delta<0).sum())}/10; interval includes zero')
    _save(plt,fig,out,'fixed_reference_comparison')
    # Figure 5: canonical measurements only, never presentation overrides.
    beta=_rows(data/'beta_metrics.csv'); fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for ax, metric, label in zip(axes,['KL_pstar_to_q','quality_coverage_Q1000'],['Forward KL · median and IQR','C(1000) · mean and IQR']):
        _metric_panel(ax,beta,'beta',metric); ax.set_ylabel(label)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Figure 5 · Historical reference-band sweep, 200 paired instances\nCoverage: top floor(10% of valid states), truncating score ties; differs from Eq.9',fontsize=10)
    _save(plt,fig,out,'figure5_main_sweep')
    # Figure 6: stored mechanistic diagnostic and verified historical device data.
    r,h=np.load(data/'recovery_seed118.npz'),np.load(data/'hardware_curves.npz'); q=r['Q']; idx=5
    fig,axes=plt.subplots(1,4,figsize=(13,3.5),layout='constrained')
    for ax in axes[:3]:
        ax.plot(q,r['target_curve'],color='.2',label='Target'); ax.plot(q,r['uniform_curve'],':',color='.65',label='Uniform cube')
    axes[0].plot(q,r['parity_curves'][idx],color='#d64545',label='Parity σ1,K512')
    axes[0].plot(q,r['spectral_curves'][idx],'-.',color='.4',label='Legacy spectral')
    for v in r['parity_curves']: axes[1].plot(q,v,color='#e4b7b7',lw=.8)
    axes[1].plot(q,r['parity_curves'][idx],color='#d64545',label='Parity σ1,K512'); axes[1].plot(q,r['iqp_mse_curve'],color='#4c78a8',label='MSE')
    for v in r['spectral_curves']: axes[2].plot(q,v,color='.75',lw=.8)
    axes[2].plot(q,r['spectral_curves'][idx],color='.4',label='Legacy spectral σ1,K512')
    axes[3].plot(h['q_grid'],h['target_mean'],color='.2',label='Target')
    axes[3].plot(h['q_grid'],h['uniform_mean'],':',color='.65',label='Historical uniform cube')
    for prefix,color in [('parity','#d64545'),('mse','#4c78a8')]:
        for mode,style in [('hw','-'),('sim','--')]:
            y,sd=h[f'{prefix}_{mode}_mean'],h[f'{prefix}_{mode}_std']
            axes[3].plot(h['q_grid'],y,style,color=color,label=f'{prefix} {mode}')
            if mode=='hw': axes[3].fill_between(h['q_grid'],y-sd,y+sd,color=color,alpha=.12)
    for ax,title in zip(axes,['(a) Fixed band, seed118','(b) Parity family, seed118','(c) Legacy spectral family','(d) Historical IBM, mean ± SD']):
        ax.set(xlabel='Samples Q',ylabel='Expected recovery R(Q)',title=title,xlim=(0,2000),ylim=(0,.95));ax.legend(fontsize=6,frameon=False,loc='upper left');ax.grid(alpha=.15)
    fig.suptitle('Figure 6 · Historical recovery; elite sets truncate score ties at floor(10% of valid states)\nSpectral proxy uses full-cube weighting; hardware parity uses oracle band selection',fontsize=10)
    _save(plt,fig,out,'figure6_recovery')
    # Figure 7: no implicit recomputation of the expensive n=20 runs.
    fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
    _metric_panel(ax,_rows(data/'size_metrics.csv'),'n','KL_pstar_to_q')
    ax.set(ylabel='Forward KL · median and IQR',title='Figure 7 · Historical size sweep at β = 0.9; 10 seeds per n');ax.legend(frameon=False,fontsize=8)
    _save(plt,fig,out,'figure7_size_sweep')
    (out/'reference-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
