"""Figures from fresh checkpoints; protocols and parity settings stay separate."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .core import coverage

COLORS = {"iqp-parity": "#c73737", "iqp-mse": "#3478b5", "maxent": "#7955a0",
          "ising-parity": "#bf8524", "ising-nll": "#487f57", "transformer": "#393939"}


def _save(fig, out, name):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / (name + ".png"), dpi=180)
    fig.savefig(out / (name + ".pdf"))
    plt.close(fig)


def _interval(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        return dict(mean=None,ci95_halfwidth=None,seeds=len(values),undefined_seed_count=int((~np.isfinite(values)).sum()))
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
    return dict(mean=mean, ci95_halfwidth=half, seeds=len(values))


def _number(value):
    return float(value) if value is not None else float("nan")


def _rows(paths):
    result = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            result.extend(dict(row, checkpoint=path) for row in json.loads(str(z["rows"])))
    return result


def _sweep(rows, out, xkey, fixed, manifest):
    for key in sorted({tuple(r[f] for f in fixed) for r in rows}):
        selected = [r for r in rows if tuple(r[f] for f in fixed)==key]
        xs = sorted({r[xkey] for r in selected})
        if len(xs)<2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        undefined = {}
        for model in COLORS:
            for ax, metric in zip(axes, ("kl", "coverage_1000")):
                values = [np.array([_number(r[metric]) for r in selected if r["model"]==model and r[xkey]==x]) for x in xs]
                if not all(len(v) for v in values):
                    continue
                invalid = [x for x,v in zip(xs,values) if not np.all(np.isfinite(v))]
                if invalid:
                    undefined[f"{model}/{metric}"] = invalid
                    ax.text(.02,.98-.07*len(undefined),f"{model}: {len(invalid)} nonfinite groups",transform=ax.transAxes,va="top",fontsize=7)
                quantile = lambda a: [np.quantile(v,a) if np.all(np.isfinite(v)) else np.nan for v in values]
                ax.plot(xs,quantile(.5),"o-",ms=3,label=model,color=COLORS[model])
                ax.fill_between(xs,quantile(.25),quantile(.75),alpha=.10,color=COLORS[model])
        axes[0].set_ylabel("Forward KL · median and IQR")
        axes[1].set_ylabel("Discoveries per draw · Q=1000")
        for ax in axes:
            ax.set_xlabel("Sharpness β" if xkey=="beta" else "Qubits n")
        axes[1].legend(fontsize=7)
        description = ", ".join(f"{k}={v:g}" if isinstance(v,(int,float)) else f"{k}={v}" for k,v in zip(fixed,key))
        fig.suptitle(description, fontsize=10)
        name = "fresh_" + xkey + "_" + "_".join(str(v) for v in key)
        _save(fig,out,name)
        manifest[name] = dict(group=dict(zip(fixed,key)),statistic="median and IQR; no pooling across fixed settings",nonfinite_groups=undefined,
                             seed_counts={str(x):len({r['seed'] for r in selected if r[xkey]==x}) for x in xs})


def _diagnostic(rows, out, manifest):
    # Fixed, outcome-independent display slice: n12/beta.9 if present, else first.
    ns = sorted({r["n"] for r in rows}); n = 12 if 12 in ns else ns[0]
    betas = sorted({r["beta"] for r in rows if r["n"]==n}); beta = .9 if .9 in betas else betas[0]
    selected = [r for r in rows if r["n"]==n and r["beta"]==beta]
    settings = sorted({(r["sigma"],r["k"]) for r in selected})
    ref = (1.,512) if (1.,512) in settings else settings[0]
    seeds = sorted({r["seed"] for r in selected}); representative = 118 if 118 in seeds else seeds[0]
    ref_rows = [r for r in selected if (r["sigma"],r["k"])==ref]
    pairs = {seed:{r["model"]:r for r in ref_rows if r["seed"]==seed} for seed in seeds}
    paired = {s:p for s,p in pairs.items() if "iqp-parity" in p and "iqp-mse" in p}
    if not paired or representative not in paired:
        return
    delta = [_number(p["iqp-parity"]["kl"])-_number(p["iqp-mse"]["kl"]) for p in paired.values()]
    fig, axes = plt.subplots(1,3,figsize=(11.6,3.4),layout="constrained")
    with np.load(paired[representative]["iqp-parity"]["checkpoint"],allow_pickle=False) as z:
        valid, scores = z["support"],z["scores"]
        for name,label,color in [("p","target","#222222"),("iqp-parity_q","IQP parity",COLORS["iqp-parity"]),("iqp-mse_q","IQP MSE",COLORS["iqp-mse"])]:
            axes[0].plot(np.bincount(scores[valid],weights=z[name][valid]),"o-",ms=3,label=label,color=color)
    axes[0].set(xlabel="Bracketed zero-run score ℓ",ylabel="Probability mass",title=f"Fixed reference · seed {representative}")
    axes[0].legend(fontsize=7)
    sigmas, ks = sorted({s[0] for s in settings}),sorted({s[1] for s in settings})
    grid = np.full((len(sigmas),len(ks)),np.nan)
    for r in selected:
        if r["seed"]==representative and r["model"]=="iqp-parity":
            grid[sigmas.index(r["sigma"]),ks.index(r["k"])] = _number(r["kl"])
    heatmap = axes[1].imshow(grid,cmap="YlOrRd",aspect="auto")
    for i,j in np.ndindex(grid.shape):
        if np.isfinite(grid[i,j]):
            color = "white" if heatmap.norm(grid[i,j]) > .75 else "#222222"
            axes[1].text(j,i,f"{grid[i,j]:.3f}",ha="center",va="center",fontsize=8,color=color)
        else:
            axes[1].text(j,i,"∞" if np.isinf(grid[i,j]) else "undefined",ha="center",va="center",fontsize=8)
    axes[1].set(xticks=range(len(ks)),xticklabels=ks,yticks=range(len(sigmas)),yticklabels=sigmas,xlabel="K",ylabel="σ",title="IQP parity · forward KL")
    stats = _interval(delta)
    if stats["mean"] is None:
        axes[2].text(.5,.5,f"Paired contrast undefined\n{stats['undefined_seed_count']} nonfinite seed(s)",ha="center",transform=axes[2].transAxes)
    else:
        axes[2].scatter(delta,np.zeros(len(delta)),s=20,color="#999999",label="paired seeds")
        axes[2].errorbar(stats["mean"],1,xerr=stats["ci95_halfwidth"],fmt="o",capsize=4,color="#222222",label="mean and 95% t CI")
    axes[2].axvline(0,color="#999999",lw=.8)
    axes[2].xaxis.set_major_locator(MaxNLocator(4))
    axes[2].set(yticks=[],ylim=(-.5,1.6),xlabel="KL(parity) − KL(MSE)\nNegative favors parity",title=f"Fixed σ={ref[0]:g}, K={ref[1]}")
    if stats["mean"] is not None:
        axes[2].legend(fontsize=7,loc="upper left")
    protocol = selected[0]["protocol"]
    fig.suptitle(f"Fresh diagnostics · {protocol} · n={n}, β={beta:g} · no best-setting selection",fontsize=11)
    _save(fig,out,"fresh_diagnostics")
    manifest["fresh_diagnostics"] = dict(n=n,beta=beta,reference=ref,representative_seed=representative,paired_seed_ids=list(paired),paired_delta_kl=stats)
    _recovery(selected,settings,ref,out,manifest)
    if len(settings)>1:
        _oracle(selected,settings,out,manifest)


def _oracle(rows,settings,out,manifest):
    choices = []
    for seed in sorted({r["seed"] for r in rows}):
        candidates = [r for r in rows if r["seed"]==seed and r["model"]=="iqp-parity"]
        if {(r['sigma'],r['k']) for r in candidates} != set(settings):
            continue
        if any(np.isnan(_number(r['kl'])) for r in candidates):
            manifest['fresh_oracle_diagnostic'] = dict(status='undefined: missing parity KL',seed=seed)
            return
        best = min(candidates,key=lambda r:_number(r['kl']))
        mse = next((r for r in rows if r['seed']==seed and r['model']=='iqp-mse' and (r['sigma'],r['k'])==(best['sigma'],best['k'])),None)
        if mse is not None:
            choices.append(dict(seed=seed,sigma=best['sigma'],k=best['k'],parity_kl=_number(best['kl']),mse_kl=_number(mse['kl'])))
    if not choices:
        return
    delta = [r['parity_kl']-r['mse_kl'] for r in choices]; stats = _interval(delta)
    if stats['mean'] is not None and stats['ci95_halfwidth'] is not None:
        stats.update(ci95_low=stats['mean']-stats['ci95_halfwidth'],ci95_high=stats['mean']+stats['ci95_halfwidth'])
    fig,ax = plt.subplots(figsize=(8.5,3.3),layout='constrained')
    if stats['mean'] is not None:
        ax.scatter(delta,np.zeros(len(delta)),color='#999999',label='seed pairs')
        ax.errorbar(stats['mean'],1,xerr=stats['ci95_halfwidth'],fmt='o',capsize=4,color='#c73737',label='mean and 95% t CI')
        ax.legend(fontsize=8)
    else:
        ax.text(.5,.5,'Nonfinite paired contrast; see JSON',ha='center',transform=ax.transAxes)
    ax.axvline(0,color='#999999',lw=.8)
    ax.set(yticks=[],ylim=(-.5,1.7),xlabel='KL(oracle-selected parity) - KL(MSE at the same selected setting)\nNegative favors parity')
    fig.suptitle(f"Exploratory oracle · {rows[0]['protocol']} · n={rows[0]['n']}, β={rows[0]['beta']:g}\nMinimum true KL over {len(settings)} parity runs per seed: {len(settings)*rows[0]['steps']} candidate updates\nSelected-setting MSE: {rows[0]['steps']} updates; unequal selection budgets",fontsize=10)
    _save(fig,out,'fresh_oracle_diagnostic')
    manifest['fresh_oracle_diagnostic'] = dict(selection='true-KL minimum per seed; exploratory oracle, not a fixed-reference comparison',
        protocol=rows[0]['protocol'],n=rows[0]['n'],beta=rows[0]['beta'],
        candidates_per_seed=len(settings),parity_candidate_updates_per_seed=len(settings)*rows[0]['steps'],
        selected_mse_updates_per_seed=rows[0]['steps'],paired_delta_kl=stats,
        choices=[{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()} for row in choices])


def _recovery(rows,settings,ref,out,manifest):
    empty = sorted({r["seed"] for r in rows if r.get("elite_size")==0})
    if empty:
        manifest["fresh_recovery"] = dict(status="undefined: empty unseen elite",seed_ids=empty)
        return
    paths = {(r["sigma"],r["k"],r["seed"]):r["checkpoint"] for r in rows}
    common = sorted(set.intersection(*[{s for a,b,s in paths if (a,b)==setting} for setting in settings]))
    if not common:
        return
    Q = np.linspace(0,2000,81).astype(int)
    curves = {}
    for sigma,k in settings:
        for seed in common:
            with np.load(paths[(sigma,k,seed)],allow_pickle=False) as z:
                valid = z["support"]
                models = {name:z[name] for name in ("iqp-parity_q","iqp-mse_q","spectral_cube","spectral_support","spectral_unique_support") if name in z}
                models.update(target=z['p'],uniform_cube=np.ones(len(valid))/len(valid),uniform_support=valid/valid.sum())
                for name,q in models.items():
                    curves.setdefault((sigma,k,name),[]).append(coverage(q,z["elite"],Q)["recovery"])
    means = {key:np.mean(value,axis=0) for key,value in curves.items()}
    fig,axes = plt.subplots(1,3,figsize=(13,4.8),layout="constrained")
    labels = [("target","Target p*","#222222"),("iqp-parity_q","IQP parity","#c73737"),("iqp-mse_q","IQP MSE","#3478b5"),("spectral_cube","Spectral repeats · cube","#a080b5"),("spectral_support","Spectral repeats · S","#65477d"),("spectral_unique_support","Spectral unique · S","#af761f"),("uniform_cube","Uniform cube","#bbbbbb"),("uniform_support","Uniform S","#555555")]
    for name,label,color in labels:
        if (*ref,name) in means:
            axes[0].plot(Q,means[(*ref,name)],label=label,color=color,lw=1.5)
    axes[0].set_title(f"Fixed σ={ref[0]:g}, K={ref[1]}")
    for setting,color in zip(settings,plt.cm.viridis(np.linspace(.05,.9,len(settings)))):
        for ax,name in zip(axes[1:],("iqp-parity_q","spectral_support")):
            if (*setting,name) in means:
                ax.plot(Q,means[(*setting,name)],color=color,label=f"σ={setting[0]:g}, K={setting[1]}",lw=1.2)
    for ax in axes[1:]:
        for name,label,color in [labels[0],labels[-1],labels[-2],('iqp-mse_q','Fixed-reference IQP MSE','#3478b5')]:
            if (*ref,name) in means:
                ax.plot(Q,means[(*ref,name)],color=color,label=label,lw=1.8,ls='--' if name=='iqp-mse_q' else ':')
    axes[1].set_title("Every IQP parity setting")
    axes[2].set_title("Every spectral setting · repeats, S")
    for ax in axes:
        ax.set(xlabel="Independent draws Q",ylabel="Mean fraction of unseen elite recovered",xlim=(0,2000),ylim=(0,1))
        ax.legend(fontsize=6,loc="upper center",bbox_to_anchor=(.5,-.18),ncol=2)
    fig.suptitle(f"Fresh recovery · {rows[0]['protocol']} · n={rows[0]['n']}, β={rows[0]['beta']:g} · {len(common)} common seeds",fontsize=11)
    _save(fig,out,"fresh_recovery")
    keys = sorted(curves)
    np.savez_compressed(out/"fresh_recovery_data.npz",Q=Q,seeds=common,
                        labels=np.array([f"sigma={a:g},K={b},{name}" for a,b,name in keys]),
                        seed_curves=np.array([curves[key] for key in keys]))
    manifest["fresh_recovery"] = dict(seed_ids=common,reference=ref,statistic="arithmetic mean over common seeds; all settings shown",Q=Q.tolist())


def render(outdir):
    """Render fresh beta/size sweeps and a fixed diagnostic/recovery slice."""
    out = Path(outdir)
    paths = sorted(out.glob("n*_b*_s*_sigma*_k*.npz"))
    rows = _rows(paths)
    if not rows:
        raise ValueError("No fresh checkpoints found")
    if len({(r["protocol"],r["m"],r["steps"]) for r in rows})!=1:
        raise ValueError("Separate directories are required for different protocols/data sizes/budgets")
    plt.rcParams.update({"font.size":9,"axes.titlesize":10,"pdf.fonttype":42})
    manifest = {"checkpoints":[p.name for p in paths],"selection":"Main displays use fixed reference (1,512), n12, beta.9, representative seed118 if present, otherwise first available; only the separately labeled oracle diagnostic selects on true KL"}
    _sweep(rows,out,"beta",("protocol","n","sigma","k"),manifest)
    _sweep(rows,out,"n",("protocol","beta","sigma","k"),manifest)
    _diagnostic(rows,out,manifest)
    (out/"fresh_figures.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    return manifest
