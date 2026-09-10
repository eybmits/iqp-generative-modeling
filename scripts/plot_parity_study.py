#!/usr/bin/env python3
"""Portable plots for the frozen parity study; never trains or changes selection.

Chart contract: every paired observation per fixed contrast; absolute-KL scatter
with an identity reference, plus paired-difference dots and a t interval.
Blue and neutral marks distinguish data, means, and references without relying
on sign colors. Separate exploratory intervals retain their declared levels.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
BLUE, INK, GREY = "#27638E", "#272B30", "#969DA4"
OBJECTIVES = {"parity": "Sampled parity", "expected-parity": "Expected parity",
              "mse": "Raw MSE", "scaled-mse": "Scaled MSE"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, out, name):
    for suffix in ("png", "pdf", "svg"):
        metadata = {"Creator": "iqp-parity-study", "CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Date": None} if suffix == "svg" else {"Software": "iqp-parity-study"}
        fig.savefig(out / f"{name}.{suffix}", dpi=180, metadata=metadata)
    plt.close(fig)


def count(n, architecture):
    return n * (n - 1) // 2 if architecture == "dense" else 2 * n


def label(config, n):
    return f"{OBJECTIVES[config['objective']]} · {config['architecture']}, {count(n, config['architecture'])} angles · lr={config['lr']:g}"


def edges(n, architecture):
    return list(itertools.combinations(range(n), 2)) if architecture == "dense" else sorted({tuple(sorted((i, (i+d) % n))) for i in range(n) for d in (1, 2)})


def graph(ax, n, architecture, title, color):
    positions = np.column_stack((np.cos(np.pi/2 - np.arange(n)*2*np.pi/n), np.sin(np.pi/2 - np.arange(n)*2*np.pi/n)))
    for a, b in edges(n, architecture):
        ax.plot(*positions[[a, b]].T, color=color, alpha=.28 if architecture == "dense" else .6, lw=.8, zorder=1)
    ax.scatter(*positions.T, s=190, facecolor="white", edgecolor=color, lw=1.2, zorder=2)
    for i, (x, y) in enumerate(positions):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, color=INK)
    ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal")
    ax.axis("off"); ax.set_title(title, fontsize=11, pad=12)


def architecture(protocol, selected, out, version):
    n, config = protocol["n"], selected["parity"]
    fig = plt.figure(figsize=(12.5, 7.4), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.2, 1))
    ax = fig.add_subplot(grid[0, :]); ax.set(xlim=(0, 10), ylim=(-1.25, n+.6)); ax.axis("off")
    for i in range(n):
        y = n-1-i
        ax.plot([.65, 9.35], [y, y], color="#C7CBD0", lw=.8, zorder=0)
        ax.text(.52, y, f"q{i}: |0⟩", ha="right", va="center", fontsize=7)
        for x, text in ((1.3, "H"), (7.25, "H"), (8.9, "M")):
            ax.add_patch(Rectangle((x-.20, y-.32), .4, .64, facecolor="white", edgecolor=INK, lw=.7))
            ax.text(x, y, text, ha="center", va="center", fontsize=7)
    ax.add_patch(Rectangle((2.35, -.4), 3.9, n-.2, facecolor="#F2F5F7", edgecolor=BLUE, lw=1.2))
    ax.text(4.3, (n-1)/2, f"Commuting RZZ block\n{config['architecture']} graph · {count(n, config['architecture'])} trainable angles\nOne independent angle per edge", ha="center", va="center", fontsize=12, linespacing=1.6)
    ax.text(8.9, n+.05, "Z-basis measurement", ha="center", fontsize=8)
    ax.text(5, -1.04, r"$RZZ_{ij}(\theta)=e^{-i\theta Z_iZ_j/2}$   ·   One ZZ block; no local Z gates   ·   Even-parity output support", ha="center", fontsize=10)
    graph(fig.add_subplot(grid[1, 0]), n, "ring", f"Original paper context\nNN + NNN ring · {2*n} angles", GREY)
    graph(fig.add_subplot(grid[1, 1]), n, config["architecture"], f"{'Default model · study v2' if version == 2 else 'Study v1 selected model'}\n{config['architecture'].capitalize()} · {count(n, config['architecture'])} angles", BLUE)
    ax = fig.add_subplot(grid[1, 2]); ax.axis("off")
    detail = f"Exact expectation over {2**n-1:,} nonzero masks\nGaussian/Hamming-kernel MMD²" if config["objective"] == "expected-parity" else f"{protocol['k']} sampled nonzero parity masks\nRepeated masks retain their weight"
    note = "Fixed after v1; tested on a separate cohort.\nNo replication retuning or hardware run." if version == 2 else "Selection uses development validation NLL.\nSeparate from v2's sampled-parity model."
    ax.text(0, .92, "Fixed training rule" if version == 2 else "Selected training rule", fontsize=12, weight="bold", va="top")
    ax.text(0, .78, f"{OBJECTIVES[config['objective']]} · lr={config['lr']:g}\n{detail}\n\n{protocol['m']} training draws per model\n{protocol['steps']} Adam updates; {len(protocol['confirmation_seeds'])} new paired seeds\n\nSame-architecture MSE control:\n{label(selected['same_architecture_mse'], n)}\n\n{note}", fontsize=9, va="top", linespacing=1.45)
    fig.suptitle(f"IQP architecture and training · Study v{version} · New simulation results", fontsize=15)
    save(fig, out, "architecture")


def checked_pair(contrast, rows, expected):
    tables = []
    for key in (contrast["left"], contrast["right"]):
        chosen = [r for r in rows if r["key"] == key]
        table = {int(r["seed"]): r for r in chosen}
        if len(table) != len(chosen) or set(table) != set(expected):
            raise ValueError(f"Missing or duplicate confirmation rows for {key}")
        tables.append(table)
    a, b = tables; seeds = sorted(expected)
    for seed in seeds:
        if a[seed]["sample_sha256"] != b[seed]["sample_sha256"]:
            raise ValueError("Training samples are not paired")
        if a[seed]["architecture"] == b[seed]["architecture"] and a[seed]["initial_sha256"] != b[seed]["initial_sha256"]:
            raise ValueError("Initial angles are not paired")
    left, right = (np.array([float(table[s]["kl"]) for s in seeds]) for table in tables)
    delta = left-right
    if not np.all(np.isfinite(delta)):
        raise ValueError("Nonfinite KL must be resolved explicitly, not hidden in a plot")
    half = t.ppf((1+contrast["confidence"])/2, len(seeds)-1)*delta.std(ddof=1)/np.sqrt(len(seeds))
    np.testing.assert_allclose(contrast["ci"], [delta.mean()-half, delta.mean()+half], atol=1e-12, rtol=0)
    np.testing.assert_allclose(contrast["paired_differences"], delta, atol=1e-12, rtol=0)
    np.testing.assert_allclose([contrast["parity_mean_kl"], contrast["mse_mean_kl"], contrast["mean_difference"]], [left.mean(), right.mean(), delta.mean()], atol=1e-12, rtol=0)
    if contrast["seeds"] != seeds or contrast["n"] != len(seeds):
        raise ValueError("Summary has a different confirmation cohort")
    if contrast["wins"] != int((delta < 0).sum()):
        raise ValueError("Summary paired win count differs from the plotted observations")
    return dict(left=left, right=right, delta=delta, seeds=seeds, summary=contrast)


def delta_panel(ax, pair, y=0, title=None):
    d, summary = pair["delta"], pair["summary"]
    offsets = np.linspace(-.08, .08, len(d))
    ax.scatter(d, y+offsets, s=13, color=GREY, alpha=.6, linewidth=0)
    ci = np.asarray(summary["ci"]); mean = summary["mean_difference"]
    ax.errorbar(mean, y+.36, xerr=np.array([[mean-ci[0]], [ci[1]-mean]]), fmt="D", color=BLUE, ms=5, capsize=4, lw=1.7)
    ax.axvline(0, color=INK, lw=.8, ls="--")
    ax.set(yticks=[], ylim=(y-.22, y+.9), xlabel="Paired ΔKL = parity − MSE (nats)")
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    ax.text(.02, .97, f"Mean {mean:+.4f}; {100*summary['confidence']:g}% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]\nParity lower in {summary['wins']}/{summary['n']} pairs", transform=ax.transAxes, va="top", fontsize=9)


def main_comparisons(protocol, selected, summary, pairs, columns, out, version):
    fig, axes = plt.subplots(2, len(columns), figsize=(6*len(columns), 6.7), height_ratios=(3, 1.3), squeeze=False, layout="constrained")
    for j, (name, title, control) in enumerate(columns):
        pair, ax = pairs[name], axes[0, j]
        values = np.concatenate([pair["left"], pair["right"]])
        low, high = max(0, values.min()-.035), values.max()+.035
        ax.scatter(pair["right"], pair["left"], s=27, facecolor=BLUE, edgecolor="white", lw=.4, alpha=.8)
        ax.plot([low, high], [low, high], "--", color=INK, lw=.9)
        ax.set(xlim=(low, high), ylim=(low, high), aspect="equal", xlabel=f"MSE forward KL (nats)\n{label(control, protocol['n'])}", ylabel=f"Parity forward KL (nats)\n{label(selected['parity'], protocol['n'])}")
        ax.set_title(f"{title}\nMean KL: parity {pair['left'].mean():.4f}; MSE {pair['right'].mean():.4f}", loc="left", fontsize=10)
        delta_panel(axes[1, j], pair)
    gate = all(summary[name]["ci"][1] < 0 for name, _, _ in columns)
    if gate != summary["robust_benefit_gate"]:
        raise ValueError("Benefit criterion does not match the saved intervals")
    confidence = 100 * summary[columns[0][0]]["confidence"]
    fig.suptitle(f"Study v{version} · Primary benefit criterion {'passed' if gate else 'failed'}\nn={protocol['n']}, β={protocol['beta']:g} · {len(protocol['confirmation_seeds'])} fresh paired datasets · {protocol['steps']} updates per fit\nBoth declared {confidence:g}% intervals must be below zero · New simulation study\nEach point is one paired dataset; below the diagonal means lower parity KL", fontsize=12)
    save(fig, out, "confirmation_kl")


def write_points(out, pairs):
    with (out/"paired_points.csv").open("w", newline="") as file:
        writer = csv.writer(file); writer.writerow(["comparison", "seed", "parity_key", "mse_key", "parity_kl", "mse_kl", "delta_kl", "confidence"])
        for name, pair in pairs.items():
            contrast = pair["summary"]
            for seed, left, right in zip(pair["seeds"], pair["left"], pair["right"]):
                writer.writerow([name, seed, contrast["left"], contrast["right"], left, right, left-right, contrast["confidence"]])


def check_keys(contrast, left, right):
    key = lambda c: f"{c['architecture']}_{c['objective']}_lr{c['lr']:g}"
    if (contrast["left"], contrast["right"]) != (key(left), key(right)):
        raise ValueError("Summary contrast differs from the frozen configuration choices")


def confirmation(protocol, selected, summary, rows, out):
    configs = {
        "primary": (selected["parity"], selected["mse"]),
        "same_architecture": (selected["parity"], selected["same_architecture_mse"]),
        "sampled_family": (selected["parity_family"], selected["parity_family_mse"]),
        "expected_family": (selected["expected-parity_family"], selected["expected-parity_family_mse"]),
        "original_reference": (selected["paper_parity"], selected["paper_mse"]),
    }
    for name, (left, right) in configs.items():
        check_keys(summary[name], left, right)
    pairs = {name: checked_pair(summary[name], rows, protocol["confirmation_seeds"]) for name in configs}
    columns = [("primary", "Global comparison", selected["mse"])]
    if summary["primary"]["right"] != summary["same_architecture"]["right"]:
        columns.append(("same_architecture", "Same architecture", selected["same_architecture_mse"]))
    else:
        columns[0] = ("primary", "Global and same-architecture comparison", selected["mse"])
    main_comparisons(protocol, selected, summary, pairs, columns, out, 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), layout="constrained")
    for ax, (name, title) in zip(axes, (("sampled_family", "Selected sampled-parity family"), ("expected_family", "Selected expected-parity family"), ("original_reference", "Original ring reference · both learning rates .05"))):
        pair = pairs[name]
        delta_panel(ax, pair, title=f"{title}\n{pair['summary']['left']} versus {pair['summary']['right']}")
    fig.suptitle(f"Study v1 · Exploratory paired comparisons · {len(protocol['confirmation_seeds'])} confirmation seeds\nV1 primary criterion {'passed' if summary['robust_benefit_gate'] else 'failed'}. Family intervals: 97.5%; original-ring interval: 95%.", fontsize=12)
    save(fig, out, "exploratory_kl")
    write_points(out, pairs)


def replication(protocol, selected, summary, rows, out):
    columns = [("matched_dense_mse", "Same architecture · loss comparison", selected["same_architecture_mse"]),
               ("strong_ring_mse", "Stronger ring MSE · architecture also differs", selected["mse"])]
    for name, _, control in columns:
        check_keys(summary[name], selected["parity"], control)
    pairs = {name: checked_pair(summary[name], rows, protocol["confirmation_seeds"]) for name, _, _ in columns}
    main_comparisons(protocol, selected, summary, pairs, columns, out, 2)
    write_points(out, pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("all", "v1", "v2"), default="all")
    parser.add_argument("--results", type=Path, help="Override results directory; requires --study v1 or v2")
    parser.add_argument("--protocol", type=Path, help="Override frozen protocol; requires --study v1 or v2")
    parser.add_argument("--out", type=Path, default=ROOT/"runs/study-figures")
    parser.add_argument("--architecture-only", action="store_true")
    args = parser.parse_args()
    if args.study == "all" and (args.results or args.protocol):
        parser.error("Path overrides require --study v1 or v2")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "iqp-parity-study"})
    for version in ([1, 2] if args.study == "all" else [int(args.study[-1])]):
        results = args.results or ROOT/("results/study" if version == 1 else "results/replication")
        protocol_path = args.protocol or ROOT/("protocols/parity-study-v1.json" if version == 1 else "protocols/sampled-parity-replication-v2.json")
        seal_path = results/("selection.json" if version == 1 else "lock.json")
        protocol, seal = read(protocol_path), read(seal_path)
        if sha(protocol_path) != seal["fingerprint"]["protocol_sha256"]:
            raise ValueError("Protocol differs from the frozen selection or replication lock")
        if version == 1:
            selected = seal["selected"]
        else:
            if protocol != seal["protocol"]:
                raise ValueError("Replication lock differs from the protocol")
            configs = protocol["configurations"]
            selected = dict(parity=configs["sampled_parity"], mse=configs["strong_ring_mse"], same_architecture_mse=configs["matched_dense_mse"])
        sources = {"protocol": sha(protocol_path), "selection_or_lock": sha(seal_path), "renderer": sha(__file__)}
        summary, rows = None, None
        if not args.architecture_only:
            path = results/("confirmation-summary.json" if version == 1 else "summary.json")
            if not path.is_file():
                raise SystemExit(f"Study v{version} summary is not complete; use --architecture-only until it exists.")
            summary = read(path)
            metrics_path = results/("confirmation/metrics.csv" if version == 1 else "checkpoints/metrics.csv")
            with metrics_path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            sources.update(confirmation_summary=sha(path), confirmation_metrics=sha(metrics_path))
        out = args.out/f"v{version}"
        out.mkdir(parents=True, exist_ok=True)
        architecture(protocol, selected, out, version)
        if summary is not None:
            (confirmation if version == 1 else replication)(protocol, selected, summary, rows, out)
        names = ["architecture"] + ([] if args.architecture_only else ["confirmation_kl"] + (["exploratory_kl"] if version == 1 else []))
        generated = [out/f"{name}.{suffix}" for name in names for suffix in ("png", "pdf", "svg")]
        if not args.architecture_only:
            generated.append(out/"paired_points.csv")
        (out/"plot_manifest.json").write_text(json.dumps({"scope": f"Study v{version}; new simulation results, not a printed-paper reproduction; no pooling across studies", "source_sha256": sources, "architecture_only": args.architecture_only, "outputs": {p.name: sha(p) for p in sorted(generated)}}, indent=2)+"\n")
        print(f"Study v{version} figures written to {out}")


if __name__ == "__main__":
    main()
