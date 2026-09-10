"""Frozen development/confirmation study; original reproduction stays unchanged.

python -m iqp_repro.study develop --out results/study
python -m iqp_repro.study confirm --out results/study
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.stats import t

from . import core


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def edge_list(n, architecture):
    if architecture == "ring":
        return core.pairs(n)
    if architecture == "dense":
        return list(itertools.combinations(range(n), 2))
    raise ValueError("architecture must be ring or dense")


def parity_weights(n, objective, masks=None, sigma=1.):
    """Nonnegative normalized weights on the Walsh spectrum, zero mode excluded."""
    if objective == "parity":
        indices = core.mask_indices(masks)
        if not len(indices) or np.any(indices == 0):
            raise ValueError("nonempty, nonzero masks required")
        return np.bincount(indices, minlength=2**n) / len(indices)
    if objective != "expected-parity" or sigma <= 0:
        raise ValueError("unknown parity objective or invalid sigma")
    rho = -0.5 * np.expm1(-1 / (2 * sigma**2))
    orders = core.bits_table(n).sum(axis=1)
    weights = rho**orders * (1-rho)**(n-orders)
    weights[0] = 0
    return weights / weights.sum()


class Circuit:
    """One H--commuting RZZ--H block. Dense changes only the edge set."""

    def __init__(self, n, architecture):
        self.n = n
        self.edges = edge_list(n, architecture)
        self.indices = np.array([(1 << (n-i-1)) | (1 << (n-j-1)) for i,j in self.edges])
        self.size = 2**n

    def state(self, theta):
        if np.shape(theta) != (len(self.edges),):
            raise ValueError("one angle per edge required")
        coefficients = np.zeros(self.size)
        coefficients[self.indices] = theta
        diagonal = np.exp(-0.5j * core.fwht(coefficients)) / np.sqrt(self.size)
        amplitude = core.fwht(diagonal) / np.sqrt(self.size)
        return diagonal, amplitude

    def probabilities(self, theta):
        return np.abs(self.state(theta)[1])**2

    def loss_gradient(self, theta, empirical, objective, weights=None):
        diagonal, amplitude = self.state(theta)
        difference = np.abs(amplitude)**2 - empirical
        if objective in {"parity", "expected-parity"}:
            error = core.fwht(difference)
            value = np.dot(weights, error**2)
            dq = core.fwht(2 * weights * error)
        elif objective in {"mse", "scaled-mse"}:
            denominator = self.size//2 if objective == "mse" else 1
            value = np.dot(difference, difference) / denominator
            dq = 2 * difference / denominator
        else:
            raise ValueError("unknown objective")
        reverse = core.fwht(2 * dq * amplitude) / np.sqrt(self.size)
        phase_derivative = 0.5 * np.imag(np.conj(reverse) * diagonal)
        gradient = core.fwht(phase_derivative)[self.indices]
        return float(value), gradient


def train(config, empirical, masks, protocol):
    circuit = Circuit(protocol["n"], config["architecture"])
    theta = .01 * np.random.default_rng(config["seed"] + 10000 + 7*protocol["k"]).standard_normal(len(circuit.edges))
    initial = theta.copy()
    weights = parity_weights(protocol["n"], config["objective"], masks, protocol["sigma"]) if "parity" in config["objective"] else None
    optimizer = core.Adam(config["lr"], protocol["optimizer"]["beta1"], protocol["optimizer"]["beta2"], protocol["optimizer"]["epsilon"])
    loss_history, gradient_history = [], []
    for _ in range(protocol["steps"]):
        value, gradient = circuit.loss_gradient(theta, empirical, config["objective"], weights)
        loss_history.append(value)
        gradient_history.append(np.linalg.norm(gradient))
        theta = optimizer.update(theta, gradient)
    value, gradient = circuit.loss_gradient(theta, empirical, config["objective"], weights)
    loss_history.append(value)
    gradient_history.append(np.linalg.norm(gradient))
    q = circuit.probabilities(theta)
    q /= q.sum()
    return dict(q=q, theta=theta, initial=initial, loss_history=np.array(loss_history), gradient_history=np.array(gradient_history))


def configuration_key(config):
    return f'{config["architecture"]}_{config["objective"]}_lr{config["lr"]:g}'


def _run_one(config, protocol, out, fingerprint):
    path = Path(out) / f'{configuration_key(config)}_seed{config["seed"]}.npz'
    specification = dict(config=config, protocol=protocol, fingerprint=fingerprint)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data["specification"]) != canonical(specification):
                raise ValueError(f"stale checkpoint: {path}")
            return json.loads(str(data["metrics"]))
    started = time.perf_counter()
    p, support, scores = core.target(protocol["n"], protocol["beta"])
    samples = np.random.default_rng(config["seed"]+7).choice(len(p), protocol["m"], p=p)
    empirical = core.empirical(samples, protocol["n"])
    masks = core.sample_masks(protocol["n"], protocol["sigma"], protocol["k"], config["seed"]+222)
    result = train(config, empirical, masks, protocol)
    elite = core.elite(scores, support, samples)
    discovery = core.coverage(result["q"], elite, [1000])
    kl = core.forward_kl(p, result["q"])
    if not np.isfinite(kl):
        save_json(path.with_suffix(".failure.json"), dict(specification=specification,
                  error="nonfinite exact KL", kl=str(kl)))
        raise ValueError(f"nonfinite exact KL; retain failure, do not omit: {config}")
    validation = np.random.default_rng(config["seed"]+50000).choice(len(p), protocol["validation_samples"], p=p)
    validation_nll = float(-np.mean(np.log(result["q"][validation])))
    metrics = dict(config, key=configuration_key(config), kl=kl, validation_nll=validation_nll,
                   recovery_1000=float(discovery["recovery"][0]), coverage_1000=float(discovery["yield"][0]),
                   parameters=len(result["theta"]), seconds=time.perf_counter()-started,
                   sample_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
                   initial_sha256=hashlib.sha256(result["initial"].tobytes()).hexdigest())
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result, p=p, samples=samples, validation=validation, masks=masks, elite=elite,
                        specification=canonical(specification), metrics=canonical(metrics))
    temporary.replace(path)
    return metrics


def run_batch(configs, protocol, folder, fingerprint, jobs):
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_run_one, c, protocol, folder, fingerprint) for c in configs]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 20 == 0 or i == len(futures):
                print(f"{folder.name}: {i}/{len(futures)} completed", flush=True)
    rows.sort(key=lambda r:(r["key"], r["seed"]))
    with (folder/"metrics.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def rank_configurations(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["key"], []).append(row)
    ranked = []
    for key, values in groups.items():
        config = {name:values[0][name] for name in ("architecture", "objective", "lr")}
        ranked.append(dict(config, key=key, mean_kl=float(np.mean([v["kl"] for v in values])),
                           mean_validation_nll=float(np.mean([v["validation_nll"] for v in values])),
                           mean_recovery_1000=float(np.mean([v["recovery_1000"] for v in values])), runs=len(values)))
    return sorted(ranked, key=lambda r:(r["mean_validation_nll"], r["key"]))


def select(ranking):
    parity = next(r for r in ranking if "parity" in r["objective"])
    mse = next(r for r in ranking if "mse" in r["objective"])
    same = next(r for r in ranking if "mse" in r["objective"] and r["architecture"] == parity["architecture"])
    selected = dict(parity=parity, mse=mse, same_architecture_mse=same)
    for objective in ("parity", "expected-parity"):
        family = next(r for r in ranking if r["objective"] == objective)
        selected[objective+"_family"] = family
        selected[objective+"_family_mse"] = next(r for r in ranking if "mse" in r["objective"] and r["architecture"] == family["architecture"])
    selected["paper_parity"] = dict(architecture="ring", objective="parity", lr=.05, key="ring_parity_lr0.05")
    selected["paper_mse"] = dict(architecture="ring", objective="mse", lr=.05, key="ring_mse_lr0.05")
    return selected


def paired_summary(left, right, rows, confidence=.95):
    a = {r["seed"]:r for r in rows if r["key"] == left["key"]}
    b = {r["seed"]:r for r in rows if r["key"] == right["key"]}
    if a.keys() != b.keys() or len(a) < 2:
        raise ValueError("missing paired seeds")
    seeds = sorted(a)
    for seed in seeds:
        if a[seed]["sample_sha256"] != b[seed]["sample_sha256"]:
            raise ValueError("unpaired training samples")
        if left["architecture"] == right["architecture"] and a[seed]["initial_sha256"] != b[seed]["initial_sha256"]:
            raise ValueError("unpaired initialization")
    deltas = np.array([a[s]["kl"]-b[s]["kl"] for s in seeds])
    half = float(t.ppf((1+confidence)/2, len(seeds)-1)*deltas.std(ddof=1)/np.sqrt(len(seeds)))
    return dict(left=left["key"], right=right["key"], n=len(seeds), confidence=confidence,
                parity_mean_kl=float(np.mean([a[s]["kl"] for s in seeds])), mse_mean_kl=float(np.mean([b[s]["kl"] for s in seeds])),
                mean_difference=float(deltas.mean()), ci=[float(deltas.mean()-half), float(deltas.mean()+half)],
                wins=int(np.sum(deltas<0)), seeds=seeds, paired_differences=deltas.tolist(),
                parity_recovery_1000=float(np.mean([a[s]["recovery_1000"] for s in seeds])),
                mse_recovery_1000=float(np.mean([b[s]["recovery_1000"] for s in seeds])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "confirm"])
    parser.add_argument("--protocol", type=Path, default=Path("protocols/parity-study-v1.json"))
    parser.add_argument("--out", type=Path, default=Path("results/study"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if set(protocol["development_seeds"]) & set(protocol["confirmation_seeds"]):
        raise ValueError("development and confirmation must be disjoint")
    fingerprint = dict(study_sha256=digest(__file__), core_sha256=digest(core.__file__),
                       protocol_sha256=digest(args.protocol), python=sys.version, numpy=np.__version__,
                       platform=platform.platform())
    args.out.mkdir(parents=True, exist_ok=True)
    lock_path = args.out/"lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != fingerprint:
        raise ValueError("protocol, code or environment changed; use a new output directory")
    save_json(lock_path, fingerprint)
    if args.stage == "develop":
        if (args.out/"confirmation").exists():
            raise ValueError("cannot reselect after confirmation starts")
        configs = [dict(architecture=a, objective=o, lr=lr, seed=s)
                   for a,o,lr,s in itertools.product(protocol["architectures"], protocol["objectives"], protocol["learning_rates"], protocol["development_seeds"])]
        rows = run_batch(configs, protocol, args.out/"development", fingerprint, args.jobs)
        ranking = rank_configurations(rows)
        save_json(args.out/"development-ranking.json", ranking)
        save_json(args.out/"selection.json", dict(fingerprint=fingerprint, selected=select(ranking),
                  development_metrics_sha256=digest(args.out/"development"/"metrics.csv")))
        print(canonical(select(ranking)), flush=True)
    else:
        selection_path = args.out/"selection.json"
        selection = json.loads(selection_path.read_text())
        if selection["fingerprint"] != fingerprint or selection["development_metrics_sha256"] != digest(args.out/"development"/"metrics.csv"):
            raise ValueError("selection provenance changed")
        seal_path = args.out/"confirmation-seal.json"
        seal = dict(selection_sha256=digest(selection_path), fingerprint=fingerprint)
        if seal_path.exists() and json.loads(seal_path.read_text()) != seal:
            raise ValueError("selection changed after confirmation started")
        save_json(seal_path, seal)
        selected = selection["selected"]
        unique = {c["key"]:{k:c[k] for k in ("architecture", "objective", "lr")} for c in selected.values()}
        configs = [dict(c, seed=s) for c in unique.values() for s in protocol["confirmation_seeds"]]
        rows = run_batch(configs, protocol, args.out/"confirmation", fingerprint, args.jobs)
        results = {}
        for name,left,right,confidence in [
            ("primary", "parity", "mse", .95),
            ("same_architecture", "parity", "same_architecture_mse", .95),
            ("sampled_family", "parity_family", "parity_family_mse", .975),
            ("expected_family", "expected-parity_family", "expected-parity_family_mse", .975),
            ("original_reference", "paper_parity", "paper_mse", .95)]:
            results[name] = paired_summary(selected[left], selected[right], rows, confidence)
        results["robust_benefit_gate"] = all(results[k]["ci"][1] < 0 for k in ("primary", "same_architecture"))
        results["training_runs"] = len(rows)
        save_json(args.out/"confirmation-summary.json", results)
        print(canonical(results), flush=True)


if __name__ == "__main__":
    main()
