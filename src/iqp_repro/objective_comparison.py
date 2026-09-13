"""Reproduce historical losses, select learning rates, then confirm on new data.

python -m iqp_repro.objective_comparison historical --out runs/objectives
python -m iqp_repro.objective_comparison develop --out runs/objectives
python -m iqp_repro.objective_comparison confirm --out runs/objectives
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
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

from . import core, loss_search, objective_losses, study


def configuration_key(config):
    return f'{config["objective"]}_lr{config["lr"]:g}'


def fingerprint():
    return dict(source_hashes={Path(module.__file__).name: study.digest(module.__file__)
                              for module in (core, loss_search, objective_losses, study)}
                | {Path(__file__).name: study.digest(__file__)},
                python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(),
                threads={name: os.environ.get(name) for name in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})


def train_configuration(config, seed, protocol):
    """One final-iterate fit; normalization uses only its initial training gradient."""
    started = time.perf_counter()
    n, k = protocol["n"], protocol["k"]
    p, support, _ = core.target(n, protocol["beta"])
    samples = np.random.default_rng(seed + protocol["sample_seed_offset"]).choice(len(p), protocol["m"], p=p)
    validation = np.random.default_rng(seed + protocol["validation_seed_offset"]).choice(
        len(p), protocol["validation_samples"], p=p)
    empirical = core.empirical(samples, n)
    masks = core.sample_masks(n, protocol["sigma"], k, seed + protocol["mask_seed_offset"])
    weights = study.parity_weights(n, "parity", masks)
    circuit = loss_search.Circuit(n, protocol["architecture"])
    initial = .01 * np.random.default_rng(seed + protocol["initial_seed_offset"]).standard_normal(len(circuit.edges))
    theta = initial.copy()
    reference_norm = float(np.linalg.norm(circuit.loss_gradient(initial, empirical, "parity", weights)[1]))
    initial_norm = float(np.linalg.norm(objective_losses.loss_gradient(
        circuit, initial, empirical, config["objective"], weights)[1]))
    if not (np.isfinite(reference_norm) and np.isfinite(initial_norm) and reference_norm > 0 and initial_norm > 0):
        raise FloatingPointError("initial gradient norms must be finite and positive")
    scale = reference_norm / initial_norm
    options = protocol["optimizer"]
    optimizer = core.Adam(config["lr"], options["beta1"], options["beta2"], options["epsilon"])
    history, gradient_history = [], []
    for step in range(protocol["steps"] + 1):
        value, gradient = objective_losses.loss_gradient(circuit, theta, empirical, config["objective"], weights)
        if not (np.isfinite(value) and np.isfinite(gradient).all()):
            raise FloatingPointError(f"nonfinite objective or gradient at step {step}")
        history.append(value)
        gradient_history.append(float(np.linalg.norm(scale * gradient)))
        if step < protocol["steps"]:
            theta = optimizer.update(theta, scale * gradient)
    q = circuit.probabilities(theta)
    if not (np.isfinite(q).all() and np.isfinite(theta).all() and q.sum() > 0):
        raise FloatingPointError("invalid final circuit state")
    q /= q.sum()
    with np.errstate(divide="ignore"):
        validation_nll = float(-np.mean(np.log(q[validation])))
    row = dict(config, key=configuration_key(config), seed=int(seed), status="ok", error="",
               kl=core.forward_kl(p, q), validation_nll=validation_nll, parameters=len(theta),
               scale=scale, initial_gradient_norm=initial_norm, reference_gradient_norm=reference_norm,
               scaled_initial_gradient_norm=scale * initial_norm,
               sample_sha256=loss_search.array_digest(samples), initial_sha256=loss_search.array_digest(initial),
               mask_sha256=loss_search.array_digest(masks), validation_sha256=loss_search.array_digest(validation),
               seconds=time.perf_counter() - started)
    return dict(metrics=row, q=q, theta=theta, initial=initial, samples=samples, validation=validation,
                masks=masks, loss_history=np.array(history), scaled_gradient_history=np.array(gradient_history))


def run_one(config, seed, protocol, folder, provenance):
    path = Path(folder) / f'{configuration_key(config)}_seed{seed}.npz'
    failed = path.with_suffix(".failure.json")
    specification = dict(config=config, seed=seed, protocol=protocol, fingerprint=provenance)
    if path.exists() and failed.exists():
        raise ValueError(f"conflicting checkpoint records: {path}")
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data["specification"])) != specification:
                raise ValueError(f"stale checkpoint: {path}")
            return json.loads(str(data["metrics"]))
    if failed.exists():
        saved = json.loads(failed.read_text())
        if saved["specification"] != specification:
            raise ValueError(f"stale failure: {failed}")
        return saved["metrics"]
    try:
        result = train_configuration(config, seed, protocol)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        row = dict(config, key=configuration_key(config), seed=seed, status="failed", error=repr(exc),
                   kl="Infinity", validation_nll="Infinity", parameters=len(loss_search.Circuit(
                       protocol["n"], protocol["architecture"]).edges),
                   **{key: "unavailable" for key in (
                       "scale", "initial_gradient_norm", "reference_gradient_norm", "scaled_initial_gradient_norm",
                       "sample_sha256", "initial_sha256", "mask_sha256", "validation_sha256", "seconds")})
        loss_search.save_json(failed, dict(specification=specification, metrics=row))
        return row
    row = loss_search.json_safe(result.pop("metrics"))
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **result, metrics=study.canonical(row), specification=study.canonical(specification))
    tmp.replace(path)
    return row


def ranking(rows, configs, seeds):
    ranked = []
    for config in configs:
        key = configuration_key(config)
        group = [row for row in rows if row["key"] == key]
        if sorted(row["seed"] for row in group) != sorted(seeds):
            raise ValueError(f"incomplete or duplicate cohort: {key}")
        ranked.append(dict(config, key=key, fits=len(group), failures=sum(row["status"] != "ok" for row in group),
                           mean_kl=float(np.mean([float(row["kl"]) for row in group])),
                           mean_validation_nll=float(np.mean([float(row["validation_nll"]) for row in group]))))
    return sorted(ranked, key=lambda row: (row["mean_validation_nll"], row["key"]))


def paired_comparison(rows, first_key, second_key, seeds, comparisons=11):
    for key in (first_key, second_key):
        group = [r for r in rows if r["key"] == key]
        if sorted(r["seed"] for r in group) != sorted(seeds):
            raise ValueError("comparison requires complete cohorts without duplicate rows")
    left = {r["seed"]: r for r in rows if r["key"] == first_key}
    right = {r["seed"]: r for r in rows if r["key"] == second_key}
    if len(left) != len(seeds) or len(right) != len(seeds) or set(left) != set(seeds) or set(right) != set(seeds):
        raise ValueError("comparison requires complete paired cohorts")
    result = dict(first=first_key, second=second_key, pairs=len(seeds), bonferroni_factor=comparisons)
    if any(left[s]["status"] != "ok" or right[s]["status"] != "ok" or
           not np.isfinite(float(left[s]["kl"])) or not np.isfinite(float(right[s]["kl"])) for s in seeds):
        return dict(result, valid=False, mean_difference=None, nominal_95_ci=None, adjusted_95_ci=None,
                    reason="failed or nonfinite pair; no seeds omitted")
    for seed in seeds:
        for field in ("sample_sha256", "initial_sha256", "mask_sha256", "validation_sha256"):
            if not left[seed].get(field) or left[seed][field] == "unavailable":
                raise ValueError(f"missing pairing digest {field}: {seed}")
            if left[seed][field] != right[seed][field]:
                raise ValueError(f"unpaired {field}: {seed}")
    difference = np.array([float(left[s]["kl"]) - float(right[s]["kl"]) for s in seeds])
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / np.sqrt(len(seeds)))
    def interval(alpha):
        half = float(t.ppf(1 - alpha / 2, len(seeds) - 1) * se)
        return [mean - half, mean + half]
    return dict(result, valid=True, mean_difference=mean, nominal_95_ci=interval(.05),
                adjusted_95_ci=interval(.05 / comparisons), first_wins=int(np.sum(difference < 0)))


def save_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_configurations(ranked, objectives):
    chosen = []
    for objective in objectives:
        candidates = [r for r in ranked if r["objective"] == objective and not r["failures"]
                      and np.isfinite(float(r["mean_validation_nll"]))]
        if not candidates:
            raise ValueError(f"no eligible development setting for {objective}")
        best = min(candidates, key=lambda r: (float(r["mean_validation_nll"]), r["key"]))
        chosen.append({key: best[key] for key in ("objective", "lr")})
    return chosen


def run(stage, out, protocol_path, jobs):
    out, protocol_path = Path(out), Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    provenance = fingerprint()
    out.mkdir(parents=True, exist_ok=True)
    lock = dict(protocol=protocol, fingerprint=provenance)
    lock_path = out / "lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("code, protocol or environment changed; use a new output directory")
    loss_search.save_json(lock_path, lock)
    folder = out / "checkpoints"
    folder.mkdir(exist_ok=True)
    destination = out / stage
    destination.mkdir(exist_ok=True)
    if stage == "historical":
        configs, seeds = protocol["historical_configurations"], protocol["historical_seeds"]
    elif stage == "develop":
        configs = [dict(objective=o, lr=lr) for o, lr in itertools.product(protocol["objectives"], protocol["learning_rates"])]
        seeds = protocol["development_seeds"]
    else:
        selection_path = out / "develop" / "selection.json"
        selection = json.loads(selection_path.read_text())
        if selection["lock_sha256"] != study.digest(lock_path):
            raise ValueError("selection belongs to a different protocol/code/environment")
        if selection["metrics_sha256"] != study.digest(out / "develop" / "metrics.csv"):
            raise ValueError("development evidence changed after selection")
        configs, seeds = selection["configurations"], protocol["confirmation_seeds"]
        if set(c["objective"] for c in configs) != set(protocol["objectives"]) or len(configs) != len(protocol["objectives"]):
            raise ValueError("one selected configuration per objective required")
        seal_path = destination / "seal.json"
        seal = dict(configurations=configs, seeds=seeds, lock_sha256=study.digest(lock_path),
                    selection_sha256=study.digest(selection_path), comparisons=11)
        if seal_path.exists():
            prior = json.loads(seal_path.read_text())
            if {k: v for k, v in prior.items() if k != "frozen_at_utc"} != seal:
                raise ValueError("confirmation seal changed")
        else:
            loss_search.save_json(seal_path, dict(seal, frozen_at_utc=datetime.now(timezone.utc).isoformat()))
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_one, c, seed, protocol, folder, provenance) for c in configs for seed in seeds]
        for count, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if count % 20 == 0 or count == len(futures):
                print(f'{stage}: {count}/{len(futures)} fits retained ({time.perf_counter() - started:.1f}s)', flush=True)
    rows.sort(key=lambda row: (row["key"], row["seed"]))
    save_csv(destination / "metrics.csv", rows)
    ranked = ranking(rows, configs, seeds)
    summary = dict(stage=stage, seeds=seeds, fits=len(rows), failures=sum(r["status"] != "ok" for r in rows),
                   ranking=ranked, comparisons=[], wall_seconds=time.perf_counter() - started)
    if stage == "develop":
        chosen = select_configurations(ranked, protocol["objectives"])
        loss_search.save_json(destination / "selection.json", dict(configurations=chosen,
                              lock_sha256=study.digest(lock_path), metrics_sha256=study.digest(destination / "metrics.csv"),
                              selection_metric="mean independent validation NLL; exact KL never selects"))
    else:
        selected = {c["objective"]: configuration_key(c) for c in configs}
        pairs = [(selected["parity"], selected[o]) for o in protocol["objectives"] if o != "parity"]
        pairs += [(selected["mse"], selected[o]) for o in protocol["objectives"] if o not in ("parity", "mse")]
        if stage == "historical":
            # Include both NLL learning rates, without choosing the better test result.
            pairs += [(selected["parity"], "nll_lr0.1"), (selected["mse"], "nll_lr0.1")]
        summary["comparisons"] = [paired_comparison(rows, a, b, seeds, len(pairs)) for a, b in pairs]
    loss_search.save_json(destination / "summary.json", summary)
    loss_search.save_json(destination / "execution.json", dict(executable=sys.executable, jobs=jobs,
                          completed_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(loss_search.json_safe(sorted(ranked, key=lambda r: r["mean_kl"])), indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["historical", "develop", "confirm"])
    parser.add_argument("--protocol", type=Path, default=Path("protocols/objective-comparison.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    run(args.stage, args.out, args.protocol, args.jobs)


if __name__ == "__main__":
    main()
