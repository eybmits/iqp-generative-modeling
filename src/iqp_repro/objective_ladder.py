"""Seven gradient-matched objectives across the existing IQP ring radii.

python -m iqp_repro.objective_ladder develop --out runs/objective-ladder
python -m iqp_repro.objective_ladder confirm --out runs/objective-ladder
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

from . import core, loss_search, objective_comparison, objective_losses, parameter_ladder, study


def configuration_key(config):
    return (f'r{config["radius"]}_{config["objective"]}'
            f'_sigma{config["sigma"]:g}_lr{config["lr"]:g}')


def configurations(protocol):
    return [dict(radius=r, objective=o, sigma=protocol["sigma_by_radius"][str(r)], lr=lr)
            for r, o, lr in itertools.product(protocol["radii"], protocol["objectives"], protocol["learning_rates"])]


def validate_protocol(protocol):
    for field in ("radii", "objectives", "learning_rates", "development_seeds", "confirmation_seeds"):
        if not protocol[field] or len(set(protocol[field])) != len(protocol[field]):
            raise ValueError(f"empty or duplicate {field}")
    expected = {"parity", "mse", "nll", "spherical", "hellinger", "js", "tv"}
    if set(protocol["objectives"]) != expected:
        raise ValueError("exactly the seven declared objective families are required")
    if any(len(protocol[field]) < 2 for field in ("development_seeds", "confirmation_seeds")):
        raise ValueError("at least two paired seeds per cohort required")
    if set(protocol["development_seeds"]) & set(protocol["confirmation_seeds"]):
        raise ValueError("development and confirmation overlap")
    for radius in protocol["radii"]:
        parameter_ladder.Circuit(protocol["n"], radius)
        sigma = protocol["sigma_by_radius"][str(radius)]
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("positive finite reference sigma required")
    if any(not np.isfinite(lr) or lr <= 0 for lr in protocol["learning_rates"]):
        raise ValueError("positive finite learning rates required")
    if any(not isinstance(protocol[f], int) or protocol[f] < 1
           for f in ("m", "k", "steps", "validation_samples")):
        raise ValueError("positive integer budgets required")


def fingerprint():
    result = objective_comparison.fingerprint()
    result["source_hashes"].update({"objective_ladder.py": study.digest(__file__),
                                    "parameter_ladder.py": study.digest(parameter_ladder.__file__)})
    return result


def train_configuration(config, seed, protocol):
    """Same objective/Adam kernels as the 36-parameter study; vary the edge set."""
    started = time.perf_counter()
    n = protocol["n"]
    p, _, _ = core.target(n, protocol["beta"])
    samples = np.random.default_rng(seed + protocol["sample_seed_offset"]).choice(len(p), protocol["m"], p=p)
    validation = np.random.default_rng(seed + protocol["validation_seed_offset"]).choice(
        len(p), protocol["validation_samples"], p=p)
    empirical = core.empirical(samples, n)
    masks = core.sample_masks(n, config["sigma"], protocol["k"], seed + protocol["mask_seed_offset"])
    weights = study.parity_weights(n, "parity", masks)
    circuit = parameter_ladder.Circuit(n, config["radius"])
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
    history, gradients = [], []
    for step in range(protocol["steps"] + 1):
        value, gradient = objective_losses.loss_gradient(circuit, theta, empirical, config["objective"], weights)
        if not (np.isfinite(value) and np.isfinite(gradient).all()):
            raise FloatingPointError(f"nonfinite loss or gradient at step {step}")
        history.append(value)
        gradients.append(float(np.linalg.norm(scale * gradient)))
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
                masks=masks, loss_history=np.array(history), scaled_gradient_history=np.array(gradients))


def run_one(config, seed, protocol, folder, provenance):
    path = Path(folder) / f'{configuration_key(config)}_seed{seed}.npz'
    failure = path.with_suffix(".failure.json")
    specification = dict(config=config, seed=seed, protocol=protocol, fingerprint=provenance)
    if path.exists() and failure.exists():
        raise ValueError("both success and failure records exist")
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved["specification"])) != specification:
                raise ValueError(f"stale checkpoint: {path}")
            return json.loads(str(saved["metrics"]))
    if failure.exists():
        saved = json.loads(failure.read_text())
        if saved["specification"] != specification:
            raise ValueError(f"stale failure: {failure}")
        return saved["metrics"]
    try:
        result = train_configuration(config, seed, protocol)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        row = dict(config, key=configuration_key(config), seed=seed, status="failed", error=repr(exc),
                   kl="Infinity", validation_nll="Infinity",
                   parameters=len(parameter_ladder.Circuit(protocol["n"], config["radius"]).edges),
                   **{k: "unavailable" for k in ("scale", "initial_gradient_norm", "reference_gradient_norm",
                      "scaled_initial_gradient_norm", "sample_sha256", "initial_sha256", "mask_sha256",
                      "validation_sha256", "seconds")})
        loss_search.save_json(failure, dict(specification=specification, metrics=row))
        return row
    row = loss_search.json_safe(result.pop("metrics"))
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result, metrics=study.canonical(row), specification=study.canonical(specification))
    temporary.replace(path)
    return row


def summarize_ranking(rows, configs, seeds):
    expected = {(configuration_key(c), seed) for c in configs for seed in seeds}
    if len(rows) != len(expected) or {(r["key"], r["seed"]) for r in rows} != expected:
        raise ValueError("incomplete or duplicate cohort")
    ranked = []
    for config in configs:
        key = configuration_key(config)
        group = [row for row in rows if row["key"] == key]
        ranked.append(dict(config, key=key, fits=len(group), parameters=int(group[0]["parameters"]),
                           failures=sum(r["status"] != "ok" for r in group),
                           mean_kl=float(np.mean([float(r["kl"]) for r in group])),
                           mean_validation_nll=float(np.mean([float(r["validation_nll"]) for r in group]))))
    return sorted(ranked, key=lambda r: (r["radius"], r["mean_validation_nll"], r["key"]))


def select_configurations(ranking, protocol):
    result = []
    for radius, objective in itertools.product(protocol["radii"], protocol["objectives"]):
        eligible = [r for r in ranking if r["radius"] == radius and r["objective"] == objective
                    and r["failures"] == 0 and np.isfinite(float(r["mean_validation_nll"]))]
        if not eligible:
            raise ValueError(f"no eligible setting for radius {radius}, {objective}")
        best = min(eligible, key=lambda r: (float(r["mean_validation_nll"]), r["key"]))
        result.append({k: best[k] for k in ("radius", "objective", "sigma", "lr")})
    return result


def confirmation_summary(rows, configs, protocol):
    seeds = protocol["confirmation_seeds"]
    factor = len(protocol["radii"]) * (len(protocol["objectives"]) - 1)
    comparisons, stability, architectures = [], [], []
    for radius in protocol["radii"]:
        selected = {c["objective"]: configuration_key(c) for c in configs if c["radius"] == radius}
        local_comparisons = []
        for objective in protocol["objectives"]:
            group = sorted([r for r in rows if r["key"] == selected[objective]], key=lambda r: r["seed"])
            values = np.array([float(r["kl"]) for r in group])
            valid = all(r["status"] == "ok" for r in group) and np.isfinite(values).all()
            item = dict(radius=radius, objective=objective, parameters=int(group[0]["parameters"]), valid=bool(valid))
            item.update({name: float(fn(values)) if valid else None for name, fn in
                         (("mean", np.mean), ("median", np.median), ("sd", lambda a: a.std(ddof=1)),
                          ("p90", lambda a: np.quantile(a, .9)), ("p95", lambda a: np.quantile(a, .95)), ("max", np.max))})
            stability.append(item)
            if objective != "parity":
                result = objective_comparison.paired_comparison(rows, selected["parity"], selected[objective], seeds, factor)
                local_comparisons.append(dict(result, radius=radius, control=objective))
        comparisons.extend(local_comparisons)
        local = [r for r in stability if r["radius"] == radius]
        all_finite = all(r["valid"] for r in local)
        best = min(local, key=lambda r: (r["mean"], r["objective"]))["objective"] if all_finite else None
        passed = sum(c["valid"] and c["adjusted_95_ci"][1] < 0 for c in local_comparisons)
        architectures.append(dict(radius=radius, parameters=int(local[0]["parameters"]), best_mean_objective=best,
                                   adjusted_parity_wins=passed, planned_controls=len(local_comparisons),
                                   parity_beats_all_adjusted=passed == len(local_comparisons)))
    return dict(comparisons=comparisons, stability=stability, architectures=architectures,
                bonferroni_factor=factor, passing_architectures=sum(a["parity_beats_all_adjusted"] for a in architectures))


def read_rows(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("radius", "seed", "parameters"):
            row[key] = int(row[key])
        for key in ("sigma", "lr", "kl", "validation_nll"):
            row[key] = float(row[key])
    return rows


def run(stage, out, protocol_path, jobs=4):
    out, protocol_path = Path(out), Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    provenance = fingerprint()
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / "lock.json"
    lock = dict(protocol=protocol, fingerprint=provenance)
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("code, protocol or environment changed; choose a new output directory")
    loss_search.save_json(lock_path, lock)
    destination, folder = out / stage, out / "checkpoints"
    destination.mkdir(exist_ok=True)
    folder.mkdir(exist_ok=True)
    if stage == "develop":
        configs, seeds = configurations(protocol), protocol["development_seeds"]
    elif stage == "confirm":
        selection_path = out / "develop/selection.json"
        selection = json.loads(selection_path.read_text())
        if selection["lock_sha256"] != study.digest(lock_path) or selection["metrics_sha256"] != study.digest(out / "develop/metrics.csv"):
            raise ValueError("development provenance changed")
        # Recompute selection from the complete grid before permitting any confirmation fit.
        ranked = summarize_ranking(read_rows(out / "develop/metrics.csv"), configurations(protocol), protocol["development_seeds"])
        configs = select_configurations(ranked, protocol)
        if configs != selection["configurations"]:
            raise ValueError("selection differs from complete validation evidence")
        seeds = protocol["confirmation_seeds"]
        seal = dict(configurations=configs, seeds=seeds, lock_sha256=study.digest(lock_path),
                    selection_sha256=study.digest(selection_path), comparisons=len(protocol["radii"]) * 6)
        seal_path = destination / "seal.json"
        if seal_path.exists():
            prior = json.loads(seal_path.read_text())
            if {k: v for k, v in prior.items() if k != "frozen_at_utc"} != seal:
                raise ValueError("confirmation seal changed")
        else:
            loss_search.save_json(seal_path, dict(seal, frozen_at_utc=datetime.now(timezone.utc).isoformat()))
    else:
        raise ValueError("stage must be develop or confirm")
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_one, c, seed, protocol, folder, provenance) for c in configs for seed in seeds]
        for count, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if count % 100 == 0 or count == len(futures):
                print(f'{stage}: {count}/{len(futures)} fits ({time.perf_counter() - started:.1f}s)', flush=True)
    rows.sort(key=lambda r: (r["key"], r["seed"]))
    with (destination / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    ranked = summarize_ranking(rows, configs, seeds)
    summary = dict(stage=stage, seeds=seeds, fits=len(rows), failures=sum(r["status"] != "ok" for r in rows), ranking=ranked)
    if stage == "develop":
        selected = select_configurations(ranked, protocol)
        loss_search.save_json(destination / "selection.json", dict(configurations=selected,
                              lock_sha256=study.digest(lock_path), metrics_sha256=study.digest(destination / "metrics.csv")))
        print(json.dumps(selected, indent=2), flush=True)
    else:
        summary.update(confirmation_summary(rows, configs, protocol))
        print(json.dumps(summary["architectures"], indent=2), flush=True)
    loss_search.save_json(destination / "summary.json", summary)
    loss_search.save_json(destination / "execution.json", dict(executable=sys.executable, jobs=jobs,
                          wall_seconds=time.perf_counter() - started, completed_at_utc=datetime.now(timezone.utc).isoformat()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "confirm"])
    parser.add_argument("--protocol", type=Path, default=Path("protocols/objective-ladder.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    run(args.stage, args.out, args.protocol, args.jobs)


if __name__ == "__main__":
    main()
