"""Bounded IQP development search; validation NLL selects, target KL diagnoses.

python -m iqp_repro.loss_search --out runs/loss-search/iqp --jobs 3

This command uses the historical development seeds only. It never runs a
confirmation cohort. A failure is retained and ranked with infinite validation
NLL; a valid distribution with infinite exact KL is likewise retained.
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
import traceback

import numpy as np
import scipy

from . import core, study


class Circuit(study.Circuit):
    """Original ring or dense circuit, plus a ring with distances one to three."""

    def __init__(self, n, architecture):
        if architecture != "ring3":
            super().__init__(n, architecture)
            return
        if not isinstance(n, (int, np.integer)) or n < 4:
            raise ValueError("ring3 requires an integer n >= 4")
        self.n = n
        self.edges = sorted({tuple(sorted((i, (i + d) % n)))
                             for i in range(n) for d in (1, 2, 3)
                             if i != (i + d) % n})
        self.indices = np.array([(1 << (n-i-1)) | (1 << (n-j-1))
                                 for i, j in self.edges])
        self.size = 2**n


def development_protocol():
    return dict(
        stage="development", n=12, betas=[.9, 1.2, 1.5, 1.8], m=200,
        development_seeds=list(range(111, 121)), steps=600, k=512,
        validation_samples=2000,
        optimizer=dict(name="PennyLane-style Adam", beta1=.9, beta2=.99,
                       epsilon=1e-8, epsilon_placement="before second-moment bias correction"),
        architectures=["ring", "ring3", "dense"],
        parity_sigmas=[.75, 1., 1.5, 2.], parity_learning_rates=[.02, .05, .1],
        mse_objectives=["mse", "scaled-mse"],
        mse_learning_rates=[.005, .01, .02, .05, .1, .2],
        mse_sigma=1.,
        data_rng="default_rng(seed+7); m IID draws with replacement",
        mask_rng="default_rng(seed+222); nonzero Bernoulli masks, duplicates retained",
        initialization=".01*default_rng(seed+10000+7*k).standard_normal(parameter_count)",
        validation_rng="default_rng(seed+50000); independent IID target draws",
        selection="Mean validation NLL over all ten development seeds; target KL is diagnostic only",
        stopping_rule="All 2880 specified fits once; final iterate after 600 updates; retain failures and infinite metrics",
        claim_boundary="Exploratory architecture and optimizer development on historical seeds, not independent confirmation",
    )


def configurations(protocol):
    result = []
    for beta, architecture in itertools.product(protocol["betas"], protocol["architectures"]):
        result.extend(dict(beta=beta, architecture=architecture, objective="parity", sigma=sigma, lr=lr)
                      for sigma, lr in itertools.product(protocol["parity_sigmas"], protocol["parity_learning_rates"]))
        result.extend(dict(beta=beta, architecture=architecture, objective=objective,
                           sigma=protocol["mse_sigma"], lr=lr)
                      for objective, lr in itertools.product(protocol["mse_objectives"], protocol["mse_learning_rates"]))
    return result


def configuration_key(config):
    return (f'b{config["beta"]:g}_{config["architecture"]}_{config["objective"]}'
            f'_sigma{config["sigma"]:g}_lr{config["lr"]:g}')


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def save_json(path, value):
    study.save_json(path, json_safe(value))


def train_configuration(config, seed, protocol):
    """Train one final-iterate IQP fit with shared data and initialization.

    Config fields: beta, architecture (ring/ring3/dense), objective
    (parity/mse/scaled-mse), sigma, lr. Protocol fields: n,m,k,steps,
    validation_samples, optimizer (beta1,beta2,epsilon). Returns NumPy arrays
    q, logq, theta, initial, history, loss_history, gradient_history, samples,
    validation, masks, p, support, scores, elite, plus a metrics dictionary.
    Exact zero probabilities remain zero; logq may contain negative infinity.
    """
    if config["objective"] not in {"parity", "mse", "scaled-mse"}:
        raise ValueError("objective must be parity, mse, or scaled-mse")
    if protocol["steps"] < 1 or config["lr"] <= 0:
        raise ValueError("positive learning rate and at least one update required")
    started = time.perf_counter()
    p, support, scores = core.target(protocol["n"], config["beta"])
    samples = np.random.default_rng(seed + 7).choice(len(p), protocol["m"], p=p)
    validation = np.random.default_rng(seed + 50000).choice(len(p), protocol["validation_samples"], p=p)
    empirical = core.empirical(samples, protocol["n"])
    masks = core.sample_masks(protocol["n"], config["sigma"], protocol["k"], seed + 222)
    circuit = Circuit(protocol["n"], config["architecture"])
    initial = .01 * np.random.default_rng(seed + 10000 + 7*protocol["k"]).standard_normal(len(circuit.edges))
    theta = initial.copy()
    weights = study.parity_weights(protocol["n"], "parity", masks) if config["objective"] == "parity" else None
    options = protocol["optimizer"]
    optimizer = core.Adam(config["lr"], options["beta1"], options["beta2"], options["epsilon"])
    history, gradients = [], []
    for _ in range(protocol["steps"]):
        loss, gradient = circuit.loss_gradient(theta, empirical, config["objective"], weights)
        history.append(loss)
        gradients.append(float(np.linalg.norm(gradient)))
        theta = optimizer.update(theta, gradient)
    loss, gradient = circuit.loss_gradient(theta, empirical, config["objective"], weights)
    history.append(loss)
    gradients.append(float(np.linalg.norm(gradient)))
    q = circuit.probabilities(theta)
    if not (np.isfinite(theta).all() and np.isfinite(q).all() and
            np.isfinite(history).all() and np.isfinite(gradients).all() and q.sum() > 0):
        raise FloatingPointError("nonfinite optimization state or invalid Born law")
    q /= q.sum()
    with np.errstate(divide="ignore"):
        logq = np.log(q)
    elite = core.elite(scores, support, samples)
    discovery = core.coverage(q, elite, [1000])
    metrics = dict(config, seed=int(seed), key=configuration_key(config),
                   family="parity" if config["objective"] == "parity" else "mse",
                   parameters=len(theta), status="ok", kl=core.forward_kl(p, q),
                   validation_nll=float(-np.mean(logq[validation])),
                   recovery_1000=float(discovery["recovery"][0]),
                   coverage_1000=float(discovery["yield"][0]),
                   seconds=time.perf_counter()-started,
                   sample_sha256=array_digest(samples), initial_sha256=array_digest(initial),
                   mask_sha256=array_digest(masks), validation_sha256=array_digest(validation), error="")
    history = np.array(history)
    return dict(q=q, logq=logq, theta=theta, initial=initial, history=history,
                loss_history=history, gradient_history=np.array(gradients),
                samples=samples, validation=validation, masks=masks, p=p,
                support=support, scores=scores, elite=elite, metrics=metrics)


def fingerprint():
    return dict(loss_search_sha256=study.digest(__file__), core_sha256=study.digest(core.__file__),
                study_sha256=study.digest(study.__file__), python=sys.version,
                numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                thread_environment={name: os.environ.get(name) for name in
                                    ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]})


def run_one(config, seed, protocol, folder, provenance):
    folder = Path(folder)
    path = folder / f"{configuration_key(config)}_seed{seed}.npz"
    failed = path.with_suffix(".failure.json")
    specification = dict(config=config, seed=seed, protocol=protocol, fingerprint=provenance)
    if path.exists() and failed.exists():
        raise ValueError(f"both success and failure artifacts exist: {path}")
    if path.exists():
        with np.load(path, allow_pickle=False) as arrays:
            if json.loads(str(arrays["specification"])) != specification:
                raise ValueError(f"checkpoint provenance changed: {path}")
            return json.loads(str(arrays["metrics"]))
    if failed.exists():
        saved = json.loads(failed.read_text())
        if saved["specification"] != specification:
            raise ValueError(f"failure provenance changed: {failed}")
        return saved["metrics"]
    try:
        result = train_configuration(config, seed, protocol)
    except Exception as error:
        metrics = dict(config, seed=seed, key=configuration_key(config),
                       family="parity" if config["objective"] == "parity" else "mse",
                       parameters=len(Circuit(protocol["n"], config["architecture"]).edges),
                       status="failed", kl=float("inf"), validation_nll=float("inf"),
                       recovery_1000=float("nan"), coverage_1000=float("nan"), seconds=float("nan"),
                       sample_sha256="unavailable", initial_sha256="unavailable",
                       mask_sha256="unavailable", validation_sha256="unavailable",
                       error=f"{type(error).__name__}: {error}")
        save_json(failed, dict(specification=specification, metrics=metrics, traceback=traceback.format_exc()))
        return json_safe(metrics)
    metrics = json_safe(result.pop("metrics"))
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result, metrics=study.canonical(metrics),
                        specification=study.canonical(specification))
    temporary.replace(path)
    return metrics


def summarize(rows, protocol):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["key"], []).append(row)
    ranked = []
    for key, group in grouped.items():
        if sorted(row["seed"] for row in group) != sorted(protocol["development_seeds"]):
            raise ValueError(f"incomplete development seeds: {key}")
        item = {name: group[0][name] for name in ["beta", "architecture", "objective", "sigma", "lr", "family", "parameters"]}
        item.update(key=key, fits=len(group), failures=sum(row["status"] != "ok" for row in group),
                    nonfinite_validation_nll=sum(not np.isfinite(float(row["validation_nll"])) for row in group),
                    mean_validation_nll=float(np.mean([float(row["validation_nll"]) for row in group])),
                    mean_kl=float(np.mean([float(row["kl"]) for row in group])),
                    mean_recovery_1000=float(np.mean([float(row["recovery_1000"]) for row in group])))
        ranked.append(item)
    ranked.sort(key=lambda row: (row["beta"], row["family"], row["architecture"], row["mean_validation_nll"], row["key"]))
    selections = []
    for beta, family, architecture in itertools.product(protocol["betas"], ["parity", "mse"], protocol["architectures"]):
        candidates = [row for row in ranked if (row["beta"], row["family"], row["architecture"]) == (beta, family, architecture)]
        chosen = min(candidates, key=lambda row: (row["mean_validation_nll"], row["key"]))
        selections.append(dict(beta=beta, family=family, architecture=architecture,
                               candidates=len(candidates), selected=chosen,
                               selectable=bool(np.isfinite(chosen["mean_validation_nll"]))))
    overall = []
    for beta, family in itertools.product(protocol["betas"], ["parity", "mse"]):
        candidates = [row for row in ranked if (row["beta"], row["family"]) == (beta, family)]
        chosen = min(candidates, key=lambda row: (row["mean_validation_nll"], row["key"]))
        overall.append(dict(beta=beta, family=family, candidates=len(candidates), selected=chosen,
                            selectable=bool(np.isfinite(chosen["mean_validation_nll"]))))
    return dict(stage="development", fits=len(rows), failed_fits=sum(row["status"] != "ok" for row in rows),
                selection_metric="mean validation NLL over every specified development seed; key breaks exact ties",
                claim_boundary=protocol["claim_boundary"], ranking=ranked,
                selections_by_architecture=selections, selections_overall=overall)


def run_development(out, jobs=3, protocol_path=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    folder = out / "checkpoints"
    folder.mkdir(exist_ok=True)
    protocol, provenance = development_protocol(), fingerprint()
    seal = dict(protocol=protocol, fingerprint=provenance)
    if protocol_path is not None:
        source = json.loads(Path(protocol_path).read_text())
        for field in ["n", "betas", "m", "k", "steps", "development_seeds", "validation_samples",
                      "architectures", "parity_learning_rates", "mse_learning_rates", "mse_objectives"]:
            if source[field] != protocol[field]:
                raise ValueError(f"canonical development protocol differs: {field}")
        if source["sigmas"] != protocol["parity_sigmas"] or any(
                source[field] != value for field, value in
                [("sample_seed_offset", 7), ("validation_seed_offset", 50000), ("mask_seed_offset", 222)]):
            raise ValueError("canonical development masks or RNG specification differs")
        if any(source["optimizer"][field] != protocol["optimizer"][field]
               for field in ["beta1", "beta2", "epsilon"]):
            raise ValueError("canonical development optimizer differs")
        provenance["canonical_protocol_sha256"] = study.digest(protocol_path)
        seal["canonical_protocol"] = source
    lock = out / "lock.json"
    if lock.exists() and json.loads(lock.read_text()) != seal:
        raise ValueError("development protocol, code, or environment changed; choose a new output directory")
    save_json(lock, seal)
    save_json(out / "execution.json", dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                                         workers=jobs, executable=sys.executable))
    tasks = [(config, seed) for config in configurations(protocol) for seed in protocol["development_seeds"]]
    if len(tasks) != 2880:
        raise ValueError("bounded development grid must contain exactly 2880 fits")
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_one, config, seed, protocol, folder, provenance) for config, seed in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 20 == 0 or i == len(tasks):
                print(f"Development: {i}/{len(tasks)} fits retained", flush=True)
    rows.sort(key=lambda row: (row["key"], row["seed"]))
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, protocol)
    save_json(out / "summary.json", summary)
    print(json.dumps(json_safe(summary["selections_overall"]), indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/loss-advantage-development.json"))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    run_development(args.out, args.jobs, args.protocol)


if __name__ == "__main__":
    main()
