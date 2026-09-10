#!/usr/bin/env python3
"""Reproducible classical development controls for the separate loss study.

python scripts/check_loss_baselines.py --out runs/loss-baselines --jobs 2
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
import torch

from iqp_repro import classical, cli, core, support_baselines
from iqp_repro.cli import metrics as scientific_metrics


def clean(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(float(value))
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def canonical(value):
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration_key(config):
    return (f'{config["model"]}_{config["architecture"]}_'
            f's{config["sigma"]:g}_lr{config["lr"]:g}_t{config["steps"]}_'
            f'{"support" if config.get("support_aware", True) else "paper"}')


def configuration_grid(protocol):
    configurations = []
    for model, grid in protocol["classical"].items():
        for architecture, sigma, lr, steps in itertools.product(
                grid["architectures"], grid["sigmas"], grid["learning_rates"], grid["steps"]):
            configurations.append(dict(model=model, architecture=architecture,
                                       sigma=sigma, lr=lr, steps=steps, support_aware=True, l2=0.))
    # The unmodified paper settings remain separately identifiable controls.
    for model, architecture, lr in [("ising-parity", "ring", .05),
                                    ("ising-nll", "dense", .05),
                                    ("maxent", "features", .05),
                                    ("transformer", "full", .001)]:
        configurations.append(dict(model=model, architecture=architecture, sigma=1.,
                                   lr=lr, steps=600, support_aware=False, l2=0.))
    return configurations


def train_paper(config, bits, samples, masks, seed):
    n = bits.shape[1]
    empirical = core.empirical(samples, n)
    indices = core.mask_indices(masks)
    moments = core.fwht(empirical)[indices]
    model = config["model"]
    if model == "ising-parity":
        result = classical.train_ising(bits, indices, moments, empirical,
                  topology="nn_nnn", loss="parity", seed=seed+30001,
                  lr=config["lr"], steps=config["steps"])
        parameters = n + len(core.pairs(n))
    elif model == "ising-nll":
        result = classical.train_ising(bits, indices, moments, empirical,
                  topology="dense", loss="nll", seed=seed+30004,
                  lr=config["lr"], steps=config["steps"])
        parameters = n + n*(n-1)//2
    elif model == "maxent":
        result = classical.train_maxent(indices, moments, n=n, seed=seed+36001,
                  lr=config["lr"], steps=config["steps"])
        parameters = len(indices)
    elif model == "transformer":
        result = classical.train_transformer(bits, samples, seed=seed+35501,
                  lr=config["lr"], epochs=config["steps"])
        parameters = sum(p.numel() for p in classical.ARTransformer(n).parameters())
    else:
        raise ValueError(model)
    result["parameters"] = parameters
    return result


def train_configuration(config, seed, protocol):
    started = time.perf_counter()
    torch.set_num_threads(1)
    p, support, scores = core.target(protocol["n"], config["beta"])
    bits = core.bits_table(protocol["n"])
    samples = np.random.default_rng(seed+7).choice(len(p), protocol["m"], p=p)
    validation = np.random.default_rng(seed+50000).choice(len(p), protocol["validation_samples"], p=p)
    masks = core.sample_masks(protocol["n"], config["sigma"], protocol["k"], seed+222)
    if config.get("support_aware", True):
        offsets = {"ising-parity": 30001, "ising-nll": 30004, "maxent": 36001, "transformer": 35501}
        result = support_baselines.train_baseline(config, bits, samples, masks, seed+offsets[config["model"]])
    else:
        result = train_paper(config, bits, samples, masks, seed)
    elite = core.elite(scores, support, samples)
    unseen = support & (core.empirical(samples, protocol["n"]) == 0)
    metrics = scientific_metrics(p, result["q"], support, unseen, logq=result["logq"])
    recovery = core.coverage(result["q"], elite, [1000])
    metrics.update(config, key=configuration_key(config), seed=seed,
                   recovery_1000=float(recovery["recovery"][0]), coverage_1000=float(recovery["yield"][0]),
                   validation_nll=float(-np.mean(result["logq"][validation])),
                   parameters=int(result["parameters"]), seconds=time.perf_counter()-started,
                   sample_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
                   validation_sha256=hashlib.sha256(validation.tobytes()).hexdigest())
    result.update(p=p, samples=samples, validation=validation, masks=masks,
                  support=support, elite=elite, metrics=metrics)
    return result


def run_one(config, seed, protocol, folder, fingerprint):
    path = Path(folder) / f'b{config["beta"]:g}_{configuration_key(config)}_seed{seed}.npz'
    specification = canonical(dict(config=config, seed=seed, protocol=protocol, fingerprint=fingerprint))
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data["specification"]) != specification:
                raise ValueError(f"stale checkpoint: {path}")
            return json.loads(str(data["metrics"]))
    try:
        result = train_configuration(config, seed, protocol)
    except Exception as exc:
        save_json(path.with_suffix(".failure.json"), dict(specification=json.loads(specification), error=repr(exc)))
        raise
    row = result.pop("metrics")
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **result, specification=specification, metrics=canonical(row))
    tmp.replace(path)
    return clean(row)


def ranking(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["beta"], row["key"]), []).append(row)
    result = []
    for (beta, key), values in grouped.items():
        first = values[0]
        result.append(dict(beta=beta, key=key,
            config={k: first[k] for k in ("model", "architecture", "sigma", "lr", "steps", "support_aware", "l2")},
            runs=len(values), mean_validation_nll=float(np.mean([float(v["validation_nll"]) for v in values])),
            mean_kl=float(np.mean([float(v["kl"]) for v in values]))))
    return sorted(result, key=lambda r: (r["beta"], r["config"]["model"], r["mean_validation_nll"], r["key"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/loss-advantage-development.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(protocol_sha256=digest(args.protocol), script_sha256=digest(__file__),
        support_baselines_sha256=digest(support_baselines.__file__), classical_sha256=digest(classical.__file__),
        metrics_source_sha256=digest(cli.__file__),
        core_sha256=digest(core.__file__), python=sys.version, numpy=np.__version__, torch=torch.__version__,
        platform=platform.platform())
    lock = args.out/"lock.json"
    if lock.exists() and json.loads(lock.read_text()) != fingerprint:
        raise ValueError("code, protocol or environment changed; use a fresh output directory")
    save_json(lock, fingerprint)
    configurations = configuration_grid(protocol)
    if args.models:
        configurations = [c for c in configurations if c["model"] in args.models]
    jobs = [(dict(config, beta=beta), seed) for beta, config, seed in
            itertools.product(protocol["betas"], configurations, protocol["development_seeds"])]
    folder = args.out/"checkpoints"
    folder.mkdir(exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, config, seed, protocol, folder, fingerprint) for config, seed in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 40 == 0 or i == len(futures):
                print(f"Classical development: {i}/{len(futures)}", flush=True)
    rows.sort(key=lambda r: (r["beta"], r["key"], r["seed"]))
    with (args.out/"metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ranked = ranking(rows)
    save_json(args.out/"ranking.json", ranked)
    best = {}
    for row in ranked:
        best.setdefault(f'{row["beta"]:g}/{row["config"]["model"]}/'
                        f'{"support" if row["config"]["support_aware"] else "paper"}', row)
    save_json(args.out/"selection.json", best)
    print(json.dumps(clean(best), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
