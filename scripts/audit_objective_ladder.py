#!/usr/bin/env python3
"""Audit every retained architecture/objective fit and all 24 paired intervals."""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

from iqp_repro import objective_losses, parameter_ladder

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/audit_objective_comparison.py"
SPEC = importlib.util.spec_from_file_location("previous_objective_audit", HELPER)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
require, same, digest = base.require, base.same, base.digest


def key(config):
    return f'r{config["radius"]}_{config["objective"]}_sigma{config["sigma"]:g}_lr{config["lr"]:g}'


def read_rows(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        for name in ("seed", "radius", "parameters"):
            r[name] = int(r[name])
        for name in ("sigma", "lr", "kl", "validation_nll"):
            r[name] = float(r[name])
    return rows


def ranking(rows, configs, seeds):
    expected = {(key(c), seed) for c in configs for seed in seeds}
    require(len(rows) == len(expected) and {(r["key"], r["seed"]) for r in rows} == expected,
            "incomplete or duplicate cohort")
    result = []
    for c in configs:
        group = [r for r in rows if r["key"] == key(c)]
        result.append(dict(c, key=key(c), fits=len(group), parameters=group[0]["parameters"],
                           failures=sum(r["status"] != "ok" for r in group),
                           mean_kl=float(np.mean([r["kl"] for r in group])),
                           mean_validation_nll=float(np.mean([r["validation_nll"] for r in group]))))
    return sorted(result, key=lambda r: (r["radius"], r["mean_validation_nll"], r["key"]))


def comparisons_and_stability(rows, configs, protocol):
    factor = len(protocol["radii"]) * 6
    seeds = protocol["confirmation_seeds"]
    lookup = {(r["key"], r["seed"]): r for r in rows}
    comparisons, stability, architectures = [], [], []
    for radius in protocol["radii"]:
        local = {c["objective"]: c for c in configs if c["radius"] == radius}
        values, valid = {}, {}
        for objective in protocol["objectives"]:
            group = [lookup[(key(local[objective]), s)] for s in sorted(seeds)]
            a = np.array([r["kl"] for r in group])
            values[objective] = a
            ok = np.isfinite(a).all() and all(r["status"] == "ok" for r in group)
            valid[objective] = bool(ok)
            stability.append(dict(radius=radius, objective=objective, parameters=group[0]["parameters"], valid=bool(ok),
                                  mean=float(np.mean(a)) if ok else None, median=float(np.median(a)) if ok else None,
                                  sd=float(np.std(a, ddof=1)) if ok else None,
                                  p90=float(np.quantile(a, .9)) if ok else None,
                                  p95=float(np.quantile(a, .95)) if ok else None, max=float(np.max(a)) if ok else None))
        passed = 0
        for objective in protocol["objectives"]:
            if objective == "parity":
                continue
            c = dict(first=key(local["parity"]), second=key(local[objective]), pairs=len(seeds), bonferroni_factor=factor)
            if not (valid["parity"] and valid[objective]):
                c.update(valid=False, mean_difference=None, nominal_95_ci=None, adjusted_95_ci=None,
                         reason="failed or nonfinite pair; no seeds omitted")
            else:
                d = values["parity"] - values[objective]
                mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
                half = float(t.ppf(.975, len(d)-1) * se)
                adjusted = float(t.ppf(1 - .05/(2*factor), len(d)-1) * se)
                c.update(valid=True, mean_difference=mean, nominal_95_ci=[mean-half, mean+half],
                         adjusted_95_ci=[mean-adjusted, mean+adjusted], first_wins=int((d < 0).sum()))
                passed += mean + adjusted < 0
            comparisons.append(dict(c, radius=radius, control=objective))
        best = min(values, key=lambda o: (values[o].mean(), o)) if all(valid.values()) else None
        architectures.append(dict(radius=radius, parameters=len(parameter_ladder.Circuit(protocol["n"], radius).edges),
                                  best_mean_objective=best, adjusted_parity_wins=int(passed), planned_controls=6,
                                  parity_beats_all_adjusted=bool(passed == 6)))
    return dict(comparisons=comparisons, stability=stability, architectures=architectures,
                bonferroni_factor=factor, passing_architectures=sum(a["parity_beats_all_adjusted"] for a in architectures))


def audit(runs, protocol_path):
    runs, protocol_path = Path(runs), Path(protocol_path)
    required = ["lock.json", "develop/metrics.csv", "develop/summary.json", "develop/selection.json",
                "confirm/metrics.csv", "confirm/summary.json", "confirm/seal.json"]
    require(all((runs / p).is_file() for p in required), "incomplete study")
    p = json.loads(protocol_path.read_text())
    lock = json.loads((runs / "lock.json").read_text())
    require(lock["protocol"] == p, "protocol differs from lock")
    for filename, expected in lock["fingerprint"]["source_hashes"].items():
        require(digest(ROOT / "src/iqp_repro" / filename) == expected, "source hash mismatch: " + filename)
    require(not set(p["development_seeds"]) & set(p["confirmation_seeds"]), "overlapping seed cohorts")
    dev_configs = [dict(radius=r, objective=o, sigma=p["sigma_by_radius"][str(r)], lr=lr)
                   for r, o, lr in itertools.product(p["radii"], p["objectives"], p["learning_rates"])]
    selection = json.loads((runs / "develop/selection.json").read_text())
    require(selection["lock_sha256"] == digest(runs / "lock.json"), "selection lock differs")
    require(selection["metrics_sha256"] == digest(runs / "develop/metrics.csv"), "selection evidence differs")
    dev_rows = read_rows(runs / "develop/metrics.csv")
    ranked = ranking(dev_rows, dev_configs, p["development_seeds"])
    selected = []
    for radius, objective in itertools.product(p["radii"], p["objectives"]):
        eligible = [r for r in ranked if r["radius"] == radius and r["objective"] == objective
                    and r["failures"] == 0 and np.isfinite(r["mean_validation_nll"])]
        require(bool(eligible), "no eligible development setting")
        best = min(eligible, key=lambda r: (r["mean_validation_nll"], r["key"]))
        selected.append({k: best[k] for k in ("radius", "objective", "sigma", "lr")})
    same(selection["configurations"], selected, "validation-only selection")
    seal = json.loads((runs / "confirm/seal.json").read_text())
    same({k: v for k, v in seal.items() if k != "frozen_at_utc"},
         dict(configurations=selected, seeds=p["confirmation_seeds"], lock_sha256=digest(runs / "lock.json"),
              selection_sha256=digest(runs / "develop/selection.json"), comparisons=len(p["radii"]) * 6), "confirmation seal")
    require(bool(seal.get("frozen_at_utc")), "missing freeze timestamp")
    size = 2**p["n"]
    support = np.array([x.bit_count() % 2 == 0 for x in range(size)])
    scores = np.array([max(map(len, format(x, f'0{p["n"]}b').split("1")[1:-1]), default=0) for x in range(size)])
    target = np.zeros(size)
    logits = p["beta"] * scores[support]
    target[support] = np.exp(logits - logits.max())
    target /= target.sum()
    inputs, cross_radius_hashes, inventory, stages = {}, {}, {}, {}
    errors = {name: 0. for name in ("probabilities", "kl", "validation_nll", "raw_loss", "initial_gradient_norm", "scaled_gradient_norm", "summary")}
    for stage, configs, seeds in (("develop", dev_configs, p["development_seeds"]),
                                 ("confirm", selected, p["confirmation_seeds"])):
        rows = dev_rows if stage == "develop" else read_rows(runs / "confirm/metrics.csv")
        ranked = ranking(rows, configs, seeds)
        cfg = {key(c): c for c in configs}
        for row in rows:
            c, seed = cfg[row["key"]], row["seed"]
            circuit = parameter_ladder.Circuit(p["n"], c["radius"])
            label = f'{row["key"]}_seed{seed}'
            require(row["status"] in ("ok", "failed"), "invalid fit status")
            suffix = ".npz" if row["status"] == "ok" else ".failure.json"
            path = runs / "checkpoints" / (label + suffix)
            require(path.name not in inventory, "duplicate checkpoint")
            inventory[path.name] = digest(path)
            spec = dict(config=c, seed=seed, protocol=p, fingerprint=lock["fingerprint"])
            if row["status"] == "failed":
                failure = json.loads(path.read_text())
                require(failure["specification"] == spec, "failure provenance differs")
                require(row["kl"] == row["validation_nll"] == np.inf, "failure omitted from metrics")
                continue
            with np.load(path, allow_pickle=False) as saved:
                require(json.loads(str(saved["specification"])) == spec, "checkpoint provenance differs")
                metrics = json.loads(str(saved["metrics"]))
                require(metrics.keys() == row.keys(), "metric fields differ")
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        same(float(row[name]), float(value), label + "/" + name)
                    else:
                        require(str(row[name]) == str(value), "CSV/checkpoint mismatch")
                require(row["parameters"] == len(circuit.edges), "wrong parameter count")
                cache_key = (seed, c["radius"])
                if cache_key not in inputs:
                    inputs[cache_key] = base.reconstruct_inputs(dict(p, sigma=c["sigma"]), circuit, seed, target)
                expected_inputs = inputs[cache_key]
                for name, field in (("samples", "sample_sha256"), ("validation", "validation_sha256"),
                                    ("initial", "initial_sha256"), ("masks", "mask_sha256")):
                    require(np.array_equal(saved[name], expected_inputs[name]), "deterministic " + name + " mismatch")
                    require(base.array_digest(expected_inputs[name]) == row[field], "input hash mismatch")
                shared = (row["sample_sha256"], row["validation_sha256"])
                require(seed not in cross_radius_hashes or cross_radius_hashes[seed] == shared, "unpaired architecture data")
                cross_radius_hashes[seed] = shared
                q = circuit.probabilities(saved["theta"])
                q /= q.sum()
                err = float(np.max(np.abs(q - saved["q"])))
                require(np.isfinite(q).all() and np.isfinite(saved["q"]).all() and err < 2e-12, "Born law differs")
                errors["probabilities"] = max(errors["probabilities"], err)
                require(np.max(q[~support]) < 1e-25, "support leakage")
                with np.errstate(divide="ignore"):
                    kl = float(np.sum(target[support] * (np.log(target[support]) - np.log(q[support]))))
                    nll = float(-np.mean(np.log(q[expected_inputs["validation"]])))
                same(row["kl"], kl, label + "/KL", errors, "kl")
                same(row["validation_nll"], nll, label + "/validation", errors, "validation_nll")
                empirical = np.bincount(expected_inputs["samples"], minlength=size) / p["m"]
                mask_ids = expected_inputs["masks"].astype(int) @ (1 << np.arange(p["n"]-1, -1, -1))
                weights = np.bincount(mask_ids, minlength=size) / p["k"]
                grad0 = objective_losses.loss_gradient(circuit, saved["initial"], empirical, c["objective"], weights)[1]
                reference = objective_losses.loss_gradient(circuit, saved["initial"], empirical, "parity", weights)[1]
                scale = np.linalg.norm(reference) / np.linalg.norm(grad0)
                for name, value in (("scale", scale), ("initial_gradient_norm", np.linalg.norm(grad0)),
                                    ("reference_gradient_norm", np.linalg.norm(reference)),
                                    ("scaled_initial_gradient_norm", np.linalg.norm(reference))):
                    same(float(row[name]), float(value), label + "/" + name, errors, "initial_gradient_norm")
                for name in ("loss_history", "scaled_gradient_history"):
                    require(saved[name].shape == (p["steps"]+1,) and np.isfinite(saved[name]).all(), "incomplete history")
                for index, theta in ((0, saved["initial"]), (-1, saved["theta"])):
                    law = circuit.probabilities(theta)
                    raw = base.scalar_loss(law, empirical, c["objective"], mask_ids)
                    same(float(saved["loss_history"][index]), raw, label + "/loss", errors, "raw_loss")
                    gradient = grad0 if index == 0 else objective_losses.loss_gradient(circuit, theta, empirical, c["objective"], weights)[1]
                    same(float(saved["scaled_gradient_history"][index]), float(np.linalg.norm(scale * gradient)),
                         label + "/gradient", errors, "scaled_gradient_norm")
        expected = dict(stage=stage, seeds=seeds, fits=len(rows), failures=sum(r["status"] != "ok" for r in rows), ranking=ranked)
        if stage == "confirm":
            expected.update(comparisons_and_stability(rows, configs, p))
        same(json.loads((runs / stage / "summary.json").read_text()), expected, "independent summary", errors)
        stages[stage] = dict(fits=len(rows), failed=expected["failures"], nonfinite_kl=sum(not np.isfinite(r["kl"]) for r in rows))
    actual = {p.name for p in (runs / "checkpoints").glob("*.npz")} | {p.name for p in (runs / "checkpoints").glob("*.failure.json")}
    require(actual == set(inventory), "checkpoint inventory differs")
    return dict(status="pass", stages=stages, checkpoints=len(inventory), unique_datasets=len(cross_radius_hashes),
                architecture_dataset_pairs=len(inputs), max_absolute_errors=errors,
                source_hashes=lock["fingerprint"]["source_hashes"], protocol_sha256=digest(protocol_path),
                evidence_sha256={name: digest(runs / name) for name in required},
                checkpoint_manifest_sha256=hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
                auditor_sha256=digest(__file__), audit_helper_sha256=digest(HELPER),
                limits="No retraining. Frozen tested circuit kernels reconstruct probabilities and gradients; scalar losses, RNG inputs, validation selection, intervals and descriptive stability are checked independently. Hash consistency does not prove historical freeze timing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols/objective-ladder.json")
    args = parser.parse_args()
    try:
        result = audit(args.runs, args.protocol)
    except (ValueError, KeyError, OSError, FloatingPointError) as error:
        print(f"Audit failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
