#!/usr/bin/env python3
"""Read-only audit of all completed objective-comparison cohorts; never retrain.

python scripts/audit_objective_comparison.py --runs runs/objective-comparison \
    --out results/objective-comparison/validation.json

The audit reconstructs inputs and scalar objectives independently, uses the
tested circuit kernels to check endpoint gradients, and independently derives
validation selection and paired intervals. It refuses incomplete studies and
writes its compact JSON report only after all checks pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

from iqp_repro import loss_search, objective_losses


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = {"core.py", "study.py", "loss_search.py", "objective_losses.py", "objective_comparison.py"}
STAGES = ("historical", "develop", "confirm")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def key(config):
    return f'{config["objective"]}_lr{float(config["lr"]):g}'


def walsh(values):
    transformed = np.array(values, copy=True)
    width = 1
    while width < transformed.size:
        blocks = transformed.reshape(-1, 2 * width)
        left, right = blocks[:, :width].copy(), blocks[:, width:].copy()
        blocks[:, :width], blocks[:, width:] = left + right, left - right
        width *= 2
    return transformed


def scalar_loss(q, p, objective, mask_ids):
    """Independent scalar definitions, with no probability clipping."""
    if objective == "parity":
        return float(np.mean(walsh(q - p)[mask_ids]**2))
    if objective in ("mse", "scaled-mse"):
        return float(np.sum((q - p)**2) / (len(q)//2 if objective == "mse" else 1))
    if objective == "spherical":
        return float(1 - np.sum(p*q) / np.sqrt(np.sum(q*q)))
    if objective == "hellinger":
        return float(np.sum((np.sqrt(p) - np.sqrt(q))**2) / 2)
    if objective == "tv":
        return float(np.sum(np.abs(q - p)) / 2)
    p_positive, q_positive = p > 0, q > 0
    with np.errstate(divide="ignore"):
        if objective == "nll":
            return float(-np.sum(p[p_positive] * np.log(q[p_positive])))
        require(objective == "js", "unsupported objective in scalar audit")
        midpoint = (p + q) / 2
        return float((np.sum(p[p_positive] * np.log(p[p_positive] / midpoint[p_positive]))
                      + np.sum(q[q_positive] * np.log(q[q_positive] / midpoint[q_positive]))) / 2)


def same(actual, expected, label, errors=None, bucket="summary"):
    """Recursive exact structure comparison with explicit numeric tolerances."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label + ": differing fields")
        for name, value in expected.items():
            same(actual[name], value, label + "/" + name, errors, bucket)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ": differing length")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{label}/{index}", errors, bucket)
    elif isinstance(expected, (float, np.floating)):
        actual = float(actual)
        require(bool(np.isclose(actual, expected, rtol=1e-10, atol=2e-12, equal_nan=False)), label + ": numeric mismatch")
        if errors is not None and np.isfinite(actual) and np.isfinite(expected):
            errors[bucket] = max(errors.get(bucket, 0.), abs(actual - expected))
    else:
        require(actual == expected, label + ": mismatch")


def reconstruct_inputs(protocol, circuit, seed, target):
    n = protocol["n"]
    samples = np.random.default_rng(seed + protocol["sample_seed_offset"]).choice(
        circuit.size, protocol["m"], p=target)
    validation = np.random.default_rng(seed + protocol["validation_seed_offset"]).choice(
        circuit.size, protocol["validation_samples"], p=target)
    initial = .01 * np.random.default_rng(seed + protocol["initial_seed_offset"]).standard_normal(len(circuit.edges))
    rng = np.random.default_rng(seed + protocol["mask_seed_offset"])
    probability = -.5 * np.expm1(-1 / (2 * protocol["sigma"]**2))
    masks = rng.binomial(1, probability, (protocol["k"], n)).astype(np.int8)
    missing = np.flatnonzero(masks.sum(axis=1) == 0)
    while len(missing):
        masks[missing] = rng.binomial(1, probability, (len(missing), n))
        missing = np.flatnonzero(masks.sum(axis=1) == 0)
    return dict(samples=samples, validation=validation, initial=initial, masks=masks)


def independent_ranking(rows, configs, seeds):
    ranking = []
    expected = {(key(config), seed) for config in configs for seed in seeds}
    actual = {(row["key"], row["seed"]) for row in rows}
    require(len(actual) == len(rows) == len(expected) and actual == expected, "incomplete or duplicate cohort")
    for config in configs:
        group = [row for row in rows if row["key"] == key(config)]
        ranking.append(dict(config, key=key(config), fits=len(group),
                            failures=sum(row["status"] != "ok" for row in group),
                            mean_kl=float(np.mean([row["kl"] for row in group])),
                            mean_validation_nll=float(np.mean([row["validation_nll"] for row in group]))))
    return sorted(ranking, key=lambda row: (row["mean_validation_nll"], row["key"]))


def independent_comparisons(rows, configs, seeds, objectives, historical=False):
    selected = {config["objective"]: key(config) for config in configs}
    pairs = [(selected["parity"], selected[objective]) for objective in objectives if objective != "parity"]
    pairs += [(selected["mse"], selected[objective]) for objective in objectives if objective not in ("parity", "mse")]
    if historical:
        pairs += [(selected["parity"], "nll_lr0.1"), (selected["mse"], "nll_lr0.1")]
    indexed = {(row["key"], row["seed"]): row for row in rows}
    results = []
    for first, second in pairs:
        paired = [(indexed[first, seed], indexed[second, seed]) for seed in seeds]
        common = dict(first=first, second=second, pairs=len(seeds), bonferroni_factor=len(pairs))
        if any(a["status"] != "ok" or b["status"] != "ok" or
               not np.isfinite(a["kl"]) or not np.isfinite(b["kl"]) for a, b in paired):
            results.append(dict(common, valid=False, mean_difference=None, nominal_95_ci=None,
                                adjusted_95_ci=None, reason="failed or nonfinite pair; no seeds omitted"))
            continue
        differences = np.array([a["kl"] - b["kl"] for a, b in paired])
        mean = float(np.mean(differences))
        standard_error = np.sqrt(np.sum((differences - mean)**2) / (len(seeds) * (len(seeds) - 1)))
        def interval(alpha):
            half_width = float(t.ppf(1 - alpha / 2, len(seeds) - 1) * standard_error)
            return [mean - half_width, mean + half_width]
        results.append(dict(common, valid=True, mean_difference=mean,
                            nominal_95_ci=interval(.05), adjusted_95_ci=interval(.05 / len(pairs)),
                            first_wins=int(np.count_nonzero(differences < 0))))
    return results


def audit(runs, protocol_path=ROOT / "protocols/objective-comparison.json"):
    runs, protocol_path = Path(runs), Path(protocol_path)
    required = ["lock.json", "develop/selection.json", "confirm/seal.json"]
    required += [f"{stage}/{name}" for stage in STAGES for name in ("metrics.csv", "summary.json")]
    missing = [name for name in required if not (runs / name).is_file()]
    require(not missing, "incomplete study; missing required cohort artifacts: " + ", ".join(missing))
    lock, protocol = read_json(runs / "lock.json"), read_json(protocol_path)
    require(lock["protocol"] == protocol, "protocol differs from frozen lock")
    source_hashes = lock["fingerprint"]["source_hashes"]
    require(set(source_hashes) == SOURCE_NAMES, "incomplete frozen source hash map")
    for name, expected in source_hashes.items():
        require(digest(ROOT / "src/iqp_repro" / name) == expected, "source hash mismatch: " + name)
    seed_lists = [protocol[name] for name in ("historical_seeds", "development_seeds", "confirmation_seeds")]
    require(all(len(seeds) >= 2 and len(seeds) == len(set(seeds)) for seeds in seed_lists), "invalid seed cohorts")
    require(sum(map(len, seed_lists)) == len(set(itertools.chain.from_iterable(seed_lists))), "cohorts overlap")
    selection = read_json(runs / "develop/selection.json")
    require(selection["lock_sha256"] == digest(runs / "lock.json"), "selection lock hash mismatch")
    require(selection["metrics_sha256"] == digest(runs / "develop/metrics.csv"), "selection evidence hash mismatch")
    seal = read_json(runs / "confirm/seal.json")
    expected_seal = dict(configurations=selection["configurations"], seeds=protocol["confirmation_seeds"],
                         lock_sha256=digest(runs / "lock.json"),
                         selection_sha256=digest(runs / "develop/selection.json"), comparisons=11)
    same({name: value for name, value in seal.items() if name != "frozen_at_utc"}, expected_seal, "confirmation seal")
    require(bool(seal.get("frozen_at_utc")), "missing confirmation freeze timestamp")
    configs = dict(historical=protocol["historical_configurations"],
                   develop=[dict(objective=objective, lr=lr) for objective, lr in
                            itertools.product(protocol["objectives"], protocol["learning_rates"])],
                   confirm=selection["configurations"])
    require(len(configs["confirm"]) == len(protocol["objectives"]) and
            {c["objective"] for c in configs["confirm"]} == set(protocol["objectives"]),
            "confirmation requires one configuration per objective")
    circuit = loss_search.Circuit(protocol["n"], protocol["architecture"])
    support = np.array([state.bit_count() % 2 == 0 for state in range(circuit.size)])
    scores = np.array([max(map(len, format(state, f'0{protocol["n"]}b').split("1")[1:-1]), default=0)
                       for state in range(circuit.size)])
    target = np.zeros(circuit.size)
    logits = protocol["beta"] * scores[support]
    target[support] = np.exp(logits - logits.max())
    target /= target.sum()
    errors = {name: 0. for name in ("probabilities", "kl", "validation_nll", "raw_loss", "scaled_loss",
                                   "initial_gradient_norm", "scaled_gradient_norm", "summary")}
    inventory, stage_counts, input_cache = {}, {}, {}
    for stage, seeds in zip(STAGES, seed_lists):
        with (runs / stage / "metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["seed"], row["lr"] = int(row["seed"]), float(row["lr"])
            row["kl"], row["validation_nll"] = float(row["kl"]), float(row["validation_nll"])
        ranking = independent_ranking(rows, configs[stage], seeds)
        indexed_configs = {key(config): config for config in configs[stage]}
        pairing = {}
        for row in rows:
            config, seed = indexed_configs[row["key"]], row["seed"]
            label = f'{stage}/{row["key"]}_seed{seed}'
            require(row["objective"] == config["objective"] and row["lr"] == config["lr"], label + ": config mismatch")
            require(row["status"] in ("ok", "failed"), label + ": invalid status")
            stem = f'{row["key"]}_seed{seed}'
            path = runs / "checkpoints" / (stem + (".npz" if row["status"] == "ok" else ".failure.json"))
            require(path.is_file(), label + ": missing retained checkpoint")
            require(path.name not in inventory, label + ": duplicate retained checkpoint")
            inventory[path.name] = digest(path)
            specification = dict(config=config, seed=seed, protocol=protocol, fingerprint=lock["fingerprint"])
            if row["status"] == "failed":
                record = read_json(path)
                require(record["specification"] == specification, label + ": failure provenance mismatch")
                require(row["kl"] == row["validation_nll"] == np.inf, label + ": failure must be retained as infinity")
                require(record["metrics"].keys() == row.keys(), label + ": failure metric fields differ")
                for name, expected in record["metrics"].items():
                    if name in ("kl", "validation_nll") or isinstance(expected, (int, float)):
                        same(float(row[name]), float(expected), label + "/" + name)
                    else:
                        require(str(row[name]) == str(expected), label + "/" + name + ": failure CSV mismatch")
                continue
            with np.load(path, allow_pickle=False) as saved:
                require(json.loads(str(saved["specification"])) == specification, label + ": checkpoint provenance mismatch")
                metrics = json.loads(str(saved["metrics"]))
                require(metrics.keys() == row.keys(), label + ": CSV metric fields differ")
                for name, expected in metrics.items():
                    if isinstance(expected, (int, float)):
                        same(float(row[name]), float(expected), label + "/" + name)
                    else:
                        require(str(row[name]) == str(expected), label + "/" + name + ": CSV mismatch")
                require(int(row["parameters"]) == len(circuit.edges), label + ": parameter count")
                if seed not in input_cache:
                    input_cache[seed] = reconstruct_inputs(protocol, circuit, seed, target)
                inputs = input_cache[seed]
                input_hashes = []
                for name, field in (("samples", "sample_sha256"), ("validation", "validation_sha256"),
                                    ("initial", "initial_sha256"), ("masks", "mask_sha256")):
                    require(np.array_equal(saved[name], inputs[name]), label + ": deterministic " + name + " mismatch")
                    expected_hash = array_digest(inputs[name])
                    require(row[field] == expected_hash, label + ": input digest mismatch: " + field)
                    input_hashes.append(expected_hash)
                require(seed not in pairing or pairing[seed] == input_hashes, label + ": unpaired inputs")
                pairing[seed] = input_hashes
                q = circuit.probabilities(saved["theta"])
                q /= q.sum()
                require(np.isfinite(q).all() and np.isfinite(saved["q"]).all(), label + ": nonfinite Born law")
                q_error = float(np.max(np.abs(q - saved["q"])))
                require(q_error < 2e-12, label + ": Born law differs from saved theta")
                errors["probabilities"] = max(errors["probabilities"], q_error)
                require(np.all(saved["q"] >= 0) and abs(saved["q"].sum() - 1) < 2e-12,
                        label + ": invalid saved probability distribution")
                require(np.max(saved["q"][~support]) < 1e-25, label + ": odd-support leakage")
                with np.errstate(divide="ignore"):
                    kl = float(np.sum(target[support] * (np.log(target[support]) - np.log(q[support]))))
                    nll = float(-np.mean(np.log(q[inputs["validation"]])))
                same(row["kl"], kl, label + ": exact unfloored KL", errors, "kl")
                same(row["validation_nll"], nll, label + ": validation NLL", errors, "validation_nll")
                empirical = np.bincount(inputs["samples"], minlength=circuit.size) / protocol["m"]
                mask_ids = inputs["masks"].astype(int) @ (1 << np.arange(protocol["n"] - 1, -1, -1))
                weights = np.bincount(mask_ids, minlength=circuit.size) / protocol["k"]
                initial_value, initial_gradient = objective_losses.loss_gradient(
                    circuit, inputs["initial"], empirical, config["objective"], weights)
                reference_gradient = objective_losses.loss_gradient(circuit, inputs["initial"], empirical, "parity", weights)[1]
                initial_norm, reference_norm = float(np.linalg.norm(initial_gradient)), float(np.linalg.norm(reference_gradient))
                scale = reference_norm / initial_norm
                for name, expected in (("initial_gradient_norm", initial_norm), ("reference_gradient_norm", reference_norm),
                                       ("scale", scale), ("scaled_initial_gradient_norm", reference_norm)):
                    same(float(row[name]), expected, label + "/" + name, errors, "initial_gradient_norm")
                for name in ("loss_history", "scaled_gradient_history"):
                    require(saved[name].shape == (protocol["steps"] + 1,) and np.isfinite(saved[name]).all(),
                            label + ": incomplete or nonfinite " + name)
                for index, angles in ((0, inputs["initial"]), (-1, saved["theta"])):
                    law = circuit.probabilities(angles)
                    raw_value = scalar_loss(law, empirical, config["objective"], mask_ids)
                    value, gradient = ((initial_value, initial_gradient) if index == 0 else
                                       objective_losses.loss_gradient(circuit, angles, empirical, config["objective"], weights))
                    same(value, raw_value, label + ": independent objective", errors, "raw_loss")
                    same(float(saved["loss_history"][index]), raw_value, label + ": endpoint raw loss", errors, "raw_loss")
                    same(float(saved["loss_history"][index] * scale), raw_value * scale,
                         label + ": endpoint scaled loss", errors, "scaled_loss")
                    same(float(saved["scaled_gradient_history"][index]), float(np.linalg.norm(scale * gradient)),
                         label + ": endpoint scaled gradient", errors, "scaled_gradient_norm")
        summary = read_json(runs / stage / "summary.json")
        expected_summary = dict(stage=stage, seeds=seeds, fits=len(rows), failures=sum(row["status"] != "ok" for row in rows),
                                ranking=ranking, comparisons=[] if stage == "develop" else
                                independent_comparisons(rows, configs[stage], seeds, protocol["objectives"], stage == "historical"))
        same({name: value for name, value in summary.items() if name != "wall_seconds"}, expected_summary,
             stage + ": independently recomputed summary", errors)
        if stage == "develop":
            chosen = []
            for objective in protocol["objectives"]:
                eligible = [row for row in ranking if row["objective"] == objective and row["failures"] == 0
                            and np.isfinite(row["mean_validation_nll"])]
                require(bool(eligible), "no eligible development configuration: " + objective)
                winner = min(eligible, key=lambda row: (row["mean_validation_nll"], row["key"]))
                chosen.append({name: winner[name] for name in ("objective", "lr")})
            same(selection["configurations"], chosen, "validation-only development selection")
        stage_counts[stage] = dict(fits=len(rows), successful=sum(row["status"] == "ok" for row in rows),
                                   failed=sum(row["status"] != "ok" for row in rows),
                                   nonfinite_kl=sum(not np.isfinite(row["kl"]) for row in rows),
                                   paired_comparisons=len(expected_summary["comparisons"]))
    actual_files = {path.name for path in (runs / "checkpoints").glob("*.npz")}
    actual_files |= {path.name for path in (runs / "checkpoints").glob("*.failure.json")}
    require(actual_files == set(inventory), "checkpoint inventory differs from complete cohort grid")
    manifest_hash = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(status="pass", stages=stage_counts, checkpoint_files=len(inventory),
                unique_paired_seeds=len(input_cache), max_absolute_errors=errors,
                source_hashes=source_hashes, protocol_sha256=digest(protocol_path),
                locked_protocol_sha256=hashlib.sha256(json.dumps(lock["protocol"], sort_keys=True,
                                                               separators=(",", ":")).encode()).hexdigest(),
                evidence_sha256={name: digest(runs / name) for name in required},
                checkpoint_manifest_sha256=manifest_hash, auditor_sha256=digest(__file__),
                limits="No retraining. Endpoint circuit gradients use the tested frozen objective kernels; scalar losses, RNG reconstruction, selection and paired inference are independently computed. Hash consistency does not prove historical freeze timing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols/objective-comparison.json")
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
