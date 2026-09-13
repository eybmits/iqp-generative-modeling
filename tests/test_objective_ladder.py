"""Architecture extension: numerical compatibility, pairing and frozen inference."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from iqp_repro import core, objective_comparison, objective_ladder as ladder, objective_losses, parameter_ladder, study


def protocol():
    p = json.loads((Path(__file__).parents[1] / "protocols/objective-ladder.json").read_text())
    p.update(m=30, k=16, steps=3, validation_samples=40, initial_seed_offset=10112,
             development_seeds=[111, 112], confirmation_seeds=[700001, 700002], learning_rates=[.02, .1])
    return p


@pytest.mark.parametrize("objective", ["parity", "mse", "nll", "spherical", "hellinger", "js", "tv"])
def test_radius_three_preserves_previous_training_bit_for_bit(objective):
    p = protocol()
    config = dict(radius=3, objective=objective, sigma=.75, lr=.1)
    new = ladder.train_configuration(config, 111, p)
    old = objective_comparison.train_configuration(dict(objective=objective, lr=.1), 111,
                                                   dict(p, architecture="ring3", sigma=.75))
    for name in ("theta", "q", "initial", "samples", "validation", "masks", "loss_history", "scaled_gradient_history"):
        np.testing.assert_array_equal(new[name], old[name])
    for name in ("kl", "validation_nll", "scale", "initial_gradient_norm", "reference_gradient_norm"):
        assert new["metrics"][name] == old["metrics"][name]


@pytest.mark.parametrize("radius,parameters", [(3, 36), (4, 48), (5, 60), (6, 66)])
def test_parity_matches_original_architecture_ladder_and_preserves_support(radius, parameters):
    p = protocol()
    c = dict(radius=radius, objective="parity", sigma=p["sigma_by_radius"][str(radius)], lr=.05)
    new = ladder.train_configuration(c, 112, p)
    old = parameter_ladder.train_configuration(c, 112, p)
    for name in ("theta", "q", "initial", "samples", "validation", "masks", "loss_history"):
        np.testing.assert_array_equal(new[name], old[name])
    assert new["metrics"]["parameters"] == parameters
    odd = core.bits_table(12).sum(axis=1) % 2 == 1
    assert np.all(new["q"][odd] == 0)


@pytest.mark.parametrize("radius", [4, 5, 6])
@pytest.mark.parametrize("objective", ["parity", "mse", "nll", "spherical", "hellinger", "js", "tv"])
def test_larger_circuit_gradients_against_central_differences(radius, objective):
    circuit = parameter_ladder.Circuit(12, radius)
    rng = np.random.default_rng(33)
    theta = rng.normal(0, .4, len(circuit.edges))
    target = core.target(12, .9)[0]
    empirical = core.empirical(rng.choice(len(target), 200, p=target), 12)
    masks = core.sample_masks(12, 1., 32, 44)
    weights = study.parity_weights(12, "parity", masks)
    analytic = objective_losses.loss_gradient(circuit, theta, empirical, objective, weights)[1]
    for index in (0, len(theta)//2, len(theta)-1):
        left, right = theta.copy(), theta.copy()
        left[index] -= 1e-6
        right[index] += 1e-6
        numerical = (objective_losses.loss_gradient(circuit, right, empirical, objective, weights)[0]
                     - objective_losses.loss_gradient(circuit, left, empirical, objective, weights)[0]) / 2e-6
        assert analytic[index] == pytest.approx(numerical, rel=5e-5, abs=2e-8)


def test_same_samples_across_radii_and_initials_within_radius():
    p = protocol()
    results = [ladder.train_configuration(dict(radius=r, objective=o,
               sigma=p["sigma_by_radius"][str(r)], lr=.1), 111, p)
               for r in [3, 4, 5, 6] for o in ["parity", "mse"]]
    assert len({r["metrics"]["sample_sha256"] for r in results}) == 1
    assert len({r["metrics"]["validation_sha256"] for r in results}) == 1
    for left, right in zip(results[::2], results[1::2]):
        assert left["metrics"]["initial_sha256"] == right["metrics"]["initial_sha256"]
        assert left["metrics"]["reference_gradient_norm"] == pytest.approx(right["metrics"]["scaled_initial_gradient_norm"])


def test_selection_is_separate_per_architecture_and_ignores_target_kl():
    p = protocol()
    ranked = [dict(c, key=ladder.configuration_key(c), failures=0,
                   mean_validation_nll=c["lr"] if c["radius"] % 2 else -c["lr"], mean_kl=-c["lr"])
              for c in ladder.configurations(p)]
    chosen = ladder.select_configurations(ranked, p)
    assert len(chosen) == 28
    assert all(c["lr"] == (.02 if c["radius"] % 2 else .1) for c in chosen)
    changed = deepcopy(ranked)
    for r in changed:
        r["mean_kl"] = -100 * r["mean_kl"]
    assert chosen == ladder.select_configurations(changed, p)


def test_joint_correction_and_stability_do_not_conflate_mean_with_spread():
    p = protocol()
    seeds = [101, 102, 103, 104]
    p["confirmation_seeds"] = seeds
    configs = [c for c in ladder.configurations(p) if c["lr"] == .1]
    rows = []
    for c in configs:
        for index, seed in enumerate(seeds):
            kl = [.1, .2, .3, .7][index] if c["objective"] == "parity" else .4
            rows.append(dict(c, key=ladder.configuration_key(c), seed=seed, status="ok", kl=kl,
                             parameters=len(parameter_ladder.Circuit(12, c["radius"]).edges),
                             sample_sha256=str(seed), validation_sha256=str(seed),
                             initial_sha256=str((seed, c["radius"])), mask_sha256=str((seed, c["radius"]))))
    result = ladder.confirmation_summary(rows, configs, p)
    assert result["bonferroni_factor"] == len(result["comparisons"]) == 24
    assert result["passing_architectures"] == 0
    assert all(r["best_mean_objective"] == "parity" for r in result["architectures"])
    assert all(c["adjusted_95_ci"][1] > 0 for c in result["comparisons"])
    parity = next(r for r in result["stability"] if r["objective"] == "parity")
    mse = next(r for r in result["stability"] if r["objective"] == "mse")
    assert parity["mean"] < mse["mean"]
    assert parity["sd"] > mse["sd"]
    assert parity["p95"] > mse["p95"]


def test_protocol_validates_shared_budget_and_disjoint_confirmatory_cohort():
    p = protocol()
    ladder.validate_protocol(p)
    for field, bad in [("confirmation_seeds", p["development_seeds"]),
                       ("radii", [3, 3]), ("learning_rates", [0.]), ("steps", 0)]:
        with pytest.raises(ValueError):
            ladder.validate_protocol(dict(p, **{field: bad}))


def test_retained_failures_and_stale_checkpoint_guard(tmp_path, monkeypatch):
    p = protocol()
    c = ladder.configurations(p)[0]
    def fail(*args):
        raise FloatingPointError("test failure")
    monkeypatch.setattr(ladder, "train_configuration", fail)
    row = ladder.run_one(c, 111, p, tmp_path, {"version": 1})
    assert row["status"] == "failed" and row["kl"] == "Infinity"
    assert ladder.run_one(c, 111, p, tmp_path, {"version": 1}) == row
    with pytest.raises(ValueError, match="stale failure"):
        ladder.run_one(c, 111, p, tmp_path, {"version": 2})
