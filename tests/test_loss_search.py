"""Independent architecture, original-trajectory, and development-boundary checks."""
import copy
import itertools
import json

import numpy as np
import pytest

from iqp_repro import core, loss_search


def small_protocol():
    protocol = loss_search.development_protocol()
    protocol.update(n=6, m=30, k=32, steps=8, validation_samples=50)
    return protocol


@pytest.mark.parametrize("objective", ["parity", "mse"])
def test_original_ring_training_trajectory_and_shared_validation(objective):
    protocol = small_protocol()
    config = dict(beta=.9, architecture="ring", objective=objective, sigma=1., lr=.05)
    result = loss_search.train_configuration(config, 111, protocol)
    empirical = core.empirical(result["samples"], protocol["n"])
    reference = core.train_iqp(protocol["n"], empirical, result["masks"],
                               steps=protocol["steps"], lr=.05, seed_init=111+10000+7*protocol["k"],
                               loss=objective, mse_domain="support")
    np.testing.assert_allclose(result["theta"], reference["theta"], atol=2e-13, rtol=0)
    np.testing.assert_allclose(result["q"], reference["q"], atol=2e-14, rtol=0)
    np.testing.assert_allclose(result["history"], reference["loss_history"], atol=2e-14, rtol=0)
    expected_validation = np.random.default_rng(111+50000).choice(len(result["p"]), 50, p=result["p"])
    np.testing.assert_array_equal(result["validation"], expected_validation)
    assert result["metrics"]["validation_nll"] == pytest.approx(-np.mean(result["logq"][expected_validation]))


def explicit_ring3(n, theta):
    bits = np.array(list(itertools.product((0, 1), repeat=n)))
    signs = 1 - 2*bits
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if min(j-i, n-(j-i)) <= 3]
    hadamard = np.array([[1., 1.], [1., -1.]])/np.sqrt(2)
    full_h = np.array([[1.]])
    for _ in range(n):
        full_h = np.kron(full_h, hadamard)
    phase = np.exp(-.5j * sum(angle*signs[:, i]*signs[:, j] for angle, (i, j) in zip(theta, edges)))
    return np.abs(full_h @ (phase/np.sqrt(2**n)))**2


@pytest.mark.parametrize("objective", ["parity", "mse", "scaled-mse"])
def test_ring3_support_and_finite_difference_against_explicit_gates(objective):
    n = 8
    bits = np.array(list(itertools.product((0, 1), repeat=n)))
    support = bits.sum(axis=1) % 2 == 0
    rng = np.random.default_rng(78)
    empirical = np.zeros(2**n)
    empirical[support] = rng.random(support.sum())
    empirical /= empirical.sum()
    circuit = loss_search.Circuit(n, "ring3")
    theta = .3*rng.normal(size=len(circuit.edges))
    masks = bits[[1, 2, 3, 3, 7, 31, 65, 90]]
    characters = 1 - 2 * ((masks @ bits.T) % 2)
    weights = np.bincount(core.mask_indices(masks), minlength=2**n)/len(masks) if objective == "parity" else None

    def oracle(angles):
        difference = explicit_ring3(n, angles) - empirical
        if objective == "parity":
            return float(np.mean((characters @ difference)**2))
        return float(np.sum(difference**2)/(2**(n-1) if objective == "mse" else 1))

    q = circuit.probabilities(theta)
    np.testing.assert_allclose(q, explicit_ring3(n, theta), atol=2e-15)
    assert q.sum() == pytest.approx(1., abs=2e-15)
    assert q[~support].sum() < 1e-28
    assert len(loss_search.Circuit(12, "ring3").edges) == 36
    value, gradient = circuit.loss_gradient(theta, empirical, objective, weights)
    assert value == pytest.approx(oracle(theta), abs=2e-15)
    step = 1e-6
    numerical = np.array([(oracle(theta+step*direction)-oracle(theta-step*direction))/(2*step)
                          for direction in np.eye(len(theta))])
    np.testing.assert_allclose(gradient, numerical, atol=2e-9, rtol=2e-6)


def test_bounded_development_grid_and_shared_inputs():
    protocol = loss_search.development_protocol()
    configs = loss_search.configurations(protocol)
    assert len(configs)*len(protocol["development_seeds"]) == 2880
    assert {config["beta"] for config in configs} == {.9, 1.2, 1.5, 1.8}
    assert protocol["development_seeds"] == list(range(111, 121))
    short = small_protocol()
    results = [loss_search.train_configuration(dict(beta=.9, architecture="ring3", objective=objective,
                                                   sigma=sigma, lr=.02), 111, short)
               for objective, sigma in [("parity", .75), ("parity", 1.5), ("mse", 1.), ("scaled-mse", 1.)]]
    for result in results[1:]:
        for name in ["samples", "validation", "initial"]:
            np.testing.assert_array_equal(result[name], results[0][name])


def test_provenance_resume_and_retained_failures(tmp_path, monkeypatch):
    protocol = small_protocol()
    config = dict(beta=.9, architecture="ring3", objective="mse", sigma=1., lr=.02)
    provenance = {"test": "frozen"}
    first = loss_search.run_one(config, 111, protocol, tmp_path, provenance)
    assert loss_search.run_one(config, 111, protocol, tmp_path, provenance) == first
    changed = copy.deepcopy(protocol)
    changed["steps"] += 1
    with pytest.raises(ValueError, match="provenance"):
        loss_search.run_one(config, 111, changed, tmp_path, provenance)
    def fail(*args):
        raise FloatingPointError("deliberate failure")
    monkeypatch.setattr(loss_search, "train_configuration", fail)
    failed = loss_search.run_one(config, 112, protocol, tmp_path, provenance)
    assert failed["status"] == "failed" and failed["validation_nll"] == "Infinity"
    assert len(list(tmp_path.glob("*.failure.json"))) == 1
    assert loss_search.run_one(config, 112, protocol, tmp_path, provenance) == failed


def test_selection_uses_nll_not_target_kl():
    protocol = loss_search.development_protocol()
    rows = []
    for config in loss_search.configurations(protocol):
        for seed in protocol["development_seeds"]:
            rows.append(dict(config, seed=seed, key=loss_search.configuration_key(config),
                             family="parity" if config["objective"] == "parity" else "mse",
                             parameters=len(loss_search.Circuit(12, config["architecture"]).edges),
                             status="ok", validation_nll=config["lr"], kl=-config["lr"], recovery_1000=.4))
    summary = loss_search.summarize(rows, protocol)
    for selection in summary["selections_overall"]:
        expected = .02 if selection["family"] == "parity" else .005
        assert selection["selected"]["lr"] == expected
