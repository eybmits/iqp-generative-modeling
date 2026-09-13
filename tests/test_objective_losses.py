"""Independent formulas, numerical circuit gradients, and zero-support behavior."""
import numpy as np
import pytest

from iqp_repro import core, loss_search, objective_losses, study


def example():
    circuit = loss_search.Circuit(6, "ring3")
    support = np.array([index.bit_count() % 2 == 0 for index in range(circuit.size)])
    rng = np.random.default_rng(2806)
    theta = .4 * rng.normal(size=len(circuit.edges))
    empirical = np.zeros(circuit.size)
    empirical[support] = rng.uniform(.2, 1., support.sum())
    empirical /= empirical.sum()
    return circuit, theta, empirical, support


def independent_loss(q, p, objective):
    """Scalar formulas, deliberately separate from the analytic reverse pass."""
    if objective == "nll":
        return -sum(p[index] * np.log(q[index]) for index in np.flatnonzero(p))
    if objective == "spherical":
        return 1 - sum(p*q) / np.sqrt(sum(q*q))
    if objective == "hellinger":
        return sum((np.sqrt(p) - np.sqrt(q))**2) / 2
    if objective == "js":
        midpoint = (p + q) / 2
        return (sum(p[i] * np.log(p[i]/midpoint[i]) for i in np.flatnonzero(p))
                + sum(q[i] * np.log(q[i]/midpoint[i]) for i in np.flatnonzero(q))) / 2
    return sum(abs(q - p)) / 2


@pytest.mark.parametrize("objective", ["nll", "spherical", "hellinger", "js", "tv"])
@pytest.mark.parametrize("sparse_empirical", [False, True])
def test_analytic_gradient_and_value_match_independent_finite_differences(objective, sparse_empirical):
    circuit, theta, empirical, support = example()
    if sparse_empirical:
        empirical[np.flatnonzero(support)[::3]] = 0
        empirical /= empirical.sum()
    q = circuit.probabilities(theta)
    assert np.all(q[support] > 0)
    np.testing.assert_array_equal(q[~support], 0)
    value, gradient = objective_losses.loss_gradient(circuit, theta, empirical, objective)
    assert value == pytest.approx(independent_loss(q, empirical, objective), abs=2e-14)
    step = 1e-6
    numerical = np.array([
        (independent_loss(circuit.probabilities(theta + step*direction), empirical, objective)
         - independent_loss(circuit.probabilities(theta - step*direction), empirical, objective)) / (2*step)
        for direction in np.eye(len(theta))
    ])
    np.testing.assert_allclose(gradient, numerical, atol=2e-9, rtol=3e-6)


@pytest.mark.parametrize("objective", ["parity", "mse", "scaled-mse"])
def test_original_kernels_are_preserved_exactly(objective):
    circuit, theta, empirical, _ = example()
    masks = core.sample_masks(circuit.n, 1., 32, 101)
    weights = study.parity_weights(circuit.n, "parity", masks) if objective == "parity" else None
    value, gradient = objective_losses.loss_gradient(circuit, theta, empirical, objective, weights)
    original_value, original_gradient = circuit.loss_gradient(theta, empirical, objective, weights)
    assert value == original_value
    np.testing.assert_array_equal(gradient, original_gradient)


def test_nll_has_no_probability_floor_and_rejects_observed_exact_zero():
    circuit, _, _, _ = example()
    theta = np.zeros(len(circuit.edges))
    theta[0] = 1e-9
    empirical = np.zeros(circuit.size)
    empirical[circuit.indices[0]] = 1
    q = circuit.probabilities(theta)
    assert 0 < q[circuit.indices[0]] < 1e-16
    value, gradient = objective_losses.loss_gradient(circuit, theta, empirical, "nll")
    assert value == pytest.approx(-np.log(q[circuit.indices[0]]), abs=1e-13)
    assert gradient[0] == pytest.approx(-1 / np.tan(theta[0]/2), rel=1e-12)
    with pytest.raises(FloatingPointError, match="observed exact zero"):
        objective_losses.loss_gradient(circuit, np.zeros_like(theta), empirical, "nll")


@pytest.mark.parametrize("objective", ["nll", "spherical", "hellinger", "js", "tv"])
def test_unobserved_exact_zeros_are_finite_and_target_point_mass_is_stationary(objective):
    circuit, theta, _, _ = example()
    empirical = np.zeros(circuit.size)
    empirical[0] = 1
    value, gradient = objective_losses.loss_gradient(circuit, np.zeros_like(theta), empirical, objective)
    assert value == pytest.approx(0., abs=1e-14)
    np.testing.assert_allclose(gradient, 0, atol=1e-14)


def test_js_has_a_finite_circuit_derivative_at_an_observed_zero():
    circuit, theta, _, _ = example()
    empirical = np.zeros(circuit.size)
    empirical[circuit.indices[0]] = 1
    theta = np.zeros_like(theta)
    value, gradient = objective_losses.loss_gradient(circuit, theta, empirical, "js")
    assert value == pytest.approx(np.log(2))
    np.testing.assert_array_equal(gradient, 0)
    with pytest.raises(FloatingPointError, match="undefined"):
        objective_losses.loss_gradient(circuit, theta, empirical, "hellinger")


def test_tv_uses_zero_subgradient_at_equality():
    circuit, theta, _, _ = example()
    value, gradient = objective_losses.loss_gradient(circuit, theta, circuit.probabilities(theta), "tv")
    assert value == 0
    np.testing.assert_array_equal(gradient, 0)


def test_unknown_objective_is_rejected():
    circuit, theta, empirical, _ = example()
    with pytest.raises(ValueError, match="unknown objective"):
        objective_losses.loss_gradient(circuit, theta, empirical, "not-a-loss")
