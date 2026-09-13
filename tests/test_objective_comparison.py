"""Independent checks of matching, selection, and paired inference."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t

from iqp_repro import core, loss_search, objective_comparison as comparison


def small_protocol():
    return dict(n=6, beta=.9, architecture="ring3", m=30, k=20, sigma=.75,
                steps=6, validation_samples=40, sample_seed_offset=7,
                mask_seed_offset=222, initial_seed_offset=10000 + 7 * 20,
                validation_seed_offset=50000,
                optimizer=dict(beta1=.9, beta2=.99, epsilon=1e-8))


def test_parity_trajectory_and_rngs_match_existing_trainer_exactly():
    protocol = small_protocol()
    config = dict(objective="parity", lr=.05)
    new = comparison.train_configuration(config, 111, protocol)
    old = loss_search.train_configuration(
        dict(config, beta=.9, architecture="ring3", sigma=.75), 111, protocol)
    for field in ("q", "theta", "initial", "samples", "validation", "masks", "loss_history"):
        np.testing.assert_array_equal(new[field], old[field])
    np.testing.assert_array_equal(new["scaled_gradient_history"], old["gradient_history"])
    assert new["metrics"]["scale"] == 1.
    assert new["metrics"]["kl"] == old["metrics"]["kl"]
    assert new["metrics"]["validation_nll"] == old["metrics"]["validation_nll"]


def test_scale_matched_mse_and_summed_brier_have_identical_trajectories():
    protocol = small_protocol()
    mse = comparison.train_configuration(dict(objective="mse", lr=.05), 112, protocol)
    summed = comparison.train_configuration(dict(objective="scaled-mse", lr=.05), 112, protocol)
    for field in ("q", "theta", "initial", "samples", "validation", "masks", "scaled_gradient_history"):
        np.testing.assert_array_equal(mse[field], summed[field])
    np.testing.assert_array_equal(summed["loss_history"], 2**(protocol["n"] - 1) * mse["loss_history"])
    assert mse["metrics"]["scale"] == 2**(protocol["n"] - 1) * summed["metrics"]["scale"]


def test_one_initial_scalar_is_held_fixed_for_every_optimizer_update(monkeypatch):
    protocol = small_protocol()
    raw_gradients, supplied_gradients = [], []
    original_loss = comparison.objective_losses.loss_gradient
    original_adam = core.Adam

    def record_loss(*args, **kwargs):
        value, gradient = original_loss(*args, **kwargs)
        raw_gradients.append(gradient.copy())
        return value, gradient

    class RecordingAdam(original_adam):
        def update(self, theta, gradient):
            supplied_gradients.append(gradient.copy())
            return super().update(theta, gradient)

    monkeypatch.setattr(comparison.objective_losses, "loss_gradient", record_loss)
    monkeypatch.setattr(core, "Adam", RecordingAdam)
    result = comparison.train_configuration(dict(objective="mse", lr=.05), 113, protocol)
    scale = result["metrics"]["scale"]
    assert len(raw_gradients) == protocol["steps"] + 2  # Initial calibration plus all iterates.
    assert len(supplied_gradients) == protocol["steps"]
    for unscaled, supplied in zip(raw_gradients[1:-1], supplied_gradients):
        np.testing.assert_array_equal(supplied, scale * unscaled)
    expected = [np.linalg.norm(scale * gradient) for gradient in raw_gradients[1:]]
    np.testing.assert_array_equal(result["scaled_gradient_history"], expected)
    assert result["metrics"]["scaled_initial_gradient_norm"] == pytest.approx(
        result["metrics"]["reference_gradient_norm"], rel=2e-15)
    assert not np.isclose(expected[-1], expected[0], rtol=.05)


def candidate(objective, lr, validation, kl, failures=0):
    config = dict(objective=objective, lr=lr)
    return dict(config, key=comparison.configuration_key(config), fits=10,
                failures=failures, mean_validation_nll=validation, mean_kl=kl)


def test_selection_uses_validation_nll_instead_of_exact_kl():
    rows = [candidate("parity", .1, 4., .1), candidate("parity", .2, 3., .9),
            candidate("mse", .1, 2., .8), candidate("mse", .2, 4., .2)]
    selected = comparison.select_configurations(rows, ["parity", "mse"])
    assert selected == [dict(objective="parity", lr=.2), dict(objective="mse", lr=.1)]
    for row in rows:
        row["mean_kl"] = -1000 * row["mean_kl"]
    assert comparison.select_configurations(rows, ["parity", "mse"]) == selected


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_selection_skips_nonfinite_candidates_even_if_first(invalid):
    rows = [candidate("mse", .005, invalid, .001), candidate("mse", .1, 3., .5)]
    assert comparison.select_configurations(rows, ["mse"]) == [dict(objective="mse", lr=.1)]


def test_selection_skips_failures_and_breaks_ties_by_configuration_key():
    rows = [candidate("mse", .005, -100., .001, failures=1),
            candidate("mse", .2, 3., .1), candidate("mse", .1, 3., .9)]
    assert comparison.select_configurations(rows, ["mse"]) == [dict(objective="mse", lr=.1)]
    with pytest.raises(ValueError):
        comparison.select_configurations(rows[:1], ["mse"])


def paired_rows():
    rows = []
    for objective, values in [("parity", [.3, .5, .7, .2]), ("mse", [.4, .7, .6, .6])]:
        config = dict(objective=objective, lr=.1)
        for seed, kl in enumerate(values, 1):
            rows.append(dict(config, key=comparison.configuration_key(config), seed=seed,
                             status="ok", kl=kl, sample_sha256=f"sample-{seed}",
                             initial_sha256=f"initial-{seed}", mask_sha256=f"masks-{seed}",
                             validation_sha256=f"validation-{seed}"))
    return rows


def compare(rows):
    return comparison.paired_comparison(rows, "parity_lr0.1", "mse_lr0.1", [1, 2, 3, 4], 11)


def test_paired_intervals_use_within_seed_differences_and_all_planned_comparisons():
    result = compare(paired_rows())
    difference = np.array([-.1, -.2, .1, -.4])
    expected_mean = difference.mean()
    se = difference.std(ddof=1) / 2
    assert result["valid"]
    assert result["pairs"] == 4
    assert result["first_wins"] == 3
    assert result["mean_difference"] == pytest.approx(expected_mean)
    for name, alpha in [("nominal_95_ci", .05), ("adjusted_95_ci", .05 / 11)]:
        half = t.ppf(1 - alpha / 2, 3) * se
        np.testing.assert_allclose(result[name], [expected_mean - half, expected_mean + half], atol=1e-14)


@pytest.mark.parametrize("metric", [np.nan, np.inf, -np.inf, "Infinity"])
def test_nonfinite_pair_invalidates_inference_without_omitting_seed(metric):
    rows = paired_rows()
    rows[0]["kl"] = metric
    result = compare(rows)
    assert not result["valid"]
    assert result["pairs"] == 4
    assert result["nominal_95_ci"] is None
    assert result["adjusted_95_ci"] is None


def test_failed_pair_invalidates_inference_even_with_finite_metric():
    rows = paired_rows()
    rows[0]["status"] = "failed"
    result = compare(rows)
    assert not result["valid"]
    assert result["pairs"] == 4


def test_pairing_rejects_missing_duplicate_or_wrong_seed():
    rows = paired_rows()
    for altered in (rows[1:], rows + [deepcopy(rows[0])],
                    [dict(rows[0], seed=999)] + rows[1:]):
        with pytest.raises(ValueError):
            compare(altered)


@pytest.mark.parametrize("field", ["sample_sha256", "initial_sha256", "mask_sha256", "validation_sha256"])
def test_pairing_checks_every_shared_random_input(field):
    rows = paired_rows()
    rows[0][field] = "a different random draw"
    with pytest.raises(ValueError):
        compare(rows)


@pytest.mark.parametrize("missing", ["", "unavailable", None])
def test_pairing_requires_present_hashes_even_when_equal(missing):
    rows = paired_rows()
    rows[0]["sample_sha256"] = rows[4]["sample_sha256"] = missing
    with pytest.raises(ValueError):
        compare(rows)


def test_protocol_preserves_historical_rngs_and_disjoint_confirmation_streams():
    root = Path(__file__).parents[1]
    protocol = json.loads((root / "protocols/objective-comparison.json").read_text())
    assert protocol["initial_seed_offset"] == 10000 + 7 * protocol["k"]
    assert protocol["sample_seed_offset"] == 7
    assert protocol["mask_seed_offset"] == 222
    assert protocol["validation_seed_offset"] == 50000
    new = set(protocol["confirmation_seeds"])
    prior = set(protocol["development_seeds"]) | set(protocol["historical_seeds"])
    for name in ("loss-advantage-confirmation.json", "parameter-ladder-confirmation.json"):
        prior.update(json.loads((root / "protocols" / name).read_text())["confirmation_seeds"])
    offsets = [0, protocol["sample_seed_offset"], protocol["mask_seed_offset"],
               protocol["initial_seed_offset"], protocol["validation_seed_offset"]]
    new_streams = [{seed + offset for seed in new} for offset in offsets[1:]]
    prior_streams = {seed + offset for seed in prior for offset in offsets}
    assert not new.intersection(prior_streams)
    for index, stream in enumerate(new_streams):
        assert not stream.intersection(prior_streams)
        for other in new_streams[index + 1:]:
            assert not stream.intersection(other)
