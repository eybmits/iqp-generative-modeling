"""Independent graph, numerical, selection, and audit-boundary ladder tests."""
import copy
import itertools
import json

import numpy as np
import pytest
from scipy.linalg import hadamard
from scipy.stats import t

from iqp_repro import loss_search, parameter_ladder as ladder, study


def circular_edges(n, radius):
    return [(i, j) for i in range(n) for j in range(i+1, n)
            if min(j-i, n-j+i) <= radius]


def bits(n):
    return np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.int8)


def small_protocol():
    return dict(n=6, beta=.9, m=30, k=16, steps=3, validation_samples=40,
                radii=[1, 2, 3], parity_sigmas=[.75, 1.5],
                parity_learning_rates=[.02, .1], mse_objectives=["mse", "scaled-mse"],
                mse_learning_rates=[.02, .1], mse_sigma=1.,
                development_seeds=[700101, 700102],
                confirmation_seeds=[800101, 800102, 800103, 800104],
                optimizer=dict(beta1=.9, beta2=.99, epsilon=1e-8))


def development_rows(protocol):
    """NLL and diagnostic KL deliberately prefer opposite learning rates."""
    rows = []
    for config in ladder.configurations(protocol):
        for position, seed in enumerate(protocol["development_seeds"]):
            penalty = {"parity": .1, "mse": 1., "scaled-mse": .5}[config["objective"]]
            nll = penalty + config["lr"] + .01*config["sigma"] + .001*position
            rows.append(dict(config, beta=protocol["beta"], seed=seed,
                key=ladder.configuration_key(config), parameters=len(circular_edges(protocol["n"], config["radius"])),
                status="ok", validation_nll=nll, kl=10-nll,
                coverage_1000=.1, recovery_1000=.3,
                sample_sha256=f"data-{seed}", validation_sha256=f"validation-{seed}",
                initial_sha256=f"initial-{config['radius']}-{seed}", mask_sha256=f"mask-{config['sigma']}-{seed}",
                error=""))
    return rows


def freeze(protocol):
    summary = ladder.summarize_development(development_rows(protocol), protocol)
    return ladder.make_confirmation_protocol(summary, protocol, ladder.source_hashes(),
                                             {"development_metrics_sha256": "a"*64})


def confirmation_rows(protocol, winning_radii=()):
    """Four varying paired differences support independently calculable CIs."""
    rows = []
    for position, seed in enumerate(protocol["confirmation_seeds"]):
        for role, config in protocol["configurations"].items():
            parity = config["objective"] == "parity"
            value = 1. if parity else (2.+.02*position if config["radius"] in winning_radii else .5+.01*position)
            rows.append(dict(config, role=role, beta=protocol["protocol"]["beta"], seed=seed,
                             key=ladder.configuration_key(config),
                             parameters=len(circular_edges(protocol["protocol"]["n"], config["radius"])),
                             status="ok", kl=value, validation_nll=1000. if parity else 0.,
                             recovery_1000=.3, coverage_1000=.1,
                             sample_sha256=f"data-{seed}", validation_sha256=f"validation-{seed}",
                             initial_sha256=f"initial-{config['radius']}-{seed}",
                             mask_sha256=f"mask-{config['sigma']}-{seed}", error=""))
    return rows


def test_production_parameter_counts_nested_graphs_and_unique_edges():
    previous = set()
    for radius, count in enumerate([12, 24, 36, 48, 60, 66], 1):
        circuit = ladder.Circuit(12, radius)
        assert circuit.edges == circular_edges(12, radius)
        assert len(circuit.edges) == len(set(circuit.edges)) == count
        assert previous < set(circuit.edges)
        previous = set(circuit.edges)


@pytest.mark.parametrize("radius,reference", [(2, "ring"), (3, "ring3"), (6, "dense")])
def test_ladder_matches_frozen_circuit_architectures(radius, reference):
    original = loss_search.Circuit(12, reference) if reference == "ring3" else study.Circuit(12, reference)
    fresh = ladder.Circuit(12, radius)
    assert fresh.edges == original.edges
    theta = .4*np.random.default_rng(617).standard_normal(len(original.edges))
    np.testing.assert_allclose(fresh.probabilities(theta), original.probabilities(theta), atol=2e-15, rtol=0)


@pytest.mark.parametrize("objective", ["parity", "mse"])
def test_radius_four_born_law_support_and_gradient_from_explicit_gates(objective):
    # At n=10 radius four is distinct from dense, unlike a small saturated ring.
    n, radius = 10, 4
    words = bits(n)
    support = words.sum(axis=1)%2 == 0
    graph = circular_edges(n, radius)
    spins = 1-2*words
    eigenvalues = np.column_stack([spins[:, i]*spins[:, j] for i, j in graph])
    transform = hadamard(2**n).astype(float)/np.sqrt(2**n)
    rng = np.random.default_rng(912)
    theta = .3*rng.standard_normal(len(graph))
    empirical = np.zeros(2**n)
    empirical[support] = rng.random(support.sum())
    empirical /= empirical.sum()
    mask_ids = np.array([1, 2, 3, 3, 7, 15, 23, 31, 129, 513, 777])
    characters = 1-2*((words[mask_ids] @ words.T)%2)
    weights = np.bincount(mask_ids, minlength=2**n)/len(mask_ids) if objective == "parity" else None

    def explicit_law(angles):
        phases = np.exp(-.5j*(eigenvalues @ angles))/np.sqrt(2**n)
        return np.abs(transform @ phases)**2

    def loss(angles):
        difference = explicit_law(angles)-empirical
        if objective == "parity":
            return float(np.mean((characters @ difference)**2))
        return float(np.dot(difference, difference)/support.sum())

    circuit = ladder.Circuit(n, radius)
    q = circuit.probabilities(theta)
    np.testing.assert_allclose(q, explicit_law(theta), atol=2e-15, rtol=0)
    assert q.sum() == pytest.approx(1., abs=2e-15)
    assert q[~support].sum() < 1e-28
    value, gradient = circuit.loss_gradient(theta, empirical, objective, weights)
    assert value == pytest.approx(loss(theta), abs=2e-15)
    epsilon = 1e-6
    numerical = np.array([(loss(theta+epsilon*direction)-loss(theta-epsilon*direction))/(2*epsilon)
                          for direction in np.eye(len(theta))])
    np.testing.assert_allclose(gradient, numerical, atol=2e-9, rtol=3e-6)


def test_tiny_training_shares_inputs_and_initial_angles_between_losses():
    protocol = small_protocol()
    configs = [dict(radius=2, objective=objective, sigma=1., lr=.05)
               for objective in ["parity", "mse"]]
    results = [ladder.train_configuration(config, 700101, protocol) for config in configs]
    for field in ["samples", "validation", "initial", "masks", "p"]:
        np.testing.assert_array_equal(results[0][field], results[1][field])
    p = results[0]["p"]
    expected_data = np.random.default_rng(700101+7).choice(len(p), protocol["m"], p=p)
    expected_validation = np.random.default_rng(700101+50000).choice(len(p), protocol["validation_samples"], p=p)
    expected_initial = .01*np.random.default_rng(700101+10000+7*protocol["k"]).standard_normal(len(circular_edges(protocol["n"], 2)))
    np.testing.assert_array_equal(results[0]["samples"], expected_data)
    np.testing.assert_array_equal(results[0]["validation"], expected_validation)
    np.testing.assert_array_equal(results[0]["initial"], expected_initial)
    for result in results:
        assert len(result["loss_history"]) == protocol["steps"]+1
        assert result["metrics"]["validation_nll"] == pytest.approx(-np.mean(result["logq"][expected_validation]))
        assert result["q"].sum() == pytest.approx(1., abs=2e-15)


@pytest.mark.parametrize("alteration", ["missing", "duplicate", "wrong_seed"])
def test_development_rejects_incomplete_or_duplicate_configuration_seed_grids(alteration):
    protocol = small_protocol()
    rows = development_rows(protocol)
    if alteration == "missing":
        rows.pop()
    elif alteration == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    else:
        rows[-1]["seed"] = 999999
    with pytest.raises(ValueError):
        ladder.summarize_development(rows, protocol)


def test_development_selection_uses_validation_nll_and_retains_full_ranking():
    protocol = small_protocol()
    rows = development_rows(protocol)
    report = ladder.summarize_development(rows, protocol)
    assert report["fits"] == len(rows)
    assert len(report["ranking"]) == len(ladder.configurations(protocol))
    for selection in report["selections"]:
        assert selection["eligible"]
        assert selection["parity"]["lr"] == .02
        assert selection["parity"]["sigma"] == .75
        assert selection["mse_tuned"]["lr"] == .02
        assert selection["mse_tuned"]["objective"] == "scaled-mse"
        assert selection["mse_matched"]["objective"] == "mse"
        assert selection["mse_matched"]["lr"] == selection["parity"]["lr"]
        candidates = [row for row in report["ranking"] if row["radius"] == selection["radius"] and row["objective"] == "parity"]
        # Selecting by target KL would demonstrably pick another configuration.
        assert selection["parity"]["key"] != min(candidates, key=lambda row:row["mean_kl"])["key"]


@pytest.mark.parametrize("bad_outcome", ["failed", "Infinity", "NaN"])
def test_ineligible_family_is_retained_but_cannot_be_frozen_for_confirmation(bad_outcome):
    protocol = small_protocol()
    rows = development_rows(protocol)
    for row in rows:
        if row["radius"] == 1 and row["objective"] == "parity":
            if bad_outcome == "failed":
                row.update(status="failed", validation_nll=-999., error="synthetic failed fit")
            else:
                row["validation_nll"] = bad_outcome
    report = ladder.summarize_development(rows, protocol)
    assert report["fits"] == len(rows)
    assert len(report["ranking"]) == len(ladder.configurations(protocol))
    assert not next(value for value in report["selections"] if value["radius"] == 1)["eligible"]
    with pytest.raises(ValueError):
        ladder.make_confirmation_protocol(report, protocol, ladder.source_hashes(), {})


def test_one_failed_candidate_cannot_win_by_a_misleading_finite_nll():
    protocol = small_protocol()
    rows = development_rows(protocol)
    poisoned = rows[0]["key"]
    rows[0].update(status="failed", validation_nll=-999., error="synthetic failed fit")
    report = ladder.summarize_development(rows, protocol)
    assert report["failed_fits"] == 1
    assert len(report["ranking"]) == len(ladder.configurations(protocol))
    assert report["selections"][0]["parity"]["key"] != poisoned


@pytest.mark.parametrize("alteration", ["missing", "duplicate", "wrong_seed"])
def test_confirmation_rejects_incomplete_duplicate_or_unexpected_pairs(alteration):
    frozen = freeze(small_protocol())
    rows = confirmation_rows(frozen, winning_radii=[1, 2])
    if alteration == "missing":
        rows.pop()
    elif alteration == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    else:
        rows[-1]["seed"] = 999999
    with pytest.raises(ValueError):
        ladder.summarize_confirmation(rows, frozen)


@pytest.mark.parametrize("field", ["sample_sha256", "validation_sha256", "mask_sha256", "initial_sha256"])
def test_confirmation_requires_shared_observations_masks_and_within_radius_initialization(field):
    frozen = freeze(small_protocol())
    rows = confirmation_rows(frozen, winning_radii=[1, 2])
    rows[-1][field] = "mismatched"
    with pytest.raises(ValueError):
        ladder.summarize_confirmation(rows, frozen)


def six_radius_protocol():
    protocol = small_protocol()
    protocol.update(n=12, radii=[1, 2, 3, 4, 5, 6])
    return protocol


@pytest.mark.parametrize("number_wins,expected_majority", [(3, False), (4, True)])
def test_exact_four_of_six_majority_and_independent_paired_statistics(number_wins, expected_majority):
    frozen = freeze(six_radius_protocol())
    rows = confirmation_rows(frozen, winning_radii=range(1, number_wins+1))
    report = ladder.summarize_confirmation(rows, frozen)
    assert report["planned_comparisons"] == 12
    assert report["primary_majority"]["required_wins"] == 4
    assert report["primary_majority"]["total_count"] == 6
    assert report["primary_majority"]["confirmed_wins"] == number_wins
    assert report["primary_majority"]["confirmed_majority"] is expected_majority
    result = next(value for value in report["comparisons"] if value["radius"] == 1 and value["kind"] == "tuned")
    differences = np.array([-1., -1.02, -1.04, -1.06])
    mean = differences.mean()
    standard_error = differences.std(ddof=1)/np.sqrt(4)
    np.testing.assert_allclose(result["ci95"], mean + np.array([-1, 1])*t.ppf(.975, 3)*standard_error)
    np.testing.assert_allclose(result["adjusted_ci95"], mean + np.array([-1, 1])*t.ppf(1-.05/24, 3)*standard_error)
    assert result["se"] == pytest.approx(standard_error)
    assert result["mean_difference"] == pytest.approx(mean)
    assert result["relative_improvement"] == pytest.approx(1-1./2.03)
    assert result["wins"] == result["n"] == result["finite_pairs"] == 4
    assert result["confirmed_win"] and not result["confirmed_loss"]
    reversed_result = next(value for value in report["comparisons"]
                           if value["radius"] == number_wins+1 and value["kind"] == "tuned")
    assert reversed_result["confirmed_loss"] and not reversed_result["confirmed_win"]
    assert reversed_result["adjusted_ci95"][0] > 0 and reversed_result["wins"] == 0


def test_identical_mse_controls_keep_both_comparisons_and_bonferroni_factor():
    protocol = six_radius_protocol()
    rows = development_rows(protocol)
    for row in rows:
        if row["objective"] == "mse":
            row["validation_nll"] = row["lr"]
    summary = ladder.summarize_development(rows, protocol)
    frozen = ladder.make_confirmation_protocol(summary, protocol, ladder.source_hashes(), {})
    assert len(frozen["comparisons"]) == 12
    for radius in protocol["radii"]:
        pair = [value for value in frozen["comparisons"] if value["radius"] == radius]
        assert {value["kind"] for value in pair} == {"tuned", "matched"}
        assert frozen["configurations"][pair[0]["control"]] == frozen["configurations"][pair[1]["control"]]
    report = ladder.summarize_confirmation(confirmation_rows(frozen, winning_radii=protocol["radii"]), frozen)
    expected_half = t.ppf(1-.05/24, 3)*np.std([-1., -1.02, -1.04, -1.06], ddof=1)/2
    for comparison in report["comparisons"]:
        assert comparison["adjusted_ci95"][1]-comparison["mean_difference"] == pytest.approx(expected_half)


@pytest.mark.parametrize("bad_outcome", ["failed", "Infinity", "NaN"])
def test_failed_or_nonfinite_pair_stays_in_denominator_and_blocks_boundary_majority(bad_outcome):
    frozen = freeze(six_radius_protocol())
    rows = confirmation_rows(frozen, winning_radii=[1, 2, 3, 4])
    role = next(value["parity"] for value in frozen["comparisons"] if value["radius"] == 4)
    selected = next(row for row in rows if row["role"] == role)
    if bad_outcome == "failed":
        selected.update(status="failed", kl=.001, error="synthetic failed fit")
    else:
        selected["kl"] = bad_outcome
    report = ladder.summarize_confirmation(rows, frozen)
    assert report["primary_majority"]["confirmed_wins"] == 3
    assert not report["primary_majority"]["confirmed_majority"]
    for comparison in report["comparisons"]:
        if comparison["radius"] == 4:
            assert comparison["n"] == 4 and comparison["finite_pairs"] == 3
            assert comparison["ci95"] is None and comparison["adjusted_ci95"] is None
            assert not comparison["confirmed_win"] and not comparison["confirmed_loss"]


def test_checkpoint_resume_preserves_results_and_rejects_changed_scientific_inputs(tmp_path, monkeypatch):
    protocol = small_protocol()
    config = dict(radius=1, objective="mse", sigma=1., lr=.02)
    provenance = {"source": "frozen"}
    first = ladder.run_one(config, 700101, protocol, tmp_path, provenance)

    def unexpected_retraining(*args, **kwargs):
        raise AssertionError("a valid checkpoint must be reused")

    monkeypatch.setattr(ladder, "train_configuration", unexpected_retraining)
    assert ladder.run_one(config, 700101, protocol, tmp_path, provenance) == first
    changed = copy.deepcopy(protocol)
    changed["steps"] += 1
    with pytest.raises(ValueError):
        ladder.run_one(config, 700101, changed, tmp_path, provenance)
    with pytest.raises(ValueError):
        ladder.run_one(config, 700101, protocol, tmp_path, {"source": "changed"})


@pytest.mark.parametrize("family,bad_outcome", itertools.product(["parity", "control"], ["failed", "Infinity", "NaN"]))
def test_relative_improvement_is_undefined_for_any_failed_or_nonfinite_pair(family, bad_outcome):
    frozen = freeze(small_protocol())
    rows = confirmation_rows(frozen, winning_radii=[1, 2])
    comparison = next(item for item in frozen["comparisons"] if item["radius"] == 1 and item["kind"] == "tuned")
    row = next(item for item in rows if item["role"] == comparison[family])
    if bad_outcome == "failed":
        row.update(status="failed", error="synthetic failed fit with finite diagnostic KL")
    else:
        row["kl"] = bad_outcome
    summary = ladder.summarize_confirmation(rows, frozen)
    result = next(item for item in summary["comparisons"] if item["radius"] == 1 and item["kind"] == "tuned")
    assert result["relative_improvement"] is None
    assert result["n"] == 4 and result["finite_pairs"] == 3
    assert not result["confirmed_win"] and not result["confirmed_loss"]
    unaffected = next(item for item in summary["comparisons"] if item["radius"] == 2 and item["kind"] == "tuned")
    assert unaffected["relative_improvement"] == pytest.approx(1-1./2.03)


@pytest.mark.parametrize("control_kl", [0., -.1])
def test_relative_improvement_is_undefined_for_nonpositive_control_mean(control_kl):
    frozen = freeze(small_protocol())
    rows = confirmation_rows(frozen, winning_radii=[1, 2])
    comparison = next(item for item in frozen["comparisons"] if item["radius"] == 1 and item["kind"] == "tuned")
    for row in rows:
        if row["role"] == comparison["control"]:
            row["kl"] = control_kl
    result = next(item for item in ladder.summarize_confirmation(rows, frozen)["comparisons"]
                  if item["radius"] == 1 and item["kind"] == "tuned")
    assert result["relative_improvement"] is None
    assert result["n"] == result["finite_pairs"] == 4


def test_failure_checkpoint_is_retained_and_never_silently_retried(tmp_path, monkeypatch):
    protocol = small_protocol()
    config = dict(radius=1, objective="parity", sigma=.75, lr=.02)

    def fail(*args, **kwargs):
        raise FloatingPointError("deliberate numerical failure")

    monkeypatch.setattr(ladder, "train_configuration", fail)
    row = ladder.run_one(config, 700101, protocol, tmp_path, {"source":"frozen"})
    assert row["status"] == "failed" and float(row["kl"]) == float("inf")
    saved = list(tmp_path.glob("*.failure.json"))
    assert len(saved) == 1 and not list(tmp_path.glob("*.npz"))
    assert "deliberate numerical failure" in json.loads(saved[0].read_text())["metrics"]["error"]
    assert ladder.run_one(config, 700101, protocol, tmp_path, {"source":"frozen"}) == row
    with pytest.raises(ValueError):
        ladder.run_one(config, 700101, protocol, tmp_path, {"source":"changed"})


@pytest.mark.parametrize("alteration", ["source_hash", "seed_list", "missing_comparison"])
def test_bad_frozen_confirmation_is_rejected_before_scheduling_any_training(tmp_path, monkeypatch, alteration):
    frozen = freeze(small_protocol())
    if alteration == "source_hash":
        frozen["source_hashes"][next(iter(frozen["source_hashes"]))] = "0"*64
    elif alteration == "seed_list":
        frozen["confirmation_seeds"] = [999991, 999992]
    else:
        frozen["comparisons"].pop()

    def must_not_execute(*args, **kwargs):
        raise AssertionError("invalid frozen evidence must fail before generating data")

    monkeypatch.setattr(ladder, "_execute", must_not_execute)
    with pytest.raises(ValueError):
        ladder.run_confirmation(tmp_path, frozen, jobs=1)
