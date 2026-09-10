"""Independent numerical and study-boundary checks; no confirmatory training."""
import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from iqp_repro import core, study


def bit_table(n):
    return np.array(list(itertools.product((0, 1), repeat=n)), dtype=int)


def explicit_edges(n, architecture):
    if architecture == "dense":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    return sorted({tuple(sorted((i, (i + distance) % n)))
                   for i in range(n) for distance in (1, 2)
                   if i != (i + distance) % n})


def explicit_probabilities(n, architecture, theta):
    """Apply actual dense H and diagonal RZZ matrices to |0...0>."""
    H = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
    full_h = np.array([[1.]])
    for _ in range(n):
        full_h = np.kron(full_h, H)
    bits = bit_table(n)
    spins = 1 - 2 * bits
    phase = np.ones(2**n, dtype=complex)
    for angle, (i, j) in zip(theta, explicit_edges(n, architecture)):
        phase *= np.exp(-.5j * angle * spins[:, i] * spins[:, j])
    initial = np.zeros(2**n)
    initial[0] = 1
    amplitude = full_h @ (phase * (full_h @ initial))
    return np.abs(amplitude)**2


def example(n=6):
    rng = np.random.default_rng(271)
    bits = bit_table(n)
    even = bits.sum(axis=1) % 2 == 0
    empirical = np.zeros(2**n)
    empirical[even] = rng.random(even.sum())
    empirical /= empirical.sum()
    masks = bits[[1, 2, 3, 3, 5, 7, 12, 31, 2**n-1]]
    return bits, empirical, masks


def small_protocol(steps=1):
    return dict(n=6, beta=.9, m=1, sigma=1., k=32, steps=steps,
                validation_samples=100,
                optimizer=dict(beta1=.9, beta2=.99, epsilon=1e-8),
                architectures=["ring", "dense"],
                objectives=["parity", "expected-parity", "mse", "scaled-mse"],
                learning_rates=[.05], development_seeds=[10, 11],
                confirmation_seeds=[20, 21, 22])


@pytest.mark.parametrize("architecture", ["ring", "dense"])
def test_circuit_matches_explicit_gates_and_even_support(architecture):
    n = 6
    theta = np.random.default_rng(8).normal(size=len(explicit_edges(n, architecture)))
    q = study.Circuit(n, architecture).probabilities(theta)
    np.testing.assert_allclose(q, explicit_probabilities(n, architecture, theta), atol=2e-15)
    assert q.sum() == pytest.approx(1., abs=2e-15)
    assert q[bit_table(n).sum(axis=1) % 2 == 1].sum() < 1e-28
    assert len(study.Circuit(12, architecture).edges) == (24 if architecture == "ring" else 66)


@pytest.mark.parametrize("architecture", ["ring", "dense"])
@pytest.mark.parametrize("objective", ["parity", "expected-parity", "mse", "scaled-mse"])
def test_all_objectives_against_independent_gate_finite_differences(architecture, objective):
    n, sigma = 6, 1.2
    bits, empirical, masks = example(n)
    theta = .4 * np.random.default_rng(13).normal(size=len(explicit_edges(n, architecture)))
    circuit = study.Circuit(n, architecture)
    weights = study.parity_weights(n, objective, masks, sigma) if "parity" in objective else None
    characters = 1 - 2 * ((masks @ bits.T) % 2)
    distance = np.abs(bits[:, None, :] - bits[None, :, :]).sum(axis=2)
    rho = (1 - np.exp(-1 / (2 * sigma**2))) / 2
    mass_zero = (1-rho)**n
    kernel = (np.exp(-distance / (2*sigma**2)) - mass_zero) / (1-mass_zero)

    def oracle(angles):
        delta = explicit_probabilities(n, architecture, angles) - empirical
        if objective == "parity":
            return np.mean((characters @ delta)**2)
        if objective == "expected-parity":
            return delta @ kernel @ delta
        return delta @ delta / (2**(n-1) if objective == "mse" else 1)

    value, gradient = circuit.loss_gradient(theta, empirical, objective, weights)
    assert value == pytest.approx(oracle(theta), abs=3e-15)
    h = 1e-6
    numerical = np.array([(oracle(theta + h*e) - oracle(theta - h*e)) / (2*h)
                          for e in np.eye(len(theta))])
    np.testing.assert_allclose(gradient, numerical, atol=2e-9, rtol=2e-6)


@pytest.mark.parametrize("sigma", [.5, 1., 2.])
def test_expected_parity_is_conditioned_bernoulli_kernel(sigma):
    n = 4
    bits = bit_table(n)
    rho = (1 - np.exp(-1 / (2*sigma**2))) / 2
    probabilities = np.prod(np.where(bits, rho, 1-rho), axis=1)
    probabilities[0] = 0
    probabilities /= probabilities.sum()
    weights = study.parity_weights(n, "expected-parity", sigma=sigma)
    np.testing.assert_allclose(weights, probabilities, atol=2e-16)
    assert weights[0] == 0
    assert weights.sum() == pytest.approx(1.)
    W = 1 - 2 * ((bits @ bits.T) % 2)
    distances = np.abs(bits[:, None] - bits[None, :]).sum(axis=2)
    kernel = (np.exp(-distances/(2*sigma**2)) - (1-rho)**n) / (1-(1-rho)**n)
    np.testing.assert_allclose(W.T @ np.diag(weights) @ W, kernel, atol=2e-15)


def test_sampled_weights_keep_duplicate_multiplicity():
    masks = bit_table(4)[[1, 3, 3, 3, 7]]
    weights = study.parity_weights(4, "parity", masks)
    assert weights[1] == pytest.approx(.2)
    assert weights[3] == pytest.approx(.6)
    assert weights[7] == pytest.approx(.2)
    with pytest.raises(ValueError, match="nonzero masks"):
        study.parity_weights(4, "parity", bit_table(4)[[0, 1]])


def test_parseval_and_support_mse_scaling_include_gradients():
    n = 6
    _, empirical, _ = example(n)
    circuit = study.Circuit(n, "dense")
    theta = .3*np.random.default_rng(18).normal(size=len(circuit.edges))
    uniform = np.ones(2**n)/2**n  # Include constant mode, whose error is zero.
    spectral_value, spectral_gradient = circuit.loss_gradient(theta, empirical, "parity", uniform)
    scaled_value, scaled_gradient = circuit.loss_gradient(theta, empirical, "scaled-mse")
    raw_value, raw_gradient = circuit.loss_gradient(theta, empirical, "mse")
    assert spectral_value == pytest.approx(scaled_value, abs=2e-15)
    assert scaled_value == pytest.approx(2**(n-1)*raw_value, abs=2e-15)
    np.testing.assert_allclose(spectral_gradient, scaled_gradient, atol=2e-15)
    np.testing.assert_allclose(scaled_gradient, 2**(n-1)*raw_gradient, atol=2e-15)


def test_adam_loss_scaling_requires_matching_epsilon_scaling():
    rng = np.random.default_rng(19)
    theta = rng.normal(size=12)
    raw, rescaled, reduced_epsilon = theta.copy(), theta.copy(), theta.copy()
    scale = 2048.
    a = core.Adam(.02, eps=1e-8)
    b = core.Adam(.02, eps=scale*1e-8)
    c = core.Adam(.02, eps=1e-8/scale)
    d = core.Adam(.02, eps=1e-8)
    fixed_epsilon_scaled = theta.copy()
    for gradient in 1e-7*rng.normal(size=(20, 12)):
        raw = a.update(raw, gradient)
        rescaled = b.update(rescaled, scale*gradient)
        reduced_epsilon = c.update(reduced_epsilon, gradient)
        fixed_epsilon_scaled = d.update(fixed_epsilon_scaled, scale*gradient)
    np.testing.assert_array_equal(raw, rescaled)
    np.testing.assert_array_equal(reduced_epsilon, fixed_epsilon_scaled)
    assert np.max(np.abs(raw-fixed_epsilon_scaled)) > .001


@pytest.mark.parametrize("architecture", ["ring", "dense"])
def test_all_objectives_start_at_same_angles_and_use_final_iterate(architecture):
    protocol = small_protocol(steps=3)
    _, empirical, masks = example(protocol["n"])
    results = [study.train(dict(architecture=architecture, objective=objective, lr=.02, seed=13),
                           empirical, masks, protocol) for objective in protocol["objectives"]]
    for objective, result in zip(protocol["objectives"], results):
        np.testing.assert_array_equal(result["initial"], results[0]["initial"])
        assert len(result["loss_history"]) == 4
        weights = study.parity_weights(protocol["n"], objective, masks) if "parity" in objective else None
        value, _ = study.Circuit(protocol["n"], architecture).loss_gradient(result["theta"], empirical, objective, weights)
        assert result["loss_history"][-1] == pytest.approx(value)
        np.testing.assert_allclose(result["q"], explicit_probabilities(protocol["n"], architecture, result["theta"]), atol=2e-15)


@pytest.mark.parametrize("objective", ["parity", "mse"])
def test_original_ring_ten_update_trajectory(objective):
    protocol = small_protocol(steps=10)
    _, empirical, masks = example(protocol["n"])
    config = dict(architecture="ring", objective=objective, lr=.05, seed=111)
    fresh = study.train(config, empirical, masks, protocol)
    original = core.train_iqp(protocol["n"], empirical, masks, steps=10, lr=.05,
                              seed_init=111+10000+7*protocol["k"], loss=objective,
                              mse_domain="support")
    np.testing.assert_allclose(fresh["theta"], original["theta"], atol=2e-13, rtol=0)
    np.testing.assert_allclose(fresh["q"], original["q"], atol=2e-14, rtol=0)


def test_checkpoint_cache_binds_protocol_and_validation_draws(tmp_path, monkeypatch):
    protocol = small_protocol()
    configs = [dict(architecture="ring", objective=o, lr=.02, seed=13)
               for o in ("parity", "scaled-mse")]
    fingerprint = {"code": "frozen"}
    rows = [study._run_one(c, protocol, tmp_path, fingerprint) for c in configs]
    archives = []
    for config in configs:
        with np.load(tmp_path/f'{study.configuration_key(config)}_seed13.npz', allow_pickle=False) as z:
            archives.append({key:z[key] for key in z.files})
    np.testing.assert_array_equal(archives[0]["samples"], archives[1]["samples"])
    np.testing.assert_array_equal(archives[0]["validation"], archives[1]["validation"])
    np.testing.assert_array_equal(archives[0]["initial"], archives[1]["initial"])
    expected = np.random.default_rng(13+50000).choice(len(archives[0]["p"]), protocol["validation_samples"], p=archives[0]["p"])
    np.testing.assert_array_equal(archives[0]["validation"], expected)
    for row, archive in zip(rows, archives):
        assert row["validation_nll"] == pytest.approx(-np.log(archive["q"][expected]).mean())
    monkeypatch.setattr(study, "train", lambda *a, **kw: pytest.fail("cached run retrained"))
    assert study._run_one(configs[0], protocol, tmp_path, fingerprint) == rows[0]
    with pytest.raises(ValueError, match="stale checkpoint"):
        study._run_one(configs[0], dict(protocol, steps=2), tmp_path, fingerprint)
    with pytest.raises(ValueError, match="stale checkpoint"):
        study._run_one(configs[0], protocol, tmp_path, {"code": "different"})


def fake_rows(configs, confirmation=False):
    values = {("dense", "parity"):1., ("dense", "expected-parity"):1.1,
              ("ring", "parity"):1.2, ("ring", "expected-parity"):1.3,
              ("ring", "mse"):1.05, ("ring", "scaled-mse"):1.2,
              ("dense", "mse"):1.5, ("dense", "scaled-mse"):1.6}
    rows = []
    for config in configs:
        architecture, objective = config["architecture"], config["objective"]
        validation = values[(architecture, objective)]
        kl = 10-validation  # Deliberately reverse exact-target KL ranking.
        if confirmation:
            kl = .3 if "parity" in objective else (.4 if architecture == "ring" else .2)
        rows.append(dict(config, key=study.configuration_key(config), kl=kl,
                         validation_nll=validation, recovery_1000=.5,
                         sample_sha256=str(config["seed"]),
                         initial_sha256=f'{architecture}/{config["seed"]}'))
    return rows


def test_ranking_and_selection_ignore_exact_target_kl():
    configs = [dict(architecture=a, objective=o, lr=.05, seed=s)
               for a, o, s in itertools.product(["ring", "dense"], small_protocol()["objectives"], [10, 11])]
    rows = fake_rows(configs)
    ranking = study.rank_configurations(rows)
    selected = study.select(ranking)
    assert selected["parity"]["key"] == "dense_parity_lr0.05"
    assert selected["mse"]["key"] == "ring_mse_lr0.05"
    assert selected["same_architecture_mse"]["key"] == "dense_mse_lr0.05"
    changed = [dict(row, kl=1000-row["kl"]) for row in rows]
    assert [r["key"] for r in study.rank_configurations(changed)] == [r["key"] for r in ranking]
    ties = [dict(row, validation_nll=1.) for row in rows]
    assert [r["key"] for r in study.rank_configurations(ties)] == sorted({r["key"] for r in ties})


def test_paired_statistics_reject_missing_data_and_mismatched_initializations():
    left = dict(architecture="ring", objective="parity", lr=.05, key="ring_parity_lr0.05")
    right = dict(architecture="ring", objective="mse", lr=.05, key="ring_mse_lr0.05")
    rows = []
    for seed, delta in zip([20, 21, 22], [-1., -2., -3.]):
        for config, kl in [(left, 5+delta), (right, 5.)]:
            rows.append(dict(config, seed=seed, kl=kl, recovery_1000=.5,
                             sample_sha256=str(seed), initial_sha256=str(seed)))
    result = study.paired_summary(left, right, rows)
    assert result["n"] == 3
    assert result["wins"] == 3
    assert result["mean_difference"] == -2.
    assert result["ci"][1] == pytest.approx(-2+4.302652729749462/np.sqrt(3))
    with pytest.raises(ValueError, match="missing paired seeds"):
        study.paired_summary(left, right, rows[:-1])
    for key, message in [("sample_sha256", "unpaired training samples"), ("initial_sha256", "unpaired initialization")]:
        corrupted = copy.deepcopy(rows)
        corrupted[0][key] = "different"
        with pytest.raises(ValueError, match=message):
            study.paired_summary(left, right, corrupted)


def prepare_fake_study(tmp_path, monkeypatch):
    protocol_path = tmp_path/"protocol.json"
    protocol_path.write_text(json.dumps(small_protocol()))
    out = tmp_path/"study"
    calls = []

    def fake_batch(configs, protocol, folder, fingerprint, jobs):
        folder.mkdir(parents=True, exist_ok=True)
        rows = fake_rows(configs, confirmation=folder.name == "confirmation")
        (folder/"metrics.csv").write_text(json.dumps(rows))
        calls.append(folder.name)
        return rows

    monkeypatch.setattr(study, "run_batch", fake_batch)

    def invoke(stage):
        monkeypatch.setattr(sys, "argv", ["study", stage, "--protocol", str(protocol_path), "--out", str(out)])
        study.main()

    return out, calls, invoke


def test_confirmation_requires_both_primary_and_same_architecture_and_seals_selection(tmp_path, monkeypatch):
    out, calls, invoke = prepare_fake_study(tmp_path, monkeypatch)
    invoke("develop")
    invoke("confirm")
    result = json.loads((out/"confirmation-summary.json").read_text())
    assert result["primary"]["ci"][1] < 0
    assert result["same_architecture"]["ci"][0] > 0
    assert result["robust_benefit_gate"] is False
    assert (out/"confirmation-seal.json").exists()
    selection = json.loads((out/"selection.json").read_text())
    selection["selected"]["parity"]["lr"] = .1
    (out/"selection.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="selection changed after confirmation started"):
        invoke("confirm")
    assert calls == ["development", "confirmation"]
    with pytest.raises(ValueError, match="cannot reselect"):
        invoke("develop")


def test_confirmation_rejects_changed_development_evidence(tmp_path, monkeypatch):
    out, calls, invoke = prepare_fake_study(tmp_path, monkeypatch)
    invoke("develop")
    with (out/"development"/"metrics.csv").open("a") as file:
        file.write("changed evidence")
    with pytest.raises(ValueError, match="selection provenance changed"):
        invoke("confirm")
    assert calls == ["development"]
    assert not (out/"confirmation").exists()


def test_interrupted_checkpoint_write_leaves_no_final_cache_and_can_resume(tmp_path, monkeypatch):
    protocol = small_protocol()
    config = dict(architecture="ring", objective="mse", lr=.02, seed=13)
    final_path = tmp_path/f'{study.configuration_key(config)}_seed13.npz'
    original_save = np.savez_compressed

    def interrupted_save(path, **arrays):
        Path(path).write_bytes(b"interrupted incomplete archive")
        raise OSError("simulated interruption")

    monkeypatch.setattr(study.np, "savez_compressed", interrupted_save)
    with pytest.raises(OSError, match="simulated interruption"):
        study._run_one(config, protocol, tmp_path, {"code": "frozen"})
    assert not final_path.exists()
    assert final_path.with_suffix(".tmp.npz").exists()
    monkeypatch.setattr(study.np, "savez_compressed", original_save)
    row = study._run_one(config, protocol, tmp_path, {"code": "frozen"})
    assert final_path.exists()
    assert not final_path.with_suffix(".tmp.npz").exists()
    with np.load(final_path, allow_pickle=False) as saved:
        assert json.loads(str(saved["metrics"])) == row


def test_infinite_kl_aborts_and_preserves_failure_specification(tmp_path, monkeypatch):
    protocol = small_protocol()
    config = dict(architecture="ring", objective="mse", lr=.02, seed=13)
    fingerprint = {"code": "frozen"}

    def impossible_model(config, empirical, masks, protocol):
        bits = bit_table(protocol["n"])
        q = (bits.sum(axis=1) % 2 == 0).astype(float)
        q[0] = 0  # Positive target probability must produce true infinite KL.
        q /= q.sum()
        return dict(q=q, theta=np.zeros(12), initial=np.zeros(12),
                    loss_history=np.zeros(2), gradient_history=np.zeros(2))

    monkeypatch.setattr(study, "train", impossible_model)
    with pytest.raises(ValueError, match="nonfinite exact KL"):
        study._run_one(config, protocol, tmp_path, fingerprint)
    path = tmp_path/f'{study.configuration_key(config)}_seed13.failure.json'
    failure = json.loads(path.read_text())
    assert failure["error"] == "nonfinite exact KL"
    assert failure["kl"] == "inf"
    assert failure["specification"] == dict(config=config, protocol=protocol, fingerprint=fingerprint)
    assert not list(tmp_path.glob("*.npz"))
