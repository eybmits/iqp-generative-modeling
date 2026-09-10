"""Small independent numerical checks, requiring no simulator/service."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from iqp_repro.core import (
    Adam, bits_table, coverage, elite, empirical, forward_kl, fwht,
    iqp_loss_gradient, iqp_probabilities, mask_indices, pairs, sample_masks,
    spectral, target, train_iqp,
)


def dense_parities(masks, bits):
    return 1.0 - 2.0 * ((masks.astype(int) @ bits.T.astype(int)) % 2)


def test_target_bracketed_linear_score_and_support():
    p, valid, scores = target(6, 0.9)
    assert scores[int("100001", 2)] == 4
    assert scores[int("000010", 2)] == 0  # boundary zeros do not count
    assert scores[int("110011", 2)] == 2
    assert valid.sum() == 32
    assert p[~valid].sum() == 0
    assert_allclose(p.sum(), 1)
    # The upstream score was 1+ell on valid states: its shift cancels in p.
    legacy = np.exp(0.9 * (scores[valid] + 1))
    assert_allclose(p[valid], legacy / legacy.sum(), rtol=1e-14)


def test_walsh_matches_dense_and_preserves_float32():
    bits = bits_table(4)
    dense = dense_parities(bits, bits)
    x = np.random.default_rng(1).normal(size=16)
    assert_allclose(fwht(x), dense @ x, atol=1e-14)
    assert_allclose(fwht(fwht(x)) / len(x), x, atol=1e-14)
    assert fwht(x.astype(np.float32)).dtype == np.float32


def test_mask_rejection_and_duplicates():
    masks = sample_masks(5, 3, 100, 333)
    rng = np.random.default_rng(333)
    prob = 0.5 * (1 - np.exp(-1 / 18))
    expected = rng.binomial(1, prob, size=(100, 5)).astype(np.int8)
    zero = np.flatnonzero(expected.sum(axis=1) == 0)
    while len(zero):
        expected[zero] = rng.binomial(1, prob, size=(len(zero), 5))
        zero = np.flatnonzero(expected.sum(axis=1) == 0)
    assert_array_equal(masks, expected)
    assert masks.sum(axis=1).min() > 0
    assert len(np.unique(mask_indices(masks))) < len(masks)


def test_circuit_against_dense_unitary_and_even_support():
    n = 5
    theta = np.random.default_rng(42).normal(size=len(pairs(n)))
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    H = h
    for _ in range(n - 1):
        H = np.kron(H, h)
    spins = 1 - 2 * bits_table(n)
    phases = sum(t * spins[:, i] * spins[:, j] for t, (i, j) in zip(theta, pairs(n)))
    unitary = H @ np.diag(np.exp(-0.5j * phases)) @ H
    q = iqp_probabilities(n, theta)
    assert_allclose(q, np.abs(unitary[:, 0])**2, atol=2e-15)
    assert_allclose(q.sum(), 1, atol=1e-14)
    assert_allclose(q[bits_table(n).sum(axis=1) % 2 == 1], 0, atol=1e-30)
    assert len(pairs(12)) == 24
    assert pairs(2) == [(0, 1)]


@pytest.mark.parametrize("loss", ["parity", "mse"])
def test_exact_reverse_gradient(loss):
    n = 5
    p, _, _ = target(n, 0.9)
    theta = np.random.default_rng(11).normal(size=len(pairs(n)))
    masks = sample_masks(n, 2, 31, 12)
    _, gradient = iqp_loss_gradient(n, theta, p, masks, loss)
    directions = np.eye(len(theta)) * 1e-6
    numerical = [(iqp_loss_gradient(n, theta + d, p, masks, loss)[0]
                  - iqp_loss_gradient(n, theta - d, p, masks, loss)[0]) / 2e-6
                 for d in directions]
    assert_allclose(gradient, numerical, rtol=2e-6, atol=1e-10)


def test_paper_mse_domain_and_pennylane_adam_convention():
    p, _, _ = target(5, 0.9)
    theta = np.full(len(pairs(5)), 0.02)
    cube, gc = iqp_loss_gradient(5, theta, p, loss="mse", mse_domain="cube")
    support, gs = iqp_loss_gradient(5, theta, p, loss="mse", mse_domain="support")
    assert_allclose(support, 2 * cube)
    assert_allclose(gs, 2 * gc)
    tiny = np.array([1e-10])
    got = Adam().update(np.array([0.0]), tiny)
    expected = -0.05 * tiny / (np.abs(tiny) + 1e-8 / np.sqrt(0.01))
    assert_allclose(got, expected, rtol=1e-14)


def test_spectral_projection_and_duplicate_multiplicity():
    p, valid, _ = target(5, 0.7)
    masks = sample_masks(5, 2, 100, 5)
    P = dense_parities(masks, bits_table(5))
    linear = (1 + P.T @ (P @ p)) / len(p)
    clipped = np.maximum(linear, 0)
    assert_allclose(spectral(p, masks), clipped / clipped.sum(), atol=1e-15)
    clipped[~valid] = 0
    corrected = spectral(p, masks, valid)
    assert_allclose(corrected, clipped / clipped.sum(), atol=1e-15)
    assert corrected[~valid].sum() == 0


def test_kl_zero_mass_and_coverage_extremes():
    assert np.isinf(forward_kl([0.5, 0.5], [1, 0]))
    assert np.isfinite(forward_kl([0.5, 0.5], [1, 0], eps=1e-12))
    assert forward_kl([1, 0], [1, 0]) == 0
    certain = coverage([1, 0], [True, False], [0, 1, 100])
    assert_array_equal(certain["discoveries"], [0, 1, 1])
    tiny = coverage([1 - 1e-15, 1e-15], [False, True], [1000])
    assert_allclose(tiny["discoveries"], [1e-12], rtol=1e-12)
    assert np.isnan(coverage([1, 0], [False, False], [1])["recovery"]).all()


def test_kl_support_leakage_decomposition():
    p, valid, _ = target(5, 0.9)
    q = np.random.default_rng(9).uniform(size=len(p))
    q /= q.sum()
    support_mass = q[valid].sum()
    conditioned = np.where(valid, q / support_mass, 0)
    assert_allclose(forward_kl(p, q), forward_kl(p, conditioned) - np.log(support_mass), atol=1e-14)


def test_quantile_includes_score_ties_and_excludes_training():
    scores = np.array([0, 1, 2, 2])
    valid = np.ones(4, dtype=bool)
    assert_array_equal(elite(scores, valid, [], 0.25), [False, False, True, True])
    assert_array_equal(elite(scores, valid, [2], 0.25), [False, False, False, True])
    assert elite(scores, valid, [], 0.25, method="topk").sum() == 1


def test_seeded_training_end_to_end():
    p, support, _ = target(6, 0.9)
    samples = np.random.default_rng(118).choice(len(p), 200, p=p)
    emp = empirical(samples, 6)
    masks = sample_masks(6, 1, 64, 333)
    for loss in ["parity", "mse"]:
        result = train_iqp(6, emp, masks, steps=80, seed_init=10559, loss=loss)
        assert result["loss_history"][-1] < result["loss_history"][0]
        assert np.isfinite(forward_kl(p, result["q"]))
        assert result["q"][~support].sum() == 0
        assert_allclose(result["q"].sum(), 1)
