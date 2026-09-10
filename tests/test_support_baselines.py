"""Check exact support, restricted likelihood gradients, and prefix generation."""
import numpy as np
import pytest
import torch

from iqp_repro import classical, core
from iqp_repro.support_baselines import (
    _ising_evaluate, _supported_distribution, edge_list, train_baseline,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def data():
    bits = core.bits_table(4)
    samples = np.array([0, 0, 0, 3, 3, 5, 6, 9, 10, 12])
    indices = np.array([1, 2, 3, 3, 5, 7, 15])
    masks = ((indices[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.int8)
    return bits, samples, masks


def check_support(result, bits):
    support = bits.sum(axis=1) % 2 == 0
    assert result['q'].shape == (len(bits),)
    assert np.all(result['q'][~support] == 0.)
    assert np.all(np.isneginf(result['logq'][~support]))
    assert np.isfinite(result['logq'][support]).all()
    assert np.isfinite(result['loss_history']).all()
    assert result['q'].sum() == pytest.approx(1., abs=1e-12)
    np.testing.assert_allclose(np.exp(result['logq']), result['q'], atol=1e-15)


@pytest.mark.parametrize('model', ['ising-parity', 'ising-nll', 'maxent', 'transformer'])
@pytest.mark.parametrize('l2', [0., .01])
def test_training_returns_normalized_even_law_and_finite_final_loss(data, model, l2):
    bits, samples, masks = data
    config = dict(model=model, lr=.01, steps=4, l2=l2)
    result = train_baseline(config, bits, samples, masks, seed=19)
    check_support(result, bits)
    assert len(result['loss_history']) == 5
    assert result['parameters'] > 0


@pytest.mark.parametrize('loss', ['parity', 'nll'])
def test_restricted_ising_gradient_matches_finite_differences(data, loss):
    bits, samples, masks = data
    fields = [8, 4, 2, 1]
    features = np.array([fields[i] | fields[j] for i,j in edge_list(4, 'ring')] + fields)
    empirical = core.empirical(samples, 4)
    indices = core.mask_indices(masks)
    support = bits.sum(axis=1) % 2 == 0
    theta = .1*np.random.default_rng(17).standard_normal(len(features))
    kwargs = dict(features=features, empirical=empirical, indices=indices,
                  support=support, loss=loss, l2=.07)
    value, gradient, q, logq = _ising_evaluate(theta, **kwargs)
    epsilon = 1e-6
    numerical = []
    for direction in np.eye(len(theta)):
        plus = _ising_evaluate(theta+epsilon*direction, **kwargs)[0]
        minus = _ising_evaluate(theta-epsilon*direction, **kwargs)[0]
        numerical.append((plus-minus)/(2*epsilon))
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-5)
    assert np.isfinite(value)
    assert q[~support].sum() == 0.
    if loss == 'nll':
        observed = np.bincount(samples, minlength=len(bits)) > 0
        assert value == pytest.approx(-empirical[observed] @ logq[observed]+.035*(theta @ theta))


def test_support_normalizer_ignores_arbitrarily_large_forbidden_logits():
    support = np.array([True, False, False, True])
    q, logq = _supported_distribution(np.array([0., 10000., 10000., 0.]), support)
    np.testing.assert_array_equal(q, [.5, 0., 0., .5])
    assert np.isfinite(logq[support]).all()


def test_maxent_restricted_one_statistic_has_known_solution(data):
    bits, _, _ = data
    samples = np.array([0]*7 + [3]*3)
    duplicate_masks = np.array([[0, 0, 0, 1], [0, 0, 0, 1]])
    result = train_baseline(dict(model='maxent', lr=.05, steps=350),
                            bits, samples, duplicate_masks, seed=91)
    support = bits.sum(axis=1) % 2 == 0
    expected = np.where(support, (1+.4*(1-2*bits[:, -1]))/support.sum(), 0.)
    np.testing.assert_allclose(result['q'], expected, atol=2e-7)
    check_support(result, bits)


def test_transformer_completion_preserves_trained_prefix_law_and_direct_seed(data):
    bits, samples, masks = data
    config = dict(model='transformer', lr=.01, steps=3)
    result = train_baseline(config, bits, samples, masks, seed=23)
    prefix = classical.train_transformer(core.bits_table(3), samples >> 1,
                                         seed=23, epochs=3, lr=.01)
    # Completion is x -> 2*x + parity(x), not conditioning a 4-bit AR model.
    prefixes = np.arange(8)
    completed = 2*prefixes + (core.bits_table(3).sum(axis=1) % 2)
    np.testing.assert_array_equal(result['q'][completed], prefix['q'])
    check_support(result, bits)
    assert result['parameters'] == sum(p.numel() for p in classical.ARTransformer(3).parameters())


@pytest.mark.parametrize('architecture, count', [('ring', 36), ('ring3', 48), ('dense', 78)])
def test_ising_graph_capacity_and_direct_initialization_seed(architecture, count):
    bits = core.bits_table(12)
    masks = np.eye(12, dtype=np.int8)
    result = train_baseline(dict(model='ising-parity', architecture=architecture, lr=.05, steps=0),
                            bits, np.array([0, 3]), masks, seed=71)
    assert result['parameters'] == count
    np.testing.assert_array_equal(result['theta'], .01*np.random.default_rng(71).standard_normal(count))
    assert all(i < j for i,j in edge_list(12, architecture))


def test_spectral_uniform_shrinkage_and_rejecting_odd_training_data(data):
    bits, samples, masks = data
    result = train_baseline(dict(model='spectral', alpha=1.), bits, samples, masks, seed=0)
    support = bits.sum(axis=1) % 2 == 0
    np.testing.assert_array_equal(result['q'], support/support.sum())
    assert result['parameters'] == 0
    assert result['fitted_moments'] == 6
    with pytest.raises(ValueError, match='even parity'):
        train_baseline(dict(model='ising-nll', steps=1), bits, np.array([1, 2]), masks, seed=0)
