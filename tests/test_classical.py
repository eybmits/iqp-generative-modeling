import numpy as np
import pytest
import torch

from iqp_repro.classical import (
    ARTransformer, autoregressive_logq, train_ising, train_maxent,
    train_transformer,
)
from iqp_repro.core import bits_table


@pytest.fixture(autouse=True)
def paper_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def case():
    bits = bits_table(4)
    masks = np.asarray([1, 2, 3, 3, 5, 7, 15])
    mask_bits = ((masks[:, None] >> np.arange(3, -1, -1)) & 1)
    P = (1 - 2 * ((mask_bits @ bits.T) % 2)).astype(float)
    samples = np.asarray([0, 0, 0, 3, 3, 5, 6, 9, 10, 12])
    empirical = np.bincount(samples, minlength=16) / len(samples)
    return bits, masks, P, samples, empirical


def check_distribution(result):
    assert np.isfinite(result["logq"]).all()
    assert np.isfinite(result["loss_history"]).all()
    assert np.all(result["q"] >= 0)
    assert result["q"].sum() == pytest.approx(1, abs=1e-12)
    np.testing.assert_allclose(np.exp(result["logq"]), result["q"], atol=1e-15)


def test_maxent_known_one_statistic_solution_and_duplicate_masks():
    # For one binary sufficient statistic, the exact fitted moment is z and
    # each state in the corresponding half of the cube has mass (1 +/- z)/N.
    z = 0.4
    result = train_maxent(np.array([1, 1]), np.array([z, z]), n=4, steps=350)
    expected = (1 + z * (1 - 2 * (np.arange(16) & 1))) / 16
    check_distribution(result)
    np.testing.assert_allclose(result["q"], expected, atol=2e-7)


def test_dense_and_compressed_parity_features_agree():
    bits, masks, P, _, empirical = case()
    z = P @ empirical
    dense = train_maxent(P, z, steps=40)
    compressed = train_maxent(masks, z, n=4, steps=40, method="walsh")
    np.testing.assert_allclose(dense["q"], compressed["q"], atol=2e-7)
    for topology, loss in [("nn_nnn", "parity"), ("dense", "nll")]:
        kwargs = dict(seed=17, steps=40, topology=topology, loss=loss)
        a = train_ising(bits, P, z, empirical, **kwargs)
        b = train_ising(bits, masks, z, empirical, **kwargs)
        check_distribution(a)
        np.testing.assert_allclose(a["q"], b["q"], atol=1e-12)
        assert a["loss_history"][-1] < a["loss_history"][0]


def test_maxent_and_ising_keep_full_support():
    bits, masks, P, _, empirical = case()
    result = train_maxent(masks, P @ empirical, n=4, steps=20)
    check_distribution(result)
    assert result["q"][bits.sum(1) % 2 == 1].sum() > 0
    zero = train_maxent(masks, P @ empirical, n=4, steps=0)
    np.testing.assert_allclose(zero["q"], np.full(16, 1 / 16))
    assert zero["loss_history"].size == 1


def test_transformer_capacity_causality_and_chain_normalization():
    torch.manual_seed(2)
    model = ARTransformer(12)
    assert sum(p.numel() for p in model.parameters()) == 9057
    model.eval()
    # Alter tokens strictly after position 2: first three output logits cannot
    # change. The driver supplies BOS followed by preceding bits only.
    tokens = torch.tensor([[2, 0, 1, 0, 0], [2, 0, 1, 1, 1]])
    with torch.no_grad():
        output = model(tokens)
    torch.testing.assert_close(output[0, :3], output[1, :3], atol=1e-7, rtol=0)
    logq = autoregressive_logq(model, bits_table(5), batch_size=7)
    assert np.exp(logq).sum() == pytest.approx(1, abs=2e-7)
    assert np.isfinite(logq).all()


def test_transformer_log_probability_has_no_clipping_floor():
    model = ARTransformer(4)
    with torch.no_grad():
        model.out.weight.zero_()
        model.out.bias.fill_(100)
    logq = autoregressive_logq(model, bits_table(4))
    assert logq[0] == pytest.approx(-400)
    assert np.exp(logq).sum() == pytest.approx(1)


def test_transformer_training_is_finite_and_repeatable():
    bits, _, _, samples, _ = case()
    first = train_transformer(bits, samples, seed=9, epochs=4)
    second = train_transformer(bits, samples, seed=9, epochs=4)
    check_distribution(first)
    np.testing.assert_array_equal(first["q"], second["q"])
    assert first["loss_history"][-1] < first["loss_history"][0]
