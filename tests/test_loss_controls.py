"""Shared observations and metric definitions across the new study's families."""
import importlib.util
from pathlib import Path

import numpy as np

from iqp_repro import core, loss_search


spec = importlib.util.spec_from_file_location("check_loss_baselines",
       Path(__file__).parents[1]/"scripts"/"check_loss_baselines.py")
baselines = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baselines)


def test_new_families_share_training_validation_and_masks():
    protocol = dict(n=6, m=30, k=20, steps=3, validation_samples=40,
                    optimizer=dict(beta1=.9, beta2=.99, epsilon=1e-8))
    iq = loss_search.train_configuration(dict(beta=.9, architecture="ring3",
           objective="parity", sigma=.75, lr=.05), 111, protocol)
    base = dict(beta=.9, model="ising-parity", architecture="ring3", sigma=.75,
                lr=.05, steps=3, support_aware=True, l2=0.)
    aware = baselines.train_configuration(base, 111, protocol)
    paper = baselines.train_configuration(dict(base, architecture="ring", support_aware=False), 111, protocol)
    for key in ("p", "samples", "validation", "masks"):
        np.testing.assert_array_equal(iq[key], aware[key])
        np.testing.assert_array_equal(iq[key], paper[key])
    for result in (aware, paper):
        unseen = result["support"] & (core.empirical(result["samples"], 6) == 0)
        assert result["metrics"]["unseen_mass"] == float(result["q"][unseen].sum())
        p, logq = result["p"], result["logq"]
        positive = p > 0
        assert result["metrics"]["kl"] == float(np.sum(p[positive]*(np.log(p[positive])-logq[positive])))
    assert aware["q"][~aware["support"]].sum() == 0
    assert paper["q"][~paper["support"]].sum() > 0
