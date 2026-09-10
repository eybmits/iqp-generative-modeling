"""Rendering must disclose infinite KL and refuse mixed experiment cohorts."""
import json
import numpy as np
import pytest
from iqp_repro.core import target, forward_kl
from iqp_repro.run_figures import render


def checkpoint(out, beta, protocol="matched"):
    p, support, scores = target(4,beta)
    delta = np.zeros(len(p)); delta[0] = 1
    uniform = support / support.sum()
    rows = []
    for model,q in [("iqp-parity",delta),("iqp-mse",uniform)]:
        kl = forward_kl(p,q)
        rows.append(dict(n=4,beta=beta,seed=111,sigma=1.,k=512,m=50,steps=10,
                         protocol=protocol,model=model,kl="Infinity" if np.isinf(kl) else kl,
                         coverage_1000=0.,elite_size=0))
    np.savez_compressed(out/f"n4_b{beta:g}_s111_sigma1_k512.npz",rows=np.array(json.dumps(rows)),
                        p=p,support=support,scores=scores,elite=np.zeros(len(p),dtype=bool),
                        **{"iqp-parity_q":delta,"iqp-mse_q":uniform})


def test_nonfinite_kl_is_disclosed_and_empty_elite_undefined(tmp_path):
    checkpoint(tmp_path,.9)
    checkpoint(tmp_path,1.)
    manifest = render(tmp_path)
    contrast = manifest["fresh_diagnostics"]["paired_delta_kl"]
    assert contrast["mean"] is None
    assert contrast["undefined_seed_count"] == 1
    assert manifest["fresh_recovery"]["status"] == "undefined: empty unseen elite"
    sweep = manifest["fresh_beta_matched_4_1.0_512"]
    assert sweep["nonfinite_groups"]["iqp-parity/kl"] == [.9,1.]
    assert (tmp_path/"fresh_diagnostics.pdf").exists()
    assert "NaN" not in (tmp_path/"fresh_figures.json").read_text()


def test_mixed_protocols_are_rejected(tmp_path):
    checkpoint(tmp_path,.9)
    checkpoint(tmp_path,1.,protocol="source")
    with pytest.raises(ValueError,match="Separate directories"):
        render(tmp_path)
