"""Full development/confirmation audit and tampering checks at small n."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from iqp_repro import objective_ladder

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("ladder_audit", ROOT / "scripts/audit_objective_ladder.py")
auditor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auditor)


@pytest.fixture(scope="module")
def complete_study(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ladder-audit")
    p = json.loads((ROOT / "protocols/objective-ladder.json").read_text())
    p.update(n=6, radii=[2, 3], sigma_by_radius={"2": 1., "3": .75}, m=20, k=8,
             steps=1, validation_samples=30, learning_rates=[.1],
             development_seeds=[101, 102], confirmation_seeds=[9001, 9002])
    protocol = folder / "protocol.json"
    protocol.write_text(json.dumps(p))
    runs = folder / "runs"
    for stage in ("develop", "confirm"):
        objective_ladder.run(stage, runs, protocol, jobs=1)
    return runs, protocol


@pytest.fixture
def copied(complete_study, tmp_path):
    runs, protocol = complete_study
    dest = tmp_path / "runs"
    shutil.copytree(runs, dest)
    return dest, protocol


def test_all_checkpoints_and_architecture_comparisons_audit(complete_study):
    runs, protocol = complete_study
    result = auditor.audit(runs, protocol)
    assert result["status"] == "pass"
    assert result["checkpoints"] == 56
    assert result["unique_datasets"] == 4
    assert result["architecture_dataset_pairs"] == 8
    assert result["max_absolute_errors"]["probabilities"] == 0
    assert result["max_absolute_errors"]["summary"] < 2e-12


def test_refuses_missing_confirmation(copied):
    runs, protocol = copied
    (runs / "confirm/seal.json").unlink()
    with pytest.raises(ValueError, match="incomplete study"):
        auditor.audit(runs, protocol)


@pytest.mark.parametrize("field", ["theta", "samples"])
def test_changed_probabilities_or_input_are_detected(copied, field):
    runs, protocol = copied
    path = next((runs / "checkpoints").glob("r2_mse*_seed9001.npz"))
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    if field == "theta":
        arrays[field][0] += .01
    else:
        arrays[field][0] ^= 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="Born law differs|deterministic samples"):
        auditor.audit(runs, protocol)


def test_wrong_joint_correction_is_detected(copied):
    runs, protocol = copied
    path = runs / "confirm/summary.json"
    summary = json.loads(path.read_text())
    summary["comparisons"][0]["bonferroni_factor"] = 1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="independent summary"):
        auditor.audit(runs, protocol)


def test_changed_tail_statistic_is_detected(copied):
    runs, protocol = copied
    path = runs / "confirm/summary.json"
    summary = json.loads(path.read_text())
    summary["stability"][0]["p95"] *= .9
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="independent summary"):
        auditor.audit(runs, protocol)


def test_confirmation_rejects_forged_selection_before_training(copied):
    runs, protocol = copied
    path = runs / "develop/selection.json"
    selection = json.loads(path.read_text())
    selection["configurations"][0]["lr"] = .2
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="selection differs"):
        objective_ladder.run("confirm", runs, protocol, jobs=1)
    with pytest.raises(ValueError, match="validation-only selection"):
        auditor.audit(runs, protocol)
