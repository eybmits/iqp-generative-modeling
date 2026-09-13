"""Auditor integration and tampering checks on a tiny complete three-stage study."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from iqp_repro import objective_comparison


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("objective_audit", ROOT / "scripts/audit_objective_comparison.py")
auditor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auditor)


def save_json(path, document):
    Path(path).write_text(json.dumps(document, indent=2) + "\n")


@pytest.fixture(scope="module")
def completed_study(tmp_path_factory):
    folder = tmp_path_factory.mktemp("objective-audit-fixture")
    protocol = json.loads((ROOT / "protocols/objective-comparison.json").read_text())
    protocol.update(n=4, m=20, k=8, steps=1, validation_samples=30, learning_rates=[.1],
                    development_seeds=[101, 102], historical_seeds=[501, 502],
                    confirmation_seeds=[9001, 9002])
    protocol_path = folder / "protocol.json"
    save_json(protocol_path, protocol)
    runs = folder / "runs"
    for stage in ("historical", "develop", "confirm"):
        objective_comparison.run(stage, runs, protocol_path, jobs=1)
    return runs, protocol_path


@pytest.fixture
def study_copy(completed_study, tmp_path):
    original, protocol = completed_study
    copied = tmp_path / "runs"
    shutil.copytree(original, copied)
    return copied, protocol


def test_complete_study_audits_all_retained_states_and_statistics(completed_study):
    runs, protocol = completed_study
    result = auditor.audit(runs, protocol)
    assert result["status"] == "pass"
    assert result["checkpoint_files"] == 44
    assert result["unique_paired_seeds"] == 6
    assert [result["stages"][name]["fits"] for name in auditor.STAGES] == [16, 14, 14]
    assert result["max_absolute_errors"]["probabilities"] == 0
    assert result["max_absolute_errors"]["kl"] < 2e-12
    assert result["max_absolute_errors"]["summary"] < 2e-12


def test_missing_confirmation_fails_without_writing_output(study_copy, monkeypatch, capsys, tmp_path):
    runs, protocol = study_copy
    (runs / "confirm/summary.json").unlink()
    destination = tmp_path / "validation.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--runs", str(runs), "--protocol", str(protocol), "--out", str(destination)])
    with pytest.raises(SystemExit) as error:
        auditor.main()
    assert error.value.code == 1
    assert "incomplete study" in capsys.readouterr().err
    assert not destination.exists()


@pytest.mark.parametrize("field, message", [("theta", "Born law differs"), ("samples", "deterministic samples")])
def test_changed_checkpoint_arrays_are_detected(study_copy, field, message):
    runs, protocol = study_copy
    checkpoint = next((runs / "checkpoints").glob("mse_lr0.1_seed501.npz"))
    with np.load(checkpoint, allow_pickle=False) as data:
        changed = {name: data[name].copy() for name in data.files}
    if field == "theta":
        changed[field][0] += .01
    else:
        changed[field][0] ^= 1
    np.savez_compressed(checkpoint, **changed)
    with pytest.raises(ValueError, match=message):
        auditor.audit(runs, protocol)


def test_wrong_summary_mean_is_detected(study_copy):
    runs, protocol = study_copy
    path = runs / "historical/summary.json"
    document = json.loads(path.read_text())
    document["ranking"][0]["mean_kl"] += .01
    save_json(path, document)
    with pytest.raises(ValueError, match="independently recomputed summary"):
        auditor.audit(runs, protocol)


def test_changed_selection_is_rejected_even_with_updated_seal_hash(study_copy):
    runs, protocol = study_copy
    path = runs / "develop/selection.json"
    selection = json.loads(path.read_text())
    selection["configurations"][0]["lr"] = .2
    save_json(path, selection)
    seal_path = runs / "confirm/seal.json"
    seal = json.loads(seal_path.read_text())
    seal["configurations"] = selection["configurations"]
    seal["selection_sha256"] = auditor.digest(path)
    save_json(seal_path, seal)
    with pytest.raises(ValueError, match="validation-only development selection"):
        auditor.audit(runs, protocol)


def test_source_hash_tampering_is_detected(study_copy):
    runs, protocol = study_copy
    path = runs / "lock.json"
    lock = json.loads(path.read_text())
    lock["fingerprint"]["source_hashes"]["core.py"] = "0" * 64
    save_json(path, lock)
    with pytest.raises(ValueError, match="source hash mismatch"):
        auditor.audit(runs, protocol)


def test_duplicate_or_missing_seed_is_rejected_before_statistics():
    config = dict(objective="mse", lr=.1)
    rows = [dict(config, key="mse_lr0.1", seed=seed, status="ok", kl=.3, validation_nll=3.) for seed in (1, 2)]
    for invalid in (rows[:1], rows + [rows[0]]):
        with pytest.raises(ValueError, match="incomplete or duplicate cohort"):
            auditor.independent_ranking(invalid, [config], [1, 2])
