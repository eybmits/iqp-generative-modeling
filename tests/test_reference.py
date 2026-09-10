"""Check the historical-data boundary independently of fresh model training."""
import hashlib
import json
import shutil

import numpy as np
import pytest
from scipy.stats import t

from iqp_repro.figures import REFERENCE, render_reference, validate_reference


@pytest.fixture(scope='module')
def reference_report():
    return validate_reference()


def test_hardware_counts_reproduce_paired_statistics(reference_report):
    with np.load(REFERENCE / 'hardware.npz', allow_pickle=False) as raw:
        assert np.all(raw['counts_cube'].sum(axis=-1) == 10000)
        for budget in [1000, 2000]:
            # Direct occupancy expression, independent of the log1p/expm1 implementation.
            recovery = np.array([[np.mean(1 - (1 - counts[mask.astype(bool)] / 10000) ** budget)
                                  for counts, mask in zip(model, raw['elite_unseen_masks'])]
                                 for model in raw['counts_cube']])
            delta = recovery[0] - recovery[1]
            actual = reference_report['hardware']['paired_recovery'][str(budget)]
            halfwidth = t.ppf(.975, 9) * delta.std(ddof=1) / np.sqrt(10)
            assert actual['mean'] == pytest.approx(delta.mean(), abs=1e-12)
            assert actual['ci95_low'] == pytest.approx(delta.mean() - halfwidth, abs=1e-12)
            assert actual['ci95_high'] == pytest.approx(delta.mean() + halfwidth, abs=1e-12)
            assert actual['parity_wins'] == int((delta > 0).sum())
    assert reference_report['hardware_curve_max_absolute_reconstruction_error'] < 1e-10
    overlap = reference_report['hardware']['mean_plus_minus_sd_overlap']
    assert overlap['positive_saved_budget_points'] == 158
    assert overlap['overlapping_points'] == 158
    assert overlap['overlap_fraction'] == 1
    assert reference_report['legacy_score_display'] == list(range(11))


def test_reference_rejects_tampered_source(tmp_path):
    data = tmp_path / 'reference'
    shutil.copytree(REFERENCE, data)
    with (data / 'beta_metrics.csv').open('ab') as handle:
        handle.write(b'corrupted\n')
    with pytest.raises(ValueError, match='checksum mismatch: beta_metrics.csv'):
        validate_reference(data)


def test_shot_count_invariant_survives_updated_checksum(tmp_path):
    data = tmp_path / 'reference'
    shutil.copytree(REFERENCE, data)
    path = data / 'hardware.npz'
    with np.load(path, allow_pickle=False) as raw:
        arrays = {key: raw[key] for key in raw.files}
    arrays['counts_cube'][0, 0, 0] += 1
    np.savez_compressed(path, **arrays)
    manifest_path = data / 'SOURCES.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['hardware.npz']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Expected 20 historical 10000-shot observations'):
        validate_reference(data)


def test_reference_rerender_has_all_figures_and_report(tmp_path):
    report = render_reference(tmp_path)
    assert len(list(tmp_path.glob('*.pdf'))) == 8
    assert len(list(tmp_path.glob('*.png'))) == 8
    assert all(p.read_bytes().startswith(b'%PDF-') for p in tmp_path.glob('*.pdf'))
    saved = json.loads((tmp_path / 'reference-summary.json').read_text())
    assert saved['hardware'] == report['hardware']
    assert saved['paper_printed_mean_supported_by_raw_arrays'] is False
