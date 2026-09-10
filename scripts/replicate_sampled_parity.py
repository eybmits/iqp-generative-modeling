"""One frozen follow-up of the sampled-parity secondary result; no selection."""

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np

from iqp_repro import core, study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('protocols/sampled-parity-replication-v2.json'))
    parser.add_argument('--out', type=Path, default=Path('runs/replication'))
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if set(protocol['confirmation_seeds']) & set(protocol['previously_used_seeds']):
        raise ValueError('replication seeds must be new')
    if len(set(protocol['confirmation_seeds'])) != len(protocol['confirmation_seeds']):
        raise ValueError('duplicate replication seed')
    fingerprint = dict(script_sha256=study.digest(__file__), study_sha256=study.digest(study.__file__),
                       core_sha256=study.digest(core.__file__), protocol_sha256=study.digest(args.protocol),
                       python=sys.version, numpy=np.__version__, platform=platform.platform())
    args.out.mkdir(parents=True, exist_ok=True)
    seal = dict(fingerprint=fingerprint, protocol=protocol)
    lock = args.out/'lock.json'
    if lock.exists() and json.loads(lock.read_text()) != seal:
        raise ValueError('frozen protocol, code or environment changed; choose a new output directory')
    study.save_json(lock, seal)
    selected = {label:dict(c, key=study.configuration_key(c)) for label,c in protocol['configurations'].items()}
    configurations = [dict(c, seed=s) for c in protocol['configurations'].values() for s in protocol['confirmation_seeds']]
    rows = study.run_batch(configurations, protocol, args.out/'checkpoints', fingerprint, args.jobs)
    comparisons = {}
    for name in ['matched_dense_mse', 'strong_ring_mse']:
        comparisons[name] = study.paired_summary(selected['sampled_parity'], selected[name], rows, .975)
    comparisons['robust_benefit_gate'] = all(comparisons[name]['ci'][1] < 0 for name in ['matched_dense_mse', 'strong_ring_mse'])
    comparisons['training_runs'] = len(rows)
    comparisons['selection'] = 'Fixed before all replication draws; adaptively motivated by study-v1 secondary result; no pooling with v1'
    study.save_json(args.out/'summary.json', comparisons)
    print(study.canonical(comparisons), flush=True)


if __name__ == '__main__':
    main()
