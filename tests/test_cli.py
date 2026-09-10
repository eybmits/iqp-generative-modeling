import numpy as np
import pytest
from iqp_repro.cli import metrics, interval, summarize, write_json
from iqp_repro import core


def test_support_kl_and_true_missing_mass():
    p=np.array([.5,0,0,.5]); q=np.ones(4)/4
    s=p>0
    result=metrics(p,q,s,s)
    assert result['kl']==pytest.approx(np.log(2))
    assert result['support_kl']==pytest.approx(np.log(2))
    assert result['conditional_kl']==pytest.approx(0)
    assert metrics(p,np.array([1.,0,0,0]),s,s)['kl']==float('inf')


def test_ci_requires_replicates():
    assert interval([1])['ci95_halfwidth'] is None
    assert interval([1,1,1])['ci95_halfwidth']==0


def test_paired_summary_clusters_beta_by_seed(tmp_path):
    rows=[]
    for seed in [111,112]:
        for beta in [.1,.2]:
            for model,value in [('iqp-parity',1.),('iqp-mse',2.)]:
                rows.append(dict(n=6,beta=beta,seed=seed,sigma=1.,k=32,model=model,
                    kl=value,coverage_1000=.1,samples_sha256=str(seed)))
    import json
    np.savez(tmp_path/'n6_b0_s0_sigma1_k32.npz',rows=np.array(json.dumps(rows)))
    summary=summarize(tmp_path)
    contrast=summary['paired_contrasts'][0]
    assert contrast['instances']==4
    assert contrast['paired_delta_kl']['n']==2
    assert contrast['paired_delta_kl']['mean']==-1
    assert contrast['paired_delta_kl']['ci95_halfwidth']==0


def test_nonfinite_paired_ci_exports_honestly(tmp_path):
    import json
    rows=[]
    for seed,a,b in [(111,'Infinity',2.),(112,'Infinity','Infinity')]:
        for model,value in [('iqp-parity',a),('iqp-mse',b)]:
            rows.append(dict(n=6,beta=.9,seed=seed,sigma=1.,k=32,model=model,
                kl=value,coverage_1000=.1,samples_sha256=str(seed)))
    np.savez(tmp_path/'n6_b0_s0_sigma1_k32.npz',rows=np.array(json.dumps(rows)))
    summarize(tmp_path)
    saved=json.loads((tmp_path/'summary.json').read_text())
    contrast=saved['paired_contrasts'][0]
    assert contrast['undefined_pairs']==1
    assert contrast['paired_delta_kl']['ci95_halfwidth'] is None
    assert contrast['paired_delta_kl']['interval_status']=='undefined_nonfinite_values'
    assert saved['paired_instances'][0]['delta_kl']=='Infinity'
    assert saved['paired_instances'][1]['delta_kl'] is None


def test_resume_rejects_stale_protocol_metadata(tmp_path):
    import json
    from iqp_repro.cli import train_instance
    task=dict(n=6,beta=.9,seed=111,sigma=1.,k=32,m=50,steps=1,
              models=['iqp-parity'],protocol='matched',preset='smoke',threads=1,
              implementation_sha256='changed',out=str(tmp_path))
    np.savez(tmp_path/'n6_b0.9_s111_sigma1_k32.npz',rows=np.array('[]'),
             task=np.array(json.dumps(dict(task,implementation_sha256='old'))))
    with pytest.raises(ValueError,match='Checkpoint protocol mismatch'):
        train_instance(task)


def test_resume_rejects_mixed_environments(tmp_path):
    from iqp_repro.cli import check_environment
    path=tmp_path/'environment.json'
    runtime={'python':'3.13.12','threads':4,'versions':{'torch':'2.10.0'}}
    check_environment(path,runtime)
    check_environment(path,runtime)
    with pytest.raises(ValueError,match='Resume environment differs'):
        check_environment(path,dict(runtime,threads=1))


def test_json_serializes_numpy_scalar_counts(tmp_path):
    import json
    path=tmp_path/'counts.json'
    write_json(path,{'n':np.int64(2),'valid':np.bool_(True),'x':np.float64(float('inf'))})
    assert json.loads(path.read_text())=={'n':2,'valid':True,'x':'Infinity'}


def test_cli_training_and_summary_default_to_numerical_outputs(tmp_path):
    """A real training invocation must finish without importing plotting code."""
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src'))
    invocation = '''
import importlib.abc
import sys
class NoPlotImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('matplotlib') or fullname == 'iqp_repro.run_figures':
            raise AssertionError('Numerical reproduction imported plotting code')
sys.meta_path.insert(0, NoPlotImports())
from iqp_repro.cli import main
main()
'''
    command = [sys.executable, '-c', invocation]
    subprocess.run(command + ['run', '--preset', 'smoke', '--n', '4', '--steps', '1',
                   '--models', 'iqp-parity', 'iqp-mse', '--threads', '1', '--out', str(tmp_path)],
                   env=env, check=True, capture_output=True, text=True)
    original = json.loads((tmp_path / 'summary.json').read_text())
    assert original['rows'] == 2
    assert len(original['paired_contrasts']) == 1
    subprocess.run(command + ['summarize', '--no-plots', '--out', str(tmp_path)],
                   env=env, check=True, capture_output=True, text=True)
    assert json.loads((tmp_path / 'summary.json').read_text()) == original
    assert {path.suffix for path in tmp_path.iterdir()} <= {'.json', '.csv', '.npz'}
