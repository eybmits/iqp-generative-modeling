"""Selection uses complete development evidence, never exact target KL."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'select_loss_advantage', Path(__file__).parents[1]/'scripts/select_loss_advantage.py')
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


def cohort(profiles=None):
    profiles = profiles or {
        .9: dict(p=1., mse=1.3, paper=1.4, support=1.5),
        1.2: dict(p=5., mse=5.6, paper=5.6, support=5.6)}
    development = dict(n=12, m=200, k=512, steps=600, validation_samples=2000,
        betas=list(profiles), development_seeds=[111,112], architectures=['ring','ring3'],
        sigmas=[.75,1.], parity_learning_rates=[.1], mse_learning_rates=[.1,.2],
        mse_objectives=['mse','scaled-mse'], optimizer=dict(beta1=.9,beta2=.99,epsilon=1e-8),
        classical={
            model:dict(architectures=[architecture],sigmas=[1.],learning_rates=[lr],steps=[50])
            for model,architecture,lr in [('ising-parity','ring',.05),('ising-nll','dense',.05),
                                         ('maxent','features',.05),('transformer','prefix',.001)]},
        reserved_confirmation_seeds=dict(start=4001,stop_inclusive=4120))
    expected = selector.expected_configurations(development)
    rows, rankings = {}, {}
    digest = lambda *parts: hashlib.sha256(repr(parts).encode()).hexdigest()
    for kind,configs in expected.items():
        rows[kind],rankings[kind] = [],[]
        driver = selector.loss_search if kind=='iqp' else selector._baseline_driver()
        for config in configs:
            profile = profiles[config['beta']]
            key = driver.configuration_key(config)
            if kind == 'iqp':
                parity = config['objective']=='parity'
                nll = (profile['p'] + (.1 if config['architecture']=='ring' else 0.)
                       + (.1 if config['sigma']!=.75 else 0.)) if parity else (
                       profile['mse'] + (.2 if config['architecture']=='ring' else 0.)
                       + (.05 if config['lr']==.1 else 0.)
                       + (.02 if config['objective']=='scaled-mse' else 0.))
                parameters = len(selector.loss_search.Circuit(development['n'],config['architecture']).edges)
                ranking = dict(config,key=key,family='parity' if parity else 'mse',parameters=parameters,
                    fits=2,failures=0,nonfinite_validation_nll=0,mean_validation_nll=nll,
                    mean_kl=.5,mean_recovery_1000=.2)
            else:
                nll = profile['support'] if config['support_aware'] else profile['paper']
                ranking = dict(beta=config['beta'],key=key,
                    config={k:v for k,v in config.items() if k!='beta'},runs=2,
                    mean_validation_nll=nll,mean_kl=.5)
            rankings[kind].append(ranking)
            for seed in development['development_seeds']:
                row = dict(config,key=key,seed=seed,kl=.5,validation_nll=nll,recovery_1000=.2,
                    sample_sha256=digest('sample',config['beta'],seed),
                    validation_sha256=digest('validation',config['beta'],seed))
                if kind=='iqp':
                    row.update(status='ok',parameters=parameters,
                        initial_sha256=digest('initial',config['beta'],seed,config['architecture']),
                        mask_sha256=digest('masks',config['beta'],seed,config['sigma']))
                rows[kind].append(row)
    summary = dict(fits=len(rows['iqp']),failed_fits=0,ranking=rankings['iqp'])
    return development, summary, rankings['classical'], rows['iqp'], rows['classical']


def choose(data):
    development,iq,cl,_,_ = data
    return selector.choose(iq['ranking'],cl,development)


def signature(selection):
    picked = selection['selected']
    return (selection['selection_tier'],picked['beta'],picked['parity']['key'],
            {k:v['key'] for k,v in picked['mse'].items()},
            {k:v['key'] for k,v in picked['classical'].items()})


def test_exact_target_kl_cannot_change_choice():
    data = cohort()
    expected = signature(choose(data))
    for i,row in enumerate(data[1]['ranking']+data[2]):
        row['mean_kl'] = 1000-i if i%2 else i/10000
    assert signature(choose(data)) == expected


def test_beta_selection_uses_within_beta_gaps_not_raw_nll():
    data = cohort()
    selected = choose(data)
    assert selected['selected']['beta'] == 1.2
    assert selected['selected']['parity']['mean_validation_nll'] == 5.
    assert min(r['mean_validation_nll'] for r in data[1]['ranking'] if r['objective']=='parity') == 1.


def test_support_tier_takes_priority_over_better_paper_only_gap():
    data = cohort({.9:dict(p=3.,mse=3.2,paper=3.2,support=3.1),
                   1.2:dict(p=1.,mse=2.,paper=2.,support=.9)})
    selected = choose(data)
    assert selected['selection_tier'] == 'support'
    assert selected['selected']['beta'] == .9


def test_paper_fallback_retains_all_stronger_controls_and_fixed_masks():
    data = cohort({.9:dict(p=3.,mse=3.2,paper=3.2,support=2.9),
                   1.2:dict(p=1.,mse=2.,paper=2.,support=.9)})
    selected = choose(data)
    assert selected['selection_tier'] == 'paper'
    frozen = selector.freeze(selected,data[0],{}, {})
    assert len([r for r in frozen['comparisons'] if r['group']=='support']) == 4
    assert len([r for r in frozen['comparisons'] if r['group']=='paper']) == 4
    assert frozen['k'] == 512
    assert frozen['confirmation_seeds'] == list(range(5001,5121))
    assert '6-qubit unit smoke' in frozen['seed_reservation_note']


def test_identical_mse_roles_deduplicate_after_mask_sigma_adjustment():
    data = cohort()
    for row in data[1]['ranking']:
        if 'mse' in row['objective'] and row['lr']==.2:
            row['mean_validation_nll'] += .5
    selected = choose(data)
    frozen = selector.freeze(selected,data[0],{}, {})
    assert {frozen['role_aliases'][role] for role in ['mse_matched','mse_same','mse_global']} == {'mse_matched'}
    assert len([r for r in frozen['comparisons'] if r['group']=='loss']) == 1
    assert frozen['configurations']['mse_matched']['config']['sigma'] == .75
    assert frozen['k'] == data[0]['k']


def test_no_eligible_or_all_failed_parity_blocks_confirmation():
    data = cohort({.9:dict(p=2.,mse=1.,paper=3.,support=3.)})
    with pytest.raises(ValueError,match='do not draw confirmation data'):
        choose(data)
    data = cohort()
    for row in data[1]['ranking']:
        if row['objective']=='parity':
            row.update(mean_validation_nll='Infinity',failures=2)
    with pytest.raises(ValueError,match='do not draw confirmation data'):
        choose(data)


def test_nan_is_rejected_instead_of_silently_affecting_order():
    data = cohort()
    data[1]['ranking'][0]['mean_validation_nll'] = 'NaN'
    with pytest.raises(ValueError,match='Undefined'):
        choose(data)


def validate(data):
    d,iq,cl,ir,cr = data
    return selector.validate_rankings(iq,cl,ir,cr,d)


def test_complete_synthetic_grid_recomputes_rankings():
    data = cohort()
    iq,cl,report = validate(data)
    assert report['rankings_recomputed'] is True
    assert len(iq) == len(data[1]['ranking'])
    assert len(cl) == len(data[2])
    assert report['shared_hash_groups']['sample_sha256'] == 4


@pytest.mark.parametrize('corruption', ['duplicate_csv','duplicate_ranking','missing_seed','changed_mean','changed_sample'])
def test_counts_cannot_hide_incomplete_or_corrupted_evidence(corruption):
    data = cohort()
    if corruption=='duplicate_csv':
        data[3][-1] = copy.deepcopy(data[3][0])
    elif corruption=='duplicate_ranking':
        data[2][-1] = copy.deepcopy(data[2][0])
    elif corruption=='missing_seed':
        data[4][-1]['seed'] = 113
    elif corruption=='changed_mean':
        data[1]['ranking'][0]['mean_validation_nll'] += .001
    else:
        data[4][-1]['sample_sha256'] = 'f'*64
    with pytest.raises(ValueError):
        validate(data)


def test_source_lock_mismatch_is_rejected_before_evidence_is_selected(tmp_path):
    development,_,_,_,_ = cohort()
    iq,cl = tmp_path/'iqp',tmp_path/'classical'
    iq.mkdir(); cl.mkdir()
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(development))
    iq_protocol = dict(development,parity_sigmas=development['sigmas'],mse_sigma=1.)
    (iq/'lock.json').write_text(json.dumps(dict(canonical_protocol=development,protocol=iq_protocol,
                                              fingerprint={'loss_search_sha256':'stale'})))
    (cl/'lock.json').write_text('{}')
    with pytest.raises(ValueError,match='source/protocol hash mismatch'):
        selector.validate_inputs(iq,cl,protocol,development,Path(__file__).parents[1])
