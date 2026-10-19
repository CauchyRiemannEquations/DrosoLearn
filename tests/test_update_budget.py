"""Invented-graph pipeline checks, never registered graph/seeds."""
import copy
import json
import numpy as np
import pytest
from drosolearn import update_budget as study
from drosolearn.graph import Projection
from drosolearn.representation_update import catalog


def fixture_config():
    cfg = copy.deepcopy(study.SCIENCE)
    cfg.update(scope='integration_test',model_seeds=[90091],training_trials=8,
               checkpoints=[0,4,8],population_pass_count=1,positive_seed_count=1)
    return cfg


def tiny_graph():
    n=20
    return Projection(np.arange(n),np.arange(n),np.ones(n),n,n,
                      list(map(str,range(n))),list(map(str,range(n))),{})


@pytest.fixture
def produced(monkeypatch,tmp_path):
    cfg=fixture_config()
    monkeypatch.setattr(study,'load_projection',lambda path:tiny_graph())
    monkeypatch.setattr(study,'build_graphs',lambda graph,seed:{name:graph.copy() for name in study.TOPOLOGIES})
    path=tmp_path/cfg['input_catalog'];path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'study_id':'representation_update_v1','representations':['raw','identity'],
                                'catalog':{key:value.tolist() for key,value in catalog().items()}}),encoding='utf8')
    arrays,rows=study.produce(cfg,tmp_path)
    return cfg,arrays,rows,tmp_path


@pytest.mark.parametrize('key,value',[('training_trials',404),('head_lr',.2),('rng_namespace',1),
    ('model_seeds',[17211]),('budget_convention','equal_actual_total'),('input_catalog','new.json')])
def test_science_fixed_before_execution(key,value):
    cfg=copy.deepcopy(study.SCIENCE);cfg[key]=value
    with pytest.raises(ValueError):study.validate_config(cfg)


def test_all_trials_costs_and_fixed_witnesses_recorded(produced):
    cfg,a,rows,root=produced
    assert len(rows)==144
    assert a['checkpoint_probabilities'].shape==(3,2,3,1,3,4)
    assert np.all(a['checkpoint_probabilities'][...,0,:]==.5)
    assert all(value.dtype.kind in 'biuf' and np.isfinite(value).all() for value in a.values())
    assert np.allclose(a['trial_head_steps'],a['trial_expected_head_steps'],atol=1e-12)
    assert np.allclose(a['trial_cumulative_head_path_l2'],np.cumsum(a['trial_head_steps'],axis=-1))
    assert np.allclose(a['trial_cumulative_head_squared_l2'],np.cumsum(a['trial_head_steps']**2,axis=-1))
    assert np.max(a['trial_head_steps'][:,:,[0,2]]) <= .1+1e-12
    assert np.max(a['trial_cumulative_head_path_l2'][:,:,[0,2]]) <= .8+1e-12
    assert not np.count_nonzero(a['trial_internal_steps'])
    for top in study.TOPOLOGIES:
        for rep in study.REPRESENTATIONS:
            for key in ('weights','encoder','bias','probe_kc','probe_mbon'):
                assert np.array_equal(a['w_'+top+'_'+rep+'_'+key],a['finalfixed_'+top+'_'+rep+'_'+key])
    s=study.summarize(cfg,a,rows);json.dumps(s,allow_nan=False)
    assert s['counts']['training_feedback_trials']==144
    assert s['counts']['static_probe_cases']==216
    assert s['resources']['total_producer_SVD_calls']==18
    assert s['budget_validation']['common_bounds_valid_all']
    assert s['budget_validation']['analytic_probability_ceiling_valid_all']
    assert not s['budget_validation']['actual_cumulative_cost_equalized']


def test_zero_reward_and_observed_choice_only(produced):
    cfg,a,rows,root=produced
    zero=a['trial_actions']==0
    assert np.all(a['trial_rewards'][zero]==0)
    assert np.all(a['trial_head_steps'][zero]==0)
    assert np.all(a['trial_update_factors'][zero]==0)
    assert np.all(a['trial_head_update_calls'][..., -1]==8)


def test_common_scalar_matches_l2_not_own_score(produced):
    cfg,a,rows,root=produced
    for rep in study.REPRESENTATIONS:
        prefix='b_real_'+rep+'_'
        norms=a[prefix+'direction_norms'][0]
        own=a[prefix+'own_influence'][0]
        assert np.allclose(norms[[0,2]],1)
        assert np.allclose(own[[0,1]],1)
        assert np.allclose(own[2],1/norms[1])
        assert np.any(norms[1]>1.1)


def test_cost_bounds_include_off_input_influence_and_all_updates(produced):
    cfg,a,rows,root=produced
    for top in study.TOPOLOGIES:
        for rep in study.REPRESENTATIONS:
            influence=a['b_'+top+'_'+rep+'_influences']
            upper=2*.1*np.abs(influence).sum(axis=2)
            assert np.array_equal(upper,a['c_'+top+'_'+rep+'_absolute_logit_ceiling'])


def test_primary_budget_gate_requires_acquisition_and_all_conditions(produced):
    cfg,a,rows,root=produced
    b={key:value.copy() for key,value in a.items()}
    b['checkpoint_probabilities'][0,0,0,0,-1]=.5
    b['checkpoint_probabilities'][0,0,2,0,-1]=[.95,.95,.05,.05]
    for top in study.TOPOLOGIES:
        for rep in study.REPRESENTATIONS:b['c_'+top+'_'+rep+'_correct_probability_ceiling'][:]=1.
    assert study.summarize(cfg,b,rows)['H_budget']['supported']
    b['b_ann_identity_numeric_budget_qualified'][:]=False
    assert not study.summarize(cfg,b,rows)['H_budget']['supported']


def test_budget_bound_and_probability_corruption_not_hidden(produced):
    cfg,a,rows,root=produced
    with pytest.raises(ValueError):study.summarize(cfg,a,rows[:-1])
    b={key:value.copy() for key,value in a.items()}
    b['trial_head_steps'][2,1,2,0,0]=.11
    assert not study.summarize(cfg,b,rows)['budget_validation']['common_bounds_valid_all']
    b['checkpoint_probabilities'].flat[0]=float('nan')
    with pytest.raises(ValueError):study.summarize(cfg,b,rows)


def test_old_catalog_tampering_rejected(produced):
    cfg,a,rows,root=produced
    path=root/cfg['input_catalog'];value=json.loads(path.read_text())
    value['catalog']['lexicographic_ids'][0]=1
    path.write_text(json.dumps(value),encoding='utf8')
    with pytest.raises(ValueError):study.load_catalog(cfg,root)
