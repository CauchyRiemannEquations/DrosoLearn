"""Invented-only pipeline and independent mechanism/acquisition decision checks."""
import copy,json
import numpy as np
import pytest
from drosolearn import spillover_tradeoff as study
from drosolearn.graph import Projection
from drosolearn.representation_update import catalog


def fixture_config():
    cfg=copy.deepcopy(study.SCIENCE)
    cfg.update(scope='integration_test',model_seeds=[90092],training_trials=8,
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
    return cfg,arrays,rows


@pytest.mark.parametrize('key,value',[('training_trials',404),('head_lr',.2),('rng_namespace',1),
    ('alphas',[0.,.5,1.]),('primary_alpha',.75),('effect_threshold',.01),('model_seeds',[17311])])
def test_design_is_fixed_before_science(key,value):
    cfg=copy.deepcopy(study.SCIENCE);cfg[key]=value
    with pytest.raises(ValueError):study.validate_config(cfg)


def test_all_five_alphas_costs_and_fixed_witnesses_recorded(produced):
    cfg,a,rows=produced
    assert len(rows)==240 and a['checkpoint_probabilities'].shape==(3,2,5,1,3,4)
    assert np.array_equal(a['alphas'],[0.,.25,.5,.75,1.])
    assert np.all(a['checkpoint_probabilities'][...,0,:]==.5)
    assert all(v.dtype.kind in 'biuf' and np.isfinite(v).all() for v in a.values())
    assert np.allclose(a['trial_head_steps'],a['trial_expected_head_steps'])
    assert np.allclose(a['trial_cumulative_head_path_l2'],np.cumsum(a['trial_head_steps'],axis=-1))
    assert np.max(a['trial_head_steps'])<=.1+1e-12
    assert np.max(a['trial_cumulative_head_path_l2'])<=.8+1e-12
    for top in study.TOPOLOGIES:
        for rep in study.REPRESENTATIONS:
            for key in ('weights','encoder','bias','probe_kc','probe_mbon'):
                assert np.array_equal(a['w_'+top+'_'+rep+'_'+key],a['finalfixed_'+top+'_'+rep+'_'+key])
    s=study.summarize(cfg,a,rows);json.dumps(s,allow_nan=False)
    assert s['counts']['training_feedback_trials']==240 and s['counts']['static_probe_cases']==360
    assert s['counts']['internal_update_calls']==s['counts']['value_update_calls']==0
    assert s['resources']['total_producer_SVD_calls']==18
    assert s['budget_validation']['common_bounds_valid_all']
    assert s['budget_validation']['analytic_probability_ceiling_valid_all']


def synthetic_claims(produced,central,left,right):
    cfg,a,rows=produced;b={key:value.copy() for key,value in a.items()}
    for ai,score in [(0,left),(2,central),(4,right)]:
        b['checkpoint_probabilities'][0,0,ai,0,-1]=[score,score,1-score,1-score]
    for top in study.TOPOLOGIES:
        for rep in study.REPRESENTATIONS:b['c_'+top+'_'+rep+'_correct_probability_ceiling'][:]=1.
    return cfg,b,rows


@pytest.mark.parametrize('middle,left,right,tradeoff,acquisition',[
    (.9,.7,.7,True,True),(.7,.6,.6,True,False),(.85,.83,.83,False,True),(.8,.5,.85,False,True)])
def test_tradeoff_requires_both_endpoints_and_is_separate_from_acquisition(produced,middle,left,right,tradeoff,acquisition):
    cfg,a,rows=synthetic_claims(produced,middle,left,right)
    s=study.summarize(cfg,a,rows)
    assert s['H_tradeoff']['supported'] is tradeoff
    assert s['H_acquisition']['supported'] is acquisition
    assert not s['H_tradeoff']['requires_acquisition']
    assert not s['H_acquisition']['requires_tradeoff_advantage']


def test_best_descriptive_interior_never_replaces_primary(produced):
    cfg,a,rows=synthetic_claims(produced,.6,.55,.6)
    a['checkpoint_probabilities'][0,0,1,0,-1]=[.99,.99,.01,.01]
    s=study.summarize(cfg,a,rows)
    assert not s['H_tradeoff']['supported'] and not s['H_acquisition']['supported']
    assert s['primary_alpha']==.5
    assert [f['alpha'] for f in s['families'] if f['role']=='primary']==[.5]


def test_all_static_and_actual_cost_gates_apply_to_both_hypotheses(produced):
    cfg,a,rows=synthetic_claims(produced,.9,.7,.7)
    assert study.summarize(cfg,a,rows)['H_tradeoff']['supported']
    b={key:value.copy() for key,value in a.items()};b['t_ann_identity_numeric_spillover_qualified'][:]=False
    s=study.summarize(cfg,b,rows)
    assert not s['H_tradeoff']['supported'] and not s['H_acquisition']['supported']
    a['trial_head_steps'][2,1,3,0,0]=.11
    assert not study.summarize(cfg,a,rows)['budget_validation']['common_bounds_valid_all']


def test_hidden_nogo_reward_and_trial_loss_are_not_silent(produced):
    cfg,a,rows=produced
    zero=a['trial_actions']==0
    assert np.all(a['trial_rewards'][zero]==0) and np.all(a['trial_head_steps'][zero]==0)
    assert np.all(a['trial_head_update_calls'][...,-1]==8)
    with pytest.raises(ValueError):study.summarize(cfg,a,rows[:-1])
    b={key:value.copy() for key,value in a.items()};b['checkpoint_probabilities'].flat[0]=float('nan')
    with pytest.raises(ValueError):study.summarize(cfg,b,rows)


@pytest.mark.parametrize('criterion',['energy','endpoint'])
def test_explicit_squared_cost_and_endpoint_path_criteria_gate_both_claims(produced,criterion):
    cfg,a,rows=synthetic_claims(produced,.9,.7,.7)
    assert study.summarize(cfg,a,rows)['H_tradeoff']['supported']
    if criterion=='energy':a['trial_cumulative_head_squared_l2'][2,1,3,0,-1]=1.
    else:a['checkpoint_heads'][2,1,3,0,-1,0]=10.
    s=study.summarize(cfg,a,rows)
    assert not s['H_tradeoff']['supported'] and not s['H_acquisition']['supported']
