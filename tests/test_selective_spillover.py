"""Reward-information and head/state separation on invented data only."""
from copy import deepcopy
import numpy as np
import pytest
from drosolearn.selective_geometry import selective_geometry
from drosolearn.selective_spillover import SelectiveHead,SCIENCE,information_checks,validate_config


@pytest.fixture
def geo():
    return selective_geometry(np.array([[1,0,0,0,.4,.2],[0,1,0,0,.4,.2],
        [0,0,1,0,.4,.2],[0,0,0,1,.4,.2]],float))


@pytest.mark.parametrize('method',['shared','isolated','feedback_selective'])
def test_hidden_targets_cannot_enter_nonoracle_constructor(geo,method):
    with pytest.raises(ValueError,match='hidden target'):SelectiveHead(geo,method,.1,np.array([1,1,-1,-1]))


@pytest.mark.parametrize('signs',[None,[1,1,0,-1],[1,1,-1],[1,1,-1,np.nan]])
def test_oracle_requires_explicit_full_valid_target_information(geo,signs):
    with pytest.raises(ValueError):SelectiveHead(geo,'oracle_selective',.1,signs)


def test_initial_memory_is_unknown_even_in_privileged_diagnostic(geo):
    for method in ['shared','isolated','feedback_selective','oracle_selective']:
        actor=SelectiveHead(geo,method,.1,[1,1,-1,-1] if method=='oracle_selective' else None)
        assert not actor.known.any() and np.array_equal(actor.probabilities(),np.full(4,.5))
        assert actor.discoveries==actor.memory_observation_calls==actor.calls==0


def test_nogo_is_neither_new_label_nor_output_learning(geo):
    actor=SelectiveHead(geo,'feedback_selective',.1);e=actor.observe(2,0,0)
    assert not actor.known.any() and not actor.head.any() and not e['harm_mask'].any()
    assert not e['routing_signs'].any() and e['head_step']==e['memory_changed']==0
    assert actor.calls==1 and actor.memory_observation_calls==0


def test_current_go_reward_is_available_before_same_trial_routing(geo):
    actor=SelectiveHead(geo,'feedback_selective',.1)
    actor.observe(0,1,1);e=actor.observe(2,1,-1)
    assert e['known_before'].tolist()==[1,0,0,0]
    assert e['known_after'].tolist()==e['routing_signs'].tolist()==[1,0,-1,0]
    assert e['harm_mask'].tolist()==[True,False,False,False]
    assert abs(e['influence'][0])<1e-8 and e['desired_influence'][1]==geo['base_influences'][2,1]
    assert e['desired_influence'][3]==geo['base_influences'][2,3]


def test_same_target_help_is_preserved_raw_but_scaled_at_unit_length(geo):
    actor=SelectiveHead(geo,'feedback_selective',.1)
    actor.observe(0,1,1);actor.observe(1,1,1);actor.observe(2,1,-1)
    e=actor.observe(0,1,1)
    assert e['desired_influence'][1]==geo['base_influences'][0,1]
    assert e['desired_influence'][2]==0
    assert e['raw_norm']>=1 and np.linalg.norm(e['direction'])==pytest.approx(1.)
    assert e['influence'][1]==pytest.approx(e['desired_influence'][1]*e['normalization_scale'])


def test_reward_repetition_does_not_invent_extra_discovery(geo):
    actor=SelectiveHead(geo,'feedback_selective',.1)
    first=actor.observe(1,1,1);second=actor.observe(1,1,1)
    assert first['memory_changed']==1 and second['memory_changed']==0
    assert actor.discoveries==1 and actor.memory_observation_calls==2 and actor.calls==2


def test_conflicting_observation_fails_before_mutating_memory_or_head(geo):
    actor=SelectiveHead(geo,'feedback_selective',.1);actor.observe(1,1,1)
    before=actor.head.copy()
    with pytest.raises(ValueError,match='Contradictory'):actor.observe(1,1,-1)
    assert actor.known.tolist()==[0,1,0,0] and np.array_equal(actor.head,before)
    assert actor.memory_observation_calls==actor.discoveries==1


@pytest.mark.parametrize('method',['shared','isolated'])
def test_control_directions_do_not_use_observed_signs(geo,method):
    actor=SelectiveHead(geo,method,.1);actor.observe(0,1,1);e=actor.observe(2,1,-1)
    assert actor.known.tolist()==[1,0,-1,0] and not e['routing_signs'].any() and not e['harm_mask'].any()
    expected=geo['shared_directions'][2] if method=='shared' else geo['isolated_directions'][2]
    assert np.array_equal(e['direction'],expected)


def test_known_table_is_not_used_as_direct_output_policy(geo):
    actor=SelectiveHead(geo,'feedback_selective',.1)
    for i,r in enumerate([1,1,-1,-1]):actor.observe(i,1,r)
    assert actor.lookup_probabilities().tolist()==[1.,1.,0.,0.]
    assert not np.array_equal(actor.lookup_probabilities(),actor.probabilities())
    assert np.all((actor.probabilities()>0)&(actor.probabilities()<1))


def test_oracle_and_feedback_routes_match_once_all_signs_observed(geo):
    feedback=SelectiveHead(geo,'feedback_selective',.1)
    oracle=SelectiveHead(geo,'oracle_selective',.1,[1,1,-1,-1])
    for i,r in enumerate([1,1,-1,-1]):feedback.observe(i,1,r);oracle.observe(i,1,r)
    ef=feedback.observe(0,1,1);eo=oracle.observe(0,1,1)
    assert np.array_equal(ef['direction'],eo['direction'])
    assert np.array_equal(ef['routing_signs'],eo['routing_signs'])


@pytest.mark.parametrize('action,reward',[(0,1),(0,-1),(1,0),(1,2)])
def test_task_observation_contract_rejects_counterfactual_rewards(geo,action,reward):
    actor=SelectiveHead(geo,'feedback_selective',.1)
    with pytest.raises(ValueError):actor.observe(0,action,reward)


@pytest.mark.parametrize('key,value',[('head_lr',.2),('rng_namespace',5),('methods',['feedback_selective']),
    ('effect_threshold',.001),('model_seeds',[17401])])
def test_science_configuration_cannot_be_tuned_after_results(key,value):
    cfg=deepcopy(SCIENCE);cfg[key]=value
    with pytest.raises(ValueError):validate_config(cfg)


@pytest.fixture(scope='module')
def saved_fixture():
    from test_selective_spillover_verify import fixture_config,tiny_graph,audit
    cfg=fixture_config();arrays,_=audit.expected_artifacts(cfg,tiny_graph())
    return cfg,arrays


@pytest.mark.parametrize('field',['checkpoint_known','checkpoint_memory_observation_calls','checkpoint_memory_discoveries'])
def test_checkpoint_memory_must_match_actual_observation_path(saved_fixture,field):
    cfg,original=saved_fixture;arrays=deepcopy(original);arrays[field].flat[0]=1
    if field=='checkpoint_known':arrays['checkpoint_lookup_probabilities'].flat[0]=1.
    assert not all(information_checks(cfg,arrays).values())


def test_checkpoint_probability_must_come_from_saved_output_head(saved_fixture):
    cfg,original=saved_fixture;arrays=deepcopy(original)
    arrays['checkpoint_probabilities'].flat[-1]=.999
    assert not information_checks(cfg,arrays)['head_policy_does_not_directly_lookup_memory']
