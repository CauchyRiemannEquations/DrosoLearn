"""Reward-only switching boundaries on invented features; no science cohort."""
from copy import deepcopy
import numpy as np
import pytest
from drosolearn.selective_geometry import selective_geometry


@pytest.fixture
def geometry():
    raw=np.array([[1.,0.,0.,0.,.4,.2],[0.,1.,0.,0.,.4,.2],
        [0.,0.,1.,0.,.4,.2],[0.,0.,0.,1.,.4,.2]])
    return selective_geometry(raw)


def learner(geometry,method='gated_balanced'):
    from drosolearn.gated_learning import GatedHead
    return GatedHead(geometry,method,.1)


def reveal_four(actor):
    return [actor.observe(i,1,r) for i,r in enumerate([1,1,-1,-1])]


@pytest.mark.parametrize('method',['shared','isolated','feedback_selective','balanced','gated_balanced'])
def test_all_methods_begin_with_unknown_reward_memory_and_half_choice(geometry,method):
    actor=learner(geometry,method)
    assert np.array_equal(actor.known,np.zeros(4,np.int8))
    assert not actor.head.any() and np.array_equal(actor.probabilities(),np.full(4,.5))
    assert actor.calls==actor.discoveries==actor.memory_observation_calls==0


@pytest.mark.parametrize('method',['shared','isolated','feedback_selective','balanced','gated_balanced'])
def test_no_method_can_receive_an_oracle_answer_table(geometry,method):
    from drosolearn.gated_learning import GatedHead
    with pytest.raises((TypeError,ValueError)):GatedHead(geometry,method,.1,np.array([1,1,-1,-1]))


def test_fourth_discovery_stays_isolated_and_only_next_trial_can_open_gate(geometry):
    actor=learner(geometry)
    records=reveal_four(actor)
    assert [e['gate_known_count_before'] for e in records]==[0,1,2,3]
    assert [bool(e['gate_full_before']) for e in records]==[False]*4
    assert [e['direction_mode'] for e in records]==[1]*4
    assert [e['gate_switch_event'] for e in records]==[0]*4
    assert actor.known.tolist()==[1,1,-1,-1] and not actor.cases
    next_step=actor.observe(0,1,1)
    assert next_step['gate_known_count_before']==4 and bool(next_step['gate_full_before'])
    assert next_step['direction_mode']==3 and next_step['gate_switch_event']==1
    assert len(actor.cases)==1 and next_step['case_index']>=0


def test_gate_can_first_activate_on_nogo_without_running_a_cone_solver(geometry,monkeypatch):
    import drosolearn.balanced_learning as previous
    import drosolearn.gated_learning as current
    actor=learner(geometry);reveal_four(actor)
    def forbidden(*args,**kwargs):raise AssertionError('NoGo must not compute a balanced direction')
    monkeypatch.setattr(previous,'cone_direction',forbidden)
    monkeypatch.setattr(current,'cone_direction',forbidden,raising=False)
    before=actor.head.copy();memory=actor.known.copy()
    e=actor.observe(0,0,0)
    assert e['direction_mode']==3 and e['gate_switch_event']==1 and bool(e['gate_full_before'])
    assert e['case_index']==-1 and not e['direction'].any()
    assert not e['head_step'] and not e['update_factor'] and not actor.cases
    assert np.array_equal(before,actor.head) and np.array_equal(memory,actor.known)
    e2=actor.observe(1,0,0)
    assert e2['direction_mode']==3 and e2['gate_switch_event']==0


def test_early_nogo_retains_isolated_unit_direction_but_has_zero_actual_update(geometry):
    actor=learner(geometry);e=actor.observe(2,0,0)
    assert e['direction_mode']==1 and e['gate_known_count_before']==0
    assert not e['gate_full_before'] and e['gate_switch_event']==0
    assert np.array_equal(e['direction'],geometry['isolated_directions'][2])
    assert np.linalg.norm(e['direction'])==pytest.approx(1.)
    assert e['head_step']==e['update_factor']==e['memory_changed']==0
    assert not actor.known.any() and not actor.head.any() and not actor.cases


def test_repeating_known_rewards_cannot_open_gate_with_missing_sign(geometry):
    actor=learner(geometry)
    for index,reward in [(0,1),(1,1),(2,-1)]*4:
        e=actor.observe(index,1,reward)
        assert e['direction_mode']==1 and e['gate_switch_event']==0
    assert actor.known.tolist()==[1,1,-1,0] and actor.discoveries==3
    assert actor.memory_observation_calls==12 and not actor.cases
    absent=actor.observe(3,0,0)
    assert absent['direction_mode']==1 and absent['known_after'][3]==0


@pytest.mark.parametrize('method,expected',[('shared',0),('isolated',1),('feedback_selective',2),('balanced',3)])
def test_controls_keep_their_own_direction_mode_and_never_emit_a_gate_switch(geometry,method,expected):
    actor=learner(geometry,method)
    for i,r in enumerate([1,1,-1,-1]):
        e=actor.observe(i,1,r)
        assert e['direction_mode']==expected and e['gate_switch_event']==0
        assert e['gate_known_count_before']==i and not e['gate_full_before']
    e=actor.observe(0,0,0)
    assert e['direction_mode']==expected and e['gate_switch_event']==0 and e['gate_full_before']


def test_gated_and_isolated_have_identical_factual_prefix_through_fourth_discovery(geometry):
    gated=learner(geometry);isolated=learner(geometry,'isolated')
    signs=[1,1,-1,-1]
    sources=[0,1,1,2,3,0,2,3];actions=[1,0,1,1,0,0,0,1]
    for source,action in zip(sources,actions):
        assert gated.probabilities().tobytes()==isolated.probabilities().tobytes()
        reward=signs[source] if action else 0
        g=gated.observe(source,action,reward);i=isolated.observe(source,action,reward)
        for name in ['direction','known_before','known_after','routing_signs','direction_residuals']:
            assert np.array_equal(g[name],i[name]),name
        for name in ['update_factor','head_step','memory_changed','memory_observation_calls','memory_discoveries']:
            assert g[name]==i[name],name
        assert gated.head.tobytes()==isolated.head.tobytes()
        assert gated.known.tobytes()==isolated.known.tobytes()
        assert g['direction_mode']==1 and g['gate_switch_event']==0
    assert gated.known.tolist()==[1,1,-1,-1]
    g=gated.observe(0,1,1);i=isolated.observe(0,1,1)
    assert g['direction_mode']==3 and i['direction_mode']==1 and g['gate_switch_event']==1


def test_gate_never_closes_after_valid_full_observation_and_event_occurs_once(geometry):
    actor=learner(geometry);reveal_four(actor)
    events=[]
    for t in range(8):
        index=t%4;action=int(t%3!=0);reward=([1,1,-1,-1][index] if action else 0)
        e=actor.observe(index,action,reward)
        events.append(e['gate_switch_event'])
        assert e['direction_mode']==3 and e['gate_full_before']
    assert events==[1]+[0]*7
    for case in actor.cases:assert np.all(np.abs(case['key'][1:])==1)


def test_invalid_trial_does_not_consume_the_first_gate_activation(geometry):
    actor=learner(geometry);reveal_four(actor)
    before=actor.head.copy()
    with pytest.raises(ValueError):actor.observe(0,0,1)
    assert np.array_equal(actor.head,before) and actor.known.tolist()==[1,1,-1,-1]
    first=actor.observe(0,0,0)
    assert first['gate_switch_event']==1
    assert actor.observe(1,0,0)['gate_switch_event']==0


def test_common_handwritten_uniforms_generate_identical_choices_until_full_observation(geometry):
    gated=learner(geometry);isolated=learner(geometry,'isolated')
    signs=[1,1,-1,-1]
    sources=[0,1,2,3,0,1,2,3]
    uniforms=[.1,.9,.1,.9,.1,.1,.1,.1]
    discovery=None
    for trial,(source,u) in enumerate(zip(sources,uniforms),start=1):
        gp=float(gated.probabilities()[source]);ip=float(isolated.probabilities()[source])
        assert gp==ip
        ga,ia=int(u<gp),int(u<ip)
        assert ga==ia
        ge=gated.observe(source,ga,signs[source] if ga else 0)
        ie=isolated.observe(source,ia,signs[source] if ia else 0)
        assert gated.head.tobytes()==isolated.head.tobytes()
        assert gated.known.tobytes()==isolated.known.tobytes()
        assert ge['direction_mode']==1 and ge['gate_switch_event']==0
        if np.count_nonzero(gated.known)==4:
            discovery=trial
            break
    assert discovery==8
    assert gated.observe(0,0,0)['gate_switch_event']==1


def test_output_policy_still_uses_head_rather_than_the_fully_known_reward_table(geometry):
    actor=learner(geometry);reveal_four(actor);actor.observe(0,1,1)
    probability=actor.probabilities()
    assert np.all((probability>0.)&(probability<1.))
    assert np.allclose(probability,1./(1.+np.exp(-(geometry['unit_rows']@actor.head))))
    assert not np.array_equal(probability,(actor.known.astype(float)+1.)/2.)


def test_early_isolation_and_full_knowledge_have_no_meaningful_unknown_other_harm(geometry):
    from drosolearn.balanced_learning import trial_effects
    actor=learner(geometry);signs=np.array([1,1,-1,-1]);harm=0.
    for trial in range(16):
        source=trial%4;before=actor.head.copy()
        e=actor.observe(source,1,int(signs[source]))
        evaluated=trial_effects(geometry['unit_rows'],before,actor.head,signs,source,e['known_before'],e['known_after'])
        harm+=evaluated['unknown_other_negative_correct_probability_sum']
        if e['gate_full_before']:assert evaluated['unknown_other_exposure_count']==0
        else:assert np.max(np.abs(np.delete(evaluated['aligned_margin_delta'],source)))<=1e-8
    assert harm<=1e-8


@pytest.mark.parametrize('method',['shared','isolated','feedback_selective','balanced','gated_balanced'])
def test_each_method_obeys_same_movement_caps_and_preserves_internal_geometry(geometry,method):
    frozen={key:geometry[key].copy() for key in ['unit_rows','pseudoinverse','base_influences','shared_directions','isolated_directions']}
    actor=learner(geometry,method);path=energy=0.
    for t in range(16):
        index=t%4;before=actor.head.copy();e=actor.observe(index,1,[1,1,-1,-1][index])
        step=np.linalg.norm(actor.head-before);path+=step;energy+=step**2
        assert step<=.1+1e-8 and np.linalg.norm(e['direction'])<=1.+1e-8
        assert step==pytest.approx(abs(e['update_factor'])*np.linalg.norm(e['direction']),abs=1e-12)
    assert path<=1.6+1e-8 and energy<=.16+1e-8 and np.linalg.norm(actor.head)<=path+1e-8
    for key,value in frozen.items():assert np.array_equal(geometry[key],value),key


@pytest.mark.parametrize('action,reward',[(0,1),(0,-1),(1,0),(1,2),(2,1)])
def test_counterfactual_or_invalid_obtained_feedback_is_rejected(geometry,action,reward):
    actor=learner(geometry)
    with pytest.raises(ValueError):actor.observe(0,action,reward)
    assert not actor.known.any() and not actor.head.any()


def test_contradictory_deterministic_feedback_does_not_mutate_state(geometry):
    actor=learner(geometry);actor.observe(0,1,1)
    before=(actor.head.copy(),actor.known.copy(),actor.calls,actor.memory_observation_calls)
    with pytest.raises(ValueError):actor.observe(0,1,-1)
    assert np.array_equal(actor.head,before[0]) and np.array_equal(actor.known,before[1])
    assert (actor.calls,actor.memory_observation_calls)==before[2:]


@pytest.mark.parametrize('key,value',[('model_seeds',[17601]),('training_trials',800),('head_lr',.2),
    ('methods',['gated_balanced']),('rng_namespace',202610201),('effect_threshold',.001),
    ('positive_seed_count',1),('population_pass_count',1),('numeric_tolerance',.01),
    ('gate_switch_rule','known_after_four_then_balanced'),('protection_nonworse_seed_count',8),
    ('protection_absolute_tolerance',.001)])
def test_frozen_prospective_design_cannot_be_relaxed_after_results(key,value):
    from drosolearn.gated_learning import SCIENCE,validate_config
    cfg=deepcopy(SCIENCE);cfg[key]=value
    with pytest.raises(ValueError):validate_config(cfg)
