"""Autonomous U/P head rules on invented features, with no science cohort."""
from copy import deepcopy
import numpy as np
import pytest
from drosolearn.selective_geometry import selective_geometry


def study():
    import drosolearn.objective_learning as module
    return module


@pytest.fixture
def geometry():
    return selective_geometry(np.array([[1.,0.,0.,0.,.4,.2],[0.,1.,0.,0.,.4,.2],
        [0.,0.,1.,0.,.4,.2],[0.,0.,0.,1.,.4,.2]]))


def actor(geometry,method='gated_probability'):
    return study().make_head(geometry,method,.1)


def reveal_four(head):return [head.observe(i,1,r) for i,r in enumerate([1,1,-1,-1])]


@pytest.mark.parametrize('method',['shared','isolated','feedback_selective','balanced','gated_balanced',
    'gated_self_blend','gated_uniform','gated_probability'])
def test_all_new_cohort_methods_start_without_oracle_reward_knowledge(geometry,method):
    head=actor(geometry,method)
    assert head.known.tolist()==[0,0,0,0] and not head.head.any()
    assert np.array_equal(head.probabilities(),np.full(4,.5))
    assert head.calls==head.discoveries==head.memory_observation_calls==0
    with pytest.raises((TypeError,ValueError)):study().make_head(geometry,method,.1,np.array([1,1,-1,-1]))


@pytest.mark.parametrize('method,mode,kind',[('gated_uniform',5,0),('gated_probability',6,1)])
def test_last_discovery_stays_isolated_and_only_next_trial_can_optimize(geometry,method,mode,kind):
    head=actor(geometry,method);records=reveal_four(head)
    assert [e['gate_known_count_before'] for e in records]==[0,1,2,3]
    assert [e['direction_mode'] for e in records]==[1]*4
    assert all(e['gate_switch_event']==0 and not e['objective_applicable'] for e in records)
    assert not head.objective_cases and head.objective_requests==0
    for e in records:
        assert e['objective_kind']==e['objective_case_index']==-1
        assert not e['objective_logits_before'].any() and not e['objective_vector'].any()
    e=head.observe(0,1,1)
    assert e['direction_mode']==mode and e['objective_kind']==kind and e['objective_applicable']
    assert e['gate_full_before'] and e['gate_switch_event']==1 and e['objective_case_index']==0
    assert len(head.objective_cases)==head.objective_requests==1 and not head.cases


@pytest.mark.parametrize('method,mode',[('gated_uniform',5),('gated_probability',6)])
def test_first_full_nogo_records_switch_without_objective_or_solver(geometry,method,mode,monkeypatch):
    head=actor(geometry,method);reveal_four(head)
    def forbidden(*args,**kwargs):raise AssertionError('Full NoGo must skip objectives and optimizers')
    with monkeypatch.context() as patch:
        patch.setattr(study(),'objective_problem',forbidden);patch.setattr(study(),'cone_direction',forbidden)
        before=head.head.copy();memory=head.known.copy();e=head.observe(0,0,0)
        assert e['direction_mode']==mode and e['gate_switch_event']==1 and e['gate_full_before']
        assert not e['objective_applicable'] and e['objective_kind']==e['objective_case_index']==-1
        assert not e['direction'].any() and e['head_step']==e['update_factor']==0.
        assert np.array_equal(head.head,before) and np.array_equal(head.known,memory)
        assert not head.objective_cases and head.objective_requests==0
    assert head.observe(0,1,1)['gate_switch_event']==0


@pytest.mark.parametrize('method',['gated_uniform','gated_probability'])
def test_missing_sign_cannot_be_revealed_by_nogo_or_repeated_known_rewards(geometry,method):
    head=actor(geometry,method)
    for source,reward in [(0,1),(1,1),(2,-1)]*3:
        e=head.observe(source,1,reward)
        assert e['direction_mode']==1 and not e['objective_applicable'] and not e['gate_full_before']
    e=head.observe(3,0,0)
    assert e['known_after'][3]==0 and e['gate_switch_event']==0 and e['direction_mode']==1
    assert head.discoveries==3 and head.objective_requests==0 and not head.objective_cases


def test_all_five_isolating_paths_have_identical_prefix_through_final_discovery(geometry):
    heads=[actor(geometry,m) for m in ['isolated','gated_balanced','gated_self_blend','gated_uniform','gated_probability']]
    sources=[0,1,1,2,3,0,2,3];actions=[1,0,1,1,0,0,0,1];signs=[1,1,-1,-1]
    for source,action in zip(sources,actions):
        assert len({head.probabilities().tobytes() for head in heads})==1
        records=[head.observe(source,action,signs[source] if action else 0) for head in heads]
        assert len({head.head.tobytes() for head in heads})==len({head.known.tobytes() for head in heads})==1
        for field in ['direction','known_before','known_after','routing_signs','direction_residuals']:
            assert all(np.array_equal(e[field],records[0][field]) for e in records),field
        assert all(e['direction_mode']==1 and e['gate_switch_event']==0 for e in records)
    assert [head.observe(0,1,1)['direction_mode'] for head in heads]==[1,3,4,5,6]


@pytest.mark.parametrize('method',['shared','isolated','feedback_selective','balanced','gated_balanced','gated_self_blend'])
def test_six_original_controls_reuse_exact_preserved_head_behavior(geometry,method):
    from drosolearn.self_blend_learning import SelfBlendHead
    old=SelfBlendHead(geometry,method,.1);new=actor(geometry,method)
    assert type(new) is SelfBlendHead
    for t in range(12):
        source=t%4;action=int(t%5!=4);reward=[1,1,-1,-1][source] if action else 0
        oe=old.observe(source,action,reward);ne=new.observe(source,action,reward)
        assert old.head.tobytes()==new.head.tobytes() and old.known.tobytes()==new.known.tobytes()
        for field in ['direction','known_before','known_after','routing_signs','direction_residuals']:
            assert np.array_equal(oe[field],ne[field]),field
        for field in ['direction_mode','gate_switch_event','update_factor','head_step','blend_applicable']:
            assert oe[field]==ne[field],field


@pytest.mark.parametrize('source,kind',[(0,0),(2,0),(0,1),(2,1)])
def test_u_p_objectives_use_observed_source_sign_and_one_quarter_normalization(source,kind):
    d=np.eye(4);k=np.array([1,1,-1,-1],np.int8);z=np.array([.2,.8,-1.4,-.1])
    a,c=study().objective_problem(d,source,k,z,kind);expected=k[source]*k[:,None]*d
    assert np.array_equal(a,expected)
    p=1./(1.+np.exp(-z));weights=np.ones(4) if kind==0 else p*(1-p)
    assert np.allclose(c,weights@expected/4.,atol=1e-14)


def test_probability_direction_and_signed_update_use_own_actual_preupdate_state():
    geo=selective_geometry(np.eye(4));head=actor(geo);reveal_four(head)
    head.head[:]=np.array([.2,.8,-1.4,-.1]);old=head.head.copy();k=head.known.copy();source=2
    beforep=1./(1.+np.exp(-old));a=k[source]*k[:,None]*np.eye(4);c=(beforep*(1-beforep))@a/4.
    expected=c/np.linalg.norm(c);factor=.1*(1-beforep[source])*k[source]
    e=head.observe(source,1,-1)
    assert np.allclose(e['objective_logits_before'],old,atol=1e-14)
    assert np.allclose(e['objective_vector'],c,atol=1e-14)
    assert np.allclose(e['direction'],expected,atol=1e-14)
    assert e['update_factor']==pytest.approx(factor) and factor<0.
    assert np.allclose(head.head,old+factor*expected,atol=1e-14)
    afterp=head.probabilities();afterc=(afterp*(1-afterp))@a/4.
    assert not np.allclose(e['direction'],afterc/np.linalg.norm(afterc),atol=1e-10,rtol=1e-10)


def test_u_reuses_static_source_case_and_p_records_each_current_go_case(geometry):
    uniform=actor(geometry,'gated_uniform');probability=actor(geometry,'gated_probability')
    for head in [uniform,probability]:reveal_four(head)
    u1=uniform.observe(0,1,1);u2=uniform.observe(0,1,1)
    p1=probability.observe(0,1,1);p2=probability.observe(0,1,1)
    assert u1['objective_case_index']==u2['objective_case_index']==0
    assert len(uniform.objective_cases)==len(uniform.objective_cache)==1 and uniform.objective_requests==2
    assert p1['objective_case_index']==0 and p2['objective_case_index']==1
    assert len(probability.objective_cases)==2 and not probability.objective_cache and probability.objective_requests==2
    assert not np.array_equal(p1['objective_logits_before'],p2['objective_logits_before'])
    assert not np.array_equal(p1['objective_vector'],p2['objective_vector'])
    for head in [uniform,probability]:
        for case in head.objective_cases:
            cert=case['certificate'];assert len(cert)==8 and np.max(cert[4:])<=1e-8
            assert cert[0]==pytest.approx(case['objective']@case['direction'])
            assert abs(cert[0]-cert[2])<=1e-8 and cert[3]==pytest.approx(1.)


def test_symmetric_identity_case_is_a_null_for_u_vs_p_direction_and_trajectory():
    geo=selective_geometry(np.eye(4));u=actor(geo,'gated_uniform');p=actor(geo,'gated_probability');signs=[1,1,-1,-1]
    for t in range(16):
        source=t%4;action=int(t not in (5,10));reward=signs[source] if action else 0
        ue=u.observe(source,action,reward);pe=p.observe(source,action,reward)
        assert np.allclose(u.head,p.head,atol=1e-14)
        assert np.array_equal(u.known,p.known) and np.allclose(ue['direction'],pe['direction'],atol=1e-14)
    assert len(p.objective_cases)>len(u.objective_cases)


@pytest.mark.parametrize('method',['gated_uniform','gated_probability'])
def test_invalid_full_feedback_does_not_consume_switch_or_mutate_counters(geometry,method):
    head=actor(geometry,method);reveal_four(head)
    old=(head.head.copy(),head.known.copy(),head.calls,head.memory_observation_calls,head.objective_requests)
    with pytest.raises(ValueError):head.observe(0,1,-1)
    assert np.array_equal(head.head,old[0]) and np.array_equal(head.known,old[1])
    assert (head.calls,head.memory_observation_calls,head.objective_requests)==old[2:]
    assert not head.gate_activated and not head.objective_cases
    assert head.observe(0,0,0)['gate_switch_event']==1


def test_p_domain_failure_is_closed_and_full_nogo_skips_the_derivative_guard():
    geo=selective_geometry(np.eye(4));head=actor(geo);reveal_four(head);head.head[0]=40.
    e=head.observe(1,0,0)
    assert e['gate_switch_event']==1 and not e['objective_applicable'] and head.objective_requests==0
    before=(head.head.copy(),head.known.copy(),head.calls,head.memory_observation_calls,head.objective_requests)
    with pytest.raises(ValueError):head.observe(1,1,1)
    assert np.array_equal(head.head,before[0]) and np.array_equal(head.known,before[1])
    assert (head.calls,head.memory_observation_calls,head.objective_requests)==before[2:] and not head.objective_cases
    # U has no clipped derivative to interpret and keeps its declared objective.
    u=actor(geo,'gated_uniform');reveal_four(u);u.head[0]=40.
    assert u.observe(1,1,1)['objective_applicable']


@pytest.mark.parametrize('method',['gated_uniform','gated_probability'])
def test_actual_updates_obey_common_caps_preserve_geometry_and_protect_unknown_others(geometry,method):
    from drosolearn.balanced_learning import trial_effects
    head=actor(geometry,method);y=np.array([1,1,-1,-1]);path=energy=unknown=0.;events=[]
    original={key:geometry[key].copy() for key in ['unit_rows','pseudoinverse','base_influences','isolated_directions']}
    for t in range(20):
        source=t%4;before=head.head.copy();prob=float(head.probabilities()[source]);e=head.observe(source,1,int(y[source]))
        factor=.1*(1-prob)*y[source]
        assert e['update_factor']==pytest.approx(factor)
        assert np.allclose(head.head,before+factor*e['direction'],atol=1e-14)
        step=float(np.linalg.norm(head.head-before));path+=step;energy+=step**2;events.append(e['gate_switch_event'])
        assert step<=.1+1e-8 and np.linalg.norm(e['direction'])<=1.+1e-8
        effects=trial_effects(geometry['unit_rows'],before,head.head,y,source,e['known_before'],e['known_after'])
        unknown+=effects['unknown_other_negative_correct_probability_sum']
        if e['gate_full_before']:assert effects['unknown_other_exposure_count']==0
        else:assert np.max(np.abs(np.delete(effects['aligned_margin_delta'],source)))<=1e-8
    assert events==[0]*4+[1]+[0]*15
    assert path<=2.+1e-8 and energy<=.2+1e-8 and np.linalg.norm(head.head)<=path+1e-8 and unknown<=1e-8
    for key,value in original.items():assert np.array_equal(geometry[key],value),key
    assert np.allclose(head.probabilities(),1./(1.+np.exp(-(geometry['unit_rows']@head.head))))
    assert not np.array_equal(head.probabilities(),(head.known.astype(float)+1.)/2.)


@pytest.mark.parametrize('mutation',['partial','float_memory','bad_sign','wrong_rows','complex_rows','nan_logits','wrong_logits','bad_source','bad_kind','p_boundary'])
def test_objective_inputs_reject_hidden_or_invalid_knowledge_and_states(mutation):
    d=np.eye(4);source=0;k=np.array([1,1,-1,-1],np.int8);z=np.zeros(4);kind=1
    if mutation=='partial':k[3]=0
    elif mutation=='float_memory':k=k.astype(float)
    elif mutation=='bad_sign':k[3]=2
    elif mutation=='wrong_rows':d=d[:3]
    elif mutation=='complex_rows':d=d.astype(complex)
    elif mutation=='nan_logits':z[0]=np.nan
    elif mutation=='wrong_logits':z=z[:3]
    elif mutation=='bad_source':source=4
    elif mutation=='bad_kind':kind=3
    else:z[0]=40.
    with pytest.raises(ValueError):study().objective_problem(d,source,k,z,kind)


@pytest.mark.parametrize('key,value',[('model_seeds',[17801]),('rng_namespace',202610203),('training_trials',800),
    ('head_lr',.2),('methods',['gated_probability']),('effect_threshold',.001),('positive_seed_count',1),
    ('population_pass_count',1),('protection_nonworse_seed_count',8),('numeric_tolerance',.01),
    ('protection_absolute_tolerance',.001),('state_effect_threshold',.001),('state_positive_seed_count',1),
    ('gain_controls',['balanced']),('gate_switch_rule','known_after_four_then_objective'),
    ('probability_gradient_domain','clipped_derivative_fallback')])
def test_fixed_new_cohort_objectives_and_all_hypothesis_thresholds_cannot_be_relaxed(key,value):
    cfg=deepcopy(study().SCIENCE);cfg[key]=value
    with pytest.raises(ValueError):study().validate_config(cfg)
