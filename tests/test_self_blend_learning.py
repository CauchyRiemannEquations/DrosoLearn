"""Prospective fixed blend and gate checks on invented feature rows only."""
from copy import deepcopy

import numpy as np
import pytest
from drosolearn.selective_geometry import selective_geometry


def study():
    import drosolearn.self_blend_learning as module
    return module


@pytest.fixture
def geometry():
    return selective_geometry(np.array([[1.,0.,0.,0.,.4,.2],[0.,1.,0.,0.,.4,.2],
        [0.,0.,1.,0.,.4,.2],[0.,0.,0.,1.,.4,.2]]))


def actor(geometry,method='gated_self_blend'):
    return study().SelfBlendHead(geometry,method,.1)


def reveal_four(head):
    return [head.observe(index,1,reward) for index,reward in enumerate([1,1,-1,-1])]


@pytest.mark.parametrize('method',['shared','isolated','feedback_selective','balanced','gated_balanced','gated_self_blend'])
def test_new_cohort_methods_start_without_privileged_reward_knowledge(geometry,method):
    head=actor(geometry,method)
    assert head.known.tolist()==[0,0,0,0] and not head.head.any()
    assert np.array_equal(head.probabilities(),np.full(4,.5))
    assert head.calls==head.discoveries==head.memory_observation_calls==0
    with pytest.raises((TypeError,ValueError)):
        study().SelfBlendHead(geometry,method,.1,np.array([1,1,-1,-1]))


def test_analytic_fixed_mix_uses_directions_rather_than_changing_objective_weights():
    geo=selective_geometry(np.eye(4));known=np.ones(4,np.int8)
    balanced=np.array([.6,.8,0.,0.]);original=balanced.copy()
    result=study().blend_direction(geo,0,known,balanced)
    raw=np.array([.9,.2,0.,0.]);expected=raw/np.linalg.norm(raw)
    assert np.allclose(result['direction'],expected,atol=1e-14)
    assert np.array_equal(result['isolated'],np.array([1.,0.,0.,0.]))
    assert np.array_equal(result['balanced'],balanced) and np.array_equal(balanced,original)
    assert result['raw_norm']==pytest.approx(np.sqrt(.85))
    assert len(result['residuals'])==5 and np.max(result['residuals'])<=1e-8
    assert result['direction'][0]>.6 and result['direction'][1]>=.25*.8
    assert result['direction'][0]>=.75
    # It is a normalized convex direction, not a 75% probability or reward share.
    assert result['direction'][1]!=pytest.approx(.25*.8)


@pytest.mark.parametrize('source,known,balanced',[
    (0,[1,1,-1,-1],[.6,.8,0.,0.]),
    (2,[1,1,-1,-1],[-.5,0.,np.sqrt(.75),0.]),
    (3,[-1,1,-1,1],[0.,.6,0.,.8])])
def test_unit_influence_safety_and_self_help_floors_use_current_observed_signs(source,known,balanced):
    geo=selective_geometry(np.eye(4));known=np.array(known,np.int8);b=np.array(balanced)
    e=study().blend_direction(geo,source,known,b);w=e['direction'];d=geo['unit_rows']
    align=known[source]*known;other=np.arange(4)!=source;influence=d@w;bi=d@b;ii=d@e['isolated']
    assert .5-1e-8<=e['raw_norm']<=1.+1e-8
    assert np.linalg.norm(w)==pytest.approx(1.)
    assert np.min(align*influence)>=-1e-8
    assert influence[source]>=.75*ii[source]-1e-8
    assert np.all(align[other]*influence[other]>=.25*align[other]*bi[other]-1e-8)
    assert np.max(e['residuals'])<=1e-8


def test_equal_ingredients_remain_the_same_unit_direction():
    geo=selective_geometry(np.eye(4));e=study().blend_direction(geo,1,np.ones(4,np.int8),np.eye(4)[1])
    assert np.array_equal(e['direction'],np.eye(4)[1]) and e['raw_norm']==1.


def test_isolation_cannot_hide_an_unsafe_source_in_the_balanced_ingredient():
    geo=selective_geometry(np.eye(4));known=np.ones(4,np.int8)
    # The mixture itself would be safe; its supposed safe B ingredient is not.
    unsafe=np.array([-.6,.8,0.,0.]);raw=.75*np.eye(4)[0]+.25*unsafe
    assert np.all(raw>=0.) and np.linalg.norm(unsafe)==pytest.approx(1.)
    with pytest.raises(ValueError):study().blend_direction(geo,0,known,unsafe)


def test_mixing_cannot_hide_cross_input_contamination_in_the_isolated_ingredient():
    geo=selective_geometry(np.eye(4));known=np.ones(4,np.int8)
    geo['isolated_directions'][0]=np.array([.8,.6,0.,0.])
    assert np.linalg.norm(geo['isolated_directions'][0])==pytest.approx(1.)
    with pytest.raises(ValueError):study().blend_direction(geo,0,known,np.array([.6,.8,0.,0.]))


def test_fourth_discovery_uses_isolation_and_only_next_trial_can_mix(geometry):
    head=actor(geometry);records=reveal_four(head)
    assert [e['gate_known_count_before'] for e in records]==[0,1,2,3]
    assert [e['direction_mode'] for e in records]==[1]*4
    assert [e['gate_switch_event'] for e in records]==[0]*4
    assert not head.cases
    for e in records:
        assert not e['blend_applicable'] and e['blend_raw_norm']==0.
        assert not e['blend_residuals'].any()
    first=head.observe(0,1,1)
    assert first['direction_mode']==4 and first['gate_known_count_before']==4
    assert first['gate_switch_event']==1 and first['gate_full_before'] and first['blend_applicable']
    assert len(head.cases)==1 and first['case_index']>=0
    assert np.max(first['blend_residuals'])<=1e-8


def test_gate_activation_on_nogo_does_not_compute_balanced_or_mixed_direction(geometry,monkeypatch):
    import drosolearn.balanced_learning as parent
    head=actor(geometry);reveal_four(head)
    def forbidden(*args,**kwargs):raise AssertionError('NoGo cannot request a balanced or mixed direction')
    monkeypatch.setattr(parent,'cone_direction',forbidden)
    monkeypatch.setattr(study(),'blend_direction',forbidden)
    oldhead=head.head.copy();memory=head.known.copy()
    e=head.observe(0,0,0)
    assert e['direction_mode']==4 and e['gate_switch_event']==1 and e['gate_full_before']
    assert e['case_index']==-1 and not e['blend_applicable'] and not e['direction'].any()
    assert not e['blend_residuals'].any() and e['blend_raw_norm']==0.
    assert e['head_step']==e['update_factor']==0. and not head.cases
    assert np.array_equal(head.head,oldhead) and np.array_equal(head.known,memory)
    assert head.observe(1,0,0)['gate_switch_event']==0


def test_early_nogo_does_not_reveal_missing_sign_or_consume_an_optimizer(geometry):
    head=actor(geometry);e=head.observe(3,0,0)
    assert e['direction_mode']==1 and not e['gate_full_before'] and e['gate_switch_event']==0
    assert np.array_equal(e['direction'],geometry['isolated_directions'][3])
    assert e['update_factor']==e['head_step']==e['memory_changed']==0.
    assert not e['blend_applicable'] and not head.known.any() and not head.cases
    for source,reward in [(0,1),(1,1),(2,-1)]*3:
        assert head.observe(source,1,reward)['direction_mode']==1
    assert head.known.tolist()==[1,1,-1,0] and not head.cases


def test_all_three_isolating_paths_have_identical_prefix_through_fourth_discovery(geometry):
    heads=[actor(geometry,m) for m in ['isolated','gated_balanced','gated_self_blend']]
    sources=[0,1,1,2,3,0,2,3];actions=[1,0,1,1,0,0,0,1];signs=[1,1,-1,-1]
    for source,action in zip(sources,actions):
        assert len({head.probabilities().tobytes() for head in heads})==1
        evidence=[head.observe(source,action,signs[source] if action else 0) for head in heads]
        assert len({head.head.tobytes() for head in heads})==1
        assert len({head.known.tobytes() for head in heads})==1
        for field in ['direction','known_before','known_after','routing_signs','direction_residuals']:
            assert all(np.array_equal(e[field],evidence[0][field]) for e in evidence),field
        assert all(e['direction_mode']==1 and e['gate_switch_event']==0 and not e['blend_applicable'] for e in evidence)
    assert heads[-1].known.tolist()==[1,1,-1,-1]
    modes=[head.observe(0,1,1)['direction_mode'] for head in heads]
    assert modes==[1,3,4]


@pytest.mark.parametrize('method',['shared','isolated','feedback_selective','balanced','gated_balanced'])
def test_five_old_methods_keep_their_factual_head_memory_and_directions(geometry,method):
    from drosolearn.gated_learning import GatedHead
    old=GatedHead(geometry,method,.1);new=actor(geometry,method)
    for t in range(12):
        source=t%4;action=int(t%5!=4);reward=[1,1,-1,-1][source] if action else 0
        oe=old.observe(source,action,reward);ne=new.observe(source,action,reward)
        assert old.head.tobytes()==new.head.tobytes() and old.known.tobytes()==new.known.tobytes()
        for field in ['direction','known_before','known_after','routing_signs','direction_residuals']:
            assert np.array_equal(oe[field],ne[field]),field
        for field in ['direction_mode','gate_known_count_before','gate_full_before','gate_switch_event','update_factor','head_step']:
            assert oe[field]==ne[field],field
        assert not ne['blend_applicable'] and ne['blend_raw_norm']==0.


def test_full_go_uses_actual_cached_balanced_ingredient_and_fresh_same_rule_mix(geometry):
    head=actor(geometry);reveal_four(head)
    first=head.observe(0,1,1);oldcases=len(head.cases)
    second=head.observe(0,1,1)
    assert first['case_index']==second['case_index'] and len(head.cases)==oldcases==1
    for e in [first,second]:
        b=head.cases[e['case_index']]['direction'];mix=study().blend_direction(geometry,0,head.known,b)
        assert np.allclose(e['direction'],mix['direction'],atol=1e-14)
        assert np.allclose(e['blend_isolated_direction'],geometry['isolated_directions'][0],atol=1e-14)
        assert np.allclose(e['blend_balanced_direction'],b,atol=1e-14)
        assert e['blend_raw_norm']==pytest.approx(mix['raw_norm'])
    assert second['gate_switch_event']==0 and first['gate_switch_event']==1
    assert second['update_factor']!=first['update_factor']


def test_invalid_feedback_does_not_consume_gate_event_or_modify_learned_state(geometry):
    head=actor(geometry);reveal_four(head);old=head.head.copy();memory=head.known.copy()
    with pytest.raises(ValueError):head.observe(0,0,1)
    assert np.array_equal(head.head,old) and np.array_equal(head.known,memory)
    assert head.observe(0,0,0)['gate_switch_event']==1
    with pytest.raises(ValueError):head.observe(0,1,-1)
    assert head.observe(1,0,0)['gate_switch_event']==0


def test_real_output_update_uses_signed_scalar_and_keeps_unknown_harm_protection(geometry):
    from drosolearn.balanced_learning import trial_effects
    head=actor(geometry);signs=np.array([1,1,-1,-1]);unknown=0.;path=energy=0.
    original={key:geometry[key].copy() for key in ['unit_rows','pseudoinverse','base_influences','shared_directions','isolated_directions']}
    for trial in range(16):
        source=trial%4;p=float(head.probabilities()[source]);before=head.head.copy()
        e=head.observe(source,1,int(signs[source]));factor=.1*(1.-p)*signs[source]
        assert e['update_factor']==pytest.approx(factor)
        assert np.allclose(head.head,before+factor*e['direction'],atol=1e-14)
        step=float(np.linalg.norm(head.head-before));path+=step;energy+=step**2
        assert step<=.1+1e-8 and np.linalg.norm(e['direction'])<=1.+1e-8
        assert step==pytest.approx(abs(factor)*np.linalg.norm(e['direction']),abs=1e-12)
        effects=trial_effects(geometry['unit_rows'],before,head.head,signs,source,e['known_before'],e['known_after'])
        unknown+=effects['unknown_other_negative_correct_probability_sum']
        if e['gate_full_before']:
            assert effects['unknown_other_exposure_count']==0 and e['blend_applicable']
        else:
            assert np.max(np.abs(np.delete(effects['aligned_margin_delta'],source)))<=1e-8
    assert unknown<=1e-8 and path<=1.6+1e-8 and energy<=.16+1e-8
    for key,values in original.items():assert np.array_equal(values,geometry[key]),key
    assert np.allclose(head.probabilities(),1./(1.+np.exp(-(geometry['unit_rows']@head.head))))
    assert not np.array_equal(head.probabilities(),(head.known.astype(float)+1.)/2.)


@pytest.mark.parametrize('mutation',['partial_memory','float_memory','wrong_sign','wrong_direction_shape',
    'nan_direction','complex_direction','nonunit_direction','unsafe_direction','invalid_source'])
def test_blend_rejects_missing_knowledge_nonunit_or_unsafe_ingredients(mutation):
    geo=selective_geometry(np.eye(4));known=np.ones(4,np.int8);source=0;b=np.array([.6,.8,0.,0.])
    if mutation=='partial_memory':known[3]=0
    elif mutation=='float_memory':known=known.astype(float)
    elif mutation=='wrong_sign':known[3]=2
    elif mutation=='wrong_direction_shape':b=b[:3]
    elif mutation=='nan_direction':b[0]=np.nan
    elif mutation=='complex_direction':b=b.astype(complex)
    elif mutation=='nonunit_direction':b*=.5
    elif mutation=='unsafe_direction':b[1]=-.8
    else:source=4
    with pytest.raises(ValueError):study().blend_direction(geo,source,known,b)


@pytest.mark.parametrize('key,value',[('model_seeds',[17701]),('training_trials',800),('head_lr',.2),
    ('rng_namespace',202610202),('methods',['gated_self_blend']),('effect_threshold',.001),
    ('positive_seed_count',1),('population_pass_count',1),('numeric_tolerance',.01),
    ('gate_switch_rule','known_after_four_then_blend'),('protection_nonworse_seed_count',8),
    ('protection_absolute_tolerance',.001)])
def test_registered_scientific_scope_thresholds_and_new_cohort_cannot_be_relaxed(key,value):
    cfg=deepcopy(study().SCIENCE);cfg[key]=value
    with pytest.raises(ValueError):study().validate_config(cfg)


def test_registered_mix_coefficients_and_original_balanced_objective_are_fixed():
    cfg=deepcopy(study().SCIENCE)
    assert cfg['balanced_own_objective_weight']==cfg['balanced_help_objective_weight']==.5
    assert study().BLEND_RESIDUALS==('exact_mix_l2','unit_norm_abs','known_safety_deficit','own_floor_deficit','help_floor_deficit')
    for key in ['blend_isolated_weight','blend_balanced_weight']:
        changed=deepcopy(cfg);changed[key]=.5
        with pytest.raises(ValueError):study().validate_config(changed)
