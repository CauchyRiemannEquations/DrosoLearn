"""Objective alignment on invented saved states; no real cohort or learner."""
from copy import deepcopy
import numpy as np
import pytest


def study():
    import drosolearn.probability_alignment as module
    return module


def sigmoid(z):return 1./(1.+np.exp(-np.clip(z,-40.,40.)))


def invented_parent(never_full=False):
    """Twelve explicit steps with identity features and certified legacy B=I."""
    n=12;shape=(3,2,6,1,n);y=np.array([1,1,-1,-1])
    cfg=deepcopy(study().SCIENCE);cfg.update(scope='integration_test',model_seeds=[90091],training_trials=n)
    p={'model_seeds':np.array([90091]),'targets':np.array([1,1,0,0]),
        'schedules':np.array([[0,1,2,3]*3],np.int8),'checkpoints':np.array([0,4,8,12])}
    for name in ['trial_update_factors','trial_probabilities','trial_rewards','trial_head_steps']:
        p[name]=np.zeros(shape)
    p['trial_actions']=np.zeros(shape,np.int8)
    for name in ['trial_directions','trial_correct_probability_delta','trial_aligned_margin_delta']:
        p[name]=np.zeros(shape+(4,))
    for name in ['trial_known_before','trial_known_after']:p[name]=np.zeros(shape+(4,),np.int8)
    p['checkpoint_heads']=np.zeros(shape[:-1]+(4,4));p['checkpoint_probabilities']=np.full(shape[:-1]+(4,4),.5)
    for ti,top in enumerate(['real','rewired','ann']):
        for ri,rep in enumerate(['raw','identity']):
            p['g_'+top+'_'+rep+'_unit_rows']=np.eye(4)[None]
            p['g_'+top+'_'+rep+'_isolated_directions']=np.eye(4)[None]
            for mi in range(6):
                head=np.zeros(4);known=np.zeros(4,np.int8)
                for t,source in enumerate(p['schedules'][0]):
                    at=(ti,ri,mi,0,t);before=head.copy();memory=known.copy();full=np.all(memory!=0)
                    action=0 if t in (5,10) or (never_full and source==3) else 1
                    prob=float(sigmoid(head[source]));reward=int(y[source]) if action else 0
                    factor=.1*(action-prob)*reward;direction=np.eye(4)[source].copy()
                    if full and not action:direction[:]=0.
                    head+=factor*direction
                    if action:known[source]=reward
                    for name,value in [('trial_actions',action),('trial_rewards',reward),('trial_probabilities',prob),
                        ('trial_update_factors',factor),('trial_directions',direction),('trial_known_before',memory),
                        ('trial_known_after',known),('trial_head_steps',np.linalg.norm(head-before)),
                        ('trial_correct_probability_delta',y*(sigmoid(head)-sigmoid(before))),
                        ('trial_aligned_margin_delta',y*(head-before))]:p[name][at]=value
                    if t+1 in (4,8,12):
                        ki=(4,8,12).index(t+1)+1;ix=(ti,ri,mi,0,ki)
                        p['checkpoint_heads'][ix]=head;p['checkpoint_probabilities'][ix]=sigmoid(head)
    return cfg,p


@pytest.mark.parametrize('source',range(4))
def test_uniform_and_probability_objectives_have_source_sign_and_four_input_normalization(source):
    d=np.eye(4);k=np.array([1,1,-1,-1],np.int8);z=np.array([.1,.7,-1.4,-.3])
    a,c,h=study().objectives(d,source,k,z,1e-8);p=sigmoid(z)
    assert np.array_equal(a,k[source]*k[:,None]*d)
    assert c.shape==(3,4) and h.shape==(4,) and h.dtype==bool and not h.any()
    assert np.array_equal(c[0],d[source])
    assert np.allclose(c[1],k[source]*(k[:,None]*d).sum(0)/4.,atol=1e-14)
    assert np.allclose(c[2],k[source]*(k[:,None]*(p*(1-p))[:,None]*d).sum(0)/4.,atol=1e-14)


def test_legacy_help_subset_is_distinct_from_the_four_input_evaluation_objective():
    raw=np.array([[1.,0.,0.,0.,.4],[0.,1.,0.,0.,.4],[0.,0.,1.,0.,.4],[0.,0.,0.,1.,.4]])
    d=raw/np.linalg.norm(raw,axis=1)[:,None];k=np.array([1,1,-1,-1],np.int8)
    a,c,h=study().objectives(d,0,k,np.zeros(4),1e-8)
    assert h.tolist()==[False,True,False,False]
    assert np.allclose(c[0],.5*d[0]+.5*d[1])
    assert np.allclose(c[1],(d[0]+d[1]-d[2]-d[3])/4.)
    assert not np.allclose(c[0],c[1])


def test_equal_sensitivity_makes_probability_a_positive_scaling_of_uniform():
    d=np.eye(4);k=np.array([1,1,-1,-1],np.int8)
    _,c,_=study().objectives(d,2,k,np.zeros(4),1e-8)
    assert np.allclose(c[2],.25*c[1])


@pytest.mark.parametrize('source',[0,2])
def test_probability_objective_is_the_actual_signed_step_first_derivative(source):
    d=np.eye(4);k=np.array([1,1,-1,-1],np.int8);z=np.array([.1,.7,-1.4,-.3])
    _,c,_=study().objectives(d,source,k,z,1e-8);w=k[source]*k/2.;eps=1e-6
    g=k[source]*eps;finite=np.mean(k*(sigmoid(z+g*(d@w))-sigmoid(z)))
    assert finite/eps==pytest.approx(c[2]@w,abs=1e-8)
    # The whole finite update is not declared equal to this first-order term.
    large=k[source]*.1;exact=np.mean(k*(sigmoid(z+large*(d@w))-sigmoid(z)))
    assert not np.isclose(exact,.1*(c[2]@w),rtol=1e-5,atol=1e-8)


def test_objectives_do_not_modify_saved_features_memory_or_preupdate_state():
    args=[np.eye(4),1,np.array([1,1,-1,-1],np.int8),np.array([.1,.2,-.3,-.4]),1e-8]
    before=[x.copy() if isinstance(x,np.ndarray) else x for x in args]
    study().objectives(*args)
    assert all(np.array_equal(a,b) for a,b in zip(args,before))


@pytest.mark.parametrize('mutation',['partial','float_memory','bad_sign','wrong_rows','complex_rows',
    'wrong_state','nan_state','positive_clip_boundary','negative_clip_boundary','outside_clip','bad_source'])
def test_objectives_reject_hidden_knowledge_bad_axes_and_nondifferentiable_clipping(mutation):
    d=np.eye(4);k=np.array([1,1,-1,-1],np.int8);z=np.zeros(4);source=0
    if mutation=='partial':k[3]=0
    elif mutation=='float_memory':k=k.astype(float)
    elif mutation=='bad_sign':k[3]=2
    elif mutation=='wrong_rows':d=d[:3]
    elif mutation=='complex_rows':d=d.astype(complex)
    elif mutation=='wrong_state':z=z[:3]
    elif mutation=='nan_state':z[0]=np.nan
    elif mutation=='positive_clip_boundary':z[0]=40.
    elif mutation=='negative_clip_boundary':z[0]=-40.
    elif mutation=='outside_clip':z[0]=41.
    else:source=4
    with pytest.raises(ValueError):study().objectives(d,source,k,z,1e-8)


@pytest.mark.parametrize('key,value',[('model_seeds',[17701]),('training_trials',800),('numeric_tolerance',.1),
    ('baseline_commit','0'*40),('new_confirmatory_hypothesis',True),
    ('representations',['identity']),('topologies',['real'])])
def test_fixed_retrospective_scope_cannot_select_favorable_records_or_thresholds(key,value):
    cfg=deepcopy(study().SCIENCE);cfg[key]=value
    with pytest.raises(ValueError):study().validate_config(cfg)


def test_synthetic_inventory_preserves_full_before_gate_and_uses_no_partial_or_nogo_solver():
    cfg,parent=invented_parent();a,rows=study().analyze(cfg,parent)
    assert len(a)==27 and len(rows)==90
    assert a['reference_full_before'][0,0,0].tolist()==[False]*4+[True]*8
    assert a['eligible'][0,0,0].tolist()==[False]*4+[True,False,True,True,True,True,False,True]
    assert np.count_nonzero(a['eligible'])==36
    for key in ['helpful_mask','objectives','candidate_directions','candidate_influences','candidate_q_changes',
        'candidate_margin_changes','candidate_tangent_changes','candidate_head_steps','candidate_objective_scores']:
        assert not a[key][~a['eligible']].any(),key
    assert np.all(a['case_indices'][~a['eligible']]==-1)
    assert np.all(a['case_indices'][...,:3]==-1)


def test_uniform_cache_reuses_source_but_probability_has_one_key_per_actual_saved_state():
    cfg,parent=invented_parent();a,rows=study().analyze(cfg,parent);summary=study().summarize_arrays(cfg,a,rows,parent)
    keys=a['case_keys'];assert keys.shape==(60,9)
    assert np.count_nonzero(keys[:,3]==0)==24 and np.count_nonzero(keys[:,3]==1)==36
    assert np.isin(keys[keys[:,3]==0,4],range(4)).all()
    assert np.isin(keys[keys[:,3]==1,4],[4,6,7,8,9,11]).all()
    assert a['case_indices'][0,0,0,4,3]==a['case_indices'][0,0,0,8,3]
    assert a['case_indices'][0,0,0,4,4]!=a['case_indices'][0,0,0,8,4]
    counts=summary['counts'];assert counts['new_optimization_calls']==60
    assert counts['unique_uniform_optimizations']==24 and counts['state_probability_optimizations']==36
    assert counts['candidate_destination_effects']==720 and counts['saved_reference_records']==72


def test_negative_source_signed_updates_give_exact_sigmoid_effects_and_equal_actual_step_caps():
    cfg,parent=invented_parent();a,_=study().analyze(cfg,parent);at=(0,0,0,6)
    assert a['eligible'][at] and a['factual_update_factors'][at]<0.
    z=a['factual_logits'][at];g=a['factual_update_factors'][at];d=a['unit_rows'][0,0,0]
    directions=a['candidate_directions'][at];influence=directions@d.T;y=2*parent['targets']-1
    expected=y*(sigmoid(z+g*influence)-sigmoid(z))
    assert np.allclose(a['candidate_q_changes'][at],expected,atol=1e-14)
    assert np.allclose(a['candidate_margin_changes'][at],y*g*influence,atol=1e-14)
    assert np.allclose(a['candidate_head_steps'][at],abs(g),atol=1e-14)
    assert np.min(expected)>=-1e-8
    assert np.allclose(a['candidate_q_changes'][at][0],parent['trial_correct_probability_delta'][0,0,4,0,6])


def test_tangent_prediction_is_exactly_the_signed_gradient_dot_not_the_finite_q():
    cfg,parent=invented_parent();a,_=study().analyze(cfg,parent);e=a['eligible']
    expected=np.abs(a['factual_update_factors'][e,None])*a['candidate_objective_scores'][e,:,2]
    assert np.allclose(a['candidate_tangent_changes'][e].mean(-1),expected,atol=1e-14)
    # A nonlinear output increment can differ while all directions are safe.
    assert np.max(np.abs(a['candidate_q_changes'][e]-a['candidate_tangent_changes'][e]))>1e-8
    scores=a['candidate_objective_scores'][e]
    assert np.min(scores[:,4,2,None]-scores[:,:,2])>=-1e-8
    assert np.min(scores[:,3,1,None]-scores[:,:,1])>=-1e-8


def test_rows_and_all_families_keep_own_other_total_units_and_no_new_learning_claim():
    cfg,parent=invented_parent();a,rows=study().analyze(cfg,parent);s=study().summarize_arrays(cfg,a,rows,parent)
    assert len(s['families'])==30 and s['group_order']==['own','other','total']
    for family in s['families']:
        seed=family['seeds'][0]
        q=np.array(seed['exact_Q_sum']);assert q[0]+q[1]==pytest.approx(q[2])
        assert seed['eligible_trials']==6
        assert sum(seed[name] for name in ['positive_vs_U_trials','negative_vs_U_trials','zero_vs_U_trials'])==6
        assert np.allclose(np.asarray(seed['exact_Q_mean'])*6,q)
        assert seed['mean_head_step']<=.1+1e-8
    for row in rows:
        assert len(row)==17 and row['eligible_trials']==6
        assert row['exact_Q_sum']==pytest.approx(6*row['exact_Q_mean'])
        assert row['head_path_l2_sum']==pytest.approx(6*row['mean_step_l2'])
    interpretation=s['interpretation']
    assert interpretation['tangent_global_maximum_is_expected_integrity_property']
    assert not interpretation['finite_probability_global_optimum_claimed']
    assert not interpretation['summed_fixed_state_gains_predict_new_trajectory']
    assert not interpretation['parent_H_gain'] and not interpretation['parent_H_acquisition'] and interpretation['parent_H_protection']
    assert s['finite_alignment_summary']['new_learned_performance_claim'] is False


def test_empty_fully_observed_scope_retains_all_rows_null_means_and_zero_optimizer_calls():
    cfg,parent=invented_parent(never_full=True);a,rows=study().analyze(cfg,parent);s=study().summarize_arrays(cfg,a,rows,parent)
    assert not a['reference_full_before'].any() and not a['eligible'].any()
    assert a['case_keys'].shape==(0,9) and s['counts']['new_optimization_calls']==0
    assert len(rows)==90 and all(row['eligible_trials']==0 and row['exact_Q_mean'] is None for row in rows)
    for family in s['families']:
        seed=family['seeds'][0]
        assert seed['exact_Q_mean']==[None]*3 and seed['objective_score_mean']==[None]*3
        assert seed['worst_Q_difference_vs_U'] is None and seed['mean_head_step'] is None
    assert s['finite_alignment_summary']['mean_exact_P_minus_U_per_update'] is None


def test_analysis_does_not_draw_randomness_change_parent_or_construct_a_new_training_path(monkeypatch):
    cfg,parent=invented_parent();old={key:value.copy() for key,value in parent.items()}
    def forbidden(*args,**kwargs):raise AssertionError('Saved-state alignment cannot draw new random numbers')
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    a,rows=study().analyze(cfg,parent);s=study().summarize_arrays(cfg,a,rows,parent)
    for key,value in old.items():assert np.array_equal(parent[key],value),key
    for key in ['new_training_trials','new_head_update_calls','new_internal_update_calls','new_value_update_calls','new_random_draws','fresh_model_constructions']:
        assert s['counts'][key]==0
    assert s['counts']['new_optimization_calls']>0


@pytest.mark.parametrize('key',['trial_correct_probability_delta','trial_update_factors','checkpoint_heads'])
def test_corrupted_saved_factual_probability_scalar_or_head_is_rejected(key):
    cfg,parent=invented_parent()
    parent[key][0,0,4,0].flat[0]+=.1
    with pytest.raises(ValueError):study().analyze(cfg,parent)


def test_zero_scalar_actual_go_is_kept_when_interior_sigmoid_rounds_to_one():
    cfg,parent=invented_parent();y=2*parent['targets']-1
    # Arithmetic edge only: this invented early prefix has a large movement and
    # is NOT a valid example of the original 0.1 step-capped learning cohort.
    # A 12-step capped fixture could not reach this floating-point boundary.
    for ti in range(3):
        for ri in range(2):
            base=(ti,ri,4,0);head=np.zeros(4)
            for t,source in enumerate(parent['schedules'][0]):
                at=base+(t,);before=head.copy();action=int(parent['trial_actions'][at])
                reward=int(y[source]) if action else 0;direction=parent['trial_directions'][at].copy()
                if t==0:direction=np.array([760.,0.,0.,0.])
                prob=float(sigmoid(head[source]));factor=.1*(action-prob)*reward;head+=factor*direction
                for name,value in [('trial_probabilities',prob),('trial_update_factors',factor),
                    ('trial_directions',direction),('trial_head_steps',np.linalg.norm(head-before)),
                    ('trial_correct_probability_delta',y*(sigmoid(head)-sigmoid(before))),
                    ('trial_aligned_margin_delta',y*(head-before))]:parent[name][at]=value
                if t+1 in (4,8,12):
                    k=(4,8,12).index(t+1)+1
                    parent['checkpoint_heads'][base+(k,)]=head
                    parent['checkpoint_probabilities'][base+(k,)]=sigmoid(head)
    assert sigmoid(38.)==1. and parent['trial_update_factors'][0,0,4,0,4]==0.
    a,rows=study().analyze(cfg,parent);s=study().summarize_arrays(cfg,a,rows,parent);at=(0,0,0,4)
    assert a['eligible'][at] and a['factual_update_factors'][at]==0.
    assert not a['candidate_q_changes'][at].any() and not a['candidate_tangent_changes'][at].any()
    assert not a['candidate_head_steps'][at].any() and not a['candidate_margin_changes'][at].any()
    assert np.all(a['case_indices'][at][3:]>=0) and s['counts']['eligible_actual_Go_records']==36
