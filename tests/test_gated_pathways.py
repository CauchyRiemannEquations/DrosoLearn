"""Common-time, individual-knowledge accounting on handwritten saved paths."""
from copy import deepcopy
import numpy as np
import pytest


def study():
    import drosolearn.gated_pathways as module
    return module


def sigmoid(values):return 1./(1.+np.exp(-np.clip(values,-40.,40.)))


def fixture_data(gate_never=False):
    """Eight explicit saved steps, no graph, RNG, actor or solver object."""
    n=8;shape=(3,2,5,1,n);signs=np.array([1,1,-1,-1])
    cfg=deepcopy(study().SCIENCE);cfg.update(scope='integration_test',model_seeds=[90091],training_trials=n)
    parent={'model_seeds':np.array([90091]),'targets':np.array([1,1,0,0]),
        'schedules':np.array([[0,1,2,3,0,1,2,3]],np.int8),'checkpoints':np.array([0,4,8])}
    for name in ['trial_update_factors','trial_head_steps','trial_head_step_squared_l2','trial_cumulative_head_path_l2','trial_cumulative_head_squared_l2']:
        parent[name]=np.zeros(shape)
    parent['trial_actions']=np.zeros(shape,np.int8)
    for name in ['trial_directions','trial_correct_probability_delta','trial_aligned_margin_delta']:
        parent[name]=np.zeros(shape+(4,))
    for name in ['trial_known_before','trial_known_after']:parent[name]=np.zeros(shape+(4,),np.int8)
    parent['checkpoint_heads']=np.zeros(shape[:-1]+(3,4))
    parent['checkpoint_probabilities']=np.full(shape[:-1]+(3,4),.5)
    parent['first_balanced_mode_trials']=np.full(shape[:-1],n+1,np.int32)
    for ti,top in enumerate(['real','rewired','ann']):
        for ri,rep in enumerate(['raw','identity']):
            parent['g_'+top+'_'+rep+'_unit_rows']=np.eye(4)[None]
            for mi in range(5):
                head=np.zeros(4);known=np.zeros(4,np.int8);path=energy=0.
                for t,source in enumerate(parent['schedules'][0]):
                    at=(ti,ri,mi,0,t);before=head.copy();full=np.count_nonzero(known)==4
                    action=1
                    if mi==0:action=0
                    if mi==1 and t==3:action=0
                    if mi in (2,4) and t==1:action=0
                    if mi==4 and gate_never:action=0
                    known_before=known.copy();p=float(sigmoid(head[source]))
                    reward=int(signs[source]) if action else 0
                    g=.1*(action-p)*reward
                    w=np.eye(4)[source]
                    if action==0 and (mi==3 or (mi==4 and full)):w=np.zeros(4)
                    head+=g*w
                    if action:known[source]=reward
                    step=float(np.linalg.norm(g*w));path+=step;energy+=step**2
                    for name,value in [('trial_actions',action),('trial_update_factors',g),('trial_directions',w),
                        ('trial_head_steps',step),('trial_head_step_squared_l2',step**2),('trial_cumulative_head_path_l2',path),
                        ('trial_cumulative_head_squared_l2',energy),('trial_known_before',known_before),('trial_known_after',known),
                        ('trial_correct_probability_delta',signs*(sigmoid(head)-sigmoid(before))),
                        ('trial_aligned_margin_delta',signs*(head-before))]:parent[name][at]=value
                    if mi==4 and full and parent['first_balanced_mode_trials'][ti,ri,mi,0]==n+1:
                        parent['first_balanced_mode_trials'][ti,ri,mi,0]=t+1
                    if t+1 in (4,8):
                        k=(4,8).index(t+1)+1
                        parent['checkpoint_heads'][ti,ri,mi,0,k]=head
                        parent['checkpoint_probabilities'][ti,ri,mi,0,k]=sigmoid(head)
    return cfg,parent


def test_same_reference_time_split_does_not_claim_same_knowledge():
    cfg,parent=fixture_data();arrays,rows=study().analyze(cfg,parent)
    assert len(arrays)==12 and len(rows)==180
    # Gated discovers fourth sign at6; its common post starts trial7.
    assert arrays['first_full_trials'][0,0,:,0].tolist()==[9,8,6,4,6]
    assert arrays['first_gated_balanced_trials'][0,0,0]==7
    assert arrays['gated_phase_before'][0,0,0].tolist()==[0,0,0,0,0,0,1,1]
    assert arrays['own_full_before'][0,0,3,0].tolist()==[False]*4+[True]*4
    assert arrays['own_full_before'][0,0,1,0].tolist()==[False]*8
    assert np.any((arrays['gated_phase_before'][0,0,0]==0)&arrays['own_full_before'][0,0,3,0])
    assert np.any((arrays['gated_phase_before'][0,0,0]==1)&~arrays['own_full_before'][0,0,1,0])


def test_post_reference_phase_can_have_unknown_destinations_in_other_method():
    cfg,parent=fixture_data();arrays,_=study().analyze(cfg,parent)
    post=arrays['gated_phase_before'][0,0,0]==1
    assert np.any(arrays['effect_categories'][0,0,1,0][post]==2)
    assert not np.any(arrays['effect_categories'][0,0,4,0][post]==2)
    assert arrays['effect_categories'][0,0,1,0,6,3]==2


def test_phase_costs_are_repeated_by_category_but_endpoint_contributions_are_not():
    cfg,parent=fixture_data();arrays,rows=study().analyze(cfg,parent)
    selected=[row for row in rows if row['topology']=='real' and row['representation']=='raw' and row['method']=='gated_balanced']
    assert len(selected)==6
    for phase in ['pre','post']:
        grouped=[row for row in selected if row['phase']==phase]
        assert len({row['trial_count'] for row in grouped})==1
        assert len({row['go_trials'] for row in grouped})==1
        assert len({row['head_path_l2'] for row in grouped})==1
        for row in grouped:
            assert row['endpoint_contribution']==pytest.approx(row['probability_signed_sum']/4.)
            assert row['probability_positive_mass']-row['probability_negative_mass']==pytest.approx(row['probability_signed_sum'])
            if row['go_trials']:assert row['gain_per_Go']==pytest.approx(row['endpoint_contribution']/row['go_trials'])
            if row['head_path_l2']:assert row['gain_per_path']==pytest.approx(row['endpoint_contribution']/row['head_path_l2'])


def test_zero_go_zero_path_ratios_are_null_without_deleting_records():
    cfg,parent=fixture_data();arrays,rows=study().analyze(cfg,parent)
    shared=[row for row in rows if row['method']=='shared']
    assert len(shared)==36
    assert all(row['gain_per_Go'] is None and row['gain_per_path'] is None for row in shared)
    assert all(row['probability_signed_sum']==row['head_path_l2']==row['head_squared_l2']==0 for row in shared)
    assert sum(row['exposures'] for row in rows)==3*2*5*8*4


def test_all_factual_contributions_and_phase_costs_reconstruct_parent_totals():
    cfg,parent=fixture_data();arrays,rows=study().analyze(cfg,parent)
    summary=study().summarize_arrays(cfg,arrays,rows,parent)
    assert len(summary['families'])==30 and len(summary['contrasts'])==24
    for family in summary['families']:
        seed=family['seeds'][0]
        parts=np.asarray(seed['phase_category_endpoint_contributions'])
        assert parts.sum()==pytest.approx(seed['final_expected_accuracy']-seed['initial_expected_accuracy'])
        costs=seed['phase_costs']
        assert sum(c['trial_count'] for c in costs)==8
        assert sum(c['go_trials'] for c in costs)==sum(c['nonzero_head_steps'] for c in costs)
        assert sum(c['endpoint_contribution'] for c in costs)==pytest.approx(parts.sum())
    for contrast in summary['contrasts']:
        seed=contrast['seeds'][0]
        assert np.asarray(seed['phase_category_endpoint_difference']).sum()==pytest.approx(seed['endpoint_difference'])
        assert sum(seed['phase_endpoint_difference'])==pytest.approx(seed['endpoint_difference'])


def test_knowledge_mismatch_and_first_discovery_differences_use_own_path():
    cfg,parent=fixture_data();arrays,rows=study().analyze(cfg,parent)
    summary=study().summarize_arrays(cfg,arrays,rows,parent)
    families={x['method']:x for x in summary['families'] if x['topology']=='real' and x['representation']=='raw'}
    assert families['balanced']['seeds'][0]['own_full_vs_reference_phase_disagreement_trials']==2
    assert families['isolated']['seeds'][0]['own_full_vs_reference_phase_disagreement_trials']==2
    assert families['gated_balanced']['seeds'][0]['own_full_vs_reference_phase_disagreement_trials']==0
    contrasts={x['control']:x for x in summary['contrasts'] if x['topology']=='real' and x['representation']=='raw'}
    assert contrasts['balanced']['seeds'][0]['own_full_trial_difference']==2
    assert contrasts['isolated']['seeds'][0]['own_full_trial_difference']==-2


def test_never_opening_reference_leaves_post_empty_and_keeps_sentinel_and_nulls():
    cfg,parent=fixture_data(gate_never=True);arrays,rows=study().analyze(cfg,parent)
    assert np.all(arrays['first_gated_balanced_trials']==9)
    assert not arrays['gated_phase_before'].any()
    post=[row for row in rows if row['phase']=='post']
    assert len(post)==90
    assert all(row['trial_count']==row['exposures']==row['go_trials']==row['head_path_l2']==0 for row in post)
    assert all(row['gain_per_Go'] is None and row['gain_per_path'] is None for row in post)
    summary=study().summarize_arrays(cfg,arrays,rows,parent)
    assert all(f['seeds'][0]['phase_costs'][1]['gain_per_Go'] is None for f in summary['families'])


def test_analysis_has_no_rng_or_solver_and_preserves_parent_verdicts(monkeypatch):
    cfg,parent=fixture_data()
    def forbidden(*args,**kwargs):raise AssertionError('No new random draws or optimization in saved-path accounting')
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    monkeypatch.setattr(np.linalg,'svd',forbidden)
    arrays,rows=study().analyze(cfg,parent);summary=study().summarize_arrays(cfg,arrays,rows,parent)
    for name in ['new_training_trials','new_head_update_calls','new_internal_update_calls','new_value_update_calls',
        'new_random_draws','fresh_model_constructions','new_optimization_calls']:assert summary['counts'][name]==0
    interpretation=summary['interpretation']
    assert not interpretation['parent_H_gain'] and not interpretation['parent_H_acquisition'] and interpretation['parent_H_protection']
    assert not interpretation['new_confirmatory_hypothesis'] and not interpretation['novel_input_generalization_tested']


@pytest.mark.parametrize('key,value',[('model_seeds',[17601]),('training_trials',800),('numeric_tolerance',.1),
    ('phase_definition','each_method_known_after_full'),('reference_method','balanced'),
    ('category_definition','reference_method_knowledge'),('contrast_controls',['balanced']),
    ('new_confirmatory_hypothesis',True),('baseline_commit','0'*40)])
def test_frozen_saved_analysis_cannot_select_more_favorable_scope(key,value):
    cfg=deepcopy(study().SCIENCE);cfg[key]=value
    with pytest.raises(ValueError):study().validate_config(cfg)


def test_nonlinear_probability_effects_and_categorical_partition_use_exact_inputs():
    d=np.eye(4);head=np.array([0.,3.,-3.,1.]);w=np.array([1.,1.,-1.,-1.]);signs=np.array([1,1,-1,-1])
    q,m=study().probability_effects(d,head,np.asarray(.05),w,signs)
    assert np.allclose(q,signs*(sigmoid(head+.05*w)-sigmoid(head)),atol=1e-14)
    assert np.allclose(m,signs*.05*w,atol=1e-14)
    assert q[0]>q[1]>0.
    cat=study().decompose_categories(np.array([1,0,-1,0]),np.asarray(1))
    assert cat.tolist()==[1,0,1,2]


def test_batched_probability_effects_preserve_every_saved_step():
    d=np.eye(4);heads=np.array([[0.,1.,2.,3.],[1.,0.,-1.,-2.]])
    w=np.array([[1.,0.,0.,0.],[0.,0.,1.,0.]]);g=np.array([0.,-.04]);y=np.array([1,1,-1,-1])
    q,m=study().probability_effects(d,heads,g,w,y)
    assert q.shape==m.shape==(2,4) and not q[0].any() and not m[0].any()
    assert np.allclose(q,y*(sigmoid(heads+g[:,None]*w)-sigmoid(heads)),atol=1e-14)


@pytest.mark.parametrize('mutation',['rows','head','direction','scalar','sign','nan','complex'])
def test_effect_helper_rejects_bad_axes_and_nonfinite_values(mutation):
    args=[np.eye(4),np.zeros(4),np.asarray(.03),np.ones(4),np.array([1,1,-1,-1])]
    if mutation=='rows':args[0]=args[0][:3]
    elif mutation=='head':args[1]=np.zeros(3)
    elif mutation=='direction':args[3]=np.zeros((1,4))
    elif mutation=='scalar':args[2]=np.array([.03])
    elif mutation=='sign':args[4][0]=0
    elif mutation=='nan':args[1][0]=np.nan
    else:args[3]=args[3].astype(complex)
    with pytest.raises(ValueError):study().probability_effects(*args)


@pytest.mark.parametrize('known,source',[(np.asarray(0),np.asarray(0)),(np.zeros(3,int),np.asarray(0)),
    (np.array([1,2,0,0]),np.asarray(0)),(np.array([1.,0.,0.,0.]),np.asarray(0)),
    (np.zeros(4,int),np.asarray(1.)),(np.zeros(4,int),np.asarray(4)),(np.zeros((2,4),int),np.asarray(1))])
def test_category_helper_rejects_invalid_memory_or_source_axes(known,source):
    with pytest.raises(ValueError):study().decompose_categories(known,source)


def test_empty_or_zero_effect_cells_are_kept_and_small_effects_not_thresholded():
    cfg,parent=fixture_data();arrays,rows=study().analyze(cfg,parent)
    assert all(row['nonzero_probability_effects']==row['positive_probability_effects']+row['negative_probability_effects'] for row in rows)
    assert any(row['exposures']>0 and row['nonzero_probability_effects']==0 for row in rows)
    assert any(row['exposures']==0 and row['probability_positive_mass']==row['probability_negative_mass']==0 for row in rows)
    q,_=study().probability_effects(np.eye(4),np.zeros(4),np.asarray(1e-8),np.array([1.,-1.,0.,0.]),np.ones(4,int))
    assert q[0]>0 and q[1]<0 and abs(q[0])<1e-8 and abs(q[1])<1e-8


def test_parent_arrays_are_not_mutated_and_costs_match_saved_totals():
    cfg,parent=fixture_data();original={key:value.copy() for key,value in parent.items()}
    arrays,rows=study().analyze(cfg,parent)
    summary=study().summarize_arrays(cfg,arrays,rows,parent)
    for key,value in original.items():assert np.array_equal(parent[key],value),key
    for ti in range(3):
        for ri in range(2):
            for mi in range(5):
                seed=summary['families'][ti*10+ri*5+mi]['seeds'][0]
                at=(ti,ri,mi,0)
                assert sum(c['head_path_l2'] for c in seed['phase_costs'])==pytest.approx(parent['trial_cumulative_head_path_l2'][at+(-1,)])
                assert sum(c['head_squared_l2'] for c in seed['phase_costs'])==pytest.approx(parent['trial_cumulative_head_squared_l2'][at+(-1,)])


@pytest.mark.parametrize('field',['trial_correct_probability_delta','trial_aligned_margin_delta','trial_head_steps',
    'trial_head_step_squared_l2','trial_cumulative_head_path_l2','trial_cumulative_head_squared_l2',
    'checkpoint_heads','checkpoint_probabilities','first_balanced_mode_trials'])
def test_saved_arithmetic_or_switch_identity_corruption_is_rejected(field):
    cfg,parent=fixture_data()
    if field=='first_balanced_mode_trials':parent[field][0,0,4,0]+=1
    else:parent[field].flat[0]+=.03
    with pytest.raises(ValueError):study().analyze(cfg,parent)
