"""Saved self-blend factor algebra on explicit invented paths, never a cohort."""
from copy import deepcopy
import numpy as np
import pytest


def study():
    import drosolearn.self_blend_factors as module
    return module


def sigmoid(z):return 1./(1.+np.exp(-np.clip(z,-40.,40.)))


def invented_saved_paths(gate_never=False):
    """Twelve handmade steps, one fake seed, no actor/RNG/solver construction."""
    n=12;shape=(3,2,6,1,n);signs=np.array([1,1,-1,-1])
    cfg=deepcopy(study().SCIENCE)
    cfg.update(scope='integration_test',model_seeds=[90091],training_trials=n)
    p={'model_seeds':np.array([90091]),'targets':np.array([1,1,0,0]),
        'schedules':np.array([[0,1,2,3]*3],np.int8),'checkpoints':np.array([0,4,8,12])}
    for name in ['trial_update_factors','trial_head_steps','trial_head_step_squared_l2',
        'trial_cumulative_head_path_l2','trial_cumulative_head_squared_l2','trial_probabilities','trial_blend_raw_norm']:
        p[name]=np.zeros(shape)
    for name in ['trial_actions','trial_rewards']:p[name]=np.zeros(shape,np.int8)
    for name in ['trial_directions','trial_actual_influence','trial_correct_probability_delta','trial_aligned_margin_delta',
        'trial_blend_isolated_directions','trial_blend_balanced_directions']:p[name]=np.zeros(shape+(4,))
    for name in ['trial_known_before','trial_known_after']:p[name]=np.zeros(shape+(4,),np.int8)
    p['trial_blend_applicable']=np.zeros(shape,bool)
    p['checkpoint_heads']=np.zeros(shape[:-1]+(4,4));p['checkpoint_logits']=np.zeros(shape[:-1]+(4,4))
    p['checkpoint_probabilities']=np.full(shape[:-1]+(4,4),.5)
    for name in ['first_balanced_mode_trials','first_blend_mode_trials']:p[name]=np.full(shape[:-1],n+1,np.int32)
    for ti,top in enumerate(['real','rewired','ann']):
        for ri,rep in enumerate(['raw','identity']):
            p['g_'+top+'_'+rep+'_unit_rows']=np.eye(4)[None]
            for mi in range(6):
                head=np.zeros(4);known=np.zeros(4,np.int8);path=energy=0.
                for t,source in enumerate(p['schedules'][0]):
                    at=(ti,ri,mi,0,t);old=head.copy();before=known.copy();full=np.count_nonzero(before)==4
                    action=1
                    if mi==0:action=0
                    elif mi==1 and t in (4,7):action=0
                    elif mi==2 and t in (3,7):action=0
                    elif mi==3 and t in (5,9):action=0
                    elif mi==4 and t in (4,7):action=0
                    elif mi==5 and t in (5,7):action=0
                    if mi==5 and gate_never:action=0
                    prob=float(sigmoid(head[source]));reward=int(signs[source]) if action else 0
                    if action:known[source]=reward
                    isolated=np.eye(4)[source].copy();balanced=isolated*.8
                    for j in range(4):
                        if j!=source and known[j]:balanced[j]=.2*known[source]*known[j]
                    balanced/=np.linalg.norm(balanced)
                    direction=isolated.copy()
                    if mi in (2,3) or (mi==4 and full):direction=balanced.copy()
                    if mi==5 and full:
                        if action:
                            raw=.75*isolated+.25*balanced;length=float(np.linalg.norm(raw));direction=raw/length
                            p['trial_blend_applicable'][at]=True;p['trial_blend_raw_norm'][at]=length
                            p['trial_blend_isolated_directions'][at]=isolated
                            p['trial_blend_balanced_directions'][at]=balanced
                        else:direction[:]=0.
                    if not action and (mi==3 or (mi==4 and full)):direction[:]=0.
                    factor=.1*(action-prob)*reward;delta=signs*(sigmoid(old+factor*direction)-sigmoid(old))
                    head+=factor*direction;step=abs(factor)*float(np.linalg.norm(direction));path+=step;energy+=step**2
                    for name,value in [('trial_actions',action),('trial_rewards',reward),('trial_probabilities',prob),
                        ('trial_update_factors',factor),('trial_directions',direction),('trial_actual_influence',direction),
                        ('trial_head_steps',step),('trial_head_step_squared_l2',step**2),('trial_cumulative_head_path_l2',path),
                        ('trial_cumulative_head_squared_l2',energy),('trial_known_before',before),('trial_known_after',known),
                        ('trial_correct_probability_delta',delta),('trial_aligned_margin_delta',signs*factor*direction)]:p[name][at]=value
                    if mi in (4,5) and full:
                        name='first_balanced_mode_trials' if mi==4 else 'first_blend_mode_trials'
                        if p[name][ti,ri,mi,0]==n+1:p[name][ti,ri,mi,0]=t+1
                    if t+1 in (4,8,12):
                        ki=(4,8,12).index(t+1)+1;ix=(ti,ri,mi,0,ki)
                        p['checkpoint_heads'][ix]=head;p['checkpoint_logits'][ix]=head;p['checkpoint_probabilities'][ix]=sigmoid(head)
    phase=np.count_nonzero(p['trial_known_before'][0,0,5,0],axis=-1)==4
    own=np.arange(4)==p['schedules'][0,:,None]
    prior_plot={'factual_post_pathways':[]}
    for mi,method in enumerate(['shared','isolated','feedback_selective','balanced','gated_balanced','gated_self_blend']):
        q=p['trial_correct_probability_delta'][0,0,mi,0]
        prior_plot['factual_post_pathways'].append({'method':method,
            'own_endpoint_contribution_by_seed':[float(q[phase[:,None]&own].sum()/4.)],
            'other_endpoint_contribution_by_seed':[float(q[phase[:,None]&~own].sum()/4.)]})
    return cfg,p,prior_plot


@pytest.fixture
def bridge_arguments():
    return (np.array([.1,-.3,.4,-.2]),np.array([.4,-.1,.2,-.5]),np.asarray(.04),np.asarray(-.03),
        np.array([1.,.2,-.1,.3]),np.array([.8,.3,-.2,.1]),np.array([1,1,-1,-1]))


def test_all_eight_exact_cells_use_the_declared_state_scalar_direction_bits(bridge_arguments):
    zc,zn,gc,gn,fc,fn,y=bridge_arguments;cells=study().bridge_cells(*bridge_arguments)
    assert cells.shape==(8,4)
    for code in range(8):
        z=zn if code&4 else zc;g=gn if code&2 else gc;f=fn if code&1 else fc
        assert np.allclose(cells[code],y*(sigmoid(z+g*f)-sigmoid(z)),atol=1e-14)


def test_crossed_cells_preserve_batches_and_leave_saved_values_unchanged(bridge_arguments):
    original=[x.copy() for x in bridge_arguments];zc,zn,gc,gn,fc,fn,y=bridge_arguments
    batch=study().bridge_cells(np.stack([zc,zn]),np.stack([zn,zc]),np.array([gc,gn]),np.array([gn,gc]),
        np.stack([fc,fn]),np.stack([fn,fc]),y)
    assert batch.shape==(2,8,4) and np.allclose(batch[0,::-1],batch[1])
    for value,old in zip(bridge_arguments,original):assert np.array_equal(value,old)


@pytest.mark.parametrize('bit,index',[(4,0),(2,1),(1,2)])
def test_single_factor_stays_in_its_own_allocation(bit,index):
    cells=np.array([[1.,-2.,3.,-4.] if code&bit else [0.,0.,0.,0.] for code in range(8)])
    result=study().shapley3(cells)
    assert np.allclose(result[index],cells[bit]) and not np.delete(result,index,axis=0).any()


def test_pair_and_triple_interactions_are_symmetric_and_integer_safe():
    cells=np.zeros((8,4),int);cells[[6,7]]=[2,-4,6,-8]
    split=study().shapley3(cells)
    assert np.allclose(split[0],cells[7]/2.) and np.allclose(split[1],cells[7]/2.) and not split[2].any()
    cells[:]=0;cells[7]=[3,-6,9,-12]
    assert np.allclose(study().shapley3(cells),np.broadcast_to(cells[7]/3.,(3,4)))


def test_help_and_harm_are_split_before_allocation_and_can_have_negative_allocations(bridge_arguments):
    cells=study().bridge_cells(*bridge_arguments);signed=study().shapley3(cells)
    positive=study().shapley3(np.maximum(cells,0.));negative=study().shapley3(np.maximum(-cells,0.))
    assert np.allclose(signed,positive-negative,atol=1e-14)
    assert np.allclose(signed.sum(0),cells[7]-cells[0],atol=1e-14)
    assert np.allclose(positive.sum(0),np.maximum(cells[7],0.)-np.maximum(cells[0],0.))
    assert np.allclose(negative.sum(0),np.maximum(-cells[7],0.)-np.maximum(-cells[0],0.))
    assert np.any(positive<0.) or np.any(negative<0.)
    assert not np.allclose(negative,np.maximum(-signed,0.))


def test_a_better_saturated_state_can_have_a_negative_state_allocation():
    cells=study().bridge_cells(np.zeros(4),np.full(4,8.),np.asarray(.1),np.asarray(.1),np.ones(4),np.ones(4),np.ones(4,int))
    split=study().shapley3(cells)
    assert np.all(sigmoid(np.full(4,8.))>.5) and np.all(split[0]<0.) and not split[1:].any()


def test_current_fourth_discovery_is_pre_and_all_actual_post_action_cases_survive():
    cfg,parent,plot=invented_saved_paths();a,rows=study().analyze(cfg,parent,plot)
    assert len(a)==14 and len(rows)==360
    assert a['first_post_trials'][0,0,0]==5
    assert a['reference_post'][0,0,0].tolist()==[False]*4+[True]*8
    assert a['pair_action_cases'][0,0,0,0,4:8].tolist()==[2,1,3,0]
    assert np.array_equal(a['bridge_eligible'],a['reference_post'][:,:,None]&(a['pair_action_cases']==3))
    for key in ['bridge_q_cells','bridge_q_shapley','bridge_positive_shapley','bridge_negative_shapley','factual_pair_diagnostics']:
        assert not a[key][~a['bridge_eligible']].any()


def test_all30_families_reconstruct_signed_positive_negative_and_one_sided_factual_totals():
    cfg,parent,plot=invented_saved_paths();a,rows=study().analyze(cfg,parent,plot)
    s=study().summarize_arrays(cfg,a,rows,parent,plot)
    assert len(s['families'])==30 and s['control_order']==['gated_balanced','isolated','balanced','feedback_selective','shared']
    for family in s['families']:
        seed=family['seeds'][0];parts=np.asarray(seed['component_endpoint_contributions'])
        positive=np.asarray(seed['component_positive_endpoint_allocations']);negative=np.asarray(seed['component_negative_endpoint_allocations'])
        assert np.allclose(parts,positive-negative,atol=1e-14)
        assert np.allclose(parts.sum(1),seed['factual_post_endpoint_difference'],atol=1e-14)
        assert parts.sum()==pytest.approx(seed['factual_post_total_difference'])
        assert seed['case_counts']==[int(x) for x in seed['case_counts']] and sum(seed['case_counts'])==8
    primary=s['families'][0]['seeds'][0];parts=np.asarray(primary['component_endpoint_contributions'])
    assert parts[0,3]<0 and parts[0,4]>0 and not parts[:,5].any()
    assert s['component_order'][4]=='self_blend_only_Go'
    for row in rows:
        assert row['endpoint_contribution']==pytest.approx(row['probability_signed_sum']/4.)
        assert row['probability_signed_sum']==pytest.approx(row['positive_mass_allocation']-row['negative_mass_allocation'])


def test_actual_corner_probabilities_and_primary_conditional_diagnostics_match_saved_prefix():
    cfg,parent,plot=invented_saved_paths();a,_=study().analyze(cfg,parent,plot)
    for ci,control in enumerate([4,1,3,2,0]):
        mask=a['bridge_eligible'][:,:,ci]
        assert np.allclose(a['bridge_q_cells'][:,:,ci][mask,0],a['probability_changes'][:,:,control][mask])
        assert np.allclose(a['bridge_q_cells'][:,:,ci][mask,7],a['probability_changes'][:,:,5][mask])
    at=(0,0,0,0,6);source=2;assert a['bridge_eligible'][at]
    for side,mi in enumerate([4,5]):
        base=(0,0,mi,0);g=parent['trial_update_factors'][base];w=parent['trial_directions'][base]
        old=(g[:6,None]*w[:6]).sum(0);p=float(sigmoid(old[source]));direction=w[6];other=np.arange(4)!=source
        expected=[-old[source],1-p,p*(1-p),abs(g[6]),abs(g[6])*np.linalg.norm(direction),direction[source],
            np.mean(-np.array([1,1,-1,-1])[other]*direction[other]),np.linalg.norm(direction)]
        assert np.allclose(a['factual_pair_diagnostics'][at+(side,)],expected,atol=1e-14)


def test_partial_control_and_empty_pairs_are_explicit_not_silently_equalized():
    cfg,parent,plot=invented_saved_paths();a,rows=study().analyze(cfg,parent,plot);s=study().summarize_arrays(cfg,a,rows,parent,plot)
    selective=next(f for f in s['families'] if f['control']=='feedback_selective')
    assert selective['seeds'][0]['control_partial_before_on_eligible_trials']>0
    shared=next(f for f in s['families'] if f['control']=='shared')
    assert shared['seeds'][0]['both_Go_trials']==0 and shared['seeds'][0]['diagnostic_means']==[[None]*8,[None]*8]


def test_never_opening_reference_keeps_zero_post_null_diagnostics_and_sentinel():
    cfg,parent,plot=invented_saved_paths(gate_never=True);a,rows=study().analyze(cfg,parent,plot);s=study().summarize_arrays(cfg,a,rows,parent,plot)
    assert np.all(a['first_post_trials']==13) and not a['reference_post'].any() and not a['bridge_eligible'].any()
    assert len(rows)==360 and all(row['component_trials']==0 for row in rows)
    for family in s['families']:
        assert family['seeds'][0]['factual_post_total_difference']==0.
        assert family['seeds'][0]['diagnostic_means']==[[None]*8,[None]*8]


def test_saved_inputs_unchanged_and_no_rng_or_optimizer_is_created(monkeypatch):
    cfg,parent,plot=invented_saved_paths();old={key:v.copy() for key,v in parent.items()};oldplot=deepcopy(plot)
    def forbidden(*args,**kwargs):raise AssertionError('Saved factor arithmetic cannot learn, draw randomness or optimize')
    monkeypatch.setattr(np.random,'default_rng',forbidden);monkeypatch.setattr(np.linalg,'svd',forbidden)
    a,rows=study().analyze(cfg,parent,plot);s=study().summarize_arrays(cfg,a,rows,parent,plot)
    for key,v in old.items():assert np.array_equal(parent[key],v),key
    assert plot==oldplot
    for name in ['new_training_trials','new_head_update_calls','new_internal_update_calls','new_value_update_calls','new_random_draws','fresh_model_constructions','new_optimization_calls']:assert s['counts'][name]==0
    assert s['interpretation']['parent_H_gain'] is s['interpretation']['parent_H_acquisition'] is False
    assert s['interpretation']['parent_H_protection'] is True


@pytest.mark.parametrize('key,value',[('model_seeds',[17701]),('training_trials',800),('numeric_tolerance',.1),
    ('bridge_eligibility','all_trials'),('phase_definition','known_after_count_equals_four'),('new_confirmatory_hypothesis',True),
    ('baseline_commit','0'*40),('contrast_controls',['isolated']),('bridge_bits',['direction','scalar','state']),('destination_groups',['other'])])
def test_frozen_descriptive_scope_cannot_select_favorable_budget_or_records(key,value):
    cfg=deepcopy(study().SCIENCE);cfg[key]=value
    with pytest.raises(ValueError):study().validate_config(cfg)


@pytest.mark.parametrize('mutation',['state_shape','direction_shape','scalar_shape','nan_state','infinite_scalar','complex_direction','invalid_sign','string_sign'])
def test_bridge_rejects_invalid_axes_nonfinite_or_nonreal_values(bridge_arguments,mutation):
    values=[x.copy() for x in bridge_arguments]
    if mutation=='state_shape':values[1]=values[1][:3]
    elif mutation=='direction_shape':values[5]=values[5][:3]
    elif mutation=='scalar_shape':values[2]=np.array([.1])
    elif mutation=='nan_state':values[0][0]=np.nan
    elif mutation=='infinite_scalar':values[2]=np.asarray(np.inf)
    elif mutation=='complex_direction':values[5]=values[5].astype(complex)
    elif mutation=='invalid_sign':values[6][0]=0
    else:values[6]=values[6].astype(str)
    with pytest.raises(ValueError):study().bridge_cells(*values)


@pytest.mark.parametrize('cells',[np.zeros(8),np.zeros((7,4)),np.full((8,4),np.nan),np.full((8,4),np.inf),np.zeros((8,4),complex),np.full((8,4),'0')])
def test_shapley_rejects_bad_tables(cells):
    with pytest.raises(ValueError):study().shapley3(cells)


@pytest.mark.parametrize('key',['trial_correct_probability_delta','trial_aligned_margin_delta','checkpoint_heads','checkpoint_probabilities','first_blend_mode_trials'])
def test_corrupted_actual_effect_checkpoint_or_reference_switch_is_rejected(key):
    cfg,parent,plot=invented_saved_paths();parent[key].flat[0]+=1
    if key=='first_blend_mode_trials':parent[key][0,0,5,0]+=1
    with pytest.raises(ValueError):study().analyze(cfg,parent,plot)


def test_changed_previously_published_post_ledger_is_rejected():
    cfg,parent,plot=invented_saved_paths();plot['factual_post_pathways'][5]['own_endpoint_contribution_by_seed'][0]+=.01
    with pytest.raises(ValueError):study().analyze(cfg,parent,plot)


def test_saved_mixture_coefficients_and_five_equations_use_same_both_go_events_only():
    cfg,parent,plot=invented_saved_paths();a,_=study().analyze(cfg,parent,plot)
    records=study().mixture_geometry(cfg,a,parent)
    assert len(records)==6
    assert study().MIX_RESIDUALS==('balanced_ingredient_control_residual_max','unit_other_scaling_residual_max',
        'unit_own_mixture_residual_max','raw_norm_residual_max','mixed_direction_equation_residual_max')
    for ti in range(3):
        for ri in range(2):
            record=records[ti*2+ri]['seeds'][0];mask=a['bridge_eligible'][ti,ri,0,0]
            norm=parent['trial_blend_raw_norm'][ti,ri,5,0][mask]
            assert record['has_both_Go_evidence'] and record['eligible_both_Go_trials']==mask.sum()
            assert record['raw_norm_mean']==pytest.approx(norm.mean())
            assert record['balanced_weight_after_normalization_mean']==pytest.approx((.25/norm).mean())
            assert record['balanced_weight_after_normalization_min']==pytest.approx((.25/norm).min())
            assert record['balanced_weight_after_normalization_max']==pytest.approx((.25/norm).max())
            assert record['self_weight_after_normalization_mean']==pytest.approx((.75/norm).mean())
            assert all(record[key]<=1e-8 for key in study().MIX_RESIDUALS)
            assert np.array_equal(parent['trial_blend_balanced_directions'][ti,ri,5,0][mask],parent['trial_directions'][ti,ri,4,0][mask])


def test_missing_both_go_evidence_is_null_not_zero_coefficient_or_claimed_identity():
    cfg,parent,plot=invented_saved_paths(gate_never=True);a,_=study().analyze(cfg,parent,plot)
    for family in study().mixture_geometry(cfg,a,parent):
        record=family['seeds'][0]
        assert not record['has_both_Go_evidence'] and record['eligible_both_Go_trials']==0
        assert all(record[key] is None for key in ['raw_norm_mean','balanced_weight_after_normalization_mean',
            'balanced_weight_after_normalization_min','balanced_weight_after_normalization_max','self_weight_after_normalization_mean'])
        assert all(record[key]==0. for key in study().MIX_RESIDUALS)


@pytest.mark.parametrize('field',['trial_blend_balanced_directions','trial_blend_raw_norm',
    'trial_blend_isolated_directions','trial_blend_applicable','trial_directions'])
def test_corrupted_stored_same_event_mixture_evidence_is_rejected(field):
    cfg,parent,plot=invented_saved_paths();a,_=study().analyze(cfg,parent,plot);at=(0,0,5,0,6)
    if field=='trial_blend_applicable':parent[field][at]=False
    elif field=='trial_blend_raw_norm':parent[field][at]+=.1
    else:parent[field][at+(0,)]+=.1
    with pytest.raises(ValueError):study().mixture_geometry(cfg,a,parent)


def test_direction_identity_audit_does_not_expand_to_a_one_sided_go_event():
    cfg,parent,plot=invented_saved_paths();a,_=study().analyze(cfg,parent,plot)
    expected=study().mixture_geometry(cfg,a,parent)
    assert a['pair_action_cases'][0,0,0,0,4]==2 and not a['bridge_eligible'][0,0,0,0,4]
    # Its bytes remain protected by the parent hash, but this particular identity
    # audit makes no claim about the absent control action at this trial.
    parent['trial_blend_balanced_directions'][0,0,5,0,4,0]+=.1
    assert study().mixture_geometry(cfg,a,parent)==expected


@pytest.mark.parametrize('mutation',['method_order','missing_method','bad_seed_length','nonfinite'])
def test_parent_plot_binding_rejects_reordered_incomplete_or_nonfinite_ledger(mutation):
    cfg,parent,plot=invented_saved_paths();records=plot['factual_post_pathways']
    if mutation=='method_order':records[0],records[1]=records[1],records[0]
    elif mutation=='missing_method':records.pop()
    elif mutation=='bad_seed_length':records[0]['own_endpoint_contribution_by_seed']=[]
    else:records[0]['own_endpoint_contribution_by_seed']=[np.nan]
    with pytest.raises(ValueError):study().analyze(cfg,parent,plot)
