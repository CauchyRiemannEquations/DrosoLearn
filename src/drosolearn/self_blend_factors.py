"""Symmetric one-step accounting of recorded post-gate output updates.

Mixed corners are arithmetic on saved states/scalars/directions, never a new
learning path or a policy that receives future rewards.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from .representation_envelope import require

TOPS=('real','rewired','ann');REPS=('raw','identity')
METHODS=('shared','isolated','feedback_selective','balanced','gated_balanced','gated_self_blend')
CONTROLS=('gated_balanced','isolated','balanced','feedback_selective','shared')
GROUPS=('own','other');COMPONENTS=('state','scalar','direction','control_only_Go','self_blend_only_Go','neither_Go')
DIAGNOSTICS=('aligned_source_logit_before','correct_source_probability','logistic_sensitivity_value',
    'abs_update_factor','head_step_l2','own_unit_direction_influence','other_unit_aligned_influence_mean','direction_l2')
SCIENCE={'study_id':'self_blend_factors_v1','scope':'retrospective_descriptive_self_blend_factors',
    'module':'drosolearn.self_blend_factors','protocol':'docs/self-blend-factors-protocol.md',
    'learning_result':'results/self_blend_learning_v1','prior_plot':'results/self_blend_figures_v1/plotted_values.json',
    'baseline_tag':'self-blend-learning-v1','baseline_commit':'0d1a7abaf35e564eb397a52bb9f44a4f1b723907',
    'topologies':list(TOPS),'representations':list(REPS),'methods':list(METHODS),'contrast_controls':list(CONTROLS),
    'model_seeds':list(range(17701,17711)),'training_trials':400,'numeric_tolerance':1e-8,
    'phase_definition':'reference_self_blend_known_before_count_equals_four',
    'bridge_eligibility':'reference_post_and_both_actual_Go','bridge_bits':['state','scalar','direction'],
    'bridge_allocation':'three_factor_symmetric_Shapley_exact_sigmoid','destination_groups':list(GROUPS),
    'new_confirmatory_hypothesis':False,'external_manifest':None}


def validate_config(cfg):
    require(type(cfg) is dict,'Configuration dictionary required')
    fixture=cfg.get('scope')=='integration_test';exempt={'scope','model_seeds','training_trials'} if fixture else set()
    for key,value in SCIENCE.items():
        if key not in exempt:require(cfg.get(key)==value,'Fixed post-gate factor accounting differs: '+key)
    if fixture:
        require(type(cfg['model_seeds']) is list and 1<=len(cfg['model_seeds'])<=2
            and len(set(cfg['model_seeds']))==len(cfg['model_seeds'])
            and all(type(s) is int and 90001<=s<=90100 for s in cfg['model_seeds']),'Invented fixture seeds only')
        require(type(cfg['training_trials']) is int and 4<=cfg['training_trials']<=16,'Bounded fixture length')


def sigmoid(z):return 1./(1.+np.exp(-np.clip(z,-40.,40.)))


def bridge_cells(zC,zG,gC,gG,fC,fG,signs):
    zC,zG,gC,gG,fC,fG,signs=map(np.asarray,(zC,zG,gC,gG,fC,fG,signs))
    require(zC.ndim>=1 and zC.shape[-1]==4 and zG.shape==fC.shape==fG.shape==zC.shape
        and gC.shape==gG.shape==zC.shape[:-1] and signs.shape==(4,) and np.isin(signs,[-1,1]).all()
        and all(x.dtype.kind in 'fiu' and np.isfinite(x).all() for x in (zC,zG,gC,gG,fC,fG,signs)),
        'Finite saved one-step states, scalars, directions and signs')
    result=[]
    for bits in range(8):
        z=zG if bits&4 else zC;g=gG if bits&2 else gC;f=fG if bits&1 else fC
        result.append(signs*(sigmoid(z+g[...,None]*f)-sigmoid(z)))
    return np.stack(result,axis=-2)


def shapley3(cells):
    values=np.asarray(cells)
    require(values.ndim>=2 and values.shape[-2]==8 and values.dtype.kind in 'fiu' and np.isfinite(values).all(),
        'Eight finite real one-step cells')
    values=values.astype(float,copy=False);result=[]
    for bit in (4,2,1):
        effect=np.zeros_like(values[...,0,:])
        for subset in range(8):
            if subset&bit:continue
            weight=1./3. if subset.bit_count() in (0,2) else 1./6.
            effect+=weight*(values[...,subset|bit,:]-values[...,subset,:])
        result.append(effect)
    return np.stack(result,axis=-2)


def factual(cfg,parent):
    n=cfg['training_trials'];seeds=cfg['model_seeds'];F=(3,2,6,len(seeds),n);tol=cfg['numeric_tolerance']
    require(np.array_equal(parent['model_seeds'],seeds) and parent['trial_actions'].shape==F
        and parent['schedules'].shape==(len(seeds),n) and np.isin(parent['schedules'],range(4)).all()
        and parent['targets'].shape==(4,) and np.isin(parent['targets'],[0,1]).all(),'Fixed complete parent axes')
    require(np.isin(parent['trial_actions'],[0,1]).all(),'Recorded binary actual actions')
    q=np.empty(F+(4,));margin=q.copy();states={};residual=0.;y=2*parent['targets'].astype(int)-1
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for si in range(len(seeds)):
                d=parent['g_'+top+'_'+rep+'_unit_rows'][si]
                for mi in range(6):
                    at=(ti,ri,mi,si);g=parent['trial_update_factors'][at];w=parent['trial_directions'][at]
                    require(g.shape==(n,) and w.shape==(n,d.shape[1]) and np.isfinite(g).all()
                        and np.isfinite(w).all() and np.isfinite(d).all(),'Finite saved update ingredients')
                    head=np.concatenate((np.zeros((1,d.shape[1])),np.cumsum(g[:,None]*w,axis=0)))
                    z=head[:-1]@d.T;f=w@d.T;inc=g[:,None]*f
                    q[at]=y*(sigmoid(z+inc)-sigmoid(z));margin[at]=y*inc
                    checkhead=head[parent['checkpoints']]
                    for delta in (q[at]-parent['trial_correct_probability_delta'][at],
                        margin[at]-parent['trial_aligned_margin_delta'][at],
                        checkhead-parent['checkpoint_heads'][at],
                        sigmoid(checkhead@d.T)-parent['checkpoint_probabilities'][at]):
                        residual=max(residual,float(np.max(np.abs(delta))))
                    states[at]=(z,g,f,w)
    require(residual<=tol,'Saved factual effects and head/probe reconstruction')
    return q,margin,states,residual


def analyze(cfg,parent,prior_plot):
    validate_config(cfg);n=cfg['training_trials'];seeds=cfg['model_seeds'];tol=cfg['numeric_tolerance']
    q,m,states,residual=factual(cfg,parent);F=q.shape[:-1];P=(3,2,5,len(seeds),n)
    post=np.count_nonzero(parent['trial_known_before'][:,:,5],axis=-1)==4
    require(prior_plot_residual(cfg,q,post,parent['schedules'],prior_plot)<=tol,'Original saved primary post bookkeeping differs')
    a={'probability_changes':q,'margin_changes':m,'reference_post':post,
        'pair_action_cases':np.empty(P,dtype=np.int8),'bridge_eligible':np.empty(P,dtype=bool),
        'bridge_q_cells':np.zeros(P+(8,4)),
        'bridge_q_shapley':np.zeros(P+(3,4)),'bridge_positive_shapley':np.zeros(P+(3,4)),
        'bridge_negative_shapley':np.zeros(P+(3,4)),'factual_pair_diagnostics':np.zeros(P+(2,8)),
        'first_post_trials':np.empty((3,2,len(seeds)),dtype=np.int32)}
    for name in ('model_seeds','schedules','targets'):a[name]=parent[name].copy()
    y=2*a['targets'].astype(int)-1
    for ti in range(3):
        for ri in range(2):
            for si in range(len(seeds)):
                reference=post[ti,ri,si];opened=np.flatnonzero(reference)
                a['first_post_trials'][ti,ri,si]=int(opened[0]+1) if opened.size else n+1
                require(a['first_post_trials'][ti,ri,si]==parent['first_blend_mode_trials'][ti,ri,5,si],
                    'Recorded reference gate first-post identity')
                source=parent['schedules'][si];own=np.arange(4)==source[:,None];G=(ti,ri,5,si)
                for ci,control in enumerate(CONTROLS):
                    C=(ti,ri,METHODS.index(control),si);at=(ti,ri,ci,si)
                    cases=parent['trial_actions'][C]+2*parent['trial_actions'][G]
                    eligible=reference&(cases==3);a['pair_action_cases'][at]=cases;a['bridge_eligible'][at]=eligible
                    zC,gC,fC,wC=states[C];zG,gG,fG,wG=states[G]
                    cells=bridge_cells(zC[eligible],zG[eligible],gC[eligible],gG[eligible],fC[eligible],fG[eligible],y)
                    a['bridge_q_cells'][at][eligible]=cells
                    a['bridge_q_shapley'][at][eligible]=shapley3(cells)
                    a['bridge_positive_shapley'][at][eligible]=shapley3(np.maximum(cells,0.))
                    a['bridge_negative_shapley'][at][eligible]=shapley3(np.maximum(-cells,0.))
                    if eligible.any():
                        require(np.max(np.abs(cells[:,0]-q[C][eligible]))<=tol
                            and np.max(np.abs(cells[:,-1]-q[G][eligible]))<=tol,'Actual endpoints of crossed factual corners')
                        src=source[eligible];idx=np.arange(len(src));other=~own[eligible]
                        for j,(z,g,f,w) in enumerate(((zC,gC,fC,wC),(zG,gG,fG,wG))):
                            z,g,f,w=z[eligible],g[eligible],f[eligible],w[eligible]
                            p=sigmoid(z[idx,src]);correct=np.where(y[src]==1,p,1-p);norm=np.linalg.norm(w,axis=-1)
                            diag=np.column_stack((y[src]*z[idx,src],correct,p*(1-p),np.abs(g),np.abs(g)*norm,
                                f[idx,src],(y[src,None]*y*f*other).sum(-1)/3.,norm))
                            a['factual_pair_diagnostics'][at][eligible,j]=diag
    return a,make_rows(cfg,a)


def components_for_pair(a,ti,ri,ci,si):
    at=(ti,ri,ci,si);post=a['reference_post'][ti,ri,si];cases=a['pair_action_cases'][at]
    mi=METHODS.index(CONTROLS[ci]);qC=a['probability_changes'][ti,ri,mi,si];qG=a['probability_changes'][ti,ri,5,si]
    values=[]
    for key,op in (('bridge_q_shapley',lambda q:q),('bridge_positive_shapley',lambda q:np.maximum(q,0.)),
        ('bridge_negative_shapley',lambda q:np.maximum(-q,0.))):
        v=np.zeros((6,len(post),4));v[:3]=np.moveaxis(a[key][at],1,0)
        v[3]=-op(qC)*(post&(cases==1))[:,None];v[4]=op(qG)*(post&(cases==2))[:,None]
        values.append(v)
    eligible=int(a['bridge_eligible'][at].sum())
    counts=[eligible]*3+[int(np.count_nonzero(post&(cases==k))) for k in (1,2,0)]
    return (*values,counts)


def make_rows(cfg,a):
    rows=[]
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for ci,control in enumerate(CONTROLS):
                for si,seed in enumerate(cfg['model_seeds']):
                    v,pos,neg,counts=components_for_pair(a,ti,ri,ci,si)
                    own=np.arange(4)==a['schedules'][si,:,None];at=(ti,ri,ci,si)
                    for group,mask in zip(GROUPS,(own,~own)):
                        for component,name in enumerate(COMPONENTS):
                            raw,plus,minus=(float((x[component]*mask).sum()) for x in (v,pos,neg))
                            rows.append({'topology':top,'representation':rep,'control':control,'seed':seed,
                                'destination_group':group,'component':name,
                                'reference_post_trials':int(a['reference_post'][ti,ri,si].sum()),
                                'both_Go_trials':int(a['bridge_eligible'][at].sum()),'component_trials':counts[component],
                                'probability_signed_sum':raw,'positive_mass_allocation':plus,'negative_mass_allocation':minus,
                                'endpoint_contribution':raw/4.,'positive_endpoint_allocation':plus/4.,'negative_endpoint_allocation':minus/4.})
    return rows


def load_inputs(cfg,root):
    with np.load(Path(root)/cfg['learning_result']/'arrays.npz',allow_pickle=False) as h:
        parent={key:h[key] for key in h.files}
    prior_plot=json.loads((Path(root)/cfg['prior_plot']).read_text(encoding='utf8'))
    return parent,prior_plot


def produce(cfg,root):return analyze(cfg,*load_inputs(cfg,root))


def summarize(cfg,a,rows,root):return summarize_arrays(cfg,a,rows,*load_inputs(cfg,root))


def prior_plot_residual(cfg,q,post,schedules,prior_plot):
    require(type(prior_plot) is dict,'Original saved plotting-value dictionary required')
    records=prior_plot.get('factual_post_pathways')
    require(type(records) is list and len(records)==6 and all(type(r) is dict for r in records)
        and [r.get('method') for r in records]==list(METHODS),
        'Original six-method primary post bookkeeping required')
    residual=0.
    for mi,record in enumerate(records):
        own=[];other=[]
        for si in range(len(cfg['model_seeds'])):
            mask=np.arange(4)==schedules[si,:,None];delta=q[0,0,mi,si]*post[0,0,si,:,None]
            own.append(float((delta*mask).sum()/4.));other.append(float((delta*~mask).sum()/4.))
        for name,values in [('own_endpoint_contribution_by_seed',own),('other_endpoint_contribution_by_seed',other)]:
            original=np.asarray(record.get(name))
            require(original.shape==(len(cfg['model_seeds']),) and original.dtype.kind in 'fiu'
                and np.isfinite(original).all(),'Original finite complete seed bookkeeping')
            residual=max(residual,float(np.max(np.abs(np.asarray(values)-original))))
    return residual


MIX_RESIDUALS=('balanced_ingredient_control_residual_max','unit_other_scaling_residual_max',
    'unit_own_mixture_residual_max','raw_norm_residual_max','mixed_direction_equation_residual_max')


def mixture_geometry(cfg,a,parent):
    """Same actual both-Go source: decode, do not optimize, the saved mixture."""
    records=[];tol=cfg['numeric_tolerance']
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            values=[]
            for si,seed in enumerate(cfg['model_seeds']):
                e=a['bridge_eligible'][ti,ri,0,si];C=(ti,ri,4,si);G=(ti,ri,5,si)
                record={'seed':seed,'eligible_both_Go_trials':int(e.sum()),'has_both_Go_evidence':bool(e.any())}
                for name in ('raw_norm_mean','balanced_weight_after_normalization_mean','balanced_weight_after_normalization_min',
                    'balanced_weight_after_normalization_max','self_weight_after_normalization_mean'):record[name]=None
                record.update({name:0. for name in MIX_RESIDUALS})
                if e.any():
                    d=parent['g_'+top+'_'+rep+'_unit_rows'][si];src=parent['schedules'][si,e];idx=np.arange(len(src))
                    i=parent['trial_blend_isolated_directions'][G][e];b=parent['trial_blend_balanced_directions'][G][e]
                    w=parent['trial_directions'][G][e];c=parent['trial_directions'][C][e]
                    norm=parent['trial_blend_raw_norm'][G][e];m=.75*i+.25*b
                    require(np.all(parent['trial_blend_applicable'][G][e]) and np.all(norm>0.)
                        and np.all(np.isfinite(norm)),'Saved actual post both-Go mixture only')
                    coef=.25/norm;fG=w@d.T;fC=c@d.T;fI=i@d.T;other=np.arange(4)[None,:]!=src[:,None]
                    record.update(raw_norm_mean=float(norm.mean()),balanced_weight_after_normalization_mean=float(coef.mean()),
                        balanced_weight_after_normalization_min=float(coef.min()),balanced_weight_after_normalization_max=float(coef.max()),
                        self_weight_after_normalization_mean=float((.75/norm).mean()),
                        balanced_ingredient_control_residual_max=float(np.max(np.abs(b-c))),
                        unit_other_scaling_residual_max=float(np.max(np.abs((fG-coef[:,None]*fC)[other]))),
                        unit_own_mixture_residual_max=float(np.max(np.abs(fG[idx,src]-(.75/norm)*fI[idx,src]-coef*fC[idx,src]))),
                        raw_norm_residual_max=float(np.max(np.abs(norm-np.linalg.norm(m,axis=-1)))),
                        mixed_direction_equation_residual_max=float(np.max(np.abs(w-m/norm[:,None]))))
                    require(all(record[key]<=tol for key in MIX_RESIDUALS),'Recorded same-state unit mixture geometry differs')
                values.append(record)
            records.append({'topology':top,'representation':rep,'seeds':values})
    return records


def mean_nested(values):
    if isinstance(values[0],list):return [mean_nested(list(v)) for v in zip(*values)]
    defined=[v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def summarize_arrays(cfg,a,rows,parent,prior_plot):
    validate_config(cfg);tol=cfg['numeric_tolerance'];q=a['probability_changes'];post=a['reference_post'];families=[]
    factual_res=max(float(np.max(np.abs(q-parent['trial_correct_probability_delta']))),
        float(np.max(np.abs(a['margin_changes']-parent['trial_aligned_margin_delta']))),factual(cfg,parent)[3])
    groupres=componentres=0.
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for ci,control in enumerate(CONTROLS):
                records=[];mi=METHODS.index(control)
                for si,seed in enumerate(cfg['model_seeds']):
                    at=(ti,ri,ci,si);C=(ti,ri,mi,si);G=(ti,ri,5,si);pm=post[ti,ri,si]
                    own=np.arange(4)==a['schedules'][si,:,None];v,pos,neg,counts=components_for_pair(a,ti,ri,ci,si)
                    component=[];cpos=[];cneg=[];fact=[];fpos=[];fneg=[]
                    for mask in (own,~own):
                        mask=mask*pm[:,None]
                        component.append((v*mask).sum((1,2))/4.);cpos.append((pos*mask).sum((1,2))/4.)
                        cneg.append((neg*mask).sum((1,2))/4.)
                        fact.append(float(((q[G]-q[C])*mask).sum()/4.))
                        fpos.append(float(((np.maximum(q[G],0.)-np.maximum(q[C],0.))*mask).sum()/4.))
                        fneg.append(float(((np.maximum(-q[G],0.)-np.maximum(-q[C],0.))*mask).sum()/4.))
                        prev=((parent['trial_correct_probability_delta'][G]-parent['trial_correct_probability_delta'][C])*mask).sum()/4.
                        groupres=max(groupres,abs(fact[-1]-prev))
                    component=np.array(component);cpos=np.array(cpos);cneg=np.array(cneg)
                    componentres=max(componentres,float(np.max(np.abs(component.sum(-1)-fact))),
                        float(np.max(np.abs(cpos.sum(-1)-fpos))),float(np.max(np.abs(cneg.sum(-1)-fneg))))
                    eligible=a['bridge_eligible'][at];diag=a['factual_pair_diagnostics'][at][eligible]
                    bothfull=(np.count_nonzero(parent['trial_known_before'][C],axis=-1)==4)
                    records.append({'seed':seed,'reference_post_trials':int(pm.sum()),'both_Go_trials':int(eligible.sum()),
                        'case_counts':[int(np.count_nonzero(pm&(a['pair_action_cases'][at]==case))) for case in range(4)],
                        'component_endpoint_contributions':component.tolist(),
                        'component_positive_endpoint_allocations':cpos.tolist(),'component_negative_endpoint_allocations':cneg.tolist(),
                        'factual_post_endpoint_difference':fact,'factual_post_positive_endpoint_difference':fpos,
                        'factual_post_negative_endpoint_difference':fneg,'factual_post_total_difference':float(sum(fact)),
                        'diagnostic_means':diag.mean(0).tolist() if len(diag) else [[None]*8 for _ in range(2)],
                        'both_methods_full_before_on_eligible_trials':int(np.count_nonzero(eligible&bothfull)),
                        'control_partial_before_on_eligible_trials':int(np.count_nonzero(eligible&~bothfull))})
                means={key:mean_nested([r[key] for r in records]) for key in records[0] if key!='seed'}
                families.append({'topology':top,'representation':rep,'control':control,
                    'role':'primary' if (ti,ri,ci)==(0,0,0) else 'descriptive_control','seeds':records,'means':means})
    cells=a['bridge_q_cells'];plus=np.maximum(cells,0.);minus=np.maximum(-cells,0.);eligible=a['bridge_eligible']
    geometry=mixture_geometry(cfg,a,parent)
    residuals={'prior_primary_post_bookkeeping':prior_plot_residual(cfg,q,post,parent['schedules'],prior_plot),
        'saved_unit_mixture_geometry':max(r[key] for f in geometry for r in f['seeds'] for key in MIX_RESIDUALS),
        'factual_parent_effects':float(factual_res),'post_group_endpoint_reconstruction':float(groupres),
        'shapley_signed_efficiency':float(np.max(np.abs(a['bridge_q_shapley'].sum(-2)-(cells[...,-1,:]-cells[...,0,:])))),
        'shapley_positive_efficiency':float(np.max(np.abs(a['bridge_positive_shapley'].sum(-2)-(plus[...,-1,:]-plus[...,0,:])))),
        'shapley_negative_efficiency':float(np.max(np.abs(a['bridge_negative_shapley'].sum(-2)-(minus[...,-1,:]-minus[...,0,:])))),
        'positive_negative_allocation_identity':float(np.max(np.abs(a['bridge_q_shapley']-a['bridge_positive_shapley']+a['bridge_negative_shapley']))),
        'post_component_reconstruction':float(componentres),
        'zero_padded_ineligible':float(max(np.max(np.abs(a[key][~eligible]),initial=0.) for key in (
            'bridge_q_cells','bridge_q_shapley','bridge_positive_shapley','bridge_negative_shapley','factual_pair_diagnostics')))}
    require(all(np.isfinite(v) and v<=tol for v in residuals.values()),'Exact post-gate factor accounting residual failed')
    require(len(a)==14 and len(rows)==3*2*5*len(cfg['model_seeds'])*2*6,'Complete factor accounting closure')
    return {'study_id':cfg['study_id'],'status':'completed','scope':cfg['scope'],'families':families,
        'mixture_geometry':geometry,'group_order':list(GROUPS),'component_order':list(COMPONENTS),'diagnostic_order':list(DIAGNOSTICS),'control_order':list(CONTROLS),
        'validation':{'all_exact_accounting_gates_passed':True,'residuals':residuals,
            'common_post_uses_prefeedback_memory':True,'only_actual_both_Go_crossed':True,'own_other_partition':True,
            'all_action_cases_retained':True,'tiny_signs_preserved':True,'same_recorded_fact_corners_match':True,'no_future_policy_or_new_training':True},
        'counts':{'analyzed_parent_trial_records':int(np.prod(q.shape[:-1])),'analyzed_destination_effects':int(q.size),
            'paired_trial_records':int(np.prod(eligible.shape)),'reference_gate_paths':int(np.prod(post.shape[:-1])),
            'comparison_paths':int(np.prod(eligible.shape[:-1])),'measurement_records':len(rows),'numeric_arrays':len(a),
            'paired_post_trial_records':int(post.sum()*5),'eligible_both_Go_trials':int(eligible.sum()),
            'corner_destination_effects':int(eligible.sum()*32),'new_training_trials':0,'new_head_update_calls':0,
            'new_internal_update_calls':0,'new_value_update_calls':0,'new_random_draws':0,'fresh_model_constructions':0,'new_optimization_calls':0},
        'interpretation':{'retrospective_descriptive_only':True,'parent_H_gain':False,'parent_H_acquisition':False,'parent_H_protection':True,
            'new_confirmatory_hypothesis':False,'accounting_allocation_not_unique_causal_explanation':True,
            'mixed_one_step_cells_not_relearned_policy':True,'unpaired_actions_not_assigned_to_zero_direction':True,
            'same_realized_movement_budget_claimed':False,'internal_synapses_and_probe_activity_fixed':True,
            'observed_memory_is_separate_from_output_weights':True,'novel_input_generalization_tested':False,
            'new_human_experiment':False,'shared_neural_mechanism_claim':False,'education_effect_claim':False}}
