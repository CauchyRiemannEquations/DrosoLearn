"""Exact descriptive accounting of previously completed balanced learning.

No learner, new graph, RNG, alternative routing solver or future policy is run.
Crossed one-step cells are arithmetic on already recorded states and directions.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .representation_envelope import require

TOPS=('real','rewired','ann');REPS=('raw','identity')
METHODS=('shared','isolated','feedback_selective','balanced')
PHASES=('partial','full');CATEGORIES=('own','known_other','unknown_other')
SCIENCE={'study_id':'balanced_pathways_v1','scope':'retrospective_descriptive_balanced_pathways',
    'module':'drosolearn.balanced_pathways','protocol':'docs/balanced-pathways-protocol.md',
    'parent_result':'results/balanced_learning_v1','baseline_tag':'balanced-learning-v1',
    'baseline_commit':'b836a780b91d81425e534dad7284485f284799ce',
    'topologies':list(TOPS),'representations':list(REPS),'methods':list(METHODS),
    'model_seeds':list(range(17501,17511)),'training_trials':400,'numeric_tolerance':1e-8,
    'phase_definition':'known_before_count_equals_four','category_definition':'own_known_before_other_unknown_before_other',
    'bridge_eligibility':'both_actual_Go_only','bridge_bits':['state','scalar','direction'],
    'bridge_allocation':'three_factor_symmetric_Shapley_exact_sigmoid',
    'objective_eligibility':'both_actual_Go_and_both_known_before_four',
    'new_confirmatory_hypothesis':False,'external_manifest':None}


def validate_config(cfg):
    fixture=cfg.get('scope')=='integration_test'
    exempt={'scope','model_seeds','training_trials'} if fixture else set()
    for key,value in SCIENCE.items():
        if key not in exempt:require(cfg.get(key)==value,'Fixed saved-path design differs: '+key)
    if fixture:
        require(type(cfg['model_seeds']) is list and 1<=len(cfg['model_seeds'])<=2
            and len(set(cfg['model_seeds']))==len(cfg['model_seeds'])
            and all(type(s) is int and 90001<=s<=90100 for s in cfg['model_seeds']), 'Invented fixture seeds only')
        require(type(cfg['training_trials']) is int and 4<=cfg['training_trials']<=16,'Bounded fixture length')


def sigmoid(z):return 1./(1.+np.exp(-np.clip(z,-40.,40.)))


def probability_effects(D,heads_before,g,w,signs):
    D,heads_before,g,w,signs=map(np.asarray,(D,heads_before,g,w,signs))
    require(D.ndim==2 and D.shape[0]==4 and heads_before.ndim>=1 and heads_before.shape[-1]==D.shape[1]
        and w.shape==heads_before.shape and g.shape==heads_before.shape[:-1] and signs.shape==(4,)
        and np.isin(signs,[-1,1]).all() and all(x.dtype.kind in 'fiu' and np.isfinite(x).all()
            for x in (D,heads_before,g,w,signs)),'Finite real factual effect axes and target signs')
    before=heads_before@D.T
    margin=g[...,None]*(w@D.T)
    return signs*(sigmoid(before+margin)-sigmoid(before)),signs*margin


def decompose_categories(known_before,sources):
    known=np.asarray(known_before);sources=np.asarray(sources)
    require(known.ndim>=1 and known.shape[-1]==4 and known.dtype.kind in 'biu' and np.isin(known,[-1,0,1]).all()
        and sources.dtype.kind in 'iu' and sources.shape==known.shape[:-1] and np.isin(sources,range(4)).all(),
        'Known-before integer axes and source indices')
    result=np.where(known==0,2,1).astype(np.int8)
    return np.where(np.arange(4)==sources[...,None],0,result).astype(np.int8)


def bridge_cells(zS,zB,gS,gB,fS,fB,signs):
    zS,zB,gS,gB,fS,fB,signs=map(np.asarray,(zS,zB,gS,gB,fS,fB,signs))
    require(zS.ndim>=1 and zS.shape[-1]==4 and zB.shape==fS.shape==fB.shape==zS.shape
        and gS.shape==gB.shape==zS.shape[:-1] and signs.shape==(4,) and np.isin(signs,[-1,1]).all()
        and all(x.dtype.kind in 'fiu' and np.isfinite(x).all() for x in (zS,zB,gS,gB,fS,fB,signs)),
        'Finite real paired one-step axes and target signs')
    result=[]
    for bits in range(8):
        z=np.asarray(zB if bits&4 else zS);g=np.asarray(gB if bits&2 else gS)
        f=np.asarray(fB if bits&1 else fS)
        result.append(np.asarray(signs)*(sigmoid(z+g[...,None]*f)-sigmoid(z)))
    return np.stack(result,axis=-2)


def shapley3(cells):
    values=np.asarray(cells);require(values.ndim>=2 and values.shape[-2]==8
        and values.dtype.kind in 'fiu' and np.isfinite(values).all(),'Eight finite real one-step cells')
    values=values.astype(float,copy=False)
    result=[]
    for bit in (4,2,1):
        effect=np.zeros_like(values[...,0,:])
        for subset in range(8):
            if subset&bit:continue
            weight=1./3. if subset.bit_count() in (0,2) else 1./6.
            effect+=weight*(values[...,subset|bit,:]-values[...,subset,:])
        result.append(effect)
    return np.stack(result,axis=-2)


def analyze(cfg,parent):
    validate_config(cfg);n=cfg['training_trials'];seeds=cfg['model_seeds'];tol=cfg['numeric_tolerance']
    F=(3,2,4,len(seeds),n);P=(3,2,len(seeds),n);y=2*parent['targets']-1
    require(np.array_equal(parent['model_seeds'],seeds) and parent['trial_actions'].shape==F,'Fixed complete parent paths')
    a={name:np.empty(F+(4,),dtype=np.int8 if name=='effect_categories' else float)
        for name in ('probability_changes','margin_changes','effect_categories')}
    a['phase_before']=np.empty(F,dtype=np.int8);a['first_full_trials']=np.empty(F[:-1],dtype=np.int32)
    a['bridge_eligible']=np.empty(P,dtype=bool);a['bridge_q_cells']=np.zeros(P+(8,4))
    for name in ('bridge_q_shapley','bridge_harm_shapley'):a[name]=np.zeros(P+(3,4))
    a['pair_unknown_membership']=np.empty(P+(4,),dtype=np.int8)
    a['objective_eligible']=np.empty(P,dtype=bool);a['objective_values']=np.zeros(P+(2,))
    a['objective_own_help_influences']=np.zeros(P+(2,2));a['objective_matched_probability_gain']=np.zeros(P)
    a['objective_help_counts']=np.zeros(P,dtype=np.int8);a['pair_action_cases']=np.empty(P,dtype=np.int8)
    for name in ('model_seeds','schedules','targets'):a[name]=parent[name].copy()
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for si in range(len(seeds)):
                d=parent['g_'+top+'_'+rep+'_unit_rows'][si];source=parent['schedules'][si];states=[]
                for mi in range(4):
                    at=(ti,ri,mi,si);g=parent['trial_update_factors'][at];w=parent['trial_directions'][at]
                    steps=g[:,None]*w;heads=np.concatenate((np.zeros((1,d.shape[1])),np.cumsum(steps,axis=0)))
                    q,margin=probability_effects(d,heads[:-1],g,w,y)
                    require(np.max(np.abs(q-parent['trial_correct_probability_delta'][at]))<=tol
                        and np.max(np.abs(margin-parent['trial_aligned_margin_delta'][at]))<=tol,'Factual same-state effect reconstruction')
                    require(np.max(np.abs(heads[parent['checkpoints']]-parent['checkpoint_heads'][at]))<=tol,'Factual output-prefix reconstruction')
                    a['probability_changes'][at]=q;a['margin_changes'][at]=margin
                    a['effect_categories'][at]=decompose_categories(parent['trial_known_before'][at],source)
                    a['phase_before'][at]=(np.count_nonzero(parent['trial_known_before'][at],axis=-1)==4)
                    full=np.flatnonzero(np.count_nonzero(parent['trial_known_after'][at],axis=-1)==4)
                    a['first_full_trials'][at]=int(full[0]+1) if full.size else n+1
                    states.append((heads[:-1]@d.T,g,w@d.T))
                at=(ti,ri,si);S=(ti,ri,2,si);B=(ti,ri,3,si)
                case=parent['trial_actions'][S]+2*parent['trial_actions'][B];eligible=case==3
                a['pair_action_cases'][at]=case;a['bridge_eligible'][at]=eligible
                zS,gS,fS=states[2];zB,gB,fB=states[3]
                cells=bridge_cells(zS[eligible],zB[eligible],gS[eligible],gB[eligible],fS[eligible],fB[eligible],y)
                a['bridge_q_cells'][at][eligible]=cells
                a['bridge_q_shapley'][at][eligible]=shapley3(cells)
                a['bridge_harm_shapley'][at][eligible]=shapley3(np.maximum(-cells,0.))
                uS=a['effect_categories'][S]==2;uB=a['effect_categories'][B]==2
                a['pair_unknown_membership'][at]=np.where(uS&uB,1,np.where(uS,2,np.where(uB,3,0)))
                full=eligible&(a['phase_before'][S]==1)&(a['phase_before'][B]==1)
                a['objective_eligible'][at]=full
                if np.any(full):
                    src=source[full];gram=(d@d.T)[src];h=(y[src,None]*y*gram>tol)&(np.arange(4)!=src[:,None])
                    count=h.sum(-1);a['objective_help_counts'][at][full]=count
                    for method,f in enumerate((fS[full],fB[full])):
                        own=f[np.arange(len(src)),src]
                        help=np.divide((h*y[src,None]*y*f).sum(-1),count,out=np.zeros(len(src)),where=count>0)
                        a['objective_own_help_influences'][at][full,method,0]=own
                        a['objective_own_help_influences'][at][full,method,1]=help
                        a['objective_values'][at][full,method]=np.where(count>0,.5*(own+help),own)
                        safe=y[src,None]*y*f
                        require(np.min(safe)>=-tol,'Same full-observation comparison direction is feasible')
                    require(np.min(np.diff(a['objective_values'][at][full],axis=-1))>=-tol,'Recorded full-known balanced objective is not worse')
                    a['objective_matched_probability_gain'][at][full]=(a['bridge_q_cells'][at][full,1]-a['bridge_q_cells'][at][full,0]).mean(-1)
    return a,make_rows(cfg,a)


def make_rows(cfg,a):
    rows=[]
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for mi,method in enumerate(METHODS):
                for si,seed in enumerate(cfg['model_seeds']):
                    at=(ti,ri,mi,si)
                    for ph,phase in enumerate(PHASES):
                        phase_mask=a['phase_before'][at]==ph
                        for ci,category in enumerate(CATEGORIES):
                            mask=phase_mask[:,None]&(a['effect_categories'][at]==ci)
                            q=a['probability_changes'][at][mask];m=a['margin_changes'][at][mask]
                            rows.append({'topology':top,'representation':rep,'method':method,'seed':seed,
                                'phase':phase,'category':category,'trial_count':int(phase_mask.sum()),'exposures':int(mask.sum()),
                                'nonzero_probability_effects':int(np.count_nonzero(q)),
                                'positive_probability_effects':int(np.count_nonzero(q>0)),
                                'negative_probability_effects':int(np.count_nonzero(q<0)),
                                'probability_positive_mass':float(np.maximum(q,0.).sum()),
                                'probability_negative_mass':float(np.maximum(-q,0.).sum()),'probability_signed_sum':float(q.sum()),
                                'margin_positive_mass':float(np.maximum(m,0.).sum()),
                                'margin_negative_mass':float(np.maximum(-m,0.).sum()),'margin_signed_sum':float(m.sum())})
    return rows


def load_parent(cfg,root):
    with np.load(Path(root)/cfg['parent_result']/'arrays.npz',allow_pickle=False) as f:return {name:f[name] for name in f.files}


def produce(cfg,root):return analyze(cfg,load_parent(cfg,root))


def summarize(cfg,arrays,rows,root):return summarize_arrays(cfg,arrays,rows,load_parent(cfg,root))


def summarize_arrays(cfg,a,rows,parent):
    validate_config(cfg);tol=cfg['numeric_tolerance'];n=cfg['training_trials'];seeds=cfg['model_seeds'];y=2*a['targets']-1
    p=parent['checkpoint_probabilities'];correct=np.where(a['targets']==1,p,1-p)
    initial=correct[...,0,:].mean(-1);final=correct[...,-1,:].mean(-1)
    q=a['probability_changes'];m=a['margin_changes'];contributions=np.zeros(q.shape[:4]+(2,3))
    positive=contributions.copy();negative=contributions.copy();families=[];contrasts=[]
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for mi,method in enumerate(METHODS):
                records=[]
                for si,seed in enumerate(seeds):
                    at=(ti,ri,mi,si)
                    for ph in range(2):
                        for ci in range(3):
                            mask=(a['phase_before'][at]==ph)[:,None]&(a['effect_categories'][at]==ci)
                            values=q[at][mask];ix=at+(ph,ci)
                            contributions[ix]=float(values.sum()/4.)
                            positive[ix]=float(np.maximum(values,0.).sum());negative[ix]=float(np.maximum(-values,0.).sum())
                    records.append({'seed':seed,'initial_expected_accuracy':float(initial[at]),'final_expected_accuracy':float(final[at]),
                        'reconstructed_accuracy_gain':float(contributions[at].sum()),
                        'phase_category_endpoint_contributions':contributions[at].tolist(),
                        'phase_category_positive_probability_mass':positive[at].tolist(),
                        'phase_category_negative_probability_mass':negative[at].tolist(),
                        'first_full_trial':int(a['first_full_trials'][at]),
                        'actual_Go_trials':int(parent['trial_actions'][at].sum()),
                        'actual_head_path_l2':float(parent['trial_cumulative_head_path_l2'][at+(-1,)])})
                families.append({'topology':top,'representation':rep,'method':method,'seeds':records,
                    'mean_initial_expected_accuracy':float(initial[ti,ri,mi].mean()),
                    'mean_final_expected_accuracy':float(final[ti,ri,mi].mean()),
                    'mean_phase_category_endpoint_contributions':contributions[ti,ri,mi].mean(0).tolist(),
                    'mean_phase_category_positive_probability_mass':positive[ti,ri,mi].mean(0).tolist(),
                    'mean_phase_category_negative_probability_mass':negative[ti,ri,mi].mean(0).tolist()})
            pair_records=[]
            for si,seed in enumerate(seeds):
                at=(ti,ri,si);S=(ti,ri,2,si);B=(ti,ri,3,si);difference=q[B]-q[S]
                harmS=np.maximum(-q[S],0.);harmB=np.maximum(-q[B],0.)
                membership=a['pair_unknown_membership'][at];both=membership==1;onlyS=membership==2;onlyB=membership==3
                eligible=a['bridge_eligible'][at];obj=a['objective_eligible'][at];cases=a['pair_action_cases'][at]
                ph=a['bridge_q_shapley'][at].sum((0,2))/4.
                uh=(a['bridge_harm_shapley'][at]*both[:,None,:]).sum((0,2))
                unknownparts=[float(((harmB-harmS)*both).sum()),-float((harmS*onlyS).sum()),float((harmB*onlyB).sum())]
                objdiff=np.diff(a['objective_values'][at][obj],axis=-1).ravel()
                pg=a['objective_matched_probability_gain'][at][obj];influence=a['objective_own_help_influences'][at][obj]
                average=lambda x:float(np.mean(x)) if np.size(x) else None
                record={'seed':seed,'endpoint_difference':float(final[B]-final[S]),
                    'phase_category_endpoint_difference':(contributions[B]-contributions[S]).tolist(),
                    'action_case_counts':[int(np.count_nonzero(cases==k)) for k in range(4)],
                    'action_case_endpoint_difference':[float(difference[cases==k].sum()/4.) for k in range(4)],
                    'both_Go_probability_shapley':ph.tolist(),
                    'both_Go_probability_difference':float(difference[eligible].sum()/4.),
                    'unknown_damage_difference':float((harmB*(a['effect_categories'][B]==2)).sum()-(harmS*(a['effect_categories'][S]==2)).sum()),
                    'unknown_membership_damage_difference':unknownparts,
                    'both_Go_shared_unknown_harm_shapley':uh.tolist(),
                    'both_Go_shared_unknown_harm_difference':float(((harmB-harmS)*both*eligible[:,None]).sum()),
                    'unpaired_Go_shared_unknown_harm_difference':float(((harmB-harmS)*both*(~eligible)[:,None]).sum()),
                    'full_known_objective_trials':int(obj.sum()),'objective_gain_mean':average(objdiff),
                    'objective_improved_trials':int(np.count_nonzero(objdiff>tol)),
                    'objective_matched_probability_gain_mean':average(pg),
                    'objective_improved_but_probability_not_improved_trials':int(np.count_nonzero((objdiff>tol)&(pg<=0.))),
                    'objective_own_influence_change_mean':average(influence[:,1,0]-influence[:,0,0]),
                    'objective_help_influence_change_mean':average(influence[:,1,1]-influence[:,0,1]),
                    'full_known_direction_cosine_mean':average(np.sum(parent['trial_directions'][S][obj]*parent['trial_directions'][B][obj],axis=-1)),
                    'unknown_mask_disagreement_destination_count':int(np.count_nonzero(onlyS|onlyB)),
                    'first_full_trial_difference':int(a['first_full_trials'][B]-a['first_full_trials'][S])}
                pair_records.append(record)
            vector_keys=('phase_category_endpoint_difference','action_case_counts','action_case_endpoint_difference',
                'both_Go_probability_shapley','unknown_membership_damage_difference','both_Go_shared_unknown_harm_shapley')
            scalar_keys=('endpoint_difference','both_Go_probability_difference','unknown_damage_difference',
                'both_Go_shared_unknown_harm_difference','unpaired_Go_shared_unknown_harm_difference',
                'full_known_objective_trials','objective_gain_mean','objective_improved_trials',
                'objective_matched_probability_gain_mean','objective_improved_but_probability_not_improved_trials',
                'objective_own_influence_change_mean','objective_help_influence_change_mean',
                'full_known_direction_cosine_mean',
                'unknown_mask_disagreement_destination_count','first_full_trial_difference')
            means={key:np.asarray([r[key] for r in pair_records]).mean(0).tolist() for key in vector_keys}
            for key in scalar_keys:
                values=[r[key] for r in pair_records if r[key] is not None]
                means[key]=float(np.mean(values)) if values else None
            contrasts.append({'topology':top,'representation':rep,'role':'primary' if (ti,ri)==(0,0) else 'descriptive_control',
                'seeds':pair_records,'means':means})
    endpoint_residual=float(np.max(np.abs(contributions.sum((-1,-2))-(final-initial))))
    shapley_residual=float(np.max(np.abs(a['bridge_q_shapley'].sum(-2)-(a['bridge_q_cells'][...,-1,:]-a['bridge_q_cells'][...,0,:]))))
    harm_cells=np.maximum(-a['bridge_q_cells'],0.)
    harm_shapley_residual=float(np.max(np.abs(a['bridge_harm_shapley'].sum(-2)-(harm_cells[...,-1,:]-harm_cells[...,0,:]))))
    contrast_residual=max(abs(r['endpoint_difference']-np.asarray(r['phase_category_endpoint_difference']).sum()) for c in contrasts for r in c['seeds'])
    bridge_residual=max(abs(r['endpoint_difference']-sum(r['action_case_endpoint_difference'])) for c in contrasts for r in c['seeds'])
    unknown_residual=max(abs(r['unknown_damage_difference']-sum(r['unknown_membership_damage_difference'])) for c in contrasts for r in c['seeds'])
    bridge_total_residual=max(abs(r['both_Go_probability_difference']-sum(r['both_Go_probability_shapley'])) for c in contrasts for r in c['seeds'])
    unknown_bridge_residual=max(abs(r['both_Go_shared_unknown_harm_difference']-sum(r['both_Go_shared_unknown_harm_shapley'])) for c in contrasts for r in c['seeds'])
    residuals={'endpoint_telescoping':endpoint_residual,'per_destination_probability_Shapley':shapley_residual,
        'per_destination_harm_Shapley':harm_shapley_residual,'endpoint_phase_category_difference':float(contrast_residual),
        'endpoint_action_case_difference':float(bridge_residual),'unknown_membership_difference':float(unknown_residual),
        'paired_Go_probability_bridge':float(bridge_total_residual),'paired_Go_shared_unknown_harm_bridge':float(unknown_bridge_residual)}
    require(all(np.isfinite(x) and x<=tol for x in residuals.values()),'Saved-path exact accounting residual failed')
    require(len(rows)==3*2*4*len(seeds)*2*3 and len(a)==19,'Complete saved-path output closure')
    eligible=int(a['bridge_eligible'].sum());full=int(a['objective_eligible'].sum())
    return {'study_id':cfg['study_id'],'status':'completed','scope':cfg['scope'],'families':families,'contrasts':contrasts,
        'phase_order':list(PHASES),'category_order':list(CATEGORIES),'bridge_factor_order':['state','scalar','direction'],
        'action_case_order':['neither_Go','selective_only_Go','balanced_only_Go','both_Go'],
        'unknown_membership_order':['both_unknown_difference','selective_only_negative','balanced_only_positive'],
        'validation':{'all_exact_accounting_gates_passed':True,'residuals':residuals,
            'last_discovery_trial_assigned_to_partial':True,'zero_effects_and_roundoff_signs_preserved':True,
            'both_Go_only_bridge_no_NoGo_zero_direction_ambiguity':True,
            'full_known_objectives_compare_same_information_constraints':True},
        'counts':{'analyzed_parent_trial_records':int(np.prod(q.shape[:-1])),
            'analyzed_destination_effects':int(q.size),'paired_trial_records':int(np.prod(a['bridge_eligible'].shape)),
            'measurement_records':len(rows),'numeric_arrays':len(a),'bridge_eligible_trials':eligible,
            'bridge_evaluated_corner_destination_effects':eligible*32,'full_known_objective_trials':full,
            'new_training_trials':0,'new_head_update_calls':0,'new_internal_update_calls':0,'new_value_update_calls':0,
            'new_random_draws':0,'fresh_model_constructions':0,'new_optimization_calls':0},
        'interpretation':{'retrospective_descriptive_only':True,'parent_H_gain':False,'parent_H_acquisition':False,'parent_H_unknown':False,
            'new_confirmatory_hypothesis':False,'exact_path_accounting_not_unique_causal_explanation':True,
            'crossed_one_step_cells_are_not_relearned_policy':True,'crossed_directions_can_have_different_original_information':True,
            'unknown_harm_bridge_intersection_mask_fixed_across_cells':True,'same_realized_movement_budget_claimed':False,
            'internal_synapses_and_probe_activity_fixed':True,'observed_memory_is_separate_from_output_weights':True,
            'novel_input_generalization_tested':False,'new_human_experiment':False,'shared_neural_mechanism_claim':False,
            'education_effect_claim':False}}
