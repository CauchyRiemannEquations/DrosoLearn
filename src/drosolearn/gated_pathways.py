"""Descriptive accounting of immutable observed-gated learning paths.

Only recorded output steps are evaluated. No RNG, graph, learner, solver,
alternative policy, or extra training is constructed.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .representation_envelope import require

TOPS=('real','rewired','ann');REPS=('raw','identity')
METHODS=('shared','isolated','feedback_selective','balanced','gated_balanced')
CONTROLS=('balanced','isolated','feedback_selective','shared')
PHASES=('pre','post');CATEGORIES=('own','known_other','unknown_other')
SCIENCE={'study_id':'gated_pathways_v1','scope':'retrospective_descriptive_gated_pathways',
    'module':'drosolearn.gated_pathways','protocol':'docs/gated-pathways-protocol.md',
    'parent_result':'results/gated_learning_v1','baseline_tag':'gated-learning-v1',
    'baseline_commit':'b89e9f2baf1aed91b4451538c85ba3e9458df53b',
    'topologies':list(TOPS),'representations':list(REPS),'methods':list(METHODS),
    'model_seeds':list(range(17601,17611)),'training_trials':400,'numeric_tolerance':1e-8,
    'phase_definition':'reference_gated_known_before_count_equals_four',
    'category_definition':'own_each_method_known_before_other_unknown_before_other',
    'reference_method':'gated_balanced','contrast_controls':list(CONTROLS),
    'new_confirmatory_hypothesis':False,'external_manifest':None}


def validate_config(cfg):
    require(type(cfg) is dict,'Configuration dictionary required')
    fixture=cfg.get('scope')=='integration_test'
    exempt={'scope','model_seeds','training_trials'} if fixture else set()
    for key,value in SCIENCE.items():
        if key not in exempt:require(cfg.get(key)==value,'Fixed gated-path accounting differs: '+key)
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
    before=heads_before@D.T;step=g[...,None]*(w@D.T)
    return signs*(sigmoid(before+step)-sigmoid(before)),signs*step


def decompose_categories(known_before,sources):
    known=np.asarray(known_before);sources=np.asarray(sources)
    require(known.ndim>=1 and known.shape[-1]==4 and known.dtype.kind in 'biu' and np.isin(known,[-1,0,1]).all()
        and sources.dtype.kind in 'iu' and sources.shape==known.shape[:-1] and np.isin(sources,range(4)).all(),
        'Known-before integer axes and source indices')
    result=np.where(known==0,2,1).astype(np.int8)
    return np.where(np.arange(4)==sources[...,None],0,result).astype(np.int8)


def analyze(cfg,parent):
    validate_config(cfg);n=cfg['training_trials'];seeds=cfg['model_seeds'];tol=cfg['numeric_tolerance']
    F=(3,2,5,len(seeds),n);P=(3,2,len(seeds),n)
    require(np.array_equal(parent['model_seeds'],seeds) and parent['trial_actions'].shape==F
        and parent['schedules'].shape==(len(seeds),n) and parent['targets'].shape==(4,)
        and np.isin(parent['targets'],[0,1]).all(),'Fixed complete parent cohort and axes')
    y=2*parent['targets'].astype(int)-1
    a={name:np.empty(F+(4,),dtype=np.int8 if name=='effect_categories' else float)
        for name in ('probability_changes','margin_changes','effect_categories')}
    a['own_full_before']=np.count_nonzero(parent['trial_known_before'],axis=-1)==4
    require(a['own_full_before'].shape==F,'Complete factual observation histories')
    a['gated_phase_before']=a['own_full_before'][:,:,4].astype(np.int8)
    require(a['gated_phase_before'].shape==P,'Common reference phase axes')
    a['first_full_trials']=np.empty(F[:-1],dtype=np.int32)
    a['first_gated_balanced_trials']=np.empty(P[:-1],dtype=np.int32)
    a['head_step_l2']=np.empty(F);a['head_step_squared_l2']=np.empty(F)
    for name in ('model_seeds','schedules','targets'):a[name]=parent[name].copy()
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for si in range(len(seeds)):
                d=parent['g_'+top+'_'+rep+'_unit_rows'][si];source=parent['schedules'][si]
                opened=np.flatnonzero(a['gated_phase_before'][ti,ri,si])
                a['first_gated_balanced_trials'][ti,ri,si]=int(opened[0]+1) if opened.size else n+1
                require(a['first_gated_balanced_trials'][ti,ri,si]==parent['first_balanced_mode_trials'][ti,ri,4,si],
                    'Reference switching trial identity')
                for mi in range(5):
                    at=(ti,ri,mi,si);g=parent['trial_update_factors'][at];w=parent['trial_directions'][at]
                    steps=g[:,None]*w;heads=np.concatenate((np.zeros((1,d.shape[1])),np.cumsum(steps,axis=0)))
                    q,m=probability_effects(d,heads[:-1],g,w,y)
                    require(np.max(np.abs(q-parent['trial_correct_probability_delta'][at]))<=tol
                        and np.max(np.abs(m-parent['trial_aligned_margin_delta'][at]))<=tol,'Recorded same-state destination effects')
                    checkheads=heads[parent['checkpoints']]
                    require(np.max(np.abs(checkheads-parent['checkpoint_heads'][at]))<=tol
                        and np.max(np.abs(sigmoid(checkheads@d.T)-parent['checkpoint_probabilities'][at]))<=tol,
                        'Recorded factual head prefixes and static probabilities')
                    a['probability_changes'][at]=q;a['margin_changes'][at]=m
                    a['effect_categories'][at]=decompose_categories(parent['trial_known_before'][at],source)
                    lengths=np.linalg.norm(steps,axis=-1)
                    a['head_step_l2'][at]=lengths;a['head_step_squared_l2'][at]=lengths**2
                    require(np.max(np.abs(lengths-parent['trial_head_steps'][at]))<=tol
                        and np.max(np.abs(lengths**2-parent['trial_head_step_squared_l2'][at]))<=tol
                        and np.max(np.abs(np.cumsum(lengths)-parent['trial_cumulative_head_path_l2'][at]))<=tol
                        and np.max(np.abs(np.cumsum(lengths**2)-parent['trial_cumulative_head_squared_l2'][at]))<=tol,
                        'Recorded actual step lengths and cumulative costs')
                    full=np.flatnonzero(np.count_nonzero(parent['trial_known_after'][at],axis=-1)==4)
                    a['first_full_trials'][at]=int(full[0]+1) if full.size else n+1
    return a,make_rows(cfg,a,parent)


def ratio(value,denominator):return float(value/denominator) if denominator else None


def make_rows(cfg,a,parent):
    rows=[]
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for mi,method in enumerate(METHODS):
                for si,seed in enumerate(cfg['model_seeds']):
                    at=(ti,ri,mi,si)
                    for ph,phase in enumerate(PHASES):
                        pm=a['gated_phase_before'][ti,ri,si]==ph
                        go=int(parent['trial_actions'][at][pm].sum());length=float(a['head_step_l2'][at][pm].sum())
                        energy=float(a['head_step_squared_l2'][at][pm].sum())
                        for ci,category in enumerate(CATEGORIES):
                            mask=pm[:,None]&(a['effect_categories'][at]==ci)
                            q=a['probability_changes'][at][mask];m=a['margin_changes'][at][mask];gain=float(q.sum()/4.)
                            rows.append({'topology':top,'representation':rep,'method':method,'seed':seed,
                                'phase':phase,'category':category,'trial_count':int(pm.sum()),'exposures':int(mask.sum()),
                                'nonzero_probability_effects':int(np.count_nonzero(q)),
                                'positive_probability_effects':int(np.count_nonzero(q>0)),
                                'negative_probability_effects':int(np.count_nonzero(q<0)),
                                'probability_positive_mass':float(np.maximum(q,0.).sum()),
                                'probability_negative_mass':float(np.maximum(-q,0.).sum()),'probability_signed_sum':float(q.sum()),
                                'margin_positive_mass':float(np.maximum(m,0.).sum()),
                                'margin_negative_mass':float(np.maximum(-m,0.).sum()),'margin_signed_sum':float(m.sum()),
                                'go_trials':go,'nonzero_head_steps':int(np.count_nonzero(a['head_step_l2'][at][pm])),
                                'head_path_l2':length,'head_squared_l2':energy,'gain_per_Go':ratio(gain,go),
                                'gain_per_path':ratio(gain,length),'endpoint_contribution':gain})
    return rows


def load_parent(cfg,root):
    with np.load(Path(root)/cfg['parent_result']/'arrays.npz',allow_pickle=False) as f:return {name:f[name] for name in f.files}


def produce(cfg,root):return analyze(cfg,load_parent(cfg,root))


def summarize(cfg,arrays,rows,root):return summarize_arrays(cfg,arrays,rows,load_parent(cfg,root))


def summarize_arrays(cfg,a,rows,parent):
    validate_config(cfg);tol=cfg['numeric_tolerance'];seeds=cfg['model_seeds']
    q=a['probability_changes'];m=a['margin_changes'];p=parent['checkpoint_probabilities']
    correct=np.where(a['targets']==1,p,1-p);initial=correct[...,0,:].mean(-1);final=correct[...,-1,:].mean(-1)
    contributions=np.zeros(q.shape[:4]+(2,3));positive=contributions.copy();negative=contributions.copy()
    phasecosts={};families=[];contrasts=[]
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for mi,method in enumerate(METHODS):
                records=[]
                for si,seed in enumerate(seeds):
                    at=(ti,ri,mi,si);cost=[]
                    for ph in range(2):
                        pm=a['gated_phase_before'][ti,ri,si]==ph
                        for ci in range(3):
                            mask=pm[:,None]&(a['effect_categories'][at]==ci);v=q[at][mask];ix=at+(ph,ci)
                            contributions[ix]=float(v.sum()/4.);positive[ix]=float(np.maximum(v,0.).sum())
                            negative[ix]=float(np.maximum(-v,0.).sum())
                        go=int(parent['trial_actions'][at][pm].sum());length=float(a['head_step_l2'][at][pm].sum())
                        gain=float(q[at][pm].sum()/4.)
                        cost.append({'trial_count':int(pm.sum()),'go_trials':go,
                            'nonzero_head_steps':int(np.count_nonzero(a['head_step_l2'][at][pm])),
                            'head_path_l2':length,'head_squared_l2':float(a['head_step_squared_l2'][at][pm].sum()),
                            'endpoint_contribution':gain,'gain_per_Go':ratio(gain,go),'gain_per_path':ratio(gain,length)})
                    phasecosts[at]=cost
                    records.append({'seed':seed,'initial_expected_accuracy':float(initial[at]),'final_expected_accuracy':float(final[at]),
                        'reconstructed_accuracy_gain':float(contributions[at].sum()),
                        'phase_category_endpoint_contributions':contributions[at].tolist(),
                        'phase_category_positive_probability_mass':positive[at].tolist(),
                        'phase_category_negative_probability_mass':negative[at].tolist(),
                        'first_full_trial':int(a['first_full_trials'][at]),
                        'reference_first_balanced_trial':int(a['first_gated_balanced_trials'][ti,ri,si]),
                        'phase_costs':cost,'own_full_vs_reference_phase_disagreement_trials':int(np.count_nonzero(
                            a['own_full_before'][at]!=a['gated_phase_before'][ti,ri,si]))})
                means=[]
                for ph in range(2):
                    means.append({key:float(np.mean(values)) if (values:=[r['phase_costs'][ph][key] for r in records
                        if r['phase_costs'][ph][key] is not None]) else None for key in records[0]['phase_costs'][ph]})
                families.append({'topology':top,'representation':rep,'method':method,'seeds':records,
                    'mean_initial_expected_accuracy':float(initial[ti,ri,mi].mean()),
                    'mean_final_expected_accuracy':float(final[ti,ri,mi].mean()),
                    'mean_phase_category_endpoint_contributions':contributions[ti,ri,mi].mean(0).tolist(),
                    'mean_phase_category_positive_probability_mass':positive[ti,ri,mi].mean(0).tolist(),
                    'mean_phase_category_negative_probability_mass':negative[ti,ri,mi].mean(0).tolist(),'mean_phase_costs':means})
            for control in CONTROLS:
                mi=METHODS.index(control);records=[]
                for si,seed in enumerate(seeds):
                    G=(ti,ri,4,si);C=(ti,ri,mi,si)
                    records.append({'seed':seed,'endpoint_difference':float(final[G]-final[C]),
                        'phase_category_endpoint_difference':(contributions[G]-contributions[C]).tolist(),
                        'phase_category_positive_probability_mass_difference':(positive[G]-positive[C]).tolist(),
                        'phase_category_negative_probability_mass_difference':(negative[G]-negative[C]).tolist(),
                        'phase_endpoint_difference':(contributions[G].sum(-1)-contributions[C].sum(-1)).tolist(),
                        'phase_head_path_difference':[phasecosts[G][ph]['head_path_l2']-phasecosts[C][ph]['head_path_l2'] for ph in range(2)],
                        'phase_Go_difference':[phasecosts[G][ph]['go_trials']-phasecosts[C][ph]['go_trials'] for ph in range(2)],
                        'own_full_trial_difference':int(a['first_full_trials'][G]-a['first_full_trials'][C]),
                        'unknown_mask_disagreement_destination_count':int(np.count_nonzero(
                            (a['effect_categories'][G]==2)!=(a['effect_categories'][C]==2)))})
                means={key:np.asarray([r[key] for r in records]).mean(0).tolist() for key in records[0] if key!='seed'}
                contrasts.append({'topology':top,'representation':rep,'control':control,
                    'role':'primary' if (ti,ri,mi)==(0,0,3) else 'descriptive_control','seeds':records,'means':means})
    costpath=costenergy=0.
    for at,cost in phasecosts.items():
        costpath=max(costpath,abs(sum(r['head_path_l2'] for r in cost)-parent['trial_cumulative_head_path_l2'][at+(-1,)]))
        costenergy=max(costenergy,abs(sum(r['head_squared_l2'] for r in cost)-parent['trial_cumulative_head_squared_l2'][at+(-1,)]))
    residuals={'endpoint_telescoping':float(np.max(np.abs(contributions.sum((-1,-2))-(final-initial)))),
        'positive_negative_identity':float(np.max(np.abs(contributions-(positive-negative)/4.))),
        'phase_category_partition':float(np.max(np.abs(contributions.sum((-1,-2))-q.sum((-1,-2))/4.))),
        'contrast_endpoint':float(max(abs(r['endpoint_difference']-np.asarray(r['phase_category_endpoint_difference']).sum())
            for c in contrasts for r in c['seeds'])), 'phase_cost_path':float(costpath),'phase_cost_energy':float(costenergy)}
    require(all(np.isfinite(v) and v<=tol for v in residuals.values()),'Exact common-phase accounting residual failed')
    expected_rows=3*2*5*len(seeds)*2*3
    require(len(rows)==expected_rows and len(a)==12,'Complete common-phase output closure')
    return {'study_id':cfg['study_id'],'status':'completed','scope':cfg['scope'],'families':families,'contrasts':contrasts,
        'phase_order':list(PHASES),'category_order':list(CATEGORIES),'contrast_control_order':list(CONTROLS),
        'validation':{'all_exact_accounting_gates_passed':True,'residuals':residuals,
            'common_reference_phase_uses_prefeedback_memory':True,'last_discovery_trial_assigned_to_pre':True,
            'each_method_categories_use_own_observations':True,'zero_effects_and_roundoff_signs_preserved':True,
            'no_counterfactual_policy_or_direction_evaluated':True},
        'counts':{'analyzed_parent_trial_records':int(np.prod(q.shape[:-1])),'analyzed_destination_effects':int(q.size),
            'reference_gate_paths':3*2*len(seeds),'paired_comparison_paths':3*2*len(seeds)*4,
            'paired_comparison_trial_records':3*2*len(seeds)*4*cfg['training_trials'],
            'measurement_records':len(rows),'numeric_arrays':len(a),'new_training_trials':0,'new_head_update_calls':0,
            'new_internal_update_calls':0,'new_value_update_calls':0,'new_random_draws':0,'fresh_model_constructions':0,'new_optimization_calls':0},
        'interpretation':{'retrospective_descriptive_only':True,'parent_H_gain':False,'parent_H_acquisition':False,
            'parent_H_protection':True,'new_confirmatory_hypothesis':False,'exact_path_accounting_not_unique_causal_explanation':True,
            'reference_gate_is_common_time_split_not_all_methods_knowledge':True,'same_realized_movement_budget_claimed':False,
            'internal_synapses_and_probe_activity_fixed':True,'observed_memory_is_separate_from_output_weights':True,
            'novel_input_generalization_tested':False,'new_human_experiment':False,'shared_neural_mechanism_claim':False,'education_effect_claim':False}}
