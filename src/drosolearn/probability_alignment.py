"""Frozen-state objective alignment, not another learned trajectory.

Observed signs define candidate objectives. Hidden targets are evaluator-only.
The probability direction maximizes a tangent, not the finite sigmoid gain.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .representation_envelope import require
from .cone_geometry import cone_direction

TOPS=('real','rewired','ann');REPS=('raw','identity')
CANDIDATES=('balanced','isolated','self_blend','uniform_score','state_probability')
OBJECTIVES=('legacy','uniform','probability');GROUPS=('own','other','total')
SCIENCE={'study_id':'probability_alignment_v1','scope':'retrospective_descriptive_probability_alignment',
    'module':'drosolearn.probability_alignment','protocol':'docs/probability-alignment-protocol.md',
    'learning_result':'results/self_blend_learning_v1','reference_method':'gated_balanced','reference_method_index':4,
    'baseline_tag':'self-blend-factors-v1','baseline_commit':'2715f1eada36207ce77709f08874688b1425d676',
    'topologies':list(TOPS),'representations':list(REPS),'candidates':list(CANDIDATES),'objective_order':list(OBJECTIVES),
    'destination_groups':list(GROUPS),'model_seeds':list(range(17701,17711)),'training_trials':400,
    'numeric_tolerance':1e-8,'svd_rcond':1e-12,'head_lr':.1,'blend_isolated_weight':.75,'blend_balanced_weight':.25,
    'eligibility':'reference_own_known_before_full_and_actual_Go','probability_gradient_domain':'all_abs_logits_strictly_less_than_40',
    'primary_comparison':'state_probability_minus_uniform_score_exact_finite_Q_per_eligible_update',
    'new_confirmatory_hypothesis':False,'external_manifest':None}


def validate_config(cfg):
    fixture=cfg.get('scope')=='integration_test';exempt={'scope','model_seeds','training_trials'} if fixture else set()
    for key,value in SCIENCE.items():
        if key not in exempt:require(cfg.get(key)==value,'Fixed objective-alignment design differs: '+key)
    if fixture:
        require(type(cfg['model_seeds']) is list and 1<=len(cfg['model_seeds'])<=2
            and len(set(cfg['model_seeds']))==len(cfg['model_seeds'])
            and all(type(s) is int and 90001<=s<=90100 for s in cfg['model_seeds']),'Invented fixture seeds only')
        require(type(cfg['training_trials']) is int and 4<=cfg['training_trials']<=16,'Bounded fixture budget')


def sigmoid(z):return 1./(1.+np.exp(-np.clip(z,-40.,40.)))


def objectives(d,source,known,z,tol):
    d,k,z=map(np.asarray,(d,known,z))
    require(d.ndim==2 and d.shape[0]==4 and type(source) is int and 0<=source<4
        and k.shape==z.shape==(4,) and k.dtype.kind in 'iu' and np.isin(k,[-1,1]).all()
        and all(v.dtype.kind in 'fiu' and np.isfinite(v).all() for v in (d,z))
        and np.all(np.abs(z)<40.),'Finite full observed signs and strict unclipped gradient domain required')
    a=k[source]*k[:,None]*d
    helpful=(np.arange(4)!=source)&(k[source]*k*(d@d[source])>tol)
    legacy=d[source].copy()
    if helpful.any():legacy=.5*d[source]+.5*np.mean(k[source]*k[helpful,None]*d[helpful],axis=0)
    p=sigmoid(z);uniform=(k[source]*k)@d/4.;probability=(k[source]*k*p*(1-p))@d/4.
    return a,np.stack((legacy,uniform,probability)),helpful


def certificate(a,c,proof):
    w,v,mu=proof['direction'],proof['projected_objective'],proof['multipliers']
    return np.asarray([c@w,np.linalg.norm(v),proof['dual_upper_bound'],np.linalg.norm(w),
        max(0.,float(-(a@w).min())),np.max(np.abs(v-c-a.T@mu)),
        np.max(np.abs(mu*(a@v))),abs(float(np.linalg.norm(v)-c@w))])


def analyze(cfg,parent):
    validate_config(cfg);seeds=cfg['model_seeds'];n=cfg['training_trials'];tol=cfg['numeric_tolerance'];mi=4
    F=(3,2,len(seeds),n);C=parent['trial_directions'].shape[-1]
    require(np.array_equal(parent['model_seeds'],seeds) and parent['trial_actions'].shape==(3,2,6,len(seeds),n)
        and parent['schedules'].shape==(len(seeds),n),'Complete fixed original learning axes')
    a={name:parent[name].copy() for name in ('model_seeds','schedules','targets')}
    for key,dtype,tail in [('reference_full_before',bool,()),('eligible',bool,()),('factual_logits',float,(4,)),
        ('factual_update_factors',float,()),('observed_signs_before',np.int8,(4,)),('helpful_mask',bool,(4,)),
        ('objectives',float,(3,C)),('candidate_directions',float,(5,C)),('candidate_influences',float,(5,4)),
        ('candidate_q_changes',float,(5,4)),('candidate_margin_changes',float,(5,4)),
        ('candidate_tangent_changes',float,(5,4)),('candidate_head_steps',float,(5,)),
        ('candidate_objective_scores',float,(5,3)),('factual_probability_changes',float,(4,))]:
        a[key]=np.zeros(F+tail,dtype=dtype)
    a['case_indices']=np.full(F+(5,),-1,dtype=np.int32)
    D=[];I=[];cases=[];uniform_cache={};y=2*parent['targets'].astype(int)-1
    require(y.shape==(4,) and np.isin(y,[-1,1]).all(),'Four evaluator signs')
    for ti,top in enumerate(TOPS):
        ds=[];isos=[]
        for ri,rep in enumerate(REPS):
            d=parent['g_'+top+'_'+rep+'_unit_rows'];iso=parent['g_'+top+'_'+rep+'_isolated_directions']
            ds.append(d.copy());isos.append(iso.copy())
            for si in range(len(seeds)):
                at=(ti,ri,si);old=(ti,ri,mi,si)
                g=parent['trial_update_factors'][old];w=parent['trial_directions'][old]
                heads=np.concatenate((np.zeros((1,C)),np.cumsum(g[:,None]*w,axis=0)))
                z=heads[:-1]@d[si].T;known=parent['trial_known_before'][old];actions=parent['trial_actions'][old]
                full=np.count_nonzero(known,axis=-1)==4;eligible=full&(actions==1)
                q=y*(sigmoid(z+g[:,None]*(w@d[si].T))-sigmoid(z))
                p=sigmoid(z)[np.arange(n),parent['schedules'][si]]
                rewards=parent['trial_rewards'][old]
                require(np.max(np.abs(heads[parent['checkpoints']]-parent['checkpoint_heads'][old]))<=tol
                    and np.max(np.abs(q-parent['trial_correct_probability_delta'][old]))<=tol
                    and np.max(np.abs(cfg['head_lr']*(actions-p)*rewards-g))<=tol,
                    'Original head, factual effects and state-consistent signed update must reconstruct')
                a['reference_full_before'][at]=full;a['eligible'][at]=eligible
                a['factual_logits'][at]=z;a['factual_update_factors'][at]=g
                a['observed_signs_before'][at]=known;a['factual_probability_changes'][at]=q
                for t in np.flatnonzero(eligible):
                    t=int(t);src=int(parent['schedules'][si,t]);k=known[t];index=at+(t,)
                    require(np.array_equal(k,y) and rewards[t]==k[src] and g[t]*k[src]>=0,
                        'Direction inputs must be actual observed full memory, with obtained Go reward')
                    A,objs,H=objectives(d[si],src,k,z[t],tol);a['objectives'][index]=objs;a['helpful_mask'][index]=H
                    B=w[t];isolated=iso[si,src];mix=.75*isolated+.25*B;mixnorm=np.linalg.norm(mix)
                    require(mixnorm>0.,'Nonzero fixed mixture required')
                    directions=[B,isolated,mix/mixnorm]
                    for kind,oi in ((0,1),(1,2)):
                        address=src if kind==0 else t;key=(si,ti,ri,kind,address,*map(int,k))
                        cached=uniform_cache.get(key) if kind==0 else None
                        if cached is None:
                            proof=cone_direction(A,objs[oi],cfg['svd_rcond'],tol);cert=certificate(A,objs[oi],proof)
                            require(proof['numeric_qualified'] and abs(cert[3]-1)<=tol and max(cert[4:])<=tol,
                                'Qualified nonzero unit direction and global support certificate required')
                            cached=len(cases);cases.append({'key':key,'objective':objs[oi].copy(),
                                'direction':proof['direction'].copy(),'projection':proof['projected_objective'].copy(),
                                'multipliers':proof['multipliers'].copy(),'certificate':cert})
                            if kind==0:uniform_cache[key]=cached
                        directions.append(cases[cached]['direction']);a['case_indices'][index+(kind+3,)]=cached
                    directions=np.stack(directions);influence=directions@d[si].T
                    steps=abs(g[t])*np.linalg.norm(directions,axis=-1)
                    require(np.max(np.abs(np.linalg.norm(directions,axis=-1)-1))<=tol
                        and np.min((A@directions.T))>=-tol and np.max(steps)<=cfg['head_lr']+tol,
                        'Equal unit directions, observed safety and common actual single-step ceiling required')
                    a['candidate_directions'][index]=directions;a['candidate_influences'][index]=influence
                    a['candidate_q_changes'][index]=y*(sigmoid(z[t]+g[t]*influence)-sigmoid(z[t]))
                    a['candidate_margin_changes'][index]=y*g[t]*influence
                    a['candidate_tangent_changes'][index]=k*sigmoid(z[t])*(1-sigmoid(z[t]))*g[t]*influence
                    a['candidate_head_steps'][index]=steps;a['candidate_objective_scores'][index]=directions@objs.T
        D.append(np.stack(ds));I.append(np.stack(isos))
    a['unit_rows']=np.stack(D);a['isolated_directions']=np.stack(I)
    a['case_keys']=np.asarray([case['key'] for case in cases],dtype=np.int32).reshape(-1,9)
    for plural,singular,width in [('objectives','objective',C),('directions','direction',C),('projections','projection',C),
        ('multipliers','multipliers',4),('certificates','certificate',8)]:
        a['case_'+plural]=np.asarray([case[singular] for case in cases],dtype=float).reshape(-1,width)
    require(len(a)==27,'Exact numeric objective audit inventory')
    return a,make_rows(cfg,a)


def grouped(values,sources):
    own=values[np.arange(len(sources)),sources]/4.;total=values.sum(-1)/4.
    return np.column_stack((own,total-own,total))


def make_rows(cfg,a):
    rows=[]
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for ci,candidate in enumerate(CANDIDATES):
                for si,seed in enumerate(cfg['model_seeds']):
                    at=(ti,ri,si);e=a['eligible'][at];src=a['schedules'][si,e];count=int(e.sum())
                    q=grouped(a['candidate_q_changes'][at][e,ci],src)
                    b=grouped(a['candidate_q_changes'][at][e,0],src);u=grouped(a['candidate_q_changes'][at][e,3],src)
                    margin=grouped(a['candidate_margin_changes'][at][e,ci],src)
                    tangent=grouped(a['candidate_tangent_changes'][at][e,ci],src)
                    steps=a['candidate_head_steps'][at][e,ci];scores=a['candidate_objective_scores'][at][e,ci]
                    for gi,group in enumerate(GROUPS):
                        rows.append({'topology':top,'representation':rep,'candidate':candidate,'seed':seed,
                            'destination_group':group,'eligible_trials':count,'exact_Q_sum':float(q[:,gi].sum()),
                            'exact_Q_mean':float(q[:,gi].mean()) if count else None,
                            'exact_Q_difference_from_B_sum':float((q-b)[:,gi].sum()),
                            'exact_Q_difference_from_U_sum':float((q-u)[:,gi].sum()),
                            'margin_mean_sum':float(margin[:,gi].sum()),'tangent_Q_sum':float(tangent[:,gi].sum()),
                            'head_path_l2_sum':float(steps.sum()),'objective_legacy_score_mean':float(scores[:,0].mean()) if count else None,
                            'objective_uniform_score_mean':float(scores[:,1].mean()) if count else None,
                            'objective_probability_score_mean':float(scores[:,2].mean()) if count else None,
                            'mean_step_l2':float(steps.mean()) if count else None})
    return rows


def mean_nested(values):
    if isinstance(values[0],list):return [mean_nested(list(v)) for v in zip(*values)]
    defined=[v for v in values if v is not None];return float(np.mean(defined)) if defined else None


def summarize_arrays(cfg,a,rows,parent):
    validate_config(cfg);tol=cfg['numeric_tolerance'];e=a['eligible'];res={}
    res['original_balanced_finite_gain']=float(np.max(np.abs(a['candidate_q_changes'][...,0,:][e]-a['factual_probability_changes'][e]),initial=0.))
    scores=a['candidate_objective_scores'][e]
    res['legacy_global_support']=float(np.maximum(scores[:,:,0]-scores[:,0:1,0],0).max(initial=0.))
    res['uniform_global_support']=float(np.maximum(scores[:,:,1]-scores[:,3:4,1],0).max(initial=0.))
    res['probability_tangent_global_support']=float(np.maximum(scores[:,:,2]-scores[:,4:5,2],0).max(initial=0.))
    res['unit_norm']=float(np.max(np.abs(np.linalg.norm(a['candidate_directions'][e],axis=-1)-1),initial=0.))
    res['single_step_equal_cost']=float(np.max(np.abs(a['candidate_head_steps'][e]-np.abs(a['factual_update_factors'][e,None])),initial=0.))
    res['observed_known_margin_safety']=float(np.maximum(-a['candidate_margin_changes'][e],0).max(initial=0.))
    res['case_certificate_residual']=float(a['case_certificates'][:,4:].max(initial=0.))
    res['probability_tangent_identity']=float(np.max(np.abs(a['candidate_tangent_changes'][e].mean(-1)
        -np.abs(a['factual_update_factors'][e,None])*scores[:,:,2]),initial=0.))
    require(all(np.isfinite(value) and value<=tol for value in res.values()),'Objective, finite-state, safety or tangent certificate audit failed')
    for key in ('helpful_mask','objectives','candidate_directions','candidate_influences','candidate_q_changes',
        'candidate_margin_changes','candidate_tangent_changes','candidate_head_steps','candidate_objective_scores'):
        require(not np.any(a[key][~e]),'Excluded partial/NoGo records must remain zero: '+key)
    require(np.all(a['case_indices'][~e]==-1),'Excluded solver references must remain sentinel')
    families=[]
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for ci,candidate in enumerate(CANDIDATES):
                records=[]
                for si,seed in enumerate(cfg['model_seeds']):
                    at=(ti,ri,si);mask=e[at];source=a['schedules'][si,mask];count=int(mask.sum())
                    q=grouped(a['candidate_q_changes'][at][mask,ci],source)
                    B=grouped(a['candidate_q_changes'][at][mask,0],source);U=grouped(a['candidate_q_changes'][at][mask,3],source)
                    tangent=grouped(a['candidate_tangent_changes'][at][mask,ci],source);diff=q[:,2]-U[:,2]
                    records.append({'seed':seed,'eligible_trials':count,'exact_Q_sum':q.sum(0).tolist(),
                        'exact_Q_mean':q.mean(0).tolist() if count else [None]*3,
                        'difference_from_B_sum':(q-B).sum(0).tolist(),'difference_from_B_mean':(q-B).mean(0).tolist() if count else [None]*3,
                        'difference_from_U_sum':(q-U).sum(0).tolist(),'difference_from_U_mean':(q-U).mean(0).tolist() if count else [None]*3,
                        'positive_vs_U_trials':int((diff>0).sum()),'negative_vs_U_trials':int((diff<0).sum()),
                        'zero_vs_U_trials':int((diff==0).sum()),'negative_beyond_tolerance_vs_U_trials':int((diff < -tol).sum()),
                        'worst_Q_difference_vs_U':float(diff.min()) if count else None,
                        'mean_head_step':float(a['candidate_head_steps'][at][mask,ci].mean()) if count else None,
                        'head_path_l2_sum':float(a['candidate_head_steps'][at][mask,ci].sum()),
                        'objective_score_mean':a['candidate_objective_scores'][at][mask,ci].mean(0).tolist() if count else [None]*3,
                        'tangent_Q_sum':tangent.sum(0).tolist(),'tangent_error_mean':(q-tangent).mean(0).tolist() if count else [None]*3,
                        'old_helpful_count_mean':float(a['helpful_mask'][at][mask].sum(-1).mean()) if count else None})
                means={key:mean_nested([r[key] for r in records]) for key in records[0] if key!='seed'}
                families.append({'topology':top,'representation':rep,'candidate':candidate,'role':'primary' if (ti,ri,ci)==(0,0,4) else 'descriptive_control',
                    'seeds':records,'means':means})
    primary=next(f for f in families if f['role']=='primary');records=primary['seeds']
    pcase=int((a['case_keys'][:,3]==1).sum());ucase=int((a['case_keys'][:,3]==0).sum())
    require(pcase==int(e.sum()) and ucase<=6*len(cfg['model_seeds'])*4 and len(rows)==6*5*len(cfg['model_seeds'])*3,
        'Actual state-dependent and static optimization counts differ')
    return {'study_id':cfg['study_id'],'scope':cfg['scope'],'status':'completed','families':families,
        'candidate_order':list(CANDIDATES),'objective_order':list(OBJECTIVES),'group_order':list(GROUPS),
        'finite_alignment_summary':{'mean_exact_P_minus_U_per_update':primary['means']['difference_from_U_mean'][2],
            'positive_seed_means':sum(r['difference_from_U_mean'][2] is not None and r['difference_from_U_mean'][2]>0 for r in records),
            'pooled_positive_trials':sum(r['positive_vs_U_trials'] for r in records),'pooled_negative_trials':sum(r['negative_vs_U_trials'] for r in records),
            'pooled_zero_trials':sum(r['zero_vs_U_trials'] for r in records),
            'pooled_negative_beyond_tolerance_trials':sum(r['negative_beyond_tolerance_vs_U_trials'] for r in records),
            'worst_Q_difference':min((r['worst_Q_difference_vs_U'] for r in records if r['worst_Q_difference_vs_U'] is not None),default=None),
            'new_learned_performance_claim':False},
        'validation':{'all_numeric_information_and_support_checks_passed':True,'residuals':res,
            'new_objectives_use_own_observed_signs_only':True,'same_factual_state_action_reward_and_scalar':True,
            'all_eligible_logits_strictly_inside_clipping_domain':True,'excluded_partial_and_NoGo_solver_calls':0},
        'counts':{'saved_reference_records':int(e.size),'saved_reference_destination_effects':int(e.size*4),
            'eligible_actual_Go_records':int(e.sum()),'candidate_destination_effects':int(e.sum()*5*4),
            'unique_uniform_optimizations':ucase,'state_probability_optimizations':pcase,'new_optimization_calls':ucase+pcase,
            'candidate_single_update_evaluations':int(e.sum()*5),'numeric_arrays':len(a),'measurement_records':len(rows),
            'new_training_trials':0,'new_head_update_calls':0,'new_internal_update_calls':0,'new_value_update_calls':0,
            'new_random_draws':0,'fresh_model_constructions':0},
        'interpretation':{'retrospective_descriptive_only':True,'new_confirmatory_hypothesis':False,
            'tangent_global_maximum_is_expected_integrity_property':True,'finite_probability_global_optimum_claimed':False,
            'summed_fixed_state_gains_predict_new_trajectory':False,'actual_future_policy_relearned':False,
            'state_weighting_effect_separated_from_uniform_objective_control':True,'processing_cost_equalized':False,
            'parent_H_gain':False,'parent_H_acquisition':False,'parent_H_protection':True,
            'novel_input_generalization_tested':False,'new_human_experiment':False,'shared_neural_mechanism_claim':False,'education_effect_claim':False}}


def load_inputs(cfg,root):
    with np.load(Path(root)/cfg['learning_result']/'arrays.npz',allow_pickle=False) as h:return {name:h[name] for name in h.files}


def produce(cfg,root):return analyze(cfg,load_inputs(cfg,root))


def summarize(cfg,a,rows,root):return summarize_arrays(cfg,a,rows,load_inputs(cfg,root))
