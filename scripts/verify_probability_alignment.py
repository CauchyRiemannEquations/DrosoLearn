"""Independent state-consistent saved-step probability alignment audit.

No new producer or cone solver is imported. Only a frozen parent's own full-
observation actual Go events enter counterfactual single-step arithmetic.
"""
from __future__ import annotations
import argparse,csv,json,sys
from datetime import datetime,timezone
from itertools import combinations
from pathlib import Path
import numpy as np

TOPS=("real","rewired","ann")
REPS=("raw","identity")
SIGNS=np.asarray([1.,1.,-1.,-1.])
CANDIDATES=('balanced','isolated','self_blend','uniform_score','state_probability')
OBJECTIVES=('legacy','uniform','probability')
GROUPS=('own','other','total')
FIELDS=('model_seeds','schedules','targets','reference_full_before','eligible','factual_logits',
    'factual_update_factors','observed_signs_before','helpful_mask','objectives','candidate_directions',
    'candidate_influences','candidate_q_changes','candidate_margin_changes','candidate_tangent_changes',
    'candidate_head_steps','candidate_objective_scores','factual_probability_changes','case_indices',
    'unit_rows','isolated_directions','case_keys','case_objectives','case_directions','case_projections',
    'case_multipliers','case_certificates')

def sigmoid(z):
    return 1./(1.+np.exp(-np.clip(z,-40.,40.)))


def independent_objectives(rows,source,observed,probabilities,tolerance=1e-8):
    """No evaluator signs, other learner's state or future outcome enters c."""
    d,k,p=map(np.asarray,(rows,observed,probabilities))
    demand(d.ndim==2 and d.shape[0]==4 and d.dtype.kind in 'fiu' and np.isfinite(d).all()
        and type(source) is int and source in range(4) and k.shape==(4,) and k.dtype.kind in 'biu'
        and np.isin(k,(-1,1)).all() and p.shape==(4,) and p.dtype.kind in 'fiu'
        and np.isfinite(p).all() and np.all((p>=0.)&(p<=1.)),
        "Four already observed signs and current finite four-input probabilities")
    a=k[source]*k[:,None]*d
    helpful=np.flatnonzero((np.arange(4)!=source)&(a@d[source]>tolerance))
    balanced=d[source].copy()
    if helpful.size:balanced=.5*d[source]+.5*a[helpful].mean(axis=0)
    return {"constraints":a,"balanced":balanced,"uniform":a.mean(axis=0),
        "probability":(p*(1-p))@a/4.,"helpful_indices":helpful}


def certify_case(objective,constraints,direction,projection,multipliers,tolerance=1e-8):
    """Certify an actual stored U/P solution using an independently solved bound."""
    c,a,w,p,mu=map(np.asarray,(objective,constraints,direction,projection,multipliers))
    demand(c.ndim==w.ndim==p.ndim==mu.ndim==1 and c.shape==w.shape==p.shape
        and a.shape==(4,c.size) and mu.shape==(4,) and all(x.dtype.kind in 'fiu' and np.isfinite(x).all() for x in (c,a,w,p,mu)),
        "Finite ambient U/P certificate axes")
    witness=validate_dual_witness(c,a,w,mu,tolerance)
    optimum,_=independent_support_bound(c,a,tolerance)
    achieved=float(c@w)
    demand(abs(achieved-optimum)<=tolerance and abs(witness['dual_upper_bound']-optimum)<=tolerance,
        "Independent ambient least-squares active-face global support optimum")
    certificate=np.asarray([achieved,np.linalg.norm(p),witness['dual_upper_bound'],np.linalg.norm(w),
        max(-float(np.min(a@w)),0.),float(np.max(np.abs(p-c-a.T@mu))),
        float(np.max(np.abs(mu*(a@p)))),abs(float(np.linalg.norm(p))-achieved)])
    demand(np.all(certificate[4:]<=tolerance),"U/P projection, stationarity and complementary support certificate")
    return certificate


def same_state_effects(rows,logits,signed_factor,directions,evaluator_signs):
    """Evaluate directions at one saved factual state; no new path is created."""
    d,z,w,y=map(np.asarray,(rows,logits,directions,evaluator_signs));g=np.asarray(signed_factor)
    demand(d.ndim==2 and d.shape[0]==4 and z.shape==y.shape==(4,) and w.ndim==2 and w.shape[1]==d.shape[1]
        and g.ndim==0 and all(x.dtype.kind in 'fiu' and np.isfinite(x).all() for x in (d,z,w,y,g))
        and np.isin(y,(-1,1)).all(),"Same-state finite candidate effects")
    f=w@d.T;inc=float(g)*f
    probability=y*(sigmoid(z[None,:]+inc)-sigmoid(z)[None,:])
    margin=y*inc
    return {"influences":f,"probability_changes":probability,"margin_changes":margin,
        "step_lengths":abs(float(g))*np.linalg.norm(w,axis=-1)}


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, tolerance=1e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    demand(actual.shape == expected.shape and actual.dtype.kind in "biuf"
           and np.isfinite(actual).all(), label + " finite axes")
    if expected.dtype.kind in "biu":
        demand(actual.dtype.kind in "biu" and np.array_equal(actual, expected), label + " exact")
    else:
        demand(np.allclose(actual, expected, atol=tolerance, rtol=1e-9), label + " arithmetic")


def compare_tree(actual, expected, label="summary"):
    if isinstance(expected, dict):
        demand(isinstance(actual, dict) and actual.keys() == expected.keys(), label + " keys")
        for key, value in expected.items():
            compare_tree(actual[key], value, label + "." + key)
    elif isinstance(expected, list):
        demand(isinstance(actual, list) and len(actual) == len(expected), label + " length")
        for i, value in enumerate(expected):
            compare_tree(actual[i], value, label + "." + str(i))
    elif type(expected) is bool:
        demand(type(actual) is bool and actual == expected, label + " flag")
    elif type(expected) is int:
        demand(type(actual) is int and actual == expected, label + " integer")
    elif type(expected) is float:
        demand(type(actual) in (float, int) and np.isfinite(actual)
               and np.isclose(actual, expected, atol=1e-9, rtol=1e-9), label + " numeric")
    else:
        demand(actual == expected, label + " value")


def independent_support_bound(objective, constraints, tolerance=1e-8):
    """Independently minimize certified dual bounds over every active face.

    A slightly infeasible primal direction can overstate a support value when
    nearly parallel constraints amplify roundoff. Every candidate here instead
    has exactly nonnegative multipliers, including the all-zero candidate. Its
    norm(c+A.T.mu) is a valid upper bound regardless of primal roundoff. The
    true optimum's active subset is included in this finite enumeration.
    """
    c, a = np.asarray(objective, dtype=float), np.asarray(constraints, dtype=float)
    demand(c.ndim == 1 and a.ndim == 2 and a.shape[1] == c.size and a.shape[0] <= 4
           and np.isfinite(c).all() and np.isfinite(a).all() and np.isfinite(tolerance)
           and tolerance > 0., "Independent dual enumeration finite axes")
    best_mu = np.zeros(a.shape[0])
    best_upper = float(np.linalg.norm(c))
    for count in range(1, a.shape[0] + 1):
        for subset in combinations(range(a.shape[0]), count):
            face = a[list(subset)]
            selected = -np.linalg.lstsq(face.T, c, rcond=1e-12)[0]
            if np.any(selected < -tolerance):
                continue
            mu = np.zeros(a.shape[0])
            mu[list(subset)] = np.maximum(selected, 0.)
            upper = float(np.linalg.norm(c + a.T @ mu))
            if upper < best_upper:
                best_upper, best_mu = upper, mu
    demand(np.isfinite(best_upper) and np.isfinite(best_mu).all() and np.all(best_mu >= 0.),
           "Independent nonnegative dual support bound")
    return best_upper, best_mu


def validate_dual_witness(objective, constraints, direction, multipliers, tolerance=1e-8):
    """Check the algebraic unit-ball upper bound, independently of stationarity.

    For nonnegative mu and A.w >= 0, c.w <= (c+A.T.mu).w <=
    ||c+A.T.mu||. A matching primal value therefore certifies the support
    optimum even if a solver's projection has small stationarity roundoff.
    """
    c, a, w, mu = (np.asarray(x, dtype=float) for x in
                    (objective, constraints, direction, multipliers))
    demand(c.ndim == w.ndim == mu.ndim == 1 and c.shape == w.shape
           and a.ndim == 2 and a.shape == (mu.size, c.size)
           and all(np.isfinite(x).all() for x in (c, a, w, mu)),
           "Independent dual witness finite axes")
    demand(np.all(mu >= 0.), "Independent dual multipliers exactly nonnegative")
    vector = c + a.T @ mu
    upper, achieved = float(np.linalg.norm(vector)), float(c @ w)
    gap = upper - achieved
    residuals = np.asarray([
        max(float(np.linalg.norm(w)) - 1., 0.),
        max(-float(np.min(a @ w)), 0.) if mu.size else 0.,
        abs(gap), max(-gap, 0.),
    ])
    demand(np.isfinite(residuals).all() and residuals.max() <= tolerance,
           "Independent primal/dual support bound")
    return {"dual_vector": vector, "dual_upper_bound": upper,
            "achieved_objective": achieved, "dual_optimality_gap": gap,
            "residuals": residuals}


def validate_science(cfg,allow_fixture=False):
    fixed={'study_id':'probability_alignment_v1','scope':'retrospective_descriptive_probability_alignment',
        'module':'drosolearn.probability_alignment','protocol':'docs/probability-alignment-protocol.md',
        'learning_result':'results/self_blend_learning_v1','reference_method':'gated_balanced','reference_method_index':4,
        'baseline_tag':'self-blend-factors-v1','baseline_commit':'2715f1eada36207ce77709f08874688b1425d676',
        'topologies':list(TOPS),'representations':list(REPS),'candidates':list(CANDIDATES),'objective_order':list(OBJECTIVES),
        'destination_groups':list(GROUPS),'model_seeds':list(range(17701,17711)),'training_trials':400,
        'numeric_tolerance':1e-8,'svd_rcond':1e-12,'head_lr':.1,'blend_isolated_weight':.75,'blend_balanced_weight':.25,
        'eligibility':'reference_own_known_before_full_and_actual_Go','probability_gradient_domain':'all_abs_logits_strictly_less_than_40',
        'primary_comparison':'state_probability_minus_uniform_score_exact_finite_Q_per_eligible_update',
        'new_confirmatory_hypothesis':False,'external_manifest':None}
    demand(isinstance(cfg,dict),'Independent fixed alignment design')
    fixture=cfg.get('scope')=='integration_test'
    demand(not fixture or allow_fixture,'Invented alignment fixture cannot enter the actual analysis')
    for key,value in fixed.items():
        if not (fixture and key in ('scope','model_seeds','training_trials')):
            demand(cfg.get(key)==value,'Fixed independent alignment choice: '+key)
    if fixture:
        seeds,n=cfg.get('model_seeds'),cfg.get('training_trials')
        demand(type(seeds) is list and 1<=len(seeds)<=2 and len(set(seeds))==len(seeds)
            and all(type(seed) is int and 90001<=seed<=90100 for seed in seeds),'Invented alignment seeds only')
        demand(type(n) is int and 4<=n<=16,'Bounded invented alignment record count')


def reconstruct_reference(cfg,parent):
    """Rebuild the one frozen reference's state and actual signed Go factor."""
    s,n=len(cfg['model_seeds']),cfg['training_trials'];f=(3,2,s,n)
    close(parent['model_seeds'],np.asarray(cfg['model_seeds']),'Frozen parent seed cohort')
    close(parent['targets'],np.asarray([1,1,0,0]),'Frozen environment evaluator targets')
    order=parent['schedules'];actions=parent['trial_actions']
    demand(order.shape==(s,n) and order.dtype.kind in 'iu' and np.isin(order,range(4)).all()
        and actions.shape==(3,2,6,s,n) and actions.dtype.kind in 'biu' and np.isin(actions,(0,1)).all(),
        'Complete source/action axes')
    known=parent['trial_known_before'][:,:,4]
    demand(known.shape==f+(4,) and known.dtype.kind in 'iu' and np.isin(known,(-1,0,1)).all(),
        'Actual own observed reward signs before feedback')
    g=parent['trial_update_factors'][:,:,4]
    demand(g.shape==f and g.dtype.kind in 'fiu' and np.isfinite(g).all(),'Frozen signed scalar axes')
    z=np.empty(f+(4,));q=np.empty_like(z);features=[];isolation=[]
    for ti,top in enumerate(TOPS):
        feature_reps=[];isolation_reps=[]
        for ri,rep in enumerate(REPS):
            prefix='g_'+top+'_'+rep+'_';rows=parent[prefix+'unit_rows'];iso=parent[prefix+'isolated_directions']
            demand(rows.ndim==3 and rows.shape[:2]==(s,4) and iso.shape==rows.shape and np.isfinite(rows).all(),
                'Frozen common four-input feature and isolated-direction axes')
            close(np.linalg.norm(rows,axis=-1),np.ones((s,4)),'Frozen decision row unit norms')
            for si in range(s):
                d=rows[si];at=(ti,ri,si);old=(ti,ri,4,si)
                # Ambient minimum-norm right inverses, with no new graph or encoder.
                raw=np.linalg.lstsq(d,np.eye(4),rcond=1e-12)[0].T
                close(d@raw.T,np.eye(4),'Frozen feature full-row-rank identity')
                close(iso[si],raw/np.linalg.norm(raw,axis=-1)[:,None],'Independent isolated directions')
                w=parent['trial_directions'][old]
                demand(w.shape==(n,d.shape[1]) and np.isfinite(w).all(),'Frozen reference directions')
                heads=np.vstack((np.zeros((1,d.shape[1])),np.cumsum(g[at][:,None]*w,axis=0)))
                z[at]=heads[:-1]@d.T
                q[at]=SIGNS*(sigmoid(z[at]+g[at][:,None]*(w@d.T))-sigmoid(z[at]))
                prob=sigmoid(z[at])[np.arange(n),order[si]]
                reward=parent['trial_rewards'][old]
                close(reward,actions[old]*SIGNS[order[si]],'Actual obtained reference Go rewards')
                close(g[at],cfg['head_lr']*(actions[old]-prob)*reward,'Actual same-state signed factor')
                close(heads[parent['checkpoints']],parent['checkpoint_heads'][old],'Reference checkpoint head prefixes')
                close(q[at],parent['trial_correct_probability_delta'][old],'Reference exact factual probability effects')
                observed=np.zeros(4,np.int8)
                for t,source in enumerate(order[si]):
                    close(known[at+(t,)],observed,'Reference memory contains no future signs')
                    if actions[old+(t,)]:observed[int(source)]=int(reward[t])
            feature_reps.append(rows.copy());isolation_reps.append(iso.copy())
        features.append(np.stack(feature_reps));isolation.append(np.stack(isolation_reps))
    return {'logits':z,'q':q,'g':g,'known':known,'full':np.all(known!=0,axis=-1),
        'unit_rows':np.stack(features),'isolated':np.stack(isolation)}


def expected_artifacts(cfg,parent,actual):
    state=reconstruct_reference(cfg,parent);s,n=len(cfg['model_seeds']),cfg['training_trials'];tol=cfg['numeric_tolerance']
    f=(3,2,s,n);dimension=state['unit_rows'].shape[-1]
    expected={name:parent[name].copy() for name in ('model_seeds','schedules','targets')}
    expected.update(reference_full_before=state['full'],eligible=state['full']&(parent['trial_actions'][:,:,4]==1),
        factual_logits=state['logits'],factual_update_factors=state['g'],observed_signs_before=state['known'])
    for name,tail,dtype in [('helpful_mask',(4,),bool),('objectives',(3,dimension),float),
        ('candidate_directions',(5,dimension),float),('candidate_influences',(5,4),float),
        ('candidate_q_changes',(5,4),float),('candidate_margin_changes',(5,4),float),('candidate_tangent_changes',(5,4),float),
        ('candidate_head_steps',(5,),float),('candidate_objective_scores',(5,3),float)]:
        expected[name]=np.zeros(f+tail,dtype=dtype)
    expected['factual_probability_changes']=state['q'];expected['case_indices']=np.full(f+(5,),-1,np.int32)
    expected['unit_rows']=state['unit_rows'];expected['isolated_directions']=state['isolated']
    keys=[];vectors={name:[] for name in ('objectives','directions','projections','multipliers','certificates')};cache={};legacy_bounds={}
    for ti in range(3):
        for ri in range(2):
            for si in range(s):
                d=state['unit_rows'][ti,ri,si]
                for t0 in np.flatnonzero(expected['eligible'][ti,ri,si]):
                    t=int(t0);at=(ti,ri,si,t);source=int(parent['schedules'][si,t]);k=state['known'][at];z=state['logits'][at];g=state['g'][at]
                    demand(np.all(np.abs(z)<40.) and g*k[source]>=0.,'Eligible gradient is unclipped and actual signed Go is consistent')
                    close(k,SIGNS,'Full obtained signs agree with the environment only in the evaluator audit')
                    record=independent_objectives(d,source,k,sigmoid(z),tol)
                    a=record['constraints'];objs=np.stack([record['balanced'],record['uniform'],record['probability']])
                    h=np.zeros(4,bool);h[record['helpful_indices']]=True
                    expected['helpful_mask'][at]=h;expected['objectives'][at]=objs
                    b=parent['trial_directions'][ti,ri,4,si,t];i=state['isolated'][ti,ri,si,source];mix=.75*i+.25*b
                    legacy_key=(ti,ri,si,source,*map(int,k))
                    if legacy_key not in legacy_bounds:legacy_bounds[legacy_key]=independent_support_bound(objs[0],a,tol)[0]
                    demand(abs(float(objs[0]@b)-legacy_bounds[legacy_key])<=tol,
                        'Original balanced direction independently achieves the legacy global support')
                    demand(np.linalg.norm(mix)>0.,'The fixed mixture has a nonzero norm')
                    directions=[b,i,mix/np.linalg.norm(mix)]
                    for kind,oi in ((0,1),(1,2)):
                        address=source if kind==0 else t;key=(si,ti,ri,kind,address,*map(int,k))
                        index=cache.get(key) if kind==0 else None
                        if index is None:
                            index=len(keys);demand(index<len(actual['case_keys']),'Every actual U/P state has a certificate')
                            close(actual['case_keys'][index],np.asarray(key),'First-encounter independent U/P case order')
                            objective=objs[oi];w=actual['case_directions'][index];projection=actual['case_projections'][index];mu=actual['case_multipliers'][index]
                            close(actual['case_objectives'][index],objective,'Current observed-only U/P objective')
                            cert=certify_case(objective,a,w,projection,mu,tol)
                            demand(abs(cert[3]-1.)<=tol,'Registered U/P requires a nonzero unit solution')
                            close(actual['case_certificates'][index],cert,'Independent actual support certificate arithmetic')
                            keys.append(key)
                            for name,value in [('objectives',objective),('directions',w),('projections',projection),('multipliers',mu),('certificates',cert)]:vectors[name].append(value.copy())
                            if kind==0:cache[key]=index
                        directions.append(vectors['directions'][index]);expected['case_indices'][at+(kind+3,)]=index
                    w=np.stack(directions);effect=same_state_effects(d,z,g,w,SIGNS)
                    demand(np.min(a@w.T)>=-tol and np.max(np.abs(np.linalg.norm(w,axis=-1)-1.))<=tol,
                        'Every candidate is safe under own observed signs and has the same unit norm')
                    for name,key in [('candidate_influences','influences'),('candidate_q_changes','probability_changes'),('candidate_margin_changes','margin_changes'),('candidate_head_steps','step_lengths')]:expected[name][at]=effect[key]
                    expected['candidate_directions'][at]=w
                    expected['candidate_tangent_changes'][at]=k*sigmoid(z)*(1-sigmoid(z))*g*effect['influences']
                    expected['candidate_objective_scores'][at]=w@objs.T
    demand(len(keys)==len(actual['case_keys']),'No unused, duplicate or skipped U/P certificate')
    expected['case_keys']=np.asarray(keys,np.int32).reshape(-1,9)
    for name,width in [('objectives',dimension),('directions',dimension),('projections',dimension),('multipliers',4),('certificates',8)]:
        expected['case_'+name]=np.asarray(vectors[name],float).reshape(-1,width)
    demand(set(expected)==set(FIELDS) and len(expected)==27,'Independent complete 27-array closure')
    return expected


def group_effects(effect,sources):
    """Each destination contributes one quarter of the four-input evaluator."""
    result=np.zeros((len(sources),3))
    for t,source in enumerate(sources):
        result[t,0]=effect[t,int(source)]/4.
        result[t,1]=np.delete(effect[t],int(source)).sum()/4.
        result[t,2]=effect[t].sum()/4.
    return result


def expected_rows(cfg,a):
    rows=[]
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for ci,name in enumerate(CANDIDATES):
                for si,seed in enumerate(cfg['model_seeds']):
                    at=(ti,ri,si);eligible=a['eligible'][at];count=int(eligible.sum());sources=a['schedules'][si,eligible]
                    q=group_effects(a['candidate_q_changes'][at][eligible,ci],sources)
                    base=group_effects(a['candidate_q_changes'][at][eligible,0],sources)
                    uniform=group_effects(a['candidate_q_changes'][at][eligible,3],sources)
                    margin=group_effects(a['candidate_margin_changes'][at][eligible,ci],sources)
                    tangent=group_effects(a['candidate_tangent_changes'][at][eligible,ci],sources)
                    steps=a['candidate_head_steps'][at][eligible,ci];scores=a['candidate_objective_scores'][at][eligible,ci]
                    for gi,group in enumerate(GROUPS):
                        rows.append({'topology':top,'representation':rep,'candidate':name,'seed':seed,
                            'destination_group':group,'eligible_trials':count,'exact_Q_sum':float(q[:,gi].sum()),
                            'exact_Q_mean':float(q[:,gi].mean()) if count else None,
                            'exact_Q_difference_from_B_sum':float((q[:,gi]-base[:,gi]).sum()),
                            'exact_Q_difference_from_U_sum':float((q[:,gi]-uniform[:,gi]).sum()),
                            'margin_mean_sum':float(margin[:,gi].sum()),'tangent_Q_sum':float(tangent[:,gi].sum()),
                            'head_path_l2_sum':float(steps.sum()),
                            'objective_legacy_score_mean':float(scores[:,0].mean()) if count else None,
                            'objective_uniform_score_mean':float(scores[:,1].mean()) if count else None,
                            'objective_probability_score_mean':float(scores[:,2].mean()) if count else None,
                            'mean_step_l2':float(steps.mean()) if count else None})
    return rows


def nullable_mean(values):
    if isinstance(values[0],list):return [nullable_mean([v[i] for v in values]) for i in range(len(values[0]))]
    defined=[v for v in values if v is not None]
    return float(sum(defined)/len(defined)) if defined else None


def maximum(values):
    return float(np.max(values,initial=0.))


def expected_summary(cfg,a,rows,parent):
    e=a['eligible'];tol=cfg['numeric_tolerance'];scores=a['candidate_objective_scores'][e]
    residuals={
        'original_balanced_finite_gain':maximum(np.abs(a['candidate_q_changes'][...,0,:][e]-a['factual_probability_changes'][e])),
        'legacy_global_support':maximum(np.maximum(scores[:,:,0]-scores[:,0:1,0],0.)),
        'uniform_global_support':maximum(np.maximum(scores[:,:,1]-scores[:,3:4,1],0.)),
        'probability_tangent_global_support':maximum(np.maximum(scores[:,:,2]-scores[:,4:5,2],0.)),
        'unit_norm':maximum(np.abs(np.linalg.norm(a['candidate_directions'][e],axis=-1)-1.)),
        'single_step_equal_cost':maximum(np.abs(a['candidate_head_steps'][e]-np.abs(a['factual_update_factors'][e,None]))),
        'observed_known_margin_safety':maximum(np.maximum(-a['candidate_margin_changes'][e],0.)),
        'case_certificate_residual':maximum(a['case_certificates'][:,4:]),
        'probability_tangent_identity':maximum(np.abs(a['candidate_tangent_changes'][e].mean(-1)
            -np.abs(a['factual_update_factors'][e,None])*scores[:,:,2]))}
    demand(all(np.isfinite(v) and v<=tol for v in residuals.values()),'Independent support, equal movement, safety and tangent gates')
    for key in ('helpful_mask','objectives','candidate_directions','candidate_influences','candidate_q_changes',
        'candidate_margin_changes','candidate_tangent_changes','candidate_head_steps','candidate_objective_scores'):
        demand(not np.any(a[key][~e]),'Excluded records are zero: '+key)
    demand(np.all(a['case_indices'][~e]==-1),'No excluded partial/NoGo solver calls')
    families=[]
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for ci,name in enumerate(CANDIDATES):
                records=[]
                for si,seed in enumerate(cfg['model_seeds']):
                    at=(ti,ri,si);mask=e[at];sources=a['schedules'][si,mask];count=int(mask.sum())
                    q=group_effects(a['candidate_q_changes'][at][mask,ci],sources)
                    base=group_effects(a['candidate_q_changes'][at][mask,0],sources)
                    uniform=group_effects(a['candidate_q_changes'][at][mask,3],sources)
                    tangent=group_effects(a['candidate_tangent_changes'][at][mask,ci],sources)
                    difference=q[:,2]-uniform[:,2];steps=a['candidate_head_steps'][at][mask,ci]
                    records.append({'seed':seed,'eligible_trials':count,'exact_Q_sum':q.sum(0).tolist(),
                        'exact_Q_mean':q.mean(0).tolist() if count else [None]*3,
                        'difference_from_B_sum':(q-base).sum(0).tolist(),
                        'difference_from_B_mean':(q-base).mean(0).tolist() if count else [None]*3,
                        'difference_from_U_sum':(q-uniform).sum(0).tolist(),
                        'difference_from_U_mean':(q-uniform).mean(0).tolist() if count else [None]*3,
                        'positive_vs_U_trials':int(np.count_nonzero(difference>0.)),
                        'negative_vs_U_trials':int(np.count_nonzero(difference<0.)),
                        'zero_vs_U_trials':int(np.count_nonzero(difference==0.)),
                        'negative_beyond_tolerance_vs_U_trials':int(np.count_nonzero(difference < -tol)),
                        'worst_Q_difference_vs_U':float(difference.min()) if count else None,
                        'mean_head_step':float(steps.mean()) if count else None,'head_path_l2_sum':float(steps.sum()),
                        'objective_score_mean':a['candidate_objective_scores'][at][mask,ci].mean(0).tolist() if count else [None]*3,
                        'tangent_Q_sum':tangent.sum(0).tolist(),
                        'tangent_error_mean':(q-tangent).mean(0).tolist() if count else [None]*3,
                        'old_helpful_count_mean':float(a['helpful_mask'][at][mask].sum(-1).mean()) if count else None})
                families.append({'topology':top,'representation':rep,'candidate':name,
                    'role':'primary' if (ti,ri,ci)==(0,0,4) else 'descriptive_control','seeds':records,
                    'means':{key:nullable_mean([record[key] for record in records]) for key in records[0] if key!='seed'}})
    primary=[family for family in families if family['role']=='primary'][0];records=primary['seeds']
    kinds=a['case_keys'][:,3];uniform_count=int(np.count_nonzero(kinds==0));probability_count=int(np.count_nonzero(kinds==1))
    demand(probability_count==int(e.sum()) and uniform_count<=24*len(cfg['model_seeds'])
        and len(rows)==90*len(cfg['model_seeds']),'Exact state-dependent/static optimization and measurement inventory')
    return {'study_id':cfg['study_id'],'scope':cfg['scope'],'status':'completed','families':families,
        'candidate_order':list(CANDIDATES),'objective_order':list(OBJECTIVES),'group_order':list(GROUPS),
        'finite_alignment_summary':{'mean_exact_P_minus_U_per_update':primary['means']['difference_from_U_mean'][2],
            'positive_seed_means':sum(r['difference_from_U_mean'][2] is not None and r['difference_from_U_mean'][2]>0 for r in records),
            'pooled_positive_trials':sum(r['positive_vs_U_trials'] for r in records),
            'pooled_negative_trials':sum(r['negative_vs_U_trials'] for r in records),'pooled_zero_trials':sum(r['zero_vs_U_trials'] for r in records),
            'pooled_negative_beyond_tolerance_trials':sum(r['negative_beyond_tolerance_vs_U_trials'] for r in records),
            'worst_Q_difference':min((r['worst_Q_difference_vs_U'] for r in records if r['worst_Q_difference_vs_U'] is not None),default=None),
            'new_learned_performance_claim':False},
        'validation':{'all_numeric_information_and_support_checks_passed':True,'residuals':residuals,
            'new_objectives_use_own_observed_signs_only':True,'same_factual_state_action_reward_and_scalar':True,
            'all_eligible_logits_strictly_inside_clipping_domain':True,'excluded_partial_and_NoGo_solver_calls':0},
        'counts':{'saved_reference_records':int(e.size),'saved_reference_destination_effects':int(e.size*4),
            'eligible_actual_Go_records':int(e.sum()),'candidate_destination_effects':int(e.sum()*20),
            'unique_uniform_optimizations':uniform_count,'state_probability_optimizations':probability_count,
            'new_optimization_calls':uniform_count+probability_count,'candidate_single_update_evaluations':int(e.sum()*5),
            'numeric_arrays':27,'measurement_records':len(rows),'new_training_trials':0,'new_head_update_calls':0,
            'new_internal_update_calls':0,'new_value_update_calls':0,'new_random_draws':0,'fresh_model_constructions':0},
        'interpretation':{'retrospective_descriptive_only':True,'new_confirmatory_hypothesis':False,
            'tangent_global_maximum_is_expected_integrity_property':True,'finite_probability_global_optimum_claimed':False,
            'summed_fixed_state_gains_predict_new_trajectory':False,'actual_future_policy_relearned':False,
            'state_weighting_effect_separated_from_uniform_objective_control':True,'processing_cost_equalized':False,
            'parent_H_gain':False,'parent_H_acquisition':False,'parent_H_protection':True,
            'novel_input_generalization_tested':False,'new_human_experiment':False,'shared_neural_mechanism_claim':False,'education_effect_claim':False}}


def check_artifacts(actual,wanted,rows,wanted_rows):
    demand(set(actual)==set(wanted)==set(FIELDS),'Complete independent 27-array closure')
    for name,value in wanted.items():close(actual[name],value,'Independent state-consistent objective array: '+name)
    demand(len(rows)==len(wanted_rows),'All ordered seed/group measurement rows')
    for actual_row,expected_row in zip(rows,wanted_rows):
        demand(tuple(actual_row)==tuple(expected_row),'Exact 17-column measurement schema')
        for key,value in expected_row.items():
            if value is None:demand(actual_row[key] is None or actual_row[key]=='','Undefined exposure mean stays null: '+key)
            elif isinstance(value,str):demand(actual_row[key]==value,'Exact measurement identity: '+key)
            else:
                number=float(actual_row[key]);demand(np.isfinite(number) and (number==value if type(value) is int else np.isclose(number,value,atol=1e-9,rtol=1e-9)),
                    'Independent count or measurement arithmetic: '+key)


def verify_science(cfg,parent,arrays,rows,summary,allow_fixture=False):
    validate_science(cfg,allow_fixture)
    wanted=expected_artifacts(cfg,parent,arrays);wanted_rows=expected_rows(cfg,wanted)
    check_artifacts(arrays,wanted,rows,wanted_rows)
    expected=expected_summary(cfg,wanted,wanted_rows,parent)
    compare_tree(summary,expected,'Independent full probability-alignment summary')
    cases=len(wanted['case_keys']);eligible=int(wanted['eligible'].sum())
    return {'numeric_arrays_checked':27,'saved_reference_records_checked':int(wanted['eligible'].size),
        'saved_reference_destination_effects_checked':int(wanted['factual_probability_changes'].size),
        'eligible_actual_Go_records_checked':eligible,'candidate_destination_effects_checked':eligible*20,
        'measurement_records_checked':len(rows),'head_prefixes_and_factual_scalar_effects_checked':True,
        'own_prefeedback_observed_signs_and_full_gate_checked':True,'all_five_same_state_signed_updates_checked':True,
        'uniform_and_probability_global_support_independently_reoptimized':True,
        'independent_ambient_support_reoptimizations':cases,'independent_active_faces_per_support':16,
        'legacy_balanced_global_support_independently_reoptimized':True,
        'independent_legacy_support_reoptimizations':int(np.count_nonzero(wanted['case_keys'][:,3]==0)),
        'signed_finite_probability_failures_and_null_means_retained':True,
        'new_training_trials':0,'new_random_draws':0,'fresh_model_constructions':0,
        'scientific_producer_imported':False,'new_cone_solver_imported':False}


def load_rows(path):
    with Path(path).open(encoding='utf8',newline='') as handle:
        reader=csv.DictReader(handle)
        demand(reader.fieldnames is not None and len(reader.fieldnames)==len(set(reader.fieldnames)),'Unique measurement field names')
        return list(reader)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    for name in ('config','registration','result-dir','publication-proof','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.root.resolve()/'src'))
    from drosolearn.probability_envelope import verify_envelope,sha,write_new
    cfg,_,metadata=verify_envelope(args.root,args.config,args.registration,args.result_dir,args.publication_proof)
    with np.load(args.root/cfg['learning_result']/'arrays.npz',allow_pickle=False) as handle:parent={name:handle[name] for name in handle.files}
    with np.load(args.result_dir/'arrays.npz',allow_pickle=False) as handle:arrays={name:handle[name] for name in handle.files}
    rows=load_rows(args.result_dir/'measurements.csv');summary=json.loads((args.result_dir/'summary.json').read_text(encoding='utf8'))
    checks=verify_science(cfg,parent,arrays,rows,summary)
    demand(metadata['counts']==summary['counts'],'Metadata exact independent numeric inventory')
    report={'status':'passed','study_id':cfg['study_id'],'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'registration_sha256':sha(args.registration),'summary_sha256':sha(args.result_dir/'summary.json'),
        'scientific_producer_imported':False,'new_helper_imported':False,'counts':checks}
    write_new(args.out,report);print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
