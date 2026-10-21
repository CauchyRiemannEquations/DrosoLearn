"""Saved-only accounting of actual U/P state, scalar, direction and choice paths."""
import json
from pathlib import Path
import numpy as np

TOPS=('real','rewired','ann')
REPS=('raw','identity')
METHODS=('shared','isolated','feedback_selective','balanced','gated_balanced',
    'gated_self_blend','gated_uniform','gated_probability')
COMPARISONS=(('gated_uniform','gated_balanced'),('gated_probability','gated_balanced'),
    ('gated_probability','gated_uniform'))
COMPONENTS=('state','scalar','direction','control_only_Go','candidate_only_Go','neither_Go')
GROUPS=('own','other')
DIAGNOSTICS=('aligned_source_logit_before','correct_source_probability','logistic_sensitivity_value',
    'abs_update_factor','head_step_l2','own_unit_direction_influence',
    'other_unit_aligned_influence_mean','direction_l2','observed_reward','source_Go_probability',
    'own_aligned_margin_step','other_aligned_margin_step_mean')


def require(value,message):
    if not value:raise ValueError(message)


def validate_config(cfg):
    fixture=cfg.get('scope')=='integration_test'
    require(cfg.get('study_id')=='objective_pathways_v1' and (fixture or
        cfg.get('scope')=='retrospective_descriptive_objective_pathways'),'Fixed saved-only study')
    for key,value in [('topologies',TOPS),('representations',REPS),('methods',METHODS),
        ('destination_groups',GROUPS),('bridge_bits',('state','scalar','direction'))]:
        require(cfg.get(key)==list(value),'Fixed '+key)
    require(cfg.get('comparisons')==[list(x) for x in COMPARISONS],'Fixed three comparisons')
    require(type(cfg.get('training_trials')) is int and cfg['training_trials']>0,'Positive trial count')
    seeds=cfg.get('model_seeds')
    require(type(seeds) is list and bool(seeds) and len(set(seeds))==len(seeds)
        and all(type(s) is int for s in seeds),'Unique recorded computational seeds')
    require(cfg.get('numeric_tolerance')==1e-8 and cfg.get('new_confirmatory_hypothesis') is False,
        'Fixed arithmetic tolerance, no new success test')
    require((fixture and all(90001<=s<=90100 for s in seeds) and 4<=cfg['training_trials']<=16)
        or (not fixture and seeds==list(range(17801,17811)) and cfg['training_trials']==400),
        'Separate invented fixtures from exact complete actual parent cohort')
    for key,value in {'module':'drosolearn.objective_pathways','protocol':'docs/objective-pathways-protocol.md',
        'learning_result':'results/objective_learning_v1','baseline_tag':'objective-learning-v1',
        'baseline_commit':'b96a6c9d88a9f83cbe4e093272eea0ca77f245fe',
        'phase_definition':'all_three_gated_known_before_count_equals_four',
        'bridge_eligibility':'common_post_and_both_actual_Go',
        'bridge_allocation':'three_factor_symmetric_Shapley_exact_sigmoid',
        'choice_reward_phases':['all','pre','post'],'choice_reward_actions':[0,1],
        'choice_reward_values':[-1,0,1],'external_manifest':None}.items():
        require(key in cfg and cfg[key]==value,'Fixed scientific field '+key)


def sigmoid(x):return 1./(1.+np.exp(-np.clip(x,-40.,40.)))


def bridge_cells(zc,zn,gc,gn,fc,fn,y):
    require(zc.ndim==2 and zc.shape[1]==4 and zn.shape==zc.shape and fc.shape==fn.shape==zc.shape
        and gc.shape==gn.shape==(len(zc),) and y.shape==(4,)
        and all(np.isfinite(x).all() for x in (zc,zn,gc,gn,fc,fn,y)),'Finite compatible crossed ingredients')
    out=[]
    for bits in range(8):
        z=zn if bits&4 else zc;g=gn if bits&2 else gc;f=fn if bits&1 else fc
        out.append(y*(sigmoid(z+g[:,None]*f)-sigmoid(z)))
    return np.stack(out,axis=1)


def shapley3(cells):
    require(cells.shape[-2:]==(8,4) and np.isfinite(cells).all(),'Eight finite four-destination cells')
    result=[]
    for bit in (4,2,1):
        v=np.zeros_like(cells[...,0,:])
        for subset in range(8):
            if subset&bit:continue
            weight=1./6. if subset.bit_count()==1 else 1./3.
            v+=weight*(cells[...,subset|bit,:]-cells[...,subset,:])
        result.append(v)
    return np.stack(result,axis=-2)


def factual(cfg,p):
    validate_config(cfg);S=len(cfg['model_seeds']);N=cfg['training_trials'];F=(3,2,8,S,N)
    require(p['trial_actions'].shape==F and np.array_equal(p['model_seeds'],cfg['model_seeds']),
        'Complete eight-method parent axes')
    require(p['schedules'].shape==(S,N) and np.isin(p['schedules'],range(4)).all()
        and p['uniforms'].shape==(S,N) and np.all((p['uniforms']>=0)&(p['uniforms']<1))
        and p['targets'].shape==(4,) and np.isin(p['targets'],[0,1]).all(),'Fixed source, targets, uniforms')
    q=np.empty(F+(4,));margin=np.empty_like(q);zsource=np.empty(F);states={};res=0.;y=2*p['targets']-1
    def check(v):
        nonlocal res
        require(np.isfinite(v).all(),'Finite factual arithmetic')
        res=max(res,float(np.max(np.abs(v),initial=0.)))
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for si in range(S):
                d=p['g_'+top+'_'+rep+'_unit_rows'][si];src=p['schedules'][si];ix=np.arange(N)
                require(d.shape[0]==4 and np.isfinite(d).all(),'Saved fixed feature rows')
                for mi in range(8):
                    at=(ti,ri,mi,si);g=p['trial_update_factors'][at];w=p['trial_directions'][at]
                    require(w.shape==(N,d.shape[1]) and np.isfinite(w).all() and np.isfinite(g).all(),
                        'Finite complete saved update ingredients')
                    head=np.concatenate((np.zeros((1,d.shape[1])),np.cumsum(g[:,None]*w,axis=0)))
                    z=head[:-1]@d.T;f=w@d.T;zsource[at]=z[ix,src];prob=sigmoid(zsource[at])
                    act=p['trial_actions'][at];reward=p['trial_rewards'][at]
                    require(np.isin(act,[0,1]).all() and np.array_equal(act,p['uniforms'][si]<prob),
                        'Recorded own probability and common uniform action')
                    require(np.array_equal(reward,act*y[src]),'Only actual Go obtains signed reward')
                    check(prob-p['trial_probabilities'][at]);check(g-.1*(act-prob)*reward)
                    known=np.zeros(4,dtype=int)
                    for t,source in enumerate(src):
                        require(np.array_equal(known,p['trial_known_before'][at+(t,)]),'Own prefeedback memory')
                        if act[t]:known[source]=reward[t]
                        require(np.array_equal(known,p['trial_known_after'][at+(t,)]),'Obtained reward memory only')
                    delta=g[:,None]*f;q[at]=y*(sigmoid(z+delta)-sigmoid(z));margin[at]=y*delta
                    check(q[at]-p['trial_correct_probability_delta'][at])
                    check(margin[at]-p['trial_aligned_margin_delta'][at])
                    check(head[p['checkpoints']]-p['checkpoint_heads'][at])
                    check(sigmoid(head[p['checkpoints']]@d.T)-p['checkpoint_probabilities'][at])
                    unknown=(p['trial_known_before'][at]==0)&(np.arange(4)!=src[:,None])
                    check(np.sum(np.maximum(-q[at],0.)*unknown,axis=-1)
                        -p['trial_unknown_other_negative_correct_probability_sum'][at])
                    require(np.all(g[act==0]==0.),'NoGo has no actual output update')
                    if mi in (4,6,7):
                        full=np.count_nonzero(p['trial_known_before'][at],axis=-1)==4
                        require(np.all(w[full&(act==0)]==0.),'Full gated NoGo has no computed candidate direction')
                    steps=np.abs(g)*np.linalg.norm(w,axis=-1)
                    check(steps-p['trial_head_steps'][at])
                    check(np.cumsum(steps)-p['trial_cumulative_head_path_l2'][at])
                    check(np.cumsum(steps**2)-p['trial_cumulative_head_squared_l2'][at])
                    require(steps.max(initial=0.)<=.1+cfg['numeric_tolerance']
                        and steps.sum()<=N*.1+cfg['numeric_tolerance']
                        and np.sum(steps**2)<=N*.01+cfg['numeric_tolerance'], 'Same recorded correction ceilings')
                    states[at]=(z,g,f,w)
    require(res<=cfg['numeric_tolerance'],'Factual reconstruction, choices, rewards, memory, cost or effect mismatch')
    return q,margin,zsource,states,res


def analyze(cfg,p):
    q,m,zsource,states,res=factual(cfg,p);S=len(cfg['model_seeds']);N=cfg['training_trials'];P=(3,2,3,S,N)
    post=np.count_nonzero(p['trial_known_before'][:,:,4],axis=-1)==4
    require(all(np.array_equal(post,np.count_nonzero(p['trial_known_before'][:,:,mi],axis=-1)==4)
        for mi in (6,7)),'Identical actual prefeedback full gates')
    require(np.all(np.diff(post.astype(int),axis=-1)>=0),'Gate monotonicity')
    a={k:p[k].copy() for k in ('model_seeds','schedules','targets','uniforms')}
    a.update(probability_changes=q,margin_changes=m,source_logits_before=zsource,reference_post=post,
        first_post_trials=np.full((3,2,S),N+1,dtype=np.int32),
        pair_action_cases=np.empty(P,dtype=np.int8),bridge_eligible=np.empty(P,dtype=bool),
        bridge_q_cells=np.zeros(P+(8,4)),bridge_q_shapley=np.zeros(P+(3,4)),
        bridge_positive_shapley=np.zeros(P+(3,4)),bridge_negative_shapley=np.zeros(P+(3,4)),
        factual_pair_diagnostics=np.zeros(P+(2,12)))
    y=2*p['targets']-1
    for ti in range(3):
        for ri in range(2):
            for si in range(S):
                mask=post[ti,ri,si];where=np.flatnonzero(mask)
                if len(where):a['first_post_trials'][ti,ri,si]=int(where[0]+1)
                src=p['schedules'][si];other=np.arange(4)!=src[:,None]
                for ci,(new,control) in enumerate(COMPARISONS):
                    C=(ti,ri,METHODS.index(control),si);G=(ti,ri,METHODS.index(new),si);at=(ti,ri,ci,si)
                    for field in ('trial_actions','trial_rewards','trial_update_factors','trial_directions',
                        'trial_known_before','trial_known_after'):
                        require(np.array_equal(p[field][C][~mask],p[field][G][~mask]),'Identical actual gated prefix')
                    cases=p['trial_actions'][C]+2*p['trial_actions'][G];e=mask&(cases==3)
                    a['pair_action_cases'][at]=cases;a['bridge_eligible'][at]=e
                    zc,gc,fc,wc=states[C];zn,gn,fn,wn=states[G]
                    cells=bridge_cells(zc[e],zn[e],gc[e],gn[e],fc[e],fn[e],y)
                    a['bridge_q_cells'][at][e]=cells
                    for key,values in [('bridge_q_shapley',cells),('bridge_positive_shapley',np.maximum(cells,0.)),
                        ('bridge_negative_shapley',np.maximum(-cells,0.))]:a[key][at][e]=shapley3(values)
                    require(np.max(np.abs(cells[:,0]-q[C][e]),initial=0.)<=cfg['numeric_tolerance']
                        and np.max(np.abs(cells[:,-1]-q[G][e]),initial=0.)<=cfg['numeric_tolerance'],
                        'Actual two endpoints of eight cells')
                    for j,(z,g,f,w,idx) in enumerate(((zc,gc,fc,wc,C),(zn,gn,fn,wn,G))):
                        zz,gg,ff,ww=z[e],g[e],f[e],w[e];ss=src[e];row=np.arange(e.sum())
                        prob=sigmoid(zz[row,ss]);norm=np.linalg.norm(ww,axis=-1)
                        aligned_other=(y[ss,None]*y*ff*other[e]).sum(-1)/3.
                        diag=np.column_stack((y[ss]*zz[row,ss],np.where(y[ss]>0,prob,1-prob),
                            prob*(1-prob),np.abs(gg),np.abs(gg)*norm,ff[row,ss],aligned_other,norm,
                            p['trial_rewards'][idx][e],prob,y[ss]*gg*ff[row,ss],
                            (y*gg[:,None]*ff*other[e]).sum(-1)/3.))
                        a['factual_pair_diagnostics'][at][e,j]=diag
    return a,make_rows(cfg,a)


def components_for_pair(a,ti,ri,ci,si):
    at=(ti,ri,ci,si);post=a['reference_post'][ti,ri,si];cases=a['pair_action_cases'][at]
    new,control=COMPARISONS[ci];qC=a['probability_changes'][ti,ri,METHODS.index(control),si]
    qN=a['probability_changes'][ti,ri,METHODS.index(new),si];values=[]
    for key,op in [('bridge_q_shapley',lambda x:x),('bridge_positive_shapley',lambda x:np.maximum(x,0.)),
        ('bridge_negative_shapley',lambda x:np.maximum(-x,0.))]:
        v=np.zeros((6,len(post),4));v[:3]=np.moveaxis(a[key][at],1,0)
        v[3]=-op(qC)*(post&(cases==1))[:,None];v[4]=op(qN)*(post&(cases==2))[:,None];values.append(v)
    counts=[int(a['bridge_eligible'][at].sum())]*3+[int(np.count_nonzero(post&(cases==c))) for c in (1,2,0)]
    return (*values,counts)


def make_rows(cfg,a):
    rows=[]
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for ci,(new,control) in enumerate(COMPARISONS):
                for si,seed in enumerate(cfg['model_seeds']):
                    v,pos,neg,counts=components_for_pair(a,ti,ri,ci,si)
                    own=np.arange(4)==a['schedules'][si,:,None];at=(ti,ri,ci,si)
                    for group,mask in zip(GROUPS,(own,~own)):
                        for j,name in enumerate(COMPONENTS):
                            raw,plus,minus=(float((x[j]*mask).sum()) for x in (v,pos,neg))
                            rows.append(dict(topology=top,representation=rep,comparison=new+'_minus_'+control,seed=seed,
                                destination_group=group,component=name,reference_post_trials=int(a['reference_post'][ti,ri,si].sum()),
                                both_Go_trials=int(a['bridge_eligible'][at].sum()),component_trials=counts[j],
                                probability_signed_sum=raw,positive_mass_allocation=plus,negative_mass_allocation=minus,
                                endpoint_contribution=raw/4.,positive_endpoint_allocation=plus/4.,negative_endpoint_allocation=minus/4.))
    return rows


def mean_nested(values):
    if isinstance(values[0],list):return [mean_nested(list(v)) for v in zip(*values)]
    defined=[v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def path_summary(cfg,a,p):
    families=[];y=2*a['targets']-1;N=cfg['training_trials'];res=0.
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for mi,method in enumerate(METHODS):
                records=[]
                for si,seed in enumerate(cfg['model_seeds']):
                    at=(ti,ri,mi,si);src=a['schedules'][si];act=p['trial_actions'][at];r=p['trial_rewards'][at]
                    g=p['trial_update_factors'][at];w=p['trial_directions'][at];q=a['probability_changes'][at]
                    step=np.abs(g)*np.linalg.norm(w,axis=-1);prob=sigmoid(a['source_logits_before'][at])
                    own=np.arange(4)==src[:,None];unknown=(p['trial_known_before'][at]==0)&~own
                    endpoints=np.where(a['targets']==1,p['checkpoint_probabilities'][at][-1],1-p['checkpoint_probabilities'][at][-1])
                    start=np.where(a['targets']==1,p['checkpoint_probabilities'][at][0],1-p['checkpoint_probabilities'][at][0])
                    res=max(res,float(np.max(np.abs(q.sum(0)-(endpoints-start)))))
                    phases={};strata=[]
                    for phase,mask in [('all',np.ones(N,dtype=bool)),('pre',~a['reference_post'][ti,ri,si]),
                        ('post',a['reference_post'][ti,ri,si])]:
                        size=int(mask.sum());correct=np.where(y[src]>0,prob,1-prob)
                        phases[phase]={'trials':size,'Go':int(act[mask].sum()),'NoGo':int((1-act[mask]).sum()),
                            'reward_positive':int(np.count_nonzero(r[mask]>0)),'reward_negative':int(np.count_nonzero(r[mask]<0)),
                            'obtained_reward_sum':int(r[mask].sum()),'expected_obtained_reward_sum':float((prob*y[src])[mask].sum()),
                            'source_Go_probability_mean':float(prob[mask].mean()) if size else None,
                            'correct_source_probability_mean':float(correct[mask].mean()) if size else None,
                            'head_path_l2':float(step[mask].sum()),'head_squared_l2':float((step[mask]**2).sum()),
                            'unknown_other_damage':float((np.maximum(-q,0.)*unknown*mask[:,None]).sum()),
                            'own_endpoint_contribution':float((q*own*mask[:,None]).sum()/4.),
                            'other_endpoint_contribution':float((q*~own*mask[:,None]).sum()/4.)}
                        for source in range(4):
                            for action in (0,1):
                                for reward in (-1,0,1):
                                    keep=mask&(src==source)&(act==action)&(r==reward);n=int(keep.sum())
                                    strata.append({'phase':phase,'source':source,'action':action,'reward':reward,'trials':n,
                                        'obtained_reward_sum':int(r[keep].sum()),'absolute_scalar_sum':float(np.abs(g[keep]).sum()),
                                        'head_path_l2':float(step[keep].sum()),'source_probability_mean':float(prob[keep].mean()) if n else None,
                                        'own_endpoint_contribution':float((q*own*keep[:,None]).sum()/4.),
                                        'other_endpoint_contribution':float((q*~own*keep[:,None]).sum()/4.)})
                    records.append({'seed':seed,'initial_correct_probabilities':start.tolist(),
                        'final_correct_probabilities':endpoints.tolist(),'final_mean':float(endpoints.mean()),
                        'phases':phases,'source_action_reward_strata':strata})
                families.append({'topology':top,'representation':rep,'method':method,'seeds':records,
                    'final_mean':float(np.mean([r['final_mean'] for r in records]))})
    require(res<=cfg['numeric_tolerance'],'Every saved path telescopes to four final probabilities')
    return families,res


def summarize_arrays(cfg,a,rows,p):
    tol=cfg['numeric_tolerance'];families=[];account_res=0.;q=a['probability_changes'];y=2*a['targets']-1
    _,_,_,_,factual_res=factual(cfg,p)
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for ci,(new,control) in enumerate(COMPARISONS):
                records=[];ni=METHODS.index(new);co=METHODS.index(control)
                for si,seed in enumerate(cfg['model_seeds']):
                    at=(ti,ri,ci,si);C=(ti,ri,co,si);G=(ti,ri,ni,si);post=a['reference_post'][ti,ri,si]
                    own=np.arange(4)==a['schedules'][si,:,None];v,pos,neg,counts=components_for_pair(a,ti,ri,ci,si)
                    allocations=[];ap=[];an=[];fact=[];pre=[];fp=[];fn=[]
                    for mask in (own,~own):
                        alloc=[(x*mask).sum((1,2))/4. for x in (v,pos,neg)]
                        allocations.append(alloc[0].tolist());ap.append(alloc[1].tolist());an.append(alloc[2].tolist())
                        delta=q[G]-q[C];f=float((delta*mask*post[:,None]).sum()/4.);fact.append(f)
                        pre.append(float((delta*mask*~post[:,None]).sum()/4.))
                        positive=float(((np.maximum(q[G],0.)-np.maximum(q[C],0.))*mask*post[:,None]).sum()/4.)
                        negative=float(((np.maximum(-q[G],0.)-np.maximum(-q[C],0.))*mask*post[:,None]).sum()/4.)
                        fp.append(positive);fn.append(negative)
                        account_res=max(account_res,abs(sum(alloc[0])-f),abs(sum(alloc[1])-positive),abs(sum(alloc[2])-negative))
                    e=a['bridge_eligible'][at];diag=a['factual_pair_diagnostics'][at][e];cases=a['pair_action_cases'][at]
                    diff=p['checkpoint_probabilities'][G][-1]-p['checkpoint_probabilities'][C][-1]
                    endpoint=float((y*diff).mean());account_res=max(account_res,abs(sum(fact)+sum(pre)-endpoint))
                    actiondiv=np.flatnonzero(p['trial_actions'][G]!=p['trial_actions'][C])
                    scalar_res=float(np.max(np.abs(p['trial_update_factors'][G][e]-p['trial_update_factors'][C][e]
                        +.1*p['trial_rewards'][C][e]*(p['trial_probabilities'][G][e]-p['trial_probabilities'][C][e])),initial=0.))
                    account_res=max(account_res,scalar_res)
                    records.append({'seed':seed,'reference_post_trials':int(post.sum()),'both_Go_trials':int(e.sum()),
                        'case_counts':[int(np.count_nonzero(post&(cases==k))) for k in range(4)],
                        'component_endpoint_contributions':allocations,'component_positive_endpoint_allocations':ap,
                        'component_negative_endpoint_allocations':an,'factual_pre_endpoint_difference':pre,
                        'factual_post_endpoint_difference':fact,'factual_post_positive_endpoint_difference':fp,
                        'factual_post_negative_endpoint_difference':fn,'final_endpoint_difference':endpoint,
                        'diagnostic_means':diag.mean(0).tolist() if len(diag) else [[None]*12 for _ in range(2)],
                        'first_actual_action_divergence':int(actiondiv[0]+1) if len(actiondiv) else None,
                        'both_Go_scalar_probability_identity_residual':scalar_res})
                means={key:mean_nested([r[key] for r in records]) for key in records[0] if key!='seed'}
                families.append({'topology':top,'representation':rep,'candidate':new,'control':control,
                    'comparison':new+'_minus_'+control,'role':'primary' if (ti,ri)==(0,0) else 'descriptive_control',
                    'seeds':records,'means':means})
    paths,telescope=path_summary(cfg,a,p);cells=a['bridge_q_cells'];e=a['bridge_eligible']
    residuals={'factual_parent_reconstruction':factual_res,'complete_endpoint_accounting':account_res,
        'all_path_telescoping':telescope,'shapley_efficiency':float(np.max(np.abs(a['bridge_q_shapley'].sum(-2)-(cells[...,-1,:]-cells[...,0,:])))),
        'signed_positive_negative_identity':float(np.max(np.abs(a['bridge_q_shapley']-a['bridge_positive_shapley']+a['bridge_negative_shapley']))),
        'ineligible_zero_padding':float(max(np.max(np.abs(a[key][~e]),initial=0.) for key in
            ('bridge_q_cells','bridge_q_shapley','bridge_positive_shapley','bridge_negative_shapley','factual_pair_diagnostics')))}
    require(len(a)==16 and len(rows)==3*2*3*len(cfg['model_seeds'])*2*6,'Complete fixed analysis artifacts')
    require(all(np.isfinite(v) and v<=tol for v in residuals.values()),'Exact saved-only accounting gate')
    return {'study_id':cfg['study_id'],'status':'completed','scope':cfg['scope'],'families':families,'actual_paths':paths,
        'component_order':list(COMPONENTS),'diagnostic_order':list(DIAGNOSTICS),'group_order':list(GROUPS),
        'comparison_order':[list(x) for x in COMPARISONS],
        'validation':{'all_exact_accounting_gates_passed':True,'residuals':residuals,
            'same_prefeedback_full_gate_and_prefix_checked':True,'only_actual_both_Go_crossed':True,
            'one_sided_actual_choices_retained':True,'recorded_common_uniform_and_own_probabilities_checked':True,
            'both_Go_rewards_identical_and_scalar_difference_from_probability':True},
        'counts':{'analyzed_parent_trial_records':int(np.prod(q.shape[:-1])),'analyzed_destination_effects':int(q.size),
            'reference_gate_paths':int(np.prod(a['reference_post'].shape[:-1])),'comparison_paths':int(np.prod(e.shape[:-1])),
            'paired_trial_records':int(e.size),'paired_post_trial_records':int(a['reference_post'].sum()*3),
            'eligible_both_Go_trials':int(e.sum()),'corner_destination_effects':int(e.sum()*32),
            'numeric_arrays':len(a),'measurement_records':len(rows),'actual_path_strata':len(paths)*len(cfg['model_seeds'])*72,
            **{k:0 for k in ('new_training_trials','new_head_update_calls','new_internal_update_calls','new_value_update_calls',
                'new_random_draws','fresh_model_constructions','new_optimization_calls')}},
        'interpretation':{'retrospective_descriptive_only':True,'new_confirmatory_hypothesis':False,
            'accounting_allocation_not_unique_causal_explanation':True,'mixed_cells_not_relearned_policy':True,
            'NoGo_potential_rewards_or_directions_not_generated':True,'same_realized_movement_budget_claimed':False,
            'parent_U_gain':False,'parent_P_gain':False,'parent_U_acquisition':False,'parent_P_acquisition':False,
            'parent_U_protection':True,'parent_P_protection':True,'parent_state_gain':False,
            'novel_input_generalization_tested':False,'new_human_experiment':False,
            'shared_neural_mechanism_claim':False,'education_effect_claim':False}}


def load_inputs(cfg,root):
    with np.load(Path(root)/cfg['learning_result']/'arrays.npz',allow_pickle=False) as h:
        return {key:h[key] for key in h.files}


def produce(cfg,root):return analyze(cfg,load_inputs(cfg,root))


def summarize(cfg,a,rows,root):
    result=summarize_arrays(cfg,a,rows,load_inputs(cfg,root))
    prior=json.loads((Path(root)/cfg['learning_result']/'summary.json').read_text(encoding='utf8'))
    original={(f['topology'],f['representation'],f['method']):f for f in prior['families']}
    for family in result['actual_paths']:
        old=original[family['topology'],family['representation'],family['method']]
        require(abs(family['final_mean']-old['mean_final_expected_accuracy'])<=cfg['numeric_tolerance'],
            'Parent published final endpoint mean')
        require(all(x['seed']==z['seed'] and abs(x['final_mean']-z['final_expected_accuracy'])<=cfg['numeric_tolerance']
            for x,z in zip(family['seeds'],old['seeds'])),'Parent published complete seed endpoints')
    return result
