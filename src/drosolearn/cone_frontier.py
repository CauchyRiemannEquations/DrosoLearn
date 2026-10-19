"""Certified same-record unit-cap direction comparisons, without new learning."""
from pathlib import Path
import numpy as np
from .cone_geometry import cone_direction

TOPS=['real','rewired','ann'];REPS=['raw','identity'];PHASES=['all','partial','full']
CANDIDATES=['parent_feedback','shared_reference','isolated_reference','observed_own','observed_balanced','oracle_own','oracle_balanced']
SCIENCE={'study_id':'cone_frontier_v1','scope':'retrospective_descriptive_cone_frontier','module':'drosolearn.cone_frontier',
    'protocol':'docs/cone-frontier-protocol.md','source_result':'results/selective_spillover_v1',
    'topologies':TOPS,'representations':REPS,'model_seeds':list(range(17401,17411)),'training_trials':400,
    'head_lr':.1,'numeric_tolerance':1e-8,'neutral_tolerance':1e-8,'svd_rcond':1e-12,'labels':[1,1,-1,-1],
    'parent_method':'feedback_selective','phases':PHASES,'candidates':CANDIDATES,'external_manifest':None}
METRICS=['trial_count','updated_trials','own_margin','helpful_margin','harmful_margin','unknown_harmful_margin',
    'total_margin','cumulative_step_l2','max_step_l2','own_retention','help_retention','joint_nondegrading_updates',
    'joint_strict_improvement_updates','known_safety_violations']
INTERPRETATION={'prior_outcomes_known':True,'confirmatory_behavioral_hypothesis':False,
    'same_saved_scalar_not_autonomous_policy':True,'no_unknown_signs_in_observed_constraints':True,
    'oracle_is_full_information_safety_reference':True,'oracle_not_universal_observed_upper_bound':True,
    'own_optimum_not_joint_improvement_guarantee':True,'balanced_not_complete_Pareto_frontier':True,
    'score_not_accuracy':True,'unknown_harm_evaluator_only':True,'unit_cap_not_equal_processing_cost':True,
    'generalization_tested':False,'new_human_experiment':False,'shared_neural_mechanism_claim':False,'education_effect_claim':False}


def require(value,message):
    if not value:raise ValueError(message)


def validate_config(cfg):
    require(type(cfg) is dict,'Configuration object required')
    fixture=cfg.get('scope')=='integration_test';exempt={'scope','model_seeds','training_trials'} if fixture else set()
    for key,value in SCIENCE.items():
        if key not in exempt:require(cfg.get(key)==value,'Fixed cone analysis differs:'+key)
    if fixture:
        seeds,n=cfg['model_seeds'],cfg['training_trials']
        require(type(seeds) is list and 1<=len(seeds)<=2 and len(set(seeds))==len(seeds)
                and all(type(x) is int and 90001<=x<=90100 for x in seeds),'Invented fixture seeds only')
        require(type(n) is int and 4<=n<=16 and n%4==0,'Bounded invented saved path')


def candidate_problem(d,source,signs,objective):
    d=np.asarray(d,dtype=float);k=np.asarray(signs)
    require(d.ndim==2 and d.shape[0]==4 and np.isfinite(d).all(),'Four finite feature rows')
    require(type(source) in (int,np.int64,np.int32) and 0<=source<4,'Valid source')
    require(k.shape==(4,) and np.isin(k,[-1,0,1]).all() and k[source]!=0,'Actually known source reward required')
    require(objective in ['own','balanced'],'Prespecified objective')
    known=np.flatnonzero(k!=0);a=k[source]*k[known,None]*d[known]
    base=d@d[source];helpful=np.flatnonzero((k!=0)&(k[source]*k*base>1e-8)&(np.arange(4)!=source))
    c=d[source].copy()
    if objective=='balanced' and len(helpful):c=.5*c+.5*(k[source]*k[helpful,None]*d[helpful]).mean(0)
    return a,c,known,helpful


def stats(values):
    values=list(values);valid=[float(v) for v in values if v is not None]
    require(np.isfinite(valid).all(),'Finite valid statistics')
    return {'mean':float(np.mean(valid)) if valid else None,'sd':float(np.std(valid,ddof=1)) if len(valid)>1 else (0. if valid else None),
            'valid_count':len(valid),'total_count':len(values)}


def analyze_saved(cfg,parent):
    validate_config(cfg);seeds=cfg['model_seeds'];n=cfg['training_trials'];tol=cfg['numeric_tolerance'];y=np.asarray(cfg['labels'])
    require(np.array_equal(parent['model_seeds'],seeds) and np.array_equal(parent['targets'],(y>0).astype(int)),'Fixed parent axes')
    source=parent['schedules'];require(source.shape==(len(seeds),n) and np.isin(source,range(4)).all(),'Complete saved sources')
    g=parent['trial_update_factors'][:,:,3];known=parent['trial_known_after'][:,:,3]
    require(g.shape==(3,2,len(seeds),n) and known.shape==g.shape+(4,) and np.isfinite(g).all(),'Complete finite parent feedback path')
    action=parent['trial_actions'][:,:,3];reward=parent['trial_rewards'][:,:,3];p=parent['trial_probabilities'][:,:,3]
    require(np.isin(action,[0,1]).all() and np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        and np.max(np.abs(g))<=cfg['head_lr']+tol and np.all(g*y[source]>=-tol),'Valid saved probabilities, actions and bounded scalar')
    require(np.array_equal(reward,action*y[source]) and np.array_equal(g!=0,action==1)
            and np.max(np.abs(g-cfg['head_lr']*(action-p)*reward))<=tol,'Actual feedback scalar identity')
    require(np.isin(known,[-1,0,1]).all() and np.array_equal(parent['trial_routing_signs'][:,:,3],known),'Observed-only routing')
    shape=g.shape;dim=parent['g_real_raw_unit_rows'].shape[-1]
    arrays={'case_indices':np.full(shape+(4,),-1,dtype=np.int32),
        'candidate_influences':np.zeros(shape+(7,4)),'candidate_margin_delta':np.zeros(shape+(7,4)),
        'candidate_step_lengths':np.zeros(shape+(7,)), 'known_after':known.copy(),'sources':source.copy(),
        'update_factors':g.copy(),'full_known_after':np.all(known!=0,axis=-1)}
    cache={};records=[];directions=[];projections=[];multipliers=[];objectives=[];constraints=[];certificates=[]
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            prefix='g_'+top+'_'+rep+'_'
            for si,seed in enumerate(seeds):
                d=parent[prefix+'unit_rows'][si];gram=d@d.T
                require(d.shape==(4,dim) and np.max(np.abs(np.linalg.norm(d,axis=1)-1))<=tol
                        and np.max(np.abs(gram-parent[prefix+'base_influences'][si]))<=tol,'Frozen unit feature identity')
                observed=np.zeros(4,dtype=np.int8)
                for t,i0 in enumerate(source[si]):
                    i=int(i0);at=(ti,ri,si,t);original=(ti,ri,3,si,t)
                    require(np.array_equal(parent['trial_known_before'][original],observed),'Actual reward memory continuity')
                    if action[at]:observed[i]=reward[at]
                    require(np.array_equal(known[at],observed),'Go-only observed reward memory')
                    if g[at]==0:continue
                    require(known[at][i]==y[i],'Current actual Go source sign known')
                    previous=parent['trial_directions'][original]
                    require(np.max(np.abs(d@previous-parent['trial_actual_influence'][original]))<=tol,'Stored parent direction influence')
                    refs=[previous,d[i],parent[prefix+'isolated_directions'][si,i]]
                    for ci,w in enumerate(refs):
                        arrays['candidate_influences'][at+(ci,)]=d@w
                        arrays['candidate_step_lengths'][at+(ci,)]=abs(g[at])*np.linalg.norm(w)
                    for offset,(scope,objective) in enumerate([(0,'own'),(0,'balanced'),(1,'own'),(1,'balanced')]):
                        k=known[at].copy() if scope==0 else y.copy();oi=int(objective=='balanced')
                        key=(ti,ri,si,i,scope,oi,*map(int,k))
                        if key not in cache:
                            a,c,indices,_=candidate_problem(d,i,k,objective)
                            result=cone_direction(a,c,cfg['svd_rcond'],tol)
                            require(result['numeric_qualified'],'Certified optimizer required')
                            padded_a=np.zeros((4,dim));padded_a[indices]=a
                            lam=np.zeros(4);lam[indices]=result['multipliers']
                            cache[key]=len(records);records.append(key);directions.append(result['direction'])
                            projections.append(result['projected_objective']);multipliers.append(lam)
                            objectives.append(c);constraints.append(padded_a)
                            certificates.append([result['achieved_objective'],result['objective_upper_bound'],result['dual_upper_bound'],
                                result['direction_norm'],result['residuals']['primal_violation'],result['residuals']['stationarity'],
                                result['residuals']['complementarity'],result['residuals']['support_gap']])
                        index=cache[key];arrays['case_indices'][at+(offset,)]=index;w=directions[index]
                        arrays['candidate_influences'][at+(offset+3,)]=d@w
                        arrays['candidate_step_lengths'][at+(offset+3,)]=abs(g[at])*np.linalg.norm(w)
                    arrays['candidate_margin_delta'][at]=arrays['candidate_influences'][at]*g[at]*y
    require(records,'At least one actual saved Go update required')
    arrays.update(case_keys=np.asarray(records,dtype=np.int32),case_directions=np.stack(directions),case_projections=np.stack(projections),
        case_multipliers=np.stack(multipliers),case_objectives=np.stack(objectives),case_constraints=np.stack(constraints),
        case_certificates=np.asarray(certificates))
    require(all(np.isfinite(x).all() for x in arrays.values()),'Finite derived case/trial arrays')
    return arrays,measurements(cfg,arrays,parent)


def measurements(cfg,arrays,parent):
    rows=[];y=np.asarray(cfg['labels']);tol=cfg['numeric_tolerance'];n=cfg['training_trials']
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            for ci,candidate in enumerate(CANDIDATES):
                for si,seed in enumerate(cfg['model_seeds']):
                    source=arrays['sources'][si].astype(int);at=(ti,ri,si);delta=arrays['candidate_margin_delta'][at][:,ci]
                    baseline=arrays['candidate_margin_delta'][at][:,0];influence=arrays['candidate_influences'][at][:,ci]
                    d=parent['g_'+top+'_'+rep+'_unit_rows'][si];gram=d@d.T
                    own=np.eye(4,dtype=bool)[source];cross=~own
                    helpful=(y[source,None]*y*gram[source]>cfg['neutral_tolerance'])&cross
                    unknown=arrays['known_after'][at]==0;nonzero=arrays['update_factors'][at]!=0
                    own_delta=delta[np.arange(n),source];oldown=baseline[np.arange(n),source]
                    help_delta=np.sum(np.where(helpful,np.maximum(delta,0),0),axis=1)
                    oldhelp=np.sum(np.where(helpful,np.maximum(baseline,0),0),axis=1)
                    k=arrays['known_after'][at];safety=np.any((k[np.arange(n),source,None]*k*influence < -tol)&(k!=0),axis=1)&nonzero
                    joint=(own_delta>=oldown-tol)&(help_delta>=oldhelp-tol)&nonzero
                    strict=joint&((own_delta>oldown+tol)|(help_delta>oldhelp+tol))
                    for phase in PHASES:
                        mask=np.ones(n,bool) if phase=='all' else arrays['full_known_after'][at] if phase=='full' else ~arrays['full_known_after'][at]
                        row={'topology':top,'representation':rep,'candidate':candidate,'seed':seed,'phase':phase,
                            'trial_count':int(mask.sum()),'updated_trials':int(np.count_nonzero(mask&nonzero)),
                            'own_margin':float(own_delta[mask].sum()),'helpful_margin':float(help_delta[mask].sum()),
                            'harmful_margin':float(np.where(cross,np.maximum(-delta,0),0)[mask].sum()),
                            'unknown_harmful_margin':float(np.where(cross&unknown,np.maximum(-delta,0),0)[mask].sum()),
                            'total_margin':float(delta[mask].sum()),
                            'cumulative_step_l2':float(arrays['candidate_step_lengths'][at][:,ci][mask].sum()),
                            'max_step_l2':float(arrays['candidate_step_lengths'][at][:,ci][mask].max()) if mask.any() else 0.}
                        od=float(oldown[mask].sum());hd=float(oldhelp[mask].sum())
                        row.update(own_retention=row['own_margin']/od if od>0 else None,help_retention=row['helpful_margin']/hd if hd>0 else None,
                            joint_nondegrading_updates=int(np.count_nonzero(joint&mask)),joint_strict_improvement_updates=int(np.count_nonzero(strict&mask)),
                            known_safety_violations=int(np.count_nonzero(safety&mask)))
                        rows.append(row)
    return rows


def produce(cfg,root):
    validate_config(cfg)
    with np.load(Path(root)/cfg['source_result']/'arrays.npz',allow_pickle=False) as saved:parent={key:saved[key] for key in saved.files}
    return analyze_saved(cfg,parent)


def summarize(cfg,arrays,rows):
    validate_config(cfg);seeds=cfg['model_seeds'];tol=cfg['numeric_tolerance']
    by={(r['topology'],r['representation'],r['candidate'],r['seed'],r['phase']):r for r in rows}
    require(len(by)==len(rows)==6*7*len(seeds)*3,'Complete unique candidate records');families=[];balance=True;safety=True
    additive=[k for k in METRICS if k not in ['max_step_l2','own_retention','help_retention']]
    for top in TOPS:
        for rep in REPS:
            for candidate in CANDIDATES:
                phases={}
                for phase in PHASES:
                    group=[by[top,rep,candidate,seed,phase] for seed in seeds]
                    pooled={}
                    for name,field in [('own_retention','own_margin'),('help_retention','helpful_margin')]:
                        denominator=sum(by[top,rep,'parent_feedback',seed,phase][field] for seed in seeds)
                        pooled[name]=sum(r[field] for r in group)/denominator if denominator>0 else None
                    phases[phase]={'metrics':{key:stats([r[key] for r in group]) for key in METRICS},'pooled':pooled}
                    if candidate in ['parent_feedback','observed_own','observed_balanced','oracle_own','oracle_balanced']:
                        safety&=all(r['known_safety_violations']==0 for r in group)
                for seed in seeds:
                    for key in additive:
                        x=[by[top,rep,candidate,seed,p][key] for p in PHASES];balance&=abs(x[0]-x[1]-x[2])<=tol
                role='primary_descriptive' if (top,rep,candidate)==('real','raw','observed_balanced') else (
                    'prespecified_complement' if candidate=='observed_own' else 'privileged_diagnostic' if candidate.startswith('oracle_') else 'descriptive_control')
                families.append({'topology':top,'representation':rep,'candidate':candidate,'role':role,'phases':phases})
    cert=arrays['case_certificates'];error=float(max(np.max(np.abs(cert[:,4:])),np.max(np.abs(cert[:,2]-cert[:,0]))))
    length=arrays['candidate_step_lengths'];step_violation=float(np.maximum(length-np.abs(arrays['update_factors'])[...,None],0).max())
    unit=bool(step_violation<=tol and np.max(cert[:,3])<=1+tol)
    require(unit and safety and balance and error<=tol,'Frozen unit-cap/safety/certificate/phase gate')
    count=6*len(seeds)*cfg['training_trials']
    return {'study_id':cfg['study_id'],'status':'completed','scope':cfg['scope'],'candidates':CANDIDATES,'phases':PHASES,'families':families,
        'validation':{'all_unit_caps_passed':True,'all_observed_and_oracle_constraints_passed':True,'all_case_certificates_passed':True,
            'all_phase_balances_passed':True,'max_certificate_residual':error,'max_step_bound_violation':step_violation},
        'counts':{'analyzed_parent_trial_records':count,'nonzero_parent_updates':int(np.count_nonzero(arrays['update_factors'])),
            'candidate_trial_records':count*7,'candidate_destination_effects':count*7*4,'unique_optimization_cases':len(arrays['case_keys']),
            'measurement_records':len(rows),'conditions':6,'model_seeds':len(seeds),'new_training_trials':0,'new_head_update_calls':0,
            'new_internal_update_calls':0,'new_value_update_calls':0,'new_random_draws':0,'fresh_model_constructions':0},
        'interpretation':dict(INTERPRETATION)}


def render(summary,arrays,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(exist_ok=False);plotted={'ratios':[]}
    fig,axes=plt.subplots(2,3,figsize=(15,8));fig.subplots_adjust(left=.065,right=.98,bottom=.13,top=.81,wspace=.30,hspace=.56)
    names=['parent_feedback','observed_own','observed_balanced','oracle_own','oracle_balanced'];x=np.arange(5)
    for ti,top in enumerate(TOPS):
        for ri,rep in enumerate(REPS):
            ax=axes[ri,ti]
            for metric,offset,color,label in [('own_retention',-.16,'#287a9f','Own correction'),('help_retention',.16,'#179a83','Helpful cross')]:
                values=[next(f for f in summary['families'] if (f['topology'],f['representation'],f['candidate'])==(top,rep,name))['phases']['all']['pooled'][metric] for name in names]
                ax.bar(x+offset,values,.3,color=color,label=label);plotted['ratios'].append({'topology':top,'representation':rep,'metric':metric,'candidates':names,'values':values})
            ax.axhline(1,color='#888888',ls=':',lw=.8);ax.set_xticks(x,['stored','own','balanced','oracle own','oracle bal'],rotation=25)
            ax.set_title(top+' / '+rep)
            if ti==0:ax.set_ylabel('Score mass / stored selective score mass')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False)
    fig.suptitle('Same saved updates, unit cap and known-sign safety; not new learning',y=.98)
    fig.savefig(out/'cone-retention.png',dpi=150);plt.close(fig)
    return plotted
