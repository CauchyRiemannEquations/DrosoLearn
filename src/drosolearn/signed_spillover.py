"""Descriptive signed accounting of an immutable, already learned trajectory.

No actor, graph constructor, random generator or learning method is called.
Labels are used only by this retrospective measurement of saved data.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .signed_geometry import signed_static, checkpoint_drift

SCIENCE = {
    'study_id':'signed_spillover_v1','scope':'retrospective_descriptive_signed_spillover',
    'module':'drosolearn.signed_spillover','protocol':'docs/signed-spillover-protocol.md',
    'source_result':'results/spillover_tradeoff_v1',
    'topologies':['real','rewired','ann'],'representations':['raw','identity'],
    'alphas':[0.,.25,.5,.75,1.],'model_seeds':list(range(17301,17311)),
    'training_trials':400,'checkpoints':[0,50,100,200,400],'head_lr':.1,
    'numeric_tolerance':1e-8,'neutral_tolerance':1e-8,'labels':[1,1,-1,-1],
    'categories':['self','cross_same','cross_opposite','cross_all','all'],'external_manifest':None,
}
RESIDUALS = ['static_M','trial_g','trial_p','trial_own_score','trial_other_score_l2',
    'trial_head_step_l2','checkpoint_head','checkpoint_logits','checkpoint_probabilities',
    'margin_telescoping','probability_telescoping']
METRICS = ['static_cells','static_helpful_cells','static_harmful_cells','static_neutral_cells',
    'static_helpful_mass','static_harmful_mass','static_net_mass','actual_cells',
    'margin_helpful_cells','margin_harmful_cells','margin_neutral_cells',
    'margin_helpful_sum','margin_harmful_sum','margin_net_sum',
    'probability_helpful_cells','probability_harmful_cells','probability_neutral_cells',
    'probability_helpful_sum','probability_harmful_sum','probability_net_sum']
INTERPRETATION = {
    'prior_outcomes_known':True,'confirmatory_behavioral_hypothesis':False,
    'source_labels_used_in_analysis_only':True,'pathwise_accounting_not_isolated_causal_effect':True,
    'uniform_source_checkpoint_expectation_not_actual_next_schedule':True,
    'probability_expectation_not_sigmoid_of_expected_margin':True,
    'neutral_counts_do_not_zero_exact_masses':True,'internal_synapses_fixed_in_parent':True,
    'generalization_tested':False,'new_human_experiment':False,'shared_neural_mechanism_claim':False,
    'education_effect_claim':False,'output_cost_optimization_performed':False,
}


def require(value,message):
    if not value:raise ValueError(message)


def validate_config(cfg):
    require(type(cfg) is dict,'Configuration object required')
    fixture=cfg.get('scope')=='integration_test'
    exempt={'scope','model_seeds','training_trials','checkpoints'} if fixture else set()
    for key,value in SCIENCE.items():
        if key not in exempt:require(cfg.get(key)==value,'Fixed signed-analysis design differs: '+key)
    if fixture:
        seeds,n,steps=cfg['model_seeds'],cfg['training_trials'],cfg['checkpoints']
        require(type(seeds) is list and 1<=len(seeds)<=2 and len(set(seeds))==len(seeds)
                and all(type(s) is int and 90001<=s<=90100 for s in seeds),'Invented fixture seeds only')
        require(type(n) is int and 4<=n<=16 and n%4==0,'Bounded saved fixture')
        require(steps==sorted(set(steps)) and steps[0]==0 and steps[-1]==n
                and all(type(x) is int and 0<=x<=n for x in steps),'Fixture checkpoint axis')


def masks(labels):
    own=np.eye(4,dtype=bool);same=labels[:,None]==labels[None,:]
    return {'self':own,'cross_same':same&~own,'cross_opposite':~same,
        'cross_all':~own,'all':np.ones((4,4),dtype=bool)}


def sigmoid(z):return 1./(1.+np.exp(-np.clip(z,-40.,40.)))


def mass(values,tolerance):
    values=np.asarray(values,dtype=float)
    return {'helpful_cells':int(np.count_nonzero(values>tolerance)),
        'harmful_cells':int(np.count_nonzero(values < -tolerance)),
        'neutral_cells':int(np.count_nonzero(np.abs(values)<=tolerance)),
        'helpful':float(np.maximum(values,0).sum()),
        'harmful':float(np.maximum(-values,0).sum()),'net':float(values.sum())}


def analyze_saved(cfg,parent,parent_rows):
    """Project saved scalars; arithmetic reconstruction is not new experience."""
    validate_config(cfg)
    labels=np.asarray(cfg['labels']);n=cfg['training_trials'];seeds=cfg['model_seeds']
    shape=(3,2,5,len(seeds));tol=cfg['numeric_tolerance'];ntol=cfg['neutral_tolerance']
    axes={'model_seeds':seeds,'alphas':cfg['alphas'],'checkpoints':cfg['checkpoints'],
        'targets':(labels>0).astype(int)}
    for name,expected in axes.items():require(np.array_equal(parent[name],expected),'Parent axis differs: '+name)
    require(parent['schedules'].shape==(len(seeds),n),'Complete saved schedules required')
    for order in parent['schedules']:
        require(np.array_equal(np.bincount(order,minlength=4),np.full(4,n//4)),'Balanced saved source schedule')
    for key in ['internal_update_calls','value_update_calls','internal_steps','value_steps']:
        require(not np.any(parent['trial_'+key]),'Parent internal/value state changed')
    arrays={
        'static_target_aligned':np.empty(shape+(4,4)),
        'trial_margin_delta':np.empty(shape+(n,4)),
        'trial_probability_delta':np.empty(shape+(n,4)),
        'trial_pre_logits':np.empty(shape+(n,4)),
        'checkpoint_expected_margin_by_source':np.empty(shape+(len(cfg['checkpoints']),4,4)),
        'checkpoint_expected_probability_by_source':np.empty(shape+(len(cfg['checkpoints']),4,4)),
        'reconstruction_residuals':np.empty(shape+(len(RESIDUALS),)),
        'final_correct_probability':np.empty(shape+(4,)),
    }
    for name in ['probabilities','actions','rewards','update_factors','own_score_steps','other_score_steps','head_steps']:
        require(parent['trial_'+name].shape==shape+(n,),'Parent trial shape: '+name)
    require(len(parent_rows)==int(np.prod(shape))*n,'Complete saved CSV rows')
    # CSV order need not match the NPZ axis order. All source identities are explicit.
    seen=set()
    for row in parent_rows:
        ti=cfg['topologies'].index(row['topology']);ri=cfg['representations'].index(row['representation'])
        ai=cfg['alphas'].index(float(row['alpha']));si=seeds.index(int(row['seed']));t=int(row['trial'])-1
        at=(ti,ri,ai,si,t);require(0<=t<n and at not in seen,'Unique saved trial identity');seen.add(at)
        source=int(parent['schedules'][si,t])
        require(int(row['example_index'])==source and int(row['target'])==int(labels[source]>0),'CSV source/target differs')
        for field,key in [('probability_go','probabilities'),('action','actions'),('reward','rewards'),
                          ('update_factor','update_factors')]:
            require(abs(float(row[field])-float(parent['trial_'+key][at]))<=tol,'Saved CSV value differs: '+field)
    rows=[];groups=masks(labels)
    for ti,topology in enumerate(cfg['topologies']):
        for ri,representation in enumerate(cfg['representations']):
            prefix='t_'+topology+'_'+representation+'_'
            d_all=parent[prefix+'base_budget_base_geometry_unit_rows']
            w_all=parent[prefix+'directions'];m_all=parent[prefix+'influences']
            require(d_all.shape[:2]==(len(seeds),4) and w_all.shape==(len(seeds),5,4,d_all.shape[-1]),'Saved feature/direction shapes')
            require(m_all.shape==(len(seeds),5,4,4),'Saved influence shapes')
            for ai,alpha in enumerate(cfg['alphas']):
                for si,seed in enumerate(seeds):
                    at=(ti,ri,ai,si);d=d_all[si];w=w_all[si,ai];matrix=w@d.T
                    require(np.isfinite(d).all() and np.isfinite(w).all(),'Finite frozen geometry')
                    static=signed_static(matrix,labels,ntol)['target_aligned']
                    arrays['static_target_aligned'][at]=static
                    source=parent['schedules'][si].astype(int);g=parent['trial_update_factors'][at]
                    action=parent['trial_actions'][at];reward=parent['trial_rewards'][at]
                    saved_p=parent['trial_probabilities'][at]
                    require(np.all((action==0)|(action==1)) and np.array_equal(reward,action*labels[source]),'Obtained reward only')
                    require(np.all(g*labels[source]>=-tol),'Saved scalar must point toward source target')
                    z0=parent['checkpoint_logits'][at+(0,)];h0=parent['checkpoint_heads'][at+(0,)]
                    score_steps=g[:,None]*matrix[source]
                    logits_path=np.concatenate((z0[None],z0[None]+np.cumsum(score_steps,axis=0)),axis=0)
                    head_steps=g[:,None]*w[source]
                    heads_path=np.concatenate((h0[None],h0[None]+np.cumsum(head_steps,axis=0)),axis=0)
                    pre=logits_path[:-1];post=logits_path[1:]
                    c=score_steps*labels
                    q=sigmoid(logits_path*labels)
                    qdelta=np.diff(q,axis=0)
                    arrays['trial_margin_delta'][at]=c
                    arrays['trial_probability_delta'][at]=qdelta
                    arrays['trial_pre_logits'][at]=pre
                    checkpoints=np.asarray(cfg['checkpoints'],dtype=int)
                    saved_logits=parent['checkpoint_logits'][at]
                    saved_probs=parent['checkpoint_probabilities'][at]
                    saved_heads=parent['checkpoint_heads'][at]
                    final_correct=np.where(labels>0,saved_probs[-1],1-saved_probs[-1])
                    arrays['final_correct_probability'][at]=final_correct
                    others=np.ones((n,4),dtype=bool);others[np.arange(n),source]=False
                    own=score_steps[np.arange(n),source]
                    residuals=[np.max(np.abs(matrix-m_all[si,ai])),
                        np.max(np.abs(g-cfg['head_lr']*(action-saved_p)*reward)),
                        np.max(np.abs(sigmoid(pre)[np.arange(n),source]-saved_p)),
                        np.max(np.abs(own-parent['trial_own_score_steps'][at])),
                        np.max(np.abs(np.linalg.norm(np.where(others,score_steps,0),axis=1)-parent['trial_other_score_steps'][at])),
                        np.max(np.abs(np.linalg.norm(head_steps,axis=1)-parent['trial_head_steps'][at])),
                        np.max(np.abs(heads_path[checkpoints]-saved_heads)),
                        np.max(np.abs(logits_path[checkpoints]-saved_logits)),
                        np.max(np.abs(sigmoid(logits_path[checkpoints])-saved_probs)),
                        np.max(np.abs(c.sum(0)-labels*(saved_logits[-1]-saved_logits[0]))),
                        np.max(np.abs(qdelta.sum(0)-(final_correct-np.where(labels>0,saved_probs[0],1-saved_probs[0]))))]
                    arrays['reconstruction_residuals'][at]=residuals
                    require(np.isfinite(residuals).all() and max(residuals)<=tol,'Saved-trajectory arithmetic mismatch')
                    for ki in range(len(checkpoints)):
                        drift=checkpoint_drift(matrix,labels,saved_logits[ki],cfg['head_lr'])
                        arrays['checkpoint_expected_margin_by_source'][at+(ki,)]=drift['margin_contributions']
                        arrays['checkpoint_expected_probability_by_source'][at+(ki,)]=drift['probability_contributions']
                    for category,mask in groups.items():
                        row={'topology':topology,'representation':representation,'alpha':alpha,'seed':seed,'category':category}
                        sm=mass(static[mask],ntol);tm=mask[source]
                        mm=mass(c[tm],ntol);pm=mass(qdelta[tm],ntol)
                        row['static_cells']=int(mask.sum())
                        for key in ['helpful_cells','harmful_cells','neutral_cells']:row['static_'+key]=sm[key]
                        for key in ['helpful','harmful','net']:row['static_'+key+'_mass']=sm[key]
                        row['actual_cells']=int(tm.sum())
                        for prefix_,values in [('margin',mm),('probability',pm)]:
                            for key in ['helpful_cells','harmful_cells','neutral_cells']:row[prefix_+'_'+key]=values[key]
                            for key in ['helpful','harmful','net']:row[prefix_+'_'+key+'_sum']=values[key]
                        rows.append(row)
    require(all(np.isfinite(value).all() for value in arrays.values()),'Finite derived arrays required')
    return arrays,rows


def produce(cfg,root):
    validate_config(cfg);directory=Path(root)/cfg['source_result']
    with np.load(directory/'arrays.npz',allow_pickle=False) as saved:
        parent={key:saved[key] for key in saved.files if key.startswith(('trial_','checkpoint_','t_'))
                or key in ['model_seeds','alphas','checkpoints','targets','schedules']}
    with (directory/'measurements.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    return analyze_saved(cfg,parent,rows)


def stats(values):
    values=np.asarray(values,dtype=float)
    return {'mean':float(values.mean()),'sd':float(values.std(ddof=1)) if values.size>1 else 0.}


def summarize(cfg,arrays,rows):
    validate_config(cfg);labels=np.asarray(cfg['labels']);groups=masks(labels)
    lookup={(x['topology'],x['representation'],x['alpha'],x['seed'],x['category']):x for x in rows}
    require(len(lookup)==len(rows)==3*2*5*len(cfg['model_seeds'])*5,'Exact complete unique measurement rows')
    families=[];checkpoints=[];balanced=[];classified=[]
    for ti,topology in enumerate(cfg['topologies']):
        for ri,representation in enumerate(cfg['representations']):
            for ai,alpha in enumerate(cfg['alphas']):
                categories={}
                for seed in cfg['model_seeds']:
                    partition={name:lookup[topology,representation,alpha,seed,name] for name in cfg['categories']}
                    for metric in METRICS:
                        for left,right in [('all',['self','cross_same','cross_opposite']),('cross_all',['cross_same','cross_opposite'])]:
                            difference=abs(partition[left][metric]-sum(partition[name][metric] for name in right))
                            balanced.append(difference==0 if metric.endswith('_cells') else difference<=cfg['numeric_tolerance'])
                for category in cfg['categories']:
                    selected=[lookup[topology,representation,alpha,seed,category] for seed in cfg['model_seeds']]
                    categories[category]={key:stats([row[key] for row in selected]) for key in METRICS}
                    for row in selected:
                        for prefix,suffix in [('static','mass'),('margin','sum'),('probability','sum')]:
                            balanced.append(abs(row[prefix+'_helpful_'+suffix]-row[prefix+'_harmful_'+suffix]-row[prefix+'_net_'+suffix])<=cfg['numeric_tolerance'])
                            cells=row['static_cells' if prefix=='static' else 'actual_cells']
                            classified.append(sum(row[prefix+'_'+k+'_cells'] for k in ['helpful','harmful','neutral'])==cells)
                at=(ti,ri,ai)
                families.append({'topology':topology,'representation':representation,'alpha':alpha,
                    'role':'primary_descriptive' if (ti,ri)==(0,0) else 'descriptive_control',
                    'final_expected_accuracy':float(arrays['final_correct_probability'][at].mean()),'categories':categories})
                for ki,step in enumerate(cfg['checkpoints']):
                    categories={}
                    for category,mask in groups.items():
                        margin=arrays['checkpoint_expected_margin_by_source'][at][:,ki][:,mask].sum(-1)/4
                        probability=arrays['checkpoint_expected_probability_by_source'][at][:,ki][:,mask].sum(-1)/4
                        ms,ps=stats(margin),stats(probability)
                        categories[category]={'mean_margin_drift':ms['mean'],'sd_margin_drift':ms['sd'],
                            'mean_probability_drift':ps['mean'],'sd_probability_drift':ps['sd']}
                    checkpoints.append({'topology':topology,'representation':representation,'alpha':alpha,
                        'checkpoint':step,'categories':categories})
    residuals=arrays['reconstruction_residuals'].reshape(-1,len(RESIDUALS)).max(0)
    validation={'residual_axis':RESIDUALS,'max_residual_by_axis':residuals.tolist(),
        'all_reconstruction_and_telescoping_checks_passed':bool(np.all(residuals<=cfg['numeric_tolerance'])),
        'all_mass_balances_passed':bool(all(balanced)),'all_classification_counts_complete':bool(all(classified))}
    require(all(validation[x] for x in ['all_reconstruction_and_telescoping_checks_passed','all_mass_balances_passed','all_classification_counts_complete']),
        'Signed accounting validation failed')
    s=len(cfg['model_seeds']);n=cfg['training_trials'];k=len(cfg['checkpoints']);learners=30*s
    counts={'analyzed_parent_trial_records':learners*n,'trial_destination_cells':learners*n*4,
        'cross_destination_cells':learners*n*3,'static_ordered_input_pairs':learners*16,
        'checkpoint_destination_cells':learners*k*4,'checkpoint_source_destination_cells':learners*k*16,
        'analysis_groups':learners,'conditions':30,'model_seeds':s,'measurement_records':len(rows),
        'new_training_trials':0,'new_head_update_calls':0,'new_internal_update_calls':0,'new_value_update_calls':0,
        'new_random_draws':0,'fresh_model_constructions':0}
    return {'study_id':cfg['study_id'],'status':'completed','scope':cfg['scope'],'categories':cfg['categories'],
        'families':families,'checkpoint_expectations':checkpoints,'validation':validation,'counts':counts,
        'interpretation':dict(INTERPRETATION)}


def render(summary,arrays,out):
    """Static views of independently checked accounting, with no new inference."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(exist_ok=False)
    families=summary['families'];alphas=[0,.25,.5,.75,1]
    plotted={'actual_cross_probability_variation':[],'primary_static_signed_matrices':[]}
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,constrained_layout=True)
    for ti,topology in enumerate(['real','rewired','ann']):
        for ri,representation in enumerate(['raw','identity']):
            selected=[x for x in families if x['topology']==topology and x['representation']==representation]
            helpful=[x['categories']['cross_all']['probability_helpful_sum']['mean']/4*100 for x in selected]
            harmful=[-x['categories']['cross_all']['probability_harmful_sum']['mean']/4*100 for x in selected]
            net=[x['categories']['cross_all']['probability_net_sum']['mean']/4*100 for x in selected]
            ax=axes[ri,ti];x=np.arange(5)
            ax.bar(x-.12,helpful,.24,label='Helpful cross changes',color='#247ba0')
            ax.bar(x+.12,harmful,.24,label='Harmful cross changes',color='#d1495b')
            ax.plot(x,net,'o-',color='#222222',label='Net cross contribution')
            ax.axhline(0,color='#666666',lw=.7);ax.set_xticks(x,[str(a) for a in alphas])
            ax.set_title(topology+' / '+representation);ax.set_xlabel('Direction mixture alpha')
            if ti==0:ax.set_ylabel('Saved-path probability contribution (pp)')
            plotted['actual_cross_probability_variation'].append({'topology':topology,'representation':representation,
                'alphas':alphas,'helpful_pp':helpful,'harmful_signed_pp':harmful,'net_pp':net})
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
    fig.suptitle('Cross-input changes along the saved path; cumulative variation can exceed 100 pp',y=1.07)
    fig.savefig(out/'probability-accounting.png',dpi=150,bbox_inches='tight');plt.close(fig)
    matrices=arrays['static_target_aligned'][0,0,[0,2,4]].mean(axis=1)
    limit=max(1e-8,float(np.abs(matrices).max()));fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    labels=['ABAB','BABA','ABBA','BAAB']
    for ax,alpha,matrix in zip(axes,[0,.5,1],matrices):
        im=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit)
        ax.set_xticks(range(4),labels,rotation=30);ax.set_yticks(range(4),labels)
        ax.set_title('real / raw, alpha='+str(alpha));ax.set_xlabel('Destination input');ax.set_ylabel('Source input')
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',
                color='white' if abs(matrix[i,j])>.6*limit else 'black',fontsize=9)
        plotted['primary_static_signed_matrices'].append({'alpha':alpha,'mean_target_aligned_influence':matrix.tolist()})
    fig.colorbar(im,ax=axes,label='Unit source-correct score influence: positive helps',shrink=.85)
    fig.savefig(out/'signed-static-influence.png',dpi=150,bbox_inches='tight');plt.close(fig)
    return plotted
