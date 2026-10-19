"""Three-stage score accounting on a frozen, already observed learning path.

The pre-normalization reference uses the recorded scalar; it is not an
autonomous alternative learner and may exceed the original movement budget.
"""
from pathlib import Path
import csv
import numpy as np

TOPOLOGIES=['real','rewired','ann']
REPRESENTATIONS=['raw','identity']
METHODS=['shared','isolated','oracle_selective','feedback_selective']
PHASES=['all','partial','full']
CATEGORIES=['self','cross_helpful','cross_harmful','cross_neutral','cross_unknown',
            'cross_same','cross_opposite','cross_all','all']
SCIENCE={'study_id':'normalization_loss_v1','scope':'retrospective_descriptive_normalization_loss',
    'module':'drosolearn.normalization_loss','protocol':'docs/normalization-loss-protocol.md',
    'source_result':'results/selective_spillover_v1','topologies':TOPOLOGIES,'representations':REPRESENTATIONS,
    'methods':METHODS,'model_seeds':list(range(17401,17411)),'training_trials':400,
    'checkpoints':[0,50,100,200,400],'head_lr':.1,'numeric_tolerance':1e-8,'neutral_tolerance':1e-8,
    'labels':[1,1,-1,-1],'phases':PHASES,'categories':CATEGORIES,'external_manifest':None}
RESIDUALS=['gram','actual_influence','desired_influence','scaled_desired','reciprocal_scale','own_score',
           'other_score','head_step','scalar','unit_direction','checkpoint_logits','stage_accounting']
METRICS=['trial_count','updated_trials','cells','updated_cells','scale_mean_all','scale_mean_updated',
    'raw_norm_mean','scale_min','scale_max','baseline_helpful','baseline_harmful','baseline_net',
    'desired_helpful','desired_harmful','desired_net','actual_helpful','actual_harmful','actual_net',
    'suppression_net','normalization_net','helpful_loss','harmful_reduction','helpful_retention','harmful_retention']
COUNT_METRICS=['trial_count','updated_trials','cells','updated_cells']
ADDITIVE=['cells','updated_cells','baseline_helpful','baseline_harmful','baseline_net',
    'desired_helpful','desired_harmful','desired_net','actual_helpful','actual_harmful','actual_net',
    'suppression_net','normalization_net','helpful_loss','harmful_reduction']
INTERPRETATION={'prior_outcomes_known':True,'confirmatory_behavioral_hypothesis':False,
    'saved_scalar_reference_not_counterfactual_policy':True,'score_loss_not_accuracy_loss':True,
    'phase_is_descriptive_not_causal':True,'true_labels_evaluator_only':True,'unknown_category_overlaps':True,
    'neutral_counts_do_not_zero_exact_masses':True,'internal_synapses_fixed_in_parent':True,
    'generalization_tested':False,'new_human_experiment':False,'shared_neural_mechanism_claim':False,
    'education_effect_claim':False}


def require(value,message):
    if not value:raise ValueError(message)


def validate_config(cfg):
    require(type(cfg) is dict,'Configuration object required')
    fixture=cfg.get('scope')=='integration_test'
    exempt={'scope','model_seeds','training_trials','checkpoints'} if fixture else set()
    for key,value in SCIENCE.items():
        if key not in exempt:require(cfg.get(key)==value,'Fixed normalization analysis differs: '+key)
    if fixture:
        seeds,n,grid=cfg['model_seeds'],cfg['training_trials'],cfg['checkpoints']
        require(type(seeds) is list and 1<=len(seeds)<=2 and len(set(seeds))==len(seeds)
                and all(type(x) is int and 90001<=x<=90100 for x in seeds),'Invented fixture seeds required')
        require(type(n) is int and 4<=n<=16 and n%4==0,'Bounded saved fixture')
        require(type(grid) is list and len(grid)>=2 and grid==sorted(set(grid)) and grid[0]==0 and grid[-1]==n
                and all(type(x) is int and 0<=x<=n for x in grid),'Fixture checkpoint grid')


def stats(values):
    values=list(values);valid=[float(v) for v in values if v is not None]
    require(all(np.isfinite(valid)),'Finite valid statistics required')
    return {'mean':float(np.mean(valid)) if valid else None,
        'sd':float(np.std(valid,ddof=1)) if len(valid)>1 else (0. if valid else None),
        'valid_count':len(valid),'total_count':len(values)}


def category_masks(sources,labels,gram_rows,known_after,tol):
    sources=np.asarray(sources);labels=np.asarray(labels);gram_rows=np.asarray(gram_rows)
    known_after=np.asarray(known_after)
    require(sources.ndim==1 and sources.dtype.kind in 'iu' and np.isin(sources,range(4)).all(),'Four valid source indices')
    n=len(sources)
    require(labels.shape==(4,) and np.isin(labels,[-1,1]).all(),'Four evaluation labels')
    require(gram_rows.shape==known_after.shape==(n,4) and np.isfinite(gram_rows).all()
            and np.isin(known_after,[-1,0,1]).all() and np.isfinite(tol) and tol>0,'Finite masks and positive tolerance')
    own=np.arange(4)[None,:]==sources[:,None];cross=~own
    aligned=labels[sources,None]*labels[None,:]*gram_rows
    same=labels[sources,None]==labels[None,:]
    return {'self':own,'cross_helpful':cross&(aligned>tol),'cross_harmful':cross&(aligned < -tol),
        'cross_neutral':cross&(np.abs(aligned)<=tol),'cross_unknown':cross&(known_after==0),
        'cross_same':cross&same,'cross_opposite':cross&~same,'cross_all':cross,'all':np.ones((n,4),bool)}


def row_metrics(B,P,A,mask,phase,scales,raw_norms,nonzero):
    B,P,A=(np.asarray(x,dtype=float) for x in (B,P,A))
    mask,phase,nonzero=(np.asarray(x) for x in (mask,phase,nonzero))
    scales=np.asarray(scales,dtype=float);raw_norms=np.asarray(raw_norms,dtype=float)
    require(B.ndim==2 and B.shape[1]==4 and P.shape==A.shape==mask.shape==B.shape,'Score/mask shape Nx4 required')
    n=B.shape[0]
    require(mask.dtype==phase.dtype==nonzero.dtype==bool and phase.shape==nonzero.shape==scales.shape==raw_norms.shape==(n,),
            'Boolean masks and scalar vectors required')
    require(all(np.isfinite(x).all() for x in (B,P,A,scales,raw_norms))
            and np.all(scales>0) and np.all(raw_norms>0),'Finite values and positive normalization lengths')
    selection=mask&phase[:,None];updated=phase&nonzero
    row={'trial_count':int(phase.sum()),'updated_trials':int(updated.sum()),'cells':int(selection.sum()),
        'updated_cells':int(np.count_nonzero(selection&nonzero[:,None])),
        'scale_mean_all':float(scales[phase].mean()) if phase.any() else None,
        'scale_mean_updated':float(scales[updated].mean()) if updated.any() else None,
        'raw_norm_mean':float(raw_norms[phase].mean()) if phase.any() else None,
        'scale_min':float(scales[phase].min()) if phase.any() else None,
        'scale_max':float(scales[phase].max()) if phase.any() else None}
    for key,stage in [('baseline',B),('desired',P),('actual',A)]:
        values=stage[selection]
        row[key+'_helpful']=float(np.maximum(values,0).sum())
        row[key+'_harmful']=float(np.maximum(-values,0).sum())
        row[key+'_net']=float(values.sum())
    row['suppression_net']=float((B-P)[selection].sum())
    row['normalization_net']=float((P-A)[selection].sum())
    row['helpful_loss']=row['desired_helpful']-row['actual_helpful']
    row['harmful_reduction']=row['desired_harmful']-row['actual_harmful']
    for kind in ['helpful','harmful']:
        denominator=row['desired_'+kind]
        row[kind+'_retention']=row['actual_'+kind]/denominator if denominator>0 else None
    return row


def analyze_saved(cfg,parent,parent_rows):
    validate_config(cfg);seeds=cfg['model_seeds'];n=cfg['training_trials'];labels=np.asarray(cfg['labels'])
    tol=cfg['numeric_tolerance'];shape=(3,2,4,len(seeds),n)
    for key,expected in [('model_seeds',seeds),('checkpoints',cfg['checkpoints']),('targets',(labels>0).astype(int))]:
        require(np.array_equal(parent[key],expected),'Parent axis differs: '+key)
    require(parent['schedules'].shape==(len(seeds),n),'Saved source schedule shape')
    for order in parent['schedules']:
        require(np.isin(order,range(4)).all() and np.array_equal(np.bincount(order,minlength=4),np.full(4,n//4)),
                'Balanced four-input schedule required')
    for key in ['internal_steps','value_steps','internal_update_calls','value_update_calls']:
        require(parent['trial_'+key].shape==shape and not np.any(parent['trial_'+key]),'Parent fixed internal/value state')
    require(len(parent_rows)==int(np.prod(shape)),'Complete parent CSV rows required')
    seen=set()
    for row in parent_rows:
        ti=TOPOLOGIES.index(row['topology']);ri=REPRESENTATIONS.index(row['representation'])
        mi=METHODS.index(row['method']);si=seeds.index(int(row['seed']));t=int(row['trial'])-1
        at=(ti,ri,mi,si,t);require(0<=t<n and at not in seen,'Unique parent CSV trial');seen.add(at)
        require(int(row['example_index'])==parent['schedules'][si,t],'CSV source mismatch')
        for field,key in [('probability_go','probabilities'),('action','actions'),('reward','rewards'),('update_factor','update_factors')]:
            require(abs(float(row[field])-float(parent['trial_'+key][at]))<=tol,'CSV/NPZ mismatch: '+field)
    arrays={key:np.empty(shape+(4,)) for key in ['baseline_margin_delta','desired_margin_delta',
        'actual_margin_delta','suppression_margin_delta','normalization_margin_delta']}
    for key in ['scales','raw_norms']:arrays[key]=np.empty(shape)
    for key in ['full_known_after','nonzero_update']:arrays[key]=np.empty(shape,dtype=bool)
    arrays['cross_class']=np.empty(shape+(4,),dtype=np.int8)
    arrays['reconstruction_residuals']=np.empty(shape[:-1]+(12,));rows=[]
    for ti,top in enumerate(TOPOLOGIES):
        for ri,rep in enumerate(REPRESENTATIONS):
            for mi,method in enumerate(METHODS):
                for si,seed in enumerate(seeds):
                    at=(ti,ri,mi,si);source=parent['schedules'][si].astype(int)
                    d=parent['g_'+top+'_'+rep+'_unit_rows'][si];gram=d@d.T;base=gram[source]
                    g=parent['trial_update_factors'][at];directions=parent['trial_directions'][at]
                    actual=parent['trial_actual_influence'][at];desired=parent['trial_desired_influence'][at]
                    scales=parent['trial_normalization_scales'][at];norms=parent['trial_raw_direction_norms'][at]
                    known=parent['trial_known_after'][at];routing=parent['trial_routing_signs'][at]
                    action=parent['trial_actions'][at];reward=parent['trial_rewards'][at];p=parent['trial_probabilities'][at]
                    require(g.shape==scales.shape==norms.shape==(n,) and actual.shape==desired.shape==known.shape==(n,4),
                            'Complete finite scalar and influence shapes')
                    require(np.isin(action,[0,1]).all() and np.array_equal(reward,action*labels[source]),'Actual obtained rewards only')
                    require(np.all(g*labels[source]>=-tol),'Source-aligned obtained scalar')
                    observed=np.zeros(4,dtype=np.int8)
                    for t,i in enumerate(source):
                        require(np.array_equal(parent['trial_known_before'][at+(t,)],observed),'Observed-before memory mismatch')
                        if action[t]:observed[i]=reward[t]
                        require(np.array_equal(known[t],observed),'Go-only observed-after memory mismatch')
                    route=np.broadcast_to(labels,(n,4)) if method=='oracle_selective' else known
                    if method in METHODS[:2]:route=np.zeros((n,4),dtype=np.int8)
                    require(np.array_equal(route,routing),'Routing information mismatch')
                    harm=(route[np.arange(n),source,None]*route*base < -tol)&(route!=0)&(route[np.arange(n),source,None]!=0)
                    harm[np.arange(n),source]=False
                    if method in METHODS[:2]:harm[:]=False
                    require(np.array_equal(harm,parent['trial_harm_mask'][at]),'Identified harm mask mismatch')
                    expected=base.copy()
                    if method=='isolated':expected=np.eye(4)[source]
                    elif method in METHODS[2:]:expected[harm]=0.
                    B=labels*g[:,None]*base;P=labels*g[:,None]*desired;A=labels*g[:,None]*actual
                    for key,value in [('baseline_margin_delta',B),('desired_margin_delta',P),('actual_margin_delta',A),
                        ('suppression_margin_delta',B-P),('normalization_margin_delta',P-A),('scales',scales),('raw_norms',norms),
                        ('full_known_after',np.all(known!=0,axis=1)),('nonzero_update',g!=0)]:arrays[key][at]=value
                    masks=category_masks(source,labels,base,known,cfg['neutral_tolerance'])
                    classes=np.zeros((n,4),dtype=np.int8);classes[masks['self']]=2
                    classes[masks['cross_helpful']]=1;classes[masks['cross_harmful']]=-1;arrays['cross_class'][at]=classes
                    deltas=g[:,None]*actual;own=deltas[np.arange(n),source]
                    others=deltas.copy();others[np.arange(n),source]=0
                    initial=parent['checkpoint_logits'][at+(0,)]
                    path=np.concatenate([initial[None],initial[None]+np.cumsum(deltas,axis=0)])
                    residuals=[np.max(np.abs(gram-parent['g_'+top+'_'+rep+'_base_influences'][si])),
                        np.max(np.abs(directions@d.T-actual)),np.max(np.abs(desired-expected)),
                        np.max(np.abs(actual-desired*scales[:,None])),np.max(np.abs(norms*scales-1)),
                        np.max(np.abs(own-parent['trial_own_score_steps'][at])),
                        np.max(np.abs(np.linalg.norm(others,axis=1)-parent['trial_other_score_steps'][at])),
                        np.max(np.abs(np.abs(g)-parent['trial_head_steps'][at])),
                        np.max(np.abs(g-cfg['head_lr']*(action-p)*reward)),
                        np.max(np.abs(np.linalg.norm(directions,axis=1)-1)),
                        np.max(np.abs(path[cfg['checkpoints']]-parent['checkpoint_logits'][at])),
                        np.max(np.abs((B-A)-((B-P)+(P-A))))]
                    require(np.isfinite(residuals).all() and max(residuals)<=tol,'Frozen trajectory arithmetic mismatch')
                    require(np.array_equal(g!=0,action==1),'Nonzero updates distinct from NoGo direction-only records')
                    arrays['reconstruction_residuals'][at]=residuals
                    phases={'all':np.ones(n,bool),'partial':~arrays['full_known_after'][at],'full':arrays['full_known_after'][at]}
                    for phase in PHASES:
                        for category in CATEGORIES:
                            row={'topology':top,'representation':rep,'method':method,'seed':seed,'phase':phase,'category':category}
                            row.update(row_metrics(B,P,A,masks[category],phases[phase],scales,norms,g!=0));rows.append(row)
    require(all(np.isfinite(value).all() for value in arrays.values()),'All derived arrays finite')
    return arrays,rows


def produce(cfg,root):
    validate_config(cfg);folder=Path(root)/cfg['source_result']
    with np.load(folder/'arrays.npz',allow_pickle=False) as saved:
        parent={key:saved[key] for key in saved.files if key.startswith(('trial_','checkpoint_','g_'))
                or key in ['model_seeds','checkpoints','targets','schedules']}
    with (folder/'measurements.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    return analyze_saved(cfg,parent,rows)


def summarize(cfg,arrays,rows):
    validate_config(cfg);seeds=cfg['model_seeds'];tol=cfg['numeric_tolerance']
    lookup={(x['topology'],x['representation'],x['method'],x['seed'],x['phase'],x['category']):x for x in rows}
    require(len(rows)==len(lookup)==24*len(seeds)*3*9,'Complete unique derived records')
    valid=True;families=[]
    for ti,top in enumerate(TOPOLOGIES):
        for ri,rep in enumerate(REPRESENTATIONS):
            for mi,method in enumerate(METHODS):
                phases={}
                for phase in PHASES:
                    categories={};pooled={}
                    for cat in CATEGORIES:
                        selected=[lookup[top,rep,method,seed,phase,cat] for seed in seeds]
                        categories[cat]={key:stats([x[key] for x in selected]) for key in METRICS}
                        pooled[cat]={}
                        for kind in ['helpful','harmful']:
                            denominator=sum(x['desired_'+kind] for x in selected)
                            pooled[cat][kind+'_retention']=sum(x['actual_'+kind] for x in selected)/denominator if denominator>0 else None
                        for row in selected:
                            for name in ['baseline','desired','actual']:
                                valid&=abs(row[name+'_helpful']-row[name+'_harmful']-row[name+'_net'])<=tol
                            valid&=abs(row['baseline_net']-row['actual_net']-row['suppression_net']-row['normalization_net'])<=tol
                    for seed in seeds:
                        for left,right in [('all',['self','cross_all']),('cross_all',['cross_helpful','cross_harmful','cross_neutral']),
                                           ('cross_all',['cross_same','cross_opposite'])]:
                            for metric in ADDITIVE:
                                valid&=abs(lookup[top,rep,method,seed,phase,left][metric]-sum(lookup[top,rep,method,seed,phase,k][metric] for k in right))<=tol
                    phases[phase]={'categories':categories,'pooled_categories':pooled}
                for seed in seeds:
                    for cat in CATEGORIES:
                        for metric in ADDITIVE+['trial_count','updated_trials']:
                            values=[lookup[top,rep,method,seed,p,cat][metric] for p in PHASES]
                            valid&=abs(values[0]-values[1]-values[2])<=tol
                families.append({'topology':top,'representation':rep,'method':method,
                    'role':'primary_descriptive' if (ti,ri,mi)==(0,0,3) else ('privileged_diagnostic' if mi==2 else 'descriptive_control'),
                    'phases':phases})
    residual=np.max(arrays['reconstruction_residuals'].reshape(-1,12),axis=0)
    require(valid and np.all(residual<=tol),'Stage/partition balance or reconstruction failed')
    count=24*len(seeds)*cfg['training_trials']
    return {'study_id':cfg['study_id'],'status':'completed','scope':cfg['scope'],'categories':CATEGORIES,'phases':PHASES,
        'families':families,'validation':{'residual_axis':RESIDUALS,'max_residual_by_axis':residual.tolist(),
            'all_reconstruction_checks_passed':True,'all_stage_and_partition_balances_passed':True},
        'counts':{'analyzed_parent_trial_records':count,'trial_destination_cells':count*4,'cross_destination_cells':count*3,
            'conditions':24,'model_seeds':len(seeds),'measurement_records':len(rows),'new_training_trials':0,
            'new_head_update_calls':0,'new_internal_update_calls':0,'new_value_update_calls':0,'new_random_draws':0,
            'fresh_model_constructions':0},'interpretation':dict(INTERPRETATION)}


def render(summary,arrays,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(exist_ok=False);plotted={'retention':[],'primary_phases':[]}
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    fig.subplots_adjust(left=.065,right=.98,bottom=.10,top=.82,wspace=.27,hspace=.48)
    for ti,top in enumerate(TOPOLOGIES):
        for ri,rep in enumerate(REPRESENTATIONS):
            ax=axes[ri,ti];x=np.arange(4)
            selected=[next(f for f in summary['families'] if (f['topology'],f['representation'],f['method'])==(top,rep,m)) for m in METHODS]
            for cat,offset,color,label in [('self',-.16,'#287a9f','Own correction'),('cross_helpful',.16,'#179a83','Retained helpful cross')]:
                values=[f['phases']['all']['pooled_categories'][cat]['helpful_retention'] for f in selected]
                plotted['retention'].append({'topology':top,'representation':rep,'category':cat,'values':values})
                ax.bar(x+offset,[v*100 if v is not None else 0 for v in values],.30,color=color,label=label)
                for j,v in enumerate(values):
                    if v is None:ax.text(j+offset,2,'N/A',ha='center',va='bottom',fontsize=8,rotation=90)
            ax.set_xticks(x,['shared','isolated','oracle','feedback'],rotation=20);ax.set_ylim(0,110)
            ax.set_title(top+' / '+rep);ax.axhline(100,color='#888888',ls=':',lw=.8)
            if ti==0:ax.set_ylabel('Remaining score mass / raw desired mass (%)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False)
    fig.suptitle('Normalization retention on saved paths; pooled ratios, not accuracy',fontsize=15,y=.98)
    fig.savefig(out/'normalization-retention.png',dpi=150);plt.close(fig)
    primary=next(f for f in summary['families'] if f['role']=='primary_descriptive')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for cat,color,label in [('self','#287a9f','Own correction'),('cross_helpful','#179a83','Helpful cross')]:
        values=[primary['phases'][p]['pooled_categories'][cat]['helpful_retention'] for p in PHASES]
        axes[0].plot(range(3),[100*v if v is not None else np.nan for v in values],'o-',label=label,color=color)
        plotted['primary_phases'].append({'category':cat,'phases':PHASES,'helpful_retention':values})
    axes[0].set_xticks(range(3),['All trials','Some signs unknown','All signs observed'])
    axes[0].set_ylim(0,105);axes[0].set_ylabel('Remaining desired helpful score mass (%)');axes[0].legend(frameon=False)
    rows=primary['phases']['all']['categories'];stages=['baseline','desired','actual']
    for cat,color,label in [('self','#287a9f','Own helpful'),('cross_helpful','#179a83','Cross helpful'),('cross_harmful','#d1495b','Cross harmful magnitude')]:
        kind='harmful' if cat=='cross_harmful' else 'helpful'
        values=[rows[cat][p+'_'+kind]['mean'] for p in stages]
        axes[1].plot(range(3),values,'o-',color=color,label=label)
    axes[1].set_xticks(range(3),['Shared reference','Harm removed (raw)','Unit movement'])
    axes[1].set_ylabel('Mean cumulative score mass per seed');axes[1].legend(frameon=False)
    fig.suptitle('real / raw / observed-reward selective: arithmetic stages, not alternative policies')
    fig.savefig(out/'primary-normalization-stages.png',dpi=150);plt.close(fig)
    return plotted
