"""Exploratory saved-trace descriptions and a planned metric counterexample."""
from pathlib import Path
import numpy as np
from .human_shell import behavior_metrics
from .human_shell_source import load_cohort
from .prospective import read

EVENTS = ('nogo_hidden', 'go_empty', 'go_gain', 'go_loss')
POLICIES = ('uniform', 'persistence', 'q')
METRICS = ('acquisition_gain', 'reversal_early', 'reversal_recovery')
CONDITIONS = ('fixed','plastic','rewired_fixed','rewired_plastic','ann_fixed','ann_plastic','linear','q','uniform','bias','persistence')


def validate_config(cfg):
    fixed = {'study_id':'shell_behavior_v1',
             'scope':'exploratory_saved_trace_description_and_planned_metric_counterexample',
             'synthetic_seeds':list(range(16001,16009)),
             'synthetic_policies':list(POLICIES), 'trials_per_person':216,
             'rng_namespace':2026101902, 'q_alpha':.1, 'q_beta':4.,
             'trajectory_parent':'results/human_shell_rng_v2/trajectories.npz',
             'source_manifest':'data/human/tnu_shell_2024/source-manifest.json',
             'external_manifest':'data/human/tnu_shell_2024/source-manifest.json'}
    if any(cfg.get(k)!=v for k,v in fixed.items()):raise ValueError('Fixed descriptive and calibration scope required')
    original=read(Path(__file__).resolve().parents[2]/'configs/human_shell_rng_v2.json')
    if cfg['test_seeds']!=original['test_seeds'] or cfg['test_ids']!=original['test_ids']:
        raise ValueError('Original fixed held-out schedules required')
    if cfg['conditions']!=list(CONDITIONS):raise ValueError('All original controls required')


def regime(shell, phase):
    return ('RNRN','NRNR','RPRP','PRPR','PNPN','NPNP')[shell-1][phase]


def pairing(rows):
    result=[]
    for shell in range(1,7):
        locations=[i for i,r in enumerate(rows) if r['shell']==shell]
        phase=0
        for left,right in zip(locations[:-1],locations[1:]):
            if rows[left]['state']!=rows[right]['state']:
                phase+=1
            else:
                result.append((left,right,shell-1,phase))
        if phase!=3:raise ValueError('Four original shell phases required')
    return result


def transition_counts(records, ids, actions):
    k,p,n=actions.shape
    counts=np.zeros((k,p,6,4,4,3),dtype=np.int64)
    for pi,pid in enumerate(ids):
        rows=records[pid]
        if len(rows)!=n:raise ValueError('Original action alignment required')
        for left,right,shell,phase in pairing(rows):
            for stream in range(k):
                action=int(actions[stream,pi,left]);following=int(actions[stream,pi,right])
                event=0 if action==0 else (1,2,3)[rows[left]['content']]
                counts[stream,pi,shell,phase,event]+=np.array([1,following,int(action==following)])
    return counts


def simulate(rows, seed, pid, policy, flip=False, alpha=.1, beta=4.):
    if policy not in POLICIES:raise ValueError('Known calibration policy required')
    n=len(rows);rng=np.random.default_rng(np.random.SeedSequence([2026101902,seed,int(pid[1:])]))
    u=rng.random(n);p=np.empty(n);a=np.empty(n,dtype=np.int8);r=np.empty(n,dtype=np.int8)
    steps=np.zeros(n);values=np.zeros((6,2));last=np.full(6,-1,dtype=int)
    for i,row in enumerate(rows):
        s=row['shell']-1
        probability=.5 if policy=='uniform' else (.5 if last[s]<0 else (.9 if last[s] else .1))
        if policy=='q':probability=float(1/(1+np.exp(-np.clip(beta*(values[s,1]-values[s,0]),-40,40))))
        action=int(u[i]<probability)
        reward=(0,1,-1)[row['content']] if action else 0
        if flip:reward=-reward
        p[i],a[i],r[i]=probability,action,reward
        if policy=='q':
            old=values[s,action];values[s,action]+=alpha*(reward-values[s,action]);steps[i]=abs(values[s,action]-old)
        last[s]=action
    return p,a,r,steps,values


def metric_vector(rows, actions):
    return np.array(list(behavior_metrics(rows,actions).values()))


def table_row(**kwargs):
    fields=('kind','condition','seed','participant_id','shell','phase','regime','event','statistic',
            'numerator','denominator','people_with_denominator','value',*METRICS,'head_updates','internal_updates','value_updates')
    result={key:None for key in fields};result.update(kwargs);return result


def measurement_rows(cfg, arrays):
    records=[];counts=arrays['transition_counts']
    streams=[('human',0)]+[(c,s) for c in cfg['conditions'] for s in cfg['test_seeds']]
    for ki,(condition,seed) in enumerate(streams):
        for sh in range(6):
            for phase in range(4):
                for ei,event in enumerate(EVENTS):
                    cell=counts[ki,:,sh,phase,ei];n=int(cell[:,0].sum());people=int(np.count_nonzero(cell[:,0]))
                    for index,stat in ((1,'next_go'),(2,'stay')):
                        numerator=int(cell[:,index].sum())
                        records.append(table_row(kind='saved_transition',condition=condition,seed=seed,shell=sh+1,phase=phase+1,
                            regime=regime(sh+1,phase),event=event,statistic=stat,numerator=numerator,denominator=n,
                            people_with_denominator=people,value=numerator/n if n else None))
    for fi,condition in enumerate((*POLICIES,'scripted_q_tape')):
        values=arrays['synthetic_metrics'][fi] if fi<3 else arrays['scripted_metrics']
        for si,seed in enumerate(cfg['synthetic_seeds']):
            for pi,pid in enumerate(cfg['test_ids']):
                records.append(table_row(kind='synthetic_metric',condition=condition,seed=seed,participant_id=pid,
                    **dict(zip(METRICS,map(float,values[si,pi]))),head_updates=0,internal_updates=0,
                    value_updates=cfg['trials_per_person'] if condition=='q' else 0))
    for si,seed in enumerate(cfg['synthetic_seeds']):
        for pi,pid in enumerate(cfg['test_ids']):
            unequal=int(np.count_nonzero(arrays['synthetic_actions'][2,si,pi]!=arrays['counterfactual_actions'][si,pi]))
            records.append(table_row(kind='reward_intervention',condition='q',seed=seed,participant_id=pid,
                statistic='choice_disagreement',numerator=unequal,denominator=cfg['trials_per_person'],value=unequal/cfg['trials_per_person'],
                head_updates=0,internal_updates=0,value_updates=cfg['trials_per_person']))
    return records


def produce(cfg, root, cache=None):
    validate_config(cfg)
    records,_=load_cohort(cache,read(Path(root)/cfg['source_manifest']))
    ids=cfg['test_ids'];records={pid:records[pid] for pid in ids}
    human=np.array([[r['action'] for r in records[pid]] for pid in ids],dtype=np.int8)
    with np.load(Path(root)/cfg['trajectory_parent'],allow_pickle=False) as z:
        old=z['actions'];expected=(2,11,5,29,216)
        if old.shape!=expected:raise ValueError('Frozen original trajectory axes required')
        actions=np.concatenate((human[None],old[1].reshape(55,29,216)),axis=0)
    arrays={'transition_counts':transition_counts(records,ids,actions)}
    shape=(3,len(cfg['synthetic_seeds']),len(ids),216)
    for key,dtype in [('synthetic_probabilities',float),('synthetic_actions',np.int8),('synthetic_rewards',np.int8),('synthetic_value_steps',float)]:
        arrays[key]=np.empty(shape,dtype=dtype)
    for key,dtype in [('counterfactual_probabilities',float),('counterfactual_actions',np.int8),('counterfactual_rewards',np.int8),('counterfactual_value_steps',float)]:
        arrays[key]=np.empty(shape[1:],dtype=dtype)
    arrays['final_value_tables']=np.zeros((2,shape[1],shape[2],6,2))
    arrays['synthetic_metrics']=np.empty((3,shape[1],shape[2],3))
    arrays['scripted_metrics']=np.empty((shape[1],shape[2],3))
    for si,seed in enumerate(cfg['synthetic_seeds']):
        for pi,pid in enumerate(ids):
            for fi,policy in enumerate(POLICIES):
                p,a,r,steps,q=simulate(records[pid],seed,pid,policy)
                for name,value in zip(('synthetic_probabilities','synthetic_actions','synthetic_rewards','synthetic_value_steps'),(p,a,r,steps)):
                    arrays[name][fi,si,pi]=value
                arrays['synthetic_metrics'][fi,si,pi]=metric_vector(records[pid],a)
                if policy=='q':arrays['final_value_tables'][0,si,pi]=q
            p,a,r,steps,q=simulate(records[pid],seed,pid,'q',flip=True)
            for name,value in zip(('counterfactual_probabilities','counterfactual_actions','counterfactual_rewards','counterfactual_value_steps'),(p,a,r,steps)):
                arrays[name][si,pi]=value
            arrays['final_value_tables'][1,si,pi]=q
    # An explicitly constructed open-loop tape is a counterexample, not an eligible
    # fair learner or a claim about the true causes of human behavior.
    arrays['scripted_actions']=arrays['synthetic_actions'][2].copy()
    for si in range(shape[1]):
        for pi,pid in enumerate(ids):
            arrays['scripted_metrics'][si,pi]=metric_vector(records[pid],arrays['scripted_actions'][si,pi])
    return arrays,measurement_rows(cfg,arrays)


def summarize(cfg, arrays, rows):
    totals=arrays['transition_counts'][...,0].sum(axis=(1,2,3,4))
    streams,people=arrays['transition_counts'].shape[:2]
    n=arrays['scripted_actions'].size
    trial_count=arrays['synthetic_actions'].shape[-1]
    candidates=streams*people*(trial_count-6)
    exact=np.array_equal(arrays['synthetic_metrics'][2],arrays['scripted_metrics'])
    groups=[]
    for fi,policy in enumerate(POLICIES):
        groups.append({'policy':policy,'metric_means':dict(zip(METRICS,map(float,arrays['synthetic_metrics'][fi].mean(axis=(0,1))))),
                       'online_value_update_calls':n if policy=='q' else 0})
    different=int(np.count_nonzero(arrays['counterfactual_actions']!=arrays['scripted_actions']))
    return {'status':'completed','scope':'exploratory_saved_trace_description_and_planned_metric_counterexample',
        'counts':{'human_records_reanalyzed':people,'old_model_streams':(streams-1)*people,'old_total_streams':streams*people,
                  'old_recorded_trials_reduced':streams*people*trial_count,'next_same_shell_candidates':candidates,
                  'excluded_phase_boundary_pairs':candidates-int(totals.sum()),'eligible_within_phase_pairs':int(totals.sum()),
                  'new_synthetic_policy_choices':4*n,'scripted_tape_reads':n,'new_human_experiments':0,
                  'new_neural_parameter_updates':0,'new_value_update_calls':2*n,
                  'nonzero_value_update_steps':int(np.count_nonzero(arrays['synthetic_value_steps'])+np.count_nonzero(arrays['counterfactual_value_steps']))},
        'event_order':list(EVENTS),'synthetic_policy_results':groups,
        'saved_transition_summary':{'stream_order':['human']+list(CONDITIONS),'per_stream_eligible_pairs':list(map(int,totals)),
            'conditional_rates_are_descriptive_not_causal_reward_effects':True,'missing_denominators_are_null':True},
        'metric_identification':{'planned_counterexample_verified':bool(exact),
            'three_summaries_uniquely_identify_reward_learning':False if exact else None,
            'scripted_policy_online_updates':0,'scripted_policy_uses_precomputed_action_tape':True,
            'scripted_policy_is_eligible_fair_learning_control':False,
            'notice':'Exact equality was planned by construction; it limits mechanism identification and is not a new human learning effect.'},
        'reward_intervention':{'q_vs_fixed_tape_disagreeing_choices':different,'evaluated_choices':n,
            'disagreement_fraction':different/n,'fixed_tape_choice_change':0,
            'notice':'Reward signs reversed in a synthetic intervention; original human rewards and original comparison are unchanged.'},
        'original_similarity_verdict_changed':False,'new_human_similarity_confirmation':False,
        'shared_neural_mechanism_claim':False,'education_effect_claim':False,
        'notice':'Known old outcomes informed this separate exploratory plan. Do not interpret pooled transitions, shared schedules, computational seeds or tape copies as independent people.'}
