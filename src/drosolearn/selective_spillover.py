"""Prospective output learning with an explicit observed-reward routing memory."""
from __future__ import annotations
from itertools import product
import numpy as np
from .graph import load_projection
from .human_shell_models import build_graphs
from .pattern_rule import VectorActor
from .update_budget import load_catalog,require
from .selective_geometry import selective_geometry,selective_direction

TOPOLOGIES=('real','rewired','ann')
REPRESENTATIONS=('raw','identity')
METHODS=('shared','isolated','oracle_selective','feedback_selective')
RESIDUALS=('raw_target_residual','normalization_residual','unit_norm_residual','scaled_target_residual',
    'own_raw_preservation_residual','raw_norm_lower_bound_residual','harm_zero_residual')
SCIENCE={'study_id':'selective_spillover_v1','scope':'prospective_feedback_selective_spillover',
    'module':'drosolearn.selective_spillover','protocol':'docs/selective-spillover-protocol.md',
    'topologies':list(TOPOLOGIES),'representations':list(REPRESENTATIONS),'methods':list(METHODS),
    'model_seeds':list(range(17401,17411)),'training_trials':400,'checkpoints':[0,50,100,200,400],
    'head_lr':.1,'rng_namespace':202610196,'svd_rcond':1e-12,'numeric_tolerance':1e-8,'exact_condition_max':1e6,
    'acquisition_threshold':.8,'acquisition_gain':.25,'class_threshold':.7,'population_pass_count':8,
    'effect_threshold':.05,'positive_seed_count':8,'normalization':'per_input_initial_fixed_MBON_gap_unit_L2',
    'budget_convention':'unit_directions_common_eta_ceiling_with_observed_reward_memory',
    'input_catalog':'data/synthetic/representation_update_v1/input_catalog.json',
    'graph':'data/flywire_783_mb_left_kc512_s701','external_manifest':None}


def validate_config(cfg):
    fixture=cfg.get('scope')=='integration_test'
    exempt={'scope','model_seeds','training_trials','checkpoints','population_pass_count','positive_seed_count'} if fixture else set()
    for key,value in SCIENCE.items():
        if key not in exempt:require(cfg.get(key)==value,'Fixed selective design differs: '+key)
    if fixture:
        seeds,n,steps=cfg['model_seeds'],cfg['training_trials'],cfg['checkpoints']
        require(type(seeds) is list and 1<=len(seeds)<=2 and len(set(seeds))==len(seeds)
            and all(type(s) is int and 90001<=s<=90100 for s in seeds),'Invented fixture seeds only')
        require(type(n) is int and 4<=n<=16 and n%4==0,'Bounded fixture budget')
        require(steps==sorted(set(steps)) and steps[0]==0 and steps[-1]==n
            and all(type(t) is int and 0<=t<=n and t%4==0 for t in steps),'Fixture checkpoint grid')
        require(cfg['population_pass_count']==cfg['positive_seed_count']==len(seeds),'Fixture pass counts')


def sigmoid(z):return 1./(1.+np.exp(-np.clip(z,-40.,40.)))


class SelectiveHead:
    """Environment labels can enter only the explicitly privileged oracle arm.

    The feedback arm receives index, actual action and obtained reward only.
    Its four-sign auxiliary memory is not an internal KC/MBON synaptic weight.
    """
    def __init__(self,geometry,method,rate,oracle_signs=None):
        require(method in METHODS and np.isfinite(rate) and rate>0,'Fixed method and positive rate')
        require(geometry['numeric_selective_qualified'],'Qualified fixed geometry')
        self.geometry=geometry;self.features=geometry['unit_rows'];self.method=method;self.rate=rate
        self.head=np.zeros(self.features.shape[1]);self.known=np.zeros(4,dtype=np.int8)
        self.calls=0;self.memory_observation_calls=0;self.discoveries=0
        if method=='oracle_selective':
            values=np.asarray(oracle_signs)
            require(values.shape==(4,) and np.isin(values,[-1,1]).all(),'Oracle requires explicit privileged full signs')
            self.oracle_signs=values.astype(np.int8).copy()
        else:
            require(oracle_signs is None,'Reward-only/control learner must not receive hidden target table')
            self.oracle_signs=None

    def logits(self):return self.features@self.head
    def probabilities(self):return sigmoid(self.logits())
    def lookup_probabilities(self):return (self.known.astype(float)+1.)/2.

    def observe(self,index,action,reward):
        require(type(index) is int and 0<=index<4,'Observed source index')
        require(type(action) is int and action in (0,1)
            and type(reward) is int and ((action==0 and reward==0) or (action==1 and reward in (-1,1))),
            'Only obtained Go +/-1 or NoGo0 is allowed')
        p=float(self.probabilities()[index]);before=self.known.copy();changed=0
        if action:
            require(self.known[index] in (0,reward),'Contradictory deterministic reward observation')
            changed=int(self.known[index]==0);self.known[index]=reward
            self.memory_observation_calls+=1;self.discoveries+=changed
        routing=np.zeros(4,dtype=np.int8)
        if self.method in ('oracle_selective','feedback_selective'):
            routing=self.oracle_signs.copy() if self.method=='oracle_selective' else self.known.copy()
            evidence=selective_direction(self.geometry,index,routing)
        else:
            d=self.features;gram=self.geometry['base_influences']
            if self.method=='shared':
                raw=d[index].copy();direction=raw.copy();desired=gram[index].copy();scale=1.
            else:
                raw=self.geometry['base_budget']['base_geometry']['directions'][index].copy()
                direction=self.geometry['isolated_directions'][index].copy();desired=np.eye(4)[index]
                scale=1./np.linalg.norm(raw)
            raw_norm=float(np.linalg.norm(raw));influence=direction@d.T;raw_influence=raw@d.T
            evidence={'direction':direction,'desired_influence':desired,'influence':influence,
                'harm_mask':np.zeros(4,dtype=bool),'raw_norm':raw_norm,'normalization_scale':float(scale),
                'raw_target_residual':float(np.max(np.abs(raw_influence-desired))),
                'normalization_residual':float(np.max(np.abs(direction-raw*scale))),
                'unit_norm_residual':float(abs(np.linalg.norm(direction)-1.)),
                'scaled_target_residual':float(np.max(np.abs(influence-desired*scale))),
                'own_raw_preservation_residual':float(abs(raw_influence[index]-gram[index,index])),
                'raw_norm_lower_bound_residual':float(max(1.-raw_norm,0.)),'harm_zero_residual':0.}
        residuals=np.array([evidence[name] for name in RESIDUALS])
        require(np.isfinite(residuals).all() and residuals.max()<=self.geometry['tolerance'],'Dynamic direction numeric gate')
        factor=self.rate*(action-p)*reward;old=self.head.copy();self.head+=factor*evidence['direction'];self.calls+=1
        step=self.head-old;scores=self.features@step
        return {**evidence,'known_before':before,'known_after':self.known.copy(),'routing_signs':routing,
            'memory_changed':changed,'memory_observation_calls':self.memory_observation_calls,'memory_discoveries':self.discoveries,
            'direction_residuals':residuals,'update_factor':factor,'head_step':float(np.linalg.norm(step)),
            'own_score_step':float(scores[index]),'other_score_step':float(np.linalg.norm(np.delete(scores,index))),
            'expected_head_step':float(abs(factor)*np.linalg.norm(evidence['direction']))}


def produce(cfg,root):
    validate_config(cfg);data=load_catalog(cfg,root);graph=load_projection(root/cfg['graph'])
    seeds,n,grid=cfg['model_seeds'],cfg['training_trials'],cfg['checkpoints'];c=graph.n_mbon
    shape=(3,2,4,len(seeds),n);cshape=shape[:-1]+(len(grid),);arrays=dict(data)
    integer={'actions','rewards','head_update_calls','internal_update_calls','value_update_calls',
        'memory_changed','memory_observation_calls','memory_discoveries'}
    scalar=('probabilities','actions','rewards','head_steps','own_score_steps','other_score_steps','internal_steps',
        'value_steps','head_update_calls','internal_update_calls','value_update_calls','update_factors','expected_head_steps',
        'head_step_squared_l2','cumulative_head_path_l2','cumulative_head_squared_l2','head_norms','raw_direction_norms',
        'normalization_scales','memory_changed','memory_observation_calls','memory_discoveries')
    for field in scalar:arrays['trial_'+field]=np.zeros(shape,dtype=np.int32 if field in integer else float)
    for field in ('known_before','known_after','routing_signs'):arrays['trial_'+field]=np.empty(shape+(4,),dtype=np.int8)
    arrays['trial_harm_mask']=np.empty(shape+(4,),dtype=bool)
    for field in ('desired_influence','actual_influence'):arrays['trial_'+field]=np.empty(shape+(4,))
    arrays['trial_directions']=np.empty(shape+(c,));arrays['trial_direction_residuals']=np.empty(shape+(7,))
    for field in ('logits','probabilities','lookup_probabilities'):arrays['checkpoint_'+field]=np.empty(cshape+(4,))
    arrays['checkpoint_heads']=np.empty(cshape+(c,));arrays['checkpoint_known']=np.empty(cshape+(4,),dtype=np.int8)
    for field in ('head_calls','internal_calls','value_calls','memory_observation_calls','memory_discoveries'):
        arrays['checkpoint_'+field]=np.zeros(cshape,dtype=np.int32)
    arrays['first_discovery_trials']=np.full(shape[:-1]+(4,),n+1,dtype=np.int32)
    arrays['schedules']=np.empty((len(seeds),n),dtype=np.int8);arrays['uniforms']=np.empty((len(seeds),n))
    arrays['model_seeds']=np.asarray(seeds,dtype=np.int64);arrays['checkpoints']=np.asarray(grid,dtype=np.int32)
    prepared=[];static={};rows=[];lookup={step:i for i,step in enumerate(grid)}
    for si,seed in enumerate(seeds):
        graphs=build_graphs(graph,seed)
        for ti,topology in enumerate(TOPOLOGIES):
            for ri,representation in enumerate(REPRESENTATIONS):
                inputs=data['inputs'][ri]
                base=VectorActor(graphs[topology],seed,cfg['head_lr'],False,inputs,inputs);witness=base.witness()
                gap=witness['probe_mbon'][:,1]-witness['probe_mbon'][:,0]
                geo=selective_geometry(gap,cfg['svd_rcond'],cfg['numeric_tolerance'],cfg['exact_condition_max'])
                max_residual=np.zeros(7);preflight_count=0
                for signs in product((-1,0,1),repeat=4):
                    for source in range(4):
                        evidence=selective_direction(geo,source,np.asarray(signs,dtype=np.int8))
                        max_residual=np.maximum(max_residual,[evidence[name] for name in RESIDUALS]);preflight_count+=1
                require(preflight_count==324,'Complete label-free sign-state preflight')
                prefix=topology+'_'+representation+'_'
                fields=('unit_rows','pseudoinverse','base_influences','shared_directions','isolated_directions',
                    'retained_rank','full_row_condition','projector_identity_residual','numeric_selective_qualified')
                for field in fields:static.setdefault('g_'+prefix+field,[]).append(np.asarray(geo[field]))
                static.setdefault('g_'+prefix+'preflight_direction_count',[]).append(np.asarray(preflight_count))
                static.setdefault('g_'+prefix+'preflight_max_residuals',[]).append(max_residual)
                for field in ('weights','encoder','bias','probe_kc','probe_mbon'):
                    static.setdefault('w_'+prefix+field,[]).append(witness[field].copy())
                prepared.append((si,ti,ri,seed,topology,representation,base,witness,geo,prefix))
    # Every geometry and all 81 sign states qualify before behavioral RNG.
    for si,seed in enumerate(seeds):
        rng=np.random.default_rng(np.random.SeedSequence([cfg['rng_namespace'],seed,0]))
        arrays['schedules'][si]=np.concatenate([rng.permutation(4) for _ in range(n//4)])
        arrays['uniforms'][si]=np.random.default_rng(np.random.SeedSequence([cfg['rng_namespace'],seed,1])).random(n)
    true_signs=2*data['targets']-1
    for si,ti,ri,seed,topology,representation,base,witness,geo,prefix in prepared:
        for mi,method in enumerate(METHODS):
            actor=SelectiveHead(geo,method,cfg['head_lr'],true_signs if method=='oracle_selective' else None)
            path=energy=0.
            def checkpoint(step):
                at=(ti,ri,mi,si,lookup[step])
                for field,value in [('logits',actor.logits()),('probabilities',actor.probabilities()),
                    ('lookup_probabilities',actor.lookup_probabilities()),('heads',actor.head),('known',actor.known),
                    ('head_calls',actor.calls),('memory_observation_calls',actor.memory_observation_calls),('memory_discoveries',actor.discoveries)]:
                    arrays['checkpoint_'+field][at]=value
            checkpoint(0)
            for t,source in enumerate(arrays['schedules'][si]):
                source=int(source);p=float(actor.probabilities()[source]);a=int(arrays['uniforms'][si,t]<p)
                reward=int(true_signs[source]) if a else 0;e=actor.observe(source,a,reward)
                at=(ti,ri,mi,si,t);path+=e['head_step'];energy+=e['head_step']**2;head_norm=float(np.linalg.norm(actor.head))
                if e['memory_changed']:arrays['first_discovery_trials'][ti,ri,mi,si,source]=t+1
                values={'probabilities':p,'actions':a,'rewards':reward,'head_steps':e['head_step'],
                    'own_score_steps':e['own_score_step'],'other_score_steps':e['other_score_step'],'head_update_calls':actor.calls,
                    'update_factors':e['update_factor'],'expected_head_steps':e['expected_head_step'],'head_step_squared_l2':e['head_step']**2,
                    'cumulative_head_path_l2':path,'cumulative_head_squared_l2':energy,'head_norms':head_norm,
                    'raw_direction_norms':e['raw_norm'],'normalization_scales':e['normalization_scale'],
                    'memory_changed':e['memory_changed'],'memory_observation_calls':actor.memory_observation_calls,'memory_discoveries':actor.discoveries}
                for field,value in values.items():arrays['trial_'+field][at]=value
                for field,key in [('known_before','known_before'),('known_after','known_after'),('routing_signs','routing_signs'),
                    ('harm_mask','harm_mask'),('desired_influence','desired_influence'),('actual_influence','influence'),
                    ('directions','direction'),('direction_residuals','direction_residuals')]:arrays['trial_'+field][at]=e[key]
                require(e['head_step']<=cfg['head_lr']+cfg['numeric_tolerance'] and path<=n*cfg['head_lr']+cfg['numeric_tolerance']
                    and energy<=n*cfg['head_lr']**2+cfg['numeric_tolerance'] and head_norm<=path+cfg['numeric_tolerance'],
                    'Common actual output movement ceiling violated')
                row={'topology':topology,'representation':representation,'method':method,'seed':seed,'trial':t+1,
                    'example_index':source,'target':int(data['targets'][source]),'probability_go':p,'action':a,'reward':reward,
                    'head_update_calls':actor.calls,'internal_update_calls':0,'value_update_calls':0,'head_step_l2':e['head_step'],
                    'own_score_step':e['own_score_step'],'other_score_step_l2':e['other_score_step'],'internal_step_l2':0.,'value_step_l2':0.,
                    'head_nonzero_step':int(e['head_step']>0),'internal_nonzero_step':0,'value_nonzero_step':0,'update_factor':e['update_factor'],
                    'direction_l2':float(np.linalg.norm(e['direction'])),'expected_head_step_l2':e['expected_head_step'],
                    'head_step_squared_l2':e['head_step']**2,'cumulative_head_path_l2':path,'cumulative_head_squared_l2':energy,'head_norm':head_norm,
                    'raw_direction_norm':e['raw_norm'],'normalization_scale':e['normalization_scale'],'memory_changed':e['memory_changed'],
                    'memory_observation_calls':actor.memory_observation_calls,'memory_discoveries':actor.discoveries,
                    'direction_residual_max':float(e['direction_residuals'].max())}
                for field,key in [('known_before','known_before'),('known_after','known_after'),('routing_sign','routing_signs'),('harm','harm_mask'),
                    ('desired_influence','desired_influence'),('actual_influence','influence')]:
                    for j in range(4):row[field+'_'+str(j)]=int(e[key][j]) if field in ('known_before','known_after','routing_sign','harm') else float(e[key][j])
                rows.append(row)
                if t+1 in lookup:checkpoint(t+1)
        final=base.witness()
        for field in ('weights','encoder','bias','probe_kc','probe_mbon'):
            require(np.array_equal(witness[field],final[field]),'Fixed internal weights/encoder/activity changed')
            static.setdefault('finalfixed_'+prefix+field,[]).append(final[field])
    arrays.update({name:np.stack(values) for name,values in static.items()})
    return arrays,rows


def information_checks(cfg,arrays):
    flags={name:True for name in ('observed_memory_from_obtained_Go_rewards_only','NoGo_does_not_reveal_unknown_sign',
        'current_Go_feedback_precedes_direction','controls_ignore_observed_memory_for_direction','oracle_privileged_labels_separate',
        'feedback_routing_equals_observed_after','head_policy_does_not_directly_lookup_memory','static_lookup_not_autonomous_experiment',
        'memory_writes_equal_first_discoveries')}
    y=2*arrays['targets']-1;n=cfg['training_trials'];tol=cfg['numeric_tolerance']
    for ti,top in enumerate(TOPOLOGIES):
        for ri,rep in enumerate(REPRESENTATIONS):
            for mi,method in enumerate(METHODS):
                for si in range(len(cfg['model_seeds'])):
                    at=(ti,ri,mi,si);known=np.zeros(4,dtype=np.int8);calls=discovered=0
                    first=np.full(4,n+1,dtype=int)
                    for t,source in enumerate(arrays['schedules'][si]):
                        source=int(source);a=int(arrays['trial_actions'][at+(t,)]);r=int(arrays['trial_rewards'][at+(t,)])
                        flags['observed_memory_from_obtained_Go_rewards_only'] &= np.array_equal(arrays['trial_known_before'][at+(t,)],known)
                        before=known.copy();changed=int(a==1 and known[source]==0)
                        if a:
                            flags['observed_memory_from_obtained_Go_rewards_only'] &= r in (-1,1) and known[source] in (0,r)
                            known[source]=r;calls+=1;discovered+=changed
                            if changed:first[source]=t+1
                        else:flags['NoGo_does_not_reveal_unknown_sign'] &= r==0 and np.array_equal(before,arrays['trial_known_after'][at+(t,)])
                        flags['observed_memory_from_obtained_Go_rewards_only'] &= np.array_equal(known,arrays['trial_known_after'][at+(t,)])
                        flags['memory_writes_equal_first_discoveries'] &= (arrays['trial_memory_changed'][at+(t,)]==changed
                            and arrays['trial_memory_observation_calls'][at+(t,)]==calls and arrays['trial_memory_discoveries'][at+(t,)]==discovered)
                        route=arrays['trial_routing_signs'][at+(t,)]
                        if method=='feedback_selective':
                            flags['feedback_routing_equals_observed_after'] &= np.array_equal(route,known)
                            flags['current_Go_feedback_precedes_direction'] &= np.array_equal(route,arrays['trial_known_after'][at+(t,)])
                        elif method=='oracle_selective':flags['oracle_privileged_labels_separate'] &= np.array_equal(route,y)
                        else:flags['controls_ignore_observed_memory_for_direction'] &= not np.any(route)
                    flags['memory_writes_equal_first_discoveries'] &= np.array_equal(first,arrays['first_discovery_trials'][at])
                    g=arrays['trial_update_factors'][at];directions=arrays['trial_directions'][at]
                    path=np.concatenate((np.zeros((1,directions.shape[-1])),np.cumsum(g[:,None]*directions,axis=0)))
                    d=arrays['g_'+top+'_'+rep+'_unit_rows'][si]
                    predicted=sigmoid(path[:-1]@d.T)[np.arange(n),arrays['schedules'][si]]
                    flags['head_policy_does_not_directly_lookup_memory'] &= bool(np.max(np.abs(predicted-arrays['trial_probabilities'][at]))<=tol)
                    flags['head_policy_does_not_directly_lookup_memory'] &= bool(np.max(np.abs(path[cfg['checkpoints']]-arrays['checkpoint_heads'][at]))<=tol)
                    flags['head_policy_does_not_directly_lookup_memory'] &= bool(np.max(np.abs(sigmoid(path[cfg['checkpoints']]@d.T)-arrays['checkpoint_probabilities'][at]))<=tol)
                    for ki,step in enumerate(cfg['checkpoints']):
                        saved_known=np.zeros(4,dtype=np.int8) if step==0 else arrays['trial_known_after'][at+(step-1,)]
                        flags['observed_memory_from_obtained_Go_rewards_only'] &= np.array_equal(arrays['checkpoint_known'][at+(ki,)],saved_known)
                        for field in ('memory_observation_calls','memory_discoveries'):
                            expected=0 if step==0 else arrays['trial_'+field][at+(step-1,)]
                            flags['memory_writes_equal_first_discoveries'] &= arrays['checkpoint_'+field][at+(ki,)]==expected
    flags['static_lookup_not_autonomous_experiment']=bool(np.array_equal(arrays['checkpoint_lookup_probabilities'],(arrays['checkpoint_known'].astype(float)+1.)/2.))
    return {key:bool(value) for key,value in flags.items()}


def summarize(cfg,arrays,rows):
    validate_config(cfg);p=arrays['checkpoint_probabilities'];targets=arrays['targets'];tol=cfg['numeric_tolerance']
    correct=np.where(targets==1,p,1-p);initial=correct[...,0,:];final=correct[...,-1,:];scores=final.mean(-1)
    acquired=(scores>=cfg['acquisition_threshold'])&(scores-initial.mean(-1)>=cfg['acquisition_gain'])
    acquired&=(final[...,:2].mean(-1)>=cfg['class_threshold'])&(final[...,2:].mean(-1)>=cfg['class_threshold'])
    maxstep=arrays['trial_head_steps'].max(-1);path=arrays['trial_cumulative_head_path_l2'][...,-1]
    energy=arrays['trial_cumulative_head_squared_l2'][...,-1];endpoint=np.linalg.norm(arrays['checkpoint_heads'][...,-1,:],axis=-1)
    n=cfg['training_trials'];eta=cfg['head_lr'];energy_valid=energy<=n*eta**2+tol;endpoint_valid=endpoint<=path+tol
    within=(maxstep<=eta+tol)&(path<=n*eta+tol)&energy_valid&endpoint_valid
    stepres=float(np.max(np.abs(arrays['trial_head_steps']-arrays['trial_expected_head_steps'])))
    dynres=float(arrays['trial_direction_residuals'].max());information=information_checks(cfg,arrays)
    require(all(information.values()),'Observed reward information/state gate failed')
    lookup=np.where(targets==1,arrays['checkpoint_lookup_probabilities'],1-arrays['checkpoint_lookup_probabilities'])[...,-1,:].mean(-1)
    families=[];contrasts=[];geometries=[];static_valid=True
    for ti,top in enumerate(TOPOLOGIES):
        for ri,rep in enumerate(REPRESENTATIONS):
            prefix='g_'+top+'_'+rep+'_';geoseeds=[]
            for si,seed in enumerate(cfg['model_seeds']):
                record={'seed':seed}
                for key in ('retained_rank','full_row_condition','projector_identity_residual','numeric_selective_qualified','preflight_direction_count','preflight_max_residuals'):
                    value=arrays[prefix+key][si];record[key]=value.tolist()
                static_valid &= bool(record['numeric_selective_qualified'] and record['retained_rank']==4 and record['preflight_direction_count']==324
                    and max(record['preflight_max_residuals'])<=tol)
                geoseeds.append(record)
            geometries.append({'topology':top,'representation':rep,'seeds':geoseeds})
            for mi,method in enumerate(METHODS):
                records=[]
                for si,seed in enumerate(cfg['model_seeds']):
                    at=(ti,ri,mi,si)
                    records.append({'seed':seed,'initial_expected_accuracy':float(initial[at].mean()),'final_expected_accuracy':float(scores[at]),
                        'gain':float(scores[at]-initial[at].mean()),'class_1_expected_accuracy':float(final[at][:2].mean()),
                        'class_0_expected_accuracy':float(final[at][2:].mean()),'acquisition_pass':bool(acquired[at]),
                        'final_head_l2':float(endpoint[at]),'cumulative_head_path_l2':float(path[at]),'cumulative_head_squared_l2':float(energy[at]),
                        'max_head_step_l2':float(maxstep[at]),'within_all_common_cost_bounds':bool(within[at]),
                        'observed_signs_final':arrays['checkpoint_known'][at+(-1,)].tolist(),
                        'first_discovery_trials':arrays['first_discovery_trials'][at].tolist(),
                        'memory_observation_calls':int(arrays['trial_memory_observation_calls'][at+(-1,)]),
                        'distinct_signs_observed':int(arrays['trial_memory_discoveries'][at+(-1,)]),'lookup_expected_accuracy':float(lookup[at])})
                families.append({'topology':top,'representation':rep,'method':method,
                    'role':'primary' if (ti,ri,mi)==(0,0,3) else ('privileged_diagnostic' if mi==2 else 'control'),
                    'mean_final_expected_accuracy':float(scores[ti,ri,mi].mean()),
                    'sd_final_expected_accuracy':float(scores[ti,ri,mi].std(ddof=1)) if len(records)>1 else 0.,
                    'acquisition_pass_seeds':int(acquired[ti,ri,mi].sum()),'mean_cumulative_head_path_l2':float(path[ti,ri,mi].mean()),
                    'mean_final_memory_known':float(arrays['trial_memory_discoveries'][ti,ri,mi,:,-1].mean()),
                    'mean_memory_lookup_accuracy':float(lookup[ti,ri,mi].mean()),'seeds':records})
            contrasts.append({'topology':top,'representation':rep,'feedback_minus_shared_by_seed':(scores[ti,ri,3]-scores[ti,ri,0]).tolist(),
                'feedback_minus_isolated_by_seed':(scores[ti,ri,3]-scores[ti,ri,1]).tolist(),
                'oracle_minus_feedback_by_seed':(scores[ti,ri,2]-scores[ti,ri,3]).tolist(),'descriptive_only':(ti,ri)!=(0,0)})
    numeric=bool(static_valid and within.all() and stepres<=tol and dynres<=tol and all(information.values()))
    comparisons={}
    for name,key in [('shared','feedback_minus_shared_by_seed'),('isolated','feedback_minus_isolated_by_seed')]:
        effect=np.asarray(contrasts[0][key]);comparisons[name]={'mean_effect':float(effect.mean()),'positive_seeds':int(np.count_nonzero(effect>0)),
            'practical_advantage_pass':bool(effect.mean()>=cfg['effect_threshold'] and np.count_nonzero(effect>0)>=cfg['positive_seed_count'])}
    passes=int(acquired[0,0,3].sum());seedcount=len(cfg['model_seeds']);learners=24*seedcount
    require(len(rows)==learners*n,'Complete actual feedback records')
    return {'study_id':cfg['study_id'],'status':'completed','methods':list(METHODS),'families':families,'contrasts':contrasts,
        'H_selective':{'supported':bool(numeric and all(x['practical_advantage_pass'] for x in comparisons.values())),
            'endpoint_comparisons':comparisons,'all_numeric_and_information_gates_passed':numeric,'requires_acquisition':False,'oracle_can_replace_primary':False},
        'H_acquisition':{'supported':bool(numeric and passes>=cfg['population_pass_count']),'acquisition_pass_seeds':passes,
            'all_numeric_and_information_gates_passed':numeric,'requires_selective_advantage':False,'oracle_can_replace_primary':False},
        'budget_validation':{'conditional_scalar_matching_only':True,'actual_cumulative_cost_equalized':False,
            'common_unit_direction_step_ceiling':eta,'common_unit_direction_path_ceiling':n*eta,
            'common_unit_direction_squared_step_sum_ceiling':n*eta**2,'common_bounds_valid_all':bool(within.all()),
            'squared_step_sum_valid_all':bool(energy_valid.all()),'head_endpoint_within_path_all':bool(endpoint_valid.all()),
            'head_step_expected_residual_max':stepres,'all_static_numeric_gates_passed':bool(static_valid),
            'all_dynamic_direction_gates_passed':bool(dynres<=tol),'max_dynamic_direction_residual':dynres},
        'information_validation':information,'geometry':geometries,
        'counts':{'training_feedback_trials':len(rows),'static_probe_cases':int(np.prod(p.shape)),'model_initializations':learners,
            'fixed_feature_constructions':6*seedcount,'model_seeds':seedcount,'head_update_calls':int(arrays['trial_head_update_calls'][...,-1].sum()),
            'head_nonzero_steps':int(np.count_nonzero(arrays['trial_head_steps'])),'internal_update_calls':0,'value_update_calls':0,
            'internal_nonzero_steps':0,'value_nonzero_steps':0,'auxiliary_memory_observation_calls':int(arrays['trial_memory_observation_calls'][...,-1].sum()),
            'auxiliary_memory_discoveries':int(arrays['trial_memory_changed'].sum()),'preflight_direction_evaluations':6*seedcount*324,
            'runtime_selective_direction_evaluations':12*seedcount*n,'lookup_static_cases':int(np.prod(p.shape)),'lookup_autonomous_training_trials':0},
        'resources':{'head_coordinates':int(arrays['checkpoint_heads'].shape[-1]),'observed_sign_memory_cells_per_learner':4,
            'oracle_extra_given_target_signs':4,'pseudoinverse_coefficients_per_group':int(4*arrays['checkpoint_heads'].shape[-1]),
            'direction_solver_SVD_per_group':1,'diagnostic_SVD_per_group':2,'total_producer_SVD_calls':18*seedcount,'dynamic_SVD_calls':0,
            'exact_actual_cumulative_budget_matching_claimed':False,'extra_solver_and_information_processing_cost_equalized':False},
        'interpretation':{'artificial_history_buffer':True,'internal_synapses_fixed':True,'identity_dictionary_is_supplied':True,
            'observed_reward_memory_is_additional_learning_state':True,'oracle_is_privileged_diagnostic':True,
            'helpful_raw_targets_preserved_before_unit_normalization':True,'original_helpful_magnitudes_preserved_after_normalization':False,
            'feedback_uses_hidden_targets_for_routing':False,'prior_failure_or_seed_discarded':False,'generalization_tested':False,
            'new_human_experiment':False,'shared_neural_mechanism_claim':False,'education_effect_claim':False}}


def render(summary,arrays,out):
    """Presentation of verified trajectories; no new training or selection."""
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(exist_ok=False)
    targets=arrays['targets'];p=arrays['checkpoint_probabilities'];grid=arrays['checkpoints']
    correct=np.where(targets==1,p,1-p).mean(-1)
    reference=np.where(targets==1,arrays['checkpoint_lookup_probabilities'],1-arrays['checkpoint_lookup_probabilities']).mean(-1)
    colors=['#555555','#7654a3','#e08214','#16817a'];plotted={'learning':[],'primary_memory':[]}
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    fig.subplots_adjust(left=.06,right=.985,bottom=.08,top=.83,hspace=.4,wspace=.2)
    for ti,top in enumerate(TOPOLOGIES):
        for ri,rep in enumerate(REPRESENTATIONS):
            ax=axes[ri,ti]
            for mi,method in enumerate(METHODS):
                mean=correct[ti,ri,mi].mean(0)
                ax.plot(grid,mean,'o--' if mi==2 else 'o-',color=colors[mi],label=method,markersize=3)
                plotted['learning'].append({'topology':top,'representation':rep,'method':method,
                    'checkpoints':grid.tolist(),'mean_expected_accuracy':mean.tolist()})
            ax.axhline(.8,color='#888888',lw=.8,ls=':');ax.set_ylim(.45,1.02)
            ax.set(title=top+' / '+rep,xlabel='Obtained-feedback trials')
            if ti==0:ax.set_ylabel('Expected correct choice probability')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.suptitle('Selective suppression: output learning on four artificial inputs',y=.98,fontsize=14)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=4,frameon=False,fontsize=10)
    fig.savefig(out/'learning-curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));fig.subplots_adjust(left=.07,right=.98,bottom=.2,top=.82,wspace=.25)
    for mi,method in enumerate(METHODS):
        count=np.count_nonzero(arrays['checkpoint_known'][0,0,mi],axis=-1).mean(0)
        axes[0].plot(grid,count,'o--' if mi==2 else 'o-',color=colors[mi],label=method,markersize=4)
        plotted['primary_memory'].append({'method':method,'checkpoints':grid.tolist(),'mean_observed_signs':count.tolist(),
            'mean_final_head_accuracy':float(correct[0,0,mi,:,-1].mean()),
            'mean_final_static_lookup_accuracy':float(reference[0,0,mi,:,-1].mean())})
    axes[0].set(xlabel='Obtained-feedback trials',ylabel='Observed reward signs (0 to 4)',ylim=(-.1,4.2))
    x=np.arange(4);head=correct[0,0,:,:, -1].mean(1);ref=reference[0,0,:,:, -1].mean(1)
    axes[1].bar(x-.16,head,.32,label='Learned output head',color=colors)
    axes[1].bar(x+.16,ref,.32,label='Static memory-only reference',color='#cccccc')
    axes[1].set_xticks(x,['shared','isolated','oracle','feedback'],rotation=20)
    axes[1].set(ylabel='Expected correct choice probability',ylim=(0,1.05))
    axes[1].legend(loc='lower right',fontsize=8)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Real / raw: observed memory and output are separate states',y=.99,fontsize=13)
    fig.savefig(out/'memory-and-output.png',dpi=150);plt.close(fig)
    return plotted
