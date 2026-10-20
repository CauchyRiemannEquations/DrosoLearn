"""Prospective U/P output learning after actual feedback discovery.

The producer/diagnostic scaffolding is adapted additively from the preserved
self_blend_learning.py. Its six controls use the original SelfBlendHead.
No environment target enters either new head's objective or safety constraints.
"""
from __future__ import annotations
from itertools import product
import numpy as np
from .graph import load_projection
from .human_shell_models import build_graphs
from .pattern_rule import VectorActor
from .update_budget import load_catalog,require
from .selective_geometry import selective_geometry,selective_direction
from .balanced_learning import trial_effects,sigmoid,RESIDUALS
from .self_blend_learning import SelfBlendHead,SCIENCE as PARENT_SCIENCE,BLEND_RESIDUALS
from .cone_geometry import cone_direction

TOPOLOGIES=('real','rewired','ann');REPRESENTATIONS=('raw','identity')
METHODS=('shared','isolated','feedback_selective','balanced','gated_balanced','gated_self_blend',
    'gated_uniform','gated_probability')
NEW_METHODS=METHODS[6:]
SCIENCE={**PARENT_SCIENCE,'study_id':'objective_learning_v1','scope':'prospective_observed_objective_learning',
    'module':'drosolearn.objective_learning','protocol':'docs/objective-learning-protocol.md',
    'methods':list(METHODS),'model_seeds':list(range(17801,17811)),'rng_namespace':202610204,
    'gate_switch_rule':'known_before_four_then_own_uniform_or_current_probability_objective',
    'probability_gradient_domain':'all_abs_own_current_logits_strictly_less_than_40',
    'gain_controls':['gated_balanced','isolated','gated_self_blend'],
    'state_effect_threshold':.05,'state_positive_seed_count':8,
    'measurement_storage':'lossless_deterministic_gzip_csv_crlf',
    'baseline_tag':'probability-alignment-v1','baseline_commit':'9159aa764d44183c21468974ba3b0f208d676731'}


def validate_config(cfg):
    fixture=cfg.get('scope')=='integration_test'
    exempt={'scope','model_seeds','training_trials','checkpoints','population_pass_count',
        'positive_seed_count','protection_nonworse_seed_count','state_positive_seed_count'} if fixture else set()
    for key,value in SCIENCE.items():
        if key not in exempt:require(cfg.get(key)==value,'Fixed prospective objective design differs: '+key)
    if fixture:
        seeds,n,grid=cfg['model_seeds'],cfg['training_trials'],cfg['checkpoints']
        require(type(seeds) is list and 1<=len(seeds)<=2 and len(set(seeds))==len(seeds)
            and all(type(s) is int and 90001<=s<=90100 for s in seeds),'Invented fixture seeds only')
        require(type(n) is int and 4<=n<=16 and n%4==0,'Bounded invented trial budget')
        require(grid==sorted(set(grid)) and grid[0]==0 and grid[-1]==n
            and all(type(t) is int and 0<=t<=n and t%4==0 for t in grid),'Invented checkpoint grid')
        require(all(cfg[key]==len(seeds) for key in ('population_pass_count','positive_seed_count',
            'protection_nonworse_seed_count','state_positive_seed_count')),'Invented pass counts')


def objective_problem(rows,source,observed,logits,kind):
    """Only the head's four already observed signs and its own current state."""
    d,k,z=map(np.asarray,(rows,observed,logits))
    require(d.ndim==2 and d.shape[0]==4 and d.dtype.kind in 'fiu' and np.isfinite(d).all()
        and type(source) is int and 0<=source<4 and k.shape==z.shape==(4,)
        and k.dtype.kind in 'iu' and np.isin(k,[-1,1]).all() and z.dtype.kind in 'fiu'
        and np.isfinite(z).all() and type(kind) is int and kind in (0,1),
        'Full own observed signs, current finite real logits and explicit U/P kind required')
    a=k[source]*k[:,None]*d
    if kind==1:
        require(np.all(np.abs(z)<40.),'Current P derivative requires strictly unclipped logits')
        p=sigmoid(z);c=(p*(1-p))@a/4.
    else:c=a.mean(0)
    return a,c


def objective_certificate(a,c,proof):
    w,v,mu=proof['direction'],proof['projected_objective'],proof['multipliers']
    return np.asarray([c@w,np.linalg.norm(v),proof['dual_upper_bound'],np.linalg.norm(w),
        max(0.,float(-(a@w).min())),np.max(np.abs(v-c-a.T@mu)),
        np.max(np.abs(mu*(a@v))),abs(float(np.linalg.norm(v)-c@w))])


def blank_objective(dimension):
    return {'objective_applicable':False,'objective_kind':-1,'objective_case_index':-1,
        'objective_logits_before':np.zeros(4),'objective_vector':np.zeros(dimension),
        'objective_certificate_residual_max':0.}


class ObjectiveHead(SelfBlendHead):
    """The two new policies; early isolation is exactly the preserved control."""
    def __init__(self,geometry,method,rate):
        require(method in NEW_METHODS,'Only U/P methods use the new head')
        super().__init__(geometry,'isolated',rate)
        self.method=method;self.objective_cases=[];self.objective_cache={};self.objective_requests=0

    def observe(self,index,action,reward):
        require(type(index) is int and 0<=index<4 and type(action) is int and action in (0,1)
            and type(reward) is int and ((action==0 and reward==0) or (action==1 and reward in (-1,1))),
            'Only actual obtained Go reward or NoGo zero may enter a head')
        original=self.method;before=self.known.copy();count=int(np.count_nonzero(before));full=count==4
        if not full:
            self.method='isolated'
            try:e=super().observe(index,action,reward)
            finally:self.method=original
            return {**e,**blank_objective(len(self.head))}
        kind=NEW_METHODS.index(original);switch=int(not self.gate_activated)
        z=self.logits();p=float(sigmoid(z)[index]);d=self.geometry['unit_rows']
        direction=np.zeros_like(self.head);projection=direction.copy();raw_norm=scale=0.
        extra=blank_objective(len(self.head))
        if action:
            require(before[index]==reward,'Full observed memory agrees with obtained reward')
            a,c=objective_problem(d,index,before,z,kind)
            address=index if kind==0 else self.calls;key=(address,index,*map(int,before))
            cached=self.objective_cache.get(key) if kind==0 else None
            if cached is None:
                proof=cone_direction(a,c,self.geometry['rcond'],self.geometry['tolerance'])
                cert=objective_certificate(a,c,proof)
                require(proof['numeric_qualified'] and abs(cert[3]-1)<=self.geometry['tolerance']
                    and np.max(cert[4:])<=self.geometry['tolerance'],
                    'Actual U/P requires qualified nonzero unit global support certificate')
                cached=len(self.objective_cases)
                self.objective_cases.append({'key':key,'objective':c.copy(),'logits':z.copy(),
                    'direction':proof['direction'].copy(),'projection':proof['projected_objective'].copy(),
                    'multipliers':proof['multipliers'].copy(),'certificate':cert})
                if kind==0:self.objective_cache[key]=cached
            case=self.objective_cases[cached];direction=case['direction'];projection=case['projection']
            raw_norm=float(np.linalg.norm(projection));scale=1/raw_norm
            extra={'objective_applicable':True,'objective_kind':kind,'objective_case_index':cached,
                'objective_logits_before':z.copy(),'objective_vector':c.copy(),
                'objective_certificate_residual_max':float(np.max(case['certificate'][4:]))}
            self.memory_observation_calls+=1;self.objective_requests+=1
        self.gate_activated=True
        factor=self.rate*(action-p)*reward;old=self.head.copy();self.head+=factor*direction;self.calls+=1
        step=self.head-old;scores=d@step
        return {'direction':direction.copy(),'desired_influence':projection@d.T,'influence':direction@d.T,
            'harm_mask':np.zeros(4,bool),'raw_norm':raw_norm,'normalization_scale':scale,
            'direction_residuals':np.zeros(7),'direction_residuals_applicable':False,'case_index':-1,
            'known_before':before,'known_after':self.known.copy(),'routing_signs':self.known.copy(),
            'memory_changed':0,'memory_observation_calls':self.memory_observation_calls,
            'memory_discoveries':self.discoveries,'update_factor':factor,'head_step':float(np.linalg.norm(step)),
            'own_score_step':float(scores[index]),'other_score_step':float(np.linalg.norm(np.delete(scores,index))),
            'expected_head_step':abs(float(factor))*float(np.linalg.norm(direction)),
            'blend_applicable':False,'blend_isolated_direction':np.zeros_like(self.head),
            'blend_balanced_direction':np.zeros_like(self.head),'blend_raw_norm':0.,'blend_residuals':np.zeros(5),
            'direction_mode':kind+5,'gate_known_count_before':count,'gate_full_before':True,
            'gate_switch_event':switch,**extra}


def make_head(geometry,method,rate):
    return ObjectiveHead(geometry,method,rate) if method in NEW_METHODS else SelfBlendHead(geometry,method,rate)


def produce(cfg,root):
    validate_config(cfg);data=load_catalog(cfg,root);graph=load_projection(root/cfg['graph'])
    seeds,n,grid=cfg['model_seeds'],cfg['training_trials'],cfg['checkpoints'];c=graph.n_mbon
    shape=(3,2,8,len(seeds),n);cshape=shape[:-1]+(len(grid),);arrays=dict(data)
    integer={'actions','rewards','head_update_calls','internal_update_calls','value_update_calls',
        'memory_changed','memory_observation_calls','memory_discoveries','unknown_other_exposure_count'}
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
    arrays['trial_case_indices']=np.full(shape,-1,dtype=np.int32)
    arrays['trial_direction_residuals_applicable']=np.empty(shape,dtype=bool)
    for field in ('unknown_other_mask_before','unknown_other_mask_after'):
        arrays['trial_'+field]=np.empty(shape+(4,),dtype=bool)
    for field in ('correct_probability_delta','aligned_margin_delta'):
        arrays['trial_'+field]=np.empty(shape+(4,))
    for field in ('unknown_other_negative_correct_probability_sum','unknown_other_positive_correct_probability_sum','unknown_other_negative_margin_sum','unknown_other_exposure_count'):
        arrays['trial_'+field]=np.empty(shape,dtype=np.int32 if field.endswith('count') else float)
    arrays['trial_direction_modes']=np.empty(shape,dtype=np.int8)
    arrays['trial_gate_known_counts_before']=np.empty(shape,dtype=np.int8)
    arrays['trial_gate_full_before']=np.empty(shape,dtype=bool)
    arrays['trial_gate_switch_events']=np.empty(shape,dtype=np.int8)
    arrays['first_balanced_mode_trials']=np.full(shape[:-1],n+1,dtype=np.int32)
    arrays['first_blend_mode_trials']=np.full(shape[:-1],n+1,dtype=np.int32)
    arrays['trial_blend_applicable']=np.zeros(shape,dtype=bool)
    arrays['trial_blend_isolated_directions']=np.zeros(shape+(c,))
    arrays['trial_blend_balanced_directions']=np.zeros(shape+(c,))
    arrays['trial_blend_raw_norm']=np.zeros(shape)
    arrays['trial_blend_residuals']=np.zeros(shape+(5,))
    for name in ('first_uniform_mode_trials','first_probability_mode_trials'):
        arrays[name]=np.full(shape[:-1],n+1,dtype=np.int32)
    arrays['trial_objective_applicable']=np.zeros(shape,dtype=bool)
    arrays['trial_objective_logits_before']=np.zeros(shape+(4,))
    arrays['trial_objective_vectors']=np.zeros(shape+(c,))
    arrays['trial_objective_kind']=np.full(shape,-1,dtype=np.int8)
    arrays['trial_objective_case_indices']=np.full(shape,-1,dtype=np.int32)
    cases=[];objective_cases=[]
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
    # Every geometry and all 81 selective sign states qualify before behavioral RNG.
    for si,seed in enumerate(seeds):
        rng=np.random.default_rng(np.random.SeedSequence([cfg['rng_namespace'],seed,0]))
        arrays['schedules'][si]=np.concatenate([rng.permutation(4) for _ in range(n//4)])
        arrays['uniforms'][si]=np.random.default_rng(np.random.SeedSequence([cfg['rng_namespace'],seed,1])).random(n)
    true_signs=2*data['targets']-1
    for si,ti,ri,seed,topology,representation,base,witness,geo,prefix in prepared:
        for mi,method in enumerate(METHODS):
            actor=make_head(geo,method,cfg['head_lr']);case_start=len(cases);objective_start=len(objective_cases)
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
                reward=int(true_signs[source]) if a else 0;oldhead=actor.head.copy();e=actor.observe(source,a,reward)
                effects=trial_effects(geo['unit_rows'],oldhead,actor.head,true_signs,source,e['known_before'],e['known_after'])
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
                case_index=case_start+e['case_index'] if e['case_index']>=0 else -1
                arrays['trial_case_indices'][at]=case_index
                arrays['trial_direction_residuals_applicable'][at]=e['direction_residuals_applicable']
                for field,value in effects.items():arrays['trial_'+field][at]=value
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
                row['case_index']=case_index;row['direction_residuals_applicable']=int(e['direction_residuals_applicable'])
                for field in ('unknown_other_negative_correct_probability_sum','unknown_other_positive_correct_probability_sum','unknown_other_negative_margin_sum','unknown_other_exposure_count'):
                    row[field]=effects[field]
                for field,key in [('unknown_other_before','unknown_other_mask_before'),('unknown_other_after','unknown_other_mask_after'),('correct_probability_delta','correct_probability_delta'),('aligned_margin_delta','aligned_margin_delta')]:
                    for j in range(4):row[field+'_'+str(j)]=int(effects[key][j]) if field.startswith('unknown_other') else float(effects[key][j])
                for field,key in [('direction_modes','direction_mode'),('gate_known_counts_before','gate_known_count_before'),
                    ('gate_full_before','gate_full_before'),('gate_switch_events','gate_switch_event')]:
                    arrays['trial_'+field][at]=e[key];row[key]=int(e[key])
                if e['gate_switch_event']:
                    arrays[{4:'first_balanced_mode_trials',5:'first_blend_mode_trials',6:'first_uniform_mode_trials',7:'first_probability_mode_trials'}[mi]][ti,ri,mi,si]=t+1
                for name,key in [('applicable','blend_applicable'),('isolated_directions','blend_isolated_direction'),('balanced_directions','blend_balanced_direction'),('raw_norm','blend_raw_norm'),('residuals','blend_residuals')]:
                    arrays['trial_blend_'+name][at]=e[key]
                row.update(blend_applicable=int(e['blend_applicable']),blend_raw_norm=e['blend_raw_norm'],blend_residual_max=float(e['blend_residuals'].max()))
                extra={**blank_objective(c),**{key:value for key,value in e.items() if key.startswith('objective_')}}
                oi=objective_start+extra['objective_case_index'] if extra['objective_case_index']>=0 else -1
                arrays['trial_objective_case_indices'][at]=oi
                for field,key in [('applicable','objective_applicable'),('logits_before','objective_logits_before'),
                    ('vectors','objective_vector'),('kind','objective_kind')]:arrays['trial_objective_'+field][at]=extra[key]
                row.update(objective_applicable=int(extra['objective_applicable']),objective_kind=extra['objective_kind'],
                    objective_case_index=oi,objective_vector_l2=float(np.linalg.norm(extra['objective_vector'])),
                    objective_certificate_residual_max=extra['objective_certificate_residual_max'])
                for j in range(4):row['objective_logit_before_'+str(j)]=float(extra['objective_logits_before'][j])
                rows.append(row)
                if t+1 in lookup:checkpoint(t+1)
            for case in actor.cases:cases.append({'key':(si,ti,ri,mi,*case['key']),**{k:v for k,v in case.items() if k!='key'}})
            for case in getattr(actor,'objective_cases',[]):
                objective_cases.append({'key':(si,ti,ri,mi,*case['key']),**{k:v for k,v in case.items() if k!='key'}})
        final=base.witness()
        for field in ('weights','encoder','bias','probe_kc','probe_mbon'):
            require(np.array_equal(witness[field],final[field]),'Fixed internal weights/encoder/activity changed')
            static.setdefault('finalfixed_'+prefix+field,[]).append(final[field])
    arrays.update({name:np.stack(values) for name,values in static.items()})
    arrays['case_keys']=np.asarray([case['key'] for case in cases],dtype=np.int32).reshape(-1,9)
    for field,width in [('directions',c),('projections',c),('multipliers',4),('certificates',8)]:
        singular={'directions':'direction','projections':'projection','multipliers':'multipliers','certificates':'certificate'}[field]
        arrays['case_'+field]=np.asarray([case[singular] for case in cases]).reshape(-1,width)
    arrays['objective_case_keys']=np.asarray([case['key'] for case in objective_cases],dtype=np.int32).reshape(-1,10)
    for plural,singular,width in [('objectives','objective',c),('logits','logits',4),('directions','direction',c),
        ('projections','projection',c),('multipliers','multipliers',4),('certificates','certificate',8)]:
        arrays['objective_case_'+plural]=np.asarray([case[singular] for case in objective_cases],dtype=float).reshape(-1,width)
    require(len(arrays)==215,'Exact prospective objective learning numeric inventory')
    return arrays,rows



def information_checks(cfg,arrays):
    flags={key:True for key in ('observed_memory_from_obtained_Go_rewards_only','NoGo_does_not_reveal_unknown_sign',
        'current_Go_feedback_precedes_direction','controls_and_gated_isolation_ignore_memory_for_direction',
        'feedback_balanced_and_gate_balanced_routing_equals_observed_after','head_policy_does_not_directly_lookup_memory',
        'static_lookup_not_autonomous_experiment','memory_writes_equal_first_discoveries','no_oracle_learner_present')}
    n=cfg['training_trials'];tol=cfg['numeric_tolerance']
    for ti,top in enumerate(TOPOLOGIES):
        for ri,rep in enumerate(REPRESENTATIONS):
            for mi,method in enumerate(METHODS):
                for si in range(len(cfg['model_seeds'])):
                    at=(ti,ri,mi,si);known=np.zeros(4,dtype=np.int8);calls=discoveries=0
                    first=np.full(4,n+1,dtype=int)
                    for t,source in enumerate(arrays['schedules'][si]):
                        source=int(source);ix=at+(t,);a=int(arrays['trial_actions'][ix]);r=int(arrays['trial_rewards'][ix])
                        before=known.copy();changed=int(a==1 and known[source]==0)
                        flags['observed_memory_from_obtained_Go_rewards_only'] &= np.array_equal(before,arrays['trial_known_before'][ix])
                        if a:
                            flags['observed_memory_from_obtained_Go_rewards_only'] &= r in (-1,1) and known[source] in (0,r)
                            known[source]=r;calls+=1;discoveries+=changed
                            if changed:first[source]=t+1
                        else:flags['NoGo_does_not_reveal_unknown_sign'] &= r==0 and np.array_equal(before,arrays['trial_known_after'][ix])
                        flags['observed_memory_from_obtained_Go_rewards_only'] &= np.array_equal(known,arrays['trial_known_after'][ix])
                        flags['memory_writes_equal_first_discoveries'] &= (arrays['trial_memory_changed'][ix]==changed
                            and arrays['trial_memory_observation_calls'][ix]==calls and arrays['trial_memory_discoveries'][ix]==discoveries)
                        route=arrays['trial_routing_signs'][ix]
                        if mi in (2,3) or (mi>=4 and arrays['trial_direction_modes'][ix] in (3,4,5,6)):
                            flags['feedback_balanced_and_gate_balanced_routing_equals_observed_after'] &= np.array_equal(route,known)
                            flags['current_Go_feedback_precedes_direction'] &= np.array_equal(route,arrays['trial_known_after'][ix])
                        else:flags['controls_and_gated_isolation_ignore_memory_for_direction'] &= not np.any(route)
                    flags['memory_writes_equal_first_discoveries'] &= np.array_equal(first,arrays['first_discovery_trials'][at])
                    g=arrays['trial_update_factors'][at];directions=arrays['trial_directions'][at]
                    heads=np.concatenate((np.zeros((1,directions.shape[-1])),np.cumsum(g[:,None]*directions,axis=0)))
                    d=arrays['g_'+top+'_'+rep+'_unit_rows'][si]
                    p=sigmoid(heads[:-1]@d.T)[np.arange(n),arrays['schedules'][si]]
                    flags['head_policy_does_not_directly_lookup_memory'] &= bool(np.max(np.abs(p-arrays['trial_probabilities'][at]))<=tol)
                    flags['head_policy_does_not_directly_lookup_memory'] &= np.array_equal(arrays['trial_actions'][at],arrays['uniforms'][si]<p)
                    flags['head_policy_does_not_directly_lookup_memory'] &= bool(np.max(np.abs(heads[cfg['checkpoints']]-arrays['checkpoint_heads'][at]))<=tol)
                    flags['head_policy_does_not_directly_lookup_memory'] &= bool(np.max(np.abs(sigmoid(heads[cfg['checkpoints']]@d.T)-arrays['checkpoint_probabilities'][at]))<=tol)
                    for ki,step in enumerate(cfg['checkpoints']):
                        expected=np.zeros(4,dtype=np.int8) if step==0 else arrays['trial_known_after'][at+(step-1,)]
                        flags['observed_memory_from_obtained_Go_rewards_only'] &= np.array_equal(arrays['checkpoint_known'][at+(ki,)],expected)
                        for field in ('memory_observation_calls','memory_discoveries'):
                            expected=0 if step==0 else arrays['trial_'+field][at+(step-1,)]
                            flags['memory_writes_equal_first_discoveries'] &= arrays['checkpoint_'+field][at+(ki,)]==expected
    flags['static_lookup_not_autonomous_experiment']=bool(np.array_equal(arrays['checkpoint_lookup_probabilities'],(arrays['checkpoint_known'].astype(float)+1.)/2.))
    return {key:bool(value) for key,value in flags.items()}



def blend_checks(cfg,a):
    active=(a['trial_direction_modes']==4)&(a['trial_actions']==1);tol=cfg['numeric_tolerance']
    residual=float(a['trial_blend_residuals'].max())
    flags={'only_actual_post_Go_has_blend_diagnostics':bool(np.array_equal(active,a['trial_blend_applicable'])),
        'all_unit_mix_and_unit_score_retention_residuals_pass':bool(residual<=tol),
        'inactive_blend_padding_zero':bool(all(not np.any(a['trial_blend_'+f][~active]) for f in
            ('isolated_directions','balanced_directions','raw_norm','residuals'))),
        'raw_mix_norm_in_half_to_one':bool(np.all((a['trial_blend_raw_norm'][active]>=.5-tol)&(a['trial_blend_raw_norm'][active]<=1+tol)))}
    return {**flags,'max_residual':residual,'residual_names':list(BLEND_RESIDUALS)}


def summarize(cfg,arrays,rows):
    validate_config(cfg);tol=cfg['numeric_tolerance'];n=cfg['training_trials'];eta=cfg['head_lr']
    targets=arrays['targets'];p=arrays['checkpoint_probabilities'];correct=np.where(targets==1,p,1-p)
    initial=correct[...,0,:];final=correct[...,-1,:];scores=final.mean(-1)
    acquired=(scores>=cfg['acquisition_threshold'])&(scores-initial.mean(-1)>=cfg['acquisition_gain'])
    acquired&=(final[...,:2].mean(-1)>=cfg['class_threshold'])&(final[...,2:].mean(-1)>=cfg['class_threshold'])
    path=arrays['trial_cumulative_head_path_l2'][...,-1];energy=arrays['trial_cumulative_head_squared_l2'][...,-1]
    endpoint=np.linalg.norm(arrays['checkpoint_heads'][...,-1,:],axis=-1);maxstep=arrays['trial_head_steps'].max(-1)
    within=(maxstep<=eta+tol)&(path<=n*eta+tol)&(energy<=n*eta**2+tol)&(endpoint<=path+tol)
    stepres=float(np.max(np.abs(arrays['trial_head_steps']-arrays['trial_expected_head_steps'])))
    applicable=arrays['trial_direction_residuals_applicable'];controlres=float(arrays['trial_direction_residuals'][applicable].max())
    certres=float(arrays['case_certificates'][:,4:].max()) if len(arrays['case_keys']) else 0.
    information=information_checks(cfg,arrays);require(all(information.values()),'Observed information integrity failed')
    known=arrays['trial_known_after']!=0;margin=arrays['trial_aligned_margin_delta'];negative_known=known&(margin < -tol)
    unknown=arrays['trial_unknown_other_negative_correct_probability_sum'].sum(-1)
    unknown_help=arrays['trial_unknown_other_positive_correct_probability_sum'].sum(-1)
    unknown_margin=arrays['trial_unknown_other_negative_margin_sum'].sum(-1)
    exposure=arrays['trial_unknown_other_exposure_count'].sum(-1)
    require(np.array_equal(arrays['trial_unknown_other_mask_before'],arrays['trial_unknown_other_mask_after']),
        'Unknown-other scoring mask changed across current reward')
    all_unknown_before=np.count_nonzero(arrays['trial_known_before']==0,axis=-1)
    all_unknown_after=np.count_nonzero(arrays['trial_known_after']==0,axis=-1)
    require(np.array_equal(all_unknown_before-all_unknown_after,arrays['trial_memory_changed']),
        'Source-inclusive unknown count may decrease only on a first actual Go reward')
    families=[];geometries=[];contrasts=[];static_valid=True
    for ti,top in enumerate(TOPOLOGIES):
        for ri,rep in enumerate(REPRESENTATIONS):
            prefix='g_'+top+'_'+rep+'_';geoseeds=[]
            for si,seed in enumerate(cfg['model_seeds']):
                record={'seed':seed}
                for key in ('retained_rank','full_row_condition','projector_identity_residual','numeric_selective_qualified','preflight_direction_count','preflight_max_residuals'):
                    record[key]=arrays[prefix+key][si].tolist()
                static_valid &= bool(record['numeric_selective_qualified'] and record['retained_rank']==4
                    and record['preflight_direction_count']==324 and max(record['preflight_max_residuals'])<=tol)
                geoseeds.append(record)
            geometries.append({'topology':top,'representation':rep,'seeds':geoseeds})
            for mi,method in enumerate(METHODS):
                records=[]
                for si,seed in enumerate(cfg['model_seeds']):
                    at=(ti,ri,mi,si);actions=arrays['trial_actions'][at]
                    sampled=(actions==targets[arrays['schedules'][si]])
                    records.append({'seed':seed,'initial_expected_accuracy':float(initial[at].mean()),
                        'final_expected_accuracy':float(scores[at]),'gain':float(scores[at]-initial[at].mean()),
                        'class_1_expected_accuracy':float(final[at][:2].mean()),'class_0_expected_accuracy':float(final[at][2:].mean()),
                        'acquisition_pass':bool(acquired[at]),'sampled_accuracy_all_trials':float(sampled.mean()),
                        'sampled_accuracy_last_100_trials':float(sampled[-min(100,n):].mean()),
                        'final_head_l2':float(endpoint[at]),'cumulative_head_path_l2':float(path[at]),
                        'cumulative_head_squared_l2':float(energy[at]),'max_head_step_l2':float(maxstep[at]),
                        'within_all_common_cost_bounds':bool(within[at]),'observed_signs_final':arrays['checkpoint_known'][at+(-1,)].tolist(),
                        'first_discovery_trials':arrays['first_discovery_trials'][at].tolist(),
                        'first_balanced_mode_trial':int(arrays['first_balanced_mode_trials'][at]),
                        'first_blend_mode_trial':int(arrays['first_blend_mode_trials'][at]),
                        'first_uniform_mode_trial':int(arrays['first_uniform_mode_trials'][at]),
                        'first_probability_mode_trial':int(arrays['first_probability_mode_trials'][at]),
                        'memory_observation_calls':int(arrays['trial_memory_observation_calls'][at+(-1,)]),
                        'distinct_signs_observed':int(arrays['trial_memory_discoveries'][at+(-1,)]),
                        'unknown_other_negative_correct_probability_sum':float(unknown[at]),
                        'unknown_other_positive_correct_probability_sum':float(unknown_help[at]),
                        'unknown_other_negative_margin_sum':float(unknown_margin[at]),
                        'unknown_other_exposure_count':int(exposure[at]),
                        'unknown_other_negative_correct_probability_per_exposure':float(unknown[at]/exposure[at]) if exposure[at] else None,
                        'known_destination_negative_margin_events':int(negative_known[at].sum())})
                families.append({'topology':top,'representation':rep,'method':method,
                    'role':'co_primary' if (ti,ri)==(0,0) and mi>=6 else 'control',
                    'mean_final_expected_accuracy':float(scores[ti,ri,mi].mean()),
                    'sd_final_expected_accuracy':float(scores[ti,ri,mi].std(ddof=1)) if len(records)>1 else 0.,
                    'acquisition_pass_seeds':int(acquired[ti,ri,mi].sum()),
                    'mean_cumulative_head_path_l2':float(path[ti,ri,mi].mean()),
                    'mean_unknown_other_negative_correct_probability_sum':float(unknown[ti,ri,mi].mean()),
                    'mean_unknown_other_exposure_count':float(exposure[ti,ri,mi].mean()),'seeds':records})
            contrasts.append({'topology':top,'representation':rep,'descriptive_only':(ti,ri)!=(0,0),
                'endpoint_differences':{method:{name:(scores[ti,ri,mi]-scores[ti,ri,METHODS.index(name)]).tolist()
                    for name in cfg['gain_controls']} for mi,method in ((6,NEW_METHODS[0]),(7,NEW_METHODS[1]))},
                'probability_minus_uniform_by_seed':(scores[ti,ri,7]-scores[ti,ri,6]).tolist()})
    gates=gate_checks(cfg,arrays);require(all(gates.values()),'Observed gate/prefix integrity failed')
    blend=blend_checks(cfg,arrays);require(all(blend[k] for k in blend if isinstance(blend[k],bool)),'Old blend integrity failed')
    objective=objective_checks(cfg,arrays)
    require(all(value for value in objective.values() if type(value) is bool),'Actual U/P objective integrity failed')
    safe=bool(not np.any(negative_known[:,:,1:]))
    numeric=bool(static_valid and within.all() and stepres<=tol and controlres<=tol and certres<=tol and safe
        and all(information.values()) and objective['max_certificate_residual']<=tol)
    require(numeric,'Actual-learning numeric, known-safety, information or common-cost gate failed')
    gain={};acquisition={};protection={}
    for mi,method in ((6,NEW_METHODS[0]),(7,NEW_METHODS[1])):
        comparisons={}
        for name in cfg['gain_controls']:
            effect=scores[0,0,mi]-scores[0,0,METHODS.index(name)]
            comparisons[name]={'mean_effect':float(effect.mean()),'positive_seeds':int(np.count_nonzero(effect>0)),
                'practical_advantage_pass':bool(effect.mean()>=cfg['effect_threshold'] and np.count_nonzero(effect>0)>=cfg['positive_seed_count'])}
        gain[method]={'supported':bool(numeric and all(x['practical_advantage_pass'] for x in comparisons.values())),
            'endpoint_comparisons':comparisons,'all_numeric_and_information_gates_passed':numeric,'requires_acquisition':False}
        passes=int(acquired[0,0,mi].sum())
        acquisition[method]={'supported':bool(numeric and passes>=cfg['population_pass_count']),
            'acquisition_pass_seeds':passes,'all_numeric_and_information_gates_passed':numeric,'requires_gain':False}
        checks={}
        for name,index in [('gated_balanced',4),('isolated',1)]:
            diff=unknown[0,0,mi]-unknown[0,0,index];count=int(np.count_nonzero(diff<=tol))
            checks[name]={'mean_effect':float(diff.mean()),'nonworsening_seeds':count,
                'all_seeds_nonworsening':bool(count>=cfg['protection_nonworse_seed_count'])}
        maximum=float(unknown[0,0,mi].max())
        protection[method]={'supported':bool(numeric and maximum<=cfg['protection_absolute_tolerance']
                and all(r['all_seeds_nonworsening'] for r in checks.values())),
            'maximum_primary_seed_unknown_damage':maximum,'absolute_tolerance':cfg['protection_absolute_tolerance'],
            'comparisons':checks,'operational_protection_expected_by_design':True,
            'novel_learning_mechanism_demonstrated':False,'all_numeric_and_information_gates_passed':numeric}
    state=scores[0,0,7]-scores[0,0,6]
    seedcount=len(cfg['model_seeds']);learners=48*seedcount
    require(len(rows)==learners*n and all(len(row)==96 for row in rows),'Complete 96-column autonomous trial record')
    active=arrays['trial_objective_applicable'];kind=arrays['trial_objective_kind'];keys=arrays['objective_case_keys']
    ucount=int(np.count_nonzero(keys[:,3]==6));pcount=int(np.count_nonzero(keys[:,3]==7))
    return {'study_id':cfg['study_id'],'status':'completed','methods':list(METHODS),'families':families,
        'contrasts':contrasts,'geometry':geometries,'H_gain':gain,'H_acquisition':acquisition,'H_protection':protection,
        'H_state':{'supported':bool(numeric and state.mean()>=cfg['state_effect_threshold']
                and np.count_nonzero(state>0)>=cfg['state_positive_seed_count']),
            'comparison':'gated_probability_minus_gated_uniform','mean_effect':float(state.mean()),
            'positive_seeds':int(np.count_nonzero(state>0)),'all_numeric_and_information_gates_passed':numeric,
            'requires_acquisition':False},
        'gate_validation':gates,'blend_validation':blend,'objective_validation':objective,
        'validation':{'all_numeric_and_information_gates_passed':numeric,'all_common_cost_bounds_passed':bool(within.all()),
            'all_static_geometry_gates_passed':bool(static_valid),'all_control_direction_gates_passed':bool(controlres<=tol),
            'all_balanced_case_certificates_passed':bool(certres<=tol),'all_nonshared_known_destinations_safe':safe,
            'head_step_expected_residual_max':stepres,'control_direction_residual_max':controlres,'balanced_certificate_residual_max':certres,
            'unknown_other_masks_equal_before_after':True,'source_inclusive_unknown_count_drop_equals_discoveries':True},
        'information_validation':information,
        'budget_validation':{'common_step_ceiling':eta,'common_path_ceiling':n*eta,'common_squared_step_sum_ceiling':n*eta**2,
            'actual_cumulative_cost_equalized':False,'extra_solver_and_information_processing_cost_equalized':False},
        'counts':{'training_feedback_trials':len(rows),'static_probe_cases':int(np.prod(p.shape)),
            'model_initializations':learners,'fixed_feature_constructions':6*seedcount,'model_seeds':seedcount,
            'head_update_calls':int(arrays['trial_head_update_calls'][...,-1].sum()),
            'head_nonzero_steps':int(np.count_nonzero(arrays['trial_head_steps'])),
            'internal_update_calls':0,'value_update_calls':0,'internal_nonzero_steps':0,'value_nonzero_steps':0,
            'auxiliary_memory_observation_calls':int(arrays['trial_memory_observation_calls'][...,-1].sum()),
            'auxiliary_memory_discoveries':int(arrays['trial_memory_changed'].sum()),
            'preflight_direction_evaluations':6*seedcount*324,'runtime_selective_direction_evaluations':6*seedcount*n,
            'balanced_direction_requests':int(np.count_nonzero(np.isin(arrays['trial_direction_modes'],[3,4])&(arrays['trial_actions']==1))),
            'always_balanced_direction_requests':int(arrays['trial_actions'][:,:,3].sum()),
            'gated_balanced_direction_requests':int(np.count_nonzero((arrays['trial_direction_modes'][:,:,4]==3)&(arrays['trial_actions'][:,:,4]==1))),
            'gated_isolated_direction_evaluations':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,4]==1)),
            'gated_balanced_mode_trials':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,4]==3)),
            'gated_gate_activations':int(arrays['trial_gate_switch_events'][:,:,4].sum()),
            'self_blend_gate_activations':int(arrays['trial_gate_switch_events'][:,:,5].sum()),
            'self_blend_direction_requests':int(arrays['trial_blend_applicable'].sum()),
            'self_blend_mode_trials':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,5]==4)),
            'self_blend_isolated_direction_evaluations':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,5]==1)),
            'balanced_unique_solver_cases':len(arrays['case_keys']),
            'balanced_NoGo_solver_skips':int(np.count_nonzero(np.isin(arrays['trial_direction_modes'],[3,4])&(arrays['trial_actions']==0))),
            'uniform_direction_requests':int(np.count_nonzero(active&(kind==0))),
            'probability_direction_requests':int(np.count_nonzero(active&(kind==1))),
            'uniform_unique_solver_cases':ucount,'probability_unique_solver_cases':pcount,
            'objective_unique_solver_cases':ucount+pcount,'total_unique_solver_cases':len(arrays['case_keys'])+ucount+pcount,
            'uniform_gate_activations':int(arrays['trial_gate_switch_events'][:,:,6].sum()),
            'probability_gate_activations':int(arrays['trial_gate_switch_events'][:,:,7].sum()),
            'uniform_mode_trials':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,6]==5)),
            'probability_mode_trials':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,7]==6)),
            'uniform_isolated_direction_evaluations':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,6]==1)),
            'probability_isolated_direction_evaluations':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,7]==1)),
            'objective_NoGo_solver_skips':int(np.count_nonzero(np.isin(arrays['trial_direction_modes'],[5,6])&(arrays['trial_actions']==0))),
            'unknown_other_destination_exposures':int(exposure.sum()),
            'source_inclusive_unknown_exposures_before_feedback':int(all_unknown_before.sum()),
            'source_inclusive_unknown_exposures_after_feedback':int(all_unknown_after.sum()),
            'lookup_static_cases':int(np.prod(p.shape)),'lookup_autonomous_training_trials':0,
            'numeric_arrays':len(arrays),'measurement_columns':96},
        'resources':{'head_coordinates':int(arrays['checkpoint_heads'].shape[-1]),'observed_sign_memory_cells_per_learner':4,
            'privileged_target_signs_given_to_any_learner':0,'balanced_solver_cache_only_actual_observations':True,
            'balanced_solver_calls_equal_unique_actual_Go_states':True,'processing_cost_equalized':False,
            'uniform_cache_by_actual_source_and_full_observed_signs':True,'probability_fresh_case_per_actual_full_Go':True},
        'interpretation':{'artificial_history_buffer':True,'internal_synapses_fixed':True,'identity_dictionary_is_supplied':True,
            'observed_reward_memory_is_additional_learning_state':True,'true_labels_used_for_unknown_harm_evaluation_only':True,
            'any_learner_uses_hidden_targets_for_routing':False,'prior_failure_or_seed_discarded':False,
            'unknown_means_reward_unobserved_not_novel_input':True,'generalization_tested':False,
            'probability_direction_tangent_not_finite_global_optimum':True,'own_actual_current_state_used_for_each_policy':True,
            'new_human_experiment':False,'shared_neural_mechanism_claim':False,'education_effect_claim':False}}


def gate_checks(cfg,a):
    flags={name:True for name in ('mode_uses_known_before_only','fourth_discovery_update_is_isolated',
        'first_balanced_mode_follows_full_discovery','first_blend_mode_follows_full_discovery',
        'first_uniform_mode_follows_full_discovery','first_probability_mode_follows_full_discovery',
        'exact_isolated_prefix_through_full_discovery','exact_gated_balanced_prefix_through_full_discovery',
        'gated_and_isolated_discovery_trials_equal','partial_gated_has_no_balanced_case',
        'only_full_observation_states_in_gated_case_cache','full_NoGo_has_zero_direction_and_no_case',
        'gate_switch_once_at_first_full_mode','all_methods_record_memory_before_gate')}
    n=cfg['training_trials'];counts=np.count_nonzero(a['trial_known_before'],axis=-1)
    flags['all_methods_record_memory_before_gate']=bool(np.array_equal(counts,a['trial_gate_known_counts_before'])
        and np.array_equal(counts==4,a['trial_gate_full_before']))
    flags['mode_uses_known_before_only']=bool(all(np.all(a['trial_direction_modes'][:,:,mi]==mi) for mi in range(4))
        and all(np.array_equal(a['trial_direction_modes'][:,:,mi],np.where(counts[:,:,mi]==4,mode,1))
            for mi,mode in ((4,3),(5,4),(6,5),(7,6))))
    flags['gate_switch_once_at_first_full_mode']=bool(not np.any(a['trial_gate_switch_events'][:,:,:4]))
    for mi,field,flag in [(4,'first_balanced_mode_trials','first_balanced_mode_follows_full_discovery'),
        (5,'first_blend_mode_trials','first_blend_mode_follows_full_discovery'),
        (6,'first_uniform_mode_trials','first_uniform_mode_follows_full_discovery'),
        (7,'first_probability_mode_trials','first_probability_mode_follows_full_discovery')]:
        flags[flag]=bool(all(np.all(a[field][:,:,other]==n+1) for other in range(8) if other!=mi))
        for ti in range(3):
            for ri in range(2):
                for si in range(len(cfg['model_seeds'])):
                    g=(ti,ri,mi,si);isolated=(ti,ri,1,si)
                    discovery=int(a['first_discovery_trials'][g].max());expected=min(discovery+1,n+1)
                    flags['gated_and_isolated_discovery_trials_equal'] &= np.array_equal(a['first_discovery_trials'][g],a['first_discovery_trials'][isolated])
                    flags[flag] &= a[field][g]==expected
                    event=np.zeros(n,dtype=np.int8)
                    if expected<=n:event[expected-1]=1
                    flags['gate_switch_once_at_first_full_mode'] &= np.array_equal(event,a['trial_gate_switch_events'][g])
                    if discovery<=n:flags['fourth_discovery_update_is_isolated'] &= a['trial_direction_modes'][g+(discovery-1,)]==1
                    prefix=min(discovery,n)
                    for name,value in a.items():
                        if name.startswith('trial_'):left=value[g][:prefix];right=value[isolated][:prefix]
                        elif name.startswith('checkpoint_'):
                            mask=np.asarray(cfg['checkpoints'])<=prefix;left=value[g][mask];right=value[isolated][mask]
                        else:continue
                        flags['exact_isolated_prefix_through_full_discovery'] &= left.dtype==right.dtype and left.shape==right.shape and left.tobytes()==right.tobytes()
                        if mi>=5:
                            reference=value[(ti,ri,4,si)][:prefix] if name.startswith('trial_') else value[(ti,ri,4,si)][mask]
                            flags['exact_gated_balanced_prefix_through_full_discovery'] &= left.tobytes()==reference.tobytes()
        partial=a['trial_direction_modes'][:,:,mi]==1
        flags['partial_gated_has_no_balanced_case'] &= bool(np.all(a['trial_case_indices'][:,:,mi][partial]==-1)
            and np.all(a['trial_objective_case_indices'][:,:,mi][partial]==-1))
        no_go=(a['trial_direction_modes'][:,:,mi]!=1)&(a['trial_actions'][:,:,mi]==0)
        flags['full_NoGo_has_zero_direction_and_no_case'] &= bool(not np.any(a['trial_directions'][:,:,mi][no_go])
            and np.all(a['trial_case_indices'][:,:,mi][no_go]==-1) and np.all(a['trial_objective_case_indices'][:,:,mi][no_go]==-1))
    for key in a['case_keys']:
        if key[3]>=4:flags['only_full_observation_states_in_gated_case_cache'] &= bool(np.all(np.isin(key[5:],[-1,1])))
    for key in a['objective_case_keys']:
        flags['only_full_observation_states_in_gated_case_cache'] &= bool(np.all(np.isin(key[6:],[-1,1])))
    return {name:bool(value) for name,value in flags.items()}


def objective_checks(cfg,a):
    tol=cfg['numeric_tolerance'];modes=a['trial_direction_modes'];actions=a['trial_actions'];active=np.isin(modes,[5,6])&(actions==1)
    flags={'only_actual_full_Go_has_objective_diagnostics':bool(np.array_equal(active,a['trial_objective_applicable'])),
        'inactive_objective_padding_zero_and_sentinel':bool(all(not np.any(a[key][~active]) for key in
            ('trial_objective_logits_before','trial_objective_vectors')) and np.all(a['trial_objective_kind'][~active]==-1)
            and np.all(a['trial_objective_case_indices'][~active]==-1)),
        'current_own_preupdate_state_and_observed_objective_checked':True,
        'uniform_source_cache_probability_each_actual_trial':True,'all_probability_logits_strictly_unclipped':True,
        'all_objective_global_support_certificates_passed':True}
    used=set();certmax=float(a['objective_case_certificates'][:,4:].max(initial=0.))
    for ti,top in enumerate(TOPOLOGIES):
        for ri,rep in enumerate(REPRESENTATIONS):
            for mi in (6,7):
                for si in range(len(cfg['model_seeds'])):
                    at=(ti,ri,mi,si);d=a['g_'+top+'_'+rep+'_unit_rows'][si]
                    heads=np.concatenate((np.zeros((1,d.shape[1])),np.cumsum(a['trial_update_factors'][at][:,None]*a['trial_directions'][at],axis=0)))
                    for trial in np.flatnonzero(active[at]):
                        t=int(trial);ix=at+(t,);src=int(a['schedules'][si,t]);kind=mi-6;z=d@heads[t]
                        A,c=objective_problem(d,src,a['trial_known_before'][ix],z,kind)
                        index=int(a['trial_objective_case_indices'][ix]);key=a['objective_case_keys'][index]
                        address=src if kind==0 else t
                        expected=(si,ti,ri,mi,address,src,*map(int,a['trial_known_before'][ix]))
                        flags['current_own_preupdate_state_and_observed_objective_checked'] &= bool(a['trial_objective_kind'][ix]==kind
                            and np.max(np.abs(z-a['trial_objective_logits_before'][ix]))<=tol
                            and np.max(np.abs(c-a['trial_objective_vectors'][ix]))<=tol
                            and np.max(np.abs(c-a['objective_case_objectives'][index]))<=tol
                            and np.max(np.abs(a['trial_directions'][ix]-a['objective_case_directions'][index]))<=tol)
                        flags['uniform_source_cache_probability_each_actual_trial'] &= np.array_equal(key,expected)
                        if kind==1:flags['all_probability_logits_strictly_unclipped'] &= bool(np.all(np.abs(z)<40.))
                        used.add(index)
    keys=a['objective_case_keys'];ucount=int(np.count_nonzero(keys[:,3]==6));pcount=int(np.count_nonzero(keys[:,3]==7))
    flags['uniform_source_cache_probability_each_actual_trial'] &= bool(pcount==np.count_nonzero(active&(modes==6))
        and ucount<=6*len(cfg['model_seeds'])*4 and len(keys)==len({tuple(k) for k in keys})
        and used==set(range(len(keys))))
    flags['all_objective_global_support_certificates_passed'] &= certmax<=tol
    return {**{key:bool(value) for key,value in flags.items()},'max_certificate_residual':certmax}
