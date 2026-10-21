"""Invented saved records only; never open actual parent arrays in development."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from drosolearn import objective_pathways as m


def fixture():
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/objective_pathways_v1.json').read_text(encoding='utf8'))
    cfg.update(scope='integration_test',model_seeds=[90001,90002],training_trials=16)
    S=2;N=16;F=(3,2,8,S,N);d=np.eye(4)+.1
    d/=np.linalg.norm(d,axis=1,keepdims=True);y=np.array([1.,1.,-1.,-1.])
    p={'model_seeds':np.array(cfg['model_seeds']),'targets':(y>0).astype(int),
        'schedules':np.tile(np.arange(N)%4,(S,1)),'uniforms':np.tile(np.array([.01]*4+[.49,.51,.55,.45]*3),(S,1)),
        'checkpoints':np.array([0,4,N]),'trial_actions':np.zeros(F,dtype=np.int8),'trial_rewards':np.zeros(F,dtype=np.int8),
        'trial_probabilities':np.zeros(F),'trial_update_factors':np.zeros(F),'trial_directions':np.zeros(F+(4,)),
        'trial_head_steps':np.zeros(F),'trial_cumulative_head_path_l2':np.zeros(F),'trial_cumulative_head_squared_l2':np.zeros(F),
        'trial_head_step_squared_l2':np.zeros(F),
        'trial_unknown_other_mask_before':np.zeros(F+(4,),dtype=bool),'trial_unknown_other_mask_after':np.zeros(F+(4,),dtype=bool),
        'trial_known_before':np.zeros(F+(4,),dtype=np.int8),'trial_known_after':np.zeros(F+(4,),dtype=np.int8),
        'trial_correct_probability_delta':np.zeros(F+(4,)),'trial_aligned_margin_delta':np.zeros(F+(4,)),
        'trial_unknown_other_negative_correct_probability_sum':np.zeros(F),
        'checkpoint_heads':np.zeros(F[:-1]+(3,4)),'checkpoint_probabilities':np.full(F[:-1]+(3,4),.5)}
    for key in ('balanced','blend','uniform','probability'):p['first_'+key+'_mode_trials']=np.full(F[:-1],N+1,dtype=np.int32)
    inverse=np.linalg.inv(d)
    for ti,top in enumerate(m.TOPS):
        for ri,rep in enumerate(m.REPS):
            p['g_'+top+'_'+rep+'_unit_rows']=np.tile(d,(S,1,1))
            for mi in range(8):
                for si in range(S):
                    at=(ti,ri,mi,si);h=np.zeros(4);known=np.zeros(4,dtype=np.int8)
                    for t,src in enumerate(p['schedules'][si]):
                        ix=at+(t,);z=d@h;prob=m.sigmoid(z[src]);a=int(p['uniforms'][si,t]<prob);r=int(a*y[src])
                        p['trial_known_before'][ix]=known
                        isolated=inverse[:,src];isolated/=np.linalg.norm(isolated)
                        w=isolated.copy()
                        if np.all(known) and mi>=4:
                            global_direction=y[src]*(y@d);global_direction/=np.linalg.norm(global_direction)
                            alpha={4:.4,5:.2,6:.8,7:.9}[mi]
                            w=(1-alpha)*isolated+alpha*global_direction
                            if mi==7:w+=.03*np.tanh(z)
                            w/=np.linalg.norm(w)
                        if not a:w*=0
                        g=.1*(a-prob)*r;after=h+g*w;change=y*(m.sigmoid(d@after)-m.sigmoid(z))
                        for key,val in [('actions',a),('rewards',r),('probabilities',prob),('update_factors',g),
                            ('directions',w),('correct_probability_delta',change),('aligned_margin_delta',y*(d@(g*w)))]:
                            p['trial_'+key][ix]=val
                        unknown=(known==0)&(np.arange(4)!=src)
                        p['trial_unknown_other_mask_before'][ix]=unknown
                        p['trial_unknown_other_negative_correct_probability_sum'][ix]=(np.maximum(-change,0)*unknown).sum()
                        if a:known[src]=r
                        p['trial_unknown_other_mask_after'][ix]=(known==0)&(np.arange(4)!=src)
                        p['trial_known_after'][ix]=known;h=after
                        if t+1 in p['checkpoints']:
                            k=int(np.flatnonzero(p['checkpoints']==t+1)[0]);p['checkpoint_heads'][at+(k,)]=h
                            p['checkpoint_probabilities'][at+(k,)]=m.sigmoid(d@h)
                    step=np.abs(p['trial_update_factors'][at])*np.linalg.norm(p['trial_directions'][at],axis=-1)
                    p['trial_head_steps'][at]=step;p['trial_cumulative_head_path_l2'][at]=np.cumsum(step)
                    p['trial_cumulative_head_squared_l2'][at]=np.cumsum(step**2)
                    p['trial_head_step_squared_l2'][at]=step**2
                    for method,key in ((4,'balanced'),(5,'blend'),(6,'uniform'),(7,'probability')):
                        if mi==method:p['first_'+key+'_mode_trials'][at]=5
    return cfg,p


def test_complete_fixture_accounting():
    cfg,p=fixture();a,rows=m.analyze(cfg,p);s=m.summarize_arrays(cfg,a,rows,p)
    assert len(a)==16 and len(rows)==432
    assert s['counts']['analyzed_parent_trial_records']==1536
    assert s['counts']['actual_path_strata']==6912
    assert len(s['families'])==18 and len(s['actual_paths'])==48
    assert all(x<=1e-8 for x in s['validation']['residuals'].values())


@pytest.mark.parametrize('bit',[1,2,4])
def test_shapley_additive_bits(bit):
    cells=np.array([[float(bool(i&bit))]*4 for i in range(8)])
    got=m.shapley3(cells);assert np.allclose(got.sum(0),1.)
    assert np.allclose(got[{4:0,2:1,1:2}[bit]],1.)


@pytest.mark.parametrize('kind',['directions','update_factors','probabilities','correct_probability_delta',
    'aligned_margin_delta','unknown_other_negative_correct_probability_sum'])
def test_factual_tamper_rejected(kind):
    cfg,p=fixture();p['trial_'+kind].flat[0]+=.2
    with pytest.raises(ValueError):m.analyze(cfg,p)


@pytest.mark.parametrize('kind',['actions','rewards','known_before','known_after'])
def test_discrete_parent_tamper_rejected(kind):
    cfg,p=fixture();p['trial_'+kind].flat[0]=0 if kind in ('actions','rewards') else -1
    with pytest.raises(ValueError):m.analyze(cfg,p)


@pytest.mark.parametrize('key',['comparisons','methods','topologies','representations','destination_groups','bridge_bits'])
def test_configuration_axes_are_fixed(key):
    cfg,_=fixture();cfg[key]=list(reversed(cfg[key]))
    with pytest.raises(ValueError):m.validate_config(cfg)


def test_parent_not_mutated():
    cfg,p=fixture();before={k:v.copy() for k,v in p.items()};a,rows=m.analyze(cfg,p);m.summarize_arrays(cfg,a,rows,p)
    assert all(np.array_equal(v,p[k]) for k,v in before.items())


def test_no_new_random_solver_or_learning_import():
    source=Path(m.__file__).read_text(encoding='utf8')
    assert all(token not in source for token in ('np.random','scipy.optimize','SelfBlendHead','FlyWire','run_learning('))


def test_empty_crossing_mean_is_null():
    assert m.mean_nested([None,None]) is None
    assert m.mean_nested([[None,1.],[None,3.]])==[None,2.]


def test_signed_positive_negative_allocation():
    x=np.arange(32,dtype=float).reshape(8,4)/100-.1
    assert np.allclose(m.shapley3(x),m.shapley3(np.maximum(x,0))-m.shapley3(np.maximum(-x,0)))


def test_one_sided_cases_preserved():
    cfg,p=fixture();a,rows=m.analyze(cfg,p)
    assert np.any(a['pair_action_cases']==1) or np.any(a['pair_action_cases']==2)
    assert np.all(a['bridge_q_cells'][~a['bridge_eligible']]==0)
    assert np.all(a['factual_pair_diagnostics'][~a['bridge_eligible']]==0)


def test_strata_include_empty_impossible_combinations():
    cfg,p=fixture();a,rows=m.analyze(cfg,p);s=m.summarize_arrays(cfg,a,rows,p)
    record=s['actual_paths'][0]['seeds'][0]
    assert len(record['source_action_reward_strata'])==72
    assert any(x['trials']==0 and x['source_probability_mean'] is None for x in record['source_action_reward_strata'])
    assert sum(x['trials'] for x in record['source_action_reward_strata'] if x['phase']=='all')==16


def test_target_nan_rejected():
    cfg,p=fixture();p['targets']=np.array([1.,1.,np.nan,0.])
    with pytest.raises(ValueError):m.analyze(cfg,p)
