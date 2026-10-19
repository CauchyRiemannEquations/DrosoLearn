"""Invented saved-state fixtures, never the frozen research arrays."""
from copy import deepcopy
import numpy as np
import pytest
from drosolearn.signed_spillover import SCIENCE, analyze_saved, summarize, sigmoid, validate_config, mass


def invented_parent():
    cfg=deepcopy(SCIENCE)
    cfg.update(scope='integration_test',model_seeds=[90001],training_trials=8,checkpoints=[0,4,8])
    shape=(3,2,5,1,8);y=np.array([1,1,-1,-1]);source=np.tile(np.arange(4),2)
    parent={'model_seeds':np.array([90001]),'alphas':np.array(cfg['alphas']),
        'checkpoints':np.array(cfg['checkpoints']),'targets':(y>0).astype(int),'schedules':source[None]}
    for name in ['probabilities','actions','rewards','update_factors','own_score_steps','other_score_steps','head_steps',
                 'internal_update_calls','value_update_calls','internal_steps','value_steps']:
        parent['trial_'+name]=np.zeros(shape)
    parent['checkpoint_logits']=np.empty((3,2,5,1,3,4))
    parent['checkpoint_probabilities']=np.empty((3,2,5,1,3,4))
    parent['checkpoint_heads']=np.empty((3,2,5,1,3,6))
    rows=[]
    for ti,top in enumerate(cfg['topologies']):
        for ri,rep in enumerate(cfg['representations']):
            raw=np.array([[1,0,0,0,.3,.1],[0,1,0,0,-.2,.4],[0,0,1,0,.4,.3],[0,0,0,1,.1,-.2]],float)
            raw[:,-2:]*=1+ti*.1+ri*.05
            d=raw/np.linalg.norm(raw,axis=1)[:,None]
            u=np.linalg.pinv(d).T;v=u/np.linalg.norm(u,axis=1)[:,None]
            w=np.stack([(1-a)*d+a*v for a in cfg['alphas']]);w/=np.linalg.norm(w,axis=-1)[...,None]
            prefix='t_'+top+'_'+rep+'_'
            parent[prefix+'base_budget_base_geometry_unit_rows']=d[None]
            parent[prefix+'directions']=w[None]
            parent[prefix+'influences']=(w@d.T)[None]
            for ai,alpha in enumerate(cfg['alphas']):
                head=np.zeros(6);ki=0;at=(ti,ri,ai,0)
                for t in range(9):
                    if t in cfg['checkpoints']:
                        parent['checkpoint_heads'][at+(ki,)]=head
                        parent['checkpoint_logits'][at+(ki,)]=d@head
                        parent['checkpoint_probabilities'][at+(ki,)]=sigmoid(d@head);ki+=1
                    if t==8:break
                    i=int(source[t]);p=float(sigmoid(d@head)[i]);a=int(t not in (2,5));reward=a*int(y[i])
                    g=cfg['head_lr']*(a-p)*reward;step=g*w[ai,i];scores=d@step;head=head+step
                    for name,value in [('probabilities',p),('actions',a),('rewards',reward),('update_factors',g),
                                       ('own_score_steps',scores[i]),('other_score_steps',np.linalg.norm(np.delete(scores,i))),
                                       ('head_steps',np.linalg.norm(step))]:
                        parent['trial_'+name][at+(t,)]=value
                    rows.append({'topology':top,'representation':rep,'alpha':alpha,'seed':90001,'trial':t+1,
                        'example_index':i,'target':int(y[i]>0),'probability_go':p,'action':a,'reward':reward,'update_factor':g})
    return cfg,parent,rows


def test_full_fixture_decomposition_and_expected_effects():
    cfg,parent,rows=invented_parent();arrays,out=analyze_saved(cfg,parent,rows)
    result=summarize(cfg,arrays,out)
    assert len(arrays)==8 and len(out)==150 and len(result['families'])==30
    assert len(result['checkpoint_expectations'])==90
    assert result['counts']['analyzed_parent_trial_records']==240
    assert result['counts']['new_training_trials']==result['counts']['new_random_draws']==0
    assert result['validation']['all_reconstruction_and_telescoping_checks_passed']
    assert arrays['trial_margin_delta'].shape==(3,2,5,1,8,4)
    cross=[x for x in out if x['category']=='cross_all']
    assert any(x['margin_harmful_sum']>0 for x in cross)
    for family in result['families']:
        categories=family['categories']
        for metric in ['margin_net_sum','probability_net_sum','actual_cells']:
            assert categories['cross_all'][metric]['mean']==pytest.approx(
                categories['cross_same'][metric]['mean']+categories['cross_opposite'][metric]['mean'])
            assert categories['all'][metric]['mean']==pytest.approx(
                categories['self'][metric]['mean']+categories['cross_all'][metric]['mean'])
        gain=categories['all']['probability_net_sum']['mean']/4
        assert gain==pytest.approx(family['final_expected_accuracy']-.5,abs=1e-14)


@pytest.mark.parametrize('name',['trial_probabilities','trial_update_factors','trial_own_score_steps',
    'trial_other_score_steps','trial_head_steps','checkpoint_logits','checkpoint_probabilities','checkpoint_heads'])
def test_changed_saved_values_fail(name):
    cfg,parent,rows=invented_parent();parent[name].flat[-1]+=.01
    with pytest.raises(ValueError):analyze_saved(cfg,parent,rows)


@pytest.mark.parametrize('name',['trial_internal_update_calls','trial_value_update_calls','trial_internal_steps','trial_value_steps'])
def test_internal_or_value_changes_are_rejected(name):
    cfg,parent,rows=invented_parent();parent[name].flat[0]=1
    with pytest.raises(ValueError,match='internal/value'):analyze_saved(cfg,parent,rows)


def test_duplicate_saved_csv_record_rejected():
    cfg,parent,rows=invented_parent();rows[-1]=rows[0]
    with pytest.raises(ValueError,match='Unique'):analyze_saved(cfg,parent,rows)


def test_zero_nogo_reward_rule_required():
    cfg,parent,rows=invented_parent();parent['trial_rewards'].flat[2]=1
    rows[2]['reward']=1
    with pytest.raises(ValueError,match='Obtained reward'):analyze_saved(cfg,parent,rows)


def test_static_influence_transpose_tamper_fails():
    cfg,parent,rows=invented_parent()
    parent['t_real_raw_influences'][0,0,0,1]+=.02
    with pytest.raises(ValueError,match='arithmetic'):analyze_saved(cfg,parent,rows)


def test_balanced_schedule_required():
    cfg,parent,rows=invented_parent();parent['schedules'][0,0]=1
    with pytest.raises(ValueError,match='Balanced'):analyze_saved(cfg,parent,rows)


def test_counts_and_exact_masses_are_separate():
    result=mass(np.array([1e-12,-2e-12,0]),1e-8)
    assert result['neutral_cells']==3
    assert result['helpful']==1e-12 and result['harmful']==2e-12 and result['net']==-1e-12


def test_scientific_configuration_cannot_change_design():
    for key,value in [('neutral_tolerance',.1),('labels',[1,-1,1,-1]),('alphas',[0,1]),('head_lr',.2),('model_seeds',[17302])]:
        cfg=deepcopy(SCIENCE);cfg[key]=value
        with pytest.raises(ValueError):validate_config(cfg)


def test_fixture_cannot_use_research_seeds():
    cfg=deepcopy(SCIENCE);cfg['scope']='integration_test'
    with pytest.raises(ValueError):validate_config(cfg)


def test_summary_rejects_missing_or_duplicate_measurements():
    cfg,parent,rows=invented_parent();arrays,out=analyze_saved(cfg,parent,rows)
    for wrong in [out[:-1],out[:-1]+[out[0]]]:
        with pytest.raises(ValueError,match='measurement'):summarize(cfg,arrays,wrong)


def test_summary_rejects_broken_signed_mass_identity():
    cfg,parent,rows=invented_parent();arrays,out=analyze_saved(cfg,parent,rows)
    out[0]['probability_helpful_sum']+=.1
    with pytest.raises(ValueError,match='validation'):summarize(cfg,arrays,out)


def test_summary_rejects_broken_category_partition_even_if_row_mass_balances():
    cfg,parent,rows=invented_parent();arrays,out=analyze_saved(cfg,parent,rows)
    row=next(x for x in out if x['category']=='all')
    row['probability_helpful_sum']+=.01;row['probability_net_sum']+=.01
    with pytest.raises(ValueError,match='validation'):summarize(cfg,arrays,out)
