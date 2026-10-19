"""Small invented records only; scientific data and seeds are not sampled."""
import copy
from pathlib import Path
import numpy as np
import pytest
from drosolearn import shell_behavior as b
from drosolearn.prospective import read

def invented_rows():
    rows=[]
    for exposure in range(40):
        for shell in range(1,7):
            if shell not in (3,4) and exposure>=34:continue
            phase=exposure//10;kind=b.regime(shell,phase)
            content=0 if exposure%5==0 or kind=='N' else (1 if kind=='R' else 2)
            rows.append({'trial':len(rows)+1,'shell':shell,'state':1+phase%2,'content':content,'action':int(exposure%3!=0)})
    return rows

def test_pairing_keeps_next_same_shell_and_excludes_reversal_boundaries():
    rows=invented_rows();pairs=b.pairing(rows)
    assert len(rows)==216 and len(pairs)==192
    assert all(rows[i]['shell']==rows[j]['shell'] and rows[i]['state']==rows[j]['state'] for i,j,_,_ in pairs)
    assert all(not any(rows[t]['shell']==rows[i]['shell'] for t in range(i+1,j)) for i,j,_,_ in pairs)

def test_incomplete_phase_structure_is_rejected():
    rows=[r for r in invented_rows() if r['state']==1]
    with pytest.raises(ValueError):b.pairing(rows)

def test_nogo_hidden_content_is_not_an_event_classifier():
    rows=invented_rows();ids=['A9001'];actions=np.zeros((1,1,216),dtype=np.int8)
    first=b.transition_counts({'A9001':rows},ids,actions)
    altered=copy.deepcopy(rows)
    for row in altered:row['content']=2-row['content']
    second=b.transition_counts({'A9001':altered},ids,actions)
    assert np.array_equal(first,second)
    assert first[...,0,0].sum()==192 and first[...,1:,0].sum()==0
    assert first[...,0,2].sum()==192 and first[...,0,1].sum()==0

def test_go_events_have_separate_empty_gain_loss_counts():
    rows=invented_rows();actions=np.ones((1,1,216),dtype=np.int8)
    c=b.transition_counts({'A9001':rows},['A9001'],actions)
    assert c[...,0,0].sum()==0 and all(c[...,i,0].sum()>0 for i in (1,2,3))
    assert c[...,0].sum()==c[...,1].sum()==c[...,2].sum()==192

@pytest.mark.parametrize('policy',b.POLICIES)
def test_simulation_is_deterministic_and_nogo_reward_is_zero(policy):
    rows=invented_rows();first=b.simulate(rows,71,'A9001',policy);again=b.simulate(rows,71,'A9001',policy)
    assert all(np.array_equal(a,c) for a,c in zip(first,again))
    assert np.all(first[2][first[1]==0]==0)
    assert np.isfinite(first[0]).all()

@pytest.mark.parametrize('policy',['uniform','persistence'])
def test_nonlearning_choices_ignore_reward_intervention(policy):
    rows=invented_rows();base=b.simulate(rows,71,'A9001',policy);flip=b.simulate(rows,71,'A9001',policy,True)
    assert np.array_equal(base[1],flip[1]) and np.count_nonzero(base[3])==0
    assert np.array_equal(base[4],np.zeros((6,2)))

def test_q_observation_mask_for_forced_nogo(monkeypatch):
    class NoGoRng:
        def random(self,n):return np.ones(n)
    monkeypatch.setattr(b.np.random,'default_rng',lambda seed:NoGoRng())
    values=b.simulate(invented_rows(),71,'A9001','q')
    assert np.all(values[1]==0) and np.all(values[2]==0) and np.all(values[0]==.5)
    assert np.count_nonzero(values[4])==0

def test_reward_learning_can_react_to_sign_intervention_on_invented_records():
    rows=invented_rows();base=b.simulate(rows,71,'A9001','q');flip=b.simulate(rows,71,'A9001','q',True)
    assert np.any(base[1]!=flip[1]) and np.any(base[4]!=flip[4])

def test_fixed_tape_preserves_all_three_metrics_without_online_updates():
    rows=invented_rows();q=b.simulate(rows,71,'A9001','q')[1];tape=q.copy()
    assert np.array_equal(b.metric_vector(rows,q),b.metric_vector(rows,tape))

def test_table_schema_contains_missing_denominator_as_none():
    row=b.table_row(kind='test',denominator=0,value=None)
    assert row['value'] is None and set(b.METRICS)<=set(row)

@pytest.mark.parametrize('key,value',[('synthetic_seeds',[1]),('q_alpha',.2),('q_beta',8),('rng_namespace',1),('trials_per_person',432),('test_ids',['A9001']*29)])
def test_real_scientific_config_cannot_be_retuned(key,value):
    root=Path(__file__).resolve().parents[1];cfg=read(root/'configs/shell_behavior_v1.json');cfg[key]=value
    with pytest.raises(ValueError):b.validate_config(cfg)
