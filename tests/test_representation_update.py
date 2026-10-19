"""Synthetic graph fixtures only: reward API, source-frozen design and full recording."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from drosolearn import representation_update as study
from drosolearn.graph import Projection
from drosolearn.representation_geometry import geometry


def fixture_config():
    cfg = copy.deepcopy(study.SCIENCE)
    cfg.update(scope='integration_test',model_seeds=[90081],training_trials=8,
               checkpoints=[0,4,8],population_pass_count=1,positive_seed_count=1)
    return cfg


def tiny_projection():
    n = 16
    return Projection(np.arange(n),np.arange(n),np.ones(n),n,n,
                      list(map(str,range(n))),list(map(str,range(n))),{})


def test_exact_finite_catalog_has_no_identity_or_norm_loss():
    data = study.catalog()
    assert data['lexicographic_ids'].tolist() == [0,3,1,2]
    assert data['inputs'].shape == (2,4,7)
    assert np.allclose(np.linalg.norm(data['inputs'],axis=-1),np.sqrt(2/3))
    assert len(set(map(tuple,data['inputs'][0]))) == 4
    assert len(set(map(tuple,data['inputs'][1]))) == 4
    assert np.array_equal(data['inputs'][0,0]+data['inputs'][0,1],data['inputs'][0,2]+data['inputs'][0,3])


@pytest.mark.parametrize('field,value', [('training_trials',404),('head_lr',.2),('rng_namespace',2),
                                        ('model_seeds',[17111]),('normalization','old')])
def test_scientific_design_cannot_be_changed_silently(field,value):
    cfg = copy.deepcopy(study.SCIENCE); cfg[field] = value
    with pytest.raises(ValueError): study.validate_config(cfg)


@pytest.mark.parametrize('rule',['shared','projected'])
def test_obtained_reward_only_update_and_own_step_match(rule):
    rows = np.array([[1.,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]])
    g = geometry(rows)
    directions = g['unit_rows'] if rule == 'shared' else g['directions']
    actor = study.RewardHead(g['unit_rows'],directions,.1)
    before = actor.features.copy()
    step, own, other = actor.observe(0,1,1)
    assert step > 0 and own == pytest.approx(.05)
    assert actor.calls == 1
    if rule == 'projected': assert other < 1e-12
    else: assert other > .01
    assert actor.observe(1,0,0) == (0.,0.,0.)
    assert actor.calls == 2 and np.array_equal(before,actor.features)
    with pytest.raises(ValueError): actor.observe(1,0,-1)
    with pytest.raises(ValueError): actor.observe(4,1,1)


def test_numeric_optional_schema_stable_and_json_summary_ready():
    a = study.numeric_fields({'head_l2':None,'nested':{'full_row_condition':2.}})
    assert not a['head_l2_defined'] and a['head_l2'] == 0
    assert a['nested_full_row_condition_defined']


@pytest.fixture
def produced(monkeypatch,tmp_path):
    cfg = fixture_config()
    monkeypatch.setattr(study,'load_projection',lambda unused: tiny_projection())
    monkeypatch.setattr(study,'build_graphs',lambda graph,seed:{key:graph.copy() for key in study.TOPOLOGIES})
    folder = tmp_path / cfg['input_catalog']; folder.parent.mkdir(parents=True)
    folder.write_text(json.dumps({'study_id':cfg['study_id'],'representations':list(study.REPRESENTATIONS),
                     'catalog':{k:v.tolist() for k,v in study.catalog().items()}}),encoding='utf8')
    arrays,rows = study.produce(cfg,tmp_path)
    return cfg,arrays,rows,tmp_path


def test_end_to_end_invented_graph_records_all_choices_and_fixed_witnesses(produced):
    cfg,a,rows,root = produced
    assert len(rows) == 96 and a['checkpoint_probabilities'].shape == (3,2,2,1,3,4)
    assert np.all(a['checkpoint_probabilities'][...,0,:] == .5)
    assert all(np.isfinite(v).all() and v.dtype.kind != 'O' for v in a.values())
    assert not np.count_nonzero(a['trial_internal_steps'])
    assert not np.count_nonzero(a['trial_value_update_calls'])
    for topology in study.TOPOLOGIES:
        for rep in study.REPRESENTATIONS:
            for key in ('weights','encoder','bias','probe_kc','probe_mbon'):
                assert np.array_equal(a['w_'+topology+'_'+rep+'_'+key],a['finalfixed_'+topology+'_'+rep+'_'+key])
    summary = study.summarize(cfg,a,rows)
    json.dumps(summary,allow_nan=False)
    assert summary['counts']['training_feedback_trials'] == 96
    assert summary['counts']['static_probe_cases'] == 144
    assert summary['counts']['internal_update_calls'] == 0
    assert summary['analytic_input_margin_feasible'] == {'raw':False,'identity':True}


def test_no_silent_trial_loss_and_probability_corruption(produced):
    cfg,a,rows,root = produced
    with pytest.raises(ValueError): study.summarize(cfg,a,rows[:-1])
    b = dict(a); b['checkpoint_probabilities'] = a['checkpoint_probabilities'].copy()
    b['checkpoint_probabilities'].flat[0] = float('nan')
    with pytest.raises(ValueError): study.summarize(cfg,b,rows)


def test_input_tampering_rejected(produced):
    cfg,a,rows,root = produced
    path = root / cfg['input_catalog']; content = json.loads(path.read_text())
    content['catalog']['targets'][0] = 0
    path.write_text(json.dumps(content),encoding='utf8')
    with pytest.raises(ValueError): study.load_catalog(cfg,root)
