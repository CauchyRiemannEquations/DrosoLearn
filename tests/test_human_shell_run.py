"""Freeze/output guards using only invented files, never actual human choices."""
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('shell_runner',Path(__file__).resolve().parents[1]/'scripts/run_human_shell.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

@pytest.mark.parametrize('value',['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'])
def test_reject_ambiguous_json(tmp_path,value):
    path=tmp_path/'bad.json';path.write_text(value)
    with pytest.raises(ValueError):runner.read(path)

def test_exclusive_json_does_not_replace_bytes(tmp_path):
    path=tmp_path/'record.json';runner.write_new(path,{'before':1});before=path.read_bytes()
    with pytest.raises(FileExistsError):runner.write_new(path,{'after':2})
    assert path.read_bytes()==before

@pytest.mark.parametrize('name',['../outside','C:/outside','/outside','sub\\a'])
def test_frozen_path_is_contained(tmp_path,name):
    with pytest.raises(ValueError):runner.safe(tmp_path,name)

def test_duplicate_inventory_rejected(tmp_path):
    (tmp_path/'a').write_text('a')
    with pytest.raises(ValueError):runner.hashes(tmp_path,['a','a'])

def test_hash_witness_changes_when_bytes_change(tmp_path):
    file=tmp_path/'a';file.write_text('before');before=runner.hashes(tmp_path,['a'])
    file.write_text('after')
    assert runner.hashes(tmp_path,['a'])!=before

def test_csv_schema_cannot_silently_drop_columns(tmp_path):
    with pytest.raises(ValueError):runner.write_csv(tmp_path/'out.csv',[{'a':1},{'b':2}])

def test_csv_empty_table_is_an_error(tmp_path):
    with pytest.raises(ValueError):runner.write_csv(tmp_path/'out.csv',[])

def test_csv_new_file_roundtrip_and_exclusivity(tmp_path):
    file=tmp_path/'out.csv';runner.write_csv(file,[{'action':1,'nll':.4}]);before=file.read_bytes()
    assert b'action,nll' in before
    with pytest.raises(FileExistsError):runner.write_csv(file,[{'action':0,'nll':.9}])
    assert file.read_bytes()==before

def test_scientific_config_is_validated_without_executing_it():
    cfg=runner.read(runner.ROOT/'configs/human_shell_v1.json')
    runner.validate_config(cfg)
    cfg['test_ids'][0]=cfg['calibration_ids'][0]
    with pytest.raises(ValueError):runner.validate_config(cfg)

@pytest.mark.parametrize('key,value',[('equivalence_margin',.5),('prediction_margin',0),('bootstrap_repeats',5),('rates',[.1,1,10]),('test_seeds',[1])])
def test_no_posthoc_budget_or_margin_changes(key,value):
    cfg=runner.read(runner.ROOT/'configs/human_shell_v1.json');cfg[key]=value
    with pytest.raises(ValueError):runner.validate_config(cfg)

def test_three_figures_from_an_invented_verified_summary(tmp_path,monkeypatch):
    import numpy as np
    from drosolearn import human_shell_source
    plot_spec=importlib.util.spec_from_file_location('shell_plot',runner.ROOT/'scripts/plot_human_shell.py')
    plotting=importlib.util.module_from_spec(plot_spec);plot_spec.loader.exec_module(plotting)
    conditions=['fixed','plastic','rewired_fixed','rewired_plastic','ann_fixed','ann_plastic','linear','q','uniform','bias','persistence']
    pids=['A9001','A9002'];records={}
    for pid in pids:
        rows=[]
        for shell in (3,4):
            for state in (1,2,1,2):
                rows.extend({'trial':len(rows)+j+1,'shell':shell,'state':state,'content':0,'action':j%2} for j in range(7))
        records[pid]=rows
    monkeypatch.setattr(human_shell_source,'load_cohort',lambda cache,manifest:(records,{}))
    result=tmp_path/'result';result.mkdir()
    reg={'config':{'conditions':conditions,'test_ids':pids,'manifest':'data/human/tnu_shell_2024/source-manifest.json'},'source_files':{'scripts/plot_human_shell.py':runner.sha(runner.ROOT/'scripts/plot_human_shell.py')}}
    runner.write_new(result/'registration.json',reg)
    runner.write_new(result/'metadata.json',{'status':'completed'})
    runner.write_new(result/'verification.json',{'status':'passed'})
    summary={'behavior_family_comparisons':[{'condition':c,'metrics':[{'difference':0.,'ci90':[-.05,.05]} for _ in range(3)]} for c in conditions],
             'conditional_family_mean_nll':{c:.7+i*.01 for i,c in enumerate(conditions)}}
    runner.write_new(result/'summary.json',summary)
    runner.write_new(result/'artifact_hashes.json',{'summary.json':runner.sha(result/'summary.json')})
    np.savez_compressed(result/'trajectories.npz',actions=np.zeros((2,11,2,2,56),dtype=np.int8))
    out=tmp_path/'figures';plotting.plot(result,out,tmp_path)
    assert {p.name for p in out.glob('*.png')}=={'behavior_comparison.png','heldout_prediction.png','reversal_curves.png'}
    assert all(p.stat().st_size>1000 for p in out.glob('*.png'))
    assert runner.read(out/'provenance.json')['summary_sha256']==runner.sha(result/'summary.json')
    with pytest.raises(ValueError):plotting.plot(result,out,tmp_path)
