"""New cone inventory/count/failure checks; inherited byte gates stay unchanged."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from drosolearn import cone_envelope as env


def configuration():
    return json.loads((Path(__file__).resolve().parents[1]/env.CONFIG).read_text(encoding='utf-8'))


def summary():
    return {'study_id':env.STUDY_ID,'scope':'retrospective_descriptive_cone_frontier','status':'completed','counts':{
        'analyzed_parent_trial_records':24000,'candidate_trial_records':168000,'candidate_destination_effects':672000,
        'measurement_records':1260,'conditions':6,'model_seeds':10,'nonzero_parent_updates':12000,'unique_optimization_cases':1000,
        'new_training_trials':0,'new_head_update_calls':0,'new_internal_update_calls':0,'new_value_update_calls':0,
        'new_random_draws':0,'fresh_model_constructions':0}}


def test_exact_new_closure():
    cfg=configuration();root=Path(__file__).resolve().parents[1]
    env.validate_envelope(root,cfg,root/env.CONFIG)
    assert len(env.SOURCE_FILES)==16 and len(env.PARENT_FILES)==11
    assert 'src/drosolearn/cone_geometry.py' in env.SOURCE_FILES
    assert 'results/selective_spillover_v1/arrays.npz' in env.PARENT_FILES


@pytest.mark.parametrize('group',['source_files','parent_files'])
def test_omitted_or_duplicate_frozen_file_fails(group):
    cfg=configuration();cfg[group]=cfg[group][1:]+[cfg[group][1]]
    root=Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError):env.validate_envelope(root,cfg,root/env.CONFIG)


@pytest.mark.parametrize('field',['new_training','new_seed_draws','new_effect_hypothesis_test','shared_neural_mechanism_claim'])
def test_descriptive_scope_cannot_be_silently_upgraded(field):
    cfg=configuration();cfg['interpretation'][field]=True;root=Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError):env.validate_envelope(root,cfg,root/env.CONFIG)


def test_new_work_counts_are_six_feedback_paths_not_all_parent_paths():
    env.validate_counts(configuration(),summary(),1260)


def test_parent_result_directory_is_passed_as_directory(tmp_path,monkeypatch):
    from drosolearn import normalization_envelope as prior
    for name in ['configs/normalization_loss_v1.json','results/registration_normalization_loss_v1.json',
                 'results/normalization_loss_publication_proof.json']:
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_text('{}')
    folder=tmp_path/'results/normalization_loss_v1';folder.mkdir()
    monkeypatch.setattr(env,'git',lambda *args:env.BASE_COMMIT.encode())
    monkeypatch.setattr(env,'ancestor',lambda *args:None)
    def capture(root,config,registration,result,proof):
        assert result==folder and result.is_dir()
        raise RuntimeError('directory reached inherited validator')
    monkeypatch.setattr(prior,'verify_envelope',capture)
    with pytest.raises(RuntimeError,match='directory reached'):env.parent_gate(tmp_path)


@pytest.mark.parametrize('key,value',[('analyzed_parent_trial_records',96000),('new_training_trials',1),
    ('new_random_draws',False),('unique_optimization_cases',0),('unique_optimization_cases',999999),
    ('nonzero_parent_updates',24001),('conditions',24),('candidate_destination_effects',168000)])
def test_counts_reject_scope_expansion_learning_boolean_or_unbounded_cache(key,value):
    s=summary();s['counts'][key]=value
    with pytest.raises(ValueError):env.validate_counts(configuration(),s,1260)


def test_count_receipt_requires_exact_row_closure():
    with pytest.raises(ValueError):env.validate_counts(configuration(),summary(),1259)


def test_run_failure_is_exclusive_and_keeps_failure_metadata(tmp_path,monkeypatch):
    root=tmp_path;out=root/env.RESULT;out.parent.mkdir();reg=root/'reg.json';reg.write_text('{}')
    proof=root/'proof.json';proof.write_text('{}');config=root/'cfg.json';config.write_text('{}')
    cfg=configuration();registration={'source_git_commit':'0'*40}
    monkeypatch.setattr(env,'validate_registration',lambda *args:(cfg,registration))
    monkeypatch.setattr(env,'publication',lambda *args:('proof.json',{'checked_at_utc':'2020-01-01T00:00:00+00:00'}))
    def fail(*args):raise RuntimeError('invented pre-analysis failure')
    monkeypatch.setattr(env.importlib,'import_module',lambda *args:SimpleNamespace(validate_config=lambda *args:None,produce=fail))
    with pytest.raises(RuntimeError):env.run(root,config,reg,proof,out)
    meta=json.loads((out/'metadata.json').read_text());assert meta['status']=='failed' and meta['new_training_calls']==0
    assert (out/'registration.json').read_bytes()==reg.read_bytes()
    assert 'metadata.json' in json.loads((out/'artifact_hashes.json').read_text())
    with pytest.raises(ValueError):env.run(root,config,reg,proof,out)
