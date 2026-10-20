"""Targeted gates for a new actual learner, distinct from its saved-path parent."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drosolearn import self_blend_envelope as env


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {'study_id': env.STUDY_ID, 'module': env.MODULE, 'protocol': env.PROTOCOL,
            'scope': 'prospective_observed_gated_self_blend_learning', 'external_manifest': None,
            'model_seeds': list(range(17701, 17711)), 'training_trials': 400,
            'checkpoints': [0, 50, 100, 200, 400],
            **{group: sorted(values) for group, values in (
                ('source_files', env.SOURCE_FILES), ('parent_files', env.PARENT_FILES),
                ('graph_files', env.GRAPH_FILES), ('input_files', env.INPUT_FILES))},
            'interpretation': {'new_human_experiment': False, 'shared_neural_mechanism_claim': False,
                               'education_effect_claim': False}}


def summary():
    return {'study_id': env.STUDY_ID, 'status': 'completed', 'counts': {
        'training_feedback_trials': 144000, 'static_probe_cases': 7200,
        'model_initializations': 360, 'fixed_feature_constructions': 60,
        'model_seeds': 10, 'head_update_calls': 144000, 'head_nonzero_steps': 47000,
        'internal_update_calls': 0, 'value_update_calls': 0,
        'internal_nonzero_steps': 0, 'value_nonzero_steps': 0}}


def test_exact_actual_science_and_historical_closure():
    cfg = configuration()
    env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)
    assert len(env.SOURCE_FILES) == 44 and len(env.PARENT_FILES) == 11
    assert env.NUMERIC_ARRAY_COUNT == 201
    assert len(env.GRAPH_FILES) == 4 and len(env.INPUT_FILES) == 2
    assert {'scripts/verify_selective_spillover.py', 'scripts/verify_cone_frontier.py',
            'src/drosolearn/cone_geometry.py'} <= env.SOURCE_FILES
    assert {'results/gated_factors_v1/arrays.npz',
            'results/gated_factors_v1/measurements.csv'} <= env.PARENT_FILES


@pytest.mark.parametrize('group', ['source_files', 'parent_files', 'graph_files', 'input_files'])
@pytest.mark.parametrize('mutation', ['omit', 'duplicate', 'extra'])
def test_inventory_cannot_omit_duplicate_or_expand(group, mutation):
    cfg = configuration()
    if mutation == 'omit':
        cfg[group] = cfg[group][1:]
    elif mutation == 'duplicate':
        cfg[group][-1] = cfg[group][0]
    else:
        cfg[group].append('undeclared.data')
    with pytest.raises(ValueError):
        env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


@pytest.mark.parametrize('scope', ['integration_test', 'retrospective_descriptive_cone_frontier', None])
def test_fixture_or_prior_readonly_scope_cannot_be_registered(scope):
    cfg = configuration(); cfg['scope'] = scope
    with pytest.raises(ValueError):
        env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


@pytest.mark.parametrize('field', ['new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim'])
def test_interpretation_is_bounded(field):
    cfg = configuration(); cfg['interpretation'][field] = True
    with pytest.raises(ValueError):
        env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


def test_real_training_count_contract():
    env.validate_counts(configuration(), summary(), 144000)


@pytest.mark.parametrize('field,value', [
    ('training_feedback_trials', 0), ('static_probe_cases', 4000), ('model_initializations', 60),
    ('fixed_feature_constructions', 240), ('head_update_calls', 0), ('model_seeds', True),
    ('internal_update_calls', 1), ('internal_nonzero_steps', False), ('value_update_calls', 1),
    ('value_nonzero_steps', 1), ('head_nonzero_steps', 0), ('head_nonzero_steps', 144001),
    ('head_nonzero_steps', True),
])
def test_actual_work_cannot_be_relabelled_zero_learning_or_boolean(field, value):
    value_summary = summary(); value_summary['counts'][field] = value
    with pytest.raises(ValueError):
        env.validate_counts(configuration(), value_summary, 144000)


def test_no_budget_extension_or_incomplete_rows():
    cfg = configuration(); cfg['training_trials'] = 800
    with pytest.raises(ValueError):
        env.validate_counts(cfg, summary(), 144000)
    with pytest.raises(ValueError):
        env.validate_counts(configuration(), summary(), 143999)


def test_parent_gate_starts_with_latest_inherited_byte_gate_before_any_science(tmp_path, monkeypatch):
    from drosolearn import gated_factors_envelope as prior
    def stop(root):
        assert root == tmp_path
        raise RuntimeError('inherited saved-byte gate')
    monkeypatch.setattr(prior, 'parent_gate', stop)
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('parent scientific import'))
    with pytest.raises(RuntimeError, match='inherited saved-byte gate'): env.parent_gate(tmp_path)


def test_no_science_import_before_registration_gate(tmp_path, monkeypatch):
    out = tmp_path / env.RESULT
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (_ for _ in ()).throw(ValueError('freeze missing')))
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('premature science import'))
    with pytest.raises(ValueError, match='freeze missing'):
        env.run(tmp_path, tmp_path / 'c', tmp_path / 'r', tmp_path / 'p', out)
    assert not out.exists()


def test_no_science_import_before_remote_publication_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), {}))
    monkeypatch.setattr(env, 'publication', lambda *args: (_ for _ in ()).throw(ValueError('remote bytes missing')))
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('premature science import'))
    out = tmp_path / env.RESULT
    with pytest.raises(ValueError, match='remote bytes missing'):
        env.run(tmp_path, tmp_path / 'c', tmp_path / 'r', tmp_path / 'p', out)
    assert not out.exists()


def test_failed_actual_run_is_exclusive_and_keeps_failure_metadata(tmp_path, monkeypatch):
    out = tmp_path / env.RESULT; out.parent.mkdir()
    registration = tmp_path / 'reg.json'; registration.write_text('{}')
    config = tmp_path / 'cfg.json'; config.write_text('{}')
    proof = tmp_path / 'proof.json'; proof.write_text('{}')
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), {'source_git_commit': '0' * 40}))
    monkeypatch.setattr(env, 'publication', lambda *args: ('proof.json', {'checked_at_utc': '2020-01-01T00:00:00+00:00'}))
    def fail(*args):
        raise RuntimeError('invented actual-learning failure')
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: SimpleNamespace(validate_config=lambda *args: None, produce=fail))
    with pytest.raises(RuntimeError, match='invented actual-learning failure'):
        env.run(tmp_path, config, registration, proof, out)
    metadata = json.loads((out / 'metadata.json').read_text())
    assert metadata['status'] == 'failed' and metadata['analysis_only'] is False
    assert metadata['stage'] == 'actual_learning'
    assert 'counts' not in metadata
    assert (out / 'registration.json').read_bytes() == registration.read_bytes()
    assert 'metadata.json' in json.loads((out / 'artifact_hashes.json').read_text())
    with pytest.raises(ValueError, match='New fixed result directory'):
        env.run(tmp_path, config, registration, proof, out)


def proof_fixture(tmp_path, monkeypatch):
    registration = tmp_path / env.REGISTRATION
    registration.parent.mkdir(); registration.write_bytes(b'{"frozen":true}\n')
    proof = tmp_path / 'proof.json'
    reg = {'source_git_commit': 'a' * 40, 'registered_at_utc': '2020-01-01T00:00:00+00:00'}
    value = {'study_id': env.STUDY_ID, 'registration_sha256': env.sha(registration),
             'source_commit': reg['source_git_commit'], 'remote_bytes_match': True,
             'visibility': 'private', 'published_commit': 'b' * 40,
             'url': env.REPOSITORY + '/blob/' + 'b' * 40 + '/' + env.REGISTRATION,
             'checked_at_utc': '2020-01-02T00:00:00+00:00'}
    proof.write_text(json.dumps(value))
    monkeypatch.setattr(env, 'ancestor', lambda *args: None)
    monkeypatch.setattr(env, 'git', lambda *args: registration.read_bytes())
    return registration, reg, proof, value


def test_publication_binds_exact_private_bytes_commit_url_and_prior_time(tmp_path, monkeypatch):
    registration, reg, proof, value = proof_fixture(tmp_path, monkeypatch)
    assert env.publication(tmp_path, registration, reg, proof)[1] == value


@pytest.mark.parametrize('field,value', [
    ('remote_bytes_match', False), ('visibility', 'public'), ('registration_sha256', '0' * 64),
    ('source_commit', 'c' * 40), ('published_commit', 'not-a-commit'),
    ('url', 'https://example.test/not-registered-bytes'),
    ('checked_at_utc', '2019-12-31T00:00:00+00:00'),
])
def test_bad_publication_receipt_cannot_start_training(tmp_path, monkeypatch, field, value):
    registration, reg, proof, data = proof_fixture(tmp_path, monkeypatch)
    data[field] = value; proof.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        env.publication(tmp_path, registration, reg, proof)


def test_local_published_commit_bytes_must_equal_registration(tmp_path, monkeypatch):
    registration, reg, proof, _ = proof_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(env, 'git', lambda *args: b'changed')
    with pytest.raises(ValueError, match='registration bytes differ'):
        env.publication(tmp_path, registration, reg, proof)


def test_verify_rejects_readonly_metadata_before_raw_array_loading(tmp_path, monkeypatch):
    out = tmp_path / env.RESULT; out.mkdir(parents=True)
    registration = tmp_path / 'reg.json'; registration.write_text('{}')
    config = tmp_path / 'cfg.json'; config.write_text('{}')
    proof = tmp_path / 'proof.json'; proof.write_text('{}')
    (out / 'registration.json').write_bytes(registration.read_bytes())
    reg = {'source_git_commit': 'a' * 40, 'registered_at_utc': '2020-01-01T00:00:00+00:00'}
    metadata = {'study_id': env.STUDY_ID, 'status': 'completed', 'stage': 'completed',
                'source_git_commit': reg['source_git_commit'], 'registration_sha256': env.sha(registration),
                'config_sha256': env.sha(config), 'publication_proof': 'proof.json',
                'publication_proof_sha256': env.sha(proof), 'analysis_only': True,
                'new_human_experiment': False, 'shared_neural_mechanism_tested': False,
                'education_effect_tested': False}
    (out / 'metadata.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), reg))
    monkeypatch.setattr(env, 'publication', lambda *args: ('proof.json', {'checked_at_utc': '2020-01-02T00:00:00+00:00'}))
    with pytest.raises(ValueError, match='actual-learning identity'):
        env.verify_envelope(tmp_path, config, registration, out, proof)


def factor_prefreeze_fixture(tmp_path, monkeypatch):
    """Manufacture archival receipts and XML; no actual parent arrays are read."""
    def put(name, value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value) if not isinstance(value,str) else value)
        return path
    times=['2020-01-01T00:0'+str(i)+':00+00:00' for i in range(9)]
    baseline='results/gated_factors_baseline_preservation.json'
    validation='results/gated_factors_prefreeze_validation.json'
    exception='results/gated_factors_prefreeze_evidence_exception.json'
    junit='results/gated_factors_prefreeze_tests/gated_factors_prefreeze_final.xml'
    failures=['results/gated_factors_development/gated_factors_verify_development_failure_01.xml',
        'results/gated_factors_development/gated_factors_verify_failure_01.py',
        'results/gated_factors_development/gated_factors_verify_tests_failure_01.py']
    classes=[('tests.test_gated_factors',45),('tests.test_gated_factors_envelope',48),
        ('tests.test_gated_factors_verify',65),('tests.test_balanced_pathways',67),('tests.test_gated_pathways',44)]
    xml='<testsuites><testsuite>'+''.join('<testcase classname="'+cls+'" name="case'+str(i)+'"/>'
        for cls,count in classes for i in range(count))+'</testsuite></testsuites>'
    put(junit,xml)
    failed='<testsuites><testsuite>'+''.join('<testcase name="case'+str(i)+'">'+
        ('<failure>original traceback with trailing spaces   </failure>' if i==0 else '')+'</testcase>'
        for i in range(9))+'</testsuite></testsuites>'
    for name in failures:put(name,failed if name.endswith('.xml') else '# original failed version\n')
    put(baseline,{'verified_at_utc':times[0]})
    reg={'source_git_commit':'b'*40,'source_files':{str(i):'0'*64 for i in range(20)},
        'gate':{'inherited':True},'registered_at_utc':times[3]}
    tests={'status':'passed','frozen_source_hashes':reg['source_files'],'parent_byte_gate':reg['gate'],
        'unique_passed_tests':269,'new_unique_passed_tests':158,'unchanged_related_regression_passed_tests':111,
        'actual_new_saved_factor_aggregation_before_freeze':0,'checked_at_utc':times[1],
        'test_receipts':[{'path':junit,'sha256':env.sha(tmp_path/junit),'passed_unique_tests':269}],
        'development_failure_artifacts':[{'path':name,'sha256':env.sha(tmp_path/name)} for name in failures]}
    put(validation,tests)
    put(exception,{'status':'passed','original_failure_xml':failures[0],
        'original_failure_xml_sha256':env.sha(tmp_path/failures[0]),'development_failure_test_count':9,
        'development_failure_failed_tests':1,'scientific_sources_configuration_thresholds_or_parent_inputs_changed':False,
        'actual_factor_aggregation_started':False,'prefreeze_validation_sha256':env.sha(tmp_path/validation),
        'checked_at_utc':times[2]})
    values={name:env.sha(tmp_path/name) for name in [baseline,validation,exception,junit,*failures]}
    archive={'prefreeze_receipts_committed_source_sha256':values,
        'original_development_failure_and_raw_whitespace_evidence_retained':True,
        'prefreeze_publication_analysis_verification_utc_order':times[:8],'verified_at_utc':times[8]}
    meta={'started_at_utc':times[5],'completed_at_utc':times[6]}
    proof={'checked_at_utc':times[4]};verification={'verified_at_utc':times[7]};calls=[]
    monkeypatch.setattr(env,'committed',lambda root,commit,hashes:calls.append((commit,hashes)))
    return archive,reg,meta,proof,verification,calls


def test_parent_prefreeze_reuses_exact_source_bytes_tests_and_failure_xml_without_learning(tmp_path,monkeypatch):
    args=factor_prefreeze_fixture(tmp_path,monkeypatch)
    gate=env.archived_factor_prefreeze_gate(tmp_path,*args[:5])
    assert gate['prefreeze_source_committed_receipts_checked']==7
    assert gate['raw_failed_xml_whitespace_preserved_without_science_change'] is True
    assert args[-1][0][0]==args[1]['source_git_commit'] and len(args[-1][0][1])==7


@pytest.mark.parametrize('field,value',[
    ('original_development_failure_and_raw_whitespace_evidence_retained',False),
    ('prefreeze_publication_analysis_verification_utc_order',[]),
    ('prefreeze_receipts_committed_source_sha256',{}),
])
def test_parent_seal_cannot_omit_failure_exception_or_chronology(tmp_path,monkeypatch,field,value):
    args=factor_prefreeze_fixture(tmp_path,monkeypatch);args[0][field]=value
    with pytest.raises(ValueError):env.archived_factor_prefreeze_gate(tmp_path,*args[:5])


@pytest.mark.parametrize('name',[
    'results/gated_factors_prefreeze_evidence_exception.json',
    'results/gated_factors_development/gated_factors_verify_development_failure_01.xml',
    'results/gated_factors_development/gated_factors_verify_failure_01.py',
])
def test_parent_original_failure_or_exception_bytes_cannot_be_rewritten(tmp_path,monkeypatch,name):
    args=factor_prefreeze_fixture(tmp_path,monkeypatch);(tmp_path/name).write_text('altered')
    with pytest.raises(ValueError):env.archived_factor_prefreeze_gate(tmp_path,*args[:5])


def test_parent_exception_and_final_tests_must_precede_registration(tmp_path,monkeypatch):
    args=factor_prefreeze_fixture(tmp_path,monkeypatch)
    args[1]['registered_at_utc']='2019-12-31T00:00:00+00:00'
    args[0]['prefreeze_publication_analysis_verification_utc_order'][3]=args[1]['registered_at_utc']
    with pytest.raises(ValueError):env.archived_factor_prefreeze_gate(tmp_path,*args[:5])
