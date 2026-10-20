"""Synthetic-only scope, counts, publication, archive and failure-retention gates."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drosolearn import self_blend_factors_envelope as env

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {'study_id': env.STUDY_ID, 'module': env.MODULE, 'protocol': env.PROTOCOL,
        'scope': 'retrospective_descriptive_self_blend_factors', 'external_manifest': None,
        'model_seeds': list(range(17701, 17711)), 'training_trials': 400,
        'source_files': sorted(env.SOURCE_FILES), 'parent_files': sorted(env.PARENT_FILES),
        'interpretation': {key: False for key in ('new_human_experiment',
            'shared_neural_mechanism_claim', 'education_effect_claim',
            'new_training', 'new_seed_draws', 'new_effect_hypothesis_test')}}


def summary():
    return {'study_id': env.STUDY_ID, 'status': 'completed',
        'scope': 'retrospective_descriptive_self_blend_factors', 'counts': {
            'analyzed_parent_trial_records': 144000, 'analyzed_destination_effects': 576000,
            'reference_gate_paths': 60, 'comparison_paths': 300,
            'paired_trial_records': 120000, 'measurement_records': 3600, 'numeric_arrays': 14,
            'paired_post_trial_records': 20, 'eligible_both_Go_trials': 7, 'corner_destination_effects': 224,
            **{key: 0 for key in ('new_training_trials', 'new_head_update_calls', 'new_internal_update_calls',
                'new_value_update_calls', 'new_random_draws', 'fresh_model_constructions', 'new_optimization_calls')}}}


def test_exact_new_runtime_and_direct_parent_closures():
    env.validate_envelope(ROOT, configuration(), ROOT / env.CONFIG)
    assert len(env.SOURCE_FILES) == 22 and len(env.PARENT_FILES) == 13
    assert 'src/drosolearn/self_blend_envelope.py' in env.REUSED_SOURCE
    assert 'src/drosolearn/gated_learning.py' not in env.SOURCE_FILES
    assert 'src/drosolearn/gated_pathways.py' not in env.SOURCE_FILES
    assert {'results/self_blend_learning_v1/arrays.npz', 'results/self_blend_figures_v1/plotted_values.json'} <= env.PARENT_FILES


@pytest.mark.parametrize('group', ['source_files', 'parent_files'])
@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'extra'])
def test_no_inventory_omission_duplicate_or_expansion(group, mode):
    cfg = configuration()
    if mode == 'missing': cfg[group] = cfg[group][1:]
    elif mode == 'duplicate': cfg[group][-1] = cfg[group][0]
    else: cfg[group].append('unfrozen.txt')
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


@pytest.mark.parametrize('field', ['new_training', 'new_seed_draws', 'new_effect_hypothesis_test', 'shared_neural_mechanism_claim'])
def test_readonly_interpretation_cannot_be_upgraded(field):
    cfg = configuration(); cfg['interpretation'][field] = True
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


@pytest.mark.parametrize('scope', ['integration_test', 'prospective_observed_gated_learning', None])
def test_new_learning_or_fixture_cannot_be_registered_as_saved_analysis(scope):
    cfg = configuration(); cfg['scope'] = scope
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


def test_fixed_path_and_no_external_data():
    with pytest.raises(ValueError): env.validate_envelope(ROOT, configuration(), ROOT / 'configs/other.json')
    cfg = configuration(); cfg['external_manifest'] = 'data/other.json'
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


def test_exact_saved_factor_work_counts_and_empty_eligible_boundary():
    env.validate_counts(configuration(), summary(), 3600)
    empty = summary()
    for key in ('paired_post_trial_records', 'eligible_both_Go_trials', 'corner_destination_effects'):
        empty['counts'][key] = 0
    env.validate_counts(configuration(), empty, 3600)


@pytest.mark.parametrize('field,value', [
    ('analyzed_parent_trial_records', 96000), ('analyzed_destination_effects', 384000),
    ('reference_gate_paths', 300), ('comparison_paths', 60), ('paired_trial_records', 144000),
    ('measurement_records', 1800), ('numeric_arrays', 12), ('new_training_trials', 1),
    ('new_head_update_calls', 1), ('new_internal_update_calls', False), ('new_value_update_calls', 1),
    ('new_random_draws', 1), ('fresh_model_constructions', 1), ('new_optimization_calls', 1),
    ('paired_post_trial_records', -1), ('paired_post_trial_records', 120001),
    ('eligible_both_Go_trials', 21), ('eligible_both_Go_trials', True),
    ('corner_destination_effects', 225), ('corner_destination_effects', 224.0),
])
def test_wrong_axes_any_new_work_and_invalid_bridge_counts_fail(field, value):
    changed = summary(); changed['counts'][field] = value
    with pytest.raises(ValueError): env.validate_counts(configuration(), changed, 3600)


def test_fixed_parent_cohort_budget_and_exact_count_keys():
    cfg = configuration(); cfg['model_seeds'][0] = 17501
    with pytest.raises(ValueError): env.validate_counts(cfg, summary(), 3600)
    cfg = configuration(); cfg['training_trials'] = 800
    with pytest.raises(ValueError): env.validate_counts(cfg, summary(), 3600)
    with pytest.raises(ValueError): env.validate_counts(configuration(), summary(), 3599)
    changed = summary(); changed['counts']['extra'] = 0
    with pytest.raises(ValueError): env.validate_counts(configuration(), changed, 3600)


def test_parent_entry_reuses_inherited_byte_gate_before_any_scientific_import(tmp_path, monkeypatch):
    from drosolearn import self_blend_envelope as prior
    def stop(root):
        assert root == tmp_path
        raise RuntimeError('inherited byte-only gate reached')
    monkeypatch.setattr(prior, 'parent_gate', stop)
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('scientific import'))
    with pytest.raises(RuntimeError, match='inherited byte-only gate reached'): env.parent_gate(tmp_path)


@pytest.mark.parametrize('gate', ['validate_registration', 'publication'])
def test_no_scientific_arithmetic_import_before_frozen_private_publication(tmp_path, monkeypatch, gate):
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), {}))
    def fail(*args): raise ValueError('missing frozen publication')
    monkeypatch.setattr(env, gate, fail)
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('premature producer import'))
    out = tmp_path / env.RESULT
    with pytest.raises(ValueError, match='missing frozen publication'):
        env.run(tmp_path, tmp_path / 'c', tmp_path / 'r', tmp_path / 'p', out)
    assert not out.exists()


def test_failure_keeps_exclusive_registration_and_zero_new_work_metadata(tmp_path, monkeypatch):
    out = tmp_path / env.RESULT; out.parent.mkdir()
    cfg = tmp_path / 'cfg.json'; cfg.write_text('{}')
    reg = tmp_path / 'reg.json'; reg.write_text('{}')
    proof = tmp_path / 'proof.json'; proof.write_text('{}')
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), {'source_git_commit': 'a' * 40}))
    monkeypatch.setattr(env, 'publication', lambda *args: ('proof.json', {'checked_at_utc': '2020-01-01T00:00:00+00:00'}))
    def fail(*args): raise RuntimeError('invented saved-path failure')
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: SimpleNamespace(validate_config=lambda *args: None, produce=fail))
    with pytest.raises(RuntimeError): env.run(tmp_path, cfg, reg, proof, out)
    meta = json.loads((out / 'metadata.json').read_text())
    assert meta['status'] == 'failed' and meta['analysis_only'] is True
    assert meta['new_training_calls'] == meta['new_random_seed_draws'] == meta['new_graph_or_feature_constructions'] == meta['new_optimization_calls'] == 0
    assert (out / 'registration.json').read_bytes() == reg.read_bytes()
    assert 'metadata.json' in json.loads((out / 'artifact_hashes.json').read_text())
    with pytest.raises(ValueError): env.run(tmp_path, cfg, reg, proof, out)


def learning_prefreeze_fixture(tmp_path, monkeypatch):
    """Manufacture prior receipts; no actual learning archive or NPZ is opened."""
    def put(name, value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value) if not isinstance(value,str) else value)
        return path
    times=['2020-01-01T00:0'+str(i)+':00+00:00' for i in range(8)]
    validation='results/self_blend_prefreeze_validation.json';baseline='results/self_blend_baseline_preservation.json'
    junit='results/self_blend_prefreeze_tests/self_blend_prefreeze_final.xml'
    first_name='results/self_blend_development/failure_01.json'
    second_name='results/self_blend_development/core_failure_01_manifest.json'
    zip1='results/self_blend_development/failure_01.zip';zip2='results/self_blend_development/core_failure_01.zip'
    put(zip1,'fake preserved raw ZIP1');put(zip2,'fake preserved raw ZIP2')
    classes={'tests.test_balanced_learning':61,'tests.test_gated_learning':47,
        'tests.test_self_blend_envelope':56,'tests.test_self_blend_learning':47,
        'tests.test_self_blend_learning_verify':131}
    put(junit,'<testsuites><testsuite>'+''.join('<testcase classname="'+cls+'" name="case'+str(i)+'"/>'
        for cls,count in classes.items() for i in range(count))+'</testsuite></testsuites>')
    first={'actual_cohort_started':False,'failed':6,'passed':95,'artifacts':{'source.py':'a'*64}}
    second={'actual_new_cohort_runs':0,'synthetic_tests_only':True,'failed_tests':6,'passed_tests':39,
        'member_sha256':{'core.py':'b'*64},'archive_sha256':env.sha(tmp_path/zip2),'preserved_at_utc':times[0]}
    put(first_name,first);put(second_name,second);put(baseline,{'verified_at_utc':times[0]})
    reg={'source_git_commit':'c'*40,'source_files':{str(i):'a'*64 for i in range(44)},
        'gate':{'inherited':True},'registered_at_utc':times[2]}
    tests={'status':'passed','frozen_source_hashes':reg['source_files'],'parent_byte_gate':reg['gate'],
        'unique_passed_tests':342,'new_unique_passed_tests':234,'unchanged_related_regression_passed_tests':108,
        'actual_new_training_before_freeze':0,'new_cohort_graph_feature_rng_calls_before_freeze':0,
        'synthetic_fixtures_only_before_freeze':True,'checked_at_utc':times[1],
        'test_receipts':[{'path':junit,'sha256':env.sha(tmp_path/junit),'passed_unique_tests':342}],
        'development_failure_artifacts':[{'path':name,'sha256':env.sha(tmp_path/name)}
            for name in (first_name,second_name,zip1,zip2)]}
    put(validation,tests)
    values={name:env.sha(tmp_path/name) for name in (validation,baseline,junit,first_name,second_name,zip1,zip2)}
    archive={'prefreeze_receipts_committed_source_sha256':values,
        'prefreeze_passed_unique_tests_independently_counted':342,'prefreeze_unique_tests_by_class':classes,
        'original_development_failure_ZIPs_and_member_hashes_checked':[
            {'archive':zip1,'sha256':values[zip1],'member_sha256':first['artifacts'],'tests':101,'failed':6},
            {'archive':zip2,'sha256':values[zip2],'member_sha256':second['member_sha256'],'tests':45,'failed':6}],
        'prefreeze_publication_learning_verification_utc_order':times[:7],'verified_at_utc':times[7]}
    meta={'started_at_utc':times[4],'completed_at_utc':times[5]};proof={'checked_at_utc':times[3]}
    verification={'verified_at_utc':times[6]};calls=[]
    monkeypatch.setattr(env,'committed',lambda root,commit,hashes:calls.append((commit,hashes)))
    return archive,reg,meta,proof,verification,calls


def test_parent_actual_replay_and_zip_member_checks_reused_only_by_frozen_bytes(tmp_path,monkeypatch):
    args=learning_prefreeze_fixture(tmp_path,monkeypatch)
    gate=env.archived_learning_prefreeze_gate(tmp_path,*args[:5])
    assert gate['source_committed_prefreeze_receipts_checked']==7
    assert gate['prior_ZIP_member_hash_validation_reused_by_raw_bytes'] is True
    assert args[-1][0][0]==args[1]['source_git_commit'] and len(args[-1][0][1])==7


@pytest.mark.parametrize('field,value',[
    ('prefreeze_passed_unique_tests_independently_counted',341),
    ('prefreeze_unique_tests_by_class',{}),
    ('original_development_failure_ZIPs_and_member_hashes_checked',[]),
    ('prefreeze_publication_learning_verification_utc_order',[]),
])
def test_actual_parent_seal_cannot_drop_old_tests_failures_or_chronology(tmp_path,monkeypatch,field,value):
    args=learning_prefreeze_fixture(tmp_path,monkeypatch);args[0][field]=value
    with pytest.raises(ValueError):env.archived_learning_prefreeze_gate(tmp_path,*args[:5])


@pytest.mark.parametrize('name',[
    'results/self_blend_development/failure_01.zip',
    'results/self_blend_development/core_failure_01_manifest.json',
    'results/self_blend_prefreeze_tests/self_blend_prefreeze_final.xml',
])
def test_actual_parent_original_failure_or_test_bytes_cannot_be_rewritten(tmp_path,monkeypatch,name):
    args=learning_prefreeze_fixture(tmp_path,monkeypatch);(tmp_path/name).write_text('altered')
    with pytest.raises(ValueError):env.archived_learning_prefreeze_gate(tmp_path,*args[:5])


def test_actual_parent_development_and_tests_must_precede_registration(tmp_path,monkeypatch):
    args=learning_prefreeze_fixture(tmp_path,monkeypatch)
    args[1]['registered_at_utc']='2019-12-31T00:00:00+00:00'
    args[0]['prefreeze_publication_learning_verification_utc_order'][2]=args[1]['registered_at_utc']
    with pytest.raises(ValueError):env.archived_learning_prefreeze_gate(tmp_path,*args[:5])
