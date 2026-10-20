"""Synthetic-only publication, positive solver work and historical byte gates."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drosolearn import probability_envelope as env

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {'study_id': env.STUDY_ID, 'module': env.MODULE, 'protocol': env.PROTOCOL,
        'scope': 'retrospective_descriptive_probability_alignment', 'external_manifest': None,
        'model_seeds': list(range(17701, 17711)), 'training_trials': 400,
        'source_files': sorted(env.SOURCE_FILES), 'parent_files': sorted(env.PARENT_FILES),
        'interpretation': {key: False for key in ('new_human_experiment', 'shared_neural_mechanism_claim',
            'education_effect_claim', 'new_training', 'new_seed_draws', 'new_effect_hypothesis_test')}}


def summary():
    return {'study_id': env.STUDY_ID, 'status': 'completed',
        'scope': 'retrospective_descriptive_probability_alignment', 'counts': {
            'saved_reference_records': 24000, 'saved_reference_destination_effects': 96000,
            'eligible_actual_Go_records': 7, 'candidate_destination_effects': 140,
            'unique_uniform_optimizations': 3, 'state_probability_optimizations': 7,
            'new_optimization_calls': 10, 'candidate_single_update_evaluations': 35,
            'numeric_arrays': 27, 'measurement_records': 900,
            **{key: 0 for key in ('new_training_trials', 'new_head_update_calls', 'new_internal_update_calls',
                'new_value_update_calls', 'new_random_draws', 'fresh_model_constructions')}}}


def test_exact_pure_integrity_math_and_direct_parent_closures():
    env.validate_envelope(ROOT, configuration(), ROOT / env.CONFIG)
    assert len(env.SOURCE_FILES) == 24 and len(env.PARENT_FILES) == 17
    assert {'src/drosolearn/cone_geometry.py', 'src/drosolearn/self_blend_factors_envelope.py'} <= env.REUSED_SOURCE
    assert 'src/drosolearn/self_blend_learning.py' not in env.SOURCE_FILES
    assert len(env.ORIGINAL_FILES) == 6 and env.ORIGINAL_FILES <= env.PARENT_FILES


@pytest.mark.parametrize('group', ['source_files', 'parent_files'])
@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'extra'])
def test_exact_inventory_cannot_omit_duplicate_or_expand(group, mode):
    cfg = configuration()
    if mode == 'missing': cfg[group] = cfg[group][1:]
    elif mode == 'duplicate': cfg[group][-1] = cfg[group][0]
    else: cfg[group].append('unfrozen.py')
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


@pytest.mark.parametrize('field', ['new_training', 'new_seed_draws', 'new_effect_hypothesis_test', 'shared_neural_mechanism_claim'])
def test_analysis_cannot_be_upgraded_to_new_learning_or_mechanism(field):
    cfg = configuration(); cfg['interpretation'][field] = True
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


@pytest.mark.parametrize('scope', ['integration_test', 'prospective_observed_gated_learning', None])
def test_actual_learning_or_fixture_cannot_be_registered_as_objective_audit(scope):
    cfg = configuration(); cfg['scope'] = scope
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


def test_positive_new_direction_optimization_is_separate_from_zero_new_learning():
    env.validate_counts(configuration(), summary(), 900)


@pytest.mark.parametrize('field,value', [
    ('saved_reference_records', 144000), ('saved_reference_destination_effects', 576000),
    ('eligible_actual_Go_records', 0), ('eligible_actual_Go_records', 24001),
    ('candidate_destination_effects', 141), ('candidate_single_update_evaluations', 34),
    ('unique_uniform_optimizations', 0), ('unique_uniform_optimizations', 241),
    ('unique_uniform_optimizations', 8), ('state_probability_optimizations', 6),
    ('new_optimization_calls', 0), ('new_optimization_calls', 11), ('new_optimization_calls', True),
    ('numeric_arrays', 14), ('measurement_records', 3600), ('new_training_trials', 1),
    ('new_head_update_calls', 1), ('new_internal_update_calls', False), ('new_value_update_calls', 1),
    ('new_random_draws', 1), ('fresh_model_constructions', 1),
])
def test_wrong_work_axes_zero_solver_claim_or_new_learning_fails(field, value):
    changed = summary(); changed['counts'][field] = value
    with pytest.raises(ValueError): env.validate_counts(configuration(), changed, 900)


def test_fixed_cohort_budget_rows_and_exact_counter_keys():
    cfg = configuration(); cfg['model_seeds'][0] = 17601
    with pytest.raises(ValueError): env.validate_counts(cfg, summary(), 900)
    cfg = configuration(); cfg['training_trials'] = 800
    with pytest.raises(ValueError): env.validate_counts(cfg, summary(), 900)
    with pytest.raises(ValueError): env.validate_counts(configuration(), summary(), 899)
    changed = summary(); changed['counts']['unregistered_counter'] = 0
    with pytest.raises(ValueError): env.validate_counts(configuration(), changed, 900)


def test_parent_entry_reuses_latest_byte_gate_before_any_scientific_import(tmp_path, monkeypatch):
    from drosolearn import self_blend_factors_envelope as prior
    def stop(root):
        assert root == tmp_path
        raise RuntimeError('prior byte-only gate')
    monkeypatch.setattr(prior, 'parent_gate', stop)
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('premature science import'))
    with pytest.raises(RuntimeError, match='prior byte-only gate'): env.parent_gate(tmp_path)


@pytest.mark.parametrize('gate', ['validate_registration', 'publication'])
def test_no_actual_state_math_or_solver_import_before_frozen_private_publication(tmp_path, monkeypatch, gate):
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), {}))
    def fail(*args): raise ValueError('frozen private evidence missing')
    monkeypatch.setattr(env, gate, fail)
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('premature solver/science import'))
    out = tmp_path / env.RESULT
    with pytest.raises(ValueError, match='frozen private evidence missing'):
        env.run(tmp_path, tmp_path / 'c', tmp_path / 'r', tmp_path / 'p', out)
    assert not out.exists()


def run_fixture(tmp_path, monkeypatch, fail=False):
    (tmp_path / 'results').mkdir()
    for name in ('cfg.json', 'reg.json', 'proof.json'): (tmp_path / name).write_text('{}')
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), {'source_git_commit': 'a' * 40}))
    monkeypatch.setattr(env, 'publication', lambda *args: ('proof.json', {'checked_at_utc': '2020-01-01T00:00:00+00:00'}))
    def produce(*args):
        if fail: raise RuntimeError('invented solver failure')
        return {str(i): np.zeros(1) for i in range(27)}, [{'fake': 0} for _ in range(900)]
    module = SimpleNamespace(validate_config=lambda *args: None, produce=produce, summarize=lambda *args: summary())
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: module)
    return (tmp_path, tmp_path / 'cfg.json', tmp_path / 'reg.json', tmp_path / 'proof.json', tmp_path / env.RESULT)


def test_completed_metadata_records_positive_solver_calls_and_zero_learning(tmp_path, monkeypatch):
    args = run_fixture(tmp_path, monkeypatch)
    meta = env.run(*args)
    assert meta['status'] == 'completed' and meta['analysis_only'] is True
    assert meta['new_optimization_calls'] == 10
    assert meta['new_training_calls'] == meta['new_random_seed_draws'] == meta['new_graph_or_feature_constructions'] == 0


def test_failure_is_exclusive_and_does_not_invent_zero_completed_solver_work(tmp_path, monkeypatch):
    args = run_fixture(tmp_path, monkeypatch, fail=True)
    with pytest.raises(RuntimeError): env.run(*args)
    meta = json.loads((args[-1] / 'metadata.json').read_text())
    assert meta['status'] == 'failed' and meta['new_optimization_calls'] is None and 'counts' not in meta
    assert (args[-1] / 'registration.json').read_bytes() == args[2].read_bytes()
    assert 'metadata.json' in json.loads((args[-1] / 'artifact_hashes.json').read_text())
    with pytest.raises(ValueError): env.run(*args)


def parent_prefreeze_fixture(tmp_path, monkeypatch):
    def put(name, value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value) if not isinstance(value,str) else value)
        return path
    validation='results/self_blend_factors_prefreeze_validation.json';baseline='results/self_blend_factors_baseline_preservation.json'
    junit='results/self_blend_factors_prefreeze_tests/self_blend_factors_prefreeze_final.xml'
    note_name='results/self_blend_factors_prefreeze_finalization.json';zip_name='results/self_blend_factors_prefreeze_style_attempt01.zip'
    test='tests/test_self_blend_factors_envelope.py';times=['2020-01-01T00:0'+str(i)+':00+00:00' for i in range(9)]
    classes={'tests.test_gated_factors':45,'tests.test_self_blend_factors':56,'tests.test_self_blend_factors_envelope':50,
        'tests.test_self_blend_factors_verify':97,'tests.test_self_blend_learning':47}
    put(junit,'<testsuites><testsuite>'+''.join('<testcase classname="'+cls+'" name="case'+str(i)+'"/>'
        for cls,count in classes.items() for i in range(count))+'</testsuite></testsuites>')
    put(zip_name,'fake raw ZIP previously independently byte-checked')
    reg={'source_git_commit':'b'*40,'source_files':{**{str(i):'0'*64 for i in range(21)},test:'f'*64},
        'gate':{'inherited':True},'registered_at_utc':times[3]}
    note={'status':'passed','archive':zip_name,'archive_sha256':env.sha(tmp_path/zip_name),'test_path':test,
        'final_test_sha256':reg['source_files'][test],'removed_only_trailing_blank_line':True,
        'scientific_design_thresholds_or_parent_inputs_changed':False,'actual_saved_factor_aggregation_started':False,
        'member_sha256':{test:'e'*64},'checked_at_utc':times[1]}
    put(note_name,note);put(baseline,{'verified_at_utc':times[0]})
    tests={'status':'passed','frozen_source_hashes':reg['source_files'],'parent_byte_gate':reg['gate'],
        'unique_passed_tests':295,'new_unique_passed_tests':203,'unchanged_related_regression_passed_tests':92,
        'actual_new_saved_factor_aggregation_before_freeze':0,'new_training_random_graph_or_optimization_calls':0,
        'synthetic_fixtures_only_before_freeze':True,'checked_at_utc':times[2],
        'test_receipts':[{'path':junit,'sha256':env.sha(tmp_path/junit),'passed_unique_tests':295}],
        'development_failure_artifacts':[{'path':name,'sha256':env.sha(tmp_path/name)} for name in (note_name,zip_name)]}
    put(validation,tests);values={name:env.sha(tmp_path/name) for name in (validation,baseline,junit,note_name,zip_name)}
    style={'original_style_attempt_zip_sha256':values[zip_name],'finalization_note_sha256':values[note_name],
        'original_member_sha256':note['member_sha256'],'only_trailing_empty_LF_lines_removed_from_one_unregistered_test':True,
        'all_other21_source_hashes_and_original_295_passing_test_identities_preserved':True,'non_LF_original_bytes_identical':True,
        'removed_trailing_LF_bytes':2,'removed_trailing_LF_bytes_independently_counted':2,'final_trailing_LF_bytes':1,
        'original_notice_count_corrected_by_rawbytes':True,
        'original_tests_finalization_refreshed_tests_registration_utc_order':[times[0],times[1],times[2],times[3]]}
    meta={'started_at_utc':times[5],'completed_at_utc':times[6]};proof={'checked_at_utc':times[4]}
    verification={'verified_at_utc':times[7]};calls=[]
    archive={'prefreeze_receipts_committed_source_sha256':values,'development_failure_artifact_bytes_preserved':tests['development_failure_artifacts'],
        'prefreeze_passed_unique_tests_independently_counted':295,'prefreeze_unique_tests_by_class':classes,
        'original_prefreeze_style_attempt_and_EOF_only_finalization':style,'original_notice_count_corrected_by_rawbytes':True,
        'removed_trailing_LF_bytes':2,'prefreeze_publication_analysis_verification_utc_order':
        [times[0],times[2],times[3],times[4],times[5],times[6],times[7]],'verified_at_utc':times[8]}
    monkeypatch.setattr(env,'committed',lambda root,commit,hashes:calls.append((commit,hashes)))
    return archive,reg,meta,proof,verification,calls


def test_parent_295_test_and_exact_two_LF_style_receipts_reused_by_source_bytes(tmp_path,monkeypatch):
    args=parent_prefreeze_fixture(tmp_path,monkeypatch)
    gate=env.archived_probability_parent_gate(tmp_path,*args[:5])
    assert gate['source_committed_prefreeze_and_style_receipts_checked']==5
    assert gate['no_parent_scientific_replay_or_solver_calls'] is True
    assert args[-1][0][0]==args[1]['source_git_commit'] and len(args[-1][0][1])==5


@pytest.mark.parametrize('field,value', [('removed_trailing_LF_bytes',1),
    ('original_notice_count_corrected_by_rawbytes',False),('prefreeze_passed_unique_tests_independently_counted',294),
    ('prefreeze_publication_analysis_verification_utc_order',[])])
def test_parent_style_counts_and_source_chronology_cannot_be_rewritten(tmp_path,monkeypatch,field,value):
    args=parent_prefreeze_fixture(tmp_path,monkeypatch);args[0][field]=value
    with pytest.raises(ValueError):env.archived_probability_parent_gate(tmp_path,*args[:5])


def test_parent_preserved_raw_style_ZIP_cannot_change(tmp_path,monkeypatch):
    args=parent_prefreeze_fixture(tmp_path,monkeypatch)
    (tmp_path/'results/self_blend_factors_prefreeze_style_attempt01.zip').write_text('changed')
    with pytest.raises(ValueError):env.archived_probability_parent_gate(tmp_path,*args[:5])
