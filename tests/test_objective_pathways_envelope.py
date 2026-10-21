"""Synthetic-only immutability, chronology and zero-new-learning boundaries."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drosolearn import objective_pathways_envelope as env

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {'study_id': env.STUDY_ID, 'module': env.MODULE, 'protocol': env.PROTOCOL,
        'scope': 'retrospective_descriptive_objective_pathways', 'external_manifest': None,
        'model_seeds': list(range(17801, 17811)), 'training_trials': 400,
        'source_files': sorted(env.SOURCE_FILES), 'parent_files': sorted(env.PARENT_FILES),
        'interpretation': {key: False for key in ('new_human_experiment',
            'shared_neural_mechanism_claim', 'education_effect_claim',
            'new_training', 'new_seed_draws', 'new_effect_hypothesis_test')}}


def summary():
    return {'study_id': env.STUDY_ID, 'status': 'completed',
        'scope': 'retrospective_descriptive_objective_pathways', 'counts': {
            'analyzed_parent_trial_records': 192000, 'analyzed_destination_effects': 768000,
            'reference_gate_paths': 60, 'comparison_paths': 180,
            'paired_trial_records': 72000, 'measurement_records': 2160, 'numeric_arrays': 16,
            'actual_path_strata': 34560,
            'paired_post_trial_records': 20, 'eligible_both_Go_trials': 7, 'corner_destination_effects': 224,
            **{key: 0 for key in ('new_training_trials', 'new_head_update_calls', 'new_internal_update_calls',
                'new_value_update_calls', 'new_random_draws', 'fresh_model_constructions', 'new_optimization_calls')}}}


def test_exact_minimal_byte_closure_and_lossless_parent():
    env.validate_envelope(ROOT, configuration(), ROOT / env.CONFIG)
    assert len(env.REUSED_SOURCE) == 16 and len(env.SOURCE_FILES) == 25 and len(env.PARENT_FILES) == 16
    assert 'src/drosolearn/objective_envelope.py' in env.REUSED_SOURCE
    assert 'src/drosolearn/objective_learning.py' not in env.SOURCE_FILES
    assert 'src/drosolearn/cone_geometry.py' not in env.SOURCE_FILES
    assert 'results/objective_learning_v1/measurements.csv.gz' in env.PARENT_FILES
    assert 'results/objective_learning_v1/measurements.csv' not in env.PARENT_FILES
    assert env.BASE_TAG == 'objective-learning-v1'
    assert env.BASE_COMMIT == 'b96a6c9d88a9f83cbe4e093272eea0ca77f245fe'
    assert len(env.NUMERIC_ARRAYS) == 16 and env.MEASUREMENT_COLUMN_COUNT == 15


@pytest.mark.parametrize('group', ['source_files', 'parent_files'])
@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'extra'])
def test_no_inventory_omission_duplicate_or_expansion(group, mode):
    cfg = configuration()
    if mode == 'missing': cfg[group] = cfg[group][1:]
    elif mode == 'duplicate': cfg[group][-1] = cfg[group][0]
    else: cfg[group].append('unfrozen.txt')
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


@pytest.mark.parametrize('field', ['new_training', 'new_seed_draws', 'new_effect_hypothesis_test',
    'shared_neural_mechanism_claim', 'new_human_experiment', 'education_effect_claim'])
def test_readonly_interpretation_cannot_be_upgraded(field):
    cfg = configuration(); cfg['interpretation'][field] = True
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


@pytest.mark.parametrize('scope', ['integration_test', 'prospective_observed_objective_learning', None])
def test_new_learning_or_fixture_cannot_be_registered_as_saved_analysis(scope):
    cfg = configuration(); cfg['scope'] = scope
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


def test_fixed_path_and_no_external_data():
    with pytest.raises(ValueError): env.validate_envelope(ROOT, configuration(), ROOT / 'configs/other.json')
    cfg = configuration(); cfg['external_manifest'] = 'data/other.json'
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


def test_zero_bridge_eligible_is_retained_without_replacement():
    env.validate_counts(configuration(), summary(), 2160)
    empty = summary()
    for key in ('paired_post_trial_records', 'eligible_both_Go_trials', 'corner_destination_effects'):
        empty['counts'][key] = 0
    env.validate_counts(configuration(), empty, 2160)


@pytest.mark.parametrize('field,value', [
    ('analyzed_parent_trial_records', 144000), ('analyzed_destination_effects', 576000),
    ('reference_gate_paths', 180), ('comparison_paths', 60), ('paired_trial_records', 120000),
    ('measurement_records', 3600), ('numeric_arrays', 14), ('actual_path_strata', 3456), ('new_training_trials', 1),
    ('new_head_update_calls', 1), ('new_internal_update_calls', False), ('new_value_update_calls', 1),
    ('new_random_draws', 1), ('fresh_model_constructions', 1), ('new_optimization_calls', 1),
    ('paired_post_trial_records', -1), ('paired_post_trial_records', 72001),
    ('eligible_both_Go_trials', 21), ('eligible_both_Go_trials', True),
    ('corner_destination_effects', 225), ('corner_destination_effects', 224.0),
])
def test_wrong_axes_new_work_and_invalid_bridge_counts_fail(field, value):
    changed = summary(); changed['counts'][field] = value
    with pytest.raises(ValueError): env.validate_counts(configuration(), changed, 2160)


def test_fixed_parent_cohort_budget_and_exact_count_keys():
    cfg = configuration(); cfg['model_seeds'][0] = 17701
    with pytest.raises(ValueError): env.validate_counts(cfg, summary(), 2160)
    cfg = configuration(); cfg['training_trials'] = 800
    with pytest.raises(ValueError): env.validate_counts(cfg, summary(), 2160)
    with pytest.raises(ValueError): env.validate_counts(configuration(), summary(), 2159)
    changed = summary(); changed['counts']['extra'] = 0
    with pytest.raises(ValueError): env.validate_counts(configuration(), changed, 2160)


def test_inherited_byte_gate_precedes_parent_data_or_science_import(tmp_path, monkeypatch):
    from drosolearn import objective_envelope as prior
    def stop(root):
        assert root == tmp_path
        raise RuntimeError('inherited byte-only gate reached')
    monkeypatch.setattr(prior, 'parent_gate', stop)
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('scientific import'))
    with pytest.raises(RuntimeError, match='inherited byte-only gate reached'): env.parent_gate(tmp_path)


@pytest.mark.parametrize('gate', ['validate_registration', 'publication'])
def test_no_arithmetic_import_before_frozen_private_publication(tmp_path, monkeypatch, gate):
    monkeypatch.setattr(env, 'validate_registration', lambda *args: (configuration(), {}))
    def fail(*args): raise ValueError('missing frozen publication')
    monkeypatch.setattr(env, gate, fail)
    monkeypatch.setattr(env.importlib, 'import_module', lambda *args: pytest.fail('premature producer import'))
    out = tmp_path / env.RESULT
    with pytest.raises(ValueError, match='missing frozen publication'):
        env.run(tmp_path, tmp_path / 'c', tmp_path / 'r', tmp_path / 'p', out)
    assert not out.exists()


def test_failure_preserves_exclusive_registration_and_zero_work_metadata(tmp_path, monkeypatch):
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


def prefreeze_fixture(tmp_path, monkeypatch):
    """Only invented JSON/XML text; no actual parent arrays are opened."""
    def put(name, value):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if not isinstance(value, str) else value)
    times = ['2020-01-01T00:0' + str(i) + ':00+00:00' for i in range(8)]
    validation = 'results/objective_learning_prefreeze_validation.json'
    baseline = 'results/objective_learning_baseline_preservation.json'
    junit = 'results/objective_learning_prefreeze_tests/objective_learning_prefreeze_final.xml'
    classes = {'tests.test_cone_geometry': 45, 'tests.test_objective_learning': 59,
        'tests.test_objective_learning_envelope': 62, 'tests.test_objective_learning_verify': 134,
        'tests.test_probability_alignment': 38, 'tests.test_self_blend_learning': 47}
    put(junit, '<testsuites><testsuite>' + ''.join('<testcase classname="' + cls + '" name="case' + str(i) + '"/>'
        for cls, count in classes.items() for i in range(count)) + '</testsuite></testsuites>')
    put(baseline, {'verified_at_utc': times[0]})
    reg = {'source_git_commit': 'c' * 40, 'source_files': {str(i): 'a' * 64 for i in range(49)},
        'gate': {'inherited': True}, 'registered_at_utc': times[2]}
    tests = {'status': 'passed', 'frozen_source_hashes': reg['source_files'], 'parent_byte_gate': reg['gate'],
        'unique_passed_tests': 385, 'new_unique_passed_tests': 255, 'unchanged_related_regression_passed_tests': 130,
        'unique_passed_tests_by_class': classes, 'synthetic_fixtures_only_before_freeze': True,
        'actual_new_training_before_freeze': 0, 'new_cohort_graph_feature_rng_calls_before_freeze': 0,
        'actual_new_cohort_objective_optimizations_before_freeze': 0, 'checked_at_utc': times[1],
        'test_receipts': [{'path': junit, 'sha256': env.sha(tmp_path / junit), 'passed_unique_tests': 385}],
        'development_failure_artifacts': []}
    put(validation, tests)
    values = {name: env.sha(tmp_path / name) for name in (validation, baseline, junit)}
    archive = {'prefreeze_receipts_committed_source_sha256': values,
        'prefreeze_passed_unique_tests_independently_counted': 385, 'prefreeze_unique_tests_by_class': classes,
        'development_failure_artifact_bytes_preserved': [], 'development_ZIP_manifests_and_original_member_hashes': {},
        'prefreeze_publication_learning_verification_utc_order': times[:7], 'verified_at_utc': times[7]}
    meta = {'started_at_utc': times[4], 'completed_at_utc': times[5]}
    proof = {'checked_at_utc': times[3]}; verified = {'verified_at_utc': times[6]}; calls = []
    monkeypatch.setattr(env, 'committed', lambda root, commit, hashes: calls.append((commit, deepcopy(hashes))))
    return archive, reg, meta, proof, verified, calls


def test_original_385_test_and_zero_development_receipts_reused_as_bytes(tmp_path, monkeypatch):
    args = prefreeze_fixture(tmp_path, monkeypatch)
    gate = env.archived_learning_prefreeze_gate(tmp_path, *args[:5])
    assert gate['source_committed_prefreeze_receipts_checked'] == 3
    assert gate['original_385_unique_tests_and_zero_new_development_failures_checked'] is True
    assert args[-1][0][0] == args[1]['source_git_commit'] and len(args[-1][0][1]) == 3


@pytest.mark.parametrize('field,value', [
    ('prefreeze_passed_unique_tests_independently_counted', 384),
    ('prefreeze_unique_tests_by_class', {}),
    ('development_failure_artifact_bytes_preserved', [{'path': 'invented.zip'}]),
    ('development_ZIP_manifests_and_original_member_hashes', {'invented.zip': {}}),
    ('prefreeze_publication_learning_verification_utc_order', []),
])
def test_original_test_partition_failures_and_chronology_cannot_change(tmp_path, monkeypatch, field, value):
    args = prefreeze_fixture(tmp_path, monkeypatch); args[0][field] = value
    with pytest.raises(ValueError): env.archived_learning_prefreeze_gate(tmp_path, *args[:5])


@pytest.mark.parametrize('name', [
    'results/objective_learning_prefreeze_validation.json',
    'results/objective_learning_prefreeze_tests/objective_learning_prefreeze_final.xml',
    'results/objective_learning_baseline_preservation.json',
])
def test_actual_parent_prefreeze_bytes_cannot_be_rewritten(tmp_path, monkeypatch, name):
    args = prefreeze_fixture(tmp_path, monkeypatch); (tmp_path / name).write_text('altered')
    with pytest.raises(ValueError): env.archived_learning_prefreeze_gate(tmp_path, *args[:5])


def test_original_tests_must_precede_registration_and_actual_learning(tmp_path, monkeypatch):
    args = prefreeze_fixture(tmp_path, monkeypatch)
    args[1]['registered_at_utc'] = '2019-12-31T00:00:00+00:00'
    args[0]['prefreeze_publication_learning_verification_utc_order'][2] = args[1]['registered_at_utc']
    with pytest.raises(ValueError): env.archived_learning_prefreeze_gate(tmp_path, *args[:5])


def test_no_learner_graph_solver_or_rng_imports_in_new_envelope():
    tree = ast.parse((ROOT / 'src/drosolearn/objective_pathways_envelope.py').read_text(encoding='utf8'))
    imported = {node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names}
    assert not any(name.startswith(('scipy', 'random')) or name in ('graph', 'objective_learning', 'cone_geometry')
        for name in imported)
