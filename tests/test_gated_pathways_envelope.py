"""Synthetic-only scope, count, publication and inherited archival-gate checks."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drosolearn import gated_pathways_envelope as env

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {'study_id': env.STUDY_ID, 'module': env.MODULE, 'protocol': env.PROTOCOL,
        'scope': 'retrospective_descriptive_gated_pathways', 'external_manifest': None,
        'model_seeds': list(range(17601, 17611)), 'training_trials': 400,
        'source_files': sorted(env.SOURCE_FILES), 'parent_files': sorted(env.PARENT_FILES),
        'interpretation': {key: False for key in ('new_human_experiment',
            'shared_neural_mechanism_claim', 'education_effect_claim',
            'new_training', 'new_seed_draws', 'new_effect_hypothesis_test')}}


def summary():
    return {'study_id': env.STUDY_ID, 'status': 'completed',
        'scope': 'retrospective_descriptive_gated_pathways', 'counts': {
            'analyzed_parent_trial_records': 120000, 'analyzed_destination_effects': 480000,
            'reference_gate_paths': 60, 'paired_comparison_paths': 240,
            'paired_comparison_trial_records': 96000, 'measurement_records': 1800, 'numeric_arrays': 12,
            **{key: 0 for key in ('new_training_trials', 'new_head_update_calls', 'new_internal_update_calls',
                'new_value_update_calls', 'new_random_draws', 'fresh_model_constructions', 'new_optimization_calls')}}}


def test_exact_new_runtime_closure():
    env.validate_envelope(ROOT, configuration(), ROOT / env.CONFIG)
    assert len(env.SOURCE_FILES) == 19 and len(env.PARENT_FILES) == 11
    assert {'src/drosolearn/pathways_envelope.py', 'src/drosolearn/gated_envelope.py'} <= env.REUSED_SOURCE
    assert 'src/drosolearn/gated_learning.py' not in env.SOURCE_FILES
    assert {'results/gated_learning_v1/arrays.npz', 'results/gated_learning_v1/measurements.csv'} <= env.PARENT_FILES


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
def test_new_learning_or_fixture_cannot_be_registered_as_saved_path_analysis(scope):
    cfg = configuration(); cfg['scope'] = scope
    with pytest.raises(ValueError): env.validate_envelope(ROOT, cfg, ROOT / env.CONFIG)


def test_exact_saved_path_work_counts():
    env.validate_counts(configuration(), summary(), 1800)


@pytest.mark.parametrize('field,value', [
    ('analyzed_parent_trial_records', 96000), ('analyzed_destination_effects', 384000),
    ('reference_gate_paths', 300), ('paired_comparison_paths', 60), ('paired_comparison_trial_records', 120000),
    ('measurement_records', 1440), ('numeric_arrays', 195), ('new_training_trials', 1),
    ('new_head_update_calls', 1), ('new_internal_update_calls', False), ('new_value_update_calls', 1),
    ('new_random_draws', 1), ('fresh_model_constructions', 1), ('new_optimization_calls', 1),
])
def test_wrong_axes_or_any_new_work_or_boolean_counter_fails(field, value):
    changed = summary(); changed['counts'][field] = value
    with pytest.raises(ValueError): env.validate_counts(configuration(), changed, 1800)


def test_fixed_parent_cohort_budget_and_count_closure():
    cfg = configuration(); cfg['model_seeds'][0] = 17501
    with pytest.raises(ValueError): env.validate_counts(cfg, summary(), 1800)
    cfg = configuration(); cfg['training_trials'] = 800
    with pytest.raises(ValueError): env.validate_counts(cfg, summary(), 1800)
    with pytest.raises(ValueError): env.validate_counts(configuration(), summary(), 1799)
    changed = summary(); changed['counts']['extra'] = 0
    with pytest.raises(ValueError): env.validate_counts(configuration(), changed, 1800)


def test_parent_entry_reuses_inherited_byte_gate_before_any_scientific_import(tmp_path, monkeypatch):
    from drosolearn import gated_envelope as prior
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


def test_failure_keeps_exclusive_raw_registration_and_zero_new_work_metadata(tmp_path, monkeypatch):
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


def archival_fixture(tmp_path, monkeypatch):
    names = ('results/gated_learning_baseline_preservation.json',
        'results/gated_learning_prefreeze_validation.json',
        'results/gated_learning_prefreeze_finalization.json',
        'results/gated_learning_prefreeze_tests/gated_prefreeze_tests_final.xml')
    times = ['2020-01-01T00:0' + str(i) + ':00+00:00' for i in range(9)]
    source = {**{'source_' + str(i) + '.py': '0' * 64 for i in range(38)}, 'tests/test_gated_learning_verify.py': '1' * 64}
    reg = {'source_git_commit': 'b' * 40, 'source_files': source, 'registered_at_utc': times[4]}
    tests = {'status': 'passed', 'frozen_source_hashes': source, 'unique_passed_tests': 242, 'checked_at_utc': times[2]}
    final = {'status': 'passed', 'passed_unique_tests': 242, 'all39_final_source_hashes_verified': True,
        'new_cohort_graph_or_learning_started': False, 'configuration_seeds_budget_thresholds_or_model_source_changed': False,
        'premature_unpublished_source_commit': 'a' * 40, 'checked_at_utc': times[3]}
    docs = [{'started_at_utc': times[0], 'verified_at_utc': times[1]}, tests, final]
    for name, value in zip(names[:3], docs):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
    final['final_prefreeze_receipt_sha256'] = env.sha(tmp_path / names[1])
    (tmp_path / names[2]).write_text(json.dumps(final))
    junit = tmp_path / names[3]; junit.parent.mkdir(parents=True); junit.write_text('<testsuites/>')
    hashes = {name: env.sha(tmp_path / name) for name in names}
    changed = [names[1], names[2], names[3], 'tests/test_gated_learning_verify.py']
    keys = ('baseline_started', 'baseline_verified', 'final_tests_checked', 'finalization_checked',
        'registered', 'remote_bytes_checked', 'learning_started', 'learning_completed', 'independently_verified')
    archive = {'prefreeze_receipts_committed_source_sha256': hashes,
        'prefreeze_passed_unique_tests_independently_counted': 242,
        'prefreeze_finalization_retained_initial_commit': 'a' * 40,
        'configuration_and_model_source_unchanged_during_finalization': True,
        'prefreeze_finalization_changed_paths': changed,
        'prefreeze_to_independent_verification_utc_order': dict(zip(keys, times)), 'verified_at_utc': times[-1]}
    meta = {'started_at_utc': times[6], 'completed_at_utc': times[7]}
    proof = {'checked_at_utc': times[5]}; verified = {'verified_at_utc': times[8]}
    calls = []
    monkeypatch.setattr(env, 'committed', lambda root, commit, values: calls.append((commit, values)))
    monkeypatch.setattr(env, 'ancestor', lambda *args: calls.append(('ancestor', args)))
    monkeypatch.setattr(env, 'git', lambda *args: ('\n'.join(changed) + '\n').encode())
    return archive, reg, meta, proof, verified, calls


def test_original_prefreeze_links_are_checked_without_rerunning_parent_science(tmp_path, monkeypatch):
    archive, reg, meta, proof, verified, calls = archival_fixture(tmp_path, monkeypatch)
    gate = env.archived_prefreeze_gate(tmp_path, archive, reg, meta, proof, verified)
    assert gate['source_committed_prefreeze_receipts_checked'] == 4
    assert calls[0][0] == reg['source_git_commit'] and len(calls[0][1]) == 4
    assert len(calls[-1][1]) == 38


@pytest.mark.parametrize('field,value', [('prefreeze_passed_unique_tests_independently_counted', 241),
    ('prefreeze_finalization_retained_initial_commit', 'c' * 40),
    ('configuration_and_model_source_unchanged_during_finalization', False),
    ('prefreeze_finalization_changed_paths', ['src/drosolearn/gated_learning.py'])])
def test_broken_original_finalization_binding_fails(tmp_path, monkeypatch, field, value):
    archive, reg, meta, proof, verified, _ = archival_fixture(tmp_path, monkeypatch)
    archive[field] = value
    with pytest.raises(ValueError): env.archived_prefreeze_gate(tmp_path, archive, reg, meta, proof, verified)


def test_prefreeze_working_bytes_changed_after_sealing_fail(tmp_path, monkeypatch):
    archive, reg, meta, proof, verified, _ = archival_fixture(tmp_path, monkeypatch)
    (tmp_path / 'results/gated_learning_prefreeze_finalization.json').write_text('{}')
    with pytest.raises(ValueError): env.archived_prefreeze_gate(tmp_path, archive, reg, meta, proof, verified)


def test_original_registration_timestamp_cannot_precede_finalization(tmp_path, monkeypatch):
    archive, reg, meta, proof, verified, _ = archival_fixture(tmp_path, monkeypatch)
    reg['registered_at_utc'] = '2019-12-31T00:00:00+00:00'
    archive['prefreeze_to_independent_verification_utc_order']['registered'] = reg['registered_at_utc']
    with pytest.raises(ValueError): env.archived_prefreeze_gate(tmp_path, archive, reg, meta, proof, verified)
