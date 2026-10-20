"""Synthetic-only scope, counts, publication, archive and failure-retention gates."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drosolearn import gated_factors_envelope as env

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return {'study_id': env.STUDY_ID, 'module': env.MODULE, 'protocol': env.PROTOCOL,
        'scope': 'retrospective_descriptive_gated_factors', 'external_manifest': None,
        'model_seeds': list(range(17601, 17611)), 'training_trials': 400,
        'source_files': sorted(env.SOURCE_FILES), 'parent_files': sorted(env.PARENT_FILES),
        'interpretation': {key: False for key in ('new_human_experiment',
            'shared_neural_mechanism_claim', 'education_effect_claim',
            'new_training', 'new_seed_draws', 'new_effect_hypothesis_test')}}


def summary():
    return {'study_id': env.STUDY_ID, 'status': 'completed',
        'scope': 'retrospective_descriptive_gated_factors', 'counts': {
            'analyzed_parent_trial_records': 120000, 'analyzed_destination_effects': 480000,
            'reference_gate_paths': 60, 'comparison_paths': 240,
            'paired_trial_records': 96000, 'measurement_records': 2880, 'numeric_arrays': 14,
            'paired_post_trial_records': 20, 'eligible_both_Go_trials': 7, 'corner_destination_effects': 224,
            **{key: 0 for key in ('new_training_trials', 'new_head_update_calls', 'new_internal_update_calls',
                'new_value_update_calls', 'new_random_draws', 'fresh_model_constructions', 'new_optimization_calls')}}}


def test_exact_new_runtime_and_direct_parent_closures():
    env.validate_envelope(ROOT, configuration(), ROOT / env.CONFIG)
    assert len(env.SOURCE_FILES) == 20 and len(env.PARENT_FILES) == 12
    assert 'src/drosolearn/gated_pathways_envelope.py' in env.REUSED_SOURCE
    assert 'src/drosolearn/gated_learning.py' not in env.SOURCE_FILES
    assert 'src/drosolearn/gated_pathways.py' not in env.SOURCE_FILES
    assert {'results/gated_learning_v1/arrays.npz', 'results/gated_pathways_v1/arrays.npz'} <= env.PARENT_FILES


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
    env.validate_counts(configuration(), summary(), 2880)
    empty = summary()
    for key in ('paired_post_trial_records', 'eligible_both_Go_trials', 'corner_destination_effects'):
        empty['counts'][key] = 0
    env.validate_counts(configuration(), empty, 2880)


@pytest.mark.parametrize('field,value', [
    ('analyzed_parent_trial_records', 96000), ('analyzed_destination_effects', 384000),
    ('reference_gate_paths', 300), ('comparison_paths', 60), ('paired_trial_records', 120000),
    ('measurement_records', 1800), ('numeric_arrays', 12), ('new_training_trials', 1),
    ('new_head_update_calls', 1), ('new_internal_update_calls', False), ('new_value_update_calls', 1),
    ('new_random_draws', 1), ('fresh_model_constructions', 1), ('new_optimization_calls', 1),
    ('paired_post_trial_records', -1), ('paired_post_trial_records', 96001),
    ('eligible_both_Go_trials', 21), ('eligible_both_Go_trials', True),
    ('corner_destination_effects', 225), ('corner_destination_effects', 224.0),
])
def test_wrong_axes_any_new_work_and_invalid_bridge_counts_fail(field, value):
    changed = summary(); changed['counts'][field] = value
    with pytest.raises(ValueError): env.validate_counts(configuration(), changed, 2880)


def test_fixed_parent_cohort_budget_and_exact_count_keys():
    cfg = configuration(); cfg['model_seeds'][0] = 17501
    with pytest.raises(ValueError): env.validate_counts(cfg, summary(), 2880)
    cfg = configuration(); cfg['training_trials'] = 800
    with pytest.raises(ValueError): env.validate_counts(cfg, summary(), 2880)
    with pytest.raises(ValueError): env.validate_counts(configuration(), summary(), 2879)
    changed = summary(); changed['counts']['extra'] = 0
    with pytest.raises(ValueError): env.validate_counts(configuration(), changed, 2880)


def test_parent_entry_reuses_inherited_byte_gate_before_any_scientific_import(tmp_path, monkeypatch):
    from drosolearn import gated_pathways_envelope as prior
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


def archival_fixture(tmp_path, monkeypatch):
    """Build fake prior bytes; never open, replay or aggregate the actual study."""
    from drosolearn import gated_pathways_envelope as prior
    def put(name, value):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if not isinstance(value, str) else value)
        return path
    times = ['2020-01-01T00:0' + str(i) + ':00+00:00' for i in range(8)]
    for name in env.REUSED_SOURCE: put(name, 'prior frozen source')
    sources = {f'src/prior_{i}.py': env.sha(put(f'src/prior_{i}.py', 'source')) for i in range(19)}
    parents = {f'data/prior_{i}.json': env.sha(put(f'data/prior_{i}.json', '{}')) for i in range(11)}
    reg = {'study_id': 'gated_pathways_v1', 'source_git_commit': 'b' * 40,
        'registered_at_utc': times[2], 'source_files': sources, 'parent_files': parents, 'gate': {'fake': True}}
    reg_path = put('results/registration_gated_pathways_v1.json', reg)
    prefix = 'results/gated_pathways_v1/'
    put(prefix + 'registration.json', reg)
    proof_path = put('results/gated_pathways_publication_proof.json', {'checked_at_utc': times[3]})
    counts = {key: 0 for key in ('new_training_trials', 'new_head_update_calls', 'new_internal_update_calls',
        'new_value_update_calls', 'new_random_draws', 'fresh_model_constructions', 'new_optimization_calls')}
    counts.update(analyzed_parent_trial_records=120000, analyzed_destination_effects=480000,
                  numeric_arrays=12, measurement_records=1800)
    interpretation = {'parent_H_gain': False, 'parent_H_acquisition': False, 'parent_H_protection': True,
        'new_confirmatory_hypothesis': False, 'exact_path_accounting_not_unique_causal_explanation': True,
        'reference_gate_is_common_time_split_not_all_methods_knowledge': True}
    put(prefix + 'summary.json', {'study_id': 'gated_pathways_v1', 'status': 'completed', 'counts': counts,
        'interpretation': interpretation, 'validation': {'all_exact_accounting_gates_passed': True}})
    meta = {'study_id': 'gated_pathways_v1', 'status': 'completed', 'stage': 'completed', 'analysis_only': True,
        'registration_sha256': env.sha(reg_path), 'source_git_commit': reg['source_git_commit'], 'counts': counts,
        'publication_proof': 'results/gated_pathways_publication_proof.json',
        'publication_proof_sha256': env.sha(proof_path), 'started_at_utc': times[4], 'completed_at_utc': times[5]}
    put(prefix + 'metadata.json', meta)
    flags = ('all_factual_head_prefixes_and_parent_effects_reconstructed',
        'all_methods_use_same_reference_prefeedback_time_boundary',
        'own_method_observations_used_for_known_unknown_categories',
        'last_discovery_trial_pre_and_next_trial_post_checked',
        'positive_negative_masses_and_endpoint_contributions_separated',
        'costs_Go_counts_and_nullable_descriptive_ratios_checked',
        'all_four_controls_and_all_thirty_families_checked',
        'parent_full_autonomous_replay_and_byte_gate_reused_by_frozen_hashes',
        'no_counterfactual_direction_policy_rng_graph_or_solver_calls')
    checks = {key: True for key in flags}
    checks.update(arrays=12, parent_trial_records_checked=120000, destination_effects_checked=480000,
        measurement_records_checked=1800, new_training_trials=0, new_random_draws=0, new_optimization_calls=0)
    verification = {'study_id': 'gated_pathways_v1', 'status': 'passed', 'scientific_producer_imported': False,
        'new_helper_imported': False, 'registration_sha256': env.sha(reg_path),
        'summary_sha256': env.sha(tmp_path / (prefix + 'summary.json')), 'counts': checks, 'verified_at_utc': times[6]}
    verify_path = put(prefix + 'verification.json', verification)
    put(prefix + 'arrays.npz', 'fake archived raw bytes')
    put(prefix + 'measurements.csv', 'fake archived CSV bytes')
    put('results/gated_learning_v1/arrays.npz', 'fake original learning raw bytes')
    artifacts = {name: env.sha(tmp_path / (prefix + name)) for name in env.PRIMARY}
    put(prefix + 'artifact_hashes.json', artifacts)
    baseline = put('results/gated_pathways_baseline_preservation.json', {'status': 'passed',
        'allowed_evolving_indexes': sorted(env.EVOLVING_INDEXES), 'verified_at_utc': times[0]})
    tests = {'status': 'passed', 'frozen_source_hashes': sources, 'parent_byte_gate': reg['gate'],
        'unique_passed_tests': 256, 'new_unique_passed_tests': 142, 'unchanged_related_regression_passed_tests': 114,
        'actual_new_saved_path_aggregation_before_freeze': 0, 'checked_at_utc': times[1]}
    put('results/gated_pathways_prefreeze_validation.json', tests)
    put('results/gated_pathways_prefreeze_tests/gated_pathways_prefreeze_final2.xml', '<testsuites/>')
    status = {key: False for key in ('level4_similarity_supported', 'shared_neural_mechanism_verified',
        'human_shell_H_similarity', 'human_shell_H_prediction', 'gated_learning_H_gain', 'gated_learning_H_acquisition')}
    status['gated_learning_H_protection'] = True
    for name in env.EVOLVING_INDEXES: put(name, status)
    prepaths = ['results/gated_pathways_prefreeze_validation.json', 'results/gated_pathways_baseline_preservation.json',
        'results/gated_pathways_prefreeze_tests/gated_pathways_prefreeze_final2.xml']
    paths = set(sources) | set(prepaths) | {prefix + name for name in (*env.PRIMARY, 'verification.json', 'artifact_hashes.json')} | env.EVOLVING_INDEXES
    archive = {'status': 'passed', 'all_old_files_Git_blob_hash_preserved_except_three_indexes': True,
        'scientific_verification_sha256': env.sha(verify_path), 'registration_sha256': env.sha(reg_path),
        'source_git_commit': reg['source_git_commit'], 'counts': counts, 'baseline_hash_receipt_sha256': env.sha(baseline),
        'allowed_evolving_indexes': sorted(env.EVOLVING_INDEXES),
        'prefreeze_receipts_committed_source_sha256': {name: env.sha(tmp_path / name) for name in prepaths},
        'prefreeze_publication_analysis_verification_utc_order': times[:7], 'verified_at_utc': times[7],
        'current_file_sha256': {name: env.sha(tmp_path / name) for name in paths}}
    archive_path = put('results/gated_pathways_archive_validation.json', archive)
    calls = []
    monkeypatch.setattr(prior, 'parent_gate', lambda *args: {'inherited': True})
    monkeypatch.setattr(prior, 'publication', lambda *args: ('results/gated_pathways_publication_proof.json', env.read(proof_path)))
    monkeypatch.setattr(env, 'committed', lambda root, commit, values: calls.append((commit, values)))
    monkeypatch.setattr(env, 'ancestor', lambda *args: None)
    monkeypatch.setattr(env, 'git', lambda root, *args: (env.BASE_COMMIT.encode() if args[0] == 'rev-parse' else json.dumps(status).encode()))
    return archive_path, archive, calls


def test_completed_parent_archive_checks_prefreeze_source_snapshot_without_science(tmp_path, monkeypatch):
    _, archive, calls = archival_fixture(tmp_path, monkeypatch)
    gate = env.parent_gate(tmp_path)
    assert gate['parent_gate_no_new_learner_graph_seed_or_optimizer_calls'] is True
    assert gate['parent_saved_accounting_and_original_autonomous_replay_reused_not_rerun'] is True
    assert gate['pinned_baseline_files'] == 23
    assert any(set(values) == set(archive['prefreeze_receipts_committed_source_sha256']) for _, values in calls)


@pytest.mark.parametrize('field,value', [
    ('scientific_verification_sha256', '0' * 64), ('baseline_hash_receipt_sha256', '0' * 64),
    ('source_git_commit', 'c' * 40), ('allowed_evolving_indexes', ['README.md']),
    ('prefreeze_publication_analysis_verification_utc_order', []),
])
def test_broken_completed_parent_archive_bindings_fail(tmp_path, monkeypatch, field, value):
    path, archive, _ = archival_fixture(tmp_path, monkeypatch)
    archive[field] = value; path.write_text(json.dumps(archive))
    with pytest.raises(ValueError): env.parent_gate(tmp_path)


def test_parent_prefreeze_working_bytes_changed_after_sealing_fail(tmp_path, monkeypatch):
    archival_fixture(tmp_path, monkeypatch)
    path = tmp_path / 'results/gated_pathways_prefreeze_tests/gated_pathways_prefreeze_final2.xml'
    path.write_text('altered archived receipt')
    with pytest.raises(ValueError): env.parent_gate(tmp_path)
