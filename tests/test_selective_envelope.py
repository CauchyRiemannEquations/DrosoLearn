"""Invented Git/byte fixtures only; no actual study seeds or model responses."""
from datetime import datetime, timezone
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from drosolearn import selective_envelope as env


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        path.write_text(json.dumps(value, sort_keys=True), encoding='utf8')
    else:
        path.write_text(value, encoding='utf8')
    return path


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE).decode().strip()


def commit(root):
    git(root, 'add', '.')
    git(root, 'commit', '-q', '-m', 'Synthetic integrity fixture')
    return git(root, 'rev-parse', 'HEAD')


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / 'repository'
    root.mkdir()
    git(root, 'init', '-q', '--initial-branch=main')
    git(root, 'config', 'user.name', 'Integrity Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'config', 'core.autocrlf', 'false')
    for name in env.SOURCE_FILES | env.GRAPH_FILES | env.INPUT_FILES:
        put(root, name, 'synthetic file\n')
    put(root, 'historical-receipt.json', {'invented': True})
    source = commit(root)
    study, prefix = 'signed_spillover_v1', 'results/signed_spillover_v1/'
    registration = {'study_id': study, 'registered_at_utc': '2020-01-01T00:00:00+00:00',
        'source_git_commit': source, 'source_files': env.inventory(root, sorted(env.REUSED_SOURCE)),
        'parent_files': env.inventory(root, ['historical-receipt.json'])}
    reg_path = put(root, 'results/registration_' + study + '.json', registration)
    put(root, prefix + 'registration.json', registration)
    counts = {'new_training_trials': 0, 'new_head_update_calls': 0,
        'new_internal_update_calls': 0, 'new_value_update_calls': 0, 'new_random_draws': 0,
        'fresh_model_constructions': 0, 'analyzed_parent_trial_records': 120000}
    summary_path = put(root, prefix + 'summary.json', {'study_id': study,
        'status': 'completed', 'counts': counts})
    meta = {'study_id': study, 'status': 'completed', 'stage': 'completed',
        'started_at_utc': '2020-01-01T00:00:01+00:00',
        'completed_at_utc': '2020-01-01T00:00:02+00:00',
        'registration_sha256': env.sha(reg_path), 'source_git_commit': source, 'counts': counts}
    put(root, prefix + 'metadata.json', meta)
    replay = {key: True for key in env.PARENT_REPLAY_TRUE}
    replay.update(parent_trial_records_checked=120000, trial_destination_effects_checked=480000,
                  checkpoint_source_destination_effects_checked=24000)
    verify = {'status': 'passed', 'study_id': study, 'scientific_producer_imported': False,
        'new_helper_imported': False, 'verified_at_utc': '2020-01-01T00:00:03+00:00',
        'registration_sha256': env.sha(reg_path), 'summary_sha256': env.sha(summary_path), 'counts': replay}
    verification_path = put(root, prefix + 'verification.json', verify)
    put(root, prefix + 'arrays.npz', 'synthetic historical bytes, not real arrays')
    put(root, prefix + 'measurements.csv', 'invented,value\n0,1\n')
    put(root, prefix + 'artifact_hashes.json', {name: env.sha(root / prefix / name) for name in env.PRIMARY})
    prior = {'study_id': 'spillover_tradeoff_v1', 'status': 'completed',
        'H_tradeoff': {'supported': False},
        'H_acquisition': {'supported': False, 'acquisition_pass_seeds': 0},
        'budget_validation': {key: True for key in ('conditional_scalar_matching_only',
            'squared_step_sum_valid_all', 'head_endpoint_within_path_all', 'common_bounds_valid_all',
            'all_static_numeric_gates_passed', 'analytic_probability_ceiling_valid_all')}}
    prior['budget_validation']['actual_cumulative_cost_equalized'] = False
    prior_path = put(root, 'results/spillover_tradeoff_v1/summary.json', prior)
    put(root, 'results/spillover_tradeoff_v1/verification.json', {'status': 'passed',
        'scientific_producer_imported': False, 'summary_sha256': env.sha(prior_path)})
    put(root, 'results/signed_spillover_publication_proof.json', {'invented': True})
    receipt_path = put(root, 'results/signed_spillover_baseline_preservation.json', {'status': 'passed'})
    for name in env.EVOLVING_INDEXES:
        put(root, name, {'level4_similarity_supported': False, 'shared_neural_mechanism_verified': False,
            'human_shell_H_similarity': False, 'human_shell_H_prediction': False})
    snapshot_names = set(registration['source_files']) | {'historical-receipt.json'} | env.EVOLVING_INDEXES
    snapshot_names |= {prefix + name for name in (*env.PRIMARY, 'verification.json', 'artifact_hashes.json')}
    snapshot_names |= {'results/spillover_tradeoff_v1/summary.json', 'results/spillover_tradeoff_v1/verification.json'}
    snapshot = env.inventory(root, sorted(snapshot_names))
    put(root, 'results/signed_spillover_archive_validation.json', {'status': 'passed',
        'all_old_baseline_files_literal_preserved_except_three_indexes': True,
        'literal_baseline_receipt_sha256': env.sha(receipt_path),
        'scientific_verification_sha256': env.sha(verification_path),
        'registration_sha256': env.sha(reg_path), 'source_git_commit': source,
        'counts': counts, 'current_file_sha256': snapshot})
    cfg = {'study_id': env.STUDY_ID, 'module': env.MODULE, 'protocol': env.PROTOCOL,
        'scope': 'synthetic_integrity_fixture', 'source_files': sorted(env.SOURCE_FILES),
        'parent_files': sorted(env.PARENT_FILES), 'graph_files': sorted(env.GRAPH_FILES),
        'input_files': sorted(env.INPUT_FILES), 'external_manifest': None,
        'interpretation': {'new_human_experiment': False, 'shared_neural_mechanism_claim': False,
                          'education_effect_claim': False}}
    config = put(root, env.CONFIG, cfg)
    baseline = commit(root)
    git(root, 'tag', env.BASE_TAG)
    monkeypatch.setattr(env, 'BASE_COMMIT', baseline)
    calls = []
    def validate_config(value):
        calls.append('validate')
    def produce(value, current_root):
        calls.append('produce')
        return {'synthetic_values': np.array([1., 2.])}, [{'index': 0, 'value': 1.}]
    def summarize(value, arrays, rows):
        calls.append('summarize')
        return {'counts': {'synthetic_records': len(rows)}, 'scope': 'integrity_fixture_only'}
    module = SimpleNamespace(validate_config=validate_config, produce=produce, summarize=summarize)
    def import_module(name):
        assert name == env.MODULE
        calls.append('import')
        return module
    monkeypatch.setattr(env, 'importlib', SimpleNamespace(import_module=import_module))
    return SimpleNamespace(root=root, config=config, cfg=cfg, baseline=baseline,
        registration=root / env.REGISTRATION, result=root / env.RESULT,
        proof=root / 'results/selective_spillover_publication_proof.json', calls=calls, module=module)


def publish(item):
    reg = env.register(item.root, item.config, item.registration)
    published = commit(item.root)
    proof = {'study_id': env.STUDY_ID, 'source_commit': reg['source_git_commit'],
        'published_commit': published, 'registration_sha256': env.sha(item.registration),
        'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'remote_bytes_match': True,
        'visibility': 'private', 'url': env.REPOSITORY + '/blob/' + published + '/' + env.REGISTRATION}
    put(item.root, item.proof.relative_to(item.root).as_posix(), proof)
    return reg


@pytest.fixture
def published(repository):
    publish(repository)
    return repository


@pytest.fixture
def completed(published):
    env.run(published.root, published.config, published.registration, published.proof, published.result)
    return published


def test_exact_finite_inventories():
    assert len(env.SOURCE_FILES) == len(env.REUSED_SOURCE) + 11
    assert len(env.PARENT_FILES) == 8
    assert len(env.GRAPH_FILES) == 4 and len(env.INPUT_FILES) == 2
    assert not any(name.endswith(('/arrays.npz', '/measurements.csv')) for name in env.PARENT_FILES)


@pytest.mark.parametrize('payload', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'])
def test_json_duplicate_and_nonfinite_rejected(tmp_path, payload):
    path = tmp_path / 'bad.json'
    path.write_text(payload)
    with pytest.raises(ValueError):
        env.read(path)


@pytest.mark.parametrize('name', ['../outside', 'a/../file', '/absolute', 'C:/absolute',
                                'a\\file', './file', 'a//file', '', 'missing'])
def test_noncanonical_or_missing_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        env.safe(tmp_path, name)


@pytest.mark.parametrize('key', ['source_files', 'parent_files', 'graph_files', 'input_files'])
def test_inventory_omission_duplicate_rejected(repository, key):
    cfg = {**repository.cfg, key: repository.cfg[key][1:] + [repository.cfg[key][-1]]}
    with pytest.raises(ValueError, match='inventory'):
        env.validate_envelope(repository.root, cfg, repository.config)


@pytest.mark.parametrize('key', ['new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim'])
def test_unbounded_interpretation_rejected(repository, key):
    cfg = {**repository.cfg, 'interpretation': {**repository.cfg['interpretation'], key: True}}
    with pytest.raises(ValueError, match='interpretation'):
        env.validate_envelope(repository.root, cfg, repository.config)


def test_development_scope_rejected(repository):
    with pytest.raises(ValueError, match='fixture'):
        env.validate_envelope(repository.root, {**repository.cfg, 'scope': 'integration_test'}, repository.config)


def test_external_cache_dependency_rejected(repository):
    with pytest.raises(ValueError, match='external'):
        env.validate_envelope(repository.root, {**repository.cfg, 'external_manifest': 'data/other.json'}, repository.config)


def test_registration_freezes_four_committed_inventories(repository):
    reg = env.register(repository.root, repository.config, repository.registration)
    assert reg['source_git_commit'] == repository.baseline
    for group in ('source_files', 'parent_files', 'graph_files', 'input_files'):
        assert reg[group] == env.inventory(repository.root, repository.cfg[group])
    assert repository.calls == ['import', 'validate']
    assert not repository.result.exists()
    assert reg['gate']['prior_readonly_analysis_not_new_learning'] is True
    assert reg['gate']['prior_spillover_learning_not_qualified'] is True
    assert reg['gate']['parent_archive_raw_arrays_and_csv_literal_checked'] is True


@pytest.mark.parametrize('group', ['source_files', 'parent_files', 'graph_files', 'input_files'])
def test_uncommitted_bytes_fail_before_import(repository, group):
    name = next(name for name in repository.cfg[group] if name != env.CONFIG)
    with (repository.root / name).open('a') as handle:
        handle.write('\nchanged')
    with pytest.raises(ValueError, match='Git and working'):
        env.register(repository.root, repository.config, repository.registration)
    assert repository.calls == []


def test_sealed_reused_source_changed_and_committed_still_rejected(repository):
    put(repository.root, sorted(env.REUSED_SOURCE)[0], 'changed source')
    commit(repository.root)
    with pytest.raises(ValueError, match='Git and working'):
        env.register(repository.root, repository.config, repository.registration)
    assert repository.calls == []


@pytest.mark.parametrize('name', ['results/signed_spillover_v1/arrays.npz',
    'results/signed_spillover_v1/measurements.csv', 'historical-receipt.json'])
def test_historical_raw_or_receipt_changed_not_hidden_by_cached_gate(repository, name):
    put(repository.root, name, 'modified historical bytes')
    commit(repository.root)
    with pytest.raises(ValueError, match='immutable baseline snapshot'):
        env.register(repository.root, repository.config, repository.registration)
    assert repository.calls == []


def reseal_parent(item):
    """Make a deliberately internally consistent but semantically wrong old study."""
    root, prefix = item.root, 'results/signed_spillover_v1/'
    reg_path = root / 'results/registration_signed_spillover_v1.json'
    meta_path, summary_path = root / prefix / 'metadata.json', root / prefix / 'summary.json'
    verification = env.read(root / prefix / 'verification.json')
    verification['summary_sha256'] = env.sha(summary_path)
    verification_path = put(root, prefix + 'verification.json', verification)
    put(root, prefix + 'artifact_hashes.json', {name: env.sha(root / prefix / name) for name in env.PRIMARY})
    archive = env.read(root / 'results/signed_spillover_archive_validation.json')
    archive['scientific_verification_sha256'] = env.sha(verification_path)
    archive['registration_sha256'] = env.sha(reg_path)
    archive['counts'] = env.read(summary_path)['counts']
    archive['current_file_sha256'] = env.inventory(root, sorted(archive['current_file_sha256']))
    put(root, 'results/signed_spillover_archive_validation.json', archive)


@pytest.mark.parametrize('defect', ['failed_status', 'imported_producer', 'imported_helper',
    'incomplete_replay', 'new_learning_in_old_audit', 'wrong_record_count',
    'failed_archive', 'modified_old_files', 'negative_tradeoff_rewritten',
    'negative_acquisition_rewritten', 'invalid_prior_bounds', 'human_positive_claim'])
def test_semantic_parent_gates_block_even_after_consistent_resealing(repository, monkeypatch, defect):
    root, prefix = repository.root, 'results/signed_spillover_v1/'
    meta = env.read(root / prefix / 'metadata.json')
    summary = env.read(root / prefix / 'summary.json')
    verification = env.read(root / prefix / 'verification.json')
    archive = env.read(root / 'results/signed_spillover_archive_validation.json')
    prior = env.read(root / 'results/spillover_tradeoff_v1/summary.json')
    if defect == 'failed_status':
        meta['status'] = 'failed'
    elif defect == 'imported_producer':
        verification['scientific_producer_imported'] = True
    elif defect == 'imported_helper':
        verification['new_helper_imported'] = True
    elif defect == 'incomplete_replay':
        verification['counts']['obtained_reward_only_nogo_zero_checked'] = False
    elif defect == 'new_learning_in_old_audit':
        summary['counts']['new_training_trials'] = 1
        meta['counts'] = summary['counts']
    elif defect == 'wrong_record_count':
        verification['counts']['parent_trial_records_checked'] -= 1
    elif defect == 'failed_archive':
        archive['status'] = 'failed'
    elif defect == 'modified_old_files':
        archive['all_old_baseline_files_literal_preserved_except_three_indexes'] = False
    elif defect == 'negative_tradeoff_rewritten':
        prior['H_tradeoff']['supported'] = True
    elif defect == 'negative_acquisition_rewritten':
        prior['H_acquisition']['acquisition_pass_seeds'] = 1
    elif defect == 'invalid_prior_bounds':
        prior['budget_validation']['common_bounds_valid_all'] = False
    elif defect == 'human_positive_claim':
        status = env.read(root / 'docs/repository-status.json')
        status['level4_similarity_supported'] = True
        put(root, 'docs/repository-status.json', status)
    put(root, prefix + 'metadata.json', meta)
    put(root, prefix + 'summary.json', summary)
    put(root, prefix + 'verification.json', verification)
    put(root, 'results/signed_spillover_archive_validation.json', archive)
    prior_path = put(root, 'results/spillover_tradeoff_v1/summary.json', prior)
    pv = env.read(root / 'results/spillover_tradeoff_v1/verification.json')
    pv['summary_sha256'] = env.sha(prior_path)
    put(root, 'results/spillover_tradeoff_v1/verification.json', pv)
    reseal_parent(repository)
    baseline = commit(root)
    git(root, 'tag', '-f', env.BASE_TAG)
    monkeypatch.setattr(env, 'BASE_COMMIT', baseline)
    with pytest.raises(ValueError):
        env.register(root, repository.config, repository.registration)
    assert repository.calls == []


def test_register_never_overwrites(repository):
    env.register(repository.root, repository.config, repository.registration)
    before = repository.registration.read_bytes()
    with pytest.raises(ValueError, match='New fixed'):
        env.register(repository.root, repository.config, repository.registration)
    assert repository.registration.read_bytes() == before


def test_published_round_trip_and_verification_never_import_producer(completed):
    before = list(completed.calls)
    cfg, reg, meta = env.verify_envelope(completed.root, completed.config,
        completed.registration, completed.result, completed.proof)
    assert cfg == completed.cfg and reg['source_git_commit'] == completed.baseline
    assert meta['counts'] == {'synthetic_records': 1}
    assert meta['measurement_records'] == meta['numeric_arrays'] == 1
    assert completed.calls == before


@pytest.mark.parametrize('field,value', [('remote_bytes_match', False), ('visibility', 'public'),
    ('registration_sha256', '0' * 64), ('source_commit', '0' * 40),
    ('published_commit', None), ('study_id', 'another'), ('url', 'https://example.invalid')])
def test_wrong_publication_blocks_before_import(published, field, value):
    proof = env.read(published.proof)
    proof[field] = value
    put(published.root, published.proof.relative_to(published.root).as_posix(), proof)
    before = list(published.calls)
    with pytest.raises(ValueError):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert published.calls == before and not published.result.exists()


@pytest.mark.parametrize('stamp', ['2020-01-01T00:00:00+00:00', '2999-01-01T00:00:00+00:00',
                                  '2026-01-01T00:00:00', '2026-01-01T00:00:00+09:00'])
def test_wrong_publication_chronology_blocks_run(published, stamp):
    proof = env.read(published.proof)
    proof['checked_at_utc'] = stamp
    put(published.root, published.proof.relative_to(published.root).as_posix(), proof)
    before = list(published.calls)
    with pytest.raises(ValueError):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert published.calls == before and not published.result.exists()


def test_actual_failure_sealed_and_retry_prohibited(published):
    def fail(cfg, root):
        raise RuntimeError('synthetic producer failure')
    published.module.produce = fail
    with pytest.raises(RuntimeError, match='synthetic'):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    meta = env.read(published.result / 'metadata.json')
    assert meta['status'] == 'failed' and meta['error_type'] == 'RuntimeError'
    artifacts = env.read(published.result / 'artifact_hashes.json')
    assert set(artifacts) == {'registration.json', 'metadata.json'}
    before = {name: (published.result / name).read_bytes() for name in artifacts}
    with pytest.raises(ValueError, match='New fixed result'):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert all((published.result / name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize('bad', [np.array([np.nan]), np.array([np.inf]), np.array(['x'], dtype=object), np.array(['x'])])
def test_nonfinite_non_numeric_arrays_fail_sealed(published, bad):
    published.module.produce = lambda cfg, root: ({'bad': bad}, [{'x': 1}])
    with pytest.raises(ValueError, match='Finite numeric'):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert env.read(published.result / 'metadata.json')['status'] == 'failed'
    assert not (published.result / 'arrays.npz').exists()


@pytest.mark.parametrize('artifact', env.PRIMARY)
def test_primary_byte_change_rejected(completed, artifact):
    with (completed.result / artifact).open('ab') as handle:
        handle.write(b'\n')
    with pytest.raises(ValueError):
        env.verify_envelope(completed.root, completed.config, completed.registration, completed.result, completed.proof)


@pytest.mark.parametrize('field', ['numeric_arrays', 'measurement_records'])
def test_metadata_count_change_rejected_after_rehash(completed, field):
    path = completed.result / 'metadata.json'
    meta = env.read(path)
    meta[field] += 1
    path.write_text(json.dumps(meta))
    hashes = env.read(completed.result / 'artifact_hashes.json')
    hashes['metadata.json'] = env.sha(path)
    (completed.result / 'artifact_hashes.json').write_text(json.dumps(hashes))
    with pytest.raises(ValueError):
        env.verify_envelope(completed.root, completed.config, completed.registration, completed.result, completed.proof)


def test_duplicate_csv_column_rejected_after_rehash(completed):
    path = completed.result / 'measurements.csv'
    path.write_text('index,index\n0,1\n')
    hashes = env.read(completed.result / 'artifact_hashes.json')
    hashes['measurements.csv'] = env.sha(path)
    (completed.result / 'artifact_hashes.json').write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match='Unique saved'):
        env.verify_envelope(completed.root, completed.config, completed.registration, completed.result, completed.proof)


def test_output_already_completed_blocks_duplicate(completed):
    before = (completed.result / 'metadata.json').read_bytes()
    with pytest.raises(ValueError, match='New fixed result'):
        env.run(completed.root, completed.config, completed.registration, completed.proof, completed.result)
    assert (completed.result / 'metadata.json').read_bytes() == before
