"""Invented Git/file fixtures only; no actual parent arrays or study arithmetic."""
from datetime import datetime, timezone
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from drosolearn import signed_envelope as env


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
    git(root, 'commit', '-q', '-m', 'Synthetic-integrity-fixture')
    return git(root, 'rev-parse', 'HEAD')


def seal_parent(item, monkeypatch):
    """Bind freshly invented receipts before testing a semantic parent defect."""
    root, prefix = item.root, 'results/spillover_tradeoff_v1/'
    verification = env.read(root / prefix / 'verification.json')
    verification['summary_sha256'] = env.sha(root / prefix / 'summary.json')
    put(root, prefix + 'verification.json', verification)
    put(root, prefix + 'artifact_hashes.json', {name: env.sha(root / prefix / name) for name in env.PRIMARY})
    archive = env.read(root / 'results/spillover_archive_validation.json')
    archive['scientific_verification_sha256'] = env.sha(root / prefix / 'verification.json')
    archive['current_file_sha256'] = {name: env.sha(root / name) for name in item.snapshot_names}
    put(root, 'results/spillover_archive_validation.json', archive)
    item.baseline = commit(root)
    git(root, 'tag', '-f', env.BASE_TAG)
    monkeypatch.setattr(env, 'BASE_COMMIT', item.baseline)


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / 'repository'
    root.mkdir()
    git(root, 'init', '-q', '--initial-branch=main')
    git(root, 'config', 'user.name', 'Integrity Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'config', 'core.autocrlf', 'false')
    for name in env.SOURCE_FILES | env.EVOLVING_INDEXES:
        put(root, name, 'invented file\n')
    groups = {key: {'parent/' + key + '.txt': None} for key in
              ('source_files', 'parent_files', 'graph_files', 'input_files')}
    for values in groups.values():
        for name in values:
            put(root, name, 'invented old closure\n')
            values[name] = env.sha(root / name)
    source = commit(root)
    prefix = 'results/spillover_tradeoff_v1/'
    registration = {'study_id': 'spillover_tradeoff_v1',
        'registered_at_utc': '2020-01-01T00:00:00+00:00', 'source_git_commit': source, **groups}
    reg_path = put(root, 'results/registration_spillover_tradeoff_v1.json', registration)
    put(root, prefix + 'registration.json', registration)
    counts = {'invented_parent_records': 1}
    summary_path = put(root, prefix + 'summary.json', {'study_id': 'spillover_tradeoff_v1',
        'status': 'completed', 'counts': counts,
        'H_tradeoff': {'supported': False, 'numeric_spillover_qualified_all': True},
        'H_acquisition': {'supported': False, 'acquisition_pass_seeds': 0,
                          'numeric_spillover_qualified_all': True},
        'budget_validation': {**{key: True for key in (
            'conditional_scalar_matching_only', 'squared_step_sum_valid_all',
            'head_endpoint_within_path_all', 'common_bounds_valid_all',
            'all_static_numeric_gates_passed', 'analytic_probability_ceiling_valid_all')},
            'actual_cumulative_cost_equalized': False}})
    put(root, prefix + 'metadata.json', {'study_id': 'spillover_tradeoff_v1',
        'status': 'completed', 'stage': 'completed', 'source_git_commit': source,
        'registration_sha256': env.sha(reg_path), 'counts': counts,
        'started_at_utc': '2020-01-01T00:00:01+00:00',
        'completed_at_utc': '2020-01-01T00:00:02+00:00'})
    put(root, prefix + 'verification.json', {'study_id': 'spillover_tradeoff_v1',
        'status': 'passed', 'scientific_producer_imported': False,
        'registration_sha256': env.sha(reg_path), 'summary_sha256': env.sha(summary_path),
        'verified_at_utc': '2020-01-01T00:00:03+00:00',
        'counts': {**{key: True for key in env.REPLAY_TRUE},
            'independently_replayed_training_actions': 120000, 'checkpoint_static_probes': 6000,
            'internal_update_calls': 0, 'value_update_calls': 0,
            'actual_autonomous_costs_equalized': False, 'extra_solver_cost_equalized': False}})
    put(root, prefix + 'arrays.npz', 'invented binary bytes; not loaded in gate\n')
    put(root, prefix + 'measurements.csv', 'invented,csv\n1,2\n')
    put(root, prefix + 'artifact_hashes.json', {name: env.sha(root / prefix / name) for name in env.PRIMARY})
    put(root, 'results/spillover_tradeoff_publication_proof.json', {'fixture': 'pinned old proof bytes'})
    snapshot_names = set(env.EVOLVING_INDEXES)
    for values in groups.values():
        snapshot_names.update(values)
    snapshot_names.update(prefix + name for name in (*env.PRIMARY, 'verification.json', 'artifact_hashes.json'))
    put(root, 'results/spillover_archive_validation.json', {'status': 'passed',
        'all_old_baseline_files_literal_preserved_except_three_indexes': True,
        'source_git_commit': source, 'registration_sha256': env.sha(reg_path), 'counts': counts,
        'scientific_verification_sha256': env.sha(root / prefix / 'verification.json'),
        'current_file_sha256': {name: env.sha(root / name) for name in snapshot_names}})
    cfg = {'study_id': env.STUDY_ID, 'module': env.MODULE, 'protocol': env.PROTOCOL,
        'scope': 'retrospective_descriptive_signed_spillover',
        'source_files': sorted(env.SOURCE_FILES), 'parent_files': sorted(env.PARENT_FILES),
        'external_manifest': None,
        'interpretation': {key: False for key in (
            'new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim',
            'new_training', 'new_seed_draws', 'new_effect_hypothesis_test')}}
    config = put(root, env.CONFIG, cfg)
    baseline = commit(root)
    git(root, 'tag', env.BASE_TAG)
    monkeypatch.setattr(env, 'BASE_COMMIT', baseline)
    calls = []
    def validate_config(value):
        calls.append('validate')
    def produce(value, current_root):
        calls.append('produce')
        return {'invented_values': np.array([1., 2.])}, [{'index': 0, 'value': 1.}]
    def summarize(value, arrays, rows):
        calls.append('summarize')
        return {'counts': {'invented_analysis_records': len(rows)}, 'scope': 'fixture_only'}
    module = SimpleNamespace(validate_config=validate_config, produce=produce, summarize=summarize)
    def import_module(name):
        assert name == env.MODULE
        calls.append('import')
        return module
    monkeypatch.setattr(env, 'importlib', SimpleNamespace(import_module=import_module))
    return SimpleNamespace(root=root, config=config, cfg=cfg, baseline=baseline,
        registration=root / env.REGISTRATION, result=root / env.RESULT,
        proof=root / 'results/signed_spillover_publication_proof.json', calls=calls,
        module=module, snapshot_names=snapshot_names)


def publish(item):
    reg = env.register(item.root, item.config, item.registration)
    published = commit(item.root)
    put(item.root, item.proof.relative_to(item.root).as_posix(), {
        'study_id': env.STUDY_ID, 'source_commit': reg['source_git_commit'],
        'published_commit': published, 'registration_sha256': env.sha(item.registration),
        'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'remote_bytes_match': True,
        'visibility': 'private',
        'url': env.REPOSITORY + '/blob/' + published + '/' + env.REGISTRATION})
    return reg


@pytest.fixture
def published(repository):
    publish(repository)
    return repository


@pytest.fixture
def completed(published):
    env.run(published.root, published.config, published.registration, published.proof, published.result)
    return published


def test_exact_bounded_closure():
    assert len(env.SOURCE_FILES) == 15
    assert len(env.REUSED_SOURCE) == 4
    assert len(env.PARENT_FILES) == 10
    assert {'results/spillover_tradeoff_v1/arrays.npz',
            'results/spillover_tradeoff_v1/measurements.csv'} <= env.PARENT_FILES
    assert not any('/graph.py' in name or '/model.py' in name for name in env.SOURCE_FILES)


@pytest.mark.parametrize('key', ['source_files', 'parent_files'])
def test_inventory_omission_duplicate_rejected(repository, key):
    cfg = {**repository.cfg, key: repository.cfg[key][1:] + [repository.cfg[key][-1]]}
    with pytest.raises(ValueError, match='inventory'):
        env.validate_envelope(repository.root, cfg, repository.config)


@pytest.mark.parametrize('scope', ['integration_test', 'prospective_spillover_tradeoff_diagnosis', None])
def test_scope_must_be_read_only_descriptive(repository, scope):
    with pytest.raises(ValueError, match='read-only descriptive'):
        env.validate_envelope(repository.root, {**repository.cfg, 'scope': scope}, repository.config)


@pytest.mark.parametrize('key', ['new_human_experiment', 'shared_neural_mechanism_claim',
    'education_effect_claim', 'new_training', 'new_seed_draws', 'new_effect_hypothesis_test'])
def test_unbounded_interpretation_rejected(repository, key):
    cfg = {**repository.cfg, 'interpretation': {**repository.cfg['interpretation'], key: True}}
    with pytest.raises(ValueError, match='interpretation'):
        env.validate_envelope(repository.root, cfg, repository.config)


def test_external_data_rejected(repository):
    with pytest.raises(ValueError, match='external'):
        env.validate_envelope(repository.root, {**repository.cfg, 'external_manifest': 'other.json'}, repository.config)


def test_register_freezes_all_input_and_source_bytes_without_analysis(repository):
    reg = env.register(repository.root, repository.config, repository.registration)
    assert reg['source_files'] == env.inventory(repository.root, sorted(env.SOURCE_FILES))
    assert reg['parent_files'] == env.inventory(repository.root, sorted(env.PARENT_FILES))
    assert reg['gate']['parent_arrays_and_trial_csv_sha256_checked'] is True
    assert reg['gate']['new_training_calls'] == reg['gate']['new_random_seed_draws'] == 0
    assert repository.calls == ['import', 'validate']
    assert not repository.result.exists()


@pytest.mark.parametrize('name', ['src/drosolearn/signed_spillover.py',
    'results/spillover_tradeoff_v1/arrays.npz', 'results/spillover_tradeoff_v1/measurements.csv',
    'parent/source_files.txt'])
def test_changed_source_or_saved_raw_bytes_fail_before_import(repository, name):
    with (repository.root / name).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError):
        env.register(repository.root, repository.config, repository.registration)
    assert repository.calls == []
    assert not repository.registration.exists()


@pytest.mark.parametrize('name', sorted(env.EVOLVING_INDEXES))
def test_live_indexes_may_evolve_while_baseline_snapshot_is_preserved(repository, name):
    put(repository.root, name, 'new current index; old blob retained\n')
    reg = env.register(repository.root, repository.config, repository.registration)
    assert reg['gate']['parent_archive_baseline_snapshot_checked'] is True


@pytest.mark.parametrize('defect', [
    'failed_meta', 'failed_verification', 'producer_imported', 'tradeoff_supported',
    'acquisition_supported', 'one_acquisition', 'numeric_tradeoff_false',
    'numeric_acquisition_false', 'incomplete_replay', 'cost_equalized_claim',
    'solver_equalized_claim', 'fewer_replayed_trials', 'fewer_static_probes',
    'internal_update', 'value_update', 'invalid_bound', 'cumulative_cost_equalized',
    'failed_archive', 'old_files_changed', 'archive_source_identity',
])
def test_semantically_closed_parent_cannot_import_analysis(repository, monkeypatch, defect):
    root, prefix = repository.root, 'results/spillover_tradeoff_v1/'
    meta = env.read(root / prefix / 'metadata.json')
    verification = env.read(root / prefix / 'verification.json')
    summary = env.read(root / prefix / 'summary.json')
    archive = env.read(root / 'results/spillover_archive_validation.json')
    if defect == 'failed_meta':
        meta['status'] = 'failed'
    elif defect == 'failed_verification':
        verification['status'] = 'failed'
    elif defect == 'producer_imported':
        verification['scientific_producer_imported'] = True
    elif defect == 'tradeoff_supported':
        summary['H_tradeoff']['supported'] = True
    elif defect == 'acquisition_supported':
        summary['H_acquisition']['supported'] = True
    elif defect == 'one_acquisition':
        summary['H_acquisition']['acquisition_pass_seeds'] = 1
    elif defect == 'numeric_tradeoff_false':
        summary['H_tradeoff']['numeric_spillover_qualified_all'] = False
    elif defect == 'numeric_acquisition_false':
        summary['H_acquisition']['numeric_spillover_qualified_all'] = False
    elif defect == 'incomplete_replay':
        verification['counts']['all_checkpoints_and_trial_head_states_replayed'] = False
    elif defect == 'cost_equalized_claim':
        verification['counts']['actual_autonomous_costs_equalized'] = True
    elif defect == 'solver_equalized_claim':
        verification['counts']['extra_solver_cost_equalized'] = True
    elif defect == 'fewer_replayed_trials':
        verification['counts']['independently_replayed_training_actions'] -= 1
    elif defect == 'fewer_static_probes':
        verification['counts']['checkpoint_static_probes'] -= 1
    elif defect == 'internal_update':
        verification['counts']['internal_update_calls'] = 1
    elif defect == 'value_update':
        verification['counts']['value_update_calls'] = 1
    elif defect == 'invalid_bound':
        summary['budget_validation']['common_bounds_valid_all'] = False
    elif defect == 'cumulative_cost_equalized':
        summary['budget_validation']['actual_cumulative_cost_equalized'] = True
    elif defect == 'failed_archive':
        archive['status'] = 'failed'
    elif defect == 'old_files_changed':
        archive['all_old_baseline_files_literal_preserved_except_three_indexes'] = False
    elif defect == 'archive_source_identity':
        archive['source_git_commit'] = 'a' * 40
    put(root, prefix + 'metadata.json', meta)
    put(root, prefix + 'summary.json', summary)
    put(root, prefix + 'verification.json', verification)
    put(root, 'results/spillover_archive_validation.json', archive)
    seal_parent(repository, monkeypatch)
    with pytest.raises(ValueError):
        env.register(root, repository.config, repository.registration)
    assert repository.calls == []
    assert not repository.registration.exists()


def test_parent_snapshot_must_match_baseline_blob(repository, monkeypatch):
    path = repository.root / 'results/spillover_archive_validation.json'
    archive = env.read(path)
    archive['current_file_sha256']['README.md'] = '0' * 64
    put(repository.root, path.relative_to(repository.root).as_posix(), archive)
    baseline = commit(repository.root)
    git(repository.root, 'tag', '-f', env.BASE_TAG)
    monkeypatch.setattr(env, 'BASE_COMMIT', baseline)
    with pytest.raises(ValueError, match='Git and working bytes'):
        env.register(repository.root, repository.config, repository.registration)
    assert repository.calls == []


def test_register_never_overwrites(repository):
    env.register(repository.root, repository.config, repository.registration)
    before = repository.registration.read_bytes()
    with pytest.raises(ValueError, match='New fixed'):
        env.register(repository.root, repository.config, repository.registration)
    assert repository.registration.read_bytes() == before


def test_successful_analysis_and_verify_import_no_parent_or_new_learner(completed):
    before = list(completed.calls)
    cfg, reg, meta = env.verify_envelope(completed.root, completed.config,
        completed.registration, completed.result, completed.proof)
    assert cfg == completed.cfg
    assert meta['analysis_only'] is True
    assert meta['new_training_calls'] == meta['new_random_seed_draws'] == 0
    assert meta['counts'] == {'invented_analysis_records': 1}
    assert completed.calls == before
    assert completed.calls == ['import', 'validate', 'import', 'validate', 'produce', 'summarize']
    assert set(env.read(completed.result / 'artifact_hashes.json')) == set(env.PRIMARY)


@pytest.mark.parametrize('field,value', [('remote_bytes_match', False), ('visibility', 'public'),
    ('registration_sha256', '0' * 64), ('source_commit', '0' * 40),
    ('published_commit', None), ('study_id', 'other'), ('url', 'https://example.invalid')])
def test_publication_identity_mutations_block_analysis(published, field, value):
    proof = env.read(published.proof)
    proof[field] = value
    put(published.root, published.proof.relative_to(published.root).as_posix(), proof)
    before = list(published.calls)
    with pytest.raises(ValueError):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert published.calls == before
    assert not published.result.exists()


@pytest.mark.parametrize('stamp', ['2020-01-01T00:00:00+00:00', '2999-01-01T00:00:00+00:00',
    '2026-01-01T00:00:00', '2026-01-01T00:00:00+09:00'])
def test_invalid_publication_chronology_blocks_analysis(published, stamp):
    proof = env.read(published.proof)
    proof['checked_at_utc'] = stamp
    put(published.root, published.proof.relative_to(published.root).as_posix(), proof)
    before = list(published.calls)
    with pytest.raises(ValueError):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert published.calls == before
    assert not published.result.exists()


def test_frozen_source_change_blocks_analysis(published):
    put(published.root, 'src/drosolearn/signed_geometry.py', 'changed after registration')
    before = list(published.calls)
    with pytest.raises(ValueError, match='Frozen file'):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert published.calls == before


def test_failed_analysis_saved_and_cannot_be_retried(published):
    def fail(cfg, root):
        raise RuntimeError('invented saved-results arithmetic failure')
    published.module.produce = fail
    with pytest.raises(RuntimeError, match='invented'):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    meta = env.read(published.result / 'metadata.json')
    assert meta['status'] == 'failed' and meta['error_type'] == 'RuntimeError'
    hashes = env.read(published.result / 'artifact_hashes.json')
    assert set(hashes) == {'registration.json', 'metadata.json'}
    before = {name: (published.result / name).read_bytes() for name in hashes}
    with pytest.raises(ValueError, match='New fixed result'):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert all((published.result / name).read_bytes() == value for name, value in before.items())
    with pytest.raises(ValueError, match='Completed frozen'):
        env.verify_envelope(published.root, published.config, published.registration, published.result, published.proof)


@pytest.mark.parametrize('bad', [np.array([np.nan]), np.array([np.inf]),
    np.array(['x'], dtype=object), np.array(['x'])])
def test_invalid_derived_arrays_preserve_failure(published, bad):
    published.module.produce = lambda cfg, root: ({'invalid': bad}, [{'x': 1}])
    with pytest.raises(ValueError, match='Finite numeric'):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert env.read(published.result / 'metadata.json')['status'] == 'failed'
    assert not (published.result / 'arrays.npz').exists()


@pytest.mark.parametrize('artifact', env.PRIMARY)
def test_all_saved_primary_byte_changes_rejected(completed, artifact):
    with (completed.result / artifact).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError):
        env.verify_envelope(completed.root, completed.config, completed.registration, completed.result, completed.proof)


@pytest.mark.parametrize('field,value', [('new_training_calls', 1), ('new_random_seed_draws', 1),
    ('new_effect_hypothesis_test', True), ('analysis_only', False), ('measurement_records', 2)])
def test_refreshed_metadata_hash_does_not_hide_invalid_claims(completed, field, value):
    path = completed.result / 'metadata.json'
    meta = env.read(path)
    meta[field] = value
    put(completed.result, 'metadata.json', meta)
    hashes = env.read(completed.result / 'artifact_hashes.json')
    hashes['metadata.json'] = env.sha(path)
    put(completed.result, 'artifact_hashes.json', hashes)
    with pytest.raises(ValueError):
        env.verify_envelope(completed.root, completed.config, completed.registration, completed.result, completed.proof)


def test_completed_output_blocks_duplicate(completed):
    before = (completed.result / 'metadata.json').read_bytes()
    with pytest.raises(ValueError, match='New fixed result'):
        env.run(completed.root, completed.config, completed.registration, completed.proof, completed.result)
    assert (completed.result / 'metadata.json').read_bytes() == before
