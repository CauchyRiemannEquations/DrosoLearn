"""Synthetic Git/byte mutation tests; no study seeds, models or feedback."""
from datetime import datetime, timezone
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from drosolearn import update_budget_envelope as env


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
    for study in ('representation_update_v1',):
        prefix = 'results/' + study + '/'
        registration = {'study_id': study, 'registered_at_utc': '2020-01-01T00:00:00+00:00',
                        'source_git_commit': 'a' * 40}
        reg_path = put(root, 'results/registration_' + study + '.json', registration)
        summary_path = put(root, prefix + 'summary.json', {'study_id': study, 'status': 'completed', 'counts': {'synthetic_records': 1}, 'H_update': {'supported': True, 'treated_acquisition_pass_seeds': 10, 'numeric_update_qualified_all': True}})
        meta = {'study_id': study, 'status': 'completed', 'stage': 'completed',
                'registered_at_utc': registration['registered_at_utc'],
                'started_at_utc': '2020-01-01T00:00:01+00:00',
                'completed_at_utc': '2020-01-01T00:00:02+00:00',
                'registration_sha256': env.sha(reg_path), 'source_git_commit': 'a' * 40,
                'counts': {'synthetic_records': 1}}
        meta_path = put(root, prefix + 'metadata.json', meta)
        put(root, prefix + 'verification.json', {'status': 'passed', 'study_id': study,
            'scientific_producer_imported': False, 'verified_at_utc': '2020-01-01T00:00:03+00:00',
            'registration_sha256': env.sha(reg_path), 'summary_sha256': env.sha(summary_path), 'counts': {'all_checkpoints_and_trial_head_states_replayed': True, 'fixed_internal_weights_encoder_and_probe_activity_checked': True, 'head_L2_budget_not_equalized': True}})
        put(root, prefix + 'artifact_hashes.json', {'registration.json': env.sha(reg_path),
            'metadata.json': env.sha(meta_path), 'summary.json': env.sha(summary_path),
            'arrays.npz': 'b' * 64, 'measurements.csv': 'c' * 64})
    put(root, 'results/representation_update_archive_validation.json', {'status': 'passed',
        'all_old_baseline_files_literal_preserved_except_three_indexes': True,
        'scientific_verification_sha256': env.sha(root / prefix / 'verification.json'),
        'registration_sha256': env.sha(reg_path), 'counts': {'synthetic_records': 1}})
    cfg = {'study_id': env.STUDY_ID, 'module': env.MODULE, 'protocol': env.PROTOCOL,
           'scope': 'synthetic_integrity_fixture', 'source_files': sorted(env.SOURCE_FILES),
           'parent_files': sorted(env.PARENT_FILES), 'graph_files': sorted(env.GRAPH_FILES),
           'input_files': sorted(env.INPUT_FILES), 'external_manifest': None,
           'interpretation': {'new_human_experiment': False,
             'shared_neural_mechanism_claim': False, 'education_effect_claim': False}}
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
        proof=root / 'results/update_budget_prefreeze_evidence.json', calls=calls, module=module)


def publish(item):
    reg = env.register(item.root, item.config, item.registration)
    published = commit(item.root)
    proof = {'study_id': env.STUDY_ID, 'source_commit': reg['source_git_commit'],
             'published_commit': published, 'registration_sha256': env.sha(item.registration),
             'checked_at_utc': datetime.now(timezone.utc).isoformat(),
             'remote_bytes_match': True, 'visibility': 'private',
             'url': env.REPOSITORY + '/blob/' + published + '/' + env.REGISTRATION}
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
    assert len(env.SOURCE_FILES) == 24
    assert len(env.REUSED_SOURCE) == 13
    assert len(env.PARENT_FILES) == 6
    assert len(env.GRAPH_FILES) == 4
    assert len(env.INPUT_FILES) == 2
    assert all('interference_order_v1/arrays.npz' not in name for name in env.PARENT_FILES)


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


def test_write_new_never_overwrites(tmp_path):
    path = tmp_path / 'receipt.json'
    env.write_new(path, {'first': 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        env.write_new(path, {'second': 2})
    assert path.read_bytes() == before


def test_inventory_duplicate_rejected(tmp_path):
    (tmp_path / 'x').write_text('x')
    with pytest.raises(ValueError, match='unique'):
        env.inventory(tmp_path, ['x', 'x'])


@pytest.mark.parametrize('key', ['source_files', 'parent_files', 'graph_files', 'input_files'])
def test_exact_inventory_no_omission_or_duplicate(repository, key):
    cfg = dict(repository.cfg)
    cfg[key] = cfg[key][1:] + [cfg[key][-1]]
    with pytest.raises(ValueError, match='inventory'):
        env.validate_envelope(repository.root, cfg, repository.config)


@pytest.mark.parametrize('key', ['new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim'])
def test_unbounded_interpretation_rejected(repository, key):
    cfg = dict(repository.cfg)
    cfg['interpretation'] = {**cfg['interpretation'], key: True}
    with pytest.raises(ValueError, match='interpretation'):
        env.validate_envelope(repository.root, cfg, repository.config)


def test_development_scope_rejected(repository):
    cfg = {**repository.cfg, 'scope': 'integration_test'}
    with pytest.raises(ValueError, match='fixture'):
        env.validate_envelope(repository.root, cfg, repository.config)


def test_external_cache_dependency_rejected(repository):
    cfg = {**repository.cfg, 'external_manifest': 'data/other.json'}
    with pytest.raises(ValueError, match='external'):
        env.validate_envelope(repository.root, cfg, repository.config)


def test_register_all_four_committed_inventories(repository):
    reg = env.register(repository.root, repository.config, repository.registration)
    assert reg['source_git_commit'] == repository.baseline
    for group in ('source_files', 'parent_files', 'graph_files', 'input_files'):
        assert reg[group] == env.inventory(repository.root, repository.cfg[group])
    assert repository.calls == ['import', 'validate']
    assert not repository.result.exists()


@pytest.mark.parametrize('group', ['source_files', 'parent_files', 'graph_files', 'input_files'])
def test_uncommitted_bytes_fail_before_producer_import(repository, group):
    name = next(name for name in repository.cfg[group] if name != env.CONFIG)
    with (repository.root / name).open('a') as handle:
        handle.write('\nchanged')
    with pytest.raises(ValueError, match='Git and working'):
        env.register(repository.root, repository.config, repository.registration)
    assert repository.calls == []
    assert not repository.registration.exists()


def test_changed_sealed_source_rejected_even_after_new_commit(repository):
    name = sorted(env.REUSED_SOURCE)[0]
    put(repository.root, name, 'changed reused source\n')
    commit(repository.root)
    with pytest.raises(ValueError, match='Git and working'):
        env.register(repository.root, repository.config, repository.registration)
    assert repository.calls == []


def test_wrong_baseline_tag_rejected(repository):
    put(repository.root, 'extra.txt', 'new')
    commit(repository.root)
    git(repository.root, 'tag', '-f', env.BASE_TAG)
    with pytest.raises(ValueError, match='Pinned completed'):
        env.register(repository.root, repository.config, repository.registration)
    assert repository.calls == []


@pytest.mark.parametrize('defect', [
    'failed_status', 'imported_producer', 'unsupported_update', 'only_nine_acquisitions',
    'numeric_update_unqualified', 'replay_incomplete', 'unmatched_cost_unrecorded',
    'failed_archive', 'changed_old_files', 'archive_registration_mismatch',
])
def test_closed_parent_gate_blocks_before_import(repository, monkeypatch, defect):
    """A pinned parent is insufficient when the required prior evidence is closed."""
    root = repository.root
    prefix = 'results/representation_update_v1/'
    reg_path = root / 'results/registration_representation_update_v1.json'
    meta = env.read(root / prefix / 'metadata.json')
    summary = env.read(root / prefix / 'summary.json')
    verification = env.read(root / prefix / 'verification.json')
    archive = env.read(root / 'results/representation_update_archive_validation.json')
    if defect == 'failed_status':
        meta['status'] = 'failed'
    elif defect == 'imported_producer':
        verification['scientific_producer_imported'] = True
    elif defect == 'unsupported_update':
        summary['H_update']['supported'] = False
    elif defect == 'only_nine_acquisitions':
        summary['H_update']['treated_acquisition_pass_seeds'] = 9
    elif defect == 'numeric_update_unqualified':
        summary['H_update']['numeric_update_qualified_all'] = False
    elif defect == 'replay_incomplete':
        verification['counts']['all_checkpoints_and_trial_head_states_replayed'] = False
    elif defect == 'unmatched_cost_unrecorded':
        verification['counts']['head_L2_budget_not_equalized'] = False
    elif defect == 'failed_archive':
        archive['status'] = 'failed'
    elif defect == 'changed_old_files':
        archive['all_old_baseline_files_literal_preserved_except_three_indexes'] = False
    elif defect == 'archive_registration_mismatch':
        archive['registration_sha256'] = '0' * 64
    meta_path = put(root, prefix + 'metadata.json', meta)
    summary_path = put(root, prefix + 'summary.json', summary)
    verification['summary_sha256'] = env.sha(summary_path)
    verification_path = put(root, prefix + 'verification.json', verification)
    archive['scientific_verification_sha256'] = env.sha(verification_path)
    put(root, 'results/representation_update_archive_validation.json', archive)
    artifacts = env.read(root / prefix / 'artifact_hashes.json')
    artifacts.update({'registration.json': env.sha(reg_path),
        'metadata.json': env.sha(meta_path), 'summary.json': env.sha(summary_path)})
    put(root, prefix + 'artifact_hashes.json', artifacts)
    baseline = commit(root)
    git(root, 'tag', '-f', env.BASE_TAG)
    monkeypatch.setattr(env, 'BASE_COMMIT', baseline)
    with pytest.raises(ValueError):
        env.register(root, repository.config, repository.registration)
    assert repository.calls == []
    assert not repository.registration.exists()


def test_registration_no_overwrite(repository):
    env.register(repository.root, repository.config, repository.registration)
    original = repository.registration.read_bytes()
    with pytest.raises(ValueError, match='New fixed'):
        env.register(repository.root, repository.config, repository.registration)
    assert repository.registration.read_bytes() == original


def test_register_rejects_arbitrary_result_name(repository):
    with pytest.raises(ValueError, match='fixed registration'):
        env.register(repository.root, repository.config, repository.root / 'results/other.json')


def test_publication_round_trip_and_completed_verification(completed):
    before = len(completed.calls)
    cfg, reg, meta = env.verify_envelope(completed.root, completed.config,
        completed.registration, completed.result, completed.proof)
    assert cfg == completed.cfg
    assert reg['source_git_commit'] == completed.baseline
    assert meta['counts'] == {'synthetic_records': 1}
    assert meta['measurement_records'] == meta['numeric_arrays'] == 1
    assert len(completed.calls) == before  # Envelope verification imports no producer.
    assert set(env.read(completed.result / 'artifact_hashes.json')) == set(env.PRIMARY)


@pytest.mark.parametrize('field,value', [('remote_bytes_match', False), ('visibility', 'public'),
    ('registration_sha256', '0' * 64), ('source_commit', '0' * 40),
    ('published_commit', None), ('study_id', 'another'), ('url', 'https://example.invalid')])
def test_publication_identity_mutations_block_before_import(published, field, value):
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
def test_invalid_publication_chronology_blocks_run(published, stamp):
    proof = env.read(published.proof)
    proof['checked_at_utc'] = stamp
    put(published.root, published.proof.relative_to(published.root).as_posix(), proof)
    before = list(published.calls)
    with pytest.raises(ValueError):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert published.calls == before
    assert not published.result.exists()


def test_registration_changed_after_publication_blocked(published):
    reg = env.read(published.registration)
    reg['notice'] += ' changed'
    put(published.root, env.REGISTRATION, reg)
    before = list(published.calls)
    with pytest.raises(ValueError, match='remote-registration'):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert published.calls == before


def test_frozen_source_change_after_registration_blocks_run(published):
    path = published.root / 'src/drosolearn/update_budget.py'
    path.write_text('changed after freeze')
    before = list(published.calls)
    with pytest.raises(ValueError, match='Frozen file'):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert published.calls == before


def test_actual_failure_preserved_and_cannot_retry(published):
    def fail(cfg, root):
        raise RuntimeError('synthetic producer failure')
    published.module.produce = fail
    with pytest.raises(RuntimeError, match='synthetic'):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    meta = env.read(published.result / 'metadata.json')
    assert meta['status'] == 'failed'
    assert meta['error_type'] == 'RuntimeError'
    artifacts = env.read(published.result / 'artifact_hashes.json')
    assert set(artifacts) == {'registration.json', 'metadata.json'}
    before = {name: (published.result / name).read_bytes() for name in artifacts}
    with pytest.raises(ValueError, match='New fixed result'):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert all((published.result / name).read_bytes() == value for name, value in before.items())
    with pytest.raises(ValueError, match='Completed frozen'):
        env.verify_envelope(published.root, published.config, published.registration, published.result, published.proof)


@pytest.mark.parametrize('bad', [np.array([np.nan]), np.array([np.inf]), np.array(['x'], dtype=object), np.array(['x'])])
def test_nonfinite_or_non_numeric_arrays_preserve_failure(published, bad):
    published.module.produce = lambda cfg, root: ({'bad': bad}, [{'x': 1}])
    with pytest.raises(ValueError, match='Finite numeric'):
        env.run(published.root, published.config, published.registration, published.proof, published.result)
    assert env.read(published.result / 'metadata.json')['status'] == 'failed'
    assert not (published.result / 'arrays.npz').exists()


@pytest.mark.parametrize('artifact', env.PRIMARY)
def test_any_primary_byte_mutation_detected(completed, artifact):
    with (completed.result / artifact).open('ab') as handle:
        handle.write(b'\n')
    with pytest.raises(ValueError):
        env.verify_envelope(completed.root, completed.config, completed.registration, completed.result, completed.proof)


def test_metadata_row_count_checked_even_with_refreshed_artifact_hash(completed):
    path = completed.result / 'metadata.json'
    meta = env.read(path)
    meta['measurement_records'] += 1
    path.write_text(json.dumps(meta))
    hashes = env.read(completed.result / 'artifact_hashes.json')
    hashes['metadata.json'] = env.sha(path)
    (completed.result / 'artifact_hashes.json').write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match='row count'):
        env.verify_envelope(completed.root, completed.config, completed.registration, completed.result, completed.proof)


def test_unexpected_primary_inventory_rejected(completed):
    path = completed.result / 'artifact_hashes.json'
    hashes = env.read(path)
    hashes['extra.json'] = 'a' * 64
    path.write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match='Exact five'):
        env.verify_envelope(completed.root, completed.config, completed.registration, completed.result, completed.proof)


def test_output_already_completed_blocks_duplicate(completed):
    before = (completed.result / 'metadata.json').read_bytes()
    with pytest.raises(ValueError, match='New fixed result'):
        env.run(completed.root, completed.config, completed.registration, completed.proof, completed.result)
    assert (completed.result / 'metadata.json').read_bytes() == before
