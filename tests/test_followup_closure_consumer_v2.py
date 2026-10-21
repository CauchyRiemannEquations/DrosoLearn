"""Invented-only regression checks for the additional historical-schema consumer."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('strict_closure_consumer_v2',
    REPO / 'scripts/verify_followup_closure_consumer_v2.py')
C = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(C)
OLD_SPEC = importlib.util.spec_from_file_location('unchanged_closure_consumer_for_fixture',
    REPO / 'scripts/verify_followup_closure.py')
V = importlib.util.module_from_spec(OLD_SPEC); OLD_SPEC.loader.exec_module(V)


def put(root, name, value):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    if type(value) is bytes:
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()


def init(root):
    root.mkdir(); git(root, 'init', '-q'); git(root, 'config', 'user.name', 'Invented fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid'); git(root, 'config', 'core.autocrlf', 'false')
    put(root, '.gitattributes', b'* -text\n')


def commit(root, label):
    git(root, 'add', '--all'); git(root, 'commit', '-qm', label)
    return git(root, 'rev-parse', 'HEAD')


@pytest.fixture(scope='module')
def invented_human(tmp_path_factory):
    root = tmp_path_factory.mktemp('human-parent') / 'invented'; init(root)
    cfg = dict(C.FIXED)
    old_config = {'study_id': 'human_shell_rng_v2', 'fixture_only': True}
    put(root, 'configs/human_shell_rng_v2.json', old_config)
    for name in C.HUMAN_SOURCE_SCRIPTS:
        put(root, name, b'# Invented original verifier bytes; never executed.\n')
    source = commit(root, 'Invented human source')
    source_files = {name: C.sha(root / name) for name in (*C.HUMAN_SOURCE_SCRIPTS, 'configs/human_shell_rng_v2.json')}
    registration = {'study_id': cfg['human_study_id'], 'source_git_commit': source,
        'registered_at_utc': '2000-01-01T00:00:00+00:00', 'config': old_config,
        'config_sha256': source_files['configs/human_shell_rng_v2.json'], 'source_files': source_files}
    summary = {name: None for name in C.HUMAN_SUMMARY_FIELDS}; summary['status'] = 'completed'
    put(root, cfg['human_registration'], registration); put(root, cfg['human_summary'], summary)
    metadata = {'study_id': cfg['human_study_id'], 'status': 'completed', 'stage': 'completed',
        'source_git_commit': source, 'registration_sha256': C.sha(root / cfg['human_registration']),
        'config_sha256': registration['config_sha256'], 'started_at_utc': '2000-01-01T00:00:01+00:00',
        'completed_at_utc': '2000-01-01T00:00:02+00:00'}
    put(root, cfg['human_metadata'], metadata)
    verified = {'study_id': cfg['human_study_id'], 'status': 'passed', 'verifier_imports_producer': False,
        'verifier_modifies_original_results': False, 'all_frozen_inputs_match': True,
        'parent_arithmetic_verification_passed': True, 'registration_sha256': metadata['registration_sha256'],
        'summary_sha256': C.sha(root / cfg['human_summary']), 'verified_at_utc': '2000-01-01T00:00:03+00:00'}
    put(root, cfg['human_verification'], verified)
    hashes = {name: hashlib.sha256(b'invented unused primary bytes').hexdigest() for name in C.HUMAN_PRIMARY_NAMES}
    for field in ('human_summary', 'human_metadata', 'human_registration'):
        hashes[Path(cfg[field]).name] = C.sha(root / cfg[field])
    put(root, cfg['human_artifact_hashes'], hashes)
    for name in C.HUMAN_PATHS:
        put(root, name, b'Invented original anchor path.\n')
    put(root, 'other/summary.json', {'study_id': 'invented_other'})
    put(root, 'other/verification.json', {'study_id': 'invented_other', 'status': 'passed'})
    put(root, 'other/note.md', b'Invented other anchor.\n')
    completed = commit(root, 'Invented human completion'); git(root, 'tag', cfg['human_completion_tag'])
    anchor = dict(id=cfg['human_anchor_id'], summary=cfg['human_summary'], verification=cfg['human_verification'],
        paths=list(C.HUMAN_PATHS))
    return root, cfg, anchor, completed


def human_copy(tmp_path, invented_human, monkeypatch):
    root, cfg, anchor, completed = invented_human
    target = tmp_path / 'invented-human-copy'; shutil.copytree(root, target)
    monkeypatch.setattr(C, 'HUMAN_COMPLETION_COMMIT', completed)
    return target, copy.deepcopy(cfg), copy.deepcopy(anchor)


def test_original_uniform_schema_assumption_rejects_legacy_fixture(invented_human):
    root, cfg, anchor, completed = invented_human
    with pytest.raises(ValueError, match='Original anchor evidence identities'):
        V.check_anchors(V.SavedEvidence(root), [anchor])


def test_strict_original_schema_accepts_unchanged_invented_human(tmp_path, invented_human, monkeypatch):
    root, cfg, anchor = human_copy(tmp_path, invented_human, monkeypatch)
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file() and '.git' not in p.parts}
    C.check_human_anchor(V.SavedEvidence(root), anchor, cfg)
    assert len(C.HUMAN_SUMMARY_FIELDS) == 20
    assert before == {name: (root / name).read_bytes() for name in before}


@pytest.mark.parametrize('kind', ['added_study_id', 'missing_field', 'unknown_field', 'status', 'identity',
    'metadata_status', 'metadata_stage', 'summary_sha', 'registration_sha', 'source_commit', 'config_sha',
    'config_values', 'producer_imported', 'results_modified', 'unfrozen', 'parent_unverified',
    'primary_inventory', 'primary_sha', 'source_script', 'completion_tag', 'chronology'])
def test_human_original_schema_and_byte_bindings_reject_mutation(tmp_path, invented_human, monkeypatch, kind):
    root, cfg, anchor = human_copy(tmp_path, invented_human, monkeypatch)
    summary = C.read(root / cfg['human_summary']); meta = C.read(root / cfg['human_metadata'])
    verified = C.read(root / cfg['human_verification']); reg = C.read(root / cfg['human_registration'])
    artifacts = C.read(root / cfg['human_artifact_hashes'])
    if kind == 'added_study_id': summary['study_id'] = cfg['human_study_id']
    elif kind == 'missing_field': summary.pop('H_similarity')
    elif kind == 'unknown_field': summary['extra'] = False
    elif kind == 'status': summary['status'] = 'running'
    elif kind == 'identity': verified['study_id'] = 'another_study'
    elif kind == 'metadata_status': meta['status'] = 'running'
    elif kind == 'metadata_stage': meta['stage'] = 'running'
    elif kind == 'summary_sha': verified['summary_sha256'] = '0' * 64
    elif kind == 'registration_sha': meta['registration_sha256'] = '0' * 64
    elif kind == 'source_commit': meta['source_git_commit'] = '0' * 40
    elif kind == 'config_sha': meta['config_sha256'] = '0' * 64
    elif kind == 'config_values': reg['config']['fixture_only'] = False
    elif kind == 'producer_imported': verified['verifier_imports_producer'] = True
    elif kind == 'results_modified': verified['verifier_modifies_original_results'] = True
    elif kind == 'unfrozen': verified['all_frozen_inputs_match'] = False
    elif kind == 'parent_unverified': verified['parent_arithmetic_verification_passed'] = False
    elif kind == 'primary_inventory': artifacts['unexpected.json'] = '0' * 64
    elif kind == 'primary_sha': artifacts['summary.json'] = '0' * 64
    elif kind == 'source_script': put(root, C.HUMAN_SOURCE_SCRIPTS[0], b'Changed source.\n')
    elif kind == 'completion_tag': monkeypatch.setattr(C, 'HUMAN_COMPLETION_COMMIT', '0' * 40)
    elif kind == 'chronology': verified['verified_at_utc'] = '1999-01-01T00:00:00+00:00'
    for field, value in (('human_summary', summary), ('human_metadata', meta), ('human_verification', verified),
                         ('human_registration', reg), ('human_artifact_hashes', artifacts)):
        put(root, cfg[field], value)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        C.check_human_anchor(V.SavedEvidence(root), anchor, cfg)


@pytest.mark.parametrize('field', ['id', 'summary', 'verification', 'paths'])
def test_human_path_exception_cannot_spread_to_another_document(tmp_path, invented_human, monkeypatch, field):
    root, cfg, anchor = human_copy(tmp_path, invented_human, monkeypatch)
    anchor[field] = [] if field == 'paths' else 'another'
    with pytest.raises(ValueError, match='Exact single historical human anchor paths'):
        C.check_human_anchor(V.SavedEvidence(root), anchor, cfg)


def test_other_anchors_use_original_function_and_restore_on_exit(tmp_path, invented_human, monkeypatch):
    root, cfg, anchor = human_copy(tmp_path, invented_human, monkeypatch)
    calls = []
    def original(evidence, anchors):
        calls.extend(anchors); return V.check_anchors(evidence, anchors)
    module = SimpleNamespace(check_anchors=original)
    other = dict(id='invented_other', summary='other/summary.json', verification='other/verification.json', paths=['other/note.md'])
    with C.original_schema_consumer(module, cfg):
        module.check_anchors(V.SavedEvidence(root), [anchor, other])
    assert calls == [other] and module.check_anchors is original
    put(root, 'other/verification.json', {'study_id': 'wrong', 'status': 'passed'})
    with pytest.raises(ValueError, match='Original anchor evidence identities'):
        with C.original_schema_consumer(module, cfg):
            module.check_anchors(V.SavedEvidence(root), [anchor, other])
    assert module.check_anchors is original


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'relabeled'])
def test_exactly_one_human_exception_and_no_relabeling(tmp_path, invented_human, monkeypatch, kind):
    root, cfg, anchor = human_copy(tmp_path, invented_human, monkeypatch)
    module = SimpleNamespace(check_anchors=V.check_anchors)
    anchors = [] if kind == 'missing' else [anchor, copy.deepcopy(anchor)]
    if kind == 'relabeled': anchors[-1]['id'] = 'another'
    with pytest.raises(ValueError):
        with C.original_schema_consumer(module, cfg):
            module.check_anchors(V.SavedEvidence(root), anchors)
    assert module.check_anchors is V.check_anchors


@pytest.fixture(scope='module')
def invented_consumer(tmp_path_factory):
    root = tmp_path_factory.mktemp('consumer-parent') / 'invented'; init(root)
    # Only the new fixed configuration is read. Every saved source/result here is invented.
    cfg = C.read(REPO / C.CONFIG)
    old_sources = set(cfg['parent_files']) - C.PARENT_ADDITIONS
    for name in old_sources:
        put(root, name, b'Invented frozen original source bytes.\n')
    old_commit = commit(root, 'Invented original source freeze')
    original = {'study_id': 'followup_closure_v1', 'source_git_commit': old_commit,
        'source_files': {name: C.sha(root / name) for name in old_sources}}
    for name in C.PARENT_ADDITIONS:
        put(root, name, {'invented': True})
    put(root, cfg['original_registration'], original)
    archive_name = 'results/followup_closure_verification_attempt01.zip'
    members = sorted(old_sources | {cfg['original_registration'], cfg['original_publication_proof']} |
        {name for name in cfg['parent_files'] if name.startswith('results/followup_closure_v1/')})
    member_sha = {}
    with zipfile.ZipFile(root / archive_name, 'w') as archive:
        for name in members:
            payload = (root / name).read_bytes(); archive.writestr(name, payload)
            member_sha[name] = hashlib.sha256(payload).hexdigest()
        payload = b'{"invented_first_tool_failure":true}\n'; archive.writestr('original_tool_result.json', payload)
        member_sha['original_tool_result.json'] = hashlib.sha256(payload).hexdigest()
    failure = dict(status='preserved_original_independent_consumer_failure', study_id='followup_closure_v1',
        captured_at_utc='2000-01-01T00:00:00+00:00', first_verifier_exit_code=1,
        first_verifier_receipt_written=False, failure='ValueError: Original anchor evidence identities',
        source_git_commit=old_commit, original_registration_sha256=C.sha(root / cfg['original_registration']),
        original_frozen_source_files_unchanged=True, original_primary_files_unchanged=True,
        original_verifier_or_producer_rerun=False, new_learning_or_optimization_calls=0,
        scientific_result_or_acquisition_gate_waived=False, archive=archive_name,
        archive_sha256=C.sha(root / archive_name), member_sha256=member_sha)
    put(root, cfg['original_failure'], failure)
    for name in C.SOURCE_FILES:
        put(root, name, cfg if name == C.CONFIG else b'Invented newly registered consumer source bytes.\n')
    source = commit(root, 'Invented additive consumer freeze')
    registration = dict(study_id=C.STUDY, registration_type='retrospective_strict_schema_consumer_correction',
        registered_at_utc='2000-01-01T00:00:01+00:00', source_git_commit=source, config=cfg,
        config_sha256=C.sha(root / C.CONFIG), original_independent_failure_preserved=True,
        scientific_success_criteria_changed=False, original_source_or_primary_files_changed=False,
        new_learning_rng_optimization_effect_human_calls=0,
        **{key: {name: C.sha(root / name) for name in cfg[key]} for key in ('source_files', 'parent_files')})
    put(root, C.REGISTRATION, registration); published = commit(root, 'Invented new registration publication')
    proof = dict(study_id=C.STUDY, registration_sha256=C.sha(root / C.REGISTRATION), source_commit=source,
        published_commit=published, remote_bytes_match=True, visibility='private',
        checked_at_utc='2000-01-01T00:00:02+00:00', url=C.REPOSITORY + '/blob/' + published + '/' + C.REGISTRATION)
    put(root, C.PUBLICATION, proof)
    return root, cfg


def consumer_copy(tmp_path, invented_consumer):
    source, cfg = invented_consumer; root = tmp_path / 'invented-consumer-copy'; shutil.copytree(source, root)
    return root, copy.deepcopy(cfg)


def test_separate_correction_freeze_and_publication_accept_invented_git(invented_consumer):
    root, cfg = invented_consumer
    actual, registration, proof, failure = C.validate_consumer(root, root / C.REGISTRATION, root / C.PUBLICATION)
    assert actual == cfg and len(registration['source_files']) == 4 and len(registration['parent_files']) == 54
    assert failure['first_verifier_exit_code'] == 1 and proof['visibility'] == 'private'
    record = C.correction_record(root, cfg, registration)
    assert record['human_original_tag_primary_files_checked'] == 5
    assert record['source_files'] == 4 and record['parent_files'] == 54


@pytest.mark.parametrize('kind', ['own_source', 'parent', 'old_source', 'config_scope', 'config_extra',
    'config_boolean', 'registration_extra', 'registration_identity', 'registration_type', 'registration_config',
    'registration_sha', 'registration_time', 'registration_work', 'source_commit', 'source_inventory',
    'parent_inventory', 'proof_registration_sha', 'proof_source', 'proof_private', 'proof_remote',
    'proof_url', 'proof_published', 'proof_time', 'registration_path', 'proof_path'])
def test_correction_gate_rejects_changed_provenance(tmp_path, invented_consumer, kind):
    root, cfg = consumer_copy(tmp_path, invented_consumer)
    reg = C.read(root / C.REGISTRATION); proof = C.read(root / C.PUBLICATION)
    reg_path = root / C.REGISTRATION; proof_path = root / C.PUBLICATION
    if kind == 'own_source': put(root, 'scripts/verify_followup_closure_consumer_v2.py', b'changed')
    elif kind == 'parent': put(root, cfg['human_summary'], {'changed': True})
    elif kind == 'old_source': put(root, cfg['original_verifier'], b'changed')
    elif kind == 'config_scope': cfg['scope'] = 'relaxed'
    elif kind == 'config_extra': cfg['extra'] = False
    elif kind == 'config_boolean': cfg['new_learning_rng_optimization_effect_human_calls'] = False
    elif kind == 'registration_extra': reg['extra'] = False
    elif kind == 'registration_identity': reg['study_id'] = 'another'
    elif kind == 'registration_type': reg['registration_type'] = 'another'
    elif kind == 'registration_config': reg['config'] = {}
    elif kind == 'registration_sha': reg['config_sha256'] = '0' * 64
    elif kind == 'registration_time': reg['registered_at_utc'] = '1999-01-01T00:00:00+00:00'
    elif kind == 'registration_work': reg['new_learning_rng_optimization_effect_human_calls'] = False
    elif kind == 'source_commit': reg['source_git_commit'] = '0' * 40
    elif kind == 'source_inventory': reg['source_files'].pop(next(iter(reg['source_files'])))
    elif kind == 'parent_inventory': reg['parent_files'].pop(next(iter(reg['parent_files'])))
    elif kind == 'proof_registration_sha': proof['registration_sha256'] = '0' * 64
    elif kind == 'proof_source': proof['source_commit'] = '0' * 40
    elif kind == 'proof_private': proof['visibility'] = 'public'
    elif kind == 'proof_remote': proof['remote_bytes_match'] = False
    elif kind == 'proof_url': proof['url'] += 'changed'
    elif kind == 'proof_published': proof['published_commit'] = '0' * 40
    elif kind == 'proof_time': proof['checked_at_utc'] = '1999-01-01T00:00:00+00:00'
    elif kind == 'registration_path': reg_path = root / 'wrong-registration.json'
    elif kind == 'proof_path': proof_path = root / 'wrong-proof.json'
    put(root, C.CONFIG, cfg); put(root, C.REGISTRATION, reg); put(root, C.PUBLICATION, proof)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        C.validate_consumer(root, reg_path, proof_path)


@pytest.mark.parametrize('field', ['status', 'study_id', 'first_verifier_exit_code', 'first_verifier_receipt_written',
    'failure', 'source_git_commit', 'original_registration_sha256', 'original_frozen_source_files_unchanged',
    'original_primary_files_unchanged', 'original_verifier_or_producer_rerun', 'new_learning_or_optimization_calls',
    'scientific_result_or_acquisition_gate_waived', 'archive_sha256', 'member_sha256'])
def test_preserved_failure_cannot_be_rewritten(tmp_path, invented_consumer, field):
    root, cfg = consumer_copy(tmp_path, invented_consumer)
    failure = C.read(root / cfg['original_failure'])
    old = failure[field]
    failure[field] = not old if type(old) is bool else ({} if type(old) is dict else 'changed')
    put(root, cfg['original_failure'], failure)
    with pytest.raises(ValueError):
        C.check_failure(root, cfg, C.read(root / cfg['original_registration']))


def test_original_failed_output_cannot_be_backfilled(tmp_path, invented_consumer):
    root, cfg = consumer_copy(tmp_path, invented_consumer)
    put(root, cfg['original_result_dir'] + '/verification.json', {'status': 'passed'})
    with pytest.raises(ValueError, match='must remain absent'):
        C.check_failure(root, cfg, C.read(root / cfg['original_registration']))


def test_safe_new_outputs_support_later_read_only_recheck(tmp_path, invented_consumer):
    root, cfg = consumer_copy(tmp_path, invented_consumer)
    assert C.output_path(root, root / cfg['output'], cfg) == root / cfg['output']
    assert C.output_path(root, root / 'results/another_consumer_check/verification.json', cfg).is_relative_to(root)


@pytest.mark.parametrize('name', ['results/followup_closure_v1/verification.json',
    'results/followup_closure_v1/nested/another.json', C.REGISTRATION, C.PUBLICATION,
    'scripts/verify_followup_closure_consumer_v2.py', '../outside.json'])
def test_output_cannot_replace_or_enter_original_result(tmp_path, invented_consumer, name):
    root, cfg = consumer_copy(tmp_path, invented_consumer)
    with pytest.raises(ValueError): C.output_path(root, root / name, cfg)


@pytest.mark.parametrize('text', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'])
def test_new_registration_json_is_strict(tmp_path, text):
    path = tmp_path / 'invented.json'; path.write_text(text, encoding='utf8')
    with pytest.raises(ValueError): C.read(path)


def test_adapter_imports_no_scientific_producer_and_reuses_all_other_checks():
    source = Path(C.__file__).read_text(encoding='utf8')
    assert 'verifier.verify_science(original_cfg, root, arrays, rows, summary)' in source
    assert 'from drosolearn.closure_envelope import verify_envelope, write_new' in source
    assert not any(token in source for token in ('import drosolearn.followup_closure',
        'from drosolearn.followup_closure import', 'default_rng(', 'SeedSequence(', '.optimize(', '.fit('))
    assert 'verifier.check_anchors = original' in source
