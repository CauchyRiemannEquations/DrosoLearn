"""Independent closure verifier tests use invented evidence only."""
import importlib.util
import hashlib
import copy
import json
import shutil
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('independent_followup_closure',
    Path(__file__).resolve().parents[1] / 'scripts/verify_followup_closure.py')
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


@pytest.mark.parametrize('pointer, expected', [('', {'a/b': {'~c': [3, False, None]}, '': 4}),
    ('/a~1b/~0c/0', 3), ('/a~1b/~0c/1', False), ('/a~1b/~0c/2', None), ('/', 4)])
def test_pointer_literal_values(pointer, expected):
    assert V.canonical(V.resolve_pointer({'a/b': {'~c': [3, False, None]}, '': 4}, pointer)) == V.canonical(expected)


@pytest.mark.parametrize('pointer', ['a', '/a~2b', '/a~', '/list/01', '/list/-1', '/list/1', '/list/x', '/leaf/x', '/missing'])
def test_pointer_invalid_or_missing_rejected(pointer):
    with pytest.raises(ValueError):
        V.resolve_pointer({'list': [1], 'leaf': 2}, pointer)


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}'])
def test_nonliteral_json_rejected(tmp_path, raw):
    path = tmp_path / 'evidence.json'; path.write_text(raw, encoding='utf8')
    with pytest.raises(ValueError):
        V.read_json(path)


@pytest.mark.parametrize('name', ['../escape', '/abs', 'C:/abs', 'folder\\file', '-unsafe', 'a/../../escape',
    'a//file', 'a/./file', 'line\nbreak', 'carriage\rreturn', 'null\0byte'])
def test_uncontained_evidence_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        V.contained(tmp_path, name)


def test_unfrozen_evidence_rejected(tmp_path):
    (tmp_path / 'data.json').write_text('{}', encoding='utf8')
    with pytest.raises(ValueError):
        V.contained(tmp_path, 'data.json', {'other.json'})


def test_canonical_distinguishes_boolean_integer_and_float():
    assert len({V.canonical(False), V.canonical(0), V.canonical(0.)}) == 3


def test_streamed_sha_and_no_input_mutation(tmp_path):
    path = tmp_path / 'saved.bin'; path.write_bytes(b'exact\r\n\0bytes')
    before = path.read_bytes()
    assert V.digest(path) == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before


@pytest.mark.parametrize('value', ['2026-10-21T00:00:00', '2026-10-21T09:00:00+09:00', 'invalid'])
def test_invalid_or_non_utc_timestamps_rejected(value):
    with pytest.raises(ValueError):
        V.stamp(value)


def test_explicit_utc_timestamp_preserved():
    assert V.stamp('2026-10-21T00:00:00+00:00').utcoffset().total_seconds() == 0


@pytest.fixture(scope='module')
def invented(tmp_path_factory):
    from tests.test_followup_closure import fixture, m
    cfg, manifest, root = fixture(tmp_path_factory.mktemp('independent-closure'))
    arrays, rows, summary = m.analyze(cfg, root)
    return cfg, manifest, root, arrays, rows, summary


def changed_copy(tmp_path, invented):
    cfg, manifest, root, arrays, rows, summary = invented
    target = tmp_path / 'independent-mutated-copy'
    shutil.copytree(root, target)
    return copy.deepcopy(cfg), copy.deepcopy(manifest), target, copy.deepcopy(arrays), copy.deepcopy(rows), copy.deepcopy(summary)


def test_full_independent_tiny_git_fixture(invented):
    cfg, manifest, root, arrays, rows, summary = invented
    result = V.verify_science(cfg, root, arrays, rows, summary, allow_fixture=True)
    assert result['verified_followup_studies'] == 17
    assert result['actual_learning_registrations'] == 8 and result['saved_record_registrations'] == 9
    assert result['independently_checked_claims'] == 1 and result['exact_numeric_ledgers'] == 2
    assert result['past_full_science_reruns'] == result['new_training_trials'] == 0


def test_actual_gate_rejects_invented_fixture(invented):
    cfg, manifest, root, arrays, rows, summary = invented
    with pytest.raises(ValueError):
        V.verify_science(cfg, root, arrays, rows, summary)


@pytest.mark.parametrize('kind', ['summary', 'metadata', 'verification', 'archive', 'publication_proof',
    'protocol', 'findings', 'reproduction', 'artifact_hashes'])
def test_original_study_bytes_cannot_be_replaced(tmp_path, invented, kind):
    cfg, manifest, root, arrays, rows, summary = changed_copy(tmp_path, invented)
    path = root / manifest['studies'][0][kind]
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        V.check_study(V.SavedEvidence(root), manifest['studies'][0])


@pytest.mark.parametrize('kind', ['arrays.npz', 'measurements.csv', 'registration.json'])
def test_original_primary_bytes_cannot_be_replaced(tmp_path, invented, kind):
    cfg, manifest, root, arrays, rows, summary = changed_copy(tmp_path, invented)
    path = root / manifest['studies'][0]['result_dir'] / kind
    path.write_bytes(path.read_bytes() + b'altered')
    with pytest.raises(ValueError):
        V.check_study(V.SavedEvidence(root), manifest['studies'][0])


@pytest.mark.parametrize('field', ['counts', 'summary_sha256', 'registration_sha256', 'study_id',
    'scientific_producer_imported', 'new_helper_imported', 'verified_at_utc'])
def test_separate_reproduction_receipt_cannot_claim_different_work(tmp_path, invented, field):
    cfg, manifest, root, arrays, rows, summary = changed_copy(tmp_path, invented)
    path = root / cfg['reproduction_receipt']; record = V.read_json(path)
    if field == 'counts': record[field] = {'invented_records': 5}
    elif field == 'verified_at_utc': record[field] = '1999-01-01T00:00:00+00:00'
    elif field in ('scientific_producer_imported', 'new_helper_imported'): record[field] = True
    else: record[field] = 'changed'
    path.write_text(json.dumps(record), encoding='utf8')
    with pytest.raises(ValueError):
        V.expected_reproduction(cfg, V.SavedEvidence(root))


@pytest.mark.parametrize('kind', ['kind', 'optimization', 'supported_level5', 'supported_level4',
    'supported_level3', 'duplicate_claim', 'study_order', 'boolean_version'])
def test_closed_manifest_scope_cannot_expand(invented, kind):
    manifest = copy.deepcopy(invented[1])
    if kind == 'kind': manifest['studies'][0]['kind'] = 'saved_record_analysis'
    elif kind == 'optimization': manifest['studies'][6]['original_new_optimization'] = False
    elif kind.startswith('supported_level'):
        manifest['claims'][0].update(level=int(kind[-1]), verdict='supported')
    elif kind == 'duplicate_claim': manifest['claims'].append(copy.deepcopy(manifest['claims'][0]))
    elif kind == 'study_order': manifest['studies'].reverse()
    else: manifest['manifest_version'] = True
    with pytest.raises(ValueError):
        V.manifest_contract(manifest)


def test_no_scientific_producer_or_training_import():
    source = Path(V.__file__).read_text(encoding='utf8')
    assert all(token not in source for token in ('from drosolearn.followup_closure import',
        'import drosolearn.followup_closure', 'np.random', 'scipy.optimize', 'SelfBlendHead'))


def test_wrong_release_commit_fails(invented):
    cfg, manifest, root, arrays, rows, summary = invented
    study = copy.deepcopy(manifest['studies'][0]); study['completion_commit'] = 'a' * 40
    with pytest.raises(ValueError):
        V.check_study(V.SavedEvidence(root), study)


def test_literal_boolean_evidence_cannot_be_numeric_zero(tmp_path, invented):
    cfg, manifest, root, arrays, rows, summary = changed_copy(tmp_path, invented)
    manifest['studies'][0]['hypothesis_observations'][0]['expected'] = 0
    with pytest.raises(ValueError):
        V.check_study(V.SavedEvidence(root), manifest['studies'][0])


def test_direct_evidence_must_appear_in_frozen_inventory(invented):
    cfg, manifest, root, arrays, rows, summary = invented
    cfg = copy.deepcopy(cfg); cfg['parent_files'] = []
    with pytest.raises(ValueError):
        V.expected_evidence(cfg, root)


@pytest.mark.parametrize('kind', ['flag', 'ledger_dtype', 'missing_array', 'statement', 'matched',
    'question_boolean', 'counter', 'human_claim', 'unmatched_claim', 'extra_summary_key'])
def test_output_tamper_rejected_against_independently_checked_fixture(invented, monkeypatch, kind):
    cfg, manifest, root, original_arrays, original_rows, original_summary = invented
    wanted = copy.deepcopy((original_arrays, original_rows, original_summary))
    monkeypatch.setattr(V, 'expected_evidence', lambda config, folder: copy.deepcopy(wanted))
    arrays, rows, summary = copy.deepcopy(wanted)
    if kind == 'flag': arrays['study_integrity_flags'][0, 0] = False
    elif kind == 'ledger_dtype': arrays['claim_ledger'] = arrays['claim_ledger'].astype(float)
    elif kind == 'missing_array': del arrays['claim_ledger']
    elif kind == 'statement': rows[0]['statement'] = 'Unsupported new assertion'
    elif kind == 'matched': rows[0]['matched'] = 0
    elif kind == 'question_boolean': rows[0]['question'] = True
    elif kind == 'counter': summary['counts']['new_training_trials'] = 1
    elif kind == 'human_claim': summary['interpretation']['human_similarity_supported'] = True
    elif kind == 'unmatched_claim': summary['claims'][0]['matched'] = False
    else: summary['extra'] = 'unregistered addition'
    with pytest.raises(ValueError):
        V.verify_science(cfg, root, arrays, rows, summary, allow_fixture=True)
