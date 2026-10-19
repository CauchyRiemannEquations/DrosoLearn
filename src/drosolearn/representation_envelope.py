"""Prospective integrity for one new diagnosis; no scientific arithmetic.

The historical follow-up envelope and its source inventory remain unchanged.
This module imports the new producer only after every byte and parent gate passes.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

STUDY_ID = 'representation_update_v1'
MODULE = 'drosolearn.representation_update'
PROTOCOL = 'docs/representation-update-protocol.md'
CONFIG = 'configs/representation_update_v1.json'
REGISTRATION = 'results/registration_representation_update_v1.json'
RESULT = 'results/representation_update_v1'
BASE_TAG = 'phase5-synthesis-v1'
BASE_COMMIT = '5978fd127c0f0787a5f2068c9d0e5fc74b35e62c'
REPOSITORY = 'https://github.com/CauchyRiemannEquations/DrosoLearn'
PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json')
REUSED_SOURCE = frozenset({
    'pyproject.toml', 'requirements-lock.txt', 'src/drosolearn/__init__.py',
    'src/drosolearn/graph.py', 'src/drosolearn/model.py',
    'src/drosolearn/human_shell_models.py', 'src/drosolearn/pattern_rule.py',
    'scripts/verify_human_shell.py', 'scripts/verify_pattern_rule.py',
})
SOURCE_FILES = REUSED_SOURCE | frozenset({
    CONFIG, PROTOCOL, 'src/drosolearn/representation_update.py',
    'src/drosolearn/representation_envelope.py', 'scripts/run_representation_update.py',
    'scripts/verify_representation_update.py', 'tests/test_representation_update.py',
    'tests/test_representation_envelope.py', 'tests/test_representation_update_verify.py',
    'src/drosolearn/representation_geometry.py', 'tests/test_representation_geometry.py',
})
PARENT_FILES = frozenset({
    'results/registration_pattern_rule_v1.json',
    'results/pattern_rule_v1/metadata.json', 'results/pattern_rule_v1/summary.json',
    'results/pattern_rule_v1/verification.json', 'results/pattern_rule_v1/artifact_hashes.json',
    'results/registration_phase5_synthesis_v1.json',
    'results/phase5_synthesis_v1/metadata.json', 'results/phase5_synthesis_v1/summary.json',
    'results/phase5_synthesis_v1/verification.json', 'results/phase5_synthesis_v1/artifact_hashes.json',
    'results/phase5_synthesis_archive_validation.json',
})
GRAPH_FILES = frozenset('data/flywire_783_mb_left_kc512_s701/' + name for name in
                        ('edges.csv', 'neurons.json', 'annotations.csv', 'provenance.json'))
OLD_INPUT = 'data/synthetic/pattern_rule_v1/input_catalog.json'
INPUT_FILES = frozenset({OLD_INPUT, 'data/synthetic/representation_update_v1/input_catalog.json'})
PINNED_FILES = REUSED_SOURCE | PARENT_FILES | GRAPH_FILES | {OLD_INPUT}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=unique,
                      parse_constant=invalid)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(root, name):
    require(type(name) is str and bool(name) and '\\' not in name and ':' not in name
            and not any(character in name for character in ('\n', '\r', '\0'))
            and not PurePosixPath(name).is_absolute()
            and '..' not in PurePosixPath(name).parts
            and str(PurePosixPath(name)) == name,
            'Canonical contained POSIX path required')
    root = Path(root).resolve()
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path.is_file(), 'Existing contained file required: ' + name)
    return path


def relative(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    require(path.is_relative_to(root), 'Path must remain inside this repository')
    return path.relative_to(root).as_posix()


def inventory(root, names):
    require(type(names) is list and bool(names) and len(names) == len(set(names)),
            'Nonempty explicit unique inventory required')
    return {name: sha(safe(root, name)) for name in names}


def write_new(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def utc(value):
    require(type(value) is str, 'UTC timestamp string required')
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
            'Timezone-aware UTC timestamp required')
    return parsed


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE)


def ancestor(root, older, newer='HEAD'):
    require(type(older) is str and re.fullmatch(r'[0-9a-f]{40}', older) is not None,
            'Exact Git commit required')
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', older, newer], cwd=root,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def committed(root, commit, hashes):
    require(type(hashes) is dict and bool(hashes), 'Explicit committed file hashes required')
    for name, checksum in hashes.items():
        require(type(checksum) is str and re.fullmatch(r'[0-9a-f]{64}', checksum) is not None,
                'SHA256 digest required')
        safe(root, name)
    expressions = ''.join(commit + ':' + name + '\n' for name in hashes).encode('utf8')
    raw = subprocess.run(['git', 'cat-file', '--batch'], cwd=root, input=expressions,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    stream = io.BytesIO(raw)
    for name, checksum in hashes.items():
        header = stream.readline().split()
        require(len(header) == 3 and header[1] == b'blob' and header[2].isdigit(),
                'Committed literal file blob required: ' + name)
        size = int(header[2])
        blob = stream.read(size)
        require(len(blob) == size and stream.read(1) == b'\n'
                and hashlib.sha256(blob).hexdigest() == checksum,
                'Git and working bytes differ: ' + name)
    require(stream.read() == b'', 'Exact ordered Git blob response required')


def validate_envelope(root, cfg, config):
    require(type(cfg) is dict and cfg.get('study_id') == STUDY_ID
            and cfg.get('module') == MODULE and cfg.get('protocol') == PROTOCOL,
            'Fixed diagnosis implementation required')
    require(cfg.get('scope') != 'integration_test', 'Synthetic fixture cannot be registered as science')
    require(relative(root, config) == CONFIG, 'Fixed configuration path required')
    for key, expected in (('source_files', SOURCE_FILES), ('parent_files', PARENT_FILES),
                          ('graph_files', GRAPH_FILES), ('input_files', INPUT_FILES)):
        values = cfg.get(key)
        require(type(values) is list and len(values) == len(expected) and set(values) == expected,
                'Exact unique frozen inventory required: ' + key)
    interpretation = cfg.get('interpretation', {})
    require(all(interpretation.get(key) is False for key in
                ('new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim')),
            'Bounded interpretation required')
    require(cfg.get('external_manifest') is None, 'No external human/raw cache in this diagnosis')


def parent_gate(root):
    require(git(root, 'rev-parse', BASE_TAG).decode().strip() == BASE_COMMIT,
            'Pinned completed Phase5 release required')
    ancestor(root, BASE_COMMIT)
    pinned = {name: sha(safe(root, name)) for name in sorted(PINNED_FILES)}
    committed(root, BASE_COMMIT, pinned)
    for study in ('pattern_rule_v1', 'phase5_synthesis_v1'):
        prefix = 'results/' + study + '/'
        reg_path = safe(root, 'results/registration_' + study + '.json')
        reg, meta = read(reg_path), read(safe(root, prefix + 'metadata.json'))
        summary = read(safe(root, prefix + 'summary.json'))
        verification = read(safe(root, prefix + 'verification.json'))
        artifacts = read(safe(root, prefix + 'artifact_hashes.json'))
        require(reg.get('study_id') == meta.get('study_id') == verification.get('study_id') == study
                and meta.get('status') == 'completed' and meta.get('stage') == 'completed'
                and verification.get('status') == 'passed'
                and verification.get('scientific_producer_imported') is False,
                'Completed independently verified parent required: ' + study)
        require(meta.get('registration_sha256') == verification.get('registration_sha256') == sha(reg_path)
                and meta.get('source_git_commit') == reg.get('source_git_commit')
                and verification.get('summary_sha256') == sha(safe(root, prefix + 'summary.json'))
                and meta.get('counts') == summary.get('counts'), 'Parent evidence identity mismatch')
        require(set(artifacts) == set(PRIMARY)
                and all(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value)
                        for value in artifacts.values()), 'Complete parent primary hash receipt required')
        require(artifacts['registration.json'] == sha(reg_path)
                and artifacts['metadata.json'] == sha(safe(root, prefix + 'metadata.json'))
                and artifacts['summary.json'] == sha(safe(root, prefix + 'summary.json')),
                'Bound small parent receipt hashes differ')
        times = [utc(reg['registered_at_utc']), utc(meta['started_at_utc']),
                 utc(meta['completed_at_utc']), utc(verification['verified_at_utc'])]
        require(times == sorted(times) and times[-1] <= datetime.now(timezone.utc),
                'Completed parent UTC chronology required')
    archive = read(safe(root, 'results/phase5_synthesis_archive_validation.json'))
    require(archive.get('status') == 'passed' and archive.get('bounded_phase5_synthesis_completed') is True
            and archive.get('all_old_baseline_files_literal_preserved_except_three_indexes') is True,
            'Completed bounded Phase5 archive required')
    return {'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT,
            'small_parent_receipts_independently_verified': ['pattern_rule_v1', 'phase5_synthesis_v1'],
            'prior_scientific_replays_reused_not_rerun': True,
            'pinned_baseline_files': len(pinned), 'external_human_data_required': False,
            'large_interference_array_required': False, 'negative_prior_results_preserved': True}


def frozen_inventory(root, cfg):
    return {group: inventory(root, cfg[group]) for group in
            ('source_files', 'parent_files', 'graph_files', 'input_files')}


def register(root, config, out):
    root, config, out = Path(root).resolve(), Path(config).resolve(), Path(out).resolve()
    require(relative(root, out) == REGISTRATION and not out.exists(), 'New fixed registration path required')
    cfg = read(config)
    validate_envelope(root, cfg, config)
    frozen = frozen_inventory(root, cfg)
    commit = git(root, 'rev-parse', 'HEAD').decode().strip()
    for values in frozen.values():
        committed(root, commit, values)
    gate = parent_gate(root)
    importlib.import_module(MODULE).validate_config(cfg)
    reg = {'study_id': STUDY_ID, 'registration_type': 'prospective_representation_update_diagnosis',
           'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': commit,
           'config': cfg, 'config_sha256': sha(config), 'registration_path': REGISTRATION,
           'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, **frozen, 'gate': gate,
           'notice': 'Prior failed and supported outcomes known; new diagnosis criteria and finite budget frozen before this computation. Computational seeds are not animals.'}
    write_new(out, reg)
    return reg


def validate_registration(root, config, registration):
    root = Path(root).resolve()
    require(relative(root, registration) == REGISTRATION, 'Fixed registration location required')
    cfg, reg = read(config), read(registration)
    validate_envelope(root, cfg, config)
    require(reg.get('study_id') == STUDY_ID and reg.get('config') == cfg
            and reg.get('config_sha256') == sha(config) and reg.get('registration_path') == REGISTRATION
            and reg.get('baseline_tag') == BASE_TAG and reg.get('baseline_commit') == BASE_COMMIT,
            'Registered study/config identity changed')
    frozen = frozen_inventory(root, cfg)
    require(all(reg.get(group) == values for group, values in frozen.items()), 'Frozen file inventory changed')
    ancestor(root, reg['source_git_commit'])
    ancestor(root, BASE_COMMIT, reg['source_git_commit'])
    for values in frozen.values():
        committed(root, reg['source_git_commit'], values)
    require(reg.get('gate') == parent_gate(root), 'Frozen prior evidence gate changed')
    require(utc(reg['registered_at_utc']) <= datetime.now(timezone.utc), 'Prior registration required')
    return cfg, reg


def publication(root, registration, reg, proof):
    proof_name = relative(root, proof)
    safe(root, proof_name)
    value = read(proof)
    require(value.get('study_id') == STUDY_ID and value.get('registration_sha256') == sha(registration)
            and value.get('source_commit') == reg['source_git_commit']
            and value.get('remote_bytes_match') is True and value.get('visibility') == 'private',
            'Exact private remote-registration byte proof required')
    published = value.get('published_commit')
    require(type(published) is str and re.fullmatch(r'[0-9a-f]{40}', published) is not None,
            'Exact published Git commit required')
    ancestor(root, reg['source_git_commit'], published)
    ancestor(root, published)
    require(value.get('url') == REPOSITORY + '/blob/' + published + '/' + REGISTRATION,
            'Exact published registration URL required')
    require(git(root, 'show', published + ':' + REGISTRATION) == Path(registration).read_bytes(),
            'Published commit registration bytes differ')
    registered, checked = utc(reg['registered_at_utc']), utc(value['checked_at_utc'])
    require(registered < checked <= datetime.now(timezone.utc), 'Remote publication must follow registration')
    return proof_name, value


def csv_write(path, rows):
    require(type(rows) is list and bool(rows) and all(type(row) is dict for row in rows),
            'Nonempty measurement dictionaries required')
    fields = list(rows[0])
    require(bool(fields) and all(set(row) == set(fields) for row in rows), 'Uniform measurement schema required')
    with Path(path).open('x', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(root, config, registration, proof, out):
    import numpy as np
    root, out = Path(root).resolve(), Path(out).resolve()
    require(relative(root, out) == RESULT and not out.exists(), 'New fixed result directory required')
    cfg, reg = validate_registration(root, config, registration)
    proof_name, evidence = publication(root, registration, reg, proof)
    module = importlib.import_module(MODULE)
    module.validate_config(cfg)
    start = datetime.now(timezone.utc)
    require(utc(evidence['checked_at_utc']) < start, 'Remote byte proof must precede computation')
    out.mkdir()
    with (out / 'registration.json').open('xb') as handle:
        handle.write(Path(registration).read_bytes())
    meta = {'study_id': STUDY_ID, 'status': 'running', 'stage': 'produce',
            'started_at_utc': start.isoformat(), 'source_git_commit': reg['source_git_commit'],
            'registration_sha256': sha(registration), 'config_sha256': sha(config),
            'publication_proof': proof_name, 'publication_proof_sha256': sha(proof),
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__},
            'new_human_experiment': False, 'shared_neural_mechanism_tested': False,
            'education_effect_tested': False}
    try:
        arrays, rows = module.produce(cfg, root)
        require(type(arrays) is dict and bool(arrays)
                and all(type(key) is str and key and np.asarray(value).dtype.kind in 'biufc'
                        and np.all(np.isfinite(value)) for key, value in arrays.items()),
                'Finite numeric non-object raw arrays required')
        np.savez_compressed(out / 'arrays.npz', **arrays)
        csv_write(out / 'measurements.csv', rows)
        meta['stage'] = 'prespecified_summary'
        summary = module.summarize(cfg, arrays, rows)
        require(type(summary) is dict and type(summary.get('counts')) is dict,
                'Explicit prespecified summary counts required')
        write_new(out / 'summary.json', summary)
        meta.update(status='completed', stage='completed', counts=summary['counts'],
                    measurement_records=len(rows), numeric_arrays=len(arrays))
    except Exception as error:
        meta.update(status='failed', error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        meta['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_new(out / 'metadata.json', meta)
        write_new(out / 'artifact_hashes.json', {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()})
    return meta


def verify_envelope(root, config, registration, result_dir, publication_proof):
    import numpy as np
    root, directory = Path(root).resolve(), Path(result_dir).resolve()
    require(relative(root, directory) == RESULT and directory.is_dir(), 'Fixed completed result directory required')
    cfg, reg = validate_registration(root, config, registration)
    proof_name, evidence = publication(root, registration, reg, publication_proof)
    meta = read(directory / 'metadata.json')
    require((directory / 'registration.json').read_bytes() == Path(registration).read_bytes(),
            'Saved registration copy differs')
    require(meta.get('study_id') == STUDY_ID and meta.get('status') == 'completed'
            and meta.get('stage') == 'completed' and meta.get('source_git_commit') == reg['source_git_commit']
            and meta.get('registration_sha256') == sha(registration) and meta.get('config_sha256') == sha(config)
            and meta.get('publication_proof') == proof_name
            and meta.get('publication_proof_sha256') == sha(publication_proof)
            and all(meta.get(key) is False for key in
                    ('new_human_experiment', 'shared_neural_mechanism_tested', 'education_effect_tested')),
            'Completed frozen run identity and interpretation required')
    times = [utc(reg['registered_at_utc']), utc(evidence['checked_at_utc']),
             utc(meta['started_at_utc']), utc(meta['completed_at_utc'])]
    require(times[0] < times[1] < times[2] <= times[3] <= datetime.now(timezone.utc),
            'Prospective publication-before-run UTC chronology required')
    artifacts = read(directory / 'artifact_hashes.json')
    require(set(artifacts) == set(PRIMARY), 'Exact five primary artifact hashes required')
    for name, checksum in artifacts.items():
        require(sha(safe(directory, name)) == checksum, 'Primary artifact changed: ' + name)
    summary = read(directory / 'summary.json')
    require(type(meta.get('counts')) is dict and meta['counts'] == summary.get('counts'),
            'Summary and metadata counts differ')
    with (directory / 'measurements.csv').open(encoding='utf8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    with np.load(directory / 'arrays.npz', allow_pickle=False) as arrays:
        require(type(meta.get('numeric_arrays')) is int and meta['numeric_arrays'] == len(arrays.files)
                and len(arrays.files) > 0
                and all(arrays[key].dtype.kind in 'biufc' and np.all(np.isfinite(arrays[key]))
                        for key in arrays.files), 'Saved finite numeric array inventory differs')
    require(type(meta.get('measurement_records')) is int and meta['measurement_records'] == len(rows)
            and len(rows) > 0, 'Saved measurement row count differs')
    return cfg, reg, meta
