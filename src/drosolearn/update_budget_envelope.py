"""Prospective byte, chronology, and parent gates for the bounded budget study.

Only integrity primitives are reused. Scientific production stays lazy, after all
byte checks pass; independent envelope verification never imports the producer.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib
from pathlib import Path
import re
import sys

from .representation_envelope import (
    ancestor, committed, csv_write, git, inventory, read, relative, require,
    safe, sha, utc, write_new,
)

STUDY_ID = 'update_budget_v1'
MODULE = 'drosolearn.update_budget'
PROTOCOL = 'docs/update-budget-protocol.md'
CONFIG = 'configs/update_budget_v1.json'
REGISTRATION = 'results/registration_update_budget_v1.json'
RESULT = 'results/update_budget_v1'
BASE_TAG = 'representation-update-v1'
BASE_COMMIT = '440f6b57c6176605db37d9a5397e3f82764a5b29'
REPOSITORY = 'https://github.com/CauchyRiemannEquations/DrosoLearn'
PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json')
REUSED_SOURCE = frozenset({
    'pyproject.toml', 'requirements-lock.txt', 'src/drosolearn/__init__.py',
    'src/drosolearn/graph.py', 'src/drosolearn/model.py',
    'src/drosolearn/human_shell_models.py', 'src/drosolearn/pattern_rule.py',
    'src/drosolearn/representation_geometry.py',
    'src/drosolearn/representation_update.py', 'src/drosolearn/representation_envelope.py',
    'scripts/verify_human_shell.py', 'scripts/verify_pattern_rule.py',
    'scripts/verify_representation_update.py',
})
SOURCE_FILES = REUSED_SOURCE | frozenset({
    CONFIG, PROTOCOL, 'src/drosolearn/update_budget.py',
    'src/drosolearn/update_budget_envelope.py', 'scripts/run_update_budget.py',
    'scripts/verify_update_budget.py', 'tests/test_update_budget.py',
    'tests/test_update_budget_envelope.py', 'tests/test_update_budget_verify.py',
    'src/drosolearn/update_budget_geometry.py', 'tests/test_update_budget_geometry.py',
})
PARENT_FILES = frozenset({
    'results/registration_representation_update_v1.json',
    'results/representation_update_v1/metadata.json',
    'results/representation_update_v1/summary.json',
    'results/representation_update_v1/verification.json',
    'results/representation_update_v1/artifact_hashes.json',
    'results/representation_update_archive_validation.json',
})
GRAPH_FILES = frozenset('data/flywire_783_mb_left_kc512_s701/' + name for name in
                        ('edges.csv', 'neurons.json', 'annotations.csv', 'provenance.json'))
INPUT_FILES = frozenset({
    'data/synthetic/pattern_rule_v1/input_catalog.json',
    'data/synthetic/representation_update_v1/input_catalog.json',
})
PINNED_FILES = REUSED_SOURCE | PARENT_FILES | GRAPH_FILES | INPUT_FILES


def validate_envelope(root, cfg, config):
    require(type(cfg) is dict and cfg.get('study_id') == STUDY_ID
            and cfg.get('module') == MODULE and cfg.get('protocol') == PROTOCOL,
            'Fixed budget-study implementation required')
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
            'Pinned completed representation release required')
    ancestor(root, BASE_COMMIT)
    pinned = {name: sha(safe(root, name)) for name in sorted(PINNED_FILES)}
    committed(root, BASE_COMMIT, pinned)
    study = 'representation_update_v1'
    prefix = 'results/' + study + '/'
    reg_path = safe(root, 'results/registration_' + study + '.json')
    reg, meta = read(reg_path), read(safe(root, prefix + 'metadata.json'))
    summary = read(safe(root, prefix + 'summary.json'))
    verification = read(safe(root, prefix + 'verification.json'))
    artifacts = read(safe(root, prefix + 'artifact_hashes.json'))
    require(reg.get('study_id') == meta.get('study_id') == verification.get('study_id') == study
            and meta.get('status') == 'completed' and meta.get('stage') == 'completed'
            and summary.get('study_id') == study and summary.get('status') == 'completed'
            and verification.get('status') == 'passed'
            and verification.get('scientific_producer_imported') is False,
            'Completed independently verified representation parent required')
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
    replay = verification.get('counts', {})
    require(replay.get('all_checkpoints_and_trial_head_states_replayed') is True
            and replay.get('fixed_internal_weights_encoder_and_probe_activity_checked') is True
            and replay.get('head_L2_budget_not_equalized') is True,
            'Prior replay and documented unmatched head-step cost required')
    update = summary.get('H_update', {})
    require(update.get('supported') is True and update.get('treated_acquisition_pass_seeds') == 10
            and update.get('numeric_update_qualified_all') is True,
            'Successful acquired prior projected-update control required')
    archive = read(safe(root, 'results/representation_update_archive_validation.json'))
    require(archive.get('status') == 'passed'
            and archive.get('all_old_baseline_files_literal_preserved_except_three_indexes') is True
            and archive.get('scientific_verification_sha256') == sha(safe(root, prefix + 'verification.json'))
            and archive.get('registration_sha256') == sha(reg_path)
            and archive.get('counts') == summary.get('counts'),
            'Verified literal historical archive and parent identity required')
    return {'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT,
            'small_parent_receipts_independently_verified': [study],
            'prior_scientific_replays_reused_not_rerun': True,
            'pinned_baseline_files': len(pinned), 'external_human_data_required': False,
            'large_interference_array_required': False, 'negative_prior_results_preserved': True,
            'prior_learning_qualified': True, 'prior_head_L2_cost_unmatched': True}


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
    reg = {'study_id': STUDY_ID, 'registration_type': 'prospective_update_budget_diagnosis',
           'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': commit,
           'config': cfg, 'config_sha256': sha(config), 'registration_path': REGISTRATION,
           'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, **frozen, 'gate': gate,
           'notice': 'Prior supported and failed outcomes known; separate fixed-budget diagnosis with new seeds and criteria frozen before computation. Computational seeds are not animals.'}
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
