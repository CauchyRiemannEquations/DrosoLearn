"""Prospective integrity gates for actual observed-feedback balanced learning.

Historical analyses are checked by bytes and completed independent receipts.
New graph/learner work is imported lazily only after registration and publication.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import re
import sys

from .representation_envelope import (
    ancestor, committed, csv_write, git, inventory, read, relative, require,
    safe, sha, utc, write_new,
)
from .selective_envelope import GRAPH_FILES, INPUT_FILES, REUSED_SOURCE as SELECTIVE_REUSED

STUDY_ID = 'balanced_learning_v1'
MODULE = 'drosolearn.balanced_learning'
PROTOCOL = 'docs/balanced-learning-protocol.md'
CONFIG = 'configs/balanced_learning_v1.json'
REGISTRATION = 'results/registration_balanced_learning_v1.json'
RESULT = 'results/balanced_learning_v1'
BASE_TAG = 'cone-frontier-v1'
BASE_COMMIT = '56e1a31537083825e83d4402cfa9445e7e519b9b'
REPOSITORY = 'https://github.com/CauchyRiemannEquations/DrosoLearn'
PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json')
REUSED_SOURCE = SELECTIVE_REUSED | frozenset({
    'src/drosolearn/selective_geometry.py', 'src/drosolearn/selective_spillover.py',
    'src/drosolearn/selective_envelope.py', 'src/drosolearn/normalization_envelope.py',
    'src/drosolearn/cone_geometry.py', 'src/drosolearn/cone_envelope.py',
    'scripts/verify_selective_spillover.py', 'scripts/verify_cone_frontier.py',
    'scripts/verify_normalization_loss.py',
})
SOURCE_FILES = REUSED_SOURCE | frozenset({
    CONFIG, PROTOCOL, 'src/drosolearn/balanced_learning.py',
    'src/drosolearn/balanced_envelope.py', 'scripts/run_balanced_learning.py',
    'scripts/verify_balanced_learning.py', 'tests/test_balanced_learning.py',
    'tests/test_balanced_envelope.py', 'tests/test_balanced_learning_verify.py',
})
PARENT_FILES = frozenset({
    'results/registration_cone_frontier_v1.json',
    'results/cone_frontier_publication_proof.json',
    'results/cone_frontier_archive_validation.json',
    'results/cone_frontier_baseline_preservation.json',
    *('results/cone_frontier_v1/' + name for name in
      (*PRIMARY, 'verification.json', 'artifact_hashes.json')),
})
EVOLVING_INDEXES = frozenset({'README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'})
CONE_REPLAY_TRUE = frozenset({
    'active_faces_reoptimized_in_original_feature_space',
    'nonnegative_dual_upper_bound_independently_checked',
    'certified_direction_not_exact_degenerate_vector_identity',
    'observed_objectives_constraints_use_only_actual_observations',
    'oracle_full_information_separate_from_observed_mode',
    'no_go_cases_skipped_and_all_zero_effects', 'same_saved_scalar_not_autonomous_policy',
    'parent_full_trajectory_replay_reused_by_frozen_hashes',
    'no_rng_learner_or_graph_construction_calls',
})


def validate_envelope(root, cfg, config):
    require(type(cfg) is dict and cfg.get('study_id') == STUDY_ID
            and cfg.get('module') == MODULE and cfg.get('protocol') == PROTOCOL
            and cfg.get('scope') == 'prospective_observed_balanced_learning',
            'Fixed prospective actual-learning implementation required')
    require(relative(root, config) == CONFIG, 'Fixed configuration path required')
    for key, expected in (('source_files', SOURCE_FILES), ('parent_files', PARENT_FILES),
                          ('graph_files', GRAPH_FILES), ('input_files', INPUT_FILES)):
        values = cfg.get(key)
        require(type(values) is list and len(values) == len(expected) and set(values) == expected,
                'Exact unique frozen inventory required: ' + key)
    require(cfg.get('external_manifest') is None, 'No new human/raw external data in this experiment')
    interpretation = cfg.get('interpretation', {})
    require(all(interpretation.get(key) is False for key in (
        'new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim')),
        'Bounded actual-learning interpretation required')


def parent_gate(root):
    from .cone_envelope import parent_gate as inherited_parent_gate, publication as cone_publication
    inherited = inherited_parent_gate(root)
    require(git(root, 'rev-parse', BASE_TAG).decode().strip() == BASE_COMMIT,
            'Pinned completed cone-frontier release required')
    ancestor(root, BASE_COMMIT)
    pinned = inventory(root, sorted(REUSED_SOURCE | PARENT_FILES | GRAPH_FILES | INPUT_FILES))
    committed(root, BASE_COMMIT, pinned)
    prefix = 'results/cone_frontier_v1/'
    reg_path = safe(root, 'results/registration_cone_frontier_v1.json')
    reg, meta = read(reg_path), read(safe(root, prefix + 'metadata.json'))
    summary = read(safe(root, prefix + 'summary.json'))
    verification_path = safe(root, prefix + 'verification.json')
    verification = read(verification_path)
    require(reg.get('study_id') == meta.get('study_id') == summary.get('study_id')
            == verification.get('study_id') == 'cone_frontier_v1'
            and meta.get('status') == meta.get('stage') == summary.get('status') == 'completed'
            and meta.get('analysis_only') is True and verification.get('status') == 'passed'
            and verification.get('scientific_producer_imported') is False
            and verification.get('new_helper_imported') is False,
            'Completed independently checked descriptive cone parent required')
    require(safe(root, prefix + 'registration.json').read_bytes() == reg_path.read_bytes()
            and meta.get('registration_sha256') == verification.get('registration_sha256') == sha(reg_path)
            and meta.get('source_git_commit') == reg.get('source_git_commit')
            and verification.get('summary_sha256') == sha(safe(root, prefix + 'summary.json'))
            and meta.get('counts') == summary.get('counts'), 'Parent evidence identity mismatch')
    proof_name, proof = cone_publication(root, reg_path, reg,
                                       safe(root, 'results/cone_frontier_publication_proof.json'))
    require(meta.get('publication_proof') == proof_name
            and meta.get('publication_proof_sha256') == sha(safe(root, proof_name)),
            'Parent remote-publication proof binding required')
    times = [utc(reg['registered_at_utc']), utc(proof['checked_at_utc']),
             utc(meta['started_at_utc']), utc(meta['completed_at_utc']),
             utc(verification['verified_at_utc'])]
    require(times[0] < times[1] < times[2] <= times[3] <= times[4] <= datetime.now(timezone.utc),
            'Parent publication-before-arithmetic and independent UTC chronology required')
    artifacts = read(safe(root, prefix + 'artifact_hashes.json'))
    require(type(artifacts) is dict and set(artifacts) == set(PRIMARY),
            'Complete parent primary artifact receipt required')
    require(all(type(digest) is str and re.fullmatch(r'[0-9a-f]{64}', digest)
                and sha(safe(root, prefix + name)) == digest for name, digest in artifacts.items()),
            'Parent raw/primary artifact bytes changed')
    replay = verification.get('counts', {})
    require(all(replay.get(key) is True for key in CONE_REPLAY_TRUE)
            and replay.get('arrays') == 15 and replay.get('parent_trial_records_checked') == 24000
            and replay.get('candidate_trial_records_checked') == 168000
            and replay.get('candidate_destination_effects_checked') == 672000
            and replay.get('measurement_records_checked') == 1260
            and replay.get('new_training_trials') == replay.get('new_random_draws') == 0,
            'Prior complete independent geometric frontier receipt required')
    counts = summary.get('counts', {})
    require(all(counts.get(key) == 0 for key in (
        'new_training_trials', 'new_head_update_calls', 'new_internal_update_calls',
        'new_value_update_calls', 'new_random_draws', 'fresh_model_constructions'))
        and counts.get('analyzed_parent_trial_records') == 24000,
        'Prior saved-scalar frontier must not be relabeled as autonomous learning')
    require(all(summary.get('validation', {}).get(key) is True for key in (
        'all_unit_caps_passed', 'all_observed_and_oracle_constraints_passed',
        'all_case_certificates_passed', 'all_phase_balances_passed')),
        'Prior bounded and information-restricted geometric certificates required')
    for group in ('source_files', 'parent_files'):
        values = reg.get(group)
        require(type(values) is dict and bool(values), 'Parent frozen closure required')
        require(all(sha(safe(root, name)) == digest for name, digest in values.items()),
                'Parent frozen bytes changed: ' + group)
        committed(root, reg['source_git_commit'], values)
    ancestor(root, reg['source_git_commit'], BASE_COMMIT)
    archive = read(safe(root, 'results/cone_frontier_archive_validation.json'))
    require(archive.get('status') == 'passed'
            and archive.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True
            and archive.get('scientific_verification_sha256') == sha(verification_path)
            and archive.get('registration_sha256') == sha(reg_path)
            and archive.get('source_git_commit') == reg['source_git_commit']
            and archive.get('counts') == counts, 'Verified completed parent archive required')
    receipt = safe(root, 'results/cone_frontier_baseline_preservation.json')
    require(archive.get('baseline_hash_receipt_sha256') == sha(receipt)
            and read(receipt).get('status') == 'passed', 'Original Git-blob baseline receipt required')
    snapshot = archive.get('current_file_sha256')
    require(type(snapshot) is dict and bool(snapshot)
            and set(reg['source_files']) | {prefix + name for name in
                (*PRIMARY, 'verification.json', 'artifact_hashes.json')}
            | EVOLVING_INDEXES <= set(snapshot), 'Complete parent archive snapshot required')
    committed(root, BASE_COMMIT, snapshot)
    require(all(sha(safe(root, name)) == digest for name, digest in snapshot.items()
                if name not in EVOLVING_INDEXES), 'Historical immutable snapshot changed')
    require(all(snapshot.get(prefix + name) == digest for name, digest in artifacts.items()),
            'Parent raw artifact and archive hashes differ')
    current_status = read(safe(root, 'docs/repository-status.json'))
    old_status = json.loads(git(root, 'show', BASE_COMMIT + ':docs/repository-status.json'))
    require(all(status.get(key) is False for status in (current_status, old_status) for key in (
        'level4_similarity_supported', 'shared_neural_mechanism_verified',
        'human_shell_H_similarity', 'human_shell_H_prediction',
        'selective_spillover_H_selective', 'selective_spillover_H_acquisition')),
        'Prior negative learning/human outcomes and closed mechanism gates required')
    return {'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT,
            'inherited_parent_gate': inherited, 'pinned_baseline_files': len(pinned),
            'historical_archive_snapshot_files': len(snapshot),
            'prior_cone_independent_certificate_and_information_gates_reused': True,
            'prior_frontier_is_not_actual_learning': True,
            'parent_source_data_Git_and_archive_bytes_checked': True,
            'prior_negative_learning_and_human_results_preserved': True,
            'parent_gate_no_new_learner_graph_or_seed_calls': True,
            'whole_repository_literal_pass_not_repeated': True}


def frozen_inventory(root, cfg):
    return {group: inventory(root, cfg[group]) for group in
            ('source_files', 'parent_files', 'graph_files', 'input_files')}


def validate_counts(cfg, summary, row_count):
    seeds, trials = cfg.get('model_seeds'), cfg.get('training_trials')
    checkpoints = cfg.get('checkpoints')
    require(type(seeds) is list and len(seeds) == 10 and len(set(seeds)) == 10
            and type(trials) is int and trials == 400
            and checkpoints == [0, 50, 100, 200, 400], 'Fixed actual-learning dimensions required')
    conditions, groups = 24, 6
    expected = {'training_feedback_trials': conditions * len(seeds) * trials,
                'static_probe_cases': conditions * len(seeds) * len(checkpoints) * 4,
                'model_initializations': conditions * len(seeds),
                'fixed_feature_constructions': groups * len(seeds),
                'model_seeds': len(seeds), 'head_update_calls': conditions * len(seeds) * trials,
                'internal_update_calls': 0, 'value_update_calls': 0,
                'internal_nonzero_steps': 0, 'value_nonzero_steps': 0}
    counts = summary.get('counts', {})
    require(type(counts) is dict and all(type(counts.get(key)) is int and counts[key] == value
                for key, value in expected.items()) and row_count == expected['training_feedback_trials'],
            'Actual nonzero training and exact static/feature/model work counts required')
    require(type(counts.get('head_nonzero_steps')) is int
            and 0 < counts['head_nonzero_steps'] <= expected['head_update_calls'],
            'Actual nonzero output learning must be recorded')
    require(summary.get('study_id') == STUDY_ID and summary.get('status') == 'completed',
            'Completed prospective actual-learning summary required')


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
    reg = {'study_id': STUDY_ID, 'registration_type': 'prospective_observed_balanced_learning',
           'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': commit,
           'config': cfg, 'config_sha256': sha(config), 'registration_path': REGISTRATION,
           'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, **frozen, 'gate': gate,
           'notice': 'Prior geometric and failed learning outcomes known; actual observed-feedback balanced learning with a separate new cohort, fixed unit caps and unknown-harm assessment. Frozen before new graph, seed, or learner computation. Computational seeds are not animals.'}
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
            'Registered prospective study/config identity changed')
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
    require(utc(evidence['checked_at_utc']) < start, 'Remote byte proof must precede new learning')
    out.mkdir()
    with (out / 'registration.json').open('xb') as handle:
        handle.write(Path(registration).read_bytes())
    meta = {'study_id': STUDY_ID, 'status': 'running', 'stage': 'actual_learning',
            'started_at_utc': start.isoformat(), 'source_git_commit': reg['source_git_commit'],
            'registration_sha256': sha(registration), 'config_sha256': sha(config),
            'publication_proof': proof_name, 'publication_proof_sha256': sha(proof),
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__},
            'analysis_only': False, 'new_human_experiment': False,
            'shared_neural_mechanism_tested': False, 'education_effect_tested': False}
    try:
        arrays, rows = module.produce(cfg, root)
        require(type(arrays) is dict and bool(arrays)
                and all(type(key) is str and key and np.asarray(value).dtype.kind in 'biufc'
                        and np.all(np.isfinite(value)) for key, value in arrays.items()),
                'Finite numeric non-object actual-learning arrays required')
        np.savez_compressed(out / 'arrays.npz', **arrays)
        csv_write(out / 'measurements.csv', rows)
        meta['stage'] = 'prespecified_summary'
        summary = module.summarize(cfg, arrays, rows)
        validate_counts(cfg, summary, len(rows))
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
    require(meta.get('study_id') == STUDY_ID and meta.get('status') == meta.get('stage') == 'completed'
            and meta.get('source_git_commit') == reg['source_git_commit']
            and meta.get('registration_sha256') == sha(registration) and meta.get('config_sha256') == sha(config)
            and meta.get('publication_proof') == proof_name
            and meta.get('publication_proof_sha256') == sha(publication_proof)
            and meta.get('analysis_only') is False
            and all(meta.get(key) is False for key in (
                'new_human_experiment', 'shared_neural_mechanism_tested', 'education_effect_tested')),
            'Completed frozen actual-learning identity and bounded interpretation required')
    times = [utc(reg['registered_at_utc']), utc(evidence['checked_at_utc']),
             utc(meta['started_at_utc']), utc(meta['completed_at_utc'])]
    require(times[0] < times[1] < times[2] <= times[3] <= datetime.now(timezone.utc),
            'Prospective publication-before-learning UTC chronology required')
    artifacts = read(directory / 'artifact_hashes.json')
    require(type(artifacts) is dict and set(artifacts) == set(PRIMARY), 'Exact five primary artifact hashes required')
    for name, checksum in artifacts.items():
        require(sha(safe(directory, name)) == checksum, 'Primary artifact changed: ' + name)
    summary = read(directory / 'summary.json')
    require(type(meta.get('counts')) is dict and meta['counts'] == summary.get('counts'),
            'Summary and metadata counts differ')
    with (directory / 'measurements.csv').open(encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
                'Unique saved measurement columns required')
        rows = list(reader)
    with np.load(directory / 'arrays.npz', allow_pickle=False) as arrays:
        require(type(meta.get('numeric_arrays')) is int and meta['numeric_arrays'] == len(arrays.files)
                and len(arrays.files) > 0
                and all(arrays[key].dtype.kind in 'biufc' and np.all(np.isfinite(arrays[key]))
                        for key in arrays.files), 'Saved finite numeric array inventory differs')
    require(type(meta.get('measurement_records')) is int and meta['measurement_records'] == len(rows)
            and len(rows) > 0, 'Saved measurement row count differs')
    validate_counts(cfg, summary, len(rows))
    return cfg, reg, meta
