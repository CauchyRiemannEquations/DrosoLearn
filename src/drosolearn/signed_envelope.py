"""Byte and chronology gates for a read-only, descriptive signed-spillover audit.

Saved parent arrays and trial CSV bytes are analysis inputs. No graph constructor,
learner, or scientific parent producer is imported by this integrity module.
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

STUDY_ID = 'signed_spillover_v1'
MODULE = 'drosolearn.signed_spillover'
PROTOCOL = 'docs/signed-spillover-protocol.md'
CONFIG = 'configs/signed_spillover_v1.json'
REGISTRATION = 'results/registration_signed_spillover_v1.json'
RESULT = 'results/signed_spillover_v1'
BASE_TAG = 'spillover-tradeoff-v1'
BASE_COMMIT = 'a9a60a82f73310c983f64589e631b2b8a9e3a052'
REPOSITORY = 'https://github.com/CauchyRiemannEquations/DrosoLearn'
PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json')
EVOLVING_INDEXES = frozenset({'README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'})
REUSED_SOURCE = frozenset({
    'pyproject.toml', 'requirements-lock.txt', 'src/drosolearn/__init__.py',
    'src/drosolearn/representation_envelope.py',
})
# This is an exact bounded closure, not a dynamically growing source-directory scan.
SOURCE_FILES = REUSED_SOURCE | frozenset({
    CONFIG, PROTOCOL, 'src/drosolearn/signed_spillover.py',
    'src/drosolearn/signed_geometry.py', 'src/drosolearn/signed_envelope.py',
    'scripts/run_signed_spillover.py', 'scripts/verify_signed_spillover.py',
    'tests/test_signed_spillover.py', 'tests/test_signed_geometry.py',
    'tests/test_signed_envelope.py', 'tests/test_signed_spillover_verify.py',
})
PARENT_FILES = frozenset({
    'results/registration_spillover_tradeoff_v1.json',
    'results/spillover_tradeoff_publication_proof.json',
    'results/spillover_archive_validation.json',
    *(('results/spillover_tradeoff_v1/' + name) for name in (*PRIMARY, 'verification.json', 'artifact_hashes.json')),
})
REPLAY_TRUE = frozenset({
    'all_checkpoints_and_trial_head_states_replayed',
    'fixed_internal_weights_encoder_and_probe_activity_checked',
    'five_fixed_alpha_directions_and_analytic_own_influences_checked',
    'exact_frozen_shared_and_projected_l2_endpoints_checked',
    'all_actual_step_path_energy_and_head_norms_replayed',
    'all_unit_direction_actual_step_and_path_ceilings_checked',
    'balanced_presentation_absolute_logit_and_probability_ceilings_checked',
    'primary_midpoint_contrasted_against_both_endpoints_checked',
    'acquisition_gate_separate_from_tradeoff_gate_checked',
    'no_best_alpha_selected_after_observation', 'no_hidden_nogo_outcome_updates',
    'no_generalization_or_human_mechanism_claim',
})


def validate_envelope(root, cfg, config):
    require(type(cfg) is dict and cfg.get('study_id') == STUDY_ID
            and cfg.get('module') == MODULE and cfg.get('protocol') == PROTOCOL
            and cfg.get('scope') == 'retrospective_descriptive_signed_spillover',
            'Fixed read-only descriptive audit implementation required')
    require(relative(root, config) == CONFIG, 'Fixed configuration path required')
    for key, expected in (('source_files', SOURCE_FILES), ('parent_files', PARENT_FILES)):
        values = cfg.get(key)
        require(type(values) is list and len(values) == len(expected) and set(values) == expected,
                'Exact unique frozen inventory required: ' + key)
    require(cfg.get('external_manifest') is None, 'No external data in this saved-results audit')
    interpretation = cfg.get('interpretation', {})
    require(all(interpretation.get(key) is False for key in (
        'new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim',
        'new_training', 'new_seed_draws', 'new_effect_hypothesis_test')),
        'Bounded read-only descriptive interpretation required')


def parent_gate(root):
    require(git(root, 'rev-parse', BASE_TAG).decode().strip() == BASE_COMMIT,
            'Pinned completed spillover-tradeoff release required')
    ancestor(root, BASE_COMMIT)
    pinned = inventory(root, sorted(REUSED_SOURCE | PARENT_FILES))
    committed(root, BASE_COMMIT, pinned)
    parent = 'spillover_tradeoff_v1'
    prefix = 'results/' + parent + '/'
    reg_path = safe(root, 'results/registration_' + parent + '.json')
    reg, meta = read(reg_path), read(safe(root, prefix + 'metadata.json'))
    summary = read(safe(root, prefix + 'summary.json'))
    verification_path = safe(root, prefix + 'verification.json')
    verification = read(verification_path)
    artifacts = read(safe(root, prefix + 'artifact_hashes.json'))
    require(reg.get('study_id') == meta.get('study_id') == summary.get('study_id')
            == verification.get('study_id') == parent
            and meta.get('status') == meta.get('stage') == summary.get('status') == 'completed'
            and verification.get('status') == 'passed'
            and verification.get('scientific_producer_imported') is False,
            'Completed independently verified spillover parent required')
    require((safe(root, prefix + 'registration.json')).read_bytes() == reg_path.read_bytes()
            and meta.get('registration_sha256') == verification.get('registration_sha256') == sha(reg_path)
            and meta.get('source_git_commit') == reg.get('source_git_commit')
            and verification.get('summary_sha256') == sha(safe(root, prefix + 'summary.json'))
            and meta.get('counts') == summary.get('counts'), 'Parent evidence identity mismatch')
    require(type(artifacts) is dict and set(artifacts) == set(PRIMARY),
            'Complete parent primary hash receipt required')
    for name, digest in artifacts.items():
        require(type(digest) is str and re.fullmatch(r'[0-9a-f]{64}', digest) is not None
                and sha(safe(root, prefix + name)) == digest,
                'Parent raw or primary artifact changed: ' + name)
    times = [utc(reg['registered_at_utc']), utc(meta['started_at_utc']),
             utc(meta['completed_at_utc']), utc(verification['verified_at_utc'])]
    require(times == sorted(times) and times[-1] <= datetime.now(timezone.utc),
            'Completed parent UTC chronology required')
    replay = verification.get('counts', {})
    require(all(replay.get(key) is True for key in REPLAY_TRUE)
            and replay.get('actual_autonomous_costs_equalized') is False
            and replay.get('extra_solver_cost_equalized') is False,
            'Prior complete independent replay with unequal autonomous costs required')
    require(replay.get('independently_replayed_training_actions') == 120000
            and replay.get('checkpoint_static_probes') == 6000
            and replay.get('internal_update_calls') == replay.get('value_update_calls') == 0,
            'Pinned full parent replay counts required')
    tradeoff, acquisition = summary.get('H_tradeoff', {}), summary.get('H_acquisition', {})
    require(tradeoff.get('supported') is False and acquisition.get('supported') is False
            and acquisition.get('acquisition_pass_seeds') == 0
            and tradeoff.get('numeric_spillover_qualified_all') is True
            and acquisition.get('numeric_spillover_qualified_all') is True,
            'Prior negative tradeoff and acquisition outcomes must remain preserved')
    budget = summary.get('budget_validation', {})
    require(all(budget.get(key) is True for key in (
        'conditional_scalar_matching_only', 'squared_step_sum_valid_all',
        'head_endpoint_within_path_all', 'common_bounds_valid_all',
        'all_static_numeric_gates_passed', 'analytic_probability_ceiling_valid_all'))
        and budget.get('actual_cumulative_cost_equalized') is False,
        'Prior fixed-budget and numeric gates required')
    ancestor(root, reg['source_git_commit'], BASE_COMMIT)
    for group in ('source_files', 'parent_files', 'graph_files', 'input_files'):
        hashes = reg.get(group)
        require(type(hashes) is dict and bool(hashes), 'Complete parent frozen inventory required')
        require(all(sha(safe(root, name)) == digest for name, digest in hashes.items()),
                'Parent frozen source/data bytes changed: ' + group)
        committed(root, reg['source_git_commit'], hashes)
    archive = read(safe(root, 'results/spillover_archive_validation.json'))
    require(archive.get('status') == 'passed'
            and archive.get('all_old_baseline_files_literal_preserved_except_three_indexes') is True
            and archive.get('scientific_verification_sha256') == sha(verification_path)
            and archive.get('registration_sha256') == sha(reg_path)
            and archive.get('source_git_commit') == reg['source_git_commit']
            and archive.get('counts') == summary.get('counts'),
            'Passed historical archive and exact parent identity required')
    snapshot = archive.get('current_file_sha256')
    require(type(snapshot) is dict and bool(snapshot)
            and set(reg['source_files']) | set(reg['graph_files']) | set(reg['input_files'])
            | {prefix + name for name in (*PRIMARY, 'verification.json', 'artifact_hashes.json')}
            | EVOLVING_INDEXES <= set(snapshot), 'Complete historical archive snapshot required')
    # The three live indexes may evolve. Their archived bytes must still occur
    # literally in the pinned completed parent commit, never in a rewritten receipt.
    for name, digest in snapshot.items():
        safe(root, name)
        if name not in EVOLVING_INDEXES:
            require(sha(safe(root, name)) == digest, 'Historical immutable snapshot changed: ' + name)
    committed(root, BASE_COMMIT, snapshot)
    return {'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT,
            'prior_full_replay_reused_not_rerun': True,
            'parent_arrays_and_trial_csv_sha256_checked': True,
            'parent_source_and_data_source_commit_bytes_checked': True,
            'parent_archive_baseline_snapshot_checked': True,
            'prior_negative_tradeoff_and_acquisition_preserved': True,
            'evolving_indexes_checked_in_baseline_commit_only': sorted(EVOLVING_INDEXES),
            'pinned_baseline_files': len(pinned), 'archive_snapshot_files': len(snapshot),
            'new_training_calls': 0, 'new_random_seed_draws': 0}


def frozen_inventory(root, cfg):
    return {group: inventory(root, cfg[group]) for group in ('source_files', 'parent_files')}


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
    reg = {'study_id': STUDY_ID, 'registration_type': 'retrospective_descriptive_signed_spillover',
           'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': commit,
           'config': cfg, 'config_sha256': sha(config), 'registration_path': REGISTRATION,
           'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, **frozen, 'gate': gate,
           'notice': 'Prior outcomes known; decomposition and reporting frozen before this saved-results audit. Descriptive analysis only, without new training, seeds, animals or effect hypothesis tests.'}
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
            'Registered audit/config identity changed')
    frozen = frozen_inventory(root, cfg)
    require(all(reg.get(group) == values for group, values in frozen.items()), 'Frozen file inventory changed')
    ancestor(root, reg['source_git_commit'])
    ancestor(root, BASE_COMMIT, reg['source_git_commit'])
    for values in frozen.values():
        committed(root, reg['source_git_commit'], values)
    require(reg.get('gate') == parent_gate(root), 'Frozen parent evidence gate changed')
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
    require(registered < checked <= datetime.now(timezone.utc), 'Publication must follow registration')
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
    require(utc(evidence['checked_at_utc']) < start, 'Remote byte proof must precede saved-results arithmetic')
    out.mkdir()
    with (out / 'registration.json').open('xb') as handle:
        handle.write(Path(registration).read_bytes())
    meta = {'study_id': STUDY_ID, 'status': 'running', 'stage': 'saved_results_analysis',
            'started_at_utc': start.isoformat(), 'source_git_commit': reg['source_git_commit'],
            'registration_sha256': sha(registration), 'config_sha256': sha(config),
            'publication_proof': proof_name, 'publication_proof_sha256': sha(proof),
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__},
            'analysis_only': True, 'new_training_calls': 0, 'new_random_seed_draws': 0,
            'new_human_experiment': False, 'shared_neural_mechanism_tested': False,
            'education_effect_tested': False, 'new_effect_hypothesis_test': False}
    try:
        arrays, rows = module.produce(cfg, root)
        require(type(arrays) is dict and bool(arrays)
                and all(type(key) is str and key and np.asarray(value).dtype.kind in 'biufc'
                        and np.all(np.isfinite(value)) for key, value in arrays.items()),
                'Finite numeric non-object analysis arrays required')
        np.savez_compressed(out / 'arrays.npz', **arrays)
        csv_write(out / 'measurements.csv', rows)
        meta['stage'] = 'prespecified_descriptive_summary'
        summary = module.summarize(cfg, arrays, rows)
        require(type(summary) is dict and type(summary.get('counts')) is dict,
                'Explicit descriptive summary counts required')
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
            and meta.get('analysis_only') is True
            and meta.get('new_training_calls') == meta.get('new_random_seed_draws') == 0
            and all(meta.get(key) is False for key in (
                'new_human_experiment', 'shared_neural_mechanism_tested',
                'education_effect_tested', 'new_effect_hypothesis_test')),
            'Completed frozen read-only analysis identity required')
    times = [utc(reg['registered_at_utc']), utc(evidence['checked_at_utc']),
             utc(meta['started_at_utc']), utc(meta['completed_at_utc'])]
    require(times[0] < times[1] < times[2] <= times[3] <= datetime.now(timezone.utc),
            'Registration-publication-analysis UTC chronology required')
    artifacts = read(directory / 'artifact_hashes.json')
    require(type(artifacts) is dict and set(artifacts) == set(PRIMARY),
            'Exact five primary artifact hashes required')
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
                        for key in arrays.files), 'Saved finite analysis array inventory differs')
    require(type(meta.get('measurement_records')) is int and meta['measurement_records'] == len(rows)
            and len(rows) > 0, 'Saved measurement row count differs')
    return cfg, reg, meta
