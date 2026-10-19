"""Prospective byte, chronology, and parent gates for the bounded selective-spillover study.

Only integrity primitives are reused. Scientific production stays lazy, after all
byte checks pass; independent envelope verification never imports the producer.
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

STUDY_ID = 'selective_spillover_v1'
MODULE = 'drosolearn.selective_spillover'
PROTOCOL = 'docs/selective-spillover-protocol.md'
CONFIG = 'configs/selective_spillover_v1.json'
REGISTRATION = 'results/registration_selective_spillover_v1.json'
RESULT = 'results/selective_spillover_v1'
BASE_TAG = 'signed-spillover-v1'
BASE_COMMIT = '4ee00454500b03e2ccae08ba22f2a6625526be83'
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
    'src/drosolearn/update_budget_geometry.py', 'src/drosolearn/update_budget.py',
    'src/drosolearn/update_budget_envelope.py', 'scripts/verify_update_budget.py',
})
SOURCE_FILES = REUSED_SOURCE | frozenset({
    CONFIG, PROTOCOL, 'src/drosolearn/selective_spillover.py',
    'src/drosolearn/selective_geometry.py', 'src/drosolearn/selective_envelope.py',
    'scripts/run_selective_spillover.py', 'scripts/verify_selective_spillover.py',
    'tests/test_selective_spillover.py', 'tests/test_selective_geometry.py',
    'tests/test_selective_envelope.py', 'tests/test_selective_spillover_verify.py',
})
PARENT_FILES = frozenset({
    'results/registration_signed_spillover_v1.json',
    'results/signed_spillover_publication_proof.json',
    'results/signed_spillover_archive_validation.json',
    'results/signed_spillover_baseline_preservation.json',
    'results/signed_spillover_v1/metadata.json',
    'results/signed_spillover_v1/summary.json',
    'results/signed_spillover_v1/verification.json',
    'results/signed_spillover_v1/artifact_hashes.json',
})
GRAPH_FILES = frozenset('data/flywire_783_mb_left_kc512_s701/' + name for name in
                        ('edges.csv', 'neurons.json', 'annotations.csv', 'provenance.json'))
INPUT_FILES = frozenset({
    'data/synthetic/pattern_rule_v1/input_catalog.json',
    'data/synthetic/representation_update_v1/input_catalog.json',
})
PINNED_FILES = REUSED_SOURCE | PARENT_FILES | GRAPH_FILES | INPUT_FILES
EVOLVING_INDEXES = frozenset({'README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'})
PARENT_REPLAY_TRUE = frozenset({
    'no_rng_or_learner_calls', 'source_destination_orientation_checked',
    'all_saved_checkpoints_checked', 'obtained_reward_only_nogo_zero_checked',
    'actual_margin_and_probability_telescoping_checked',
    'exact_masses_separate_from_tolerance_counts_checked',
    'finite_go_branch_expectation_not_sigmoid_of_expected_margin_checked',
    'pathwise_groups_not_isolated_causal_effect', 'no_new_behavioral_hypothesis_claim',
})


def validate_envelope(root, cfg, config):
    require(type(cfg) is dict and cfg.get('study_id') == STUDY_ID
            and cfg.get('module') == MODULE and cfg.get('protocol') == PROTOCOL,
            'Fixed selective-spillover implementation required')
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
            'Pinned completed signed-spillover release required')
    ancestor(root, BASE_COMMIT)
    pinned = {name: sha(safe(root, name)) for name in sorted(PINNED_FILES)}
    committed(root, BASE_COMMIT, pinned)
    study = 'signed_spillover_v1'
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
            and verification.get('scientific_producer_imported') is False
            and verification.get('new_helper_imported') is False,
            'Completed independently verified signed-spillover parent required')
    require(meta.get('registration_sha256') == verification.get('registration_sha256') == sha(reg_path)
            and meta.get('source_git_commit') == reg.get('source_git_commit')
            and verification.get('summary_sha256') == sha(safe(root, prefix + 'summary.json'))
            and meta.get('counts') == summary.get('counts'), 'Parent evidence identity mismatch')
    require(type(artifacts) is dict and set(artifacts) == set(PRIMARY)
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
    require(all(replay.get(key) is True for key in PARENT_REPLAY_TRUE)
            and replay.get('parent_trial_records_checked') == 120000
            and replay.get('trial_destination_effects_checked') == 480000
            and replay.get('checkpoint_source_destination_effects_checked') == 24000,
            'Prior complete independent saved-path analysis required')
    counts = summary.get('counts', {})
    require(all(counts.get(key) == 0 for key in (
                'new_training_trials', 'new_head_update_calls', 'new_internal_update_calls',
                'new_value_update_calls', 'new_random_draws', 'fresh_model_constructions'))
            and counts.get('analyzed_parent_trial_records') == 120000,
            'Prior read-only analysis must remain separate from new learning')
    prior = read(safe(root, 'results/spillover_tradeoff_v1/summary.json'))
    prior_verify = read(safe(root, 'results/spillover_tradeoff_v1/verification.json'))
    require(prior.get('study_id') == 'spillover_tradeoff_v1'
            and prior.get('status') == 'completed'
            and prior_verify.get('status') == 'passed'
            and prior_verify.get('scientific_producer_imported') is False
            and prior_verify.get('summary_sha256') == sha(safe(root, 'results/spillover_tradeoff_v1/summary.json'))
            and prior.get('H_tradeoff', {}).get('supported') is False
            and prior.get('H_acquisition', {}).get('supported') is False
            and prior.get('H_acquisition', {}).get('acquisition_pass_seeds') == 0,
            'Prior negative tradeoff and acquisition outcomes must remain preserved')
    validation = prior.get('budget_validation', {})
    require(all(validation.get(key) is True for key in (
                'conditional_scalar_matching_only', 'squared_step_sum_valid_all',
                'head_endpoint_within_path_all', 'common_bounds_valid_all',
                'all_static_numeric_gates_passed', 'analytic_probability_ceiling_valid_all'))
            and validation.get('actual_cumulative_cost_equalized') is False,
            'Prior qualified numeric bounds and unequal actual costs required')
    archive_path = safe(root, 'results/signed_spillover_archive_validation.json')
    archive = read(archive_path)
    require(archive.get('status') == 'passed'
            and archive.get('all_old_baseline_files_literal_preserved_except_three_indexes') is True
            and archive.get('scientific_verification_sha256') == sha(safe(root, prefix + 'verification.json'))
            and archive.get('registration_sha256') == sha(reg_path)
            and archive.get('source_git_commit') == reg['source_git_commit']
            and archive.get('counts') == summary.get('counts'),
            'Verified literal historical archive and parent identity required')
    receipt_path = safe(root, 'results/signed_spillover_baseline_preservation.json')
    require(archive.get('literal_baseline_receipt_sha256') == sha(receipt_path)
            and read(receipt_path).get('status') == 'passed',
            'Original literal baseline receipt required')
    snapshot = archive.get('current_file_sha256')
    require(type(snapshot) is dict and bool(snapshot)
            and set(reg['source_files']) | {prefix + name for name in (*PRIMARY, 'verification.json', 'artifact_hashes.json')}
            | EVOLVING_INDEXES <= set(snapshot), 'Complete parent archive snapshot required')
    # This bounded parent snapshot is checked literally, including its saved raw
    # arrays and CSV. It is separate from the earlier whole-repository 2 GB pass.
    committed(root, BASE_COMMIT, snapshot)
    require(all(sha(safe(root, name)) == digest for name, digest in snapshot.items()
                if name not in EVOLVING_INDEXES), 'Historical immutable baseline snapshot changed')
    require(all(snapshot.get(prefix + name) == artifacts[name] for name in PRIMARY),
            'Parent artifact receipt and archive snapshot differ')
    ancestor(root, reg['source_git_commit'], BASE_COMMIT)
    for group in ('source_files', 'parent_files'):
        values = reg.get(group)
        require(type(values) is dict and bool(values), 'Complete parent source/data closure required')
        require(all(sha(safe(root, name)) == digest for name, digest in values.items()),
                'Parent frozen source/data bytes changed: ' + group)
        committed(root, reg['source_git_commit'], values)
    old_status = json.loads(git(root, 'show', BASE_COMMIT + ':docs/repository-status.json'))
    current_status = read(safe(root, 'docs/repository-status.json'))
    require(all(status.get(key) is False for status in (old_status, current_status) for key in (
                'level4_similarity_supported', 'shared_neural_mechanism_verified',
                'human_shell_H_similarity', 'human_shell_H_prediction')),
            'Prior negative human comparisons and closed mechanism gates required')
    return {'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT,
            'small_parent_receipts_independently_verified': [study, 'spillover_tradeoff_v1'],
            'prior_scientific_replays_reused_not_rerun': True,
            'pinned_baseline_files': len(pinned), 'external_human_data_required': False,
            'large_interference_array_required': False, 'negative_prior_results_preserved': True,
            'parent_archive_baseline_snapshot_checked': True,
            'whole_repository_literal_pass_not_repeated': True,
            'parent_archive_raw_arrays_and_csv_literal_checked': True,
            'parent_source_commit_source_and_data_bytes_checked': True,
            'prior_readonly_analysis_not_new_learning': True,
            'prior_human_similarity_and_mechanism_negative_preserved': True,
            'historical_archive_snapshot_files': len(snapshot),
            'prior_spillover_learning_not_qualified': True,
            'prior_numeric_budget_qualified': True,
            'prior_actual_autonomous_costs_not_equalized': True}



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
    reg = {'study_id': STUDY_ID, 'registration_type': 'prospective_selective_spillover_diagnosis',
           'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': commit,
           'config': cfg, 'config_sha256': sha(config), 'registration_path': REGISTRATION,
           'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, **frozen, 'gate': gate,
           'notice': 'Prior supported and failed outcomes known; separate obtained-feedback selective-spillover diagnosis with new seeds and criteria frozen before computation. Computational seeds are not animals.'}
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
    return cfg, reg, meta
