"""Byte and chronology gates for a read-only, descriptive gated-pathways audit.

Saved parent arrays and trial CSV bytes are analysis inputs. No graph constructor,
learner, or scientific parent producer is imported by this integrity module.
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

STUDY_ID = 'gated_pathways_v1'
MODULE = 'drosolearn.gated_pathways'
PROTOCOL = 'docs/gated-pathways-protocol.md'
CONFIG = 'configs/gated_pathways_v1.json'
REGISTRATION = 'results/registration_gated_pathways_v1.json'
RESULT = 'results/gated_pathways_v1'
BASE_TAG = 'gated-learning-v1'
BASE_COMMIT = 'b89e9f2baf1aed91b4451538c85ba3e9458df53b'
REPOSITORY = 'https://github.com/CauchyRiemannEquations/DrosoLearn'
PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json')
REUSED_SOURCE = frozenset({
    'pyproject.toml', 'requirements-lock.txt', 'src/drosolearn/__init__.py',
    'src/drosolearn/representation_envelope.py', 'src/drosolearn/balanced_envelope.py',
    'src/drosolearn/selective_envelope.py', 'src/drosolearn/normalization_envelope.py',
    'src/drosolearn/cone_envelope.py', 'src/drosolearn/pathways_envelope.py',
    'src/drosolearn/gated_envelope.py',
})
SOURCE_FILES = REUSED_SOURCE | frozenset({
    CONFIG, PROTOCOL, 'src/drosolearn/gated_pathways.py',
    'src/drosolearn/gated_pathways_envelope.py', 'scripts/run_gated_pathways.py',
    'scripts/verify_gated_pathways.py', 'tests/test_gated_pathways.py',
    'tests/test_gated_pathways_envelope.py', 'tests/test_gated_pathways_verify.py',
})
PARENT_FILES = frozenset({
    'results/registration_gated_learning_v1.json',
    'results/gated_learning_publication_proof.json',
    'results/gated_learning_archive_validation.json',
    'results/gated_learning_baseline_preservation.json',
    *('results/gated_learning_v1/' + name for name in
      (*PRIMARY, 'verification.json', 'artifact_hashes.json')),
})
EVOLVING_INDEXES = frozenset({'README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'})


def archived_prefreeze_gate(root, archive, reg, meta, proof, verification):
    """Reuse the parent's sealed finalization, with its original Git-byte links."""
    expected = {'results/gated_learning_baseline_preservation.json',
        'results/gated_learning_prefreeze_validation.json',
        'results/gated_learning_prefreeze_finalization.json',
        'results/gated_learning_prefreeze_tests/gated_prefreeze_tests_final.xml'}
    hashes = archive.get('prefreeze_receipts_committed_source_sha256')
    require(type(hashes) is dict and set(hashes) == expected
            and all(sha(safe(root, name)) == digest for name, digest in hashes.items()),
            'Exact archived prefreeze source-committed receipts required')
    committed(root, reg['source_git_commit'], hashes)
    tests = read(safe(root, 'results/gated_learning_prefreeze_validation.json'))
    final = read(safe(root, 'results/gated_learning_prefreeze_finalization.json'))
    baseline = read(safe(root, 'results/gated_learning_baseline_preservation.json'))
    require(tests.get('status') == final.get('status') == 'passed'
            and len(reg['source_files']) == 39 and tests.get('frozen_source_hashes') == reg['source_files']
            and tests.get('unique_passed_tests') == final.get('passed_unique_tests')
                == archive.get('prefreeze_passed_unique_tests_independently_counted') == 242
            and final.get('all39_final_source_hashes_verified') is True
            and final.get('new_cohort_graph_or_learning_started') is False
            and final.get('configuration_seeds_budget_thresholds_or_model_source_changed') is False
            and final.get('final_prefreeze_receipt_sha256') == hashes['results/gated_learning_prefreeze_validation.json'],
            'Parent final prefreeze/test identity and rejected initial snapshot must be preserved')
    initial = final.get('premature_unpublished_source_commit')
    require(archive.get('prefreeze_finalization_retained_initial_commit') == initial
            and archive.get('configuration_and_model_source_unchanged_during_finalization') is True,
            'Retained initial snapshot and unchanged-model archival facts required')
    ancestor(root, initial, reg['source_git_commit'])
    changed = {'results/gated_learning_prefreeze_finalization.json',
        'results/gated_learning_prefreeze_tests/gated_prefreeze_tests_final.xml',
        'results/gated_learning_prefreeze_validation.json', 'tests/test_gated_learning_verify.py'}
    require(set(archive.get('prefreeze_finalization_changed_paths', [])) == changed
            and set(git(root, 'diff', '--name-only', initial, reg['source_git_commit']).decode().splitlines()) == changed,
            'Only the four documented prefreeze-finalization paths may differ')
    committed(root, initial, {name: digest for name, digest in reg['source_files'].items()
                             if name != 'tests/test_gated_learning_verify.py'})
    chronology = archive.get('prefreeze_to_independent_verification_utc_order', {})
    keys = ('baseline_started', 'baseline_verified', 'final_tests_checked', 'finalization_checked',
            'registered', 'remote_bytes_checked', 'learning_started', 'learning_completed', 'independently_verified')
    require(set(chronology) == set(keys)
            and chronology['baseline_started'] == baseline['started_at_utc']
            and chronology['baseline_verified'] == baseline['verified_at_utc']
            and chronology['final_tests_checked'] == tests['checked_at_utc']
            and chronology['finalization_checked'] == final['checked_at_utc']
            and chronology['registered'] == reg['registered_at_utc']
            and chronology['remote_bytes_checked'] == proof['checked_at_utc']
            and chronology['learning_started'] == meta['started_at_utc']
            and chronology['learning_completed'] == meta['completed_at_utc']
            and chronology['independently_verified'] == verification['verified_at_utc'],
            'Archived original prefreeze-to-verification timestamps must match their receipts')
    times = [utc(chronology[key]) for key in keys]
    require(times == sorted(times) and times[3] < times[4] < times[5] < times[6]
            and times[-1] <= utc(archive['verified_at_utc']) <= datetime.now(timezone.utc),
            'Preserved UTC registration-before-learning and finalization-before-registration required')
    return {'source_committed_prefreeze_receipts_checked': len(hashes),
            'retained_initial_commit_ancestry_and_exact_four_finalized_paths_checked': True,
            'initial_model_and_configuration_source_bytes_unchanged': True,
            'parent_finalization_precedes_original_registration_and_learning': True}

def validate_envelope(root, cfg, config):
    require(type(cfg) is dict and cfg.get('study_id') == STUDY_ID
            and cfg.get('module') == MODULE and cfg.get('protocol') == PROTOCOL
            and cfg.get('scope') == 'retrospective_descriptive_gated_pathways',
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
    from .gated_envelope import parent_gate as inherited_gate, publication as prior_publication
    inherited = inherited_gate(root)
    require(git(root, 'rev-parse', BASE_TAG).decode().strip() == BASE_COMMIT,
            'Pinned completed gated-learning release required')
    ancestor(root, BASE_COMMIT)
    pinned = inventory(root, sorted(REUSED_SOURCE | PARENT_FILES))
    committed(root, BASE_COMMIT, pinned)
    prefix = 'results/gated_learning_v1/'
    reg_path = safe(root, 'results/registration_gated_learning_v1.json')
    reg, meta = read(reg_path), read(safe(root, prefix + 'metadata.json'))
    summary = read(safe(root, prefix + 'summary.json'))
    verification_path = safe(root, prefix + 'verification.json')
    verification = read(verification_path)
    require(reg.get('study_id') == meta.get('study_id') == summary.get('study_id')
            == verification.get('study_id') == 'gated_learning_v1'
            and meta.get('status') == meta.get('stage') == summary.get('status') == 'completed'
            and meta.get('analysis_only') is False and verification.get('status') == 'passed'
            and verification.get('scientific_producer_imported') is False
            and verification.get('new_helper_imported') is False,
            'Completed independently checked actual gated-learning parent required')
    require(safe(root, prefix + 'registration.json').read_bytes() == reg_path.read_bytes()
            and meta.get('registration_sha256') == verification.get('registration_sha256') == sha(reg_path)
            and meta.get('source_git_commit') == reg.get('source_git_commit')
            and verification.get('summary_sha256') == sha(safe(root, prefix + 'summary.json'))
            and meta.get('counts') == summary.get('counts'), 'Parent evidence identity mismatch')
    proof_name, proof = prior_publication(root, reg_path, reg,
        safe(root, 'results/gated_learning_publication_proof.json'))
    require(meta.get('publication_proof') == proof_name
            and meta.get('publication_proof_sha256') == sha(safe(root, proof_name)),
            'Parent publication-proof binding required')
    times = [utc(reg['registered_at_utc']), utc(proof['checked_at_utc']),
             utc(meta['started_at_utc']), utc(meta['completed_at_utc']),
             utc(verification['verified_at_utc'])]
    require(times[0] < times[1] < times[2] <= times[3] <= times[4] <= datetime.now(timezone.utc),
            'Parent prospective publication, learning, independent chronology required')
    artifacts = read(safe(root, prefix + 'artifact_hashes.json'))
    require(type(artifacts) is dict and set(artifacts) == set(PRIMARY),
            'Complete parent primary artifact receipt required')
    require(all(type(digest) is str and re.fullmatch(r'[0-9a-f]{64}', digest)
                and sha(safe(root, prefix + name)) == digest for name, digest in artifacts.items()),
            'Parent raw or primary artifact bytes changed')
    checks = verification.get('counts', {})
    replay_true = ('all_fresh_autonomous_choices_rewards_and_heads_replayed',
        'fixed_graph_encoding_weights_and_probe_activity_checked',
        'selective_324_hypothetical_directions_preflight_before_behavioral_rng_checked',
        'balanced_no_go_skips_optimizer_and_uses_zero_direction_checked',
        'balanced_uses_only_actual_own_obtained_feedback_checked',
        'independent_ambient_lstsq_dual_support_optima_checked',
        'case_certificates_are_global_support_bounds_not_solver_identity',
        'unknown_other_masks_before_and_after_match_checked',
        'evaluator_hidden_signs_do_not_enter_observed_direction_checked',
        'per_trial_correct_probability_and_aligned_margin_changes_checked',
        'all_three_hypotheses_separately_recomputed',
        'common_actual_movement_ceiling_not_equal_realized_budget_checked')
    require(all(checks.get(key) is True for key in replay_true)
            and checks.get('arrays') == 195 and checks.get('independently_replayed_training_actions') == 120000
            and checks.get('checkpoint_static_probes') == 6000 and checks.get('conditions') == 30
            and checks.get('model_seeds') == 10 and checks.get('head_update_calls') == 120000
            and checks.get('internal_update_calls') == checks.get('value_update_calls') == 0
            and checks.get('unique_balanced_states_independently_reoptimized') == 720
            and checks.get('extra_optimizer_and_memory_processing_cost_equalized') is False
            and checks.get('new_generalization_or_human_mechanism_claim') is False,
            'Prior complete actual-learning independent replay required')
    require(all(summary.get(key, {}).get('supported') is False for key in ('H_gain', 'H_acquisition'))
            and summary.get('H_protection', {}).get('supported') is True
            and summary['H_acquisition'].get('acquisition_pass_seeds') == 0
            and all(summary[key].get('all_numeric_and_information_gates_passed') is True
                    for key in ('H_gain', 'H_acquisition', 'H_protection')),
            'Prior unsupported gain/acquisition and operational protection outcome required')
    protection = summary['H_protection']
    require(protection.get('operational_protection_expected_by_design') is True
            and protection.get('novel_learning_mechanism_demonstrated') is False
            and protection.get('maximum_primary_seed_unknown_damage') <= protection.get('absolute_tolerance')
            and set(protection.get('comparisons', {})) == {'balanced', 'feedback_selective'}
            and all(value.get('nonworsening_seeds') == 10 and value.get('all_seeds_nonworsening') is True
                    for value in protection['comparisons'].values()),
            'Operational protection must not be relabeled as useful learning or a new mechanism')
    gate_keys = {'mode_uses_known_before_only', 'fourth_discovery_update_is_isolated',
        'first_balanced_mode_follows_full_discovery', 'exact_isolated_prefix_through_full_discovery',
        'gated_and_isolated_discovery_trials_equal', 'partial_gated_has_no_balanced_case',
        'only_full_observation_states_in_gated_case_cache', 'full_NoGo_has_zero_direction_and_no_case',
        'gate_switch_once_at_first_full_mode', 'all_methods_record_memory_before_gate'}
    require(set(summary.get('gate_validation', {})) == gate_keys
            and all(summary['gate_validation'][key] is True for key in gate_keys),
            'All ten actual prefeedback switch and identical-prefix gates required')
    require(summary.get('validation', {}).get('all_numeric_and_information_gates_passed') is True
            and all(value is True for value in summary.get('information_validation', {}).values())
            and summary.get('budget_validation', {}).get('actual_cumulative_cost_equalized') is False
            and summary['budget_validation'].get('extra_solver_and_information_processing_cost_equalized') is False,
            'Prior actual information and qualified unequal-cost gates required')
    for group in ('source_files', 'parent_files', 'graph_files', 'input_files'):
        values = reg.get(group)
        require(type(values) is dict and bool(values), 'Complete parent frozen closure required')
        require(all(sha(safe(root, name)) == digest for name, digest in values.items()),
                'Parent frozen bytes changed: ' + group)
        committed(root, reg['source_git_commit'], values)
    ancestor(root, reg['source_git_commit'], BASE_COMMIT)
    archive = read(safe(root, 'results/gated_learning_archive_validation.json'))
    require(archive.get('status') == 'passed'
            and archive.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True
            and archive.get('scientific_verification_sha256') == sha(verification_path)
            and archive.get('registration_sha256') == sha(reg_path)
            and archive.get('source_git_commit') == reg['source_git_commit']
            and archive.get('counts') == summary['counts'], 'Completed parent archive binding required')
    receipt = safe(root, 'results/gated_learning_baseline_preservation.json')
    require(archive.get('baseline_hash_receipt_sha256') == sha(receipt)
            and read(receipt).get('status') == 'passed', 'Original Git-blob baseline receipt required')
    require(archive.get('allowed_evolving_indexes') == sorted(EVOLVING_INDEXES)
            and read(receipt).get('allowed_evolving_indexes') == sorted(EVOLVING_INDEXES),
            'The exact three historical evolving indexes are required')
    finalization_gate = archived_prefreeze_gate(root, archive, reg, meta, proof, verification)
    snapshot = archive.get('current_file_sha256')
    require(type(snapshot) is dict and bool(snapshot)
            and set(reg['source_files']) | set(reg['graph_files']) | set(reg['input_files'])
            | {prefix + name for name in (*PRIMARY, 'verification.json', 'artifact_hashes.json')}
            | EVOLVING_INDEXES <= set(snapshot), 'Complete parent archive snapshot required')
    committed(root, BASE_COMMIT, snapshot)
    require(all(sha(safe(root, name)) == digest for name, digest in snapshot.items()
                if name not in EVOLVING_INDEXES), 'Historical immutable snapshot changed')
    require(all(snapshot.get(prefix + name) == digest for name, digest in artifacts.items()),
            'Parent raw artifact and archive hashes differ')
    old_status = json.loads(git(root, 'show', BASE_COMMIT + ':docs/repository-status.json'))
    current_status = read(safe(root, 'docs/repository-status.json'))
    require(all(status.get(key) is False for status in (old_status, current_status) for key in (
        'level4_similarity_supported', 'shared_neural_mechanism_verified',
        'human_shell_H_similarity', 'human_shell_H_prediction',
        'gated_learning_H_gain', 'gated_learning_H_acquisition'))
        and all(status.get('gated_learning_H_protection') is True for status in (old_status, current_status)),
        'Prior negative learning/human outcomes and closed mechanism gates required')
    return {'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT,
            'inherited_parent_gate': inherited, 'pinned_baseline_files': len(pinned),
            'historical_archive_snapshot_files': len(snapshot),
            'prior_complete_120000_action_replay_and_720_optima_reused_not_rerun': True,
            'parent_raw_arrays_CSV_source_data_Git_and_archive_bytes_checked': True,
            'prior_unsupported_gain_acquisition_and_operational_protection_preserved': True,
            'original_parent_prefreeze_finalization_gate': finalization_gate,
            'parent_gate_no_new_learner_graph_or_seed_calls': True,
            'whole_repository_literal_pass_not_repeated': True}


def frozen_inventory(root, cfg):
    return {group: inventory(root, cfg[group]) for group in ('source_files', 'parent_files')}


def validate_counts(cfg, summary, row_count):
    require(cfg.get('model_seeds') == list(range(17601, 17611))
            and type(cfg.get('training_trials')) is int and cfg['training_trials'] == 400,
            'Fixed saved actual-learning cohort and budget required')
    expected = {'analyzed_parent_trial_records': 120000, 'analyzed_destination_effects': 480000,
                'reference_gate_paths': 60, 'paired_comparison_paths': 240,
                'paired_comparison_trial_records': 96000, 'measurement_records': 1800, 'numeric_arrays': 12,
                'new_training_trials': 0, 'new_head_update_calls': 0,
                'new_internal_update_calls': 0, 'new_value_update_calls': 0,
                'new_random_draws': 0, 'fresh_model_constructions': 0, 'new_optimization_calls': 0}
    counts = summary.get('counts', {})
    require(type(counts) is dict and set(counts) == set(expected),
            'Exact saved-path work count keys required')
    require(all(type(counts[key]) is int and counts[key] == value for key, value in expected.items())
            and row_count == expected['measurement_records'],
            'Exact saved-path counts and zero new training/random/model/optimization work required')
    require(summary.get('study_id') == STUDY_ID and summary.get('status') == 'completed'
            and summary.get('scope') == 'retrospective_descriptive_gated_pathways',
            'Completed saved-path comparison required')


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
    reg = {'study_id': STUDY_ID, 'registration_type': 'retrospective_descriptive_gated_pathways',
           'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': commit,
           'config': cfg, 'config_sha256': sha(config), 'registration_path': REGISTRATION,
           'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, **frozen, 'gate': gate,
           'notice': 'Parent outcomes are already known; this saved-results descriptive decomposition and reporting are frozen before its arithmetic. Descriptive analysis only, without new training, seeds, animals or effect hypothesis tests.'}
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
            'new_graph_or_feature_constructions': 0, 'new_optimization_calls': 0,
            'new_human_experiment': False, 'shared_neural_mechanism_tested': False,
            'education_effect_tested': False, 'new_effect_hypothesis_test': False}
    try:
        arrays, rows = module.produce(cfg, root)
        require(type(arrays) is dict and len(arrays) == 12
                and all(type(key) is str and key and np.asarray(value).dtype.kind in 'biufc'
                        and np.all(np.isfinite(value)) for key, value in arrays.items()),
                'Finite numeric non-object analysis arrays required')
        np.savez_compressed(out / 'arrays.npz', **arrays)
        csv_write(out / 'measurements.csv', rows)
        meta['stage'] = 'prespecified_descriptive_summary'
        summary = module.summarize(cfg, arrays, rows, root)
        require(type(summary) is dict and type(summary.get('counts')) is dict,
                'Explicit descriptive summary counts required')
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
            and meta.get('analysis_only') is True
            and all(type(meta.get(key)) is int and meta[key] == 0 for key in (
                'new_training_calls', 'new_random_seed_draws', 'new_graph_or_feature_constructions',
                'new_optimization_calls'))
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
        require(type(meta.get('numeric_arrays')) is int and meta['numeric_arrays'] == len(arrays.files) == 12
                and all(arrays[key].dtype.kind in 'biufc' and np.all(np.isfinite(arrays[key]))
                        for key in arrays.files), 'Saved finite analysis array inventory differs')
    require(type(meta.get('measurement_records')) is int and meta['measurement_records'] == len(rows)
            and len(rows) > 0, 'Saved measurement row count differs')
    validate_counts(cfg, summary, len(rows))
    return cfg, reg, meta
