"""Byte and chronology gates for a read-only, descriptive self-blend-factors audit.

Saved parent arrays and trial CSV bytes are analysis inputs. No graph constructor,
learner, or scientific parent producer is imported by this integrity module.
"""
from __future__ import annotations

import csv
from collections import Counter
import xml.etree.ElementTree as ET
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

STUDY_ID = 'self_blend_factors_v1'
MODULE = 'drosolearn.self_blend_factors'
PROTOCOL = 'docs/self-blend-factors-protocol.md'
CONFIG = 'configs/self_blend_factors_v1.json'
REGISTRATION = 'results/registration_self_blend_factors_v1.json'
RESULT = 'results/self_blend_factors_v1'
BASE_TAG = 'self-blend-learning-v1'
BASE_COMMIT = '0d1a7abaf35e564eb397a52bb9f44a4f1b723907'
REPOSITORY = 'https://github.com/CauchyRiemannEquations/DrosoLearn'
PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json')
REUSED_SOURCE = frozenset({
    'pyproject.toml', 'requirements-lock.txt', 'src/drosolearn/__init__.py',
    'src/drosolearn/representation_envelope.py', 'src/drosolearn/balanced_envelope.py',
    'src/drosolearn/selective_envelope.py', 'src/drosolearn/normalization_envelope.py',
    'src/drosolearn/cone_envelope.py', 'src/drosolearn/pathways_envelope.py',
    'src/drosolearn/gated_envelope.py', 'src/drosolearn/gated_pathways_envelope.py',
    'src/drosolearn/gated_factors_envelope.py', 'src/drosolearn/self_blend_envelope.py',
})
SOURCE_FILES = REUSED_SOURCE | frozenset({
    CONFIG, PROTOCOL, 'src/drosolearn/self_blend_factors.py',
    'src/drosolearn/self_blend_factors_envelope.py', 'scripts/run_self_blend_factors.py',
    'scripts/verify_self_blend_factors.py', 'tests/test_self_blend_factors.py',
    'tests/test_self_blend_factors_envelope.py', 'tests/test_self_blend_factors_verify.py',
})
PARENT_FILES = frozenset({
    'results/registration_self_blend_learning_v1.json',
    'results/self_blend_learning_publication_proof.json',
    'results/self_blend_archive_validation.json',
    'results/self_blend_baseline_preservation.json',
    'results/self_blend_figures_v1/plotted_values.json',
    'results/self_blend_figures_v1/provenance.json',
    *('results/self_blend_learning_v1/' + name for name in
      (*PRIMARY, 'verification.json', 'artifact_hashes.json')),
})
EVOLVING_INDEXES = frozenset({'README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'})


def validate_envelope(root, cfg, config):
    require(type(cfg) is dict and cfg.get('study_id') == STUDY_ID
            and cfg.get('module') == MODULE and cfg.get('protocol') == PROTOCOL
            and cfg.get('scope') == 'retrospective_descriptive_self_blend_factors',
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


def archived_learning_prefreeze_gate(root, archive, reg, meta, proof, verification):
    validation = 'results/self_blend_prefreeze_validation.json'
    baseline = 'results/self_blend_baseline_preservation.json'
    junit = 'results/self_blend_prefreeze_tests/self_blend_prefreeze_final.xml'
    failures = {'results/self_blend_development/failure_01.zip', 'results/self_blend_development/failure_01.json',
        'results/self_blend_development/core_failure_01.zip', 'results/self_blend_development/core_failure_01_manifest.json'}
    paths = {validation, baseline, junit} | failures
    values = archive.get('prefreeze_receipts_committed_source_sha256')
    require(type(values) is dict and set(values) == paths
            and all(sha(safe(root, name)) == digest for name, digest in values.items()),
            'Exact completed-parent prefreeze and original failure ZIP bytes required')
    committed(root, reg['source_git_commit'], values)
    tests = read(safe(root, validation))
    require(tests.get('status') == 'passed' and tests.get('frozen_source_hashes') == reg['source_files']
            and tests.get('parent_byte_gate') == reg['gate'] and tests.get('unique_passed_tests') == 342
            and tests.get('new_unique_passed_tests') == 234 and tests.get('unchanged_related_regression_passed_tests') == 108
            and type(tests.get('actual_new_training_before_freeze')) is int and tests['actual_new_training_before_freeze'] == 0
            and type(tests.get('new_cohort_graph_feature_rng_calls_before_freeze')) is int
            and tests['new_cohort_graph_feature_rng_calls_before_freeze'] == 0
            and tests.get('synthetic_fixtures_only_before_freeze') is True,
            'Original complete 342-test no-premature-cohort snapshot required')
    receipts = tests.get('test_receipts'); failed_receipts = tests.get('development_failure_artifacts')
    require(type(receipts) is list and len(receipts) == 1 and receipts[0].get('path') == junit
            and receipts[0].get('sha256') == values[junit] and receipts[0].get('passed_unique_tests') == 342
            and type(failed_receipts) is list and len(failed_receipts) == 4
            and {item.get('path') for item in failed_receipts} == failures
            and all(item.get('sha256') == values[item['path']] for item in failed_receipts),
            'Exact original final JUnit and four development ZIP/manifest paths required')
    expected_classes = {'tests.test_balanced_learning': 61, 'tests.test_gated_learning': 47,
        'tests.test_self_blend_envelope': 56, 'tests.test_self_blend_learning': 47,
        'tests.test_self_blend_learning_verify': 131}
    cases = ET.parse(safe(root, junit)).findall('.//testcase')
    require(len(cases) == len({(c.get('classname'), c.get('name')) for c in cases}) == 342
            and all(c.find(key) is None for c in cases for key in ('failure', 'error', 'skipped'))
            and dict(Counter(c.get('classname') for c in cases)) == expected_classes
            and archive.get('prefreeze_passed_unique_tests_independently_counted') == 342
            and archive.get('prefreeze_unique_tests_by_class') == expected_classes,
            'Original 342 unique passing tests and exact class partition required')
    zipped = archive.get('original_development_failure_ZIPs_and_member_hashes_checked')
    first = read(safe(root, 'results/self_blend_development/failure_01.json'))
    second = read(safe(root, 'results/self_blend_development/core_failure_01_manifest.json'))
    require(type(zipped) is list and len(zipped) == 2
            and first.get('actual_cohort_started') is False and first.get('failed') == 6 and first.get('passed') == 95
            and second.get('actual_new_cohort_runs') == 0 and second.get('synthetic_tests_only') is True
            and second.get('failed_tests') == 6 and second.get('passed_tests') == 39,
            'Both prior invented failures must remain distinct from actual learning')
    expected_zip = {
        'results/self_blend_development/failure_01.zip': (first['artifacts'], 101, 6),
        'results/self_blend_development/core_failure_01.zip': (second['member_sha256'], 45, 6)}
    require({item.get('archive') for item in zipped} == set(expected_zip)
            and all(item.get('sha256') == values[item['archive']]
                and item.get('member_sha256') == expected_zip[item['archive']][0]
                and item.get('tests') == expected_zip[item['archive']][1]
                and item.get('failed') == expected_zip[item['archive']][2] for item in zipped)
            and second.get('archive_sha256') == values['results/self_blend_development/core_failure_01.zip'],
            'Prior independent ZIP-member validation must bind current source-committed raw archives')
    chronology = [read(safe(root, baseline))['verified_at_utc'], tests['checked_at_utc'], reg['registered_at_utc'],
        proof['checked_at_utc'], meta['started_at_utc'], meta['completed_at_utc'], verification['verified_at_utc']]
    times = list(map(utc, chronology))
    require(archive.get('prefreeze_publication_learning_verification_utc_order') == chronology
            and times == sorted(times) and times[2] < times[3] < times[4]
            and utc(second['preserved_at_utc']) <= times[1]
            and times[-1] <= utc(archive['verified_at_utc']) <= datetime.now(timezone.utc),
            'Original development/freeze/registration/publication/learning UTC chronology required')
    return {'source_committed_prefreeze_receipts_checked': 7, 'original_342_tests_and_four_failure_archive_paths_checked': True,
        'prior_ZIP_member_hash_validation_reused_by_raw_bytes': True,
        'parent_prefreeze_and_failure_preservation_precede_actual_learning': True}


def parent_gate(root):
    from .self_blend_envelope import parent_gate as inherited_gate, publication as prior_publication
    inherited = inherited_gate(root)
    require(git(root, 'rev-parse', BASE_TAG).decode().strip() == BASE_COMMIT,
            'Pinned completed self-blend-learning release required')
    ancestor(root, BASE_COMMIT)
    pinned = inventory(root, sorted(REUSED_SOURCE | PARENT_FILES))
    committed(root, BASE_COMMIT, pinned)
    prefix = 'results/self_blend_learning_v1/'
    reg_path = safe(root, 'results/registration_self_blend_learning_v1.json')
    reg, meta = read(reg_path), read(safe(root, prefix + 'metadata.json'))
    summary = read(safe(root, prefix + 'summary.json'))
    verification_path = safe(root, prefix + 'verification.json')
    verification = read(verification_path)
    require(reg.get('study_id') == meta.get('study_id') == summary.get('study_id')
            == verification.get('study_id') == 'self_blend_learning_v1'
            and meta.get('status') == meta.get('stage') == summary.get('status') == 'completed'
            and meta.get('analysis_only') is False and verification.get('status') == 'passed'
            and verification.get('scientific_producer_imported') is False
            and verification.get('new_helper_imported') is False,
            'Completed independently checked actual self-blend-learning parent required')
    require(safe(root, prefix + 'registration.json').read_bytes() == reg_path.read_bytes()
            and meta.get('registration_sha256') == verification.get('registration_sha256') == sha(reg_path)
            and meta.get('source_git_commit') == reg.get('source_git_commit')
            and verification.get('summary_sha256') == sha(safe(root, prefix + 'summary.json'))
            and meta.get('counts') == summary.get('counts'), 'Parent evidence identity mismatch')
    proof_name, proof = prior_publication(root, reg_path, reg,
        safe(root, 'results/self_blend_learning_publication_proof.json'))
    require(meta.get('publication_proof') == proof_name
            and meta.get('publication_proof_sha256') == sha(safe(root, proof_name)),
            'Parent publication-proof binding required')
    times = [utc(reg['registered_at_utc']), utc(proof['checked_at_utc']),
             utc(meta['started_at_utc']), utc(meta['completed_at_utc']),
             utc(verification['verified_at_utc'])]
    require(times[0] < times[1] < times[2] <= times[3] <= times[4] <= datetime.now(timezone.utc),
            'Parent prospective publication, learning and independent UTC chronology required')
    artifacts = read(safe(root, prefix + 'artifact_hashes.json'))
    require(type(artifacts) is dict and set(artifacts) == set(PRIMARY),
            'Complete parent primary artifact receipt required')
    require(all(type(digest) is str and re.fullmatch(r'[0-9a-f]{64}', digest)
                and sha(safe(root, prefix + name)) == digest for name, digest in artifacts.items()),
            'Parent raw or primary artifact bytes changed')
    checks = verification.get('counts', {})
    replay = ('all_fresh_autonomous_choices_rewards_and_heads_replayed',
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
    expected = {'arrays': 201, 'independently_replayed_training_actions': 144000,
        'model_seeds': 10, 'conditions': 36, 'checkpoint_static_probes': 7200,
        'head_update_calls': 144000, 'internal_update_calls': 0, 'value_update_calls': 0,
        'unique_balanced_states_independently_reoptimized': 943}
    require(all(checks.get(key) is True for key in replay)
            and all(type(checks.get(key)) is int and checks[key] == value for key, value in expected.items())
            and checks.get('extra_optimizer_and_memory_processing_cost_equalized') is False
            and checks.get('new_generalization_or_human_mechanism_claim') is False,
            'Prior complete actual action and 943-state independent replay required')
    counts = summary.get('counts', {})
    expected = {'training_feedback_trials': 144000, 'static_probe_cases': 7200,
        'model_initializations': 360, 'fixed_feature_constructions': 60, 'model_seeds': 10,
        'head_update_calls': 144000, 'internal_update_calls': 0, 'value_update_calls': 0,
        'internal_nonzero_steps': 0, 'value_nonzero_steps': 0, 'balanced_unique_solver_cases': 943}
    require(all(type(counts.get(key)) is int and counts[key] == value for key, value in expected.items())
            and type(counts.get('head_nonzero_steps')) is int and 0 < counts['head_nonzero_steps'] <= 144000
            and meta.get('numeric_arrays') == 201 and meta.get('measurement_records') == 144000,
            'Parent actual learning must not be relabeled as a zero-learning analysis')
    require(summary.get('H_gain', {}).get('supported') is False
            and summary.get('H_acquisition', {}).get('supported') is False
            and summary.get('H_protection', {}).get('supported') is True
            and summary.get('validation', {}).get('all_numeric_and_information_gates_passed') is True
            and all(value is True for value in summary.get('gate_validation', {}).values())
            and all(value is True for value in summary.get('information_validation', {}).values()),
            'Prior unsupported gain/acquisition and operational protection must remain preserved')
    for group, size in (('source_files', 44), ('parent_files', 11), ('graph_files', 4), ('input_files', 2)):
        values = reg.get(group)
        require(type(values) is dict and len(values) == size
                and all(sha(safe(root, name)) == digest for name, digest in values.items()),
                'Exact parent frozen closure or bytes changed: ' + group)
        committed(root, reg['source_git_commit'], values)
    ancestor(root, reg['source_git_commit'], BASE_COMMIT)
    archive = read(safe(root, 'results/self_blend_archive_validation.json'))
    require(archive.get('status') == 'passed'
            and archive.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True
            and archive.get('scientific_verification_sha256') == sha(verification_path)
            and archive.get('registration_sha256') == sha(reg_path)
            and archive.get('source_git_commit') == reg['source_git_commit'] and archive.get('counts') == counts
            and archive.get('H_gain') is False and archive.get('H_acquisition') is False
            and archive.get('H_protection') is True, 'Completed parent archive binding required')
    baseline_path = safe(root, 'results/self_blend_baseline_preservation.json'); baseline = read(baseline_path)
    require(archive.get('baseline_hash_receipt_sha256') == sha(baseline_path)
            and baseline.get('status') == 'passed'
            and archive.get('allowed_evolving_indexes') == baseline.get('allowed_evolving_indexes') == sorted(EVOLVING_INDEXES),
            'Exact three-index original Git-blob baseline receipt required')
    prefreeze_gate = archived_learning_prefreeze_gate(root, archive, reg, meta, proof, verification)
    snapshot = archive.get('current_file_sha256')
    required = set(reg['source_files']) | set(reg['graph_files']) | set(reg['input_files']) | PARENT_FILES | EVOLVING_INDEXES
    # The archive validation receipt cannot hash itself; it is pinned directly above.
    required.discard('results/self_blend_archive_validation.json')
    require(type(snapshot) is dict and required <= set(snapshot), 'Complete parent immutable archive snapshot required')
    committed(root, BASE_COMMIT, snapshot)
    require(all(sha(safe(root, name)) == digest for name, digest in snapshot.items() if name not in EVOLVING_INDEXES),
            'Historical immutable source/raw/presentation snapshot changed')
    require(all(snapshot.get(prefix + name) == digest for name, digest in artifacts.items()),
            'Parent raw artifacts and archive hashes differ')
    provenance = read(safe(root, 'results/self_blend_figures_v1/provenance.json'))
    require(provenance.get('verified_arrays_only') is True and provenance.get('new_training_trials') == 0
            and provenance.get('new_parameter_selection') is False
            and provenance.get('arrays_sha256') == artifacts['arrays.npz']
            and provenance.get('summary_sha256') == artifacts['summary.json']
            and provenance.get('verification_sha256') == sha(verification_path)
            and provenance.get('plotted_values_sha256') == sha(safe(root, 'results/self_blend_figures_v1/plotted_values.json')),
            'Prior plotted accounting must bind exact independently checked actual arrays')
    old_status = json.loads(git(root, 'show', BASE_COMMIT + ':docs/repository-status.json'))
    current_status = read(safe(root, 'docs/repository-status.json'))
    require(all(status.get(key) is False for status in (old_status, current_status) for key in (
        'level4_similarity_supported', 'shared_neural_mechanism_verified', 'human_shell_H_similarity', 'human_shell_H_prediction',
        'self_blend_learning_H_gain', 'self_blend_learning_H_acquisition'))
        and all(status.get('self_blend_learning_H_protection') is True for status in (old_status, current_status)),
        'Prior negative human/gain/acquisition outcomes and operational protection required')
    return {'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, 'inherited_parent_gate': inherited,
        'pinned_baseline_files': len(pinned), 'historical_archive_snapshot_files': len(snapshot),
        'prior_complete_144000_action_replay_and_943_B_states_reused_not_rerun': True,
        'original_learning_prefreeze_and_development_ZIP_gate': prefreeze_gate,
        'raw_source_graph_input_and_plotted_accounting_Git_archive_bytes_checked': True,
        'prior_unsupported_gain_acquisition_and_operational_protection_preserved': True,
        'parent_gate_no_new_learner_graph_seed_or_optimizer_calls': True,
        'whole_repository_literal_pass_not_repeated': True}


def frozen_inventory(root, cfg):
    return {group: inventory(root, cfg[group]) for group in ('source_files', 'parent_files')}


def validate_counts(cfg, summary, row_count):
    require(cfg.get('model_seeds') == list(range(17701, 17711))
            and type(cfg.get('training_trials')) is int and cfg['training_trials'] == 400,
            'Fixed saved actual-learning cohort and budget required')
    expected = {'analyzed_parent_trial_records': 144000, 'analyzed_destination_effects': 576000,
                'reference_gate_paths': 60, 'comparison_paths': 300,
                'paired_trial_records': 120000, 'measurement_records': 3600, 'numeric_arrays': 14,
                'new_training_trials': 0, 'new_head_update_calls': 0,
                'new_internal_update_calls': 0, 'new_value_update_calls': 0,
                'new_random_draws': 0, 'fresh_model_constructions': 0, 'new_optimization_calls': 0}
    counts = summary.get('counts', {})
    variable = {'paired_post_trial_records', 'eligible_both_Go_trials', 'corner_destination_effects'}
    require(type(counts) is dict and set(counts) == set(expected) | variable,
            'Exact saved-factor work count keys required')
    require(all(type(counts[key]) is int and counts[key] == value for key, value in expected.items())
            and all(type(counts[key]) is int for key in variable)
            and 0 <= counts['eligible_both_Go_trials'] <= counts['paired_post_trial_records'] <= 120000
            and counts['corner_destination_effects'] == 32 * counts['eligible_both_Go_trials']
            and type(row_count) is int and row_count == expected['measurement_records'],
            'Exact saved-factor counts and zero new training/random/model/optimization work required')
    require(summary.get('study_id') == STUDY_ID and summary.get('status') == 'completed'
            and summary.get('scope') == 'retrospective_descriptive_self_blend_factors',
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
    reg = {'study_id': STUDY_ID, 'registration_type': 'retrospective_descriptive_self_blend_factors',
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
        require(type(arrays) is dict and len(arrays) == 14
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
        require(type(meta.get('numeric_arrays')) is int and meta['numeric_arrays'] == len(arrays.files) == 14
                and all(arrays[key].dtype.kind in 'biufc' and np.all(np.isfinite(arrays[key]))
                        for key in arrays.files), 'Saved finite analysis array inventory differs')
    require(type(meta.get('measurement_records')) is int and meta['measurement_records'] == len(rows)
            and len(rows) > 0, 'Saved measurement row count differs')
    validate_counts(cfg, summary, len(rows))
    return cfg, reg, meta
