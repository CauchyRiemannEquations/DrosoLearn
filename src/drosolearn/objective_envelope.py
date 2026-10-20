"""Prospective integrity gates for whole-evaluator U/P output learning.

Parent saved analyses are reused through immutable source, raw-data and independent
receipts. New graph/cohort/learner work starts only after frozen publication.
"""
from __future__ import annotations

import csv
import gzip
import io
from collections import Counter
import hashlib
from zipfile import ZipFile
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

from .representation_envelope import (
    ancestor, committed, csv_write, git, inventory, read, relative, require,
    safe, sha, utc, write_new,
)
from .selective_envelope import GRAPH_FILES, INPUT_FILES
from .self_blend_envelope import SOURCE_FILES as SELF_BLEND_SOURCE

STUDY_ID = 'objective_learning_v1'
MODULE = 'drosolearn.objective_learning'
PROTOCOL = 'docs/objective-learning-protocol.md'
CONFIG = 'configs/objective_learning_v1.json'
REGISTRATION = 'results/registration_objective_learning_v1.json'
RESULT = 'results/objective_learning_v1'
BASE_TAG = 'probability-alignment-v1'
BASE_COMMIT = '9159aa764d44183c21468974ba3b0f208d676731'
REPOSITORY = 'https://github.com/CauchyRiemannEquations/DrosoLearn'
PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv.gz', 'summary.json')
PARENT_PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json')
OLD_STUDY_ONLY = frozenset({'configs/self_blend_learning_v1.json',
    'docs/self-blend-learning-protocol.md', 'scripts/run_self_blend_learning.py',
    'tests/test_self_blend_learning.py', 'tests/test_self_blend_envelope.py',
    'tests/test_self_blend_learning_verify.py'})
REUSED_SOURCE = (SELF_BLEND_SOURCE - OLD_STUDY_ONLY) | frozenset({
    'src/drosolearn/self_blend_factors_envelope.py', 'src/drosolearn/probability_envelope.py'})
SOURCE_FILES = REUSED_SOURCE | frozenset({
    CONFIG, PROTOCOL, 'src/drosolearn/objective_learning.py',
    'src/drosolearn/objective_envelope.py', 'scripts/run_objective_learning.py',
    'scripts/verify_objective_learning.py', 'tests/test_objective_learning.py',
    'tests/test_objective_learning_envelope.py', 'tests/test_objective_learning_verify.py'})
PARENT_FILES = frozenset({
    'results/registration_probability_alignment_v1.json',
    'results/probability_alignment_publication_proof.json',
    'results/probability_alignment_archive_validation.json',
    'results/probability_alignment_baseline_preservation.json',
    'results/probability_alignment_presentation_validation.json',
    'results/probability_alignment_archive_finalization_attempt01.json',
    'results/probability_alignment_archive_finalization_attempt01.zip',
    *('results/probability_alignment_v1/' + name for name in
      (*PARENT_PRIMARY, 'verification.json', 'artifact_hashes.json'))})
EVOLVING_INDEXES = frozenset({'README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'})
NUMERIC_ARRAY_COUNT = 215
MEASUREMENT_COLUMN_COUNT = 96
METHODS = ('shared', 'isolated', 'feedback_selective', 'balanced', 'gated_balanced',
    'gated_self_blend', 'gated_uniform', 'gated_probability')


def validate_envelope(root, cfg, config):
    require(type(cfg) is dict and cfg.get('study_id') == STUDY_ID
            and cfg.get('module') == MODULE and cfg.get('protocol') == PROTOCOL
            and cfg.get('scope') == 'prospective_observed_objective_learning',
            'Fixed prospective actual-learning implementation required')
    require(relative(root, config) == CONFIG, 'Fixed configuration path required')
    for key, expected in (('source_files', SOURCE_FILES), ('parent_files', PARENT_FILES),
                          ('graph_files', GRAPH_FILES), ('input_files', INPUT_FILES)):
        values = cfg.get(key)
        require(type(values) is list and len(values) == len(expected) and set(values) == expected,
                'Exact unique frozen inventory required: ' + key)
    require(cfg.get('external_manifest') is None, 'No new human/raw external data in this experiment')
    require(cfg.get('measurement_storage') == 'lossless_deterministic_gzip_csv_crlf',
            'Fixed lossless deterministic gzip CSV storage required')
    interpretation = cfg.get('interpretation', {})
    require(all(interpretation.get(key) is False for key in (
        'new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim')),
        'Bounded actual-learning interpretation required')


def archived_alignment_prefreeze_gate(root, archive, reg, meta, proof, verified):
    """Reuse the exact completed 261-test snapshot and original failed fixtures."""
    baseline = 'results/probability_alignment_baseline_preservation.json'
    validation = 'results/probability_alignment_prefreeze_validation.json'
    junit = 'results/probability_alignment_prefreeze_tests/probability_alignment_prefreeze_final.xml'
    development = {'results/probability_alignment_development/probability_alignment_envelope_development_failure_01' + ext
        for ext in ('.zip', '.json')}
    paths = {baseline, validation, junit} | development
    values = archive.get('prefreeze_receipts_committed_source_sha256')
    require(type(values) is dict and set(values) == paths
            and all(sha(safe(root, name)) == digest for name, digest in values.items()),
            'Exact original parent prefreeze and failed-fixture bytes required')
    committed(root, reg['source_git_commit'], values)
    tests = read(safe(root, validation))
    classes = {'tests.test_cone_geometry': 45, 'tests.test_probability_alignment': 38,
        'tests.test_probability_alignment_envelope': 48, 'tests.test_probability_alignment_verify': 83,
        'tests.test_self_blend_learning': 47}
    require(tests.get('status') == 'passed' and tests.get('frozen_source_hashes') == reg['source_files']
            and tests.get('parent_byte_gate') == reg['gate'] and tests.get('unique_passed_tests') == 261
            and tests.get('new_unique_passed_tests') == 169
            and tests.get('unchanged_related_regression_passed_tests') == 92
            and tests.get('unique_passed_tests_by_class') == classes
            and tests.get('synthetic_fixtures_only_before_freeze') is True
            and all(type(tests.get(key)) is int and tests[key] == 0 for key in (
                'actual_new_saved_alignment_aggregation_before_freeze',
                'actual_new_direction_optimizations_before_freeze', 'new_training_random_graph_calls_before_freeze')),
            'Original parent 261-test frozen no-premature-work snapshot required')
    receipts = tests.get('test_receipts'); failures = tests.get('development_failure_artifacts')
    require(type(receipts) is list and len(receipts) == 1 and receipts[0].get('path') == junit
            and receipts[0].get('sha256') == values[junit] and receipts[0].get('passed_unique_tests') == 261
            and type(failures) is list and len(failures) == 2
            and {item.get('path') for item in failures} == development
            and all(item.get('sha256') == values[item['path']] for item in failures)
            and archive.get('development_failure_artifact_bytes_preserved') == failures,
            'Original final JUnit and two failed-fixture artifacts must remain bound')
    cases = ET.parse(safe(root, junit)).findall('.//testcase')
    require(len(cases) == len({(case.get('classname'), case.get('name')) for case in cases}) == 261
            and all(case.find(key) is None for case in cases for key in ('failure', 'error', 'skipped'))
            and dict(Counter(case.get('classname') for case in cases)) == classes
            and archive.get('prefreeze_unique_tests_by_class') == classes
            and archive.get('prefreeze_passed_unique_tests_independently_counted') == 261,
            'Original exact unique parent test partition required')
    failed_zip = next(name for name in development if name.endswith('.zip'))
    failed_note = read(safe(root, next(name for name in development if name.endswith('.json'))))
    member_hashes = {name.replace('\\', '/'): digest for name, digest in failed_note['member_sha256'].items()}
    require(failed_note.get('archive_sha256') == values[failed_zip]
            and failed_note.get('test_count') == 48 and failed_note.get('failed_test_count') == 2
            and failed_note.get('actual_saved_parent_aggregation_started') is False
            and failed_note.get('actual_new_learning_or_direction_optimization_started') is False
            and failed_note.get('scientific_design_or_threshold_change') is False
            and len(member_hashes) == len(failed_note['member_sha256']) == 2,
            'Original two failed invented directory fixtures must not become scientific failures')
    with ZipFile(safe(root, failed_zip)) as archive_zip:
        require(len(archive_zip.namelist()) == 2 and set(archive_zip.namelist()) == set(member_hashes)
                and all(hashlib.sha256(archive_zip.read(name)).hexdigest() == digest for name, digest in member_hashes.items()),
                'Original failed-fixture ZIP member bytes changed')
        raw_cases = ET.fromstring(archive_zip.read('probability_alignment_envelope_tests.xml')).findall('.//testcase')
        require(len(raw_cases) == 48 and sum(case.find('failure') is not None for case in raw_cases) == 2,
                'Original forty-six passing and two failing invented cases required')
    chronology = [read(safe(root, baseline))['verified_at_utc'], tests['checked_at_utc'], reg['registered_at_utc'],
        proof['checked_at_utc'], meta['started_at_utc'], meta['completed_at_utc'], verified['verified_at_utc']]
    require(archive.get('prefreeze_publication_analysis_verification_utc_order') == chronology
            and list(map(utc, chronology)) == sorted(map(utc, chronology)),
            'Original baseline/freeze/publication/analysis/independent chronology differs')
    return {'source_committed_prefreeze_and_original_fixture_receipts_checked': len(paths),
        'original_261_unique_tests_and_two_failed_fixture_artifacts_checked': True,
        'Windows_manifest_member_names_uniquely_normalized_without_changing_original_bytes': True}


def archived_alignment_finalization_gate(root, archive, verified):
    """Bind original QA, separate archive diagnostics and the timestamp correction."""
    qa_name = 'results/probability_alignment_presentation_validation.json'
    qa = read(safe(root, qa_name))
    require(qa.get('status') == 'passed' and qa.get('study_id') == 'probability_alignment_v1'
            and archive.get('presentation_validation_sha256') == sha(safe(root, qa_name)),
            'Original unchanged parent presentation verdict required')
    for group in ('document_sha256', 'additional_file_sha256'):
        values = qa.get(group)
        require(type(values) is dict and bool(values)
                and all(sha(safe(root, name)) == digest for name, digest in values.items() if name not in EVOLVING_INDEXES),
                'Original non-index parent presentation evidence changed')
    note_name = 'results/probability_alignment_archive_finalization_attempt01.json'
    note = read(safe(root, note_name))
    expected = {'scripts/check_probability_alignment_archive.py', qa_name, 'tool_reported_failure.txt'}
    require(archive.get('archive_consumer_timestamp_finalization') == note
            and note.get('status') == 'preserved_archive_glue_failure'
            and note.get('archive') == 'results/probability_alignment_archive_finalization_attempt01.zip'
            and note.get('archive_sha256') == sha(safe(root, note['archive']))
            and note.get('failure_diagnostic_is_tool_text_transcription_not_raw_stderr') is True
            and note.get('scientific_registered_source_config_inputs_thresholds_results_or_verification_changed') is False
            and note.get('existing_presentation_receipt_not_rewritten') is True
            and note.get('actual_learning_or_direction_optimization_repeated') is False
            and type(note.get('member_sha256')) is dict and set(note['member_sha256']) == expected,
            'Original timestamp-consumer failure must remain distinct from registered science')
    with ZipFile(safe(root, note['archive'])) as archive_zip:
        require(len(archive_zip.namelist()) == 3 and set(archive_zip.namelist()) == expected
                and all(hashlib.sha256(archive_zip.read(name)).hexdigest() == digest for name, digest in note['member_sha256'].items())
                and archive_zip.read(qa_name) == safe(root, qa_name).read_bytes(),
                'Original three-member timestamp-attempt ZIP and QA bytes changed')
    require(utc(verified['verified_at_utc']) <= utc(qa['exactchecked_at_utc']) <= utc(note['checked_at_utc'])
            <= utc(archive['verified_at_utc']) <= datetime.now(timezone.utc),
            'Actual exactchecked_at_utc QA key and original correction chronology required')
    smoke = archive.get('separate_archive_tool_development_preservation', {})
    require(smoke.get('registered_scientific_failure') is False
            and smoke.get('six_final_synthetic_receipt_checks_passed') is True
            and smoke.get('original_draft_and_explicit_reproduced_failure_preserved') is True
            and smoke.get('no_original_stderr_file_falsely_claimed_for_transcribed_diagnostics') is True
            and sha(safe(root, smoke['archive'])) == smoke.get('archive_sha256')
            and sha(safe(root, smoke['manifest'])) == smoke.get('manifest_sha256'),
            'Separate original archive-tool smoke diagnostics must remain preserved')
    return {'original_QA_and_timestamp_failure_ZIP_bytes_checked': True,
        'actual_exactchecked_at_utc_key_and_unchanged_QA_chronology_checked': True,
        'separate_archive_tool_smoke_and_registered_scientific_failures_distinguished': True}


def parent_gate(root):
    from .probability_envelope import parent_gate as inherited_gate, publication as prior_publication, validate_counts as prior_counts
    inherited = inherited_gate(root)
    require(git(root, 'rev-parse', BASE_TAG).decode().strip() == BASE_COMMIT,
            'Pinned completed probability-alignment release required')
    ancestor(root, BASE_COMMIT)
    pinned = inventory(root, sorted(REUSED_SOURCE | PARENT_FILES | GRAPH_FILES | INPUT_FILES))
    committed(root, BASE_COMMIT, pinned)
    prefix = 'results/probability_alignment_v1/'
    reg_path = safe(root, 'results/registration_probability_alignment_v1.json')
    reg = read(reg_path); meta = read(safe(root, prefix + 'metadata.json'))
    summary = read(safe(root, prefix + 'summary.json')); verified_path = safe(root, prefix + 'verification.json')
    verified = read(verified_path)
    require(reg.get('study_id') == meta.get('study_id') == summary.get('study_id') == verified.get('study_id') == 'probability_alignment_v1'
            and meta.get('status') == meta.get('stage') == summary.get('status') == 'completed'
            and meta.get('analysis_only') is True and verified.get('status') == 'passed'
            and verified.get('scientific_producer_imported') is False and verified.get('new_helper_imported') is False,
            'Completed independently checked saved-state probability-alignment parent required')
    require(safe(root, prefix + 'registration.json').read_bytes() == reg_path.read_bytes()
            and meta.get('registration_sha256') == verified.get('registration_sha256') == sha(reg_path)
            and meta.get('source_git_commit') == reg.get('source_git_commit')
            and verified.get('summary_sha256') == sha(safe(root, prefix + 'summary.json'))
            and meta.get('counts') == summary.get('counts'), 'Original parent identities differ')
    proof_name, proof = prior_publication(root, reg_path, reg, safe(root, 'results/probability_alignment_publication_proof.json'))
    require(meta.get('publication_proof') == proof_name and meta.get('publication_proof_sha256') == sha(safe(root, proof_name)),
            'Parent private publication proof binding required')
    times = list(map(utc, [reg['registered_at_utc'], proof['checked_at_utc'], meta['started_at_utc'],
        meta['completed_at_utc'], verified['verified_at_utc']]))
    require(times == sorted(times) and times[0] < times[1] < times[2] and times[-1] <= datetime.now(timezone.utc),
            'Original private publication-before-analysis chronology required')
    artifacts = read(safe(root, prefix + 'artifact_hashes.json'))
    require(type(artifacts) is dict and set(artifacts) == set(PARENT_PRIMARY)
            and all(sha(safe(root, prefix + name)) == digest for name, digest in artifacts.items()),
            'Complete parent raw primary artifact hash receipt required')
    prior_counts(reg['config'], summary, meta['measurement_records'])
    counts = summary['counts']; checks = verified.get('counts', {})
    require(all(type(checks.get(key)) is int and checks[key] == value for key, value in {
        'numeric_arrays_checked': 27, 'saved_reference_records_checked': 24000,
        'saved_reference_destination_effects_checked': 96000, 'measurement_records_checked': 900,
        'eligible_actual_Go_records_checked': counts['eligible_actual_Go_records'],
        'candidate_destination_effects_checked': counts['candidate_destination_effects'],
        'independent_ambient_support_reoptimizations': counts['new_optimization_calls'],
        'independent_legacy_support_reoptimizations': counts['unique_uniform_optimizations'],
        'independent_active_faces_per_support': 16, 'new_training_trials': 0,
        'new_random_draws': 0, 'fresh_model_constructions': 0}.items())
        and all(checks.get(key) is True for key in ('head_prefixes_and_factual_scalar_effects_checked',
            'own_prefeedback_observed_signs_and_full_gate_checked', 'all_five_same_state_signed_updates_checked',
            'uniform_and_probability_global_support_independently_reoptimized',
            'legacy_balanced_global_support_independently_reoptimized', 'signed_finite_probability_failures_and_null_means_retained'))
        and checks.get('new_cone_solver_imported') is False,
        'Original complete state/effect replay and explicitly positive support optimization receipt required')
    interpretation = summary.get('interpretation', {})
    require(all(interpretation.get(key) is False for key in ('parent_H_gain', 'parent_H_acquisition',
        'new_confirmatory_hypothesis', 'finite_probability_global_optimum_claimed',
        'summed_fixed_state_gains_predict_new_trajectory', 'actual_future_policy_relearned'))
        and interpretation.get('parent_H_protection') is True
        and summary.get('validation', {}).get('all_numeric_information_and_support_checks_passed') is True,
        'Parent saved one-step effects must not become actual learned performance')
    for group, size in (('source_files', 24), ('parent_files', 17)):
        values = reg.get(group)
        require(type(values) is dict and len(values) == size
                and all(sha(safe(root, name)) == digest for name, digest in values.items()),
                'Complete unchanged original parent closure required: ' + group)
        committed(root, reg['source_git_commit'], values)
    ancestor(root, reg['source_git_commit'], BASE_COMMIT)
    archive = read(safe(root, 'results/probability_alignment_archive_validation.json'))
    require(archive.get('status') == 'passed'
            and archive.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True
            and archive.get('scientific_verification_sha256') == sha(verified_path)
            and archive.get('registration_sha256') == sha(reg_path)
            and archive.get('source_git_commit') == reg['source_git_commit'] and archive.get('counts') == counts,
            'Original completed probability archive binding required')
    baseline_name = 'results/probability_alignment_baseline_preservation.json'
    baseline = read(safe(root, baseline_name))
    require(baseline.get('status') == 'passed' and archive.get('baseline_hash_receipt_sha256') == sha(safe(root, baseline_name))
            and archive.get('allowed_evolving_indexes') == baseline.get('allowed_evolving_indexes') == sorted(EVOLVING_INDEXES),
            'Exact three-index prior raw Git-blob baseline required')
    prefreeze = archived_alignment_prefreeze_gate(root, archive, reg, meta, proof, verified)
    finalization = archived_alignment_finalization_gate(root, archive, verified)
    snapshot = archive.get('current_file_sha256')
    require(type(snapshot) is dict and set(reg['source_files']) | set(reg['parent_files'])
            | set(archive['prefreeze_receipts_committed_source_sha256']) | PARENT_FILES
            - {'results/probability_alignment_archive_validation.json'} <= set(snapshot),
            'Complete original parent archive snapshot required')
    committed(root, BASE_COMMIT, snapshot)
    require(all(sha(safe(root, name)) == digest for name, digest in snapshot.items() if name not in EVOLVING_INDEXES),
            'Completed parent snapshot bytes changed')
    old_status = json.loads(git(root, 'show', BASE_COMMIT + ':docs/repository-status.json'))
    current = read(safe(root, 'docs/repository-status.json'))
    require(all(status.get(key) is False for status in (old_status, current) for key in (
        'level4_similarity_supported', 'shared_neural_mechanism_verified', 'human_shell_H_similarity',
        'human_shell_H_prediction', 'gated_learning_H_gain', 'gated_learning_H_acquisition'))
        and all(status.get('gated_learning_H_protection') is True
            and status.get('probability_alignment_completed') is True for status in (old_status, current)),
        'Prior unsupported learning/human outcomes and operational protection must remain preserved')
    return {'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, 'inherited_parent_gate': inherited,
        'pinned_baseline_files': len(pinned), 'historical_archive_snapshot_files': len(snapshot),
        'original_24000_saved_records_96000_effects_and_positive_optimizations_reused_not_recomputed': True,
        'parent_prefreeze_and_failed_fixture_gate': prefreeze, 'parent_QA_and_archive_finalization_gate': finalization,
        'source_data_publication_Git_archive_and_raw_failure_bytes_checked': True,
        'parent_gate_no_new_learning_random_graph_or_optimization_calls': True,
        'whole_old_repository_hash_pass_not_repeated': True}


def frozen_inventory(root, cfg):
    return {group: inventory(root, cfg[group]) for group in
            ('source_files', 'parent_files', 'graph_files', 'input_files')}


def compressed_csv_write(path, rows):
    """Preserve the original UTF-8/CRLF CSV bytes in a deterministic gzip stream."""
    require(type(rows) is list and bool(rows) and all(type(row) is dict for row in rows),
            'Nonempty measurement dictionaries required')
    fields = list(rows[0])
    require(bool(fields) and all(set(row) == set(fields) for row in rows), 'Uniform measurement schema required')
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=9) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf8', newline='') as text:
                writer = csv.DictWriter(text, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


def validate_counts(cfg, summary, row_count):
    seeds, trials = cfg.get('model_seeds'), cfg.get('training_trials')
    checkpoints = cfg.get('checkpoints')
    require(type(seeds) is list and len(seeds) == 10 and seeds == list(range(17801, 17811))
            and type(trials) is int and trials == 400
            and checkpoints == [0, 50, 100, 200, 400] and cfg.get('methods') == list(METHODS),
            'Fixed eight-policy actual-learning dimensions required')
    conditions, groups = 48, 6
    expected = {'training_feedback_trials': conditions * len(seeds) * trials,
                'static_probe_cases': conditions * len(seeds) * len(checkpoints) * 4,
                'model_initializations': conditions * len(seeds),
                'fixed_feature_constructions': groups * len(seeds),
                'model_seeds': len(seeds), 'head_update_calls': conditions * len(seeds) * trials,
                'internal_update_calls': 0, 'value_update_calls': 0,
                'internal_nonzero_steps': 0, 'value_nonzero_steps': 0,
                'preflight_direction_evaluations': 19440, 'runtime_selective_direction_evaluations': 24000,
                'lookup_static_cases': 9600, 'lookup_autonomous_training_trials': 0,
                'numeric_arrays': NUMERIC_ARRAY_COUNT, 'measurement_columns': MEASUREMENT_COLUMN_COUNT}
    variable = {'head_nonzero_steps', 'auxiliary_memory_observation_calls', 'auxiliary_memory_discoveries',
        'balanced_direction_requests', 'always_balanced_direction_requests', 'gated_balanced_direction_requests',
        'gated_isolated_direction_evaluations', 'gated_balanced_mode_trials', 'gated_gate_activations',
        'self_blend_gate_activations', 'self_blend_direction_requests', 'self_blend_mode_trials',
        'self_blend_isolated_direction_evaluations', 'balanced_unique_solver_cases', 'balanced_NoGo_solver_skips',
        'uniform_direction_requests', 'probability_direction_requests', 'uniform_unique_solver_cases',
        'probability_unique_solver_cases', 'objective_unique_solver_cases', 'total_unique_solver_cases',
        'uniform_gate_activations', 'probability_gate_activations', 'uniform_mode_trials', 'probability_mode_trials',
        'uniform_isolated_direction_evaluations', 'probability_isolated_direction_evaluations',
        'objective_NoGo_solver_skips', 'unknown_other_destination_exposures',
        'source_inclusive_unknown_exposures_before_feedback', 'source_inclusive_unknown_exposures_after_feedback'}
    counts = summary.get('counts', {})
    require(type(counts) is dict and set(counts) == set(expected) | variable
            and all(type(counts.get(key)) is int and counts[key] == value for key, value in expected.items())
            and all(type(counts[key]) is int and counts[key] >= 0 for key in variable)
            and type(row_count) is int and row_count == expected['training_feedback_trials'],
            'Actual nonzero training and exact static/feature/model work counts required')
    require(type(counts.get('head_nonzero_steps')) is int
            and 0 < counts['head_nonzero_steps'] <= expected['head_update_calls'],
            'Actual nonzero output learning must be recorded')
    per_method = 24000
    require(counts['balanced_direction_requests'] == counts['always_balanced_direction_requests']
                + counts['gated_balanced_direction_requests'] + counts['self_blend_direction_requests']
            and counts['always_balanced_direction_requests'] <= per_method
            and counts['balanced_unique_solver_cases'] <= counts['balanced_direction_requests']
            and counts['balanced_NoGo_solver_skips'] == per_method + counts['gated_balanced_mode_trials']
                + counts['self_blend_mode_trials'] - counts['balanced_direction_requests'],
            'Exact legacy B/W requests, actual cached cases and NoGo skips required')
    for prefix, request in (('gated', 'gated_balanced_direction_requests'),
                            ('self_blend', 'self_blend_direction_requests'),
                            ('uniform', 'uniform_direction_requests'), ('probability', 'probability_direction_requests')):
        mode = 'gated_balanced_mode_trials' if prefix == 'gated' else prefix + '_mode_trials'
        isolated = prefix + '_isolated_direction_evaluations'
        activation = 'gated_gate_activations' if prefix == 'gated' else prefix + '_gate_activations'
        require(counts[mode] + counts[isolated] == per_method and counts[request] <= counts[mode]
                and counts[activation] <= 60, 'Exact per-policy gate, isolated prefix and request accounting required')
    require(counts['uniform_unique_solver_cases'] <= min(240, counts['uniform_direction_requests'])
            and counts['probability_unique_solver_cases'] == counts['probability_direction_requests']
            and counts['objective_unique_solver_cases'] == counts['uniform_unique_solver_cases']
                + counts['probability_unique_solver_cases']
            and counts['total_unique_solver_cases'] == counts['balanced_unique_solver_cases']
                + counts['objective_unique_solver_cases']
            and counts['objective_NoGo_solver_skips'] == counts['uniform_mode_trials']
                + counts['probability_mode_trials'] - counts['uniform_direction_requests']
                - counts['probability_direction_requests'],
            'Actual U cached cases and P fresh full-Go optimizations must remain explicit and distinct')
    require(counts['auxiliary_memory_discoveries'] <= min(1920, counts['auxiliary_memory_observation_calls'])
            and counts['auxiliary_memory_observation_calls'] <= expected['head_update_calls']
            and counts['source_inclusive_unknown_exposures_before_feedback']
                - counts['source_inclusive_unknown_exposures_after_feedback'] == counts['auxiliary_memory_discoveries']
            and counts['source_inclusive_unknown_exposures_before_feedback'] <= 4 * expected['head_update_calls']
            and counts['unknown_other_destination_exposures'] <= min(3 * expected['head_update_calls'],
                counts['source_inclusive_unknown_exposures_after_feedback']),
            'Obtained-reward memory and source-inclusive versus other-destination exposures must differ correctly')
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
    reg = {'study_id': STUDY_ID, 'registration_type': 'prospective_observed_objective_learning',
           'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': commit,
           'config': cfg, 'config_sha256': sha(config), 'registration_path': REGISTRATION,
           'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, **frozen, 'gate': gate,
           'notice': 'Prior geometric and failed learning outcomes known; actual observed-feedback U/P objective learning with a separate new cohort, fixed unit caps and unknown-harm assessment. Frozen before new graph, seed, or learner computation. Computational seeds are not animals.'}
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
            'measurement_storage': cfg['measurement_storage'],
            'analysis_only': False, 'new_human_experiment': False,
            'shared_neural_mechanism_tested': False, 'education_effect_tested': False}
    try:
        arrays, rows = module.produce(cfg, root)
        require(type(arrays) is dict and len(arrays) == NUMERIC_ARRAY_COUNT
                and all(type(key) is str and key and np.asarray(value).dtype.kind in 'biufc'
                        and np.all(np.isfinite(value)) for key, value in arrays.items()),
                'Finite numeric non-object actual-learning arrays required')
        require(type(rows) is list and bool(rows)
                and all(type(row) is dict and len(row) == MEASUREMENT_COLUMN_COUNT for row in rows),
                'Exact ninety-six-column actual trial records required')
        np.savez_compressed(out / 'arrays.npz', **arrays)
        compressed_csv_write(out / 'measurements.csv.gz', rows)
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
            and meta.get('measurement_storage') == cfg['measurement_storage']
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
    with gzip.open(directory / 'measurements.csv.gz', 'rt', encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None
                and len(reader.fieldnames) == len(set(reader.fieldnames)) == MEASUREMENT_COLUMN_COUNT,
                'Exact ninety-six unique saved measurement columns required')
        rows = list(reader)
    with np.load(directory / 'arrays.npz', allow_pickle=False) as arrays:
        require(type(meta.get('numeric_arrays')) is int and meta['numeric_arrays'] == len(arrays.files) == NUMERIC_ARRAY_COUNT
                and all(arrays[key].dtype.kind in 'biufc' and np.all(np.isfinite(arrays[key]))
                        for key in arrays.files), 'Saved finite numeric array inventory differs')
    require(type(meta.get('measurement_records')) is int and meta['measurement_records'] == len(rows)
            and len(rows) > 0, 'Saved measurement row count differs')
    validate_counts(cfg, summary, len(rows))
    return cfg, reg, meta
