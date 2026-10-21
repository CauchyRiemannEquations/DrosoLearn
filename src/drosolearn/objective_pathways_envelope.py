"""Frozen byte gates for descriptive saved objective-learning pathways.

The parent actual cohort is immutable input. This module imports no learner,
graph builder or optimizer and performs no new scientific parent replay.
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
from .self_blend_factors_envelope import REUSED_SOURCE as BASE_ENVELOPES

STUDY_ID = 'objective_pathways_v1'
MODULE = 'drosolearn.objective_pathways'
PROTOCOL = 'docs/objective-pathways-protocol.md'
CONFIG = 'configs/objective_pathways_v1.json'
REGISTRATION = 'results/registration_objective_pathways_v1.json'
RESULT = 'results/objective_pathways_v1'
BASE_TAG = 'objective-learning-v1'
BASE_COMMIT = 'b96a6c9d88a9f83cbe4e093272eea0ca77f245fe'
REPOSITORY = 'https://github.com/CauchyRiemannEquations/DrosoLearn'
PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json')
PARENT_PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv.gz', 'summary.json')
REUSED_SOURCE = BASE_ENVELOPES | frozenset({
    'src/drosolearn/self_blend_factors_envelope.py',
    'src/drosolearn/probability_envelope.py', 'src/drosolearn/objective_envelope.py'})
SOURCE_FILES = REUSED_SOURCE | frozenset({
    CONFIG, PROTOCOL, 'src/drosolearn/objective_pathways.py',
    'src/drosolearn/objective_pathways_envelope.py', 'scripts/run_objective_pathways.py',
    'scripts/verify_objective_pathways.py', 'tests/test_objective_pathways.py',
    'tests/test_objective_pathways_envelope.py', 'tests/test_objective_pathways_verify.py'})
PARENT_FILES = frozenset({
    'results/registration_objective_learning_v1.json',
    'results/objective_learning_publication_proof.json',
    'results/objective_learning_archive_validation.json',
    'results/objective_learning_baseline_preservation.json',
    'results/objective_learning_presentation_validation.json',
    'results/objective_learning_prefreeze_validation.json',
    'results/objective_learning_prefreeze_tests/objective_learning_prefreeze_final.xml',
    'results/objective_learning_figures_v1/plotted_values.json',
    'results/objective_learning_figures_v1/provenance.json',
    *('results/objective_learning_v1/' + name for name in
      (*PARENT_PRIMARY, 'verification.json', 'artifact_hashes.json'))})
EVOLVING_INDEXES = frozenset({'README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'})
NUMERIC_ARRAYS = frozenset(('model_seeds', 'schedules', 'targets', 'uniforms',
    'probability_changes', 'margin_changes', 'source_logits_before', 'reference_post',
    'first_post_trials', 'pair_action_cases', 'bridge_eligible', 'bridge_q_cells',
    'bridge_q_shapley', 'bridge_positive_shapley', 'bridge_negative_shapley',
    'factual_pair_diagnostics'))
MEASUREMENT_COLUMN_COUNT = 15


def validate_envelope(root, cfg, config):
    require(type(cfg) is dict and cfg.get('study_id') == STUDY_ID
            and cfg.get('module') == MODULE and cfg.get('protocol') == PROTOCOL
            and cfg.get('scope') == 'retrospective_descriptive_objective_pathways',
            'Fixed read-only descriptive implementation required')
    require(relative(root, config) == CONFIG, 'Fixed configuration path required')
    for key, expected in (('source_files', SOURCE_FILES), ('parent_files', PARENT_FILES)):
        values = cfg.get(key)
        require(type(values) is list and len(values) == len(expected) and set(values) == expected,
                'Exact unique frozen inventory required: ' + key)
    require(cfg.get('external_manifest') is None, 'No external data in this saved-results analysis')
    interpretation = cfg.get('interpretation', {})
    require(all(interpretation.get(key) is False for key in (
        'new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim',
        'new_training', 'new_seed_draws', 'new_effect_hypothesis_test')),
        'Bounded read-only descriptive interpretation required')


def archived_learning_prefreeze_gate(root, archive, reg, meta, proof, verified):
    validation = 'results/objective_learning_prefreeze_validation.json'
    baseline = 'results/objective_learning_baseline_preservation.json'
    junit = 'results/objective_learning_prefreeze_tests/objective_learning_prefreeze_final.xml'
    expected = {validation, baseline, junit}
    values = archive.get('prefreeze_receipts_committed_source_sha256')
    require(type(values) is dict and set(values) == expected
            and all(sha(safe(root, name)) == digest for name, digest in values.items()),
            'Exact original completed-parent prefreeze receipts required')
    committed(root, reg['source_git_commit'], values)
    tests = read(safe(root, validation))
    classes = {'tests.test_cone_geometry': 45, 'tests.test_objective_learning': 59,
        'tests.test_objective_learning_envelope': 62, 'tests.test_objective_learning_verify': 134,
        'tests.test_probability_alignment': 38, 'tests.test_self_blend_learning': 47}
    require(tests.get('status') == 'passed' and tests.get('frozen_source_hashes') == reg['source_files']
            and tests.get('parent_byte_gate') == reg['gate'] and tests.get('unique_passed_tests') == 385
            and tests.get('new_unique_passed_tests') == 255
            and tests.get('unchanged_related_regression_passed_tests') == 130
            and tests.get('unique_passed_tests_by_class') == classes
            and tests.get('synthetic_fixtures_only_before_freeze') is True
            and all(type(tests.get(key)) is int and tests[key] == 0 for key in (
                'actual_new_training_before_freeze', 'new_cohort_graph_feature_rng_calls_before_freeze',
                'actual_new_cohort_objective_optimizations_before_freeze')),
            'Original exact 385-test prefreeze with no premature cohort work required')
    receipts = tests.get('test_receipts')
    require(type(receipts) is list and len(receipts) == 1 and receipts[0].get('path') == junit
            and receipts[0].get('sha256') == values[junit] and receipts[0].get('passed_unique_tests') == 385
            and tests.get('development_failure_artifacts') == []
            and archive.get('development_failure_artifact_bytes_preserved') == []
            and archive.get('development_ZIP_manifests_and_original_member_hashes') == {},
            'Original single successful JUnit and zero new development failures must remain distinct')
    cases = ET.parse(safe(root, junit)).findall('.//testcase')
    require(len(cases) == len({(c.get('classname'), c.get('name')) for c in cases}) == 385
            and all(c.find(key) is None for c in cases for key in ('failure', 'error', 'skipped'))
            and dict(Counter(c.get('classname') for c in cases)) == classes
            and archive.get('prefreeze_passed_unique_tests_independently_counted') == 385
            and archive.get('prefreeze_unique_tests_by_class') == classes,
            'Exact original 385 unique cases and original class partition required')
    chronology = [read(safe(root, baseline))['verified_at_utc'], tests['checked_at_utc'], reg['registered_at_utc'],
        proof['checked_at_utc'], meta['started_at_utc'], meta['completed_at_utc'], verified['verified_at_utc']]
    times = list(map(utc, chronology))
    require(archive.get('prefreeze_publication_learning_verification_utc_order') == chronology
            and times == sorted(times) and times[2] < times[3] < times[4]
            and times[-1] <= utc(archive['verified_at_utc']) <= datetime.now(timezone.utc),
            'Original freeze/publication/actual-learning/independent UTC chronology required')
    return {'source_committed_prefreeze_receipts_checked': 3,
        'original_385_unique_tests_and_zero_new_development_failures_checked': True,
        'all_prior_ancestor_failures_preserved_by_inherited_byte_gate': True,
        'original_prefreeze_precedes_actual_learning': True}


def parent_gate(root):
    from .objective_envelope import parent_gate as inherited_gate, publication as prior_publication, validate_counts as prior_counts
    inherited = inherited_gate(root)
    require(git(root, 'rev-parse', BASE_TAG).decode().strip() == BASE_COMMIT,
            'Pinned completed objective-learning release required')
    ancestor(root, BASE_COMMIT)
    pinned = inventory(root, sorted(REUSED_SOURCE | PARENT_FILES))
    committed(root, BASE_COMMIT, pinned)
    prefix = 'results/objective_learning_v1/'
    reg_path = safe(root, 'results/registration_objective_learning_v1.json')
    reg = read(reg_path); meta = read(safe(root, prefix + 'metadata.json'))
    summary = read(safe(root, prefix + 'summary.json')); verified_path = safe(root, prefix + 'verification.json')
    verified = read(verified_path)
    require(reg.get('study_id') == meta.get('study_id') == summary.get('study_id') == verified.get('study_id') == 'objective_learning_v1'
            and meta.get('status') == meta.get('stage') == summary.get('status') == 'completed'
            and meta.get('analysis_only') is False and verified.get('status') == 'passed'
            and verified.get('scientific_producer_imported') is False and verified.get('new_helper_imported') is False,
            'Completed independently checked actual U/P learning parent required')
    require(safe(root, prefix + 'registration.json').read_bytes() == reg_path.read_bytes()
            and meta.get('registration_sha256') == verified.get('registration_sha256') == sha(reg_path)
            and meta.get('source_git_commit') == reg.get('source_git_commit')
            and verified.get('summary_sha256') == sha(safe(root, prefix + 'summary.json'))
            and meta.get('counts') == summary.get('counts'), 'Original parent evidence identities differ')
    proof_name, proof = prior_publication(root, reg_path, reg, safe(root, 'results/objective_learning_publication_proof.json'))
    require(meta.get('publication_proof') == proof_name and meta.get('publication_proof_sha256') == sha(safe(root, proof_name)),
            'Parent private publication proof binding required')
    times = list(map(utc, [reg['registered_at_utc'], proof['checked_at_utc'], meta['started_at_utc'],
        meta['completed_at_utc'], verified['verified_at_utc']]))
    require(times == sorted(times) and times[0] < times[1] < times[2] and times[-1] <= datetime.now(timezone.utc),
            'Original private publication-before-actual-learning chronology required')
    artifacts = read(safe(root, prefix + 'artifact_hashes.json'))
    require(type(artifacts) is dict and set(artifacts) == set(PARENT_PRIMARY)
            and all(sha(safe(root, prefix + name)) == digest for name, digest in artifacts.items()),
            'Exact parent five primary hashes including lossless gzip required')
    prior_counts(reg['config'], summary, meta['measurement_records'])
    counts = summary['counts']; checks = verified.get('counts', {})
    expected = {'arrays': 215, 'independently_replayed_training_actions': 192000,
        'model_seeds': 10, 'conditions': 48, 'checkpoint_static_probes': 9600,
        'head_update_calls': 192000, 'internal_update_calls': 0, 'value_update_calls': 0,
        'unique_balanced_states_independently_reoptimized': 972,
        'unique_objective_states_independently_reoptimized': 12017,
        'unique_uniform_states_independently_reoptimized': 240,
        'unique_probability_states_independently_reoptimized': 11777}
    boolean_checks = ('all_fresh_autonomous_choices_rewards_and_heads_replayed',
        'fixed_graph_encoding_weights_and_probe_activity_checked',
        'balanced_uses_only_actual_own_obtained_feedback_checked',
        'independent_ambient_lstsq_dual_support_optima_checked',
        'unknown_other_masks_before_and_after_match_checked',
        'evaluator_hidden_signs_do_not_enter_observed_direction_checked',
        'per_trial_correct_probability_and_aligned_margin_changes_checked',
        'all_candidate_gain_acquisition_protection_and_P_minus_U_state_hypotheses_recomputed',
        'each_P_case_uses_own_actual_preupdate_head_and_new_actual_Go_trial',
        'common_actual_movement_ceiling_not_equal_realized_budget_checked')
    require(all(type(checks.get(key)) is int and checks[key] == value for key, value in expected.items())
            and all(checks.get(key) is True for key in boolean_checks)
            and checks.get('extra_optimizer_and_memory_processing_cost_equalized') is False
            and checks.get('new_generalization_or_human_mechanism_claim') is False
            and meta.get('numeric_arrays') == 215 and meta.get('measurement_records') == 192000
            and meta.get('measurement_storage') == 'lossless_deterministic_gzip_csv_crlf',
            'Original complete 192000-choice replay and all U/P state certificates required')
    require(all(summary.get('H_gain', {}).get(method, {}).get('supported') is False
                and summary.get('H_acquisition', {}).get(method, {}).get('supported') is False
                and summary.get('H_protection', {}).get(method, {}).get('supported') is True
                for method in ('gated_uniform', 'gated_probability'))
            and summary.get('H_state', {}).get('supported') is False
            and summary.get('validation', {}).get('all_numeric_and_information_gates_passed') is True
            and all(value is True for value in summary.get('gate_validation', {}).values())
            and all(value is True for value in summary.get('objective_validation', {}).values()
                if type(value) is bool),
            'All seven original gain/acquisition/protection/state verdicts required')
    for group, size in (('source_files', 49), ('parent_files', 14), ('graph_files', 4), ('input_files', 2)):
        values = reg.get(group)
        require(type(values) is dict and len(values) == size
                and all(sha(safe(root, name)) == digest for name, digest in values.items()),
                'Exact unchanged original source/raw closure required: ' + group)
        committed(root, reg['source_git_commit'], values)
    ancestor(root, reg['source_git_commit'], BASE_COMMIT)
    archive = read(safe(root, 'results/objective_learning_archive_validation.json'))
    require(archive.get('status') == 'passed'
            and archive.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True
            and archive.get('scientific_verification_sha256') == sha(verified_path)
            and archive.get('registration_sha256') == sha(reg_path)
            and archive.get('source_git_commit') == reg['source_git_commit'] and archive.get('counts') == counts,
            'Completed objective-learning archive binding required')
    baseline_path = safe(root, 'results/objective_learning_baseline_preservation.json'); baseline = read(baseline_path)
    require(archive.get('baseline_hash_receipt_sha256') == sha(baseline_path)
            and baseline.get('status') == 'passed'
            and archive.get('allowed_evolving_indexes') == baseline.get('allowed_evolving_indexes') == sorted(EVOLVING_INDEXES),
            'Exact original prior Git-blob receipt and only three evolving indexes required')
    prefreeze = archived_learning_prefreeze_gate(root, archive, reg, meta, proof, verified)
    snapshot = archive.get('current_file_sha256')
    required = set(reg['source_files']) | set(reg['graph_files']) | set(reg['input_files']) | PARENT_FILES | EVOLVING_INDEXES
    required.discard('results/objective_learning_archive_validation.json')
    require(type(snapshot) is dict and len(snapshot) == 95 and required <= set(snapshot),
            'Complete original ninety-five-file archive snapshot required')
    committed(root, BASE_COMMIT, snapshot)
    require(all(sha(safe(root, name)) == digest for name, digest in snapshot.items() if name not in EVOLVING_INDEXES),
            'Historical immutable objective-learning source/raw/presentation snapshot changed')
    qa_name = 'results/objective_learning_presentation_validation.json'; qa = read(safe(root, qa_name))
    require(qa.get('status') == 'passed' and qa.get('study_id') == 'objective_learning_v1'
            and archive.get('presentation_validation_sha256') == sha(safe(root, qa_name))
            and all(type(qa.get(group)) is dict and bool(qa[group])
                and all(sha(safe(root, name)) == digest for name, digest in qa[group].items() if name not in EVOLVING_INDEXES)
                for group in ('document_sha256', 'additional_file_sha256'))
            and all(value is True for value in qa.get('checks', {}).values())
            and times[-1] <= utc(qa['checked_at_utc']) <= utc(archive['verified_at_utc']),
            'Original QA docs/figures/full-gzip validation and actual checked_at_utc chronology required')
    provenance = read(safe(root, 'results/objective_learning_figures_v1/provenance.json'))
    require(provenance.get('verified_arrays_only') is True
            and provenance.get('new_training_trials') == provenance.get('new_optimization_calls') == 0
            and provenance.get('arrays_sha256') == artifacts['arrays.npz']
            and provenance.get('summary_sha256') == artifacts['summary.json']
            and provenance.get('verification_sha256') == sha(verified_path)
            and provenance.get('plotted_values_sha256') == sha(safe(root, 'results/objective_learning_figures_v1/plotted_values.json')),
            'Original independently checked actual-learning figure provenance required')
    old_status = json.loads(git(root, 'show', BASE_COMMIT + ':docs/repository-status.json'))
    current_status = read(safe(root, 'docs/repository-status.json'))
    require(all(status.get(key) is False for status in (old_status, current_status) for key in (
        'level4_similarity_supported', 'shared_neural_mechanism_verified',
        'human_shell_H_similarity', 'human_shell_H_prediction', 'objective_learning_H_state'))
        and all(status.get('objective_learning_H_gain', {}).get(method) is False
            and status.get('objective_learning_H_acquisition', {}).get(method) is False
            and status.get('objective_learning_H_protection', {}).get(method) is True
            for status in (old_status, current_status) for method in ('gated_uniform', 'gated_probability')),
            'Original human negatives and U/P unsupported gains/acquisition/state with protection required')
    return {'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, 'inherited_parent_gate': inherited,
        'pinned_baseline_files': len(pinned), 'historical_archive_snapshot_files': len(snapshot),
        'original_192000_actual_choices_and_215_arrays_replay_reused_not_repeated': True,
        'original_learning_prefreeze_gate': prefreeze,
        'source49_parent14_graph4_input2_and_gzip_primary_hashes_checked': True,
        'original_QA_385_tests_and_seven_verdicts_preserved': True,
        'parent_gate_no_new_learning_random_graph_or_optimization_calls': True,
        'whole_old_repository_hash_pass_not_repeated': True}

def frozen_inventory(root, cfg):
    return {group: inventory(root, cfg[group]) for group in ('source_files', 'parent_files')}


def validate_counts(cfg, summary, row_count):
    require(cfg.get('model_seeds') == list(range(17801, 17811))
            and type(cfg.get('training_trials')) is int and cfg['training_trials'] == 400,
            'Fixed saved actual-learning cohort and budget required')
    expected = {'analyzed_parent_trial_records': 192000, 'analyzed_destination_effects': 768000,
                'reference_gate_paths': 60, 'comparison_paths': 180,
                'paired_trial_records': 72000, 'measurement_records': 2160, 'numeric_arrays': 16,
                'actual_path_strata': 34560,
                'new_training_trials': 0, 'new_head_update_calls': 0,
                'new_internal_update_calls': 0, 'new_value_update_calls': 0,
                'new_random_draws': 0, 'fresh_model_constructions': 0, 'new_optimization_calls': 0}
    counts = summary.get('counts', {})
    variable = {'paired_post_trial_records', 'eligible_both_Go_trials', 'corner_destination_effects'}
    require(type(counts) is dict and set(counts) == set(expected) | variable,
            'Exact saved-factor work count keys required')
    require(all(type(counts[key]) is int and counts[key] == value for key, value in expected.items())
            and all(type(counts[key]) is int for key in variable)
            and 0 <= counts['eligible_both_Go_trials'] <= counts['paired_post_trial_records'] <= 72000
            and counts['corner_destination_effects'] == 32 * counts['eligible_both_Go_trials']
            and type(row_count) is int and row_count == expected['measurement_records'],
            'Exact saved-factor counts and zero new training/random/model/optimization work required')
    require(summary.get('study_id') == STUDY_ID and summary.get('status') == 'completed'
            and summary.get('scope') == 'retrospective_descriptive_objective_pathways',
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
    reg = {'study_id': STUDY_ID, 'registration_type': 'retrospective_descriptive_objective_pathways',
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
        require(type(arrays) is dict and set(arrays) == NUMERIC_ARRAYS
                and all(type(key) is str and key and np.asarray(value).dtype.kind in 'biufc'
                        and np.all(np.isfinite(value)) for key, value in arrays.items()),
                'Finite numeric non-object analysis arrays required')
        require(type(rows) is list and bool(rows)
                and all(type(row) is dict and len(row) == MEASUREMENT_COLUMN_COUNT for row in rows),
                'Exact fifteen-column descriptive rows required')
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
        require(reader.fieldnames is not None
                and len(reader.fieldnames) == len(set(reader.fieldnames)) == MEASUREMENT_COLUMN_COUNT,
                'Exact fifteen unique saved measurement columns required')
        rows = list(reader)
    with np.load(directory / 'arrays.npz', allow_pickle=False) as arrays:
        require(type(meta.get('numeric_arrays')) is int and meta['numeric_arrays'] == len(arrays.files) == 16
                and set(arrays.files) == NUMERIC_ARRAYS
                and all(arrays[key].dtype.kind in 'biufc' and np.all(np.isfinite(arrays[key]))
                        for key in arrays.files), 'Saved finite analysis array inventory differs')
    require(type(meta.get('measurement_records')) is int and meta['measurement_records'] == len(rows)
            and len(rows) > 0, 'Saved measurement row count differs')
    validate_counts(cfg, summary, len(rows))
    return cfg, reg, meta
