"""Byte and chronology gates for a read-only, descriptive probability-alignment audit.

Saved parent arrays are analysis inputs. No graph constructor, learner, or scientific
parent producer is imported here. The registered analysis performs new U/P direction
optimizations and counts those separately from its zero new learning trials.
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

STUDY_ID = 'probability_alignment_v1'
MODULE = 'drosolearn.probability_alignment'
PROTOCOL = 'docs/probability-alignment-protocol.md'
CONFIG = 'configs/probability_alignment_v1.json'
REGISTRATION = 'results/registration_probability_alignment_v1.json'
RESULT = 'results/probability_alignment_v1'
BASE_TAG = 'self-blend-factors-v1'
BASE_COMMIT = '2715f1eada36207ce77709f08874688b1425d676'
REPOSITORY = 'https://github.com/CauchyRiemannEquations/DrosoLearn'
PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json')
REUSED_SOURCE = frozenset({
    'pyproject.toml', 'requirements-lock.txt', 'src/drosolearn/__init__.py',
    'src/drosolearn/representation_envelope.py', 'src/drosolearn/balanced_envelope.py',
    'src/drosolearn/selective_envelope.py', 'src/drosolearn/normalization_envelope.py',
    'src/drosolearn/cone_envelope.py', 'src/drosolearn/pathways_envelope.py',
    'src/drosolearn/gated_envelope.py', 'src/drosolearn/gated_pathways_envelope.py',
    'src/drosolearn/gated_factors_envelope.py', 'src/drosolearn/self_blend_envelope.py',
    'src/drosolearn/self_blend_factors_envelope.py', 'src/drosolearn/cone_geometry.py',
})
SOURCE_FILES = REUSED_SOURCE | frozenset({
    CONFIG, PROTOCOL, 'src/drosolearn/probability_alignment.py',
    'src/drosolearn/probability_envelope.py', 'scripts/run_probability_alignment.py',
    'scripts/verify_probability_alignment.py', 'tests/test_probability_alignment.py',
    'tests/test_probability_alignment_envelope.py', 'tests/test_probability_alignment_verify.py',
})
ORIGINAL_FILES = frozenset('results/self_blend_learning_v1/' + name for name in
    ('arrays.npz', 'metadata.json', 'registration.json', 'summary.json', 'verification.json', 'artifact_hashes.json'))
PARENT_FILES = ORIGINAL_FILES | frozenset({
    'results/registration_self_blend_factors_v1.json',
    'results/self_blend_factors_publication_proof.json',
    'results/self_blend_factors_archive_validation.json',
    'results/self_blend_factors_baseline_preservation.json',
    *('results/self_blend_factors_v1/' + name for name in
      (*PRIMARY, 'verification.json', 'artifact_hashes.json')),
})
EVOLVING_INDEXES = frozenset({'README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'})


def validate_envelope(root, cfg, config):
    require(type(cfg) is dict and cfg.get('study_id') == STUDY_ID
            and cfg.get('module') == MODULE and cfg.get('protocol') == PROTOCOL
            and cfg.get('scope') == 'retrospective_descriptive_probability_alignment',
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


def archived_probability_parent_gate(root, archive, reg, meta, proof, verification):
    validation = 'results/self_blend_factors_prefreeze_validation.json'
    baseline = 'results/self_blend_factors_baseline_preservation.json'
    junit = 'results/self_blend_factors_prefreeze_tests/self_blend_factors_prefreeze_final.xml'
    note_path = 'results/self_blend_factors_prefreeze_finalization.json'
    zip_path = 'results/self_blend_factors_prefreeze_style_attempt01.zip'
    expected = {validation, baseline, junit, note_path, zip_path}
    values = archive.get('prefreeze_receipts_committed_source_sha256')
    require(type(values) is dict and set(values) == expected
            and all(sha(safe(root, name)) == digest for name, digest in values.items()),
            'Exact completed-parent prefreeze/style ZIP source-committed bytes required')
    committed(root, reg['source_git_commit'], values)
    tests = read(safe(root, validation)); note = read(safe(root, note_path))
    require(tests.get('status') == 'passed' and tests.get('frozen_source_hashes') == reg['source_files']
            and tests.get('parent_byte_gate') == reg['gate'] and tests.get('unique_passed_tests') == 295
            and tests.get('new_unique_passed_tests') == 203 and tests.get('unchanged_related_regression_passed_tests') == 92
            and tests.get('actual_new_saved_factor_aggregation_before_freeze') == 0
            and tests.get('new_training_random_graph_or_optimization_calls') == 0
            and tests.get('synthetic_fixtures_only_before_freeze') is True,
            'Original complete 295-test no-premature-arithmetic snapshot required')
    receipts = tests.get('test_receipts'); development = tests.get('development_failure_artifacts')
    require(type(receipts) is list and len(receipts) == 1 and receipts[0].get('path') == junit
            and receipts[0].get('sha256') == values[junit] and receipts[0].get('passed_unique_tests') == 295
            and type(development) is list and len(development) == 2
            and {item.get('path') for item in development} == {note_path, zip_path}
            and all(item.get('sha256') == values[item['path']] for item in development)
            and archive.get('development_failure_artifact_bytes_preserved') == development,
            'Original successful JUnit and exact two style-finalization receipts required')
    cases = ET.parse(safe(root, junit)).findall('.//testcase')
    classes = {'tests.test_gated_factors': 45, 'tests.test_self_blend_factors': 56,
        'tests.test_self_blend_factors_envelope': 50, 'tests.test_self_blend_factors_verify': 97,
        'tests.test_self_blend_learning': 47}
    require(len(cases) == len({(c.get('classname'), c.get('name')) for c in cases}) == 295
            and all(c.find(key) is None for c in cases for key in ('failure', 'error', 'skipped'))
            and dict(Counter(c.get('classname') for c in cases)) == classes
            and archive.get('prefreeze_passed_unique_tests_independently_counted') == 295
            and archive.get('prefreeze_unique_tests_by_class') == classes,
            'Original 295 unique passing tests and exact class partition required')
    style = archive.get('original_prefreeze_style_attempt_and_EOF_only_finalization', {})
    test_path = 'tests/test_self_blend_factors_envelope.py'
    require(note.get('status') == 'passed' and note.get('archive') == zip_path
            and note.get('archive_sha256') == values[zip_path]
            and note.get('test_path') == test_path
            and note.get('final_test_sha256') == reg['source_files'][test_path]
            and note.get('removed_only_trailing_blank_line') is True
            and note.get('scientific_design_thresholds_or_parent_inputs_changed') is False
            and note.get('actual_saved_factor_aggregation_started') is False
            and style.get('original_style_attempt_zip_sha256') == values[zip_path]
            and style.get('finalization_note_sha256') == values[note_path]
            and style.get('original_member_sha256') == note.get('member_sha256')
            and style.get('only_trailing_empty_LF_lines_removed_from_one_unregistered_test') is True
            and style.get('all_other21_source_hashes_and_original_295_passing_test_identities_preserved') is True
            and style.get('non_LF_original_bytes_identical') is True
            and style.get('removed_trailing_LF_bytes') == style.get('removed_trailing_LF_bytes_independently_counted') == 2
            and style.get('final_trailing_LF_bytes') == 1
            and style.get('original_notice_count_corrected_by_rawbytes') is True
            and archive.get('original_notice_count_corrected_by_rawbytes') is True
            and archive.get('removed_trailing_LF_bytes') == 2,
            'Prior raw-ZIP EOF-only correction must retain the independently counted two LF bytes')
    order = style.get('original_tests_finalization_refreshed_tests_registration_utc_order')
    require(type(order) is list and len(order) == 4 and order[1:] == [note['checked_at_utc'], tests['checked_at_utc'], reg['registered_at_utc']]
            and list(map(utc, order)) == sorted(map(utc, order)) and utc(order[-2]) < utc(order[-1]),
            'Original style correction and refreshed passing suite must precede registration')
    chronology = [read(safe(root, baseline))['verified_at_utc'], tests['checked_at_utc'], reg['registered_at_utc'],
        proof['checked_at_utc'], meta['started_at_utc'], meta['completed_at_utc'], verification['verified_at_utc']]
    times = list(map(utc, chronology))
    require(archive.get('prefreeze_publication_analysis_verification_utc_order') == chronology
            and times == sorted(times) and times[2] < times[3] < times[4]
            and times[-1] <= utc(archive['verified_at_utc']) <= datetime.now(timezone.utc),
            'Original source/registration/publication/analysis/verification UTC chronology required')
    return {'source_committed_prefreeze_and_style_receipts_checked': 5,
        'original_295_passing_tests_and_style_ZIP_preserved': True,
        'independent_two_LF_only_finalization_and_all_other_sources_reused_by_frozen_hashes': True,
        'no_parent_scientific_replay_or_solver_calls': True}


def parent_gate(root):
    from .self_blend_factors_envelope import parent_gate as inherited_gate, publication as prior_publication
    inherited = inherited_gate(root)
    require(git(root, 'rev-parse', BASE_TAG).decode().strip() == BASE_COMMIT,
            'Pinned completed self-blend-factors release required')
    ancestor(root, BASE_COMMIT)
    pinned = inventory(root, sorted(REUSED_SOURCE | PARENT_FILES))
    committed(root, BASE_COMMIT, pinned)
    prefix = 'results/self_blend_factors_v1/'
    reg_path = safe(root, 'results/registration_self_blend_factors_v1.json')
    reg, meta = read(reg_path), read(safe(root, prefix + 'metadata.json'))
    summary = read(safe(root, prefix + 'summary.json'))
    verification_path = safe(root, prefix + 'verification.json')
    verification = read(verification_path)
    require(reg.get('study_id') == meta.get('study_id') == summary.get('study_id')
            == verification.get('study_id') == 'self_blend_factors_v1'
            and meta.get('status') == meta.get('stage') == summary.get('status') == 'completed'
            and meta.get('analysis_only') is True and verification.get('status') == 'passed'
            and verification.get('scientific_producer_imported') is False
            and verification.get('new_helper_imported') is False,
            'Completed independently checked descriptive self-blend-factors parent required')
    require(safe(root, prefix + 'registration.json').read_bytes() == reg_path.read_bytes()
            and meta.get('registration_sha256') == verification.get('registration_sha256') == sha(reg_path)
            and meta.get('source_git_commit') == reg.get('source_git_commit')
            and verification.get('summary_sha256') == sha(safe(root, prefix + 'summary.json'))
            and meta.get('counts') == summary.get('counts'), 'Parent evidence identity mismatch')
    proof_name, proof = prior_publication(root, reg_path, reg,
        safe(root, 'results/self_blend_factors_publication_proof.json'))
    require(meta.get('publication_proof') == proof_name
            and meta.get('publication_proof_sha256') == sha(safe(root, proof_name)),
            'Parent publication-proof binding required')
    times = [utc(reg['registered_at_utc']), utc(proof['checked_at_utc']),
             utc(meta['started_at_utc']), utc(meta['completed_at_utc']),
             utc(verification['verified_at_utc'])]
    require(times[0] < times[1] < times[2] <= times[3] <= times[4] <= datetime.now(timezone.utc),
            'Parent publication-before-analysis and independent UTC chronology required')
    artifacts = read(safe(root, prefix + 'artifact_hashes.json'))
    require(type(artifacts) is dict and set(artifacts) == set(PRIMARY),
            'Complete parent primary artifact receipt required')
    require(all(type(digest) is str and re.fullmatch(r'[0-9a-f]{64}', digest)
                and sha(safe(root, prefix + name)) == digest for name, digest in artifacts.items()),
            'Parent raw or primary artifact bytes changed')
    checks = verification.get('counts', {})
    replay_true = ('factual_head_prefixes_and_parent_effects_checked',
        'original_primary_post_display_bookkeeping_checked', 'same_actual_paired_Go_unit_mixture_equations_checked',
        'only_post_both_actual_Go_eight_corners_checked', 'state_scalar_direction_symmetric_allocations_checked',
        'signed_positive_and_negative_mass_allocations_checked', 'own_other_and_all_unpaired_actual_actions_accounted',
        'factual_diagnostics_and_nullable_seed_means_checked', 'partial_control_information_not_silently_equalized',
        'no_rng_model_graph_solver_or_future_policy_calls')
    require(all(checks.get(key) is True for key in replay_true)
            and checks.get('arrays') == 14 and checks.get('parent_trial_records_checked') == 144000
            and checks.get('destination_effects_checked') == 576000
            and checks.get('measurement_records_checked') == 3600
            and all(type(checks.get(key)) is int and checks[key] == 0 for key in (
                'new_training_trials', 'new_random_draws', 'new_optimization_calls')),
            'Prior complete saved-factor independent arithmetic required')
    counts = summary.get('counts', {})
    require(all(type(counts.get(key)) is int and counts[key] == 0 for key in (
        'new_training_trials', 'new_head_update_calls', 'new_internal_update_calls',
        'new_value_update_calls', 'new_random_draws', 'fresh_model_constructions', 'new_optimization_calls'))
        and counts.get('analyzed_parent_trial_records') == 144000
        and counts.get('analyzed_destination_effects') == 576000
        and counts.get('numeric_arrays') == 14 and counts.get('measurement_records') == 3600,
        'Prior saved accounting must not be relabeled as additional learning')
    interpretation = summary.get('interpretation', {})
    require(interpretation.get('parent_H_gain') is False
            and interpretation.get('parent_H_acquisition') is False
            and interpretation.get('parent_H_protection') is True
            and interpretation.get('new_confirmatory_hypothesis') is False
            and interpretation.get('accounting_allocation_not_unique_causal_explanation') is True
            and interpretation.get('mixed_one_step_cells_not_relearned_policy') is True
            and summary.get('validation', {}).get('all_exact_accounting_gates_passed') is True,
            'Prior unsupported gain/acquisition and operational protection interpretation required')
    for group in ('source_files', 'parent_files'):
        values = reg.get(group)
        require(type(values) is dict and bool(values), 'Complete parent frozen closure required')
        require(all(sha(safe(root, name)) == digest for name, digest in values.items()),
                'Parent frozen bytes changed: ' + group)
        committed(root, reg['source_git_commit'], values)
    require(len(reg['source_files']) == 22 and len(reg['parent_files']) == 13,
            'Exact parent source/parent inventory required')
    ancestor(root, reg['source_git_commit'], BASE_COMMIT)
    archive = read(safe(root, 'results/self_blend_factors_archive_validation.json'))
    require(archive.get('status') == 'passed'
            and archive.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True
            and archive.get('scientific_verification_sha256') == sha(verification_path)
            and archive.get('registration_sha256') == sha(reg_path)
            and archive.get('source_git_commit') == reg['source_git_commit']
            and archive.get('counts') == counts, 'Completed parent archive binding required')
    receipt = safe(root, 'results/self_blend_factors_baseline_preservation.json')
    require(archive.get('baseline_hash_receipt_sha256') == sha(receipt)
            and read(receipt).get('status') == 'passed'
            and archive.get('allowed_evolving_indexes') == read(receipt).get('allowed_evolving_indexes')
                == sorted(EVOLVING_INDEXES), 'Exact three-index prior Git-blob baseline receipt required')
    prefreeze_gate = archived_probability_parent_gate(root, archive, reg, meta, proof, verification)
    prepaths = set(archive['prefreeze_receipts_committed_source_sha256'])
    snapshot = archive.get('current_file_sha256')
    require(type(snapshot) is dict and bool(snapshot)
            and set(reg['source_files']) | prepaths
            | {prefix + name for name in (*PRIMARY, 'verification.json', 'artifact_hashes.json')}
            | EVOLVING_INDEXES | ORIGINAL_FILES <= set(snapshot), 'Complete parent archive snapshot required')
    committed(root, BASE_COMMIT, snapshot)
    require(all(sha(safe(root, name)) == digest for name, digest in snapshot.items()
                if name not in EVOLVING_INDEXES), 'Historical immutable snapshot changed')
    require(all(snapshot.get(prefix + name) == digest for name, digest in artifacts.items()),
            'Parent raw artifact and archive hashes differ')
    old_status = json.loads(git(root, 'show', BASE_COMMIT + ':docs/repository-status.json'))
    current_status = read(safe(root, 'docs/repository-status.json'))
    require(all(status.get(key) is False for status in (old_status, current_status) for key in (
        'level4_similarity_supported', 'shared_neural_mechanism_verified',
        'human_shell_H_similarity', 'human_shell_H_prediction', 'gated_learning_H_gain', 'gated_learning_H_acquisition'))
        and all(status.get('gated_learning_H_protection') is True for status in (old_status, current_status)),
        'Prior negative learning/human outcomes and operational protection must remain preserved')
    return {'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, 'inherited_parent_gate': inherited,
        'pinned_baseline_files': len(pinned), 'historical_archive_snapshot_files': len(snapshot),
        'prior_complete_saved_factors_and_original_144000_action_replay_reused_not_rerun': True,
        'original_parent_prefreeze_style_finalization_gate': prefreeze_gate,
        'parent_raw_source_data_Git_archive_and_prefreeze_bytes_checked': True,
        'prior_unsupported_gain_acquisition_and_operational_protection_preserved': True,
        'parent_gate_no_new_learner_graph_seed_or_optimizer_calls': True,
        'whole_repository_literal_pass_not_repeated': True}


def frozen_inventory(root, cfg):
    return {group: inventory(root, cfg[group]) for group in ('source_files', 'parent_files')}


def validate_counts(cfg, summary, row_count):
    require(cfg.get('model_seeds') == list(range(17701, 17711))
            and type(cfg.get('training_trials')) is int and cfg['training_trials'] == 400,
            'Fixed original actual-learning cohort required')
    fixed = {'saved_reference_records': 24000, 'saved_reference_destination_effects': 96000,
        'numeric_arrays': 27, 'measurement_records': 900, 'new_training_trials': 0,
        'new_head_update_calls': 0, 'new_internal_update_calls': 0, 'new_value_update_calls': 0,
        'new_random_draws': 0, 'fresh_model_constructions': 0}
    variable = {'eligible_actual_Go_records', 'candidate_destination_effects',
        'unique_uniform_optimizations', 'state_probability_optimizations',
        'new_optimization_calls', 'candidate_single_update_evaluations'}
    counts = summary.get('counts')
    require(type(counts) is dict and set(counts) == set(fixed) | variable
            and all(type(counts[key]) is int and counts[key] == value for key, value in fixed.items())
            and all(type(counts[key]) is int for key in variable)
            and type(row_count) is int and row_count == 900,
            'Exact objective-analysis counts, 900 rows and zero new learning/model/random work required')
    eligible = counts['eligible_actual_Go_records']; uniform = counts['unique_uniform_optimizations']
    require(0 < eligible <= 24000 and 0 < uniform <= min(240, eligible)
            and counts['state_probability_optimizations'] == eligible
            and counts['candidate_destination_effects'] == 20 * eligible
            and counts['candidate_single_update_evaluations'] == 5 * eligible
            and counts['new_optimization_calls'] == uniform + eligible,
            'Actual eligible-only U/P optimizations must be positive, bounded and explicitly counted')
    require(summary.get('study_id') == STUDY_ID and summary.get('status') == 'completed'
            and summary.get('scope') == 'retrospective_descriptive_probability_alignment',
            'Completed descriptive probability-alignment arithmetic required')


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
    reg = {'study_id': STUDY_ID, 'registration_type': 'retrospective_descriptive_probability_alignment',
           'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': commit,
           'config': cfg, 'config_sha256': sha(config), 'registration_path': REGISTRATION,
           'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, **frozen, 'gate': gate,
           'notice': 'Parent outcomes are already known; this saved-state probability alignment and new U/P direction optimizations are frozen before their arithmetic. Descriptive analysis only, without new training, seeds, animals or effect hypothesis tests.'}
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
            'new_graph_or_feature_constructions': 0, 'new_optimization_calls': None,
            'new_human_experiment': False, 'shared_neural_mechanism_tested': False,
            'education_effect_tested': False, 'new_effect_hypothesis_test': False}
    try:
        arrays, rows = module.produce(cfg, root)
        require(type(arrays) is dict and len(arrays) == 27
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
                    measurement_records=len(rows), numeric_arrays=len(arrays),
                    new_optimization_calls=summary['counts']['new_optimization_calls'])
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
                'new_training_calls', 'new_random_seed_draws', 'new_graph_or_feature_constructions'))
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
    require(type(meta.get('new_optimization_calls')) is int
            and meta['new_optimization_calls'] == summary['counts']['new_optimization_calls'] > 0,
            'New direction optimization work must be explicit and positive')
    with (directory / 'measurements.csv').open(encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
                'Unique saved measurement columns required')
        rows = list(reader)
    with np.load(directory / 'arrays.npz', allow_pickle=False) as arrays:
        require(type(meta.get('numeric_arrays')) is int and meta['numeric_arrays'] == len(arrays.files) == 27
                and all(arrays[key].dtype.kind in 'biufc' and np.all(np.isfinite(arrays[key]))
                        for key in arrays.files), 'Saved finite analysis array inventory differs')
    require(type(meta.get('measurement_records')) is int and meta['measurement_records'] == len(rows)
            and len(rows) > 0, 'Saved measurement row count differs')
    validate_counts(cfg, summary, len(rows))
    return cfg, reg, meta
