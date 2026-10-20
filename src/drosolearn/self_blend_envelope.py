"""Prospective integrity gates for self-emphasized post-observation output learning.

Parent saved analyses are reused through immutable source, raw-data and independent
receipts. New graph/cohort/learner work starts only after frozen publication.
"""
from __future__ import annotations

import csv
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
from .gated_envelope import REUSED_SOURCE as GATED_REUSED

STUDY_ID = 'self_blend_learning_v1'
MODULE = 'drosolearn.self_blend_learning'
PROTOCOL = 'docs/self-blend-learning-protocol.md'
CONFIG = 'configs/self_blend_learning_v1.json'
REGISTRATION = 'results/registration_self_blend_learning_v1.json'
RESULT = 'results/self_blend_learning_v1'
BASE_TAG = 'gated-factors-v1'
BASE_COMMIT = 'd94922361e8a17e156bed0d5c4f240750f75f5ef'
REPOSITORY = 'https://github.com/CauchyRiemannEquations/DrosoLearn'
PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json')
REUSED_SOURCE = GATED_REUSED | frozenset({
    'src/drosolearn/gated_learning.py', 'src/drosolearn/gated_envelope.py',
    'scripts/verify_gated_learning.py', 'src/drosolearn/gated_pathways_envelope.py',
    'src/drosolearn/gated_factors_envelope.py',
})
SOURCE_FILES = REUSED_SOURCE | frozenset({
    CONFIG, PROTOCOL, 'src/drosolearn/self_blend_learning.py',
    'src/drosolearn/self_blend_envelope.py', 'scripts/run_self_blend_learning.py',
    'scripts/verify_self_blend_learning.py', 'tests/test_self_blend_learning.py',
    'tests/test_self_blend_envelope.py', 'tests/test_self_blend_learning_verify.py',
})
PARENT_FILES = frozenset({
    'results/registration_gated_factors_v1.json',
    'results/gated_factors_publication_proof.json',
    'results/gated_factors_archive_validation.json',
    'results/gated_factors_baseline_preservation.json',
    *('results/gated_factors_v1/' + name for name in
      (*PRIMARY, 'verification.json', 'artifact_hashes.json')),
})
EVOLVING_INDEXES = frozenset({'README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'})
NUMERIC_ARRAY_COUNT = 201


def validate_envelope(root, cfg, config):
    require(type(cfg) is dict and cfg.get('study_id') == STUDY_ID
            and cfg.get('module') == MODULE and cfg.get('protocol') == PROTOCOL
            and cfg.get('scope') == 'prospective_observed_gated_self_blend_learning',
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


def archived_factor_prefreeze_gate(root, archive, reg, meta, proof, verification):
    """Bind the completed parent's tests, early failure bytes and style exception."""
    validation = 'results/gated_factors_prefreeze_validation.json'
    baseline = 'results/gated_factors_baseline_preservation.json'
    exception = 'results/gated_factors_prefreeze_evidence_exception.json'
    junit = 'results/gated_factors_prefreeze_tests/gated_factors_prefreeze_final.xml'
    failed_xml = 'results/gated_factors_development/gated_factors_verify_development_failure_01.xml'
    failures = {failed_xml, 'results/gated_factors_development/gated_factors_verify_failure_01.py',
        'results/gated_factors_development/gated_factors_verify_tests_failure_01.py'}
    paths = {validation, baseline, exception, junit} | failures
    values = archive.get('prefreeze_receipts_committed_source_sha256')
    require(type(values) is dict and set(values) == paths
            and all(sha(safe(root, name)) == digest for name, digest in values.items()),
            'Exact completed-parent prefreeze, early failure and exception bytes required')
    committed(root, reg['source_git_commit'], values)
    tests = read(safe(root, validation)); note = read(safe(root, exception))
    require(tests.get('status') == 'passed' and tests.get('frozen_source_hashes') == reg['source_files']
            and tests.get('parent_byte_gate') == reg['gate'] and tests.get('unique_passed_tests') == 269
            and tests.get('new_unique_passed_tests') == 158
            and tests.get('unchanged_related_regression_passed_tests') == 111
            and tests.get('actual_new_saved_factor_aggregation_before_freeze') == 0,
            'Original completed-parent 269-test frozen snapshot required')
    receipts = tests.get('test_receipts'); failed_receipts = tests.get('development_failure_artifacts')
    require(type(receipts) is list and len(receipts) == 1 and receipts[0].get('path') == junit
            and receipts[0].get('sha256') == values[junit] and receipts[0].get('passed_unique_tests') == 269
            and type(failed_receipts) is list and len(failed_receipts) == 3
            and {item.get('path') for item in failed_receipts} == failures
            and all(item.get('sha256') == values[item['path']] for item in failed_receipts),
            'Exact original successful JUnit and three early failure receipts required')
    cases = ET.parse(safe(root, junit)).findall('.//testcase')
    require(len(cases) == len({(c.get('classname'), c.get('name')) for c in cases}) == 269
            and all(c.find(key) is None for c in cases for key in ('failure', 'error', 'skipped')),
            'Original 269 unique passing JUnit cases required')
    classes = {'tests.test_gated_factors', 'tests.test_gated_factors_envelope', 'tests.test_gated_factors_verify'}
    require(sum(c.get('classname') in classes for c in cases) == 158
            and sum(c.get('classname') == 'tests.test_balanced_pathways' for c in cases) == 67
            and sum(c.get('classname') == 'tests.test_gated_pathways' for c in cases) == 44,
            'Exact original 158 new plus 111 unchanged regression cases required')
    failed = ET.parse(safe(root, failed_xml)).findall('.//testcase')
    require(len(failed) == 9 and sum(c.find('failure') is not None for c in failed) == 1
            and all(c.find(key) is None for c in failed for key in ('error', 'skipped'))
            and sum(c.find('failure') is None for c in failed) == 8,
            'Original eight-pass one-fail development evidence must remain intact')
    require(note.get('status') == 'passed' and note.get('original_failure_xml') == failed_xml
            and note.get('original_failure_xml_sha256') == values[failed_xml]
            and note.get('development_failure_test_count') == 9 and note.get('development_failure_failed_tests') == 1
            and note.get('scientific_sources_configuration_thresholds_or_parent_inputs_changed') is False
            and note.get('actual_factor_aggregation_started') is False
            and note.get('prefreeze_validation_sha256') == values[validation]
            and archive.get('original_development_failure_and_raw_whitespace_evidence_retained') is True,
            'Original raw-evidence whitespace exception must preserve science and precede analysis')
    chronology = [read(safe(root, baseline))['verified_at_utc'], tests['checked_at_utc'], note['checked_at_utc'],
        reg['registered_at_utc'], proof['checked_at_utc'], meta['started_at_utc'], meta['completed_at_utc'],
        verification['verified_at_utc']]
    require(archive.get('prefreeze_publication_analysis_verification_utc_order') == chronology,
            'Original completed-parent prefreeze chronology differs')
    times = list(map(utc, chronology))
    require(times == sorted(times) and times[3] < times[4] < times[5]
            and times[-1] <= utc(archive['verified_at_utc']) <= datetime.now(timezone.utc),
            'Original source/exception/registration/publication/analysis chronology required')
    return {'prefreeze_source_committed_receipts_checked': 7,
        'original_269_unique_tests_and_three_failure_artifacts_checked': True,
        'raw_failed_xml_whitespace_preserved_without_science_change': True,
        'parent_prefreeze_and_exception_precede_original_analysis': True}


def parent_gate(root):
    from .gated_factors_envelope import parent_gate as inherited_gate, publication as prior_publication
    inherited = inherited_gate(root)
    require(git(root, 'rev-parse', BASE_TAG).decode().strip() == BASE_COMMIT,
            'Pinned completed gated-factors release required')
    ancestor(root, BASE_COMMIT)
    pinned = inventory(root, sorted(REUSED_SOURCE | PARENT_FILES | GRAPH_FILES | INPUT_FILES))
    committed(root, BASE_COMMIT, pinned)
    prefix = 'results/gated_factors_v1/'
    reg_path = safe(root, 'results/registration_gated_factors_v1.json')
    reg, meta = read(reg_path), read(safe(root, prefix + 'metadata.json'))
    summary = read(safe(root, prefix + 'summary.json'))
    verification_path = safe(root, prefix + 'verification.json')
    verification = read(verification_path)
    require(reg.get('study_id') == meta.get('study_id') == summary.get('study_id')
            == verification.get('study_id') == 'gated_factors_v1'
            and meta.get('status') == meta.get('stage') == summary.get('status') == 'completed'
            and meta.get('analysis_only') is True and verification.get('status') == 'passed'
            and verification.get('scientific_producer_imported') is False
            and verification.get('new_helper_imported') is False,
            'Completed independently checked descriptive gated-factors parent required')
    require(safe(root, prefix + 'registration.json').read_bytes() == reg_path.read_bytes()
            and meta.get('registration_sha256') == verification.get('registration_sha256') == sha(reg_path)
            and meta.get('source_git_commit') == reg.get('source_git_commit')
            and verification.get('summary_sha256') == sha(safe(root, prefix + 'summary.json'))
            and meta.get('counts') == summary.get('counts'), 'Parent evidence identity mismatch')
    proof_name, proof = prior_publication(root, reg_path, reg,
        safe(root, 'results/gated_factors_publication_proof.json'))
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
    replay_true = ('factual_head_prefixes_and_prior_accounting_checked',
        'only_post_both_actual_Go_eight_corners_checked', 'state_scalar_direction_symmetric_allocations_checked',
        'signed_positive_and_negative_mass_allocations_checked', 'own_other_and_all_unpaired_actual_actions_accounted',
        'factual_diagnostics_and_nullable_seed_means_checked', 'partial_control_information_not_silently_equalized',
        'no_rng_model_graph_solver_or_future_policy_calls')
    require(all(checks.get(key) is True for key in replay_true)
            and checks.get('arrays') == 14 and checks.get('parent_trial_records_checked') == 120000
            and checks.get('destination_effects_checked') == 480000
            and checks.get('measurement_records_checked') == 2880
            and all(type(checks.get(key)) is int and checks[key] == 0 for key in (
                'new_training_trials', 'new_random_draws', 'new_optimization_calls')),
            'Prior complete saved-factor independent arithmetic required')
    counts = summary.get('counts', {})
    require(all(type(counts.get(key)) is int and counts[key] == 0 for key in (
        'new_training_trials', 'new_head_update_calls', 'new_internal_update_calls',
        'new_value_update_calls', 'new_random_draws', 'fresh_model_constructions', 'new_optimization_calls'))
        and counts.get('analyzed_parent_trial_records') == 120000
        and counts.get('analyzed_destination_effects') == 480000
        and counts.get('numeric_arrays') == 14 and counts.get('measurement_records') == 2880
        and type(counts.get('eligible_both_Go_trials')) is int
        and type(counts.get('paired_post_trial_records')) is int
        and 0 <= counts['eligible_both_Go_trials'] <= counts['paired_post_trial_records'] <= 96000
        and counts.get('corner_destination_effects') == counts['eligible_both_Go_trials'] * 32,
        'Prior saved factor arithmetic must remain separate from actual learning')
    interpretation = summary.get('interpretation', {})
    require(interpretation.get('parent_H_gain') is False
            and interpretation.get('parent_H_acquisition') is False
            and interpretation.get('parent_H_protection') is True
            and interpretation.get('new_confirmatory_hypothesis') is False
            and interpretation.get('accounting_allocation_not_unique_causal_explanation') is True
            and interpretation.get('mixed_one_step_cells_not_relearned_policy') is True
            and interpretation.get('unpaired_actions_not_assigned_to_zero_direction') is True
            and summary.get('validation', {}).get('all_exact_accounting_gates_passed') is True,
            'Prior unsupported gain/acquisition and descriptive factor interpretation required')
    for group in ('source_files', 'parent_files'):
        values = reg.get(group)
        require(type(values) is dict and bool(values), 'Complete parent frozen closure required')
        require(all(sha(safe(root, name)) == digest for name, digest in values.items()),
                'Parent frozen bytes changed: ' + group)
        committed(root, reg['source_git_commit'], values)
    require(len(reg['source_files']) == 20 and len(reg['parent_files']) == 12,
            'Exact parent source/parent inventory required')
    ancestor(root, reg['source_git_commit'], BASE_COMMIT)
    archive = read(safe(root, 'results/gated_factors_archive_validation.json'))
    require(archive.get('status') == 'passed'
            and archive.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True
            and archive.get('scientific_verification_sha256') == sha(verification_path)
            and archive.get('registration_sha256') == sha(reg_path)
            and archive.get('source_git_commit') == reg['source_git_commit']
            and archive.get('counts') == counts, 'Completed parent archive binding required')
    receipt = safe(root, 'results/gated_factors_baseline_preservation.json')
    require(archive.get('baseline_hash_receipt_sha256') == sha(receipt)
            and read(receipt).get('status') == 'passed'
            and archive.get('allowed_evolving_indexes') == read(receipt).get('allowed_evolving_indexes')
                == sorted(EVOLVING_INDEXES), 'Exact three-index prior Git-blob baseline receipt required')
    prefreeze_gate = archived_factor_prefreeze_gate(root, archive, reg, meta, proof, verification)
    prepaths = set(archive['prefreeze_receipts_committed_source_sha256'])
    snapshot = archive.get('current_file_sha256')
    require(type(snapshot) is dict and bool(snapshot)
            and set(reg['source_files']) | prepaths
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
        'human_shell_H_similarity', 'human_shell_H_prediction', 'gated_learning_H_gain', 'gated_learning_H_acquisition'))
        and all(status.get('gated_learning_H_protection') is True for status in (old_status, current_status)),
        'Prior negative learning/human outcomes and operational protection must remain preserved')
    return {'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, 'inherited_parent_gate': inherited,
        'pinned_baseline_files': len(pinned), 'historical_archive_snapshot_files': len(snapshot),
        'parent_saved_factor_accounting_and_original_autonomous_replay_reused_not_rerun': True,
        'original_factor_prefreeze_failure_and_exception_gate': prefreeze_gate,
        'parent_raw_source_data_Git_archive_and_prefreeze_bytes_checked': True,
        'prior_unsupported_gain_acquisition_and_operational_protection_preserved': True,
        'parent_gate_no_new_learner_graph_seed_or_optimizer_calls': True,
        'whole_repository_literal_pass_not_repeated': True}


def frozen_inventory(root, cfg):
    return {group: inventory(root, cfg[group]) for group in
            ('source_files', 'parent_files', 'graph_files', 'input_files')}


def validate_counts(cfg, summary, row_count):
    seeds, trials = cfg.get('model_seeds'), cfg.get('training_trials')
    checkpoints = cfg.get('checkpoints')
    require(type(seeds) is list and len(seeds) == 10 and seeds == list(range(17701, 17711))
            and type(trials) is int and trials == 400
            and checkpoints == [0, 50, 100, 200, 400], 'Fixed actual-learning dimensions required')
    conditions, groups = 36, 6
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
    reg = {'study_id': STUDY_ID, 'registration_type': 'prospective_observed_gated_self_blend_learning',
           'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': commit,
           'config': cfg, 'config_sha256': sha(config), 'registration_path': REGISTRATION,
           'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, **frozen, 'gate': gate,
           'notice': 'Prior geometric and failed learning outcomes known; actual observed-feedback self-blend learning with a separate new cohort, fixed unit caps and unknown-harm assessment. Frozen before new graph, seed, or learner computation. Computational seeds are not animals.'}
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
        require(type(arrays) is dict and len(arrays) == NUMERIC_ARRAY_COUNT
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
        require(type(meta.get('numeric_arrays')) is int and meta['numeric_arrays'] == len(arrays.files) == NUMERIC_ARRAY_COUNT
                and all(arrays[key].dtype.kind in 'biufc' and np.all(np.isfinite(arrays[key]))
                        for key in arrays.files), 'Saved finite numeric array inventory differs')
    require(type(meta.get('measurement_records')) is int and meta['measurement_records'] == len(rows)
            and len(rows) > 0, 'Saved measurement row count differs')
    validate_counts(cfg, summary, len(rows))
    return cfg, reg, meta
