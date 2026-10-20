"""Byte and chronology gates for a read-only, descriptive gated-factors audit.

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

STUDY_ID = 'gated_factors_v1'
MODULE = 'drosolearn.gated_factors'
PROTOCOL = 'docs/gated-factors-protocol.md'
CONFIG = 'configs/gated_factors_v1.json'
REGISTRATION = 'results/registration_gated_factors_v1.json'
RESULT = 'results/gated_factors_v1'
BASE_TAG = 'gated-pathways-v1'
BASE_COMMIT = 'e74604a67479299825e2ab5572fd3efd8f9970d6'
REPOSITORY = 'https://github.com/CauchyRiemannEquations/DrosoLearn'
PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json')
REUSED_SOURCE = frozenset({
    'pyproject.toml', 'requirements-lock.txt', 'src/drosolearn/__init__.py',
    'src/drosolearn/representation_envelope.py', 'src/drosolearn/balanced_envelope.py',
    'src/drosolearn/selective_envelope.py', 'src/drosolearn/normalization_envelope.py',
    'src/drosolearn/cone_envelope.py', 'src/drosolearn/pathways_envelope.py',
    'src/drosolearn/gated_envelope.py', 'src/drosolearn/gated_pathways_envelope.py',
})
SOURCE_FILES = REUSED_SOURCE | frozenset({
    CONFIG, PROTOCOL, 'src/drosolearn/gated_factors.py',
    'src/drosolearn/gated_factors_envelope.py', 'scripts/run_gated_factors.py',
    'scripts/verify_gated_factors.py', 'tests/test_gated_factors.py',
    'tests/test_gated_factors_envelope.py', 'tests/test_gated_factors_verify.py',
})
PARENT_FILES = frozenset({
    'results/registration_gated_pathways_v1.json',
    'results/gated_pathways_publication_proof.json',
    'results/gated_pathways_archive_validation.json',
    'results/gated_pathways_baseline_preservation.json',
    'results/gated_learning_v1/arrays.npz',
    *('results/gated_pathways_v1/' + name for name in
      (*PRIMARY, 'verification.json', 'artifact_hashes.json')),
})
EVOLVING_INDEXES = frozenset({'README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'})


def validate_envelope(root, cfg, config):
    require(type(cfg) is dict and cfg.get('study_id') == STUDY_ID
            and cfg.get('module') == MODULE and cfg.get('protocol') == PROTOCOL
            and cfg.get('scope') == 'retrospective_descriptive_gated_factors',
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
    from .gated_pathways_envelope import parent_gate as inherited_gate, publication as prior_publication
    inherited = inherited_gate(root)
    require(git(root, 'rev-parse', BASE_TAG).decode().strip() == BASE_COMMIT,
            'Pinned completed gated-pathways release required')
    ancestor(root, BASE_COMMIT)
    pinned = inventory(root, sorted(REUSED_SOURCE | PARENT_FILES))
    committed(root, BASE_COMMIT, pinned)
    prefix = 'results/gated_pathways_v1/'
    reg_path = safe(root, 'results/registration_gated_pathways_v1.json')
    reg, meta = read(reg_path), read(safe(root, prefix + 'metadata.json'))
    summary = read(safe(root, prefix + 'summary.json'))
    verification_path = safe(root, prefix + 'verification.json')
    verification = read(verification_path)
    require(reg.get('study_id') == meta.get('study_id') == summary.get('study_id')
            == verification.get('study_id') == 'gated_pathways_v1'
            and meta.get('status') == meta.get('stage') == summary.get('status') == 'completed'
            and meta.get('analysis_only') is True and verification.get('status') == 'passed'
            and verification.get('scientific_producer_imported') is False
            and verification.get('new_helper_imported') is False,
            'Completed independently checked descriptive gated-pathways parent required')
    require(safe(root, prefix + 'registration.json').read_bytes() == reg_path.read_bytes()
            and meta.get('registration_sha256') == verification.get('registration_sha256') == sha(reg_path)
            and meta.get('source_git_commit') == reg.get('source_git_commit')
            and verification.get('summary_sha256') == sha(safe(root, prefix + 'summary.json'))
            and meta.get('counts') == summary.get('counts'), 'Parent evidence identity mismatch')
    proof_name, proof = prior_publication(root, reg_path, reg,
        safe(root, 'results/gated_pathways_publication_proof.json'))
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
    replay_true = ('all_factual_head_prefixes_and_parent_effects_reconstructed',
        'all_methods_use_same_reference_prefeedback_time_boundary',
        'own_method_observations_used_for_known_unknown_categories',
        'last_discovery_trial_pre_and_next_trial_post_checked',
        'positive_negative_masses_and_endpoint_contributions_separated',
        'costs_Go_counts_and_nullable_descriptive_ratios_checked',
        'all_four_controls_and_all_thirty_families_checked',
        'parent_full_autonomous_replay_and_byte_gate_reused_by_frozen_hashes',
        'no_counterfactual_direction_policy_rng_graph_or_solver_calls')
    require(all(checks.get(key) is True for key in replay_true)
            and checks.get('arrays') == 12 and checks.get('parent_trial_records_checked') == 120000
            and checks.get('destination_effects_checked') == 480000
            and checks.get('measurement_records_checked') == 1800
            and checks.get('new_training_trials') == checks.get('new_random_draws')
                == checks.get('new_optimization_calls') == 0,
            'Prior complete saved-path independent arithmetic required')
    counts = summary.get('counts', {})
    require(all(type(counts.get(key)) is int and counts[key] == 0 for key in (
        'new_training_trials', 'new_head_update_calls', 'new_internal_update_calls',
        'new_value_update_calls', 'new_random_draws', 'fresh_model_constructions', 'new_optimization_calls'))
        and counts.get('analyzed_parent_trial_records') == 120000
        and counts.get('analyzed_destination_effects') == 480000
        and counts.get('numeric_arrays') == 12 and counts.get('measurement_records') == 1800,
        'Prior saved accounting must not be relabeled as additional learning')
    interpretation = summary.get('interpretation', {})
    require(interpretation.get('parent_H_gain') is False
            and interpretation.get('parent_H_acquisition') is False
            and interpretation.get('parent_H_protection') is True
            and interpretation.get('new_confirmatory_hypothesis') is False
            and interpretation.get('exact_path_accounting_not_unique_causal_explanation') is True
            and interpretation.get('reference_gate_is_common_time_split_not_all_methods_knowledge') is True
            and summary.get('validation', {}).get('all_exact_accounting_gates_passed') is True,
            'Prior unsupported gain/acquisition and operational protection interpretation required')
    for group in ('source_files', 'parent_files'):
        values = reg.get(group)
        require(type(values) is dict and bool(values), 'Complete parent frozen closure required')
        require(all(sha(safe(root, name)) == digest for name, digest in values.items()),
                'Parent frozen bytes changed: ' + group)
        committed(root, reg['source_git_commit'], values)
    require(len(reg['source_files']) == 19 and len(reg['parent_files']) == 11,
            'Exact parent source/parent inventory required')
    ancestor(root, reg['source_git_commit'], BASE_COMMIT)
    archive = read(safe(root, 'results/gated_pathways_archive_validation.json'))
    require(archive.get('status') == 'passed'
            and archive.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True
            and archive.get('scientific_verification_sha256') == sha(verification_path)
            and archive.get('registration_sha256') == sha(reg_path)
            and archive.get('source_git_commit') == reg['source_git_commit']
            and archive.get('counts') == counts, 'Completed parent archive binding required')
    receipt = safe(root, 'results/gated_pathways_baseline_preservation.json')
    require(archive.get('baseline_hash_receipt_sha256') == sha(receipt)
            and read(receipt).get('status') == 'passed'
            and archive.get('allowed_evolving_indexes') == read(receipt).get('allowed_evolving_indexes')
                == sorted(EVOLVING_INDEXES), 'Exact three-index prior Git-blob baseline receipt required')
    prefreeze = archive.get('prefreeze_receipts_committed_source_sha256')
    prepaths = {'results/gated_pathways_prefreeze_validation.json',
        'results/gated_pathways_baseline_preservation.json',
        'results/gated_pathways_prefreeze_tests/gated_pathways_prefreeze_final2.xml'}
    require(type(prefreeze) is dict and set(prefreeze) == prepaths
            and all(sha(safe(root, name)) == digest for name, digest in prefreeze.items()),
            'Exact original parent prefreeze/JUnit/source-byte receipt required')
    committed(root, reg['source_git_commit'], prefreeze)
    tests = read(safe(root, 'results/gated_pathways_prefreeze_validation.json'))
    require(tests.get('status') == 'passed' and tests.get('frozen_source_hashes') == reg['source_files']
            and tests.get('parent_byte_gate') == reg['gate'] and tests.get('unique_passed_tests') == 256
            and tests.get('new_unique_passed_tests') == 142
            and tests.get('unchanged_related_regression_passed_tests') == 114
            and tests.get('actual_new_saved_path_aggregation_before_freeze') == 0,
            'Original complete 256-test prefreeze snapshot required')
    chronology = archive.get('prefreeze_publication_analysis_verification_utc_order')
    expected_times = [read(receipt)['verified_at_utc'], tests['checked_at_utc'], reg['registered_at_utc'],
        proof['checked_at_utc'], meta['started_at_utc'], meta['completed_at_utc'], verification['verified_at_utc']]
    require(chronology == expected_times, 'Archived original parent UTC timestamps differ')
    parsed = list(map(utc, chronology))
    require(parsed == sorted(parsed) and parsed[2] < parsed[3] < parsed[4]
            and parsed[-1] <= utc(archive['verified_at_utc']) <= datetime.now(timezone.utc),
            'Original parent prefreeze-before-registration and publication-before-analysis chronology required')
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
        'parent_saved_accounting_and_original_autonomous_replay_reused_not_rerun': True,
        'parent_raw_source_data_Git_archive_and_prefreeze_bytes_checked': True,
        'prior_unsupported_gain_acquisition_and_operational_protection_preserved': True,
        'parent_gate_no_new_learner_graph_seed_or_optimizer_calls': True,
        'whole_repository_literal_pass_not_repeated': True}


def frozen_inventory(root, cfg):
    return {group: inventory(root, cfg[group]) for group in ('source_files', 'parent_files')}


def validate_counts(cfg, summary, row_count):
    require(cfg.get('model_seeds') == list(range(17601, 17611))
            and type(cfg.get('training_trials')) is int and cfg['training_trials'] == 400,
            'Fixed saved actual-learning cohort and budget required')
    expected = {'analyzed_parent_trial_records': 120000, 'analyzed_destination_effects': 480000,
                'reference_gate_paths': 60, 'comparison_paths': 240,
                'paired_trial_records': 96000, 'measurement_records': 2880, 'numeric_arrays': 14,
                'new_training_trials': 0, 'new_head_update_calls': 0,
                'new_internal_update_calls': 0, 'new_value_update_calls': 0,
                'new_random_draws': 0, 'fresh_model_constructions': 0, 'new_optimization_calls': 0}
    counts = summary.get('counts', {})
    variable = {'paired_post_trial_records', 'eligible_both_Go_trials', 'corner_destination_effects'}
    require(type(counts) is dict and set(counts) == set(expected) | variable,
            'Exact saved-factor work count keys required')
    require(all(type(counts[key]) is int and counts[key] == value for key, value in expected.items())
            and all(type(counts[key]) is int for key in variable)
            and 0 <= counts['eligible_both_Go_trials'] <= counts['paired_post_trial_records'] <= 96000
            and counts['corner_destination_effects'] == 32 * counts['eligible_both_Go_trials']
            and type(row_count) is int and row_count == expected['measurement_records'],
            'Exact saved-factor counts and zero new training/random/model/optimization work required')
    require(summary.get('study_id') == STUDY_ID and summary.get('status') == 'completed'
            and summary.get('scope') == 'retrospective_descriptive_gated_factors',
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
    reg = {'study_id': STUDY_ID, 'registration_type': 'retrospective_descriptive_gated_factors',
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
