"""Seal bounded project closure without rerunning any original science.

The original first Phase5, all post-Phase5 releases and failed attempts remain
byte-identical. Only the three explicit status indexes may change.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from drosolearn.closure_envelope import (
    BASE_COMMIT, BASE_TAG, INDEXES, PRIMARY, REPRO, RESULT, committed, git,
    read, require, safe, sha, utc, verify_envelope, write_new,
)

STUDY = 'followup_closure_v1'
NEW_CLASSES = frozenset({'tests.test_followup_closure', 'tests.test_followup_closure_envelope',
    'tests.test_followup_closure_verify'})
OLD_CLASSES = {'tests.test_objective_pathways': 27, 'tests.test_objective_pathways_verify': 94}
DOCS = frozenset({'docs/followup-closure-' + name + '.md' for name in
    ('final-report', 'evidence-appendix', 'reproduction', 'limitations', 'plain-language')}) | frozenset(INDEXES)
COUNT_CONTRACT = {'studies': 17, 'claims': 24, 'literal_observations': 80,
    'actual_learning_registrations': 8, 'saved_record_registrations': 9,
    'numeric_arrays': 2, 'measurement_records': 24, 'new_training_trials': 0,
    'new_model_graph_rng_optimization_calls': 0, 'new_effect_tests': 0,
    'new_human_experiments': 0, 'past_full_science_reruns': 0, 'saved_record_independent_rechecks': 1}
CORRECTION = 'followup_closure_consumer_v2'
CORRECTION_CONFIG = 'configs/followup_closure_consumer_v2.json'
CORRECTION_REGISTRATION = 'results/registration_followup_closure_consumer_v2.json'
CORRECTION_PUBLICATION = 'results/followup_closure_consumer_v2_publication_proof.json'
CORRECTION_VERIFICATION = 'results/followup_closure_consumer_v2/verification.json'
CORRECTION_PREFREEZE = 'results/followup_closure_consumer_v2_prefreeze_validation.json'
CORRECTION_JUNIT = 'results/followup_closure_consumer_v2_prefreeze_tests/final.xml'
CORRECTION_SOURCES = frozenset({CORRECTION_CONFIG, 'docs/followup-closure-consumer-v2-protocol.md',
    'scripts/verify_followup_closure_consumer_v2.py', 'tests/test_followup_closure_consumer_v2.py'})
FIRST_FAILURE = 'results/followup_closure_verification_attempt01.json'
FIRST_FAILURE_ZIP = 'results/followup_closure_verification_attempt01.zip'
HUMAN_FILES = frozenset({'results/human_shell_rng_v2/' + n for n in
    ('summary.json', 'verification.json', 'registration.json', 'metadata.json', 'artifact_hashes.json')})
HUMAN_SCHEMA = frozenset({'H_prediction', 'H_similarity', 'autonomous_nll_interpreted', 'autonomous_trials',
    'behavior_family_comparisons', 'behavior_metric_order', 'bootstrap', 'condition_order',
    'conditional_family_mean_brier', 'conditional_family_mean_nll', 'conditional_trials', 'educational_claim',
    'human_metrics', 'human_test_people', 'learning_qualification', 'mechanism_claim', 'mode_order',
    'model_seeds', 'status', 'test_modeled_trials'})
HUMAN_COMMIT = '3ee71a97fa9d6ffc7fce3bd4d7c3832437977171'
ORIGINAL_SIX = frozenset({RESULT + '/' + n for n in (*PRIMARY, 'artifact_hashes.json')})


def correction_parent_paths(original):
    """Reconstruct the extra consumer's 54 immutable inputs independently."""
    return set(original['source_files']) | set(ORIGINAL_SIX) | set(HUMAN_FILES) | {
        'results/registration_followup_closure_v1.json', 'results/followup_closure_publication_proof.json',
        REPRO, 'results/followup_closure_reproduction_v1/invocation.json', FIRST_FAILURE, FIRST_FAILURE_ZIP,
        'scripts/verify_human_shell.py', 'scripts/verify_human_shell_rng.py'}


def original_consumer_failure_gate(root, original):
    """Preserve the real first rejection; never invent a successful original receipt."""
    note = read(safe(root, FIRST_FAILURE))
    require(note.get('status') == 'preserved_original_independent_consumer_failure'
        and note.get('study_id') == STUDY and type(note.get('first_verifier_exit_code')) is int
        and note['first_verifier_exit_code'] == 1 and note.get('first_verifier_receipt_written') is False
        and note.get('failure') == 'ValueError: Original anchor evidence identities'
        and note.get('archive') == FIRST_FAILURE_ZIP
        and note.get('archive_sha256') == sha(safe(root, FIRST_FAILURE_ZIP))
        and note.get('source_git_commit') == original['source_git_commit']
        and note.get('original_registration_sha256') == sha(safe(root, 'results/registration_followup_closure_v1.json'))
        and note.get('original_frozen_source_files_unchanged') is True
        and note.get('original_primary_files_unchanged') is True
        and note.get('original_verifier_or_producer_rerun') is False
        and type(note.get('new_learning_or_optimization_calls')) is int and note['new_learning_or_optimization_calls'] == 0
        and note.get('scientific_result_or_acquisition_gate_waived') is False
        and not safe(root, RESULT + '/verification.json').exists(),
        'The first actual consumer failure and absent original receipt must remain unchanged')
    members = note.get('member_sha256')
    expected = set(original['source_files']) | set(ORIGINAL_SIX) | {
        'results/registration_followup_closure_v1.json', 'results/followup_closure_publication_proof.json',
        'original_tool_result.json'}
    require(type(members) is dict and set(members) == expected and len(expected) == 44,
        'Exact original 35 sources, six outputs, registration/proof and original tool capture')
    with ZipFile(safe(root, FIRST_FAILURE_ZIP)) as archive:
        require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == expected
            and all(hashlib.sha256(archive.read(n)).hexdigest() == digest for n, digest in members.items()),
            'Exact original consumer failure ZIP bytes')
        tool = json.loads(archive.read('original_tool_result.json'))
        require(type(tool.get('exit_code')) is int and tool['exit_code'] == 1
            and tool.get('chunk_id') == note.get('original_tool_chunk_id')
            and tool.get('session_id') == note.get('original_tool_session_id')
            and type(tool.get('output')) is str and tool['output'].endswith('ValueError: Original anchor evidence identities\r\n')
            and tool.get('argv') == ['scripts/verify_followup_closure.py', '--config', 'configs/followup_closure_v1.json',
                '--registration', 'results/registration_followup_closure_v1.json', '--result-dir', RESULT,
                '--publication-proof', 'results/followup_closure_publication_proof.json', '--out', RESULT + '/verification.json'],
            'Raw first traceback, tool identity and original exclusive output path retained')
    require(all(sha(safe(root, n)) == digest for n, digest in members.items() if n != 'original_tool_result.json'),
        'Original frozen source, primary or publication bytes differ from first-failure capture')
    return note


def correction_prefreeze_gate(root, reg):
    """Count only the separate consumer cases; do not add them to the original 267."""
    tests = read(safe(root, CORRECTION_PREFREEZE))
    require(tests.get('status') == 'passed' and tests.get('frozen_source_hashes') == reg['source_files']
        and tests.get('frozen_parent_hashes') == reg['parent_files']
        and tests.get('source_before_test_hashes') == reg['source_files']
        and tests.get('source_after_test_hashes') == reg['source_files']
        and tests.get('synthetic_fixtures_only_before_freeze') is True
        and all(type(tests.get(k)) is int and tests[k] == 0 for k in (
            'actual_corrected_verification_before_freeze', 'new_training_rng_graph_optimizer_human_effect_calls')),
        'Separate extra consumer tests must use exact source/parent freeze and no actual verification')
    receipt = tests.get('test_receipt'); dev = tests.get('development_failure_artifacts')
    require(type(receipt) is dict and receipt.get('path') == CORRECTION_JUNIT and type(dev) is list
        and all(type(d) is dict and type(d.get('path')) is str and type(d.get('sha256')) is str for d in [receipt] + dev)
        and len({d['path'] for d in [receipt] + dev}) == 1 + len(dev)
        and all(sha(safe(root, d['path'])) == d['sha256'] for d in [receipt] + dev),
        'One exact extra consumer JUnit file and every development artifact')
    directory = safe(root, 'results/followup_closure_consumer_v2_development')
    present = {p.relative_to(root).as_posix() for p in directory.rglob('*') if p.is_file()} if directory.exists() else set()
    require(present == {d['path'] for d in dev}, 'Every extra consumer development failure is preserved')
    for name in sorted(present):
        if not name.endswith('.json'):
            continue
        note = read(safe(root, name))
        if 'archive' not in note:
            continue
        archive_name = note['archive']
        if '/' not in archive_name and '\\' not in archive_name:
            archive_name = 'results/followup_closure_consumer_v2_development/' + archive_name
        members = note.get('member_sha256')
        require(archive_name in present and archive_name.endswith('.zip')
            and note.get('archive_sha256') == sha(safe(root, archive_name))
            and type(members) is dict and bool(members), 'Extra consumer development ZIP must be fully inventoried')
        normalized = {member.replace('\\', '/'): digest for member, digest in members.items()}
        require(len(normalized) == len(members), 'Distinct extra development ZIP member names')
        with ZipFile(safe(root, archive_name)) as archive:
            require(len(archive.namelist()) == len(normalized) and set(archive.namelist()) == set(normalized)
                and all(hashlib.sha256(archive.read(member)).hexdigest() == digest for member, digest in normalized.items()),
                'Every additional consumer development ZIP byte retained')
    cases = ET.parse(safe(root, CORRECTION_JUNIT)).findall('.//testcase')
    require(bool(cases) and len(cases) == len({(c.get('classname'), c.get('name')) for c in cases})
        and all(c.get('classname') == 'tests.test_followup_closure_consumer_v2' and c.get('name') for c in cases)
        and all(c.find(k) is None for c in cases for k in ('failure', 'error', 'skipped'))
        and type(tests.get('unique_passed_tests')) is int
        and type(receipt.get('passed_unique_tests')) is int
        and tests['unique_passed_tests'] == receipt['passed_unique_tests'] == len(cases)
        and tests.get('unique_passed_tests_by_class') == {'tests.test_followup_closure_consumer_v2': len(cases)},
        'All distinct additional consumer cases pass in their separate class and receipt')
    paths = {CORRECTION_PREFREEZE, CORRECTION_JUNIT} | {d['path'] for d in dev}
    committed(root, reg['source_git_commit'], paths)
    return tests, paths, len(cases)


def human_original_schema_gate(root):
    """Check historical identity through its real schema, not a fabricated field."""
    directory = 'results/human_shell_rng_v2/'
    summary = read(safe(root, directory + 'summary.json'))
    verified = read(safe(root, directory + 'verification.json'))
    registered = read(safe(root, directory + 'registration.json'))
    meta = read(safe(root, directory + 'metadata.json'))
    hashes = read(safe(root, directory + 'artifact_hashes.json'))
    require(set(summary) == HUMAN_SCHEMA and 'study_id' not in summary and summary.get('status') == 'completed'
        and verified.get('status') == 'passed' and meta.get('status') == 'completed'
        and meta.get('stage') == 'completed' and verified.get('verifier_imports_producer') is False
        and verified.get('verifier_modifies_original_results') is False
        and verified.get('all_frozen_inputs_match') is True and verified.get('parent_arithmetic_verification_passed') is True
        and all(d.get('study_id') == 'human_shell_rng_v2' for d in (verified, registered, meta))
        and verified.get('summary_sha256') == hashes.get('summary.json') == sha(safe(root, directory + 'summary.json'))
        and verified.get('registration_sha256') == meta.get('registration_sha256') == sha(safe(root, directory + 'registration.json'))
        and meta.get('config_sha256') == registered.get('config_sha256')
        and meta.get('source_git_commit') == registered.get('source_git_commit'),
        'Human anchor exact original 20-key schema, real identity and linked original bytes')
    completion = git(root, 'rev-parse', 'human-shell-comparison-v2').decode().strip()
    require(completion == HUMAN_COMMIT, 'Exact unmoved original human completion tag commit')
    source = registered.get('source_git_commit')
    require(type(source) is str and re.fullmatch('[0-9a-f]{40}', source)
        and registered.get('config', {}).get('study_id') == 'human_shell_rng_v2', 'Original human registered source and config identity')
    config_name = 'configs/human_shell_rng_v2.json'; config_bytes = git(root, 'show', source + ':' + config_name)
    require(hashlib.sha256(config_bytes).hexdigest() == registered['config_sha256']
        and json.loads(config_bytes) == registered['config']
        and registered.get('source_files', {}).get(config_name) == registered['config_sha256'],
        'Human original source-committed configuration bytes and values')
    scripts = {'scripts/verify_human_shell.py', 'scripts/verify_human_shell_rng.py'}
    require(all(name in registered['source_files'] and sha(safe(root, name)) == registered['source_files'][name] for name in scripts),
        'Human original schema consumer source bytes')
    committed(root, source, scripts)
    git(root, 'merge-base', '--is-ancestor', source, completion)
    git(root, 'merge-base', '--is-ancestor', completion, 'HEAD')
    committed(root, completion, HUMAN_FILES)
    return {'original_schema_keys': sorted(HUMAN_SCHEMA), 'original_study_id_absent': True,
        'completion_tag': 'human-shell-comparison-v2', 'completion_commit': completion,
        'tag_primary_sha256': {name: sha(safe(root, name)) for name in sorted(HUMAN_FILES)}}


def consumer_correction_gate(root, original, verified):
    """Independently bind the additional published consumer and its limited repair."""
    cfg = read(safe(root, CORRECTION_CONFIG)); reg = read(safe(root, CORRECTION_REGISTRATION))
    publication = read(safe(root, CORRECTION_PUBLICATION))
    parents = correction_parent_paths(original)
    require(cfg.get('study_id') == reg.get('study_id') == CORRECTION
        and cfg.get('scope') == 'strict_original_schema_consumer_correction_only'
        and cfg.get('original_config') == 'configs/followup_closure_v1.json'
        and cfg.get('original_registration') == 'results/registration_followup_closure_v1.json'
        and cfg.get('original_publication_proof') == 'results/followup_closure_publication_proof.json'
        and cfg.get('original_result_dir') == RESULT
        and cfg.get('original_verifier') == 'scripts/verify_followup_closure.py'
        and cfg.get('original_failure') == FIRST_FAILURE and cfg.get('output') == CORRECTION_VERIFICATION
        and cfg.get('human_anchor_id') == 'human_comparison' and cfg.get('human_study_id') == 'human_shell_rng_v2'
        and cfg.get('human_summary') == 'results/human_shell_rng_v2/summary.json'
        and cfg.get('human_verification') == 'results/human_shell_rng_v2/verification.json'
        and cfg.get('human_registration') == 'results/human_shell_rng_v2/registration.json'
        and cfg.get('human_metadata') == 'results/human_shell_rng_v2/metadata.json'
        and cfg.get('human_artifact_hashes') == 'results/human_shell_rng_v2/artifact_hashes.json'
        and cfg.get('human_completion_tag') == 'human-shell-comparison-v2'
        and all(cfg.get(k) is False for k in ('scientific_result_changed', 'scientific_success_criteria_changed',
            'original_source_or_primary_files_changed'))
        and type(cfg.get('new_learning_rng_optimization_effect_human_calls')) is int
        and cfg['new_learning_rng_optimization_effect_human_calls'] == 0,
        'Exactly scoped human original-schema correction configuration')
    require(type(cfg.get('source_files')) is list and len(cfg['source_files']) == 4
        and set(cfg['source_files']) == CORRECTION_SOURCES
        and type(cfg.get('parent_files')) is list and len(cfg['parent_files']) == 54
        and len(parents) == 54 and set(cfg['parent_files']) == parents
        and reg.get('config') == cfg and reg.get('config_sha256') == sha(safe(root, CORRECTION_CONFIG))
        and reg.get('source_files') == {n: sha(safe(root, n)) for n in sorted(CORRECTION_SOURCES)}
        and reg.get('parent_files') == {n: sha(safe(root, n)) for n in sorted(parents)}
        and type(reg.get('source_git_commit')) is str and re.fullmatch('[0-9a-f]{40}', reg['source_git_commit']),
        'Separate exact four-source and fifty-four-parent registration inventories')
    registration_fields = {'study_id', 'registration_type', 'registered_at_utc', 'source_git_commit',
        'config', 'config_sha256', 'source_files', 'parent_files', 'original_independent_failure_preserved',
        'scientific_success_criteria_changed', 'original_source_or_primary_files_changed',
        'new_learning_rng_optimization_effect_human_calls'}
    require(set(reg) == registration_fields
        and reg.get('registration_type') == 'retrospective_strict_schema_consumer_correction'
        and reg.get('original_independent_failure_preserved') is True
        and reg.get('scientific_success_criteria_changed') is False
        and reg.get('original_source_or_primary_files_changed') is False
        and type(reg.get('new_learning_rng_optimization_effect_human_calls')) is int
        and reg['new_learning_rng_optimization_effect_human_calls'] == 0,
        'Exact additional correction registration type and unchanged scientific scope')
    committed(root, reg['source_git_commit'], CORRECTION_SOURCES | parents)
    git(root, 'merge-base', '--is-ancestor', original['source_git_commit'], reg['source_git_commit'])
    published = publication.get('published_commit')
    require(publication.get('study_id') == CORRECTION and publication.get('registration_sha256') == sha(safe(root, CORRECTION_REGISTRATION))
        and publication.get('source_commit') == reg['source_git_commit']
        and publication.get('remote_bytes_match') is True and publication.get('visibility') == 'private'
        and type(published) is str and re.fullmatch('[0-9a-f]{40}', published)
        and publication.get('url') == 'https://github.com/CauchyRiemannEquations/DrosoLearn/blob/' + published + '/' + CORRECTION_REGISTRATION,
        'Exact separate private publication proof for correction registration')
    git(root, 'merge-base', '--is-ancestor', reg['source_git_commit'], published)
    git(root, 'merge-base', '--is-ancestor', published, 'HEAD')
    require(git(root, 'show', published + ':' + CORRECTION_REGISTRATION) == safe(root, CORRECTION_REGISTRATION).read_bytes(),
        'Exact correction registration Git publication bytes')
    failure = original_consumer_failure_gate(root, original)
    tests, test_paths, count = correction_prefreeze_gate(root, reg)
    human = human_original_schema_gate(root)
    expected = {'study_id': CORRECTION, 'scope': 'strict_original_schema_consumer_correction_only',
        'registration': CORRECTION_REGISTRATION, 'registration_sha256': sha(safe(root, CORRECTION_REGISTRATION)),
        'publication_proof': CORRECTION_PUBLICATION, 'publication_proof_sha256': sha(safe(root, CORRECTION_PUBLICATION)),
        'source_git_commit': reg['source_git_commit'], 'source_files': 4, 'parent_files': 54,
        'original_failure': FIRST_FAILURE, 'original_failure_sha256': sha(safe(root, FIRST_FAILURE)),
        'original_failure_archive': FIRST_FAILURE_ZIP, 'original_failure_archive_sha256': sha(safe(root, FIRST_FAILURE_ZIP)),
        'human_anchor_id': 'human_comparison', 'human_original_schema_checked': True,
        'human_original_tag_primary_files_checked': 5, 'other_anchor_checks_unchanged': True,
        'all_other_independent_checks_unchanged': True, 'original_source_or_primary_files_changed': False,
        'scientific_success_criteria_changed': False, 'new_learning_rng_optimization_effect_human_calls': 0}
    correction = verified.get('consumer_correction')
    require(type(correction) is dict and set(correction) == set(expected)
        and all(type(correction[k]) is type(v) and correction[k] == v for k, v in expected.items()),
        'Separate verification must identify exactly the limited registered consumer correction')
    require(utc(failure['captured_at_utc']) <= utc(tests['checked_at_utc']) <= utc(reg['registered_at_utc'])
        < utc(publication['checked_at_utc']) <= utc(verified['verified_at_utc']) <= datetime.now(timezone.utc),
        'Failure preservation precedes extra test freeze, separate registration/publication and verification')
    paths = set(CORRECTION_SOURCES) | parents | test_paths | {CORRECTION_REGISTRATION,
        CORRECTION_PUBLICATION, CORRECTION_VERIFICATION}
    return {'registration': reg, 'publication': publication, 'failure': failure, 'tests': tests,
        'test_paths': test_paths,
        'unique_additional_consumer_cases': count, 'paths': paths, 'human_original_identity': human,
        'consumer_correction': correction}


def prefreeze_gate(root, tests, reg, baseline_name, validation_name):
    """Bind the exact successful final cases and every original failed fixture."""
    require(tests.get('status') == 'passed' and tests.get('frozen_source_hashes') == reg['source_files']
        and tests.get('synthetic_fixtures_only_before_freeze') is True,
        'Exact source-committed synthetic prefreeze snapshot required')
    require(all(type(tests.get(name)) is int and tests[name] == 0 for name in (
        'actual_new_synthesis_before_freeze', 'actual_saved_only_reverification_before_freeze',
        'new_training_rng_graph_optimizer_human_effect_calls')),
        'No actual synthesis, saved recheck or new scientific work before freeze')
    receipts = tests.get('test_receipts'); development = tests.get('development_failure_artifacts')
    require(type(receipts) is list and len(receipts) == 1 and type(development) is list
        and receipts[0].get('path') == 'results/followup_closure_prefreeze_tests/followup_closure_prefreeze_final.xml',
        'One exact final JUnit receipt and explicit development inventory required')
    require(all(type(item) is dict and type(item.get('path')) is str and type(item.get('sha256')) is str
        for item in receipts + development)
        and len({item['path'] for item in receipts + development}) == len(receipts + development),
        'Distinct prefreeze evidence file identities')
    prepaths = {baseline_name, validation_name}
    for item in receipts + development:
        require(sha(safe(root, item['path'])) == item['sha256'], 'Original prefreeze receipt bytes differ')
        prepaths.add(item['path'])
    devnames = {item['path'] for item in development}
    directory = safe(root, 'results/followup_closure_development')
    present = {p.relative_to(root).as_posix() for p in directory.rglob('*') if p.is_file()} if directory.exists() else set()
    require(present == devnames, 'All original development records, including failures, must be frozen')
    cases = ET.parse(safe(root, receipts[0]['path'])).findall('.//testcase')
    require(bool(cases) and len(cases) == len({(c.get('classname'), c.get('name')) for c in cases})
        and all(c.get('classname') and c.get('name') for c in cases)
        and all(c.find(kind) is None for c in cases for kind in ('failure', 'error', 'skipped')),
        'Every unique final JUnit case must pass')
    classes = dict(sorted(Counter(c.get('classname') for c in cases).items()))
    new_count = sum(classes.get(name, 0) for name in NEW_CLASSES)
    require(set(classes) == NEW_CLASSES | set(OLD_CLASSES)
        and all(classes[name] == count for name, count in OLD_CLASSES.items())
        and classes == tests.get('unique_passed_tests_by_class')
        and tests.get('unique_passed_tests') == receipts[0].get('passed_unique_tests') == len(cases)
        and tests.get('new_unique_passed_tests') == new_count
        and tests.get('unchanged_related_regression_passed_tests') == sum(OLD_CLASSES.values())
        and len(cases) == new_count + sum(OLD_CLASSES.values()),
        'Exact dynamic new-case count and 121 unchanged related cases required')
    committed(root, reg['source_git_commit'], prepaths)
    zip_bindings = {}
    for name in sorted(devnames):
        if not name.endswith('.json'):
            continue
        note = read(safe(root, name))
        if 'archive' not in note:
            continue
        archive_name = note['archive']
        if '/' not in archive_name and '\\' not in archive_name:
            archive_name = 'results/followup_closure_development/' + archive_name
        require(archive_name in devnames and archive_name.endswith('.zip')
            and note.get('archive_sha256') == sha(safe(root, archive_name)),
            'Original development ZIP must remain tied to its manifest')
        expected = note.get('member_sha256')
        require(type(expected) is dict and bool(expected), 'Raw member SHA map required')
        normalized = {member.replace('\\', '/'): digest for member, digest in expected.items()}
        require(len(normalized) == len(expected), 'No duplicate ZIP identity after path separator normalization')
        with ZipFile(safe(root, archive_name)) as archive:
            require(len(archive.namelist()) == len(normalized) and set(archive.namelist()) == set(normalized)
                and all(hashlib.sha256(archive.read(member)).hexdigest() == digest for member, digest in normalized.items()),
                'Every original development ZIP byte must be preserved')
        zip_bindings[name] = {'archive': archive_name, 'archive_sha256': note['archive_sha256'],
            'member_sha256': normalized}
    return prepaths, classes, development, zip_bindings


def independent_gate(verified, summary, root, registration_name):
    require(verified.get('status') == 'passed' and verified.get('study_id') == STUDY
        and verified.get('scientific_producer_imported') is False
        and verified.get('registration_sha256') == sha(safe(root, registration_name))
        and verified.get('summary_sha256') == sha(safe(root, RESULT + '/summary.json')),
        'Independent PASS must bind this exact closure without importing its producer')
    require(summary.get('study_id') == STUDY and summary.get('status') == 'completed'
        and summary.get('scope') == 'retrospective_bounded_followup_synthesis'
        and summary.get('counts') == COUNT_CONTRACT, 'Complete fixed 17-study, 24-claim, 80-literal bounded closure')
    counts = verified.get('counts', {})
    expected = {'verified_followup_studies': 17, 'actual_learning_registrations': 8, 'saved_record_registrations': 9,
        'independently_checked_claims': 24, 'literal_observations': 80, 'exact_numeric_ledgers': 2,
        'new_training_trials': 0, 'new_model_graph_rng_optimization_calls': 0, 'new_effect_tests': 0,
        'new_human_experiments': 0, 'past_full_science_reruns': 0}
    require(all(type(counts.get(k)) is int and counts[k] == value for k, value in expected.items())
        and all(counts.get(k) is True for k in (
            'all_completion_tags_source_inputs_primary_hashes_and_chronology_checked',
            'all_historical_hypothesis_decisions_preserved', 'separate_saved_only_independent_reproduction_checked',
            'old_test_counts_not_summed_as_independent')),
        'Every independent source, result, tag, literal and retained-decision coordinate required')
    interpretation = summary.get('interpretation', {})
    require(all(interpretation.get(k) is True for k in ('bounded_local_project_completed', 'original_phase5_preserved',
        'all_failures_and_negative_decisions_preserved', 'long_term_questions_retained_without_automatic_execution'))
        and all(interpretation.get(k) is False for k in ('new_effect_or_learning_claims', 'pooled_effect_estimates_computed',
        'old_test_counts_summed_as_independent', 'novel_generalization_supported_by_this_synthesis',
        'human_similarity_supported', 'shared_neural_mechanism_verified', 'education_effect_tested',
        'automatic_continuation_enabled')),
        'Scope completion cannot become a stronger learning, human or biological claim')


def saved_reproduction_gate(root, cfg, summary, publication_time, analysis_start):
    original_name = 'results/objective_pathways_v1/verification.json'
    repeated = read(safe(root, REPRO)); original = read(safe(root, original_name))
    require(cfg['reproduction_receipt'] == REPRO and repeated.get('status') == original.get('status') == 'passed'
        and repeated.get('study_id') == original.get('study_id') == 'objective_pathways_v1'
        and repeated.get('registration_sha256') == original.get('registration_sha256')
        and repeated.get('summary_sha256') == original.get('summary_sha256')
        and repeated.get('counts') == original.get('counts')
        and repeated.get('scientific_producer_imported') is False and repeated.get('new_helper_imported') is False,
        'Separate actual saved-only numerical recheck must equal the original result and checks')
    require(utc(original['verified_at_utc']) < utc(publication_time) < utc(repeated['verified_at_utc'])
        <= utc(analysis_start), 'Publication must precede one new saved-only recheck and synthesis')
    evidence = summary.get('reproduction', {})
    require(evidence.get('receipt_path') == REPRO and evidence.get('sha256') == sha(safe(root, REPRO))
        and evidence.get('original_verification_sha256') == sha(safe(root, original_name))
        and evidence.get('summary_sha256') == repeated['summary_sha256']
        and evidence.get('counts') == original['counts'] and evidence.get('separate_new_receipt') is True
        and evidence.get('past_full_science_rerun') is False,
        'Saved-only reproduction scope and its exact new receipt must remain explicit')
    return {'original': original_name, 'original_sha256': sha(safe(root, original_name)),
        'new_receipt': REPRO, 'new_receipt_sha256': sha(safe(root, REPRO)),
        'new_training_rng_graph_optimization_or_full_science_reruns': 0}


def presentation_gate(root, verified_time, correction_test_paths):
    name = 'results/followup_closure_presentation_validation.json'; qa = read(safe(root, name))
    require(qa.get('status') == 'passed' and qa.get('study_id') == STUDY
        and type(qa.get('document_sha256')) is dict and set(qa['document_sha256']) == DOCS
        and type(qa.get('additional_file_sha256')) is dict, 'All five new documents and three status indexes require final QA')
    additional = set(ORIGINAL_SIX) | {REPRO, CORRECTION_VERIFICATION, FIRST_FAILURE, FIRST_FAILURE_ZIP,
        CORRECTION_REGISTRATION, CORRECTION_PUBLICATION, CORRECTION_PREFREEZE, CORRECTION_JUNIT} | set(CORRECTION_SOURCES)
    additional.update(correction_test_paths)
    require(additional <= set(qa['additional_file_sha256']),
        'QA must bind original six outputs, separate consumer correction, failure, tests and all checked evidence')
    for group in ('document_sha256', 'additional_file_sha256'):
        require(all(sha(safe(root, file)) == digest for file, digest in qa[group].items()),
            'No final document or reviewed evidence may change after QA')
    require(utc(verified_time) <= utc(qa['checked_at_utc']) <= datetime.now(timezone.utc),
        'Exact checked_at_utc final QA key and chronology required')
    return name, qa


def main():
    out = ROOT / 'results/followup_closure_archive_validation.json'
    require(not out.exists(), 'Exclusive new bounded project closure archive receipt')
    baseline_name = 'results/followup_closure_baseline_preservation.json'
    validation_name = 'results/followup_closure_prefreeze_validation.json'
    registration_name = 'results/registration_followup_closure_v1.json'
    publication_name = 'results/followup_closure_publication_proof.json'
    baseline = read(safe(ROOT, baseline_name))
    require(baseline.get('status') == 'passed' and baseline.get('baseline_tag') == BASE_TAG
        and baseline.get('baseline_commit') == BASE_COMMIT
        and baseline.get('hash_compared_files') == 19206 and baseline.get('hash_compared_bytes') == 2528410365
        and baseline.get('allowed_evolving_indexes') == sorted(INDEXES)
        and baseline.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True,
        'Exact successful completed-baseline Git-blob comparison receipt')
    old = set(git(ROOT, 'ls-tree', '-r', '--name-only', BASE_COMMIT).decode().splitlines())
    changed = set(git(ROOT, 'diff', '--name-only', BASE_COMMIT).decode().splitlines())
    require(old & changed <= INDEXES, 'Completed original code, settings, data or results changed')
    folder = ROOT / RESULT
    cfg, reg, meta = verify_envelope(ROOT, ROOT / 'configs/followup_closure_v1.json',
        ROOT / registration_name, folder, ROOT / publication_name)
    require(len(reg['source_files']) == 35 and len(reg['parent_files']) == 253,
        'Exact 35-source, 253-parent bounded closure inventory')
    summary = read(folder / 'summary.json'); verified = read(safe(ROOT, CORRECTION_VERIFICATION))
    independent_gate(verified, summary, ROOT, registration_name)
    correction = consumer_correction_gate(ROOT, reg, verified)
    require(meta.get('counts') == summary['counts'] and meta.get('numeric_arrays') == 2
        and meta.get('measurement_records') == 24, 'Exact metadata and checked synthesis counts')
    with np.load(folder / 'arrays.npz', allow_pickle=False) as arrays:
        require(set(arrays.files) == {'study_integrity_flags', 'claim_ledger'}
            and arrays['study_integrity_flags'].dtype.kind == 'b'
            and arrays['study_integrity_flags'].shape == (17, 8) and arrays['study_integrity_flags'].all()
            and arrays['claim_ledger'].dtype == np.dtype('int64') and arrays['claim_ledger'].shape == (24, 4)
            and np.array_equal(arrays['claim_ledger'][:, 0], np.arange(24)),
            'Two exact integer/bool ledgers with every study integrity coordinate retained')
    with (folder / 'measurements.csv').open(encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = ('claim_id', 'question', 'level', 'verdict', 'statement', 'interpretation_limit',
            'source_count', 'evidence_json', 'source_sha256_json', 'matched')
        require(tuple(reader.fieldnames or ()) == columns and len(list(reader)) == 24,
            'Exact twenty-four claim rows with ten ordered columns')
    tests = read(safe(ROOT, validation_name))
    prepaths, classes, development, zip_bindings = prefreeze_gate(ROOT, tests, reg, baseline_name, validation_name)
    publication = read(safe(ROOT, publication_name))
    repro = saved_reproduction_gate(ROOT, cfg, summary, publication['checked_at_utc'], meta['started_at_utc'])
    chronology = [baseline['verified_at_utc'], tests['checked_at_utc'], reg['registered_at_utc'],
        publication['checked_at_utc'], read(safe(ROOT, REPRO))['verified_at_utc'],
        meta['started_at_utc'], meta['completed_at_utc'], correction['failure']['captured_at_utc'],
        correction['tests']['checked_at_utc'], correction['registration']['registered_at_utc'],
        correction['publication']['checked_at_utc'], verified['verified_at_utc']]
    require(list(map(utc, chronology)) == sorted(map(utc, chronology))
        and utc(chronology[2]) < utc(chronology[3]) < utc(chronology[4])
        and utc(chronology[-1]) <= datetime.now(timezone.utc),
        'Original freeze/publication/recheck/production, failure capture, extra test freeze, correction registration/publication and verification UTC sequence')
    presentation_name, qa = presentation_gate(ROOT, verified['verified_at_utc'], correction['test_paths'])
    chronology.append(qa['checked_at_utc'])
    status = read(ROOT / 'docs/repository-status.json')
    require(status.get('followup_closure_completed') is True and status.get('bounded_project_completed') is True
        and status.get('current_phase') == 'project_completed_bounded_scope'
        and status.get('current_evidence_synthesis') == STUDY
        and status.get('current_findings_document') == 'docs/followup-closure-final-report.md'
        and status.get('phase5_completed') is True
        and status['continuation_automation']['status'] == 'PAUSED'
        and status['continuation_automation']['automatic_continuation_enabled'] is False
        and all(status.get(k) is False for k in ('level4_similarity_supported', 'shared_neural_mechanism_verified',
            'education_effect_tested')), 'Bounded scope completed and automatic continuation off; no stronger claim')
    paths = set(reg['source_files']) | set(reg['parent_files']) | set(DOCS) | prepaths | {
        registration_name, publication_name, presentation_name, REPRO,
        'scripts/check_followup_closure_archive.py', 'scripts/check_followup_closure_baseline.py'}
    paths.update(qa['additional_file_sha256'])
    paths.update(correction['paths'])
    paths.update(path.relative_to(ROOT).as_posix() for path in folder.iterdir() if path.is_file())
    write_new(out, {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'study_id': STUDY, 'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT,
        'baseline_hash_receipt_sha256': sha(safe(ROOT, baseline_name)),
        'hash_compared_files': baseline['hash_compared_files'], 'hash_compared_bytes': baseline['hash_compared_bytes'],
        'subsequent_old_file_changes': sorted(old & changed), 'allowed_evolving_indexes': sorted(INDEXES),
        'all_old_files_Git_blob_hash_preserved_except_three_indexes': True,
        'source_git_commit': reg['source_git_commit'], 'registration_sha256': sha(safe(ROOT, registration_name)),
        'prefreeze_receipts_committed_source_sha256': {name: sha(safe(ROOT, name)) for name in sorted(prepaths)},
        'prefreeze_passed_unique_tests_independently_counted': sum(classes.values()),
        'prefreeze_unique_tests_by_class': classes,
        'additional_consumer_prefreeze_passed_unique_tests_independently_counted': correction['unique_additional_consumer_cases'],
        'additional_consumer_tests_not_combined_with_original_267': True,
        'additional_consumer_prefreeze_receipt_sha256': sha(safe(ROOT, CORRECTION_PREFREEZE)),
        'prefreeze_publication_reproduction_synthesis_verification_utc_order': chronology,
        'utc_order_labels': ['baseline_preservation', 'original_prefreeze', 'original_registration',
            'original_private_publication', 'saved_only_reverification', 'original_synthesis_started',
            'original_synthesis_completed', 'original_consumer_failure_captured', 'correction_prefreeze',
            'correction_registration', 'correction_private_publication', 'corrected_independent_verification',
            'final_eight_document_QA'],
        'development_failure_artifact_bytes_preserved': development,
        'development_ZIP_manifests_and_original_member_hashes': zip_bindings,
        'saved_only_reproduction': repro, 'counts': summary['counts'],
        'scientific_verification_path': CORRECTION_VERIFICATION,
        'scientific_verification_sha256': sha(safe(ROOT, CORRECTION_VERIFICATION)),
        'original_failed_verification_path_remains_absent': True,
        'original_consumer_failure_sha256': sha(safe(ROOT, FIRST_FAILURE)),
        'original_consumer_failure_ZIP_sha256': sha(safe(ROOT, FIRST_FAILURE_ZIP)),
        'consumer_correction': correction['consumer_correction'],
        'human_original_identity_checked_without_invented_field': correction['human_original_identity'],
        'presentation_validation_sha256': sha(safe(ROOT, presentation_name)),
        'current_file_sha256': {name: sha(safe(ROOT, name)) for name in sorted(paths)},
        'method': 'Completed baseline Git-blob comparison; immutable tag/source/input/outcome chain; original 267 source-committed invented tests plus separately counted consumer cases; one fresh saved-only numerical recheck; original real consumer failure preserved; separate published original-schema consumer correction with all other independent checks unchanged; all eight final documents. No original learning, cohorts, graph, RNG, direction optimization or new effect tests.',
        'notice': 'Current authorized scope is complete. The original first Phase5, negative hypotheses, unresolved gates, all failures and long-term biological/human questions remain preserved; completion is not successful mathematical learning or common mechanism confirmation.'})
    print('Followup closure archive passed: original evidence preserved, bounded project completed, continuation off.')


if __name__ == '__main__':main()
