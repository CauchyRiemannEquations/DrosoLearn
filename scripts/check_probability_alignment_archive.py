"""Seal saved-state probability alignment and its positive optimization workload.

This checks completed byte receipts and arithmetic verdicts. It does not replay
the parent learner, repeat the new optimizations, or rerun the saved analysis.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from drosolearn.probability_envelope import (
    BASE_COMMIT, BASE_TAG, EVOLVING_INDEXES, ancestor, committed, read,
    require, safe, sha, utc, verify_envelope, write_new,
)

PREFIX = 'probability_alignment'
STUDY = 'probability_alignment_v1'
NEW_CLASSES = frozenset({'tests.test_probability_alignment',
    'tests.test_probability_alignment_envelope', 'tests.test_probability_alignment_verify'})
OLD_CLASSES = frozenset({'tests.test_cone_geometry', 'tests.test_self_blend_learning'})
FIGURES = frozenset({'primary-gains.png', 'state-weighting-by-seed.png', 'all-conditions.png'})


def archive_development_gate(root):
    """Keep post-freeze archive-tool diagnostics apart from scientific failures."""
    directory = root / 'results/probability_alignment_archive_development'
    names = {'results/probability_alignment_archive_development/archive_tool_smoke_attempt01' + ext
        for ext in ('.zip', '.json')}
    require(directory.is_dir() and {p.relative_to(root).as_posix() for p in directory.iterdir() if p.is_file()} == names,
            'Exact separate archive-tool development ZIP and manifest required')
    note_name = 'results/probability_alignment_archive_development/archive_tool_smoke_attempt01.json'
    note = read(safe(root, note_name))
    require(note.get('status') == 'preserved_archive_tool_development'
            and note.get('archive') in names and note['archive'].endswith('.zip')
            and note.get('archive_sha256') == sha(safe(root, note['archive']))
            and note.get('final_smoke_passed_checks') == 6
            and note.get('original_first_draft_failure_preserved') is True
            and note.get('fresh_original_draft_synthetic_failure_reproduction_is_explicitly_labeled') is True
            and note.get('frozen_scientific_development_directory_untouched') is True
            and all(note.get(key) is False for key in ('registered_scientific_failure',
                'scientific_registered_source_config_thresholds_or_parent_inputs_changed', 'archive_checker_main_executed'))
            and all(type(note.get(key)) is int and note[key] == 0 for key in (
                'actual_saved_parent_aggregation_calls', 'actual_new_direction_optimization_calls')),
            'Archive-tool development must remain separate from registered scientific work')
    expected = note.get('member_sha256')
    require(type(expected) is dict and len(expected) == 7, 'All seven original archive development evidence members required')
    with ZipFile(safe(root, note['archive'])) as archive:
        require(len(archive.namelist()) == 7 and set(archive.namelist()) == set(expected)
                and all(hashlib.sha256(archive.read(name)).hexdigest() == digest for name, digest in expected.items()),
                'Original archive-tool draft, diagnostic and passing receipt bytes changed')
        passed = json.loads(archive.read('probability_alignment_archive_fixture_validation.json'))
        diagnostic = json.loads(archive.read('archive_tool_development_diagnostics.json'))
        require(passed.get('status') == 'passed' and len(passed.get('checks', [])) == 6
                and len(set(passed['checks'])) == 6 and passed.get('checker_main_executed') is False
                and passed.get('synthetic_receipt_checks_only') is True
                and type(passed.get('actual_parent_aggregation_or_optimization_calls')) is int
                and passed['actual_parent_aggregation_or_optimization_calls'] == 0
                and diagnostic.get('first_draft_failure_reproduced_exit_code') != 0
                and diagnostic.get('registered_scientific_failure') is False
                and b'Existing contained file required: results/probability_alignment_development'
                    in archive.read('first_draft_synthetic_failure_reproduced_stderr.bin'),
                'Six passing invented receipt checks and explicitly labeled original-draft failure reproduction required')
    require(utc(passed['checked_at_utc']) <= utc(note['created_at_utc']) <= datetime.now(timezone.utc),
            'Archive-tool evidence preservation chronology required')
    return names, {'archive': note['archive'], 'archive_sha256': note['archive_sha256'],
        'manifest': note_name, 'manifest_sha256': sha(safe(root, note_name)),
        'original_member_sha256': expected, 'registered_scientific_failure': False,
        'six_final_synthetic_receipt_checks_passed': True,
        'original_draft_and_explicit_reproduced_failure_preserved': True,
        'no_original_stderr_file_falsely_claimed_for_transcribed_diagnostics': True}


def prefreeze_gate(root, tests, reg, baseline_name, validation_name):
    """Count the unique final cases and retain every original development byte."""
    require(tests.get('status') == 'passed' and tests.get('frozen_source_hashes') == reg['source_files']
            and tests.get('parent_byte_gate') == reg['gate']
            and tests.get('synthetic_fixtures_only_before_freeze') is True,
            'Exact frozen source, inherited byte gate and synthetic-only snapshot required')
    require(all(type(tests.get(key)) is int and tests[key] == 0 for key in (
                'actual_new_saved_alignment_aggregation_before_freeze',
                'actual_new_direction_optimizations_before_freeze',
                'new_training_random_graph_calls_before_freeze')),
            'No new actual aggregation, optimization or learning before freeze required')
    receipts = tests.get('test_receipts'); development = tests.get('development_failure_artifacts')
    require(type(receipts) is list and len(receipts) == 1
            and receipts[0]['path'].startswith('results/probability_alignment_prefreeze_tests/')
            and receipts[0]['path'].endswith('.xml') and type(development) is list,
            'One unique final suite and complete original development inventory required')
    require(all(type(item) is dict and type(item.get('path')) is str for item in receipts + development)
            and len({item['path'] for item in receipts + development}) == len(receipts + development),
            'Unique prefreeze evidence locations required')
    prepaths = {validation_name, baseline_name}
    for item in receipts + development:
        require(sha(safe(root, item['path'])) == item['sha256'], 'Original prefreeze evidence bytes changed')
        prepaths.add(item['path'])
    development_names = {item['path'] for item in development}
    directory = (root / 'results/probability_alignment_development').resolve()
    require(directory.is_relative_to(root) and directory.is_dir()
            and {p.relative_to(root).as_posix() for p in directory.iterdir() if p.is_file()}
            == development_names, 'Every original development artifact must be frozen, without hidden omissions')
    cases = ET.parse(safe(root, receipts[0]['path'])).findall('.//testcase')
    identities = {(case.get('classname'), case.get('name')) for case in cases}
    require(bool(cases) and len(cases) == len(identities)
            and all(case.get('classname') and case.get('name') for case in cases)
            and all(case.find(kind) is None for case in cases for kind in ('failure', 'error', 'skipped')),
            'Every unique final prefreeze case must pass')
    classes = dict(sorted(Counter(case.get('classname') for case in cases).items()))
    new_count = sum(classes.get(name, 0) for name in NEW_CLASSES)
    old_count = sum(classes.get(name, 0) for name in OLD_CLASSES)
    require(set(classes) == NEW_CLASSES | OLD_CLASSES
            and classes == tests.get('unique_passed_tests_by_class')
            and len(cases) == tests.get('unique_passed_tests') == receipts[0]['passed_unique_tests']
            and tests.get('new_unique_passed_tests') == new_count
            and tests.get('unchanged_related_regression_passed_tests') == old_count
            and len(cases) == new_count + old_count,
            'Exact actual JUnit class partition and dynamic counts required')
    committed(root, reg['source_git_commit'], {name: sha(safe(root, name)) for name in sorted(prepaths)})
    zip_bindings = {}
    for name in sorted(development_names):
        if not name.endswith('.json'):
            continue
        note = read(safe(root, name))
        if 'archive' not in note:
            continue
        archive_name = note['archive']
        if '/' not in archive_name and '\\' not in archive_name:
            archive_name = 'results/probability_alignment_development/' + archive_name
        require(archive_name in development_names and archive_name.endswith('.zip')
                and note.get('archive_sha256') == sha(safe(root, archive_name)),
                'Development ZIP must be tied to its original frozen manifest')
        expected = note.get('member_sha256')
        require(type(expected) is dict and bool(expected), 'Exact raw development ZIP member hashes required')
        normalized = {member.replace('\\', '/'): digest for member, digest in expected.items()}
        require(len(normalized) == len(expected), 'Manifest path-separator aliases must not hide duplicate ZIP members')
        with ZipFile(safe(root, archive_name)) as archive:
            require(len(archive.namelist()) == len(normalized) and set(archive.namelist()) == set(normalized),
                    'Development ZIP unique member inventory changed')
            require(all(hashlib.sha256(archive.read(member)).hexdigest() == digest
                        for member, digest in normalized.items()), 'Original development ZIP member bytes changed')
            failures = []
            for member in normalized:
                if member.endswith('.xml'):
                    original = ET.fromstring(archive.read(member)).findall('.//testcase')
                    failures.extend(case for case in original if case.find('failure') is not None or case.find('error') is not None)
            if 'failed_test_count' in note:
                require(type(note['failed_test_count']) is int and note['failed_test_count'] == len(failures),
                        'Original development failure count must match preserved raw JUnit')
        zip_bindings[archive_name] = {'manifest': name, 'original_manifest_member_sha256': expected,
            'ZIP_member_sha256': normalized,
            'manifest_windows_member_separators_normalized_to_ZIP': expected != normalized,
            'original_failing_cases': len(failures), 'raw_member_bytes_checked': True}
    require({name for name in development_names if name.endswith('.zip')} == set(zip_bindings),
            'No original development ZIP may be left without an exact manifest binding')
    return prepaths, classes, development, zip_bindings


def independent_gate(verified, summary, cfg):
    """Bind the independent completed verdict without evaluating another case."""
    require(verified.get('status') == 'passed' and verified.get('study_id') == STUDY
            and verified.get('scientific_producer_imported') is False
            and verified.get('new_helper_imported') is False,
            'Independent probability arithmetic without a scientific producer/helper import required')
    counts = summary['counts']
    checks = verified.get('counts', {})
    expected_counts = {'numeric_arrays_checked': 27, 'saved_reference_records_checked': 24000,
        'saved_reference_destination_effects_checked': 96000, 'measurement_records_checked': 900,
        'eligible_actual_Go_records_checked': counts['eligible_actual_Go_records'],
        'candidate_destination_effects_checked': counts['candidate_destination_effects'],
        'independent_ambient_support_reoptimizations': counts['new_optimization_calls'],
        'independent_legacy_support_reoptimizations': counts['unique_uniform_optimizations'],
        'independent_active_faces_per_support': 16, 'new_training_trials': 0,
        'new_random_draws': 0, 'fresh_model_constructions': 0}
    require(all(type(checks.get(key)) is int and checks[key] == value for key, value in expected_counts.items())
            and all(checks.get(key) is True for key in (
                'head_prefixes_and_factual_scalar_effects_checked',
                'own_prefeedback_observed_signs_and_full_gate_checked',
                'all_five_same_state_signed_updates_checked',
                'uniform_and_probability_global_support_independently_reoptimized',
                'legacy_balanced_global_support_independently_reoptimized',
                'signed_finite_probability_failures_and_null_means_retained'))
            and checks.get('scientific_producer_imported') is False
            and checks.get('new_cone_solver_imported') is False,
            'Complete independent state/effect replay and positive U/P ambient reoptimization receipt required')
    require(counts['numeric_arrays'] == 27 and counts['measurement_records'] == 900
            and counts['saved_reference_records'] == 24000 and counts['saved_reference_destination_effects'] == 96000
            and counts['candidate_destination_effects'] == 20 * counts['eligible_actual_Go_records']
            and counts['candidate_single_update_evaluations'] == 5 * counts['eligible_actual_Go_records']
            and counts['new_optimization_calls'] == counts['unique_uniform_optimizations']
                + counts['state_probability_optimizations'] > 0,
            'Actual U/P optimization work must be positive and separate from zero new learning')
    validation = summary.get('validation', {})
    require(validation.get('all_numeric_information_and_support_checks_passed') is True
            and validation.get('new_objectives_use_own_observed_signs_only') is True
            and validation.get('same_factual_state_action_reward_and_scalar') is True
            and validation.get('all_eligible_logits_strictly_inside_clipping_domain') is True
            and type(validation.get('excluded_partial_and_NoGo_solver_calls')) is int
            and validation['excluded_partial_and_NoGo_solver_calls'] == 0,
            'Same saved state, observed-only objectives, clipping and excluded-work checks required')
    residuals = validation.get('residuals', {})
    expected = {'original_balanced_finite_gain', 'legacy_global_support', 'uniform_global_support',
        'probability_tangent_global_support', 'unit_norm', 'single_step_equal_cost',
        'observed_known_margin_safety', 'case_certificate_residual', 'probability_tangent_identity'}
    require(set(residuals) == expected and all(type(value) in (int, float)
            and 0 <= value <= cfg['numeric_tolerance'] for value in residuals.values()),
            'Every frozen numeric support and probability-tangent check must pass')
    interpretation = summary.get('interpretation', {})
    require(all(interpretation.get(key) is False for key in (
        'new_confirmatory_hypothesis', 'finite_probability_global_optimum_claimed',
        'summed_fixed_state_gains_predict_new_trajectory', 'actual_future_policy_relearned',
        'processing_cost_equalized', 'parent_H_gain', 'parent_H_acquisition',
        'novel_input_generalization_tested', 'new_human_experiment',
        'shared_neural_mechanism_claim', 'education_effect_claim'))
        and all(interpretation.get(key) is True for key in ('retrospective_descriptive_only',
        'tangent_global_maximum_is_expected_integrity_property',
        'state_weighting_effect_separated_from_uniform_objective_control', 'parent_H_protection')),
        'Saved one-step probability analysis cannot be relabeled as new learned performance or finite global optimality')
    require(summary.get('finite_alignment_summary', {}).get('new_learned_performance_claim') is False
            and len(summary.get('families', [])) == 30
            and all(len(family['seeds']) == 10 for family in summary['families']),
            'All thirty conditions and three hundred repeated seed records must remain descriptive')


def main():
    out = ROOT / 'results/probability_alignment_archive_validation.json'
    require(not out.exists(), 'New exclusive probability-alignment archive receipt required')
    baseline_name = 'results/probability_alignment_baseline_preservation.json'
    validation_name = 'results/probability_alignment_prefreeze_validation.json'
    registration_name = 'results/registration_probability_alignment_v1.json'
    publication_name = 'results/probability_alignment_publication_proof.json'
    folder = ROOT / 'results/probability_alignment_v1'
    figures = ROOT / 'results/probability_alignment_figures_v1'
    baseline = read(ROOT / baseline_name)
    require(baseline.get('status') == 'passed' and baseline.get('baseline_tag') == BASE_TAG
            and baseline.get('baseline_commit') == BASE_COMMIT
            and baseline.get('allowed_evolving_indexes') == sorted(EVOLVING_INDEXES)
            and baseline.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True,
            'Exact three-index completed raw Git-blob baseline receipt required')
    old = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE_COMMIT], cwd=ROOT).decode().splitlines())
    changed = set(subprocess.check_output(['git', 'diff', '--name-only', BASE_COMMIT], cwd=ROOT).decode().splitlines())
    require(old & changed <= EVOLVING_INDEXES, 'A completed historical file changed')
    cfg, reg, meta = verify_envelope(ROOT, ROOT / 'configs/probability_alignment_v1.json',
        ROOT / registration_name, folder, ROOT / publication_name)
    require(len(reg['source_files']) == 24 and len(reg['parent_files']) == 17,
            'Exact source24/parent17 closure required')
    ancestor(ROOT, BASE_COMMIT, reg['source_git_commit'])
    verified = read(folder / 'verification.json'); summary = read(folder / 'summary.json')
    require(verified.get('summary_sha256') == sha(folder / 'summary.json')
            and verified.get('registration_sha256') == sha(ROOT / registration_name),
            'Independent verdict must bind exact current summary and registration bytes')
    independent_gate(verified, summary, cfg)
    with (folder / 'measurements.csv').open(encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)) == 17,
                'Exact seventeen unique measurement fields required')
        require(sum(1 for _ in reader) == 900, 'All nine hundred frozen measurement rows required')
    tests = read(ROOT / validation_name)
    prepaths, classes, development, zip_bindings = prefreeze_gate(ROOT, tests, reg, baseline_name, validation_name)
    archive_development_names, archive_development = archive_development_gate(ROOT)
    publication = read(ROOT / publication_name)
    chronology = [baseline['verified_at_utc'], tests['checked_at_utc'], reg['registered_at_utc'],
        publication['checked_at_utc'], meta['started_at_utc'], meta['completed_at_utc'], verified['verified_at_utc']]
    times = list(map(utc, chronology))
    require(times == sorted(times) and times[2] < times[3] < times[4]
            and times[-1] <= datetime.now(timezone.utc),
            'Baseline/freeze/registration/private-publication/analysis/independent-verification UTC chronology required')
    provenance = read(figures / 'provenance.json')
    require(provenance.get('study_id') == STUDY
            and provenance.get('renderer') == 'scripts/render_probability_alignment.py'
            and provenance.get('renderer_sha256') == sha(ROOT / provenance['renderer'])
            and provenance.get('input_sha256') == {'results/probability_alignment_v1/' + name: sha(folder / name)
                for name in ('summary.json', 'verification.json')}
            and set(provenance.get('output_sha256', {})) == FIGURES | {'plotted_values.json'}
            and all(sha(figures / name) == digest for name, digest in provenance['output_sha256'].items())
            and type(provenance.get('new_learning_calls')) is int and provenance['new_learning_calls'] == 0
            and type(provenance.get('new_optimization_calls')) is int and provenance['new_optimization_calls'] == 0,
            'Renderer-only zero work, all exact figures and scientific plotted bytes must be bound')
    plot = read(figures / 'plotted_values.json')
    require(plot.get('all_families') == summary['families']
            and plot.get('finite_alignment_summary') == summary['finite_alignment_summary']
            and plot.get('learned_trajectory_or_final_accuracy') is False
            and plot.get('equal_seed_means') is True,
            'Every plotted value must be a verified descriptive fixed-state value')
    status = read(ROOT / 'docs/repository-status.json')
    require(status.get('probability_alignment_completed') is True
            and status['continuation_automation']['status'] == 'PAUSED'
            and status['current_findings_document'] == 'docs/probability-alignment-findings.md'
            and status['level4_similarity_supported'] is status['shared_neural_mechanism_verified'] is False,
            'Completed descriptive report, bounded interpretation and paused continuation required')
    presentation_name = 'results/probability_alignment_presentation_validation.json'
    presentation = read(ROOT / presentation_name)
    require(presentation.get('status') == 'passed', 'Independent final presentation QA required')
    required_docs = {'docs/probability-alignment-findings.md', 'docs/probability-alignment-reproduction.md'} | EVOLVING_INDEXES
    require(required_docs <= set(presentation['document_sha256']) | set(presentation['additional_file_sha256']),
            'All final report, reproduction and evolving-index bytes must be presentation-checked')
    required_additional = {'scripts/render_probability_alignment.py',
        'results/probability_alignment_figures_v1/plotted_values.json',
        'results/probability_alignment_figures_v1/provenance.json'} | {
        'results/probability_alignment_figures_v1/' + name for name in FIGURES}
    require(required_additional <= set(presentation['additional_file_sha256']),
            'All three figures, renderer and provenance must be presentation-checked')
    for key in ('document_sha256', 'additional_file_sha256'):
        require(all(sha(safe(ROOT, name)) == digest for name, digest in presentation[key].items()),
                'Final presentation evidence bytes changed')
    require(utc(verified['verified_at_utc']) <= utc(presentation['exactchecked_at_utc']) <= datetime.now(timezone.utc),
            'Independent scientific verification must precede final presentation QA')
    glue_name = 'results/probability_alignment_archive_finalization_attempt01.json'
    glue = read(ROOT / glue_name)
    require(glue['scientific_registered_source_config_inputs_thresholds_results_or_verification_changed'] is False
            and glue['existing_presentation_receipt_not_rewritten'] is True
            and glue['actual_learning_or_direction_optimization_repeated'] is False
            and glue['archive_sha256'] == sha(ROOT / glue['archive']),
            'Archive consumer timestamp correction must preserve original bytes without science changes')
    with ZipFile(ROOT / glue['archive']) as original:
        require(set(original.namelist()) == set(glue['member_sha256'])
                and all(hashlib.sha256(original.read(name)).hexdigest() == digest
                    for name, digest in glue['member_sha256'].items())
                and original.read(presentation_name) == (ROOT / presentation_name).read_bytes(),
                'Original archive attempt and unchanged independent QA bytes must remain preserved')
    paths = set(reg['source_files']) | set(reg['parent_files']) | EVOLVING_INDEXES | prepaths | archive_development_names | {
        registration_name, publication_name, presentation_name,
        'docs/probability-alignment-findings.md', 'docs/probability-alignment-reproduction.md',
        'scripts/check_probability_alignment_archive.py', 'scripts/check_probability_alignment_baseline.py',
        'scripts/render_probability_alignment.py'}
    paths |= set(presentation['document_sha256']) | set(presentation['additional_file_sha256'])
    paths |= {glue_name, glue['archive']}
    for directory in (folder, figures):
        paths |= {path.relative_to(ROOT).as_posix() for path in directory.iterdir() if path.is_file()}
    write_new(out, {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT,
        'baseline_hash_receipt_sha256': sha(ROOT / baseline_name),
        'hash_compared_files': baseline['hash_compared_files'], 'hash_compared_bytes': baseline['hash_compared_bytes'],
        'subsequent_old_file_changes': sorted(old & changed), 'allowed_evolving_indexes': sorted(EVOLVING_INDEXES),
        'all_old_files_Git_blob_hash_preserved_except_three_indexes': True,
        'source_git_commit': reg['source_git_commit'], 'registration_sha256': sha(ROOT / registration_name),
        'prefreeze_receipts_committed_source_sha256': {name: sha(ROOT / name) for name in sorted(prepaths)},
        'prefreeze_passed_unique_tests_independently_counted': sum(classes.values()),
        'prefreeze_unique_tests_by_class': classes,
        'prefreeze_publication_analysis_verification_utc_order': chronology,
        'development_failure_artifact_bytes_preserved': development,
        'development_ZIP_manifests_and_original_member_hashes': zip_bindings,
        'separate_archive_tool_development_preservation': archive_development,
        'counts': summary['counts'], 'scientific_verification_sha256': sha(folder / 'verification.json'),
        'presentation_validation_sha256': sha(ROOT / presentation_name),
        'archive_consumer_timestamp_finalization': glue,
        'current_file_sha256': {name: sha(safe(ROOT, name)) for name in sorted(paths)},
        'method': 'Prior Git-blob receipt, subsequent old-path diff, source-committed synthetic test/development bytes, independent arithmetic and final presentation snapshots; no repeated learning or optimization in this checker.',
        'notice': 'The registered saved-state analysis performed positive U/P direction optimizations. It estimates single-step effects at fixed past states, not a new learned trajectory, finite global probability optimum, or shared biological mechanism.'})
    print('Probability-alignment archive passed: fixed-state effects, positive solver work and final QA bytes sealed.')


if __name__ == '__main__':
    main()
