"""Seal a completed self-blend study; never rerun learning or scientific replay."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from drosolearn.self_blend_envelope import (
    BASE_COMMIT, BASE_TAG, EVOLVING_INDEXES, committed, read, require, safe,
    sha, utc, verify_envelope, write_new,
)

DEVELOPMENT = frozenset({
    'results/self_blend_development/failure_01.zip',
    'results/self_blend_development/failure_01.json',
    'results/self_blend_development/core_failure_01.zip',
    'results/self_blend_development/core_failure_01_manifest.json',
})
NEW_CLASSES = frozenset({'tests.test_self_blend_learning', 'tests.test_self_blend_envelope',
                         'tests.test_self_blend_learning_verify'})
OLD_CLASSES = frozenset({'tests.test_gated_learning', 'tests.test_balanced_learning'})
FIGURES = frozenset({'learning-curves.png', 'all-conditions.png', 'unit-direction-and-pathways.png'})


def passing_junit(path):
    cases = ET.parse(path).findall('.//testcase')
    require(bool(cases) and len(cases) == len({(case.get('classname'), case.get('name')) for case in cases})
            and all(case.find(key) is None for case in cases for key in ('failure', 'error', 'skipped')),
            'Complete unique passing prefreeze cases required')
    return Counter(case.get('classname') for case in cases)


def preserved_zip(root, archive_name, expected_members, expected_total, expected_failed):
    """Hash raw ZIP members without extracting or altering original failure bytes."""
    require(type(expected_members) is dict and bool(expected_members), 'Original ZIP member hashes required')
    with ZipFile(safe(root, archive_name)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(expected_members)
                and not any(info.is_dir() for info in archive.infolist()), 'Exact unique raw development ZIP members required')
        for name in names:
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected_members[name],
                    'Original failed member bytes differ: ' + name)
        xml_names = [name for name in names if name.endswith('.xml')]
        require(len(xml_names) == 1, 'One original raw failed JUnit per development ZIP required')
        cases = ET.fromstring(archive.read(xml_names[0])).findall('.//testcase')
    failed = sum(case.find('failure') is not None for case in cases)
    require(len(cases) == expected_total and failed == expected_failed
            and all(case.find(key) is None for case in cases for key in ('error', 'skipped'))
            and sum(case.find('failure') is None for case in cases) == expected_total - expected_failed,
            'Original development failure and pass counts differ')
    return {'archive': archive_name, 'sha256': sha(safe(root, archive_name)),
            'member_sha256': expected_members, 'tests': len(cases), 'failed': failed}


def main():
    out = ROOT / 'results/self_blend_archive_validation.json'
    require(not out.exists(), 'New exclusive self-blend archive receipt required')
    baseline_name = 'results/self_blend_baseline_preservation.json'
    baseline = read(ROOT / baseline_name)
    require(baseline.get('status') == 'passed' and baseline.get('baseline_tag') == BASE_TAG
            and baseline.get('baseline_commit') == BASE_COMMIT
            and baseline.get('allowed_evolving_indexes') == sorted(EVOLVING_INDEXES)
            and baseline.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True,
            'Exact three-index original raw Git-blob baseline receipt required')
    old = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE_COMMIT], cwd=ROOT).decode().splitlines())
    changed = set(subprocess.check_output(['git', 'diff', '--name-only', BASE_COMMIT], cwd=ROOT).decode().splitlines())
    require(old & changed <= EVOLVING_INDEXES, 'A previously completed file changed')
    folder = ROOT / 'results/self_blend_learning_v1'
    figures = ROOT / 'results/self_blend_figures_v1'
    registration_name = 'results/registration_self_blend_learning_v1.json'
    publication_name = 'results/self_blend_learning_publication_proof.json'
    cfg, reg, meta = verify_envelope(ROOT, ROOT / 'configs/self_blend_learning_v1.json',
        ROOT / registration_name, folder, ROOT / publication_name)
    require(len(reg['source_files']) == 44 and len(reg['parent_files']) == 11
            and len(reg['graph_files']) == 4 and len(reg['input_files']) == 2,
            'Exact prospective source/parent/graph/input closure required')
    verified = read(folder / 'verification.json'); summary = read(folder / 'summary.json')
    require(verified.get('status') == 'passed' and verified.get('study_id') == 'self_blend_learning_v1'
            and verified.get('scientific_producer_imported') is False and verified.get('new_helper_imported') is False
            and verified.get('summary_sha256') == sha(folder / 'summary.json')
            and verified.get('registration_sha256') == sha(ROOT / registration_name),
            'Independent scientific receipt must bind exact completed bytes')
    vc = verified.get('counts', {}); counts = summary['counts']
    require(all(type(vc.get(key)) is int and vc[key] == value for key, value in {
        'arrays': 201, 'independently_replayed_training_actions': 144000, 'model_seeds': 10,
        'conditions': 36, 'checkpoint_static_probes': 7200, 'head_update_calls': 144000,
        'internal_update_calls': 0, 'value_update_calls': 0}.items())
        and vc.get('unique_balanced_states_independently_reoptimized') == counts['balanced_unique_solver_cases'],
        'Independent full actual learning and solver-case work counts required')
    replay_flags = ('all_fresh_autonomous_choices_rewards_and_heads_replayed',
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
    require(all(vc.get(key) is True for key in replay_flags)
            and vc.get('extra_optimizer_and_memory_processing_cost_equalized') is False
            and vc.get('new_generalization_or_human_mechanism_claim') is False,
            'Complete bounded independent autonomous replay required')
    require(summary['validation']['all_numeric_and_information_gates_passed'] is True
            and all(value is True for value in summary['gate_validation'].values())
            and all(value is True for key, value in summary['blend_validation'].items() if isinstance(value, bool)),
            'Numeric, observed-gate and blend checks must all pass')
    require(all(type(summary[name]['supported']) is bool for name in ('H_gain', 'H_acquisition', 'H_protection')),
            'All three separate scientific decisions must be preserved')
    with (folder / 'measurements.csv').open(encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)) == 87,
                'Exact 87 unique actual-trial CSV fields required')
        require(sum(1 for _ in reader) == 144000, 'Complete actual trial CSV required')
    validation_name = 'results/self_blend_prefreeze_validation.json'
    tests = read(ROOT / validation_name)
    require(tests.get('status') == 'passed' and tests.get('frozen_source_hashes') == reg['source_files']
            and tests.get('parent_byte_gate') == reg['gate']
            and type(tests.get('actual_new_training_before_freeze')) is int
            and tests['actual_new_training_before_freeze'] == 0
            and type(tests.get('new_cohort_graph_feature_rng_calls_before_freeze')) is int
            and tests['new_cohort_graph_feature_rng_calls_before_freeze'] == 0
            and tests.get('synthetic_fixtures_only_before_freeze') is True,
            'Exact source-committed prefreeze snapshot with no premature cohort work required')
    receipts = tests.get('test_receipts'); development = tests.get('development_failure_artifacts')
    require(type(receipts) is list and len(receipts) == 1
            and receipts[0]['path'].startswith('results/self_blend_prefreeze_tests/')
            and receipts[0]['path'].endswith('.xml'), 'One original final prefreeze JUnit required')
    require(type(development) is list and len(development) == 4
            and {item['path'] for item in development} == DEVELOPMENT,
            'Both original failure ZIPs and their exact two manifests required')
    prepaths = {validation_name, baseline_name}
    for item in receipts + development:
        require(sha(safe(ROOT, item['path'])) == item['sha256'], 'Prefreeze or failure receipt changed')
        prepaths.add(item['path'])
    committed(ROOT, reg['source_git_commit'], {name: sha(ROOT / name) for name in sorted(prepaths)})
    classes = passing_junit(ROOT / receipts[0]['path'])
    total = sum(classes.values()); new_count = sum(classes[key] for key in NEW_CLASSES)
    old_count = sum(classes[key] for key in OLD_CLASSES)
    require(set(classes) == NEW_CLASSES | OLD_CLASSES
            and type(tests.get('unique_passed_tests')) is int and total == tests['unique_passed_tests']
            and new_count == tests.get('new_unique_passed_tests')
            and old_count == tests.get('unchanged_related_regression_passed_tests')
            and total == new_count + old_count == receipts[0]['passed_unique_tests'],
            'Dynamic exact total of three new and two old regression test classes required')
    first = read(ROOT / 'results/self_blend_development/failure_01.json')
    require(first.get('status') == 'preserved_development_failure' and first.get('failed') == 6
            and first.get('passed') == 95 and first.get('actual_cohort_started') is False,
            'Original 101-case pre-cohort development failure must remain preserved')
    root_failure = preserved_zip(ROOT, 'results/self_blend_development/failure_01.zip', first['artifacts'], 101, 6)
    second = read(ROOT / 'results/self_blend_development/core_failure_01_manifest.json')
    require(second.get('status') == 'development_failure_retained' and second.get('study_id') == 'self_blend_learning_v1'
            and second.get('synthetic_tests_only') is True and second.get('actual_new_cohort_runs') == 0
            and second.get('total_tests') == 45 and second.get('passed_tests') == 39 and second.get('failed_tests') == 6
            and second.get('archive') == 'results/self_blend_development/core_failure_01.zip'
            and second.get('archive_sha256') == sha(ROOT / second['archive']),
            'Original core-only invented failure must remain source-bound')
    core_failure = preserved_zip(ROOT, second['archive'], second['member_sha256'], 45, 6)
    publication = read(ROOT / publication_name)
    chronology = [baseline['verified_at_utc'], tests['checked_at_utc'], reg['registered_at_utc'],
        publication['checked_at_utc'], meta['started_at_utc'], meta['completed_at_utc'], verified['verified_at_utc']]
    times = list(map(utc, chronology))
    require(times == sorted(times) and times[2] < times[3] < times[4]
            and utc(second['preserved_at_utc']) <= times[1] and times[-1] <= datetime.now(timezone.utc),
            'Development/source/registration/publication/learning/verification UTC order required')
    proof = read(figures / 'provenance.json')
    require(proof['verification_sha256'] == sha(folder / 'verification.json')
            and proof['summary_sha256'] == sha(folder / 'summary.json')
            and proof['arrays_sha256'] == sha(folder / 'arrays.npz')
            and proof['plotted_values_sha256'] == sha(figures / 'plotted_values.json')
            and proof['render_script_sha256'] == sha(ROOT / 'scripts/render_self_blend_learning.py')
            and set(proof['figure_sha256']) == FIGURES,
            'Exact three figures, source data and renderer provenance required')
    require(all(sha(figures / name) == digest for name, digest in proof['figure_sha256'].items()), 'Figure bytes changed')
    status = read(ROOT / 'docs/repository-status.json')
    require(status.get('self_blend_learning_completed') is True
            and status['continuation_automation']['status'] == 'PAUSED'
            and status['current_findings_document'] == 'docs/self-blend-learning-findings.md'
            and status['level4_similarity_supported'] is status['shared_neural_mechanism_verified'] is False,
            'Completed bounded study and paused automatic continuation required')
    presentation_name = 'results/self_blend_presentation_validation.json'
    presentation = read(ROOT / presentation_name)
    require(presentation.get('status') == 'passed', 'Independent presentation validation required')
    required_docs = {'docs/self-blend-learning-findings.md', 'docs/self-blend-learning-reproduction.md'} | EVOLVING_INDEXES
    require(required_docs <= set(presentation['document_sha256']) | set(presentation['additional_file_sha256']),
            'Final findings, reproduction and all three evolving indexes must be presentation-checked')
    required_additional = {'scripts/render_self_blend_learning.py',
        'results/self_blend_figures_v1/plotted_values.json', 'results/self_blend_figures_v1/provenance.json'} | {
        'results/self_blend_figures_v1/' + name for name in FIGURES}
    require(required_additional <= set(presentation['additional_file_sha256']), 'All figures and provenance must be presentation-checked')
    for key in ('document_sha256', 'additional_file_sha256'):
        require(all(sha(safe(ROOT, name)) == digest for name, digest in presentation[key].items()), 'Presentation bytes changed')
    paths = set(reg['source_files']) | set(reg['parent_files']) | set(reg['graph_files']) | set(reg['input_files']) | EVOLVING_INDEXES | prepaths | {
        registration_name, publication_name, presentation_name,
        'docs/self-blend-learning-findings.md', 'docs/self-blend-learning-reproduction.md',
        'scripts/check_self_blend_archive.py', 'scripts/check_self_blend_baseline.py', 'scripts/render_self_blend_learning.py'}
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
        'prefreeze_passed_unique_tests_independently_counted': total,
        'prefreeze_unique_tests_by_class': dict(sorted(classes.items())),
        'original_development_failure_ZIPs_and_member_hashes_checked': [root_failure, core_failure],
        'prefreeze_publication_learning_verification_utc_order': chronology,
        'counts': counts, 'scientific_verification_sha256': sha(folder / 'verification.json'),
        'H_gain': summary['H_gain']['supported'], 'H_acquisition': summary['H_acquisition']['supported'],
        'H_protection': summary['H_protection']['supported'],
        'current_file_sha256': {name: sha(safe(ROOT, name)) for name in sorted(paths)},
        'method': 'Prior-tree raw Git-blob hash receipt, subsequent old-path diff, source-committed prefreeze and ZIP evidence, independently replayed results and presentation snapshots. No repeated whole-tree literal pass or new scientific execution.',
        'notice': 'A single preregistered blend and new cohort; common movement ceilings do not imply equal realized costs, novel-input generalization or a shared biological mechanism.'})
    print('Self-blend archive passed: exact prospective evidence, original failures and final presentation sealed.')


if __name__ == '__main__':
    main()
