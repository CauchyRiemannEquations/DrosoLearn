"""Seal actual objective-learning work, frozen preregistration and final QA bytes.

This checker reads completed receipts. It does not repeat training, new-cohort
random or graph construction, or any B/W/U/P direction optimization.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from drosolearn.objective_envelope import (
    BASE_COMMIT, BASE_TAG, EVOLVING_INDEXES, METHODS, ancestor, committed,
    read, require, safe, sha, utc, verify_envelope, write_new,
)

STUDY = 'objective_learning_v1'
NEW_CLASSES = frozenset({'tests.test_objective_learning', 'tests.test_objective_learning_envelope',
    'tests.test_objective_learning_verify'})
OLD_CLASSES = {'tests.test_cone_geometry': 45, 'tests.test_self_blend_learning': 47,
    'tests.test_probability_alignment': 38}
FIGURES = frozenset({'learning-curves.png', 'all-conditions.png', 'protection-and-cost.png'})


def prefreeze_gate(root, tests, reg, baseline_name, validation_name):
    """Count the unique final cases and retain every original development byte."""
    require(tests.get('status') == 'passed' and tests.get('frozen_source_hashes') == reg['source_files']
            and tests.get('parent_byte_gate') == reg['gate']
            and tests.get('synthetic_fixtures_only_before_freeze') is True,
            'Exact frozen source, inherited byte gate and synthetic-only snapshot required')
    require(all(type(tests.get(key)) is int and tests[key] == 0 for key in (
                'actual_new_training_before_freeze',
                'actual_new_cohort_objective_optimizations_before_freeze',
                'new_cohort_graph_feature_rng_calls_before_freeze')),
            'No new actual aggregation, optimization or learning before freeze required')
    receipts = tests.get('test_receipts'); development = tests.get('development_failure_artifacts')
    require(type(receipts) is list and len(receipts) == 1
            and receipts[0]['path'].startswith('results/objective_learning_prefreeze_tests/')
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
    directory = (root / 'results/objective_learning_development').resolve()
    require(directory.is_relative_to(root) and (not directory.exists() or directory.is_dir())
            and ({p.relative_to(root).as_posix() for p in directory.rglob('*') if p.is_file()} if directory.exists() else set())
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
    require(set(classes) == NEW_CLASSES | set(OLD_CLASSES)
            and classes == tests.get('unique_passed_tests_by_class')
            and all(classes[name] == count for name, count in OLD_CLASSES.items())
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
            archive_name = 'results/objective_learning_development/' + archive_name
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
    """Require all actual trajectories and every B/W/U/P case to be checked."""
    require(verified.get('status') == 'passed' and verified.get('study_id') == STUDY
            and verified.get('scientific_producer_imported') is False
            and verified.get('new_helper_imported') is False,
            'Independent autonomous-learning replay without producer/head imports required')
    counts = summary['counts']; vc = verified.get('counts', {})
    expected = {'arrays': 215, 'independently_replayed_training_actions': 192000,
        'conditions': 48, 'model_seeds': 10, 'checkpoint_static_probes': 9600,
        'head_update_calls': 192000, 'internal_update_calls': 0, 'value_update_calls': 0,
        'unique_balanced_states_independently_reoptimized': counts['balanced_unique_solver_cases'],
        'unique_objective_states_independently_reoptimized': counts['objective_unique_solver_cases'],
        'unique_uniform_states_independently_reoptimized': counts['uniform_unique_solver_cases'],
        'unique_probability_states_independently_reoptimized': counts['probability_unique_solver_cases']}
    require(all(type(vc.get(key)) is int and vc[key] == value for key, value in expected.items()),
            'Exact independent actual work and dynamic per-family support case inventory required')
    flags = ('all_fresh_autonomous_choices_rewards_and_heads_replayed',
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
        'all_candidate_gain_acquisition_protection_and_P_minus_U_state_hypotheses_recomputed',
        'each_P_case_uses_own_actual_preupdate_head_and_new_actual_Go_trial',
        'common_actual_movement_ceiling_not_equal_realized_budget_checked')
    require(all(vc.get(key) is True for key in flags)
            and vc.get('extra_optimizer_and_memory_processing_cost_equalized') is False
            and vc.get('new_generalization_or_human_mechanism_claim') is False,
            'Independent own-state, feedback, safety, costs and separate hypotheses must all be checked')
    require(summary.get('methods') == list(METHODS) and len(summary.get('families', [])) == 48
            and all(len(family['seeds']) == 10 for family in summary['families'])
            and len(summary.get('geometry', [])) == 6
            and all(len(group['seeds']) == 10 for group in summary['geometry']),
            'All forty-eight prespecified conditions and their ten computational seeds required')
    require(summary.get('validation', {}).get('all_numeric_and_information_gates_passed') is True
            and all(summary.get('gate_validation', {}).values())
            and bool(summary.get('gate_validation'))
            and bool(summary.get('information_validation'))
            and all(summary['information_validation'].values()),
            'Complete numeric, pre-feedback gate and hidden-information integrity required')
    objective = summary.get('objective_validation', {})
    require(objective and all(value for value in objective.values() if type(value) is bool)
            and type(objective.get('max_certificate_residual')) in (float, int)
            and 0 <= objective['max_certificate_residual'] <= cfg['numeric_tolerance'],
            'Every actual U/P own-state certificate and cache gate must pass')
    for hypothesis in ('H_gain', 'H_acquisition', 'H_protection'):
        group = summary.get(hypothesis)
        require(type(group) is dict and set(group) == {'gated_uniform', 'gated_probability'}
                and all(type(value.get('supported')) is bool
                    and value.get('all_numeric_and_information_gates_passed') is True for value in group.values()),
                'Separate frozen U/P hypothesis verdicts must retain support or failure independently')
    require(type(summary.get('H_state', {}).get('supported')) is bool
            and summary['H_state'].get('comparison') == 'gated_probability_minus_gated_uniform'
            and summary['H_state'].get('all_numeric_and_information_gates_passed') is True,
            'Independent frozen P-versus-U endpoint verdict required')
    budget = summary.get('budget_validation', {})
    require(budget.get('common_step_ceiling') == .1 and budget.get('common_path_ceiling') == 40.
            and abs(budget.get('common_squared_step_sum_ceiling', -1.) - 4.) <= 1e-12
            and budget.get('actual_cumulative_cost_equalized') is False
            and budget.get('extra_solver_and_information_processing_cost_equalized') is False,
            'Same ceilings must not be described as equal realized movement or processing costs')
    interpretation = summary.get('interpretation', {})
    require(all(interpretation.get(key) is False for key in ('any_learner_uses_hidden_targets_for_routing',
        'prior_failure_or_seed_discarded', 'generalization_tested', 'new_human_experiment',
        'shared_neural_mechanism_claim', 'education_effect_claim'))
        and all(interpretation.get(key) is True for key in ('internal_synapses_fixed',
        'observed_reward_memory_is_additional_learning_state',
        'unknown_means_reward_unobserved_not_novel_input',
        'probability_direction_tangent_not_finite_global_optimum', 'own_actual_current_state_used_for_each_policy')),
        'Actual artificial output learning must retain its biological, generalization and optimizer limitations')


def main():
    out = ROOT / 'results/objective_learning_archive_validation.json'
    require(not out.exists(), 'New exclusive objective-learning archive receipt required')
    baseline_name = 'results/objective_learning_baseline_preservation.json'
    validation_name = 'results/objective_learning_prefreeze_validation.json'
    registration_name = 'results/registration_objective_learning_v1.json'
    publication_name = 'results/objective_learning_publication_proof.json'
    folder = ROOT / 'results/objective_learning_v1'; figures = ROOT / 'results/objective_learning_figures_v1'
    baseline = read(ROOT / baseline_name)
    require(baseline.get('status') == 'passed' and baseline.get('baseline_tag') == BASE_TAG
            and baseline.get('baseline_commit') == BASE_COMMIT
            and baseline.get('allowed_evolving_indexes') == sorted(EVOLVING_INDEXES)
            and baseline.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True,
            'Exact three-index completed raw Git-blob baseline receipt required')
    old = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE_COMMIT], cwd=ROOT).decode().splitlines())
    changed = set(subprocess.check_output(['git', 'diff', '--name-only', BASE_COMMIT], cwd=ROOT).decode().splitlines())
    require(old & changed <= EVOLVING_INDEXES, 'A completed historical file changed')
    cfg, reg, meta = verify_envelope(ROOT, ROOT / 'configs/objective_learning_v1.json',
        ROOT / registration_name, folder, ROOT / publication_name)
    require(all(len(reg[group]) == size for group, size in (
        ('source_files', 49), ('parent_files', 14), ('graph_files', 4), ('input_files', 2))),
        'Exact source49/parent14/graph4/input2 closure required')
    ancestor(ROOT, BASE_COMMIT, reg['source_git_commit'])
    verified = read(folder / 'verification.json'); summary = read(folder / 'summary.json')
    require(verified.get('summary_sha256') == sha(folder / 'summary.json')
            and verified.get('registration_sha256') == sha(ROOT / registration_name),
            'Independent verdict must bind exact current summary and registration bytes')
    independent_gate(verified, summary, cfg)
    with gzip.open(folder / 'measurements.csv.gz', 'rt', encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)) == 96,
                'Exact ninety-six unique measurement fields required')
        require(sum(1 for _ in reader) == 192000, 'All one hundred ninety-two thousand actual trial records required')
    tests = read(ROOT / validation_name)
    prepaths, classes, development, zip_bindings = prefreeze_gate(ROOT, tests, reg, baseline_name, validation_name)
    publication = read(ROOT / publication_name)
    chronology = [baseline['verified_at_utc'], tests['checked_at_utc'], reg['registered_at_utc'],
        publication['checked_at_utc'], meta['started_at_utc'], meta['completed_at_utc'], verified['verified_at_utc']]
    times = list(map(utc, chronology))
    require(times == sorted(times) and times[2] < times[3] < times[4]
            and times[-1] <= datetime.now(timezone.utc),
            'Baseline/freeze/private-publication/actual-learning/independent-verification chronology required')
    provenance = read(figures / 'provenance.json')
    require(provenance.get('study_id') == STUDY and provenance.get('verified_arrays_only') is True
            and provenance.get('arrays_sha256') == sha(folder / 'arrays.npz')
            and provenance.get('summary_sha256') == sha(folder / 'summary.json')
            and provenance.get('verification_sha256') == sha(folder / 'verification.json')
            and provenance.get('render_script_sha256') == sha(ROOT / 'scripts/render_objective_learning.py')
            and provenance.get('plotted_values_sha256') == sha(figures / 'plotted_values.json')
            and set(provenance.get('figure_sha256', {})) == FIGURES
            and all(sha(figures / name) == digest for name, digest in provenance['figure_sha256'].items())
            and type(provenance.get('new_training_trials')) is int and provenance['new_training_trials'] == 0
            and type(provenance.get('new_optimization_calls')) is int and provenance['new_optimization_calls'] == 0
            and provenance.get('new_parameter_selection') is False,
            'Renderer-only zero fresh work, all exact figures and plotted bytes must be bound')
    plot = read(figures / 'plotted_values.json')
    require(set(plot) == {'learning', 'conditions', 'protection', 'costs'}
            and len(plot['learning']) == len(plot['conditions']) == 48
            and len(plot['protection']) == len(plot['costs']) == 8,
            'All actual model conditions, learning curves, operational protection and cost values must be retained')
    status = read(ROOT / 'docs/repository-status.json')
    require(status.get('objective_learning_completed') is True
            and status['continuation_automation']['status'] == 'PAUSED'
            and status['current_findings_document'] == 'docs/objective-learning-findings.md'
            and status['level4_similarity_supported'] is status['shared_neural_mechanism_verified'] is False,
            'Completed actual-learning report, bounded interpretation and paused continuation required')
    presentation_name = 'results/objective_learning_presentation_validation.json'
    presentation = read(ROOT / presentation_name)
    require(presentation.get('status') == 'passed' and presentation.get('study_id') == STUDY,
            'Independent final presentation QA for this actual-learning study required')
    required_docs = {'docs/objective-learning-findings.md', 'docs/objective-learning-reproduction.md'} | EVOLVING_INDEXES
    require(required_docs <= set(presentation['document_sha256']) | set(presentation['additional_file_sha256']),
            'All final report, reproduction and evolving-index bytes must be presentation-checked')
    required_additional = {'scripts/render_objective_learning.py',
        'results/objective_learning_figures_v1/plotted_values.json',
        'results/objective_learning_figures_v1/provenance.json'} | {
        'results/objective_learning_figures_v1/' + name for name in FIGURES}
    require(required_additional <= set(presentation['additional_file_sha256']),
            'All three figures, renderer and provenance must be presentation-checked')
    for key in ('document_sha256', 'additional_file_sha256'):
        require(all(sha(safe(ROOT, name)) == digest for name, digest in presentation[key].items()),
                'Final presentation evidence bytes changed')
    require(utc(verified['verified_at_utc']) <= utc(provenance['generated_at_utc'])
            <= utc(presentation['checked_at_utc']) <= datetime.now(timezone.utc),
            'Independent verification must precede rendering and actual checked_at_utc presentation QA')
    paths = set(reg['source_files']) | set(reg['parent_files']) | set(reg['graph_files']) | set(reg['input_files'])
    paths |= EVOLVING_INDEXES | prepaths | {registration_name, publication_name, presentation_name,
        'docs/objective-learning-findings.md', 'docs/objective-learning-reproduction.md',
        'scripts/check_objective_learning_archive.py', 'scripts/check_objective_learning_baseline.py',
        'scripts/render_objective_learning.py'}
    paths |= set(presentation['document_sha256']) | set(presentation['additional_file_sha256'])
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
        'prefreeze_publication_learning_verification_utc_order': chronology,
        'development_failure_artifact_bytes_preserved': development,
        'development_ZIP_manifests_and_original_member_hashes': zip_bindings,
        'counts': summary['counts'], 'scientific_verification_sha256': sha(folder / 'verification.json'),
        'presentation_validation_sha256': sha(ROOT / presentation_name),
        'current_file_sha256': {name: sha(safe(ROOT, name)) for name in sorted(paths)},
        'method': 'Prior Git-blob receipt, subsequent old-path diff, source-committed synthetic tests/development bytes, independent full autonomous replay and final presentation snapshots; this checker repeats no training or optimization.',
        'notice': 'Actual artificial output-head learning, with fixed internal synapses, observed feedback memory and explicitly counted unequal solver work. Unknown means reward-unobserved training input; no new-input generalization, common neural mechanism or educational effect is claimed.'})
    print('Objective-learning archive passed: autonomous learning, dynamic solver costs and final QA bytes sealed.')


if __name__ == '__main__':
    main()
