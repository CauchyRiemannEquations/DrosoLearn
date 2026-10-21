"""Seal saved objective-pathways arithmetic and final presentation bytes.

Reads frozen receipts and manifests; no new graph, learner, random draw,
direction optimization or repeated scientific decomposition is executed.
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
from drosolearn.objective_pathways_envelope import (
    BASE_COMMIT, BASE_TAG, EVOLVING_INDEXES, ancestor, committed,
    read, require, safe, sha, utc, verify_envelope, write_new,
)

STUDY = 'objective_pathways_v1'
NEW_CLASSES = frozenset({'tests.test_objective_pathways', 'tests.test_objective_pathways_envelope',
    'tests.test_objective_pathways_verify'})
OLD_CLASSES = {'tests.test_objective_learning': 59, 'tests.test_self_blend_factors': 56}
FIGURES = frozenset({'all-contrasts.png', 'factor-accounting.png', 'choice-reward-paths.png'})
PRESENTATION_DRAFT = 'results/objective_pathways_presentation_attempt01.json'
PRESENTATION_DRAFT_ZIP = 'results/objective_pathways_presentation_attempt01.zip'
CONSUMER_ATTEMPT = 'results/objective_pathways_presentation_consumer_attempt01.json'
CONSUMER_ATTEMPT_ZIP = 'results/objective_pathways_presentation_consumer_attempt01.zip'

def prefreeze_gate(root, tests, reg, baseline_name, validation_name):
    """Count the unique final cases and retain every original development byte."""
    require(tests.get('status') == 'passed' and tests.get('frozen_source_hashes') == reg['source_files']
            and tests.get('parent_byte_gate') == reg['gate']
            and tests.get('synthetic_fixtures_only_before_freeze') is True,
            'Exact frozen source, inherited byte gate and synthetic-only snapshot required')
    require(all(type(tests.get(key)) is int and tests[key] == 0 for key in (
                'actual_new_saved_path_aggregation_before_freeze',
                'new_training_random_graph_or_optimization_calls')),
            'No new actual aggregation, optimization or learning before freeze required')
    receipts = tests.get('test_receipts'); development = tests.get('development_failure_artifacts')
    require(type(receipts) is list and len(receipts) == 1
            and receipts[0]['path'].startswith('results/objective_pathways_prefreeze_tests/')
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
    directory = (root / 'results/objective_pathways_development').resolve()
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
            archive_name = 'results/objective_pathways_development/' + archive_name
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
    require(verified.get('status') == 'passed' and verified.get('study_id') == STUDY
            and verified.get('scientific_producer_imported') is False
            and verified.get('new_helper_imported') is False,
            'Independent saved arithmetic without producer/helper imports required')
    checks = verified.get('counts', {})
    expected = {'arrays': 16, 'parent_trial_records_checked': 192000,
        'destination_effects_checked': 768000, 'measurement_records_checked': 2160,
        'actual_paths_checked': 480, 'source_action_reward_strata_checked': 34560,
        'new_training_trials': 0, 'new_random_draws': 0, 'new_optimization_calls': 0}
    require(all(type(checks.get(key)) is int and checks[key] == value for key, value in expected.items()),
            'Exact independently checked saved work and zero new task execution required')
    flags = ('factual_head_prefixes_and_parent_effects_checked',
        'actual_saved_uniform_threshold_choices_checked', 'only_actually_obtained_rewards_and_memory_checked',
        'signed_scalar_probability_identity_checked', 'same_prefeedback_gate_and_exact_prefix_checked',
        'only_post_both_actual_Go_eight_corners_checked',
        'state_scalar_direction_symmetric_allocations_checked',
        'signed_positive_and_negative_mass_allocations_checked',
        'own_other_and_all_unpaired_actual_actions_accounted',
        'all_paths_strata_costs_unknown_damage_and_endpoints_checked',
        'factual_diagnostics_and_nullable_means_checked', 'no_rng_model_graph_solver_or_future_policy_calls')
    require(all(checks.get(key) is True for key in flags),
            'Independent direction/size/state, actual actions/rewards and complete endpoint accounting required')
    require(len(summary.get('families', [])) == 18
            and all(len(family['seeds']) == 10 for family in summary['families'])
            and len(summary.get('actual_paths', [])) == 48
            and all(len(path['seeds']) == 10 for path in summary['actual_paths'])
            and all(len(seed.get('source_action_reward_strata', [])) == 72
                and set(seed.get('phases', {})) == {'all', 'pre', 'post'}
                for path in summary['actual_paths'] for seed in path['seeds']),
            'All eighteen paired contrasts and forty-eight factual families with every seed/stratum required')
    pairs = {(tuple(pair)) for pair in cfg['comparisons']}
    require(pairs == {('gated_uniform', 'gated_balanced'), ('gated_probability', 'gated_balanced'),
                     ('gated_probability', 'gated_uniform')}
            and {(family['candidate'], family['control']) for family in summary['families']} == pairs,
            'Three fixed U-minus-B, P-minus-B and P-minus-U contrasts required')
    validation = summary.get('validation', {})
    flags = ('all_exact_accounting_gates_passed', 'same_prefeedback_full_gate_and_prefix_checked',
        'only_actual_both_Go_crossed', 'one_sided_actual_choices_retained',
        'recorded_common_uniform_and_own_probabilities_checked',
        'both_Go_rewards_identical_and_scalar_difference_from_probability')
    residuals = validation.get('residuals', {})
    require(all(validation.get(key) is True for key in flags)
            and type(residuals) is dict and bool(residuals)
            and all(type(value) in (float, int) and 0 <= value <= cfg['numeric_tolerance']
                    for value in residuals.values()),
            'Complete factual reconstruction, Shapley efficiency and endpoint residual gates required')
    interpretation = summary.get('interpretation', {})
    require(all(interpretation.get(key) is True for key in ('retrospective_descriptive_only',
        'accounting_allocation_not_unique_causal_explanation', 'mixed_cells_not_relearned_policy',
        'NoGo_potential_rewards_or_directions_not_generated', 'parent_U_protection', 'parent_P_protection'))
        and all(interpretation.get(key) is False for key in ('new_confirmatory_hypothesis',
        'same_realized_movement_budget_claimed', 'parent_U_gain', 'parent_P_gain',
        'parent_U_acquisition', 'parent_P_acquisition', 'parent_state_gain',
        'novel_input_generalization_tested', 'new_human_experiment',
        'shared_neural_mechanism_claim', 'education_effect_claim')),
        'Descriptive allocation and mixed cells cannot become a relearned or uniquely causal policy claim')


def presentation_gate(root, folder, figures, verified_time):
    provenance = read(figures / 'provenance.json')
    expected_source = {'results/objective_pathways_v1/' + name for name in
        ('arrays.npz', 'summary.json', 'verification.json')} | {'scripts/render_objective_pathways.py'}
    require(provenance.get('status') == 'passed'
            and set(provenance.get('source_sha256', {})) == expected_source
            and all(sha(safe(root, name)) == digest for name, digest in provenance['source_sha256'].items())
            and set(provenance.get('artifact_sha256', {})) == FIGURES | {'plotted_values.json'}
            and all(sha(figures / name) == digest for name, digest in provenance['artifact_sha256'].items())
            and type(provenance.get('new_learning_or_solver_calls')) is int
            and provenance['new_learning_or_solver_calls'] == 0
            and provenance.get('primary_real_raw_only_for_factor_and_reward_figures') is True,
            'All three exact figures, renderer, verified results and zero-new-work provenance required')
    plot = read(figures / 'plotted_values.json')
    require(type(plot) is dict and set(plot) == {'comparison_order', 'component_order',
        'primary_component_endpoint_contributions', 'all_final_endpoint_differences',
        'primary_post_reward_and_movement_means', 'reward_and_movement_column_order'}
        and len(plot['primary_component_endpoint_contributions']) == 3
        and len(plot['all_final_endpoint_differences']) == 18
        and all(len(values) == 10 for values in plot['all_final_endpoint_differences'])
        and len(plot['primary_post_reward_and_movement_means']) == 3,
        'Saved all-comparison seeds, primary factor/reward values and explicit units required')
    name = 'results/objective_pathways_presentation_validation.json'
    qa = read(root / name)
    require(qa.get('status') == 'passed' and qa.get('study_id') == STUDY,
            'Independent final presentation QA required')
    required_docs = {'docs/objective-pathways-findings.md', 'docs/objective-pathways-reproduction.md'} | EVOLVING_INDEXES
    require(type(qa.get('document_sha256')) is dict and type(qa.get('additional_file_sha256')) is dict
            and required_docs <= set(qa['document_sha256']) | set(qa['additional_file_sha256']),
            'All final report/reproduction/evolving-index bytes must be QA-bound')
    required_additional = {'scripts/render_objective_pathways.py',
        'results/objective_pathways_figures_v1/plotted_values.json',
        'results/objective_pathways_figures_v1/provenance.json'} | {
        'results/objective_pathways_figures_v1/' + item for item in FIGURES}
    require(required_additional <= set(qa['additional_file_sha256']),
            'All figures and provenance must be independently presentation-checked')
    for key in ('document_sha256', 'additional_file_sha256'):
        require(all(sha(safe(root, item)) == digest for item, digest in qa[key].items()),
                'Final presentation evidence bytes changed')
    require(utc(verified_time) <= utc(provenance['rendered_at_utc']) <= utc(qa['checked_at_utc'])
            <= datetime.now(timezone.utc), 'Exact checked_at_utc QA key and figure chronology required')
    return name


def presentation_draft_gate(root, folder, figures, verified_time, qa_time):
    """Bind the six original draft bytes and the sole y-axis padding change."""
    note = read(safe(root, PRESENTATION_DRAFT))
    prefix = 'results/objective_pathways_figures_v1/'
    renderer = 'scripts/render_objective_pathways.py'
    members = {renderer, prefix + 'provenance.json', prefix + 'plotted_values.json'} | {
        prefix + name for name in FIGURES}
    require(note.get('status') == 'preserved_presentation_draft'
            and type(note.get('reason')) is str and bool(note['reason'])
            and note.get('archive') == PRESENTATION_DRAFT_ZIP
            and note.get('archive_sha256') == sha(safe(root, PRESENTATION_DRAFT_ZIP))
            and note.get('frozen_science_sources_config_inputs_results_or_success_criteria_changed') is False
            and type(note.get('new_training_rng_graph_optimization_or_scientific_replay_calls')) is int
            and note['new_training_rng_graph_optimization_or_scientific_replay_calls'] == 0
            and type(note.get('member_sha256')) is dict and set(note['member_sha256']) == members,
            'Original six-member presentation draft and zero-new-science note required')
    with ZipFile(safe(root, PRESENTATION_DRAFT_ZIP)) as archived:
        require(len(archived.namelist()) == 6 and set(archived.namelist()) == members,
                'Exactly six unique original presentation draft ZIP members required')
        raw = {name: archived.read(name) for name in members}
    require(all(hashlib.sha256(value).hexdigest() == note['member_sha256'][name]
                for name, value in raw.items()), 'Original presentation draft member bytes changed')
    provenance = json.loads(raw[prefix + 'provenance.json'])
    scientific = {'results/objective_pathways_v1/' + name for name in
        ('arrays.npz', 'summary.json', 'verification.json')}
    require(provenance.get('status') == 'passed'
            and set(provenance.get('source_sha256', {})) == scientific | {renderer}
            and all(provenance['source_sha256'][name] == sha(safe(root, name)) for name in scientific)
            and provenance['source_sha256'][renderer] == note['member_sha256'][renderer]
            and set(provenance.get('artifact_sha256', {})) == FIGURES | {'plotted_values.json'}
            and all(provenance['artifact_sha256'][name] == note['member_sha256'][prefix + name]
                for name in FIGURES | {'plotted_values.json'})
            and type(provenance.get('new_learning_or_solver_calls')) is int
            and provenance['new_learning_or_solver_calls'] == 0
            and provenance.get('primary_real_raw_only_for_factor_and_reward_figures') is True,
            'Every original figure and renderer must bind the same completed scientific artifacts')
    original = b"ax[0].set_title('Post-gate obtained rewards');ax[0].legend()"
    revised = b"ax[0].set_title('Post-gate obtained rewards');ax[0].set_ylim(0,140);ax[0].legend()"
    require(raw[renderer].count(original) == 1
            and safe(root, renderer).read_bytes() == raw[renderer].replace(original, revised)
            and all(safe(root, prefix + name).read_bytes() == raw[prefix + name]
                for name in ('all-contrasts.png', 'factor-accounting.png', 'plotted_values.json'))
            and safe(root, prefix + 'choice-reward-paths.png').read_bytes() != raw[prefix + 'choice-reward-paths.png'],
            'Only reward-axis display space may change; plotted values and other two figures must be exact')
    final_provenance = read(figures / 'provenance.json')
    times = [verified_time, provenance['rendered_at_utc'], note['checked_at_utc'],
             final_provenance['rendered_at_utc'], qa_time]
    require(list(map(utc, times)) == sorted(map(utc, times))
            and utc(times[-1]) <= datetime.now(timezone.utc),
            'Independent science, original draft, preserved bytes, final rendering and QA chronology required')
    return {'archive': PRESENTATION_DRAFT_ZIP, 'archive_sha256': note['archive_sha256'],
        'note': PRESENTATION_DRAFT, 'note_sha256': sha(safe(root, PRESENTATION_DRAFT)),
        'original_six_member_sha256': note['member_sha256'],
        'original_and_final_artifacts_bind_same_scientific_results': True,
        'only_single_reward_plot_y_axis_limit_added': True,
        'plotted_values_and_other_two_figures_exactly_unchanged': True,
        'new_training_rng_graph_optimization_or_scientific_replay_calls': 0,
        'scientific_verification_draft_note_final_render_QA_utc_order': times}


def presentation_consumer_gate(root, verified_time, qa_time):
    """Keep the scratch consumer and its recorded assertion diagnostic intact."""
    note = read(safe(root, CONSUMER_ATTEMPT))
    script = 'qa_objective_pathways_presentation_attempt01.py'
    diagnostic = 'objective_pathways_QA_consumer_attempt01.json'
    members = {script, diagnostic}
    require(note.get('status') == 'preserved_presentation_QA_consumer_attempt'
            and note.get('archive') == CONSUMER_ATTEMPT_ZIP
            and note.get('archive_sha256') == sha(safe(root, CONSUMER_ATTEMPT_ZIP))
            and type(note.get('member_sha256')) is dict and set(note['member_sha256']) == members
            and type(note.get('original_consumer_exit_code')) is int and note['original_consumer_exit_code'] == 1
            and note.get('all_numeric_checks_passed_before_consumer_failure') is True
            and note.get('actual_data_numeric_outputs_documents_or_frozen25_science_changed') is False
            and type(note.get('new_learning_RNG_graph_optimization_or_scientific_replay_calls')) is int
            and note['new_learning_RNG_graph_optimization_or_scientific_replay_calls'] == 0
            and all(type(note.get(key)) is str and bool(note[key]) for key in ('reason', 'corrected_clause')),
            'Original QA-consumer assertion attempt and zero-scientific-change note required')
    with ZipFile(safe(root, CONSUMER_ATTEMPT_ZIP)) as archived:
        require(len(archived.namelist()) == 2 and set(archived.namelist()) == members,
                'Exactly two unique preserved consumer script/diagnostic members required')
        raw = {name: archived.read(name) for name in members}
    require(all(hashlib.sha256(value).hexdigest() == note['member_sha256'][name]
                for name, value in raw.items()), 'Original QA-consumer source or recorded diagnostic bytes changed')
    failure = json.loads(raw[diagnostic])
    require(failure.get('status') == 'preserved_QA_consumer_assertion_failure'
            and failure.get('script') == script and failure.get('script_sha256') == note['member_sha256'][script]
            and type(failure.get('exit_code')) is int and failure['exit_code'] == 1
            and type(failure.get('traceback')) is str and 'AssertionError' in failure['traceback']
            and "assert 'objective_pathways_v1' in" in failure['traceback']
            and b"assert 'objective_pathways_v1' in" in raw[script]
            and failure.get('all_numeric_array_CSV_18family_480path_34560strata_and7report_table_checks_before_this_line') == 'passed'
            and failure.get('frozen_science_or_results_changed') is False
            and type(failure.get('new_learning_RNG_graph_solver_or_full_scientific_replay_calls')) is int
            and failure['new_learning_RNG_graph_solver_or_full_scientific_replay_calls'] == 0,
            'Recorded presentation-consumer formatting assertion must remain separate from successful numeric validation')
    times = [verified_time, note['checked_at_utc'], qa_time]
    require(list(map(utc, times)) == sorted(map(utc, times))
            and utc(times[-1]) <= datetime.now(timezone.utc),
            'Science verification, preserved QA-consumer attempt and final corrected QA chronology required')
    return {'archive': CONSUMER_ATTEMPT_ZIP, 'archive_sha256': note['archive_sha256'],
        'note': CONSUMER_ATTEMPT, 'note_sha256': sha(safe(root, CONSUMER_ATTEMPT)),
        'original_member_sha256': note['member_sha256'],
        'original_script_and_recorded_traceback_preserved': True,
        'numeric_validation_passed_before_consumer_format_assertion': True,
        'scientific_or_document_source_changes': False,
        'new_learning_RNG_graph_optimization_or_scientific_replay_calls': 0,
        'science_attempt_and_final_QA_utc_order': times}


def main():
    out = ROOT / 'results/objective_pathways_archive_validation.json'
    require(not out.exists(), 'New exclusive descriptive objective-pathways archive receipt required')
    baseline_name = 'results/objective_pathways_baseline_preservation.json'
    validation_name = 'results/objective_pathways_prefreeze_validation.json'
    registration_name = 'results/registration_objective_pathways_v1.json'
    publication_name = 'results/objective_pathways_publication_proof.json'
    baseline = read(ROOT / baseline_name)
    require(baseline.get('status') == 'passed' and baseline.get('baseline_tag') == BASE_TAG
            and baseline.get('baseline_commit') == BASE_COMMIT
            and baseline.get('allowed_evolving_indexes') == sorted(EVOLVING_INDEXES)
            and baseline.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True,
            'Exact three-index completed raw Git-blob baseline receipt required')
    old = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE_COMMIT], cwd=ROOT).decode().splitlines())
    changed = set(subprocess.check_output(['git', 'diff', '--name-only', BASE_COMMIT], cwd=ROOT).decode().splitlines())
    require(old & changed <= EVOLVING_INDEXES, 'A completed historical file changed')
    folder = ROOT / 'results/objective_pathways_v1'; figures = ROOT / 'results/objective_pathways_figures_v1'
    cfg, reg, meta = verify_envelope(ROOT, ROOT / 'configs/objective_pathways_v1.json',
        ROOT / registration_name, folder, ROOT / publication_name)
    require(len(reg['source_files']) == 25 and len(reg['parent_files']) == 16,
            'Exact source25/parent16 pure saved-analysis closure required')
    ancestor(ROOT, BASE_COMMIT, reg['source_git_commit'])
    verified = read(folder / 'verification.json'); summary = read(folder / 'summary.json')
    require(verified.get('summary_sha256') == sha(folder / 'summary.json')
            and verified.get('registration_sha256') == sha(ROOT / registration_name),
            'Independent verdict must bind current exact registration and summary bytes')
    independent_gate(verified, summary, cfg)
    with (folder / 'measurements.csv').open(encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)) == 15,
                'Exactly fifteen unique measurement columns required')
        require(sum(1 for _ in reader) == 2160, 'All two thousand one hundred sixty descriptive rows required')
    tests = read(ROOT / validation_name)
    prepaths, classes, development, zip_bindings = prefreeze_gate(ROOT, tests, reg, baseline_name, validation_name)
    publication = read(ROOT / publication_name)
    chronology = [baseline['verified_at_utc'], tests['checked_at_utc'], reg['registered_at_utc'],
        publication['checked_at_utc'], meta['started_at_utc'], meta['completed_at_utc'], verified['verified_at_utc']]
    times = list(map(utc, chronology))
    require(times == sorted(times) and times[2] < times[3] < times[4]
            and times[-1] <= datetime.now(timezone.utc),
            'Baseline/freeze/private-publication/analysis/independent-verification chronology required')
    presentation_name = presentation_gate(ROOT, folder, figures, verified['verified_at_utc'])
    draft_gate = presentation_draft_gate(ROOT, folder, figures, verified['verified_at_utc'],
        read(ROOT / presentation_name)['checked_at_utc'])
    consumer_gate = presentation_consumer_gate(ROOT, verified['verified_at_utc'],
        read(ROOT / presentation_name)['checked_at_utc'])
    status = read(ROOT / 'docs/repository-status.json')
    require(status.get('objective_pathways_completed') is True
            and status['continuation_automation']['status'] == 'PAUSED'
            and status['current_findings_document'] == 'docs/objective-pathways-findings.md'
            and status['level4_similarity_supported'] is status['shared_neural_mechanism_verified'] is False,
            'Completed descriptive report and bounded paused continuation required')
    paths = set(reg['source_files']) | set(reg['parent_files']) | EVOLVING_INDEXES | prepaths | {
        registration_name, publication_name, presentation_name, PRESENTATION_DRAFT, PRESENTATION_DRAFT_ZIP,
        CONSUMER_ATTEMPT, CONSUMER_ATTEMPT_ZIP,
        'docs/objective-pathways-findings.md', 'docs/objective-pathways-reproduction.md',
        'scripts/check_objective_pathways_archive.py', 'scripts/check_objective_pathways_baseline.py',
        'scripts/render_objective_pathways.py'}
    for directory in (folder, figures):
        paths |= {path.relative_to(ROOT).as_posix() for path in directory.iterdir() if path.is_file()}
    write_new(out, {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT,
        'baseline_hash_receipt_sha256': sha(ROOT / baseline_name),
        'hash_compared_files': baseline['hash_compared_files'], 'hash_compared_bytes': baseline['hash_compared_bytes'],
        'subsequent_old_file_changes': sorted(old & changed), 'allowed_evolving_indexes': sorted(EVOLVING_INDEXES),
        'all_old_files_Git_blob_hash_preserved_except_three_indexes': True,
        'source_git_commit': reg['source_git_commit'], 'registration_sha256': sha(ROOT / registration_name),
        'prefreeze_receipts_committed_source_sha256': {name: sha(safe(ROOT, name)) for name in sorted(prepaths)},
        'prefreeze_passed_unique_tests_independently_counted': sum(classes.values()),
        'prefreeze_unique_tests_by_class': classes,
        'prefreeze_publication_analysis_verification_utc_order': chronology,
        'development_failure_artifact_bytes_preserved': development,
        'development_ZIP_manifests_and_original_member_hashes': zip_bindings,
        'presentation_draft_preservation': draft_gate,
        'presentation_consumer_attempt_preservation': consumer_gate,
        'counts': summary['counts'], 'scientific_verification_sha256': sha(folder / 'verification.json'),
        'presentation_validation_sha256': sha(ROOT / presentation_name),
        'current_file_sha256': {name: sha(safe(ROOT, name)) for name in sorted(paths)},
        'method': 'Prior Git-blob receipt, subsequent old-path diff, frozen source-committed synthetic tests/development bytes, independent saved arithmetic and final presentation snapshots; no new task learning, optimization or repeated science replay.',
        'notice': 'Retrospective symmetric direction/size/state accounting and actual choice/reward path description. Mixed cells are not newly learned policies, unique causal effects, novel-input generalization or common biological mechanisms.'})
    print('Objective-pathways archive passed: immutable actual parent, saved arithmetic and final QA bytes sealed.')


if __name__ == '__main__':
    main()
