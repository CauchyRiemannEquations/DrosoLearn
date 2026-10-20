"""Seal descriptive self-blend factor evidence without rerunning its arithmetic."""
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
from drosolearn.self_blend_factors_envelope import (
    BASE_COMMIT, BASE_TAG, EVOLVING_INDEXES, committed, read, require, safe,
    sha, utc, verify_envelope, write_new,
)

NEW_CLASSES = frozenset({'tests.test_self_blend_factors', 'tests.test_self_blend_factors_envelope',
                         'tests.test_self_blend_factors_verify'})
OLD_CLASSES = {'tests.test_gated_factors': 45, 'tests.test_self_blend_learning': 47}
FIGURES = frozenset({'all-contrasts.png', 'factor-accounting.png', 'paired-diagnostics.png'})
STYLE_ZIP = 'results/self_blend_factors_prefreeze_style_attempt01.zip'
STYLE_NOTE = 'results/self_blend_factors_prefreeze_finalization.json'


def style_finalization_gate(root, tests, reg, final_cases):
    """Keep the original passing attempt, proving an EOF-only finalization."""
    note = read(safe(root, STYLE_NOTE))
    test_path = 'tests/test_self_blend_factors_envelope.py'
    validation = 'results/self_blend_factors_prefreeze_validation.json'
    junit = 'results/self_blend_factors_prefreeze_tests/self_blend_factors_prefreeze_final.xml'
    members = {validation, junit, test_path}
    require(note.get('status') == 'passed' and note.get('archive') == STYLE_ZIP
            and note.get('archive_sha256') == sha(safe(root, STYLE_ZIP))
            and note.get('test_path') == test_path and note.get('removed_only_trailing_blank_line') is True
            and note.get('original_passed_tests') == 295 and note.get('original_failed_tests') == 0
            and note.get('scientific_design_thresholds_or_parent_inputs_changed') is False
            and note.get('actual_saved_factor_aggregation_started') is False
            and type(note.get('member_sha256')) is dict and set(note['member_sha256']) == members,
            'Source-committed original style-stop evidence and exact three raw ZIP members required')
    with ZipFile(safe(root, STYLE_ZIP)) as archive:
        require(len(archive.namelist()) == 3 and set(archive.namelist()) == members,
                'Exact three unique original style-attempt ZIP members required')
        raw = {name: archive.read(name) for name in members}
    require(all(hashlib.sha256(value).hexdigest() == note['member_sha256'][name] for name, value in raw.items()),
            'Original passing-attempt ZIP member bytes changed')
    original = json.loads(raw[validation])
    original_cases = ET.fromstring(raw[junit]).findall('.//testcase')
    require(len(original_cases) == len({(c.get('classname'), c.get('name')) for c in original_cases}) == 295
            and all(c.find(key) is None for c in original_cases for key in ('failure', 'error', 'skipped'))
            and {(c.get('classname'), c.get('name')) for c in original_cases}
                == {(c.get('classname'), c.get('name')) for c in final_cases},
            'Original and final passing 295-test identities must match')
    require(note.get('original_test_sha256') == note['member_sha256'][test_path]
            and note.get('final_test_sha256') == reg['source_files'][test_path] == sha(safe(root, test_path))
            and raw[test_path].rstrip(b'\n') == safe(root, test_path).read_bytes().rstrip(b'\n')
            and safe(root, test_path).read_bytes().endswith(b'\n')
            and not safe(root, test_path).read_bytes().endswith(b'\n\n')
            and len(raw[test_path]) - len(safe(root, test_path).read_bytes()) == 2,
            'Only trailing empty LF lines may differ in the unregistered test source')
    expected_old_source = {**reg['source_files'], test_path: note['original_test_sha256']}
    require(original.get('status') == 'passed' and original.get('frozen_source_hashes') == expected_old_source
            and original.get('parent_byte_gate') == reg['gate']
            and original.get('unique_passed_tests') == 295 and original.get('new_unique_passed_tests') == 203
            and original.get('unchanged_related_regression_passed_tests') == 92
            and original.get('development_failure_artifacts') == []
            and original.get('actual_new_saved_factor_aggregation_before_freeze') == 0
            and original.get('new_training_random_graph_or_optimization_calls') == 0,
            'All other21 source hashes, parent evidence, criteria and no-aggregation snapshot must remain identical')
    receipts = original.get('test_receipts')
    require(type(receipts) is list and len(receipts) == 1 and receipts[0].get('path') == junit
            and receipts[0].get('sha256') == note['member_sha256'][junit]
            and receipts[0].get('passed_unique_tests') == 295,
            'Original preserved prefreeze receipt must bind its exact passing JUnit')
    require(utc(original['checked_at_utc']) <= utc(note['checked_at_utc']) <= utc(tests['checked_at_utc'])
            < utc(reg['registered_at_utc']), 'EOF finalization and refreshed passing suite must precede registration')
    return {'original_style_attempt_zip_sha256': note['archive_sha256'],
        'original_member_sha256': note['member_sha256'], 'finalization_note_sha256': sha(safe(root, STYLE_NOTE)),
        'only_trailing_empty_LF_lines_removed_from_one_unregistered_test': True,
        'original_notice_count_corrected_by_rawbytes': True,
        'removed_trailing_LF_bytes': 2,
        'final_trailing_LF_bytes': 1,
        'non_LF_original_bytes_identical': True,
        'removed_trailing_LF_bytes_independently_counted': len(raw[test_path]) - len(safe(root, test_path).read_bytes()),
        'all_other21_source_hashes_and_original_295_passing_test_identities_preserved': True,
        'original_tests_finalization_refreshed_tests_registration_utc_order': [original['checked_at_utc'],
            note['checked_at_utc'], tests['checked_at_utc'], reg['registered_at_utc']]}


def main():
    out = ROOT / 'results/self_blend_factors_archive_validation.json'
    require(not out.exists(), 'New exclusive descriptive archive receipt required')
    baseline_name = 'results/self_blend_factors_baseline_preservation.json'
    baseline = read(ROOT / baseline_name)
    require(baseline.get('status') == 'passed' and baseline.get('baseline_tag') == BASE_TAG
            and baseline.get('baseline_commit') == BASE_COMMIT
            and baseline.get('allowed_evolving_indexes') == sorted(EVOLVING_INDEXES)
            and baseline.get('all_old_files_Git_blob_hash_preserved_except_three_indexes') is True,
            'Exact three-index prior raw Git-blob baseline receipt required')
    old = set(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE_COMMIT], cwd=ROOT).decode().splitlines())
    changed = set(subprocess.check_output(['git', 'diff', '--name-only', BASE_COMMIT], cwd=ROOT).decode().splitlines())
    require(old & changed <= EVOLVING_INDEXES, 'A completed historical file changed')
    folder = ROOT / 'results/self_blend_factors_v1'; figures = ROOT / 'results/self_blend_factors_figures_v1'
    registration_name = 'results/registration_self_blend_factors_v1.json'
    publication_name = 'results/self_blend_factors_publication_proof.json'
    cfg, reg, meta = verify_envelope(ROOT, ROOT / 'configs/self_blend_factors_v1.json',
        ROOT / registration_name, folder, ROOT / publication_name)
    require(len(reg['source_files']) == 22 and len(reg['parent_files']) == 13,
            'Exact pure-analysis source22/parent13 closure required')
    verified = read(folder / 'verification.json'); summary = read(folder / 'summary.json'); counts = summary['counts']
    require(verified.get('status') == 'passed' and verified.get('study_id') == 'self_blend_factors_v1'
            and verified.get('scientific_producer_imported') is False and verified.get('new_helper_imported') is False
            and verified.get('summary_sha256') == sha(folder / 'summary.json')
            and verified.get('registration_sha256') == sha(ROOT / registration_name),
            'Independent complete descriptive arithmetic must bind current exact bytes')
    vc = verified.get('counts', {})
    require(all(type(vc.get(key)) is int and vc[key] == value for key, value in {
        'arrays': 14, 'parent_trial_records_checked': 144000, 'destination_effects_checked': 576000,
        'measurement_records_checked': 3600, 'new_training_trials': 0,
        'new_random_draws': 0, 'new_optimization_calls': 0}.items()),
        'Complete saved parent effects and zero new work verification required')
    require(all(vc.get(key) is True for key in (
        'factual_head_prefixes_and_parent_effects_checked', 'only_post_both_actual_Go_eight_corners_checked',
        'state_scalar_direction_symmetric_allocations_checked', 'signed_positive_and_negative_mass_allocations_checked',
        'own_other_and_all_unpaired_actual_actions_accounted', 'factual_diagnostics_and_nullable_seed_means_checked',
        'partial_control_information_not_silently_equalized', 'no_rng_model_graph_solver_or_future_policy_calls')),
        'Independent state/scalar/direction, actual actions and diagnostics verification required')
    require(summary['validation']['all_exact_accounting_gates_passed'] is True
            and summary['interpretation']['parent_H_gain'] is summary['interpretation']['parent_H_acquisition'] is False
            and summary['interpretation']['parent_H_protection'] is True
            and summary['interpretation']['new_confirmatory_hypothesis'] is False,
            'Exact accounting must preserve prior scientific verdicts and descriptive scope')
    require(counts['numeric_arrays'] == 14 and counts['measurement_records'] == 3600
            and counts['comparison_paths'] == 300 and counts['paired_trial_records'] == 120000
            and counts['corner_destination_effects'] == 32 * counts['eligible_both_Go_trials'],
            'Exact factor axes and eligibility-based corner counts required')
    residuals = summary['validation']['residuals']
    require(all(name in residuals and 0 <= residuals[name] <= cfg['numeric_tolerance'] for name in (
        'prior_primary_post_bookkeeping', 'saved_unit_mixture_geometry'))
        and len(summary['mixture_geometry']) == 6
        and all(len(family['seeds']) == 10 for family in summary['mixture_geometry']),
        'Original post ledger and all sixty saved mixture geometry paths required')
    with (folder / 'measurements.csv').open(encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)) == 15,
                'Exact fifteen unique descriptive CSV fields required')
        require(sum(1 for _ in reader) == 3600, 'All descriptive measurement rows required')
    validation_name = 'results/self_blend_factors_prefreeze_validation.json'
    tests = read(ROOT / validation_name)
    require(tests.get('status') == 'passed' and tests.get('frozen_source_hashes') == reg['source_files']
            and tests.get('parent_byte_gate') == reg['gate']
            and type(tests.get('actual_new_saved_factor_aggregation_before_freeze')) is int
            and tests['actual_new_saved_factor_aggregation_before_freeze'] == 0
            and type(tests.get('new_training_random_graph_or_optimization_calls')) is int
            and tests['new_training_random_graph_or_optimization_calls'] == 0
            and tests.get('synthetic_fixtures_only_before_freeze') is True,
            'Original frozen synthetic-only no-premature-aggregation snapshot required')
    receipts = tests.get('test_receipts'); development = tests.get('development_failure_artifacts')
    require(type(receipts) is list and len(receipts) == 1
            and receipts[0]['path'].startswith('results/self_blend_factors_prefreeze_tests/')
            and receipts[0]['path'].endswith('.xml'), 'One original final JUnit receipt required')
    require(type(development) is list and len(development) == 2
            and {item['path'] for item in development} == {STYLE_ZIP, STYLE_NOTE},
            'Exact original style-attempt ZIP and EOF finalization note required')
    prepaths = {validation_name, baseline_name}
    for item in receipts + development:
        require(sha(safe(ROOT, item['path'])) == item['sha256'], 'Prefreeze or development evidence bytes changed')
        prepaths.add(item['path'])
    committed(ROOT, reg['source_git_commit'], {name: sha(ROOT / name) for name in sorted(prepaths)})
    cases = ET.parse(ROOT / receipts[0]['path']).findall('.//testcase')
    require(bool(cases) and len(cases) == len({(c.get('classname'), c.get('name')) for c in cases})
            and all(c.find(key) is None for c in cases for key in ('failure', 'error', 'skipped')),
            'All original unique prefreeze cases must pass')
    classes = Counter(c.get('classname') for c in cases); new_count = sum(classes[name] for name in NEW_CLASSES)
    require(set(classes) == NEW_CLASSES | set(OLD_CLASSES) and all(classes[name] == count for name, count in OLD_CLASSES.items())
            and len(cases) == tests.get('unique_passed_tests') == receipts[0]['passed_unique_tests']
            and new_count == tests.get('new_unique_passed_tests')
            and tests.get('unchanged_related_regression_passed_tests') == 92
            and len(cases) == new_count + 92,
            'Dynamic new-test total and exact unchanged 45+47 regression partition required')
    style_gate = style_finalization_gate(ROOT, tests, reg, cases)
    publication = read(ROOT / publication_name)
    chronology = [baseline['verified_at_utc'], tests['checked_at_utc'], reg['registered_at_utc'],
        publication['checked_at_utc'], meta['started_at_utc'], meta['completed_at_utc'], verified['verified_at_utc']]
    times = list(map(utc, chronology))
    require(times == sorted(times) and times[2] < times[3] < times[4] and times[-1] <= datetime.now(timezone.utc),
            'Prefreeze/registration/private-publication/analysis/independent-verification chronology required')
    provenance = read(figures / 'provenance.json')
    require(provenance['verification_sha256'] == sha(folder / 'verification.json')
            and provenance['summary_sha256'] == sha(folder / 'summary.json')
            and provenance['arrays_sha256'] == sha(folder / 'arrays.npz')
            and provenance['plotted_values_sha256'] == sha(figures / 'plotted_values.json')
            and provenance['render_script_sha256'] == sha(ROOT / 'scripts/render_self_blend_factors.py')
            and set(provenance['figure_sha256']) == FIGURES
            and all(sha(figures / name) == digest for name, digest in provenance['figure_sha256'].items()),
            'All exact figures, renderer and plotted scientific data must be bound')
    status = read(ROOT / 'docs/repository-status.json')
    require(status.get('self_blend_factors_completed') is True
            and status['continuation_automation']['status'] == 'PAUSED'
            and status['current_findings_document'] == 'docs/self-blend-factors-findings.md'
            and status['level4_similarity_supported'] is status['shared_neural_mechanism_verified'] is False,
            'Completed descriptive report and bounded paused continuation required')
    presentation_name = 'results/self_blend_factors_presentation_validation.json'
    presentation = read(ROOT / presentation_name)
    require(presentation.get('status') == 'passed', 'Independent presentation QA required')
    required_docs = {'docs/self-blend-factors-findings.md', 'docs/self-blend-factors-reproduction.md'} | EVOLVING_INDEXES
    require(required_docs <= set(presentation['document_sha256']) | set(presentation['additional_file_sha256']),
            'All final documents and evolving indexes must be QA-bound')
    required_additional = {'scripts/render_self_blend_factors.py',
        'results/self_blend_factors_figures_v1/plotted_values.json', 'results/self_blend_factors_figures_v1/provenance.json'} | {
        'results/self_blend_factors_figures_v1/' + name for name in FIGURES}
    require(required_additional <= set(presentation['additional_file_sha256']), 'All figures and provenance must be presentation-checked')
    for key in ('document_sha256', 'additional_file_sha256'):
        require(all(sha(safe(ROOT, name)) == digest for name, digest in presentation[key].items()), 'Presentation evidence bytes changed')
    paths = set(reg['source_files']) | set(reg['parent_files']) | EVOLVING_INDEXES | prepaths | {
        registration_name, publication_name, presentation_name,
        'docs/self-blend-factors-findings.md', 'docs/self-blend-factors-reproduction.md',
        'scripts/check_self_blend_factors_archive.py', 'scripts/check_self_blend_factors_baseline.py',
        'scripts/render_self_blend_factors.py'}
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
        'prefreeze_passed_unique_tests_independently_counted': len(cases),
        'prefreeze_unique_tests_by_class': dict(sorted(classes.items())),
        'prefreeze_publication_analysis_verification_utc_order': chronology,
        'development_failure_artifact_bytes_preserved': development,
        'original_prefreeze_style_attempt_and_EOF_only_finalization': style_gate,
        'original_notice_count_corrected_by_rawbytes': True,
        'removed_trailing_LF_bytes': style_gate['removed_trailing_LF_bytes'],
        'counts': counts, 'scientific_verification_sha256': sha(folder / 'verification.json'),
        'current_file_sha256': {name: sha(safe(ROOT, name)) for name in sorted(paths)},
        'method': 'Prior-tree raw Git-blob baseline receipt, subsequent old-path diff, source-committed prefreeze/development bytes and independent scientific/presentation snapshots; no new learning, optimization or repeated scientific replay.',
        'notice': 'Retrospective symmetric single-update allocations and saved mixture geometry, not unique causal effects, new policy performance or a shared biological mechanism.'})
    print('Self-blend factor archive passed: original learning, descriptive arithmetic and final QA bytes sealed.')


if __name__ == '__main__':
    main()
