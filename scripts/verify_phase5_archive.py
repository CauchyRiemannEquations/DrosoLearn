"""Final read-only archive checks; execute only after root's final report review.

This helper runs no learner or effect test. It writes one new exclusive archive
receipt only after every literal-byte/provenance/document check has passed.
"""
from __future__ import annotations
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
PYTHON = Path(sys.executable)
BASE_TAG = 'human-shell-comparison-v2'
BASE_COMMIT = '3ee71a97fa9d6ffc7fce3bd4d7c3832437977171'
ALLOWED = {'README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'}
STUDIES = ('shell_credit_v1', 'shell_behavior_v1', 'pattern_rule_v1',
           'interference_order_v1', 'phase5_synthesis_v1')
SOURCE_COUNTS = (41, 41, 41, 47, 41)
SOURCE_CLOSURE_SLUGS = ('shell-credit', 'shell-behavior', 'pattern-rule',
                       'interference-order', 'phase5-report')
EXTERNAL_CACHE = BASE / 'work' / 'human_comparison_candidates' / 'tnu'
FINAL_DOCS = ('docs/phase5-final-report.md', 'docs/phase5-evidence-appendix.md', 'docs/phase5-reproduction.md')
FIGURE_DIRECTORY = 'results/phase5_synthesis_figures_v1'
PRIMARY = {'registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json'}
OUT = ROOT / 'results' / 'phase5_synthesis_archive_validation.json'
RAW_PARENT = 'results/interference_order_v1/arrays.npz'
PARTS_ADAPTER = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(name, root=ROOT):
    require(type(name) is str and name and '\\' not in name and ':' not in name
            and not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts,
            'Contained POSIX file name required: ' + str(name))
    result = (root / name).resolve()
    require(result.is_relative_to(root.resolve()) and result.is_file(), 'Missing contained file: ' + name)
    return result


def read(name):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(safe(name).read_text(encoding='utf8'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def digest_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha(name):
    return digest_path(safe(name))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def ancestor(commit, reference='HEAD'):
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, reference], cwd=ROOT, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def original_git_bytes(commit, name, expected_sha):
    if name == RAW_PARENT:
        require(PARTS_ADAPTER is not None, 'Independent exact-parent storage adapter must be loaded')
        return PARTS_ADAPTER.joined_parent_from_git(ROOT, commit, expected_sha)
    return git('show', commit + ':' + name)


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, 'UTC timestamp required')
    return result


def frozen_group(reg, key, checked, committed=False):
    inventory = reg.get(key)
    require(type(inventory) is dict, 'Explicit registered file inventory required: ' + key)
    for name, expected in inventory.items():
        require(sha(name) == expected, 'Registered bytes changed: ' + name)
        checked[name] = expected
        if committed:
            blob = original_git_bytes(reg['source_git_commit'], name, expected)
            require(hashlib.sha256(blob).hexdigest() == expected, 'Source commit blob changed: ' + name)


def test_receipt(name, checked, expected_source=None, before=None):
    receipt = read(name)
    require(receipt.get('passed') is True and receipt.get('source_unchanged') is True
            and receipt.get('exit_code') == 0, 'Passing unchanged-source tests required: ' + name)
    require(type(receipt.get('passed_tests')) is int and receipt['passed_tests'] > 0, 'Positive exact test count required')
    counts = receipt.get('counts_by_file')
    require(type(counts) is dict and counts and all(type(n) is int and n > 0 for n in counts.values())
            and sum(counts.values()) == receipt['passed_tests'], 'Test-count receipt disagrees with per-file counts')
    require(type(receipt.get('command')) is list and '-m' in receipt['command']
            and 'pytest' in receipt['command'], 'Actual pytest command receipt required')
    source_before, source_after = receipt.get('source_before'), receipt.get('source_after')
    require(type(source_before) is dict and source_before and source_before == source_after,
            'Test-time source maps must agree exactly')
    if expected_source is not None:
        require(source_after == expected_source, 'Test receipt must cover the exact scientific source closure')
    for source, expected in source_after.items():
        require(sha(source) == expected, 'Tested source changed after tests: ' + source)
        checked[source] = expected
    started, completed = stamp(receipt['started_at_utc']), stamp(receipt['completed_at_utc'])
    require(started <= completed <= datetime.now(timezone.utc), 'Valid completed-test chronology required')
    if before is not None:
        require(completed < before, 'Scientific registration must follow final prefreeze tests')
    checked[name] = sha(name)
    return {'receipt': name, 'passed_tests': receipt['passed_tests'], 'counts_by_file': counts,
            'started_at_utc': receipt['started_at_utc'], 'completed_at_utc': receipt['completed_at_utc'],
            'source_files': len(source_after), 'sha256': checked[name]}


def check_doc_links(name):
    text = safe(name).read_text(encoding='utf8')
    count = 0
    for angle, plain in re.findall(r'\]\((?:<([^>]+)>|([^)]*))\)', text):
        target = angle or plain
        target = target.split(' "', 1)[0].strip()
        if not target or target.startswith('#') or urlsplit(target).scheme:
            continue
        local = unquote(target.split('#', 1)[0].split('?', 1)[0])
        destination = (safe(name).parent / local).resolve()
        require(destination.is_relative_to(ROOT.resolve()) and destination.exists(), 'Broken local documentation link: ' + name + ' -> ' + target)
        count += 1
    return count


def check_ledger_tags(ledger, checked):
    """Bind every historical ledger tag to its literal numerical evidence.

    Reports may have later transparent documentation corrections, so only the
    independent verification and optional primary result are bound to tags.
    """
    require(len(ledger) == 28, 'All 28 historical completion tags required')
    bindings = []
    for study in ledger:
        tag = study['tag']
        require(type(tag) is str and bool(tag) and not tag.startswith('-'), 'Explicit safe completion tag required')
        commit = git('rev-parse', '--verify', 'refs/tags/' + tag + '^{commit}').decode().strip()
        ancestor(commit)
        names = [study['verification']]
        if study.get('primary') is not None:
            names.append(study['primary'])
        records = {}
        for name in names:
            expected = sha(name)
            literal = original_git_bytes(commit, name, expected)
            require(hashlib.sha256(literal).hexdigest() == expected,
                    'Completion-tag verification/primary bytes differ: ' + study['study_id'] + ' -> ' + name)
            checked[name] = expected
            records[name] = expected
        bindings.append({'study_id': study['study_id'], 'tag': tag, 'tag_commit': commit,
                         'verification_and_primary_sha256': records,
                         'later_corrected_report_not_bound_to_historical_tag': True})
    return bindings


def main():
    global PARTS_ADAPTER
    require(not OUT.exists(), 'Exclusive new archive output required')
    require(git('rev-parse', BASE_TAG).decode().strip() == BASE_COMMIT, 'Pinned pre-followup release required')
    old_changed = git('diff', '--name-only', '--diff-filter=MDR', BASE_COMMIT).decode().splitlines()
    require(set(old_changed) == ALLOWED, 'Only three historical indexes may change')
    preservation_run = subprocess.run([str(PYTHON), str(ROOT / 'scripts' / 'check_prephase5_baseline.py')],
                                      cwd=ROOT, capture_output=True, text=True, encoding='utf8', check=True)
    preservation = json.loads(preservation_run.stdout)
    require(preservation['passed'] is True and preservation['failures'] == []
            and preservation['tag_commit'] == BASE_COMMIT and preservation['compared_files'] > 18000
            and {item['path'] for item in preservation['excluded_files']} == ALLOWED,
            'Literal pre-followup Git preservation failed')

    # Frozen generic envelope only: no scientific producer is imported or run.
    sys.path.insert(0, str(ROOT / 'src'))
    from drosolearn.prospective import validate_registration, expected_source, parent_gate
    adapter_name = 'scripts/verify_phase5_report.py'
    synthesis_registration = read('results/registration_phase5_synthesis_v1.json')
    require(sha(adapter_name) == synthesis_registration['source_files'][adapter_name], 'Frozen independent storage adapter bytes required')
    adapter_spec = importlib.util.spec_from_file_location('archive_independent_phase5_storage', safe(adapter_name))
    PARTS_ADAPTER = importlib.util.module_from_spec(adapter_spec)
    adapter_spec.loader.exec_module(PARTS_ADAPTER)
    checked, studies, receipts, source_union = {}, {}, {}, {}
    now = datetime.now(timezone.utc)
    for index, (study, source_count) in enumerate(zip(STUDIES, SOURCE_COUNTS)):
        cfg_name = 'configs/' + study + '.json'
        reg_name = 'results/registration_' + study + '.json'
        result = 'results/' + study + '/'
        proof_name = 'results/' + study.removesuffix('_v1') + '_prefreeze_evidence.json'
        cfg, reg, proof = read(cfg_name), read(reg_name), read(proof_name)
        meta, verification, summary = read(result + 'metadata.json'), read(result + 'verification.json'), read(result + 'summary.json')
        require(cfg['study_id'] == reg['study_id'] == meta['study_id'] == verification['study_id'] == study
                and (summary.get('study_id') == study or
                     study == 'shell_behavior_v1' and 'study_id' not in summary),
                'Consistent study identity required')
        require(reg['config'] == cfg and len(reg['source_files']) == source_count
                and set(reg['source_files']) == expected_source(ROOT, study), 'Exact source closure required')
        cache = EXTERNAL_CACHE if cfg.get('external_manifest') is not None else None
        context = PARTS_ADAPTER.independent_parent_parts(ROOT) if study == 'phase5_synthesis_v1' else nullcontext()
        with context:
            validate_registration(ROOT, cfg, safe(cfg_name), reg, cache)
        require(reg['gate'] == parent_gate(ROOT, cfg), 'Historical parent gate changed')
        require(meta['status'] == 'completed' and meta['stage'] == 'completed'
                and verification['status'] == 'passed', 'Completed independently verified studies required')
        require((safe(result + 'registration.json')).read_bytes() == safe(reg_name).read_bytes(), 'Literal registration copy changed')
        reg_sha = sha(reg_name)
        require(meta['registration_sha256'] == verification['registration_sha256'] == proof['registration_sha256'] == reg_sha
                and meta['config_sha256'] == reg['config_sha256'] == sha(cfg_name), 'Registration/config identity mismatch')
        require(verification['summary_sha256'] == sha(result + 'summary.json')
                and meta['counts'] == summary['counts'], 'Verified summary and metadata disagree')
        require(all(meta.get(key) is False for key in ('new_human_experiment', 'shared_neural_mechanism_tested', 'education_effect_tested')),
                'Scientific interpretation boundary changed')
        timeline = [reg['registered_at_utc'], proof['checked_at_utc'], meta['started_at_utc'],
                    meta['completed_at_utc'], verification['verified_at_utc']]
        times = list(map(stamp, timeline))
        require(times[0] < times[1] < times[2] <= times[3] <= times[4] <= now,
                'Registration must be remotely checked before actual computation')
        require(proof['remote_bytes_match'] is True and proof['source_commit'] == reg['source_git_commit']
                and proof.get('visibility') == 'private', 'Published private preregistration proof required')
        require(proof['url'].endswith('/blob/' + proof['published_commit'] + '/' + reg_name), 'Published-registration URL identity mismatch')
        ancestor(reg['source_git_commit'], proof['published_commit']); ancestor(proof['published_commit'])
        require(hashlib.sha256(git('show', proof['published_commit'] + ':' + reg_name)).hexdigest() == reg_sha,
                'Published registration Git bytes differ')
        for group in ('source_files', 'parent_files', 'graph_files', 'input_files'):
            frozen_group(reg, group, checked, committed=group in ('source_files', 'parent_files'))
        source_union.update(reg['source_files'])
        external = reg['external_files']
        require(type(external) is dict, 'Explicit external inventory required')
        for filename, spec in external.items():
            path = safe(filename, EXTERNAL_CACHE)
            require(path.stat().st_size == spec['bytes'] and digest_path(path) == spec['sha256'], 'Original external raw bytes changed')
        artifacts = read(result + 'artifact_hashes.json')
        require(set(artifacts) == PRIMARY, 'Exact five primary artifacts required')
        for filename, expected in artifacts.items():
            name = result + filename
            require(sha(name) == expected, 'Primary result bytes changed: ' + name)
            checked[name] = expected
        receipt_name = ('results/phase5_synthesis_v1_final_test_validation.json' if study == 'phase5_synthesis_v1'
                        else 'results/' + study + '_test_validation.json')
        receipts[study] = test_receipt(receipt_name, checked, reg['source_files'], before=times[0])
        for name in (cfg_name, reg_name, proof_name, result + 'verification.json', result + 'artifact_hashes.json'):
            checked[name] = sha(name)
        tag = study.replace('_', '-')
        if study != 'phase5_synthesis_v1':
            tag_commit = git('rev-parse', tag).decode().strip(); ancestor(tag_commit)
            for name in PRIMARY | {'artifact_hashes.json', 'verification.json'}:
                path = result + name
                require(hashlib.sha256(original_git_bytes(tag_commit, path, sha(path))).hexdigest() == sha(path),
                        'Completed study tag result differs: ' + path)
        else:
            tag_commit = None  # This archive is committed before the final completion tag.
        studies[study] = {'source_commit': reg['source_git_commit'], 'published_registration_commit': proof['published_commit'],
                          'source_files': len(reg['source_files']), 'parent_files': len(reg['parent_files']),
                          'input_files': len(reg['input_files']), 'graph_files': len(reg['graph_files']),
                          'external_files': len(external), 'chronology': timeline,
                          'completed_tag': tag if tag_commit else None, 'completed_tag_commit': tag_commit,
                          'independent_verification_sha256': sha(result + 'verification.json')}

    union_name = 'results/prephase5_final_test_validation.json'
    union = test_receipt(union_name, checked, source_union)
    require(union['passed_tests'] == 643, 'Final union must retain 199 old and 444 new unique tests')
    union_counts = read(union_name)['counts_by_file']
    registered_counts = {}
    for receipt in receipts.values():
        for name, count in receipt['counts_by_file'].items():
            require(name not in registered_counts or registered_counts[name] == count, 'A shared test file changed its count')
            registered_counts[name] = count
    require(set(registered_counts) <= set(union_counts)
            and all(union_counts[name] == count for name, count in registered_counts.items()), 'Final tests must include every new study test exactly once')
    require(sum(registered_counts.values()) == 444, 'New bounded package must include 444 unique tests')

    preliminary_name = 'results/phase5_synthesis_v1_test_validation.json'
    preliminary = read(preliminary_name)
    final_synthesis_source = read('results/registration_phase5_synthesis_v1.json')['source_files']
    old_source = preliminary.get('source_after')
    require(preliminary.get('passed') is True and preliminary.get('exit_code') == 0
            and preliminary.get('source_unchanged') is True and preliminary.get('passed_tests') == 159
            and preliminary.get('source_before') == old_source and type(old_source) is dict
            and set(old_source) == set(final_synthesis_source)
            and sum(preliminary['counts_by_file'].values()) == 159,
            'Preserved preliminary Phase5 test receipt must remain internally valid')
    historical_differences = {name for name, checksum in old_source.items()
                              if checksum != final_synthesis_source[name]}
    require(historical_differences == {'configs/phase5_synthesis_v1.json'},
            'Only reviewed preliminary wording may differ from final tested config')
    require(stamp(preliminary['completed_at_utc']) < stamp(receipts['phase5_synthesis_v1']['started_at_utc']),
            'Final wording must be tested after the preserved preliminary receipt')
    checked[preliminary_name] = sha(preliminary_name)

    summary_name = 'results/phase5_synthesis_v1/summary.json'
    synthesis = read(summary_name); status = read('docs/repository-status.json')
    require(synthesis['status'] == 'completed' and len(synthesis['studies']) == 28
            and len(synthesis['questions']) == 5, 'Bounded 28-study five-question synthesis required')
    ledger_tag_bindings = check_ledger_tags(synthesis['studies'], checked)
    require(synthesis['interpretation'] == {'bounded_synthesis_complete': True, 'all_possible_phase2_phase3_work_complete': False,
                                           'level4_similarity_supported': False, 'shared_neural_mechanism_claim': False,
                                           'education_effect_claim': False}, 'Bounded synthesis interpretation changed')
    require(all(synthesis['counts'][key] == 0 for key in ('new_learning_updates', 'new_rng_draws', 'new_human_experiments', 'new_effect_tests')),
            'Retrospective synthesis cannot add a new experiment or effect test')
    require(status.get('current_phase') == 'phase5_bounded_synthesis_completed'
            and status.get('phase5_started') is True and status.get('phase5_completed') is True
            and status.get('bounded_phase5_deliverables_completed') is True
            and status.get('all_phase2_phase3_tasks_completed') is False
            and status.get('level4_similarity_supported') is False
            and status.get('shared_neural_mechanism_verified') is False
            and status.get('education_effect_tested') is False
            and status['continuation_automation']['status'] == 'PAUSED', 'Final status must report bounded completion and paused continuation')
    require(status['human_model_similarity_tested'] is True and status['direct_level4_test_completed'] is True,
            'Actual negative direct human comparison must remain recorded')
    human = read('results/human_shell_rng_v2/summary.json')
    require(human['H_similarity']['supported'] is False and human['H_prediction']['supported'] is False
            and read('results/human_task_structure_v1/audit.json')['gate']['direct_human_model_similarity_allowed'] is False,
            'Original negative human comparison and historical closed source gate must be preserved')
    final_tags = status['study_tags']
    for study in STUDIES[:-1]:
        require(final_tags[study] == study.replace('_', '-'), 'All four completed followup tags must be indexed')

    presentation_name = FIGURE_DIRECTORY + '/provenance.json'
    presentation = read(presentation_name)
    require(presentation['source_registration_sha256'] == sha('results/registration_phase5_synthesis_v1.json')
            and presentation['summary_sha256'] == sha(summary_name)
            and presentation['verification_sha256'] == sha('results/phase5_synthesis_v1/verification.json'), 'Figure evidence provenance mismatch')
    plot_source = presentation.get('plot_source', 'src/drosolearn/phase5_report.py')
    require(plot_source == 'src/drosolearn/phase5_report.py'
            and presentation['plot_source_sha256'] == sha(plot_source), 'Frozen plot renderer source differs')
    require(type(presentation.get('files')) is dict and 'plotted_values.json' in presentation['files'], 'Literal plot values and figure inventory required')
    dimensions = {}
    for name, expected in presentation['files'].items():
        relative = FIGURE_DIRECTORY + '/' + name
        require(sha(relative) == expected, 'Presentation bytes changed: ' + relative)
        if name.lower().endswith('.png'):
            with Image.open(safe(relative)) as image:
                image.verify()
            with Image.open(safe(relative)) as image:
                require(image.width > 0 and image.height > 0, 'Nonempty image required')
                dimensions[name] = [image.width, image.height]
        checked[relative] = expected
    require(bool(dimensions), 'At least one scientific summary figure required')
    checked[presentation_name] = sha(presentation_name)
    link_counts = {}
    for name in (*FINAL_DOCS, *sorted(ALLOWED)):
        link_counts[name] = check_doc_links(name); checked[name] = sha(name)
    require(safe(FINAL_DOCS[0]).stat().st_size > 0 and safe(FINAL_DOCS[1]).stat().st_size > 0
            and safe(FINAL_DOCS[2]).stat().st_size > 0, 'Nonempty final report, evidence appendix and reproduction guide required')

    presentation_validation_name = 'results/phase5_presentation_validation.json'
    presentation_validation = read(presentation_validation_name)
    require(presentation_validation['status'] == 'passed'
            and presentation_validation['numeric_plot_values_match_saved_evidence'] is True
            and presentation_validation['all_registered_conditions_retained'] is True
            and presentation_validation['dimensions'] == dimensions,
            'Separate numeric and root visual presentation validation required')
    for filename, expected in presentation_validation['figure_sha256'].items():
        require(sha(FIGURE_DIRECTORY + '/' + filename) == expected, 'Visually inspected figure changed')
    checked[presentation_validation_name] = sha(presentation_validation_name)
    archive_engineering_name = 'results/phase5_archive_engineering_evidence.json'
    archive_engineering = read(archive_engineering_name)
    require(archive_engineering['original_scientific_results_changed'] is False
            and archive_engineering['correction_scope'] == 'archive_helper_only', 'Archive helper correction scope')
    checked[archive_engineering_name] = sha(archive_engineering_name)
    for helper_source in ('scripts/verify_phase5_archive.py', 'scripts/check_prephase5_baseline.py'):
        checked[helper_source] = sha(helper_source)
    whitespace_name = 'results/prephase5_whitespace_check.json'
    whitespace = read(whitespace_name)
    require(whitespace.get('status') == 'passed' and whitespace.get('original_plain_diff_false_positive') is True
            and whitespace.get('returncode', whitespace.get('exit_code')) == 0,
            'Corrected CRLF-aware whitespace command must pass')
    command = whitespace.get('command')
    require(type(command) is list and command == ['git', '-c',
            'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', BASE_COMMIT, '--check'],
            'Record exact CRLF-aware engineering check command')
    checked[whitespace_name] = sha(whitespace_name)

    race_name = 'results/interference_order_prefreeze_engineering_evidence.json'
    race = read(race_name)
    order_reg = read('results/registration_interference_order_v1.json')
    order_meta = read('results/interference_order_v1/metadata.json')
    require(race['status'] == 'corrected_before_first_scientific_run'
            and race['source_commit'] == order_reg['source_git_commit']
            and race['registration_sha256'] == sha('results/registration_interference_order_v1.json')
            and race['registration_unchanged'] is True
            and race['scientific_draws_or_feedback_before_correction'] == 0
            and race['all_registered_source_bytes_restored'] is True
            and stamp(race['checked_at_utc']) < stamp(order_meta['started_at_utc']),
            'Source-freeze race must be restored without feedback before the first scientific run')
    require(race_name in read('results/registration_phase5_synthesis_v1.json')['input_files'],
            'Preserved source-freeze race evidence must be a frozen synthesis input')
    for change in race['changes']:
        name = change['file']
        require(change['matches_source_commit_and_registration'] is True
                and sha(name) == change['restored_frozen_sha256'] == order_reg['source_files'][name]
                and hashlib.sha256(git('show', order_reg['source_git_commit'] + ':' + name)).hexdigest() == sha(name),
                'Restored immutable source-freeze bytes differ')
    checked[race_name] = sha(race_name)
    storage_doc = 'docs/result-storage.md'
    checked[storage_doc] = sha(storage_doc)
    link_counts[storage_doc] = check_doc_links(storage_doc)

    output = {
        'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Final bounded-synthesis archival preservation, prospective provenance, unchanged-source tests, presentation and documentation; no new learning or hypothesis tests',
        'previous_literal_preservation': preservation, 'old_changed_files': old_changed,
        'allowed_evolving_indexes': sorted(ALLOWED), 'studies': studies, 'new_study_test_receipts': receipts,
        'all_28_historical_completion_tag_bindings': ledger_tag_bindings,
        'final_union_tests': union, 'all_frozen_source_input_parent_graph_external_and_primary_hashes_checked': True,
        'passed_unique_tests': 643, 'previous_unique_tests': 199, 'new_package_unique_tests': 444,
        'phase5_final_test_receipt': receipts['phase5_synthesis_v1']['receipt'],
        'phase5_preliminary_test_history': {'receipt': preliminary_name, 'sha256': sha(preliminary_name),
            'passed_tests': 159, 'historical_different_source_paths': sorted(historical_differences),
            'all_other_40_source_hashes_match_final_registration': True,
            'preliminary_config_sha_not_misrepresented_as_final': True},
        'source_and_parent_git_blobs_checked': True, 'remote_registration_proofs_precede_all_actual_computation': True,
        'one_exact_parent_lossless_git_parts': {'original_path': RAW_PARENT,
            'original_bytes': safe(RAW_PARENT).stat().st_size, 'original_sha256': sha(RAW_PARENT),
            'manifest': RAW_PARENT + '.parts.json', 'ordered_parts': [RAW_PARENT + '.part001', RAW_PARENT + '.part002'],
            'exact_join_matches_original_artifact_and_local_raw': True,
            'all_other_committed_byte_gates_unchanged': True},
        'all_old_baseline_files_literal_preserved_except_three_indexes': True,
        'presentation': {'png_dimensions': dimensions, 'provenance_sha256': sha(presentation_name),
                         'literal_plot_values_sha256': sha(FIGURE_DIRECTORY + '/plotted_values.json'),
                         'visual_inspection_not_claimed_by_this_helper': True},
        'documentation_local_link_counts': link_counts,
        'engineering_events': [{'event': 'Plain whitespace check treated preserved CRLF line endings as trailing whitespace',
                                'corrected_command_receipt': whitespace_name, 'corrected_check_passed': True,
                                'frozen_scientific_source_or_results_changed': False},
                               {'event': 'Verifier-only unregistered extra guard edits raced with the source freeze',
                                'preserved_evidence': race_name, 'restored_before_first_scientific_run': True,
                                'original_registration_unchanged': True, 'scientific_draws_or_feedback_before_correction': 0},
                               {'event': 'Original array exceeded ordinary Git single-file size limit by 47300 bytes',
                                'storage_documentation': storage_doc, 'original_result_bytes_unchanged': True,
                                'two_lossless_git_parts_match_original_sha256': True,
                                'new_learning_or_effect_tests': 0},
                               {'event': 'Two scope/wording clarifications followed the preliminary Phase5 tests',
                                'preliminary_receipt_preserved': preliminary_name,
                                'final_wording_tested_receipt': receipts['phase5_synthesis_v1']['receipt'],
                                'only_config_wording_source_hash_differs': True,
                                'all_final_registered_sources_tested_unchanged': True}],
        'bounded_phase5_synthesis_completed': True, 'all_possible_phase2_phase3_work_complete': False,
        'human_model_similarity_supported': False, 'shared_neural_mechanism_verified': False,
        'education_effect_tested': False, 'continuation_automation_status': 'PAUSED',
        'new_learning_updates': 0, 'new_rng_draws': 0, 'new_human_experiments': 0, 'new_effect_tests': 0,
        'checked_file_sha256': checked,
        'notice': 'Prior literal baseline and independent numerical verifications are reused; no old learner is rerun. This archive does not hash itself or claim future final-tag/remote state. Final completion-tag and remote equality are checked after committing this archive.',
    }
    with OUT.open('x', encoding='utf8', newline='\n') as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps({'status': 'passed', 'previous_literal_files': preservation['compared_files'],
                      'studies': len(studies), 'final_unique_passed_tests': union['passed_tests'],
                      'checked_files': len(checked), 'archive': str(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.cache is not None: EXTERNAL_CACHE = args.cache.resolve()
    if args.out is not None: OUT = args.out.resolve()
    main()
