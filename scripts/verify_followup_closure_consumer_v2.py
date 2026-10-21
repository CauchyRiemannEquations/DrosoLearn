"""Strict additive consumer for the preserved historical human-summary schema.

The original failed consumer and every scientific result remain byte-identical.
Only its anchor-identity dispatch is adapted; all other independent comparisons
are called unchanged after separately registered source and parent byte checks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile


STUDY = 'followup_closure_consumer_v2'
CONFIG = 'configs/followup_closure_consumer_v2.json'
REGISTRATION = 'results/registration_followup_closure_consumer_v2.json'
PUBLICATION = 'results/followup_closure_consumer_v2_publication_proof.json'
REPOSITORY = 'https://github.com/CauchyRiemannEquations/DrosoLearn'
SOURCE_FILES = frozenset((CONFIG, 'docs/followup-closure-consumer-v2-protocol.md',
    'scripts/verify_followup_closure_consumer_v2.py', 'tests/test_followup_closure_consumer_v2.py'))
PARENT_ADDITIONS = frozenset((
    'results/followup_closure_publication_proof.json',
    'results/followup_closure_reproduction_v1/invocation.json',
    'results/followup_closure_reproduction_v1/objective_pathways_reverification.json',
    'results/followup_closure_v1/arrays.npz', 'results/followup_closure_v1/artifact_hashes.json',
    'results/followup_closure_v1/measurements.csv', 'results/followup_closure_v1/metadata.json',
    'results/followup_closure_v1/registration.json', 'results/followup_closure_v1/summary.json',
    'results/followup_closure_verification_attempt01.json',
    'results/followup_closure_verification_attempt01.zip',
    'results/human_shell_rng_v2/artifact_hashes.json', 'results/human_shell_rng_v2/metadata.json',
    'results/human_shell_rng_v2/registration.json', 'results/human_shell_rng_v2/summary.json',
    'results/human_shell_rng_v2/verification.json', 'results/registration_followup_closure_v1.json',
    'scripts/verify_human_shell.py', 'scripts/verify_human_shell_rng.py'))
FIXED = dict(study_id=STUDY, scope='strict_original_schema_consumer_correction_only',
    original_config='configs/followup_closure_v1.json',
    original_registration='results/registration_followup_closure_v1.json',
    original_publication_proof='results/followup_closure_publication_proof.json',
    original_result_dir='results/followup_closure_v1',
    original_verifier='scripts/verify_followup_closure.py',
    original_failure='results/followup_closure_verification_attempt01.json',
    human_anchor_id='human_comparison', human_study_id='human_shell_rng_v2',
    human_summary='results/human_shell_rng_v2/summary.json',
    human_verification='results/human_shell_rng_v2/verification.json',
    human_registration='results/human_shell_rng_v2/registration.json',
    human_metadata='results/human_shell_rng_v2/metadata.json',
    human_artifact_hashes='results/human_shell_rng_v2/artifact_hashes.json',
    human_completion_tag='human-shell-comparison-v2',
    output='results/followup_closure_consumer_v2/verification.json',
    scientific_result_changed=False, scientific_success_criteria_changed=False,
    original_source_or_primary_files_changed=False, new_learning_rng_optimization_effect_human_calls=0,
    interpretation='Original consumer failed on an absent historical field. Preserve failure and original bytes; '
    'require strict original SHA/registration/metadata/tag linkage instead. All other independent comparisons remain unchanged.')
HUMAN_COMPLETION_COMMIT = '3ee71a97fa9d6ffc7fce3bd4d7c3832437977171'
HUMAN_SUMMARY_FIELDS = frozenset(('status', 'human_test_people', 'model_seeds', 'mode_order',
    'condition_order', 'behavior_metric_order', 'human_metrics', 'behavior_family_comparisons',
    'conditional_family_mean_nll', 'conditional_family_mean_brier', 'H_similarity', 'H_prediction',
    'learning_qualification', 'bootstrap', 'test_modeled_trials', 'conditional_trials',
    'autonomous_trials', 'autonomous_nll_interpreted', 'mechanism_claim', 'educational_claim'))
HUMAN_PRIMARY_NAMES = frozenset(('cohort_audit.json', 'metadata.json', 'registration.json',
    'reused_parent_data.json', 'rng_keys.json', 'selection.json', 'summary.json',
    'trajectories.npz', 'trajectory_metrics.csv', 'witnesses.npz'))
HUMAN_PATHS = ['docs/human-shell-findings.md', 'docs/human-shell-rng-design-note.md',
    'docs/human-shell-reproduction.md', 'results/human_shell_comparison_archive_validation.json']
HUMAN_SOURCE_SCRIPTS = ('scripts/verify_human_shell.py', 'scripts/verify_human_shell_rng.py')
REGISTRATION_FIELDS = frozenset(('study_id', 'registration_type', 'registered_at_utc', 'source_git_commit',
    'config', 'config_sha256', 'source_files', 'parent_files', 'original_independent_failure_preserved',
    'scientific_success_criteria_changed', 'original_source_or_primary_files_changed',
    'new_learning_rng_optimization_effect_human_calls'))


def demand(value, label):
    if not value:
        raise ValueError(label)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            demand(key not in out, 'Duplicate JSON key: ' + key)
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    value = json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=unique, parse_constant=invalid)
    canonical(value)
    return value


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def safe(root, name, exists=True):
    demand(type(name) is str and name and '\\' not in name and ':' not in name
        and not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
        and str(PurePosixPath(name)) == name and not name.startswith('-')
        and all(x not in name for x in ('\n', '\r', '\0')), 'Contained POSIX file path required')
    root = Path(root).resolve(); path = (root / name).resolve()
    demand(path.is_relative_to(root) and (not exists or path.is_file()), 'Missing or escaping named file: ' + name)
    return path


def stamp(value):
    demand(type(value) is str, 'Explicit UTC timestamp required')
    result = datetime.fromisoformat(value)
    demand(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, 'UTC timestamp required')
    return result


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE)


def full_commit(value):
    demand(type(value) is str and re.fullmatch('[0-9a-f]{40}', value) is not None, 'Full literal Git commit required')
    return value


def ancestor(root, earlier, later):
    result = subprocess.run(['git', 'merge-base', '--is-ancestor', earlier, later], cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    demand(result.returncode == 0, 'Registered consumer Git ancestry differs')


def committed(root, commit, inventory):
    full_commit(commit)
    for name, expected in inventory.items():
        demand(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected) is not None, 'Literal SHA-256 required')
        demand(sha(safe(root, name)) == expected == hashlib.sha256(git(root, 'show', commit + ':' + name)).hexdigest(),
            'Frozen committed consumer bytes differ: ' + name)


def config_contract(cfg, original_registration):
    demand(type(cfg) is dict and set(cfg) == set(FIXED) | {'source_files', 'parent_files'}, 'Exact consumer config schema')
    for key, value in FIXED.items():
        demand(canonical(cfg[key]) == canonical(value), 'Fixed consumer correction scope: ' + key)
    old_sources = original_registration.get('source_files')
    demand(type(old_sources) is dict and len(old_sources) == 35 and not (set(old_sources) & PARENT_ADDITIONS),
        'Exact original frozen 35-source inventory')
    for key, expected in (('source_files', SOURCE_FILES), ('parent_files', set(old_sources) | PARENT_ADDITIONS)):
        demand(type(cfg[key]) is list and len(cfg[key]) == len(expected) and set(cfg[key]) == expected,
            'Exact unique consumer inventory: ' + key)
    demand(len(cfg['parent_files']) == 54, 'Exact 54 preserved consumer parents')


def check_failure(root, cfg, original_registration):
    failure = read(safe(root, cfg['original_failure']))
    demand(failure.get('study_id') == 'followup_closure_v1'
        and failure.get('status') == 'preserved_original_independent_consumer_failure'
        and type(failure.get('first_verifier_exit_code')) is int and failure['first_verifier_exit_code'] == 1
        and failure.get('first_verifier_receipt_written') is False
        and failure.get('failure') == 'ValueError: Original anchor evidence identities'
        and failure.get('source_git_commit') == original_registration['source_git_commit']
        and failure.get('original_registration_sha256') == sha(safe(root, cfg['original_registration']))
        and failure.get('original_frozen_source_files_unchanged') is True
        and failure.get('original_primary_files_unchanged') is True
        and failure.get('original_verifier_or_producer_rerun') is False
        and type(failure.get('new_learning_or_optimization_calls')) is int and failure['new_learning_or_optimization_calls'] == 0
        and failure.get('scientific_result_or_acquisition_gate_waived') is False,
        'Exact preserved original failed consumer event')
    archive_name = 'results/followup_closure_verification_attempt01.zip'
    demand(failure.get('archive') == archive_name and failure.get('archive_sha256') == sha(safe(root, archive_name)),
        'Preserved original failure ZIP SHA differs')
    members = failure.get('member_sha256')
    demand(type(members) is dict and bool(members), 'Preserved failure member SHA inventory')
    with zipfile.ZipFile(safe(root, archive_name)) as archive:
        names = archive.namelist()
        demand(len(names) == len(set(names)) and set(names) == set(members), 'Exact original failure ZIP members')
        for name in names:
            demand(hashlib.sha256(archive.read(name)).hexdigest() == members[name], 'Preserved failed attempt member changed')
            if name != 'original_tool_result.json':
                demand(name in cfg['parent_files'] and sha(safe(root, name)) == members[name],
                    'Original failed attempt source/result bytes differ')
    demand(not safe(root, cfg['original_result_dir'] + '/verification.json', exists=False).exists(),
        'Original failed consumer output must remain absent')
    return failure


def validate_consumer(root, registration_path, publication_path):
    """Validate the separate correction freeze before importing the old verifier."""
    root = Path(root).resolve()
    cfg = read(safe(root, CONFIG)); original = read(safe(root, FIXED['original_registration']))
    config_contract(cfg, original)
    demand(Path(registration_path).resolve() == safe(root, REGISTRATION)
        and Path(publication_path).resolve() == safe(root, PUBLICATION), 'Fixed new consumer registration and proof paths')
    reg = read(registration_path); proof = read(publication_path)
    demand(type(reg) is dict and set(reg) == REGISTRATION_FIELDS
        and reg.get('study_id') == STUDY and reg.get('registration_type') == 'retrospective_strict_schema_consumer_correction'
        and canonical(reg.get('config')) == canonical(cfg) and reg.get('config_sha256') == sha(safe(root, CONFIG))
        and reg.get('original_independent_failure_preserved') is True
        and reg.get('scientific_success_criteria_changed') is False
        and reg.get('original_source_or_primary_files_changed') is False
        and type(reg.get('new_learning_rng_optimization_effect_human_calls')) is int
        and reg['new_learning_rng_optimization_effect_human_calls'] == 0, 'Exact separately frozen correction registration')
    source = full_commit(reg.get('source_git_commit'))
    for group in ('source_files', 'parent_files'):
        expected = {name: sha(safe(root, name)) for name in cfg[group]}
        demand(reg.get(group) == expected, 'Registered consumer inventory bytes differ: ' + group)
        committed(root, source, expected)
    committed(root, full_commit(original.get('source_git_commit')), original['source_files'])
    ancestor(root, original['source_git_commit'], source)
    published = full_commit(proof.get('published_commit'))
    demand(proof.get('study_id') == STUDY and proof.get('registration_sha256') == sha(registration_path)
        and proof.get('source_commit') == source and proof.get('remote_bytes_match') is True
        and proof.get('visibility') == 'private'
        and proof.get('url') == REPOSITORY + '/blob/' + published + '/' + REGISTRATION,
        'Exact private consumer registration byte proof')
    ancestor(root, source, published); ancestor(root, published, 'HEAD')
    demand(git(root, 'show', published + ':' + REGISTRATION) == Path(registration_path).read_bytes(),
        'Published new consumer registration bytes differ')
    failure = check_failure(root, cfg, original)
    demand(stamp(failure['captured_at_utc']) < stamp(reg['registered_at_utc']) < stamp(proof['checked_at_utc'])
        <= datetime.now(timezone.utc), 'Failure-registration-publication chronological order')
    return cfg, reg, proof, failure


def check_human_anchor(evidence, anchor, cfg):
    """Require the one original schema, its registration, and five tagged JSONs."""
    demand(anchor == dict(id=cfg['human_anchor_id'], summary=cfg['human_summary'],
        verification=cfg['human_verification'], paths=HUMAN_PATHS), 'Exact single historical human anchor paths')
    summary = evidence.read(cfg['human_summary']); verified = evidence.read(cfg['human_verification'])
    registration = evidence.read(cfg['human_registration']); meta = evidence.read(cfg['human_metadata'])
    artifacts = evidence.read(cfg['human_artifact_hashes'])
    demand(type(summary) is dict and set(summary) == HUMAN_SUMMARY_FIELDS and 'study_id' not in summary
        and summary['status'] == 'completed', 'Exact original human summary schema without a study_id')
    demand(registration.get('study_id') == meta.get('study_id') == verified.get('study_id') == cfg['human_study_id']
        and meta.get('status') == meta.get('stage') == 'completed'
        and verified.get('status') == 'passed' and verified.get('verifier_imports_producer') is False
        and verified.get('verifier_modifies_original_results') is False
        and verified.get('all_frozen_inputs_match') is True
        and verified.get('parent_arithmetic_verification_passed') is True, 'Exact original human completed identity and verdict')
    demand(evidence.sha(cfg['human_registration']) == meta.get('registration_sha256') == verified.get('registration_sha256')
        and evidence.sha(cfg['human_summary']) == verified.get('summary_sha256')
        and meta.get('source_git_commit') == registration.get('source_git_commit')
        and meta.get('config_sha256') == registration.get('config_sha256')
        and registration.get('config', {}).get('study_id') == cfg['human_study_id'],
        'Original human summary-registration-metadata-source bindings')
    demand(type(artifacts) is dict and set(artifacts) == HUMAN_PRIMARY_NAMES
        and all(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None for value in artifacts.values()),
        'Exact ten original human primary artifact hashes')
    for field in ('human_summary', 'human_metadata', 'human_registration'):
        demand(evidence.sha(cfg[field]) == artifacts[Path(cfg[field]).name], 'Original human primary JSON SHA differs')
    source = full_commit(registration.get('source_git_commit'))
    config_name = 'configs/human_shell_rng_v2.json'
    sources = registration.get('source_files')
    demand(type(sources) is dict and sources.get(config_name) == registration['config_sha256'],
        'Original human registered configuration source SHA')
    config_bytes = evidence.git('show', source + ':' + config_name)
    demand(hashlib.sha256(config_bytes).hexdigest() == registration['config_sha256']
        and canonical(json.loads(config_bytes)) == canonical(registration['config']), 'Original human committed config bytes and values')
    for name in HUMAN_SOURCE_SCRIPTS:
        demand(name in sources, 'Original human schema verifier source must be registered')
        evidence.committed(source, name, sources[name])
    tag = evidence.git('rev-parse', '--verify', 'refs/tags/' + cfg['human_completion_tag'] + '^{commit}').decode().strip()
    demand(tag == HUMAN_COMPLETION_COMMIT, 'Original human completion tag moved')
    evidence.ancestor(source, tag); evidence.ancestor(tag, 'HEAD')
    for field in ('human_summary', 'human_verification', 'human_registration', 'human_metadata', 'human_artifact_hashes'):
        evidence.committed(tag, cfg[field], evidence.sha(cfg[field]))
    times = [stamp(registration['registered_at_utc']), stamp(meta['started_at_utc']),
        stamp(meta['completed_at_utc']), stamp(verified['verified_at_utc'])]
    demand(times[0] < times[1] <= times[2] <= times[3] <= datetime.now(timezone.utc), 'Original human chronology differs')
    for name in anchor['paths']:
        evidence.sha(name)


@contextmanager
def original_schema_consumer(verifier, cfg):
    """Replace one exact identity dispatch, restoring the original on all exits."""
    original = verifier.check_anchors
    def corrected(evidence, anchors):
        human = [anchor for anchor in anchors if anchor.get('id') == cfg['human_anchor_id']]
        demand(len(human) == 1, 'Exactly one original human anchor required')
        for anchor in anchors:
            if anchor['id'] == cfg['human_anchor_id']:
                check_human_anchor(evidence, anchor, cfg)
            else:
                demand(anchor.get('summary') != cfg['human_summary']
                    and anchor.get('verification') != cfg['human_verification'], 'Human schema cannot be relabeled as another anchor')
                original(evidence, [anchor])
    verifier.check_anchors = corrected
    try:
        yield
    finally:
        verifier.check_anchors = original


def load_original(root, cfg):
    path = safe(root, cfg['original_verifier'])
    spec = importlib.util.spec_from_file_location('frozen_followup_closure_consumer', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def output_path(root, out, cfg):
    root = Path(root).resolve(); path = Path(out).resolve()
    demand(path.is_relative_to(root) and not path.exists() and path.parent != path,
        'Exclusive new workspace-contained consumer output required')
    original_dir = safe(root, cfg['original_result_dir'], exists=False)
    demand(not path.is_relative_to(original_dir)
        and path.relative_to(root).as_posix() not in set(cfg['parent_files']) | SOURCE_FILES | {REGISTRATION, PUBLICATION},
        'Consumer output cannot replace frozen sources or original results')
    return path


def correction_record(root, cfg, reg):
    return dict(study_id=STUDY, scope=cfg['scope'], registration=REGISTRATION,
        registration_sha256=sha(safe(root, REGISTRATION)), publication_proof=PUBLICATION,
        publication_proof_sha256=sha(safe(root, PUBLICATION)), source_git_commit=reg['source_git_commit'],
        source_files=4, parent_files=54, original_failure=cfg['original_failure'],
        original_failure_sha256=sha(safe(root, cfg['original_failure'])),
        original_failure_archive='results/followup_closure_verification_attempt01.zip',
        original_failure_archive_sha256=sha(safe(root, 'results/followup_closure_verification_attempt01.zip')),
        human_anchor_id=cfg['human_anchor_id'], human_original_schema_checked=True,
        human_original_tag_primary_files_checked=5, other_anchor_checks_unchanged=True,
        all_other_independent_checks_unchanged=True, original_source_or_primary_files_changed=False,
        scientific_success_criteria_changed=False, new_learning_rng_optimization_effect_human_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    for name in ('config', 'registration', 'result-dir', 'publication-proof', 'out',
                 'consumer-registration', 'consumer-publication-proof'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    cfg, reg, proof, failure = validate_consumer(root, args.consumer_registration, args.consumer_publication_proof)
    demand(Path(__file__).resolve() == safe(root, 'scripts/verify_followup_closure_consumer_v2.py'),
        'Execute the registered consumer source in this checkout')
    for value, key in ((args.config, 'original_config'), (args.registration, 'original_registration'),
                       (args.publication_proof, 'original_publication_proof'), (args.result_dir, 'original_result_dir')):
        demand(value.resolve() == safe(root, cfg[key], exists=key != 'original_result_dir'), 'Fixed original consumer argument: ' + key)
    out = output_path(root, args.out, cfg)
    verifier = load_original(root, cfg)
    sys.path.insert(0, str(root / 'src'))
    from drosolearn.closure_envelope import verify_envelope, write_new
    original_cfg, original_reg, metadata = verify_envelope(root, args.config, args.registration, args.result_dir, args.publication_proof)
    with verifier.np.load(args.result_dir / 'arrays.npz', allow_pickle=False) as handle:
        arrays = {name: handle[name] for name in handle.files}
    with (args.result_dir / 'measurements.csv').open(encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        demand(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)) == 10,
            'Ten unique closure CSV fields')
        rows = list(reader)
    summary = read(args.result_dir / 'summary.json')
    with original_schema_consumer(verifier, cfg):
        checks = verifier.verify_science(original_cfg, root, arrays, rows, summary)
    demand(canonical(metadata['counts']) == canonical(summary['counts']), 'Closure metadata and independent summary counts')
    receipt = read(safe(root, original_cfg['reproduction_receipt']))
    demand(stamp(receipt['verified_at_utc']) <= stamp(metadata['started_at_utc']) <= stamp(metadata['completed_at_utc']),
        'Saved-only reproduction precedes final synthesis')
    # Catch input mutation during the independent comparison before publishing success.
    validate_consumer(root, args.consumer_registration, args.consumer_publication_proof)
    report = dict(status='passed', study_id=original_cfg['study_id'], verified_at_utc=datetime.now(timezone.utc).isoformat(),
        registration_sha256=sha(args.registration), summary_sha256=sha(args.result_dir / 'summary.json'),
        scientific_producer_imported=False, counts=checks, consumer_correction=correction_record(root, cfg, reg))
    out.parent.mkdir(parents=True, exist_ok=True); write_new(out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
