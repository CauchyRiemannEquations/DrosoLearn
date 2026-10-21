"""Independent literal evidence verification for bounded project closure.

This verifier imports no scientific producer, learner, graph or optimizer.
It checks saved evidence and exact provenance rather than rerunning cohorts.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

import numpy as np

STUDY_IDS = ('representation_update_v1', 'update_budget_v1', 'spillover_tradeoff_v1',
    'signed_spillover_v1', 'selective_spillover_v1', 'normalization_loss_v1', 'cone_frontier_v1',
    'balanced_learning_v1', 'balanced_pathways_v1', 'gated_learning_v1', 'gated_pathways_v1',
    'gated_factors_v1', 'self_blend_learning_v1', 'self_blend_factors_v1',
    'probability_alignment_v1', 'objective_learning_v1', 'objective_pathways_v1')
LEARNING_IDS = frozenset(('representation_update_v1', 'update_budget_v1', 'spillover_tradeoff_v1',
    'selective_spillover_v1', 'balanced_learning_v1', 'gated_learning_v1',
    'self_blend_learning_v1', 'objective_learning_v1'))
OPTIMIZED_IDS = frozenset(('cone_frontier_v1', 'balanced_learning_v1', 'gated_learning_v1',
    'self_blend_learning_v1', 'probability_alignment_v1', 'objective_learning_v1'))
VERDICTS = ('supported', 'not_supported', 'not_assessable', 'descriptive', 'untested')
INTEGRITY_COLUMNS = ('metadata_completed', 'independent_passed', 'primary_hashes_match',
    'release_tag_bytes_match', 'source_input_hashes_match', 'private_publication_verified',
    'chronology_ordered', 'registered_hypothesis_literals_match')
EVOLVING = frozenset(('README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'))
INTERPRETATION = dict(bounded_local_project_completed=True, original_phase5_preserved=True,
    all_failures_and_negative_decisions_preserved=True, new_effect_or_learning_claims=False,
    pooled_effect_estimates_computed=False, old_test_counts_summed_as_independent=False,
    novel_generalization_supported_by_this_synthesis=False, human_similarity_supported=False,
    shared_neural_mechanism_verified=False, education_effect_tested=False,
    automatic_continuation_enabled=False, long_term_questions_retained_without_automatic_execution=True)


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            demand(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=unique, parse_constant=invalid)
    canonical(value)
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def contained(root, name, allowed=None):
    demand(type(name) is str and bool(name) and '\\' not in name and ':' not in name
           and not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
           and str(PurePosixPath(name)) == name and not name.startswith('-')
           and all(char not in name for char in ('\n', '\r', '\0')), 'Contained POSIX file path required')
    if allowed is not None:
        demand(name in allowed, 'Evidence path absent from frozen inventory: ' + name)
    root = Path(root).resolve()
    path = (root / name).resolve()
    demand(path.is_relative_to(root) and path.is_file(), 'Missing or escaping evidence file: ' + name)
    return path


def resolve_pointer(document, address):
    demand(type(address) is str and (address == '' or address.startswith('/')), 'RFC6901 pointer required')
    current = document
    if address == '':
        return current
    for raw in address[1:].split('/'):
        demand(re.search(r'~(?![01])', raw) is None, 'Invalid RFC6901 escape')
        key = raw.replace('~1', '/').replace('~0', '~')
        if type(current) is list:
            demand(re.fullmatch(r'0|[1-9][0-9]*', key) is not None, 'Canonical array index required')
            index = int(key)
            demand(index < len(current), 'Evidence array index out of range')
            current = current[index]
        elif type(current) is dict:
            demand(key in current, 'Missing evidence key: ' + key)
            current = current[key]
        else:
            raise ValueError('Evidence pointer traverses a scalar')
    return current


def stamp(value):
    demand(type(value) is str, 'Explicit UTC timestamp string required')
    result = datetime.fromisoformat(value)
    demand(result.tzinfo is not None and result.utcoffset().total_seconds() == 0,
           'UTC timestamp required')
    return result


def manifest_contract(manifest):
    demand(type(manifest) is dict and type(manifest.get('manifest_version')) is int and manifest['manifest_version'] == 1
           and manifest.get('study_id') == 'followup_closure_v1', 'Fixed bounded closure manifest')
    studies = manifest.get('studies')
    demand(type(studies) is list and [r.get('id') for r in studies] == list(STUDY_IDS),
           'All seventeen followup studies required in fixed order')
    required = {'id', 'kind', 'completion_tag', 'completion_commit', 'source_commit', 'registration',
        'summary', 'verification', 'archive', 'protocol', 'reproduction', 'cohort', 'result_dir',
        'publication_proof', 'artifact_hashes', 'metadata', 'findings', 'raw_files', 'hypothesis_observations',
        'original_new_optimization', 'preservation_files'}
    for study in studies:
        demand(type(study) is dict and set(study) == required, 'Exact historical study evidence schema')
        demand(study['kind'] == ('actual_learning' if study['id'] in LEARNING_IDS else 'saved_record_analysis'),
               'Actual learning and saved analysis stay distinct')
        demand(type(study['original_new_optimization']) is bool
               and study['original_new_optimization'] == (study['id'] in OPTIMIZED_IDS),
               'Original optimization work and current zero-work closure stay distinct')
        demand(type(study['preservation_files']) is list
               and len(study['preservation_files']) == len(set(study['preservation_files'])),
               'Explicit preserved failure or presentation-event file paths')
        for key in ('completion_commit', 'source_commit'):
            demand(type(study[key]) is str and re.fullmatch('[0-9a-f]{40}', study[key]) is not None,
                   'Literal full Git commit required')
        demand(type(study['completion_tag']) is str and bool(study['completion_tag'])
               and not study['completion_tag'].startswith('-'), 'Safe completion tag required')
        demand(type(study['cohort']) is dict and set(study['cohort']) ==
               {'model_seeds', 'training_trials', 'parent_learning_result'}, 'Recorded cohort identity required')
        demand(type(study['raw_files']) is list and len(study['raw_files']) == len(set(study['raw_files'])) == 5,
               'Exactly five historical primary artifacts')
        demand(type(study['hypothesis_observations']) is list, 'Explicit original hypothesis literal list')
        for item in study['hypothesis_observations']:
            demand(type(item) is dict and set(item) == {'pointer', 'expected'}, 'Exact original verdict pointer schema')
    claims = manifest.get('claims')
    demand(type(claims) is list and bool(claims), 'Nonempty bounded claim ledger')
    demand(len({c.get('id') for c in claims}) == len(claims), 'Unique closure claim ids')
    for claim in claims:
        demand(type(claim) is dict and set(claim) ==
               {'id', 'question', 'verdict', 'level', 'statement', 'limit', 'evidence'}, 'Exact closure claim schema')
        demand(type(claim['id']) is str and bool(claim['id']) and type(claim['question']) is int
               and 1 <= claim['question'] <= 5 and type(claim['level']) is int
               and 0 <= claim['level'] <= 5 and claim['verdict'] in VERDICTS, 'Valid bounded claim coordinates')
        demand(not (claim['level'] >= 3 and claim['verdict'] == 'supported'),
               'No new positive generalization/human/common-mechanism claim from this closure')
        demand(all(type(claim[k]) is str and bool(claim[k]) for k in ('statement', 'limit'))
               and type(claim['evidence']) is list and bool(claim['evidence']), 'Every claim needs meaning, limits and evidence')
        for item in claim['evidence']:
            demand(type(item) is dict and set(item) == {'file', 'pointer', 'expected'}, 'Exact frozen literal observation schema')
            canonical(item['expected'])
    anchors = manifest.get('retained_anchors')
    demand(type(anchors) is list and bool(anchors) and len({a.get('id') for a in anchors}) == len(anchors),
           'Unique nonempty retained original anchors')
    for item in anchors:
        demand(type(item) is dict and set(item) == {'id', 'summary', 'verification', 'paths'}
               and type(item['paths']) is list and bool(item['paths']), 'Exact original anchor path schema')
        demand(item['verification'] is None or item['summary'] is not None,
               'Anchor verification must name corresponding original evidence')
    canonical(manifest)


class SavedEvidence:
    """Read-only cached hashes and Git blobs; no scientific model is constructed."""
    def __init__(self, root, frozen=None):
        self.root = Path(root).resolve()
        self.frozen = frozen
        self.hashes = {}
        self.git_hashes = {}

    def path(self, name):
        return contained(self.root, name, self.frozen)

    def sha(self, name):
        if name not in self.hashes:
            self.hashes[name] = digest(self.path(name))
        return self.hashes[name]

    def read(self, name):
        return read_json(self.path(name))

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root)

    def ancestor(self, earlier, later):
        result = subprocess.run(['git', 'merge-base', '--is-ancestor', earlier, later], cwd=self.root,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        demand(result.returncode == 0, 'Historical Git ancestry changed')

    def committed(self, commit, name, expected):
        key = (commit, name)
        if key not in self.git_hashes:
            self.git_hashes[key] = hashlib.sha256(self.git('show', commit + ':' + name)).hexdigest()
        demand(self.sha(name) == self.git_hashes[key] == expected, 'Original committed bytes differ: ' + name)


def check_study(evidence, study):
    """Validate the eight fixed integrity coordinates, independently of the producer."""
    reg = evidence.read(study['registration']); meta = evidence.read(study['metadata'])
    summary = evidence.read(study['summary']); verified = evidence.read(study['verification'])
    proof = evidence.read(study['publication_proof']); archive = evidence.read(study['archive'])
    artifacts = evidence.read(study['artifact_hashes'])
    demand(reg.get('study_id') == meta.get('study_id') == summary.get('study_id') ==
           verified.get('study_id') == study['id'] and meta.get('status') == meta.get('stage') == 'completed',
           'Exact historical completed metadata and study identity')
    demand(verified.get('status') == 'passed' and verified.get('scientific_producer_imported') is False
           and archive.get('status') == 'passed', 'Historical independent and archive checks must pass')
    demand(evidence.sha(study['registration']) == meta.get('registration_sha256') == verified.get('registration_sha256')
           and reg.get('source_git_commit') == study['source_commit'] == meta.get('source_git_commit')
           and meta.get('config_sha256') == reg.get('config_sha256')
           and verified.get('summary_sha256') == evidence.sha(study['summary'])
           and canonical(meta.get('counts')) == canonical(summary.get('counts')), 'Original evidence identities and count maps')
    expected_names = {Path(name).name for name in study['raw_files']}
    demand(type(artifacts) is dict and set(artifacts) == expected_names
           and expected_names == {'registration.json', 'metadata.json', 'arrays.npz', 'summary.json',
               'measurements.csv.gz' if study['id'] == 'objective_learning_v1' else 'measurements.csv'},
           'Exact historical primary artifact receipt')
    demand((evidence.path(study['result_dir'] + '/registration.json')).read_bytes() ==
           evidence.path(study['registration']).read_bytes(), 'Original registration copy differs')
    for name in study['raw_files']:
        demand(evidence.sha(name) == artifacts[Path(name).name], 'Original primary artifact changed')
    tag_commit = evidence.git('rev-parse', '--verify', 'refs/tags/' + study['completion_tag'] + '^{commit}').decode().strip()
    demand(tag_commit == study['completion_commit'], 'Historical completion tag moved')
    evidence.ancestor(study['source_commit'], tag_commit); evidence.ancestor(tag_commit, 'HEAD')
    for key in ('registration', 'summary', 'verification', 'archive', 'protocol', 'reproduction',
                'findings', 'publication_proof', 'artifact_hashes', 'metadata'):
        evidence.committed(tag_commit, study[key], evidence.sha(study[key]))
    for name in study['raw_files']:
        evidence.committed(tag_commit, name, evidence.sha(name))
    for name in study['preservation_files']:
        evidence.committed(tag_commit, name, evidence.sha(name))
    for key in ('current_file_sha256', 'checked_file_sha256'):
        preserved = archive.get(key, {})
        demand(type(preserved) is dict, 'Original non-index archive snapshot inventory')
        for name, expected in preserved.items():
            if name not in EVOLVING:
                demand(evidence.sha(name) == expected, 'Original archived failure or non-index evidence changed')
    groups = ('source_files', 'parent_files', 'input_files', 'graph_files', 'external_files')
    source_count = input_count = 0
    for group in groups:
        entries = reg.get(group, {})
        demand(type(entries) is dict, 'Original frozen hash inventory required: ' + group)
        if group == 'source_files':
            demand(bool(entries), 'Nonempty original scientific source inventory')
        for name, expected in entries.items():
            demand(name not in EVOLVING and type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected) is not None,
                   'Explicit immutable historical inventory member')
            evidence.committed(study['source_commit'], name, expected)
        if group == 'source_files':
            source_count = len(entries)
        else:
            input_count += len(entries)
    published = proof.get('published_commit')
    demand(type(published) is str and re.fullmatch('[0-9a-f]{40}', published) is not None
           and proof.get('study_id') == study['id'] and proof.get('source_commit') == study['source_commit']
           and proof.get('registration_sha256') == evidence.sha(study['registration'])
           and proof.get('remote_bytes_match') is True and proof.get('visibility') == 'private'
           and meta.get('publication_proof') == study['publication_proof']
           and meta.get('publication_proof_sha256') == evidence.sha(study['publication_proof']),
           'Actual original private registration byte proof')
    evidence.ancestor(study['source_commit'], published); evidence.ancestor(published, tag_commit)
    evidence.committed(published, study['registration'], evidence.sha(study['registration']))
    expected_url = 'https://github.com/CauchyRiemannEquations/DrosoLearn/blob/' + published + '/' + study['registration']
    demand(proof.get('url') == expected_url, 'Original exact private publication URL')
    times = [stamp(reg['registered_at_utc']), stamp(proof['checked_at_utc']), stamp(meta['started_at_utc']),
             stamp(meta['completed_at_utc']), stamp(verified['verified_at_utc'])]
    demand(times[0] < times[1] < times[2] <= times[3] <= times[4] <= datetime.now(timezone.utc),
           'Original registration-publication-computation-verification order')
    for item in study['hypothesis_observations']:
        demand(canonical(resolve_pointer(summary, item['pointer'])) == canonical(item['expected']),
               'Original registered hypothesis literal changed')
    prior_cfg = reg.get('config', {})
    for name in ('model_seeds', 'training_trials'):
        demand(canonical(study['cohort'][name]) == canonical(prior_cfg.get(name)),
               'Original registered computational cohort identity')
    return {'source_files': source_count, 'input_files': input_count,
            'flags': [True] * len(INTEGRITY_COLUMNS)}


def validate_science(cfg, allow_fixture=False):
    expected = {'study_id': 'followup_closure_v1', 'scope': 'retrospective_bounded_followup_synthesis',
        'module': 'drosolearn.followup_closure', 'protocol': 'docs/followup-closure-protocol.md',
        'evidence_manifest': 'configs/followup_closure_evidence_v1.json',
        'baseline_tag': 'objective-pathways-v1', 'baseline_commit': 'beec7eb421b4574231f74ea0b2ede4dbc99e1ecc',
        'reproduction_receipt': 'results/followup_closure_reproduction_v1/objective_pathways_reverification.json'}
    fixture = cfg.get('scope') == 'integration_test'
    demand(not fixture or allow_fixture, 'Invented integration fixture cannot enter actual closure')
    for key, value in expected.items():
        if fixture and key == 'scope':
            continue
        demand(cfg.get(key) == value, 'Fixed bounded closure choice: ' + key)
    demand(cfg.get('study_order') == list(STUDY_IDS) and cfg.get('new_confirmatory_hypothesis') is False
           and cfg.get('new_effect_aggregation') is False and type(cfg.get('numeric_tolerance')) in (int, float)
           and cfg['numeric_tolerance'] == 0. and cfg.get('external_manifest') is None,
           'Fixed complete study order, exact literals and no new effect aggregation')
    interpretation = cfg.get('interpretation', {})
    demand(all(interpretation.get(key) is False for key in ('new_training', 'new_seed_draws',
        'new_direction_optimization', 'new_human_experiment', 'shared_neural_mechanism_claim',
        'education_effect_claim', 'automatic_continuation_enabled'))
        and interpretation.get('completed_scope_is_bounded') is True
        and interpretation.get('long_term_questions_retained') is True,
        'No new computation or stronger biological claim in bounded completion')
    canonical(cfg)


def check_anchors(evidence, anchors):
    for anchor in anchors:
        if anchor['summary'] is not None:
            evidence.sha(anchor['summary'])
        if anchor['verification'] is not None:
            summary = evidence.read(anchor['summary']); verification = evidence.read(anchor['verification'])
            demand(verification.get('status') == 'passed', 'Original anchor independent verdict must pass')
            demand(summary.get('study_id') == verification.get('study_id'), 'Original anchor evidence identities')
            if 'summary_sha256' in verification:
                demand(verification['summary_sha256'] == evidence.sha(anchor['summary']), 'Original anchor summary SHA differs')
        for name in anchor['paths']:
            evidence.sha(name)


def expected_reproduction(cfg, evidence):
    old_name = 'results/objective_pathways_v1/verification.json'
    new_name = cfg['reproduction_receipt']
    demand(new_name != old_name, 'Reproduction must use a separate new receipt')
    original = evidence.read(old_name); repeated = evidence.read(new_name)
    demand(original.get('status') == repeated.get('status') == 'passed'
           and repeated.get('study_id') == original.get('study_id') == 'objective_pathways_v1'
           and repeated.get('scientific_producer_imported') is False
           and repeated.get('new_helper_imported') is False
           and canonical(repeated.get('counts')) == canonical(original.get('counts'))
           and repeated.get('summary_sha256') == original.get('summary_sha256')
           and repeated.get('registration_sha256') == original.get('registration_sha256'),
           'Separate actual saved-only numerical recheck must match original evidence exactly')
    proof = evidence.read('results/followup_closure_publication_proof.json')
    demand(proof.get('study_id') == 'followup_closure_v1' and proof.get('remote_bytes_match') is True
           and proof.get('visibility') == 'private', 'Closure private publication proof precedes reproduction')
    demand(stamp(original['verified_at_utc']) < stamp(proof['checked_at_utc']) <
           stamp(repeated['verified_at_utc']) <= datetime.now(timezone.utc), 'Actual new read-only reproduction chronology')
    return {'receipt_path': new_name, 'sha256': evidence.sha(new_name),
        'original_verification_sha256': evidence.sha(old_name), 'summary_sha256': repeated['summary_sha256'],
        'counts': original['counts'], 'separate_new_receipt': True, 'past_full_science_rerun': False}


def expected_evidence(cfg, root):
    manifest = read_json(contained(root, cfg['evidence_manifest']))
    manifest_contract(manifest)
    frozen = set(cfg.get('source_files', [])) | set(cfg.get('parent_files', []))
    direct = {cfg['evidence_manifest']}
    for study in manifest['studies']:
        direct.update(study[key] for key in ('registration', 'summary', 'verification', 'archive', 'protocol',
            'reproduction', 'publication_proof', 'artifact_hashes', 'metadata', 'findings'))
        direct.update(study['raw_files'])
        direct.update(study['preservation_files'])
    for anchor in manifest['retained_anchors']:
        direct.update(path for path in (anchor['summary'], anchor['verification'], *anchor['paths']) if path is not None)
    direct.update(item['file'] for claim in manifest['claims'] for item in claim['evidence'])
    demand(direct <= frozen, 'All directly consumed closure evidence is frozen')
    evidence = SavedEvidence(root)
    audited = []
    flags = []
    for study in manifest['studies']:
        result = check_study(evidence, study)
        flags.append(result['flags'])
        audited.append({key: study[key] for key in ('id', 'kind', 'completion_tag', 'completion_commit', 'source_commit')}
            | {'registration_sha256': evidence.sha(study['registration']), 'summary_sha256': evidence.sha(study['summary']),
               'verification_sha256': evidence.sha(study['verification']), 'archive_sha256': evidence.sha(study['archive']),
               'integrity_flags': result['flags'], 'cohort': study['cohort'],
               'hypothesis_observations': study['hypothesis_observations']})
    check_anchors(evidence, manifest['retained_anchors'])
    rows, claims, codes = [], [], []
    for index, claim in enumerate(manifest['claims']):
        hashes = {}
        for observation in claim['evidence']:
            actual = resolve_pointer(evidence.read(observation['file']), observation['pointer'])
            demand(canonical(actual) == canonical(observation['expected']), 'Declared literal evidence differs: ' + claim['id'])
            hashes[observation['file']] = evidence.sha(observation['file'])
        claims.append(dict(claim, matched=True, source_sha256=hashes))
        codes.append([index, claim['question'], claim['level'], VERDICTS.index(claim['verdict'])])
        rows.append({'claim_id': claim['id'], 'question': claim['question'], 'level': claim['level'],
            'verdict': claim['verdict'], 'statement': claim['statement'], 'interpretation_limit': claim['limit'],
            'source_count': len(hashes), 'evidence_json': canonical(claim['evidence']),
            'source_sha256_json': canonical(hashes), 'matched': 1})
    arrays = {'study_integrity_flags': np.asarray(flags, dtype=bool), 'claim_ledger': np.asarray(codes, dtype=np.int64)}
    reproduction = expected_reproduction(cfg, evidence)
    summary = {'study_id': cfg['study_id'], 'status': 'completed', 'scope': cfg['scope'],
        'studies': audited, 'claims': claims, 'retained_anchors': manifest['retained_anchors'],
        'counts': {'studies': 17, 'claims': len(claims),
            'literal_observations': sum(len(claim['evidence']) for claim in manifest['claims']),
            'actual_learning_registrations': 8, 'saved_record_registrations': 9,
            'numeric_arrays': 2, 'measurement_records': len(rows), 'new_training_trials': 0,
            'new_model_graph_rng_optimization_calls': 0, 'new_effect_tests': 0, 'new_human_experiments': 0,
            'past_full_science_reruns': 0, 'saved_record_independent_rechecks': 1},
        'reproduction': reproduction, 'interpretation': dict(INTERPRETATION),
        'integrity_flag_order': list(INTEGRITY_COLUMNS), 'verdict_code_order': list(VERDICTS)}
    return arrays, rows, summary


def verify_science(cfg, root, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    expected_arrays, expected_rows, expected_summary = expected_evidence(cfg, root)
    demand(set(arrays) == set(expected_arrays), 'Exactly two literal closure ledgers')
    for name, expected in expected_arrays.items():
        actual = np.asarray(arrays[name])
        demand(actual.dtype == expected.dtype and np.array_equal(actual, expected), 'Exact closure ledger differs: ' + name)
    demand(len(rows) == len(expected_rows), 'Complete ordered closure claim CSV')
    for actual, expected in zip(rows, expected_rows):
        demand(tuple(actual) == tuple(expected), 'Exact ten closure CSV column names and order')
        if all(type(value) is str for value in actual.values()):
            demand(actual == {key: str(value) for key, value in expected.items()}, 'Literal closure CSV differs')
        else:
            demand(canonical(actual) == canonical(expected), 'Typed literal closure CSV differs')
    demand(canonical(summary) == canonical(expected_summary), 'Closure summary differs from independent literal evidence')
    return {'verified_followup_studies': 17, 'actual_learning_registrations': 8, 'saved_record_registrations': 9,
        'independently_checked_claims': len(expected_rows),
        'literal_observations': expected_summary['counts']['literal_observations'], 'exact_numeric_ledgers': 2,
        'all_completion_tags_source_inputs_primary_hashes_and_chronology_checked': True,
        'all_historical_hypothesis_decisions_preserved': True,
        'separate_saved_only_independent_reproduction_checked': True,
        'old_test_counts_not_summed_as_independent': True, 'new_training_trials': 0,
        'new_model_graph_rng_optimization_calls': 0, 'new_effect_tests': 0,
        'new_human_experiments': 0, 'past_full_science_reruns': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--publication-proof', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'src'))
    from drosolearn.closure_envelope import verify_envelope, write_new
    cfg, registration, metadata = verify_envelope(root, args.config, args.registration, args.result_dir, args.publication_proof)
    with np.load(args.result_dir / 'arrays.npz', allow_pickle=False) as handle:
        arrays = {name: handle[name] for name in handle.files}
    with (args.result_dir / 'measurements.csv').open(encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        demand(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)) == 10,
               'Ten unique closure CSV fields')
        rows = list(reader)
    summary = read_json(args.result_dir / 'summary.json')
    checks = verify_science(cfg, root, arrays, rows, summary)
    demand(canonical(metadata['counts']) == canonical(summary['counts']), 'Closure metadata and independent summary counts')
    receipt = read_json(contained(root, cfg['reproduction_receipt']))
    demand(stamp(receipt['verified_at_utc']) <= stamp(metadata['started_at_utc']) <= stamp(metadata['completed_at_utc']),
           'Saved-only reproduction precedes final synthesis')
    report = {'status': 'passed', 'study_id': cfg['study_id'], 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'registration_sha256': digest(args.registration), 'summary_sha256': digest(args.result_dir / 'summary.json'),
        'scientific_producer_imported': False, 'counts': checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
