"""Independently check a frozen evidence map; do not replay behavior or estimate effects."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ('quantity_ratio', 'repeated_experience', 'generalization_transfer',
            'reward_reversal', 'learning_interference', 'learning_order')
FLAGS = ('direct_level4_test_completed', 'shared_neural_mechanism_verified',
         'education_effect_tested', 'all_phase2_phase3_tasks_completed')
SOURCES = ('configs/phase4_synthesis_v1.json', 'docs/phase4-synthesis-protocol.md',
           'docs/phase4-synthesis.md', 'scripts/audit_phase4_synthesis.py',
           'scripts/verify_phase4_synthesis.py', 'tests/test_phase4_synthesis.py',
           'tests/test_phase4_synthesis_verifier.py')
EVIDENCE = (
    'data/human/honore_noel_2016/source-manifest.json',
    'data/human/honore_noel_2016/schema-audit.json',
    'data/human/task_structure_evidence_20261015/source-manifest.json',
    'data/human/stolz_2022_semantics_evidence/source-manifest.json',
    'results/human_task_structure_v1/audit.json',
    'results/human_task_structure_v1/verification.json',
    'results/human_osf_origin_v1/verification.json',
    'results/human_semantics_v1/verification.json',
    'results/conditional_transition_v1/metrics.json',
    'results/conditional_transition_v1/verification.json',
    'results/conditional_transition_v1/artifact_hashes.json',
    'results/stochastic_feedback_network_v1/summary.json',
    'results/stochastic_feedback_network_v1/verification.json',
    'results/learning_ablation_v1/summary.json',
    'results/learning_ablation_v1/verification.json',
    'results/time_mixture_v1/metrics.json',
    'results/time_mixture_v1/verification.json',
    'docs/time-mixture-findings.md', 'docs/human-data-readiness.md',
    'docs/human-semantics-findings.md', 'docs/human-task-structure-findings.md',
    'docs/encoding-findings.md', 'docs/spacing-findings.md',
    'docs/learning-ablation-findings-v2.md',
    'docs/stochastic-feedback-continuation-findings.md')
ARTIFACTS = ('registration.json', 'synthesis.json', 'metadata.json')
SEMANTIC_SCOPE = 'independent_fixed_code_counts_no_mapping_selection_or_performance_analysis'
SCOPE = 'evidence_synthesis_and_measurement_consistency_not_new_human_effect_estimation'


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        ensure(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read(file):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(file.read_text(encoding='utf8'), object_pairs_hook=_pairs,
                      parse_constant=invalid)


def evidence_path(root, relative):
    ensure(type(relative) is str and relative and '\\' not in relative,
           'A normalized repository-relative path is required')
    candidate = Path(relative)
    ensure(not candidate.is_absolute() and all(p not in ('.', '..') for p in candidate.parts),
           'Path escapes the registered repository')
    file = (root / candidate).resolve()
    ensure(file.is_relative_to(root.resolve()) and file.is_file(), 'Registered file is absent')
    return file


def utc(value):
    ensure(type(value) is str, 'UTC timestamp string required')
    result = datetime.fromisoformat(value)
    ensure(result.tzinfo is not None and result.utcoffset().total_seconds() == 0,
           'Explicit UTC timestamps required')
    return result


def integer(value):
    ensure(type(value) is int and value >= 0, 'Nonnegative integer counts required')
    return value


def validate_config(cfg):
    ensure(cfg['study_id'] == 'phase4_synthesis_v1' and cfg['scope'] == SCOPE,
           'Different synthesis or scientific scope')
    ensure([f['id'] for f in cfg['features']] == list(FEATURES), 'All six ordered features required')
    ensure(all(cfg.get(flag) is False for flag in FLAGS), 'Untested claims must remain false')
    ensure(cfg['source_files'] == list(SOURCES) and cfg['evidence_files'] == list(EVIDENCE),
           'Exact source and evidence inventories required')
    for feature in cfg['features']:
        ensure(set(feature) == {'id', 'metric', 'unit', 'human', 'model', 'difference', 'unlock'},
               'Incomplete comparison feature')
        ensure(set(feature['human']) == {'honore', 'markovic', 'openneuro'}, 'Three human sources required')
        ensure(feature['human']['openneuro'] == 'semantic_gate_closed', 'Closed semantic gate upgraded')
        ensure(all(type(feature[k]) is str and feature[k].strip()
                   for k in ('metric', 'unit', 'model', 'difference', 'unlock')),
               'Explicit metric, unit, difference and unlocking evidence required')
        ensure(all(type(v) is str and v.strip() for v in feature['human'].values()),
               'Empty human-source comparison cell')


def inspect_support(metrics, prior):
    """Recount only saved integer partitions, without rates, pooling or raw trial replay."""
    ensure(metrics['study_id'] == 'conditional_transition_v1', 'Wrong conditional measurement')
    ensure(all(metrics[k] is False for k in ('pool_group_or_species_estimates',
               'averaged_stratum_estimates', 'causal_effect_estimated')), 'Effect estimation is out of scope')
    groups, stream_ids = {}, set()
    totals = dict(streams=0, input_trials=0, candidates=0, eligible=0, excluded=0, strata=0,
                  feedback_cells=0, human_records=0, model_streams=0)
    for stream in metrics['streams']:
        name, kind = stream['stream_id'], stream['kind']
        ensure(type(name) is str and name not in stream_ids, 'Repeated or invalid stream identity')
        stream_ids.add(name)
        ensure(kind in ('human', 'model') and name.startswith(kind + '/'), 'Observation unit mismatch')
        group = 'human' if kind == 'human' else name.rsplit('/', 1)[-1]
        ensure(group in ('human', 'deterministic', 'probabilistic', 'uninformative'), 'Unknown model environment')
        expected_cells = (set(itertools.product(('initial', 'transfer'), ('1', '2'), (0, 1)))
                          if kind == 'human' else
                          {('phase1', 'larger', 0), ('phase1', 'larger', 1),
                           ('phase2', 'smaller', 0), ('phase2', 'smaller', 1)})
        observed_cells = []
        unit = groups.setdefault(group, dict(streams=0, strata=0, both_feedback=0,
            feedback_0_only=0, feedback_1_only=0, neither=0,
            feedback_0_denominator=0, feedback_1_denominator=0, eligible=0))
        unit['streams'] += 1
        sums = {f: {k: 0 for k in ('denominator', 'stay', 'switch')} for f in ('0', '1')}
        for cell in stream['strata']:
            ensure(type(cell['choice']) is int, 'Boolean is not a choice code')
            observed_cells.append((cell['phase'], cell['state'], cell['choice']))
            ensure(set(cell['by_feedback']) == {'0', '1'}, 'Both feedback cells required')
            ns = []
            for f in ('0', '1'):
                values = cell['by_feedback'][f]
                for key in sums[f]:
                    sums[f][key] += integer(values[key])
                ensure(values['stay'] + values['switch'] == values['denominator'], 'Cell partition changed')
                ns.append(values['denominator'])
                unit['feedback_' + f + '_denominator'] += values['denominator']
            status = ('both_feedback' if ns[0] and ns[1] else 'feedback_0_only' if ns[0]
                      else 'feedback_1_only' if ns[1] else 'neither')
            ensure(cell['support_status'] == status and cell['common_support'] is bool(ns[0] and ns[1]),
                   'Support status does not follow the original denominators')
            unit[status] += 1
        ensure(len(observed_cells) == len(expected_cells) and set(observed_cells) == expected_cells,
               'Missing, repeated or relabeled designed stratum')
        ensure(set(stream['by_feedback']) == {'0', '1'}, 'Original record feedback scope changed')
        for f in ('0', '1'):
            for key, value in sums[f].items():
                ensure(value == integer(stream['by_feedback'][f][key]), 'Original by-feedback partition differs')
        eligible = integer(stream['eligible'])
        ensure(sum(sums[f]['denominator'] for f in ('0', '1')) == eligible, 'Eligible pair partition differs')
        ensure(integer(stream['eligible']) + integer(stream['excluded']) == integer(stream['candidates'])
               and stream['candidates'] == integer(stream['trials']) - 1, 'Original adjacent-pair accounting differs')
        unit['strata'] += len(observed_cells)
        unit['eligible'] += eligible
        totals['streams'] += 1
        totals['human_records' if kind == 'human' else 'model_streams'] += 1
        for source, target in (('trials', 'input_trials'), ('candidates', 'candidates'),
                               ('eligible', 'eligible'), ('excluded', 'excluded')):
            totals[target] += stream[source]
        totals['strata'] += len(observed_cells)
        totals['feedback_cells'] += 2 * len(observed_cells)
    ensure(set(groups) == {'human', 'deterministic', 'probabilistic', 'uninformative'}, 'Missing source environment')
    ensure(groups['human']['streams'] == 25 and all(groups[g]['streams'] == 80
           for g in ('deterministic', 'probabilistic', 'uninformative')), 'Original record scope reduced')
    ensure(totals == {k: prior['counts'][k] for k in totals}, 'Previous independently verified count differs')
    ensure(totals == dict(streams=265, input_trials=217000, candidates=216735, eligible=213050,
           excluded=3685, strata=1160, feedback_cells=2320, human_records=25, model_streams=240),
           'Full original measurement scope required')
    return groups, totals


def _check(root, config, registration, result_dir, checked_at):
    cfg, reg = read(config), read(registration)
    validate_config(cfg)
    ensure(reg['study_id'] == cfg['study_id'] and reg['config'] == cfg
           and reg['config_sha256'] == digest(config), 'Configuration is not the registered bytes')
    for key, names in (('source_files', SOURCES), ('evidence_files', EVIDENCE)):
        ensure(set(reg[key]) == set(names), 'Registered inventory changed')
        ensure(all(re.fullmatch(r'[0-9a-f]{64}', reg[key][n]) and
               digest(evidence_path(root, n)) == reg[key][n] for n in names), 'Frozen bytes changed')
    ensure(digest(Path(__file__)) == reg['source_files']['scripts/verify_phase4_synthesis.py'],
           'Executed verifier differs from its frozen source')
    commit = reg['source_git_commit']
    ensure(type(commit) is str and re.fullmatch(r'[0-9a-f]{40}', commit), 'Full source commit required')
    ensure(subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=root,
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0, 'Source commit is not an ancestor')
    for name in SOURCES:
        blob = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=root)
        ensure(hashlib.sha256(blob).hexdigest() == reg['source_files'][name], 'Source Git blob differs')
    ensure((result_dir / 'registration.json').read_bytes() == registration.read_bytes(), 'Result registration copy changed')
    ensure(all(p.name in set(ARTIFACTS) | {'artifact_hashes.json', 'verification.json'} and p.is_file()
               for p in result_dir.iterdir()), 'Unexpected synthesis result artifact or directory')
    hashes = read(result_dir / 'artifact_hashes.json')
    ensure(set(hashes) == set(ARTIFACTS) and all(digest(result_dir/n) == hashes[n] for n in ARTIFACTS),
           'Exact three primary artifact hashes required')
    metadata = read(result_dir / 'metadata.json')
    ensure(metadata['study_id'] == cfg['study_id'] and metadata['status'] == 'completed'
           and metadata['source_git_commit'] == commit and metadata['config_sha256'] == digest(config)
           and metadata['registration_sha256'] == digest(registration), 'Execution provenance differs')
    ensure(utc(reg['registered_at_utc']) <= utc(metadata['started_at_utc']) <=
           utc(metadata['completed_at_utc']) <= checked_at, 'Prospective UTC ordering failed')
    def evidence(name):
        ensure(name in EVIDENCE, 'Undeclared evidence read')
        return read(evidence_path(root, name))
    for study in ('human_task_structure_v1', 'human_osf_origin_v1', 'human_semantics_v1',
                  'conditional_transition_v1', 'stochastic_feedback_network_v1',
                  'learning_ablation_v1', 'time_mixture_v1'):
        ensure(evidence('results/' + study + '/verification.json')['status'] in ('pass', 'passed'),
               'Required previous independent validation did not pass')
    for study, field, artifact in (
            ('human_task_structure_v1', 'result_sha256', 'audit.json'),
            ('stochastic_feedback_network_v1', 'summary_sha256', 'summary.json'),
            ('learning_ablation_v1', 'summary_sha256', 'summary.json'),
            ('time_mixture_v1', 'metrics_sha256', 'metrics.json')):
        ensure(evidence('results/' + study + '/verification.json')[field] ==
               digest(evidence_path(root, 'results/' + study + '/' + artifact)),
               'Previous independent verification does not bind its exact result bytes')
    gate = evidence('results/human_task_structure_v1/audit.json')['gate']
    ensure(all(type(v) is bool for v in gate.values()) and gate == dict(source_coded_descriptive_metric_allowed=True,
           exact_historical_task_execution_version_verified=False,
           direct_human_model_similarity_allowed=False), 'Historical meaning gate changed')
    origin = evidence('results/human_osf_origin_v1/verification.json')
    for key in ('mismatches_verified', 'raw_rows_replayed', 'unmatched_rows_preserved', 'values_replayed'):
        integer(origin[key])
    ensure(origin['source_identity_gate_passed'] is True and origin['mismatches_verified'] == 0
           and (origin['raw_rows_replayed'], origin['unmatched_rows_preserved'], origin['values_replayed'])
           == (25, 49, 75000), 'Original 25 verified and 49 unverified records must remain distinct')
    ensure(evidence('results/human_semantics_v1/verification.json')['scope'] == SEMANTIC_SCOPE,
           'A code-count audit cannot establish a semantic mapping')
    datasets = evidence('data/human/honore_noel_2016/schema-audit.json')['datasets']
    ensure(len(datasets) == 2 and all(len(d['sheets']) == 2 and
           {s['sheet'] for s in d['sheets']} == {'T1', 'T2'} and
           all(type(s['participants']) is int and s['participants'] == 56 for s in d['sheets'])
           for d in datasets), 'Two aggregate tasks with 56 people and both timepoints required')
    conditional = evidence('results/conditional_transition_v1/verification.json')
    metric_path = 'results/conditional_transition_v1/metrics.json'
    artifact_path = 'results/conditional_transition_v1/artifact_hashes.json'
    ensure(conditional['metrics_sha256'] == digest(evidence_path(root, metric_path)) ==
           evidence(artifact_path)['metrics.json'], 'Previous verifier does not bind these metric bytes')
    ensure(conditional['artifact_hashes_sha256'] == digest(evidence_path(root, artifact_path)),
           'Previous artifact inventory does not match its verification')
    support, totals = inspect_support(evidence(metric_path), conditional)
    expected = dict(study_id=cfg['study_id'], bounded_phase4_deliverables_completed=True,
        **{k:False for k in FLAGS}, direct_level4_verdict='untested_not_testable_with_current_pairing',
        feature_count=6, human_source_cells=18, features=cfg['features'],
        source_coded_descriptive_metric_allowed=True, historical_task_execution_version_verified=False,
        openneuro_semantic_mapping_verified=False, verified_original_human_records=25,
        unverified_original_human_records_retained=49, support_counts=support,
        support_is_only_denominator_presence=True, additional_learning_updates=0, rng_draws=0,
        new_human_participants=0, new_human_effect_estimates=0,
        scope='evidence map and integer recount of prior verified measurement; no new rate pooling, significance, similarity or biological claim')
    report = read(result_dir / 'synthesis.json')
    ensure(report == expected, 'Synthesis differs from independent evidence map and support recount')
    ensure(all(type(report[k]) is bool for k in ('bounded_phase4_deliverables_completed',
           'source_coded_descriptive_metric_allowed', 'historical_task_execution_version_verified',
           'openneuro_semantic_mapping_verified', 'support_is_only_denominator_presence')),
           'Inference gates must be actual Boolean declarations')
    ensure(all(report[k] is False for k in FLAGS) and all(type(report[k]) is int and report[k] == 0
           for k in ('additional_learning_updates', 'rng_draws', 'new_human_participants', 'new_human_effect_estimates')),
           'Untested claims or forbidden additional work')
    return dict(source_files=len(SOURCES), evidence_files=len(EVIDENCE), primary_artifacts=3,
                features=6, human_source_cells=18, **totals)


def verify(root, config, registration, result_dir):
    now = datetime.now(timezone.utc)
    report = dict(study_id='phase4_synthesis_v1', verified_at_utc=now.isoformat(),
                  status='failed', passed=False, errors=[],
                  scope='independent_frozen_evidence_map_and_integer_partition_check_no_raw_replay_or_effect_estimation')
    try:
        report['counts'] = _check(Path(root).resolve(), Path(config), Path(registration), Path(result_dir), now)
        report.update(status='passed', passed=True, config_sha256=digest(Path(config)),
                      registration_sha256=digest(Path(registration)),
                      synthesis_sha256=digest(Path(result_dir)/'synthesis.json'),
                      artifact_hashes_sha256=digest(Path(result_dir)/'artifact_hashes.json'))
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        report['errors'].append(str(error))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    for option in ('config', 'registration', 'result-dir', 'out'):
        parser.add_argument('--'+option, type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.root, args.config, args.registration, args.result_dir)
    with args.out.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':report['status'], 'out':str(args.out)}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
