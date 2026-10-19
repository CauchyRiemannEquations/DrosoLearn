"""Retrospective evidence ledger: no learner, RNG, new effect or biological inference."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
import numpy as np

STATUSES = ('supported', 'not_supported', 'bounded', 'not_assessable', 'design_limited', 'untested')
STUDY_IDS = ('smoke_v1', 'reversal_v1', 'encoding_v1', 'spacing_v1', 'linearity_v1',
             'reward_linear_v1', 'bandit_reversal_v1', 'head_attenuation_v1',
             'human_trial_data_audit_v1', 'human_trial_linkage_v1', 'human_semantics_v1',
             'human_osf_origin_v1', 'human_task_structure_v1', 'feedback_transition_v1',
             'metric_calibration_v1', 'stochastic_feedback_linear_v1', 'stochastic_feedback_network_v1',
             'learning_ablation_v1', 'state_decomposition_v1', 'conditional_transition_v1',
             'time_mixture_v1', 'phase4_synthesis_v1', 'human_shell_v1', 'human_shell_rng_v2',
             'shell_credit_v1', 'shell_behavior_v1', 'pattern_rule_v1', 'interference_order_v1')
INTERPRETATION = dict(bounded_synthesis_complete=True, all_possible_phase2_phase3_work_complete=False,
                      level4_similarity_supported=False, shared_neural_mechanism_claim=False,
                      education_effect_claim=False)
SPLIT_RAW = 'results/interference_order_v1/arrays.npz'
SPLIT_MANIFEST = SPLIT_RAW + '.parts.json'
SPLIT_PARTS = (SPLIT_RAW + '.part001', SPLIT_RAW + '.part002')
SPLIT_ARTIFACTS = 'results/interference_order_v1/artifact_hashes.json'


def git_bytes(root, commit, name):
    return subprocess.check_output(['git', 'show', commit + ':' + name], cwd=root)


def split_local(root, name):
    path = (Path(root)/name).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Split path must stay inside this repository')
    return path


def split_git_payload(root, commit, expected_sha):
    """Exact bytes of one large parent, stored as two ordinary Git blobs."""
    root = Path(root)
    manifest_bytes = git_bytes(root, commit, SPLIT_MANIFEST)
    require(split_local(root,SPLIT_MANIFEST).read_bytes() == manifest_bytes, 'Split manifest differs from committed bytes')
    manifest = read(split_local(root,SPLIT_MANIFEST))
    require(set(manifest) == {'format', 'original_path', 'original_bytes', 'original_sha256', 'max_part_bytes', 'parts'}
            and manifest['format'] == 'drosolearn.lossless_file_parts.v1'
            and manifest['original_path'] == SPLIT_RAW and manifest['max_part_bytes'] == 67108864
            and type(manifest['original_bytes']) is int and manifest['original_bytes'] > 0
            and manifest['original_sha256'] == expected_sha, 'Exact original-parent split identity required')
    original_artifacts = json.loads(git_bytes(root, commit, SPLIT_ARTIFACTS))
    require(original_artifacts['arrays.npz'] == expected_sha, 'Split parent must match original scientific artifact hash')
    require(type(manifest['parts']) is list and len(manifest['parts']) == 2
            and [p.get('path') for p in manifest['parts']] == list(SPLIT_PARTS), 'Exactly two ordered registered parts required')
    pieces = []
    for name, spec in zip(SPLIT_PARTS, manifest['parts']):
        require(set(spec) == {'path', 'bytes', 'sha256'} and type(spec['bytes']) is int
                and 0 < spec['bytes'] <= manifest['max_part_bytes'], 'Valid split-part schema required')
        blob = git_bytes(root, commit, name)
        require(split_local(root,name).read_bytes() == blob and len(blob) == spec['bytes']
                and hashlib.sha256(blob).hexdigest() == spec['sha256'], 'Literal committed split part mismatch')
        pieces.append(blob)
    payload = b''.join(pieces)
    require(len(payload) == manifest['original_bytes'] and hashlib.sha256(payload).hexdigest() == expected_sha,
            'Joined parent bytes differ from the original result')
    return payload


@contextmanager
def committed_parent_parts(root):
    """Storage adaptation only; every other frozen committed-byte gate is unchanged."""
    from . import prospective
    original = prospective.committed
    def checked(check_root, commit, names):
        if SPLIT_RAW not in names:
            return original(check_root, commit, names)
        require(Path(check_root).resolve() == Path(root).resolve(), 'Same repository root required')
        original(check_root, commit, {n:h for n,h in names.items() if n != SPLIT_RAW})
        payload = split_git_payload(check_root, commit, names[SPLIT_RAW])
        require(split_local(check_root,SPLIT_RAW).read_bytes() == payload, 'Existing raw parent must match exact joined Git bytes')
    prospective.committed = checked
    try:
        yield
    finally:
        prospective.committed = original


def materialize_parent(root, commit='HEAD'):
    root = Path(root)
    expected_sha = json.loads(git_bytes(root, commit, SPLIT_ARTIFACTS))['arrays.npz']
    payload = split_git_payload(root, commit, expected_sha)
    target = split_local(root,SPLIT_RAW)
    if target.exists():
        require(target.is_file() and target.read_bytes() == payload, 'Existing raw parent differs; never overwrite')
        return 'already_matches'
    with target.open('xb') as f:
        f.write(payload)
    require(digest(target) == expected_sha, 'Materialized original hash mismatch')
    return 'materialized_exact_original_bytes'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=unique, parse_constant=invalid)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def pointer(document, address):
    require(isinstance(address, str) and (address == '' or address.startswith('/')), 'RFC6901 pointer required')
    result = document
    if address:
        for raw in address[1:].split('/'):
            require(re.search(r'~(?![01])', raw) is None, 'Valid RFC6901 escape required')
            token = raw.replace('~1', '/').replace('~0', '~')
            if isinstance(result, list):
                require(token.isdigit() and str(int(token)) == token, 'Canonical list index required')
                result = result[int(token)]
            else:
                require(isinstance(result, dict), 'Pointer traverses a non-container')
                result = result[token]
    return result


def relative_file(root, name, frozen):
    require(isinstance(name, str) and name in frozen, 'Evidence path must be frozen: ' + str(name))
    path = root / name
    require(not Path(name).is_absolute() and path.resolve().is_relative_to(root.resolve()) and path.is_file(),
            'Local existing evidence file required')
    return path


def validate_config(cfg):
    require(cfg.get('study_id') == 'phase5_synthesis_v1' and
            cfg.get('scope') == 'retrospective_bounded_evidence_synthesis', 'Retrospective scope required')
    require([s['study_id'] for s in cfg['studies']] == list(STUDY_IDS), 'All 28 studies in historical order required')
    require(cfg.get('synthesis_interpretation') == INTERPRETATION, 'Bounded interpretation required')
    claims = cfg['claims']
    ids = [c['claim_id'] for c in claims]
    require(bool(claims) and len(ids) == len(set(ids)), 'Unique nonempty claim ledger required')
    for c in claims:
        require(type(c['question']) is int and 1 <= c['question'] <= 5 and type(c['level']) is int
                and 0 <= c['level'] <= 4 and c['status'] in STATUSES, 'Valid claim coordinates required')
        require(c['level'] < 4 or c['status'] != 'supported', 'No positive Level4 claim from these results')
        require(bool(c['text_ko']) and bool(c['limits_ko']) and bool(c['observations']), 'Evidence and limits required')
    qs = cfg['questions']
    require([q['question'] for q in qs] == [1, 2, 3, 4, 5], 'Five original research questions required')
    mapped = []
    for q in qs:
        require(bool(q['text_ko']) and bool(q['claim_ids']), 'Every question needs evidence')
        expected = [c['claim_id'] for c in claims if c['question'] == q['question']]
        require(q['claim_ids'] == expected, 'Question links must cover every aligned claim exactly once')
        mapped.extend(q['claim_ids'])
    require(set(mapped) == set(ids), 'Every claim mapped')


def evaluate(cfg, root):
    """Read previously known results, verify declared observations, keep all outcomes."""
    validate_config(cfg)
    root = Path(root)
    frozen = set(cfg['input_files']) | set(cfg['parent_files'])
    studies = []
    for record in cfg['studies']:
        verification_path = relative_file(root, record['verification'], frozen)
        v = read(verification_path)
        require('study_id' not in v or v['study_id'] == record['study_id'], 'Verification belongs to another study')
        require(v.get('passed') is not False and (v.get('passed') is True or v.get('status') in ('passed', 'pass')),
                'Unverified study ' + record['study_id'])
        require(v.get('errors', []) == [], 'Unresolved verification errors')
        report = relative_file(root, record['report'], frozen)
        primary = relative_file(root, record['primary'], frozen) if record.get('primary') else None
        if primary is not None and 'summary_sha256' in v and primary.name == 'summary.json':
            require(digest(primary) == v['summary_sha256'], 'Verified primary summary hash mismatch')
        studies.append(dict(record, verified=True, verification_sha256=digest(verification_path),
                            report_sha256=digest(report), primary_sha256=digest(primary) if primary else None))
    rows = []
    for claim in cfg['claims']:
        for obs in claim['observations']:
            actual = pointer(read(relative_file(root, obs['file'], frozen)), obs['pointer'])
            require(canonical(actual) == canonical(obs['value']), 'Declared observation differs: ' + claim['claim_id'])
        rows.append({k: claim[k] for k in ('claim_id', 'question', 'level', 'status', 'text_ko', 'limits_ko')}
                    | {'observations_json': canonical(claim['observations'])})
    arrays = {'study_checks': np.asarray([[i, 1] for i in range(len(studies))], dtype=np.int64),
              'claim_codes': np.asarray([[c['question'], c['level'], STATUSES.index(c['status']), len(c['observations'])]
                                         for c in cfg['claims']], dtype=np.int64)}
    summary = {'status': 'completed', 'study_id': cfg['study_id'], 'scope': cfg['scope'],
               'studies': studies, 'claims': cfg['claims'], 'questions': cfg['questions'],
               'counts': {'studies': len(studies), 'claims': len(rows),
                          'observations': sum(len(c['observations']) for c in cfg['claims']), 'questions': 5,
                          'new_learning_updates': 0, 'new_rng_draws': 0, 'new_human_experiments': 0, 'new_effect_tests': 0},
               'interpretation': dict(INTERPRETATION)}
    return arrays, rows, summary


def produce(cfg, root, cache=None):
    arrays, rows, _ = evaluate(cfg, root)
    return arrays, rows


def summarize(cfg, arrays, rows):
    # Framework supplies no root here; produce stores no hidden mutable global.
    # Evidence hashes are bound in the registration, whose committed root is explicit in config.
    root = Path(__file__).resolve().parents[2]
    expected_arrays, expected_rows, summary = evaluate(cfg, root)
    require(arrays.keys() == expected_arrays.keys() and
            all(np.array_equal(arrays[k], expected_arrays[k]) for k in arrays), 'Ledger arrays changed')
    require(rows == expected_rows, 'Ledger rows changed')
    return summary


def render_evidence_markdown(summary, out):
    """Deterministic appendix; explanatory report is separately reviewed and archived."""
    lines = ['# Phase 5 기계 판독 증거의 읽기용 부록', '',
             '기존 결과를 종합한 문서다. 새 학습·인간 실험·효과 검정은 0회다.', '']
    for q in summary['questions']:
        lines += ['## ' + str(q['question']) + '. ' + q['text_ko'], '']
        for c in summary['claims']:
            if c['question'] == q['question']:
                lines += ['- **' + c['claim_id'] + '** (' + c['status'] + ', Level ' + str(c['level']) + '): ' + c['text_ko'],
                          '  해석 범위: ' + c['limits_ko']]
                for obs in c['observations']:
                    lines.append('  근거: `' + obs['file'] + obs['pointer'] + '` = `' + canonical(obs['value']) + '`')
        lines.append('')
    lines += ['## 전체 완료 연구', '', '| 연구 | 역할 | 완료 태그 | 검산 |', '|---|---|---|---|']
    for s in summary['studies']:
        lines.append('| ' + s['study_id'] + ' | ' + s['role'] + ' | ' + s['tag'] + ' | passed |')
    with Path(out).open('x', encoding='utf8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')


def render_figures(root, out):
    """Derived descriptive plots, all conditions retained; no new significance tests."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root, out = Path(root), Path(out)
    require(not out.exists(), 'New presentation directory required')
    out.mkdir(parents=True)
    plotted = {}
    credit = read(root/'results/shell_credit_v1/summary.json')
    conditions = [r['condition'] for r in credit['acquisition']]
    points = [[r['after_accuracy'] for r in credit['outcomes'] if r['condition'] == c] for c in conditions]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for i, vals in enumerate(points):
        ax.scatter(np.full(len(vals), i), vals, s=42, alpha=.8)
        ax.hlines(np.mean(vals), i-.22, i+.22, color='black', linewidth=2)
    ax.set(xticks=range(3), xticklabels=conditions, ylim=(.4, 1),
           ylabel='Expected favorable-choice probability', title='Stationary cue learning: all five computational seeds')
    ax.axhline(.5, linestyle=':', color='gray');ax.grid(axis='y', alpha=.2)
    fig.tight_layout();fig.savefig(out/'credit_endpoints.png', dpi=160);plt.close(fig)
    plotted['credit'] = {'conditions': conditions, 'seed_points': points,
                         'notice': 'Same initial own-cue update; extra fixed solve coefficients; no biological inference.'}
    pcfg = read(root/'configs/pattern_rule_v1.json')
    with np.load(root/'results/pattern_rule_v1/arrays.npz', allow_pickle=False) as z:
        prob, target = z['checkpoint_probabilities'], z['probe_targets']
        correct = np.where(target[None, None, None] == 1, prob, 1-prob)
        train = correct[:, :, :, 0].mean(axis=-1)
        endpoint = correct[:, :, -1].mean(axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), gridspec_kw={'width_ratios':[1.1, 1]})
    colors = plt.get_cmap('tab10').colors
    for ci, condition in enumerate(pcfg['conditions']):
        axes[0].plot(pcfg['checkpoints'], train[ci].mean(axis=0), marker='.', color=colors[ci], label=condition)
    axes[0].set(xlabel='Training feedback trials', ylabel='Expected correct-choice probability', ylim=(.45, 1),
                title='Pattern acquisition: means across ten seeds')
    axes[0].axhline(.8, linestyle=':', color='gray', label='One of several acquisition criteria')
    axes[0].legend(fontsize=7, loc='upper left');axes[0].grid(alpha=.2)
    values = endpoint.mean(axis=1)[:, [0, 2, 3, 4]]
    image = axes[1].imshow(values, vmin=.45, vmax=1, cmap='viridis', aspect='auto')
    axes[1].set(xticks=range(4), xticklabels=['Trained', 'New symbols', 'Symbols+phase', 'New length'],
                yticks=range(9), yticklabels=pcfg['conditions'], title='Observed scores; acquisition gate remains mandatory')
    axes[1].tick_params(axis='x', labelrotation=22)
    for ci in range(9):
        for j in range(4):
            axes[1].text(j, ci, f'{100*values[ci,j]:.1f}', ha='center', va='center', color='white', fontsize=8)
    fig.colorbar(image, ax=axes[1], fraction=.035, label='Expected correct-choice probability')
    fig.tight_layout();fig.savefig(out/'pattern_learning.png', dpi=160);plt.close(fig)
    plotted['pattern'] = {'conditions': pcfg['conditions'], 'checkpoints': pcfg['checkpoints'],
                          'train_mean_curves': train.mean(axis=1).tolist(), 'endpoint_set_indices':[0,2,3,4],
                          'endpoint_means': values.tolist(),
                          'notice':'All neural conditions failed acquisition; OOD scores are not qualified generalization.'}
    order = read(root/'results/interference_order_v1/summary.json')
    families = order['families']
    labels = [f['condition'] for f in families]
    prefix_joint = np.asarray([[np.mean([s['tasks'][task][field] for s in family['seeds']])
                               for field,task in [('prefix_score',0),('prefix_score',1),('joint_score',0),('joint_score',1)]]
                              for family in families])
    effects = np.asarray([[f['H_interference']['mean_retention_loss'],f['H_order']['mean_order_delta']] for f in families])
    intervals = np.asarray([[f['H_interference']['ci90'],f['H_order']['ci90']] for f in families])
    qualified = [f['qualification_supported'] for f in families]
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.4))
    x = np.arange(len(families))
    for j,name in enumerate(['A alone','B alone','A after joint','B after joint']):
        axes[0].bar(x+(j-1.5)*.19,prefix_joint[:,j],width=.18,label=name)
    axes[0].axhline(.8,color='gray',linestyle=':')
    axes[0].set(xticks=x,xticklabels=labels,ylim=(0,1),ylabel='Expected optimal-action probability',
                title='Acquisition and joint learning: all ten seeds')
    axes[0].tick_params(axis='x',labelrotation=55);axes[0].legend(fontsize=8)
    for j,name in enumerate(['Retention loss','Order delta AB minus BA']):
        positions=x+(j-.5)*.34
        bars=axes[1].bar(positions,effects[:,j],width=.32,label=name)
        for i,bar in enumerate(bars):
            if not qualified[i]:bar.set_hatch('///');bar.set_alpha(.5)
        # Intervals are already registered paired-seed bootstrap outputs.
        axes[1].vlines(positions,intervals[:,j,0],intervals[:,j,1],color='black',linewidth=1)
    axes[1].axhline(0,color='black',linewidth=.7)
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Probability difference (90% paired-seed interval)',
                title='Hatched groups: raw effects, qualification failed')
    axes[1].tick_params(axis='x',labelrotation=55);axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'interference_order.png',dpi=160);plt.close(fig)
    plotted['interference_order'] = {'conditions':labels,'prefix_joint_column_order':['prefix_A','prefix_B','joint_A','joint_B'],
                                    'prefix_joint_means':prefix_joint.tolist(),'effects':effects.tolist(),
                                    'effect_columns':['retention_loss','order_AB_minus_BA'],
                                    'ci90':intervals.tolist(),'qualification_supported':qualified,
                                    'notice':'Every seed included. Unqualified effects are descriptive; Q zero differences follow separate table structure.'}
    with (out/'plotted_values.json').open('x', encoding='utf8', newline='\n') as f:
        json.dump(plotted, f, ensure_ascii=False, indent=2, allow_nan=False);f.write('\n')
    return {'matplotlib_version': matplotlib.__version__,
            'files': {p.name:digest(p) for p in sorted(out.iterdir())}}


def main():
    import argparse
    from . import prospective
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register','run','materialize-parent'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--registration',type=Path)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.action == 'materialize-parent':
        print(json.dumps({'status':materialize_parent(args.root)}));return
    require(args.config is not None and args.out is not None, 'Config and fresh output path required')
    cfg = read(args.config)
    require(cfg['study_id'] == 'phase5_synthesis_v1' and cfg.get('chunked_parent_manifest') == SPLIT_MANIFEST
            and set((SPLIT_MANIFEST, *SPLIT_PARTS)) <= set(cfg['input_files']), 'Frozen split-parent storage inventory required')
    with committed_parent_parts(args.root):
        if args.action == 'register':
            prospective.register(args.root,args.config,None,args.out)
        else:
            require(args.registration is not None,'Frozen registration required')
            prospective.run(args.root,args.config,args.registration,None,args.out)
    print(json.dumps({'status':'registered' if args.action=='register' else 'completed','out':str(args.out)}))


if __name__ == '__main__':
    main()
