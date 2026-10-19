"""Freeze and run an autonomous-action RNG repair without refitting humans."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_human_shell as original

require, read, sha, safe = original.require, original.read, original.sha, original.safe
write_new, hashes = original.write_new, original.hashes
PARENT_NAMES = ('calibration.npz', 'calibration.csv', 'registration.json', 'metadata.json',
                'selection.json', 'trajectories.npz', 'trajectory_metrics.csv', 'witnesses.npz',
                'summary.json', 'cohort_audit.json', 'verification.json', 'artifact_hashes.json')


def validate_config(root, cfg):
    parent = read(safe(root, 'configs/human_shell_v1.json'))
    original.validate_config(parent)
    require(cfg['study_id'] == 'human_shell_rng_v2'
            and cfg['scope'] == 'prospectively_registered_action_stream_repair', 'Fixed repair scope required')
    allowed = {'study_id', 'scope', 'source_files', 'parent_files', 'repair_contract'}
    require({k: v for k, v in cfg.items() if k not in allowed}
            == {k: v for k, v in parent.items() if k not in allowed}, 'Repair changed fixed science or resources')
    contract = cfg['repair_contract']
    fixed = {'original_study_id': 'human_shell_v1', 'original_result_dir': 'results/human_shell_v1',
             'new_execution': 'autonomous_only', 'action_entropy_namespace': 20261019,
             'original_action_cells': 145, 'original_unique_additive_addresses': 88,
             'required_new_unique_tuple_addresses': 145, 'new_autonomous_modeled_trials': 344520,
             'new_calibration_trials': 0, 'new_conditional_trials': 0,
             'preserve_original_calibration_and_selected_rates': True,
             'preserve_original_conditional_array_bytes': True,
             'preserve_original_conditional_csv_lines': True,
             'preserve_original_conditional_witness_bytes': True,
             'preserve_H_prediction_exactly': True, 'original_outcome_blind_design_audit': True,
             'original_source_and_results_immutable': True}
    require(all(contract.get(k) == v for k, v in fixed.items()), 'Repair contract differs')
    require(contract['action_generator'] == 'default_rng(SeedSequence([20261019, seed, int(participant_id[1:])])).random(trials_per_person)', 'Fixed tuple generator required')
    new_source = {'src/drosolearn/human_shell_rng.py', 'tests/test_human_shell_rng.py',
                  'docs/human-shell-rng-protocol.md', 'configs/human_shell_rng_v2.json',
                  'scripts/run_human_shell_rng.py', 'tests/test_human_shell_rng_run.py',
                  'scripts/verify_human_shell_rng.py', 'tests/test_human_shell_rng_verify.py',
                  'docs/human-shell-rng-design-note.md'}
    expected_source = set(parent['source_files']) | new_source
    expected_parent = set(parent['parent_files']) | {'results/human_shell_v1/'+n for n in PARENT_NAMES}
    expected_parent.add('results/human_shell_rng_design_issue_v1.json')
    require(set(cfg['source_files']) == expected_source and len(cfg['source_files']) == len(expected_source),
            'Exact original and new source closure required')
    require(set(cfg['parent_files']) == expected_parent and len(cfg['parent_files']) == len(expected_parent),
            'Exact original and new parent closure required')


def key_inventory(cfg):
    """Record addresses and untouched initial generator states; draw no uniforms."""
    import numpy as np
    require(cfg['test_ids'] and cfg['test_seeds']
            and len(set(cfg['test_ids'])) == len(cfg['test_ids'])
            and len(set(cfg['test_seeds'])) == len(cfg['test_seeds']), 'Unique nonempty RNG axes required')
    import re
    require(all(type(s) is int and s >= 0 for s in cfg['test_seeds'])
            and all(type(p) is str and re.fullmatch(r'A[0-9]{4}', p) for p in cfg['test_ids']),
            'Canonical RNG addresses required')
    old, rows = {}, []
    for seed in cfg['test_seeds']:
        for pid in cfg['test_ids']:
            entropy = [20261019, seed, int(pid[1:])]
            old.setdefault(seed + 300000 + int(pid[1:]), []).append([seed, pid])
            seq = np.random.SeedSequence(entropy)
            state = seq.generate_state(4).astype('<u4')
            pcg = np.random.default_rng(seq).bit_generator.state
            pcg_bytes = json.dumps(pcg, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            rows.append({'seed': seed, 'participant_id': pid, 'entropy': entropy,
                         'seedsequence_state_uint32': state.tolist(),
                         'seedsequence_state_sha256': hashlib.sha256(state.tobytes()).hexdigest(),
                         'pcg64_initial_state': pcg,
                         'pcg64_initial_state_sha256': hashlib.sha256(pcg_bytes).hexdigest()})
    ntuple = len({tuple(r['entropy']) for r in rows})
    nstate = len({r['seedsequence_state_sha256'] for r in rows})
    npcg = len({r['pcg64_initial_state_sha256'] for r in rows})
    require(ntuple == nstate == npcg == len(rows), 'Duplicate corrected RNG address or state')
    collisions = {str(k): v for k, v in old.items() if len(v) > 1}
    return {'cells': len(rows), 'old_unique_integer_keys': len(old),
            'old_collision_groups': len(collisions), 'old_collision_cells': sum(map(len, collisions.values())),
            'old_duplicate_uses_beyond_first': sum(len(v)-1 for v in collisions.values()),
            'old_equal_key_pairs': sum(len(v)*(len(v)-1)//2 for v in collisions.values()),
            'old_max_group_size': max(map(len, old.values())), 'new_unique_entropy_tuples': ntuple,
            'new_unique_seedsequence_states': nstate, 'new_unique_pcg64_states': npcg,
            'rng_draws_in_key_audit': 0, 'old_collisions': collisions, 'new_keys': rows}


def gate(root, cfg):
    result = original.parent_gate(root, cfg)
    parent = root / cfg['repair_contract']['original_result_dir']
    prior_reg = read(parent/'registration.json')
    check = read(parent/'verification.json')
    require(prior_reg['config'] == read(safe(root, 'configs/human_shell_v1.json')),
            'Original registered configuration changed')
    require(prior_reg['source_files'] == hashes(root, prior_reg['config']['source_files'])
            and prior_reg['graph_files'] == hashes(root, cfg['graph_files']),
            'Original source or graph changed since the first registration')
    manifest = read(safe(root, cfg['manifest']))
    expected_external = {name: {'bytes': spec['bytes'], 'sha256': spec['sha256']}
                         for name, spec in manifest['files'].items()}
    require(prior_reg['external_files'] == expected_external, 'Original external input inventory changed')
    require(check['status'] == 'passed' and check['registration_sha256'] == sha(parent/'registration.json')
            and check['summary_sha256'] == sha(parent/'summary.json')
            and read(parent/'metadata.json')['status'] == 'completed', 'Completed arithmetic-verified parent required')
    inventory = read(parent/'artifact_hashes.json')
    require(inventory == {p.name: sha(p) for p in sorted(parent.iterdir())
                          if p.is_file() and p.name not in ('artifact_hashes.json', 'verification.json')},
            'Parent artifact bytes changed')
    result.update(parent_arithmetic_verification_passed=True, original_autonomous_inference_design_limited=True,
                  action_address_repair_only=True, no_refitting=True)
    return result


def committed_files(root, commit, inventory):
    for name, digest in inventory.items():
        blob = subprocess.check_output(['git', 'show', commit+':'+name], cwd=root)
        require(hashlib.sha256(blob).hexdigest() == digest, 'Frozen file differs from Git: '+name)


def register(root, config, cache, out):
    cfg = read(config); validate_config(root, cfg)
    source, parents = hashes(root, cfg['source_files']), hashes(root, cfg['parent_files'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    parent_commit = subprocess.check_output(['git', 'rev-parse', 'human-shell-v1-design-limited'], cwd=root, text=True).strip()
    committed_files(root, commit, source)
    committed_files(root, parent_commit, parents)
    addresses = key_inventory(cfg)
    require(addresses['cells'] == 145 and addresses['old_unique_integer_keys'] == 88, 'Original design issue differs')
    reg = {'study_id': cfg['study_id'], 'registration_type': 'action_stream_repair_without_refitting',
           'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': commit,
           'parent_release_commit': parent_commit, 'parent_release_tag': 'human-shell-v1-design-limited',
           'config': cfg, 'config_sha256': sha(config), 'source_files': source, 'parent_files': parents,
           'graph_files': hashes(root, cfg['graph_files']), 'external_files': original.external_inventory(root, cfg, cache),
           'gate': gate(root, cfg),
           'notice': 'Original scientific scores were not inspected for this repair decision. Preserve the first run and change autonomous action-stream addressing only; no fitting or test-driven seed/criterion changes.'}
    write_new(out, reg)
    return reg


def read_rows(path):
    with Path(path).open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if key in ('mode', 'condition', 'participant_id'):
                continue
            if value == '': row[key] = None
            elif key == 'seed' or key.endswith('_calls') or key.endswith('_nonzero_steps'): row[key] = int(value)
            else: row[key] = float(value)
    return rows


def conditional_bytes(path):
    lines = Path(path).read_bytes().splitlines(keepends=True)
    require(bool(lines), 'Parent CSV required')
    fields = next(csv.reader(io.StringIO(lines[0].decode('utf8'))))
    require(len(set(fields)) == len(fields), 'Unique CSV fields required')
    mi = fields.index('mode'); selected = []
    for raw in lines[1:]:
        values = next(csv.reader(io.StringIO(raw.decode('utf8'))))
        require(len(values) == len(fields) and values[mi] in ('conditional', 'autonomous'), 'Malformed trajectory CSV')
        if values[mi] == 'conditional': selected.append(raw)
    return lines[0], b''.join(selected), len(selected), fields


def write_combined_csv(path, parent_csv, rows):
    header, lines, count, fields = conditional_bytes(parent_csv)
    old_rows = [r for r in read_rows(parent_csv) if r['mode'] == 'conditional']
    new_conditional = [r for r in rows if r['mode'] == 'conditional']
    require(old_rows == new_conditional, 'Conditional row values changed')
    free = [r for r in rows if r['mode'] == 'autonomous']
    require(free and all(set(r) == set(fields) for r in free), 'Autonomous row schema differs')
    buffer = io.StringIO(newline='')
    csv.DictWriter(buffer, fieldnames=fields).writerows(free)
    with Path(path).open('xb') as f: f.write(header+lines+buffer.getvalue().encode('utf8'))
    return hashlib.sha256(lines).hexdigest(), count


def copy_exact(source, dest):
    with Path(dest).open('xb') as f: f.write(Path(source).read_bytes())


def reuse_receipt(parent, arrays, witnesses, original_arrays, original_witnesses, csv_digest, csv_count):
    require(set(arrays) == set(original_arrays), 'Trajectory array inventory differs')
    for key in arrays:
        require(arrays[key].dtype == original_arrays[key].dtype
                and arrays[key][0].shape == original_arrays[key][0].shape
                and arrays[key][0].tobytes() == original_arrays[key][0].tobytes(), 'Conditional array changed')
    conditional_count = initial_count = 0
    require(set(witnesses) == set(original_witnesses), 'Witness inventory differs')
    for key in original_witnesses:
        cond, initial = key.startswith('conditional_'), key.startswith('autonomous_') and '_initial_' in key
        if cond or initial:
            require(witnesses[key].shape == original_witnesses[key].shape
                    and witnesses[key].dtype == original_witnesses[key].dtype
                    and witnesses[key].tobytes() == original_witnesses[key].tobytes(), 'Preserved witness changed')
        conditional_count += int(cond); initial_count += int(initial)
    return {'parent_study_id': 'human_shell_v1', 'parent_result_dir': 'results/human_shell_v1',
            'parent_files': {name: sha(parent/name) for name in PARENT_NAMES},
            'conditional_half_sha256': {k: hashlib.sha256(v[0].tobytes()).hexdigest() for k, v in original_arrays.items()},
            'conditional_csv_sha256': csv_digest, 'conditional_csv_records': csv_count,
            'conditional_witness_fields': conditional_count, 'unchanged_autonomous_initial_witness_fields': initial_count,
            'new_calibration_observations': 0, 'new_conditional_predictions': 0}


def run(root, config, registration, cache, out):
    import numpy as np
    from drosolearn.graph import load_projection
    from drosolearn.human_shell import summarize
    from drosolearn.human_shell_rng import produce_corrected
    from drosolearn.human_shell_source import load_cohort
    cfg, reg = read(config), read(registration); validate_config(root, cfg)
    stamp = datetime.fromisoformat(reg['registered_at_utc'])
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
            and stamp <= datetime.now(timezone.utc), 'Prior UTC registration required')
    require(reg['registration_type'] == 'action_stream_repair_without_refitting'
            and reg['config'] == cfg and reg['config_sha256'] == sha(config)
            and reg['source_files'] == hashes(root, cfg['source_files'])
            and reg['parent_files'] == hashes(root, cfg['parent_files'])
            and reg['graph_files'] == hashes(root, cfg['graph_files'])
            and reg['external_files'] == original.external_inventory(root, cfg, cache)
            and reg['gate'] == gate(root, cfg), 'Frozen science, parent, source, or input changed')
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', reg['source_git_commit'], 'HEAD'], cwd=root)
    committed_files(root, reg['source_git_commit'], reg['source_files'])
    committed_files(root, reg['parent_release_commit'], reg['parent_files'])
    require(not out.exists(), 'New output directory required')
    out.mkdir(parents=True)
    copy_exact(registration, out/'registration.json')
    parent = root/cfg['repair_contract']['original_result_dir']
    meta = {'study_id': cfg['study_id'], 'status': 'running', 'stage': 'input_validation',
            'started_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': reg['source_git_commit'],
            'registration_sha256': sha(registration), 'config_sha256': sha(config),
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__},
            'human_model_comparison_tested': False, 'shared_neural_mechanism_tested': False, 'drug_effect_tested': False}
    try:
        records, audit = load_cohort(cache, read(safe(root, cfg['manifest'])))
        require(set(records) == set(cfg['calibration_ids']+cfg['test_ids'])
                and all(len(v) == 216 for v in records.values()), 'Complete original cohort required')
        require(audit == read(parent/'cohort_audit.json'), 'Cohort source audit changed')
        for name in ('selection.json', 'cohort_audit.json'): copy_exact(parent/name, out/name)
        selection = read(parent/'selection.json')
        write_new(out/'rng_keys.json', key_inventory(cfg))
        with np.load(parent/'trajectories.npz', allow_pickle=False) as z:
            old_arrays = {k: z[k] for k in z.files}
        with np.load(parent/'witnesses.npz', allow_pickle=False) as z:
            old_witnesses = {k: z[k] for k in z.files}
        rows = read_rows(parent/'trajectory_metrics.csv')
        meta['stage'] = 'autonomous_action_stream_repair'
        print('Executing autonomous choices only; reusing calibration and conditional predictions unchanged', flush=True)
        arrays, rows, witnesses = produce_corrected(old_arrays, rows, old_witnesses,
            {pid: records[pid] for pid in cfg['test_ids']}, cfg, load_projection(root/cfg['graph']), selection)
        np.savez_compressed(out/'trajectories.npz', **arrays)
        np.savez_compressed(out/'witnesses.npz', **witnesses)
        digest, count = write_combined_csv(out/'trajectory_metrics.csv', parent/'trajectory_metrics.csv', rows)
        write_new(out/'reused_parent_data.json', reuse_receipt(parent, arrays, witnesses, old_arrays, old_witnesses, digest, count))
        meta['stage'] = 'unchanged_prespecified_summary'
        summary = summarize({pid: records[pid] for pid in cfg['test_ids']}, cfg, arrays, rows)
        prior = read(parent/'summary.json')
        require(all(summary[k] == prior[k] for k in ('H_prediction', 'human_metrics', 'conditional_family_mean_nll', 'conditional_family_mean_brier')),
                'Reused human observations or conditional prediction conclusion changed')
        write_new(out/'summary.json', summary)
        meta.update(status='completed', stage='completed', human_model_comparison_tested=True,
                    new_calibration_observations=0, new_conditional_predictions=0, reused_conditional_predictions=344520,
                    new_autonomous_choices=344520, original_human_people=49, original_human_trials=10584,
                    combined_container_observations=689040,
                    calibration_people=20, heldout_people=29, test_model_initialization_seeds=5,
                    selection_unchanged=True, conditional_predictions_unchanged=True,
                    interpretation='Autonomous-address repair only; original outcome and thresholds preserved; computational seeds are not people or animals')
    except Exception as error:
        meta.update(status='failed', error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        meta['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_new(out/'metadata.json', meta)
        write_new(out/'artifact_hashes.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['register', 'run'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Existing output cannot be overwritten')
    require(args.mode != 'run' or args.registration is not None, 'Prior registration required')
    result = register(args.root.resolve(), args.config, args.cache.resolve(), args.out) if args.mode == 'register' else run(
        args.root.resolve(), args.config, args.registration, args.cache.resolve(), args.out)
    print(json.dumps({'status': result.get('status', 'registered'), 'out': str(args.out)}))


if __name__ == '__main__': main()
