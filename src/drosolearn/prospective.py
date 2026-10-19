"""Shared integrity envelope for the bounded follow-ups; no scientific algorithms."""
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

BASE_TAG = 'human-shell-comparison-v2'
BASE_COMMIT = '3ee71a97fa9d6ffc7fce3bd4d7c3832437977171'
STUDIES = {
    'shell_credit_v1': ('shell_credit', 'shell-credit'),
    'shell_behavior_v1': ('shell_behavior', 'shell-behavior'),
    'pattern_rule_v1': ('pattern_rule', 'pattern-rule'),
    'interference_order_v1': ('interference_order', 'interference-order'),
    'phase5_synthesis_v1': ('phase5_report', 'phase5-report'),
}
ORDER = tuple(STUDIES)
COMMON_SOURCE = ['src/drosolearn/prospective.py', 'scripts/run_prephase5.py',
                 'tests/test_prospective.py', 'docs/prephase5-continuation-plan.md']
BASE_PARENT = ['results/registration_human_shell_rng_v2.json',
               'results/human_shell_rng_v2/summary.json', 'results/human_shell_rng_v2/verification.json',
               'results/human_shell_rng_v2/metadata.json', 'results/human_shell_rng_v2/artifact_hashes.json',
               'results/human_shell_comparison_archive_validation.json', 'results/human_shell_state_diagnostics.json']
PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json')


def require(ok, message):
    if not ok: raise ValueError(message)


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=pairs,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(root, name):
    require(type(name) is str and name and '\\' not in name and ':' not in name
            and not Path(name).is_absolute() and '..' not in Path(name).parts, 'Contained POSIX input path required')
    path = (Path(root)/name).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path.is_file(), 'Existing contained input required: '+name)
    return path


def inventory(root, names):
    require(type(names) is list and names and len(names) == len(set(names)), 'Explicit unique inventory required')
    return {name: sha(safe(root, name)) for name in names}


def write_new(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def study_source(study):
    module, slug = STUDIES[study]
    return [f'src/drosolearn/{module}.py', f'tests/test_{module}.py', f'docs/{slug}-protocol.md',
            f'configs/{study}.json', f'scripts/verify_{module}.py', f'tests/test_{module}_verify.py']


def expected_source(root, study):
    base = read(safe(root, BASE_PARENT[0]))['source_files']
    names = set(base) | set(COMMON_SOURCE) | set(study_source(study))
    if study == 'interference_order_v1': names |= set(study_source('pattern_rule_v1'))
    return names


def previous_parent_files(study):
    names = list(BASE_PARENT)
    for previous in ORDER[:ORDER.index(study)]:
        names.extend('results/'+previous+'/'+name for name in (*PRIMARY, 'artifact_hashes.json', 'verification.json'))
    return names


def validate_envelope(root, cfg):
    study = cfg['study_id']; require(study in STUDIES, 'Known bounded study required')
    require(cfg.get('scope') != 'integration_test', 'Development fixture cannot be registered as a scientific run')
    require(cfg['module'] == 'drosolearn.'+STUDIES[study][0]
            and cfg['protocol'] == 'docs/'+STUDIES[study][1]+'-protocol.md', 'Fixed study implementation required')
    require(set(cfg['source_files']) == expected_source(root, study)
            and len(cfg['source_files']) == len(set(cfg['source_files'])), 'Exact scientific source closure required')
    required = set(previous_parent_files(study))
    require(required <= set(cfg['parent_files']) and len(cfg['parent_files']) == len(set(cfg['parent_files'])),
            'Preceding verified studies and explicit unique parent inventory required')
    require(cfg['interpretation']['new_human_experiment'] is False
            and cfg['interpretation']['shared_neural_mechanism_claim'] is False
            and cfg['interpretation']['education_effect_claim'] is False, 'Bounded interpretation required')


def committed(root, commit, names):
    for name, digest in names.items():
        blob = subprocess.check_output(['git', 'show', commit+':'+name], cwd=root)
        require(hashlib.sha256(blob).hexdigest() == digest, 'Git and file bytes differ: '+name)


def parent_gate(root, cfg):
    tag = subprocess.check_output(['git', 'rev-parse', BASE_TAG], cwd=root, text=True).strip()
    require(tag == BASE_COMMIT, 'Fixed original comparison release required')
    base = read(safe(root, BASE_PARENT[0]))
    require(inventory(root, list(base['source_files'])) == base['source_files'], 'Original comparison source changed')
    committed(root, BASE_COMMIT, base['source_files'])
    check = read(safe(root, 'results/human_shell_rng_v2/verification.json'))
    require(check['status'] == 'passed' and check['summary_sha256'] == sha(safe(root, 'results/human_shell_rng_v2/summary.json')),
            'Verified original comparison required')
    for previous in ORDER[:ORDER.index(cfg['study_id'])]:
        directory = Path(root)/'results'/previous
        meta, check = read(directory/'metadata.json'), read(directory/'verification.json')
        require(meta['status'] == 'completed' and check['status'] == 'passed'
                and check['summary_sha256'] == sha(directory/'summary.json'), 'Earlier study must finish and verify: '+previous)
        for name, digest in read(directory/'artifact_hashes.json').items():
            require(sha(safe(directory, name)) == digest, 'Earlier result bytes changed')
    return {'original_comparison_preserved': True, 'preceding_studies_verified': list(ORDER[:ORDER.index(cfg['study_id'])]),
            'original_v1_rng_limitation_preserved': True, 'new_human_data_collected': False,
            'failed_scientific_hypotheses_do_not_block_recorded_next_study': True}


def external_inventory(root, cfg, cache):
    name = cfg.get('external_manifest')
    if name is None: return {}
    require(cache is not None and not Path(cache).resolve().is_relative_to(Path(root).resolve()), 'External raw-data cache required')
    manifest = read(safe(root, name)); result = {}
    for filename, spec in manifest['files'].items():
        path = safe(cache, filename)
        require(path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], 'Original external bytes differ')
        result[filename] = {'bytes': spec['bytes'], 'sha256': spec['sha256']}
    return result


def validate_registration(root, cfg, config, reg, cache=None):
    validate_envelope(root, cfg)
    require(reg['config'] == cfg and reg['study_id'] == cfg['study_id']
            and reg['config_sha256'] == sha(config), 'Registered criteria differ')
    for group in ('source_files', 'parent_files', 'graph_files', 'input_files'):
        names = cfg.get(group, [])
        actual = inventory(root, names) if names else {}
        require(reg[group] == actual, 'Frozen inventory changed: '+group)
    require(reg['external_files'] == external_inventory(root, cfg, cache)
            and reg['gate'] == parent_gate(root, cfg), 'Frozen source or parent gate changed')
    stamp = datetime.fromisoformat(reg['registered_at_utc'])
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
            and stamp <= datetime.now(timezone.utc), 'Prior UTC registration required')
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', reg['source_git_commit'], 'HEAD'], cwd=root)
    committed(root, reg['source_git_commit'], reg['source_files'])
    committed(root, reg['source_git_commit'], reg['parent_files'])


def register(root, config, cache, out):
    cfg = read(config); validate_envelope(root, cfg)
    module = importlib.import_module(cfg['module']); module.validate_config(cfg)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    source, parents = inventory(root, cfg['source_files']), inventory(root, cfg['parent_files'])
    committed(root, commit, source); committed(root, commit, parents)
    reg = {'study_id': cfg['study_id'], 'registration_type': 'prospective_bounded_followup',
           'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': commit,
           'config': cfg, 'config_sha256': sha(config), 'source_files': source, 'parent_files': parents,
           'graph_files': inventory(root, cfg['graph_files']) if cfg['graph_files'] else {},
           'input_files': inventory(root, cfg.get('input_files', [])) if cfg.get('input_files') else {},
           'external_files': external_inventory(root, cfg, cache), 'gate': parent_gate(root, cfg),
           'notice': 'Prior related outcomes known; new scope, fixed criteria and budget frozen before this computation. Private prospective registration, not blinded human confirmation.'}
    write_new(out, reg); return reg


def csv_write(path, rows):
    require(bool(rows), 'Nonempty measurement records required')
    fields = list(rows[0]); require(all(set(row) == set(fields) for row in rows), 'Uniform measurement schema required')
    with Path(path).open('x', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def run(root, config, registration, cache, out):
    import numpy as np
    cfg, reg = read(config), read(registration)
    validate_registration(root, cfg, config, reg, cache)
    module = importlib.import_module(cfg['module']); module.validate_config(cfg)
    require(not Path(out).exists(), 'New output directory required')
    out = Path(out); out.mkdir(parents=True)
    with (out/'registration.json').open('xb') as f: f.write(Path(registration).read_bytes())
    meta = {'study_id': cfg['study_id'], 'status': 'running', 'stage': 'produce',
            'started_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': reg['source_git_commit'],
            'registration_sha256': sha(registration), 'config_sha256': sha(config),
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__},
            'new_human_experiment': False, 'shared_neural_mechanism_tested': False, 'education_effect_tested': False}
    try:
        arrays, rows = module.produce(cfg, Path(root), cache)
        require(arrays and all(type(k) is str and np.asarray(v).dtype != object for k, v in arrays.items()), 'Numeric non-object arrays required')
        require(all(np.all(np.isfinite(v)) for v in arrays.values()), 'Finite raw arrays required')
        np.savez_compressed(out/'arrays.npz', **arrays); csv_write(out/'measurements.csv', rows)
        meta['stage'] = 'prespecified_summary'
        summary = module.summarize(cfg, arrays, rows)
        write_new(out/'summary.json', summary)
        meta.update(status='completed', stage='completed', counts=summary.get('counts', {}),
                    measurement_records=len(rows), numeric_arrays=len(arrays))
    except Exception as error:
        meta.update(status='failed', error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        meta['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_new(out/'metadata.json', meta)
        write_new(out/'artifact_hashes.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    return meta


def verify_envelope(root, config, registration, result_dir, cache=None):
    cfg, reg = read(config), read(registration)
    validate_registration(root, cfg, config, reg, cache)
    directory = Path(result_dir); meta = read(directory/'metadata.json')
    require((directory/'registration.json').read_bytes() == Path(registration).read_bytes(), 'Registration copy differs')
    require(meta['study_id'] == cfg['study_id'] and meta['status'] == 'completed' and meta['stage'] == 'completed'
            and meta['new_human_experiment'] is False and meta['shared_neural_mechanism_tested'] is False
            and meta['education_effect_tested'] is False and meta['source_git_commit'] == reg['source_git_commit']
            and meta['registration_sha256'] == sha(registration) and meta['config_sha256'] == sha(config), 'Completed frozen run required')
    stamps = [datetime.fromisoformat(x) for x in (reg['registered_at_utc'], meta['started_at_utc'], meta['completed_at_utc'])]
    require(stamps == sorted(stamps)
            and all(x.tzinfo is not None and x.utcoffset().total_seconds() == 0 for x in stamps)
            and stamps[-1] <= datetime.now(timezone.utc), 'Prospective UTC chronology required')
    items = read(directory/'artifact_hashes.json'); require(set(items) == set(PRIMARY), 'Exact primary output inventory required')
    for name, digest in items.items(): require(sha(safe(directory, name)) == digest, 'Primary artifact changed')
    require(meta['counts'] == read(directory/'summary.json').get('counts', {}), 'Metadata and summary counts differ')
    return cfg, reg, meta
