"""Byte and chronology gates for a read-only, descriptive cone-frontier audit.

Saved parent arrays and trial CSV bytes are analysis inputs. No graph constructor,
learner, or scientific parent producer is imported by this integrity module.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import re
import sys

from .representation_envelope import (
    ancestor, committed, csv_write, git, inventory, read, relative, require,
    safe, sha, utc, write_new,
)

STUDY_ID = 'cone_frontier_v1'
MODULE = 'drosolearn.cone_frontier'
PROTOCOL = 'docs/cone-frontier-protocol.md'
CONFIG = 'configs/cone_frontier_v1.json'
REGISTRATION = 'results/registration_cone_frontier_v1.json'
RESULT = 'results/cone_frontier_v1'
BASE_TAG = 'normalization-loss-v1'
BASE_COMMIT = 'f1ada01a904283a1036b4dc77b10fb3882194750'
REPOSITORY = 'https://github.com/CauchyRiemannEquations/DrosoLearn'
PRIMARY = ('registration.json', 'metadata.json', 'arrays.npz', 'measurements.csv', 'summary.json')
REUSED_SOURCE = frozenset({'pyproject.toml','requirements-lock.txt','src/drosolearn/__init__.py',
    'src/drosolearn/representation_envelope.py','src/drosolearn/normalization_envelope.py'})
SOURCE_FILES = REUSED_SOURCE | frozenset({CONFIG,PROTOCOL,'src/drosolearn/cone_geometry.py',
    'src/drosolearn/cone_frontier.py','src/drosolearn/cone_envelope.py','scripts/run_cone_frontier.py',
    'scripts/verify_cone_frontier.py','tests/test_cone_geometry.py','tests/test_cone_frontier.py',
    'tests/test_cone_envelope.py','tests/test_cone_frontier_verify.py'})
PARENT_FILES = frozenset({'results/registration_normalization_loss_v1.json',
    'results/normalization_loss_publication_proof.json','results/normalization_loss_archive_validation.json',
    *('results/normalization_loss_v1/'+name for name in (*PRIMARY,'verification.json','artifact_hashes.json')),
    'results/selective_spillover_v1/arrays.npz'})
EVOLVING_INDEXES = frozenset({'README.md', 'docs/research-roadmap.md', 'docs/repository-status.json'})

def validate_envelope(root, cfg, config):
    require(type(cfg) is dict and cfg.get('study_id') == STUDY_ID
            and cfg.get('module') == MODULE and cfg.get('protocol') == PROTOCOL
            and cfg.get('scope') == 'retrospective_descriptive_cone_frontier',
            'Fixed read-only descriptive audit implementation required')
    require(relative(root, config) == CONFIG, 'Fixed configuration path required')
    for key, expected in (('source_files', SOURCE_FILES), ('parent_files', PARENT_FILES)):
        values = cfg.get(key)
        require(type(values) is list and len(values) == len(expected) and set(values) == expected,
                'Exact unique frozen inventory required: ' + key)
    require(cfg.get('external_manifest') is None, 'No external data in this saved-results audit')
    interpretation = cfg.get('interpretation', {})
    require(all(interpretation.get(key) is False for key in (
        'new_human_experiment', 'shared_neural_mechanism_claim', 'education_effect_claim',
        'new_training', 'new_seed_draws', 'new_effect_hypothesis_test')),
        'Bounded read-only descriptive interpretation required')


def parent_gate(root):
    from .normalization_envelope import verify_envelope as prior_envelope
    require(git(root,'rev-parse',BASE_TAG).decode().strip()==BASE_COMMIT,'Pinned completed normalization release required')
    ancestor(root,BASE_COMMIT)
    cfg,reg,meta=prior_envelope(root,safe(root,'configs/normalization_loss_v1.json'),
        safe(root,'results/registration_normalization_loss_v1.json'),Path(root)/'results/normalization_loss_v1',
        safe(root,'results/normalization_loss_publication_proof.json'))
    verification=read(safe(root,'results/normalization_loss_v1/verification.json'))
    require(verification.get('status')=='passed' and verification.get('scientific_producer_imported') is False
        and verification.get('summary_sha256')==sha(safe(root,'results/normalization_loss_v1/summary.json'))
        and verification.get('registration_sha256')==meta['registration_sha256'],'Prior independent normalization audit required')
    checks=verification['counts']
    require(checks['arrays']==11 and checks['parent_trial_records_checked']==96000
        and checks['trial_destination_effects_checked']==384000 and checks['measurement_records_checked']==6480
        and checks['new_training_trials']==checks['new_random_draws']==0,'Prior bounded descriptive counts required')
    require(utc(meta['completed_at_utc'])<=utc(verification['verified_at_utc'])<=datetime.now(timezone.utc),'Prior audit chronology')
    pinned=inventory(root,sorted(REUSED_SOURCE|PARENT_FILES));committed(root,BASE_COMMIT,pinned)
    archive=read(safe(root,'results/normalization_loss_archive_validation.json'))
    require(archive['status']=='passed' and archive['scientific_verification_sha256']==sha(safe(root,'results/normalization_loss_v1/verification.json'))
        and archive['registration_sha256']==meta['registration_sha256'],'Prior normalization archive binding')
    snapshot=archive['current_file_sha256']
    for name,digest in snapshot.items():
        if name not in EVOLVING_INDEXES:require(sha(safe(root,name))==digest,'Prior immutable archive changed:'+name)
    committed(root,BASE_COMMIT,snapshot)
    require(archive['literal_baseline_receipt_sha256']==sha(safe(root,'results/normalization_loss_baseline_preservation.json'))
        and read(safe(root,'results/normalization_loss_baseline_preservation.json'))['status']=='passed','Prior literal byte receipt required')
    for group in ['source_files','parent_files']:
        require(all(sha(safe(root,name))==digest for name,digest in reg[group].items()),'Prior frozen '+group+' changed')
        committed(root,reg['source_git_commit'],reg[group])
    status=read(safe(root,'docs/repository-status.json'))
    require(status['selective_spillover_H_selective'] is False and status['selective_spillover_H_acquisition'] is False
        and status['selective_spillover_primary_acquisition_pass_seeds']==0,'Prior failures preserved')
    return {'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,'prior_normalization_independent_audit_reused':True,
        'prior_selective_full_replay_reused_not_rerun':True,'pinned_files':len(pinned),'archive_snapshot_files':len(snapshot),
        'immutable_source_parent_and_baseline_Git_bytes_checked':True,'prior_negative_human_and_acquisition_preserved':True,
        'new_training_calls':0,'new_random_seed_draws':0,'new_graph_or_feature_constructions':0}


def frozen_inventory(root, cfg):
    return {group: inventory(root, cfg[group]) for group in ('source_files', 'parent_files')}


def validate_counts(cfg,summary,row_count):
    seeds,n=cfg['model_seeds'],cfg['training_trials'];count=6*len(seeds)*n;c=summary.get('counts',{})
    expected={'analyzed_parent_trial_records':count,'candidate_trial_records':count*7,'candidate_destination_effects':count*7*4,
        'measurement_records':6*7*len(seeds)*3,'conditions':6,'model_seeds':len(seeds),'new_training_trials':0,
        'new_head_update_calls':0,'new_internal_update_calls':0,'new_value_update_calls':0,'new_random_draws':0,'fresh_model_constructions':0}
    require(set(c)==set(expected)|{'nonzero_parent_updates','unique_optimization_cases'},'Exact cone work count keys')
    require(all(type(c[k]) is int and c[k]==v for k,v in expected.items()) and row_count==expected['measurement_records'],'Fixed cone work counts')
    require(type(c['nonzero_parent_updates']) is int and 0<c['nonzero_parent_updates']<=count
        and type(c['unique_optimization_cases']) is int and 0<c['unique_optimization_cases']<=4*c['nonzero_parent_updates'],'Bounded cached optimizations')
    require(summary.get('study_id')==STUDY_ID and summary.get('status')=='completed'
        and summary.get('scope')=='retrospective_descriptive_cone_frontier','Completed geometric comparison required')


def register(root, config, out):
    root, config, out = Path(root).resolve(), Path(config).resolve(), Path(out).resolve()
    require(relative(root, out) == REGISTRATION and not out.exists(), 'New fixed registration path required')
    cfg = read(config)
    validate_envelope(root, cfg, config)
    frozen = frozen_inventory(root, cfg)
    commit = git(root, 'rev-parse', 'HEAD').decode().strip()
    for values in frozen.values():
        committed(root, commit, values)
    gate = parent_gate(root)
    importlib.import_module(MODULE).validate_config(cfg)
    reg = {'study_id': STUDY_ID, 'registration_type': 'retrospective_descriptive_cone_frontier',
           'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'source_git_commit': commit,
           'config': cfg, 'config_sha256': sha(config), 'registration_path': REGISTRATION,
           'baseline_tag': BASE_TAG, 'baseline_commit': BASE_COMMIT, **frozen, 'gate': gate,
           'notice': 'Parent outcomes are already known; this saved-results descriptive decomposition and reporting are frozen before its arithmetic. Descriptive analysis only, without new training, seeds, animals or effect hypothesis tests.'}
    write_new(out, reg)
    return reg


def validate_registration(root, config, registration):
    root = Path(root).resolve()
    require(relative(root, registration) == REGISTRATION, 'Fixed registration location required')
    cfg, reg = read(config), read(registration)
    validate_envelope(root, cfg, config)
    require(reg.get('study_id') == STUDY_ID and reg.get('config') == cfg
            and reg.get('config_sha256') == sha(config) and reg.get('registration_path') == REGISTRATION
            and reg.get('baseline_tag') == BASE_TAG and reg.get('baseline_commit') == BASE_COMMIT,
            'Registered audit/config identity changed')
    frozen = frozen_inventory(root, cfg)
    require(all(reg.get(group) == values for group, values in frozen.items()), 'Frozen file inventory changed')
    ancestor(root, reg['source_git_commit'])
    ancestor(root, BASE_COMMIT, reg['source_git_commit'])
    for values in frozen.values():
        committed(root, reg['source_git_commit'], values)
    require(reg.get('gate') == parent_gate(root), 'Frozen parent evidence gate changed')
    require(utc(reg['registered_at_utc']) <= datetime.now(timezone.utc), 'Prior registration required')
    return cfg, reg


def publication(root, registration, reg, proof):
    proof_name = relative(root, proof)
    safe(root, proof_name)
    value = read(proof)
    require(value.get('study_id') == STUDY_ID and value.get('registration_sha256') == sha(registration)
            and value.get('source_commit') == reg['source_git_commit']
            and value.get('remote_bytes_match') is True and value.get('visibility') == 'private',
            'Exact private remote-registration byte proof required')
    published = value.get('published_commit')
    require(type(published) is str and re.fullmatch(r'[0-9a-f]{40}', published) is not None,
            'Exact published Git commit required')
    ancestor(root, reg['source_git_commit'], published)
    ancestor(root, published)
    require(value.get('url') == REPOSITORY + '/blob/' + published + '/' + REGISTRATION,
            'Exact published registration URL required')
    require(git(root, 'show', published + ':' + REGISTRATION) == Path(registration).read_bytes(),
            'Published commit registration bytes differ')
    registered, checked = utc(reg['registered_at_utc']), utc(value['checked_at_utc'])
    require(registered < checked <= datetime.now(timezone.utc), 'Publication must follow registration')
    return proof_name, value


def run(root, config, registration, proof, out):
    import numpy as np
    root, out = Path(root).resolve(), Path(out).resolve()
    require(relative(root, out) == RESULT and not out.exists(), 'New fixed result directory required')
    cfg, reg = validate_registration(root, config, registration)
    proof_name, evidence = publication(root, registration, reg, proof)
    module = importlib.import_module(MODULE)
    module.validate_config(cfg)
    start = datetime.now(timezone.utc)
    require(utc(evidence['checked_at_utc']) < start, 'Remote byte proof must precede saved-results arithmetic')
    out.mkdir()
    with (out / 'registration.json').open('xb') as handle:
        handle.write(Path(registration).read_bytes())
    meta = {'study_id': STUDY_ID, 'status': 'running', 'stage': 'saved_results_analysis',
            'started_at_utc': start.isoformat(), 'source_git_commit': reg['source_git_commit'],
            'registration_sha256': sha(registration), 'config_sha256': sha(config),
            'publication_proof': proof_name, 'publication_proof_sha256': sha(proof),
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__},
            'analysis_only': True, 'new_training_calls': 0, 'new_random_seed_draws': 0,
            'new_graph_or_feature_constructions': 0,
            'new_human_experiment': False, 'shared_neural_mechanism_tested': False,
            'education_effect_tested': False, 'new_effect_hypothesis_test': False}
    try:
        arrays, rows = module.produce(cfg, root)
        require(type(arrays) is dict and bool(arrays)
                and all(type(key) is str and key and np.asarray(value).dtype.kind in 'biufc'
                        and np.all(np.isfinite(value)) for key, value in arrays.items()),
                'Finite numeric non-object analysis arrays required')
        np.savez_compressed(out / 'arrays.npz', **arrays)
        csv_write(out / 'measurements.csv', rows)
        meta['stage'] = 'prespecified_descriptive_summary'
        summary = module.summarize(cfg, arrays, rows)
        require(type(summary) is dict and type(summary.get('counts')) is dict,
                'Explicit descriptive summary counts required')
        validate_counts(cfg, summary, len(rows))
        write_new(out / 'summary.json', summary)
        meta.update(status='completed', stage='completed', counts=summary['counts'],
                    measurement_records=len(rows), numeric_arrays=len(arrays))
    except Exception as error:
        meta.update(status='failed', error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        meta['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_new(out / 'metadata.json', meta)
        write_new(out / 'artifact_hashes.json', {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()})
    return meta


def verify_envelope(root, config, registration, result_dir, publication_proof):
    import numpy as np
    root, directory = Path(root).resolve(), Path(result_dir).resolve()
    require(relative(root, directory) == RESULT and directory.is_dir(), 'Fixed completed result directory required')
    cfg, reg = validate_registration(root, config, registration)
    proof_name, evidence = publication(root, registration, reg, publication_proof)
    meta = read(directory / 'metadata.json')
    require((directory / 'registration.json').read_bytes() == Path(registration).read_bytes(),
            'Saved registration copy differs')
    require(meta.get('study_id') == STUDY_ID and meta.get('status') == meta.get('stage') == 'completed'
            and meta.get('source_git_commit') == reg['source_git_commit']
            and meta.get('registration_sha256') == sha(registration) and meta.get('config_sha256') == sha(config)
            and meta.get('publication_proof') == proof_name
            and meta.get('publication_proof_sha256') == sha(publication_proof)
            and meta.get('analysis_only') is True
            and all(type(meta.get(key)) is int and meta[key] == 0 for key in (
                'new_training_calls', 'new_random_seed_draws', 'new_graph_or_feature_constructions'))
            and all(meta.get(key) is False for key in (
                'new_human_experiment', 'shared_neural_mechanism_tested',
                'education_effect_tested', 'new_effect_hypothesis_test')),
            'Completed frozen read-only analysis identity required')
    times = [utc(reg['registered_at_utc']), utc(evidence['checked_at_utc']),
             utc(meta['started_at_utc']), utc(meta['completed_at_utc'])]
    require(times[0] < times[1] < times[2] <= times[3] <= datetime.now(timezone.utc),
            'Registration-publication-analysis UTC chronology required')
    artifacts = read(directory / 'artifact_hashes.json')
    require(type(artifacts) is dict and set(artifacts) == set(PRIMARY),
            'Exact five primary artifact hashes required')
    for name, checksum in artifacts.items():
        require(sha(safe(directory, name)) == checksum, 'Primary artifact changed: ' + name)
    summary = read(directory / 'summary.json')
    require(type(meta.get('counts')) is dict and meta['counts'] == summary.get('counts'),
            'Summary and metadata counts differ')
    with (directory / 'measurements.csv').open(encoding='utf8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
                'Unique saved measurement columns required')
        rows = list(reader)
    with np.load(directory / 'arrays.npz', allow_pickle=False) as arrays:
        require(type(meta.get('numeric_arrays')) is int and meta['numeric_arrays'] == len(arrays.files)
                and len(arrays.files) > 0
                and all(arrays[key].dtype.kind in 'biufc' and np.all(np.isfinite(arrays[key]))
                        for key in arrays.files), 'Saved finite analysis array inventory differs')
    require(type(meta.get('measurement_records')) is int and meta['measurement_records'] == len(rows)
            and len(rows) > 0, 'Saved measurement row count differs')
    validate_counts(cfg, summary, len(rows))
    return cfg, reg, meta
