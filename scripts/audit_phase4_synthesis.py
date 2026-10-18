"""Freeze and check the bounded Phase 4 evidence synthesis, without effect estimation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ('quantity_ratio', 'repeated_experience', 'generalization_transfer', 'reward_reversal', 'learning_interference', 'learning_order')
FORBIDDEN_FLAGS = ('direct_level4_test_completed', 'shared_neural_mechanism_verified', 'education_effect_tested', 'all_phase2_phase3_tasks_completed')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def path(root, name):
    result = (root / name).resolve()
    require(type(name) is str and result.is_relative_to(root.resolve()) and result.is_file(), 'Existing repository evidence required')
    return result

def read(filename):
    return json.loads(filename.read_text(encoding='utf8'))

def sha(filename):
    return hashlib.sha256(filename.read_bytes()).hexdigest()

def write_new(filename, value):
    with filename.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def inventory(root, names):
    require(type(names) is list and len(names) == len(set(names)) and bool(names), 'Unique nonempty evidence inventory required')
    return {name: sha(path(root, name)) for name in names}

def validate_config(cfg):
    require(cfg['study_id'] == 'phase4_synthesis_v1' and [row['id'] for row in cfg['features']] == list(FEATURES), 'All six ordered research features required')
    require(all(cfg.get(key) is False for key in FORBIDDEN_FLAGS), 'Untested inference flags must remain false')
    for row in cfg['features']:
        require(set(row) == {'id','metric','unit','human','model','difference','unlock'} and set(row['human']) == {'honore','markovic','openneuro'}, 'Full three-source comparability schema required')
        require(all(isinstance(row[key], str) and row[key].strip() for key in ('metric','unit','model','difference','unlock')), 'Each measurement, difference and unlocking evidence must be explicit')
        require(all(isinstance(value, str) and value for value in row['human'].values()) and row['human']['openneuro'] == 'semantic_gate_closed', 'OpenNeuro semantics must not be silently upgraded')
    require(not set(cfg['source_files']) & set(cfg['evidence_files']), 'Source and previously completed evidence must be separate')

def support_counts(streams):
    """Only recount integer support and ledger partitions, never average human or seed rates."""
    output = {}
    ids = set()
    for stream in streams:
        require(stream['stream_id'] not in ids, 'Repeated stream')
        ids.add(stream['stream_id'])
        kind = stream['kind']
        require(kind in ('human','model'), 'Unknown observation unit')
        group = 'human' if kind == 'human' else stream['stream_id'].rsplit('/',1)[-1]
        require(group in ('human','deterministic','probabilistic','uninformative'), 'Unknown environment')
        item = output.setdefault(group, {'streams':0,'strata':0,'both_feedback':0,'feedback_0_only':0,'feedback_1_only':0,'neither':0,'feedback_0_denominator':0,'feedback_1_denominator':0,'eligible':0})
        item['streams'] += 1
        subtotal = Counter()
        for cell in stream['strata']:
            item['strata'] += 1
            ns = []
            for feedback in ('0','1'):
                values = cell['by_feedback'][feedback]
                require(all(type(values[key]) is int and values[key] >= 0 for key in ('denominator','stay','switch')) and values['stay']+values['switch'] == values['denominator'], 'Exact nonnegative integer cell partition required')
                n = values['denominator']
                ns.append(n)
                for field in ('denominator','stay','switch'):
                    subtotal[feedback,field] += values[field]
                item['feedback_'+feedback+'_denominator'] += n
            status = 'both_feedback' if all(ns) else 'feedback_0_only' if ns[0] else 'feedback_1_only' if ns[1] else 'neither'
            require(cell['support_status'] == status and cell['common_support'] is all(ns), 'Support status differs from original integer denominators')
            item[status] += 1
        require(sum(subtotal[f,'denominator'] for f in ('0','1')) == stream['eligible'] and all(subtotal[f,field] == stream['by_feedback'][f][field] for f in ('0','1') for field in ('denominator','stay','switch')), 'Strata must partition each original record in all three integer counts')
        item['eligible'] += stream['eligible']
    return output

def synthesis(root, cfg):
    validate_config(cfg)
    def evidence(name):
        require(name in cfg['evidence_files'], 'Undeclared evidence dependency')
        return read(path(root,name))
    task = evidence('results/human_task_structure_v1/audit.json')
    gate = task['gate']
    require(gate['source_coded_descriptive_metric_allowed'] is True and gate['exact_historical_task_execution_version_verified'] is False and gate['direct_human_model_similarity_allowed'] is False, 'Historical source-coded gate must be preserved exactly')
    checks = ['human_task_structure_v1','human_osf_origin_v1','human_semantics_v1','conditional_transition_v1','stochastic_feedback_network_v1','learning_ablation_v1','time_mixture_v1']
    for study in checks:
        verified = evidence('results/'+study+'/verification.json')
        require(verified['status'] in ('pass','passed'), 'A required independent verification has not passed')
        if study in ('stochastic_feedback_network_v1','learning_ablation_v1'):
            require(verified['summary_sha256'] == sha(path(root,'results/'+study+'/summary.json')), 'Prior independent verification does not bind the model summary')
        if study == 'time_mixture_v1':
            require(verified['metrics_sha256'] == sha(path(root,'results/'+study+'/metrics.json')), 'Prior exact verification does not bind the counterexample')
        if study == 'human_task_structure_v1':
            require(verified['result_sha256'] == sha(path(root,'results/'+study+'/audit.json')), 'Prior task verification does not bind the source-coded gate')
    semantic = evidence('results/human_semantics_v1/verification.json')
    require(semantic['scope'] == 'independent_fixed_code_counts_no_mapping_selection_or_performance_analysis', 'Semantic audit must not be mistaken for a mapping solution')
    origin = evidence('results/human_osf_origin_v1/verification.json')
    require(origin['source_identity_gate_passed'] is True and origin['mismatches_verified'] == 0, 'Verified original-row identity required')
    metrics = evidence('results/conditional_transition_v1/metrics.json')
    verified = evidence('results/conditional_transition_v1/verification.json')
    require(sha(path(root,'results/conditional_transition_v1/metrics.json')) == verified['metrics_sha256'], 'Prior metric verification does not bind these bytes')
    require(all(metrics[key] is False for key in ('pool_group_or_species_estimates','averaged_stratum_estimates','causal_effect_estimated')), 'No pooled rate or causal estimate permitted')
    support = support_counts(metrics['streams'])
    require(sum(v['strata'] for v in support.values()) == verified['counts']['strata'] and sum(v['eligible'] for v in support.values()) == verified['counts']['eligible'], 'Support recount must retain all original strata and eligible pairs')
    human_schema = evidence('data/human/honore_noel_2016/schema-audit.json')
    require(len(human_schema['datasets']) == 2 and all(len(d['sheets']) == 2 and all(s['participants'] == 56 for s in d['sheets']) for d in human_schema['datasets']), 'Two human aggregate tasks and original 56 people must be retained')
    return {'study_id':cfg['study_id'], 'bounded_phase4_deliverables_completed':True, **{key:False for key in FORBIDDEN_FLAGS},
        'direct_level4_verdict':'untested_not_testable_with_current_pairing',
        'feature_count':len(cfg['features']), 'human_source_cells':len(cfg['features'])*3, 'features':cfg['features'],
        'source_coded_descriptive_metric_allowed':True, 'historical_task_execution_version_verified':False,
        'openneuro_semantic_mapping_verified':False, 'verified_original_human_records':origin['raw_rows_replayed'],
        'unverified_original_human_records_retained':origin['unmatched_rows_preserved'],
        'support_counts':support, 'support_is_only_denominator_presence':True,
        'additional_learning_updates':0, 'rng_draws':0, 'new_human_participants':0, 'new_human_effect_estimates':0,
        'scope':'evidence map and integer recount of prior verified measurement; no new rate pooling, significance, similarity or biological claim'}

def register(root, config, out):
    cfg = read(config)
    validate_config(cfg)
    # Registration binds bytes only; substantive recount is deferred until run.
    obj = {'study_id':cfg['study_id'], 'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'config_sha256':sha(config), 'config':cfg, 'source_files':inventory(root,cfg['source_files']),
        'evidence_files':inventory(root,cfg['evidence_files'])}
    write_new(out,obj)
    return obj

def run(root, config, registration, out):
    cfg, reg = read(config), read(registration)
    require(not out.exists(), 'Exclusive new result path required')
    require(reg['config'] == cfg and reg['config_sha256'] == sha(config) and reg['source_files'] == inventory(root,cfg['source_files']) and reg['evidence_files'] == inventory(root,cfg['evidence_files']), 'Frozen source or evidence changed')
    require(datetime.fromisoformat(reg['registered_at_utc']) <= datetime.now(timezone.utc), 'Registration must precede run')
    started = datetime.now(timezone.utc).isoformat()
    out.mkdir(parents=True)
    write_new(out/'registration.json',reg)
    meta = {'study_id':cfg['study_id'], 'status':'running', 'started_at_utc':started,'runtime':{'python':sys.version.split()[0],'dependencies':'Python standard library only'},
        'source_git_commit':reg['source_git_commit'], 'registration_sha256':sha(registration), 'config_sha256':sha(config)}
    try:
        report = synthesis(root,cfg)
        write_new(out/'synthesis.json',report)
        meta['status']='completed'
    except Exception as error:
        meta.update(status='failed',error_type=type(error).__name__,error_message=str(error))
        raise
    finally:
        meta['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        write_new(out/'metadata.json',meta)
        write_new(out/'artifact_hashes.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    return meta

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','run'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--registration',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(), 'Output must not exist')
    result=register(args.root,args.config,args.out) if args.mode=='register' else run(args.root,args.config,args.registration,args.out)
    print(json.dumps({'status':result.get('status','registered'),'out':str(args.out)}))

if __name__=='__main__':
    main()
