"""Prospectively freeze and execute the TNU recorded-trace human comparison."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def require(ok,message):
    if not ok: raise ValueError(message)

def read(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate JSON key')
            result[key]=value
        return result
    return json.loads(Path(path).read_text(encoding='utf8'),object_pairs_hook=pairs,
                      parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def safe(root,relative):
    require(type(relative) is str and relative and '\\' not in relative and ':' not in relative and not Path(relative).is_absolute() and '..' not in Path(relative).parts,'Contained POSIX path required')
    result=(root/relative).resolve()
    require(result.is_relative_to(root.resolve()) and result.is_file(),'Existing contained input required')
    return result

def write_new(path,obj):
    with Path(path).open('x',encoding='utf8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def hashes(root,names):
    require(type(names) is list and names and len(set(names))==len(names),'Explicit unique file inventory required')
    return {name:sha(safe(root,name)) for name in names}

def validate_config(cfg):
    require(cfg['study_id']=='human_shell_v1' and cfg['scope']=='prospectively_registered_exploratory_human_comparison','Fixed study scope required')
    require(cfg['conditions']==['fixed','plastic','rewired_fixed','rewired_plastic','ann_fixed','ann_plastic','linear','q','uniform','bias','persistence'] and cfg['adaptive_conditions']==cfg['conditions'][:8],'All controls required')
    require(cfg['rates']==[.03,.1,.3] and cfg['calibration_seeds']==[13001,13002,13003] and cfg['test_seeds']==list(range(13101,13106)),'Registered candidate and seed budgets required')
    require((cfg['trials_per_person'],cfg['bootstrap_repeats'],cfg['bootstrap_seed'],cfg['equivalence_margin'],cfg['prediction_margin'],cfg['learning_gain_margin'],cfg['primary_condition'])==(216,2000,14001,.1,.01,.05,'plastic'),'Fixed measurement criteria required')
    ids=cfg['calibration_ids']+cfg['test_ids']
    require(len(cfg['calibration_ids'])==20 and len(cfg['test_ids'])==29 and len(set(ids))==49 and all(re.fullmatch(r'A[0-9]{4}',pid) for pid in ids),'Full complete cohort and disjoint people required')
    require(ids==sorted(ids,key=lambda pid:hashlib.sha256(('drosolearn-shell-v1:'+pid).encode()).hexdigest()) and 'A0016' not in ids,'Prospective ID-only split differs')

def external_inventory(root,cfg,cache):
    manifest=read(safe(root,cfg['manifest']))
    require(set(manifest['files'])=={'Dataset_TNU.zip','metadata.json','readme.pdf'},'Three original cache files required')
    require(set(manifest['allowed_complete_ids'])==set(cfg['calibration_ids']+cfg['test_ids']),'Manifest and cohort differ')
    inventory={}
    for name,spec in manifest['files'].items():
        file=safe(cache,name)
        require(file.stat().st_size==spec['bytes'] and sha(file)==spec['sha256'],'Original cache bytes changed')
        inventory[name]={'bytes':spec['bytes'],'sha256':spec['sha256']}
    require(not cache.resolve().is_relative_to(root.resolve()),'Raw human data remain outside the repository')
    return inventory

def parent_gate(root,cfg):
    result=read(safe(root,'results/phase4_synthesis_v1/synthesis.json'))
    check=read(safe(root,'results/phase4_synthesis_v1/verification.json'))
    require(check['status']=='passed' and check['synthesis_sha256']==sha(safe(root,'results/phase4_synthesis_v1/synthesis.json')) and result['direct_level4_test_completed'] is False,'Previous bounded synthesis must remain unchanged')
    require(read(safe(root,'results/time_mixture_v1/verification.json'))['status']=='passed','Measurement counterexample verification required')
    old=read(safe(root,'results/human_task_structure_v1/audit.json'))
    require(old['gate']['direct_human_model_similarity_allowed'] is False,'Historical Markovic gate must not be upgraded')
    return {'previous_human_data_gate_unchanged':True,'new_source':'TNU 10066829 explicit recorded potential trace and action semantics',
            'new_source_scope':'conditional prediction and autonomous numeric trace benchmark; original visual/motor runtime not reproduced',
            'neural_mechanism_or_drug_effect_tested':False}

def register(root,config,cache,out):
    cfg=read(config);validate_config(cfg)
    record={'study_id':cfg['study_id'],'registration_type':'exploratory_human_trace_comparison',
            'registered_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'config':cfg,'config_sha256':sha(config),'source_files':hashes(root,cfg['source_files']),
            'parent_files':hashes(root,cfg['parent_files']),'graph_files':hashes(root,cfg['graph_files']),
            'external_files':external_inventory(root,cfg,cache),'gate':parent_gate(root,cfg),
            'notice':'Source schema inspected before registration; no human behavioral summaries or model fitting performed before this freeze. Private prospective Git registration, not a blinded confirmatory study.'}
    write_new(out,record)
    return record

def write_csv(path,rows):
    require(bool(rows),'A complete nonempty measurement table is required')
    fields=list(rows[0])
    require(all(set(row)==set(fields) for row in rows),'Uniform measurement table schema required')
    with Path(path).open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def run(root,config,registration,cache,out):
    import numpy as np
    from drosolearn.human_shell_source import load_cohort
    from drosolearn.graph import load_projection
    from drosolearn.human_shell import fit,produce,summarize
    cfg,reg=read(config),read(registration);validate_config(cfg)
    stamp=datetime.fromisoformat(reg['registered_at_utc'])
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds()==0 and stamp<=datetime.now(timezone.utc),'Prior UTC registration required')
    require(reg['config']==cfg and reg['config_sha256']==sha(config) and reg['source_files']==hashes(root,cfg['source_files'])
            and reg['parent_files']==hashes(root,cfg['parent_files']) and reg['graph_files']==hashes(root,cfg['graph_files'])
            and reg['external_files']==external_inventory(root,cfg,cache) and reg['gate']==parent_gate(root,cfg),'Frozen source, inputs, criteria or parent changed')
    require(not out.exists(),'New result directory required')
    out.mkdir(parents=True)
    with (out/'registration.json').open('xb') as f:f.write(Path(registration).read_bytes())
    meta={'study_id':cfg['study_id'],'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
          'source_git_commit':reg['source_git_commit'],'registration_sha256':sha(registration),'config_sha256':sha(config),
          'runtime':{'python':sys.version.split()[0],'numpy':np.__version__},'stage':'source_validation',
          'human_model_comparison_tested':False,'shared_neural_mechanism_tested':False,'drug_effect_tested':False}
    try:
        records,audit=load_cohort(cache,read(safe(root,cfg['manifest'])))
        require(set(records)==set(cfg['calibration_ids']+cfg['test_ids']) and all(len(rows)==216 for rows in records.values()),'Full source cohort required')
        write_new(out/'cohort_audit.json',audit)
        graph=load_projection(root/cfg['graph'])
        meta['stage']='calibration'
        print('Calibrating on the 20 fixed training participants',flush=True)
        selected,cal_arrays,cal_rows=fit({pid:records[pid] for pid in cfg['calibration_ids']},cfg,graph)
        write_new(out/'selection.json',selected)
        np.savez_compressed(out/'calibration.npz',**cal_arrays)
        write_csv(out/'calibration.csv',cal_rows)
        meta['stage']='heldout_prediction_and_autonomous_trace'
        print('Testing fixed settings on the 29 held-out participants',flush=True)
        arrays,trajectory_rows,witnesses=produce({pid:records[pid] for pid in cfg['test_ids']},cfg,graph,selected)
        np.savez_compressed(out/'trajectories.npz',**arrays)
        np.savez_compressed(out/'witnesses.npz',**witnesses)
        write_csv(out/'trajectory_metrics.csv',trajectory_rows)
        meta['stage']='prespecified_summary'
        summary=summarize({pid:records[pid] for pid in cfg['test_ids']},cfg,arrays,trajectory_rows)
        write_new(out/'summary.json',summary)
        meta.update(status='completed',stage='completed',human_model_comparison_tested=True,
                    original_human_people=49,original_human_trials=49*216,calibration_people=20,heldout_people=29,
                    calibration_conditional_observations=8*3*3*20*216,
                    test_conditional_predictions=11*5*29*216,autonomous_model_choices=11*5*29*216,
                    interpretation='Predictive and bounded behavioral comparison of the recorded mixed cohort; seeds are not people or animals')
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
    parser.add_argument('mode',choices=['register','run'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--registration',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(),'Existing output cannot be overwritten')
    require(args.mode!='run' or args.registration is not None,'Registration required')
    result=register(args.root.resolve(),args.config,args.cache.resolve(),args.out) if args.mode=='register' else run(args.root.resolve(),args.config,args.registration,args.cache.resolve(),args.out)
    print(json.dumps({'status':result.get('status','registered'),'out':str(args.out)}))

if __name__=='__main__':main()
