"""Literal evidence synthesis; no new scientific outcome estimates or learning."""
import json,re
from pathlib import Path
import numpy as np
from .closure_envelope import read,sha,safe,require,git,committed,utc,INDEXES

ORDER=('representation_update_v1','update_budget_v1','spillover_tradeoff_v1','signed_spillover_v1',
    'selective_spillover_v1','normalization_loss_v1','cone_frontier_v1','balanced_learning_v1',
    'balanced_pathways_v1','gated_learning_v1','gated_pathways_v1','gated_factors_v1',
    'self_blend_learning_v1','self_blend_factors_v1','probability_alignment_v1','objective_learning_v1','objective_pathways_v1')
FLAGS=('metadata_completed','independent_passed','primary_hashes_match','release_tag_bytes_match',
    'source_input_hashes_match','private_publication_verified','chronology_ordered','registered_hypothesis_literals_match')
VERDICTS=('supported','not_supported','not_assessable','descriptive','untested')
INTERPRETATION={'bounded_local_project_completed':True,'original_phase5_preserved':True,
    'all_failures_and_negative_decisions_preserved':True,'new_effect_or_learning_claims':False,
    'pooled_effect_estimates_computed':False,'old_test_counts_summed_as_independent':False,
    'novel_generalization_supported_by_this_synthesis':False,'human_similarity_supported':False,
    'shared_neural_mechanism_verified':False,'education_effect_tested':False,
    'automatic_continuation_enabled':False,'long_term_questions_retained_without_automatic_execution':True}


def pointer(document,path):
    require(type(path) is str and (path=='' or path.startswith('/')),'RFC6901 pointer')
    value=document
    if not path:return value
    for raw in path[1:].split('/'):
        require(re.search('~(?![01])',raw) is None,'Canonical pointer escape')
        key=raw.replace('~1','/').replace('~0','~')
        if type(value) is dict:
            require(key in value,'Existing literal key');value=value[key]
        elif type(value) is list:
            require(re.fullmatch('0|[1-9][0-9]*',key) is not None and int(key)<len(value),
                'Canonical bounded array pointer');value=value[int(key)]
        else:raise ValueError('Pointer traverses scalar')
    return value


def same(a,b):
    if type(a) is not type(b):return False
    if type(a) is dict:return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if type(a) is list:return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def validate_config(cfg):
    fixture=cfg.get('scope')=='integration_test'
    require(cfg.get('study_id')=='followup_closure_v1' and (fixture or
        cfg.get('scope')=='retrospective_bounded_followup_synthesis'),'Fixed descriptive closure scope')
    require(cfg.get('study_order')==list(ORDER) and cfg.get('module')=='drosolearn.followup_closure'
        and cfg.get('evidence_manifest')=='configs/followup_closure_evidence_v1.json'
        and cfg.get('protocol')=='docs/followup-closure-protocol.md','Fixed scope and complete 17-study order')
    require(cfg.get('new_confirmatory_hypothesis') is False and cfg.get('new_effect_aggregation') is False
        and cfg.get('numeric_tolerance')==0. and cfg.get('external_manifest') is None,'Exact literals; no new effect test')
    i=cfg.get('interpretation',{})
    require(all(i.get(k) is False for k in ('new_training','new_seed_draws','new_direction_optimization',
        'new_human_experiment','shared_neural_mechanism_claim','education_effect_claim','automatic_continuation_enabled'))
        and i.get('completed_scope_is_bounded') is True and i.get('long_term_questions_retained') is True,
        'Bounded completed scope with untested long-term questions')


def audit_study(root,s):
    root=Path(root);reg=read(safe(root,s['registration']));meta=read(safe(root,s['metadata']))
    summary=read(safe(root,s['summary']));verify=read(safe(root,s['verification']))
    archive=read(safe(root,s['archive']));proof=read(safe(root,s['publication_proof']))
    require(reg['study_id']==meta['study_id']==summary['study_id']==verify['study_id']==s['id'],
        'Original study identities')
    require(meta['status']=='completed' and summary['status']=='completed' and archive['status']=='passed',
        'Original completed metadata/summary/archive')
    require(verify['status']=='passed' and verify['registration_sha256']==sha(safe(root,s['registration']))
        and verify['summary_sha256']==sha(safe(root,s['summary'])),'Original independent PASS and exact outcome bytes')
    require(reg['source_git_commit']==s['source_commit']==meta['source_git_commit']
        and meta['registration_sha256']==sha(safe(root,s['registration']))
        and meta['config_sha256']==reg['config_sha256'] and same(meta['counts'],summary['counts']),
        'Original source/config/count bindings')
    artifacts=read(safe(root,s['artifact_hashes']));directory=safe(root,s['result_dir'])
    expected={'registration.json','metadata.json','arrays.npz','summary.json',
        'measurements.csv.gz' if s['id']=='objective_learning_v1' else 'measurements.csv'}
    require(set(artifacts)==expected and set(s['raw_files'])=={str((directory/n).relative_to(root)).replace('\\','/') for n in expected},
        'Five original primary files, preserving compressed versus plain trial bytes')
    require(all(sha(safe(directory,n))==h for n,h in artifacts.items()),'Original five artifact hashes')
    require((directory/'registration.json').read_bytes()==safe(root,s['registration']).read_bytes(),'Original registration copy')
    require(git(root,'rev-parse',s['completion_tag']).decode().strip()==s['completion_commit'],'Original pinned completion tag')
    git(root,'merge-base','--is-ancestor',s['source_commit'],s['completion_commit'])
    paths=set(s['raw_files'])|{s[k] for k in ('registration','summary','verification','archive','protocol','reproduction',
        'findings','publication_proof','artifact_hashes','metadata')}|set(s.get('preservation_files',[]))
    committed(root,s['completion_commit'],paths)
    for group in ('source_files','parent_files','graph_files','input_files'):
        values=reg.get(group,{})
        require(type(values) is dict,'Original immutable input map')
        for name,digest in values.items():require(sha(safe(root,name))==digest,'Original source/input SHA: '+name)
        committed(root,s['source_commit'],values)
    values=archive.get('current_file_sha256',{})
    require(type(values) is dict and all(sha(safe(root,n))==h for n,h in values.items() if n not in INDEXES),
        'Original archived non-index evidence, including failed attempts')
    require(proof.get('remote_bytes_match') is True and proof.get('visibility')=='private'
        and proof['source_commit']==s['source_commit'] and proof['registration_sha256']==sha(safe(root,s['registration']))
        and meta['publication_proof']==s['publication_proof']
        and meta['publication_proof_sha256']==sha(safe(root,s['publication_proof'])),'Original private publication evidence')
    git(root,'merge-base','--is-ancestor',s['source_commit'],proof['published_commit'])
    require(git(root,'show',proof['published_commit']+':'+s['registration'])==safe(root,s['registration']).read_bytes(),
        'Original published registration bytes')
    times=[reg['registered_at_utc'],proof['checked_at_utc'],meta['started_at_utc'],meta['completed_at_utc'],verify['verified_at_utc']]
    require(utc(times[0])<utc(times[1])<utc(times[2])<=utc(times[3])<=utc(times[4]),'Original prospective chronology')
    for observation in s['hypothesis_observations']:
        require(same(pointer(summary,observation['pointer']),observation['expected']),'Original hypothesis literal unchanged')
    return {'id':s['id'],'kind':s['kind'],'completion_tag':s['completion_tag'],'completion_commit':s['completion_commit'],
        'source_commit':s['source_commit'],'registration_sha256':sha(safe(root,s['registration'])),
        'summary_sha256':sha(safe(root,s['summary'])),'verification_sha256':sha(safe(root,s['verification'])),
        'archive_sha256':sha(safe(root,s['archive'])),'integrity_flags':[True]*8,
        'cohort':s['cohort'],'hypothesis_observations':s['hypothesis_observations']}


def reproduce_record(cfg,root):
    path=cfg['reproduction_receipt'];new=read(safe(root,path));oldpath='results/objective_pathways_v1/verification.json'
    old=read(safe(root,oldpath))
    require(new['status']==old['status']=='passed' and new['study_id']==old['study_id']=='objective_pathways_v1'
        and new['registration_sha256']==old['registration_sha256'] and new['summary_sha256']==old['summary_sha256']
        and same(new['counts'],old['counts']) and new.get('scientific_producer_imported') is False
        and new.get('new_helper_imported') is False and utc(new['verified_at_utc'])>utc(old['verified_at_utc']),
        'Same independently recomputed saved arithmetic in a separate fresh receipt')
    return {'receipt_path':path,'sha256':sha(safe(root,path)),'original_verification_sha256':sha(safe(root,oldpath)),
        'summary_sha256':new['summary_sha256'],'counts':new['counts'],'separate_new_receipt':True,'past_full_science_rerun':False}


def analyze(cfg,root):
    validate_config(cfg);manifest=read(safe(root,cfg['evidence_manifest']))
    require(manifest['study_id']==cfg['study_id'] and [s['id'] for s in manifest['studies']]==list(ORDER),
        'All original post-Phase5 studies in fixed order')
    studies=[audit_study(root,s) for s in manifest['studies']]
    require(sum(s['kind']=='actual_learning' for s in studies)==8
        and sum(s['kind']=='saved_record_analysis' for s in studies)==9,'Eight original learning and nine reused-record studies')
    records=[];rows=[];ledger=[];ids=set()
    for index,c in enumerate(manifest['claims']):
        require(c['id'] not in ids and c['verdict'] in VERDICTS and type(c['question']) is int and 1<=c['question']<=5
            and type(c['level']) is int and 0<=c['level']<=5 and c['statement'] and c['limit'] and c['evidence'],
            'Unique complete bounded claims with explicit interpretation limits')
        require(not(c['verdict']=='supported' and c['level']>=3),'No new positive generalization/human/mechanism evidence')
        ids.add(c['id']);sources={}
        for e in c['evidence']:
            require(same(pointer(read(safe(root,e['file'])),e['pointer']),e['expected']),'Exact claim literal with its original type')
            sources[e['file']]=sha(safe(root,e['file']))
        records.append({**c,'matched':True,'source_sha256':sources});ledger.append([index,c['question'],c['level'],VERDICTS.index(c['verdict'])])
        compact=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        rows.append({'claim_id':c['id'],'question':c['question'],'level':c['level'],'verdict':c['verdict'],
            'statement':c['statement'],'interpretation_limit':c['limit'],'source_count':len(sources),
            'evidence_json':compact(c['evidence']),'source_sha256_json':compact(sources),'matched':1})
    for anchor in manifest['retained_anchors']:
        if anchor['verification'] is not None:require(read(safe(root,anchor['verification']))['status']=='passed','Retained anchor independent PASS')
        for n in anchor['paths']:require(safe(root,n).is_file(),'Retained anchor source/report exists')
    arrays={'study_integrity_flags':np.array([s['integrity_flags'] for s in studies],dtype=bool),
        'claim_ledger':np.array(ledger,dtype=np.int64)}
    summary={'study_id':cfg['study_id'],'status':'completed','scope':cfg['scope'],'studies':studies,'claims':records,
        'retained_anchors':manifest['retained_anchors'],'integrity_flag_order':list(FLAGS),'verdict_code_order':list(VERDICTS),
        'counts':{'studies':17,'claims':len(records),'literal_observations':sum(len(c['evidence']) for c in records),
            'actual_learning_registrations':8,'saved_record_registrations':9,'numeric_arrays':2,'measurement_records':len(rows),
            'new_training_trials':0,'new_model_graph_rng_optimization_calls':0,'new_effect_tests':0,'new_human_experiments':0,
            'past_full_science_reruns':0,'saved_record_independent_rechecks':1},
        'reproduction':reproduce_record(cfg,root),'interpretation':INTERPRETATION.copy()}
    return arrays,rows,summary


def produce(cfg,root):
    a,rows,_=analyze(cfg,root);return a,rows


def summarize(cfg,a,rows,root):
    expected,ordered,summary=analyze(cfg,root)
    require(set(a)==set(expected) and all(np.array_equal(a[k],v) for k,v in expected.items()) and rows==ordered,
        'Exact closure arrays and evidence table');return summary
