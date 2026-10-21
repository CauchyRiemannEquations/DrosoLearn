"""Byte-preserving registration and execution envelope for bounded project closure."""
from pathlib import Path
from datetime import datetime,timezone
from functools import lru_cache
import csv,hashlib,importlib,json,re,subprocess,sys

STUDY='followup_closure_v1'
BASE_TAG='objective-pathways-v1'
BASE_COMMIT='beec7eb421b4574231f74ea0b2ede4dbc99e1ecc'
CONFIG='configs/followup_closure_v1.json'
MANIFEST='configs/followup_closure_evidence_v1.json'
PROTOCOL='docs/followup-closure-protocol.md'
REGISTRATION='results/registration_followup_closure_v1.json'
RESULT='results/followup_closure_v1'
REPRO='results/followup_closure_reproduction_v1/objective_pathways_reverification.json'
REPOSITORY='https://github.com/CauchyRiemannEquations/DrosoLearn'
INDEXES={'README.md','docs/research-roadmap.md','docs/repository-status.json'}
PRIMARY=('registration.json','metadata.json','arrays.npz','measurements.csv','summary.json')
NEW_SOURCES={CONFIG,MANIFEST,PROTOCOL,'src/drosolearn/followup_closure.py',
    'src/drosolearn/closure_envelope.py','scripts/run_followup_closure.py','scripts/verify_followup_closure.py',
    'tests/test_followup_closure.py','tests/test_followup_closure_envelope.py','tests/test_followup_closure_verify.py'}


def require(value,label):
    if not value:raise ValueError(label)


def safe(root,name):
    root=Path(root).resolve();require(type(name) is str and name and not Path(name).is_absolute(),'Relative named input')
    p=(root/name).resolve();require(p.is_relative_to(root),'Workspace-contained input');return p


def read(path):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'Duplicate JSON key');out[k]=v
        return out
    return json.loads(Path(path).read_text(encoding='utf8'),object_pairs_hook=pairs,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def write_new(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)


@lru_cache(maxsize=128)
def tree(root,commit):
    entries={}
    for record in git(root,'ls-tree','-r','-z',commit).split(b'\0'):
        if not record:continue
        head,name=record.split(b'\t',1);mode,kind,oid=head.split()
        require(kind==b'blob','Only tracked ordinary Git blobs')
        entries[name.decode('utf8')]=oid.decode()
    return entries


def committed(root,commit,files):
    entries=tree(str(Path(root).resolve()),commit)
    for name in files:
        p=safe(root,name);h=hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0')
        with p.open('rb') as f:
            while block:=f.read(1024*1024):h.update(block)
        require(entries.get(name)==h.hexdigest(),'Original Git blob differs: '+name)


def utc(value):
    d=datetime.fromisoformat(value);require(d.tzinfo is not None,'Timezone-bearing UTC evidence');return d


def source_paths(root):
    old=read(safe(root,'results/registration_objective_pathways_v1.json'))['source_files']
    require(len(old)==25,'Original 25-source reproduction snapshot');return set(old)|NEW_SOURCES


def parent_paths(manifest):
    paths=set()
    for s in manifest['studies']:
        paths.update(s[k] for k in ('registration','summary','verification','archive','protocol',
            'reproduction','findings','publication_proof','artifact_hashes','metadata'))
        paths.update(s['raw_files']);paths.update(s.get('preservation_files',[]))
    for a in manifest['retained_anchors']:
        paths.update(x for x in (a['summary'],a['verification']) if x is not None);paths.update(a['paths'])
    paths.update(e['file'] for c in manifest['claims'] for e in c['evidence'])
    paths.update(('results/objective_pathways_finalization_style_attempt01.json',
        'results/objective_pathways_finalization_style_attempt01.zip'))
    return paths


def validate_envelope(root,cfg):
    require(cfg.get('study_id')==STUDY and cfg.get('scope')=='retrospective_bounded_followup_synthesis'
        and cfg.get('module')=='drosolearn.followup_closure' and cfg.get('protocol')==PROTOCOL
        and cfg.get('evidence_manifest')==MANIFEST and cfg.get('baseline_tag')==BASE_TAG
        and cfg.get('baseline_commit')==BASE_COMMIT and cfg.get('reproduction_receipt')==REPRO,
        'Exact bounded closure configuration')
    manifest=read(safe(root,MANIFEST))
    for field,expected in (('source_files',source_paths(root)),('parent_files',parent_paths(manifest))):
        require(type(cfg.get(field)) is list and len(cfg[field])==len(expected) and set(cfg[field])==expected,
            'Exact unique closure inventory: '+field)
    require(len(cfg['source_files'])==35,'Exact 35-source closure and saved-only reproduction inventory')
    require(git(root,'rev-parse',BASE_TAG).decode().strip()==BASE_COMMIT,'Pinned last completed study')
    git(root,'merge-base','--is-ancestor',BASE_COMMIT,'HEAD')
    return manifest


def freeze_inventory(root,cfg):
    return {group:{name:sha(safe(root,name)) for name in sorted(cfg[group])}
        for group in ('source_files','parent_files')}


def register(root,config,out):
    root=Path(root).resolve();cfg=read(config)
    require(Path(config).resolve()==safe(root,CONFIG) and Path(out).resolve()==safe(root,REGISTRATION)
        and not Path(out).exists(),'Exclusive fixed closure registration')
    validate_envelope(root,cfg);importlib.import_module(cfg['module']).validate_config(cfg)
    inventory=freeze_inventory(root,cfg);commit=git(root,'rev-parse','HEAD').decode().strip()
    for paths in inventory.values():committed(root,commit,paths)
    committed(root,BASE_COMMIT,cfg['parent_files'])
    reg={'study_id':STUDY,'registration_type':'descriptive_bounded_project_closure',
        'registered_at_utc':datetime.now(timezone.utc).isoformat(),'source_git_commit':commit,
        'baseline_tag':BASE_TAG,'baseline_commit':BASE_COMMIT,'config':cfg,'config_sha256':sha(config),
        **inventory,'notice':'All original followup results and negative criteria already known. Literal evidence synthesis, byte/tag audit and one saved-only independent arithmetic reproduction; zero new learning, random draws, graph, optimizer, human or effect experiments.'}
    write_new(out,reg);return reg


def validate_registration(root,config,registration):
    root=Path(root).resolve();cfg=read(config);reg=read(registration);validate_envelope(root,cfg)
    require(Path(config).resolve()==safe(root,CONFIG) and Path(registration).resolve()==safe(root,REGISTRATION)
        and reg.get('study_id')==STUDY and reg.get('config')==cfg and reg.get('config_sha256')==sha(config),
        'Same frozen closure identity')
    inventory=freeze_inventory(root,cfg)
    require(all(reg.get(k)==v for k,v in inventory.items()),'Frozen closure source/input bytes changed')
    git(root,'merge-base','--is-ancestor',reg['source_git_commit'],'HEAD')
    for paths in inventory.values():committed(root,reg['source_git_commit'],paths)
    require(utc(reg['registered_at_utc'])<=datetime.now(timezone.utc),'Prior frozen registration')
    return cfg,reg


def publication(root,registration,reg,path):
    p=read(path);commit=p.get('published_commit')
    require(p.get('study_id')==STUDY and p.get('registration_sha256')==sha(registration)
        and p.get('source_commit')==reg['source_git_commit'] and p.get('remote_bytes_match') is True
        and p.get('visibility')=='private' and type(commit) is str and re.fullmatch('[0-9a-f]{40}',commit),
        'Literal private registration publication proof')
    require(p.get('url')==REPOSITORY+'/blob/'+commit+'/'+REGISTRATION,'Pinned private registration URL')
    git(root,'merge-base','--is-ancestor',reg['source_git_commit'],commit)
    git(root,'merge-base','--is-ancestor',commit,'HEAD')
    require(git(root,'show',commit+':'+REGISTRATION)==Path(registration).read_bytes(),'Published exact registration bytes')
    require(utc(reg['registered_at_utc'])<utc(p['checked_at_utc'])<=datetime.now(timezone.utc),'Publication after registration')
    return p


def run(root,config,registration,proof,out):
    import numpy as np
    root=Path(root).resolve();out=Path(out).resolve()
    require(out==safe(root,RESULT) and not out.exists(),'Exclusive fixed closure result directory')
    cfg,reg=validate_registration(root,config,registration);pub=publication(root,registration,reg,proof)
    reproduction=read(safe(root,REPRO))
    require(reproduction.get('status')=='passed' and utc(pub['checked_at_utc'])<utc(reproduction['verified_at_utc']),
        'One new saved-only independent receipt after private publication')
    start=datetime.now(timezone.utc);out.mkdir();(out/'registration.json').write_bytes(Path(registration).read_bytes())
    meta={'study_id':STUDY,'status':'running','started_at_utc':start.isoformat(),
        'source_git_commit':reg['source_git_commit'],'registration_sha256':sha(registration),'config_sha256':sha(config),
        'publication_proof':Path(proof).resolve().relative_to(root).as_posix(),'publication_proof_sha256':sha(proof),
        'analysis_only':True,'new_training_rng_graph_optimizer_human_effect_calls':0,
        'runtime':{'python':sys.version.split()[0],'numpy':np.__version__}}
    try:
        module=importlib.import_module(cfg['module']);a,rows=module.produce(cfg,root)
        require(set(a)=={'study_integrity_flags','claim_ledger'} and all(np.isfinite(x).all()
            and x.dtype.kind in 'biu' for x in a.values()),'Two finite integer/bool synthesis ledgers')
        np.savez_compressed(out/'arrays.npz',**a)
        with (out/'measurements.csv').open('x',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        summary=module.summarize(cfg,a,rows,root);write_new(out/'summary.json',summary)
        meta.update(status='completed',counts=summary['counts'],numeric_arrays=2,measurement_records=len(rows))
    except Exception as e:
        meta.update(status='failed',error_type=type(e).__name__,error_message=str(e));raise
    finally:
        meta['completed_at_utc']=datetime.now(timezone.utc).isoformat();write_new(out/'metadata.json',meta)
        write_new(out/'artifact_hashes.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    return meta


def verify_envelope(root,config,registration,result,proof):
    root=Path(root).resolve();directory=Path(result).resolve();cfg,reg=validate_registration(root,config,registration)
    pub=publication(root,registration,reg,proof);m=read(directory/'metadata.json')
    require(directory==safe(root,RESULT) and m.get('status')=='completed' and m.get('study_id')==STUDY
        and m.get('analysis_only') is True and m.get('new_training_rng_graph_optimizer_human_effect_calls')==0
        and m['source_git_commit']==reg['source_git_commit'] and m['registration_sha256']==sha(registration)
        and m['config_sha256']==sha(config) and m['publication_proof_sha256']==sha(proof)
        and (directory/'registration.json').read_bytes()==Path(registration).read_bytes(),'Completed immutable closure metadata')
    times=[reg['registered_at_utc'],pub['checked_at_utc'],read(safe(root,REPRO))['verified_at_utc'],
        m['started_at_utc'],m['completed_at_utc']]
    require(list(map(utc,times))==sorted(map(utc,times)),'Freeze/publication/reproduction/closure chronology')
    artifacts=read(directory/'artifact_hashes.json')
    require(set(artifacts)==set(PRIMARY) and all(sha(directory/n)==h for n,h in artifacts.items()),
        'Exact five unmodified closure primary artifacts')
    return cfg,reg,m
