"""Tiny invented JSON/Git snapshots, never actual study outcomes in development."""
import copy,json,subprocess
from pathlib import Path
import numpy as np
import pytest
from drosolearn import followup_closure as m
from drosolearn.closure_envelope import sha,write_new


def fixture(tmp_path):
    root=tmp_path/'invented';root.mkdir()
    def git(*args):return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.DEVNULL).decode().strip()
    def put(name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(value,bytes):p.write_bytes(value)
        else:write_new(p,value)
    git('init','-q');git('config','user.name','Synthetic fixture');git('config','user.email','fixture@example.invalid')
    git('config','core.autocrlf','false');put('.gitattributes',b'* -text\n');put('fixture_source.py',b'fixture_only = True\n')
    git('add','--','.gitattributes','fixture_source.py');git('commit','-qm','Invented source')
    source=git('rev-parse','HEAD');specs=[]
    actual={0,1,2,4,7,9,12,15}
    for i,study in enumerate(m.ORDER):
        folder='results/'+study;registration='results/registration_'+study+'.json'
        reg={'study_id':study,'registered_at_utc':'2000-01-01T00:00:00+00:00','source_git_commit':source,
            'config':{'model_seeds':[90001],'training_trials':4},'config_sha256':sha(root/'fixture_source.py'),'source_files':{'fixture_source.py':sha(root/'fixture_source.py')},
            'parent_files':{},'graph_files':{},'input_files':{}}
        put(registration,reg);put(folder+'/registration.json',reg)
        specs.append({'id':study,'kind':'actual_learning' if i in actual else 'saved_record_analysis',
            'completion_tag':'fixture-'+str(i),'completion_commit':None,'source_commit':source,'registration':registration,
            'summary':folder+'/summary.json','verification':folder+'/verification.json','archive':folder+'/archive.json',
            'protocol':'docs/'+study+'-protocol.md','reproduction':'docs/'+study+'-reproduction.md',
            'findings':'docs/'+study+'-findings.md','result_dir':folder,'publication_proof':folder+'/publication.json',
            'artifact_hashes':folder+'/artifact_hashes.json','metadata':folder+'/metadata.json',
            'cohort':{'model_seeds':[90001],'training_trials':4,'parent_learning_result':None},
            'hypothesis_observations':[{'pointer':'/H/supported','expected':False}],
            'raw_files':[folder+'/'+n for n in ('registration.json','metadata.json','arrays.npz','summary.json',
                'measurements.csv.gz' if study=='objective_learning_v1' else 'measurements.csv')],
            'preservation_files':[],'original_new_optimization':study in ('cone_frontier_v1','balanced_learning_v1',
                'gated_learning_v1','self_blend_learning_v1','probability_alignment_v1','objective_learning_v1')})
    git('add','--','results');git('commit','-qm','Invented preregistrations');published=git('rev-parse','HEAD')
    for s in specs:
        reg=json.loads((root/s['registration']).read_text(encoding='utf8'));folder=s['result_dir']
        proof={'study_id':s['id'],'remote_bytes_match':True,'visibility':'private','source_commit':source,
            'registration_sha256':sha(root/s['registration']),'published_commit':published,
            'checked_at_utc':'2000-01-01T00:00:01+00:00',
            'url':'https://github.com/CauchyRiemannEquations/DrosoLearn/blob/'+published+'/'+s['registration']}
        put(s['publication_proof'],proof)
        summary={'study_id':s['id'],'status':'completed','counts':{'invented_records':4},'H':{'supported':False},'metric':.5}
        put(s['summary'],summary)
        meta={'study_id':s['id'],'status':'completed','stage':'completed','counts':summary['counts'],
            'source_git_commit':source,'registration_sha256':sha(root/s['registration']),'config_sha256':reg['config_sha256'],
            'publication_proof':s['publication_proof'],'publication_proof_sha256':sha(root/s['publication_proof']),
            'started_at_utc':'2000-01-01T00:00:02+00:00','completed_at_utc':'2000-01-01T00:00:03+00:00'}
        put(s['metadata'],meta)
        put(folder+'/arrays.npz',b'invented-byte-artifact-not-an-array\n')
        put(folder+('/measurements.csv.gz' if s['id']=='objective_learning_v1' else '/measurements.csv'),b'invented,4\r\n')
        put(s['artifact_hashes'],{Path(n).name:sha(root/n) for n in s['raw_files']})
        put(s['verification'],{'status':'passed','study_id':s['id'],'verified_at_utc':'2000-01-01T00:00:04+00:00',
            'registration_sha256':sha(root/s['registration']),'summary_sha256':sha(root/s['summary']),
            'scientific_producer_imported':False,'new_helper_imported':False,'counts':{'invented_records':4}})
        put(s['archive'],{'status':'passed','current_file_sha256':{s['summary']:sha(root/s['summary'])}})
        for field in ('protocol','reproduction','findings'):put(s[field],b'Invented-only report\n')
    git('add','--','results','docs');git('commit','-qm','Invented complete snapshots');complete=git('rev-parse','HEAD')
    for s in specs:s['completion_commit']=complete;git('tag',s['completion_tag'])
    base=Path(__file__).resolve().parents[1]
    cfg=json.loads((base/'configs/followup_closure_v1.json').read_text(encoding='utf8'));cfg['scope']='integration_test'
    claims=[{'id':'invented_claim','question':2,'level':2,'verdict':'not_supported','statement':'Invented negative example',
        'limit':'Fixture only','evidence':[{'file':specs[0]['summary'],'pointer':'/H/supported','expected':False}]}]
    manifest={'manifest_version':1,'study_id':cfg['study_id'],'studies':specs,'claims':claims,
        'retained_anchors':[{'id':'invented_anchor','summary':specs[0]['summary'],'verification':specs[0]['verification'],
            'paths':[specs[0]['findings']]}]}
    put(cfg['evidence_manifest'],manifest)
    cfg['source_files']=[cfg['evidence_manifest']]
    cfg['parent_files']=sorted({s[k] for s in specs for k in ('registration','summary','verification','archive',
        'protocol','reproduction','findings','publication_proof','artifact_hashes','metadata')}|
        {p for s in specs for p in s['raw_files']})
    put('results/followup_closure_publication_proof.json',{'study_id':cfg['study_id'],'remote_bytes_match':True,
        'visibility':'private','checked_at_utc':'2000-01-01T00:00:05+00:00'})
    repro=json.loads((root/specs[-1]['verification']).read_text(encoding='utf8'))
    repro['verified_at_utc']='2000-01-02T00:00:00+00:00';put(cfg['reproduction_receipt'],repro)
    return cfg,manifest,root


def test_complete_tiny_git_fixture(tmp_path):
    cfg,manifest,root=fixture(tmp_path);a,rows,s=m.analyze(cfg,root)
    assert a['study_integrity_flags'].shape==(17,8) and a['study_integrity_flags'].all()
    assert a['claim_ledger'].tolist()==[[0,2,2,1]]
    assert s['counts']['new_training_trials']==0 and len(rows)==1 and rows[0]['matched']==1
    assert s['counts']['actual_learning_registrations']==8 and s['counts']['saved_record_registrations']==9


@pytest.mark.parametrize('path,expected',[('',{'a/b':{'~':[False,None,1.]}}),('/a~1b/~0/0',False),('/a~1b/~0/1',None),('/a~1b/~0/2',1.)])
def test_pointer_retains_exact_type(path,expected):assert m.same(m.pointer({'a/b':{'~':[False,None,1.]}},path),expected)


@pytest.mark.parametrize('path',['a','/a~2b','/a~1b/~0/01','/a~1b/~0/-','/a~1b/~0/3','/missing','/a~1b/~0/0/x'])
def test_bad_pointer_rejected(path):
    with pytest.raises(ValueError):m.pointer({'a/b':{'~':[False,None,1.]}},path)


@pytest.mark.parametrize('a,b',[(False,0),(True,1),(None,0),(1,1.),([True],[1]),({'x':1},{'x':True})])
def test_no_type_coercion(a,b):assert not m.same(a,b)


@pytest.mark.parametrize('field',['scope','study_order','module','new_confirmatory_hypothesis','new_effect_aggregation','numeric_tolerance'])
def test_fixed_synthesis_choices(field):
    root=Path(__file__).resolve().parents[1];cfg=json.loads((root/'configs/followup_closure_v1.json').read_text(encoding='utf8'))
    cfg[field]='changed'
    with pytest.raises(ValueError):m.validate_config(cfg)


@pytest.mark.parametrize('field',['summary','verification','metadata','archive','publication_proof'])
def test_immutable_study_receipt_tamper(tmp_path,field):
    cfg,manifest,root=fixture(tmp_path);s=manifest['studies'][0]
    p=root/s[field];p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError):m.audit_study(root,s)


def test_no_new_simulation_import():
    source=Path(m.__file__).read_text(encoding='utf8')
    assert all(x not in source for x in ('np.random','scipy.optimize','FlyWire','SelfBlendHead','run_learning('))


def test_reproduction_receipt_counts_must_match(tmp_path):
    cfg,manifest,root=fixture(tmp_path);p=root/cfg['reproduction_receipt'];r=json.loads(p.read_text(encoding='utf8'))
    r['counts']['invented_records']=5;p.write_text(json.dumps(r),encoding='utf8')
    with pytest.raises(ValueError):m.reproduce_record(cfg,root)
