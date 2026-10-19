import copy
import hashlib
import json
import subprocess
import numpy as np
import pytest

from drosolearn import phase5_report as p


@pytest.fixture
def evidence(tmp_path):
    (tmp_path/'audit.json').write_text(json.dumps({'passed':True,'errors':[]}),encoding='utf8')
    (tmp_path/'data.json').write_text(json.dumps({'nested':[{'gain':.125}],'a/b':{'~':False}}),encoding='utf8')
    (tmp_path/'report.md').write_text('Invented source report, no scientific data.',encoding='utf8')
    claims=[{'claim_id':'q'+str(i),'question':i,'level':0,'status':'bounded','text_ko':'합성 근거',
             'limits_ko':'생물학적 근거가 아님',
             'observations':[{'file':'data.json','pointer':'/nested/0/gain','value':.125}]} for i in range(1,6)]
    cfg={'study_id':'phase5_synthesis_v1','scope':'retrospective_bounded_evidence_synthesis',
         'studies':[{'study_id':s,'tag':'synthetic-fixture','verification':'audit.json','report':'report.md',
                     'primary':'data.json','role':'synthetic_fixture'} for s in p.STUDY_IDS],
         'claims':claims,'questions':[{'question':i,'text_ko':'합성 질문','claim_ids':['q'+str(i)]} for i in range(1,6)],
         'input_files':['audit.json','data.json','report.md'],'parent_files':[],
         'synthesis_interpretation':copy.deepcopy(p.INTERPRETATION)}
    return tmp_path,cfg


def test_full_ledger_and_counts(evidence):
    root,cfg=evidence
    arrays,rows,summary=p.evaluate(cfg,root)
    assert arrays['study_checks'].shape==(28,2)
    np.testing.assert_array_equal(arrays['claim_codes'],[[i,0,2,1] for i in range(1,6)])
    assert len(rows)==5 and summary['counts']['observations']==5
    assert all(summary['counts'][k]==0 for k in ('new_learning_updates','new_rng_draws','new_human_experiments','new_effect_tests'))
    assert all(s['verified'] for s in summary['studies'])
    assert summary['interpretation']['all_possible_phase2_phase3_work_complete'] is False


def test_framework_produce_summarize_and_array_corruption(evidence,monkeypatch):
    root,cfg=evidence
    monkeypatch.setattr(p,'__file__',str(root/'src/drosolearn/phase5_report.py'))
    arrays,rows=p.produce(cfg,root)
    summary=p.summarize(cfg,arrays,rows)
    assert summary['counts']['studies']==28
    arrays['claim_codes'][0,0]=4
    with pytest.raises(ValueError):p.summarize(cfg,arrays,rows)


@pytest.mark.parametrize('status',['not_supported','not_assessable','design_limited','untested','supported'])
def test_negative_gated_and_positive_histories_preserved(evidence,status):
    root,cfg=evidence;cfg['claims'][0]['status']=status
    _,_,summary=p.evaluate(cfg,root)
    assert summary['claims'][0]['status']==status


@pytest.mark.parametrize('change',['value','missing','outside','unfrozen','verification_false','verification_conflict'])
def test_bad_evidence_rejected(evidence,change):
    root,cfg=evidence
    if change=='value':cfg['claims'][0]['observations'][0]['value']=.126
    if change=='missing':cfg['claims'][0]['observations'][0]['pointer']='/absent'
    if change=='outside':cfg['studies'][0]['report']='../report.md';cfg['input_files'].append('../report.md')
    if change=='unfrozen':cfg['input_files'].remove('data.json')
    if change=='verification_false':(root/'audit.json').write_text('{"passed":false}',encoding='utf8')
    if change=='verification_conflict':(root/'audit.json').write_text('{"passed":false,"status":"passed"}',encoding='utf8')
    with pytest.raises((ValueError,KeyError,IndexError)):p.evaluate(cfg,root)


@pytest.mark.parametrize('change',['level5','positive4','empty_limits','missing_study','bad_status','question_link','duplicate_claim','question_order','inflated_interpretation'])
def test_invalid_scope_rejected(evidence,change):
    _,cfg=evidence
    if change=='level5':cfg['claims'][0]['level']=5
    if change=='positive4':cfg['claims'][0].update(level=4,status='supported')
    if change=='empty_limits':cfg['claims'][0]['limits_ko']=''
    if change=='missing_study':cfg['studies'].pop()
    if change=='bad_status':cfg['claims'][0]['status']='success_somehow'
    if change=='question_link':cfg['questions'][0]['claim_ids']=[]
    if change=='duplicate_claim':cfg['claims'][1]['claim_id']='q1'
    if change=='question_order':cfg['questions'].reverse()
    if change=='inflated_interpretation':cfg['synthesis_interpretation']['level4_similarity_supported']=True
    with pytest.raises(ValueError):p.validate_config(cfg)


def test_pointer_escaped_keys_and_canonical_indices():
    doc={'a/b':{'~':False},'array':[1]}
    assert p.pointer(doc,'/a~1b/~0') is False
    assert p.pointer(doc,'') is doc
    with pytest.raises(ValueError):p.pointer(doc,'/array/00')
    with pytest.raises(ValueError):p.pointer(doc,'/a~2b')


@pytest.mark.parametrize('text',['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'])
def test_invalid_json_rejected(tmp_path,text):
    file=tmp_path/'bad.json';file.write_text(text,encoding='utf8')
    with pytest.raises(ValueError):p.read(file)


def test_primary_bound_to_verification_digest(evidence):
    root,cfg=evidence
    (root/'summary.json').write_text('{}',encoding='utf8');cfg['input_files'].append('summary.json')
    cfg['studies'][0]['primary']='summary.json'
    (root/'audit.json').write_text('{"passed":true,"summary_sha256":"invalid"}',encoding='utf8')
    with pytest.raises(ValueError):p.evaluate(cfg,root)


def test_verification_study_identity_mismatch(evidence):
    root,cfg=evidence
    (root/'audit.json').write_text('{"passed":true,"study_id":"unrelated-study"}',encoding='utf8')
    with pytest.raises(ValueError):p.evaluate(cfg,root)


def test_appendix_written_once_and_all_claims_visible(evidence):
    root,cfg=evidence;summary=p.evaluate(cfg,root)[2];path=root/'appendix.md'
    p.render_evidence_markdown(summary,path)
    content=path.read_text(encoding='utf8')
    assert all(c['claim_id'] in content for c in cfg['claims'])
    assert all(s in content for s in p.STUDY_IDS)
    with pytest.raises(FileExistsError):p.render_evidence_markdown(summary,path)


def split_fixture(root,change=None):
    root.mkdir(exist_ok=True)
    subprocess.run(['git','init','-q'],cwd=root,check=True,capture_output=True)
    payload=b'invented immutable parent bytes, no scientific arrays'
    expected=hashlib.sha256(payload).hexdigest()
    chunks=[payload[:20],payload[20:]]
    manifest={'format':'drosolearn.lossless_file_parts.v1','original_path':p.SPLIT_RAW,'original_bytes':len(payload),
              'original_sha256':expected,'max_part_bytes':67108864,
              'parts':[{'path':name,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}
                       for name,blob in zip(p.SPLIT_PARTS,chunks)]}
    if change=='extra':manifest['parts'].append(manifest['parts'][1].copy())
    if change=='duplicate':manifest['parts'][1]['path']=p.SPLIT_PARTS[0]
    if change=='swapped':manifest['parts'].reverse()
    if change=='original_path':manifest['original_path']='other/arrays.npz'
    if change=='original_bytes':manifest['original_bytes']+=1
    if change=='part_bytes':manifest['parts'][0]['bytes']+=1
    if change=='part_hash':manifest['parts'][1]['sha256']='0'*64
    if change=='original_hash':manifest['original_sha256']='0'*64
    if change=='extra_key':manifest['unexpected']='ignored?'
    for name,blob in zip(p.SPLIT_PARTS,chunks):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
    (root/p.SPLIT_MANIFEST).write_text(json.dumps(manifest),encoding='utf8')
    (root/p.SPLIT_ARTIFACTS).write_text(json.dumps({'arrays.npz':expected}),encoding='utf8')
    subprocess.run(['git','add','.'],cwd=root,check=True,capture_output=True)
    subprocess.run(['git','-c','user.name=SyntheticFixture','-c','user.email=synthetic@example.invalid',
                    'commit','--no-gpg-sign','-q','-m','fixture'],cwd=root,check=True,capture_output=True)
    (root/p.SPLIT_RAW).write_bytes(payload)
    return payload,expected


def test_lossless_git_parts_and_exclusive_materialization(tmp_path):
    payload,expected=split_fixture(tmp_path)
    assert p.split_git_payload(tmp_path,'HEAD',expected)==payload
    assert p.materialize_parent(tmp_path)=='already_matches'
    raw=tmp_path/p.SPLIT_RAW;raw.unlink()
    assert p.materialize_parent(tmp_path)=='materialized_exact_original_bytes'
    assert raw.read_bytes()==payload
    raw.write_bytes(b'existing different bytes')
    with pytest.raises(ValueError):p.materialize_parent(tmp_path)
    assert raw.read_bytes()==b'existing different bytes'


@pytest.mark.parametrize('change',['extra','duplicate','swapped','original_path','original_bytes','part_bytes','part_hash','original_hash','extra_key'])
def test_invalid_committed_parts_rejected(tmp_path,change):
    _,expected=split_fixture(tmp_path,change)
    with pytest.raises(ValueError):p.split_git_payload(tmp_path,'HEAD',expected)


def test_changed_working_part_or_manifest_rejected(tmp_path):
    _,expected=split_fixture(tmp_path)
    part=tmp_path/p.SPLIT_PARTS[0];part.write_bytes(b'corrupt')
    with pytest.raises(ValueError):p.split_git_payload(tmp_path,'HEAD',expected)


def test_adapter_only_one_parent_and_restores_frozen_function(tmp_path):
    from drosolearn import prospective
    _,expected=split_fixture(tmp_path);original=prospective.committed
    with p.committed_parent_parts(tmp_path):
        prospective.committed(tmp_path,'HEAD',{p.SPLIT_RAW:expected,p.SPLIT_MANIFEST:p.digest(tmp_path/p.SPLIT_MANIFEST)})
    assert prospective.committed is original
    with pytest.raises(subprocess.CalledProcessError):
        with p.committed_parent_parts(tmp_path):
            prospective.committed(tmp_path,'HEAD',{'uncommitted-other-parent':'0'*64})
    assert prospective.committed is original


def test_adapter_rejects_changed_original(tmp_path):
    from drosolearn import prospective
    _,expected=split_fixture(tmp_path);(tmp_path/p.SPLIT_RAW).write_bytes(b'changed')
    with pytest.raises(ValueError):
        with p.committed_parent_parts(tmp_path):prospective.committed(tmp_path,'HEAD',{p.SPLIT_RAW:expected})


def test_split_path_resolves_inside_repository(tmp_path):
    assert p.split_local(tmp_path,p.SPLIT_RAW).is_relative_to(tmp_path)
    with pytest.raises(ValueError):p.split_local(tmp_path,'../outside.npz')
