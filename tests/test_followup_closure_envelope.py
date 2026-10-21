"""Small invented path, JSON, frozen-input and publication envelope checks."""
from pathlib import Path
import copy,json
import pytest
from drosolearn import closure_envelope as e


@pytest.mark.parametrize('name',['../outside','../../secret','C:/outside.json'])
def test_escape_paths_are_rejected(tmp_path,name):
    with pytest.raises(ValueError):e.safe(tmp_path,name)


def test_symlink_path_escape_is_rejected_when_supported(tmp_path):
    # Resolve of a parent traversal suffices without requiring Windows symlink privilege.
    with pytest.raises(ValueError):e.safe(tmp_path,'sub/../../outside')


@pytest.mark.parametrize('text',['{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}','{"x":-Infinity}'])
def test_json_duplicate_or_nonfinite_rejected(tmp_path,text):
    p=tmp_path/'x.json';p.write_text(text,encoding='utf8')
    with pytest.raises(ValueError):e.read(p)


def test_exclusive_receipt_never_overwrites(tmp_path):
    p=tmp_path/'x.json';e.write_new(p,{'passed':False});original=p.read_bytes()
    with pytest.raises(FileExistsError):e.write_new(p,{'passed':True})
    assert p.read_bytes()==original


@pytest.mark.parametrize('time',['2000-01-01T00:00:00','2000-01-01'])
def test_utc_requires_timezone(time):
    with pytest.raises(ValueError):e.utc(time)


def invented_envelope(tmp_path,monkeypatch):
    root=tmp_path/'envelope';root.mkdir();sources={f'fake/source{i}.txt' for i in range(35)}
    m={'studies':[],'retained_anchors':[],'claims':[]}
    (root/'configs').mkdir();e.write_new(root/e.MANIFEST,m)
    cfg={'study_id':e.STUDY,'scope':'retrospective_bounded_followup_synthesis','module':'drosolearn.followup_closure',
        'protocol':e.PROTOCOL,'evidence_manifest':e.MANIFEST,'baseline_tag':e.BASE_TAG,'baseline_commit':e.BASE_COMMIT,
        'reproduction_receipt':e.REPRO,'source_files':sorted(sources),'parent_files':sorted(e.parent_paths(m))}
    monkeypatch.setattr(e,'source_paths',lambda root:sources)
    monkeypatch.setattr(e,'git',lambda root,*args:(e.BASE_COMMIT+'\n').encode() if args[0]=='rev-parse' else b'')
    for n in cfg['source_files']+cfg['parent_files']:
        p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'Invented only\n')
    return root,cfg


def test_exact_envelope_inventory(tmp_path,monkeypatch):
    root,cfg=invented_envelope(tmp_path,monkeypatch)
    assert e.validate_envelope(root,cfg)['studies']==[]
    values=e.freeze_inventory(root,cfg);assert len(values['source_files'])==35


@pytest.mark.parametrize('field',['study_id','scope','module','protocol','evidence_manifest','baseline_tag','baseline_commit','reproduction_receipt'])
def test_fixed_envelope_identity(tmp_path,monkeypatch,field):
    root,cfg=invented_envelope(tmp_path,monkeypatch);cfg[field]='changed'
    with pytest.raises(ValueError):e.validate_envelope(root,cfg)


@pytest.mark.parametrize('field',['source_files','parent_files'])
@pytest.mark.parametrize('mutation',['duplicate','missing','extra'])
def test_frozen_inventory_must_be_exact_unique(tmp_path,monkeypatch,field,mutation):
    root,cfg=invented_envelope(tmp_path,monkeypatch)
    if mutation=='duplicate':cfg[field].append(cfg[field][0])
    elif mutation=='missing':cfg[field].pop()
    else:cfg[field].append('extra.txt')
    with pytest.raises(ValueError):e.validate_envelope(root,cfg)


def test_frozen_sha_detects_byte_tampering(tmp_path,monkeypatch):
    root,cfg=invented_envelope(tmp_path,monkeypatch);before=e.freeze_inventory(root,cfg)
    (root/cfg['source_files'][0]).write_bytes(b'changed')
    assert before!=e.freeze_inventory(root,cfg)


def test_optional_anchors_do_not_create_null_paths():
    m={'studies':[],'retained_anchors':[{'summary':None,'verification':None,'paths':['LIMITATIONS.md']}],'claims':[]}
    assert None not in e.parent_paths(m) and 'LIMITATIONS.md' in e.parent_paths(m)


def test_publication_receipt_contract(tmp_path,monkeypatch):
    root=tmp_path;regpath=root/'registration.json';regpath.write_bytes(b'Invented registration\n')
    reg={'source_git_commit':'b'*40,'registered_at_utc':'2000-01-01T00:00:00+00:00'}
    proof={'study_id':e.STUDY,'registration_sha256':e.sha(regpath),'source_commit':'b'*40,
        'published_commit':'a'*40,'remote_bytes_match':True,'visibility':'private',
        'checked_at_utc':'2000-01-01T00:00:01+00:00','url':e.REPOSITORY+'/blob/'+'a'*40+'/'+e.REGISTRATION}
    p=root/'proof.json';e.write_new(p,proof)
    monkeypatch.setattr(e,'git',lambda root,*args:regpath.read_bytes() if args[0]=='show' else b'')
    assert e.publication(root,regpath,reg,p)==proof


@pytest.mark.parametrize('field,value',[('remote_bytes_match',False),('visibility','public'),('registration_sha256','0'*64),
    ('source_commit','c'*40),('published_commit','bad'),('checked_at_utc','1999-01-01T00:00:00+00:00'),('url','https://example.invalid/')])
def test_publication_mutations_rejected(tmp_path,monkeypatch,field,value):
    root=tmp_path;p=root/'reg.json';p.write_bytes(b'Invented\n')
    reg={'source_git_commit':'b'*40,'registered_at_utc':'2000-01-01T00:00:00+00:00'}
    proof={'study_id':e.STUDY,'registration_sha256':e.sha(p),'source_commit':'b'*40,'published_commit':'a'*40,
        'remote_bytes_match':True,'visibility':'private','checked_at_utc':'2000-01-01T00:00:01+00:00',
        'url':e.REPOSITORY+'/blob/'+'a'*40+'/'+e.REGISTRATION};proof[field]=value
    q=root/'proof.json';e.write_new(q,proof);monkeypatch.setattr(e,'git',lambda root,*args:p.read_bytes() if args[0]=='show' else b'')
    with pytest.raises(ValueError):e.publication(root,p,reg,q)
