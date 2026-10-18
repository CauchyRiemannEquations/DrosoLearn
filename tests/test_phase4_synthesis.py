"""Handcrafted invariants for synthesis bookkeeping, not human effects."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('phase4_synthesis_audit',Path(__file__).resolve().parents[1]/'scripts/audit_phase4_synthesis.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

def cell(zero,one):
    status='both_feedback' if zero and one else 'feedback_0_only' if zero else 'feedback_1_only' if one else 'neither'
    return {'by_feedback':{str(f):{'denominator':n,'stay':n//2,'switch':n-n//2} for f,n in enumerate((zero,one))},'support_status':status,'common_support':bool(zero and one)}

def stream(name='human/main/27',kind='human',ns=((2,3),(0,1),(0,0))):
    return {'stream_id':name,'kind':kind,'strata':[cell(*pair) for pair in ns], 'eligible':sum(sum(pair) for pair in ns),
        'by_feedback':{str(f):{'denominator':sum(pair[f] for pair in ns),'stay':sum(pair[f]//2 for pair in ns),'switch':sum(pair[f]-pair[f]//2 for pair in ns)} for f in (0,1)}}

def test_all_cells_including_empty_and_noncontiguous_ids_remain():
    rows=[stream(),stream('human/main/0'),stream('model/9/fixed/probabilistic','model',((1,1),))]
    result=audit.support_counts(rows)
    assert result['human']['streams']==2 and result['human']['strata']==6
    assert result['human']['both_feedback']==2 and result['human']['feedback_1_only']==2 and result['human']['neither']==2
    assert result['human']['feedback_0_denominator']==4 and result['human']['feedback_1_denominator']==8
    assert result['probabilistic']['eligible']==2

@pytest.mark.parametrize('pair,status', [((1,1),'both_feedback'),((1,0),'feedback_0_only'),((0,1),'feedback_1_only'),((0,0),'neither')])
def test_support_means_only_integer_denominator_presence(pair,status):
    assert audit.support_counts([stream(ns=(pair,))])['human'][status]==1

@pytest.mark.parametrize('edit', ['negative','fraction','bool','partition','label','support','record_total','feedback_total','stay_switch_swap','unknown_kind','unknown_environment','duplicate'])
def test_reject_changed_counts_and_units(edit):
    row=stream()
    rows=[row]
    values=row['strata'][0]['by_feedback']['0']
    if edit=='negative': values['stay']=-1
    elif edit=='fraction': values['denominator']=2.0
    elif edit=='bool': values['stay']=True
    elif edit=='partition': values['switch']+=1
    elif edit=='label': row['strata'][0]['support_status']='neither'
    elif edit=='support': row['strata'][0]['common_support']=False
    elif edit=='record_total': row['eligible']+=1
    elif edit=='feedback_total': row['by_feedback']['1']['denominator']+=1
    elif edit=='stay_switch_swap': values.update(stay=2,switch=0)
    elif edit=='unknown_kind': row['kind']='fly'
    elif edit=='unknown_environment': row.update(kind='model',stream_id='model/9/fixed/unknown')
    elif edit=='duplicate': rows.append(copy.deepcopy(row))
    with pytest.raises(ValueError): audit.support_counts(rows)

def config():
    return {'study_id':'phase4_synthesis_v1','features':[{'id':key,'metric':'defined','unit':'person or seed','human':{'honore':'limited','markovic':'limited','openneuro':'semantic_gate_closed'},'model':'abstract','difference':'different','unlock':'evidence'} for key in audit.FEATURES], **{key:False for key in audit.FORBIDDEN_FLAGS}, 'source_files':['code.py'],'evidence_files':['evidence.json']}

def test_full_matrix_is_accepted_without_claiming_similarity():
    audit.validate_config(config())

@pytest.mark.parametrize('flag',audit.FORBIDDEN_FLAGS)
def test_reject_unearned_claim_upgrade(flag):
    cfg=config(); cfg[flag]=True
    with pytest.raises(ValueError): audit.validate_config(cfg)

@pytest.mark.parametrize('edit',['feature_missing','source_missing','closed_gate_upgraded','unlock_empty','source_overlap'])
def test_missing_evidence_cannot_disappear(edit):
    cfg=config()
    if edit=='feature_missing': cfg['features'].pop()
    elif edit=='source_missing': del cfg['features'][0]['human']['openneuro']
    elif edit=='closed_gate_upgraded': cfg['features'][0]['human']['openneuro']='verified'
    elif edit=='unlock_empty': cfg['features'][0]['unlock']=''
    elif edit=='source_overlap': cfg['evidence_files']=cfg['source_files'][:]
    with pytest.raises(ValueError): audit.validate_config(cfg)

def test_write_once_preserves_original_bytes(tmp_path):
    target=tmp_path/'record.json'; audit.write_new(target,{'x':1})
    before=target.read_bytes()
    with pytest.raises(FileExistsError): audit.write_new(target,{'x':2})
    assert target.read_bytes()==before

def test_evidence_path_cannot_escape(tmp_path):
    with pytest.raises(ValueError): audit.path(tmp_path,'../outside.json')

def test_duplicate_inventory_is_rejected(tmp_path):
    p=tmp_path/'a.json';p.write_text('{}')
    with pytest.raises(ValueError): audit.inventory(tmp_path,['a.json','a.json'])
