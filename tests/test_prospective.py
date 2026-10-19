"""Integrity failures must stop a registered study before computation."""
from pathlib import Path
import pytest
from drosolearn import prospective as p

@pytest.mark.parametrize('payload', ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_ambiguous_json_rejected(tmp_path, payload):
    file = tmp_path/'bad.json'; file.write_text(payload)
    with pytest.raises(ValueError): p.read(file)

def test_output_exclusivity(tmp_path):
    file = tmp_path/'once.json'; p.write_new(file, {'x':1}); before=file.read_bytes()
    with pytest.raises(FileExistsError): p.write_new(file, {'x':2})
    assert file.read_bytes()==before

@pytest.mark.parametrize('name', ['../outside', '/outside', 'C:/outside', 'sub\\file'])
def test_containment(tmp_path, name):
    with pytest.raises(ValueError): p.safe(tmp_path, name)

def test_inventory_detects_changes_and_rejects_duplicate(tmp_path):
    file = tmp_path/'a'; file.write_bytes(b'original'); old=p.inventory(tmp_path,['a'])
    file.write_bytes(b'changed'); assert p.inventory(tmp_path,['a'])!=old
    with pytest.raises(ValueError): p.inventory(tmp_path,['a','a'])

def test_csv_schema_and_exclusive_output(tmp_path):
    with pytest.raises(ValueError): p.csv_write(tmp_path/'bad.csv',[{'a':1},{'b':2}])
    file=tmp_path/'good.csv';p.csv_write(file,[{'a':1}]);before=file.read_bytes()
    with pytest.raises(FileExistsError):p.csv_write(file,[{'a':2}])
    assert file.read_bytes()==before

def test_exact_source_closure_requires_every_new_module_and_verifier():
    root=Path(__file__).resolve().parents[1]
    names=p.expected_source(root,'shell_credit_v1')
    assert {'src/drosolearn/shell_credit.py','scripts/verify_shell_credit.py','src/drosolearn/prospective.py'}<=names
    assert len(names)==41

def test_stage_dependencies_preserve_sequential_completion():
    first=p.previous_parent_files('shell_credit_v1'); last=p.previous_parent_files('phase5_synthesis_v1')
    assert first==p.BASE_PARENT
    assert len(last)==len(first)+4*7
    assert 'results/interference_order_v1/verification.json' in last
