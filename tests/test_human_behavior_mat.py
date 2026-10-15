"""Only invented numeric cell arrays; no human source files are read."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pytest
from scipy.io import savemat

SPEC = importlib.util.spec_from_file_location("human_behavior_mat", Path(__file__).resolve().parents[1] / "scripts/inspect_human_behavior_mat.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_source(tmp_path):
    cells = np.empty(2, dtype=object)
    cells[0] = np.array([1., 2., 1., 2.])
    cells[1] = np.array([2., 1., 2., 1.])
    records = []
    for name, values in (("small.mat", {"REW_action": cells}),
                         ("full.mat", {"REW_action": cells, "REW_rt": cells})):
        path = tmp_path / name
        savemat(path, values)
        raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
        records.append({"path": name, "size_bytes": len(raw), "sha256": sha,
                        "git_pointer": f"SHA256E-s{len(raw)}--{sha}.mat"})
    (tmp_path / "source-manifest.json").write_text(json.dumps({"files": records}), encoding="utf-8")
    return records


def test_only_schema_and_common_variables_are_reported(tmp_path):
    synthetic_source(tmp_path)
    report = MODULE.inspect_mats(tmp_path)
    assert report["common_variable_equality"] == {"REW_action": True}
    assert report["files"][0]["variables"]["REW_action"]["trial_vector_lengths"] == [4]
    assert report["participant_id_mapping"] == "not_assumed_or_performed"
    assert not report["performance_estimates_computed"]
    assert not report["human_model_similarity_tested"]


def test_changed_source_bytes_are_not_silently_read(tmp_path):
    records = synthetic_source(tmp_path)
    with (tmp_path / records[0]["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        MODULE.inspect_mats(tmp_path)


def test_annex_binding_must_agree_with_byte_manifest(tmp_path):
    records = synthetic_source(tmp_path)
    records[0]["git_pointer"] = "SHA256E-s10--" + "0" * 64
    (tmp_path / "source-manifest.json").write_text(json.dumps({"files": records}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        MODULE.inspect_mats(tmp_path)


def test_exact_pair_of_files_required(tmp_path):
    records = synthetic_source(tmp_path)
    (tmp_path / "source-manifest.json").write_text(json.dumps({"files": records[:1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="two selected"):
        MODULE.inspect_mats(tmp_path)
