"""Synthetic task-structure safeguards; no human source cache is read."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("human_task_structure", Path(__file__).parents[1] / "scripts/audit_human_task_structure.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def raw():
    return {"A": np.array([[1., 2., 3., np.nan, 2., 1.]]),
            "C": np.array([[1., 0., np.nan, np.nan, 1., 0.]]),
            "S": np.array([[1], [1], [1], [2], [2], [2]]),
            "R": np.array([[1, -1], [-1, 1], [1, -1], [-1, 1], [1, -1], [-1, 1]])}


def test_structure_not_accuracy_and_original_1based_state_boundary():
    value = raw()
    original = {k: v.copy() for k, v in value.items()}
    result = audit.inspect_raw(value, 6)
    assert result["structural_passed"]
    assert result["state_segment_starts_1based"] == [1, 4]
    assert result["state_segment_lengths"] == [3, 3]
    assert "accuracy" not in result
    for k in value:
        np.testing.assert_array_equal(value[k], original[k])


@pytest.mark.parametrize("key,index,value", [("C", (0, 1), 1), ("C", (0, 0), np.nan),
                                             ("C", (0, 2), 0), ("C", (0, 3), 0)])
def test_c_semantic_disagreement_is_a_failed_gate(key, index, value):
    data = raw()
    data[key][index] = value
    assert not audit.inspect_raw(data, 6)["structural_passed"]


@pytest.mark.parametrize("key,value", [("A", 0), ("A", 4), ("C", 2), ("S", 0), ("R", 0)])
def test_unknown_codes_fail(key, value):
    data = raw()
    data[key][0, 0] = value
    with pytest.raises(ValueError, match="codes"):
        audit.inspect_raw(data, 6)


@pytest.mark.parametrize("key", ["A", "C", "S", "R"])
def test_wrong_shape_fails(key):
    data = raw()
    data[key] = data[key].T
    with pytest.raises(ValueError, match="shape"):
        audit.inspect_raw(data, 6)


@pytest.mark.parametrize("key", ["A", "C", "S", "R"])
def test_nonfinite_infinite_fails(key):
    data = raw()
    data[key] = data[key].astype(float)
    data[key][0, 0] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        audit.inspect_raw(data, 6)


def test_schedule_signature_tracks_S_R_not_actions():
    data = raw()
    baseline = audit.inspect_raw(data, 6)
    data["A"][0, 0], data["C"][0, 0] = 2, 0
    assert audit.inspect_raw(data, 6)["schedule_sha256_little_endian_int64_S_then_R"] == baseline["schedule_sha256_little_endian_int64_S_then_R"]
    data["R"][0, 0] *= -1
    assert audit.inspect_raw(data, 6)["schedule_sha256_little_endian_int64_S_then_R"] != baseline["schedule_sha256_little_endian_int64_S_then_R"]


@pytest.mark.parametrize("relative", ["../escape", "a/../../escape", "a\\b", "C:/outside"])
def test_input_path_escape_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        audit.safe_path(tmp_path, relative)


def test_output_is_never_overwritten(tmp_path):
    destination = tmp_path / "new.json"
    audit.write_new(destination, {"state": "first"})
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        audit.write_new(destination, {"state": "second"})
    assert destination.read_bytes() == before


@pytest.mark.parametrize("key", ["A", "C", "S", "R"])
def test_boolean_arrays_rejected(tmp_path, key):
    data = raw()
    data[key] = data[key].astype(bool)
    with pytest.raises(ValueError, match="nonboolean"):
        audit.inspect_raw(data, 6)
