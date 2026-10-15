"""Synthetic fixtures only: no participant files or behavioral estimates."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

SPEC = importlib.util.spec_from_file_location(
    "candidate_cache_audit", Path(__file__).resolve().parents[1] /
    "scripts/audit_human_candidate_caches.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def tefits_frames():
    rows = []
    for identifier in ("s1", "s2", "s3", "s4"):
        for trial in range(1, 5):
            ratio = .2 if trial % 2 else -.2
            response = trial % 2
            rows.append([ratio, response, trial, 1., float(response == (ratio > 0)), identifier])
    columns = ["ratio", "resp", "trialNum", "sizeRat", "acc", "subID"]
    full = pd.DataFrame(rows, columns=columns)
    single = full[full.subID == "s1"].copy()
    expected = {"participants": 4, "trials_per_participant": 4,
                "columns": columns, "subIDs": ["s1", "s2", "s3", "s4"]}
    return full, single, expected


def pybefit_fixture(path):
    path.mkdir()
    state = np.array([1, 2, 1, 2])
    rewards = np.array([[1, -1], [-1, 1], [1, -1], [-1, 1]])
    savemat(path / "states_and_rewards.mat", {"regular": {"S": state, "R": rewards}})
    rows = {
        "responses": [["a", "0.0", "1.0", "2.0", "", "regular"],
                      ["b", "1.0", "0.0", "1.0", "0.0", "regular"]],
        "outcomes": [["a", "1.0", "1.0", "2.0", ""],
                     ["b", "0.0", "0.0", "0.0", "0.0"]],
        "correct_responses": [["a", "1.0", "1.0", "", ""],
                              ["b", "0.0", "0.0", "0.0", "0.0"]],
    }
    write_tables(path, rows)
    return rows, {"rows": 2, "conditions": {"regular": 2}, "duplicate_rows": []}


def write_tables(path, rows):
    for name, values in rows.items():
        header = ["subject", "0", "1", "2", "3"]
        if name == "responses":
            header += ["condition"]
        with (path / (name + ".csv")).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(values)


def test_tefits_order_coding_and_duplicate():
    full, single, expected = tefits_frames()
    result = audit.audit_tefits_frames(full, single, expected)
    assert result["single_file_exact_s1_duplicate"] is True
    assert result["rows"] == 16
    assert result["combine_single_as_extra_participant"] is False
    assert "mean" not in json.dumps(result)


@pytest.mark.parametrize("column,value", [
    ("resp", 2), ("acc", .5), ("acc", 0), ("ratio", 0),
    ("ratio", np.nan), ("sizeRat", -1), ("sizeRat", np.inf), ("trialNum", 2),
])
def test_tefits_invalid_semantics(column, value):
    full, single, expected = tefits_frames()
    full.loc[0, column] = value
    with pytest.raises(ValueError):
        audit.audit_tefits_frames(full, single, expected)


def test_tefits_single_duplicate_is_checked():
    full, single, expected = tefits_frames()
    single.loc[0, "sizeRat"] = 1.2
    with pytest.raises(ValueError, match="exact s1"):
        audit.audit_tefits_frames(full, single, expected)


def test_tefits_extra_fields_rejected():
    full, single, expected = tefits_frames()
    full["private_field"] = "unused"
    with pytest.raises(ValueError, match="columns"):
        audit.audit_tefits_frames(full, single, expected)


def test_pybefit_rewards_states_hints_and_missing(tmp_path):
    _, expected = pybefit_fixture(tmp_path / "group")
    result = audit.audit_pybefit_group(tmp_path / "group", 4, expected)
    assert not any(result["mismatch_counts"].values())
    assert result["code_sets"]["responses"] == [0, 1, 2]
    assert result["C_is_correctness_indicator_not_correct_card_ID"] is True
    assert "mean" not in json.dumps(result)


@pytest.mark.parametrize("table,row,column,value", [
    ("outcomes", 0, 1, "0.0"),
    ("correct_responses", 1, 1, "1.0"),
    ("outcomes", 0, 3, "3.0"),
    ("correct_responses", 0, 3, "0.0"),
    ("outcomes", 0, 4, "0.0"),
    ("responses", 0, 1, "3.0"),
    ("outcomes", 0, 1, "nan"),
])
def test_pybefit_invalid_code_or_semantics(tmp_path, table, row, column, value):
    rows, expected = pybefit_fixture(tmp_path / "group")
    rows[table][row][column] = value
    write_tables(tmp_path / "group", rows)
    with pytest.raises(ValueError):
        audit.audit_pybefit_group(tmp_path / "group", 4, expected)


def test_pybefit_subject_alignment_rejected(tmp_path):
    rows, expected = pybefit_fixture(tmp_path / "group")
    rows["outcomes"].reverse()
    write_tables(tmp_path / "group", rows)
    with pytest.raises(ValueError, match="alignment"):
        audit.audit_pybefit_group(tmp_path / "group", 4, expected)


def test_pybefit_duplicate_pattern_preserved(tmp_path):
    rows, expected = pybefit_fixture(tmp_path / "group")
    for values in rows.values():
        duplicate = values[0].copy()
        duplicate[0] = "c"
        values.append(duplicate)
    write_tables(tmp_path / "group", rows)
    expected.update(rows=3, conditions={"regular": 3}, duplicate_rows=[["a", "c"]])
    result = audit.audit_pybefit_group(tmp_path / "group", 4, expected)
    assert all(value == [["a", "c"]] for value in result["duplicate_rows"].values())
    expected["duplicate_rows"] = []
    with pytest.raises(ValueError, match="duplicate pattern"):
        audit.audit_pybefit_group(tmp_path / "group", 4, expected)


def test_pybefit_mat_shape_and_state_checked(tmp_path):
    _, expected = pybefit_fixture(tmp_path / "group")
    savemat(tmp_path / "group/states_and_rewards.mat",
            {"regular": {"S": np.array([1, 2, 3, 2]), "R": np.ones((4, 2))}})
    with pytest.raises(ValueError, match="state/reward code"):
        audit.audit_pybefit_group(tmp_path / "group", 4, expected)


def test_pybefit_ragged_csv_rejected(tmp_path):
    rows, expected = pybefit_fixture(tmp_path / "group")
    rows["outcomes"][0].append("extra")
    write_tables(tmp_path / "group", rows)
    with pytest.raises(ValueError, match="Ragged"):
        audit.audit_pybefit_group(tmp_path / "group", 4, expected)


def test_raw_cache_cannot_be_in_repository(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        audit.external_cache(tmp_path / "results/raw", repository=tmp_path)
    assert audit.external_cache(tmp_path, repository=tmp_path / "repository") == tmp_path.resolve()


@pytest.mark.parametrize("relative", ["../escape", "nested/../../escape", "bad\\path"])
def test_manifest_path_escape_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        audit.cache_file(tmp_path, "source", relative)


def test_verified_bytes_rejects_hash_change(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"abc")
    entry = {"path": "file", "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    assert audit.verified_bytes(path, entry) == b"abc"
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="SHA-256"):
        audit.verified_bytes(path, entry)


def test_fetch_hash_mismatch_does_not_create_file(tmp_path, monkeypatch):
    source = {"id": audit.TEFITS, "repository": "https://github.com/example/test", "commit": "a" * 40,
              "files": [{"path": "file", "url": "https://raw.githubusercontent.com/example/test/" + "a" * 40 + "/file",
                         "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}]}
    manifest = {"sources": [source, {"id": audit.PYBEFIT, "files": []}]}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return b"bad"

    monkeypatch.setattr(audit.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="Downloaded SHA-256"):
        audit.prepare_cache(tmp_path, manifest, fetch=True)
    assert not (tmp_path / audit.TEFITS / "file").exists()


def test_existing_cache_is_never_overwritten(tmp_path, monkeypatch):
    raw = b"synthetic"
    path = tmp_path / audit.TEFITS / "file"
    path.parent.mkdir()
    path.write_bytes(raw)
    source = {"id": audit.TEFITS, "repository": "https://github.com/example/test", "commit": "a" * 40,
              "files": [{"path": "file", "url": "https://raw.githubusercontent.com/example/test/" + "a" * 40 + "/file",
                         "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]}
    manifest = {"sources": [source, {"id": audit.PYBEFIT, "files": []}]}
    monkeypatch.setattr(audit.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network called"))
    assert len(audit.prepare_cache(tmp_path, manifest, fetch=True)) == 1
    assert path.read_bytes() == raw


def test_existing_output_refused_before_fetch(tmp_path, monkeypatch):
    out = tmp_path / "out.json"
    out.write_text("unchanged", encoding="utf8")
    monkeypatch.setattr(audit, "audit", lambda *args, **kwargs: pytest.fail("audit called"))
    with pytest.raises(ValueError, match="Output already"):
        audit.main(["--cache", str(tmp_path / "cache"), "--out", str(out), "--fetch"])
    assert out.read_text() == "unchanged"
