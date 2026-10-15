"""Synthetic source-identity fixtures only; no original human MAT is read."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

SPEC = importlib.util.spec_from_file_location("osf_origin_producer", Path(__file__).resolve().parents[1] /
                                               "scripts/audit_human_osf_origin.py")
origin = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(origin)


def toy_raw(second=False):
    return {"A": np.array([[2, 1, 3, np.nan] if second else [1, 2, 3, np.nan]]),
            "C": np.array([[1, 0, np.nan, np.nan] if second else [0, 1, np.nan, np.nan]]),
            "S": np.array([[1], [2], [1], [2]]),
            "R": np.array([[1, -1], [-1, 1], [1, -1], [-1, 1]])}


def make_fixture(tmp_path):
    """Return paths for 2 synthetic main rows, 1 unavailable pilot row, 4 trials."""
    cache, derived = tmp_path / "cache", tmp_path / "derived"
    cache.mkdir(); derived.mkdir()
    sources, derived_sources, coverage = [], [], {"groups": {}}

    def save_source(role, relative, raw, **extra):
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        record = {"role": role, "path": relative, "bytes": len(raw),
                  "sha256": hashlib.sha256(raw).hexdigest(), "url": "https://example.test/source", **extra}
        sources.append(record)
        return record

    nodes = []
    for index in range(2):
        fid = f"synthetic_file_{index}"
        path = cache / "osf" / (fid + ".mat")
        path.parent.mkdir(exist_ok=True)
        savemat(path, toy_raw(bool(index)))
        token = hashlib.sha256(f"example_{index}.mat".encode()).hexdigest()
        source = {"role": "osf_raw", "path": "osf/" + fid + ".mat", "bytes": path.stat().st_size,
                  "sha256": origin.digest(path), "url": "https://example.test/raw?revision=1", "version": 1,
                  "derived_group": "main", "derived_row_index": index,
                  "osf_file_id": fid, "filename_token_sha256": token}
        sources.append(source)
        nodes.append({"type": "files", "id": fid, "attributes": {"name": f"example_{index}.mat",
                      "kind": "file", "size": source["bytes"], "current_version": 1,
                      "extra": {"hashes": {"sha256": source["sha256"]}}},
                      "relationships": {"target": {"data": {"id": "75jrf"}}}})
    main_code = "raise RuntimeError('must never execute')\nfnames11=['example_0.mat']\nfnames12=[]\nfnames21=['example_1.mat']\nfnames22=[]\nfnames=fnames11+fnames12+fnames21+fnames22\n"
    pilot_code = "raise RuntimeError('must never execute')\nfnames1=['unavailable.mat']\nfnames2=[]\nfnames=fnames1+fnames2\n"
    save_source("formatter_main", "formatter/main.py", main_code.encode())
    save_source("formatter_pilot", "formatter/pilot.py", pilot_code.encode())
    save_source("formatter_license", "formatter/LICENSE.md", b"Synthetic software license")
    for index in (1, 2):
        save_source(f"catalog_pilot_{index}", f"catalog/pilot_page{index}.json",
                    json.dumps({"data": [nodes[index - 1]], "links": {"meta": {"total": 2}}}).encode())
    save_source("catalog_main", "catalog/main.json", json.dumps({"data": []}).encode())
    license_id, license_url = "synthetic-license", "https://example.test/license"
    save_source("data_license_node", "catalog/data_node.json", json.dumps({"data": {"id": "75jrf",
        "attributes": {"public": True}, "relationships": {"license": {"data": {"id": license_id}}}}}).encode())
    save_source("data_license", "catalog/license.json", json.dumps({"data": {"id": license_id,
        "attributes": {"url": license_url}}}).encode())
    for group, names in {"main": ["example_0.mat", "example_1.mat"], "pilot": ["unavailable.mat"]}.items():
        mapping = [{"processed_row_index": i, "filename_token_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "matching_osf_file_ids": [f"synthetic_file_{i}"] if group == "main" else [],
                    "matches": 1 if group == "main" else 0} for i, name in enumerate(names)]
        coverage["groups"][group] = {"mapping": mapping, "duplicate_row_groups": []}
        values = {
            "responses": [[0, 1, 2, np.nan], [1, 0, 2, np.nan]] if group == "main" else [[0, 0, 0, 0]],
            "outcomes": [[1, 1, 2, np.nan], [0, 0, 2, np.nan]] if group == "main" else [[0, 0, 0, 0]],
            "correct_responses": [[0, 1, np.nan, np.nan], [1, 0, np.nan, np.nan]] if group == "main" else [[0, 0, 0, 0]],
        }
        for role, matrix in values.items():
            relative = group + "/" + role + ".csv"
            path = derived / relative
            path.parent.mkdir(exist_ok=True)
            with path.open("w", encoding="utf8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["subject", "0", "1", "2", "3"] + (["condition"] if role == "responses" else []))
                for i, row in enumerate(matrix):
                    fields = ["" if np.isnan(value) else str(float(value)) for value in row]
                    writer.writerow([str(i), *fields] + (["regular" if i == 0 else "irregular"] if role == "responses" else []))
            derived_sources.append({"role": role, "group": group, "path": relative,
                                    "bytes": path.stat().st_size, "sha256": origin.digest(path),
                                    "url": "https://example.test/derived"})
    manifest = {"study_id": "human_osf_origin_v1",
        "scope": "partial_source_identity_verification_not_human_learning_comparison", "trials_per_row": 4,
        "fixture_kind": "synthetic_only_no_human_data",
        "selection_frozen_before_value_comparison": True,
        "value_comparison_performed_before_registration": False,
        "groups": {"main": {"rows": 2, "regular_rows": 1,
            "filename_variables": ["fnames11", "fnames12", "fnames21", "fnames22"]},
                   "pilot": {"rows": 1, "regular_rows": 1, "filename_variables": ["fnames1", "fnames2"]}},
        "coverage": coverage, "source_files": sources, "derived_sources": derived_sources,
        "expected_coverage": {"main_available_rows": 2, "main_unavailable_rows": 0,
            "pilot_available_rows": 0, "pilot_unavailable_rows": 1, "total_rows": 3,
            "selected_raw_files": 2, "unmatched_rows": 1},
        "raw_license": {"license_id": license_id, "license_url": license_url}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf8")
    return {"cache": cache, "derived_cache": derived, "manifest_path": manifest_path,
            "registration_path": tmp_path / "registration.json", "result_path": tmp_path / "result.json"}


def complete_fixture(tmp_path):
    fixture = make_fixture(tmp_path)
    registration = origin.register(fixture["cache"], fixture["derived_cache"], fixture["manifest_path"])
    fixture["registration_path"].write_text(json.dumps(registration), encoding="utf8")
    result = origin.run(fixture["registration_path"], fixture["cache"], fixture["derived_cache"], fixture["manifest_path"])
    fixture["result_path"].write_text(json.dumps(result), encoding="utf8")
    return {**fixture, "registration": registration, "result": result}


def test_exact_synthetic_registration_and_reproduction(tmp_path):
    fixture = complete_fixture(tmp_path)
    assert fixture["result"]["status"] == "pass"
    assert fixture["result"]["total_values_compared"] == 24
    assert fixture["result"]["unmatched_rows_preserved"] == 1
    assert fixture["registration"]["raw_to_CSV_comparison_performed"] is False


def test_transform_hints_missing_and_C_copy():
    result = origin.reconstruct(toy_raw(), 4)
    assert np.array_equal(result["responses"], [0, 1, 2, np.nan], equal_nan=True)
    assert np.array_equal(result["outcomes"], [1, 1, 2, np.nan], equal_nan=True)
    assert np.array_equal(result["correct_responses"], [0, 1, np.nan, np.nan], equal_nan=True)


@pytest.mark.parametrize("key,value", [("A", np.ones((4, 1))), ("R", np.ones((2, 4))),
    ("S", np.ones((1, 4))), ("C", np.ones((4, 1))), ("A", np.array([[1, 2, 4, np.nan]])),
    ("R", np.zeros((4, 2))), ("S", np.zeros((4, 1))), ("C", np.array([[0, .5, np.nan, np.nan]])),
    ("A", np.ones((1, 4), dtype=complex)), ("A", np.ones((1, 4), dtype=bool)),
    ("A", np.ones((1, 4), dtype=object)), ("A", np.array([["1", "2", "3", "4"]])),
    ("A", np.array([[1, 2, 3, np.inf]]))])
def test_transform_rejects_invalid_schema_or_codes(key, value):
    raw = toy_raw(); raw[key] = value
    with pytest.raises(ValueError): origin.reconstruct(raw, 4)


def test_C_is_not_recomputed_from_state():
    result = origin.reconstruct(toy_raw(), 4)
    assert result["correct_responses"][0] == 0  # A=1/S=1 would imply 1 if improperly recomputed.


def test_comparison_nan_position_and_exact_equality():
    assert origin.compare_values([1, np.nan], [1, np.nan])["mismatch_count"] == 0
    assert origin.compare_values([1, np.nan], [1 + 1e-12, 0])["mismatch_count"] == 2


def test_AST_never_executes_code():
    text = "raise RuntimeError('no execution')\nfnames1=['example.mat']\nfnames2=[]\nfnames=fnames1+fnames2"
    assert origin.formatter_names(text, ["fnames1", "fnames2"]) == ["example.mat"]


@pytest.mark.parametrize("text", ["fnames1=list(['a.mat'])\nfnames2=[]\nfnames=fnames1+fnames2",
    "fnames1=['a.mat']\nfnames2=[]\nfnames=fnames2+fnames1",
    "fnames1=['a.mat']\nfnames2=[]\nfnames=unknown(fnames1)"])
def test_AST_rejects_nonliteral_or_changed_order(text):
    with pytest.raises(ValueError): origin.formatter_names(text, ["fnames1", "fnames2"])


def test_cache_tamper_refuses_registration(tmp_path):
    fixture = make_fixture(tmp_path)
    with (fixture["cache"] / "formatter/main.py").open("ab") as handle: handle.write(b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        origin.register(fixture["cache"], fixture["derived_cache"], fixture["manifest_path"])


def test_unmatched_rows_are_not_removed(tmp_path):
    fixture = complete_fixture(tmp_path)
    assert fixture["result"]["coverage"]["groups"]["pilot"]["mapping"][0]["matches"] == 0
    assert len(fixture["result"]["rows"]) == 2


def test_manifest_change_after_registration_rejected(tmp_path):
    fixture = complete_fixture(tmp_path)
    with fixture["manifest_path"].open("a", encoding="utf8") as handle: handle.write(" ")
    with pytest.raises(ValueError, match="Frozen manifest"):
        origin.run(fixture["registration_path"], fixture["cache"], fixture["derived_cache"], fixture["manifest_path"])


def test_registered_input_change_rejected(tmp_path):
    fixture = complete_fixture(tmp_path)
    with (fixture["derived_cache"] / "main/responses.csv").open("ab") as handle: handle.write(b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        origin.run(fixture["registration_path"], fixture["cache"], fixture["derived_cache"], fixture["manifest_path"])


def test_real_mismatch_is_preserved_as_failed_completed_report(tmp_path):
    fixture = make_fixture(tmp_path)
    path = fixture["derived_cache"] / "main/outcomes.csv"
    content = path.read_text(encoding="utf8").replace("0,1.0,1.0", "0,0.0,1.0")
    path.write_text(content, encoding="utf8")
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf8"))
    entry = next(e for e in manifest["derived_sources"] if e["group"] == "main" and e["role"] == "outcomes")
    entry.update(bytes=path.stat().st_size, sha256=origin.digest(path))
    fixture["manifest_path"].write_text(json.dumps(manifest), encoding="utf8")
    registration = origin.register(fixture["cache"], fixture["derived_cache"], fixture["manifest_path"])
    fixture["registration_path"].write_text(json.dumps(registration), encoding="utf8")
    result = origin.run(fixture["registration_path"], fixture["cache"], fixture["derived_cache"], fixture["manifest_path"])
    assert result["status"] == "failed" and result["total_mismatches"] == 1
    assert len(result["rows"]) == 2


def test_existing_output_and_missing_registration_rejected(tmp_path):
    out = tmp_path / "exists.json"; out.write_text("unchanged", encoding="utf8")
    arguments = ["run", "--cache", str(tmp_path), "--derived-cache", str(tmp_path), "--out", str(out)]
    with pytest.raises(ValueError, match="Output already"): origin.main(arguments)
    assert out.read_text() == "unchanged"
    with pytest.raises(ValueError, match="registration"):
        origin.main(arguments[:-1] + [str(tmp_path / "new.json")])


@pytest.mark.parametrize("path", ["../outside", "a/../../outside", "a\\bad"])
def test_relative_cache_path_escape_refused(tmp_path, path):
    with pytest.raises(ValueError): origin.cached_path(tmp_path, path)


def test_fetch_restores_missing_hash_pinned_bytes_only(tmp_path, monkeypatch):
    source = b"synthetic source"
    fixture = {"source_files": [{"path": "source.bin", "url": "https://example.test/source",
        "bytes": len(source), "sha256": hashlib.sha256(source).hexdigest()}], "derived_sources": []}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, amount): return source

    monkeypatch.setattr(origin.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    origin.restore_caches(fixture, tmp_path / "cache", tmp_path / "derived")
    assert (tmp_path / "cache/source.bin").read_bytes() == source
    monkeypatch.setattr(origin.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("refetched existing file"))
    origin.restore_caches(fixture, tmp_path / "cache", tmp_path / "derived")
    (tmp_path / "cache/source.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        origin.restore_caches(fixture, tmp_path / "cache", tmp_path / "derived")


def test_downloaded_mismatch_is_not_saved(tmp_path, monkeypatch):
    source = b"expected"
    fixture = {"source_files": [{"path": "source.bin", "url": "https://example.test/source",
        "bytes": len(source), "sha256": hashlib.sha256(source).hexdigest()}], "derived_sources": []}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, amount): return b"different"

    monkeypatch.setattr(origin.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="Downloaded"):
        origin.restore_caches(fixture, tmp_path / "cache", tmp_path / "derived")
    assert not (tmp_path / "cache/source.bin").exists()


def test_metadata_restore_uses_bundled_snapshot_without_live_request(tmp_path, monkeypatch):
    repository = tmp_path / "repository"; repository.mkdir()
    snapshot = repository / "saved_metadata.json"; raw = b'{"data":[]}'
    snapshot.write_bytes(raw)
    monkeypatch.setattr(origin, "ROOT", repository)
    monkeypatch.setattr(origin.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("live metadata requested"))
    manifest = {"source_files": [{"path": "catalog/snapshot.json", "url": "https://example.test/live",
        "bundled_path": "saved_metadata.json", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}],
        "derived_sources": []}
    origin.restore_caches(manifest, tmp_path / "cache", tmp_path / "derived")
    assert (tmp_path / "cache/catalog/snapshot.json").read_bytes() == raw
