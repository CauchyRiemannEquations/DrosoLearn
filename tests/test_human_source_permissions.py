"""Synthetic metadata fixtures; no participant or stimulus content."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "human_source_permissions", Path(__file__).resolve().parents[1] /
    "scripts/check_human_source_permissions.py")
permissions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(permissions)


def node(identifier):
    return {"data": {"id": identifier, "type": "nodes", "attributes": {
        "title": "Synthetic public research project", "public": True},
        "relationships": {"license": {"data": {"id": permissions.LICENSE_ID, "type": "licenses"},
            "links": {"related": {"href": permissions.LICENSE_API}}}}}}


def license_payload():
    return {"data": {"id": permissions.LICENSE_ID, "type": "licenses", "attributes": {
        "name": "CC-By Attribution 4.0 International", "url": permissions.LICENSE_URL,
        "text": "Creative Commons Attribution 4.0 International Public License\nSynthetic fixture"}}}


def entry(name, identifier, kind="file", target="x2rau"):
    attrs = {"name": name, "kind": kind}
    if kind == "file":
        attrs.update(size=123, current_version=1, extra={"hashes": {"sha256": "a" * 64}})
    return {"id": identifier, "type": "files", "attributes": attrs,
            "relationships": {"target": {"data": {"id": target, "type": "nodes"}}}}


def page(entries, total=None):
    return {"data": entries, "links": {"meta": {"total": len(entries) if total is None else total}}}


def make_snapshot(path):
    payloads = {
        "markovic_project_node": node("h526v"), "markovic_data_node": node("75jrf"),
        "cheng_project_node": node("x2rau"), "license": license_payload(),
        "markovic_project_children": {"data": [
            {"id": "75jrf", "attributes": {"title": "Behavioural data"}}, {"id": "57wr8"}]},
        "markovic_project_providers": {"data": []},
        "markovic_data_folders": page([entry("Main", "main", "folder", "75jrf"),
                                       entry("Pilot", "pilot", "folder", "75jrf")]),
        "cheng_root_page1": page([entry("Dataset1", "d1", "folder")], total=2),
        "cheng_root_page2": page([entry("Dataset2", "d2", "folder")], total=2),
    }
    for number in range(1, 6):
        payloads[f"cheng_dataset{number}_listing"] = page([
            entry("pics", f"pics{number}", "folder"),
            entry(f"StimList_Dataset{number}.xlsx", f"stim{number}"),
            entry(f"alltrial_Dataset{number}.mat", f"trials{number}")])
    records = []
    for role, payload in payloads.items():
        raw = json.dumps(payload).encode()
        name = role + ".json"
        (path / name).write_bytes(raw)
        records.append({"path": name, "role": role, "http_status": 200,
                        "requested_url": "https://api.osf.io/v2/nodes/synthetic/",
                        "started_at_utc": "2026-10-15T01:00:00+00:00",
                        "finished_at_utc": "2026-10-15T01:00:01+00:00",
                        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    manifest = {"schema_version": 1, "scope": "synthetic_metadata_only", "files": records}
    (path / "source-manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    return manifest


def test_node_explicit_license_required():
    assert permissions.check_node(node("x2rau"), "x2rau")["public"] is True


@pytest.mark.parametrize("change", ["private", "wrong_id", "no_relation", "wrong_license", "wrong_url"])
def test_node_wrong_or_missing_relation(change):
    value = node("x2rau")
    if change == "private": value["data"]["attributes"]["public"] = False
    if change == "wrong_id": value["data"]["id"] = "other"
    if change == "no_relation": value["data"]["relationships"] = {}
    if change == "wrong_license": value["data"]["relationships"]["license"]["data"]["id"] = "MIT"
    if change == "wrong_url": value["data"]["relationships"]["license"]["links"]["related"]["href"] = "https://example.com/"
    with pytest.raises(ValueError):
        permissions.check_node(value, "x2rau")


def test_license_legalcode_url_and_header():
    assert permissions.check_license(license_payload())["url"] == permissions.LICENSE_URL


@pytest.mark.parametrize("field,value", [("name", "MIT"), ("url", "https://example.com/"), ("text", "missing")])
def test_license_wrong_attributes(field, value):
    fixture = license_payload()
    fixture["data"]["attributes"][field] = value
    with pytest.raises(ValueError):
        permissions.check_license(fixture)


def test_catalog_complete_pages_and_declared_hash_semantics():
    result = permissions.catalog_summary([
        page([entry("a.mat", "a")], total=2), page([entry("pics", "b", "folder")], total=2)], "x2rau")
    assert result["entries"] == 2
    assert result["server_hashes_are_independent_content_checks"] is False


@pytest.mark.parametrize("change", ["target", "sha256", "size", "version", "kind", "total", "duplicate"])
def test_catalog_schema_rejects_changes(change):
    value = page([entry("file.mat", "f")])
    item = value["data"][0]
    if change == "target": item["relationships"]["target"]["data"]["id"] = "other"
    if change == "sha256": item["attributes"]["extra"]["hashes"]["sha256"] = "unverified"
    if change == "size": item["attributes"]["size"] = -1
    if change == "version": item["attributes"]["current_version"] = 0
    if change == "kind": item["attributes"]["kind"] = "participant"
    if change == "total": value["links"]["meta"]["total"] = 2
    if change == "duplicate": value["data"].append(copy.deepcopy(item)); value["links"]["meta"]["total"] = 2
    with pytest.raises(ValueError):
        permissions.catalog_summary([value], "x2rau")


def test_synthetic_full_snapshot(tmp_path):
    make_snapshot(tmp_path)
    report = permissions.check_snapshot(tmp_path)
    assert report["metadata_files_verified"] == 14
    assert report["license_extended_to_github_csv"] is False
    assert report["human_performance_computed"] is False


def test_snapshot_hash_change_rejected(tmp_path):
    make_snapshot(tmp_path)
    with (tmp_path / "license.json").open("ab") as handle: handle.write(b" ")
    with pytest.raises(ValueError, match="bytes/hash"):
        permissions.check_snapshot(tmp_path)


@pytest.mark.parametrize("change", ["path", "duplicate", "http", "timezone", "url"])
def test_snapshot_manifest_errors(tmp_path, change):
    manifest = make_snapshot(tmp_path)
    if change == "path": manifest["files"][0]["path"] = "../escape.json"
    if change == "duplicate": manifest["files"].append(manifest["files"][0])
    if change == "http": manifest["files"][0]["http_status"] = 403
    if change == "timezone": manifest["files"][0]["started_at_utc"] = "2026-10-15T10:00:00+09:00"
    if change == "url": manifest["files"][0]["requested_url"] = "https://example.com/"
    (tmp_path / "source-manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(ValueError):
        permissions.check_snapshot(tmp_path)


def test_existing_output_preserved(tmp_path, monkeypatch):
    path = tmp_path / "output.json"
    path.write_text("unchanged", encoding="utf8")
    monkeypatch.setattr(permissions, "check_snapshot", lambda *args: pytest.fail("read occurred"))
    with pytest.raises(ValueError, match="Output already"):
        permissions.main(["--out", str(path)])
    assert path.read_text() == "unchanged"
