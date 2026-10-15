"""Verify the saved public OSF metadata snapshot; no remote files are fetched."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/human/osf_permissions_20261015"
LICENSE_ID = "563c1cf88c5e4a3877f9e96a"
LICENSE_API = f"https://api.osf.io/v2/licenses/{LICENSE_ID}/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode"
EXPECTED_ROLES = {
    "markovic_project_node", "markovic_data_node", "cheng_project_node", "license",
    "markovic_project_children", "markovic_project_providers", "markovic_data_folders",
    "cheng_root_page1", "cheng_root_page2",
    *(f"cheng_dataset{i}_listing" for i in range(1, 6)),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_node(payload, node_id):
    node = payload.get("data", {})
    require(node.get("type") == "nodes" and node.get("id") == node_id, "Node identity differs")
    require(node.get("attributes", {}).get("public") is True, "Node is not declared public")
    relation = node.get("relationships", {}).get("license", {})
    require(relation.get("data") == {"id": LICENSE_ID, "type": "licenses"}, "Explicit node license differs")
    require(relation.get("links", {}).get("related", {}).get("href") == LICENSE_API,
            "License relation URL differs")
    attrs = node["attributes"]
    return {"id": node_id, "title": attrs.get("title"), "public": True,
            "node_license": attrs.get("node_license"), "license_id": LICENSE_ID}


def check_license(payload):
    obj = payload.get("data", {})
    attrs = obj.get("attributes", {})
    require(obj.get("type") == "licenses" and obj.get("id") == LICENSE_ID, "License identity differs")
    require(attrs.get("name") == "CC-By Attribution 4.0 International", "License name differs")
    require(attrs.get("url") == LICENSE_URL, "License legalcode URL differs")
    require(isinstance(attrs.get("text"), str) and attrs["text"].startswith(
        "Creative Commons Attribution 4.0 International Public License"), "License text header differs")
    return {"id": LICENSE_ID, "name": attrs["name"], "url": attrs["url"]}


def catalog_summary(pages, expected_node):
    entries = []
    declared_totals = set()
    for page in pages:
        require(isinstance(page.get("data"), list), "Catalog is not a metadata list")
        total = page.get("links", {}).get("meta", {}).get("total")
        require(isinstance(total, int) and total >= 0, "Catalog total is missing")
        declared_totals.add(total)
        for item in page["data"]:
            require(item.get("type") == "files", "Catalog entry is not file metadata")
            attrs = item.get("attributes", {})
            require(attrs.get("kind") in {"file", "folder"}, "Catalog kind differs")
            require(isinstance(attrs.get("name"), str) and attrs["name"], "Catalog name is missing")
            require(item.get("relationships", {}).get("target", {}).get("data", {}).get("id") == expected_node,
                    "Catalog target node differs")
            if attrs["kind"] == "file":
                require(type(attrs.get("size")) is int and attrs["size"] >= 0, "Invalid server file size")
                require(type(attrs.get("current_version")) is int and attrs["current_version"] >= 1,
                        "Invalid server file version")
                digest = attrs.get("extra", {}).get("hashes", {}).get("sha256")
                require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
                        "Invalid server-declared file SHA-256")
            entries.append(item)
    require(len(declared_totals) == 1 and len(entries) == next(iter(declared_totals)),
            "Catalog pages are incomplete or totals differ")
    ids = [item.get("id") for item in entries]
    require(all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids)), "Duplicate catalog IDs")
    return {"entries": len(entries), "names": [item["attributes"]["name"] for item in entries],
            "server_hashes_are_independent_content_checks": False}


def check_snapshot(directory=SNAPSHOT):
    directory = Path(directory).resolve()
    manifest_path = directory / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    require(manifest.get("schema_version") == 1, "Snapshot manifest schema differs")
    payloads, records = {}, []
    for entry in manifest["files"]:
        require(entry["role"] not in payloads, "Repeated snapshot role")
        relative = Path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Metadata path escapes snapshot")
        path = (directory / relative).resolve()
        require(path.is_relative_to(directory), "Metadata symlink escapes snapshot")
        raw = path.read_bytes()
        require(entry["http_status"] == 200, "Saved metadata did not return HTTP 200")
        require(entry["requested_url"].startswith("https://api.osf.io/v2/"), "Metadata URL is not official OSF API")
        require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"],
                "Saved metadata bytes/hash differ")
        for field in ("started_at_utc", "finished_at_utc"):
            stamp = datetime.fromisoformat(entry[field])
            require(stamp.utcoffset() == timezone.utc.utcoffset(stamp), "Access timestamp is not UTC")
        payloads[entry["role"]] = json.loads(raw)
        records.append(entry)
    require(set(payloads) == EXPECTED_ROLES, "Saved metadata roles differ")
    licensed = {key: check_node(payloads[role], key) for key, role in (
        ("h526v", "markovic_project_node"), ("75jrf", "markovic_data_node"),
        ("x2rau", "cheng_project_node"))}
    license_info = check_license(payloads["license"])
    children = payloads["markovic_project_children"].get("data", [])
    require({o.get("id") for o in children} == {"75jrf", "57wr8"}, "Markovic child catalog differs")
    require(any(o.get("id") == "75jrf" and o.get("attributes", {}).get("title") == "Behavioural data"
                for o in children), "Data component correspondence differs")
    folders = catalog_summary([payloads["markovic_data_folders"]], "75jrf")
    require(set(folders["names"]) == {"Main", "Pilot"}, "Data folder names differ")
    root = catalog_summary([payloads["cheng_root_page1"], payloads["cheng_root_page2"]], "x2rau")
    datasets = {}
    for number in range(1, 6):
        dataset = catalog_summary([payloads[f"cheng_dataset{number}_listing"]], "x2rau")
        require(set(dataset["names"]) == {"pics", f"StimList_Dataset{number}.xlsx", f"alltrial_Dataset{number}.mat"},
                "Cheng dataset metadata filenames differ")
        datasets[f"Dataset{number}"] = dataset
    return {"status": "pass", "scope": manifest["scope"],
            "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "metadata_files_verified": len(records), "source_files": records,
            "explicitly_licensed_nodes": licensed, "license": license_info,
            "markovic_data_folders": folders, "cheng_root": root, "cheng_datasets": datasets,
            "participant_content_requested": False, "human_performance_computed": False,
            "github_csv_identity_checked": False, "license_extended_to_github_csv": False,
            "individual_file_content_exceptions_reviewed": False,
            "previous_access_failures_replaced": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="New output JSON path; overwrite refused")
    args = parser.parse_args(argv)
    require(not args.out or not args.out.exists(), "Output already exists")
    result = check_snapshot()
    script = Path(__file__).resolve()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, check=False, timeout=15)
    result["code_identity"] = {"git_commit": git.stdout.strip() if git.returncode == 0 else None,
                               "script": script.relative_to(ROOT).as_posix(),
                               "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
    result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["python_version"] = sys.version.split()[0]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps({"status": result["status"], "metadata_files_verified": result["metadata_files_verified"],
                      "out": str(args.out) if args.out else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
