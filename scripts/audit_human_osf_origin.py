"""Register and verify partial OSF-to-pybefit source identity; no learning estimates."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import urllib.request

import numpy as np
import scipy
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/human_osf_origin_manifest.json"
FROZEN_FILES = ["scripts/audit_human_osf_origin.py", "tests/test_human_osf_origin.py",
                "configs/human_osf_origin_manifest.json", "docs/human-osf-origin-protocol.md",
                "scripts/verify_human_osf_origin.py", "tests/test_human_osf_origin_verify.py",
                "requirements-lock.txt", "requirements-human-candidate-audit.txt",
                *["data/human/osf_origin_evidence_20261015/" + name for name in
                  ("pilot_page1.json", "pilot_page2.json", "main.json", "data_node.json", "license.json",
                   "source-manifest.json", "README.md")]]
ARRAY_ROLES = ("responses", "outcomes", "correct_responses")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def external(path):
    path = Path(path).resolve()
    require(not path.is_relative_to(ROOT), "Source caches must be outside the repository")
    return path


def cached_path(cache, relative):
    require(isinstance(relative, str) and "\\" not in relative, "Cache paths must be POSIX relative paths")
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "Cache path escapes root")
    root = Path(cache).resolve()
    target = (root / rel).resolve()
    require(target.is_relative_to(root), "Cache symlink escapes root")
    return target


def verified(path, entry):
    raw = Path(path).read_bytes()
    require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"],
            "Input bytes or SHA-256 differ: " + entry["path"])
    return raw


def restore_caches(manifest, cache, derived_cache):
    """Restore only missing hash-pinned inputs; existing bytes are never replaced."""
    cache, derived_cache = external(cache), external(derived_cache)
    for entries, base in ((manifest["source_files"], cache),
                          (manifest["derived_sources"], derived_cache)):
        for entry in entries:
            path = cached_path(base, entry["path"])
            if path.exists():
                verified(path, entry)
                continue
            if "bundled_path" in entry:
                raw = cached_path(ROOT, entry["bundled_path"]).read_bytes()
            else:
                require(entry["url"].startswith("https://"), "Only public HTTPS source URLs are allowed")
                request = urllib.request.Request(entry["url"], headers={
                    "User-Agent": "DrosoLearn-public-source-identity-audit/1.0"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    raw = response.read(entry["bytes"] + 1)
            require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"],
                    "Downloaded input bytes or SHA-256 differ: " + entry["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(raw)


def formatter_names(source, variable_order):
    """Read literal lists and concatenation order without executing upstream code."""
    tree = ast.parse(source)
    lists, concatenation = {}, None

    def names_in(expression):
        if isinstance(expression, ast.Name):
            return [expression.id]
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
            return names_in(expression.left) + names_in(expression.right)
        raise ValueError("Formatter filename concatenation is not a name-only addition")

    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id in variable_order:
                require(target.id not in lists and isinstance(statement.value, ast.List),
                        "Formatter filename lists must be unique literal lists")
                values = ast.literal_eval(statement.value)
                require(all(isinstance(value, str) and value.endswith(".mat") for value in values),
                        "Formatter filename entries must be MAT string literals")
                lists[target.id] = values
            elif target.id == "fnames":
                require(concatenation is None, "Repeated formatter filename concatenation")
                concatenation = names_in(statement.value)
    require(set(lists) == set(variable_order) and concatenation == variable_order,
            "Formatter filename list set/order differs")
    return [value for name in variable_order for value in lists[name]]


def check_inputs(manifest, cache, derived_cache):
    cache, derived_cache = external(cache), external(derived_cache)
    identities, paths = [], {}
    for kind, entries, base in (("source", manifest["source_files"], cache),
                                 ("derived", manifest["derived_sources"], derived_cache)):
        seen = set()
        for entry in entries:
            require(entry["path"] not in seen, "Repeated cache input path")
            seen.add(entry["path"])
            path = cached_path(base, entry["path"])
            verified(path, entry)
            identities.append({"kind": kind, **{key: entry[key] for key in
                ("path", "bytes", "sha256", "role")},
                **({"group": entry["group"]} if "group" in entry else {})})
            if kind == "source" and entry["role"] != "osf_raw":
                require(entry["role"] not in paths, "Repeated metadata/formatter role")
                paths[entry["role"]] = path
    return identities, paths


def inspect_coverage(manifest, paths):
    pages = [json.loads(paths[role].read_text(encoding="utf8"))
             for role in ("catalog_pilot_1", "catalog_pilot_2")]
    items = [item for page in pages for item in page["data"]]
    require(len(items) == pages[0]["links"]["meta"]["total"] == pages[1]["links"]["meta"]["total"],
            "Official filename catalog is incomplete")
    require(len({item["id"] for item in items}) == len(items), "Duplicate official catalog ID")
    by_name_hash, by_id = defaultdict(list), {}
    for item in items:
        require(item.get("type") == "files", "Official catalog item type differs")
        attrs = item["attributes"]
        require(item.get("relationships", {}).get("target", {}).get("data", {}).get("id") == "75jrf",
                "Official catalog target differs")
        if attrs.get("kind") == "file":
            name_hash = hashlib.sha256(attrs["name"].encode("utf8")).hexdigest()
            by_name_hash[name_hash].append(item["id"])
            by_id[item["id"]] = item
    groups = {}
    for group, spec in manifest["groups"].items():
        names = formatter_names(paths["formatter_" + group].read_text(encoding="utf8"), spec["filename_variables"])
        require(len(names) == spec["rows"], "Formatter row count differs")
        mapping, duplicates = [], defaultdict(list)
        for index, name in enumerate(names):
            token = hashlib.sha256(name.encode("utf8")).hexdigest()
            matches = sorted(by_name_hash.get(token, []))
            duplicates[token].append(index)
            mapping.append({"processed_row_index": index, "filename_token_sha256": token,
                            "matching_osf_file_ids": matches, "matches": len(matches)})
        expected = manifest["coverage"]["groups"][group]
        require(mapping == expected["mapping"], "Frozen filename-to-file-ID coverage differs")
        duplicate_groups = [indices for indices in duplicates.values() if len(indices) > 1]
        require(duplicate_groups == expected["duplicate_row_groups"], "Formatter duplicate groups differ")
        groups[group] = {"mapping": mapping, "duplicate_row_groups": duplicate_groups,
                         "rows": len(names), "available_rows": sum(row["matches"] == 1 for row in mapping),
                         "unmatched_rows": sum(row["matches"] == 0 for row in mapping),
                         "ambiguous_rows": sum(row["matches"] > 1 for row in mapping)}
    raw = [entry for entry in manifest["source_files"] if entry["role"] == "osf_raw"]
    selected = {(entry["derived_group"], entry["derived_row_index"], entry["osf_file_id"]) for entry in raw}
    available = {(group, row["processed_row_index"], row["matching_osf_file_ids"][0])
                 for group, summary in groups.items() for row in summary["mapping"] if row["matches"] == 1}
    require(len(selected) == len(raw) and selected == available, "Raw selection does not cover all exact matches")
    for entry in raw:
        row = groups[entry["derived_group"]]["mapping"][entry["derived_row_index"]]
        attrs = by_id[entry["osf_file_id"]]["attributes"]
        require(row["filename_token_sha256"] == entry["filename_token_sha256"], "Selected source token differs")
        require(attrs["size"] == entry["bytes"] and attrs["extra"]["hashes"]["sha256"] == entry["sha256"] and
                attrs["current_version"] == entry["version"], "Selected official source metadata differs")
    expected = manifest["expected_coverage"]
    for group in manifest["groups"]:
        require(groups[group]["available_rows"] == expected[group + "_available_rows"] and
                groups[group]["unmatched_rows"] == expected[group + "_unavailable_rows"] and
                groups[group]["ambiguous_rows"] == 0, "Coverage totals differ")
    require(len(raw) == expected["selected_raw_files"] and
            sum(g["rows"] for g in groups.values()) == expected["total_rows"] and
            sum(g["unmatched_rows"] for g in groups.values()) == expected["unmatched_rows"], "Overall coverage differs")
    node = json.loads(paths["data_license_node"].read_text(encoding="utf8"))["data"]
    license_object = json.loads(paths["data_license"].read_text(encoding="utf8"))["data"]
    require(node["id"] == "75jrf" and node["attributes"]["public"] is True and
            node["relationships"]["license"]["data"]["id"] == manifest["raw_license"]["license_id"],
            "Explicit original-data license relation differs")
    require(license_object["id"] == manifest["raw_license"]["license_id"] and
            license_object["attributes"]["url"] == manifest["raw_license"]["license_url"], "Data license URL differs")
    return {"groups": groups, "selected_raw_files": len(raw), "unmatched_rows": expected["unmatched_rows"],
            "folder_name_does_not_assign_group": True}


def source_freeze():
    return [{"path": relative, "bytes": (ROOT / relative).stat().st_size,
             "sha256": digest(ROOT / relative)} for relative in FROZEN_FILES]


def git_head():
    process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True, timeout=15)
    return process.stdout.strip()


def register(cache, derived_cache, manifest_path=MANIFEST, fetch=False):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if fetch:
        restore_caches(manifest, cache, derived_cache)
    identities, paths = check_inputs(manifest, cache, derived_cache)
    coverage = inspect_coverage(manifest, paths)
    return {"registration_type": "human_osf_origin_source_identity", "study_id": manifest["study_id"],
            "registered_at_utc": datetime.now(timezone.utc).isoformat(), "source_git_commit": git_head(),
            "manifest_sha256": digest(manifest_path), "frozen_files": source_freeze(),
            "inputs": identities, "coverage": coverage,
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
            "raw_to_CSV_comparison_performed": False, "human_performance_computed": False}


def reconstruct(raw, trials):
    expected = {"A": (1, trials), "C": (1, trials), "S": (trials, 1), "R": (trials, 2)}
    require(all(key in raw and np.asarray(raw[key]).shape == shape for key, shape in expected.items()),
            "Raw A/R/S/C keys or dimensions differ")
    require(all(np.asarray(raw[key]).dtype.kind in "iuf" for key in expected),
            "Raw arrays must have real numeric nonboolean dtypes")
    require(all(not np.isinf(np.asarray(raw[key])).any() for key in expected), "Infinite raw values are not supported")
    action = np.asarray(raw["A"], dtype=float)[0] - 1
    correct = np.asarray(raw["C"], dtype=float)[0].copy()
    states = np.asarray(raw["S"], dtype=float)[:, 0]
    rewards = np.asarray(raw["R"], dtype=float)
    require(np.isin(action[~np.isnan(action)], [0, 1, 2]).all(), "Raw action code differs")
    require(np.isin(correct[~np.isnan(correct)], [0, 1]).all(), "Raw C code differs")
    require(np.isin(states, [1, 2]).all() and np.isin(rewards, [-1, 1]).all(), "Raw state/reward code differs")
    hint, missing = action == 2, np.isnan(action)
    binary = ~hint & ~missing
    outcomes = np.zeros(trials)
    indices = np.flatnonzero(binary)
    outcomes[binary] = (rewards[indices, action[binary].astype(int)] + 1) / 2
    outcomes[missing] = np.nan
    outcomes[hint] = states[hint] + 1
    return {"responses": action, "outcomes": outcomes, "correct_responses": correct}


def read_derived(path, rows, trials, responses=False):
    with Path(path).open(encoding="utf8", newline="") as handle:
        records = list(csv.reader(handle))
    header = ["subject", *map(str, range(trials)), *( ["condition"] if responses else [])]
    require(records and records[0] == header and len(records) == rows + 1, "Derived CSV dimensions/header differ")
    require(all(len(row) == len(header) for row in records[1:]), "Ragged derived CSV")
    require([row[0] for row in records[1:]] == list(map(str, range(rows))), "Derived subject row keys differ")
    values = np.empty((rows, trials), dtype=float)
    for index, row in enumerate(records[1:]):
        for trial, token in enumerate(row[1:trials + 1]):
            value = float(token) if token != "" else np.nan
            require(token == "" or math.isfinite(value), "Only blank CSV cells may encode missing values")
            values[index, trial] = value
    return values, [row[-1] for row in records[1:]] if responses else None


def compare_values(expected, observed):
    expected, observed = np.asarray(expected, dtype=float), np.asarray(observed, dtype=float)
    require(expected.shape == observed.shape and expected.ndim == 1, "Compared array dimensions differ")
    identical = (expected == observed) | (np.isnan(expected) & np.isnan(observed))
    indices = np.flatnonzero(~identical)
    return {"values_compared": len(expected), "mismatch_count": len(indices),
            "first_mismatch_indices": indices[:20].tolist()}


def run(registration_path, cache, derived_cache, manifest_path=MANIFEST):
    started_at_utc = datetime.now(timezone.utc).isoformat()
    manifest_path, registration_path = Path(manifest_path), Path(registration_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    registration = json.loads(registration_path.read_text(encoding="utf8"))
    require(registration.get("registration_type") == "human_osf_origin_source_identity" and
            registration.get("study_id") == manifest["study_id"], "Registration identity differs")
    require(registration["manifest_sha256"] == digest(manifest_path), "Frozen manifest differs")
    require(registration["frozen_files"] == source_freeze(), "Frozen source/test/protocol files differ")
    identities, paths = check_inputs(manifest, cache, derived_cache)
    require(identities == registration["inputs"], "Frozen input identities differ")
    coverage = inspect_coverage(manifest, paths)
    require(coverage == registration["coverage"], "Frozen coverage differs")
    trials = manifest["trials_per_row"]
    tables, conditions = {}, {}
    for entry in manifest["derived_sources"]:
        group, role = entry["group"], entry["role"]
        require((group, role) not in tables, "Repeated derived role")
        values, labels = read_derived(cached_path(derived_cache, entry["path"]),
                                     manifest["groups"][group]["rows"], trials, role == "responses")
        tables[(group, role)] = values
        if labels is not None:
            conditions[group] = labels
    require(set(tables) == {(group, role) for group in manifest["groups"] for role in ARRAY_ROLES},
            "Derived CSV role set differs")
    rows = []
    for entry in manifest["source_files"]:
        if entry["role"] != "osf_raw":
            continue
        group, index = entry["derived_group"], entry["derived_row_index"]
        record = {"group": group, "row_index": index, "osf_file_id": entry["osf_file_id"],
                  "filename_token_sha256": entry["filename_token_sha256"]}
        try:
            raw = loadmat(cached_path(cache, entry["path"]), variable_names=["A", "R", "S", "C"])
            expected = reconstruct(raw, trials)
            record["comparison"] = {role: compare_values(expected[role], tables[(group, role)][index])
                                    for role in ARRAY_ROLES}
            label = "regular" if index < manifest["groups"][group]["regular_rows"] else "irregular"
            record["condition_matches"] = conditions[group][index] == label
            record["row_passed"] = record["condition_matches"] and all(
                item["mismatch_count"] == 0 for item in record["comparison"].values())
        except Exception as exc:
            record.update(row_passed=False, error_type=type(exc).__name__, error_message=str(exc))
        rows.append(record)
    passed = len(rows) == manifest["expected_coverage"]["selected_raw_files"] and all(row["row_passed"] for row in rows)
    completed_at_utc = datetime.now(timezone.utc).isoformat()
    return {"status": "pass" if passed else "failed", "study_id": manifest["study_id"],
            "scope": manifest["scope"], "registration_sha256": digest(registration_path),
            "registration_source_git_commit": registration["source_git_commit"],
            "manifest_sha256": digest(manifest_path), "source_freeze": registration["frozen_files"],
            "inputs": identities, "started_from_registered_inputs": True,
            "started_at_utc": started_at_utc, "completed_at_utc": completed_at_utc,
            "checked_at_utc": completed_at_utc,
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
            "coverage": coverage, "rows": rows,
            "total_values_compared": sum(item["values_compared"] for row in rows for item in row.get("comparison", {}).values()),
            "total_mismatches": sum(item["mismatch_count"] for row in rows for item in row.get("comparison", {}).values()),
            "gate": {"scope": "selected25_source_rows_only", "selected_rows": len(rows), "passed": passed},
            "unmatched_rows_preserved": manifest["expected_coverage"]["unmatched_rows"],
            "human_performance_computed": False, "license_extended_to_derived_CSV": False,
            "participant_names_or_original_filenames_reported": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["register", "run"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--derived-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--fetch", action="store_true", help="register only: restore missing inputs by pinned URL/hash")
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Output already exists; choose a new path")
    require(args.mode != "run" or args.registration is not None, "run requires --registration")
    require(not args.fetch or args.mode == "register", "--fetch is only allowed while registering")
    try:
        result = register(args.cache, args.derived_cache, fetch=args.fetch) if args.mode == "register" else run(
            args.registration, args.cache, args.derived_cache)
    except Exception as exc:
        result = {"status": "failed", "phase": args.mode, "error_type": type(exc).__name__,
                  "error_message": str(exc), "human_performance_computed": False,
                  "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    ok = result.get("status") != "failed"
    print(json.dumps({"status": "registered" if args.mode == "register" and ok else result.get("status"),
                      "out": str(args.out)}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
