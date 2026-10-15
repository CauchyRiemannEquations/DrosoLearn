"""Independent source-to-derived identity replay; no human learning estimates."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import ast
import csv
import hashlib
import json
import re

import numpy as np
from scipy.io import loadmat


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def safe(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Relative source path required")
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path.is_file(), "Unsafe or missing source file")
    return path


def filename_tokens(formatter, variables):
    """Read literal string lists only; never execute an upstream formatter."""
    module = ast.parse(Path(formatter).read_text(encoding="utf-8"))
    found, order = {}, None
    def concatenate(node):
        if isinstance(node, ast.Name):
            return [node.id]
        require(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add), "Invalid formatter concatenation")
        return concatenate(node.left) + concatenate(node.right)
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in variables:
                require(target.id not in found, "Duplicate formatter filename declaration")
                require(isinstance(node.value, ast.List), "Literal filename-token list required")
                value = ast.literal_eval(node.value)
                require(isinstance(value, list) and all(isinstance(x, str) and x.endswith(".mat") for x in value),
                        "Literal MAT filename-token list required")
                found[target.id] = value
            elif isinstance(target, ast.Name) and target.id == "fnames":
                require(order is None, "Duplicate formatter concatenation")
                order = concatenate(node.value)
    require(set(found) == set(variables) and order == variables, "Formatter filename declarations/order missing")
    return [token for variable in variables for token in found[variable]]


def transform_raw(arrays, trials):
    """Implement the author's formulas by trial; preserve C and NaN locations."""
    shapes = {"A": (1, trials), "R": (trials, 2), "S": (trials, 1), "C": (1, trials)}
    for name, shape in shapes.items():
        require(name in arrays, "Required raw MAT variable missing")
        value = np.asarray(arrays[name])
        require(value.shape == shape and value.dtype.kind in "iuf", "Raw MAT shape/type differs")
        require(not np.isinf(value).any(), "Infinite raw MAT measurement")
    actions = np.asarray(arrays["A"], dtype=float)[0]
    rewards = np.asarray(arrays["R"], dtype=float)
    states = np.asarray(arrays["S"], dtype=float)[:, 0]
    correct = np.asarray(arrays["C"], dtype=float)[0]
    require(set(actions[np.isfinite(actions)].tolist()) <= {1, 2, 3}, "Invalid raw action code")
    require(set(correct[np.isfinite(correct)].tolist()) <= {0, 1}, "Invalid raw correctness code")
    require(np.isin(states, [1, 2]).all() and np.isin(rewards, [-1, 1]).all(), "Invalid raw state/reward code")
    outcomes = np.empty(trials, dtype=float)
    for index, action in enumerate(actions):
        if np.isnan(action):
            outcomes[index] = np.nan
        elif action == 3:
            outcomes[index] = states[index] + 1
        else:
            outcomes[index] = (rewards[index, int(action) - 1] + 1) / 2
    return {"responses": actions - 1, "outcomes": outcomes, "correct_responses": correct.copy()}


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/human_osf_origin_manifest.json"
FROZEN = ["scripts/audit_human_osf_origin.py", "tests/test_human_osf_origin.py",
          "configs/human_osf_origin_manifest.json", "docs/human-osf-origin-protocol.md",
          "scripts/verify_human_osf_origin.py", "tests/test_human_osf_origin_verify.py",
          "requirements-lock.txt", "requirements-human-candidate-audit.txt",
          *["data/human/osf_origin_evidence_20261015/" + name for name in
            ("pilot_page1.json", "pilot_page2.json", "main.json", "data_node.json",
             "license.json", "source-manifest.json", "README.md")]]
ROLES = ("responses", "outcomes", "correct_responses")


def inspect_inputs(manifest, cache, derived_cache):
    identities, metadata = [], {}
    for kind, entries, base in (("source", manifest["source_files"], cache),
                                ("derived", manifest["derived_sources"], derived_cache)):
        require(not Path(base).resolve().is_relative_to(ROOT), "Raw/derived cache must remain external")
        seen = set()
        for item in entries:
            require(item["path"] not in seen, "Duplicate input cache path")
            seen.add(item["path"])
            require("\\" not in item["path"] and ".." not in Path(item["path"]).parts, "Unsafe input cache path")
            path = safe(base, item["path"])
            require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Input size/hash differs")
            identity = {"kind": kind, **{key: item[key] for key in ("path", "bytes", "sha256", "role")}}
            if "group" in item:
                identity["group"] = item["group"]
            identities.append(identity)
            if kind == "source" and item["role"] != "osf_raw":
                require(item["role"] not in metadata, "Duplicate input role")
                metadata[item["role"]] = path
    return identities, metadata


def coverage_from_sources(manifest, metadata):
    pages = [read_json(metadata[name]) for name in ("catalog_pilot_1", "catalog_pilot_2")]
    catalog = [entry for page in pages for entry in page["data"]]
    require(all(page["links"]["meta"]["total"] == len(catalog) for page in pages), "Official catalog pagination incomplete")
    require(len({item["id"] for item in catalog}) == len(catalog), "Duplicate official file IDs")
    by_hash, by_id = defaultdict(list), {}
    for item in catalog:
        require(item["type"] == "files" and item["relationships"]["target"]["data"]["id"] == "75jrf", "Official catalog identity differs")
        if item["attributes"]["kind"] == "file":
            token = hashlib.sha256(item["attributes"]["name"].encode("utf8")).hexdigest()
            by_hash[token].append(item["id"])
            by_id[item["id"]] = item["attributes"]
    groups = {}
    for group, spec in manifest["groups"].items():
        names = filename_tokens(metadata["formatter_" + group], spec["filename_variables"])
        require(len(names) == spec["rows"], "Formatter/derived row count differs")
        duplicate = defaultdict(list)
        mapping = []
        for index, name in enumerate(names):
            token = hashlib.sha256(name.encode("utf8")).hexdigest()
            matches = sorted(by_hash.get(token, []))
            duplicate[token].append(index)
            mapping.append({"processed_row_index": index, "filename_token_sha256": token,
                            "matching_osf_file_ids": matches, "matches": len(matches)})
        duplicates = [indices for indices in duplicate.values() if len(indices) > 1]
        declared = manifest["coverage"]["groups"][group]
        require(exact(mapping, declared["mapping"]) and exact(duplicates, declared["duplicate_row_groups"]),
                "Frozen filename/catalog coverage differs")
        groups[group] = {"mapping": mapping, "duplicate_row_groups": duplicates, "rows": len(mapping),
                         "available_rows": sum(item["matches"] == 1 for item in mapping),
                         "unmatched_rows": sum(item["matches"] == 0 for item in mapping),
                         "ambiguous_rows": sum(item["matches"] > 1 for item in mapping)}
    raw = [item for item in manifest["source_files"] if item["role"] == "osf_raw"]
    selected = {(item["derived_group"], item["derived_row_index"], item["osf_file_id"]) for item in raw}
    available = {(group, row["processed_row_index"], row["matching_osf_file_ids"][0])
                 for group, entry in groups.items() for row in entry["mapping"] if row["matches"] == 1}
    require(len(selected) == len(raw) and selected == available, "Selection does not cover all exact filename matches")
    for item in raw:
        row = groups[item["derived_group"]]["mapping"][item["derived_row_index"]]
        attrs = by_id[item["osf_file_id"]]
        require(row["filename_token_sha256"] == item["filename_token_sha256"] and attrs["size"] == item["bytes"] and
                attrs["extra"]["hashes"]["sha256"] == item["sha256"] and attrs["current_version"] == item["version"],
                "Official raw source identity/version differs")
    expected = manifest["expected_coverage"]
    for group, entry in groups.items():
        require(entry["available_rows"] == expected[group + "_available_rows"] and
                entry["unmatched_rows"] == expected[group + "_unavailable_rows"] and entry["ambiguous_rows"] == 0,
                "Coverage denominator differs")
    require(len(raw) == expected["selected_raw_files"] and sum(item["rows"] for item in groups.values()) == expected["total_rows"] and
            sum(item["unmatched_rows"] for item in groups.values()) == expected["unmatched_rows"], "Whole-row coverage differs")
    node = read_json(metadata["data_license_node"])["data"]
    license_object = read_json(metadata["data_license"])["data"]
    require(node["id"] == "75jrf" and node["attributes"]["public"] is True and
            node["relationships"]["license"]["data"]["id"] == manifest["raw_license"]["license_id"], "Original data license binding differs")
    require(license_object["id"] == manifest["raw_license"]["license_id"] and
            license_object["attributes"]["url"] == manifest["raw_license"]["license_url"], "Original data license URL differs")
    return {"groups": groups, "selected_raw_files": len(raw), "unmatched_rows": expected["unmatched_rows"],
            "folder_name_does_not_assign_group": True}


def derived_table(path, rows, trials, responses):
    with Path(path).open(encoding="utf8", newline="") as stream:
        reader = csv.reader(stream)
        require(next(reader, None) == ["subject", *map(str, range(trials)), *(["condition"] if responses else [])], "Derived CSV header differs")
        source = list(reader)
    require(len(source) == rows and all(len(row) == trials + 1 + int(responses) for row in source), "Derived CSV shape differs")
    require([row[0] for row in source] == list(map(str, range(rows))), "Derived row indices differ")
    array = np.empty((rows, trials))
    for index, row in enumerate(source):
        for trial in range(trials):
            token = row[trial + 1]
            array[index, trial] = float(token) if token else np.nan
            require(not token or np.isfinite(array[index, trial]), "Only blank derived cells may encode missingness")
    return array, [row[-1] for row in source] if responses else None


def replay_rows(manifest, cache, derived_cache):
    trials, tables, conditions = manifest["trials_per_row"], {}, {}
    for item in manifest["derived_sources"]:
        group, role = item["group"], item["role"]
        require((group, role) not in tables, "Duplicate derived role")
        table, condition = derived_table(safe(derived_cache, item["path"]), manifest["groups"][group]["rows"], trials, role == "responses")
        tables[(group, role)] = table
        if condition is not None:
            conditions[group] = condition
    require(set(tables) == {(group, role) for group in manifest["groups"] for role in ROLES}, "Derived role coverage differs")
    records = []
    for item in manifest["source_files"]:
        if item["role"] != "osf_raw":
            continue
        group, index = item["derived_group"], item["derived_row_index"]
        raw = loadmat(safe(cache, item["path"]), variable_names=["A", "R", "S", "C"])
        expected = transform_raw(raw, trials)
        comparisons = {}
        for role in ROLES:
            observed = tables[(group, role)][index]
            match = np.equal(expected[role], observed) | (np.isnan(expected[role]) & np.isnan(observed))
            mismatch = [i for i, equal in enumerate(match) if not equal]
            comparisons[role] = {"values_compared": trials, "mismatch_count": len(mismatch), "first_mismatch_indices": mismatch[:20]}
        condition = "regular" if index < manifest["groups"][group]["regular_rows"] else "irregular"
        condition_matches = conditions[group][index] == condition
        records.append({"group": group, "row_index": index, "osf_file_id": item["osf_file_id"],
                        "filename_token_sha256": item["filename_token_sha256"], "comparison": comparisons,
                        "condition_matches": condition_matches,
                        "row_passed": condition_matches and all(value["mismatch_count"] == 0 for value in comparisons.values())})
    return records


def verify(result_path, registration_path, cache, derived_cache, manifest_path=DEFAULT_MANIFEST):
    result_path, registration_path, manifest_path = Path(result_path), Path(registration_path), Path(manifest_path)
    manifest, registration, report = read_json(manifest_path), read_json(registration_path), read_json(result_path)
    require(manifest["study_id"] == "human_osf_origin_v1" and
            manifest["scope"] == "partial_source_identity_verification_not_human_learning_comparison", "Unexpected audit scope")
    require(manifest["selection_frozen_before_value_comparison"] is True and
            manifest["value_comparison_performed_before_registration"] is False, "Prospective source selection flag differs")
    require(registration["registration_type"] == "human_osf_origin_source_identity" and
            registration["study_id"] == manifest["study_id"], "Registration identity differs")
    require(registration["manifest_sha256"] == sha(manifest_path) == report["manifest_sha256"], "Manifest hash differs")
    frozen = [{"path": name, "bytes": safe(ROOT, name).stat().st_size, "sha256": sha(safe(ROOT, name))} for name in FROZEN]
    require(exact(registration["frozen_files"], frozen) and exact(report["source_freeze"], frozen), "Frozen source/protocol/test differs")
    inputs, metadata = inspect_inputs(manifest, cache, derived_cache)
    coverage = coverage_from_sources(manifest, metadata)
    require(exact(inputs, registration["inputs"]) and exact(coverage, registration["coverage"]), "Registered inputs/coverage differ")
    require(registration["raw_to_CSV_comparison_performed"] is False and registration["human_performance_computed"] is False,
            "Registration analysis scope differs")
    require(re.fullmatch(r"[0-9a-f]{40}", registration["source_git_commit"]), "Invalid source commit")
    registered = datetime.fromisoformat(registration["registered_at_utc"])
    started = datetime.fromisoformat(report["started_at_utc"])
    completed = datetime.fromisoformat(report["completed_at_utc"])
    checked = datetime.fromisoformat(report["checked_at_utc"])
    require(all(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0
                for stamp in (registered, started, completed, checked)) and
            registered <= started <= completed == checked, "Audit chronology differs")
    require(all(isinstance(runtime, dict) and set(runtime) == {"python", "numpy", "scipy"} and
                all(isinstance(value, str) and value for value in runtime.values())
                for runtime in (registration["runtime"], report["runtime"])), "Runtime provenance differs")
    records = replay_rows(manifest, cache, derived_cache)
    passed = len(records) == manifest["expected_coverage"]["selected_raw_files"] and all(row["row_passed"] for row in records)
    expected = {"status": "pass" if passed else "failed", "study_id": manifest["study_id"], "scope": manifest["scope"],
                "registration_sha256": sha(registration_path), "registration_source_git_commit": registration["source_git_commit"],
                "manifest_sha256": sha(manifest_path), "source_freeze": frozen, "inputs": inputs,
                "started_from_registered_inputs": True, "started_at_utc": report["started_at_utc"],
                "completed_at_utc": report["completed_at_utc"], "checked_at_utc": report["checked_at_utc"], "runtime": report["runtime"],
                "coverage": coverage, "rows": records,
                "total_values_compared": sum(value["values_compared"] for row in records for value in row["comparison"].values()),
                "total_mismatches": sum(value["mismatch_count"] for row in records for value in row["comparison"].values()),
                "gate": {"scope": "selected25_source_rows_only", "selected_rows": len(records), "passed": passed},
                "unmatched_rows_preserved": manifest["expected_coverage"]["unmatched_rows"], "human_performance_computed": False,
                "license_extended_to_derived_CSV": False, "participant_names_or_original_filenames_reported": False}
    require(exact(report, expected), "Source-to-derived arithmetic, coverage or scope differs")
    return {"status": "passed", "scope": "independent_partial_source_identity_replay_no_learning_estimates",
            "result_sha256": sha(result_path), "registration_sha256": sha(registration_path), "manifest_sha256": sha(manifest_path),
            "input_files_verified": len(inputs), "source_files_frozen": len(frozen), "raw_rows_replayed": len(records),
            "processed_rows_retained": manifest["expected_coverage"]["total_rows"],
            "unmatched_rows_preserved": manifest["expected_coverage"]["unmatched_rows"],
            "values_replayed": expected["total_values_compared"], "mismatches_verified": expected["total_mismatches"],
            "source_identity_gate_passed": passed, "errors": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--derived-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Verification output already exists")
    try:
        result = verify(args.result, args.registration, args.cache, args.derived_cache, args.manifest)
    except Exception as error:
        result = {"status": "failed", "scope": "independent_partial_source_identity_replay_no_learning_estimates",
                  "errors": [f"{type(error).__name__}: {error}"]}
    result["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "scope", "errors")}))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
