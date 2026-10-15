"""Independent raw task-structure replay; no human performance estimates."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import re
import struct

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/human_task_structure_v1.json"
REQUIRED_FROZEN = {
    "scripts/audit_human_task_structure.py", "tests/test_human_task_structure.py",
    "scripts/verify_human_task_structure.py", "tests/test_human_task_structure_verify.py",
    "configs/human_task_structure_v1.json", "docs/human-task-structure-protocol.md",
    "docs/human-task-structure-sources.md", "data/human/task_structure_evidence_20261015/source-manifest.json",
    "configs/human_osf_origin_manifest.json", "results/human_osf_origin_v1/comparison.json",
    "results/human_osf_origin_v1/verification.json", "requirements-lock.txt",
    "requirements-human-candidate-audit.txt"}


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

    def invalid_constant(value):
        raise VerificationError("Nonfinite JSON number")

    return json.loads(Path(path).read_text(encoding="utf8"), object_pairs_hook=unique,
                      parse_constant=invalid_constant)


def same(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def safe(base, relative):
    require(type(relative) is str and "\\" not in relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "Unsafe relative source path")
    base = Path(base).resolve()
    path = (base / relative).resolve()
    require(path.is_relative_to(base) and path.is_file(), "Missing or escaped source path")
    return path


def current_freeze(config):
    paths = config["frozen_files"]
    require(isinstance(paths, list) and all(type(path) is str for path in paths)
            and len(paths) == len(set(paths)) and REQUIRED_FROZEN <= set(paths), "Frozen source set differs")
    frozen = {name: sha(safe(ROOT, name)) for name in paths}
    manifest_path = safe(ROOT, config["source_manifest"])
    evidence = read_json(manifest_path)
    listed = set()
    for entry in evidence["files"]:
        require(entry["path"] not in listed, "Repeated primary evidence path")
        listed.add(entry["path"])
        path = safe(manifest_path.parent, entry["path"])
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "Primary evidence bytes differ")
        frozen[path.relative_to(ROOT).as_posix()] = sha(path)
    return dict(sorted(frozen.items()))


def registered_inputs(config, cache):
    cache = Path(cache).resolve()
    require(not cache.is_relative_to(ROOT), "Human cache must remain external")
    origin = read_json(safe(ROOT, config["origin_manifest"]))
    parent = read_json(safe(ROOT, config["origin_result"]))
    require(parent["status"] == "pass" and parent["gate"]["passed"] is True and
            type(parent["total_values_compared"]) is int and parent["total_values_compared"] == 75000 and
            type(parent["total_mismatches"]) is int and parent["total_mismatches"] == 0,
            "Verified parent source-origin gate differs")
    selected = [entry for entry in origin["source_files"] if entry["role"] == "osf_raw"]
    require(len(selected) == 25 and len({entry["path"] for entry in selected}) == 25 and
            len({entry["osf_file_id"] for entry in selected}) == 25 and
            len({entry["derived_row_index"] for entry in selected}) == 25 and
            all(entry["derived_group"] == "main" for entry in selected), "Fixed25 main source identity differs")
    checked = []
    for entry in selected:
        path = safe(cache, entry["path"])
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "Raw source bytes differ")
        checked.append({key: entry[key] for key in
                        ("path", "bytes", "sha256", "derived_group", "derived_row_index", "osf_file_id")})
    return checked, origin


def raw_structure(arrays, trials):
    """Replay by original trial and pack integer values with stdlib, without a producer import."""
    shapes = {"A": (1, trials), "C": (1, trials), "S": (trials, 1), "R": (trials, 2)}
    if any(name not in arrays or np.asarray(arrays[name]).shape != shape for name, shape in shapes.items()):
        raise ValueError("Source shape differs")
    if any(np.asarray(arrays[name]).dtype.kind not in "iuf" for name in shapes):
        raise ValueError("Numeric nonboolean arrays required")
    if any(np.isinf(np.asarray(arrays[name])).any() for name in shapes):
        raise ValueError("Infinite source values")
    actions = [float(value) for value in arrays["A"][0]]
    correct = [float(value) for value in arrays["C"][0]]
    states = [float(arrays["S"][trial, 0]) for trial in range(trials)]
    rewards = [float(arrays["R"][trial, card]) for trial in range(trials) for card in range(2)]
    legal = (all(math.isnan(value) or value in (1, 2, 3) for value in actions) and
             all(math.isnan(value) or value in (0, 1) for value in correct) and
             all(value in (1, 2) for value in states) and all(value in (-1, 1) for value in rewards))
    if not legal:
        raise ValueError("Source codes differ")
    logic, other_missing, starts = True, True, [1]
    for trial in range(trials):
        if actions[trial] in (1, 2):
            logic = logic and math.isfinite(correct[trial]) and correct[trial] == int(actions[trial] == states[trial])
        else:
            other_missing = other_missing and math.isnan(correct[trial])
        if trial > 0 and states[trial] != states[trial - 1]:
            starts.append(trial + 1)
    lengths = [end - start for start, end in zip(starts, starts[1:] + [trials + 1])]
    signature = hashlib.sha256(b"".join(struct.pack("<q", int(value)) for value in states + rewards)).hexdigest()
    return {"schema_passed": True, "c_equals_action_state_indicator": logic,
            "query_missing_c_remains_missing": other_missing,
            "state_segment_starts_1based": starts, "state_segment_lengths": lengths,
            "schedule_sha256_little_endian_int64_S_then_R": signature,
            "structural_passed": logic and other_missing}


def replay_rows(config, checked, origin, cache):
    records = []
    for entry in checked:
        row = {"row_index": entry["derived_row_index"], "osf_file_id": entry["osf_file_id"],
               "group": entry["derived_group"], "schedule_group": "regular" if
               entry["derived_row_index"] < origin["groups"]["main"]["regular_rows"] else "irregular"}
        try:
            arrays = loadmat(safe(cache, entry["path"]), variable_names=["A", "C", "S", "R"])
            row.update(raw_structure(arrays, config["trials_per_row"]))
        except Exception as error:
            row.update(structural_passed=False, error_type=type(error).__name__, error_message=str(error))
        records.append(row)
    groups = {}
    for label in ("regular", "irregular"):
        rows = [row for row in records if row["schedule_group"] == label]
        signatures = {row.get("schedule_sha256_little_endian_int64_S_then_R", "FAILED") for row in rows}
        groups[label] = {"rows": len(rows), "unique_schedules": len(signatures),
                         "identical_within_group": len(signatures) == 1 and "FAILED" not in signatures}
    return records, groups


def verify(result_path, registration_path, cache, config_path=None):
    config_path = Path(config_path) if config_path is not None else CONFIG
    config, registration, report = (read_json(path) for path in (config_path, registration_path, result_path))
    require(config["study_id"] == "human_task_structure_v1" and config["scope"] ==
            "exploratory_source_task_structure_not_learning_performance", "Unexpected task audit scope")
    require(type(config["trials_per_row"]) is int and config["trials_per_row"] > 0, "Invalid original trial count")
    require(registration["registration_type"] == "source_task_structure_audit" and
            registration["study_id"] == config["study_id"], "Registration identity differs")
    require(registration["config_sha256"] == sha(config_path), "Registered config hash differs")
    require(re.fullmatch(r"[0-9a-f]{40}", registration["source_git_commit"]), "Invalid registered source commit")
    frozen = current_freeze(config)
    checked, origin = registered_inputs(config, cache)
    require(same(registration["frozen_files"], frozen) and same(registration["inputs"], checked),
            "Registered source or raw input identity differs")
    stamps = [datetime.fromisoformat(value) for value in
              (registration["registered_at_utc"], report["started_at_utc"], report["completed_at_utc"])]
    require(all(value.tzinfo is not None and value.utcoffset().total_seconds() == 0 for value in stamps)
            and stamps[0] <= stamps[1] <= stamps[2], "Task audit chronology differs")
    require(isinstance(report["runtime"], dict) and set(report["runtime"]) == {"python", "numpy"} and
            all(type(value) is str and value for value in report["runtime"].values()), "Runtime provenance differs")
    rows, groups = replay_rows(config, checked, origin, cache)
    passed = len(rows) == 25 and all(row["structural_passed"] for row in rows) and all(
        group["identical_within_group"] for group in groups.values())
    expected = {"study_id": config["study_id"], "status": "pass" if passed else "failed",
                "started_at_utc": report["started_at_utc"], "completed_at_utc": report["completed_at_utc"],
                "registration_sha256": sha(registration_path), "source_git_commit": registration["source_git_commit"],
                "config_sha256": sha(config_path), "frozen_files": frozen, "inputs": checked,
                "rows": rows, "schedule_groups": groups,
                "gate": {"source_coded_descriptive_metric_allowed": passed,
                         "exact_historical_task_execution_version_verified": False,
                         "direct_human_model_similarity_allowed": False},
                "human_accuracy_or_learning_curves_computed": False,
                "scope": "source-coded option/state structure, not behavioral success", "runtime": report["runtime"]}
    require(same(report, expected), "Independent raw structure, schedules, gate or report scope differs")
    return {"status": "pass", "scope": "independent_source_task_structure_not_human_performance",
            "result_sha256": sha(result_path), "registration_sha256": sha(registration_path),
            "config_sha256": sha(config_path), "frozen_files_verified": len(frozen),
            "raw_files_replayed": len(rows), "schedule_groups_verified": len(groups),
            "source_coded_descriptive_gate_passed": passed,
            "human_accuracy_or_learning_curves_computed": False, "errors": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Verification output already exists")
    try:
        result = verify(args.result, args.registration, args.cache)
    except Exception as error:
        result = {"status": "failed", "scope": "independent_source_task_structure_not_human_performance",
                  "human_accuracy_or_learning_curves_computed": False,
                  "errors": [f"{type(error).__name__}: {error}"]}
    result["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "scope", "errors")}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
