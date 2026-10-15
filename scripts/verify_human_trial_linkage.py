"""Independent exact-record linkage replay; no human learning estimates."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import re

import numpy as np
import scipy
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


def safe(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), "Relative source path required")
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path.is_file(), "Unsafe or missing file")
    return path


CONFIG = {"audit_id": "human_trial_linkage_v1", "scope": "exact_record_linkage_and_semantics_only",
          "source_root": "data/human/stolz_2022_openneuro",
          "source_manifest": "data/human/stolz_2022_openneuro/source-manifest.json",
          "evidence_manifest": "data/human/stolz_2022_linkage_evidence/source-manifest.json",
          "performance_analysis": False, "action_map": {"left": 1, "right": 2},
          "expectation_offset": 3, "approximate_matching": False}
BASE_FROZEN = {"src/drosolearn/human_trial_linkage.py", "src/drosolearn/human_trial_data.py",
               "scripts/verify_human_trial_linkage.py", "tests/test_human_trial_linkage.py",
               "tests/test_human_trial_linkage_run.py", "tests/test_human_trial_linkage_verify.py",
               "docs/human-trial-linkage-protocol.md", "requirements-lock.txt",
               "docs/human-trial-linkage-source-review.md",
               "requirements-human-candidate-audit.txt", "pyproject.toml"}
READINESS = {"record_linkage": "unique_exact_feedback_candidates_only_not_person_identity",
             "original_trial_index": "feedback_ordinal_only_not_verified_original_trial",
             "action_identity": "fixed_left1_right2_numeric_diagnostic_not_verified_shape_ID",
             "expectation": "fixed_plus3_code_diagnostic_not_calibrated_choice_probability",
             "reversal_boundaries_or_correct_shape": "not_reconstructed", "RT_transfer": "not_performed",
             "performance_or_human_model_similarity": "not_tested"}
TASKS = ("REW", "PUN")
START = {100: "REW", 200: "PUN"}
END = {101: "REW", 201: "PUN"}
PAUSE = {150, 151}
CHOICE = {51: ("REW", 1), 52: ("REW", 2), 71: ("PUN", 1), 72: ("PUN", 2)}
EXPECT = {**{i: ("REW", i - 50) for i in range(53, 57)},
          **{i: ("PUN", i - 70) for i in range(73, 77)}}
FB = {60: ("REW", 0), 61: ("REW", 1), 80: ("PUN", 0), 81: ("PUN", 1)}
KNOWN = set(START) | set(END) | PAUSE | set(CHOICE) | set(EXPECT) | set(FB)


def event_rows(path):
    prepared = []
    active, paused = None, False
    last_onset = last_sample = -math.inf
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        require(next(reader, None) == ["onset", "duration", "sample", "value"], "Event schema differs")
        for row_number, raw in enumerate(reader, 2):
            require(len(raw) == 4, "Malformed event TSV")
            onset, duration, sample_text, marker_text = raw
            onset, duration = float(onset), float(duration)
            require(math.isfinite(onset) and math.isfinite(duration) and duration >= 0, "Invalid event time")
            require(re.fullmatch(r"[0-9]+", sample_text), "Invalid event sample")
            sample = int(sample_text)
            require(str(sample) == sample_text, "Noncanonical integer sample")
            match = re.fullmatch(r"S(?:_| )?([0-9]{2,3})", marker_text.strip())
            code = int(match[1]) if match else None
            if code in START:
                active, paused = START[code], False
            elif code in END:
                active, paused = None, False
            elif code in PAUSE:
                paused = code == 150
            prepared.append({"row": row_number, "code": code, "active": active, "paused": paused,
                             "time_reversed": onset < last_onset or sample < last_sample})
            last_onset, last_sample = onset, sample
    require(prepared, "Empty event TSV")
    return prepared


def partition(rows):
    """Use successive global feedback positions, including across task switches."""
    end_positions = [i for i, row in enumerate(rows) if row["code"] in FB]
    blocks, ordinals = [], Counter()
    start, previous_feedback = 0, None
    for end in end_positions:
        segment = rows[start:end + 1]
        event = segment[-1]
        task, flag = FB[event["code"]]
        ordinals[task] += 1
        choices = [row for row in segment[:-1] if row["code"] in CHOICE and CHOICE[row["code"]][0] == task]
        expectations = [row for row in segment[:-1] if row["code"] in EXPECT and EXPECT[row["code"]][0] == task]
        issues = []
        if len(choices) != 1:
            issues.append("choice_count_not_one")
        if len(expectations) != 1:
            issues.append("expectation_count_not_one")
        if any(row["time_reversed"] for row in segment):
            issues.append("decreasing_event_time_or_sample")
        if event["active"] != task or event["paused"]:
            issues.append("feedback_outside_matching_active_task")
        if len(choices) == 1:
            chosen = choices[0]
            if chosen["active"] != task or chosen["paused"]:
                issues.append("choice_outside_matching_active_task")
            after = [row for row in segment[:-1] if row["row"] > chosen["row"]]
            if any(row["code"] in set(START) | set(END) | PAUSE for row in after):
                issues.append("boundary_after_choice")
            if any(row["code"] not in KNOWN for row in after):
                issues.append("unknown_marker_after_choice")
            if any((row["code"] in CHOICE and CHOICE[row["code"]][0] != task) or
                   (row["code"] in EXPECT and EXPECT[row["code"]][0] != task) for row in segment[:-1]):
                issues.append("opposite_task_event_in_feedback_segment")
            if len(expectations) == 1 and not chosen["row"] < expectations[0]["row"] < event["row"]:
                issues.append("choice_expectation_feedback_order_not_strict")
        blocks.append({"task": task, "feedback_ordinal": ordinals[task], "feedback_event_row": event["row"],
                       "feedback_flag": flag, "previous_feedback_event_row": previous_feedback,
                       "segment_event_rows": [row["row"] for row in segment],
                       "choice_event_rows": [row["row"] for row in choices],
                       "expectation_event_rows": [row["row"] for row in expectations],
                       "choice_side_code": CHOICE[choices[0]["code"]][1] if len(choices) == 1 else None,
                       "expectation_mat_code": EXPECT[expectations[0]["code"]][1] if len(expectations) == 1 else None,
                       "strict_comparable": not issues, "issues": issues})
        previous_feedback = event["row"]
        start = end + 1
    tail = [row["row"] for row in rows[start:]]
    coverage = [r for block in blocks for r in block["segment_event_rows"]] + tail
    require(coverage == [row["row"] for row in rows], "Partition lost/duplicated event rows")
    return {"blocks": blocks, "tail_event_rows": tail, "event_count": len(rows)}


def mat_rows(path):
    raw = loadmat(path, simplify_cells=True)
    keys = [f"{task}_{field}" for task in TASKS for field in ("feedback", "action", "expect")]
    count = None
    for key in keys:
        vector = raw[key]
        require(isinstance(vector, np.ndarray) and vector.dtype == object and vector.ndim == 1,
                "Expected one-dimensional MAT cell vector")
        if count is None:
            count = len(vector)
        require(len(vector) == count and count > 0, "MAT row coverage differs")
        allowed = {0, 1} if key.endswith("feedback") else ({1, 2} if key.endswith("action") else {3, 4, 5, 6})
        for cell in vector:
            values = np.asarray(cell)
            require(values.ndim == 1 and values.size and np.issubdtype(values.dtype, np.number) and
                    not np.issubdtype(values.dtype, np.complexfloating) and
                    np.isfinite(values).all() and set(values.tolist()) <= allowed, "Invalid MAT vector/code")
    result = {}
    for index in range(count):
        item = {task: {field: [int(x) for x in raw[f"{task}_{field}"][index]]
                       for field in ("feedback", "action", "expect")} for task in TASKS}
        require(all(len({len(v) for v in item[task].values()}) == 1 for task in TASKS), "MAT vector lengths differ")
        result[f"mat-{index + 1:02d}"] = item
    return result


def recalculate(root, config, manifest):
    source = root / config["source_root"]
    blocks = {}
    for item in manifest["files"]:
        if item["path"].endswith("_events.tsv"):
            label = item["path"].split("/")[0]
            require(label not in blocks, "Duplicate event label")
            blocks[label] = partition(event_rows(source / item["path"]))
    require(blocks, "No event files")
    mats = mat_rows(source / "code/Data/RevEx1_behavioralData.mat")
    sequences = {label: {task: [b["feedback_flag"] for b in entry["blocks"] if b["task"] == task]
                         for task in TASKS} for label, entry in blocks.items()}
    require(all(seq for item in sequences.values() for seq in item.values()), "Empty feedback sequence")
    event_candidates = {label: [] for label in sorted(blocks)}
    mat_candidates = {label: [] for label in sorted(mats)}
    edges = []
    for label in event_candidates:
        for mat_label in mat_candidates:
            agrees = {task + "_equal": sequences[label][task] == mats[mat_label][task]["feedback"] for task in TASKS}
            both = all(agrees.values())
            edges.append({"event_label": label, "mat_row": mat_label, **agrees, "both_equal": both})
            if both:
                event_candidates[label].append(mat_label)
                mat_candidates[mat_label].append(label)
    accepted = [{"event_label": label, "mat_row": candidates[0]} for label, candidates in event_candidates.items()
                if len(candidates) == 1 and len(mat_candidates[candidates[0]]) == 1]
    numeric, counters = [], Counter()
    for pair in accepted:
        for block in blocks[pair["event_label"]]["blocks"]:
            task, index = block["task"], block["feedback_ordinal"] - 1
            action, expectation = (mats[pair["mat_row"]][task][key][index] for key in ("action", "expect"))
            strict = block["strict_comparable"]
            row = {**pair, "task": task, "feedback_ordinal": index + 1,
                   "feedback_event_row": block["feedback_event_row"], "strict_comparable": strict,
                   "event_choice_side_code": block["choice_side_code"], "mat_action_code": action,
                   "event_expectation_plus3": block["expectation_mat_code"], "mat_expect_code": expectation,
                   "action_numeric_agreement": block["choice_side_code"] == action if strict else None,
                   "expectation_numeric_agreement": block["expectation_mat_code"] == expectation if strict else None}
            numeric.append(row)
            counters["feedback_anchored_pairs"] += 1
            counters["strict_comparable" if strict else "not_strict_comparable"] += 1
            if strict:
                for kind in ("action", "expectation"):
                    counters[f"{kind}_numeric_{'agreements' if row[kind + '_numeric_agreement'] else 'disagreements'}"] += 1
    all_blocks = [b for entry in blocks.values() for b in entry["blocks"]]
    report = {"audit_id": config["audit_id"], "scope": config["scope"], "event_files": len(blocks),
              "mat_rows": len(mats), "matrix_pairs": len(edges), "exact_feedback_candidate_pairs": len(accepted),
              "unmatched_event_labels": [label for label, candidates in event_candidates.items() if not candidates],
              "unmatched_mat_rows": [label for label, candidates in mat_candidates.items() if not candidates],
              "ambiguous_event_labels": [label for label, candidates in event_candidates.items() if len(candidates) > 1],
              "ambiguous_mat_rows": [label for label, candidates in mat_candidates.items() if len(candidates) > 1],
              "event_feedback_lengths": {label: {task: len(sequence[task]) for task in TASKS}
                                         for label, sequence in sequences.items()},
              "mat_feedback_lengths": {label: {task: len(item[task]["feedback"]) for task in TASKS}
                                       for label, item in mats.items()},
              "event_feedback_blocks": len(all_blocks),
              "block_issue_counts": dict(sorted(Counter(issue for block in all_blocks for issue in block["issues"]).items())),
              "corroboration_counts": dict(counters), "event_candidates": event_candidates,
              "mat_candidates": mat_candidates, "readiness": READINESS}
    return report, edges, accepted, blocks, numeric


def check_csv(path, expected, columns):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == columns, f"CSV schema differs: {Path(path).name}")
        observed = list(reader)
    require(len(observed) == len(expected), f"CSV row count differs: {Path(path).name}")
    for index, (row, wanted) in enumerate(zip(observed, expected)):
        encoded = {key: "" if wanted[key] is None else str(wanted[key]) for key in columns}
        require(row == encoded, f"CSV semantic replay differs: {Path(path).name} row {index}")


def verify(output):
    output = Path(output).resolve()
    names = {"metadata.json", "registration.json", "linkage-report.json", "feedback-blocks.json",
             "alignment-matrix.csv", "record-links.csv", "corroboration.csv"}
    artifacts = read_json(output / "artifact_hashes.json")
    require(set(artifacts) == names, "Primary artifact coverage differs")
    for name, expected in artifacts.items():
        require(sha(safe(output, name)) == expected, f"Artifact hash differs: {name}")
    meta, reg = read_json(output / "metadata.json"), read_json(output / "registration.json")
    require(set(meta) == {"started_at_utc", "status", "registration_sha256", "source_root", "config_path",
                          "runtime", "completed_at_utc"}, "Metadata fields differ")
    require(set(reg) == {"config", "config_sha256", "source_hashes", "registered_at_utc", "git_commit", "notice"},
            "Registration fields differ")
    require(meta["status"] == "completed", "Execution not completed")
    stamps = [datetime.fromisoformat(reg["registered_at_utc"]), datetime.fromisoformat(meta["started_at_utc"]),
              datetime.fromisoformat(meta["completed_at_utc"])]
    require(all(stamp.tzinfo is not None for stamp in stamps) and stamps[0] <= stamps[1] <= stamps[2],
            "Registration/execution chronology differs")
    require(set(meta["runtime"]) == {"python", "numpy", "scipy"} and
            all(isinstance(value, str) and value for value in meta["runtime"].values()), "Runtime provenance missing")
    require(reg["notice"] == "exploratory anonymous-record linkage audit; source schemas and author code already inspected; no blinded hypothesis preregistration",
            "Exploratory scope notice differs")
    require(meta["registration_sha256"] == sha(output / "registration.json"), "Registration copy hash differs")
    root = (output / meta["source_root"]).resolve()
    config_path = safe(root, meta["config_path"])
    config = read_json(config_path)
    require(exact(config, CONFIG) and exact(reg["config"], CONFIG), "Scope/config differs")
    require(reg["config_sha256"] == sha(config_path), "Config hash differs")
    require(re.fullmatch(r"[a-f0-9]{40}", reg["git_commit"]), "Invalid source commit")
    frozen = reg["source_hashes"]
    required_frozen = set(BASE_FROZEN)
    source_manifest = None
    raw_count = 0
    for manifest_name in (config["source_manifest"], config["evidence_manifest"]):
        manifest_path = safe(root, manifest_name)
        manifest = read_json(manifest_path)
        required_frozen.add(manifest_name)
        paths = [item["path"] for item in manifest["files"]]
        require(len(paths) == len(set(paths)), "Duplicate manifest paths")
        for item in manifest["files"]:
            path = safe(manifest_path.parent, item["path"])
            name = path.relative_to(root).as_posix()
            required_frozen.add(name)
            require(path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"], "Manifest/source bytes differ")
            raw_count += 1
        if manifest_name == config["source_manifest"]:
            source_manifest = manifest
    require(set(frozen) == required_frozen, "Frozen source coverage differs")
    for name, expected in frozen.items():
        require(sha(safe(root, name)) == expected, f"Frozen source hash differs: {name}")
    description = read_json(root / config["source_root"] / "dataset_description.json")
    require(description["License"] == "CC0" and description["DatasetDOI"] ==
            "doi:10.18112/openneuro.ds004295.v1.0.0", "Dataset identity/license differs")
    report, edges, accepted, blocks, numeric = recalculate(root, config, source_manifest)
    require(exact(read_json(output / "linkage-report.json"), report), "Linkage report semantic replay differs")
    require(exact(read_json(output / "feedback-blocks.json"), blocks), "Global feedback block/tail replay differs")
    check_csv(output / "alignment-matrix.csv", edges, ["event_label", "mat_row", "REW_equal", "PUN_equal", "both_equal"])
    check_csv(output / "record-links.csv", accepted, ["event_label", "mat_row"])
    check_csv(output / "corroboration.csv", numeric,
              ["event_label", "mat_row", "task", "feedback_ordinal", "feedback_event_row", "strict_comparable",
               "event_choice_side_code", "mat_action_code", "event_expectation_plus3", "mat_expect_code",
               "action_numeric_agreement", "expectation_numeric_agreement"])
    return {"status": "passed", "scope": "independent_exact_record_linkage_replay_no_learning_analysis",
            "source_files_verified": raw_count, "frozen_files_verified": len(frozen),
            "primary_artifacts_verified": len(names), "matrix_pairs_verified": len(edges),
            "degree_one_pairs_verified": len(accepted), "event_files_verified": len(blocks),
            "event_rows_partitioned": sum(entry["event_count"] for entry in blocks.values()),
            "feedback_blocks_verified": sum(len(entry["blocks"]) for entry in blocks.values()),
            "tail_event_rows_preserved": sum(len(entry["tail_event_rows"]) for entry in blocks.values()),
            "numeric_diagnostic_rows_verified": len(numeric), "errors": [],
            "runtime": {"numpy": np.__version__, "scipy": scipy.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.output)
    except Exception as error:
        result = {"status": "failed", "scope": "independent_exact_record_linkage_replay_no_learning_analysis",
                  "errors": [f"{type(error).__name__}: {error}"]}
    result["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    with (args.output / "verification.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "scope", "errors")}))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
