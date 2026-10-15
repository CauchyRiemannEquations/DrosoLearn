"""Independent source/event replay; never estimates human performance."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import re


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_json(actual, expected):
    """Preserve JSON types: True is not an integer count or false scope flag."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_json(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same_json(a, b) for a, b in zip(actual, expected))
    return actual == expected


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def safe_file(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Relative file path required")
    destination = (Path(root) / relative).resolve()
    require(destination.is_relative_to(Path(root).resolve()), "File escapes expected root")
    require(destination.is_file(), f"Missing file: {relative}")
    return destination


def marker(value):
    match = re.fullmatch(r"S(?:_| )?([0-9]{2,3})", value.strip())
    return f"S{int(match.group(1))}" if match else None


def finite(value):
    require(not isinstance(value, bool), "Boolean measurement")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise VerificationError("Invalid measurement") from error
    require(math.isfinite(number), "Nonfinite measurement")
    return number


def original_events(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        require(next(reader, None) == ["onset", "duration", "sample", "value"], "Wrong event columns")
        rows = []
        for line, values in enumerate(reader, 2):
            require(len(values) == 4, "Malformed source TSV row")
            onset, duration, sample, value = values
            onset, duration = finite(onset), finite(duration)
            require(duration >= 0 and re.fullmatch(r"[0-9]+", sample), "Invalid duration/sample")
            rows.append({"row": line, "onset": onset, "sample": int(sample), "value": value})
    require(bool(rows), "Empty source TSV")
    return rows


def replay(events):
    """Independent finite-state replay from original TSV rows and marker numbers."""
    choices, issues, counts = [], [], Counter()
    task_counts, starts, ends = Counter(), Counter(), Counter()
    current = None
    active, paused = None, False
    previous = (-math.inf, -math.inf)

    def close(reason=None):
        nonlocal current
        if current is None:
            return
        valid = current.pop("valid")
        if reason:
            issues.append(f"row {current['choice_event_row']}: {reason}")
            valid = False
        current["complete"] = bool(valid and current["expectation_event_row"] is not None and
                                   current["feedback_event_row"] is not None)
        choices.append(current)
        current = None

    for event in events:
        line, time, sample = event["row"], event["onset"], event["sample"]
        if time < previous[0] or sample < previous[1]:
            issues.append(f"row {line}: decreasing event time/sample; source order retained")
            if current is not None:
                current["valid"] = False
        previous = (time, sample)
        code = marker(event["value"])
        counts[code or f"UNRECOGNIZED:{event['value']}"] += 1
        number = int(code[1:]) if code else -1
        if number in (100, 200):
            close("new task before feedback")
            if active is not None:
                issues.append(f"row {line}: task start while another task active")
            active = "REW" if number == 100 else "PUN"
            paused = False
            starts[active] += 1
        elif number in (101, 201):
            close("task end before complete feedback")
            task = "REW" if number == 101 else "PUN"
            if active != task:
                issues.append(f"row {line}: unmatched task end")
            ends[task] += 1
            active, paused = None, False
        elif number in (150, 151):
            close("pause boundary inside choice sequence")
            if active is None or paused == (number == 150):
                issues.append(f"row {line}: unmatched pause boundary")
            paused = number == 150
        elif number in (51, 52, 71, 72):
            close("next choice before complete feedback")
            task = "REW" if number < 70 else "PUN"
            task_counts[task] += 1
            valid = active == task and not paused
            if not valid:
                issues.append(f"row {line}: choice outside matching active task")
            current = {"task": task, "observed_choice_index": task_counts[task],
                       "choice_event_row": line, "choice_onset": time,
                       "choice_side": "left" if number in (51, 71) else "right",
                       "expectation_level": None, "expectation_event_row": None,
                       "feedback_code": None, "positive_feedback_flag": None,
                       "feedback_event_row": None, "feedback_onset": None, "valid": valid}
        elif number in (*range(53, 57), *range(73, 77)):
            task = "REW" if number < 70 else "PUN"
            if current is None or current["task"] != task or active != task or paused:
                issues.append(f"row {line}: orphan/context-mismatched expectation")
                if current is not None:
                    current["valid"] = False
            elif current["expectation_event_row"] is not None:
                issues.append(f"row {line}: multiple expectation events")
                current["valid"] = False
            else:
                current["expectation_event_row"] = line
                current["expectation_level"] = number - (53 if task == "REW" else 73)
        elif number in (60, 61, 80, 81):
            task = "REW" if number < 70 else "PUN"
            if current is None or current["task"] != task or active != task or paused:
                issues.append(f"row {line}: orphan/context-mismatched feedback")
                if current is not None:
                    current["valid"] = False
            else:
                current.update(feedback_code=code, positive_feedback_flag=number % 10,
                               feedback_event_row=line, feedback_onset=time)
                close("missing expectation event" if current["expectation_event_row"] is None else None)
        else:
            issues.append(f"row {line}: unknown event marker {event['value']}")
            if current is not None:
                current["valid"] = False
    close("file ended before complete feedback")
    if active is not None:
        issues.append("file ended with active task")
    if paused:
        issues.append("file ended during pause")
    return {"events": len(events), "choices": choices, "issues": issues,
            "marker_counts": dict(sorted(counts.items())), "task_counts": dict(task_counts),
            "task_start_counts": dict(starts), "task_end_counts": dict(ends)}


CSV_FIELDS = ["participant", "task", "observed_choice_index", "choice_event_row", "choice_onset",
              "choice_side", "expectation_level", "expectation_event_row", "feedback_code",
              "positive_feedback_flag", "feedback_event_row", "feedback_onset", "complete"]


def compare_choices(path, expected):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == CSV_FIELDS, "Unexpected choice columns; no inferred target/probability/RT allowed")
        observed = list(reader)
    require(len(observed) == len(expected), "Lost or fabricated choice rows")
    for index, (actual, wanted) in enumerate(zip(observed, expected)):
        require(set(actual) == set(CSV_FIELDS), "Malformed choice CSV row")
        for key in CSV_FIELDS:
            value = wanted[key]
            encoded = "" if value is None else str(value)
            if key in ("choice_onset", "feedback_onset") and value is not None:
                require(finite(actual[key]) == value, f"Choice row {index}: changed {key}")
            else:
                require(actual[key] == encoded, f"Choice row {index}: changed {key}")
    return len(observed)


def verify(output):
    output = Path(output).resolve()
    artifact_hashes = read_json(output / "artifact_hashes.json")
    required = {"registration.json", "metadata.json", "structure-audit.json", "observed_choices.csv"}
    require(set(artifact_hashes) == required, "Unexpected primary artifact set")
    for name, expected in artifact_hashes.items():
        require(sha(safe_file(output, name)) == expected, f"Primary artifact hash differs: {name}")
    registration = read_json(output / "registration.json")
    metadata = read_json(output / "metadata.json")
    report = read_json(output / "structure-audit.json")
    require(metadata["status"] == "completed", "Audit execution incomplete")
    require(metadata["registration_sha256"] == sha(output / "registration.json"), "Registration copy mismatch")
    root = (output / metadata["source_root"]).resolve()
    config_path = safe_file(root, metadata["config_path"])
    config = read_json(config_path)
    require(same_json(config, registration["config"]) and same_json(config, metadata["config"]), "Config copies disagree")
    require(sha(config_path) == registration["config_sha256"], "Frozen config differs")
    fixed = {"audit_id": "human_trial_data_readiness_v1", "scope": "source_and_structure_only",
             "adapter": "openneuro_ds004295_events_v1", "performance_analysis": False,
             "source_root": "data/human/stolz_2022_openneuro",
             "source_manifest": "data/human/stolz_2022_openneuro/source-manifest.json"}
    require(same_json(config, fixed), "Unregistered analysis scope or adapter")
    frozen = registration["source_hashes"]
    require(bool(frozen), "No frozen sources")
    for name, expected in frozen.items():
        require(sha(safe_file(root, name)) == expected, f"Frozen source differs: {name}")
    source = root / config["source_root"]
    manifest = read_json(root / config["source_manifest"])
    require(config["source_manifest"] in frozen, "Source manifest not frozen")
    paths = [item["path"] for item in manifest["files"]]
    require(len(paths) == len(set(paths)), "Duplicate source manifest paths")
    for item in manifest["files"]:
        path = safe_file(source, item["path"])
        relative = f"{config['source_root']}/{item['path']}"
        require(relative in frozen and frozen[relative] == item["sha256"], "Unfrozen manifest file")
        require(path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"], "Raw source bytes changed")
    frozen_paths = {"src/drosolearn/human_trial_data.py", "tests/test_human_trial_data.py",
                    "scripts/verify_human_trial_data.py", "tests/test_human_trial_data_verify.py",
                    "docs/human-trial-data-protocol.md", "requirements-lock.txt", "pyproject.toml",
                    config["source_manifest"]}
    frozen_paths.update(f"{config['source_root']}/{item['path']}" for item in manifest["files"])
    require(set(frozen) == frozen_paths, "Frozen source file coverage differs")
    description = read_json(source / "dataset_description.json")
    require(description["License"] == "CC0" and description["DatasetDOI"] ==
            "doi:10.18112/openneuro.ds004295.v1.0.0", "Source identity/license mismatch")
    declared = read_json(source / "task-task_events.json")["value"]["Levels"]
    code_set = {marker(key) for key in declared}
    expected_codes = {f"S{i}" for i in (100, 101, 200, 201, 150, 151, 51, 52, 71, 72,
                                      53, 54, 55, 56, 73, 74, 75, 76, 60, 61, 80, 81)}
    require(code_set == expected_codes, "Changed event codebook")
    file_reports, expected_choices, signatures = [], [], defaultdict(list)
    for item in manifest["files"]:
        if not item["path"].endswith("_events.tsv"):
            continue
        participant = item["path"].split("/")[0]
        events = original_events(source / item["path"])
        parsed = replay(events)
        choices = parsed.pop("choices")
        expected_choices.extend({"participant": participant, **row} for row in choices)
        file_reports.append({"path": item["path"], "participant": participant, **parsed,
                             "observed_choices": len(choices),
                             "complete_triplets": sum(row["complete"] for row in choices),
                             "choice_feedback_pairs": sum(row["feedback_event_row"] is not None for row in choices)})
        sequence = [marker(row["value"]) for row in events]
        signatures[hashlib.sha256(json.dumps(sequence).encode()).hexdigest()].append(participant)
    labels = [row["participant"] for row in file_reports]
    require(labels and len(labels) == len(set(labels)), "Nonunique or empty participant-file labels")
    require(same_json(report["files"], file_reports), "Per-source structure/issue report disagrees with independent replay")
    expected_summary = {"audit_id": config["audit_id"], "scope": config["scope"], "source_integrity_passed": True,
                        "source_files_verified": len(paths), "event_files": len(labels),
                        "participant_file_labels": labels, "event_rows": sum(row["events"] for row in file_reports),
                        "observed_choices": len(expected_choices),
                        "complete_triplets": sum(row["complete"] for row in expected_choices),
                        "issues_count": sum(len(row["issues"]) for row in file_reports),
                        "duplicate_marker_sequence_groups": [group for group in signatures.values() if len(group) > 1]}
    for key, value in expected_summary.items():
        require(same_json(report[key], value), f"Summary count/scope differs: {key}")
    require(set(report) == set(expected_summary) | {"files", "readiness"}, "Unexpected report fields or performance estimate")
    require(report["complete_triplets"] <= report["observed_choices"] <= report["event_rows"], "Invalid denominator range")
    readiness = report["readiness"]
    require(readiness["human_model_similarity"] == "not_tested", "Unsupported human/model similarity claim")
    require(readiness["human_confidence_as_choice_probability"] == "not_established", "Ordinal rating relabeled as probability")
    require(readiness["reversal_boundaries_and_correct_option"] ==
            "no per-row state in these files; paper schedules exist but practice/omission alignment and randomized shape-side mapping require verification",
            "Unsupported reversal/target recovery")
    require(readiness["reward_vs_punishment"] == "separate_contexts_required", "Reward/punishment conflation")
    require(readiness["reaction_time_bids"] == "not_in_event_columns; timing_between_events_is_not_RT", "Unsupported reaction time")
    choices_verified = compare_choices(output / "observed_choices.csv", expected_choices)
    return {"status": "passed", "scope": "independent_source_structure_replay_no_performance_analysis",
            "source_files_verified": len(paths), "frozen_files_verified": len(frozen),
            "primary_artifact_hashes_verified": len(artifact_hashes), "event_files_verified": len(labels),
            "event_rows_verified": report["event_rows"], "observed_choice_rows_verified": choices_verified,
            "complete_triplets_verified": report["complete_triplets"],
            "incomplete_choices_preserved": choices_verified - report["complete_triplets"],
            "issues_preserved": report["issues_count"], "errors": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.output)
    except Exception as error:
        result = {"status": "failed", "scope": "independent_source_structure_replay_no_performance_analysis",
                  "errors": [f"{type(error).__name__}: {error}"]}
    result["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    with (args.output / "verification.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "scope", "errors")}))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
