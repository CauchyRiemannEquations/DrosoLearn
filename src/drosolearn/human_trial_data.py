"""Source and event-structure audit only; no human performance or model fit."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess


FIELDS = ("onset", "duration", "sample", "value")
START = {"S100": "REW", "S200": "PUN"}
END = {"S101": "REW", "S201": "PUN"}
SELECTION = {"S51": ("REW", "left"), "S52": ("REW", "right"),
             "S71": ("PUN", "left"), "S72": ("PUN", "right")}
EXPECTATION = {**{f"S{x}": ("REW", x - 53) for x in range(53, 57)},
               **{f"S{x}": ("PUN", x - 73) for x in range(73, 77)}}
FEEDBACK = {"S60": ("REW", 0), "S61": ("REW", 1),
            "S80": ("PUN", 0), "S81": ("PUN", 1)}
CHOICE_FIELDS = ("participant", "task", "observed_choice_index", "choice_event_row", "choice_onset",
                 "choice_side", "expectation_level", "expectation_event_row", "feedback_code",
                 "positive_feedback_flag", "feedback_event_row", "feedback_onset", "complete")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def normalize_marker(value):
    match = re.fullmatch(r"S[ _]?([0-9]{2,3})", str(value).strip())
    return f"S{int(match[1])}" if match else None


def read_events(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Exact four-column source event schema required")
        rows = []
        for index, row in enumerate(reader, 2):
            if set(row) != set(FIELDS) or any(value is None for value in row.values()):
                raise ValueError("Malformed source event row")
            rows.append({**row, "row": index})
    if not rows:
        raise ValueError("Empty event source")
    return rows


def _number(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not an event measurement")
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError("Finite numeric event measurement required") from error
    if not math.isfinite(number):
        raise ValueError("Finite numeric event measurement required")
    return number


def parse_events(events):
    """Retain incomplete choices; never infer targets, RT, or absent trials."""
    if not events:
        raise ValueError("Events required")
    choices, issues, counts = [], [], Counter()
    task_counts, start_counts, end_counts = Counter(), Counter(), Counter()
    active, pending, paused = None, None, False
    previous_time = previous_sample = -math.inf
    seen_rows = set()

    def finish(reason=None):
        nonlocal pending
        if pending is None:
            return
        if reason:
            issues.append(f"row {pending['choice_event_row']}: {reason}")
            pending["_valid"] = False
        pending["complete"] = bool(pending.pop("_valid") and
                                   pending["expectation_event_row"] is not None and
                                   pending["feedback_event_row"] is not None)
        choices.append(pending)
        pending = None

    for index, event in enumerate(events, 2):
        if not set(FIELDS).issubset(event):
            raise ValueError("Missing event fields")
        onset, duration = (_number(event[key]) for key in ("onset", "duration"))
        sample_text = str(event["sample"])
        if not re.fullmatch(r"[0-9]+", sample_text) or duration < 0:
            raise ValueError("Nonnegative integer sample and duration required")
        sample = int(sample_text)
        row = event.get("row", index)
        if type(row) is not int or row < 2 or row in seen_rows:
            raise ValueError("Distinct original source row numbers required")
        seen_rows.add(row)
        if onset < previous_time or sample < previous_sample:
            issues.append(f"row {row}: decreasing event time/sample; source order retained")
            if pending:
                pending["_valid"] = False
        previous_time, previous_sample = onset, sample
        code = normalize_marker(event["value"])
        counts[code or f"UNRECOGNIZED:{event['value']}"] += 1
        if code in START:
            finish("new task before feedback")
            if active is not None:
                issues.append(f"row {row}: task start while another task active")
            active, paused = START[code], False
            start_counts[active] += 1
        elif code in END:
            finish("task end before complete feedback")
            if active != END[code]:
                issues.append(f"row {row}: unmatched task end")
            end_counts[END[code]] += 1
            active, paused = None, False
        elif code in ("S150", "S151"):
            finish("pause boundary inside choice sequence")
            if active is None or paused == (code == "S150"):
                issues.append(f"row {row}: unmatched pause boundary")
            paused = code == "S150"
        elif code in SELECTION:
            finish("next choice before complete feedback")
            task, side = SELECTION[code]
            valid = active == task and not paused
            if not valid:
                issues.append(f"row {row}: choice outside matching active task")
            task_counts[task] += 1
            pending = {"task": task, "observed_choice_index": task_counts[task],
                       "choice_event_row": row, "choice_onset": onset, "choice_side": side,
                       "expectation_level": None, "expectation_event_row": None,
                       "feedback_code": None, "positive_feedback_flag": None,
                       "feedback_event_row": None, "feedback_onset": None, "_valid": valid}
        elif code in EXPECTATION:
            task, level = EXPECTATION[code]
            if pending is None or pending["task"] != task or active != task or paused:
                issues.append(f"row {row}: orphan/context-mismatched expectation")
                if pending:
                    pending["_valid"] = False
            elif pending["expectation_event_row"] is not None:
                issues.append(f"row {row}: multiple expectation events")
                pending["_valid"] = False
            else:
                pending.update(expectation_level=level, expectation_event_row=row)
        elif code in FEEDBACK:
            task, flag = FEEDBACK[code]
            if pending is None or pending["task"] != task or active != task or paused:
                issues.append(f"row {row}: orphan/context-mismatched feedback")
                if pending:
                    pending["_valid"] = False
            else:
                pending.update(feedback_code=code, positive_feedback_flag=flag,
                               feedback_event_row=row, feedback_onset=onset)
                finish(None if pending["expectation_event_row"] is not None else "missing expectation event")
        else:
            issues.append(f"row {row}: unknown event marker {event['value']}")
            if pending:
                pending["_valid"] = False
    finish("file ended before complete feedback")
    if active is not None:
        issues.append("file ended with active task")
    if paused:
        issues.append("file ended during pause")
    return {"events": len(events), "choices": choices, "issues": issues,
            "marker_counts": dict(sorted(counts.items())), "task_counts": dict(task_counts),
            "task_start_counts": dict(start_counts), "task_end_counts": dict(end_counts)}


def study_hashes(root, config):
    root = Path(root)
    selected = ["src/drosolearn/human_trial_data.py", "tests/test_human_trial_data.py",
                "scripts/verify_human_trial_data.py", "tests/test_human_trial_data_verify.py",
                "docs/human-trial-data-protocol.md", "requirements-lock.txt", "pyproject.toml",
                config["source_manifest"]]
    manifest = json.loads((root / config["source_manifest"]).read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (root / config["source_root"] / item["path"]).resolve()
        if not path.is_relative_to((root / config["source_root"]).resolve()):
            raise ValueError("Source manifest path escapes the selected dataset")
        selected.append(path.relative_to(root.resolve()).as_posix())
    return {name: digest(root / name) for name in sorted(selected)}


def validate_config(config):
    fixed = {"audit_id": "human_trial_data_readiness_v1", "scope": "source_and_structure_only",
             "adapter": "openneuro_ds004295_events_v1", "performance_analysis": False,
             "source_root": "data/human/stolz_2022_openneuro",
             "source_manifest": "data/human/stolz_2022_openneuro/source-manifest.json"}
    if any(config.get(key) != value or type(config.get(key)) is not type(value) for key, value in fixed.items()):
        raise ValueError("Fixed structural-audit scope and source required")


def register(root, config_path, destination):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    record = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
              "config_sha256": digest(config_path), "source_hashes": study_hashes(root, config),
              "git_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
              "notice": "exploratory data-availability audit; schemas were inspected during source selection; no blinded hypothesis preregistration"}
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, record)
    return record


def audit(root, config):
    """Produce structural counts and ordered event associations, never accuracy."""
    root = Path(root)
    source = root / config["source_root"]
    manifest = json.loads((root / config["source_manifest"]).read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (source / item["path"]).resolve()
        if not path.is_relative_to(source.resolve()) or path.stat().st_size != item["size_bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Source bytes changed: {item['path']}")
    description = json.loads((source / "dataset_description.json").read_text())
    if description["License"] != "CC0" or description["DatasetDOI"] != "doi:10.18112/openneuro.ds004295.v1.0.0":
        raise ValueError("Source license or identity differs")
    codebook = json.loads((source / "task-task_events.json").read_text())
    declared = {normalize_marker(key) for key in codebook["value"]["Levels"]}
    expected = set(START) | set(END) | set(SELECTION) | set(EXPECTATION) | set(FEEDBACK) | {"S150", "S151"}
    if declared != expected:
        raise ValueError("Source marker codebook differs")
    files, observations, signatures = [], [], defaultdict(list)
    for item in manifest["files"]:
        if not item["path"].endswith("_events.tsv"):
            continue
        participant = item["path"].split("/")[0]
        events = read_events(source / item["path"])
        parsed = parse_events(events)
        observations += [{"participant": participant, **row} for row in parsed["choices"]]
        signature = hashlib.sha256(json.dumps([normalize_marker(event["value"]) for event in events]).encode()).hexdigest()
        signatures[signature].append(participant)
        files.append({"path": item["path"], "participant": participant,
                      **{key: value for key, value in parsed.items() if key != "choices"},
                      "observed_choices": len(parsed["choices"]),
                      "complete_triplets": sum(row["complete"] for row in parsed["choices"]),
                      "choice_feedback_pairs": sum(row["feedback_event_row"] is not None for row in parsed["choices"])})
    if len({row["participant"] for row in files}) != len(files) or not files:
        raise ValueError("Unique nonempty participant event-file coverage required")
    report = {"audit_id": config["audit_id"], "scope": config["scope"], "source_integrity_passed": True,
              "source_files_verified": len(manifest["files"]), "event_files": len(files),
              "participant_file_labels": [row["participant"] for row in files],
              "event_rows": sum(row["events"] for row in files), "observed_choices": len(observations),
              "complete_triplets": sum(row["complete"] for row in observations),
              "issues_count": sum(len(row["issues"]) for row in files), "files": files,
              "duplicate_marker_sequence_groups": [group for group in signatures.values() if len(group) > 1],
              "readiness": {"observed_choice_feedback_order": "candidate_with_event_coverage_limits",
                            "reward_vs_punishment": "separate_contexts_required",
                            "original_trial_index_or_omitted_trials": "not_in_event_files",
                            "reversal_boundaries_and_correct_option": "no per-row state in these files; paper schedules exist but practice/omission alignment and randomized shape-side mapping require verification",
                            "human_confidence_as_choice_probability": "not_established",
                            "reaction_time_bids": "not_in_event_columns; timing_between_events_is_not_RT",
                            "reaction_time_mat": "present_for_23_unlinked_rows; no_verified_participant_join",
                            "numerosity_or_mathematical_rules": "not_this_task",
                            "human_model_similarity": "not_tested"}}
    return report, observations


def run(root, config_path, registration_path, output):
    root, config_path, registration_path, output = map(lambda path: Path(path).resolve(), (root, config_path, registration_path, output))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    validate_config(config)
    if config != registration["config"] or digest(config_path) != registration["config_sha256"] or study_hashes(root, config) != registration["source_hashes"]:
        raise ValueError("Source, protocol or config changed after audit freeze")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "config": config, "registration_sha256": digest(registration_path),
                "source_root": Path(os.path.relpath(root, output)).as_posix(), "config_path": config_path.relative_to(root).as_posix(),
                "scope": "source/schema/order audit; no performance curve, model fit, or cross-species inference"}
    write_json(output / "registration.json", registration)
    write_json(output / "metadata.json", metadata)
    try:
        report, observations = audit(root, config)
        write_json(output / "structure-audit.json", report)
        with (output / "observed_choices.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CHOICE_FIELDS)
            writer.writeheader()
            writer.writerows(observations)
        metadata.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        metadata.update(status="failed", failure=repr(error), completed_at_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "register":
        record = register(args.root, args.config, args.out)
        print(json.dumps({"registered_at_utc": record["registered_at_utc"], "frozen_files": len(record["source_hashes"])}))
    else:
        if args.registration is None:
            parser.error("--registration required")
        report = run(args.root, args.config, args.registration, args.out)
        print(json.dumps({key: report[key] for key in ("source_integrity_passed", "event_files", "event_rows", "observed_choices", "complete_triplets", "issues_count")}))


if __name__ == "__main__":
    main()
