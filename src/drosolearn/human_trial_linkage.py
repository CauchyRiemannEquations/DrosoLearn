"""Exact anonymous-record linkage audit; no learning or cross-species estimates."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import csv
import json
import math
import os
import platform
import subprocess

from .human_trial_data import (START, END, SELECTION, EXPECTATION, FEEDBACK,
                               digest, normalize_marker, read_events, write_json)

TASKS = ("REW", "PUN")
PAUSE = {"S150", "S151"}
KNOWN = set(START) | set(END) | set(SELECTION) | set(EXPECTATION) | set(FEEDBACK) | PAUSE


def _sequences(records):
    if not isinstance(records, dict) or not records:
        raise ValueError("Nonempty record dictionary required")
    for label, item in records.items():
        if not isinstance(label, str) or not label or not isinstance(item, dict) or set(item) != set(TASKS):
            raise ValueError("Named records with exactly REW/PUN sequences required")
        for sequence in item.values():
            if not isinstance(sequence, list) or not sequence or any(type(x) is not int or x not in (0, 1) for x in sequence):
                raise ValueError("Nonempty strict binary integer sequence required")


def link_sequences(events, mats):
    """Build complete graph before accepting degree-one pairs on both sides."""
    _sequences(events)
    _sequences(mats)
    edges = []
    event_candidates = {label: [] for label in sorted(events)}
    mat_candidates = {label: [] for label in sorted(mats)}
    for event_label in sorted(events):
        for mat_row in sorted(mats):
            equal = {task + "_equal": events[event_label][task] == mats[mat_row][task] for task in TASKS}
            both = all(equal.values())
            edges.append({"event_label": event_label, "mat_row": mat_row, **equal, "both_equal": both})
            if both:
                event_candidates[event_label].append(mat_row)
                mat_candidates[mat_row].append(event_label)
    accepted = [{"event_label": label, "mat_row": candidates[0]}
                for label, candidates in event_candidates.items()
                if len(candidates) == 1 and len(mat_candidates[candidates[0]]) == 1]
    return {"edges": edges, "accepted": accepted,
            "event_candidates": event_candidates, "mat_candidates": mat_candidates}


def feedback_blocks(events):
    """Partition original events at every feedback; do not reconstruct trials."""
    if not events:
        raise ValueError("Nonempty event stream required")
    prepared = []
    active, paused = None, False
    previous_time = previous_sample = -math.inf
    previous_row = 1
    for index, event in enumerate(events, 2):
        row = event.get("row", index)
        if type(row) is not int or row <= previous_row:
            raise ValueError("Increasing distinct original source row numbers required")
        previous_row = row
        try:
            if any(isinstance(event[key], bool) for key in ("onset", "duration", "sample")):
                raise ValueError("Boolean event measurement")
            onset, duration = float(event["onset"]), float(event["duration"])
            sample = int(event["sample"])
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError("Numeric finite event measurement required") from error
        if not (math.isfinite(onset) and math.isfinite(duration)) or duration < 0 or sample < 0 or str(sample) != str(event["sample"]):
            raise ValueError("Finite measurements and nonnegative integer sample required")
        code = normalize_marker(event.get("value"))
        decreasing = onset < previous_time or sample < previous_sample
        previous_time, previous_sample = onset, sample
        if code in START:
            active, paused = START[code], False
        elif code in END:
            active, paused = None, False
        elif code in PAUSE:
            paused = code == "S150"
        prepared.append({"row": row, "code": code, "active": active, "paused": paused,
                         "time_reversed": decreasing, "onset": onset})
    blocks, window = [], []
    ordinal = Counter()
    previous_feedback_row = None
    for event in prepared:
        code = event["code"]
        if code not in FEEDBACK:
            window.append(event)
            continue
        task, flag = FEEDBACK[code]
        ordinal[task] += 1
        selections = [e for e in window if e["code"] in SELECTION and SELECTION[e["code"]][0] == task]
        expectations = [e for e in window if e["code"] in EXPECTATION and EXPECTATION[e["code"]][0] == task]
        issues = []
        if len(selections) != 1:
            issues.append("choice_count_not_one")
        if len(expectations) != 1:
            issues.append("expectation_count_not_one")
        if any(e["time_reversed"] for e in window + [event]):
            issues.append("decreasing_event_time_or_sample")
        if event["active"] != task or event["paused"]:
            issues.append("feedback_outside_matching_active_task")
        if len(selections) == 1:
            choice = selections[0]
            if choice["active"] != task or choice["paused"]:
                issues.append("choice_outside_matching_active_task")
            after = [e for e in window if e["row"] > choice["row"]]
            if any(e["code"] in set(START) | set(END) | PAUSE for e in after):
                issues.append("boundary_after_choice")
            if any(e["code"] not in KNOWN for e in after):
                issues.append("unknown_marker_after_choice")
            if any((e["code"] in SELECTION and SELECTION[e["code"]][0] != task) or
                   (e["code"] in EXPECTATION and EXPECTATION[e["code"]][0] != task) for e in window):
                issues.append("opposite_task_event_in_feedback_segment")
            if len(expectations) == 1 and not choice["row"] < expectations[0]["row"] < event["row"]:
                issues.append("choice_expectation_feedback_order_not_strict")
        choice_code = (1 if SELECTION[selections[0]["code"]][1] == "left" else 2) if len(selections) == 1 else None
        expect_code = EXPECTATION[expectations[0]["code"]][1] + 3 if len(expectations) == 1 else None
        blocks.append({"task": task, "feedback_ordinal": ordinal[task], "feedback_event_row": event["row"],
                       "feedback_flag": flag, "previous_feedback_event_row": previous_feedback_row,
                       "segment_event_rows": [e["row"] for e in window] + [event["row"]],
                       "choice_event_rows": [e["row"] for e in selections],
                       "expectation_event_rows": [e["row"] for e in expectations],
                       "choice_side_code": choice_code, "expectation_mat_code": expect_code,
                       "strict_comparable": not issues, "issues": issues})
        previous_feedback_row = event["row"]
        window = []
    return {"blocks": blocks, "tail_event_rows": [e["row"] for e in window], "event_count": len(prepared)}


def validate_config(config):
    expected = {"audit_id": "human_trial_linkage_v1", "scope": "exact_record_linkage_and_semantics_only",
                "source_root": "data/human/stolz_2022_openneuro",
                "source_manifest": "data/human/stolz_2022_openneuro/source-manifest.json",
                "evidence_manifest": "data/human/stolz_2022_linkage_evidence/source-manifest.json",
                "performance_analysis": False, "action_map": {"left": 1, "right": 2},
                "expectation_offset": 3, "approximate_matching": False}
    if config != expected or type(config.get("performance_analysis")) is not bool or type(config.get("approximate_matching")) is not bool or type(config.get("expectation_offset")) is not int or any(type(config.get("action_map", {}).get(side)) is not int for side in ("left", "right")):
        raise ValueError("Fixed exact-linkage-only audit configuration required")


def study_hashes(root, config):
    root = Path(root).resolve()
    paths = {"src/drosolearn/human_trial_linkage.py", "src/drosolearn/human_trial_data.py",
             "scripts/verify_human_trial_linkage.py", "tests/test_human_trial_linkage.py",
             "tests/test_human_trial_linkage_run.py", "tests/test_human_trial_linkage_verify.py",
             "docs/human-trial-linkage-protocol.md", "docs/human-trial-linkage-source-review.md",
             "requirements-lock.txt", "requirements-human-candidate-audit.txt", "pyproject.toml"}
    for manifest_name in (config["source_manifest"], config["evidence_manifest"]):
        paths.add(manifest_name)
        manifest_path = root / manifest_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            path = (manifest_path.parent / item["path"]).resolve()
            if not path.is_relative_to(manifest_path.parent.resolve()):
                raise ValueError("Unsafe manifest source path")
            paths.add(path.relative_to(root).as_posix())
    return {name: digest(root / name) for name in sorted(paths)}


def register(root, config_path, output):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    record = {"config": config, "config_sha256": digest(config_path), "source_hashes": study_hashes(root, config),
              "registered_at_utc": datetime.now(timezone.utc).isoformat(),
              "git_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
              "notice": "exploratory anonymous-record linkage audit; source schemas and author code already inspected; no blinded hypothesis preregistration"}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(output, record)
    return record


def load_mat_records(path):
    import numpy as np
    from scipy.io import loadmat
    arrays = loadmat(path, simplify_cells=True)
    names = [task + "_" + key for task in TASKS for key in ("feedback", "action", "expect")]
    row_count = None
    for name in names:
        vector = arrays[name]
        if not isinstance(vector, np.ndarray) or vector.dtype != object or vector.ndim != 1:
            raise ValueError("Expected numeric one-dimensional MATLAB cell vectors")
        if row_count is None:
            row_count = len(vector)
        if len(vector) != row_count:
            raise ValueError("MAT row coverage differs")
        for cell in vector:
            value = np.asarray(cell)
            allowed = {0, 1} if name.endswith("feedback") else ({1, 2} if name.endswith("action") else {3, 4, 5, 6})
            if value.ndim != 1 or not len(value) or not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.complexfloating) or not np.isfinite(value).all() or not set(value.tolist()) <= allowed:
                raise ValueError("Invalid MAT measurement or code")
    records = {}
    for index in range(row_count):
        record = {task: {key: [int(x) for x in arrays[task + "_" + key][index]] for key in ("feedback", "action", "expect")} for task in TASKS}
        if any(len({len(values) for values in record[task].values()}) != 1 for task in TASKS):
            raise ValueError("MAT per-row trial-vector lengths disagree")
        records[f"mat-{index + 1:02d}"] = record
    return records


def audit(root, config):
    root = Path(root).resolve()
    for manifest_name in (config["source_manifest"], config["evidence_manifest"]):
        manifest_path = root / manifest_name
        for item in json.loads(manifest_path.read_text(encoding="utf-8"))["files"]:
            path = (manifest_path.parent / item["path"]).resolve()
            if not path.is_relative_to(manifest_path.parent.resolve()) or path.stat().st_size != item["size_bytes"] or digest(path) != item["sha256"]:
                raise ValueError("Source bytes differ from manifest")
    source = root / config["source_root"]
    manifest = json.loads((root / config["source_manifest"]).read_text(encoding="utf-8"))
    description = json.loads((source / "dataset_description.json").read_text(encoding="utf-8"))
    if description["License"] != "CC0" or description["DatasetDOI"] != "doi:10.18112/openneuro.ds004295.v1.0.0":
        raise ValueError("Source dataset identity/license differs")
    blocks_by_label, sequences = {}, {}
    for item in manifest["files"]:
        if not item["path"].endswith("_events.tsv"):
            continue
        label = item["path"].split("/")[0]
        if label in sequences:
            raise ValueError("Duplicate event file label")
        blocks_by_label[label] = feedback_blocks(read_events(source / item["path"]))
        sequences[label] = {task: [b["feedback_flag"] for b in blocks_by_label[label]["blocks"] if b["task"] == task] for task in TASKS}
    mats = load_mat_records(source / "code/Data/RevEx1_behavioralData.mat")
    graph = link_sequences(sequences, {label: {task: row[task]["feedback"] for task in TASKS} for label, row in mats.items()})
    corroboration, counters = [], Counter()
    for pair in graph["accepted"]:
        label, mat_row = pair["event_label"], pair["mat_row"]
        for block in blocks_by_label[label]["blocks"]:
            task, index = block["task"], block["feedback_ordinal"] - 1
            action, expect = mats[mat_row][task]["action"][index], mats[mat_row][task]["expect"][index]
            strict = block["strict_comparable"]
            row = {**pair, "task": task, "feedback_ordinal": index + 1,
                   "feedback_event_row": block["feedback_event_row"], "strict_comparable": strict,
                   "event_choice_side_code": block["choice_side_code"], "mat_action_code": action,
                   "event_expectation_plus3": block["expectation_mat_code"], "mat_expect_code": expect,
                   "action_numeric_agreement": block["choice_side_code"] == action if strict else None,
                   "expectation_numeric_agreement": block["expectation_mat_code"] == expect if strict else None}
            corroboration.append(row)
            counters["feedback_anchored_pairs"] += 1
            counters["strict_comparable" if strict else "not_strict_comparable"] += 1
            if strict:
                counters["action_numeric_agreements" if row["action_numeric_agreement"] else "action_numeric_disagreements"] += 1
                counters["expectation_numeric_agreements" if row["expectation_numeric_agreement"] else "expectation_numeric_disagreements"] += 1
    all_blocks = [block for item in blocks_by_label.values() for block in item["blocks"]]
    report = {"audit_id": config["audit_id"], "scope": config["scope"],
              "event_files": len(sequences), "mat_rows": len(mats), "matrix_pairs": len(graph["edges"]),
              "event_feedback_lengths": {label: {task: len(values[task]) for task in TASKS} for label, values in sorted(sequences.items())},
              "mat_feedback_lengths": {label: {task: len(values[task]["feedback"]) for task in TASKS} for label, values in sorted(mats.items())},
              "exact_feedback_candidate_pairs": len(graph["accepted"]),
              "unmatched_event_labels": [label for label, candidates in graph["event_candidates"].items() if not candidates],
              "unmatched_mat_rows": [label for label, candidates in graph["mat_candidates"].items() if not candidates],
              "ambiguous_event_labels": [label for label, candidates in graph["event_candidates"].items() if len(candidates) > 1],
              "ambiguous_mat_rows": [label for label, candidates in graph["mat_candidates"].items() if len(candidates) > 1],
              "event_feedback_blocks": len(all_blocks),
              "block_issue_counts": dict(sorted(Counter(issue for block in all_blocks for issue in block["issues"]).items())),
              "corroboration_counts": dict(counters),
              "event_candidates": graph["event_candidates"], "mat_candidates": graph["mat_candidates"],
              "readiness": {"record_linkage": "unique_exact_feedback_candidates_only_not_person_identity",
                            "original_trial_index": "feedback_ordinal_only_not_verified_original_trial",
                            "action_identity": "fixed_left1_right2_numeric_diagnostic_not_verified_shape_ID",
                            "expectation": "fixed_plus3_code_diagnostic_not_calibrated_choice_probability",
                            "reversal_boundaries_or_correct_shape": "not_reconstructed",
                            "RT_transfer": "not_performed",
                            "performance_or_human_model_similarity": "not_tested"}}
    return report, graph, blocks_by_label, corroboration


def _csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(root, config_path, registration_path, output):
    root, config_path, registration_path, output = map(lambda p: Path(p).resolve(), (root, config_path, registration_path, output))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    reg = json.loads(registration_path.read_text(encoding="utf-8"))
    validate_config(config)
    if reg["config"] != config or reg["config_sha256"] != digest(config_path) or reg["source_hashes"] != study_hashes(root, config):
        raise ValueError("Source/config/protocol changed after freeze")
    output.mkdir(parents=True, exist_ok=False)
    import scipy
    import numpy as np
    meta = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "started",
            "registration_sha256": digest(registration_path), "source_root": Path(os.path.relpath(root, output)).as_posix(),
            "config_path": config_path.relative_to(root).as_posix(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    write_json(output / "registration.json", reg)
    write_json(output / "metadata.json", meta)
    try:
        report, graph, blocks, corroboration = audit(root, config)
        write_json(output / "linkage-report.json", report)
        write_json(output / "feedback-blocks.json", blocks)
        _csv(output / "alignment-matrix.csv", graph["edges"], ("event_label", "mat_row", "REW_equal", "PUN_equal", "both_equal"))
        _csv(output / "record-links.csv", graph["accepted"], ("event_label", "mat_row"))
        _csv(output / "corroboration.csv", corroboration, ("event_label", "mat_row", "task", "feedback_ordinal", "feedback_event_row", "strict_comparable", "event_choice_side_code", "mat_action_code", "event_expectation_plus3", "mat_expect_code", "action_numeric_agreement", "expectation_numeric_agreement"))
        meta.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        meta.update(status="failed", failure=repr(error), completed_at_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        (output / "metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")
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
        reg = register(args.root, args.config, args.out)
        print(json.dumps({"registered_at_utc": reg["registered_at_utc"], "frozen_files": len(reg["source_hashes"])}))
    else:
        if args.registration is None:
            parser.error("--registration required")
        report = run(args.root, args.config, args.registration, args.out)
        print(json.dumps({key: report[key] for key in ("event_files", "mat_rows", "exact_feedback_candidate_pairs", "unmatched_event_labels")}))


if __name__ == "__main__":
    main()
