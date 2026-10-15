"""Independent replay of actual-choice transition ledgers; no group contrasts."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/feedback_transition_v1.json"


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(VerificationError("Nonfinite JSON number")))


def exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def safe(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "POSIX relative input path required")
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path.is_file(), "Unsafe or missing input path")
    return path


def integer(value, minimum=0):
    require(isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value), "Canonical integer required")
    number = int(value)
    require(number >= minimum, "Integer below allowed range")
    return number


def measure_stream(stream_id, kind, coordinate, records):
    """Count original adjacency directly, without importing the primary metric."""
    require(isinstance(stream_id, str) and stream_id and kind in {"human", "model"} and records,
            "Named nonempty actual-choice stream required")
    previous_trial, previous_position = 0, 0
    for record in records:
        require(type(record["trial"]) is int and record["trial"] > previous_trial and
                type(record["source_position"]) is int and record["source_position"] > previous_position,
                "Source order must be increasing and unique")
        previous_trial, previous_position = record["trial"], record["source_position"]
        require(record["choice"] is None or record["choice"] == "query" or
                (type(record["choice"]) is int and record["choice"] in (0, 1)),
                "Canonical binary/query/missing choice required")
        require(record["feedback"] is None or record["feedback"] == "information" or
                (type(record["feedback"]) is int and record["feedback"] in (0, 1)),
                "Actual binary/information/missing feedback required")
        require(type(record["actual"]) is bool and type(record["training"]) is bool,
                "Actual/training flags must be Boolean")
        require(all(isinstance(record[key], str) and record[key] for key in ("task", "phase", "state")),
                "Task, phase and state semantics are required")
    by_feedback = {str(value): {"denominator": 0, "stay": 0, "switch": 0} for value in (0, 1)}
    ledger, reasons_count = [], Counter()
    for current, following in zip(records, records[1:]):
        reasons = []
        for prefix, record in (("current", current), ("next", following)):
            if record["choice"] is None:
                reasons.append(prefix + "_choice_missing")
            elif record["choice"] == "query":
                reasons.append(prefix + "_query")
        if current["feedback"] is None:
            reasons.append("current_feedback_missing")
        elif type(current["feedback"]) is not int or current["feedback"] not in (0, 1):
            reasons.append("current_feedback_nonbinary")
        if following["trial"] != current["trial"] + 1:
            reasons.append("trial_gap")
        for key in ("task", "phase", "state"):
            if following[key] != current[key]:
                reasons.append(key + "_boundary")
        if not current["actual"] or not following["actual"]:
            reasons.append("not_actual")
        if not current["training"] or not following["training"]:
            reasons.append("not_training")
        reasons.sort()
        eligible = not reasons
        feedback = current["feedback"]
        stayed = current["choice"] == following["choice"] if eligible else None
        if eligible:
            count = by_feedback[str(feedback)]
            count["denominator"] += 1
            count["stay" if stayed else "switch"] += 1
        reasons_count.update(reasons)
        ledger.append({"stream_id": stream_id, "source_position": str(current["source_position"]),
                       "next_source_position": str(following["source_position"]),
                       "trial": str(current["trial"]), "next_trial": str(following["trial"]),
                       "eligible": "1" if eligible else "0", "reasons": "|".join(reasons),
                       "feedback": str(feedback) if type(feedback) is int and feedback in (0, 1) else "",
                       "stay": str(int(stayed)) if stayed is not None else ""})
    for counts in by_feedback.values():
        denominator = counts["denominator"]
        counts["stay_rate"] = counts["stay"] / denominator if denominator else None
        counts["switch_rate"] = counts["switch"] / denominator if denominator else None
    included = sum(counts["denominator"] for counts in by_feedback.values())
    return ({"stream_id": stream_id, "kind": kind, "coordinate": coordinate,
             "trials": len(records), "candidates": len(ledger), "eligible": included,
             "excluded": len(ledger) - included, "reason_counts": dict(sorted(reasons_count.items())),
             "by_feedback": by_feedback}, ledger)


LEDGER_FIELDS = ("stream_id", "source_position", "next_source_position", "trial", "next_trial",
                 "eligible", "reasons", "feedback", "stay")


def source_csv(path, fields):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        require(next(reader, None) == list(fields), "CSV schema differs")
        rows = []
        for values in reader:
            require(len(values) == len(fields), "Ragged CSV row")
            rows.append(dict(zip(fields, values)))
    return rows


def human_records(path, stream_id, trials, boundary):
    """Decode A/R/S directly; C is a structure gate, never a performance score."""
    import numpy as np
    from scipy.io import loadmat
    arrays = loadmat(path, variable_names=["A", "R", "S", "C"])
    shapes = {"A": (1, trials), "R": (trials, 2), "S": (trials, 1), "C": (1, trials)}
    require(type(trials) is int and trials > 0 and type(boundary) is int and boundary > 0,
            "Fixed positive trial count and regime boundary required")
    for key, shape in shapes.items():
        require(key in arrays and arrays[key].shape == shape and arrays[key].dtype.kind in "iuf" and
                not np.isinf(arrays[key]).any(), "Human raw schema/type differs")
    action, result, state, correctness = arrays["A"][0], arrays["R"], arrays["S"][:, 0], arrays["C"][0]
    require(np.isin(state, [1, 2]).all() and np.isin(result, [-1, 1]).all(),
            "Human raw state/result alphabet differs")
    records = []
    for index in range(trials):
        chosen, actual_feedback = None, None
        if np.isnan(action[index]):
            require(np.isnan(correctness[index]), "Missing action C must remain missing")
        elif action[index] == 3:
            chosen, actual_feedback = "query", "information"
            require(np.isnan(correctness[index]), "Query C must remain missing")
        else:
            require(action[index] in (1, 2), "Human original action alphabet differs")
            require(np.isfinite(correctness[index]) and correctness[index] == int(action[index] == state[index]),
                    "Human C/action/state structure differs")
            chosen = int(action[index]) - 1
            actual_feedback = 1 if result[index, chosen] == 1 else 0
        records.append({"source_position": index + 1, "trial": index + 1, "choice": chosen,
                        "feedback": actual_feedback, "task": "markovic_card",
                        "phase": "initial" if index + 1 <= boundary else "transfer",
                        "state": str(int(state[index])), "actual": True, "training": True})
    return (stream_id, "human", "source_coded_card_identity", records)


MODEL_FIELDS = ("seed", "condition", "history", "phase", "trial", "family_id", "left_count", "right_count",
                "action", "reward", "p_right", "optimizer_factor", "internal_step_l2", "readout_step_l2",
                "kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2", "decision_left_l2", "decision_right_l2")


def model_records(path, seed, conditions, histories, trials):
    """Use logged sampled actions, mapping physical side through shown counts."""
    require(type(seed) is int and type(trials) is int and trials > 0, "Fixed model identity/count required")
    streams = {(condition, history): [] for condition in conditions for history in histories}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        fields = next(reader, None)
        required_fields = {"seed", "condition", "history", "phase", "trial", "left_count", "right_count",
                           "action", "reward", "p_right"}
        require(fields is not None and len(fields) == len(set(fields)) and required_fields <= set(fields),
                "Model actual-action CSV schema differs")
        for source_position, values in enumerate(reader, 2):
            require(len(values) == len(fields), "Ragged actual-action CSV row")
            record = dict(zip(fields, values))
            require(integer(record["seed"]) == seed and record["phase"] in ("phase1", "phase2"),
                    "Model source seed/phase differs")
            if record["phase"] != "phase2":
                continue
            key = record["condition"], record["history"]
            require(key in streams, "Unregistered model condition/history")
            trial, action, feedback = (integer(record[name]) for name in ("trial", "action", "reward"))
            left, right = integer(record["left_count"]), integer(record["right_count"])
            require(action in (0, 1) and feedback in (0, 1) and {left, right} == {2, 4},
                    "Model actual quantity/action/result differs")
            probability = float(record["p_right"])
            require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid logged model probability")
            chosen_count = left if action == 0 else right
            chosen = 0 if chosen_count < (right if action == 0 else left) else 1
            require(feedback == int(chosen == 0), "Model phase2 action/result structure differs")
            streams[key].append({"source_position": source_position, "trial": trial,
                                 "choice": chosen, "feedback": feedback, "task": "abstract_quantity",
                                 "phase": "phase2", "state": "smaller", "actual": True, "training": True})
    decoded = []
    for (condition, history), rows in streams.items():
        require(len(rows) == trials and [row["trial"] for row in rows] == list(range(1, trials + 1)),
                "Model source phase2 trial coverage differs")
        decoded.append((f"model/{seed}/{condition}/{history}/phase2", "model",
                        "smaller_larger_quantity_category", rows))
    return decoded


FROZEN_PATHS = {
    "src/drosolearn/feedback_transition.py", "src/drosolearn/__init__.py", "scripts/run_feedback_transition.py",
    "scripts/verify_feedback_transition.py", "tests/test_feedback_transition.py",
    "tests/test_feedback_transition_run.py", "tests/test_feedback_transition_verify.py",
    CONFIG_PATH, "docs/feedback-transition-protocol.md", "docs/model-human-metric-interface.md",
    "requirements-lock.txt", "requirements-human-candidate-audit.txt",
}


def checked_config(root):
    cfg = read_json(safe(root, CONFIG_PATH))
    require(cfg["study_id"] == "feedback_transition_v1" and cfg["pool_group_or_species"] is False and
            cfg["zero_denominator"] == "null" and cfg["exclude_boundaries"] == ["state", "phase", "task"],
            "Unregistered metric or inference scope")
    require(set(cfg["frozen_files"]) == FROZEN_PATHS and len(cfg["frozen_files"]) == len(FROZEN_PATHS),
            "Frozen source set differs")
    require(cfg["human"]["coordinate"] == "source_coded_card_identity" and
            cfg["human"]["historical_execution_version_verified"] is False and
            type(cfg["human"]["trials"]) is int and cfg["human"]["trials"] > 0 and
            type(cfg["human"]["regime_boundary_after_trial"]) is int and
            cfg["human"]["regime_boundary_after_trial"] > 0,
            "Human source-coded semantic definition differs")
    model = cfg["model"]
    require(model["phase"] == "phase2" and model["coordinate"] == "smaller_larger_quantity_category" and
            type(model["trials"]) is int and model["trials"] > 0,
            "Actual model scope differs")
    prior = read_json(safe(root, model["parent_config"]))
    require(exact(model["conditions"], prior["conditions"]) and exact(model["histories"], prior["histories"]) and
            exact(model["seeds"], prior["seeds"]) and exact(model["trials"], prior["trials_per_phase"]) and
            prior["pair"] == [2, 4], "Prior model coverage differs")
    for key in ("conditions", "histories", "seeds"):
        require(isinstance(model[key], list) and model[key] and len(set(model[key])) == len(model[key]),
                "Unique nonempty model coordinates required")
    require(all(type(seed) is int and seed > 0 for seed in model["seeds"]), "Integer model seed identities required")
    require(len(cfg["parent_files"]) == len(set(cfg["parent_files"])), "Duplicate parent path")
    return cfg


def check_parents(root, cfg):
    task_path = safe(root, cfg["human"]["structure_result"])
    task, task_check = read_json(task_path), read_json(safe(root, cfg["human"]["structure_verification"]))
    require(task["status"] == "pass" and task["gate"]["source_coded_descriptive_metric_allowed"] is True and
            task["gate"]["exact_historical_task_execution_version_verified"] is False and
            task["gate"]["direct_human_model_similarity_allowed"] is False,
            "Human semantic/structure gate is closed or overstated")
    require(task_check["status"] == "pass" and task_check["result_sha256"] == digest(task_path) and
            task_check["source_coded_descriptive_gate_passed"] is True,
            "Human structure independent gate differs")
    model = read_json(safe(root, cfg["model"]["verification"]))
    require(model["passed"] is True, "Model actual replay gate is closed")
    return task


def checked_inputs(root, cache, cfg, registration):
    cache = Path(cache).resolve()
    require(not cache.is_relative_to(root), "Human raw source cache must remain external")
    origin = read_json(safe(root, cfg["human"]["origin_manifest"]))
    require(exact(origin["trials_per_row"], cfg["human"]["trials"]), "Human raw trial scope differs")
    raw = [entry for entry in origin["source_files"] if entry["role"] == "osf_raw"]
    require(len(raw) == 25 and all(entry["derived_group"] == "main" for entry in raw) and
            len({entry["derived_row_index"] for entry in raw}) == len(raw) and
            len({entry["osf_file_id"] for entry in raw}) == len(raw), "Unique all-source human records required")
    expected = [{"base": "human_cache", "role": "human_raw", "path": entry["path"],
                 "bytes": entry["bytes"], "sha256": entry["sha256"], "osf_file_id": entry["osf_file_id"],
                 "row_index": entry["derived_row_index"]} for entry in raw]
    model_artifacts = read_json(safe(root, cfg["model"]["artifact_hashes"]))
    for seed in cfg["model"]["seeds"]:
        relative = f"seed_{seed}/trials.csv"
        path = cfg["model"]["root"] + "/" + relative
        source = safe(root, path)
        expected.append({"base": "repo", "role": "model_actual", "path": path,
                         "bytes": source.stat().st_size, "sha256": model_artifacts[relative], "seed": seed})
    for path in cfg["parent_files"]:
        source = safe(root, path)
        expected.append({"base": "repo", "role": "parent", "path": path,
                         "bytes": source.stat().st_size, "sha256": digest(source)})
    require(len({(entry["base"], entry["path"]) for entry in expected}) == len(expected) and
            exact(registration["inputs"], expected), "Registered source coverage or identities differ")
    for entry in expected:
        require(type(entry["bytes"]) is int and entry["bytes"] >= 0 and
                re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]), "Input digest or size is malformed")
        source = safe(cache if entry["base"] == "human_cache" else root, entry["path"])
        require(source.stat().st_size == entry["bytes"] and digest(source) == entry["sha256"],
                "Registered source bytes changed")
    return expected


def check_task_source_binding(task_parent, human_entries):
    fields = {"path", "bytes", "sha256", "derived_group", "derived_row_index", "osf_file_id"}
    prior = task_parent.get("inputs")
    require(isinstance(prior, list) and all(set(entry) == fields for entry in prior),
            "Human gate original source identities are missing")
    expected = [{"path": entry["path"], "bytes": entry["bytes"], "sha256": entry["sha256"],
                 "derived_group": "main", "derived_row_index": entry["row_index"],
                 "osf_file_id": entry["osf_file_id"]} for entry in human_entries]
    require(exact(prior, expected),
            "Human gate original source byte/row/group binding differs")


def utc(value):
    require(type(value) is str, "UTC timestamp must be text")
    stamp = datetime.fromisoformat(value)
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0, "Timezone-aware UTC required")
    return stamp


def verify(registration_path, result_dir, human_cache, root=None):
    root = Path(root if root is not None else ROOT).resolve()
    registration_path, result_dir = Path(registration_path).resolve(), Path(result_dir).resolve()
    cfg, registration = checked_config(root), read_json(registration_path)
    require(registration["study_id"] == cfg["study_id"] and
            registration["registration_type"] == "descriptive_feedback_transition" and
            exact(registration["config"], cfg) and registration["config_sha256"] == digest(safe(root, CONFIG_PATH)),
            "Registered study/config differs")
    frozen = {path: digest(safe(root, path)) for path in sorted(FROZEN_PATHS)}
    require(exact(registration["frozen_files"], frozen), "Frozen code/test/protocol bytes changed")
    require(type(registration["source_git_commit"]) is str and
            re.fullmatch(r"[a-f0-9]{40}", registration["source_git_commit"]), "Source Git identity missing")
    task_parent = check_parents(root, cfg)
    entries = checked_inputs(root, human_cache, cfg, registration)
    human_entries = [entry for entry in entries if entry["role"] == "human_raw"]
    check_task_source_binding(task_parent, human_entries)
    require({(row["row_index"], row["osf_file_id"]) for row in task_parent["rows"]} ==
            {(entry["row_index"], entry["osf_file_id"]) for entry in human_entries} and
            all(row["structural_passed"] is True for row in task_parent["rows"]),
            "Human gate does not cover every selected raw source")
    artifacts = read_json(result_dir / "artifact_hashes.json")
    require(set(artifacts) == {"metadata.json", "ledger.csv", "metrics.json"}, "Primary artifact coverage differs")
    for path, checksum in artifacts.items():
        require(checksum == digest(safe(result_dir, path)), "Primary artifact hash differs")
    metadata, report = read_json(result_dir / "metadata.json"), read_json(result_dir / "metrics.json")
    require(metadata["status"] == "completed" and metadata["registration_sha256"] == digest(registration_path) and
            metadata["source_git_commit"] == registration["source_git_commit"] and
            metadata["config_sha256"] == registration["config_sha256"] and exact(metadata["frozen_files"], frozen) and
            exact(metadata["inputs"], entries), "Completed run provenance differs")
    require(utc(registration["registered_at_utc"]) <= utc(metadata["started_at_utc"]) <= utc(metadata["completed_at_utc"]),
            "Registration/run chronology differs")
    require(set(metadata["runtime"]) == {"python", "numpy"} and
            all(type(value) is str and value for value in metadata["runtime"].values()), "Runtime provenance differs")
    summaries, ledger_count = [], 0
    with (result_dir / "ledger.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        require(next(reader, None) == list(LEDGER_FIELDS), "Ledger schema differs")
        for entry in entries:
            if entry["role"] == "human_raw":
                decoded = [human_records(safe(human_cache, entry["path"]), f"human/main/{entry['row_index']}",
                                         cfg["human"]["trials"], cfg["human"]["regime_boundary_after_trial"])]
            elif entry["role"] == "model_actual":
                decoded = model_records(safe(root, entry["path"]), entry["seed"], cfg["model"]["conditions"],
                                        cfg["model"]["histories"], cfg["model"]["trials"])
            else:
                continue
            for stream_id, kind, coordinate, records in decoded:
                summary, expected_ledger = measure_stream(stream_id, kind, coordinate, records)
                summaries.append(summary)
                for expected in expected_ledger:
                    actual = next(reader, None)
                    require(actual == [expected[field] for field in LEDGER_FIELDS],
                            "Independent original adjacency, exclusions or choice transition differs")
                    ledger_count += 1
        require(next(reader, None) is None, "Extra ledger rows")
    expected_report = {"study_id": cfg["study_id"], "streams": summaries, "pooled_species_or_group_estimates": False}
    require(exact(report, expected_report), "Independent recordwise denominator, numerator, rate or scope differs")
    expected_metadata = {
        "study_id": cfg["study_id"], "status": "completed", "started_at_utc": metadata["started_at_utc"],
        "registration_sha256": digest(registration_path), "source_git_commit": registration["source_git_commit"],
        "config_sha256": registration["config_sha256"], "frozen_files": frozen, "inputs": entries,
        "runtime": metadata["runtime"], "scope": cfg["scope"], "group_or_species_contrast_computed": False,
        "new_model_learning_executed": False, "level4_or_level5_supported": False,
        "completed_at_utc": metadata["completed_at_utc"], "streams": len(summaries),
        **{key: sum(summary[key] for summary in summaries) for key in
           ("trials", "candidates", "eligible", "excluded")},
    }
    require(exact(metadata, expected_metadata), "Run counts or interpretation scope differs")
    return {"status": "passed", "scope": "independent_recordwise_transition_no_group_or_species_contrast",
            "registration_sha256": digest(registration_path), "config_sha256": registration["config_sha256"],
            "input_files_verified": len(entries), "frozen_files_verified": len(frozen),
            "primary_artifacts_verified": len(artifacts), "human_records_replayed": len(human_entries),
            "model_streams_replayed": sum(summary["kind"] == "model" for summary in summaries),
            "streams_verified": len(summaries), "candidate_pairs_verified": ledger_count,
            "group_or_species_contrast_computed": False, "level4_or_level5_supported": False, "errors": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--human-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Verification output already exists")
    try:
        result = verify(args.registration, args.result_dir, args.human_cache)
    except Exception as error:
        result = {"status": "failed", "scope": "independent_recordwise_transition_no_group_or_species_contrast",
                  "group_or_species_contrast_computed": False, "level4_or_level5_supported": False,
                  "errors": [f"{type(error).__name__}: {error}"]}
    result["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "scope", "errors")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
