"""Recordwise feedback transitions conditional on current choice and task state.

This module creates no learning, random draws, group estimates or causal effects.
It preserves the original adjacency/exclusion definition and all designed cells.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import subprocess

from .feedback_transition import binary, human_rows, measure_stream


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ["linear", "fixed", "plastic", "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic", "frozen"]
ENVIRONMENTS = [{"id": "deterministic", "q_optimal": 1., "q_other": 0.},
                {"id": "probabilistic", "q_optimal": .8, "q_other": .2},
                {"id": "uninformative", "q_optimal": .5, "q_other": .5}]
BASE_SUMMARY_FIELDS = ("stream_id", "kind", "coordinate", "trials", "candidates", "eligible", "excluded", "reason_counts", "by_feedback")
LEDGER_FIELDS = ("stream_id", "source_position", "next_source_position", "trial", "next_trial", "eligible", "reasons", "feedback", "stay",
                 "phase", "state", "choice", "next_choice", "local_trial", "next_local_trial")
MODEL_FIELDS = ("stage", "seed", "condition", "environment", "phase", "trial", "global_trial", "family_id", "left_count", "right_count",
                "chosen_count", "action", "reward", "p_right", "q_reward", "action_uniform", "reward_uniform", "head_update_called",
                "internal_update_called", "optimizer_factor", "internal_step_l2", "readout_step_l2", "kc_left_l2", "kc_right_l2", "mbon_left_l2",
                "mbon_right_l2", "decision_left_l2", "decision_right_l2", "head_updates", "internal_updates")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))


def exact(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(b, dict):
        return a.keys() == b.keys() and all(exact(a[key], value) for key, value in b.items())
    if isinstance(b, list):
        return len(a) == len(b) and all(exact(left, right) for left, right in zip(a, b))
    return a == b


def safe(root, relative):
    require(type(relative) is str and relative and "\\" not in relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "Safe POSIX relative path required")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Path escapes declared source base")
    return path


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def utc(value):
    require(type(value) is str, "UTC timestamp must be text")
    stamp = datetime.fromisoformat(value)
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0, "Timezone-aware UTC required")
    return stamp


def validate_config(cfg):
    require(cfg["study_id"] == "conditional_transition_v1" and cfg["scope"] in ("registered_recordwise_descriptive_measurement", "integration_test"),
            "Fixed recordwise measurement identity required")
    h, m = cfg["human"], cfg["model"]
    require(h["coordinate"] == "source_coded_card_identity" and h["historical_execution_version_verified"] is False
            and m["coordinate"] == "smaller_larger_quantity_category" and m["conditions"] == CONDITIONS
            and exact(m["environments"], ENVIRONMENTS), "Fixed source-coded coordinates and all registered models/environments required")
    require(type(h["record_count"]) is int and h["record_count"] > 0 and type(h["trials"]) is int and h["trials"] >= 2
            and type(h["regime_boundary_after_trial"]) is int and 0 < h["regime_boundary_after_trial"] < h["trials"]
            and type(m["trials_per_phase"]) is int and m["trials_per_phase"] >= 2
            and type(m["seeds"]) is list and m["seeds"] and all(type(seed) is int and seed >= 0 for seed in m["seeds"])
            and len(set(m["seeds"])) == len(m["seeds"]), "Original row/phase/seed coverage required")
    if cfg["scope"] == "registered_recordwise_descriptive_measurement":
        require((h["record_count"], h["trials"], h["regime_boundary_after_trial"], m["trials_per_phase"]) == (25, 1000, 800, 400)
                and m["seeds"] == list(range(11001, 11011)), "All source-verified human rows and existing network seeds required")
    else:
        require((h["record_count"], h["trials"], h["regime_boundary_after_trial"], m["trials_per_phase"]) == (2, 8, 4, 4)
                and set(m["seeds"]) <= {17, 18}, "Only handcrafted two-record eight-row development fixtures allowed")
    streams = len(m["seeds"]) * len(CONDITIONS) * len(ENVIRONMENTS)
    expected = (h["record_count"] + streams, h["record_count"] * h["trials"] + streams * 2 * m["trials_per_phase"],
                h["record_count"] * (h["trials"] - 1) + streams * (2 * m["trials_per_phase"] - 1), h["record_count"] * 8 + streams * 4)
    fields = ("registered_streams", "registered_input_trials", "registered_candidate_pairs", "registered_strata")
    require(all(type(cfg[key]) is int for key in fields) and tuple(cfg[key] for key in fields) == expected, "Fixed complete record/cell budget differs")
    require(all(type(cfg[key]) is int and cfg[key] == 0 for key in ("learning_updates", "learner_constructions", "dataset_generations", "rng_draws"))
            and all(cfg[key] is False for key in ("pooling", "strata_averaging", "causal_inference"))
            and cfg["exclude_boundaries"] == ["state", "phase", "task"] and cfg["zero_denominator"] is None,
            "No-learning scope and original exclusion/zero-denominator rules required")
    require(cfg["common_support"] == "both_actual_feedback_denominators_positive_within_predefined_phase_state_current_choice_cell"
            and cfg["metric"] == "recordwise_actual_adjacent_stay_switch_counts_by_feedback_and_predefined_phase_state_current_choice",
            "Fixed conditional measurement and support definition required")
    for key in ("parent_files", "frozen_files"):
        require(type(cfg[key]) is list and cfg[key] and len(set(cfg[key])) == len(cfg[key]), "Explicit unique source/evidence inventories required")


def predefined_strata(kind):
    require(kind in ("human", "model"), "Known source kind required")
    if kind == "human":
        return [{"phase": phase, "state": state, "choice": choice} for phase in ("initial", "transfer") for state in ("1", "2") for choice in (0, 1)]
    return [{"phase": phase, "state": state, "choice": choice} for phase, state in (("phase1", "larger"), ("phase2", "smaller")) for choice in (0, 1)]


def measure_conditional(rows):
    """Enrich every original pair and partition eligible integer counts exactly."""
    summary, ledger = measure_stream(rows)
    design = predefined_strata(rows[0]["kind"])
    index = {(row["phase"], row["state"], row["choice"]): {**row,
             "by_feedback": {str(f): {"denominator": 0, "stay": 0, "switch": 0} for f in (0, 1)}} for row in design}
    allowed_states = {(row["phase"], row["state"]) for row in design}
    require(all(type(row.get("local_trial")) is int and row["local_trial"] > 0 and (row["phase"], row["state"]) in allowed_states
                for row in rows), "Original local trials and designed phase/state coordinates required")
    for current, following, pair in zip(rows, rows[1:], ledger):
        pair.update(phase=current["phase"], state=current["state"], choice="" if current["choice"] is None else current["choice"],
                    next_choice="" if following["choice"] is None else following["choice"],
                    local_trial=current["local_trial"], next_local_trial=following["local_trial"])
        if pair["eligible"]:
            cell = index[current["phase"], current["state"], current["choice"]]["by_feedback"][str(current["feedback"])]
            cell["denominator"] += 1
            cell["stay"] += pair["stay"]
            cell["switch"] += 1 - pair["stay"]
    for row in index.values():
        for cell in row["by_feedback"].values():
            denominator = cell["denominator"]
            cell["stay_rate"] = cell["stay"] / denominator if denominator else None
            cell["switch_rate"] = cell["switch"] / denominator if denominator else None
        zero, one = row["by_feedback"]["0"], row["by_feedback"]["1"]
        support = zero["denominator"] > 0 and one["denominator"] > 0
        row.update(stay_rate_difference_1_minus_0=one["stay_rate"] - zero["stay_rate"] if support else None,
                   common_support=support, support_status="both_feedback" if support else "feedback_0_only" if zero["denominator"] else
                   "feedback_1_only" if one["denominator"] else "neither")
    for feedback in ("0", "1"):
        require(all(sum(row["by_feedback"][feedback][field] for row in index.values()) == summary["by_feedback"][feedback][field]
                    for field in ("denominator", "stay", "switch")), "Conditional integer counts do not partition the original metric")
    summary["strata"] = list(index.values())
    return summary, ledger


def integer(record, key):
    token = record[key]
    require(type(token) is str and re.fullmatch(r"0|[1-9][0-9]*", token), "Canonical logged integer required")
    return int(token)


def number(record, key):
    token = record[key]
    require(type(token) is str and token == token.strip() and token != "", "Canonical logged number required")
    value = float(token)
    require(math.isfinite(value), "Finite logged number required")
    return value


def adapt_model_record(record, *, seed, source_position, trials_per_phase, conditions=CONDITIONS, environments=ENVIRONMENTS):
    require(type(source_position) is int and source_position >= 2 and type(seed) is int and type(trials_per_phase) is int and trials_per_phase > 0,
            "Known original CSV line and phase budget required")
    require(set(record) == set(MODEL_FIELDS) and record["stage"] == "network" and integer(record, "seed") == seed
            and record["condition"] in conditions and record["phase"] in ("phase1", "phase2"), "Registered actual network log schema differs")
    environment = next((env for env in environments if env["id"] == record["environment"]), None)
    require(environment is not None, "Unknown actual reward environment")
    phase, local = record["phase"], integer(record, "trial")
    global_trial = integer(record, "global_trial")
    require(1 <= local <= trials_per_phase and global_trial == local + (trials_per_phase if phase == "phase2" else 0),
            "Original global and local trial coordinates disagree")
    left, right, action, reward = (integer(record, key) for key in ("left_count", "right_count", "action", "reward"))
    require({left, right} == {2, 4} and action in (0, 1) and reward in (0, 1)
            and integer(record, "chosen_count") == (left, right)[action] and type(record["family_id"]) is str and record["family_id"],
            "Actual choice/quantity identity differs")
    p, q, au, ru = (number(record, key) for key in ("p_right", "q_reward", "action_uniform", "reward_uniform"))
    choice = int((left, right)[action] == 4)
    designated = 1 if phase == "phase1" else 0
    expected_q = environment["q_optimal"] if choice == designated else environment["q_other"]
    require(0 <= p <= 1 and 0 <= au < 1 and 0 <= ru < 1 and q == expected_q
            and action == int(au < p) and reward == int(ru < q), "Logged sampled action or actual stochastic reward differs")
    return {"stream_id": f"model/{seed}/{record['condition']}/{record['environment']}", "kind": "model",
            "coordinate": "smaller_larger_quantity_category", "source_position": source_position, "trial": global_trial,
            "local_trial": local, "task": "abstract_quantity", "phase": phase, "state": "larger" if phase == "phase1" else "smaller",
            "choice": choice, "feedback": reward, "actual": True, "training": True}


def model_streams(path, seed, cfg):
    model, grouped = cfg["model"], {(condition, env["id"]): [] for condition in cfg["model"]["conditions"] for env in cfg["model"]["environments"]}
    with Path(path).open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == list(MODEL_FIELDS), "Exact original30-field actual network CSV required")
        for line, record in enumerate(reader, 2):
            require(None not in record and all(value is not None for value in record.values()), "Malformed original actual CSV row")
            canonical = adapt_model_record(record, seed=seed, source_position=line, trials_per_phase=model["trials_per_phase"],
                                            conditions=model["conditions"], environments=model["environments"])
            grouped[record["condition"], record["environment"]].append(canonical)
    for condition in model["conditions"]:
        for environment in model["environments"]:
            rows = grouped[condition, environment["id"]]
            require([row["trial"] for row in rows] == list(range(1, 2 * model["trials_per_phase"] + 1)),
                    "Every actual stream must keep original global order and all800 trials")
            yield rows


def raw_entries(root, cfg):
    manifest = read_json(safe(root, cfg["human"]["origin_manifest"]))
    entries = [row for row in manifest["source_files"] if row["role"] == "osf_raw"]
    require(manifest["trials_per_row"] == cfg["human"]["trials"] and len(entries) == cfg["human"]["record_count"]
            and all(row["derived_group"] == "main" for row in entries)
            and len({row["derived_row_index"] for row in entries}) == len(entries)
            and len({row["osf_file_id"] for row in entries}) == len(entries), "All uniquely source-verified Main records required")
    return entries


def parent_gate(root, cfg):
    identity = cfg["parent_gate_identity"]
    require(set(identity) == {"published_commit", "file_sha256"} and re.fullmatch(r"[a-f0-9]{40}", identity["published_commit"])
            and set(identity["file_sha256"]) == set(cfg["parent_files"]), "Pinned parent publication identity required")
    for path, checksum in identity["file_sha256"].items():
        require(type(checksum) is str and re.fullmatch(r"[a-f0-9]{64}", checksum) and digest(safe(root, path)) == checksum,
                "A published source/semantic/verification parent changed")
    h, model = cfg["human"], cfg["model"]
    task_path = safe(root, h["structure_result"])
    task, task_checked = read_json(task_path), read_json(safe(root, h["structure_verification"]))
    require(task["status"] == "pass" and task["gate"] == {"source_coded_descriptive_metric_allowed": True,
            "exact_historical_task_execution_version_verified": False, "direct_human_model_similarity_allowed": False}
            and task_checked["status"] == "pass" and task_checked["source_coded_descriptive_gate_passed"] is True
            and task_checked["result_sha256"] == digest(task_path)
            and task_checked["registration_sha256"] == digest(safe(root, "results/registration_human_task_structure_v1.json")),
            "Human source-coded semantic gate is closed or overstated")
    sources = raw_entries(root, cfg)
    require(exact(task["inputs"], [{key: row[key] for key in ("path", "bytes", "sha256", "derived_group", "derived_row_index", "osf_file_id")}
                                  for row in sources]), "Human semantic audit does not bind these exact original sources")
    require(read_json(safe(root, "results/human_osf_origin_v1/verification.json"))["status"] == "passed", "Original human source reproduction gate is closed")
    old_reg = read_json(safe(root, "results/registration_feedback_transition_v1.json"))
    old_meta = read_json(safe(root, "results/feedback_transition_v1/metadata.json"))
    old_check = read_json(safe(root, h["prior_verification"]))
    old_hashes = read_json(safe(root, "results/feedback_transition_v1/artifact_hashes.json"))
    require(old_check["status"] == "passed" and old_check["registration_sha256"] == digest(safe(root, "results/registration_feedback_transition_v1.json"))
            and old_check["config_sha256"] == digest(safe(root, "configs/feedback_transition_v1.json"))
            and old_meta["status"] == "completed" and old_meta["registration_sha256"] == old_check["registration_sha256"]
            and old_hashes["metrics.json"] == digest(safe(root, h["prior_metrics"])), "Previously verified base measurement bytes differ")
    expected_human = [{"base": "human_cache", "role": "human_raw", "path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"],
                      "osf_file_id": row["osf_file_id"], "row_index": row["derived_row_index"]} for row in sources]
    require(exact([row for row in old_reg["inputs"] if row["role"] == "human_raw"], expected_human), "Human sources differ from the prior actual-row measurement")
    parent_config = read_json(safe(root, model["parent_config"]))
    model_check = read_json(safe(root, model["verification"]))
    model_reg_path = safe(root, "results/registration_stochastic_feedback_network_v1.json")
    model_reg = read_json(model_reg_path)
    model_meta = read_json(safe(root, model["root"] + "/metadata.json"))
    require(parent_config["stage"] == "network" and parent_config["study_id"] == "stochastic_feedback_network_v1"
            and all(exact(model[key], parent_config[key]) for key in ("seeds", "conditions", "environments", "trials_per_phase"))
            and model_check["status"] == "passed" and model_check["passed"] is True
            and model_check["registration_sha256"] == digest(model_reg_path)
            and model_check["config_sha256"] == digest(safe(root, model["parent_config"]))
            and model_check["summary_sha256"] == digest(safe(root, model["root"] + "/summary.json"))
            and model_meta["status"] == "completed" and exact(model_meta["config"], parent_config)
            and exact(model_reg["config"], parent_config) and model_meta["registration_sha256"] == model_check["registration_sha256"]
            and model_meta["source_git_commit"] == model_reg["source_git_commit"], "All actual network trajectories require exact independent verification")
    for checked in (task_checked, old_check, model_check):
        require(utc(checked["verified_at_utc"]) <= datetime.now(timezone.utc), "Parent independent verification must precede this measurement")
    return {"name": "source_verified_human_and_actual_network_recordwise_measurement", "passed": True,
            "published_commit": identity["published_commit"], "file_sha256": identity["file_sha256"],
            "historical_task_execution_version_verified": False, "direct_human_model_similarity_allowed": False}


def input_inventory(root, cfg, cache):
    cache = Path(cache).resolve()
    require(not cache.is_relative_to(Path(root).resolve()), "Original human MAT data must remain outside the repository")
    entries = [{"base": "human_cache", "role": "human_raw", "path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"],
                "osf_file_id": row["osf_file_id"], "row_index": row["derived_row_index"]} for row in raw_entries(root, cfg)]
    artifacts = read_json(safe(root, cfg["model"]["artifact_hashes"]))
    for seed in cfg["model"]["seeds"]:
        relative = f"seed_{seed}/trials.csv"
        path = cfg["model"]["root"] + "/" + relative
        source = safe(root, path)
        require(relative in artifacts and digest(source) == artifacts[relative], "Original actual network CSV differs from its pinned inventory")
        entries.append({"base": "repo", "role": "model_actual", "path": path, "bytes": source.stat().st_size, "sha256": artifacts[relative], "seed": seed})
    require(len({(row["base"], row["path"]) for row in entries}) == len(entries), "Repeated original data source")
    for row in entries:
        source = safe(cache if row["base"] == "human_cache" else root, row["path"])
        require(type(row["bytes"]) is int and row["bytes"] > 0 and type(row["sha256"]) is str and re.fullmatch(r"[a-f0-9]{64}", row["sha256"])
                and source.stat().st_size == row["bytes"] and digest(source) == row["sha256"], "Source file size or bytes differ")
    return entries


def frozen_files(root, cfg):
    return {path: digest(safe(root, path)) for path in sorted(cfg["frozen_files"])}


def register(root, config_path, cache, out):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = read_json(config_path)
    validate_config(cfg)
    evidence, inputs = parent_gate(root, cfg), input_inventory(root, cfg, cache)
    record = {"study_id": cfg["study_id"], "registration_type": "recordwise_conditional_transition_measurement",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "config": cfg, "config_sha256": digest(config_path), "frozen_files": frozen_files(root, cfg),
        "parent_files": {path: digest(safe(root, path)) for path in cfg["parent_files"]}, "input_files": inputs, "gate": evidence,
        "notice": "Prospective private Git freeze of a recordwise descriptive measurement; no blinded hypothesis, learning, pooled species comparison or causal estimate"}
    require(re.fullmatch(r"[a-f0-9]{40}", record["source_git_commit"]), "Valid source Git version required")
    write_new(out, record)
    return record


def streams(root, cfg, inputs, cache):
    from scipy.io import loadmat
    for entry in inputs:
        if entry["role"] == "human_raw":
            raw = loadmat(safe(cache, entry["path"]), variable_names=["A", "R", "S", "C"])
            rows = human_rows(raw, stream_id=f"human/main/{entry['row_index']}", trials=cfg["human"]["trials"],
                              regime_boundary_after_trial=cfg["human"]["regime_boundary_after_trial"])
            for row in rows:
                row["local_trial"] = row["trial"]
            yield rows
        else:
            yield from model_streams(safe(root, entry["path"]), entry["seed"], cfg)


def _artifacts(out):
    return {path.relative_to(out).as_posix(): digest(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"}


def run(root, config_path, registration_path, cache, out):
    root, config_path, registration_path, cache, out = (Path(path).resolve() for path in (root, config_path, registration_path, cache, out))
    cfg, reg = read_json(config_path), read_json(registration_path)
    validate_config(cfg)
    now = datetime.now(timezone.utc)
    require(utc(reg["registered_at_utc"]) <= now and re.fullmatch(r"[a-f0-9]{40}", reg["source_git_commit"]), "Prior UTC registration and source version required")
    require(reg["study_id"] == cfg["study_id"] and reg["registration_type"] == "recordwise_conditional_transition_measurement"
            and exact(reg["config"], cfg) and reg["config_sha256"] == digest(config_path) and reg["frozen_files"] == frozen_files(root, cfg)
            and reg["parent_files"] == {path: digest(safe(root, path)) for path in cfg["parent_files"]}
            and exact(reg["input_files"], input_inventory(root, cfg, cache)) and exact(reg["gate"], parent_gate(root, cfg)),
            "Registered source, semantic evidence, input identities or scope changed")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "registration.json", reg)
    metadata = {"study_id": cfg["study_id"], "status": "running", "started_at_utc": now.isoformat(), "config": cfg,
        "config_sha256": digest(config_path), "source_git_commit": reg["source_git_commit"], "registration_sha256": digest(registration_path),
        "frozen_files": reg["frozen_files"], "parent_files": reg["parent_files"], "input_files": reg["input_files"], "gate": reg["gate"],
        "runtime": {"python": platform.python_version(), "numpy": importlib.metadata.version("numpy"), "scipy": importlib.metadata.version("scipy")},
        "learning_updates": 0, "learner_constructions": 0, "dataset_generations": 0, "rng_draws": 0,
        "streams": 0, "input_trials": 0, "candidates": 0, "eligible": 0, "excluded": 0, "strata": 0,
        "pooled_group_or_species_estimates": False, "averaged_stratum_estimates": False, "causal_effect_estimated": False}
    write_new(out / "metadata.json", metadata)
    reports = []
    try:
        previous = read_json(safe(root, cfg["human"]["prior_metrics"]))
        prior_human = {row["stream_id"]: row for row in previous["streams"] if row["kind"] == "human"}
        require(len(prior_human) == cfg["human"]["record_count"], "Every old human base metric must be retained exactly")
        matched_human = set()
        with (out / "ledger.csv").open("x", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
            writer.writeheader()
            for rows in streams(root, cfg, reg["input_files"], cache):
                summary, ledger = measure_conditional(rows)
                if summary["kind"] == "human":
                    base = {key: summary[key] for key in BASE_SUMMARY_FIELDS}
                    require(summary["stream_id"] in prior_human and exact(base, prior_human[summary["stream_id"]]), "Human base metric differs from its immutable prior measurement")
                    matched_human.add(summary["stream_id"])
                reports.append(summary)
                writer.writerows(ledger)
                metadata["streams"] += 1
                metadata["input_trials"] += summary["trials"]
                metadata["candidates"] += summary["candidates"]
                metadata["eligible"] += summary["eligible"]
                metadata["excluded"] += summary["excluded"]
                metadata["strata"] += len(summary["strata"])
        require(matched_human == set(prior_human) and (metadata["streams"], metadata["input_trials"], metadata["candidates"], metadata["strata"])
                == tuple(cfg[key] for key in ("registered_streams", "registered_input_trials", "registered_candidate_pairs", "registered_strata")),
                "Registered complete record/candidate/cell coverage differs")
        write_new(out / "metrics.json", {"study_id": cfg["study_id"], "streams": reports, "pool_group_or_species_estimates": False,
                                        "averaged_stratum_estimates": False, "causal_effect_estimated": False})
        metadata.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(), human_base_metrics_exact_match=True)
    except Exception as error:
        metadata.update(status="failed", completed_at_utc=datetime.now(timezone.utc).isoformat(), error_type=type(error).__name__, error_message=str(error))
        if reports:
            write_new(out / "partial_metrics.json", {"study_id": cfg["study_id"], "streams": reports})
        (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
        write_new(out / "artifact_hashes.json", _artifacts(out))
        raise
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
    write_new(out / "artifact_hashes.json", _artifacts(out))
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--human-cache", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists() and (args.mode != "run" or args.registration is not None), "Exclusive output and prior registration required")
    result = register(args.root, args.config, args.human_cache, args.out) if args.mode == "register" else run(args.root, args.config, args.registration, args.human_cache, args.out)
    print(json.dumps({"status": result.get("status", "registered"), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
