"""Independent descriptive feedback strata from immutable actual-choice logs.

Only an earlier independent verifier's raw decoder and adjacency arithmetic are
reused. No primary converter, aggregation function or learner is called. Counts
describe eligible observed pairs, not causal reward effects or human similarity.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("immutable_independent_feedback_adjacency", Path(__file__).with_name("verify_feedback_transition.py"))
OLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OLD)
require, digest, read_json, safe = OLD.require, OLD.digest, OLD.read_json, OLD.safe
human_records, adjacent_measure = OLD.human_records, OLD.measure_stream

STUDY = "conditional_transition_v1"
CONDITIONS = ("linear", "fixed", "plastic", "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic", "frozen")
ENVIRONMENTS = ("deterministic", "probabilistic", "uninformative")
MODEL_FIELDS = ("stage", "seed", "condition", "environment", "phase", "trial", "global_trial", "family_id", "left_count", "right_count",
    "chosen_count", "action", "reward", "p_right", "q_reward", "action_uniform", "reward_uniform", "head_update_called", "internal_update_called",
    "optimizer_factor", "internal_step_l2", "readout_step_l2", "kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2", "decision_left_l2",
    "decision_right_l2", "head_updates", "internal_updates")
LEDGER_FIELDS = (*OLD.LEDGER_FIELDS, "phase", "state", "choice", "next_choice", "local_trial", "next_local_trial")
RESOURCE_FIELDS = ("learning_updates", "learner_constructions", "dataset_generations", "rng_draws")


def exact(actual, expected, label):
    require(OLD.exact(actual, expected), label)


def utc(value):
    require(type(value) is str, "Explicit UTC timestamp text")
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, "Timezone-aware UTC timestamp")
    return result


def checksum(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()


def config_check(cfg):
    require(cfg["study_id"] == STUDY and cfg["scope"] in {"registered_recordwise_descriptive_measurement", "integration_test"}, "Study identity and descriptive scope")
    h, m = cfg["human"], cfg["model"]
    require(h["coordinate"] == "source_coded_card_identity" and h["historical_execution_version_verified"] is False
            and m["coordinate"] == "smaller_larger_quantity_category", "Distinct fixed source-coded choice coordinates and historical limit")
    require(m["conditions"] == list(CONDITIONS) and [e["id"] for e in m["environments"]] == list(ENVIRONMENTS), "All eight conditions and three environments")
    exact(m["environments"], [{"id": name, "q_optimal": q1, "q_other": q0} for name, q1, q0 in
        (("deterministic", 1., 0.), ("probabilistic", .8, .2), ("uninformative", .5, .5))], "Fixed actual reward-generating environments")
    for key in RESOURCE_FIELDS:
        require(type(cfg[key]) is int and cfg[key] == 0, "No new resource use: " + key)
    require(all(cfg[key] is False for key in ("pooling", "strata_averaging", "causal_inference"))
            and cfg["zero_denominator"] is None and cfg["exclude_boundaries"] == ["state", "phase", "task"], "No pooling, causal inference or missing-cell replacement")
    require(cfg["common_support"] == "both_actual_feedback_denominators_positive_within_predefined_phase_state_current_choice_cell"
            and cfg["metric"] == "recordwise_actual_adjacent_stay_switch_counts_by_feedback_and_predefined_phase_state_current_choice", "Fixed descriptive support and strata metric")
    for key in ("frozen_files", "parent_files"):
        require(type(cfg[key]) is list and len(set(cfg[key])) == len(cfg[key]) and cfg[key], "Explicit unique source/parent closure")
    require(len(cfg["parent_files"]) == 17 and len(cfg["frozen_files"]) == 16, "Exact16-source and17-parent closure")
    seeds, hc, hn, boundary, mn = m["seeds"], h["record_count"], h["trials"], h["regime_boundary_after_trial"], m["trials_per_phase"]
    require(type(seeds) is list and seeds and len(set(seeds)) == len(seeds) and all(type(s) is int for s in seeds)
            and all(type(n) is int and n > 0 for n in (hc, hn, boundary, mn)) and boundary < hn, "Fixed unique unit and trial budgets")
    if cfg["scope"] == "registered_recordwise_descriptive_measurement":
        require(seeds == list(range(11001, 11011)) and (hc, hn, boundary, mn) == (25, 1000, 800, 400), "All previously verified human and model records")
    else:
        require(set(seeds) <= {17, 18} and hc == 2 and (hn, boundary, mn) == (8, 4, 4), "Handwritten two-record eight-trial engineering fixtures only")
    streams = hc + len(seeds) * 24
    trials = hc * hn + len(seeds) * 24 * 2 * mn
    counts = {"registered_streams": streams, "registered_input_trials": trials,
              "registered_candidate_pairs": trials - streams, "registered_strata": hc * 8 + len(seeds) * 24 * 4}
    for key, value in counts.items():
        require(type(cfg[key]) is int and cfg[key] == value, "Complete descriptive resource denominator: " + key)


def strata_coordinates(kind):
    if kind == "human":
        return [(phase, state, choice) for phase in ("initial", "transfer") for state in ("1", "2") for choice in (0, 1)]
    require(kind == "model", "Known canonical source kind")
    return [(phase, state, choice) for phase, state in (("phase1", "larger"), ("phase2", "smaller")) for choice in (0, 1)]


def measure(stream_id, kind, coordinate, records):
    """Reconstruct all fixed strata directly from the independent pair ledger."""
    design = strata_coordinates(kind)
    allowed = {(phase, state) for phase, state, _ in design}
    require(all(type(r.get("local_trial")) is int and r["local_trial"] > 0 and (r["phase"], r["state"]) in allowed for r in records),
            "Known local trials and predefined phase/state coordinates for every original row")
    base, prior_ledger = adjacent_measure(stream_id, kind, coordinate, records)
    strata = {key: {"phase": key[0], "state": key[1], "choice": key[2],
        "by_feedback": {str(f): {"denominator": 0, "stay": 0, "switch": 0} for f in (0, 1)}} for key in strata_coordinates(kind)}
    ledger = []
    for previous, current, item in zip(records[:-1], records[1:], prior_ledger, strict=True):
        row = {**item, "phase": previous["phase"], "state": previous["state"],
            "choice": "" if previous["choice"] is None else str(previous["choice"]),
            "next_choice": "" if current["choice"] is None else str(current["choice"]),
            "local_trial": str(previous["local_trial"]), "next_local_trial": str(current["local_trial"])}
        ledger.append(row)
        if item["eligible"] == "1":
            key = previous["phase"], previous["state"], previous["choice"]
            require(key in strata, "Eligible pair has a registered absolute-choice stratum")
            cell = strata[key]["by_feedback"][str(previous["feedback"])]
            stay = int(previous["choice"] == current["choice"])
            cell["denominator"] += 1
            cell["stay"] += stay
            cell["switch"] += 1 - stay
    for row in strata.values():
        for cell in row["by_feedback"].values():
            d = cell["denominator"]
            cell["stay_rate"] = cell["stay"] / d if d else None
            cell["switch_rate"] = cell["switch"] / d if d else None
        row["common_support"] = all(cell["denominator"] > 0 for cell in row["by_feedback"].values())
        present = tuple(f for f in ("0", "1") if row["by_feedback"][f]["denominator"] > 0)
        row["support_status"] = {(): "neither", ("0",): "feedback_0_only", ("1",): "feedback_1_only",
                                 ("0", "1"): "both_feedback"}[present]
        row["stay_rate_difference_1_minus_0"] = (row["by_feedback"]["1"]["stay_rate"] - row["by_feedback"]["0"]["stay_rate"]
                                                 if row["common_support"] else None)
    for f in ("0", "1"):
        for field in ("denominator", "stay", "switch"):
            require(sum(row["by_feedback"][f][field] for row in strata.values()) == base["by_feedback"][f][field],
                    "All eligible pairs appear once in absolute-choice strata")
    return {**base, "strata": list(strata.values())}, ledger


def parse_int(row, key, minimum=0):
    return OLD.integer(row[key], minimum)


def finite(row, key):
    require(type(row[key]) is str and row[key] and row[key] == row[key].strip(), "Canonical finite CSV token")
    value = float(row[key])
    require(math.isfinite(value), "Finite original CSV number: " + key)
    return value


def model_records(path, seed, *, trials_per_phase, conditions=CONDITIONS, environments=ENVIRONMENTS):
    """Read all actual actions using stable small/large coordinates, never p argmax."""
    streams = {(condition, environment): [] for condition in conditions for environment in environments}
    seen = set()
    with Path(path).open(encoding="utf8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == MODEL_FIELDS, "Exact independently verified stochastic CSV schema")
        for position, row in enumerate(reader, 2):
            require(None not in row and all(v is not None for v in row.values()), "Nonragged original actual-action CSV")
            require(row["stage"] == "network" and parse_int(row, "seed") == seed and row["phase"] in ("phase1", "phase2"), "Original stochastic source identity")
            condition, environment, phase = row["condition"], row["environment"], row["phase"]
            require((condition, environment) in streams, "Every original condition/environment retained")
            local = parse_int(row, "trial", 1)
            global_trial = parse_int(row, "global_trial", 1)
            offset = 0 if phase == "phase1" else trials_per_phase
            require(local <= trials_per_phase and global_trial == offset + local, "Exact original local/global phase coordinates")
            key = condition, environment, phase, local
            require(key not in seen, "Duplicate original stochastic trial")
            seen.add(key)
            left, right, action, reward, chosen = (parse_int(row, key) for key in ("left_count", "right_count", "action", "reward", "chosen_count"))
            require({left, right} == {2, 4} and action in (0, 1) and reward in (0, 1) and chosen == (left, right)[action], "Actual action and chosen quantity")
            require(type(row["family_id"]) is str and row["family_id"], "Known original stimulus family identifier")
            probability, action_u, reward_u, q = (finite(row, key) for key in ("p_right", "action_uniform", "reward_uniform", "q_reward"))
            require(0 <= probability <= 1 and 0 <= action_u < 1 and 0 <= reward_u < 1 and action == int(action_u < probability), "Actual sampled action, never greedy/expected action")
            choice = int(chosen == 4)
            target = 1 if phase == "phase1" else 0
            q1, q0 = {"deterministic": (1., 0.), "probabilistic": (.8, .2), "uninformative": (.5, .5)}[environment]
            require(q == (q1 if choice == target else q0) and reward == int(reward_u < q), "Logged actual sampled feedback, including stochastic unexpected outcomes")
            for field in MODEL_FIELDS[19:]:
                finite(row, field)
            for field in ("head_update_called", "internal_update_called"):
                require(parse_int(row, field) in (0, 1), "Integral actual update flag")
            for field in ("head_updates", "internal_updates"):
                parse_int(row, field)
            streams[condition, environment].append({"source_position": position, "trial": global_trial, "local_trial": local,
                "choice": choice, "feedback": reward, "task": "abstract_quantity", "phase": phase,
                "state": "larger" if phase == "phase1" else "smaller", "actual": True, "training": True})
    for key, records in streams.items():
        require([r["trial"] for r in records] == list(range(1, 2 * trials_per_phase + 1)), "Complete800-trial stream in original order")
    require(len(seen) == len(streams) * 2 * trials_per_phase, "Every actual source trial represented once")
    return streams


def parent_check(root, cfg, registration):
    expected_paths = {"configs/stochastic_feedback_network_v1.json", "results/registration_stochastic_feedback_network_v1.json",
        *("results/stochastic_feedback_network_v1/" + n + ".json" for n in ("metadata", "summary", "verification", "artifact_hashes")),
        "configs/feedback_transition_v1.json", "results/registration_feedback_transition_v1.json",
        *("results/feedback_transition_v1/" + n + ".json" for n in ("metadata", "metrics", "verification", "artifact_hashes")),
        "configs/human_osf_origin_manifest.json", "results/human_task_structure_v1/audit.json",
        "results/human_task_structure_v1/verification.json", "results/registration_human_task_structure_v1.json",
        "results/human_osf_origin_v1/verification.json"}
    require(set(cfg["parent_files"]) == expected_paths, "Exactly the17 required completed parent files")
    identity = cfg["parent_gate_identity"]
    require(type(identity) is dict and set(identity) == {"published_commit", "file_sha256"}
            and type(identity["published_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", identity["published_commit"])
            and set(identity["file_sha256"]) == set(cfg["parent_files"]), "Complete pinned completed-parent publication identity")
    for name, sha in identity["file_sha256"].items():
        require(checksum(sha) and digest(safe(root, name)) == sha, "Exact published parent evidence bytes: " + name)
    h, m = cfg["human"], cfg["model"]
    require(h["origin_manifest"] == "configs/human_osf_origin_manifest.json"
            and h["structure_result"] == "results/human_task_structure_v1/audit.json"
            and h["structure_verification"] == "results/human_task_structure_v1/verification.json"
            and h["prior_metrics"] == "results/feedback_transition_v1/metrics.json"
            and h["prior_verification"] == "results/feedback_transition_v1/verification.json"
            and m["root"] == "results/stochastic_feedback_network_v1"
            and m["parent_config"] == "configs/stochastic_feedback_network_v1.json"
            and m["verification"] == m["root"] + "/verification.json" and m["artifact_hashes"] == m["root"] + "/artifact_hashes.json", "Fixed selected parent evidence paths")
    structure, structural = read_json(safe(root, h["structure_result"])), read_json(safe(root, h["structure_verification"]))
    require(structure["status"] == "pass" and structure["gate"]["source_coded_descriptive_metric_allowed"] is True
            and structural["status"] == "pass" and structural["source_coded_descriptive_gate_passed"] is True
            and structural["result_sha256"] == digest(safe(root, h["structure_result"]))
            and structural["registration_sha256"] == digest(safe(root, "results/registration_human_task_structure_v1.json")), "Independent exact source-coded human structure gate")
    exact(structure["gate"], {"source_coded_descriptive_metric_allowed": True,
        "exact_historical_task_execution_version_verified": False, "direct_human_model_similarity_allowed": False}, "Preserved human semantic limits")
    origin = read_json(safe(root, h["origin_manifest"]))
    origin_check = read_json(safe(root, "results/human_osf_origin_v1/verification.json"))
    require(origin_check["status"] == "passed" and origin_check["source_identity_gate_passed"] is True
            and origin_check["manifest_sha256"] == digest(safe(root, h["origin_manifest"])), "Exact partial original-human source identity gate")
    old_reg = read_json(safe(root, "results/registration_feedback_transition_v1.json"))
    old_meta = read_json(safe(root, "results/feedback_transition_v1/metadata.json"))
    old_check = read_json(safe(root, h["prior_verification"]))
    require(old_meta["status"] == "completed" and old_check["status"] == "passed" and old_check["errors"] == []
            and old_check["registration_sha256"] == old_meta["registration_sha256"] == digest(safe(root, "results/registration_feedback_transition_v1.json"))
            and old_check["config_sha256"] == old_meta["config_sha256"] == old_reg["config_sha256"] == digest(safe(root, "configs/feedback_transition_v1.json")), "Previous human transition result binds exact registration and config")
    raw = [e for e in origin["source_files"] if e["role"] == "osf_raw"]
    require(origin["trials_per_row"] == h["trials"] and len(raw) == h["record_count"]
            and len({e["derived_row_index"] for e in raw}) == len({e["osf_file_id"] for e in raw}) == len(raw)
            and all(e["derived_group"] == "main" for e in raw), "All source-verified Main original records without performance selection")
    expected_structure = [{key: e[key] for key in ("path", "bytes", "sha256", "derived_group", "derived_row_index", "osf_file_id")} for e in raw]
    exact(structure["inputs"], expected_structure, "Same original human file identity and order as structure gate")
    human_entries = [e for e in old_reg["inputs"] if e["role"] == "human_raw"]
    expected_human = [{"base": "human_cache", "role": "human_raw", **{k: e[k] for k in ("path", "bytes", "sha256", "osf_file_id")},
                       "row_index": e["derived_row_index"]} for e in raw]
    exact(human_entries, expected_human, "Unchanged old human transition source identities")
    exact([e for e in old_meta["inputs"] if e["role"] == "human_raw"], human_entries, "Previous executed human sources")
    previous = read_json(safe(root, h["prior_metrics"]))
    prior_human = {e["stream_id"]: e for e in previous["streams"] if e["kind"] == "human"}
    require(len(prior_human) == len(human_entries) and set(prior_human) == {f"human/main/{e['row_index']}" for e in human_entries}, "Every previously verified human base summary retained")
    model_config = read_json(safe(root, m["parent_config"]))
    model_meta = read_json(safe(root, m["root"] + "/metadata.json"))
    model_check = read_json(safe(root, m["verification"]))
    model_reg_path = safe(root, "results/registration_stochastic_feedback_network_v1.json")
    model_reg = read_json(model_reg_path)
    require(model_meta["status"] == "completed" and model_meta["stage"] == "network" and model_check["status"] == "passed"
            and model_check["passed"] is True and model_check["errors"] == []
            and model_check["summary_sha256"] == digest(safe(root, m["root"] + "/summary.json"))
            and model_check["registration_sha256"] == model_meta["registration_sha256"] == digest(model_reg_path)
            and model_check["config_sha256"] == model_meta["config_sha256"] == model_reg["config_sha256"] == digest(safe(root, m["parent_config"])), "Exact completed stochastic model and independent replay gate")
    exact(model_meta["config"], model_config, "Executed model parent configuration")
    exact(model_reg["config"], model_config, "Registered model parent configuration")
    require(model_meta["source_git_commit"] == model_reg["source_git_commit"], "Original model source version lineage")
    require(model_config["seeds"] == m["seeds"] and model_config["conditions"] == m["conditions"]
            and model_config["environments"] == m["environments"] and model_config["trials_per_phase"] == m["trials_per_phase"], "Complete original stochastic scope and trial budget")
    inventories = {}
    for base, inventory_path, scientific_count in ((m["root"], m["artifact_hashes"], 2457),
            ("results/feedback_transition_v1", "results/feedback_transition_v1/artifact_hashes.json", 3)):
        artifacts = read_json(safe(root, inventory_path))
        require(type(artifacts) is dict and artifacts, "Complete parent primary inventory")
        for name, sha in artifacts.items():
            require(checksum(sha) and digest(safe(root, base + "/" + name)) == sha, "Every parent primary artifact byte identity")
        if cfg["scope"] == "registered_recordwise_descriptive_measurement":
            require(len(artifacts) == scientific_count, "Complete published parent primary artifact denominator")
        inventories[base] = artifacts
    require(inventories[m["root"]]["summary.json"] == model_check["summary_sha256"]
            and inventories[m["root"]]["registration.json"] == digest(model_reg_path), "Pinned model inventory binds summary and registration")
    now = utc(registration["registered_at_utc"])
    for checked in (structural, origin_check, old_check, model_check):
        require(utc(checked["verified_at_utc"]) <= now, "Every completed independent prerequisite precedes new registration")
    expected_gate = {"name": "source_verified_human_and_actual_network_recordwise_measurement", "passed": True,
        **identity, "historical_task_execution_version_verified": False, "direct_human_model_similarity_allowed": False}
    exact(registration["gate"], expected_gate, "Recorded source-coded semantic gate and exact pinned publication")
    return human_entries, prior_human, inventories


def source_check(root, reg_path, output, cache):
    require(root.is_dir() and reg_path.is_file() and output.is_dir() and cache.is_dir() and not cache.is_relative_to(root), "Existing sources/output and external human cache")
    reg, meta = read_json(reg_path), read_json(output / "metadata.json")
    cfg = reg["config"]
    config_check(cfg)
    require(reg["study_id"] == meta["study_id"] == STUDY and meta["status"] == "completed"
            and reg["registration_type"] == "recordwise_conditional_transition_measurement", "Completed registered descriptive measurement")
    exact(meta["config"], cfg, "Executed registered configuration")
    require(type(reg["source_git_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", reg["source_git_commit"])
            and meta["source_git_commit"] == reg["source_git_commit"], "Recorded source version identity")
    path = safe(root, "configs/conditional_transition_v1.json")
    require(digest(path) == reg["config_sha256"] == meta["config_sha256"], "Exact config bytes")
    exact(read_json(path), cfg, "Source configuration contents")
    require(digest(reg_path) == digest(output / "registration.json") == meta["registration_sha256"], "External/copied registration bytes")
    for field in ("frozen_files", "parent_files"):
        require(set(reg[field]) == set(cfg[field]), "Complete source/parent inventory")
        for name, sha in reg[field].items():
            require(checksum(sha) and digest(safe(root, name)) == sha, "Frozen source/parent byte identity: " + name)
        exact(meta[field], reg[field], "Executed " + field)
    for name in ("scripts/verify_conditional_transition.py", "scripts/verify_feedback_transition.py"):
        require(digest(safe(ROOT, name)) == reg["frozen_files"][name], "Executed independent verifier/helper bytes match source freeze")
    require(utc(reg["registered_at_utc"]) <= utc(meta["started_at_utc"]) <= utc(meta["completed_at_utc"])
            <= datetime.now(timezone.utc), "Registration precedes completed measurement without future timestamps")
    require(set(meta["runtime"]) == {"python", "numpy", "scipy"} and all(type(v) is str and v for v in meta["runtime"].values()), "Recorded decoder runtime")
    for field in RESOURCE_FIELDS:
        require(type(meta[field]) is int and meta[field] == 0, "No new learning or data generation: " + field)
    for field in ("pooled_group_or_species_estimates", "averaged_stratum_estimates", "causal_effect_estimated"):
        require(meta[field] is False, "No inferential interpretation: " + field)
    require(meta["human_base_metrics_exact_match"] is True, "Human base measurements remain unchanged")
    return cfg, reg, meta


def input_check(root, cache, cfg, reg, meta, human_entries, inventories):
    expected = list(human_entries)
    for seed in cfg["model"]["seeds"]:
        name = f"seed_{seed}/trials.csv"
        full = cfg["model"]["root"] + "/" + name
        path = safe(root, full)
        expected.append({"base": "repo", "role": "model_actual", "path": full, "bytes": path.stat().st_size,
                         "sha256": inventories[cfg["model"]["root"]][name], "seed": seed})
    exact(reg["input_files"], expected, "All original raw human and actual-model files without selection")
    exact(meta["input_files"], expected, "Executed input identities")
    exact(meta["gate"], reg["gate"], "Executed completed-parent semantic gate")
    require(len({(e["base"], e["path"]) for e in expected}) == len(expected), "Unique input identities")
    for entry in expected:
        path = safe(cache if entry["base"] == "human_cache" else root, entry["path"])
        require(type(entry["bytes"]) is int and entry["bytes"] > 0 and checksum(entry["sha256"])
                and path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], "Registered original input bytes and size")
    return expected


def audit(root, reg_path, output, cache):
    cfg, reg, meta = source_check(root, reg_path, output, cache)
    human_entries, prior_human, inventories = parent_check(root, cfg, reg)
    entries = input_check(root, cache, cfg, reg, meta, human_entries, inventories)
    hashes = read_json(output / "artifact_hashes.json")
    expected_artifacts = {"metadata.json", "registration.json", "metrics.json", "ledger.csv"}
    require(set(hashes) == expected_artifacts, "Complete exact primary output inventory")
    present = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} - {"artifact_hashes.json", "verification.json"}
    require(present == expected_artifacts, "No unregistered, missing or partial output")
    for name, sha in hashes.items():
        require(checksum(sha) and digest(safe(output, name)) == sha, "Every primary output exact SHA")
    summaries, pairs = [], 0
    with (output / "ledger.csv").open(encoding="utf8", newline="") as stream:
        reader = csv.reader(stream)
        require(next(reader, None) == list(LEDGER_FIELDS), "Exact enriched ledger schema")
        for entry in entries:
            if entry["role"] == "human_raw":
                decoded = human_records(safe(cache, entry["path"]), f"human/main/{entry['row_index']}",
                    cfg["human"]["trials"], cfg["human"]["regime_boundary_after_trial"])
                for row in decoded[3]:
                    row["local_trial"] = row["trial"]
                sources = [decoded]
            else:
                records = model_records(safe(root, entry["path"]), entry["seed"], trials_per_phase=cfg["model"]["trials_per_phase"],
                    conditions=tuple(cfg["model"]["conditions"]), environments=tuple(e["id"] for e in cfg["model"]["environments"]))
                sources = [(f"model/{entry['seed']}/{condition}/{environment}", "model", cfg["model"]["coordinate"], rows)
                           for (condition, environment), rows in records.items()]
            for stream_id, kind, coordinate, records in sources:
                summary, ledger = measure(stream_id, kind, coordinate, records)
                if kind == "human":
                    exact({k: v for k, v in summary.items() if k != "strata"}, prior_human[stream_id], "Exact unchanged previous human base measurement")
                summaries.append(summary)
                for expected in ledger:
                    actual = next(reader, None)
                    require(actual == [expected[field] for field in LEDGER_FIELDS], "Independent original adjacency, query/missing exclusions, source coordinates and absolute choice ledger")
                    pairs += 1
        require(next(reader, None) is None, "No extra or duplicated candidate ledger rows")
    metrics = {"study_id": STUDY, "streams": summaries, "pool_group_or_species_estimates": False,
               "averaged_stratum_estimates": False, "causal_effect_estimated": False}
    exact(read_json(output / "metrics.json"), metrics, "Every recordwise count, null cell, support flag/status and within-stratum difference")
    counts = {"streams": len(summaries), "input_trials": sum(r["trials"] for r in summaries), "candidates": pairs,
              "eligible": sum(r["eligible"] for r in summaries), "excluded": sum(r["excluded"] for r in summaries),
              "strata": sum(len(r["strata"]) for r in summaries)}
    for key, value in counts.items():
        require(type(meta[key]) is int and meta[key] == value, "Every complete raw/eligible/cell denominator: " + key)
    require(counts["candidates"] == counts["eligible"] + counts["excluded"], "Excluded reason overlap does not double-count pairs")
    for key in ("streams", "input_trials", "candidate_pairs", "strata"):
        actual = counts["candidates" if key == "candidate_pairs" else key]
        require(actual == cfg["registered_" + key], "All prospectively selected records and cells remain")
    require(len({r["stream_id"] for r in summaries}) == len(summaries), "One summary per original stream")
    return {"study_id": STUDY, "scope": "independent_recordwise_conditional_transition_no_causal_or_species_inference",
        "counts": {**counts, "feedback_cells": 2 * counts["strata"], "human_records": len(human_entries),
            "model_streams": len(summaries) - len(human_entries), "source_files": len(reg["frozen_files"]),
            "parent_files": len(reg["parent_files"]), "input_files": len(entries), "primary_artifacts": len(hashes),
            "parent_primary_artifacts": sum(len(v) for v in inventories.values()), **{k: 0 for k in RESOURCE_FIELDS}},
        "metrics_sha256": digest(output / "metrics.json"), "registration_sha256": digest(reg_path), "config_sha256": reg["config_sha256"],
        "artifact_hashes_sha256": digest(output / "artifact_hashes.json"), "input_manifest_sha256": canonical_digest(reg["input_files"]),
        "human_base_metrics_exact_match": True, "pool_group_or_species_estimates": False,
        "averaged_stratum_estimates": False, "causal_effect_estimated": False,
        "checks": ["Exact17-parent identities, all parent primary artifacts and all raw input bytes",
            "Independent human raw decoding and exact unchanged previously verified base measurements",
            "Every actual stochastic choice/reward, global/local phase coordinate and complete source CSV scope",
            "Every original adjacent pair, query/missing/boundary exclusion and absolute-choice ledger row",
            "Full8-human/4-model designed cells, exact integer partitions, null rates, support status and defined differences",
            "No averaging, pooling, new learning, generation or causal/species interpretation"]}


def verify(root, registration, result_dir, human_cache):
    paths = tuple(Path(p).resolve() for p in (root, registration, result_dir, human_cache))
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "status": "failed", "passed": False, "study_id": STUDY,
        "errors": [], "counts": {}, "not_verified": ["Historical execution-code identity beyond source-coded human semantics",
            "Adequate precision, randomized human feedback or causal identification from common_support",
            "Human/model similarity, shared neural mechanisms, education effects or new learning",
            "Authenticity of private remote publication beyond pinned local bytes (separate archive evidence)"]}
    try:
        report.update(audit(*paths))
        report.update(status="passed", passed=True)
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--human-cache", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Verification output already exists")
    if args.config is not None:
        require(args.config.resolve() == (args.root / "configs/conditional_transition_v1.json").resolve(), "Canonical source configuration path")
    report = verify(args.root, args.registration, args.result_dir, args.human_cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "errors": report["errors"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
