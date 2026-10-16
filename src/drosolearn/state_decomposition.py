"""Read-only crossed-state algebra; no learner, dataset generator or RNG calls.

The crossed parameter states are static and generally off trajectory. Algebraic
terms are not causal contributions of learning or independent neural mechanisms.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODELS = ["linear", "real", "rewired", "ann"]
BRANCHES = ["full", "head_only", "internal_only", "frozen"]
CROSSES = ("00", "10", "01", "11")
SNAPSHOT_FIELDS = {"internal_weights", "readout_weights", "kc_activity", "mbon_activity", "decision_activity", "choice_probability",
    "feature_scale", "initial_feature_gap_rms", "encoder", "bias", "pre", "post", "head_updates", "internal_updates",
    "initial_internal_weights", "internal_upper_bounds", "incoming_normalizer", "iid_p_right", "ood_p_right",
    "phase_head_updates", "phase_internal_updates"}
FIXED_FIELDS = ("encoder", "bias", "pre", "post", "feature_scale", "initial_feature_gap_rms", "initial_internal_weights",
                "internal_upper_bounds", "incoming_normalizer")
VALUE_FIELDS = tuple(f"raw_gap_{cross}" for cross in CROSSES) + (
    "head_term", "internal_term", "cross_term", "total_raw_gap_change", "bilinear_residual", "raw_interaction", "raw_interaction_residual",
) + tuple(f"aligned_raw_gap_{cross}" for cross in CROSSES) + (
    "aligned_head_term", "aligned_internal_term", "aligned_cross_term", "aligned_total_raw_gap_change",
) + tuple(f"p_right_{cross}" for cross in CROSSES) + tuple(f"p_rule_{cross}" for cross in CROSSES) + (
    "policy_interaction_right", "policy_interaction_rule", "policy_rule_change",
)
CASE_FIELDS = ("seed", "model", "branch", "trial", "split", "case", "family_id", "left_count", "right_count", "target_action", "target_sign",
               *VALUE_FIELDS, *(f"clipped_{cross}" for cross in CROSSES))
SUMMARY_FIELDS = tuple(field for field in VALUE_FIELDS if field not in ("bilinear_residual", "raw_interaction_residual"))


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


def safe(root, relative):
    require(type(relative) is str and relative and "\\" not in relative and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts, "Safe POSIX relative path required")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Input path escapes project")
    return path


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def validate_config(cfg):
    require(cfg["study_id"] == "state_decomposition_v1" and cfg["scope"] in ("registered_descriptive_diagnostic", "integration_test"),
            "Fixed static diagnostic identity required")
    require(cfg["models"] == MODELS and cfg["branches"] == BRANCHES and cfg["pair"] == [2, 4]
            and cfg["baseline_phase"] == "phase1" and cfg["target_phase"] == "phase2"
            and cfg["target"] == "smaller_quantity_all_crossed_states", "Fixed original states and shared evaluation target required")
    seeds, checkpoints, ev = cfg["seeds"], cfg["checkpoints"], cfg["eval_trials"]
    require(type(seeds) is list and seeds and all(type(seed) is int for seed in seeds) and len(set(seeds)) == len(seeds)
            and type(checkpoints) is list and checkpoints and all(type(t) is int and t >= 0 for t in checkpoints)
            and checkpoints == sorted(set(checkpoints)) and checkpoints[0] == 0 and type(ev) is int and ev > 0
            and type(cfg["baseline_trial"]) is int and cfg["baseline_trial"] > 0 and checkpoints[-1] == cfg["baseline_trial"],
            "Unique seed, checkpoint and evaluation coverage required")
    require(cfg["raw_residual_tolerance"] == 1e-10 and cfg["policy_match_tolerance"] == 1e-12 and cfg["policy_clip"] == [-40., 40.],
            "Fixed arithmetic tolerances and clipped policy required")
    require(all(type(cfg[key]) is int and cfg[key] == 0 for key in ("learning_updates", "learner_constructions", "dataset_generations", "rng_draws"))
            and cfg["all_registered_inputs_retained"] is True
            and cfg["interpretation"] == "static_off_trajectory_parameter_algebra_not_learning_causal_effect", "No-learning and interpretation scope required")
    count = len(seeds) * len(MODELS) * len(BRANCHES) * len(checkpoints) * 2 * ev
    inputs = len(seeds) * (1 + len(MODELS) + len(MODELS) * len(BRANCHES) * len(checkpoints))
    require(all(type(cfg[key]) is int for key in ("cases", "crossed_policy_values", "selected_input_files"))
            and (cfg["cases"], cfg["crossed_policy_values"], cfg["selected_input_files"]) == (count, 4 * count, inputs),
            "Full case/crossed-state/input budget arithmetic differs")
    if cfg["scope"] == "registered_descriptive_diagnostic":
        require(seeds == list(range(12001, 12011)) and checkpoints == [0, 100, 200, 300, 400]
                and cfg["baseline_trial"] == 400 and ev == 128, "All previously learned scientific states must be retained")
    else:
        require(set(seeds) <= {17, 18} and checkpoints == [0, 4, 8] and cfg["baseline_trial"] == ev == 8,
                "Only existing eight-trial seed17/18 engineering states may be reused")
    require(cfg["parent_root"] == "results/learning_ablation_v1" and cfg["parent_config"] == "configs/learning_ablation_v1.json",
            "Fixed immutable parent study required")
    for key in ("frozen_files", "parent_files"):
        require(type(cfg[key]) is list and cfg[key] and len(set(cfg[key])) == len(cfg[key]), "Explicit unique source/parent inventories required")


def selected_paths(cfg):
    base, paths = cfg["parent_root"], []
    for seed in cfg["seeds"]:
        seed_path = f"{base}/seed_{seed}"
        paths.append(seed_path + "/stimuli.npz")
        for model in MODELS:
            paths.append(f"{seed_path}/{model}/prefix/{cfg['baseline_phase']}_{cfg['baseline_trial']:04d}.npz")
            for branch in BRANCHES:
                for trial in cfg["checkpoints"]:
                    paths.append(f"{seed_path}/{model}/{branch}/{cfg['target_phase']}_{trial:04d}.npz")
    require(len(paths) == len(set(paths)) == cfg["selected_input_files"], "One declared NPZ input per identity required")
    return paths


def parent_gate(root, cfg):
    identity = cfg["parent_gate_identity"]
    require(type(identity) is dict and set(identity) == {"published_commit", "file_sha256"}
            and type(identity["published_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", identity["published_commit"])
            and type(identity["file_sha256"]) is dict and set(identity["file_sha256"]) == set(cfg["parent_files"]),
            "Exact published parent evidence identity required")
    for path, checksum in identity["file_sha256"].items():
        require(type(checksum) is str and re.fullmatch(r"[a-f0-9]{64}", checksum) and digest(safe(root, path)) == checksum,
                "Published parent evidence bytes changed")
    base = cfg["parent_root"]
    metadata, checked = read_json(safe(root, base + "/metadata.json")), read_json(safe(root, base + "/verification.json"))
    summary, artifacts = read_json(safe(root, base + "/summary.json")), read_json(safe(root, base + "/artifact_hashes.json"))
    parent_config = read_json(safe(root, cfg["parent_config"]))
    parent_registration = read_json(safe(root, "results/registration_learning_ablation_v1.json"))
    require(metadata["study_id"] == summary["study_id"] == "learning_ablation_v1" and metadata["status"] == summary["status"] == "completed"
            and checked["status"] == "passed" and checked["summary_sha256"] == digest(safe(root, base + "/summary.json"))
            and checked["registration_sha256"] == digest(safe(root, "results/registration_learning_ablation_v1.json"))
            and checked["config_sha256"] == digest(safe(root, cfg["parent_config"])), "Completed, independently verified exact parent required")
    require(metadata["config"] == parent_config and metadata["config_sha256"] == checked["config_sha256"]
            and metadata["registration_sha256"] == checked["registration_sha256"]
            and parent_registration["config"] == parent_config and parent_registration["config_sha256"] == checked["config_sha256"]
            and parent_registration["source_git_commit"] == metadata["source_git_commit"]
            and checked["artifact_hashes_sha256"] == digest(safe(root, base + "/artifact_hashes.json")),
            "Parent metadata, source version and artifact identity must match its independent verification")
    parent_time = datetime.fromisoformat(checked["verified_at_utc"])
    require(parent_time.tzinfo is not None and parent_time.utcoffset().total_seconds() == 0
            and parent_time <= datetime.now(timezone.utc), "Parent independent verification must precede this diagnostic")
    correction = read_json(safe(root, "results/learning_ablation_documentation_correction_v1.json"))
    final = read_json(safe(root, "results/learning_ablation_final_validation_v1.json"))
    input_freeze = read_json(safe(root, "results/learning_ablation_input_freeze_v1.json"))
    require(correction["status"] == "documentation_corrected" and correction["corrected_document"] == "docs/learning-ablation-findings-v2.md"
            and correction["corrected_sha256"] == digest(safe(root, correction["corrected_document"]))
            and correction["summary_sha256"] == checked["summary_sha256"]
            and correction["verification_sha256"] == digest(safe(root, base + "/verification.json"))
            and correction["scientific_values_or_criteria_changed"] is False and correction["additional_learning_updates"] == 0
            and final["status"] == "passed" and final["study_id"] == "learning_ablation_v1"
            and final["documentation_correction"] == correction
            and input_freeze["study_id"] == "learning_ablation_v1"
            and input_freeze["registration_sha256"] == checked["registration_sha256"]
            and input_freeze["config_sha256"] == checked["config_sha256"]
            and input_freeze["source_commit"] == metadata["source_git_commit"]
            and input_freeze["learning_updates"] == input_freeze["learner_constructions"] == 0,
            "Corrected parent documentation, final validation and original input freeze must remain linked")
    require(type(artifacts) is dict and artifacts and all(type(path) is str and type(checksum) is str and re.fullmatch(r"[a-f0-9]{64}", checksum)
            for path, checksum in artifacts.items()), "Complete parent artifact inventory required")
    for path, checksum in artifacts.items():
        require(digest(safe(root, base + "/" + path)) == checksum, "A recorded parent artifact changed")
    if cfg["scope"] == "registered_descriptive_diagnostic":
        require(len(artifacts) == 1058, "All1058 completed parent primary artifacts required")
    return {"name": "verified_learning_ablation_before_static_decomposition", "passed": True,
            "published_commit": identity["published_commit"], "file_sha256": identity["file_sha256"],
            "parent_primary_artifacts_verified": len(artifacts)}


def input_inventory(root, cfg):
    artifacts = read_json(safe(root, cfg["parent_root"] + "/artifact_hashes.json"))
    result = []
    for path in selected_paths(cfg):
        relative = path.removeprefix(cfg["parent_root"] + "/")
        source = safe(root, path)
        require(relative in artifacts and digest(source) == artifacts[relative], "Selected original NPZ does not match parent publication")
        result.append({"path": path, "bytes": source.stat().st_size, "sha256": artifacts[relative]})
    return result


def frozen_files(root, cfg):
    return {path: digest(safe(root, path)) for path in sorted(cfg["frozen_files"])}


def load_npz(path, fields=None):
    with np.load(path, allow_pickle=False) as handle:
        require(fields is None or set(handle.files) == set(fields), "Original NPZ schema differs")
        return {key: handle[key].copy() for key in handle.files}


def _snapshot_check(snapshot):
    require(set(snapshot) == SNAPSHOT_FIELDS and all(isinstance(value, np.ndarray) and value.dtype.kind in "iuf"
            and np.isfinite(value).all() for value in snapshot.values()), "Finite original state arrays required")
    h, weights = snapshot["readout_weights"], snapshot["internal_weights"]
    require(h.ndim == weights.ndim == 1 and h.size > 0 and snapshot["feature_scale"].shape == ()
            and float(snapshot["feature_scale"]) > 0 and snapshot["initial_feature_gap_rms"].shape == (), "Valid fixed positive scalar feature scale required")
    require(snapshot["pre"].dtype.kind in "iu" and snapshot["post"].dtype.kind in "iu"
            and snapshot["pre"].shape == snapshot["post"].shape == weights.shape, "Anatomical sparse index schema differs")
    encoder, bias = snapshot["encoder"], snapshot["bias"]
    require(encoder.ndim == 2 and encoder.shape[1] == 7 and bias.shape == (len(encoder),), "Fixed input encoder schema differs")
    if weights.size:
        require((snapshot["pre"] >= 0).all() and (snapshot["pre"] < len(encoder)).all()
                and (snapshot["post"] >= 0).all() and (snapshot["post"] < len(h)).all()
                and snapshot["internal_upper_bounds"].shape == snapshot["initial_internal_weights"].shape == weights.shape
                and snapshot["incoming_normalizer"].shape == h.shape, "Normalized anatomical state schema differs")
    else:
        require(len(h) == 7 and encoder.shape == (0, 7) and bias.size == 0, "Empty-internal model must be the registered linear representation")


def features(inputs, snapshot):
    """Read-only current architecture arithmetic; never construct a learner."""
    x = np.asarray(inputs)
    require(x.ndim == 3 and x.shape[1:] == (2, 7) and x.dtype.kind == "f" and np.isfinite(x).all(), "Finite original paired inputs required")
    scale = float(snapshot["feature_scale"])
    if snapshot["internal_weights"].size == 0:
        return x / scale
    kc = np.maximum(x @ snapshot["encoder"].T + snapshot["bias"], 0.)
    output = np.zeros((*kc.shape[:-1], len(snapshot["readout_weights"])))
    flat = kc.reshape(-1, kc.shape[-1])
    destination = output.reshape(-1, output.shape[-1])
    for index, row in enumerate(flat):
        destination[index] = np.bincount(snapshot["post"], weights=snapshot["internal_weights"] * row[snapshot["pre"]],
                                         minlength=output.shape[-1])
    return output / scale


def decompose(inputs, counts, baseline, target, tolerance=1e-10):
    _snapshot_check(baseline)
    _snapshot_check(target)
    require(all(np.array_equal(baseline[key], target[key]) for key in FIXED_FIELDS), "Encoder, topology, initial bounds or feature scale changed")
    require(baseline["readout_weights"].shape == target["readout_weights"].shape and baseline["internal_weights"].shape == target["internal_weights"].shape,
            "Crossed coefficient coordinates differ")
    counts = np.asarray(counts)
    require(counts.shape == np.asarray(inputs).shape[:2] and counts.dtype.kind in "iu" and np.all(np.sort(counts, axis=1) == [2, 4]),
            "Original two/four paired quantity identities required")
    phi0, phi1 = features(inputs, baseline), features(inputs, target)
    h0, h1 = baseline["readout_weights"], target["readout_weights"]
    psi0, psi1 = phi0[:, 1] - phi0[:, 0], phi1[:, 1] - phi1[:, 0]
    difference_head, difference_phi = h1 - h0, psi1 - psi0
    result = {}
    for label, phi, head in (("00", phi0, h0), ("10", phi0, h1), ("01", phi1, h0), ("11", phi1, h1)):
        scores = phi @ head
        result["raw_gap_" + label] = scores[:, 1] - scores[:, 0]
    result.update(head_term=psi0 @ difference_head, internal_term=difference_phi @ h0, cross_term=difference_phi @ difference_head)
    result["total_raw_gap_change"] = result["raw_gap_11"] - result["raw_gap_00"]
    result["bilinear_residual"] = result["total_raw_gap_change"] - (result["head_term"] + result["internal_term"] + result["cross_term"])
    result["raw_interaction"] = result["raw_gap_11"] - result["raw_gap_10"] - result["raw_gap_01"] + result["raw_gap_00"]
    result["raw_interaction_residual"] = result["raw_interaction"] - result["cross_term"]
    sign = 2 * np.argmin(counts, axis=1) - 1
    result["target_sign"] = sign
    for label in CROSSES:
        gap = result["raw_gap_" + label]
        result["aligned_raw_gap_" + label] = sign * gap
        result["p_right_" + label] = 1 / (1 + np.exp(-np.clip(gap, -40., 40.)))
        result["p_rule_" + label] = np.where(sign == 1, result["p_right_" + label], 1 - result["p_right_" + label])
        result["clipped_" + label] = (np.abs(gap) > 40.).astype(int)
    for field in ("head_term", "internal_term", "cross_term", "total_raw_gap_change"):
        result["aligned_" + field] = sign * result[field]
    result["policy_interaction_right"] = result["p_right_11"] - result["p_right_10"] - result["p_right_01"] + result["p_right_00"]
    result["policy_interaction_rule"] = result["p_rule_11"] - result["p_rule_10"] - result["p_rule_01"] + result["p_rule_00"]
    result["policy_rule_change"] = result["p_rule_11"] - result["p_rule_00"]
    require(all(np.isfinite(value).all() for value in result.values())
            and np.max(np.abs(result["bilinear_residual"])) <= tolerance
            and np.max(np.abs(result["raw_interaction_residual"])) <= tolerance, "Bilinear arithmetic identity residual exceeds tolerance")
    require(np.all(result["aligned_total_raw_gap_change"] * result["policy_rule_change"] >= -tolerance), "Raw/policy target orientation sign differs")
    return result


def cell_summary(seed, model, branch, trial, split, values):
    return {"seed": seed, "model": model, "branch": branch, "trial": trial, "split": split,
            "cases": len(values["head_term"]), "means": {key: float(np.mean(values[key])) for key in SUMMARY_FIELDS},
            "maximum_absolute_bilinear_residual": float(np.max(np.abs(values["bilinear_residual"]))),
            "maximum_absolute_raw_interaction_residual": float(np.max(np.abs(values["raw_interaction_residual"]))),
            "clipped_counts": {label: int(np.sum(values["clipped_" + label])) for label in CROSSES}}


def summarize(cfg, cells):
    expected = {(seed, model, branch, trial, split) for seed in cfg["seeds"] for model in MODELS for branch in BRANCHES
                for trial in cfg["checkpoints"] for split in ("iid", "ood")}
    index = {(row["seed"], row["model"], row["branch"], row["trial"], row["split"]): row for row in cells}
    require(len(index) == len(cells) and set(index) == expected, "Every original state and evaluation split must remain in summary")
    grouped = []
    for model in MODELS:
        for branch in BRANCHES:
            for trial in cfg["checkpoints"]:
                for split in ("iid", "ood"):
                    rows = [index[seed, model, branch, trial, split] for seed in cfg["seeds"]]
                    require(all(row["cases"] == cfg["eval_trials"] for row in rows), "Equal original case coverage required")
                    statistics = {key: {"mean": float(np.mean([row["means"][key] for row in rows])),
                        "sample_sd_across_seed_means": float(np.std([row["means"][key] for row in rows], ddof=1)) if len(rows) > 1 else None}
                        for key in SUMMARY_FIELDS}
                    grouped.append({"model": model, "branch": branch, "trial": trial, "split": split, "n_seeds": len(rows),
                        "cases": sum(row["cases"] for row in rows), "statistics": statistics,
                        "maximum_absolute_bilinear_residual": max(row["maximum_absolute_bilinear_residual"] for row in rows),
                        "maximum_absolute_raw_interaction_residual": max(row["maximum_absolute_raw_interaction_residual"] for row in rows),
                        "clipped_counts": {label: sum(row["clipped_counts"][label] for row in rows) for label in CROSSES}})
    return {"study_id": cfg["study_id"], "scope": cfg["scope"], "seed_summaries": cells, "cells": grouped,
            "cases": sum(row["cases"] for row in cells), "crossed_policy_values": 4 * sum(row["cases"] for row in cells),
            "learning_updates": 0, "behavior_hypotheses_tested": 0,
            "arithmetic_identity_passed": all(row["maximum_absolute_bilinear_residual"] <= cfg["raw_residual_tolerance"]
                and row["maximum_absolute_raw_interaction_residual"] <= cfg["raw_residual_tolerance"] for row in cells),
            "interpretation": "Static off-trajectory coefficient algebra; target-aligned terms avoid left/right cancellation; policy interactions include sigmoid and clipping and are not causal learning contributions"}


def register(root, config_path, out):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = read_json(config_path)
    validate_config(cfg)
    evidence = parent_gate(root, cfg)
    record = {"study_id": cfg["study_id"], "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "config": cfg, "config_sha256": digest(config_path), "frozen_files": frozen_files(root, cfg),
        "parent_files": {path: digest(safe(root, path)) for path in cfg["parent_files"]},
        "input_files": input_inventory(root, cfg), "gate": evidence,
        "notice": "Prospective private Git freeze for static descriptive arithmetic; no new learning or causal behavior hypothesis"}
    require(re.fullmatch(r"[a-f0-9]{40}", record["source_git_commit"]), "Valid source Git identity required")
    write_new(out, record)
    return record


def _artifacts(out):
    return {path.relative_to(out).as_posix(): digest(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"}


def run(root, config_path, registration_path, out):
    root, config_path, registration_path, out = (Path(path).resolve() for path in (root, config_path, registration_path, out))
    cfg, reg = read_json(config_path), read_json(registration_path)
    validate_config(cfg)
    now, timestamp = datetime.now(timezone.utc), datetime.fromisoformat(reg["registered_at_utc"])
    require(timestamp.tzinfo is not None and timestamp.utcoffset().total_seconds() == 0 and timestamp <= now
            and re.fullmatch(r"[a-f0-9]{40}", reg["source_git_commit"]), "Prior UTC registration and source identity required")
    require(reg["study_id"] == cfg["study_id"] and reg["config"] == cfg and reg["config_sha256"] == digest(config_path)
            and reg["frozen_files"] == frozen_files(root, cfg) and reg["parent_files"] == {path: digest(safe(root, path)) for path in cfg["parent_files"]}
            and reg["gate"] == parent_gate(root, cfg) and reg["input_files"] == input_inventory(root, cfg), "Frozen static analysis inputs or source changed")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "registration.json", reg)
    metadata = {"study_id": cfg["study_id"], "status": "running", "started_at_utc": now.isoformat(), "config": cfg,
        "config_sha256": digest(config_path), "source_git_commit": reg["source_git_commit"], "registration_sha256": digest(registration_path),
        "frozen_files": reg["frozen_files"], "parent_files": reg["parent_files"], "input_files": reg["input_files"], "gate": reg["gate"],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "learning_updates": 0, "learner_constructions": 0,
        "dataset_generations": 0, "rng_draws": 0, "behavior_hypotheses_tested": 0, "cases": 0, "crossed_policy_values": 0,
        "maximum_case_csv_bytes": 0, "human_or_species_comparison": False}
    write_new(out / "metadata.json", metadata)
    cells, controls = [], {"state_pairs": 0, "frozen_state_checks": 0, "head_only_internal_constant_checks": 0,
        "internal_only_head_constant_checks": 0, "linear_internal_zero_checks": 0, "initial_state_checks": 0,
        "saved_policy_entries_checked": 0, "masked_zero_term_entries_checked": 0}
    try:
        for seed in cfg["seeds"]:
            targetdir = out / f"seed_{seed}"
            targetdir.mkdir()
            seedbase = cfg["parent_root"] + f"/seed_{seed}"
            stimuli = load_npz(safe(root, seedbase + "/stimuli.npz"))
            with (targetdir / "cases.csv").open("x", encoding="utf8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS)
                writer.writeheader()
                for model in MODELS:
                    baseline = load_npz(safe(root, f"{seedbase}/{model}/prefix/{cfg['baseline_phase']}_{cfg['baseline_trial']:04d}.npz"), SNAPSHOT_FIELDS)
                    for branch in BRANCHES:
                        for trial in cfg["checkpoints"]:
                            target = load_npz(safe(root, f"{seedbase}/{model}/{branch}/{cfg['target_phase']}_{trial:04d}.npz"), SNAPSHOT_FIELDS)
                            controls["state_pairs"] += 1
                            if branch == "frozen" or trial == 0:
                                require(np.array_equal(baseline["internal_weights"], target["internal_weights"])
                                        and np.array_equal(baseline["readout_weights"], target["readout_weights"]), "Frozen or initial target coefficients changed")
                                if branch == "frozen":
                                    controls["frozen_state_checks"] += 1
                                if trial == 0:
                                    controls["initial_state_checks"] += 1
                            if branch == "head_only":
                                require(np.array_equal(baseline["internal_weights"], target["internal_weights"]), "Head-only source changed internal weights")
                                controls["head_only_internal_constant_checks"] += 1
                            if branch == "internal_only":
                                require(np.array_equal(baseline["readout_weights"], target["readout_weights"]), "Internal-only source changed readout weights")
                                controls["internal_only_head_constant_checks"] += 1
                            for split in ("iid", "ood"):
                                counts, nuisance = stimuli[split + "_counts"], stimuli[split + "_nuisance"]
                                require(counts.shape == (cfg["eval_trials"], 2) and nuisance.shape == (*counts.shape, 6)
                                        and nuisance.dtype.kind == "f" and np.isfinite(nuisance).all(), "Original evaluation stimulus dimensions differ")
                                inputs = np.concatenate((counts[..., None] / 8., .1 * (nuisance - .5)), axis=-1)
                                values = decompose(inputs, counts, baseline, target, cfg["raw_residual_tolerance"])
                                require(np.max(np.abs(values["p_right_00"] - baseline[split + "_p_right"])) <= cfg["policy_match_tolerance"]
                                        and np.max(np.abs(values["p_right_11"] - target[split + "_p_right"])) <= cfg["policy_match_tolerance"],
                                        "Reconstructed diagonal policy differs from original saved cases")
                                controls["saved_policy_entries_checked"] += 2 * cfg["eval_trials"]
                                zero_terms = set()
                                if branch == "head_only" or model == "linear":
                                    zero_terms.update(("internal_term", "cross_term"))
                                if branch == "internal_only":
                                    zero_terms.update(("head_term", "cross_term"))
                                if branch == "frozen" or trial == 0:
                                    zero_terms.update(("head_term", "internal_term", "cross_term", "total_raw_gap_change"))
                                require(all(np.array_equal(values[key], np.zeros(cfg["eval_trials"])) for key in zero_terms),
                                        "A masked or unchanged component has nonzero algebraic contribution")
                                controls["masked_zero_term_entries_checked"] += len(zero_terms) * cfg["eval_trials"]
                                if model == "linear":
                                    require(np.array_equal(values["internal_term"], np.zeros(cfg["eval_trials"]))
                                            and np.array_equal(values["cross_term"], np.zeros(cfg["eval_trials"]))
                                            and np.array_equal(values["raw_gap_00"], values["raw_gap_01"])
                                            and np.array_equal(values["raw_gap_10"], values["raw_gap_11"]), "Linear internal negative control failed")
                                    controls["linear_internal_zero_checks"] += 1
                                for index in range(cfg["eval_trials"]):
                                    row = {"seed": seed, "model": model, "branch": branch, "trial": trial, "split": split, "case": index,
                                        "family_id": str(stimuli[split + "_family_ids"][index]), "left_count": int(counts[index, 0]),
                                        "right_count": int(counts[index, 1]), "target_action": int(np.argmin(counts[index])),
                                        "target_sign": int(values["target_sign"][index])}
                                    row.update({key: int(values[key][index]) if key.startswith("clipped_") else float(values[key][index])
                                                for key in (*VALUE_FIELDS, *("clipped_" + label for label in CROSSES))})
                                    writer.writerow(row)
                                metadata["cases"] += cfg["eval_trials"]
                                metadata["crossed_policy_values"] += 4 * cfg["eval_trials"]
                                cells.append(cell_summary(seed, model, branch, trial, split, values))
            size = (targetdir / "cases.csv").stat().st_size
            require(size < 100 * 1024 * 1024, "A per-seed case CSV exceeds the declared100MiB resource bound")
            metadata["maximum_case_csv_bytes"] = max(metadata["maximum_case_csv_bytes"], size)
            print(f"completed static state decomposition seed {seed} ({cfg['seeds'].index(seed) + 1}/{len(cfg['seeds'])})", flush=True)
        require(metadata["cases"] == cfg["cases"] and metadata["crossed_policy_values"] == cfg["crossed_policy_values"], "Registered static case coverage differs")
        write_new(out / "summary.json", summarize(cfg, cells))
        write_new(out / "controls.json", controls)
        metadata.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        metadata.update(status="failed", completed_at_utc=datetime.now(timezone.utc).isoformat(), error_type=type(error).__name__, error_message=str(error))
        if cells:
            write_new(out / "partial_seed_summaries.json", cells)
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
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists() and (args.mode != "run" or args.registration is not None), "Exclusive output and prior registration required")
    result = register(args.root, args.config, args.out) if args.mode == "register" else run(args.root, args.config, args.registration, args.out)
    print(json.dumps({"status": result.get("status", "registered"), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
