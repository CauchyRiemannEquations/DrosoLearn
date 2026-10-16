"""Independent static decomposition of already saved policy states.

No producer, learner constructor, forward/update or summary code is imported.
Only named arithmetic and decoding helpers from an immutable independent
verifier are reused. Source states are read-only and no random data are drawn.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "independent_ablation_state_arithmetic", Path(__file__).with_name("verify_learning_ablation.py"))
_OLD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_OLD)
require, near, equal = _OLD.require, _OLD.near, _OLD.equal
digest, read_json, safe, utc = _OLD.digest, _OLD.read_json, _OLD.safe, _OLD.utc
number, integer, npz = _OLD.number, _OLD.integer, _OLD.npz
raw_features, inputs = _OLD.raw_features, _OLD.inputs

MODELS = ("linear", "real", "rewired", "ann")
BRANCHES = ("full", "head_only", "internal_only", "frozen")
CROSSES = ("00", "10", "01", "11")
FIXED_FIELDS = ("encoder", "bias", "pre", "post", "feature_scale", "initial_feature_gap_rms",
                "initial_internal_weights", "internal_upper_bounds", "incoming_normalizer")
TOLERANCE = 1e-10
SNAPSHOT_FIELDS = _OLD.SNAPSHOT_FIELDS
RESOURCE_FIELDS = ("learning_updates", "learner_constructions", "dataset_generations", "rng_draws")
CONTEXT_FIELDS = ("seed", "model", "branch", "trial", "split", "case", "family_id", "left_count", "right_count")
INTEGER_FIELDS = {"seed", "trial", "case", "left_count", "right_count", "target_action", "target_sign",
                  *(f"clipped_{cross}" for cross in CROSSES)}
VALUE_FIELDS = tuple([*(f"raw_gap_{c}" for c in CROSSES), "head_term", "internal_term", "cross_term",
    "total_raw_gap_change", "bilinear_residual", "raw_interaction", "raw_interaction_residual",
    *(f"p_right_{c}" for c in CROSSES), *(f"p_rule_{c}" for c in CROSSES), "policy_interaction_right",
    "policy_interaction_rule", "policy_rule_change", *(f"clipped_{c}" for c in CROSSES),
    "target_action", "target_sign", *(f"aligned_raw_gap_{c}" for c in CROSSES),
    "aligned_head_term", "aligned_internal_term", "aligned_cross_term", "aligned_total_raw_gap_change"])
CASE_FIELDS = set(CONTEXT_FIELDS) | set(VALUE_FIELDS)


def config_check(cfg):
    require(cfg["study_id"] == "state_decomposition_v1" and cfg["scope"] in {
        "registered_descriptive_diagnostic", "integration_test"}, "Static study identity and scope")
    require(cfg["models"] == list(MODELS) and cfg["branches"] == list(BRANCHES), "All models and branches retained")
    require(cfg["parent_root"] == "results/learning_ablation_v1" and cfg["parent_config"] == "configs/learning_ablation_v1.json",
            "Fixed completed ablation source paths")
    require(cfg["baseline_phase"] == "phase1" and cfg["target_phase"] == "phase2" and cfg["pair"] == [2, 4],
            "Prefix-end baseline and smaller-target reversal inputs")
    require(cfg["target"] == "smaller_quantity_all_crossed_states" and cfg["interpretation"] ==
            "static_off_trajectory_parameter_algebra_not_learning_causal_effect" and cfg["all_registered_inputs_retained"] is True,
            "Common smaller target, all inputs, descriptive off-trajectory interpretation")
    require(cfg["policy_clip"] == [-40., 40.] and cfg["raw_residual_tolerance"] == TOLERANCE
            and cfg["policy_match_tolerance"] == 1e-12, "Exact numerical clipping and immutable arithmetic/policy tolerances")
    for field in RESOURCE_FIELDS:
        require(type(cfg[field]) is int and cfg[field] == 0, f"No new resource use: {field}")
    seeds, times, ev, n = cfg["seeds"], cfg["checkpoints"], cfg["eval_trials"], cfg["baseline_trial"]
    require(type(seeds) is list and seeds and len(seeds) == len(set(seeds)) and all(type(v) is int for v in seeds), "Unique integer seeds")
    require(type(times) is list and len(times) >= 2 and all(type(t) is int and 0 <= t <= n for t in times)
            and times == sorted(set(times)) and times[0] == 0 and times[-1] == n
            and type(n) is int and n > 0 and type(ev) is int and ev > 0, "Complete ordered checkpoint budget")
    if cfg["scope"] == "registered_descriptive_diagnostic":
        require(seeds == list(range(12001, 12011)) and (n, ev, times) == (400, 128, [0, 100, 200, 300, 400]),
                "All completed scientific seeds, checkpoints and cases")
        require(len(cfg["frozen_files"]) == 75, "Complete75-file prospective source and completed-engineering-fixture closure")
    else:
        require(set(seeds) <= {17, 18} and (n, ev, times) == (8, 8, [0, 4, 8]), "Only copied completed tiny engineering inputs")
    cases = len(seeds) * 4 * 4 * len(times) * 2 * ev
    equal(cfg["cases"], cases, "Exact static case budget")
    equal(cfg["crossed_policy_values"], cases * 4, "Four crossed policies per case")
    equal(cfg["selected_input_files"], len(seeds) * (1 + 4 + 16 * len(times)), "Exact selected actual NPZ file budget")


def selected_paths(cfg):
    result = []
    for seed in cfg["seeds"]:
        base = f'{cfg["parent_root"]}/seed_{seed}'
        result.append(f"{base}/stimuli.npz")
        for model in MODELS:
            result.append(f'{base}/{model}/prefix/phase1_{cfg["baseline_trial"]:04d}.npz')
            result.extend(f"{base}/{model}/{branch}/phase2_{t:04d}.npz" for branch in BRANCHES for t in cfg["checkpoints"])
    return result


def sha_format(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def file_manifest_check(root, entries, expected):
    require(type(entries) is list and len(entries) == len(expected), "Explicit complete input file list")
    by_path = {}
    for row in entries:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "Input manifest row schema")
        name = row["path"]
        require(name not in by_path and name in expected and type(row["bytes"]) is int and row["bytes"] > 0
                and sha_format(row["sha256"]), "Unique expected input path and byte/hash format")
        path = safe(root, name)
        require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], "Registered actual input file bytes")
        by_path[name] = row
    require(set(by_path) == set(expected), "No source input selection or omission")
    return by_path


def source_check(root, registration, output, reg, meta):
    cfg = reg["config"]
    config_check(cfg)
    require(reg["study_id"] == meta["study_id"] == cfg["study_id"] and meta["status"] == "completed", "Completed static study identity")
    equal(meta["config"], cfg, "Executed static configuration")
    require(type(reg["source_git_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", reg["source_git_commit"])
            and meta["source_git_commit"] == reg["source_git_commit"], "Recorded static source version")
    require(digest(registration) == digest(output / "registration.json") == meta["registration_sha256"], "External/copied registration exact bytes")
    equal(read_json(output / "registration.json"), reg, "Registration contents")
    require(utc(reg["registered_at_utc"]) <= utc(meta["started_at_utc"]) <= utc(meta["completed_at_utc"])
            <= datetime.now(timezone.utc), "Registration precedes completed static analysis without future chronology")
    source_cfg = safe(root, "configs/state_decomposition_v1.json")
    require(digest(source_cfg) == reg["config_sha256"] == meta["config_sha256"], "Exact config bytes")
    equal(read_json(source_cfg), cfg, "Source static configuration contents")
    for key in ("frozen_files", "parent_files"):
        require(type(cfg[key]) is list and len(cfg[key]) == len(set(cfg[key])) and set(reg[key]) == set(cfg[key]), f"Exact {key} closure")
        for name, sha in reg[key].items():
            require(sha_format(sha) and digest(safe(root, name)) == sha, f"Unchanged {key} bytes: {name}")
        equal(meta[key], reg[key], f"Executed {key}")
    for name in ("scripts/verify_state_decomposition.py", "scripts/verify_learning_ablation.py", "scripts/verify_stochastic_feedback.py",
                 "src/drosolearn/encoding_audit.py", "src/drosolearn/bandit_reversal_audit.py"):
        require(name in reg["frozen_files"] and digest(safe(ROOT, name)) == reg["frozen_files"][name],
                "Executed independent verifier/helper bytes match registered source: " + name)
    entries = file_manifest_check(root, reg["input_files"], selected_paths(cfg))
    equal(meta["input_files"], reg["input_files"], "Executed exact source input manifest")
    for field in RESOURCE_FIELDS:
        require(type(meta[field]) is int and meta[field] == 0, f"No new static resource use: {field}")
    require(meta["human_or_species_comparison"] is False, "No new human/species comparison")
    return cfg, entries


def parent_check(root, cfg, reg, entries):
    base = cfg["parent_root"]
    paths = {"metadata": base + "/metadata.json", "summary": base + "/summary.json",
        "verification": base + "/verification.json", "artifacts": base + "/artifact_hashes.json",
        "registration": "results/registration_learning_ablation_v1.json", "config": cfg["parent_config"],
        "final": "results/learning_ablation_final_validation_v1.json", "doc": "docs/learning-ablation-findings-v2.md",
        "correction": "results/learning_ablation_documentation_correction_v1.json", "inputs": "results/learning_ablation_input_freeze_v1.json"}
    identity = cfg["parent_gate_identity"]
    require(set(cfg["parent_files"]) == set(paths.values()) and set(identity) == {"published_commit", "file_sha256"}
            and set(identity["file_sha256"]) == set(paths.values()), "Exact ten-file published parent identity")
    require(type(identity["published_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", identity["published_commit"]), "Parent publication commit format")
    for name, sha in identity["file_sha256"].items():
        require(sha_format(sha) and digest(safe(root, name)) == sha == reg["parent_files"][name], "Pinned published parent bytes")
    p = {key: read_json(safe(root, name)) for key, name in paths.items() if key != "doc"}
    meta, checked, registration, parent_cfg = (p[key] for key in ("metadata", "verification", "registration", "config"))
    require(meta["study_id"] == p["summary"]["study_id"] == checked["study_id"] == registration["study_id"] == "learning_ablation_v1"
            and meta["status"] == p["summary"]["status"] == "completed" and checked["status"] == "passed"
            and checked["passed"] is True and checked["errors"] == [], "Complete independently passed immutable parent")
    for key, name in (("summary_sha256", "summary"), ("registration_sha256", "registration"),
                      ("config_sha256", "config"), ("artifact_hashes_sha256", "artifacts")):
        require(checked[key] == digest(safe(root, paths[name])), "Parent independent report binds exact bytes: " + key)
    require(meta["registration_sha256"] == digest(safe(root, paths["registration"]))
            and meta["config_sha256"] == registration["config_sha256"] == digest(safe(root, paths["config"]))
            and meta["source_git_commit"] == registration["source_git_commit"], "Parent executed config/registration/version identity")
    equal(meta["config"], parent_cfg, "Parent executed configuration")
    equal(registration["config"], parent_cfg, "Parent registered configuration")
    require(parent_cfg["seeds"] == cfg["seeds"] and parent_cfg["trials_per_phase"] == cfg["baseline_trial"]
            and parent_cfg["eval_trials"] == cfg["eval_trials"] and parent_cfg["models"] == list(MODELS)
            and parent_cfg["branches"] == list(BRANCHES), "Complete existing parent coordinates")
    times = [utc(registration["registered_at_utc"]), utc(meta["started_at_utc"]), utc(meta["completed_at_utc"]),
             utc(checked["verified_at_utc"]), utc(reg["registered_at_utc"])]
    require(times == sorted(times), "Completed parent and independent verification precede static registration")
    artifacts = p["artifacts"]
    require(type(artifacts) is dict and artifacts, "Complete immutable parent primary artifact map")
    for name, sha in artifacts.items():
        require(sha_format(sha) and digest(safe(root, base + "/" + name)) == sha, "Every original parent primary artifact unchanged")
    parent_files = {path.relative_to(root / base).as_posix() for path in (root / base).rglob("*") if path.is_file()}
    require(parent_files - {"artifact_hashes.json", "verification.json"} == set(artifacts)
            and checked["counts"]["artifact_hashes"] == len(artifacts), "Complete parent primary artifact coverage")
    require(artifacts["summary.json"] == checked["summary_sha256"] and artifacts["metadata.json"] == digest(safe(root, paths["metadata"]))
            and artifacts["registration.json"] == digest(safe(root, paths["registration"])), "Original parent inventory binds gate records")
    if cfg["scope"] == "registered_descriptive_diagnostic":
        require(len(artifacts) == 1058, "All1058 published parent artifacts retained")
    for name, row in entries.items():
        require(artifacts[name.removeprefix(base + "/")] == row["sha256"], "Selected source NPZ bound to completed parent inventory")
    require(p["final"]["status"] == "passed" and p["final"]["study_id"] == "learning_ablation_v1"
            and p["final"]["all_sealed_scientific_and_original_document_bytes_unchanged"] is True,
            "Completed immutable parent final archive validation")
    correction = p["correction"]
    require(correction["status"] == "documentation_corrected" and correction["corrected_document"] == paths["doc"]
            and correction["corrected_sha256"] == digest(safe(root, paths["doc"]))
            and correction["summary_sha256"] == checked["summary_sha256"]
            and correction["verification_sha256"] == digest(safe(root, paths["verification"]))
            and correction["scientific_values_or_criteria_changed"] is False and correction["additional_learning_updates"] == 0,
            "Parent explanation correction retains exact scientific records")
    equal(p["final"]["documentation_correction"], correction, "Final parent archive retains exact explanation correction")
    frozen_inputs = p["inputs"]
    require(frozen_inputs["study_id"] == "learning_ablation_v1" and frozen_inputs["learning_updates"] == 0
            and frozen_inputs["learner_constructions"] == 0 and frozen_inputs["registration_sha256"] == checked["registration_sha256"]
            and frozen_inputs["config_sha256"] == checked["config_sha256"]
            and frozen_inputs["source_commit"] == meta["source_git_commit"], "Exact parent prospective input-freeze evidence")
    expected_gate = {"name": "verified_learning_ablation_before_static_decomposition", "passed": True,
                     **identity, "parent_primary_artifacts_verified": len(artifacts)}
    equal(reg["gate"], expected_gate, "Exact registered immutable parent gate")
    return parent_cfg, len(artifacts)


def decode_datasets(path, cfg):
    splits = ("train_phase1", "train_phase2", "iid", "ood")
    fields = {f"{split}_{field}" for split in splits for field in ("counts", "nuisance", "family_ids")}
    fields |= {f"{kind}_{phase}_uniforms" for kind in ("action", "reward") for phase in ("phase1", "phase2")}
    saved = npz(path, fields)
    result = {}
    for split in splits:
        n = cfg["eval_trials"] if split in ("iid", "ood") else cfg["baseline_trial"]
        counts, nuisance, family = (saved[f"{split}_{key}"] for key in ("counts", "nuisance", "family_ids"))
        require(counts.shape == (n, 2) and counts.dtype.kind in "iu" and np.all(np.sort(counts, axis=1) == [2, 4]), "Saved quantity pair dimensions")
        require(nuisance.shape == (n, 2, 6) and nuisance.dtype.kind == "f" and np.isfinite(nuisance).all()
                and (nuisance >= 0).all() and (nuisance <= 1).all(), "Saved six nuisance coordinates")
        require(family.shape == (n,) and family.dtype.kind in "US", "Saved case family IDs")
        result[split] = {"counts": counts, "nuisance": nuisance, "family_ids": family}
    for key in fields - {f"{split}_{field}" for split in splits for field in ("counts", "nuisance", "family_ids")}:
        require(saved[key].shape == (cfg["baseline_trial"],) and saved[key].dtype.kind == "f"
                and np.isfinite(saved[key]).all() and (saved[key] >= 0).all() and (saved[key] < 1).all(), "Existing parent uniform arrays are only read")
    return result


def snapshot_validate(saved, data, parent_cfg, linear):
    require(all(a.dtype.kind in "iuf" and np.isfinite(a).all() for a in saved.values()), "Finite numeric21-field saved state")
    h, w = saved["readout_weights"], saved["internal_weights"]
    encoder, bias, pre, post = (saved[k] for k in ("encoder", "bias", "pre", "post"))
    require(h.ndim == w.ndim == 1 and h.size > 0 and encoder.ndim == 2 and encoder.shape[1] == 7
            and bias.shape == (encoder.shape[0],) and pre.shape == post.shape == w.shape
            and pre.dtype.kind in "iu" and post.dtype.kind in "iu", "Saved encoder/sparse index/vector coordinates")
    for field in ("feature_scale", "initial_feature_gap_rms"):
        require(saved[field].shape == () and float(saved[field]) > 0, "Positive initial TRAIN scale")
    require(float(saved["feature_scale"]) == float(saved["initial_feature_gap_rms"]), "Same initial RMS and frozen scale")
    require(saved["initial_internal_weights"].shape == saved["internal_upper_bounds"].shape == w.shape
            and np.array_equal(saved["internal_upper_bounds"], 2 * saved["initial_internal_weights"])
            and (w >= 0).all() and (w <= saved["internal_upper_bounds"]).all(), "Original bounds and bounded current weights")
    if linear:
        require(w.size == 0 and h.size == 7 and encoder.shape == (0, 7) and bias.size == 0
                and saved["incoming_normalizer"].shape == (0,), "Registered no-internal linear state")
    else:
        require(w.size > 0 and np.all((pre >= 0) & (pre < len(encoder))) and np.all((post >= 0) & (post < len(h)))
                and saved["incoming_normalizer"].shape == h.shape and (saved["incoming_normalizer"] >= 1).all(), "Registered sparse network state")
    for name in ("head_updates", "internal_updates", "phase_head_updates", "phase_internal_updates"):
        require(saved[name].shape == () and saved[name].dtype.kind in "iu" and int(saved[name]) >= 0, "Saved integral learning counters read-only")
    probe = inputs(data["iid"], np.arange(1, 9))[:parent_cfg["probe_rows"]]
    kc, mb, decision, p = _OLD.state(probe, encoder, bias, pre, post, w, h, float(saved["feature_scale"]), linear)
    for name, value in (("kc_activity", kc), ("mbon_activity", mb), ("decision_activity", decision), ("choice_probability", p)):
        near(saved[name], value, "Read-only original probe state: " + name)
    for split in ("iid", "ood"):
        require(saved[split + "_p_right"].shape == (len(data[split]["counts"]),)
                and ((saved[split + "_p_right"] >= 0) & (saved[split + "_p_right"] <= 1)).all(), "Saved policy shape/range")


SUMMARY_FIELDS = tuple(k for k in VALUE_FIELDS if k not in {"target_action", "target_sign", "bilinear_residual", "raw_interaction_residual"}
                       and not k.startswith("clipped_"))
INTERPRETATION = "Static off-trajectory coefficient algebra; target-aligned terms avoid left/right cancellation; policy interactions include sigmoid and clipping and are not causal learning contributions"


def seed_summary(seed, model, branch, trial, split, values):
    return {"seed": seed, "model": model, "branch": branch, "trial": trial, "split": split,
        "cases": len(values["head_term"]), "means": {key: float(np.mean(values[key])) for key in SUMMARY_FIELDS},
        "maximum_absolute_bilinear_residual": float(np.max(np.abs(values["bilinear_residual"]))),
        "maximum_absolute_raw_interaction_residual": float(np.max(np.abs(values["raw_interaction_residual"]))),
        "clipped_counts": {key: int(np.count_nonzero(values["clipped_" + key])) for key in CROSSES}}


def summary_check(cfg, seeds, actual):
    by_key = {(r["seed"], r["model"], r["branch"], r["trial"], r["split"]): r for r in seeds}
    grouped = []
    for model in MODELS:
        for branch in BRANCHES:
            for trial in cfg["checkpoints"]:
                for split in ("iid", "ood"):
                    rows = [by_key[seed, model, branch, trial, split] for seed in cfg["seeds"]]
                    statistics = {}
                    for key in SUMMARY_FIELDS:
                        samples = np.asarray([row["means"][key] for row in rows])
                        statistics[key] = {"mean": float(samples.mean()),
                            "sample_sd_across_seed_means": float(samples.std(ddof=1)) if len(samples) > 1 else None}
                    grouped.append({"model": model, "branch": branch, "trial": trial, "split": split, "n_seeds": len(rows),
                        "cases": sum(row["cases"] for row in rows), "statistics": statistics,
                        **{key: max(row[key] for row in rows) for key in ("maximum_absolute_bilinear_residual", "maximum_absolute_raw_interaction_residual")},
                        "clipped_counts": {key: sum(row["clipped_counts"][key] for row in rows) for key in CROSSES}})
    expected = {"study_id": cfg["study_id"], "scope": cfg["scope"], "seed_summaries": seeds, "cells": grouped,
        "cases": cfg["cases"], "crossed_policy_values": cfg["crossed_policy_values"], "learning_updates": 0,
        "behavior_hypotheses_tested": 0, "arithmetic_identity_passed": True, "interpretation": INTERPRETATION}
    equal(actual, expected, "Every full-denominator static summary, aligned mean, between-seed sample SD and clipping count")
    for row in [*actual["seed_summaries"], *actual["cells"]]:
        require(row["maximum_absolute_bilinear_residual"] <= TOLERANCE and row["maximum_absolute_raw_interaction_residual"] <= TOLERANCE,
                "Every reported maximum residual below arithmetic tolerance")


def static_cases(prefix, current, data, *, linear):
    """Return all four raw/policy combinations for one unchanged input split."""
    for field in FIXED_FIELDS:
        require(np.array_equal(prefix[field], current[field]), f"Fixed source initialization: {field}")
    h0, h1 = prefix["readout_weights"], current["readout_weights"]
    w0, w1 = prefix["internal_weights"], current["internal_weights"]
    require(h0.ndim == h1.ndim == w0.ndim == w1.ndim == 1 and h0.shape == h1.shape and w0.shape == w1.shape,
            "Corresponding saved readout/internal vector dimensions")
    require(prefix["feature_scale"].shape == () and np.isfinite(prefix["feature_scale"])
            and float(prefix["feature_scale"]) > 0, "One positive frozen initial TRAIN scale")
    scale = float(prefix["feature_scale"])
    require(all(np.isfinite(v).all() for source in (prefix, current) for v in source.values()), "Finite saved source state")
    x = inputs(data, np.arange(1, 9))
    features = []
    for weights in (w0, w1):
        _, _, raw = raw_features(x, prefix["encoder"], prefix["bias"], prefix["pre"], prefix["post"], weights, h0.size, linear)
        features.append((raw[:, 1] - raw[:, 0]) / scale)
    psi0, psi1 = features
    delta_h, delta_psi = h1 - h0, psi1 - psi0
    raw_gaps = {"00": psi0 @ h0, "10": psi0 @ h1, "01": psi1 @ h0, "11": psi1 @ h1}
    terms = {"head_term": psi0 @ delta_h, "internal_term": delta_psi @ h0, "cross_term": delta_psi @ delta_h}
    total = raw_gaps["11"] - raw_gaps["00"]
    residual = total - terms["head_term"] - terms["internal_term"] - terms["cross_term"]
    interaction = raw_gaps["11"] - raw_gaps["10"] - raw_gaps["01"] + raw_gaps["00"]
    interaction_residual = interaction - terms["cross_term"]
    require(np.max(np.abs(residual)) <= TOLERANCE and np.max(np.abs(interaction_residual)) <= TOLERANCE,
            "Bilinear arithmetic identity is a numerical integrity check")
    target = (data["counts"][:, 1] < data["counts"][:, 0]).astype(int)
    sign = 2 * target - 1
    right = {key: 1 / (1 + np.exp(-np.clip(value, -40, 40))) for key, value in raw_gaps.items()}
    rule = {key: np.where(target == 1, p, 1 - p) for key, p in right.items()}
    result = {**{f"raw_gap_{key}": value for key, value in raw_gaps.items()}, **terms,
              "total_raw_gap_change": total, "bilinear_residual": residual,
              "raw_interaction": interaction, "raw_interaction_residual": interaction_residual,
              **{f"p_right_{key}": value for key, value in right.items()},
              **{f"p_rule_{key}": value for key, value in rule.items()},
              "policy_interaction_right": right["11"] - right["10"] - right["01"] + right["00"],
              "policy_interaction_rule": rule["11"] - rule["10"] - rule["01"] + rule["00"],
              "policy_rule_change": rule["11"] - rule["00"],
              **{f"clipped_{key}": (np.abs(value) > 40).astype(int) for key, value in raw_gaps.items()},
              "target_action": target, "target_sign": sign,
              **{f"aligned_raw_gap_{key}": sign * value for key, value in raw_gaps.items()},
              **{f"aligned_{key}": sign * value for key, value in terms.items()},
              "aligned_total_raw_gap_change": sign * total}
    for key, value in result.items():
        require(value.shape == (len(data["counts"]),) and np.isfinite(value).all(), f"Finite static case array: {key}")
    return result


def canonical_input_digest(entries):
    return hashlib.sha256(json.dumps(entries, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()


def audit(root, registration, output, report):
    reg, meta = read_json(registration), read_json(output / "metadata.json")
    cfg, entries = source_check(root, registration, output, reg, meta)
    parent_cfg, parent_artifacts = parent_check(root, cfg, reg, entries)
    equal(meta["gate"], reg["gate"], "Executed completed-parent gate")
    hashes = read_json(output / "artifact_hashes.json")
    expected_artifacts = {"metadata.json", "registration.json", "summary.json", "controls.json"} | {
        f"seed_{seed}/cases.csv" for seed in cfg["seeds"]}
    require(type(hashes) is dict and set(hashes) == expected_artifacts, "Complete exact static primary artifact inventory")
    files = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} - {"artifact_hashes.json", "verification.json"}
    require(files == expected_artifacts, "No omitted, extra or partial static output artifacts")
    for name, sha in hashes.items():
        require(sha_format(sha) and digest(safe(output, name)) == sha, "Static primary artifact exact bytes: " + name)
    maximum_csv = max(safe(output, f"seed_{seed}/cases.csv").stat().st_size for seed in cfg["seeds"])
    require(type(meta["maximum_case_csv_bytes"]) is int and meta["maximum_case_csv_bytes"] == maximum_csv
            and 0 < maximum_csv < 100 * 1024 * 1024, "Exact maximum case CSV size and archive file limit")
    controls = {key: 0 for key in ("state_pairs", "frozen_state_checks", "head_only_internal_constant_checks",
        "internal_only_head_constant_checks", "linear_internal_zero_checks", "initial_state_checks",
        "saved_policy_entries_checked", "masked_zero_term_entries_checked")}
    counts = {"cases": 0, "crossed_policy_values": 0, "fixed_initialization_field_checks": 0,
              "prefix_native_array_clone_checks": 0, "linear_equivalent_checkpoint_pairs": 0,
              **{key: 0 for key in RESOURCE_FIELDS}}
    summaries = []
    for seed in cfg["seeds"]:
        base = f'{cfg["parent_root"]}/seed_{seed}'
        data = decode_datasets(safe(root, base + "/stimuli.npz"), cfg)
        linear_snapshots = {}
        with safe(output, f"seed_{seed}/cases.csv").open(encoding="utf8", newline="") as stream:
            reader = csv.DictReader(stream)
            require(len(reader.fieldnames or ()) == len(CASE_FIELDS) and set(reader.fieldnames or ()) == CASE_FIELDS, "Exact wide static CSV schema")
            for model in MODELS:
                linear = model == "linear"
                prefix = npz(safe(root, f'{base}/{model}/prefix/phase1_{cfg["baseline_trial"]:04d}.npz'), SNAPSHOT_FIELDS)
                snapshot_validate(prefix, data, parent_cfg, linear)
                for branch in BRANCHES:
                    for trial in cfg["checkpoints"]:
                        current = npz(safe(root, f"{base}/{model}/{branch}/phase2_{trial:04d}.npz"), SNAPSHOT_FIELDS)
                        snapshot_validate(current, data, parent_cfg, linear)
                        for field in FIXED_FIELDS:
                            require(np.array_equal(prefix[field], current[field]), "Unchanged original encoder/topology/bounds/normalizer/scale: " + field)
                            counts["fixed_initialization_field_checks"] += 1
                        if trial == 0:
                            require(all(np.array_equal(current[k], prefix[k]) for k in SNAPSHOT_FIELDS - {"phase_head_updates", "phase_internal_updates"}),
                                    "Initial branch inherits all19 native prefix arrays exactly")
                            require(int(current["phase_head_updates"]) == int(current["phase_internal_updates"]) == 0, "Only branch phase counters reset")
                            counts["prefix_native_array_clone_checks"] += 19
                            controls["initial_state_checks"] += 1
                        controls["state_pairs"] += 1
                        if branch == "frozen" or trial == 0:
                            require(np.array_equal(prefix["readout_weights"], current["readout_weights"])
                                    and np.array_equal(prefix["internal_weights"], current["internal_weights"]), "Frozen or initial learned source state retained")
                        if branch == "frozen":
                            controls["frozen_state_checks"] += 1
                        if branch == "head_only":
                            require(np.array_equal(prefix["internal_weights"], current["internal_weights"]), "Head-only internal coefficient negative control")
                            controls["head_only_internal_constant_checks"] += 1
                        if branch == "internal_only":
                            require(np.array_equal(prefix["readout_weights"], current["readout_weights"]), "Internal-only readout coefficient negative control")
                            controls["internal_only_head_constant_checks"] += 1
                        if linear:
                            linear_snapshots[branch, trial] = current
                        for split in ("iid", "ood"):
                            values = static_cases(prefix, current, data[split], linear=linear)
                            for cross, saved in (("00", prefix), ("11", current)):
                                require(np.max(np.abs(values["p_right_" + cross] - saved[split + "_p_right"])) <= 1e-12,
                                        "Original saved diagonal p_right; same smaller target for all crosses")
                            controls["saved_policy_entries_checked"] += 2 * cfg["eval_trials"]
                            zero_terms = set()
                            if branch == "head_only" or linear:
                                zero_terms |= {"internal_term", "cross_term"}
                            if branch == "internal_only":
                                zero_terms |= {"head_term", "cross_term"}
                            if branch == "frozen" or trial == 0:
                                zero_terms |= {"head_term", "internal_term", "cross_term", "total_raw_gap_change"}
                            require(all(np.array_equal(values[key], np.zeros(cfg["eval_trials"])) for key in zero_terms), "Every inactive algebraic term exactly zero")
                            controls["masked_zero_term_entries_checked"] += len(zero_terms) * cfg["eval_trials"]
                            if linear:
                                require(np.array_equal(values["raw_gap_00"], values["raw_gap_01"])
                                        and np.array_equal(values["raw_gap_10"], values["raw_gap_11"]), "Linear no-internal crossed state equivalence")
                                controls["linear_internal_zero_checks"] += 1
                            require((values["aligned_total_raw_gap_change"] * values["policy_rule_change"] >= -TOLERANCE).all(), "Target-oriented raw/policy change sign")
                            for case in range(cfg["eval_trials"]):
                                row = next(reader, None)
                                require(row is not None and None not in row and all(v is not None for v in row.values()), "Complete nonmalformed static case row")
                                expected = {"seed": seed, "model": model, "branch": branch, "trial": trial, "split": split, "case": case,
                                    "family_id": str(data[split]["family_ids"][case]), "left_count": int(data[split]["counts"][case, 0]),
                                    "right_count": int(data[split]["counts"][case, 1]), **{key: value[case] for key, value in values.items()}}
                                for key, value in expected.items():
                                    if key in INTEGER_FIELDS:
                                        require(integer(row, key) == int(value), "Exact static case coordinate/target/strict clipping flag: " + key)
                                    elif key in {"model", "branch", "split", "family_id"}:
                                        require(row[key] == value, "Exact source case identity: " + key)
                                    else:
                                        near(number(row, key), float(value), "Independent four-cross raw, aligned, policy and interaction value: " + key)
                                for key in ("bilinear_residual", "raw_interaction_residual"):
                                    require(abs(number(row, key)) <= TOLERANCE, "Every actual CSV arithmetic residual within tolerance")
                            summaries.append(seed_summary(seed, model, branch, trial, split, values))
                            counts["cases"] += cfg["eval_trials"]
                            counts["crossed_policy_values"] += 4 * cfg["eval_trials"]
            require(next(reader, None) is None, "No duplicate or extra source case rows")
        for left, right in (("full", "head_only"), ("internal_only", "frozen")):
            for trial in cfg["checkpoints"]:
                require(all(np.array_equal(linear_snapshots[left, trial][k], linear_snapshots[right, trial][k]) for k in SNAPSHOT_FIELDS),
                        "Exact two linear factorial negative-control state pairs")
                counts["linear_equivalent_checkpoint_pairs"] += 1
        print(f"independently checked static decomposition seed {seed} ({cfg['seeds'].index(seed) + 1}/{len(cfg['seeds'])})", flush=True)
    equal(read_json(output / "controls.json"), controls, "Every frozen/masked/diagonal numerical control count")
    summary_check(cfg, summaries, read_json(output / "summary.json"))
    for key in ("cases", "crossed_policy_values"):
        equal(counts[key], cfg[key], "Complete independently counted static evaluation budget")
        equal(meta[key], counts[key], "Executed static case budget")
    require(type(meta["behavior_hypotheses_tested"]) is int and meta["behavior_hypotheses_tested"] == 0, "No new behavior hypotheses tested")
    report.update(study_id=cfg["study_id"], summary_sha256=digest(output / "summary.json"),
        registration_sha256=digest(registration), config_sha256=reg["config_sha256"],
        artifact_hashes_sha256=digest(output / "artifact_hashes.json"), input_manifest_sha256=canonical_input_digest(reg["input_files"]))
    report["counts"] = {**counts, **controls, "selected_input_files": len(entries), "parent_primary_artifacts": parent_artifacts,
        "source_hashes": len(reg["frozen_files"]), "artifact_hashes": len(hashes), "seed_summary_rows": len(summaries),
        "seeds": len(cfg["seeds"]), "models": 4, "branches": 4, "checkpoints_per_branch": len(cfg["checkpoints"])}
    report["checks"] = ["Exact source/config/ten-file completed parent gate, every parent artifact and selected NPZ bytes",
        "Prior completed parent verification and static registration chronology; no new training, construction, generation or RNG",
        "All19 inherited native prefix fields, unchanged initialization, masked coefficients and linear factorial controls",
        "Every input identity and every four-cross raw/aligned score, bilinear term, interaction and clipped sigmoid policy",
        "Same smaller target for baseline/current states and original00/11 diagonal policy within1e-12",
        "Every case residual below1e-10, exact abs(raw)>40 flags, all-denominator means and between-seed sample SD"]


def verify(root, registration, result_dir):
    root, registration, output = (Path(p).resolve() for p in (root, registration, result_dir))
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "status": "failed", "passed": False,
        "method": "independent saved-array numpy/stdlib arithmetic; explicit immutable independent-verifier helpers only",
        "counts": {}, "checks": [], "errors": [], "not_verified": [
            "Private remote publication authenticity or literal historical Git blobs (checked separately by archive validation)",
            "Causal learning contribution fractions from off-trajectory crossed coefficient states",
            "New behavioral success, independent biological mechanisms, anatomy superiority or human/fly equivalence"]}
    try:
        require(root.is_dir() and registration.is_file() and output.is_dir(), "Existing source, static result and registration required")
        audit(root, registration, output, report)
        report.update(status="passed", passed=True)
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Verification report output already exists")
    report = verify(args.root, args.registration, args.result_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "errors": report["errors"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
