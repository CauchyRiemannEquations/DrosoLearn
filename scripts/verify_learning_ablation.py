"""Independent replay of matched-prefix, factorial learning ablations.

Only immutable independent-verifier arithmetic is reused. This verifier never
imports or calls a producer, learner, update, forward or summary implementation.
It reads primary artifacts and writes only a new report through its CLI.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "independent_stochastic_feedback_arithmetic", Path(__file__).with_name("verify_stochastic_feedback.py"))
_OLD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_OLD)
# Explicit reuse is restricted to arithmetic, decoding and source validation
# helpers from an independent verifier; no old runner or old verify() is used.
require, near = _OLD._require, _OLD._near
integer, number = _OLD._integer, _OLD._number
digest, read_json, safe, utc, equal = _OLD.digest, _OLD.read_json, _OLD.safe, _OLD.utc, _OLD.equal
csv_index, npz = _OLD.csv_index, _OLD._npz
dataset, regenerate_dataset, inputs = _OLD._dataset, _OLD._regenerate_dataset, _OLD._inputs
state, raw_features = _OLD._state, _OLD._raw_features
snapshot_check, auc, metrics = _OLD._snapshot_check, _OLD._auc, _OLD.metrics
initialize, regenerate_graphs = _OLD.initialize, _OLD.regenerate_graphs

MODELS = ("linear", "real", "rewired", "ann")
BRANCHES = ("full", "head_only", "internal_only", "frozen")
MASKS = {"prefix": (True, True), "full": (True, True), "head_only": (True, False),
         "internal_only": (False, True), "frozen": (False, False)}
INITIAL_CONDITIONS = {"linear": "linear", "real": "plastic", "rewired": "rewired_plastic", "ann": "ann_plastic"}
ENV = {"id": "probabilistic", "q_optimal": .8, "q_other": .2}
RNG = {"implementation": "numpy.random.default_rng", "phase1_action_offset": 610000,
       "phase2_action_offset": 620000, "phase1_reward_offset": 630000, "phase2_reward_offset": 640000,
       "rewired_offset": 650000, "ann_offset": 660000, "encoder_offset": 670000,
       "data_multiplier": 100, "train_phase1_offset": 61, "train_phase2_offset": 64,
       "iid_offset": 62, "ood_offset": 63,
       "coupling": "same_phase_specific_action_and_reward_arrays_per_seed_reused_across_all_models_and_branches"}
CRITERIA = {"acquisition_probability": .75, "acquisition_gain": .15,
            "adaptation_probability": .75, "mean_component_effect": .05,
            "positive_seed_fraction": .8}
SNAPSHOT_FIELDS = _OLD.CHECKPOINT_FIELDS | {"phase_head_updates", "phase_internal_updates"}
TRIAL_FIELDS = {"seed", "model", "branch", "phase", "trial", "global_trial", "family_id", "left_count", "right_count",
                "chosen_count", "action", "reward", "p_right", "q_reward", "action_uniform", "reward_uniform",
                "head_enabled", "internal_enabled", "head_update_called", "internal_update_called", "optimizer_factor",
                "internal_step_l2", "readout_step_l2", "kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2",
                "decision_left_l2", "decision_right_l2", "head_updates", "internal_updates",
                "cumulative_head_updates", "cumulative_internal_updates"}
CURVE_FIELDS = {"seed", "model", "branch", "phase", "trial", "split", "p_rule", "p_larger", "p_smaller", "greedy_rule",
                "mean_policy_sensitivity", "expected_reward", "internal_change_l2", "readout_change_l2", "probe_kc_change_l2",
                "probe_mbon_change_l2", "probe_decision_change_l2", "phase_internal_change_l2", "phase_readout_change_l2",
                "phase_probe_kc_change_l2", "phase_probe_mbon_change_l2", "phase_probe_decision_change_l2",
                "lower_bound_fraction", "upper_bound_fraction", "head_updates", "internal_updates",
                "cumulative_head_updates", "cumulative_internal_updates"}


def config_check(cfg):
    require(cfg["study_id"] == "learning_ablation_v1"
            and cfg["scope"] in {"preregistered_diagnostic", "integration_test"}, "Study identity and scope")
    require(cfg["models"] == list(MODELS) and cfg["branches"] == list(BRANCHES), "All four models and factorial branches")
    require(cfg["pair"] == [2, 4] and cfg["graph"] == "data/flywire_783_mb_left_kc512_s701",
            "Pinned artificial pair and anatomical source")
    equal(cfg["rng"], RNG, "Registered independent/shared streams")
    equal(cfg["environment"], ENV, "Only informative .8/.2 actual sampled reward")
    require(cfg["history"] == "shared_high_prefix_then_deepcopied_low_reversal_2x2_masks"
            and cfg["feature_normalization"] == "fixed_initial_phase1_train_feature_gap_rms"
            and cfg["head_initialization"] == "zeros_all_models"
            and cfg["update_interface"] == "inputs_action_actual_reward_and_explicit_head_internal_masks_only_no_labels_or_reward_probabilities",
            "Fixed matched-prefix history, normalization and reward/mask-only learning interface")
    equal(cfg["criteria"], CRITERIA, "Prospective acquisition, adaptation and component-effect criteria")
    seeds = cfg["seeds"]
    require(type(seeds) is list and seeds and len(seeds) == len(set(seeds))
            and all(type(s) is int and s >= 0 for s in seeds), "Distinct nonnegative integer seeds")
    budget = tuple(cfg[k] for k in ("trials_per_phase", "checkpoint_every", "eval_trials", "probe_rows"))
    n, every, ev, probes = budget
    require(all(type(x) is int and x > 0 for x in budget)
            and n % 4 == ev % 4 == n % every == 0 and probes <= ev, "Crossing, checkpoint and probe budgets")
    require(cfg["head_lr"] == .1 and cfg["internal_lr"] == .01, "Fixed learning rates")
    if cfg["scope"] == "preregistered_diagnostic":
        require(seeds == list(range(12001, 12011)) and budget == (400, 100, 128, 8),
                "Prospective scientific seeds and resources")
    else:
        require(not set(seeds).intersection(range(12001, 12011)), "Scientific ablation seeds forbidden in fixtures")
        require(all(s in (17, 18) for s in seeds), "Only designated engineering seeds17/18")
    total = len(seeds) * 4 * 5 * n
    for field, value in (("actual_choice_feedback_trials", total),
                         ("expected_head_update_calls", len(seeds) * 4 * 3 * n),
                         ("expected_internal_update_calls", len(seeds) * 3 * 3 * n),
                         ("expected_privileged_update_calls", 0)):
        equal(cfg[field], value, f"Exact registered physical budget: {field}")


def masked_step(stimulus, action_uniform, reward_uniform, larger,
                encoder, bias, pre, post, weights, head, scale, linear,
                head_enabled, internal_enabled, head_lr, internal_lr, upper):
    kc, mb, decision, p = state(stimulus, encoder, bias, pre, post, weights, head, scale, linear)
    p = float(p)
    action = int(action_uniform < p)
    high = int(stimulus[1, 0] > stimulus[0, 0])
    target = high if larger else 1 - high
    q = .8 if action == target else .2
    reward = int(reward_uniform < q)
    factor = (action - p) * (reward - .5)
    new_head, new_weights = head.copy(), weights.copy()
    if head_enabled:
        new_head += head_lr * factor * (decision[1] - decision[0])
    internal_call = bool(internal_enabled and not linear)
    if internal_call:
        eligibility = head[post] * (kc[1, pre] - kc[0, pre]) / scale
        new_weights = np.minimum(np.maximum(weights + internal_lr * factor * eligibility, 0.), upper)
    return {"action": action, "reward": reward, "p_right": p, "q_reward": q,
            "optimizer_factor": factor, "head_update_called": int(head_enabled),
            "internal_update_called": int(internal_call),
            "internal_step_l2": float(np.linalg.norm(new_weights - weights)),
            "readout_step_l2": float(np.linalg.norm(new_head - head)),
            "kc_left_l2": float(np.linalg.norm(kc[0])), "kc_right_l2": float(np.linalg.norm(kc[1])),
            "mbon_left_l2": float(np.linalg.norm(mb[0])), "mbon_right_l2": float(np.linalg.norm(mb[1])),
            "decision_left_l2": float(np.linalg.norm(decision[0])), "decision_right_l2": float(np.linalg.norm(decision[1]))}, new_weights, new_head


def source_check(root, registration_path, output, reg, metadata):
    cfg = reg["config"]
    config_check(cfg)
    equal(metadata["config"], cfg, "Executed registered configuration")
    require(metadata["study_id"] == reg["study_id"] == cfg["study_id"]
            and metadata["status"] == "completed", "Completed matching study")
    require(type(reg["source_git_commit"]) is str and re.fullmatch(r"[a-f0-9]{40}", reg["source_git_commit"])
            and metadata["source_git_commit"] == reg["source_git_commit"], "Recorded source Git version")
    require(digest(registration_path) == digest(output / "registration.json") == metadata["registration_sha256"],
            "External/copied registration exact bytes")
    equal(read_json(output / "registration.json"), reg, "Registration contents")
    times = [utc(reg["registered_at_utc"]), utc(metadata["started_at_utc"]), utc(metadata["completed_at_utc"])]
    require(times == sorted(times), "Local registration precedes execution completion")
    config = safe(root, "configs/learning_ablation_v1.json")
    require(digest(config) == reg["config_sha256"] == metadata["config_sha256"], "Config byte identity")
    equal(read_json(config), cfg, "Source configuration contents")
    require(type(cfg["frozen_files"]) is list and cfg["frozen_files"]
            and len(cfg["frozen_files"]) == len(set(cfg["frozen_files"]))
            and set(reg["frozen_files"]) == set(cfg["frozen_files"]), "Exact frozen source closure")
    for name, sha in reg["frozen_files"].items():
        require(type(sha) is str and re.fullmatch(r"[a-f0-9]{64}", sha)
                and digest(safe(root, name)) == sha, f"Frozen source bytes: {name}")
    equal(metadata["frozen_files"], reg["frozen_files"], "Executed source closure")
    require(type(cfg["parent_files"]) is list and len(cfg["parent_files"]) == len(set(cfg["parent_files"]))
            and set(reg["parent_files"]) == set(cfg["parent_files"]), "Exact parent gate evidence closure")
    for name, sha in reg["parent_files"].items():
        require(digest(safe(root, name)) == sha, f"Unchanged parent evidence: {name}")
    equal(metadata["parent_files"], reg["parent_files"], "Executed parent evidence")
    return cfg


def parent_gate(root, cfg, reg):
    names = {"summary_sha256": "results/stochastic_feedback_network_v1/summary.json",
             "verification_sha256": "results/stochastic_feedback_network_v1/verification.json",
             "registration_sha256": "results/registration_stochastic_feedback_network_v1.json",
             "config_sha256": "configs/stochastic_feedback_network_v1.json",
             "artifact_hashes_sha256": "results/stochastic_feedback_network_v1/artifact_hashes.json"}
    require(set(names.values()) == set(cfg["parent_files"]), "Exactly five completed network gate files")
    identity = cfg["parent_gate_identity"]
    require(set(identity) == set(names) | {"published_commit"}, "Complete pinned network publication identity")
    for key, value in identity.items():
        require(type(value) is str and re.fullmatch(r"[a-f0-9]{40}" if key == "published_commit" else r"[a-f0-9]{64}", value),
                f"Parent identity digest format: {key}")
    for key, name in names.items():
        require(identity[key] == digest(safe(root, name)), f"Pinned completed parent bytes: {key}")
    summary = read_json(safe(root, names["summary_sha256"]))
    checked = read_json(safe(root, names["verification_sha256"]))
    hashes = read_json(safe(root, names["artifact_hashes_sha256"]))
    require(summary["study_id"] == "stochastic_feedback_network_v1" and summary["status"] == "completed" and summary["stage"] == "network"
            and checked["status"] == "passed", "Completed and independently verified network prerequisite")
    for key in ("summary_sha256", "registration_sha256", "config_sha256"):
        require(checked[key] == identity[key], f"Parent independent verification binds exact bytes: {key}")
    require(hashes["summary.json"] == identity["summary_sha256"], "Parent inventory binds exact summary")
    require(utc(checked["verified_at_utc"]) <= utc(reg["registered_at_utc"]), "New registration follows parent independent verification")
    expected = {"name": "verified_stochastic_network_before_ablation", "passed": True, **identity}
    equal(reg["gate"], expected, "Pinned completed parent gate")
    return expected


def replay_phase(base, seed, model, branch, cfg, datasets, streams, initial,
                 start_weights, start_head, inherited_head, inherited_internal, graph_counts, curves, trials):
    encoder, bias, pre, post, w0, h0, scale, linear, _ = initial
    phase = "phase1" if branch == "prefix" else "phase2"
    head_mask, internal_mask = MASKS[branch]
    internal_active = internal_mask and not linear
    n, every = cfg["trials_per_phase"], cfg["checkpoint_every"]
    probe = inputs(datasets["iid"], np.arange(1, 9))[:cfg["probe_rows"]]
    original_kc, original_mb, original_decision, _ = state(probe, encoder, bias, pre, post, w0, h0, scale, linear)
    phase_kc, phase_mb, phase_decision, _ = state(probe, encoder, bias, pre, post, start_weights, start_head, scale, linear)
    weights, head = start_weights.copy(), start_head.copy()
    upper = 2 * w0
    normalizer = np.empty(0) if linear else np.maximum(np.bincount(post, weights=graph_counts, minlength=h0.size), 1.)
    local_head = local_internal = 0
    measurements, snapshots = {}, {}
    path = base / model / branch

    def checkpoint(t):
        saved = npz(path / f"{phase}_{t:04d}.npz", SNAPSHOT_FIELDS)
        snapshot_check(saved, probe, encoder, bias, pre, post, weights, head, scale, linear,
                       inherited_head + local_head, inherited_internal + local_internal, f"{seed}/{model}/{branch}/{t}")
        for name, value in (("initial_internal_weights", w0), ("internal_upper_bounds", upper), ("incoming_normalizer", normalizer)):
            near(saved[name], value, f"Original initialization/clipping/normalization: {name}")
        for name, value in (("phase_head_updates", local_head), ("phase_internal_updates", local_internal)):
            require(saved[name].shape == () and np.issubdtype(saved[name].dtype, np.integer)
                    and int(saved[name]) == value, "Exact local counter independent of inherited native counter")
        snapshots[t] = saved
        kc, mb, decision, _ = state(probe, encoder, bias, pre, post, weights, head, scale, linear)
        changes = {}
        for prefix, ref_w, ref_h, ref_k, ref_m, ref_d in (
                ("", w0, h0, original_kc, original_mb, original_decision),
                ("phase_", start_weights, start_head, phase_kc, phase_mb, phase_decision)):
            for field, value in (("internal_change_l2", np.linalg.norm(weights - ref_w)),
                                 ("readout_change_l2", np.linalg.norm(head - ref_h)),
                                 ("probe_kc_change_l2", np.linalg.norm(kc - ref_k)),
                                 ("probe_mbon_change_l2", np.linalg.norm(mb - ref_m)),
                                 ("probe_decision_change_l2", np.linalg.norm(decision - ref_d))):
                changes[prefix + field] = float(value)
        changes.update(lower_bound_fraction=float(np.mean(weights <= 0)) if weights.size else 0.,
                       upper_bound_fraction=float(np.mean(weights >= upper)) if weights.size else 0.)
        for split in ("iid", "ood"):
            ds = datasets[split]
            _, _, _, probability = state(inputs(ds, np.arange(1, 9)), encoder, bias, pre, post, weights, head, scale, linear)
            near(saved[f"{split}_p_right"], probability, "All independently replayed evaluation cases")
            value = metrics(probability, ds["counts"], phase == "phase1", ENV)
            measurements[t, split] = value
            row = curves[seed, model, branch, phase, t, split]
            for field, val in {**value, **changes}.items():
                near(number(row, field), val, f"Every local/total change and metric: {field}")
            for field, val in (("head_updates", local_head), ("internal_updates", local_internal),
                               ("cumulative_head_updates", inherited_head + local_head),
                               ("cumulative_internal_updates", inherited_internal + local_internal)):
                require(integer(row, field) == val, "Curve local/native counter distinction")
        if branch == "frozen":
            require(np.array_equal(weights, start_weights) and np.array_equal(head, start_head), "Learned frozen state is retained")
    checkpoint(0)
    ds = datasets[f"train_{phase}"]
    for index, stimulus in enumerate(inputs(ds, np.arange(1, 9))):
        row = trials[seed, model, branch, phase, index + 1]
        u, v = (streams[phase, kind][index] for kind in ("action", "reward"))
        result, next_weights, next_head = masked_step(stimulus, u, v, phase == "phase1", encoder, bias, pre, post,
            weights, head, scale, linear, head_mask, internal_mask, cfg["head_lr"], cfg["internal_lr"], upper)
        require(row["family_id"] == str(ds["family_ids"][index]), "Original shared stimulus identity")
        for field, value in (("global_trial", index + 1 + (n if phase == "phase2" else 0)),
                             ("left_count", int(ds["counts"][index, 0])), ("right_count", int(ds["counts"][index, 1])),
                             ("chosen_count", int(ds["counts"][index, result["action"]])),
                             ("head_enabled", int(head_mask)), ("internal_enabled", int(internal_mask))):
            require(integer(row, field) == value, f"Trial identity and requested masks: {field}")
        for field, value in {**result, "action_uniform": u, "reward_uniform": v}.items():
            if field in {"action", "reward", "head_update_called", "internal_update_called"}:
                require(integer(row, field) == value, f"Actual selected feedback and available mask call: {field}")
            else:
                near(number(row, field), value, f"Every independent masked update/activity: {field}")
        local_head += int(head_mask)
        local_internal += int(internal_active)
        for field, value in (("head_updates", local_head), ("internal_updates", local_internal),
                             ("cumulative_head_updates", inherited_head + local_head),
                             ("cumulative_internal_updates", inherited_internal + local_internal)):
            require(integer(row, field) == value, f"Actual local/native mask counter: {field}")
        if not internal_active or not np.any(head):
            require(number(row, "internal_step_l2") == 0., "Unavailable or zero-old-head internal drift")
        if not head_mask:
            require(number(row, "readout_step_l2") == 0., "Masked output has no drift")
        weights, head = next_weights, next_head
        if (index + 1) % every == 0:
            checkpoint(index + 1)
    return {"weights": weights, "head": head, "native_head": inherited_head + local_head,
            "native_internal": inherited_internal + local_internal, "local_head": local_head,
            "local_internal": local_internal, "snapshots": snapshots, "measurements": measurements}


def component_hypothesis(name, comparison, pairs, criteria):
    require(name in {"H_I", "H_H"} and pairs and len({r["seed"] for r in pairs}) == len(pairs),
            "Two distinct all-seed component hypotheses")
    differences = [r["difference"] for r in pairs]
    require(all(type(x) in (int, float) and np.isfinite(x) for x in differences), "Finite all-seed differences")
    average = float(np.mean(differences))
    positive = sum(x > 0 for x in differences)
    mean_met = average >= criteria["mean_component_effect"]
    positive_met = positive / len(pairs) >= criteria["positive_seed_fraction"]
    return {"name": name, "model": "real", "split": "iid", "comparison": comparison, "metric": "phase2_end",
            "n_seeds": len(pairs), "all_seed_differences": pairs, "mean_effect": average, "positive_seed_count": positive,
            "mean_criterion_met": bool(mean_met), "positive_fraction_criterion_met": bool(positive_met),
            "criterion_met": bool(mean_met and positive_met)}


def summary_check(cfg, values, actual):
    n, every, c = cfg["trials_per_phase"], cfg["checkpoint_every"], cfg["criteria"]
    times = list(range(0, n + 1, every))
    outcomes, failures = [], []
    for seed in cfg["seeds"]:
        for model in MODELS:
            for branch in BRANCHES:
                for split in ("iid", "ood"):
                    prefix = [values[seed, model, "prefix", t, split]["p_rule"] for t in times]
                    after = [values[seed, model, branch, t, split]["p_rule"] for t in times]
                    acquired = bool(prefix[-1] >= c["acquisition_probability"] and prefix[-1] - prefix[0] >= c["acquisition_gain"])
                    adapted = bool(after[-1] >= c["adaptation_probability"])
                    area = auc(after, times)
                    outcomes.append({"seed": seed, "model": model, "branch": branch, "split": split,
                        "prefix_start": prefix[0], "prefix_end": prefix[-1], "prefix_gain": prefix[-1] - prefix[0],
                        "prefix_acquired": acquired, "phase2_start": after[0], "phase2_end": after[-1],
                        "phase2_gain": after[-1] - after[0], "phase2_raw_auc": area,
                        "phase2_baseline_adjusted_auc": area - after[0], "phase2_threshold_met": adapted,
                        "qualified_adaptation": bool(acquired and adapted),
                        "first_observed_threshold_checkpoint": next((t for t, v in zip(times, after, strict=True)
                                                                      if v >= c["adaptation_probability"]), None)})
                    for reason, passed in (("prefix_acquisition_threshold_not_met", acquired),
                                           ("phase2_endpoint_threshold_not_met", adapted),
                                           ("acquired_old_rule_then_phase2_threshold_not_met", not acquired or adapted)):
                        if not passed:
                            failures.append({"seed": seed, "model": model, "branch": branch, "split": split, "reason": reason})
    keyed = {(r["seed"], r["model"], r["branch"], r["split"]): r for r in outcomes}
    metric_names = ("phase2_start", "phase2_end", "phase2_gain", "phase2_raw_auc", "phase2_baseline_adjusted_auc")
    # Expressions are explicit factorial sums, independent of producer tables.
    combinations = {
        "full_minus_head_only": (1., -1., 0., 0.),
        "full_minus_internal_only": (1., 0., -1., 0.),
        "head_only_minus_frozen": (0., 1., 0., -1.),
        "internal_only_minus_frozen": (0., 0., 1., -1.),
        "full_minus_frozen": (1., 0., 0., -1.),
        "interaction": (1., -1., -1., 1.),
        "head_main_effect": (.5, .5, -.5, -.5),
        "internal_main_effect": (.5, -.5, .5, -.5),
    }
    contrasts = []
    for seed in cfg["seeds"]:
        for model in MODELS:
            for split in ("iid", "ood"):
                for comparison, factors in combinations.items():
                    for metric_name in metric_names:
                        value = float(sum(factor * keyed[seed, model, branch, split][metric_name]
                                          for branch, factor in zip(BRANCHES, factors, strict=True) if factor))
                        contrasts.append({"seed": seed, "model": model, "split": split, "comparison": comparison,
                                          "metric": metric_name, "difference": value})
    statistics = []
    for model in MODELS:
        for split in ("iid", "ood"):
            for comparison in combinations:
                for metric_name in metric_names:
                    selected = [r["difference"] for r in contrasts if
                                (r["model"], r["split"], r["comparison"], r["metric"]) == (model, split, comparison, metric_name)]
                    require(len(selected) == len(cfg["seeds"]), "Retain all seeds in every factorial contrast")
                    statistics.append({"model": model, "split": split, "comparison": comparison, "metric": metric_name,
                                       "n_seeds": len(selected), "positive_seed_count": sum(v > 0 for v in selected),
                                       "mean": float(np.mean(selected)),
                                       "sample_sd": float(np.std(selected, ddof=1)) if len(selected) > 1 else None})
    hypotheses = {}
    for name, comparison in (("H_I", "full_minus_head_only"), ("H_H", "full_minus_internal_only")):
        selected = {r["seed"]: r["difference"] for r in contrasts if
                    (r["model"], r["split"], r["comparison"], r["metric"]) == ("real", "iid", comparison, "phase2_end")}
        pairs = [{"seed": seed, "difference": selected[seed]} for seed in cfg["seeds"]]
        hypotheses[name] = component_hypothesis(name, comparison, pairs, c)
    verdicts = []
    for model in MODELS:
        for branch in BRANCHES:
            for split in ("iid", "ood"):
                selected = [keyed[seed, model, branch, split] for seed in cfg["seeds"]]
                statistics_by_metric = {}
                for field in ("prefix_start", "prefix_end", "prefix_gain", *metric_names):
                    data = [r[field] for r in selected]
                    statistics_by_metric[field] = {"mean": float(np.mean(data)),
                                                   "sample_sd": float(np.std(data, ddof=1)) if len(data) > 1 else None}
                verdicts.append({"model": model, "branch": branch, "split": split, "n_seeds": len(selected),
                                 "prefix_acquisition_count": sum(r["prefix_acquired"] for r in selected),
                                 "phase2_threshold_count": sum(r["phase2_threshold_met"] for r in selected),
                                 "qualified_adaptation_count": sum(r["qualified_adaptation"] for r in selected),
                                 "statistics": statistics_by_metric})
    expected = {"study_id": cfg["study_id"], "status": "completed", "scope": cfg["scope"], "outcomes": outcomes,
                "branch_verdicts": verdicts, "paired_contrasts": contrasts, "contrast_statistics": statistics,
                "primary_hypotheses": hypotheses, "failures": failures, "prefix_failure_exclusion": False,
                "interpretation": "Matched-prefix computational component intervention; artificial supplied quantity, no biological mechanism or topology-superiority claim"}
    equal(actual, expected, "Every independently reconstructed outcome, factorial contrast, hypothesis and failure")


def audit(root, registration_path, output, report):
    reg, metadata, summary = (read_json(p) for p in (registration_path, output / "metadata.json", output / "summary.json"))
    cfg = source_check(root, registration_path, output, reg, metadata)
    equal(metadata["gate"], parent_gate(root, cfg, reg), "Executed pinned parent gate")
    n, ev, every = cfg["trials_per_phase"], cfg["eval_trials"], cfg["checkpoint_every"]
    times, seeds = list(range(0, n + 1, every)), cfg["seeds"]
    required = {"metadata.json", "registration.json", "summary.json", "learning_curves.csv", "controls.json",
                "model_resources.json", "stages.json", "clone_evidence.json"}
    for seed in seeds:
        base = f"seed_{seed}"
        required.update(f"{base}/{name}" for name in ("stimuli.npz", "trials.csv", "real_topology.npz", "rewired_topology.npz", "ann_topology.npz"))
        for model in MODELS:
            for branch in ("prefix", *BRANCHES):
                phase = "phase1" if branch == "prefix" else "phase2"
                required.update(f"{base}/{model}/{branch}/{phase}_{t:04d}.npz" for t in times)
    hashes = read_json(output / "artifact_hashes.json")
    require(set(hashes) == required, "Exact shared-prefix primary artifact coverage")
    for name, sha in hashes.items():
        require(digest(safe(output, name)) == sha, f"Primary artifact exact bytes: {name}")
    actual_files = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    require(actual_files - {"artifact_hashes.json", "verification.json"} == required, "No unregistered primary artifacts")
    curves = csv_index(output / "learning_curves.csv", CURVE_FIELDS, ("seed", "model", "branch", "phase", "trial", "split"))
    expected_curves = {(seed, model, branch, "phase1" if branch == "prefix" else "phase2", t, split)
                       for seed in seeds for model in MODELS for branch in ("prefix", *BRANCHES)
                       for t in times for split in ("iid", "ood")}
    require(set(curves) == expected_curves, "One actual prefix and every branch evaluation coordinate")
    controls_list = read_json(output / "controls.json")
    controls = {(r["seed"], r["kind"]): r for r in controls_list}
    require(len(controls_list) == len(controls) and set(controls) == {(s, k) for s in seeds for k in ("real", "rewired", "ann")},
            "Exact independently reconstructed topology controls")
    resources, stages, clones, measurements = [], [], [], {}
    fingerprints = set()
    counts = {"actual_choice_feedback_trials": 0, "head_update_calls": 0, "internal_update_calls": 0,
              "privileged_update_calls": 0, "prefix_training_trials": 0, "branch_training_trials": 0,
              "checkpoints": 0, "case_probability_entries": 0, "prefix_clones": 0,
              "linear_equivalent_checkpoint_pairs": 0, "linear_equivalent_trial_pairs": 0,
              "fixed_initialization_field_checks": 0, "masked_parameter_checkpoint_checks": 0}
    splits = (("train_phase1", n, 61), ("train_phase2", n, 64), ("iid", ev, 62), ("ood", ev, 63))
    for seed in seeds:
        base = output / f"seed_{seed}"
        stimulus_fields = {f"{name}_{field}" for name, _, _ in splits for field in ("counts", "nuisance", "family_ids")}
        stimulus_fields.update(f"{kind}_{phase}_uniforms" for phase in ("phase1", "phase2") for kind in ("action", "reward"))
        stimuli = npz(base / "stimuli.npz", stimulus_fields)
        datasets = {}
        for name, size, offset in splits:
            ds = dataset(stimuli, name, size, (2, 4), fingerprints)
            expected = regenerate_dataset(seed * 100 + offset, name, size, (2, 4))
            require(all(np.array_equal(ds[k], v) for k, v in expected.items()), "Independent shared input RNG and canonical crossing")
            datasets[name] = ds
        streams = {}
        for phase in ("phase1", "phase2"):
            for kind in ("action", "reward"):
                values = np.random.default_rng(seed + cfg["rng"][f"{phase}_{kind}_offset"]).random(n)
                require(np.array_equal(stimuli[f"{kind}_{phase}_uniforms"], values), "Separate registered shared action/reward streams")
                streams[phase, kind] = values
        trials = csv_index(base / "trials.csv", TRIAL_FIELDS, ("seed", "model", "branch", "phase", "trial"))
        expected_trials = {(seed, model, branch, "phase1" if branch == "prefix" else "phase2", t)
                           for model in MODELS for branch in ("prefix", *BRANCHES) for t in range(1, n + 1)}
        require(set(trials) == expected_trials, "Exactly one prefix training log and four phase2 branch logs")
        graphs, encoder, bias, n_mbon = regenerate_graphs(root, cfg, base, seed,
                                                        {kind: controls[seed, kind] for kind in ("real", "rewired", "ann")})
        for model in MODELS:
            graph = None if model == "linear" else graphs[model]
            graph_counts = np.empty(0) if graph is None else graph["counts"]
            initial = initialize(INITIAL_CONDITIONS[model], graph, encoder, bias,
                                 inputs(datasets["train_phase1"], np.arange(1, 9)), n_mbon)
            ancestor = replay_phase(base, seed, model, "prefix", cfg, datasets, streams, initial,
                                    initial[4], initial[5], 0, 0, graph_counts, curves, trials)
            replays = {"prefix": ancestor}
            for branch in BRANCHES:
                replay = replay_phase(base, seed, model, branch, cfg, datasets, streams, initial,
                                      ancestor["weights"], ancestor["head"], ancestor["native_head"], ancestor["native_internal"],
                                      graph_counts, curves, trials)
                start, prefix_end = replay["snapshots"][0], ancestor["snapshots"][n]
                require(all(np.array_equal(start[k], prefix_end[k]) for k in _OLD.CHECKPOINT_FIELDS),
                        "Each branch inherits exactly all19 learned prefix arrays including native counters")
                require(int(start["phase_head_updates"]) == int(start["phase_internal_updates"]) == 0,
                        "Only branch local counters start at zero")
                clones.append({"seed": seed, "model": model, "branch": branch,
                               "all_native_checkpoint_fields_equal": True, "inherited_head_updates": ancestor["native_head"],
                               "inherited_internal_updates": ancestor["native_internal"]})
                replays[branch] = replay
                counts["prefix_clones"] += 1
            for branch, replay in replays.items():
                phase = "phase1" if branch == "prefix" else "phase2"
                head_mask, internal_mask = MASKS[branch]
                internal_active = internal_mask and model != "linear"
                for snapshot in replay["snapshots"].values():
                    for field in ("encoder", "bias", "pre", "post", "feature_scale", "initial_feature_gap_rms",
                                  "initial_internal_weights", "internal_upper_bounds", "incoming_normalizer"):
                        require(np.array_equal(snapshot[field], ancestor["snapshots"][0][field]),
                                f"Exact unchanged initial bound/scale/encoder/topology field: {field}")
                        counts["fixed_initialization_field_checks"] += 1
                    for field, enabled in (("readout_weights", head_mask), ("internal_weights", internal_active)):
                        if not enabled:
                            require(np.array_equal(snapshot[field], replay["snapshots"][0][field]),
                                    f"Exact retained masked parameter state: {field}")
                            counts["masked_parameter_checkpoint_checks"] += 1
                for key, value in replay["measurements"].items():
                    measurements[seed, model, branch, *key] = value
                resources.append({"seed": seed, "model": model, "branch": branch, "phase": phase,
                    "feature_scale": initial[6], "initial_feature_gap_rms": initial[6],
                    "readout_parameters": initial[5].size, "internal_parameters": initial[4].size,
                    "trainable_readout_parameters": initial[5].size if head_mask else 0,
                    "trainable_internal_parameters": initial[4].size if internal_active else 0,
                    "head_update_calls": replay["local_head"], "internal_update_calls": replay["local_internal"], "privileged_update_calls": 0})
                rows = [trials[seed, model, branch, phase, t] for t in range(1, n + 1)]
                chosen = sum(integer(r, "chosen_count") == (4 if phase == "phase1" else 2) for r in rows)
                reward_count = sum(integer(r, "reward") for r in rows)
                stages.append({"seed": seed, "model": model, "branch": branch, "phase": phase, "actual_trials": n,
                    "actual_designated_choice_count": chosen, "actual_designated_choice_fraction": chosen / n,
                    "actual_reward_count": reward_count, "actual_reward_rate": reward_count / n,
                    "head_update_calls": replay["local_head"], "internal_update_calls": replay["local_internal"],
                    "cumulative_head_updates": replay["native_head"], "cumulative_internal_updates": replay["native_internal"]})
                counts["actual_choice_feedback_trials"] += n
                counts["prefix_training_trials" if branch == "prefix" else "branch_training_trials"] += n
                counts["head_update_calls"] += replay["local_head"]
                counts["internal_update_calls"] += replay["local_internal"]
                counts["checkpoints"] += len(replay["snapshots"])
                counts["case_probability_entries"] += len(replay["snapshots"]) * 2 * ev
            if model == "linear":
                for left, right in (("full", "head_only"), ("internal_only", "frozen")):
                    for t in times:
                        require(all(np.array_equal(replays[left]["snapshots"][t][k], replays[right]["snapshots"][t][k])
                                    for k in SNAPSHOT_FIELDS), "Both linear no-internal-component negative-control state pairs")
                        counts["linear_equivalent_checkpoint_pairs"] += 1
                    for t in range(1, n + 1):
                        a, b = (trials[seed, model, branch, "phase2", t] for branch in (left, right))
                        require(all(a[k] == b[k] for k in TRIAL_FIELDS - {"branch", "internal_enabled"}),
                                "Both linear negative-control action/reward paths agree exactly")
                        counts["linear_equivalent_trial_pairs"] += 1
        print(f"independently audited learning ablation seed {seed} ({seeds.index(seed) + 1}/{len(seeds)})", flush=True)
    equal(read_json(output / "model_resources.json"), resources, "All physical phase-local model resources")
    equal(read_json(output / "stages.json"), stages, "Actual phase reward/choice counts and local/native counter lineage")
    equal(read_json(output / "clone_evidence.json"), clones, "Every cloned prefix evidence entry")
    summary_check(cfg, measurements, summary)
    for field in ("actual_choice_feedback_trials", "head_update_calls", "internal_update_calls", "privileged_update_calls",
                  "prefix_training_trials", "branch_training_trials"):
        equal(metadata[field], counts[field], f"Actual shared-prefix physical budget: {field}")
    require(metadata["human_or_species_comparison"] is False, "No species inference")
    report["counts"] = {**counts, "seeds": len(seeds), "models": len(MODELS), "branches": len(BRANCHES),
                        "curve_rows": len(curves), "artifact_hashes": len(hashes), "source_hashes": len(reg["frozen_files"]),
                        "canonical_fingerprints": len(fingerprints)}
    report.update(study_id=cfg["study_id"], summary_sha256=digest(output / "summary.json"),
                  registration_sha256=digest(registration_path), config_sha256=reg["config_sha256"],
                  learning_curves_sha256=digest(output / "learning_curves.csv"),
                  artifact_hashes_sha256=digest(output / "artifact_hashes.json"))
    report["checks"] = ["Exact source/config/completed-parent bytes, complete artifact coverage and UTC chronology",
        "Independent four-split nuisance crossing, canonical disjointness, topology/encoder RNG and TRAIN-only fixed RMS",
        "Each physical prefix executes once and all four branches inherit exactly its19 native arrays",
        "Every actual choice/reward and head/internal factorial mask update independently replayed with old-head eligibility",
        "Phase-local versus inherited cumulative counters and original initialization/clipping bounds stay distinct",
        "Both exact linear negative-control state/probability/action/reward paths and learned frozen states",
        "All checkpoint states, total/local weight/activity changes and locked IID/OOD probabilities reconstructed",
        "All-seed H_I/H_H, 2x2 interactions/main effects, gains/AUCs, failure ledger and actual resource counts"]


def verify(root, registration, result_dir):
    root, registration, output = map(lambda p: Path(p).resolve(), (root, registration, result_dir))
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "status": "failed", "passed": False,
              "method": "independent numpy/stdlib replay; only explicitly named immutable independent-verifier helpers reused",
              "counts": {}, "checks": [], "errors": [], "not_verified": [
                  "Private remote Git publication authenticity beyond pinned local bytes and timestamps",
                  "Equal resources or policy/feedback histories after computational mask interventions",
                  "Additive independent biological mechanisms, actual synaptic plasticity, or topology superiority",
                  "Visual quantity extraction, human/fly neural equivalence, or mathematics education effects",
                  "Intrinsic learning speed, fresh-training reversal cost, or computational seeds as animals"]}
    try:
        require(root.is_dir() and registration.is_file() and output.is_dir(), "Existing source/run/registration required")
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
    require(not args.out.exists(), "Verification output already exists")
    report = verify(args.root, args.registration, args.result_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "errors": report["errors"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
