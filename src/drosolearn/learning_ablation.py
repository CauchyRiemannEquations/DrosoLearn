"""Matched-prefix, explicitly masked output/internal learning diagnostic.

Only this new module applies selective updates. Existing learners, previous
registrations and previous scientific results remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import subprocess

import numpy as np

from .data import make_dataset, assert_disjoint, analytical_accuracy
from .graph import load_projection, random_ann, rewire
from .reward_linear import LinearLearner, NormalizedLearner
from .stochastic_feedback import digest, read_json, safe, write_new, evaluate, state_pack as original_state_pack


ROOT = Path(__file__).resolve().parents[2]
MODELS = ["linear", "real", "rewired", "ann"]
BRANCHES = ["full", "head_only", "internal_only", "frozen"]
ENVIRONMENT = {"id": "probabilistic", "q_optimal": .8, "q_other": .2}
CRITERIA = {"acquisition_probability": .75, "acquisition_gain": .15, "adaptation_probability": .75,
            "mean_component_effect": .05, "positive_seed_fraction": .8}
RNG = {"implementation": "numpy.random.default_rng", "phase1_action_offset": 610000,
       "phase2_action_offset": 620000, "phase1_reward_offset": 630000, "phase2_reward_offset": 640000,
       "rewired_offset": 650000, "ann_offset": 660000, "encoder_offset": 670000,
       "data_multiplier": 100, "train_phase1_offset": 61, "train_phase2_offset": 64,
       "iid_offset": 62, "ood_offset": 63,
       "coupling": "same_phase_specific_action_and_reward_arrays_per_seed_reused_across_all_models_and_branches"}
TRIAL_FIELDS = ("seed", "model", "branch", "phase", "trial", "global_trial", "family_id", "left_count", "right_count",
                "chosen_count", "action", "reward", "p_right", "q_reward", "action_uniform", "reward_uniform",
                "head_enabled", "internal_enabled", "head_update_called", "internal_update_called", "optimizer_factor",
                "internal_step_l2", "readout_step_l2", "kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2",
                "decision_left_l2", "decision_right_l2", "head_updates", "internal_updates",
                "cumulative_head_updates", "cumulative_internal_updates")
CURVE_FIELDS = ("seed", "model", "branch", "phase", "trial", "split", "p_rule", "p_larger", "p_smaller", "greedy_rule",
                "mean_policy_sensitivity", "expected_reward", "internal_change_l2", "readout_change_l2", "probe_kc_change_l2",
                "probe_mbon_change_l2", "probe_decision_change_l2", "phase_internal_change_l2", "phase_readout_change_l2",
                "phase_probe_kc_change_l2", "phase_probe_mbon_change_l2", "phase_probe_decision_change_l2",
                "lower_bound_fraction", "upper_bound_fraction", "head_updates", "internal_updates",
                "cumulative_head_updates", "cumulative_internal_updates")
CONTRASTS = {
    "full_minus_head_only": {"full": 1., "head_only": -1.},
    "full_minus_internal_only": {"full": 1., "internal_only": -1.},
    "head_only_minus_frozen": {"head_only": 1., "frozen": -1.},
    "internal_only_minus_frozen": {"internal_only": 1., "frozen": -1.},
    "full_minus_frozen": {"full": 1., "frozen": -1.},
    "interaction": {"full": 1., "head_only": -1., "internal_only": -1., "frozen": 1.},
    "head_main_effect": {"full": .5, "head_only": .5, "internal_only": -.5, "frozen": -.5},
    "internal_main_effect": {"full": .5, "head_only": -.5, "internal_only": .5, "frozen": -.5},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(cfg):
    require(cfg["study_id"] == "learning_ablation_v1" and cfg["scope"] in ("preregistered_diagnostic", "integration_test"),
            "Fixed ablation identity required")
    seeds = cfg["seeds"]
    require(type(seeds) is list and seeds and all(type(s) is int and s >= 0 for s in seeds) and len(set(seeds)) == len(seeds),
            "Unique integer seed coordinates required")
    require(cfg["models"] == MODELS and cfg["branches"] == BRANCHES and cfg["environment"] == ENVIRONMENT and cfg["rng"] == RNG,
            "Registered graph families, masks, reward law and streams required")
    require(cfg["pair"] == [2, 4] and cfg["graph"] == "data/flywire_783_mb_left_kc512_s701"
            and cfg["history"] == "shared_high_prefix_then_deepcopied_low_reversal_2x2_masks"
            and cfg["feature_normalization"] == "fixed_initial_phase1_train_feature_gap_rms"
            and cfg["head_initialization"] == "zeros_all_models"
            and cfg["update_interface"] == "inputs_action_actual_reward_and_explicit_head_internal_masks_only_no_labels_or_reward_probabilities",
            "Fixed representation and selected-update interface required")
    for key in ("trials_per_phase", "checkpoint_every", "eval_trials", "probe_rows"):
        require(type(cfg[key]) is int and cfg[key] > 0, "Positive integer resources required")
    n, every, ev = cfg["trials_per_phase"], cfg["checkpoint_every"], cfg["eval_trials"]
    require(n % 4 == ev % 4 == n % every == 0 and cfg["probe_rows"] <= ev, "Complete crossed data and checkpoints required")
    require(all(type(cfg[key]) in (int, float) and np.isfinite(cfg[key]) and cfg[key] >= 0 for key in ("head_lr", "internal_lr"))
            and cfg["criteria"] == CRITERIA and all(type(v) in (int, float) for v in cfg["criteria"].values()),
            "Fixed finite learning rates and prospective criteria required")
    expected = (len(seeds) * 4 * 5 * n, len(seeds) * 4 * 3 * n, len(seeds) * 3 * 3 * n, 0)
    budget_keys = ("actual_choice_feedback_trials", "expected_head_update_calls", "expected_internal_update_calls", "expected_privileged_update_calls")
    require(all(type(cfg[k]) is int for k in budget_keys) and tuple(cfg[k] for k in budget_keys) == expected,
            "Shared-prefix rather than duplicated-prefix resource accounting required")
    if cfg["scope"] == "preregistered_diagnostic":
        require(seeds == list(range(12001, 12011)) and (n, every, ev, cfg["probe_rows"], cfg["head_lr"], cfg["internal_lr"])
                == (400, 100, 128, 8, .1, .01), "Fixed new prospective budget required")
    else:
        require(set(seeds) <= {17, 18} and (n, every, ev, cfg["probe_rows"], cfg["head_lr"], cfg["internal_lr"])
                == (8, 4, 8, 2, .1, .01), "Only seed17/18, eight-trial development fixtures are allowed")
    for key in ("frozen_files", "parent_files"):
        if key in cfg:
            require(type(cfg[key]) is list and cfg[key] and len(set(cfg[key])) == len(cfg[key]), "Explicit unique byte inventories required")


def make_splits(cfg, seed):
    validate_config(cfg)
    require(type(seed) is int and seed in cfg["seeds"], "Registered input seed required")
    base, rng = seed * cfg["rng"]["data_multiplier"], cfg["rng"]
    result = {name: make_dataset(base + rng[name + "_offset"], name,
                 cfg["trials_per_phase"] if name.startswith("train_") else cfg["eval_trials"], tuple(cfg["pair"]),
                 "ood" if name == "ood" else "iid") for name in ("train_phase1", "train_phase2", "iid", "ood")}
    assert_disjoint(list(result.values()))
    require(all(analytical_accuracy(ds) == 1. and all(analytical_accuracy(ds, nuisance_index=i) == .5 for i in range(6))
                for ds in result.values()), "Structural solvability or nuisance control failed")
    return result


def selected_update(model, inputs, action, reward, *, head=True, internal=True):
    """Apply only requested coefficient changes, using the pre-update head.

    This function cannot access a target, q, evaluation dataset or other action's
    outcome. An enabled but empty linear internal component makes no update call.
    """
    require(isinstance(model, (LinearLearner, NormalizedLearner)) and type(head) is bool and type(internal) is bool
            and type(action) is int and action in (0, 1) and type(reward) is int and reward in (0, 1),
            "Known learner, Boolean component masks and actual binary action/reward required")
    kc, phi = model.features(inputs)
    p = float(model.probabilities(inputs))
    factor, old_head = (action - p) * (reward - .5), model.head.copy()
    if head:
        model.head += model.head_lr * factor * (phi[1] - phi[0])
        model.head_updates += 1
    if internal and model.weights.size:
        eligibility = old_head[model.post] * (kc[1, model.pre] - kc[0, model.pre]) / model.feature_scale
        model.weights = np.clip(model.weights + model.internal_lr * factor * eligibility, 0, model.upper)
        model.internal_updates += 1
    return p


def branch_masks(branch):
    require(branch in ("prefix", *BRANCHES), "Registered branch required")
    return branch in ("prefix", "full", "head_only"), branch in ("prefix", "full", "internal_only")


def _mean_sd(values):
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def component_hypothesis(name, comparison, pairs, criteria):
    """Arithmetic verdict; all supplied seed differences remain in its denominator."""
    require(name in ("H_I", "H_H") and pairs and all(type(row["seed"]) is int and type(row["difference"]) in (int, float)
            and np.isfinite(row["difference"]) for row in pairs) and len({row["seed"] for row in pairs}) == len(pairs),
            "Unique finite paired effects required")
    values = [row["difference"] for row in pairs]
    mean, positive = float(np.mean(values)), sum(value > 0 for value in values)
    return {"name": name, "model": "real", "split": "iid", "comparison": comparison, "metric": "phase2_end",
            "n_seeds": len(values), "all_seed_differences": pairs, "mean_effect": mean, "positive_seed_count": positive,
            "mean_criterion_met": bool(mean >= criteria["mean_component_effect"]),
            "positive_fraction_criterion_met": bool(positive / len(values) >= criteria["positive_seed_fraction"]),
            "criterion_met": bool(mean >= criteria["mean_component_effect"] and positive / len(values) >= criteria["positive_seed_fraction"])}


def summarize(cfg, curves):
    validate_config(cfg)
    n, every, criteria = cfg["trials_per_phase"], cfg["checkpoint_every"], cfg["criteria"]
    times = list(range(0, n + 1, every))
    fields = ("seed", "model", "branch", "phase", "trial", "split")
    index = {tuple(row[k] for k in fields): row for row in curves}
    coordinates = {(seed, model, branch, "phase1" if branch == "prefix" else "phase2", t, split)
        for seed in cfg["seeds"] for model in MODELS for branch in ("prefix", *BRANCHES) for t in times for split in ("iid", "ood")}
    require(len(index) == len(curves) and set(index) == coordinates
            and all(type(row["p_rule"]) in (int, float) and np.isfinite(row["p_rule"]) and 0 <= row["p_rule"] <= 1 for row in curves),
            "Complete finite prefix and every-branch checkpoint coverage required")
    outcomes, failures = [], []
    for seed in cfg["seeds"]:
        for model in MODELS:
            for branch in BRANCHES:
                for split in ("iid", "ood"):
                    first = [index[seed, model, "prefix", "phase1", t, split]["p_rule"] for t in times]
                    second = [index[seed, model, branch, "phase2", t, split]["p_rule"] for t in times]
                    acquired = bool(first[-1] >= criteria["acquisition_probability"] and first[-1] - first[0] >= criteria["acquisition_gain"])
                    adapted = bool(second[-1] >= criteria["adaptation_probability"])
                    auc = float(sum((second[i] + second[i + 1]) * .5 * every for i in range(len(times) - 1)) / n)
                    row = {"seed": seed, "model": model, "branch": branch, "split": split, "prefix_start": first[0], "prefix_end": first[-1],
                           "prefix_gain": first[-1] - first[0], "prefix_acquired": acquired, "phase2_start": second[0], "phase2_end": second[-1],
                           "phase2_gain": second[-1] - second[0], "phase2_raw_auc": auc, "phase2_baseline_adjusted_auc": auc - second[0],
                           "phase2_threshold_met": adapted, "qualified_adaptation": bool(acquired and adapted),
                           "first_observed_threshold_checkpoint": next((t for t, value in zip(times, second) if value >= criteria["adaptation_probability"]), None)}
                    outcomes.append(row)
                    for reason, passed in (("prefix_acquisition_threshold_not_met", acquired), ("phase2_endpoint_threshold_not_met", adapted),
                                           ("acquired_old_rule_then_phase2_threshold_not_met", not acquired or adapted)):
                        if not passed:
                            failures.append({"seed": seed, "model": model, "branch": branch, "split": split, "reason": reason})
    keyed = {(r["seed"], r["model"], r["branch"], r["split"]): r for r in outcomes}
    metrics = ("phase2_start", "phase2_end", "phase2_gain", "phase2_raw_auc", "phase2_baseline_adjusted_auc")
    contrasts = []
    for seed in cfg["seeds"]:
        for model in MODELS:
            for split in ("iid", "ood"):
                for comparison, coefficients in CONTRASTS.items():
                    for metric in metrics:
                        contrasts.append({"seed": seed, "model": model, "split": split, "comparison": comparison, "metric": metric,
                            "difference": float(sum(coefficient * keyed[seed, model, branch, split][metric] for branch, coefficient in coefficients.items()))})
    statistics = []
    for model in MODELS:
        for split in ("iid", "ood"):
            for comparison in CONTRASTS:
                for metric in metrics:
                    values = [r["difference"] for r in contrasts if (r["model"], r["split"], r["comparison"], r["metric"]) == (model, split, comparison, metric)]
                    statistics.append({"model": model, "split": split, "comparison": comparison, "metric": metric, "n_seeds": len(values),
                                       "positive_seed_count": sum(value > 0 for value in values), **_mean_sd(values)})
    hypotheses = {}
    for name, comparison in (("H_I", "full_minus_head_only"), ("H_H", "full_minus_internal_only")):
        pairs = [{"seed": seed, "difference": next(r["difference"] for r in contrasts if
                 (r["seed"], r["model"], r["split"], r["comparison"], r["metric"]) == (seed, "real", "iid", comparison, "phase2_end"))} for seed in cfg["seeds"]]
        hypotheses[name] = component_hypothesis(name, comparison, pairs, criteria)
    verdicts = []
    for model in MODELS:
        for branch in BRANCHES:
            for split in ("iid", "ood"):
                rows = [keyed[seed, model, branch, split] for seed in cfg["seeds"]]
                verdicts.append({"model": model, "branch": branch, "split": split, "n_seeds": len(rows),
                    "prefix_acquisition_count": sum(r["prefix_acquired"] for r in rows), "phase2_threshold_count": sum(r["phase2_threshold_met"] for r in rows),
                    "qualified_adaptation_count": sum(r["qualified_adaptation"] for r in rows),
                    "statistics": {metric: _mean_sd([row[metric] for row in rows]) for metric in ("prefix_start", "prefix_end", "prefix_gain", *metrics)}})
    return {"study_id": cfg["study_id"], "status": "completed", "scope": cfg["scope"], "outcomes": outcomes,
            "branch_verdicts": verdicts, "paired_contrasts": contrasts, "contrast_statistics": statistics,
            "primary_hypotheses": hypotheses, "failures": failures,
            "prefix_failure_exclusion": False, "interpretation": "Matched-prefix computational component intervention; artificial supplied quantity, no biological mechanism or topology-superiority claim"}


def freeze(root, cfg):
    require(cfg.get("frozen_files"), "Explicit frozen input/source inventory required")
    return {name: digest(safe(root, name)) for name in sorted(cfg["frozen_files"])}


def parents(root, cfg):
    require(cfg.get("parent_files"), "Explicit parent evidence inventory required")
    return {name: digest(safe(root, name)) for name in cfg["parent_files"]}


def gate(root, cfg):
    base = "results/stochastic_feedback_network_v1/"
    paths = {"summary_sha256": base + "summary.json", "verification_sha256": base + "verification.json",
             "artifact_hashes_sha256": base + "artifact_hashes.json", "registration_sha256": "results/registration_stochastic_feedback_network_v1.json",
             "config_sha256": "configs/stochastic_feedback_network_v1.json"}
    identity = cfg["parent_gate_identity"]
    require(type(identity) is dict and set(identity) == {*paths, "published_commit"}
            and all(type(value) is str and re.fullmatch(r"[a-f0-9]{40}" if key == "published_commit" else r"[a-f0-9]{64}", value)
                    for key, value in identity.items()), "Published parent byte identity required")
    require(all(digest(safe(root, path)) == identity[key] for key, path in paths.items()), "Verified parent byte identity changed")
    summary, checked = read_json(safe(root, paths["summary_sha256"])), read_json(safe(root, paths["verification_sha256"]))
    artifact_map = read_json(safe(root, paths["artifact_hashes_sha256"]))
    require(summary["study_id"] == "stochastic_feedback_network_v1" and summary["stage"] == "network" and summary["status"] == "completed"
            and checked["status"] == "passed" and checked["summary_sha256"] == identity["summary_sha256"]
            and checked["registration_sha256"] == identity["registration_sha256"] and checked["config_sha256"] == identity["config_sha256"]
            and artifact_map["summary.json"] == identity["summary_sha256"],
            "Independent completed network parent must bind the exact summary and registration")
    return {"name": "verified_stochastic_network_before_ablation", "passed": True, **identity}


def register(root, config_path, out):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = read_json(config_path)
    validate_config(cfg)
    evidence = gate(root, cfg)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(re.fullmatch(r"[a-f0-9]{40}", source), "Valid source Git version required")
    record = {"study_id": cfg["study_id"], "registered_at_utc": datetime.now(timezone.utc).isoformat(), "source_git_commit": source,
              "config": cfg, "config_sha256": digest(config_path), "frozen_files": freeze(root, cfg), "parent_files": parents(root, cfg),
              "gate": evidence, "notice": "Prospective private Git freeze, not accredited public or blinded preregistration"}
    write_new(out, record)
    return record


def _artifact_hashes(out):
    return {path.relative_to(out).as_posix(): digest(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"}


def run(root, config_path, regpath, out):
    root, config_path, regpath, out = (Path(path).resolve() for path in (root, config_path, regpath, out))
    cfg, reg = read_json(config_path), read_json(regpath)
    validate_config(cfg)
    now = datetime.now(timezone.utc)
    stamp = datetime.fromisoformat(reg["registered_at_utc"])
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0 and stamp <= now
            and re.fullmatch(r"[a-f0-9]{40}", reg["source_git_commit"]), "Prior UTC registration and source version required")
    require(reg["study_id"] == cfg["study_id"] and reg["config"] == cfg and reg["config_sha256"] == digest(config_path)
            and reg["frozen_files"] == freeze(root, cfg) and reg["parent_files"] == parents(root, cfg) and reg["gate"] == gate(root, cfg),
            "Frozen ablation source, settings or parent gate changed")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "registration.json", reg)
    metadata = {"study_id": cfg["study_id"], "status": "running", "started_at_utc": now.isoformat(), "config": cfg,
        "config_sha256": digest(config_path), "source_git_commit": reg["source_git_commit"], "registration_sha256": digest(regpath),
        "frozen_files": reg["frozen_files"], "parent_files": reg["parent_files"], "gate": reg["gate"],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "actual_choice_feedback_trials": 0,
        "head_update_calls": 0, "internal_update_calls": 0, "privileged_update_calls": 0,
        "prefix_training_trials": 0, "branch_training_trials": 0, "human_or_species_comparison": False}
    write_new(out / "metadata.json", metadata)
    curves, resources, controls, stages, clone_evidence = [], [], [], [], []
    n, every = cfg["trials_per_phase"], cfg["checkpoint_every"]
    try:
        real = load_projection(safe(root, cfg["graph"]))
        for seed in cfg["seeds"]:
            datasets = make_splits(cfg, seed)
            base = out / f"seed_{seed}"
            base.mkdir()
            uniforms = {f"{kind}_{phase}_uniforms": np.random.default_rng(seed + cfg["rng"][f"{phase}_{kind}_offset"]).random(n)
                        for kind in ("action", "reward") for phase in ("phase1", "phase2")}
            np.savez_compressed(base / "stimuli.npz", **{f"{name}_{field}": value for name, ds in datasets.items() for field, value in
                (("counts", ds.counts), ("nuisance", ds.nuisance), ("family_ids", np.asarray(ds.family_ids)))}, **uniforms)
            graphs = {"real": real, "rewired": rewire(real, seed + cfg["rng"]["rewired_offset"]),
                      "ann": random_ann(real, seed + cfg["rng"]["ann_offset"])}
            for kind, graph in graphs.items():
                np.savez_compressed(base / f"{kind}_topology.npz", pre=graph.pre, post=graph.post, counts=graph.weights)
                controls.append({"seed": seed, "kind": kind, "metadata": graph.metadata})
            probe = datasets["iid"].inputs()[:cfg["probe_rows"]]
            with (base / "trials.csv").open("x", encoding="utf8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
                writer.writeheader()
                for model_name in MODELS:
                    ancestor = (LinearLearner(datasets["train_phase1"].inputs(), head_lr=cfg["head_lr"]) if model_name == "linear"
                        else NormalizedLearner(graphs[model_name], seed + cfg["rng"]["encoder_offset"], datasets["train_phase1"].inputs(),
                                               True, cfg["head_lr"], cfg["internal_lr"]))
                    initial = original_state_pack(ancestor, probe, datasets)
                    def train(model, branch, phase, phase_reference):
                        path = base / model_name / branch
                        path.mkdir(parents=True)
                        larger = phase == "phase1"
                        head_mask, internal_mask = branch_masks(branch)
                        internal_active = internal_mask and model_name != "linear"
                        start_head, start_internal = model.head_updates, model.internal_updates
                        actual_chosen, actual_rewards = 0, 0
                        def checkpoint(t):
                            snap = original_state_pack(model, probe, datasets)
                            snap.update(phase_head_updates=np.asarray(model.head_updates - start_head),
                                        phase_internal_updates=np.asarray(model.internal_updates - start_internal))
                            for split in ("iid", "ood"):
                                metrics, _ = evaluate(model, datasets[split], larger, ENVIRONMENT)
                                row = {"seed": seed, "model": model_name, "branch": branch, "phase": phase, "trial": t, "split": split, **metrics,
                                    "lower_bound_fraction": float(np.mean(model.weights <= 0)) if model.weights.size else 0.,
                                    "upper_bound_fraction": float(np.mean(model.weights >= model.upper)) if model.weights.size else 0.,
                                    "head_updates": model.head_updates - start_head, "internal_updates": model.internal_updates - start_internal,
                                    "cumulative_head_updates": model.head_updates, "cumulative_internal_updates": model.internal_updates}
                                for prefix, reference in (("", initial), ("phase_", phase_reference)):
                                    for field, snapshot_key in (("internal_change_l2", "internal_weights"), ("readout_change_l2", "readout_weights"),
                                        ("probe_kc_change_l2", "kc_activity"), ("probe_mbon_change_l2", "mbon_activity"), ("probe_decision_change_l2", "decision_activity")):
                                        row[prefix + field] = float(np.linalg.norm(snap[snapshot_key] - reference[snapshot_key]))
                                curves.append(row)
                            np.savez_compressed(path / f"{phase}_{t:04d}.npz", **snap)
                        checkpoint(0)
                        dataset = datasets[f"train_{phase}"]
                        targets = dataset.target(larger)
                        for i, inputs in enumerate(dataset.inputs()):
                            kc, decision = model.features(inputs)
                            _, mbon = model.raw_activities(inputs)
                            p = float(model.probabilities(inputs))
                            action_u, reward_u = (float(uniforms[f"{kind}_{phase}_uniforms"][i]) for kind in ("action", "reward"))
                            action = int(action_u < p)
                            q = ENVIRONMENT["q_optimal"] if action == int(targets[i]) else ENVIRONMENT["q_other"]
                            reward = int(reward_u < q)
                            old_weights, old_head = model.weights.copy(), model.head.copy()
                            if branch != "frozen":
                                selected_update(model, inputs, action, reward, head=head_mask, internal=internal_mask)
                            metadata["actual_choice_feedback_trials"] += 1
                            metadata["head_update_calls"] += int(head_mask)
                            metadata["internal_update_calls"] += int(internal_active)
                            metadata["prefix_training_trials" if phase == "phase1" else "branch_training_trials"] += 1
                            actual_chosen += int(action == targets[i])
                            actual_rewards += reward
                            writer.writerow({"seed": seed, "model": model_name, "branch": branch, "phase": phase, "trial": i + 1,
                                "global_trial": i + 1 + (n if phase == "phase2" else 0), "family_id": dataset.family_ids[i],
                                "left_count": int(dataset.counts[i, 0]), "right_count": int(dataset.counts[i, 1]), "chosen_count": int(dataset.counts[i, action]),
                                "action": action, "reward": reward, "p_right": p, "q_reward": q, "action_uniform": action_u, "reward_uniform": reward_u,
                                "head_enabled": int(head_mask), "internal_enabled": int(internal_mask), "head_update_called": int(head_mask),
                                "internal_update_called": int(internal_active), "optimizer_factor": (action - p) * (reward - .5),
                                "internal_step_l2": float(np.linalg.norm(model.weights - old_weights)), "readout_step_l2": float(np.linalg.norm(model.head - old_head)),
                                "kc_left_l2": float(np.linalg.norm(kc[0])), "kc_right_l2": float(np.linalg.norm(kc[1])),
                                "mbon_left_l2": float(np.linalg.norm(mbon[0])), "mbon_right_l2": float(np.linalg.norm(mbon[1])),
                                "decision_left_l2": float(np.linalg.norm(decision[0])), "decision_right_l2": float(np.linalg.norm(decision[1])),
                                "head_updates": model.head_updates - start_head, "internal_updates": model.internal_updates - start_internal,
                                "cumulative_head_updates": model.head_updates, "cumulative_internal_updates": model.internal_updates})
                            require(np.isfinite(model.weights).all() and np.isfinite(model.head).all(), "Nonfinite learned parameters")
                            if (i + 1) % every == 0:
                                checkpoint(i + 1)
                        require(model.head_updates - start_head == (n if head_mask else 0)
                                and model.internal_updates - start_internal == (n if internal_active else 0), "Selected component counters differ")
                        stages.append({"seed": seed, "model": model_name, "branch": branch, "phase": phase, "actual_trials": n,
                            "actual_designated_choice_count": actual_chosen, "actual_designated_choice_fraction": actual_chosen / n,
                            "actual_reward_count": actual_rewards, "actual_reward_rate": actual_rewards / n,
                            "head_update_calls": model.head_updates - start_head, "internal_update_calls": model.internal_updates - start_internal,
                            "cumulative_head_updates": model.head_updates, "cumulative_internal_updates": model.internal_updates})
                        resources.append({"seed": seed, "model": model_name, "branch": branch, "phase": phase,
                            "feature_scale": model.feature_scale, "initial_feature_gap_rms": model.initial_feature_gap_rms,
                            "readout_parameters": len(model.head), "internal_parameters": len(model.weights),
                            "trainable_readout_parameters": len(model.head) if head_mask else 0,
                            "trainable_internal_parameters": len(model.weights) if internal_active else 0,
                            "head_update_calls": model.head_updates - start_head, "internal_update_calls": model.internal_updates - start_internal,
                            "privileged_update_calls": 0})
                    train(ancestor, "prefix", "phase1", initial)
                    fork = original_state_pack(ancestor, probe, datasets)
                    for branch in BRANCHES:
                        model = copy.deepcopy(ancestor)
                        clone = original_state_pack(model, probe, datasets)
                        require(fork.keys() == clone.keys() and all(np.array_equal(fork[key], clone[key]) for key in fork),
                                "Deep-copied branch differs from learned prefix")
                        clone_evidence.append({"seed": seed, "model": model_name, "branch": branch, "all_native_checkpoint_fields_equal": True,
                                               "inherited_head_updates": model.head_updates, "inherited_internal_updates": model.internal_updates})
                        train(model, branch, "phase2", fork)
            print(f"completed learning_ablation_v1 seed {seed} ({cfg['seeds'].index(seed) + 1}/{len(cfg['seeds'])})", flush=True)
        with (out / "learning_curves.csv").open("x", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
            writer.writeheader()
            writer.writerows(curves)
        for name, value in (("controls", controls), ("model_resources", resources), ("stages", stages), ("clone_evidence", clone_evidence), ("summary", summarize(cfg, curves))):
            write_new(out / f"{name}.json", value)
        require(tuple(metadata[k] for k in ("actual_choice_feedback_trials", "head_update_calls", "internal_update_calls")) ==
                tuple(cfg[k] for k in ("actual_choice_feedback_trials", "expected_head_update_calls", "expected_internal_update_calls")),
                "Final shared-prefix resource totals differ")
        metadata.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        metadata.update(status="failed", completed_at_utc=datetime.now(timezone.utc).isoformat(), error_type=type(error).__name__, error_message=str(error))
        if curves and not (out / "learning_curves.csv").exists():
            with (out / "partial_learning_curves.csv").open("x", encoding="utf8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
                writer.writeheader()
                writer.writerows(curves)
        (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
        write_new(out / "artifact_hashes.json", _artifact_hashes(out))
        raise
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")
    write_new(out / "artifact_hashes.json", _artifact_hashes(out))
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists() and (args.mode != "run" or args.registration is not None), "New output and prior registration required")
    result = register(args.root, args.config, args.out) if args.mode == "register" else run(args.root, args.config, args.registration, args.out)
    print(json.dumps({"status": result.get("status", "registered"), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
