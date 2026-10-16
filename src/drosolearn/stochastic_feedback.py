"""Prospective sampled-feedback learning; original learners remain immutable.

The environment knows a designated action and its reward probability. Learner
updates receive only the paired inputs, sampled action and sampled binary reward.
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

from .data import make_dataset, assert_disjoint, analytical_accuracy
from .experiment import CONDITIONS
from .graph import load_projection, random_ann, rewire
from .reward_linear import LinearLearner, NormalizedLearner


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENTS = [{"id": "deterministic", "q_optimal": 1., "q_other": 0.},
                {"id": "probabilistic", "q_optimal": .8, "q_other": .2},
                {"id": "uninformative", "q_optimal": .5, "q_other": .5}]
NETWORK_CONDITIONS = ["linear", "fixed", "plastic", "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic", "frozen"]
PLASTIC = {"plastic", "rewired_plastic", "ann_plastic"}
RNG = {"implementation": "numpy.random.default_rng", "phase1_action_offset": 510000,
       "phase2_action_offset": 520000, "phase1_reward_offset": 530000, "phase2_reward_offset": 540000,
       "rewired_offset": 550000, "ann_offset": 560000, "encoder_offset": 570000,
       "data_multiplier": 100, "train_phase1_offset": 51, "train_phase2_offset": 54,
       "iid_offset": 52, "ood_offset": 53,
       "coupling": "same_phase_specific_action_and_reward_arrays_per_seed_reused_across_all_conditions_and_environments"}
CRITERIA = {"acquisition_accuracy": .75, "acquisition_gain": .15, "adaptation_accuracy": .75,
            "seed_pass_fraction": .8, "paired_reward_gap": .15}
TRIAL_FIELDS = ("stage", "seed", "condition", "environment", "phase", "trial", "global_trial", "family_id",
                "left_count", "right_count", "chosen_count", "action", "reward", "p_right", "q_reward",
                "action_uniform", "reward_uniform", "head_update_called", "internal_update_called",
                "optimizer_factor", "internal_step_l2", "readout_step_l2", "kc_left_l2", "kc_right_l2",
                "mbon_left_l2", "mbon_right_l2", "decision_left_l2", "decision_right_l2",
                "head_updates", "internal_updates")
CURVE_FIELDS = ("seed", "condition", "environment", "phase", "trial", "split", "p_rule", "p_larger", "p_smaller",
                "greedy_rule", "mean_policy_sensitivity", "expected_reward", "internal_change_l2", "readout_change_l2",
                "probe_kc_change_l2", "probe_mbon_change_l2", "probe_decision_change_l2", "lower_bound_fraction",
                "upper_bound_fraction", "head_updates", "internal_updates")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    require(path.is_relative_to(root), "Path escapes root")
    return path


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def validate_config(cfg):
    stage = cfg["stage"]
    require(stage in ("linear", "network") and cfg["study_id"] == f"stochastic_feedback_{stage}_v1"
            and cfg["scope"] in ("preregistered_diagnostic", "integration_test"), "Fixed stage and study identity required")
    seeds = cfg["seeds"]
    require(type(seeds) is list and seeds and all(type(seed) is int and seed >= 0 for seed in seeds)
            and len(set(seeds)) == len(seeds), "Unique nonnegative integer seeds required")
    require(cfg["conditions"] == (["linear", "frozen"] if stage == "linear" else NETWORK_CONDITIONS)
            and cfg["environments"] == ENVIRONMENTS and cfg["rng"] == RNG, "Fixed conditions, reward probabilities and streams required")
    require(cfg["pair"] == [2, 4] and cfg["graph"] == "data/flywire_783_mb_left_kc512_s701"
            and cfg["head_initialization"] == "zeros_all_conditions"
            and cfg["feature_normalization"] == "fixed_initial_phase1_train_feature_gap_rms"
            and cfg["history"] == "high400_to_low400_no_reset"
            and cfg["frozen_policy"] == "initial_linear_state_update_calls_skipped"
            and cfg["reward_generation"] == "sample_actual_binary_reward_after_actual_action_using_independent_uniform"
            and cfg["update_interface"] == "inputs_action_actual_reward_only_no_labels_or_reward_probabilities",
            "Fixed representation, scaling, history and update interface required")
    for key in ("trials_per_phase", "checkpoint_every", "eval_trials", "probe_rows"):
        require(type(cfg[key]) is int and cfg[key] > 0, "Positive integer budgets required")
    n, ev, every = cfg["trials_per_phase"], cfg["eval_trials"], cfg["checkpoint_every"]
    require(n % 4 == ev % 4 == 0 and n % every == 0 and cfg["probe_rows"] <= ev,
            "Crossed datasets, complete checkpoints and bounded probes required")
    require(all(type(cfg[key]) in (int, float) and np.isfinite(cfg[key]) and cfg[key] >= 0 for key in ("head_lr", "internal_lr")),
            "Finite nonnegative learning rates required")
    require(cfg["criteria"] == CRITERIA and all(type(value) in (int, float) for value in cfg["criteria"].values()),
            "Prospective thresholds cannot be tuned")
    counts = len(seeds) * len(cfg["conditions"]) * len(ENVIRONMENTS) * 2 * n
    heads = len(seeds) * (len(cfg["conditions"]) - 1) * len(ENVIRONMENTS) * 2 * n
    internals = len(seeds) * len(set(cfg["conditions"]) & PLASTIC) * len(ENVIRONMENTS) * 2 * n
    require(all(type(cfg[key]) is int for key in ("actual_choice_feedback_trials", "expected_head_update_calls",
                "expected_internal_update_calls", "expected_privileged_update_calls"))
            and (cfg["actual_choice_feedback_trials"], cfg["expected_head_update_calls"], cfg["expected_internal_update_calls"],
                 cfg["expected_privileged_update_calls"]) == (counts, heads, internals, 0), "Registered resource arithmetic differs")
    scientific = set(range(10001, 10011)) | set(range(11001, 11011))
    if cfg["scope"] == "preregistered_diagnostic":
        first = 10001 if stage == "linear" else 11001
        require(seeds == list(range(first, first + 10)) and (n, every, ev, cfg["probe_rows"], cfg["head_lr"], cfg["internal_lr"])
                == (400, 100, 128, 8, .1, .01), "Fixed prospective seed and learning budget required")
    else:
        require(not set(seeds) & scientific, "Scientific seeds cannot be development fixtures")
    if "frozen_files" in cfg:
        require(type(cfg["frozen_files"]) is list and cfg["frozen_files"] and len(set(cfg["frozen_files"])) == len(cfg["frozen_files"]),
                "Unique explicit frozen source set required")
    if "parent_files" in cfg:
        require(type(cfg["parent_files"]) is list and cfg["parent_files"] and len(set(cfg["parent_files"])) == len(cfg["parent_files"]),
                "Unique gate inputs required")


def make_splits(cfg, seed):
    validate_config(cfg)
    require(type(seed) is int and seed in cfg["seeds"], "Registered dataset seed required")
    rng, n, ev = cfg["rng"], cfg["trials_per_phase"], cfg["eval_trials"]
    base = seed * rng["data_multiplier"]
    datasets = {name: make_dataset(base + rng[f"{name}_offset"], name,
                n if name.startswith("train_") else ev, tuple(cfg["pair"]), "ood" if name == "ood" else "iid")
                for name in ("train_phase1", "train_phase2", "iid", "ood")}
    assert_disjoint(list(datasets.values()))
    for dataset in datasets.values():
        require(analytical_accuracy(dataset) == 1. and all(analytical_accuracy(dataset, nuisance_index=i) == .5 for i in range(6)),
                "Structural solvability or nuisance crossing gate failed")
    return datasets


def reward_probability(env, action, target):
    require(env in ENVIRONMENTS and type(action) is int and action in (0, 1)
            and type(target) is int and target in (0, 1), "Known environment and actual binary action/target required")
    return env["q_optimal"] if action == target else env["q_other"]


def _state(model):
    return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in vars(model).items()
            if isinstance(value, (np.ndarray, int, float, bool, str))}


def _same_state(model, prior):
    current = _state(model)
    return current.keys() == prior.keys() and all(np.array_equal(current[key], value) if isinstance(value, np.ndarray)
                                                 else type(current[key]) is type(value) and current[key] == value
                                                 for key, value in prior.items())


def evaluate(model, ds, larger, env):
    require(type(larger) is bool and env in ENVIRONMENTS, "Known phase direction and environment required")
    before, rng_before = _state(model), np.random.get_state()
    p = np.asarray(model.probabilities(ds.inputs()), dtype=float)
    require(p.shape == (len(ds.counts),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid evaluation probability")
    targets, high = ds.target(larger), ds.target(True)
    p_large = float(np.where(high == 1, p, 1 - p).mean())
    p_rule = p_large if larger else 1 - p_large
    correct = np.where(np.abs(p - .5) < 1e-12, .5, ((p > .5).astype(int) == targets).astype(float))
    metrics = {"p_rule": p_rule, "p_larger": p_large, "p_smaller": 1 - p_large,
               "greedy_rule": float(correct.mean()), "mean_policy_sensitivity": float((p * (1 - p)).mean()),
               "expected_reward": env["q_other"] + (env["q_optimal"] - env["q_other"]) * p_rule}
    rng_after = np.random.get_state()
    require(_same_state(model, before) and rng_before[0] == rng_after[0] and np.array_equal(rng_before[1], rng_after[1])
            and rng_before[2:] == rng_after[2:], "Evaluation changed model or RNG state")
    return metrics, p


def state_pack(model, probe, datasets):
    pack = {**model.snapshot(probe), "encoder": model.encoder.copy(), "bias": model.bias.copy(),
            "pre": model.pre.copy(), "post": model.post.copy(), "head_updates": np.asarray(model.head_updates),
            "internal_updates": np.asarray(model.internal_updates),
            "initial_internal_weights": getattr(model, "initial_weights", np.empty(0)).copy(),
            "internal_upper_bounds": getattr(model, "upper", np.empty(0)).copy(),
            "incoming_normalizer": getattr(model, "normalizer", np.empty(0)).copy()}
    for split in ("iid", "ood"):
        pack[f"{split}_p_right"] = model.probabilities(datasets[split].inputs())
    return pack


def _mean_sd(values):
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def summarize(cfg, curves):
    validate_config(cfg)
    keys = ("seed", "condition", "environment", "phase", "trial", "split")
    index = {tuple(row[key] for key in keys): row for row in curves}
    require(len(index) == len(curves), "Duplicate checkpoint coordinates")
    n, every, c = cfg["trials_per_phase"], cfg["checkpoint_every"], cfg["criteria"]
    times = list(range(0, n + 1, every))
    expected = {(seed, condition, env["id"], phase, time, split) for seed in cfg["seeds"] for condition in cfg["conditions"]
                for env in ENVIRONMENTS for phase in ("phase1", "phase2") for time in times for split in ("iid", "ood")}
    require(set(index) == expected and all(type(row["p_rule"]) in (int, float) and np.isfinite(row["p_rule"])
            and 0 <= row["p_rule"] <= 1 for row in curves), "Full finite checkpoint coverage required")
    outcomes, failures = [], []
    for seed in cfg["seeds"]:
        for condition in cfg["conditions"]:
            for environment in ENVIRONMENTS:
                env = environment["id"]
                for split in ("iid", "ood"):
                    first, second = ([index[seed, condition, env, phase, t, split]["p_rule"] for t in times]
                                     for phase in ("phase1", "phase2"))
                    acquired = bool(first[-1] >= c["acquisition_accuracy"] and first[-1] - first[0] >= c["acquisition_gain"])
                    adapted = bool(second[-1] >= c["adaptation_accuracy"])
                    auc = float(sum((second[i] + second[i + 1]) * .5 * every for i in range(len(times) - 1)) / n)
                    row = {"seed": seed, "condition": condition, "environment": env, "split": split,
                           "phase1_start": first[0], "phase1_end": first[-1], "phase1_gain": first[-1] - first[0],
                           "phase1_threshold_met": acquired, "phase1_acquired": acquired if env != "uninformative" else None,
                           "phase2_start": second[0], "phase2_end": second[-1], "phase2_gain": second[-1] - second[0],
                           "phase2_raw_auc": auc, "phase2_baseline_adjusted_auc": auc - second[0],
                           "phase2_threshold_met": adapted, "qualified_adaptation": bool(acquired and adapted) if env != "uninformative" else None,
                           "first_observed_threshold_checkpoint": next((t for t, value in zip(times, second) if value >= c["adaptation_accuracy"]), None)}
                    outcomes.append(row)
                    if env != "uninformative":
                        for reason, passed in (("phase1_acquisition_threshold_not_met", acquired),
                                               ("phase2_endpoint_threshold_not_met", adapted),
                                               ("acquired_old_rule_then_phase2_threshold_not_met", not acquired or adapted)):
                            if not passed:
                                failures.append({"seed": seed, "condition": condition, "environment": env, "split": split, "reason": reason})
    keyed = {(r["seed"], r["condition"], r["environment"], r["split"]): r for r in outcomes}
    verdicts = []
    for condition in cfg["conditions"]:
        for environment in ENVIRONMENTS:
            env = environment["id"]
            for split in ("iid", "ood"):
                rows = [keyed[seed, condition, env, split] for seed in cfg["seeds"]]
                verdicts.append({"condition": condition, "environment": env, "split": split, "n_seeds": len(rows),
                    "phase1_threshold_count": sum(r["phase1_threshold_met"] for r in rows),
                    "phase1_acquisition_count": sum(r["phase1_acquired"] for r in rows) if env != "uninformative" else None,
                    "phase2_threshold_count": sum(r["phase2_threshold_met"] for r in rows),
                    "qualified_adaptation_count": sum(r["qualified_adaptation"] for r in rows) if env != "uninformative" else None,
                    "statistics": {metric: _mean_sd([r[metric] for r in rows]) for metric in
                                   ("phase1_start", "phase1_end", "phase1_gain", "phase2_start", "phase2_end", "phase2_gain",
                                    "phase2_raw_auc", "phase2_baseline_adjusted_auc")}})
    contrasts, network_contrasts = [], []
    metrics = ("start", "end", "gain")
    for seed in cfg["seeds"]:
        for split in ("iid", "ood"):
            for condition in cfg["conditions"]:
                for comparison, left, right in (("probabilistic_minus_uninformative", "probabilistic", "uninformative"),
                                                ("deterministic_minus_probabilistic", "deterministic", "probabilistic")):
                    for metric in (*[f"{phase}_{metric}" for phase in ("phase1", "phase2") for metric in metrics],
                                   "phase2_raw_auc", "phase2_baseline_adjusted_auc"):
                        contrasts.append({"seed": seed, "condition": condition, "split": split, "comparison": comparison,
                                          "metric": metric, "difference": keyed[seed, condition, left, split][metric] - keyed[seed, condition, right, split][metric]})
            if cfg["stage"] == "network":
                for env in (r["id"] for r in ENVIRONMENTS):
                    for left, right in (("plastic", "fixed"), ("rewired_plastic", "rewired_fixed"), ("ann_plastic", "ann_fixed"),
                                        ("rewired_fixed", "fixed"), ("rewired_plastic", "plastic"), ("ann_fixed", "fixed"), ("ann_plastic", "plastic")):
                        for metric in (*[f"{phase}_{metric}" for phase in ("phase1", "phase2") for metric in metrics],
                                       "phase2_raw_auc", "phase2_baseline_adjusted_auc"):
                            network_contrasts.append({"seed": seed, "environment": env, "split": split, "comparison": f"{left}_minus_{right}",
                                "metric": metric, "difference": keyed[seed, left, env, split][metric] - keyed[seed, right, env, split][metric]})
    acquisition_count = sum(keyed[seed, "linear", "probabilistic", "iid"]["phase1_acquired"] for seed in cfg["seeds"])
    gaps = [{"seed": seed, "difference": keyed[seed, "linear", "probabilistic", "iid"]["phase1_end"]
             - keyed[seed, "linear", "uninformative", "iid"]["phase1_end"]} for seed in cfg["seeds"]]
    gap_mean = float(np.mean([row["difference"] for row in gaps]))
    fraction_pass, gap_pass = acquisition_count / len(cfg["seeds"]) >= c["seed_pass_fraction"], gap_mean >= c["paired_reward_gap"]
    hypothesis = {"name": "H_L", "probabilistic_acquisition_count": acquisition_count, "n_seeds": len(cfg["seeds"]),
                  "acquisition_fraction": acquisition_count / len(cfg["seeds"]), "all_seed_endpoint_gaps": gaps,
                  "mean_paired_reward_gap": gap_mean, "fraction_criterion_met": bool(fraction_pass), "gap_criterion_met": bool(gap_pass),
                  "criterion_met": bool(fraction_pass and gap_pass),
                  "scope": "gate_decision" if cfg["stage"] == "linear" else "descriptive_independent_replication_not_retroactive_gate"}
    def contrast_statistics(rows, group_fields):
        grouped = {}
        for row in rows:
            grouped.setdefault(tuple(row[key] for key in group_fields), []).append(row["difference"])
        return [{**dict(zip(group_fields, identity)), "n_seeds": len(values), **_mean_sd(values)}
                for identity, values in grouped.items()]
    return {"study_id": cfg["study_id"], "stage": cfg["stage"], "status": "completed", "scope": cfg["scope"], "outcomes": outcomes,
            "family_verdicts": verdicts, "environment_contrasts": contrasts, "network_contrasts": network_contrasts,
            "environment_contrast_statistics": contrast_statistics(contrasts, ("condition", "split", "comparison", "metric")),
            "network_contrast_statistics": contrast_statistics(network_contrasts, ("environment", "split", "comparison", "metric")),
            "primary_hypothesis": hypothesis, "failures": failures,
            "interpretation": "IID and artificial nuisance OOD only; uninformative thresholds are designated-direction chance events; no fresh-control speed or topology-superiority claim"}


def freeze(root, cfg):
    require("frozen_files" in cfg and cfg["frozen_files"], "Explicit source freeze required")
    return {name: digest(safe(root, name)) for name in sorted(cfg["frozen_files"])}


def parents(root, cfg):
    require("parent_files" in cfg and cfg["parent_files"], "Explicit parent evidence freeze required")
    return {name: digest(safe(root, name)) for name in cfg["parent_files"]}


def gate(root, cfg):
    if cfg["stage"] == "linear":
        base = safe(root, "results/metric_calibration_v1/summary.json")
        verification = safe(root, "results/metric_calibration_v1/verification.json")
        summary, checked = read_json(base), read_json(verification)
        require(checked["status"] == "passed" and all(summary["analytic_claims"][key]["criterion_met"] is True for key in
                ("H1_no_learning_feedback_difference", "H2_no_learning_high_persistence", "H3_pooling_masks_phase_difference")),
                "Independent calibration and all fixed counterexamples must pass")
        hashes = read_json(safe(root, "results/metric_calibration_v1/artifact_hashes.json"))
        identity = cfg["calibration_gate_identity"]
        require(type(identity) is dict and set(identity) == {"summary_sha256", "verification_sha256", "registration_sha256",
                "artifact_hashes_sha256", "published_commit"}
                and all(type(value) is str and re.fullmatch(r"[a-f0-9]{40}" if key == "published_commit" else r"[a-f0-9]{64}", value)
                        for key, value in identity.items()), "Published calibration identity must be explicit")
        require(identity["summary_sha256"] == digest(base) and identity["verification_sha256"] == digest(verification)
                and identity["registration_sha256"] == digest(safe(root, "results/registration_metric_calibration_v1.json"))
                and identity["artifact_hashes_sha256"] == digest(safe(root, "results/metric_calibration_v1/artifact_hashes.json")),
                "Published calibration gate bytes changed")
        require(hashes["summary.json"] == digest(base)
                and checked["registration_sha256"] == digest(safe(root, "results/registration_metric_calibration_v1.json"))
                and checked["config_sha256"] == digest(safe(root, "configs/metric_calibration_v1.json")), "Calibration gate byte identity differs")
        return {"name": "calibration_before_linear", "passed": True, **identity}
    base, verification = (safe(root, f"results/stochastic_feedback_linear_v1/{name}.json") for name in ("summary", "verification"))
    summary, checked = read_json(base), read_json(verification)
    require(summary["study_id"] == "stochastic_feedback_linear_v1" and summary["primary_hypothesis"]["criterion_met"] is True
            and checked["status"] == "passed" and checked["summary_sha256"] == digest(base)
            and checked["registration_sha256"] == digest(safe(root, "results/registration_stochastic_feedback_linear_v1.json")),
            "Linear H-L and independently verified exact summary must pass before network")
    return {"name": "linear_H_L_before_network", "passed": True, "summary_sha256": digest(base), "verification_sha256": digest(verification)}


def register(root, config_path, out):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = read_json(config_path)
    validate_config(cfg)
    evidence = gate(root, cfg)
    record = {"study_id": cfg["study_id"], "stage": cfg["stage"], "registered_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "config": cfg, "config_sha256": digest(config_path), "frozen_files": freeze(root, cfg), "parent_files": parents(root, cfg),
              "gate": evidence, "notice": "Prospective private Git freeze, not accredited public or blinded preregistration"}
    require(re.fullmatch(r"[a-f0-9]{40}", record["source_git_commit"]), "Valid source Git identity required")
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
            and re.fullmatch(r"[a-f0-9]{40}", reg["source_git_commit"]), "Valid prior UTC registration and source identity required")
    require(reg["study_id"] == cfg["study_id"] and reg["stage"] == cfg["stage"] and reg["config"] == cfg
            and reg["config_sha256"] == digest(config_path) and reg["frozen_files"] == freeze(root, cfg)
            and reg["parent_files"] == parents(root, cfg) and reg["gate"] == gate(root, cfg), "Source, config or gate changed after registration")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "registration.json", reg)
    metadata = {"study_id": cfg["study_id"], "stage": cfg["stage"], "status": "running", "started_at_utc": now.isoformat(),
                "config": cfg, "config_sha256": digest(config_path), "source_git_commit": reg["source_git_commit"],
                "registration_sha256": digest(regpath), "frozen_files": reg["frozen_files"], "parent_files": reg["parent_files"],
                "gate": reg["gate"], "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                "actual_choice_feedback_trials": 0, "head_update_calls": 0, "internal_update_calls": 0, "privileged_update_calls": 0,
                "human_or_species_comparison": False}
    write_new(out / "metadata.json", metadata)
    curves, resources, controls, stages = [], [], [], []
    n, every = cfg["trials_per_phase"], cfg["checkpoint_every"]
    try:
        real = load_projection(safe(root, cfg["graph"])) if cfg["stage"] == "network" else None
        for seed in cfg["seeds"]:
            datasets = make_splits(cfg, seed)
            base = out / f"seed_{seed}"
            base.mkdir()
            uniforms = {f"{kind}_{phase}_uniforms": np.random.default_rng(seed + cfg["rng"][f"{phase}_{kind}_offset"]).random(n)
                        for kind in ("action", "reward") for phase in ("phase1", "phase2")}
            np.savez_compressed(base / "stimuli.npz", **{f"{name}_{field}": value for name, ds in datasets.items()
                for field, value in (("counts", ds.counts), ("nuisance", ds.nuisance), ("family_ids", np.asarray(ds.family_ids)))}, **uniforms)
            graphs = {} if real is None else {"real": real, "rewired": rewire(real, seed + cfg["rng"]["rewired_offset"]),
                                                        "ann": random_ann(real, seed + cfg["rng"]["ann_offset"])}
            for kind, graph in graphs.items():
                np.savez_compressed(base / f"{kind}_topology.npz", pre=graph.pre, post=graph.post, counts=graph.weights)
                controls.append({"seed": seed, "kind": kind, "metadata": graph.metadata})
            probe = datasets["iid"].inputs()[:cfg["probe_rows"]]
            with (base / "trials.csv").open("x", encoding="utf8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
                writer.writeheader()
                for condition in cfg["conditions"]:
                    for env in ENVIRONMENTS:
                        if condition in ("linear", "frozen"):
                            model = LinearLearner(datasets["train_phase1"].inputs(), head_lr=cfg["head_lr"])
                        else:
                            kind, plastic, _, _ = CONDITIONS[condition]
                            model = NormalizedLearner(graphs[kind], seed + cfg["rng"]["encoder_offset"], datasets["train_phase1"].inputs(),
                                                      plastic, cfg["head_lr"], cfg["internal_lr"])
                        path = base / condition / env["id"]
                        path.mkdir(parents=True)
                        initial = model.snapshot(probe)
                        resource = {"seed": seed, "condition": condition, "environment": env["id"], "feature_scale": model.feature_scale,
                                    "initial_feature_gap_rms": model.initial_feature_gap_rms, "readout_parameters": len(model.head),
                                    "trainable_readout_parameters": 0 if condition == "frozen" else len(model.head),
                                    "internal_parameters": len(model.weights), "trainable_internal_parameters": len(model.weights) if condition in PLASTIC else 0,
                                    "head_update_calls": 0, "internal_update_calls": 0, "privileged_update_calls": 0}
                        def checkpoint(phase, t, larger):
                            snap = state_pack(model, probe, datasets)
                            for split in ("iid", "ood"):
                                metrics, _ = evaluate(model, datasets[split], larger, env)
                                weights = model.weights
                                curves.append({"seed": seed, "condition": condition, "environment": env["id"], "phase": phase,
                                    "trial": t, "split": split, **metrics,
                                    "internal_change_l2": float(np.linalg.norm(weights - initial["internal_weights"])),
                                    "readout_change_l2": float(np.linalg.norm(model.head - initial["readout_weights"])),
                                    "probe_kc_change_l2": float(np.linalg.norm(snap["kc_activity"] - initial["kc_activity"])),
                                    "probe_mbon_change_l2": float(np.linalg.norm(snap["mbon_activity"] - initial["mbon_activity"])),
                                    "probe_decision_change_l2": float(np.linalg.norm(snap["decision_activity"] - initial["decision_activity"])),
                                    "lower_bound_fraction": float(np.mean(weights <= 0)) if weights.size else 0.,
                                    "upper_bound_fraction": float(np.mean(weights >= model.upper)) if weights.size else 0.,
                                    "head_updates": model.head_updates, "internal_updates": model.internal_updates})
                            np.savez_compressed(path / f"{phase}_{t:04d}.npz", **snap)
                        for phase, larger in (("phase1", True), ("phase2", False)):
                            checkpoint(phase, 0, larger)
                            dataset = datasets[f"train_{phase}"]
                            actions_for_rule, received = 0, 0
                            before_head, before_internal = model.head_updates, model.internal_updates
                            for i, inputs in enumerate(dataset.inputs()):
                                kc, decision = model.features(inputs)
                                _, mbon = model.raw_activities(inputs)
                                p = float(model.probabilities(inputs))
                                action_u, reward_u = (float(uniforms[f"{kind}_{phase}_uniforms"][i]) for kind in ("action", "reward"))
                                action = int(action_u < p)
                                target = int(dataset.target(larger)[i])
                                q = reward_probability(env, action, target)
                                reward = int(reward_u < q)
                                old_weights, old_head = model.weights.copy(), model.head.copy()
                                if condition != "frozen":
                                    model.update(inputs, action, reward)
                                metadata["actual_choice_feedback_trials"] += 1
                                metadata["head_update_calls"] += int(condition != "frozen")
                                metadata["internal_update_calls"] += int(condition in PLASTIC)
                                actions_for_rule += int(action == target)
                                received += reward
                                writer.writerow({"stage": cfg["stage"], "seed": seed, "condition": condition, "environment": env["id"], "phase": phase,
                                    "trial": i + 1, "global_trial": i + 1 + (n if phase == "phase2" else 0), "family_id": dataset.family_ids[i],
                                    "left_count": int(dataset.counts[i, 0]), "right_count": int(dataset.counts[i, 1]), "chosen_count": int(dataset.counts[i, action]),
                                    "action": action, "reward": reward, "p_right": p, "q_reward": q, "action_uniform": action_u, "reward_uniform": reward_u,
                                    "head_update_called": int(condition != "frozen"), "internal_update_called": int(condition in PLASTIC),
                                    "optimizer_factor": (action - p) * (reward - .5), "internal_step_l2": float(np.linalg.norm(model.weights - old_weights)),
                                    "readout_step_l2": float(np.linalg.norm(model.head - old_head)), "kc_left_l2": float(np.linalg.norm(kc[0])),
                                    "kc_right_l2": float(np.linalg.norm(kc[1])), "mbon_left_l2": float(np.linalg.norm(mbon[0])), "mbon_right_l2": float(np.linalg.norm(mbon[1])),
                                    "decision_left_l2": float(np.linalg.norm(decision[0])), "decision_right_l2": float(np.linalg.norm(decision[1])),
                                    "head_updates": model.head_updates, "internal_updates": model.internal_updates})
                                require(np.isfinite(model.weights).all() and np.isfinite(model.head).all(), "Nonfinite learned weights")
                                if (i + 1) % every == 0:
                                    checkpoint(phase, i + 1, larger)
                            stages.append({"seed": seed, "condition": condition, "environment": env["id"], "phase": phase,
                                "actual_trials": n, "actual_designated_choice_count": actions_for_rule, "actual_designated_choice_fraction": actions_for_rule / n,
                                "actual_reward_count": received, "actual_reward_rate": received / n, "head_update_calls": model.head_updates - before_head,
                                "internal_update_calls": model.internal_updates - before_internal, "cumulative_head_updates": model.head_updates,
                                "cumulative_internal_updates": model.internal_updates})
                        resource.update(head_update_calls=model.head_updates, internal_update_calls=model.internal_updates)
                        require(model.head_updates == (0 if condition == "frozen" else 2 * n)
                                and model.internal_updates == (2 * n if condition in PLASTIC else 0), "Learner update counters disagree")
                        resources.append(resource)
            print(f"completed {cfg['study_id']} seed {seed} ({cfg['seeds'].index(seed) + 1}/{len(cfg['seeds'])})", flush=True)
        with (out / "learning_curves.csv").open("x", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
            writer.writeheader()
            writer.writerows(curves)
        write_new(out / "controls.json", controls)
        write_new(out / "model_resources.json", resources)
        write_new(out / "stages.json", stages)
        write_new(out / "summary.json", summarize(cfg, curves))
        require((metadata["actual_choice_feedback_trials"], metadata["head_update_calls"], metadata["internal_update_calls"])
                == (cfg["actual_choice_feedback_trials"], cfg["expected_head_update_calls"], cfg["expected_internal_update_calls"]),
                "Completed total resources differ from registration")
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
    require(not args.out.exists(), "Output already exists")
    require(args.mode != "run" or args.registration is not None, "Run requires registration")
    result = register(args.root, args.config, args.out) if args.mode == "register" else run(args.root, args.config, args.registration, args.out)
    print(json.dumps({"status": result.get("status", "registered"), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
