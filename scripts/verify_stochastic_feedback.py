"""Independent replay of actual sampled-reward learning, never producer code.

Only previous independent audit arithmetic helpers are imported. This module
does not import a learner, producer forward/update/reward function, or summary.
Primary artifacts are read-only; the CLI writes one new exclusive report.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from drosolearn.encoding_audit import (
    _require, _near, _integer, _number, _npz, _regenerate_dataset,
    _regenerate_topology, _dataset, _inputs,
)
from drosolearn.linearity_audit import _source_projection
from drosolearn.bandit_reversal_audit import (
    _raw_features, _state, _snapshot_check, _auc,
)


ROOT = Path(__file__).resolve().parents[1]
FLAGS = {"linear": ("linear", False), "fixed": ("real", False),
         "plastic": ("real", True), "rewired_fixed": ("rewired", False),
         "rewired_plastic": ("rewired", True), "ann_fixed": ("ann", False),
         "ann_plastic": ("ann", True), "frozen": ("linear", False)}
ENVIRONMENTS = [{"id": name, "q_optimal": a, "q_other": b} for name, a, b in
                (("deterministic", 1., 0.), ("probabilistic", .8, .2),
                 ("uninformative", .5, .5))]
RNG = {"implementation": "numpy.random.default_rng", "phase1_action_offset": 510000,
       "phase2_action_offset": 520000, "phase1_reward_offset": 530000,
       "phase2_reward_offset": 540000, "rewired_offset": 550000, "ann_offset": 560000,
       "encoder_offset": 570000, "data_multiplier": 100, "train_phase1_offset": 51,
       "train_phase2_offset": 54, "iid_offset": 52, "ood_offset": 53,
       "coupling": "same_phase_specific_action_and_reward_arrays_per_seed_reused_across_all_conditions_and_environments"}
SAVED_FIELDS = {"internal_weights", "readout_weights", "kc_activity", "mbon_activity",
                "choice_probability", "encoder", "bias", "pre", "post", "decision_activity",
                "feature_scale", "initial_feature_gap_rms", "head_updates", "internal_updates"}
CHECKPOINT_FIELDS = SAVED_FIELDS | {"iid_p_right", "ood_p_right", "initial_internal_weights",
                                  "internal_upper_bounds", "incoming_normalizer"}
TRIAL_FIELDS = {"stage", "seed", "condition", "environment", "phase", "trial", "global_trial",
                "family_id", "left_count", "right_count", "chosen_count", "action", "reward",
                "p_right", "q_reward", "action_uniform", "reward_uniform", "optimizer_factor",
                "internal_step_l2", "readout_step_l2", "kc_left_l2", "kc_right_l2", "mbon_left_l2",
                "mbon_right_l2", "decision_left_l2", "decision_right_l2", "head_update_called",
                "internal_update_called", "head_updates", "internal_updates"}
CURVE_FIELDS = {"seed", "condition", "environment", "phase", "trial", "split", "p_rule", "p_larger",
                "p_smaller", "greedy_rule", "mean_policy_sensitivity", "expected_reward",
                "internal_change_l2", "readout_change_l2", "probe_kc_change_l2", "probe_mbon_change_l2",
                "probe_decision_change_l2", "lower_bound_fraction", "upper_bound_fraction",
                "head_updates", "internal_updates"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def safe(root, relative):
    root = Path(root).resolve()
    _require(type(relative) is str and relative and "\\" not in relative
             and not Path(relative).is_absolute() and ".." not in Path(relative).parts,
             "Safe relative POSIX artifact path required")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root) and path.is_file(), "Missing or escaping artifact path")
    return path


def utc(value):
    _require(type(value) is str, "UTC timestamp must be text")
    result = datetime.fromisoformat(value)
    _require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0,
             "Timezone-aware UTC timestamp required")
    return result


def equal(actual, expected, label):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"Schema: {label}")
        for key, value in expected.items():
            equal(actual[key], value, f"{label}/{key}")
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), f"List: {label}")
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            equal(a, b, f"{label}/{i}")
    elif expected is None or type(expected) in (str, bool, int):
        _require(type(actual) is type(expected) and actual == expected, f"Exact value: {label}")
    else:
        _require(type(actual) in (int, float) and type(actual) is not bool, f"Numeric value: {label}")
        _near(actual, expected, label)


def csv_index(path, fields, keys):
    result = {}
    with Path(path).open(encoding="utf8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(len(reader.fieldnames or ()) == len(fields)
                 and set(reader.fieldnames or ()) == set(fields), f"Exact CSV schema: {path}")
        for row in reader:
            _require(None not in row and all(v is not None for v in row.values()), "Malformed CSV row")
            key = tuple(_integer(row, k) if k in {"seed", "trial", "global_trial"} else row[k] for k in keys)
            _require(key not in result, f"Duplicate CSV key: {key}")
            result[key] = row
    return result


def config_check(cfg):
    stage = cfg["stage"]
    _require(stage in {"linear", "network"} and cfg["scope"] in {"preregistered_diagnostic", "integration_test"},
             "Registered stage and scope")
    _require(cfg["study_id"] == f"stochastic_feedback_{stage}_v1", "Study identity")
    _require(cfg["conditions"] == (["linear", "frozen"] if stage == "linear" else list(FLAGS)),
             "Exact prospectively specified conditions")
    equal(cfg["environments"], ENVIRONMENTS, "Three registered reward environments")
    equal(cfg["rng"], RNG, "Registered independent and shared streams")
    _require(cfg["pair"] == [2, 4] and cfg["graph"] == "data/flywire_783_mb_left_kc512_s701",
             "Fixed artificial pair and source graph")
    _require(cfg["head_initialization"] == "zeros_all_conditions"
             and cfg["feature_normalization"] == "fixed_initial_phase1_train_feature_gap_rms"
             and cfg["history"] == "high400_to_low400_no_reset"
             and cfg["frozen_policy"] == "initial_linear_state_update_calls_skipped",
             "Fixed initialization, TRAIN normalization and no-reset history")
    _require(cfg["reward_generation"] == "sample_actual_binary_reward_after_actual_action_using_independent_uniform"
             and cfg["update_interface"] == "inputs_action_actual_reward_only_no_labels_or_reward_probabilities",
             "Actual sampled feedback and reward-only learner interface")
    seeds = cfg["seeds"]
    _require(isinstance(seeds, list) and seeds and len(seeds) == len(set(seeds))
             and all(type(s) is int and s >= 0 for s in seeds), "Distinct integer seeds")
    budget = tuple(cfg[k] for k in ("trials_per_phase", "checkpoint_every", "eval_trials", "probe_rows"))
    n, every, ev, probes = budget
    _require(all(type(x) is int and x > 0 for x in budget)
             and n % 4 == ev % 4 == n % every == 0 and probes <= ev, "Crossed and checkpoint budgets")
    _require(all(type(cfg[k]) in (int, float) and np.isfinite(cfg[k]) and cfg[k] >= 0
                 for k in ("head_lr", "internal_lr")), "Finite nonnegative learning rates")
    c = cfg["criteria"]
    _require(set(c) == {"acquisition_accuracy", "acquisition_gain", "adaptation_accuracy",
                       "seed_pass_fraction", "paired_reward_gap"}
             and all(type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1 for v in c.values())
             and c["seed_pass_fraction"] > 0, "Criterion schema/ranges")
    if cfg["scope"] == "preregistered_diagnostic":
        _require(seeds == list(range(10001, 10011) if stage == "linear" else range(11001, 11011)),
                 "All prospective scientific seeds")
        _require((*budget, cfg["head_lr"], cfg["internal_lr"]) == (400, 100, 128, 8, .1, .01),
                 "Fixed scientific resources")
        equal(c, {"acquisition_accuracy": .75, "acquisition_gain": .15, "adaptation_accuracy": .75,
                  "seed_pass_fraction": .8, "paired_reward_gap": .15}, "Scientific criteria")
    else:
        _require(not any(s in set(range(10001, 10011)) | set(range(11001, 11011)) for s in seeds),
                 "Scientific seeds forbidden in development fixtures")
    presentations = len(seeds) * len(cfg["conditions"]) * 3 * 2 * n
    heads = len(seeds) * (len(cfg["conditions"]) - 1) * 3 * 2 * n
    internals = len(seeds) * sum(FLAGS[c][1] for c in cfg["conditions"]) * 3 * 2 * n
    for field, value in (("actual_choice_feedback_trials", presentations), ("expected_head_update_calls", heads),
                         ("expected_internal_update_calls", internals), ("expected_privileged_update_calls", 0)):
        equal(cfg[field], value, f"Registered exact resource arithmetic: {field}")


def metrics(probability, counts, larger, env):
    high = (counts[:, 1] > counts[:, 0]).astype(int)
    target = high if larger else 1 - high
    p_large = float(np.mean(np.where(high == 1, probability, 1 - probability)))
    p_rule = p_large if larger else 1 - p_large
    greedy = np.where(np.abs(probability - .5) < 1e-12, .5,
                      ((probability > .5).astype(int) == target).astype(float))
    return {"p_rule": p_rule, "p_larger": p_large, "p_smaller": 1 - p_large,
            "greedy_rule": float(np.mean(greedy)),
            "mean_policy_sensitivity": float(np.mean(probability * (1 - probability))),
            "expected_reward": float(env["q_other"] + (env["q_optimal"] - env["q_other"]) * p_rule)}


def initialize(condition, graph, encoder, bias, train, n_mbon):
    kind, plastic = FLAGS[condition]
    linear = kind == "linear"
    if linear:
        encoder, bias = np.empty((0, 7)), np.empty(0)
        pre, post, weights, head = np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0), np.zeros(7)
    else:
        pre, post = graph["pre"], graph["post"]
        normalizer = np.maximum(np.bincount(post, weights=graph["counts"], minlength=n_mbon), 1.)
        weights, head = graph["counts"] / normalizer[post], np.zeros(n_mbon)
    _, _, raw = _raw_features(train, encoder, bias, pre, post, weights, head.size, linear)
    scale = float(np.sqrt(np.mean(np.sum((raw[:, 1] - raw[:, 0]) ** 2, axis=-1))))
    _require(np.isfinite(scale) and scale > 0, "Positive initial phase1 TRAIN-only RMS")
    return encoder, bias, pre, post, weights, head, scale, linear, plastic


def independent_step(stimulus, action_uniform, reward_uniform, larger, env,
                     encoder, bias, pre, post, weights, head, scale, linear, plastic,
                     frozen, head_lr, internal_lr, upper):
    kc, mb, decision, probability = _state(stimulus, encoder, bias, pre, post, weights, head, scale, linear)
    probability = float(probability)
    action = int(action_uniform < probability)
    target = int(stimulus[1, 0] > stimulus[0, 0]) if larger else int(stimulus[1, 0] < stimulus[0, 0])
    q = env["q_optimal"] if action == target else env["q_other"]
    reward = int(reward_uniform < q)
    factor = (action - probability) * (reward - .5)
    new_head, new_weights = head.copy(), weights.copy()
    if not frozen:
        new_head += head_lr * factor * (decision[1] - decision[0])
        if plastic:
            eligibility = head[post] * (kc[1, pre] - kc[0, pre]) / scale
            new_weights = np.minimum(np.maximum(weights + internal_lr * factor * eligibility, 0.), upper)
    return {"action": action, "reward": reward, "p_right": probability, "q_reward": q,
            "optimizer_factor": factor, "internal_step_l2": float(np.linalg.norm(new_weights - weights)),
            "readout_step_l2": float(np.linalg.norm(new_head - head)),
            "kc_left_l2": float(np.linalg.norm(kc[0])), "kc_right_l2": float(np.linalg.norm(kc[1])),
            "mbon_left_l2": float(np.linalg.norm(mb[0])), "mbon_right_l2": float(np.linalg.norm(mb[1])),
            "decision_left_l2": float(np.linalg.norm(decision[0])), "decision_right_l2": float(np.linalg.norm(decision[1]))}, new_weights, new_head


def regenerate_graphs(root, cfg, base, seed, controls):
    source, n_kc, n_mbon, source_metadata = _source_projection(root, cfg["graph"])
    graphs = {}
    for kind in ("real", "rewired", "ann"):
        graph = _npz(base / f"{kind}_topology.npz", {"pre", "post", "counts"})
        pre, post, counts = (graph[k] for k in ("pre", "post", "counts"))
        _require(pre.ndim == post.ndim == counts.ndim == 1 and pre.shape == post.shape == counts.shape
                 and pre.shape == source["pre"].shape, "Topology shapes/edge counts")
        _require(np.issubdtype(pre.dtype, np.integer) and np.issubdtype(post.dtype, np.integer)
                 and np.all((pre >= 0) & (pre < n_kc)) and np.all((post >= 0) & (post < n_mbon)),
                 "Topology integer index bounds")
        _require(np.array_equal(post, source["post"]) and np.array_equal(counts, source["counts"])
                 and len(set(zip(pre.tolist(), post.tolist()))) == len(pre), "Source counts/posts and no duplicate pairs")
        if kind == "real":
            _require(np.array_equal(pre, source["pre"]), "Real graph equals independently parsed anatomical source")
            equal(controls[kind]["metadata"], source_metadata, "Source topology metadata")
        else:
            offset = cfg["rng"]["rewired_offset" if kind == "rewired" else "ann_offset"]
            legacy_offset = 50000 if kind == "rewired" else 60000
            expected, accepted, attempts = _regenerate_topology(source, seed + offset - legacy_offset,
                                                               kind, n_kc, n_mbon)
            _require(np.array_equal(pre, expected), "Registered control topology RNG")
            checks = {"same_edge_count": len(pre) == len(source["pre"]),
                      "same_kc_outdegree": bool(np.array_equal(np.bincount(pre, minlength=n_kc),
                                                              np.bincount(source["pre"], minlength=n_kc))),
                      "same_mbon_indegree": bool(np.array_equal(np.bincount(post, minlength=n_mbon),
                                                                np.bincount(source["post"], minlength=n_mbon))),
                      "same_incoming_weight_multisets": all(np.array_equal(np.sort(counts[post == p]),
                                                                            np.sort(source["counts"][source["post"] == p]))
                                                               for p in range(n_mbon)),
                      "duplicate_free": True,
                      "changed_edge_fraction": len(set(zip(source["pre"].tolist(), source["post"].tolist()))
                                                   - set(zip(pre.tolist(), post.tolist()))) / len(pre)}
            meta = controls[kind]["metadata"]
            equal(meta["checks"], checks, f"Topology invariants {kind}")
            _require(meta["seed"] == seed + offset, "Topology source stream seed")
            _require(all(checks[k] for k in ("same_edge_count", "same_mbon_indegree",
                                           "same_incoming_weight_multisets", "duplicate_free")), "Matched graph controls")
            if kind == "rewired":
                _require(checks["same_kc_outdegree"] and meta["accepted_swaps"] == accepted
                         and meta["attempts"] == attempts, "Exact degree-preserving rewiring counters")
        graphs[kind] = graph
    rng = np.random.default_rng(seed + cfg["rng"]["encoder_offset"])
    encoder = rng.normal(0, .8, (n_kc, 7))
    bias = rng.uniform(-.25, .25, n_kc)
    return graphs, encoder, bias, n_mbon


def source_check(root, registration_path, output, metadata, reg):
    cfg = reg["config"]
    config_check(cfg)
    equal(metadata["config"], cfg, "Result and registration configuration")
    _require(metadata["study_id"] == reg["study_id"] == cfg["study_id"], "Consistent study identity")
    _require(metadata["stage"] == reg["stage"] == cfg["stage"], "Consistent registered stage")
    _require(metadata["status"] == "completed", "Completed primary run required")
    _require(re.fullmatch(r"[0-9a-f]{40}", reg["source_git_commit"]), "Frozen source commit identity")
    _require(metadata["source_git_commit"] == reg["source_git_commit"], "Matching execution source commit")
    _require(digest(registration_path) == digest(output / "registration.json")
             == metadata["registration_sha256"], "Exact external/copied registration bytes")
    equal(read_json(output / "registration.json"), reg, "Copied registration contents")
    chronology = [utc(reg["registered_at_utc"]), utc(metadata["started_at_utc"]), utc(metadata["completed_at_utc"])]
    _require(chronology == sorted(chronology), "Registration must precede full execution")
    config_path = metadata.get("config_path", f"configs/{cfg['study_id']}.json")
    actual_config = safe(root, config_path)
    _require(digest(actual_config) == reg["config_sha256"], "Registered config byte hash")
    equal(read_json(actual_config), cfg, "Config source content")
    if "config_sha256" in metadata:
        _require(metadata["config_sha256"] == reg["config_sha256"], "Metadata config hash")
    frozen = reg["frozen_files"]
    _require(isinstance(frozen, dict) and frozen, "Nonempty source byte freeze")
    if "frozen_files" in cfg:
        _require(len(cfg["frozen_files"]) == len(set(cfg["frozen_files"]))
                 and set(frozen) == set(cfg["frozen_files"]), "Exact registered source-file closure")
    for name, sha in frozen.items():
        _require(type(sha) is str and re.fullmatch(r"[a-f0-9]{64}", sha)
                 and digest(safe(root, name)) == sha, f"Frozen source bytes: {name}")
    if "frozen_files" in metadata:
        equal(metadata["frozen_files"], frozen, "Executed source hashes")
    parents = reg["parent_files"]
    _require(isinstance(parents, dict), "Parent byte manifest")
    if "parent_files" in cfg:
        _require(set(parents) == set(cfg["parent_files"]), "Exact frozen parent inputs")
    for name, sha in parents.items():
        _require(digest(safe(root, name)) == sha, f"Unchanged parent evidence: {name}")
    if "parent_files" in metadata:
        equal(metadata["parent_files"], parents, "Executed parent evidence")
    return cfg


def replay_trajectory(base, seed, condition, env, cfg, datasets, streams, initial, curves, trials):
    encoder, bias, pre, post, initial_weights, initial_head, scale, linear, plastic = initial
    n, every = cfg["trials_per_phase"], cfg["checkpoint_every"]
    probe = _inputs(datasets["iid"], np.arange(1, 9))[:cfg["probe_rows"]]
    initial_kc, initial_mb, initial_decision, _ = _state(probe, encoder, bias, pre, post,
                                                        initial_weights, initial_head, scale, linear)
    normalizer = np.empty(0) if linear else np.maximum(
        np.bincount(post, weights=base["graph_counts"], minlength=initial_head.size), 1.)
    weights, head = initial_weights.copy(), initial_head.copy()
    upper = 2 * initial_weights
    head_updates = internal_updates = 0
    measurements, case_count, snapshots = {}, 0, 0
    path = base["path"] / condition / env["id"]

    def checkpoint(phase, t):
        nonlocal case_count, snapshots
        saved = _npz(path / f"{phase}_{t:04d}.npz", CHECKPOINT_FIELDS)
        _snapshot_check(saved, probe, encoder, bias, pre, post, weights, head, scale, linear,
                        head_updates, internal_updates, f"{seed}/{condition}/{env['id']}/{phase}/{t}")
        for field, value in (("initial_internal_weights", initial_weights),
                             ("internal_upper_bounds", upper), ("incoming_normalizer", normalizer)):
            _near(saved[field], value, f"Immutable internal scale/clipping initialization: {field}")
        kc, mb, decision, _ = _state(probe, encoder, bias, pre, post, weights, head, scale, linear)
        changes = {"internal_change_l2": float(np.linalg.norm(weights - initial_weights)),
                   "readout_change_l2": float(np.linalg.norm(head - initial_head)),
                   "probe_kc_change_l2": float(np.linalg.norm(kc - initial_kc)),
                   "probe_mbon_change_l2": float(np.linalg.norm(mb - initial_mb)),
                   "probe_decision_change_l2": float(np.linalg.norm(decision - initial_decision)),
                   "lower_bound_fraction": float(np.mean(weights <= 0)) if weights.size else 0.,
                   "upper_bound_fraction": float(np.mean(weights >= upper)) if weights.size else 0.}
        for split in ("iid", "ood"):
            ds = datasets[split]
            _, _, _, p = _state(_inputs(ds, np.arange(1, 9)), encoder, bias, pre, post, weights, head, scale, linear)
            _near(saved[f"{split}_p_right"], p, "Every locked evaluation case probability")
            value = metrics(p, ds["counts"], phase == "phase1", env)
            measurements[phase, t, split] = value
            row = curves[seed, condition, env["id"], phase, t, split]
            for field, number in {**value, **changes}.items():
                _near(_number(row, field), number, f"Independent curve {field}")
            _require(_integer(row, "head_updates") == head_updates
                     and _integer(row, "internal_updates") == internal_updates, "Checkpoint exact update counters")
            if condition == "frozen":
                _require(np.array_equal(head, initial_head) and np.array_equal(weights, initial_weights)
                         and np.array_equal(p, np.full_like(p, .5)), "Frozen state and probability unchanged")
            case_count += len(p)
        snapshots += 1

    for phase in ("phase1", "phase2"):
        checkpoint(phase, 0)
        ds = datasets[f"train_{phase}"]
        for index, stimulus in enumerate(_inputs(ds, np.arange(1, 9))):
            row = trials[seed, condition, env["id"], phase, index + 1]
            u, v = (streams[phase, kind][index] for kind in ("action", "reward"))
            result, next_weights, next_head = independent_step(
                stimulus, u, v, phase == "phase1", env, encoder, bias, pre, post, weights, head,
                scale, linear, plastic, condition == "frozen", cfg["head_lr"], cfg["internal_lr"], upper)
            _require(row["stage"] == cfg["stage"] and row["family_id"] == str(ds["family_ids"][index]),
                     "Original stage and stimulus family")
            for field, expected in (("global_trial", index + 1 + (n if phase == "phase2" else 0)),
                                    ("left_count", int(ds["counts"][index, 0])),
                                    ("right_count", int(ds["counts"][index, 1])),
                                    ("chosen_count", int(ds["counts"][index, result["action"]])),
                                    ("action", result["action"]), ("reward", result["reward"]),
                                    ("head_update_called", int(condition != "frozen")),
                                    ("internal_update_called", int(plastic and condition != "frozen"))):
                _require(_integer(row, field) == expected, f"Actual trial identity/feedback/call: {field}")
            for field, value in {**result, "action_uniform": u, "reward_uniform": v}.items():
                if field not in {"action", "reward"}:
                    _near(_number(row, field), value, f"Independent sampled trial arithmetic: {field}")
            head_updates += int(condition != "frozen")
            internal_updates += int(plastic and condition != "frozen")
            _require(_integer(row, "head_updates") == head_updates
                     and _integer(row, "internal_updates") == internal_updates, "Every post-trial update counter")
            if not plastic or not np.any(head) or condition == "frozen":
                _require(_number(row, "internal_step_l2") == 0, "Zero internal drift when update is unavailable")
            weights, head = next_weights, next_head
            if (index + 1) % every == 0:
                checkpoint(phase, index + 1)
    return {"measurements": measurements, "scale": scale, "head_parameters": initial_head.size,
            "internal_parameters": initial_weights.size if plastic else 0,
            "head_updates": head_updates, "internal_updates": internal_updates,
            "snapshots": snapshots, "case_probability_entries": case_count}


def resource_check(actual, key, replay, cfg, initial):
    seed, condition, environment = key
    weights, head, scale, plastic = initial[4], initial[5], initial[6], initial[8]
    expected = {"seed": seed, "condition": condition, "environment": environment,
                "feature_scale": scale, "initial_feature_gap_rms": scale,
                "readout_parameters": head.size,
                "trainable_readout_parameters": 0 if condition == "frozen" else head.size,
                "internal_parameters": weights.size,
                "trainable_internal_parameters": weights.size if plastic else 0,
                "head_update_calls": replay["head_updates"], "internal_update_calls": replay["internal_updates"],
                "privileged_update_calls": 0}
    equal(actual, expected, "Every independently reconstructed resource record")


def gate_check(root, cfg, reg):
    if cfg["stage"] == "linear":
        base = "results/metric_calibration_v1/"
        names = [base + suffix for suffix in ("summary.json", "verification.json", "artifact_hashes.json")]
        names += ["results/registration_metric_calibration_v1.json", "configs/metric_calibration_v1.json"]
        _require(set(names) <= set(reg["parent_files"]), "All calibration-gate inputs must be frozen")
        summary, checked, hashes = (read_json(safe(root, p)) for p in names[:3])
        required = ("H1_no_learning_feedback_difference", "H2_no_learning_high_persistence", "H3_pooling_masks_phase_difference")
        _require(summary["study_id"] == "metric_calibration_v1" and checked["status"] == "passed"
                 and all(summary["analytic_claims"][key]["criterion_met"] is True for key in required),
                 "Calibration fixed counterexamples and independent verification must pass")
        _require(hashes["summary.json"] == digest(safe(root, names[0]))
                 and checked["registration_sha256"] == digest(safe(root, names[3]))
                 and checked["config_sha256"] == digest(safe(root, names[4])), "Calibration-gate exact byte linkage")
        identity = cfg["calibration_gate_identity"]
        for field, name in (("summary_sha256", names[0]), ("verification_sha256", names[1]),
                             ("registration_sha256", names[3]), ("artifact_hashes_sha256", names[2])):
            _require(identity[field] == digest(safe(root, name)), f"Pinned completed calibration identity: {field}")
        if cfg["scope"] == "preregistered_diagnostic":
            _require(type(identity["published_commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", identity["published_commit"]),
                     "Pinned calibration publication commit declaration")
        expected = {"name": "calibration_before_linear", "passed": True, **identity}
    else:
        names = ["configs/stochastic_feedback_linear_v1.json", "results/registration_stochastic_feedback_linear_v1.json",
                 "results/stochastic_feedback_linear_v1/summary.json", "results/stochastic_feedback_linear_v1/verification.json"]
        _require(set(names) <= set(reg["parent_files"]), "All linear-gate inputs must be frozen")
        summary, checked = (read_json(safe(root, p)) for p in names[2:])
        _require(summary["study_id"] == "stochastic_feedback_linear_v1"
                 and summary["primary_hypothesis"]["criterion_met"] is True
                 and checked["status"] == "passed", "Linear H-L and independent verification required before network")
        _require(checked["summary_sha256"] == digest(safe(root, names[2]))
                 and checked["registration_sha256"] == digest(safe(root, names[1]))
                 and checked["config_sha256"] == digest(safe(root, names[0])), "Linear-gate exact byte linkage")
        _require(utc(checked["verified_at_utc"]) <= utc(reg["registered_at_utc"]),
                 "Network registration must follow independent linear verification")
        expected = {"name": "linear_H_L_before_network", "passed": True,
                    "summary_sha256": digest(safe(root, names[2])),
                    "verification_sha256": digest(safe(root, names[3]))}
    equal(reg["gate"], expected, "Independent reconstruction of registered parent gate")


def summary_check(cfg, measurements, trials, actual):
    n, every, crit = cfg["trials_per_phase"], cfg["checkpoint_every"], cfg["criteria"]
    times = list(range(0, n + 1, every))
    outcomes, failures = [], []
    for seed in cfg["seeds"]:
        for condition in cfg["conditions"]:
            for env in cfg["environments"]:
                environment = env["id"]
                for split in ("iid", "ood"):
                    key = (seed, condition, environment)
                    first = [measurements[*key, "phase1", t, split]["p_rule"] for t in times]
                    second = [measurements[*key, "phase2", t, split]["p_rule"] for t in times]
                    acquired = bool(first[-1] >= crit["acquisition_accuracy"] and first[-1] - first[0] >= crit["acquisition_gain"])
                    adapted = bool(second[-1] >= crit["adaptation_accuracy"])
                    area = _auc(second, times)
                    row = {"seed": seed, "condition": condition, "environment": environment, "split": split,
                           "phase1_start": first[0], "phase1_end": first[-1], "phase1_gain": first[-1] - first[0],
                           "phase1_threshold_met": acquired,
                           "phase1_acquired": acquired if environment != "uninformative" else None,
                           "phase2_start": second[0], "phase2_end": second[-1], "phase2_gain": second[-1] - second[0],
                           "phase2_raw_auc": area, "phase2_baseline_adjusted_auc": area - second[0],
                           "phase2_threshold_met": adapted,
                           "qualified_adaptation": bool(acquired and adapted) if environment != "uninformative" else None,
                           "first_observed_threshold_checkpoint": next((t for t, p in zip(times, second, strict=True)
                                                                         if p >= crit["adaptation_accuracy"]), None)}
                    outcomes.append(row)
                    if environment != "uninformative":
                        for reason, passed in (("phase1_acquisition_threshold_not_met", acquired),
                                               ("phase2_endpoint_threshold_not_met", adapted),
                                               ("acquired_old_rule_then_phase2_threshold_not_met", not acquired or adapted)):
                            if not passed:
                                failures.append({"seed": seed, "condition": condition, "environment": environment,
                                                 "split": split, "reason": reason})
    keyed = {(r["seed"], r["condition"], r["environment"], r["split"]): r for r in outcomes}
    numeric_fields = ("phase1_start", "phase1_end", "phase1_gain", "phase2_start", "phase2_end", "phase2_gain",
                      "phase2_raw_auc", "phase2_baseline_adjusted_auc")
    verdicts = []
    for condition in cfg["conditions"]:
        for env in cfg["environments"]:
            environment = env["id"]
            for split in ("iid", "ood"):
                rows = [keyed[seed, condition, environment, split] for seed in cfg["seeds"]]
                statistics = {}
                for field in numeric_fields:
                    values = np.asarray([r[field] for r in rows])
                    statistics[field] = {"mean": float(values.mean()),
                                         "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None}
                verdicts.append({"condition": condition, "environment": environment, "split": split, "n_seeds": len(rows),
                                 "phase1_threshold_count": sum(r["phase1_threshold_met"] for r in rows),
                                 "phase1_acquisition_count": sum(r["phase1_acquired"] for r in rows) if environment != "uninformative" else None,
                                 "phase2_threshold_count": sum(r["phase2_threshold_met"] for r in rows),
                                 "qualified_adaptation_count": sum(r["qualified_adaptation"] for r in rows) if environment != "uninformative" else None,
                                 "statistics": statistics})
    contrasts, network_contrasts = [], []
    for seed in cfg["seeds"]:
        for split in ("iid", "ood"):
            for condition in cfg["conditions"]:
                for name, left, right in (("probabilistic_minus_uninformative", "probabilistic", "uninformative"),
                                          ("deterministic_minus_probabilistic", "deterministic", "probabilistic")):
                    for field in numeric_fields:
                        contrasts.append({"seed": seed, "condition": condition, "split": split, "comparison": name,
                                          "metric": field, "difference": keyed[seed, condition, left, split][field]
                                                                                 - keyed[seed, condition, right, split][field]})
            if cfg["stage"] == "network":
                for env in cfg["environments"]:
                    environment = env["id"]
                    for left, right in (("plastic", "fixed"), ("rewired_plastic", "rewired_fixed"), ("ann_plastic", "ann_fixed"),
                                        ("rewired_fixed", "fixed"), ("rewired_plastic", "plastic"), ("ann_fixed", "fixed"), ("ann_plastic", "plastic")):
                        for field in numeric_fields:
                            network_contrasts.append({"seed": seed, "environment": environment, "split": split,
                                "comparison": f"{left}_minus_{right}", "metric": field,
                                "difference": keyed[seed, left, environment, split][field] - keyed[seed, right, environment, split][field]})
    count = sum(keyed[s, "linear", "probabilistic", "iid"]["phase1_acquired"] for s in cfg["seeds"])
    gaps = [{"seed": s, "difference": keyed[s, "linear", "probabilistic", "iid"]["phase1_end"]
                                         - keyed[s, "linear", "uninformative", "iid"]["phase1_end"]} for s in cfg["seeds"]]
    mean_gap = float(np.mean([r["difference"] for r in gaps]))
    fraction = count / len(cfg["seeds"])
    fraction_met, gap_met = fraction >= crit["seed_pass_fraction"], mean_gap >= crit["paired_reward_gap"]
    hypothesis = {"name": "H_L", "probabilistic_acquisition_count": count, "n_seeds": len(cfg["seeds"]),
                  "acquisition_fraction": fraction, "all_seed_endpoint_gaps": gaps,
                  "mean_paired_reward_gap": mean_gap, "fraction_criterion_met": bool(fraction_met),
                  "gap_criterion_met": bool(gap_met), "criterion_met": bool(fraction_met and gap_met),
                  "scope": "gate_decision" if cfg["stage"] == "linear" else "descriptive_independent_replication_not_retroactive_gate"}
    def contrast_statistics(rows, coordinates):
        groups = {}
        for row in rows:
            key = tuple(row[k] for k in coordinates)
            groups.setdefault(key, []).append(row["difference"])
        result = []
        for key, values in groups.items():
            _require(len(values) == len(cfg["seeds"]), "All-seed paired contrast group")
            result.append({**dict(zip(coordinates, key, strict=True)), "n_seeds": len(values),
                           "mean": float(np.mean(values)),
                           "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None})
        return result
    expected = {"study_id": cfg["study_id"], "stage": cfg["stage"], "status": "completed", "scope": cfg["scope"],
                "outcomes": outcomes, "family_verdicts": verdicts, "environment_contrasts": contrasts,
                "network_contrasts": network_contrasts, "primary_hypothesis": hypothesis, "failures": failures,
                "environment_contrast_statistics": contrast_statistics(contrasts, ("condition", "split", "comparison", "metric")),
                "network_contrast_statistics": contrast_statistics(network_contrasts, ("environment", "split", "comparison", "metric")),
                "interpretation": "IID and artificial nuisance OOD only; uninformative thresholds are designated-direction chance events; no fresh-control speed or topology-superiority claim"}
    equal(actual, expected, "Every independently reconstructed outcome, contrast, failure and H-L gate")


def audit(root, registration_path, output, report):
    reg, metadata, summary = (read_json(p) for p in
                              (registration_path, output / "metadata.json", output / "summary.json"))
    cfg = source_check(root, registration_path, output, metadata, reg)
    equal(metadata["gate"], reg["gate"], "Execution gate equals registered parent gate")
    gate_check(root, cfg, reg)
    n, ev, every = cfg["trials_per_phase"], cfg["eval_trials"], cfg["checkpoint_every"]
    seeds, conditions = cfg["seeds"], cfg["conditions"]
    environments = [e["id"] for e in cfg["environments"]]
    checkpoints = list(range(0, n + 1, every))
    required = {"metadata.json", "registration.json", "summary.json", "controls.json",
                "model_resources.json", "learning_curves.csv", "stages.json"}
    for seed in seeds:
        base = f"seed_{seed}"
        required.update(f"{base}/{name}" for name in ("stimuli.npz", "trials.csv"))
        if cfg["stage"] == "network":
            required.update(f"{base}/{kind}_topology.npz" for kind in ("real", "rewired", "ann"))
        for condition in conditions:
            for environment in environments:
                required.update(f"{base}/{condition}/{environment}/{phase}_{t:04d}.npz"
                                for phase in ("phase1", "phase2") for t in checkpoints)
    hashes = read_json(output / "artifact_hashes.json")
    _require(set(hashes) == required, "Exact required primary artifact coverage")
    for name, sha in hashes.items():
        _require(digest(safe(output, name)) == sha, f"Primary artifact bytes: {name}")
    actual_files = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    _require(actual_files - {"artifact_hashes.json", "verification.json"} == required,
             "No unregistered or omitted primary artifacts")
    curves = csv_index(output / "learning_curves.csv", CURVE_FIELDS,
                       ("seed", "condition", "environment", "phase", "trial", "split"))
    expected_curves = {(s, c, e, phase, t, split) for s in seeds for c in conditions for e in environments
                       for phase in ("phase1", "phase2") for t in checkpoints for split in ("iid", "ood")}
    _require(set(curves) == expected_curves, "Complete all-seed locked evaluation coverage")
    controls_list = read_json(output / "controls.json")
    controls = {(r["seed"], r["kind"]): r for r in controls_list}
    expected_controls = {(s, k) for s in seeds for k in ("real", "rewired", "ann")} if cfg["stage"] == "network" else set()
    _require(len(controls_list) == len(controls) and set(controls) == expected_controls,
             "Network-only topology control coverage")
    resource_list = read_json(output / "model_resources.json")
    resources = {(r["seed"], r["condition"], r["environment"]): r for r in resource_list}
    expected_trajectories = {(s, c, e) for s in seeds for c in conditions for e in environments}
    _require(len(resource_list) == len(resources) and set(resources) == expected_trajectories,
             "All-seed all-environment model resource coverage")
    measurements, all_trials, fingerprints = {}, {}, set()
    counters = {"actual_choice_feedback_trials": 0, "head_update_calls": 0, "internal_update_calls": 0,
                "privileged_update_calls": 0, "checkpoints": 0, "case_probability_entries": 0}
    split_specs = (("train_phase1", n, 51), ("train_phase2", n, 54), ("iid", ev, 52), ("ood", ev, 53))
    for seed in seeds:
        base = output / f"seed_{seed}"
        fields = {f"{name}_{field}" for name, _, _ in split_specs for field in ("counts", "nuisance", "family_ids")}
        fields.update(f"{kind}_{phase}_uniforms" for kind in ("action", "reward") for phase in ("phase1", "phase2"))
        stimuli = _npz(base / "stimuli.npz", fields)
        datasets = {}
        for name, size, offset in split_specs:
            ds = _dataset(stimuli, name, size, (2, 4), fingerprints)
            expected = _regenerate_dataset(seed * 100 + offset, name, size, (2, 4))
            _require(all(np.array_equal(ds[k], v) for k, v in expected.items()), f"Independent raw data regeneration: {name}")
            datasets[name] = ds
        streams = {}
        for phase in ("phase1", "phase2"):
            for kind in ("action", "reward"):
                offset = cfg["rng"][f"{phase}_{kind}_offset"]
                values = np.random.default_rng(seed + offset).random(n)
                _require(np.array_equal(stimuli[f"{kind}_{phase}_uniforms"], values),
                         "Registered separate action/reward RNG arrays shared across conditions")
                streams[phase, kind] = values
        trials = csv_index(base / "trials.csv", TRIAL_FIELDS, ("seed", "condition", "environment", "phase", "trial"))
        expected_trials = {(seed, c, e, p, t) for c in conditions for e in environments
                           for p in ("phase1", "phase2") for t in range(1, n + 1)}
        _require(set(trials) == expected_trials, "All actual trials retained without privileged update stream")
        all_trials.update(trials)
        if cfg["stage"] == "network":
            graphs, encoder, bias, n_mbon = regenerate_graphs(
                root, cfg, base, seed, {k: controls[seed, k] for k in ("real", "rewired", "ann")})
        else:
            graphs, encoder, bias, n_mbon = {}, np.empty((0, 7)), np.empty(0), 0
        for condition in conditions:
            kind, plastic = FLAGS[condition]
            graph = graphs.get(kind)
            initial = initialize(condition, graph, encoder, bias,
                                 _inputs(datasets["train_phase1"], np.arange(1, 9)), n_mbon)
            for env in cfg["environments"]:
                key = (seed, condition, env["id"])
                replay = replay_trajectory({"path": base, "graph_counts": np.empty(0) if graph is None else graph["counts"]},
                                          seed, condition, env, cfg, datasets, streams, initial, curves, trials)
                resource_check(resources[key], key, replay, cfg, initial)
                for measurement_key, value in replay["measurements"].items():
                    measurements[*key, *measurement_key] = value
                counters["actual_choice_feedback_trials"] += 2 * n
                counters["head_update_calls"] += replay["head_updates"]
                counters["internal_update_calls"] += replay["internal_updates"]
                counters["checkpoints"] += replay["snapshots"]
                counters["case_probability_entries"] += replay["case_probability_entries"]
        print(f"independently audited seed {seed} ({seeds.index(seed) + 1}/{len(seeds)})", flush=True)
    summary_check(cfg, measurements, all_trials, summary)
    stage_rows = []
    for seed in seeds:
        for condition in conditions:
            plastic = FLAGS[condition][1]
            for environment in environments:
                for phase_index, phase in enumerate(("phase1", "phase2"), 1):
                    rows = [all_trials[seed, condition, environment, phase, t] for t in range(1, n + 1)]
                    choice_count = sum(int(_integer(r, "chosen_count") == (4 if phase == "phase1" else 2)) for r in rows)
                    rewards = sum(_integer(r, "reward") for r in rows)
                    stage_rows.append({"seed": seed, "condition": condition, "environment": environment, "phase": phase,
                                       "actual_trials": n, "actual_designated_choice_count": choice_count,
                                       "actual_designated_choice_fraction": choice_count / n,
                                       "actual_reward_count": rewards, "actual_reward_rate": rewards / n,
                                       "head_update_calls": n if condition != "frozen" else 0,
                                       "internal_update_calls": n if plastic else 0,
                                       "cumulative_head_updates": phase_index * n if condition != "frozen" else 0,
                                       "cumulative_internal_updates": phase_index * n if plastic else 0})
    equal(read_json(output / "stages.json"), stage_rows, "Every phase's actual choices/rewards and physical/cumulative update counts")
    for field in ("actual_choice_feedback_trials", "head_update_calls", "internal_update_calls", "privileged_update_calls"):
        equal(metadata[field], counters[field], f"Actual physical/update budget {field}")
    _require(metadata["human_or_species_comparison"] is False, "No species-comparison claim")
    report["counts"] = {**counters, "seeds": len(seeds), "conditions": len(conditions),
                        "environments": len(environments), "trajectories": len(expected_trajectories),
                        "curve_rows": len(curves), "artifact_hashes": len(hashes),
                        "source_hashes": len(reg["frozen_files"]), "canonical_fingerprints": len(fingerprints)}
    report.update(study_id=cfg["study_id"], stage=cfg["stage"], summary_sha256=digest(output / "summary.json"),
                  registration_sha256=digest(registration_path), config_sha256=reg["config_sha256"])
    report["checks"] = ["Exact registered source/config/parent bytes and UTC execution chronology",
                        "Every split regenerated independently with joint nuisance crossing and canonical disjointness",
                        "Independent sparse topology, initialization, original normalization and clipping bounds",
                        "Every actual choice and sampled reward replayed with separate shared uniform arrays",
                        "Reward-only head and old-head internal updates; frozen update calls skipped",
                        "All checkpoint parameters, counters, raw/decision probe activities and IID/OOD case probabilities",
                        "All acquisition/adaptation flags, all-seed gaps, starting points, gains, AUCs and failure records",
                        "Registered parent gate and exact physical choice/head/internal/privileged-update budgets"]


def verify(root, registration, result_dir):
    root, registration, output = map(lambda p: Path(p).resolve(), (root, registration, result_dir))
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "status": "failed", "passed": False,
              "method": "independent numpy/stdlib replay using only immutable independent-auditor arithmetic helpers",
              "counts": {}, "checks": [], "errors": [], "not_verified": [
                  "Private remote Git authenticity; registration/publication chronology beyond recorded local evidence",
                  "Equal model capacity, feature directions, optimization resources, or biological individuals",
                  "Visual number extraction, LC11/DAN dynamics, common human/fly neural mechanism, or educational effect",
                  "Intrinsic learning speed or causal reversal cost without a matched fresh-training control",
                  "Higher p_rule under uninformative reward as evidence of genuine rule acquisition"]}
    try:
        _require(root.is_dir() and output.is_dir() and registration.is_file(), "Existing source/run/registration required")
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
    _require(not args.out.exists(), "Verification report output already exists")
    report = verify(args.root, args.registration, args.result_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "errors": report["errors"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
