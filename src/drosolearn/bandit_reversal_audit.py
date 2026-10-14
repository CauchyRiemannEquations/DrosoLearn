"""Independent arithmetic replay of normalized reward reversal histories.

Imports only previous independent auditor helpers, never any learner, producer
data generator, runner, metric or summary implementation. Expected-action
reference updates are enumerated over both possible actions independently.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import csv

import numpy as np

from .encoding_audit import (
    _require, _near, _integer, _number, _json, _hash, _npz,
    _regenerate_dataset, _dataset, _inputs, _mbon, _topologies,
)
from .linearity_audit import _source_projection

_HISTORIES = ("fresh_low", "high_to_low", "high_to_low_reset_head", "low_to_low")
_FLAGS = {"linear_normalized": ("linear", False), "fixed": ("real", False), "plastic": ("real", True),
          "rewired_fixed": ("rewired", False), "rewired_plastic": ("rewired", True),
          "ann_fixed": ("ann", False), "ann_plastic": ("ann", True)}
_SAVED_FIELDS = {"internal_weights", "readout_weights", "kc_activity", "mbon_activity", "choice_probability",
                 "encoder", "bias", "pre", "post", "decision_activity", "feature_scale", "initial_feature_gap_rms",
                 "head_updates", "internal_updates"}
_CHECKPOINT_FIELDS = _SAVED_FIELDS | {"iid_p_right", "ood_p_right"}
_TRIAL_FIELDS = {"seed", "condition", "history", "phase", "trial", "family_id", "left_count", "right_count",
                 "action", "reward", "p_right", "optimizer_factor", "internal_step_l2", "readout_step_l2",
                 "kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2", "decision_left_l2", "decision_right_l2"}
_CURVE_FIELDS = {"seed", "condition", "history", "phase", "trial", "split", "p_larger", "p_smaller",
                 "p_target", "greedy_accuracy", "mean_policy_sensitivity", "internal_change_l2",
                 "readout_change_l2", "probe_kc_change_l2", "probe_mbon_change_l2", "lower_bound_fraction",
                 "upper_bound_fraction", "probe_decision_change_l2", "larger"}
_EXPECTED_FIELDS = {"seed", "history", "phase", "trial", "family_id", "left_count", "right_count",
                    "target", "p_right", "expected_factor", "readout_step_l2"}


def _csv(path, fields, keys):
    result = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(set(reader.fieldnames or ()) == set(fields), f"CSV schema: {path}")
        for row in reader:
            _require(None not in row and all(value is not None for value in row.values()), "Malformed CSV row")
            key = tuple(_integer(row, field) if field in {"seed", "trial", "case"} else row[field] for field in keys)
            _require(key not in result, f"Duplicate CSV key: {key}")
            result[key] = row
    return result


def _larger(history, phase):
    return phase == "phase1" and history != "low_to_low"


def _raw_features(inputs, encoder, bias, pre, post, weights, head_size, linear):
    if linear:
        empty = np.empty((*inputs.shape[:-1], 0), dtype=float)
        return empty, empty.copy(), np.asarray(inputs, dtype=float)
    kc = np.maximum(inputs @ encoder.T + bias, 0)
    mb = _mbon(kc, pre, post, weights, head_size)
    return kc, mb, mb


def _state(inputs, encoder, bias, pre, post, weights, head, scale, linear):
    kc, mb, raw = _raw_features(inputs, encoder, bias, pre, post, weights, head.size, linear)
    decision = raw / scale
    scores = decision @ head
    probability = 1 / (1 + np.exp(-np.clip(scores[..., 1] - scores[..., 0], -40, 40)))
    return kc, mb, decision, probability


def _snapshot_check(saved, probe, encoder, bias, pre, post, weights, head, scale,
                    linear, head_updates, internal_updates, label):
    for field, value in (("encoder", encoder), ("bias", bias), ("pre", pre), ("post", post)):
        _require(np.array_equal(saved[field], value), f"Frozen initialization/topology: {label}/{field}")
    for field, value in (("feature_scale", scale), ("initial_feature_gap_rms", scale)):
        _require(saved[field].shape == (), f"Scalar train-only normalization: {label}/{field}")
        _near(saved[field], value, f"Frozen train-only scale: {label}/{field}")
    for field, value in (("head_updates", head_updates), ("internal_updates", internal_updates)):
        _require(saved[field].shape == () and np.issubdtype(saved[field].dtype, np.integer)
                 and int(saved[field]) == value, f"Exact integer update counter: {label}/{field}")
    _near(saved["internal_weights"], weights, f"Replayed internal state: {label}")
    _near(saved["readout_weights"], head, f"Replayed readout state: {label}")
    kc, mb, decision, probability = _state(probe, encoder, bias, pre, post, weights, head, scale, linear)
    for field, value in (("kc_activity", kc), ("mbon_activity", mb), ("decision_activity", decision),
                         ("choice_probability", probability)):
        _near(saved[field], value, f"Every reconstructed probe: {label}/{field}")
    _require(all(np.isfinite(saved[field]).all() for field in saved), f"Finite snapshot: {label}")


def _metrics(probability, counts, larger):
    high = (counts[:, 1] > counts[:, 0]).astype(int)
    target = high if larger else 1 - high
    correct = np.where(np.abs(probability - .5) < 1e-12, .5,
                       ((probability > .5).astype(int) == target).astype(float))
    p_large = float(np.mean(np.where(high == 1, probability, 1 - probability)))
    return {"p_larger": p_large, "p_smaller": 1 - p_large,
            "p_target": p_large if larger else 1 - p_large,
            "greedy_accuracy": float(np.mean(correct)),
            "mean_policy_sensitivity": float(np.mean(probability * (1 - probability)))}, target, correct


def _expected_factor(probability, target):
    # Enumerate the policy's two actions instead of copying the producer's
    # collapsed expression. This expectation is local to this reference state.
    return float(sum(mass * (action - probability) * (float(action == target) - .5)
                     for action, mass in ((0, 1 - probability), (1, probability))))


def _auc(values, checkpoints):
    return float(sum((a + b) * (right - left) / 2
                     for a, b, left, right in zip(values[:-1], values[1:], checkpoints[:-1], checkpoints[1:]))
                 / checkpoints[-1])


def _equal(actual, expected, label):
    if type(expected) is bool or expected is None:
        _require(actual is expected, f"Independent summary gate: {label}")
    elif isinstance(expected, str):
        _require(type(actual) is str and actual == expected, f"Independent summary text: {label}")
    elif isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), f"Independent summary schema: {label}")
        for key, value in expected.items():
            _equal(actual[key], value, f"{label}/{key}")
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), f"Independent summary list: {label}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            _equal(a, b, f"{label}/{index}")
    else:
        _near(actual, expected, f"Independent summary value: {label}")


def _source_freeze(output, metadata, registration, cfg):
    root = (output / metadata["source_root"]).resolve()
    _require(root.is_dir(), "Registered source root missing; use the registered code checkout")
    config = Path(metadata["config_path"])
    config = (root / config).resolve() if not config.is_absolute() else config.resolve()
    _require(config.is_file() and _hash(config) == registration["config_sha256"] and _json(config) == cfg,
             "Registered config bytes/content")
    expected = {path.relative_to(root).as_posix() for path in (root / "src").rglob("*.py") if path.is_file()}
    expected.update(path.relative_to(root).as_posix() for path in (root / cfg["graph"]).glob("*") if path.is_file())
    expected.update(("docs/bandit-reversal-protocol.md", "docs/bandit-reversal-background.md", "pyproject.toml",
                     "requirements-lock.txt", "data/source-manifest.json", "data/LICENSE.md"))
    _require(set(registration["source_hashes"]) == expected, "Registered source file set")
    for relative, digest in registration["source_hashes"].items():
        source = (root / relative).resolve()
        _require(source.is_relative_to(root) and source.is_file() and _hash(source) == digest,
                 f"Registered source byte hash: {relative}")
    return root


def _config_check(cfg):
    _require(cfg["scope"] in {"preregistered_diagnostic", "integration_test"}, "Supported study scope")
    seeds = cfg["seeds"]
    _require(seeds and len(seeds) == len(set(seeds)) and
             all(type(seed) is int and seed >= 0 for seed in seeds), "Distinct nonnegative integer seeds")
    _require(tuple(cfg["conditions"]) == tuple(_FLAGS) and tuple(cfg["histories"]) == _HISTORIES,
             "Exact seven reward learners and four matched histories")
    _require(cfg["pair"] == [2, 4], "Fixed artificial quantity pair")
    fixed = {"experiment_id": "independent_bandit_reversal_diagnostic_v1",
             "graph": "data/flywire_783_mb_left_kc512_s701",
             "head_initialization": "zeros_all_conditions",
             "feature_normalization": "fixed_initial_phase1_train_feature_gap_rms",
             "expected_reference": "conditional_expected_action_linear"}
    _require(all(cfg.get(key) == value for key, value in fixed.items()),
             "Fixed study identity, graph, zero initialization and reference design")
    budget = tuple(cfg[key] for key in ("trials_per_phase", "checkpoint_every", "eval_trials", "probe_rows"))
    n, every, ev, probe = budget
    _require(all(type(value) is int and value > 0 for value in budget) and
             n % 4 == ev % 4 == n % every == 0 and probe <= ev, "Crossing/checkpoint/probe budgets")
    _require(all(type(cfg[key]) in (int, float) and np.isfinite(cfg[key]) and cfg[key] >= 0
                 for key in ("head_lr", "internal_lr")), "Finite nonnegative learning rates")
    criteria = cfg["criteria"]
    _require(set(criteria) == {"acquisition_accuracy", "acquisition_gain", "adaptation_accuracy",
                              "endpoint_cost", "seed_pass_fraction"} and
             all(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1
                 for value in criteria.values()) and criteria["seed_pass_fraction"] > 0,
             "Fixed criterion schema and ranges")
    if cfg["scope"] == "integration_test":
        _require(not any(seed in range(7001, 7011) for seed in seeds), "Scientific seeds cannot alias fixtures")
    else:
        _require(seeds == list(range(7001, 7011)), "Ten prospective new scientific seeds")
        _require((*budget, cfg["head_lr"], cfg["internal_lr"]) == (400, 100, 128, 8, .1, .01),
                 "Fixed registered resources")
        _require(criteria == {"acquisition_accuracy": .75, "acquisition_gain": .15,
                              "adaptation_accuracy": .75, "endpoint_cost": .1, "seed_pass_fraction": .8},
                 "Fixed registered criteria")


def _replay_history(path, seed, condition, history, cfg, datasets, uniforms,
                    encoder, bias, pre, post, w0, initial_head, scale, linear,
                    plastic, records, curves, expected_reference=False):
    n, every, ev, probe_n = (cfg[key] for key in
                             ("trials_per_phase", "checkpoint_every", "eval_trials", "probe_rows"))
    checkpoints = list(range(0, n + 1, every))
    probe = _inputs(datasets["iid"], np.arange(1, 9))[:probe_n]
    initial_kc, initial_mb, initial_decision, _ = _state(probe, encoder, bias, pre, post, w0,
                                                       initial_head, scale, linear)
    snapshots, measurements = {}, {}
    weights, head = w0.copy(), initial_head.copy()
    head_updates = internal_updates = 0

    def checkpoint(phase, trial, record=True):
        saved = _npz(path / f"{phase}_{trial:04d}.npz", _CHECKPOINT_FIELDS)
        snapshots[phase, trial] = saved
        _snapshot_check(saved, probe, encoder, bias, pre, post, weights, head, scale,
                        linear, head_updates, internal_updates, f"{condition}/{history}/{phase}/{trial}")
        changes = {"internal_change_l2": float(np.linalg.norm(weights - w0)),
                   "readout_change_l2": float(np.linalg.norm(head - initial_head)),
                   "probe_kc_change_l2": float(np.linalg.norm(saved["kc_activity"] - initial_kc)),
                   "probe_mbon_change_l2": float(np.linalg.norm(saved["mbon_activity"] - initial_mb)),
                   "probe_decision_change_l2": float(np.linalg.norm(saved["decision_activity"] - initial_decision))}
        for split in ("iid", "ood"):
            data = datasets[split]
            _, _, _, probability = _state(_inputs(data, np.arange(1, 9)), encoder, bias, pre, post,
                                          weights, head, scale, linear)
            _near(saved[f"{split}_p_right"], probability, f"Full checkpoint case probability: {split}")
            if not record:
                continue
            metrics, _, _ = _metrics(probability, data["counts"], _larger(history, phase))
            measurements[phase, trial, split] = metrics
            row = curves[seed, condition, history, phase, trial, split]
            _require(_integer(row, "larger") == int(_larger(history, phase)), "Curve reward direction")
            bounds = {"lower_bound_fraction": float(np.mean(weights <= 0)) if weights.size else 0.,
                      "upper_bound_fraction": float(np.mean(weights >= 2 * w0)) if weights.size else 0.}
            for field, value in {**metrics, **changes, **bounds}.items():
                _near(_number(row, field), value, f"Every independent curve: {condition}/{history}/{phase}/{trial}/{split}/{field}")

    if history == "fresh_low":
        checkpoint("phase1", 0, record=False)
    before_boundary = after_boundary = None
    for phase in (("phase2",) if history == "fresh_low" else ("phase1", "phase2")):
        if phase == "phase2":
            before_boundary = _npz(path / "boundary_before.npz", _SAVED_FIELDS)
            _snapshot_check(before_boundary, probe, encoder, bias, pre, post, weights, head, scale,
                            linear, head_updates, internal_updates, f"{condition}/{history}/before boundary")
            old_head = head.copy()
            if history == "high_to_low_reset_head":
                head[:] = 0.
            after_boundary = _npz(path / "boundary_after.npz", _SAVED_FIELDS)
            _snapshot_check(after_boundary, probe, encoder, bias, pre, post, weights, head, scale,
                            linear, head_updates, internal_updates, f"{condition}/{history}/after boundary")
            head_reset_l2 = float(np.linalg.norm(head - old_head))
        checkpoint(phase, 0)
        data = datasets[f"train_{phase}"]
        inputs = _inputs(data, np.arange(1, 9))
        for index, stimulus in enumerate(inputs):
            row_key = (seed, history, phase, index + 1) if expected_reference else (seed, condition, history, phase, index + 1)
            row = records[row_key]
            kc, mb, raw = _raw_features(stimulus, encoder, bias, pre, post, weights, head.size, linear)
            decision = raw / scale
            probability = float(1 / (1 + np.exp(-np.clip((decision[1] - decision[0]) @ head, -40, 40))))
            _near(_number(row, "p_right"), probability, "Every pre-update trial probability")
            _require(row["family_id"] == str(data["family_ids"][index]) and
                     _integer(row, "left_count") == data["counts"][index, 0] and
                     _integer(row, "right_count") == data["counts"][index, 1], "Shared stimulus identity")
            target = int(data["counts"][index, 1] > data["counts"][index, 0])
            if not _larger(history, phase):
                target = 1 - target
            if expected_reference:
                _require(_integer(row, "target") == target, "Reference privileged target rule")
                factor = _expected_factor(probability, target)
                _near(_number(row, "expected_factor"), factor, "Independently enumerated expected-action factor")
            else:
                action, reward = _integer(row, "action"), _integer(row, "reward")
                _require(action in (0, 1) and reward in (0, 1), "Binary chosen action and reward")
                _require(action == int(uniforms[phase][index] < probability), "Every shared action stream choice")
                _require(reward == int(action == target), "Phase/history contingent reward rule")
                factor = (action - probability) * (reward - .5)
                _near(_number(row, "optimizer_factor"), factor, "Every reward-only optimizer factor")
            next_head = head + cfg["head_lr"] * factor * (decision[1] - decision[0])
            if plastic:
                eligibility = head[post] * (kc[1, pre] - kc[0, pre]) / scale
                next_weights = np.minimum(np.maximum(weights + cfg["internal_lr"] * factor * eligibility, 0), 2 * w0)
            else:
                next_weights = weights.copy()
            if expected_reference:
                _near(_number(row, "readout_step_l2"), np.linalg.norm(next_head - head), "Reference readout update")
            else:
                values = {"internal_step_l2": np.linalg.norm(next_weights - weights),
                          "readout_step_l2": np.linalg.norm(next_head - head),
                          "kc_left_l2": np.linalg.norm(kc[0]), "kc_right_l2": np.linalg.norm(kc[1]),
                          "mbon_left_l2": np.linalg.norm(mb[0]), "mbon_right_l2": np.linalg.norm(mb[1]),
                          "decision_left_l2": np.linalg.norm(decision[0]), "decision_right_l2": np.linalg.norm(decision[1])}
                for field, value in values.items():
                    _near(_number(row, field), value, f"Independently replayed trial update/activity: {field}")
                if not plastic or not np.any(head):
                    _require(_number(row, "internal_step_l2") == 0, "Fixed or zero-old-head internal update")
            weights, head = next_weights, next_head
            head_updates += 1
            internal_updates += int(plastic)
            if (index + 1) % every == 0:
                checkpoint(phase, index + 1)
    return {"snapshots": snapshots, "measurements": measurements,
            "before_boundary": before_boundary, "after_boundary": after_boundary,
            "head_reset_l2": head_reset_l2, "final_head_updates": head_updates,
            "final_internal_updates": internal_updates}


def _summary(cfg, measurements, summary):
    n = cfg["trials_per_phase"]
    checkpoints = list(range(0, n + 1, cfg["checkpoint_every"]))
    criteria, outcomes, contrasts, verdicts = cfg["criteria"], [], [], []
    conditions = (*cfg["conditions"], "expected_linear")
    for seed in cfg["seeds"]:
        for condition in conditions:
            for history in _HISTORIES:
                base = (seed, condition, history)
                sequence = [measurements[*base, "phase2", t, "iid"]["p_target"] for t in checkpoints]
                if history == "fresh_low":
                    before = after = acquired = None
                else:
                    before, after = (measurements[*base, "phase1", t, "iid"]["p_target"] for t in (0, n))
                    acquired = bool(after >= criteria["acquisition_accuracy"] and after - before >= criteria["acquisition_gain"])
                fresh_acquired = bool(sequence[-1] >= criteria["acquisition_accuracy"] and
                                      sequence[-1] - sequence[0] >= criteria["acquisition_gain"])
                target_passed = bool(sequence[-1] >= criteria["adaptation_accuracy"])
                qualified = fresh_acquired if history == "fresh_low" else bool(acquired and target_passed)
                auc = _auc(sequence, checkpoints)
                calls = n if history == "fresh_low" else 2 * n
                plastic = condition in _FLAGS and _FLAGS[condition][1]
                reference = condition == "expected_linear"
                outcomes.append({"seed": seed, "condition": condition, "history": history,
                                 "phase1_before": before, "phase1_after": after, "phase1_acquired": None if reference else acquired,
                                 "phase2_before": sequence[0], "phase2_after": sequence[-1],
                                 "phase2_gain": sequence[-1] - sequence[0], "phase2_auc": auc,
                                 "phase2_baseline_adjusted_auc": auc - sequence[0], "phase2_target_passed": None if reference else target_passed,
                                 "qualified_adaptation": None if reference else bool(qualified),
                                 "fresh_acquired": fresh_acquired if history == "fresh_low" and not reference else None,
                                 "first_observed_75_checkpoint": None if reference else next((t for t, value in zip(checkpoints, sequence)
                                                                        if value >= criteria["adaptation_accuracy"]), None),
                                 "phase2_ood_after": measurements[*base, "phase2", n, "ood"]["p_target"],
                                 "phase2_greedy_after": measurements[*base, "phase2", n, "iid"]["greedy_accuracy"],
                                 "head_updates": calls, "internal_updates": calls if plastic else 0,
                                 "algorithm": "conditional_expected_action_reference" if condition == "expected_linear"
                                              else "stochastic_reward"})
    rows = {(row["seed"], row["condition"], row["history"]): row for row in outcomes}
    for condition in cfg["conditions"]:
        selected = []
        for seed in cfg["seeds"]:
            fresh, reverse, reset, repeated = (rows[seed, condition, history] for history in _HISTORIES)
            record = {"seed": seed, "condition": condition,
                      "endpoint_cost": fresh["phase2_after"] - reverse["phase2_after"],
                      "raw_auc_cost": fresh["phase2_auc"] - reverse["phase2_auc"],
                      "baseline_adjusted_auc_cost": fresh["phase2_baseline_adjusted_auc"] - reverse["phase2_baseline_adjusted_auc"],
                      "gain_cost": fresh["phase2_gain"] - reverse["phase2_gain"],
                      "reset_minus_reverse_endpoint": reset["phase2_after"] - reverse["phase2_after"],
                      "reset_minus_fresh_endpoint": reset["phase2_after"] - fresh["phase2_after"],
                      "repeated_minus_reverse_endpoint": repeated["phase2_after"] - reverse["phase2_after"],
                      "old_rule_acquired": reverse["phase1_acquired"], "fresh_acquired": fresh["fresh_acquired"]}
            contrasts.append(record); selected.append(record)
        costs = [row["endpoint_cost"] for row in selected]
        mean_cost = float(np.mean(costs))
        positive = sum(value > 0 for value in costs)
        old_count = sum(row["old_rule_acquired"] for row in selected)
        fresh_count = sum(row["fresh_acquired"] for row in selected)
        verdicts.append({"condition": condition, "n_seeds": len(selected), "mean_endpoint_cost": mean_cost,
                         "sd_endpoint_cost": float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.,
                         "positive_seed_count": positive, "high_prefix_acquisition_count": old_count,
                         "fresh_acquisition_count": fresh_count,
                         "criterion_met": bool(mean_cost >= criteria["endpoint_cost"] and
                                               positive / len(selected) >= criteria["seed_pass_fraction"] and
                                               old_count / len(selected) >= criteria["seed_pass_fraction"] and
                                               fresh_count / len(selected) >= criteria["seed_pass_fraction"])})
    comparisons = []
    for seed in cfg["seeds"]:
        for history in _HISTORIES:
            for trial in checkpoints:
                actual = measurements[seed, "linear_normalized", history, "phase2", trial, "iid"]["p_target"]
                expected = measurements[seed, "expected_linear", history, "phase2", trial, "iid"]["p_target"]
                comparisons.append({"seed": seed, "history": history, "trial": trial,
                                    "stochastic_p_target": actual, "reference_p_target": expected,
                                    "difference": actual - expected})
    independent = {"status": "completed", "scope": cfg["scope"], "outcomes": outcomes, "paired_contrasts": contrasts,
                   "family_verdicts": verdicts,
                   "primary_hypothesis": {"name": "H_B", **next(row for row in verdicts if row["condition"] == "linear_normalized")},
                   "reference_comparisons": comparisons,
                   "actual_training_trials": len(cfg["seeds"]) * len(cfg["conditions"]) * 7 * n,
                   "expected_reference_updates": len(cfg["seeds"]) * 7 * n,
                   "interpretation": "endpoint revision cost under fixed exposure; starting point, gain and AUC separated; expected-action path is not an ensemble mean or neural mechanism"}
    _equal(summary, independent, "Every independently reconstructed summary claim")


def _audit(output, report):
    metadata, registration, summary = (_json(output / name) for name in ("metadata.json", "registration.json", "summary.json"))
    cfg = metadata["config"]
    _require(metadata.get("status") == summary.get("status") == "completed", "Study completion")
    _require(cfg == registration["config"] and summary["scope"] == cfg["scope"], "Config/scope consistency")
    _require(_hash(output / "registration.json") == metadata["registration_sha256"], "Registration byte hash")
    chronology = [datetime.fromisoformat(record[field]) for record, field in
                  ((registration, "registered_at_utc"), (metadata, "started_at_utc"), (metadata, "completed_at_utc"))]
    _require(all(value.tzinfo is not None for value in chronology) and chronology == sorted(chronology),
             "Local registration/run chronology")
    _config_check(cfg)
    _require(metadata["rng"] == {"phase1_actions": "seed+280000", "phase2_actions": "seed+300000",
                                 "encoder": "seed+70000", "rewired": "seed+50000", "ann": "seed+60000",
                                 "data": "seed*100+31 phase1,+34 phase2,+32 IID,+33 OOD"}, "Registered RNG offsets")
    _require(metadata["reference_scope"] == "label-privileged conditional expected-action drift; no rewards sampled; not ensemble expectation",
             "Distinct analytic reference interpretation")
    root = _source_freeze(output, metadata, registration, cfg)
    source_graph, n_kc, n_mbon, source_metadata = _source_projection(root, cfg["graph"])
    _require(metadata["graph"] == source_metadata, "Independently parsed source graph metadata")
    seeds, conditions = cfg["seeds"], (*cfg["conditions"], "expected_linear")
    n, every, ev, probe_n = (cfg[key] for key in ("trials_per_phase", "checkpoint_every", "eval_trials", "probe_rows"))
    checkpoints = list(range(0, n + 1, every))
    required = {"metadata.json", "registration.json", "summary.json", "controls.json", "model_resources.json",
                "boundaries.json", "learning_curves.csv"}
    for seed in seeds:
        base = f"seed_{seed}"
        required.update(f"{base}/{name}" for name in ("stimuli.npz", "trials.csv", "expected_updates.csv",
                                                     "real_topology.npz", "rewired_topology.npz", "ann_topology.npz"))
        for condition in conditions:
            for history in _HISTORIES:
                prefix = f"{base}/{condition}/{history}"
                required.update(f"{prefix}/{name}" for name in ("boundary.json", "boundary_before.npz", "boundary_after.npz"))
                required.update(f"{prefix}/phase1_{trial:04d}.npz" for trial in ([0] if history == "fresh_low" else checkpoints))
                required.update(f"{prefix}/phase2_{trial:04d}.npz" for trial in checkpoints)
    hashes = _json(output / "artifact_hashes.json")
    _require(set(hashes) == required, "Exact required primary artifact hash coverage")
    for relative, digest in hashes.items():
        path = (output / relative).resolve()
        _require(path.is_relative_to(output) and path.is_file() and _hash(path) == digest,
                 f"Safe primary artifact byte hash: {relative}")
    curves = _csv(output / "learning_curves.csv", _CURVE_FIELDS,
                  ("seed", "condition", "history", "phase", "trial", "split"))
    expected_curves = {(seed, condition, history, phase, trial, split) for seed in seeds for condition in conditions
                       for history in _HISTORIES for phase in (("phase2",) if history == "fresh_low" else ("phase1", "phase2"))
                       for trial in checkpoints
                       for split in ("iid", "ood")}
    _require(set(curves) == expected_curves, "Exact all-checkpoint evaluation coverage")
    controls_list = _json(output / "controls.json")
    controls = {(row["seed"], row["kind"]): row for row in controls_list}
    _require(len(controls_list) == len(controls) and set(controls) == {(seed, kind) for seed in seeds
             for kind in ("real", "rewired", "ann")}, "Topology control coverage")
    resources_list = _json(output / "model_resources.json")
    resources = {(row["seed"], row["condition"], row["history"]): row for row in resources_list}
    expected_keys = {(seed, condition, history) for seed in seeds for condition in conditions for history in _HISTORIES}
    _require(len(resources_list) == len(resources) and set(resources) == expected_keys, "Exact resource metadata coverage")
    boundary_list = _json(output / "boundaries.json")
    boundaries = {(row["seed"], row["condition"], row["history"]): row for row in boundary_list}
    _require(len(boundary_list) == len(boundaries) and set(boundaries) == expected_keys, "Exact boundary event coverage")
    fingerprints, measurements = set(), {}
    actual_count = expected_count = snapshot_count = boundary_snapshot_count = probability_entries = 0
    split_specs = (("train_phase1", n, 31), ("train_phase2", n, 34), ("iid", ev, 32), ("ood", ev, 33))
    for seed in seeds:
        base = output / f"seed_{seed}"
        stimuli = _npz(base / "stimuli.npz", {f"{name}_{field}" for name, _, _ in split_specs
                       for field in ("counts", "nuisance", "family_ids")}
                       | {"action_phase1_uniforms", "action_phase2_uniforms"})
        datasets = {}
        for name, size, offset in split_specs:
            datasets[name] = _dataset(stimuli, name, size, (2, 4), fingerprints)
            for field, value in _regenerate_dataset(seed * 100 + offset, name, size, (2, 4)).items():
                _require(np.array_equal(datasets[name][field], value), f"Registered raw data RNG: {name}/{field}")
        uniforms = {phase: np.random.default_rng(seed + offset).random(n)
                    for phase, offset in (("phase1", 280000), ("phase2", 300000))}
        for phase, values in uniforms.items():
            _require(np.array_equal(stimuli[f"action_{phase}_uniforms"], values), "Shared registered phase action stream")
        trials = _csv(base / "trials.csv", _TRIAL_FIELDS, ("seed", "condition", "history", "phase", "trial"))
        expected = _csv(base / "expected_updates.csv", _EXPECTED_FIELDS, ("seed", "history", "phase", "trial"))
        design = {(history, phase, trial) for history in _HISTORIES
                  for phase in (("phase2",) if history == "fresh_low" else ("phase1", "phase2"))
                  for trial in range(1, n + 1)}
        _require(set(trials) == {(seed, condition, *entry) for condition in cfg["conditions"] for entry in design}
                 and set(expected) == {(seed, *entry) for entry in design}, "Actual/reference update coverage and separation")
        actual_count += len(trials); expected_count += len(expected)
        graphs = _topologies(base, {kind: controls[seed, kind] for kind in ("real", "rewired", "ann")},
                             seed, n_kc, n_mbon, source_metadata)
        _require(all(np.array_equal(graphs["real"][field], source_graph[field]) for field in source_graph),
                 "Stored real topology equals separately parsed source")
        rng = np.random.default_rng(seed + 70000)
        network_encoder, network_bias = rng.normal(0, .8, (n_kc, 7)), rng.uniform(-.25, .25, n_kc)
        phase1_inputs = _inputs(datasets["train_phase1"], np.arange(1, 9))
        for condition in conditions:
            kind, plastic = ("linear", False) if condition == "expected_linear" else _FLAGS[condition]
            linear = kind == "linear"
            if linear:
                encoder, bias, pre, post = np.empty((0, 7)), np.empty(0), np.empty(0, dtype=int), np.empty(0, dtype=int)
                w0, initial_head = np.empty(0), np.zeros(7)
            else:
                encoder, bias = network_encoder, network_bias
                graph = graphs[kind]
                pre, post = graph["pre"], graph["post"]
                denominator = np.maximum(np.bincount(post, weights=graph["counts"], minlength=n_mbon), 1)
                w0, initial_head = graph["counts"] / denominator[post], np.zeros(n_mbon)
            _, _, initial_train_features = _raw_features(phase1_inputs, encoder, bias, pre, post, w0, initial_head.size, linear)
            gaps = initial_train_features[:, 1] - initial_train_features[:, 0]
            scale = float(np.sqrt(np.mean(np.sum(gaps ** 2, axis=-1))))
            _require(np.isfinite(scale) and scale > 0, "Positive initial phase1 TRAIN-only feature-gap RMS")
            history_replays = {}
            for history in _HISTORIES:
                key = (seed, condition, history)
                expected_resource = {"seed": seed, "condition": condition, "history": history,
                                     "feature_scale": scale, "initial_feature_gap_rms": scale,
                                     "readout_parameters": initial_head.size, "internal_parameters": len(w0) if plastic else 0,
                                     "algorithm": "conditional_expected_action_reference" if condition == "expected_linear"
                                                  else "stochastic_reward"}
                _equal(resources[key], expected_resource, f"Every train-only resource entry: {key}")
                path = base / condition / history
                replay = _replay_history(path, seed, condition, history, cfg, datasets, uniforms, encoder, bias,
                                         pre, post, w0, initial_head, scale, linear, plastic,
                                         expected if condition == "expected_linear" else trials, curves,
                                         expected_reference=condition == "expected_linear")
                history_replays[history] = replay
                for checkpoint_key, value in replay["measurements"].items():
                    measurements[*key, *checkpoint_key] = value
                pretrain = 0 if history == "fresh_low" else n
                expected_boundary = {"seed": seed, "condition": condition, "history": history,
                                     "reset_head": history == "high_to_low_reset_head",
                                     "head_change_l2": replay["head_reset_l2"], "internal_change_l2": 0.,
                                     "feature_scale_before": scale, "feature_scale_after": scale,
                                     "head_updates_before": pretrain, "head_updates_after": pretrain,
                                     "internal_updates_before": pretrain if plastic else 0,
                                     "internal_updates_after": pretrain if plastic else 0}
                _equal(_json(path / "boundary.json"), expected_boundary, f"Every local intervention: {key}")
                _equal(boundaries[key], expected_boundary, f"Every aggregate intervention: {key}")
                snapshot_count += len(replay["snapshots"]); boundary_snapshot_count += 2
                probability_entries += len(replay["snapshots"]) * 2 * ev
            for trial in checkpoints:
                high = history_replays["high_to_low"]["snapshots"]["phase1", trial]
                reset = history_replays["high_to_low_reset_head"]["snapshots"]["phase1", trial]
                _require(all(np.array_equal(high[field], reset[field]) for field in _CHECKPOINT_FIELDS),
                         "Identical high-rule pretraining state prefix")
                if not plastic:
                    fresh = history_replays["fresh_low"]["snapshots"]["phase2", trial]
                    reset = history_replays["high_to_low_reset_head"]["snapshots"]["phase2", trial]
                    _require(all(np.array_equal(fresh[field], reset[field]) for field in _CHECKPOINT_FIELDS
                                 - {"head_updates", "internal_updates"}), "Exact fixed-state reset/fresh counterfactual")
            records = expected if condition == "expected_linear" else trials
            fields = _EXPECTED_FIELDS if condition == "expected_linear" else _TRIAL_FIELDS
            for trial in range(1, n + 1):
                def row(history, phase):
                    key = (seed, history, phase, trial) if condition == "expected_linear" else (seed, condition, history, phase, trial)
                    return records[key]
                high, reset = row("high_to_low", "phase1"), row("high_to_low_reset_head", "phase1")
                _require(all(high[field] == reset[field] for field in fields - {"history"}), "Exact high-rule training prefix")
                if not plastic:
                    fresh, reset = row("fresh_low", "phase2"), row("high_to_low_reset_head", "phase2")
                    _require(all(fresh[field] == reset[field] for field in fields - {"history"}), "Exact fixed-state reset/fresh update path")
        print(f"audited seed {seed} ({seeds.index(seed) + 1}/{len(seeds)})", flush=True)
    _summary(cfg, measurements, summary)
    report["counts"].update(artifact_hashes=len(hashes), source_hashes=len(registration["source_hashes"]),
                            seeds=len(seeds), reward_conditions=len(cfg["conditions"]), histories=len(_HISTORIES),
                            actual_trajectories=len(seeds) * len(cfg["conditions"]) * len(_HISTORIES),
                            analytic_reference_trajectories=len(seeds) * len(_HISTORIES),
                            actual_training_trials=actual_count, expected_reference_updates=expected_count,
                            curve_rows=len(curves), checkpoints=snapshot_count, boundary_state_files=boundary_snapshot_count,
                            case_probability_entries=probability_entries, canonical_fingerprints=len(fingerprints),
                            boundary_events=len(boundaries), summary_outcomes=len(summary["outcomes"]),
                            reference_comparison_rows=len(summary["reference_comparisons"]))
    report["checks"].extend([
        "Exact required artifact hashes, config/source byte freeze, source file set and local registration/run chronology",
        "Four independently regenerated raw splits per seed, full joint nuisance crossing and canonical disjointness",
        "Independent source graph parsing, anatomical/rewired/ANN sparse topology and RNG reconstruction",
        "Phase1 initial TRAIN-only feature-gap RMS fixed across all histories and every saved checkpoint/boundary",
        "Every actual action, reward, trial probability, old-head internal gradient and clipping independently replayed",
        "Expected-action reference independently enumerates both action outcomes and preserves separate update counts",
        "Every checkpoint parameter, integer counter, KC/MBON/decision probe and full IID/OOD case probability reconstructed",
        "Boundary raw states and events verify readout-only reset, internal continuity, fixed scale and unchanged counters",
        "Exact high-pretraining prefixes and nonplastic reset/fresh counterfactual states/trials verified",
        "Every acquisition/qualified adaptation gate, endpoint/gain/AUC/baseline-adjusted contrast, H_B and reference comparison reconstructed",
    ])


def verify(output: Path) -> dict:
    """Write an independent report without modifying primary artifacts."""
    output = Path(output).resolve()
    report = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "method": "independent numpy/stdlib arithmetic replay; only prior independent audit helpers reused",
              "input_scope": "reward reversal on artificial scalar order; frozen initial TRAIN feature-gap scale",
              "counts": {}, "checks": [], "errors": [], "not_verified": [
                  "Visual extraction of numerosity, LC11/DAN dynamics, or shared human/fly neural mechanisms",
                  "Remote private Git publication authenticity; local chronology is self-recorded",
                  "Statistical significance, intrinsic learning rate, or educational implications",
                  "Expected-action deterministic reference as an actual reward-only learner or expected stochastic trajectory",
                  "Equal feature-gap scale or trial budgets as complete equivalence of capacity or optimization",
                  "Computational seeds as independent biological individuals"]}
    try:
        _require(output.is_dir(), f"Output directory missing: {output}")
        _audit(output, report)
        report["passed"] = True
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
