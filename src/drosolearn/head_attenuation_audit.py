"""Independent replay of shared-prefix readout attenuation counterfactuals.

Only previous independent auditors and NumPy/standard libraries are reused.
No learner, producer, data generator, metric or summary implementation is
imported. The expected-action reference enumerates both actions locally.
"""
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np

from .encoding_audit import (
    _require, _near, _integer, _number, _json, _hash, _npz,
    _regenerate_dataset, _dataset, _inputs, _topologies,
)
from .linearity_audit import _source_projection
from .bandit_reversal_audit import (
    _raw_features, _state, _snapshot_check, _metrics, _expected_factor,
    _auc, _equal, _csv,
)


_FLAGS = {"linear_normalized": ("linear", False), "fixed": ("real", False),
          "plastic": ("real", True), "rewired_fixed": ("rewired", False),
          "rewired_plastic": ("rewired", True), "ann_fixed": ("ann", False),
          "ann_plastic": ("ann", True)}
_HISTORIES = ("fresh_low", "gamma_0", "gamma_025", "gamma_05", "gamma_1")
_GAMMAS = {"gamma_0": 0., "gamma_025": .25, "gamma_05": .5, "gamma_1": 1.}
_SAVED_FIELDS = {"internal_weights", "readout_weights", "kc_activity", "mbon_activity",
                 "choice_probability", "encoder", "bias", "pre", "post", "decision_activity",
                 "feature_scale", "initial_feature_gap_rms", "head_updates", "internal_updates",
                 "initial_internal_weights", "internal_upper_bounds", "incoming_normalizer",
                 "iid_p_right", "ood_p_right", "iid_raw_margin", "ood_raw_margin"}
_CHECKPOINT_FIELDS = _SAVED_FIELDS
_TRIAL_FIELDS = {"seed", "condition", "history", "phase", "trial", "family_id", "left_count", "right_count",
                 "action", "reward", "p_right", "optimizer_factor", "internal_step_l2", "readout_step_l2",
                 "kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2", "decision_left_l2", "decision_right_l2"}
_CURVE_FIELDS = {"seed", "condition", "history", "phase", "trial", "split", "p_larger", "p_smaller",
                 "p_target", "greedy_accuracy", "mean_policy_sensitivity", "internal_change_l2",
                 "readout_change_l2", "probe_kc_change_l2", "probe_mbon_change_l2", "lower_bound_fraction",
                 "upper_bound_fraction", "probe_decision_change_l2", "larger"}
_EXPECTED_FIELDS = {"seed", "history", "phase", "trial", "family_id", "left_count", "right_count",
                    "target", "p_right", "expected_factor", "readout_step_l2"}


def _config_check(cfg):
    _require(cfg["scope"] in {"preregistered_diagnostic", "integration_test"}, "Supported study scope")
    seeds = cfg["seeds"]
    _require(seeds and len(seeds) == len(set(seeds)) and
             all(type(seed) is int and seed >= 0 for seed in seeds), "Distinct nonnegative integer seeds")
    _require(tuple(cfg["conditions"]) == tuple(_FLAGS) and tuple(cfg["histories"]) == _HISTORIES,
             "Exact seven learners and five matched continuations")
    fixed = {"experiment_id": "head_attenuation_diagnostic_v1",
             "graph": "data/flywire_783_mb_left_kc512_s701", "pair": [2, 4],
             "head_initialization": "zeros_all_conditions",
             "feature_normalization": "fixed_initial_phase1_train_feature_gap_rms",
             "expected_reference": "conditional_expected_action_linear",
             "attenuation_mode": "scale_readout_only_preserve_all_other_state", "primary_gamma": .25}
    _require(all(cfg.get(key) == value for key, value in fixed.items()), "Fixed attenuation design")
    _require(cfg["gammas"] == [0., .25, .5, 1.] and
             all(type(value) in (int, float) and np.isfinite(value) for value in cfg["gammas"])
             and type(cfg["primary_gamma"]) in (int, float) and np.isfinite(cfg["primary_gamma"]),
             "Four prospective gamma values and a numeric primary gamma")
    budget = tuple(cfg[key] for key in ("trials_per_phase", "checkpoint_every", "eval_trials", "probe_rows"))
    n, every, ev, probe = budget
    _require(all(type(value) is int and value > 0 for value in budget) and
             n % 4 == ev % 4 == n % every == 0 and probe <= ev, "Crossing/checkpoint/probe budgets")
    _require(all(type(cfg[key]) in (int, float) and np.isfinite(cfg[key]) and cfg[key] >= 0
                 for key in ("head_lr", "internal_lr")), "Finite nonnegative learning rates")
    criteria = cfg["criteria"]
    _require(set(criteria) == {"acquisition_accuracy", "acquisition_gain", "adaptation_accuracy",
                              "endpoint_benefit", "seed_pass_fraction"} and
             all(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1
                 for value in criteria.values()) and criteria["seed_pass_fraction"] > 0,
             "Fixed criterion schema and ranges")
    if cfg["scope"] == "integration_test":
        _require(not any(seed in range(8001, 8011) for seed in seeds), "Scientific seeds cannot alias fixtures")
    else:
        _require(seeds == list(range(8001, 8011)), "Ten prospective scientific seeds")
        _require((*budget, cfg["head_lr"], cfg["internal_lr"]) == (400, 100, 128, 8, .1, .01),
                 "Fixed registered resources")
        _require(criteria == {"acquisition_accuracy": .75, "acquisition_gain": .15,
                              "adaptation_accuracy": .75, "endpoint_benefit": .15, "seed_pass_fraction": .8},
                 "Fixed registered H_A criteria")


def _raw_margin(inputs, encoder, bias, pre, post, weights, head, scale, linear):
    """Use unclipped candidate scores, preserving information lost in sigmoid."""
    _, _, raw = _raw_features(inputs, encoder, bias, pre, post, weights, head.size, linear)
    scores = (raw / scale) @ head
    return scores[..., 1] - scores[..., 0]


def _raw_decisions(margin):
    return np.where(margin == 0, .5, (margin > 0).astype(float))


def _probability_decisions(probability):
    return np.where(np.abs(probability - .5) < 1e-12, .5, (probability > .5).astype(float))


def _rank_metrics(margin, probability, counts, larger):
    high = (counts[:, 1] > counts[:, 0]).astype(int)
    target = high if larger else 1 - high
    raw = _raw_decisions(margin)
    greedy = _probability_decisions(probability)
    correct = np.where(raw == .5, .5, (raw == target).astype(float))
    return {"raw_margin_greedy_accuracy": float(correct.mean()),
            "raw_margin_zero_fraction": float(np.mean(margin == 0)),
            "probability_tie_fraction": float(np.mean(greedy == .5)),
            "raw_probability_greedy_disagreement": float(np.mean(raw != greedy))}


def _update(stimulus, row, data, index, phase, cfg, uniform, encoder, bias,
            pre, post, weights, head, w0, scale, linear, plastic, reference, larger):
    kc, mb, raw = _raw_features(stimulus, encoder, bias, pre, post, weights, head.size, linear)
    decision = raw / scale
    scores = decision @ head
    probability = float(1 / (1 + np.exp(-np.clip(scores[1] - scores[0], -40, 40))))
    _near(_number(row, "p_right"), probability, "Every pre-update trial probability")
    _require(row["family_id"] == str(data["family_ids"][index]) and
             _integer(row, "left_count") == data["counts"][index, 0] and
             _integer(row, "right_count") == data["counts"][index, 1], "Shared stimulus identity")
    target = int(data["counts"][index, 1] > data["counts"][index, 0])
    if not larger:
        target = 1 - target
    if reference:
        _require(_integer(row, "target") == target, "Reference privileged target rule")
        factor = _expected_factor(probability, target)
        _near(_number(row, "expected_factor"), factor, "Enumerated two-action conditional reward drift")
    else:
        action, reward = _integer(row, "action"), _integer(row, "reward")
        _require(action in (0, 1) and reward in (0, 1), "Binary chosen action and reward")
        _require(action == int(uniform[index] < probability), "Every shared action-stream choice")
        _require(reward == int(action == target), "Phase-contingent reward rule")
        factor = (action - probability) * (reward - .5)
        _near(_number(row, "optimizer_factor"), factor, "Every reward-only optimizer factor")
    next_head = head + cfg["head_lr"] * factor * (decision[1] - decision[0])
    if plastic:
        eligibility = head[post] * (kc[1, pre] - kc[0, pre]) / scale
        next_weights = np.minimum(np.maximum(weights + cfg["internal_lr"] * factor * eligibility, 0), 2 * w0)
    else:
        next_weights = weights.copy()
    if reference:
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
    return next_weights, next_head


def _source_freeze(output, metadata, registration, cfg):
    root = (output / metadata["source_root"]).resolve()
    _require(root.is_dir(), "Registered source root missing; use the registered checkout")
    config = Path(metadata["config_path"])
    config = (root / config).resolve() if not config.is_absolute() else config.resolve()
    _require(config.is_file() and _hash(config) == registration["config_sha256"] and _json(config) == cfg,
             "Registered config bytes/content")
    expected = {path.relative_to(root).as_posix() for path in (root / "src").rglob("*.py") if path.is_file()}
    expected.update(path.relative_to(root).as_posix() for path in (root / cfg["graph"]).glob("*") if path.is_file())
    expected.update(("docs/head-attenuation-protocol.md", "docs/head-attenuation-background.md",
                     "pyproject.toml", "requirements-lock.txt", "data/source-manifest.json", "data/LICENSE.md"))
    _require(set(registration["source_hashes"]) == expected, "Registered source file set")
    for relative, digest in registration["source_hashes"].items():
        source = (root / relative).resolve()
        _require(source.is_relative_to(root) and source.is_file() and _hash(source) == digest,
                 f"Registered source byte hash: {relative}")
    return root


def _full_snapshot_check(saved, cfg, datasets, encoder, bias, pre, post, w0,
                         normalizer, weights, head, scale, linear, head_updates,
                         internal_updates, label):
    probe = _inputs(datasets["iid"], np.arange(1, 9))[:cfg["probe_rows"]]
    _snapshot_check(saved, probe, encoder, bias, pre, post, weights, head, scale,
                    linear, head_updates, internal_updates, label)
    for field, value in (("initial_internal_weights", w0), ("internal_upper_bounds", 2 * w0),
                         ("incoming_normalizer", normalizer)):
        _require(np.array_equal(saved[field], value), f"Frozen initial resource: {label}/{field}")
    for split in ("iid", "ood"):
        inputs = _inputs(datasets[split], np.arange(1, 9))
        _, _, _, probability = _state(inputs, encoder, bias, pre, post, weights, head, scale, linear)
        margin = _raw_margin(inputs, encoder, bias, pre, post, weights, head, scale, linear)
        _near(saved[f"{split}_p_right"], probability, f"Every frozen case probability: {label}/{split}")
        _near(saved[f"{split}_raw_margin"], margin, f"Every unclipped score margin: {label}/{split}")


def _boundary_checks(before, after, gamma, label):
    changed = {"readout_weights", "choice_probability", "iid_p_right", "ood_p_right",
               "iid_raw_margin", "ood_raw_margin"}
    _require(all(np.array_equal(before[field], after[field]) for field in _SAVED_FIELDS - changed),
             f"Readout-only intervention preserves all other state: {label}")
    multiplier = 1. if gamma is None else gamma
    _require(np.array_equal(after["readout_weights"], multiplier * before["readout_weights"]),
             f"Exact scaled readout clone: {label}")
    for split in ("iid", "ood"):
        original, altered = (saved[f"{split}_raw_margin"] for saved in (before, after))
        _near(altered, multiplier * original, f"Unclipped margin scales linearly: {label}/{split}")
        pre_p, post_p = (saved[f"{split}_p_right"] for saved in (before, after))
        _require(np.all(np.maximum(post_p, 1 - post_p) <= np.maximum(pre_p, 1 - pre_p) + 1e-12),
                 f"Confidence is casewise nonincreasing: {label}/{split}")
        if gamma == 0:
            _require(np.all(altered == 0) and np.all(post_p == .5), f"Zero head exact uniform policy: {label}")
        else:
            _require(np.array_equal(np.sign(original), np.sign(altered)),
                     f"Positive scaling preserves exact raw sign/rank: {label}/{split}")


def _replay_phase(path, seed, condition, history, phase, larger, cfg, datasets,
                  uniform, encoder, bias, pre, post, w0, normalizer, head0, scale, linear,
                  plastic, records, curves, weights, head, ancestor, reference):
    """Replay a physically executed phase once, then audit all frozen cases."""
    n, every, probe_n = (cfg[key] for key in ("trials_per_phase", "checkpoint_every", "probe_rows"))
    probe = _inputs(datasets["iid"], np.arange(1, 9))[:probe_n]
    initial_kc, initial_mb, initial_decision, _ = _state(
        probe, encoder, bias, pre, post, w0, head0, scale, linear)
    weights, head = weights.copy(), head.copy()
    snapshots, measurements = {}, {}

    def checkpoint(trial):
        saved = _npz(path / f"{phase}_{trial:04d}.npz", _CHECKPOINT_FIELDS)
        snapshots[trial] = saved
        _full_snapshot_check(saved, cfg, datasets, encoder, bias, pre, post, w0, normalizer,
                             weights, head, scale, linear, ancestor + trial,
                             (ancestor + trial) if plastic else 0, f"{condition}/{history}/{phase}/{trial}")
        changes = {"internal_change_l2": float(np.linalg.norm(weights - w0)),
                   "readout_change_l2": float(np.linalg.norm(head - head0)),
                   "probe_kc_change_l2": float(np.linalg.norm(saved["kc_activity"] - initial_kc)),
                   "probe_mbon_change_l2": float(np.linalg.norm(saved["mbon_activity"] - initial_mb)),
                   "probe_decision_change_l2": float(np.linalg.norm(saved["decision_activity"] - initial_decision))}
        for split in ("iid", "ood"):
            data = datasets[split]
            inputs = _inputs(data, np.arange(1, 9))
            _, _, _, probability = _state(inputs, encoder, bias, pre, post, weights, head, scale, linear)
            margin = _raw_margin(inputs, encoder, bias, pre, post, weights, head, scale, linear)
            _near(saved[f"{split}_p_right"], probability, f"Full checkpoint probability: {split}")
            _near(saved[f"{split}_raw_margin"], margin, f"Full unclipped checkpoint margin: {split}")
            metrics, _, _ = _metrics(probability, data["counts"], larger)
            measurements[trial, split] = metrics
            row = curves[seed, condition, history, phase, trial, split]
            _require(_integer(row, "larger") == int(larger), "Curve reward direction")
            bounds = {"lower_bound_fraction": float(np.mean(weights <= 0)) if weights.size else 0.,
                      "upper_bound_fraction": float(np.mean(weights >= 2 * w0)) if weights.size else 0.}
            for field, value in {**metrics, **changes, **bounds}.items():
                _near(_number(row, field), value,
                      f"Every independent curve: {condition}/{history}/{phase}/{trial}/{split}/{field}")

    checkpoint(0)
    data = datasets[f"train_{phase}"]
    for index, stimulus in enumerate(_inputs(data, np.arange(1, 9))):
        key = (seed, history, phase, index + 1) if reference else (seed, condition, history, phase, index + 1)
        weights, head = _update(stimulus, records[key], data, index, phase, cfg, uniform,
                                encoder, bias, pre, post, weights, head, w0, scale, linear,
                                plastic, reference, larger)
        if (index + 1) % every == 0:
            checkpoint(index + 1)
    return {"weights": weights, "head": head, "head_updates": ancestor + n,
            "internal_updates": (ancestor + n) if plastic else 0,
            "snapshots": snapshots, "measurements": measurements}


def _summary(cfg, measurements, summary):
    n = cfg["trials_per_phase"]
    checkpoints = list(range(0, n + 1, cfg["checkpoint_every"]))
    criteria = cfg["criteria"]
    outcomes, prefixes, contrasts, verdicts = [], [], [], []
    for seed in cfg["seeds"]:
        for condition in (*cfg["conditions"], "expected_linear"):
            reference = condition == "expected_linear"
            before, after = (measurements[seed, condition, "shared_prefix", "phase1", trial, "iid"]["p_target"]
                             for trial in (0, n))
            acquired = None if reference else bool(after >= criteria["acquisition_accuracy"] and
                                                  after - before >= criteria["acquisition_gain"])
            prefixes.append({"seed": seed, "condition": condition, "before": before,
                             "after": after, "gain": after - before, "acquired": acquired,
                             "physical_training_updates": n})
            for history in _HISTORIES:
                fresh = history == "fresh_low"
                sequence = [measurements[seed, condition, history, "phase2", trial, "iid"]["p_target"]
                            for trial in checkpoints]
                fresh_gate = bool(sequence[-1] >= criteria["acquisition_accuracy"] and
                                  sequence[-1] - sequence[0] >= criteria["acquisition_gain"])
                passed = bool(sequence[-1] >= criteria["adaptation_accuracy"])
                auc = _auc(sequence, checkpoints)
                count = n if fresh else 2 * n
                plastic = condition in _FLAGS and _FLAGS[condition][1]
                outcomes.append({"seed": seed, "condition": condition, "history": history,
                                 "gamma": _GAMMAS.get(history),
                                 "phase1_before": None if fresh else before,
                                 "phase1_after": None if fresh else after,
                                 "phase1_acquired": None if fresh else acquired,
                                 "phase2_before": sequence[0], "phase2_after": sequence[-1],
                                 "phase2_gain": sequence[-1] - sequence[0], "phase2_auc": auc,
                                 "phase2_baseline_adjusted_auc": auc - sequence[0],
                                 "frozen_no_learning_endpoint": sequence[0], "frozen_no_learning_auc": sequence[0],
                                 "phase2_target_passed": None if reference else passed,
                                 "qualified_adaptation": None if reference else bool(fresh_gate if fresh else acquired and passed),
                                 "fresh_acquired": fresh_gate if fresh and not reference else None,
                                 "first_observed_75_checkpoint": None if reference else next(
                                     (trial for trial, value in zip(checkpoints, sequence)
                                      if value >= criteria["adaptation_accuracy"]), None),
                                 "phase2_ood_after": measurements[seed, condition, history, "phase2", n, "ood"]["p_target"],
                                 "phase2_greedy_after": measurements[seed, condition, history, "phase2", n, "iid"]["greedy_accuracy"],
                                 "head_updates": count, "internal_updates": count if plastic else 0,
                                 "physical_phase2_updates": n,
                                 "algorithm": "conditional_expected_action_reference" if reference else "stochastic_reward"})
    rows = {(row["seed"], row["condition"], row["history"]): row for row in outcomes}
    for condition in cfg["conditions"]:
        selected = []
        for seed in cfg["seeds"]:
            partial, full, fresh, zero, half = (rows[seed, condition, history]
                                              for history in ("gamma_025", "gamma_1", "fresh_low", "gamma_0", "gamma_05"))
            contrast = {"seed": seed, "condition": condition,
                        "endpoint_benefit": partial["phase2_after"] - full["phase2_after"],
                        "initial_probability_difference": partial["phase2_before"] - full["phase2_before"],
                        "gain_difference": partial["phase2_gain"] - full["phase2_gain"],
                        "raw_auc_difference": partial["phase2_auc"] - full["phase2_auc"],
                        "baseline_adjusted_auc_difference": partial["phase2_baseline_adjusted_auc"] - full["phase2_baseline_adjusted_auc"],
                        "reset_minus_full_endpoint": zero["phase2_after"] - full["phase2_after"],
                        "reset_minus_fresh_endpoint": zero["phase2_after"] - fresh["phase2_after"],
                        "half_minus_full_endpoint": half["phase2_after"] - full["phase2_after"],
                        "old_rule_acquired": full["phase1_acquired"], "fresh_acquired": fresh["fresh_acquired"]}
            contrasts.append(contrast)
            selected.append(contrast)
        benefits = [row["endpoint_benefit"] for row in selected]
        mean = float(np.mean(benefits))
        positive = sum(value > 0 for value in benefits)
        old_count = sum(row["old_rule_acquired"] for row in selected)
        verdicts.append({"condition": condition, "n_seeds": len(selected), "mean_endpoint_benefit": mean,
                         "sd_endpoint_benefit": float(np.std(benefits, ddof=1)) if len(benefits) > 1 else 0.,
                         "positive_seed_count": positive, "high_prefix_acquisition_count": old_count,
                         "fresh_acquisition_count": sum(row["fresh_acquired"] for row in selected),
                         "criterion_met": bool(mean >= criteria["endpoint_benefit"] and
                                               positive / len(selected) >= criteria["seed_pass_fraction"] and
                                               old_count / len(selected) >= criteria["seed_pass_fraction"])})
    comparisons = []
    for seed in cfg["seeds"]:
        for history in _HISTORIES:
            for trial in checkpoints:
                actual = measurements[seed, "linear_normalized", history, "phase2", trial, "iid"]["p_target"]
                expected = measurements[seed, "expected_linear", history, "phase2", trial, "iid"]["p_target"]
                comparisons.append({"seed": seed, "history": history, "trial": trial,
                                    "stochastic_p_target": actual, "reference_p_target": expected,
                                    "difference": actual - expected})
    reconstructed = {"status": "completed", "scope": cfg["scope"], "outcomes": outcomes,
                     "prefix_outcomes": prefixes, "paired_contrasts": contrasts, "family_verdicts": verdicts,
                     "primary_hypothesis": {"name": "H_A", **next(row for row in verdicts if row["condition"] == "linear_normalized")},
                     "reference_comparisons": comparisons,
                     "actual_training_trials": len(cfg["seeds"]) * len(cfg["conditions"]) * 6 * n,
                     "expected_reference_updates": len(cfg["seeds"]) * 6 * n,
                     "interpretation": "readout magnitude intervention; endpoint mixes immediate baseline change and subsequent path; plastic eligibility also scales; reference is not an ensemble mean"}
    _equal(summary, reconstructed, "Every independently reconstructed summary claim")


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
             "Distinct reference interpretation")
    root = _source_freeze(output, metadata, registration, cfg)
    source_graph, n_kc, n_mbon, source_metadata = _source_projection(root, cfg["graph"])
    _require(metadata["graph"] == source_metadata, "Independently parsed source graph metadata")
    seeds, conditions = cfg["seeds"], (*cfg["conditions"], "expected_linear")
    n, every, ev = (cfg[key] for key in ("trials_per_phase", "checkpoint_every", "eval_trials"))
    checkpoints = list(range(0, n + 1, every))
    required = {"metadata.json", "registration.json", "summary.json", "controls.json", "model_resources.json",
                "boundaries.json", "learning_curves.csv"}
    for seed in seeds:
        base = f"seed_{seed}"
        required.update(f"{base}/{name}" for name in ("stimuli.npz", "trials.csv", "expected_updates.csv",
                                                     "real_topology.npz", "rewired_topology.npz", "ann_topology.npz"))
        for condition in conditions:
            required.update(f"{base}/{condition}/shared_prefix/phase1_{trial:04d}.npz" for trial in checkpoints)
            for history in _HISTORIES:
                path = f"{base}/{condition}/{history}"
                required.update(f"{path}/{name}" for name in ("boundary.json", "boundary_before.npz", "boundary_after.npz"))
                required.update(f"{path}/phase2_{trial:04d}.npz" for trial in checkpoints)
    hashes = _json(output / "artifact_hashes.json")
    _require(set(hashes) == required, "Exact required primary artifact hash coverage")
    for relative, digest in hashes.items():
        path = (output / relative).resolve()
        _require(path.is_relative_to(output) and path.is_file() and _hash(path) == digest,
                 f"Safe primary artifact byte hash: {relative}")
    curves = _csv(output / "learning_curves.csv", _CURVE_FIELDS,
                  ("seed", "condition", "history", "phase", "trial", "split"))
    design_phases = [("shared_prefix", "phase1"), *((history, "phase2") for history in _HISTORIES)]
    expected_curves = {(seed, condition, history, phase, trial, split) for seed in seeds for condition in conditions
                       for history, phase in design_phases for trial in checkpoints for split in ("iid", "ood")}
    _require(set(curves) == expected_curves, "Exact shared-prefix and all-branch checkpoint coverage")
    controls_list = _json(output / "controls.json")
    controls = {(row["seed"], row["kind"]): row for row in controls_list}
    _require(len(controls_list) == len(controls) and set(controls) == {(seed, kind) for seed in seeds
             for kind in ("real", "rewired", "ann")}, "Topology control coverage")
    keys = {(seed, condition, history) for seed in seeds for condition in conditions for history in _HISTORIES}
    resources_list = _json(output / "model_resources.json")
    resources = {(row["seed"], row["condition"], row["history"]): row for row in resources_list}
    _require(len(resources_list) == len(resources) and set(resources) == keys, "Exact resource metadata coverage")
    boundary_list = _json(output / "boundaries.json")
    boundaries = {(row["seed"], row["condition"], row["history"]): row for row in boundary_list}
    _require(len(boundary_list) == len(boundaries) and set(boundaries) == keys, "Exact boundary coverage")
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
            _require(np.array_equal(stimuli[f"action_{phase}_uniforms"], values), "Shared registered action stream")
        trials = _csv(base / "trials.csv", _TRIAL_FIELDS, ("seed", "condition", "history", "phase", "trial"))
        expected = _csv(base / "expected_updates.csv", _EXPECTED_FIELDS, ("seed", "history", "phase", "trial"))
        design = {(history, phase, trial) for history, phase in design_phases for trial in range(1, n + 1)}
        _require(set(trials) == {(seed, condition, *entry) for condition in cfg["conditions"] for entry in design}
                 and set(expected) == {(seed, *entry) for entry in design}, "Physical actual/reference update coverage")
        actual_count += len(trials)
        expected_count += len(expected)
        graphs = _topologies(base, {kind: controls[seed, kind] for kind in ("real", "rewired", "ann")},
                             seed, n_kc, n_mbon, source_metadata)
        _require(all(np.array_equal(graphs["real"][field], source_graph[field]) for field in source_graph),
                 "Stored real topology equals separately parsed source")
        rng = np.random.default_rng(seed + 70000)
        network_encoder, network_bias = rng.normal(0, .8, (n_kc, 7)), rng.uniform(-.25, .25, n_kc)
        phase1_inputs = _inputs(datasets["train_phase1"], np.arange(1, 9))
        for condition in conditions:
            kind, plastic = ("linear", False) if condition == "expected_linear" else _FLAGS[condition]
            linear, reference = kind == "linear", condition == "expected_linear"
            if linear:
                encoder, bias = np.empty((0, 7)), np.empty(0)
                pre, post = np.empty(0, dtype=int), np.empty(0, dtype=int)
                w0, normalizer, head0 = np.empty(0), np.empty(0), np.zeros(7)
            else:
                encoder, bias = network_encoder, network_bias
                graph = graphs[kind]
                pre, post = graph["pre"], graph["post"]
                normalizer = np.maximum(np.bincount(post, weights=graph["counts"], minlength=n_mbon), 1)
                w0, head0 = graph["counts"] / normalizer[post], np.zeros(n_mbon)
            _, _, initial_train = _raw_features(phase1_inputs, encoder, bias, pre, post, w0, head0.size, linear)
            gaps = initial_train[:, 1] - initial_train[:, 0]
            scale = float(np.sqrt(np.mean(np.sum(gaps ** 2, axis=-1))))
            _require(np.isfinite(scale) and scale > 0, "Positive initial phase1 TRAIN-only feature-gap RMS")
            records = expected if reference else trials
            prefix_path = base / condition / "shared_prefix"
            prefix = _replay_phase(prefix_path, seed, condition, "shared_prefix", "phase1", True, cfg, datasets,
                                   uniforms["phase1"], encoder, bias, pre, post, w0, normalizer, head0, scale,
                                   linear, plastic, records, curves, w0, head0, 0, reference)
            replays = {"shared_prefix": prefix}
            for (trial, split), value in prefix["measurements"].items():
                measurements[seed, condition, "shared_prefix", "phase1", trial, split] = value
            snapshot_count += len(prefix["snapshots"])
            for history in _HISTORIES:
                gamma = _GAMMAS.get(history)
                ancestor = 0 if gamma is None else n
                before_weights = w0.copy() if gamma is None else prefix["weights"].copy()
                before_head = head0.copy() if gamma is None else prefix["head"].copy()
                after_head = before_head.copy() if gamma is None else gamma * before_head
                path = base / condition / history
                before = _npz(path / "boundary_before.npz", _SAVED_FIELDS)
                after = _npz(path / "boundary_after.npz", _SAVED_FIELDS)
                for label, saved, head in (("before", before, before_head), ("after", after, after_head)):
                    _full_snapshot_check(saved, cfg, datasets, encoder, bias, pre, post, w0, normalizer,
                                         before_weights, head, scale, linear, ancestor, ancestor if plastic else 0,
                                         f"{condition}/{history}/boundary_{label}")
                _boundary_checks(before, after, gamma, f"{condition}/{history}")
                for split in ("iid", "ood"):
                    pre_p, post_p = before[f"{split}_p_right"], after[f"{split}_p_right"]
                    pre_margin, post_margin = before[f"{split}_raw_margin"], after[f"{split}_raw_margin"]
                    pre_conf, post_conf = np.maximum(pre_p, 1 - pre_p), np.maximum(post_p, 1 - post_p)
                    diagnostics = {"seed": seed, "condition": condition, "history": history, "split": split,
                                   "gamma": gamma, "case_count": ev,
                                   "mean_confidence_before": float(pre_conf.mean()),
                                   "mean_confidence_after": float(post_conf.mean()),
                                   "strict_confidence_decrease_cases": int(np.sum(post_conf < pre_conf)),
                                   "casewise_confidence_nonincrease": True,
                                   "raw_sign_preserved": None if gamma == 0 else bool(np.array_equal(np.sign(pre_margin), np.sign(post_margin))),
                                   "probability_greedy_disagreement_cases": int(np.sum(_probability_decisions(pre_p) != _probability_decisions(post_p)))}
                    for label, margin, probability in (("before", pre_margin, pre_p), ("after", post_margin, post_p)):
                        diagnostics.update({f"{field}_{label}": value for field, value in
                                            _rank_metrics(margin, probability, datasets[split]["counts"], False).items()})
                    report.setdefault("boundary_diagnostics", []).append(diagnostics)
                if gamma is not None:
                    _require(all(np.array_equal(before[field], prefix["snapshots"][n][field]) for field in _SAVED_FIELDS),
                             "Every branch starts from the one exact shared prefix")
                relative = f"seed_{seed}/{condition}/shared_prefix/phase1_{n:04d}.npz"
                event = {"seed": seed, "condition": condition, "history": history, "gamma": gamma,
                         "shared_prefix_path": None if gamma is None else relative,
                         "shared_prefix_sha256": None if gamma is None else _hash(output / relative),
                         "head_change_l2": float(np.linalg.norm(after_head - before_head)), "internal_change_l2": 0.,
                         "feature_scale_before": scale, "feature_scale_after": scale,
                         "head_updates_before": ancestor, "head_updates_after": ancestor,
                         "internal_updates_before": ancestor if plastic else 0,
                         "internal_updates_after": ancestor if plastic else 0}
                _equal(_json(path / "boundary.json"), event, "Every local intervention event")
                _equal(boundaries[seed, condition, history], event, "Every aggregate intervention event")
                resource = {"seed": seed, "condition": condition, "history": history,
                            "feature_scale": scale, "initial_feature_gap_rms": scale,
                            "readout_parameters": head0.size, "internal_parameters": len(w0) if plastic else 0,
                            "algorithm": "conditional_expected_action_reference" if reference else "stochastic_reward"}
                _equal(resources[seed, condition, history], resource, "Every train-only resource entry")
                replay = _replay_phase(path, seed, condition, history, "phase2", False, cfg, datasets,
                                       uniforms["phase2"], encoder, bias, pre, post, w0, normalizer, head0, scale,
                                       linear, plastic, records, curves, before_weights, after_head, ancestor, reference)
                _require(all(np.array_equal(after[field], replay["snapshots"][0][field]) for field in _SAVED_FIELDS),
                         "Intervention state equals phase2 initial state")
                replays[history] = replay
                for (trial, split), value in replay["measurements"].items():
                    measurements[seed, condition, history, "phase2", trial, split] = value
                snapshot_count += len(replay["snapshots"])
                boundary_snapshot_count += 2
            if not plastic:
                for trial in checkpoints:
                    fresh, zero = (replays[history]["snapshots"][trial] for history in ("fresh_low", "gamma_0"))
                    _require(all(np.array_equal(fresh[field], zero[field]) for field in _SAVED_FIELDS
                                 - {"head_updates", "internal_updates"}), "Exact nonplastic zero-head/fresh state path")
                fields = _EXPECTED_FIELDS if reference else _TRIAL_FIELDS
                for trial in range(1, n + 1):
                    def record(history):
                        key = (seed, history, "phase2", trial) if reference else (seed, condition, history, "phase2", trial)
                        return records[key]
                    fresh, zero = record("fresh_low"), record("gamma_0")
                    _require(all(fresh[field] == zero[field] for field in fields - {"history"}),
                             "Exact nonplastic zero-head/fresh action/reward/update path")
        print(f"audited seed {seed} ({seeds.index(seed) + 1}/{len(seeds)})", flush=True)
    _summary(cfg, measurements, summary)
    probability_entries = (snapshot_count + boundary_snapshot_count) * 2 * ev
    report["counts"].update(artifact_hashes=len(hashes), source_hashes=len(registration["source_hashes"]),
                            seeds=len(seeds), reward_conditions=len(cfg["conditions"]), continuations=len(_HISTORIES),
                            actual_shared_prefixes=len(seeds) * len(cfg["conditions"]),
                            analytic_shared_prefixes=len(seeds), actual_training_trials=actual_count,
                            expected_reference_updates=expected_count, curve_rows=len(curves), checkpoints=snapshot_count,
                            boundary_state_files=boundary_snapshot_count, case_probability_entries=probability_entries,
                            raw_margin_entries=probability_entries, canonical_fingerprints=len(fingerprints),
                            boundary_events=len(boundaries), summary_outcomes=len(summary["outcomes"]),
                            prefix_outcomes=len(summary["prefix_outcomes"]), reference_comparison_rows=len(summary["reference_comparisons"]))
    report["checks"].extend([
        "Exact artifact coverage, source/config hashes and local registration/run chronology",
        "Four independent raw split regenerations, nuisance crossing and canonical disjointness per seed",
        "Independent source graph parsing, anatomical/rewired/ANN topology and encoder RNG reconstruction",
        "Initial phase1 TRAIN-only feature-gap RMS held across prefix, every branch, fresh and all states",
        "One physically executed high prefix per family with four exact independently replayed attenuated descendants",
        "Every action, reward, old-head internal eligibility, clipping and head/internal update replayed",
        "Every expected-action update independently enumerates both actions without producer calls or success gates",
        "Every checkpoint/boundary parameter, exact integer counter, probe and complete IID/OOD probability/raw margin reconstructed",
        "Head-only boundary scaling preserves topology, initial weights, bounds, normalizer, features, scale and counters",
        "Positive gamma raw sign/rank preservation, casewise nonincreasing confidence and exact gamma0 uniform policy",
        "Nonplastic gamma0=fresh complete state and action/update paths verified, with inherited counters kept separate",
        "Every all-seed H_A gate, prefix acquisition, endpoint/initial/gain/AUC contrast, no-learning baseline and reference difference reconstructed",
    ])


def verify(output: Path) -> dict:
    """Write an independent audit report without changing primary evidence."""
    output = Path(output).resolve()
    report = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "method": "independent numpy/stdlib arithmetic replay; only prior independent audit helpers reused",
              "input_scope": "artificial scalar order, shared high prefix, readout attenuation and fixed initial TRAIN scale",
              "counts": {}, "checks": [], "errors": [], "not_verified": [
                  "Visual numerosity extraction or biological/human neural mechanisms",
                  "Remote publication authenticity; local chronology is self-recorded",
                  "Statistical significance, mathematical understanding, or educational effects",
                  "Expected-action reference as reward-only learning or a stochastic ensemble expectation",
                  "Attenuation as pure confidence attribution in plastic models; the internal gradient also changes",
                  "Computational seeds as biological individuals"]}
    try:
        _require(output.is_dir(), f"Output directory missing: {output}")
        _audit(output, report)
        report["passed"] = True
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
