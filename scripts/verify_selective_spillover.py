"""Independent replay for reward-observed selective output spillover.

The current producer and its selective geometry helper are never imported.
Only frozen earlier independent input, graph and matrix arithmetic is reused.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "selective_frozen_independent_budget", Path(__file__).with_name("verify_update_budget.py"))
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)
prior, core = budget.prior, budget.core
TOPOLOGIES, INPUTS, TARGETS = budget.TOPOLOGIES, budget.INPUTS, budget.TARGETS
INPUT_CATALOG = budget.INPUT_CATALOG
MODES = ("shared", "isolated", "oracle_selective", "feedback_selective")
SIGNS = 2 * TARGETS - 1
RESIDUALS = ("raw_target_residual", "normalization_residual", "unit_norm_residual",
             "scaled_target_residual", "own_raw_preservation_residual",
             "raw_norm_lower_bound_residual", "harm_zero_residual")


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "selective_spillover_v1", "scope": "prospective_feedback_selective_spillover",
             "module": "drosolearn.selective_spillover", "protocol": "docs/selective-spillover-protocol.md",
             "topologies": list(TOPOLOGIES), "representations": list(INPUTS), "methods": list(MODES),
             "model_seeds": list(range(17401, 17411)), "training_trials": 400,
             "checkpoints": [0, 50, 100, 200, 400], "head_lr": .1, "rng_namespace": 202610196,
             "svd_rcond": 1e-12, "numeric_tolerance": 1e-8, "exact_condition_max": 1e6,
             "acquisition_threshold": .8, "acquisition_gain": .25, "class_threshold": .7,
             "population_pass_count": 8, "effect_threshold": .05, "positive_seed_count": 8,
             "normalization": "per_input_initial_fixed_MBON_gap_unit_L2", "input_catalog": INPUT_CATALOG,
             "budget_convention": "unit_directions_common_eta_ceiling_with_observed_reward_memory",
             "graph": "data/flywire_783_mb_left_kc512_s701", "external_manifest": None}
    fixture = allow_fixture and cfg.get("scope") == "integration_test"
    exempt = {"scope", "model_seeds", "training_trials", "checkpoints", "population_pass_count", "positive_seed_count"} if fixture else set()
    for key, value in fixed.items():
        if key not in exempt:
            core.demand(cfg.get(key) == value, "Fixed independent selective study: " + key)
    if fixture:
        seeds, n, grid = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
        core.demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(set(seeds)) == len(seeds)
                    and all(type(seed) is int and 90001 <= seed <= 90100 for seed in seeds), "Invented seeds only")
        core.demand(type(n) is int and 4 <= n <= 16 and n % 4 == 0, "Invented trial budget only")
        core.demand(grid == sorted(set(grid)) and grid[0] == 0 and grid[-1] == n
                    and all(type(step) is int and 0 <= step <= n and step % 4 == 0 for step in grid), "Invented checkpoint grid")
        core.demand(cfg["population_pass_count"] == cfg["positive_seed_count"] == len(seeds), "Invented pass counts")


def independent_geometry(feature_rows, rcond=1e-12, tolerance=1e-8, exact_condition_max=1e6):
    """The frozen unit directions, dual and label-free Gram matrix."""
    base = budget.independent_budget_report(feature_rows, rcond, tolerance, exact_condition_max)
    d = base["base_geometry"]["unit_rows"]
    inverse = base["base_geometry"]["pseudoinverse"]
    diagnostic = base["base_geometry"]["unit_diagnostics"]
    projector_residual = float(np.max(np.abs(d @ inverse - np.eye(4))))
    valid = bool(base["numeric_budget_qualified"] and base["projected_l2_exact_isolation_qualified"]
                 and diagnostic["retained_rank"] == 4
                 and all(np.isfinite(item).all() for item in (d, inverse, d @ d.T))
                 and projector_residual <= tolerance)
    core.demand(valid, "Full-row-rank qualified input directions required")
    return {"base_budget": base, "unit_rows": d, "base_influences": d @ d.T,
            "pseudoinverse": inverse, "shared_directions": base["directions"][0],
            "isolated_directions": base["directions"][2],
            "full_row_rank": bool(diagnostic["full_row_rank"]),
            "retained_rank": int(diagnostic["retained_rank"]),
            "full_row_condition": float(diagnostic["full_row_condition"]),
            "projector_identity_residual": projector_residual,
            "numeric_selective_qualified": valid, "rcond": float(rcond),
            "tolerance": float(tolerance), "exact_condition_max": float(exact_condition_max),
            "direction_uses_task_labels": False,
            "observed_sign_memory_is_separate_learning_resource": True}


def independent_direction(geometry, index, known, mode, tolerance=1e-8):
    """Compute a direction from supplied observations; no implicit target lookup."""
    known = np.asarray(known)
    core.demand(type(index) is int and 0 <= index < 4 and mode in MODES,
                "Registered mode and current input required")
    core.demand(known.shape == (4,) and known.dtype.kind in "biu"
                and np.isin(known, (-1, 0, 1)).all(), "Four observed or unknown signs required")
    d, gram, pinv = geometry["unit_rows"], geometry["base_influences"], geometry["pseudoinverse"]
    signed = known[index] * known * gram[index]
    harm = signed < -tolerance
    desired = gram[index].copy()
    if mode in ("oracle_selective", "feedback_selective"):
        desired[harm] = 0.
        unnormalized = d[index] + pinv @ (desired - gram[index])
        magnitude = float(np.linalg.norm(unnormalized))
        core.demand(np.isfinite(magnitude) and magnitude > 0., "No zero selective-direction fallback")
        direction = unnormalized / magnitude if np.any(harm) else unnormalized.copy()
    elif mode == "shared":
        harm[:] = False
        direction = d[index].copy()
        unnormalized = direction.copy()
        magnitude = float(np.linalg.norm(unnormalized))
    else:
        harm[:] = False
        direction = geometry["isolated_directions"][index].copy()
        unnormalized = direction.copy()
        magnitude = float(np.linalg.norm(unnormalized))
    return {"direction": direction, "harm_mask": harm, "desired_influence": desired,
            "actual_influence": d @ direction, "unnormalized": unnormalized,
            "unnormalized_l2": magnitude, "direction_l2": float(np.linalg.norm(direction))}


def independent_selective_direction(geometry, index, known):
    """Independent complete helper witness, including every numeric gate."""
    record = independent_direction(geometry, index, known, "feedback_selective", geometry["tolerance"])
    raw, direction = record["unnormalized"], record["direction"]
    desired, harm = record["desired_influence"], record["harm_mask"]
    base = geometry["base_influences"][index]
    raw_influence = geometry["unit_rows"] @ raw
    influence = geometry["unit_rows"] @ direction
    raw_norm = record["unnormalized_l2"]
    scale = 1. / raw_norm if np.any(harm) else 1.
    residuals = {
        "raw_target_residual": float(np.max(np.abs(raw_influence - desired))),
        "normalization_residual": float(np.max(np.abs(direction - raw * scale))),
        "unit_norm_residual": float(abs(np.linalg.norm(direction) - 1.)),
        "scaled_target_residual": float(np.max(np.abs(influence - desired * scale))),
        "own_raw_preservation_residual": float(abs(raw_influence[index] - base[index])),
        "raw_norm_lower_bound_residual": float(max(1. - raw_norm, 0.)),
        "harm_zero_residual": float(np.max(np.abs(influence[harm]))) if np.any(harm) else 0.}
    finite = all(np.isfinite(value).all() for value in (direction, raw, desired, raw_influence, influence))
    valid = bool(finite and all(value <= geometry["tolerance"] for value in residuals.values()))
    core.demand(valid, "Independent selective direction must satisfy every numeric gate")
    return {"source": int(index), "observed_signs": np.asarray(known, dtype=np.int8).copy(),
            "harm_mask": harm, "direction": direction, "raw_direction": raw,
            "desired_influence": desired, "raw_influence": raw_influence, "influence": influence,
            "raw_norm": raw_norm, "normalization_scale": scale, **residuals,
            "numeric_direction_qualified": valid,
            "original_helpful_magnitudes_retained_after_normalization": False}


class IndependentHead:
    """Observed-feedback memory, separate from output weights and fixed features."""
    def __init__(self, feature_rows, mode, rate=.1):
        core.demand(mode in MODES, "Registered selective mode required")
        self.geometry = independent_geometry(feature_rows)
        self.mode, self.rate = mode, rate
        self.head = np.zeros(self.geometry["unit_rows"].shape[1])
        self.known = np.zeros(4, int)
        self.calls = 0
        self.observation_calls = self.discoveries = 0
        self.path_l2 = self.squared_l2 = 0.

    def probabilities(self):
        return 1. / (1. + np.exp(-np.clip(self.geometry["unit_rows"] @ self.head, -40., 40.)))

    def observe(self, index, action, reward):
        core.demand(type(index) is int and 0 <= index < 4 and type(action) is int and action in (0, 1)
                    and type(reward) is int and reward in (-1, 0, 1)
                    and ((action == 0 and reward == 0) or (action == 1 and reward in (-1, 1))),
                    "Chosen Go provides signed feedback; NoGo gives no label")
        probability = float(self.probabilities()[index])
        known_before = self.known.copy()
        if action == 1:
            core.demand(self.known[index] in (0, reward), "Conflicting deterministic feedback")
            self.observation_calls += 1
            self.discoveries += int(self.known[index] == 0)
            self.known[index] = reward
        routing = (SIGNS.copy() if self.mode == "oracle_selective" else self.known.copy()
                   if self.mode == "feedback_selective" else np.zeros(4, int))
        direction = independent_direction(self.geometry, index, routing, self.mode)
        scalar = self.rate * (action - probability) * reward
        old_head = self.head.copy()
        self.head += scalar * direction["direction"]
        difference = self.head - old_head
        step = float(np.linalg.norm(difference))
        self.calls += 1
        self.path_l2 += step
        self.squared_l2 += step * step
        score_steps = self.geometry["unit_rows"] @ difference
        return {**direction, "known_before": known_before, "known_after": self.known.copy(),
                "routing_signs": routing, "observation_calls": self.observation_calls,
                "discovery_calls": self.discoveries,
                "probability_go": probability, "factor": scalar, "head_step_l2": step,
                "head_step_squared_l2": step * step, "own_score_step": float(score_steps[index]),
                "other_score_step_l2": float(np.linalg.norm(np.delete(score_steps, index))),
                "expected_head_step_l2": abs(scalar) * direction["direction_l2"],
                "head_difference": difference, "score_steps": score_steps}


def independent_evidence(geometry, source, routing, method):
    if method in ("oracle_selective", "feedback_selective"):
        return independent_selective_direction(geometry, source, routing)
    d = geometry["unit_rows"]
    if method == "shared":
        raw = d[source].copy()
        direction = raw.copy()
        desired = geometry["base_influences"][source].copy()
        scale = 1.
    else:
        raw = geometry["base_budget"]["base_geometry"]["directions"][source].copy()
        direction = geometry["isolated_directions"][source].copy()
        desired = np.eye(4)[source]
        scale = 1. / float(np.linalg.norm(raw))
    magnitude = float(np.linalg.norm(raw))
    raw_influence, influence = raw @ d.T, direction @ d.T
    residuals = {
        "raw_target_residual": float(np.max(np.abs(raw_influence - desired))),
        "normalization_residual": float(np.max(np.abs(direction - raw * scale))),
        "unit_norm_residual": float(abs(np.linalg.norm(direction) - 1.)),
        "scaled_target_residual": float(np.max(np.abs(influence - desired * scale))),
        "own_raw_preservation_residual": float(abs(raw_influence[source] - geometry["base_influences"][source, source])),
        "raw_norm_lower_bound_residual": float(max(1. - magnitude, 0.)), "harm_zero_residual": 0.}
    return {"source": source, "observed_signs": routing.copy(), "harm_mask": np.zeros(4, bool),
            "direction": direction, "raw_direction": raw, "desired_influence": desired,
            "raw_influence": raw_influence, "influence": influence, "raw_norm": magnitude,
            "normalization_scale": scale, **residuals,
            "numeric_direction_qualified": all(value <= geometry["tolerance"] for value in residuals.values()),
            "original_helpful_magnitudes_retained_after_normalization": False}


def expected_artifacts(cfg, base_graph):
    """All static preflight before RNG, followed by complete autonomous replay."""
    seeds, n, checkpoints = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
    arrays = {key: value.copy() for key, value in prior.independent_catalog_arrays().items()}
    shape = (3, 2, 4, len(seeds), n)
    integer_fields = {"actions", "rewards", "head_update_calls", "internal_update_calls", "value_update_calls",
                      "memory_changed", "memory_observation_calls", "memory_discoveries"}
    fields = ("probabilities", "actions", "rewards", "head_steps", "own_score_steps", "other_score_steps",
              "internal_steps", "value_steps", "head_update_calls", "internal_update_calls", "value_update_calls",
              "update_factors", "expected_head_steps", "head_step_squared_l2", "cumulative_head_path_l2",
              "cumulative_head_squared_l2", "head_norms", "raw_direction_norms", "normalization_scales",
              "memory_changed", "memory_observation_calls", "memory_discoveries")
    for key in fields:
        arrays["trial_" + key] = np.zeros(shape, dtype=np.int32 if key in integer_fields else float)
    for key in ("known_before", "known_after", "routing_signs"):
        arrays["trial_" + key] = np.empty(shape + (4,), np.int8)
    arrays["trial_harm_mask"] = np.empty(shape + (4,), bool)
    for key in ("desired_influence", "actual_influence"):
        arrays["trial_" + key] = np.empty(shape + (4,))
    arrays["trial_directions"] = np.empty(shape + (base_graph["n_mbon"],))
    arrays["trial_direction_residuals"] = np.empty(shape + (7,))
    checkpoint_shape = shape[:-1] + (len(checkpoints),)
    for key in ("logits", "probabilities", "lookup_probabilities"):
        arrays["checkpoint_" + key] = np.empty(checkpoint_shape + (4,))
    arrays["checkpoint_heads"] = np.empty(checkpoint_shape + (base_graph["n_mbon"],))
    arrays["checkpoint_known"] = np.empty(checkpoint_shape + (4,), np.int8)
    for key in ("head_calls", "internal_calls", "value_calls", "memory_observation_calls", "memory_discoveries"):
        arrays["checkpoint_" + key] = np.zeros(checkpoint_shape, np.int32)
    arrays["first_discovery_trials"] = np.full(shape[:-1] + (4,), n + 1, np.int32)
    arrays["schedules"] = np.empty((len(seeds), n), np.int8)
    arrays["uniforms"] = np.empty((len(seeds), n))
    arrays["model_seeds"] = np.asarray(seeds, np.int64)
    arrays["checkpoints"] = np.asarray(checkpoints, np.int32)
    static, cache, rows = {}, {}, []
    for si, seed in enumerate(seeds):
        for ti, topology in enumerate(TOPOLOGIES):
            offset = 110000 if topology == "rewired" else 120000 if topology == "ann" else 0
            graph = core.transform_graph(base_graph, topology, seed + offset)
            for ri, representation in enumerate(INPUTS):
                evidence = prior.feature_evidence(graph, seed, arrays["inputs"][ri])
                witness = prior.initial_witness(evidence)
                geometry = independent_geometry(evidence["differences"], cfg["svd_rcond"], cfg["numeric_tolerance"], cfg["exact_condition_max"])
                maxima = np.zeros(7)
                count = 0
                for signs in itertools.product((-1, 0, 1), repeat=4):
                    for source in range(4):
                        rec = independent_selective_direction(geometry, source, np.asarray(signs))
                        maxima = np.maximum(maxima, [rec[key] for key in RESIDUALS])
                        count += 1
                cache[(si, ti, ri)] = (graph, witness, geometry)
                prefix = topology + "_" + representation + "_"
                for key in ("weights", "encoder", "bias", "probe_kc", "probe_mbon"):
                    for kind in ("w_", "finalfixed_"):
                        static.setdefault(kind + prefix + key, []).append(witness[key].copy())
                keys = ("unit_rows", "pseudoinverse", "base_influences", "shared_directions", "isolated_directions",
                        "retained_rank", "full_row_condition", "projector_identity_residual", "numeric_selective_qualified")
                for key in keys:
                    static.setdefault("g_" + prefix + key, []).append(np.asarray(geometry[key]))
                static.setdefault("g_" + prefix + "preflight_direction_count", []).append(np.asarray(count))
                static.setdefault("g_" + prefix + "preflight_max_residuals", []).append(maxima)
    for si, seed in enumerate(seeds):
        schedule_rng = np.random.default_rng(np.random.SeedSequence([cfg["rng_namespace"], seed, 0]))
        schedule = np.concatenate([schedule_rng.permutation(4) for _ in range(n // 4)])
        uniforms = np.random.default_rng(np.random.SeedSequence([cfg["rng_namespace"], seed, 1])).random(n)
        arrays["schedules"][si], arrays["uniforms"][si] = schedule, uniforms
        for ti, topology in enumerate(TOPOLOGIES):
            for ri, representation in enumerate(INPUTS):
                graph, witness, geometry = cache[(si, ti, ri)]
                d = geometry["unit_rows"]
                for mi, method in enumerate(MODES):
                    head, known = np.zeros(graph["n_mbon"]), np.zeros(4, np.int8)
                    calls = observations = discoveries = 0
                    path_l2 = squared_l2 = 0.
                    def checkpoint(step):
                        at = (ti, ri, mi, si, checkpoints.index(step))
                        logits = d @ head
                        arrays["checkpoint_logits"][at] = logits
                        arrays["checkpoint_probabilities"][at] = 1. / (1. + np.exp(-np.clip(logits, -40., 40.)))
                        arrays["checkpoint_lookup_probabilities"][at] = (known + 1.) / 2.
                        arrays["checkpoint_heads"][at], arrays["checkpoint_known"][at] = head, known
                        arrays["checkpoint_head_calls"][at] = calls
                        arrays["checkpoint_memory_observation_calls"][at] = observations
                        arrays["checkpoint_memory_discoveries"][at] = discoveries
                    checkpoint(0)
                    for trial, example in enumerate(schedule):
                        source = int(example)
                        target = int(TARGETS[source])
                        p = float(1. / (1. + np.exp(-np.clip((d @ head)[source], -40., 40.))))
                        action = int(uniforms[trial] < p)
                        reward = int(SIGNS[source]) if action else 0
                        before_known = known.copy()
                        changed = int(action == 1 and known[source] == 0)
                        if action:
                            core.demand(known[source] in (0, reward), "Deterministic feedback consistency")
                            known[source] = reward
                            observations += 1
                            discoveries += changed
                            if changed:
                                arrays["first_discovery_trials"][ti, ri, mi, si, source] = trial + 1
                        routing = (SIGNS.copy() if method == "oracle_selective" else known.copy()
                                   if method == "feedback_selective" else np.zeros(4, np.int8))
                        rec = independent_evidence(geometry, source, routing, method)
                        core.demand(rec["numeric_direction_qualified"], "Dynamic direction gate")
                        factor = cfg["head_lr"] * (action - p) * reward
                        before_head = head.copy()
                        head += factor * rec["direction"]
                        difference = head - before_head
                        step = float(np.linalg.norm(difference))
                        score_steps = d @ difference
                        own, other = float(score_steps[source]), float(np.linalg.norm(np.delete(score_steps, source)))
                        square = step * step
                        path_l2 += step
                        squared_l2 += square
                        head_norm = float(np.linalg.norm(head))
                        direction_l2 = float(np.linalg.norm(rec["direction"]))
                        expected_step = abs(factor) * direction_l2
                        residuals = [rec[key] for key in RESIDUALS]
                        calls += 1
                        at = (ti, ri, mi, si, trial)
                        scalar_items = (("probabilities", p), ("actions", action), ("rewards", reward),
                            ("head_steps", step), ("own_score_steps", own), ("other_score_steps", other),
                            ("head_update_calls", calls), ("update_factors", factor), ("expected_head_steps", expected_step),
                            ("head_step_squared_l2", square), ("cumulative_head_path_l2", path_l2),
                            ("cumulative_head_squared_l2", squared_l2), ("head_norms", head_norm),
                            ("raw_direction_norms", rec["raw_norm"]), ("normalization_scales", rec["normalization_scale"]),
                            ("memory_changed", changed), ("memory_observation_calls", observations), ("memory_discoveries", discoveries))
                        for key, value in scalar_items: arrays["trial_" + key][at] = value
                        for key, value in (("known_before", before_known), ("known_after", known), ("routing_signs", routing),
                                           ("harm_mask", rec["harm_mask"]), ("desired_influence", rec["desired_influence"]),
                                           ("actual_influence", rec["influence"]), ("directions", rec["direction"]),
                                           ("direction_residuals", residuals)):
                            arrays["trial_" + key][at] = value
                        row = {"topology": topology, "representation": representation, "method": method,
                               "seed": seed, "trial": trial + 1, "example_index": source, "target": target,
                               "probability_go": p, "action": action, "reward": reward, "head_update_calls": calls,
                               "internal_update_calls": 0, "value_update_calls": 0, "head_step_l2": step,
                               "own_score_step": own, "other_score_step_l2": other, "internal_step_l2": 0.,
                               "value_step_l2": 0., "head_nonzero_step": int(step > 0), "internal_nonzero_step": 0,
                               "value_nonzero_step": 0, "update_factor": factor, "direction_l2": direction_l2,
                               "expected_head_step_l2": expected_step, "head_step_squared_l2": square,
                               "cumulative_head_path_l2": path_l2, "cumulative_head_squared_l2": squared_l2,
                               "head_norm": head_norm, "raw_direction_norm": rec["raw_norm"],
                               "normalization_scale": rec["normalization_scale"], "memory_changed": changed,
                               "memory_observation_calls": observations, "memory_discoveries": discoveries,
                               "direction_residual_max": float(max(residuals))}
                        for key, vector in (("known_before", before_known), ("known_after", known), ("routing_sign", routing),
                                            ("harm", rec["harm_mask"].astype(int)), ("desired_influence", rec["desired_influence"]),
                                            ("actual_influence", rec["influence"])):
                            for j in range(4): row[key + "_" + str(j)] = int(vector[j]) if key in ("known_before", "known_after", "routing_sign", "harm") else float(vector[j])
                        rows.append(row)
                        if trial + 1 in checkpoints: checkpoint(trial + 1)
    arrays.update({key: np.stack(value) for key, value in static.items()})
    return arrays, rows


def check_artifacts(expected, expected_rows, arrays, rows):
    core.demand(set(arrays) == set(expected), "Independent selective artifact closure")
    for key, value in expected.items():
        actual = np.asarray(arrays[key])
        core.demand(actual.dtype.kind in "biuf" and np.isfinite(actual).all(), "Finite artifact: " + key)
        if value.dtype.kind in "biu":
            core.demand(actual.dtype.kind in "biu" and np.array_equal(actual, value), "Independent exact artifact: " + key)
        else:
            core.array_close(actual, value, "Independent selective replay: " + key, atol=1e-9)
    core.demand(len(rows) == len(expected_rows), "Every selective trial row required")
    for index, (actual, expected_row) in enumerate(zip(rows, expected_rows)):
        if all(type(value) is str for value in actual.values()):
            core.check_csv_row(actual, expected_row, "Selective trial " + str(index))
        else:
            core.compare_tree(actual, expected_row, "Selective trial " + str(index))


def information_checks(cfg, arrays):
    """Derive information-access assertions from full saved arrays, not claims."""
    actions, rewards = arrays["trial_actions"], arrays["trial_rewards"]
    before, after = arrays["trial_known_before"], arrays["trial_known_after"]
    routing, schedules = arrays["trial_routing_signs"], arrays["schedules"]
    projected = before.copy()
    changed_expected = np.zeros_like(actions)
    for si in range(len(cfg["model_seeds"])):
        for trial, source in enumerate(schedules[si]):
            source = int(source)
            a, r = actions[..., si, trial], rewards[..., si, trial]
            old = before[..., si, trial, source]
            projected[..., si, trial, source] = np.where(a == 1, r, old)
            changed_expected[..., si, trial] = (a == 1) & (old == 0)
    continuity = bool((before[..., 0, :] == 0).all()
                      and np.array_equal(before[..., 1:, :], after[..., :-1, :]))
    memory_valid = bool(continuity and np.array_equal(after, projected))
    observed = actions == 1
    memory_valid = bool(memory_valid and np.isin(rewards[observed], (-1, 1)).all())
    for si in range(len(cfg["model_seeds"])):
        for trial, source in enumerate(schedules[si]):
            source = int(source)
            a, r = actions[..., si, trial], rewards[..., si, trial]
            old = before[..., si, trial, source]
            memory_valid = bool(memory_valid and np.all((a == 0) | (old == 0) | (old == r)))
    nogo = actions == 0
    nogo_valid = bool((rewards[nogo] == 0).all() and np.array_equal(before[nogo], after[nogo]))
    expected_observations = np.cumsum(actions, axis=-1)
    expected_discoveries = np.cumsum(changed_expected, axis=-1)
    writes = bool(np.array_equal(arrays["trial_memory_changed"], changed_expected)
                  and np.array_equal(arrays["trial_memory_observation_calls"], expected_observations)
                  and np.array_equal(arrays["trial_memory_discoveries"], expected_discoveries))
    first = np.full_like(arrays["first_discovery_trials"], cfg["training_trials"] + 1)
    for si in range(len(cfg["model_seeds"])):
        for trial, source in enumerate(schedules[si]):
            changed = changed_expected[..., si, trial].astype(bool)
            first[..., si, int(source)] = np.where(changed, trial + 1, first[..., si, int(source)])
    writes = bool(writes and np.array_equal(first, arrays["first_discovery_trials"]))
    checkpoint_memory = np.stack([np.zeros_like(after[..., 0, :]) if step == 0 else after[..., step - 1, :]
                                  for step in cfg["checkpoints"]], axis=-2)
    memory_valid = bool(memory_valid and np.array_equal(checkpoint_memory, arrays["checkpoint_known"]))
    lookup_valid = bool(np.array_equal(arrays["checkpoint_lookup_probabilities"], (checkpoint_memory + 1.) / 2.))
    controls = bool((routing[:, :, :2] == 0).all())
    oracle = bool(np.array_equal(routing[:, :, 2], np.broadcast_to(SIGNS, routing[:, :, 2].shape)))
    feedback = bool(np.array_equal(routing[:, :, 3], after[:, :, 3]))
    probabilities = arrays["trial_probabilities"]
    action_valid = bool(np.array_equal(actions, arrays["uniforms"][None, None, None] < probabilities))
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(INPUTS):
            d_all = arrays["g_" + topology + "_" + representation + "_unit_rows"]
            for mi in range(len(MODES)):
                for si in range(len(cfg["model_seeds"])):
                    at = (ti, ri, mi, si)
                    changes = arrays["trial_update_factors"][at][:, None] * arrays["trial_directions"][at]
                    head_path = np.concatenate((np.zeros((1, changes.shape[-1])), np.cumsum(changes, axis=0)))
                    logits = head_path[:-1] @ d_all[si].T
                    predictions = 1. / (1. + np.exp(-np.clip(logits, -40., 40.)))
                    chosen = predictions[np.arange(cfg["training_trials"]), schedules[si]]
                    action_valid = bool(action_valid
                        and np.max(np.abs(chosen - probabilities[at])) <= cfg["numeric_tolerance"]
                        and np.max(np.abs(head_path[cfg["checkpoints"]] - arrays["checkpoint_heads"][at])) <= cfg["numeric_tolerance"])
    return {"observed_memory_from_obtained_Go_rewards_only": memory_valid,
            "NoGo_does_not_reveal_unknown_sign": nogo_valid,
            "current_Go_feedback_precedes_direction": feedback,
            "controls_ignore_observed_memory_for_direction": controls,
            "oracle_privileged_labels_separate": oracle,
            "feedback_routing_equals_observed_after": feedback,
            "head_policy_does_not_directly_lookup_memory": action_valid,
            "static_lookup_not_autonomous_experiment": lookup_valid,
            "memory_writes_equal_first_discoveries": writes}


def expected_summary(cfg, arrays, rows):
    probability = arrays["checkpoint_probabilities"]
    correct = probability * TARGETS + (1. - probability) * (1 - TARGETS)
    first, last = correct[..., 0, :], correct[..., -1, :]
    score = last.mean(-1)
    gain = score - first.mean(-1)
    class_one, class_zero = last[..., :2].mean(-1), last[..., 2:].mean(-1)
    qualifies = ((score >= cfg["acquisition_threshold"]) & (gain >= cfg["acquisition_gain"])
                 & (class_one >= cfg["class_threshold"]) & (class_zero >= cfg["class_threshold"]))
    eta, n, tolerance = cfg["head_lr"], cfg["training_trials"], cfg["numeric_tolerance"]
    max_step = arrays["trial_head_steps"].max(-1)
    paths = arrays["trial_cumulative_head_path_l2"][..., -1]
    energies = arrays["trial_cumulative_head_squared_l2"][..., -1]
    endpoint_norms = np.linalg.norm(arrays["checkpoint_heads"][..., -1, :], axis=-1)
    energy_valid = energies <= n * eta ** 2 + tolerance
    endpoint_valid = endpoint_norms <= paths + tolerance
    bounded = ((max_step <= eta + tolerance) & (paths <= n * eta + tolerance) & energy_valid & endpoint_valid)
    expected_residual = float(np.max(np.abs(arrays["trial_head_steps"] - arrays["trial_expected_head_steps"])))
    dynamic_residual = float(np.max(arrays["trial_direction_residuals"]))
    dynamic_valid = bool(np.isfinite(arrays["trial_direction_residuals"]).all() and dynamic_residual <= tolerance)
    information = information_checks(cfg, arrays)
    families, contrasts, geometry = [], [], []
    lookup = arrays["checkpoint_lookup_probabilities"][..., -1, :]
    lookup_correct = (lookup * TARGETS + (1. - lookup) * (1 - TARGETS)).mean(-1)
    final_known = arrays["checkpoint_known"][..., -1, :]
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(INPUTS):
            prefix = "g_" + topology + "_" + representation + "_"
            seed_geometry = []
            for si, seed in enumerate(cfg["model_seeds"]):
                seed_geometry.append({"seed": seed, "retained_rank": int(arrays[prefix + "retained_rank"][si]),
                    "full_row_condition": float(arrays[prefix + "full_row_condition"][si]),
                    "projector_identity_residual": float(arrays[prefix + "projector_identity_residual"][si]),
                    "numeric_selective_qualified": bool(arrays[prefix + "numeric_selective_qualified"][si]),
                    "preflight_direction_count": int(arrays[prefix + "preflight_direction_count"][si]),
                    "preflight_max_residuals": arrays[prefix + "preflight_max_residuals"][si].tolist()})
            geometry.append({"topology": topology, "representation": representation, "seeds": seed_geometry})
            for mi, method in enumerate(MODES):
                individuals = []
                for si, seed in enumerate(cfg["model_seeds"]):
                    individuals.append({"seed": seed,
                        "initial_expected_accuracy": float(first[ti, ri, mi, si].mean()),
                        "final_expected_accuracy": float(score[ti, ri, mi, si]), "gain": float(gain[ti, ri, mi, si]),
                        "class_1_expected_accuracy": float(class_one[ti, ri, mi, si]),
                        "class_0_expected_accuracy": float(class_zero[ti, ri, mi, si]),
                        "acquisition_pass": bool(qualifies[ti, ri, mi, si]),
                        "final_head_l2": float(endpoint_norms[ti, ri, mi, si]),
                        "cumulative_head_path_l2": float(paths[ti, ri, mi, si]),
                        "cumulative_head_squared_l2": float(energies[ti, ri, mi, si]),
                        "max_head_step_l2": float(max_step[ti, ri, mi, si]),
                        "within_all_common_cost_bounds": bool(bounded[ti, ri, mi, si]),
                        "observed_signs_final": final_known[ti, ri, mi, si].tolist(),
                        "first_discovery_trials": arrays["first_discovery_trials"][ti, ri, mi, si].tolist(),
                        "memory_observation_calls": int(arrays["trial_memory_observation_calls"][ti, ri, mi, si, -1]),
                        "distinct_signs_observed": int(np.count_nonzero(final_known[ti, ri, mi, si])),
                        "lookup_expected_accuracy": float(lookup_correct[ti, ri, mi, si])})
                outcomes = score[ti, ri, mi]
                role = "primary" if (ti, ri, mi) == (0, 0, 3) else "privileged_diagnostic" if mi == 2 else "control"
                families.append({"topology": topology, "representation": representation, "method": method, "role": role,
                    "mean_final_expected_accuracy": float(outcomes.mean()),
                    "sd_final_expected_accuracy": float(outcomes.std(ddof=1)) if len(outcomes) > 1 else 0.,
                    "acquisition_pass_seeds": int(np.count_nonzero(qualifies[ti, ri, mi])),
                    "mean_cumulative_head_path_l2": float(paths[ti, ri, mi].mean()),
                    "mean_final_memory_known": float(np.count_nonzero(final_known[ti, ri, mi], axis=-1).mean()),
                    "mean_memory_lookup_accuracy": float(lookup_correct[ti, ri, mi].mean()), "seeds": individuals})
            contrasts.append({"topology": topology, "representation": representation,
                "feedback_minus_shared_by_seed": (score[ti, ri, 3] - score[ti, ri, 0]).tolist(),
                "feedback_minus_isolated_by_seed": (score[ti, ri, 3] - score[ti, ri, 1]).tolist(),
                "oracle_minus_feedback_by_seed": (score[ti, ri, 2] - score[ti, ri, 3]).tolist(),
                "descriptive_only": (ti, ri) != (0, 0)})
    static_valid = bool(all(item["numeric_selective_qualified"] and item["retained_rank"] == 4
        and item["preflight_direction_count"] == 324 and max(item["preflight_max_residuals"]) <= tolerance
        for group in geometry for item in group["seeds"]))
    all_valid = bool(static_valid and dynamic_valid and bounded.all() and expected_residual <= tolerance and all(information.values()))
    comparisons = {}
    for name, endpoint in (("shared", 0), ("isolated", 1)):
        effect = score[0, 0, 3] - score[0, 0, endpoint]
        positive = int(np.count_nonzero(effect > 0))
        comparisons[name] = {"mean_effect": float(effect.mean()), "positive_seeds": positive,
            "practical_advantage_pass": bool(effect.mean() >= cfg["effect_threshold"] and positive >= cfg["positive_seed_count"])}
    acquired = int(np.count_nonzero(qualifies[0, 0, 3]))
    seed_count = len(cfg["model_seeds"])
    coordinates = arrays["checkpoint_heads"].shape[-1]
    return {"study_id": cfg["study_id"], "status": "completed", "methods": list(MODES),
        "families": families, "contrasts": contrasts,
        "H_selective": {"supported": bool(all_valid and all(value["practical_advantage_pass"] for value in comparisons.values())),
            "endpoint_comparisons": comparisons, "all_numeric_and_information_gates_passed": all_valid,
            "requires_acquisition": False, "oracle_can_replace_primary": False},
        "H_acquisition": {"supported": bool(all_valid and acquired >= cfg["population_pass_count"]),
            "acquisition_pass_seeds": acquired, "all_numeric_and_information_gates_passed": all_valid,
            "requires_selective_advantage": False, "oracle_can_replace_primary": False},
        "budget_validation": {"conditional_scalar_matching_only": True, "actual_cumulative_cost_equalized": False,
            "common_unit_direction_step_ceiling": eta, "common_unit_direction_path_ceiling": n * eta,
            "common_unit_direction_squared_step_sum_ceiling": n * eta ** 2,
            "common_bounds_valid_all": bool(bounded.all()), "squared_step_sum_valid_all": bool(energy_valid.all()),
            "head_endpoint_within_path_all": bool(endpoint_valid.all()), "head_step_expected_residual_max": expected_residual,
            "all_static_numeric_gates_passed": static_valid, "all_dynamic_direction_gates_passed": dynamic_valid,
            "max_dynamic_direction_residual": dynamic_residual},
        "information_validation": information, "geometry": geometry,
        "counts": {"training_feedback_trials": len(rows), "static_probe_cases": int(np.prod(probability.shape)),
            "model_initializations": 24 * seed_count, "fixed_feature_constructions": 6 * seed_count,
            "model_seeds": seed_count, "head_update_calls": int(arrays["trial_head_update_calls"][..., -1].sum()),
            "head_nonzero_steps": int(np.count_nonzero(arrays["trial_head_steps"])),
            "internal_update_calls": 0, "value_update_calls": 0, "internal_nonzero_steps": 0, "value_nonzero_steps": 0,
            "auxiliary_memory_observation_calls": int(arrays["trial_memory_observation_calls"][..., -1].sum()),
            "auxiliary_memory_discoveries": int(arrays["trial_memory_changed"].sum()),
            "preflight_direction_evaluations": 324 * 6 * seed_count,
            "runtime_selective_direction_evaluations": 2 * 6 * seed_count * n,
            "lookup_static_cases": int(np.prod(probability.shape)), "lookup_autonomous_training_trials": 0},
        "resources": {"head_coordinates": int(coordinates), "observed_sign_memory_cells_per_learner": 4,
            "oracle_extra_given_target_signs": 4, "pseudoinverse_coefficients_per_group": int(4 * coordinates),
            "direction_solver_SVD_per_group": 1, "diagnostic_SVD_per_group": 2,
            "total_producer_SVD_calls": 18 * seed_count, "dynamic_SVD_calls": 0,
            "exact_actual_cumulative_budget_matching_claimed": False,
            "extra_solver_and_information_processing_cost_equalized": False},
        "interpretation": {"artificial_history_buffer": True, "internal_synapses_fixed": True,
            "identity_dictionary_is_supplied": True, "observed_reward_memory_is_additional_learning_state": True,
            "oracle_is_privileged_diagnostic": True, "helpful_raw_targets_preserved_before_unit_normalization": True,
            "original_helpful_magnitudes_preserved_after_normalization": False,
            "feedback_uses_hidden_targets_for_routing": False, "prior_failure_or_seed_discarded": False,
            "generalization_tested": False, "new_human_experiment": False, "shared_neural_mechanism_claim": False,
            "education_effect_claim": False}}


def verify_against_graph(cfg, graph, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    expected, expected_rows = expected_artifacts(cfg, graph)
    check_artifacts(expected, expected_rows, arrays, rows)
    core.compare_tree(summary, expected_summary(cfg, arrays, rows), "Independent complete selective claims")
    return {"arrays": len(expected), "independently_replayed_training_actions": len(expected_rows),
            "model_seeds": len(cfg["model_seeds"]), "conditions": 24,
            "checkpoint_static_probes": int(np.prod(arrays["checkpoint_probabilities"].shape)),
            "head_update_calls": int(arrays["trial_head_update_calls"][..., -1].sum()),
            "internal_update_calls": 0, "value_update_calls": 0,
            "all_checkpoints_and_trial_head_states_replayed": True,
            "fixed_internal_weights_encoder_and_probe_activity_checked": True,
            "all_81_label_free_sign_states_preflight_before_behavioral_rng_checked": True,
            "feedback_memory_from_obtained_Go_only_checked": True,
            "NoGo_hidden_target_and_current_feedback_order_checked": True,
            "oracle_prior_information_separate_from_observed_memory_checked": True,
            "all_dynamic_directions_masks_targets_and_numeric_residuals_checked": True,
            "all_actual_step_path_energy_and_endpoint_norms_checked": True,
            "lookup_reference_is_static_and_not_an_autonomous_agent_checked": True,
            "primary_feedback_compared_against_both_controls_checked": True,
            "acquisition_and_selective_advantage_gates_separate_checked": True,
            "actual_autonomous_costs_equalized": False,
            "extra_solver_and_information_processing_cost_equalized": False,
            "no_generalization_or_human_mechanism_claim": True}


def verify_science(cfg, root, arrays, rows, summary):
    validate_science(cfg)
    checksum = prior.verify_input_catalog({"input_catalog": INPUT_CATALOG, "study_id": "representation_update_v1"}, root)
    graph = core.raw_graph(Path(root) / cfg["graph"])
    core.demand((graph["n_kc"], graph["n_mbon"], len(graph["pre"])) == (512, 48, 1474), "Pinned anatomical subgraph")
    report = verify_against_graph(cfg, graph, arrays, rows, summary)
    report["frozen_input_catalog_sha256"] = checksum
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--publication-proof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve() / "src"))
    from drosolearn.selective_envelope import verify_envelope, sha, write_new
    cfg, _, metadata = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.publication_proof)
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    with (args.result_dir / "measurements.csv").open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        core.demand(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), "Unique CSV columns")
        rows = list(reader)
    summary = core.read_json(args.result_dir / "summary.json")
    checks = verify_science(cfg, args.root, arrays, rows, summary)
    core.demand(metadata["counts"] == summary["counts"], "Independent metadata and summary binding")
    report = {"status": "passed", "study_id": cfg["study_id"],
              "verified_at_utc": datetime.now(timezone.utc).isoformat(), "registration_sha256": sha(args.registration),
              "summary_sha256": sha(args.result_dir / "summary.json"), "scientific_producer_imported": False,
              "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
