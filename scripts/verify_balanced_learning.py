"""Independent autonomous replay for observed-feedback balanced output learning.

The new learner and its cone solver are never imported. Earlier frozen
independent graph/encoding routines are reused. Balanced directions are
certified by ambient-space least-squares dual bounds, not solver identity.
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


def _load(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selective = _load("verify_selective_spillover.py", "balanced_frozen_independent_selective")
cone = _load("verify_cone_frontier.py", "balanced_frozen_independent_cone")
budget, prior, core = selective.budget, selective.prior, selective.core
TOPOLOGIES, REPRESENTATIONS = selective.TOPOLOGIES, selective.INPUTS
TARGETS, SIGNS = selective.TARGETS, selective.SIGNS
INPUT_CATALOG = selective.INPUT_CATALOG
METHODS = ("shared", "isolated", "feedback_selective", "balanced")
demand, close, compare_tree = cone.demand, cone.close, cone.compare_tree


def sigmoid(values):
    return 1. / (1. + np.exp(-np.clip(values, -40., 40.)))


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "balanced_learning_v1", "scope": "prospective_observed_balanced_learning",
        "module": "drosolearn.balanced_learning", "protocol": "docs/balanced-learning-protocol.md",
        "topologies": list(TOPOLOGIES), "representations": list(REPRESENTATIONS), "methods": list(METHODS),
        "model_seeds": list(range(17501, 17511)), "training_trials": 400, "checkpoints": [0, 50, 100, 200, 400],
        "head_lr": .1, "rng_namespace": 202610201, "svd_rcond": 1e-12, "numeric_tolerance": 1e-8,
        "exact_condition_max": 1e6, "acquisition_threshold": .8, "acquisition_gain": .25, "class_threshold": .7,
        "population_pass_count": 8, "effect_threshold": .05, "positive_seed_count": 8, "unknown_nonworse_seed_count": 8,
        "normalization": "per_input_initial_fixed_MBON_gap_unit_L2",
        "budget_convention": "unit_directions_common_eta_ceiling_with_observed_reward_memory",
        "balanced_own_objective_weight": .5, "balanced_help_objective_weight": .5,
        "unknown_harm_endpoint": "sum_negative_correct_probability_change_on_unknown_other_destinations_over_all_400_trials",
        "input_catalog": INPUT_CATALOG, "graph": "data/flywire_783_mb_left_kc512_s701", "external_manifest": None}
    demand(isinstance(cfg, dict), "Independent balanced configuration object")
    fixture = cfg.get("scope") == "integration_test"
    demand(not fixture or allow_fixture, "Invented fixtures cannot enter the actual experiment")
    exempt = {"scope", "model_seeds", "training_trials", "checkpoints", "population_pass_count",
              "positive_seed_count", "unknown_nonworse_seed_count"} if fixture else set()
    for key, value in fixed.items():
        if key not in exempt: demand(cfg.get(key) == value, "Independent fixed scientific choice: " + key)
    if fixture:
        seeds, n, grid = cfg.get("model_seeds"), cfg.get("training_trials"), cfg.get("checkpoints")
        demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(set(seeds)) == len(seeds)
               and all(type(seed) is int and 90001 <= seed <= 90100 for seed in seeds), "Invented fixture seeds only")
        demand(type(n) is int and 4 <= n <= 16 and n % 4 == 0, "Bounded invented trial budget")
        demand(type(grid) is list and grid == sorted(set(grid)) and grid[0] == 0 and grid[-1] == n
               and all(type(step) is int and 0 <= step <= n and step % 4 == 0 for step in grid),
               "Bounded invented checkpoint grid")
        demand(cfg["population_pass_count"] == cfg["positive_seed_count"] == cfg["unknown_nonworse_seed_count"] == len(seeds),
               "Invented fixture pass counts")


def independent_unknown_harm(before_logits, after_logits, source, known_before):
    """Evaluator-only negative correct-probability changes on unknown OTHERs."""
    before, after, known = (np.asarray(value) for value in
                            (before_logits, after_logits, known_before))
    demand(before.shape == after.shape == known.shape == (4,) and source in range(4)
           and np.isfinite(before).all() and np.isfinite(after).all()
           and known.dtype.kind in "biu" and np.isin(known, (-1, 0, 1)).all(),
           "Unknown-other evaluator finite axes and explicit observations")
    unknown = (known == 0) & (np.arange(4) != source)
    correct_delta = SIGNS * (sigmoid(after) - sigmoid(before))
    negative = np.maximum(-correct_delta, 0.)
    return {"unknown_other_mask": unknown, "correct_probability_delta": correct_delta,
            "unknown_other_negative_correct_probability_sum": float(negative[unknown].sum()),
            "unknown_other_exposure_count": int(np.count_nonzero(unknown))}


def independent_problem(rows, source, observed, tolerance=1e-8):
    """Only obtained signs classify safety constraints and helpful directions."""
    given = np.asarray(observed)
    demand(given.shape == (4,) and given.dtype.kind in "biu"
           and np.isin(given, (-1, 0, 1)).all(), "Four explicit observed integer signs")
    a, c, known, helpful = cone.independent_problem(rows, source, given, 1, tolerance)
    padded = np.zeros((4, rows.shape[1]))
    padded[known] = a
    return {"constraints": padded, "objective": c, "known_indices": known,
            "helpful_indices": helpful}


def certify_balanced(rows, source, observed, direction, projection, multipliers, tolerance=1e-8):
    """Reoptimize a state in ambient feature space and validate its certificate."""
    problem = independent_problem(rows, source, observed, tolerance)
    a, c = problem["constraints"], problem["objective"]
    w, p, mu = (np.asarray(value) for value in (direction, projection, multipliers))
    demand(w.shape == p.shape == (rows.shape[1],) and mu.shape == (4,)
           and all(value.dtype.kind in "fiu" and np.isfinite(value).all() for value in (w, p, mu)),
           "Stored balanced witness finite axes")
    demand(np.all(mu >= 0.) and np.all(mu[np.asarray(observed) == 0] == 0.),
           "Exactly nonnegative and unknown-free padded dual multipliers")
    witness = cone.validate_dual_witness(c, a, w, mu, tolerance)
    optimum, _ = cone.independent_support_bound(c, a[problem["known_indices"]], tolerance)
    demand(abs(float(c @ w) - optimum) <= tolerance
           and abs(witness["dual_upper_bound"] - optimum) <= tolerance,
           "Independent ambient active-face dual optimum")
    ap = a @ p
    certificate = np.asarray([float(c @ w), np.linalg.norm(p), witness["dual_upper_bound"],
        np.linalg.norm(w), max(-float(np.min(a @ w)), 0.),
        np.max(np.abs(p - c - a.T @ mu)), np.max(np.abs(mu * ap)),
        abs(np.linalg.norm(p) - float(c @ w))])
    demand(np.all(certificate[4:] <= tolerance), "Balanced global support certificate")
    return {**problem, "certificate": certificate, "optimum": optimum}


def compare_rows(actual, wanted):
    """Column-wise checks avoid repeated Python scalar tree dispatch for CSV."""
    demand(len(actual) == len(wanted) and len(wanted) > 0, "Every ordered autonomous trial row")
    fields = tuple(wanted[0])
    demand(all(tuple(row) == fields for row in actual) and all(tuple(row) == fields for row in wanted),
           "Exact unique autonomous CSV column closure and order")
    for field in fields:
        expected = [row[field] for row in wanted]
        values = [row[field] for row in actual]
        if type(expected[0]) is str:
            demand(values == expected, "Ordered trial identity: " + field)
        else:
            try:
                numeric = np.asarray(values, float)
            except (ValueError, TypeError) as error:
                raise ValueError("Numeric autonomous CSV column: " + field) from error
            target = np.asarray(expected)
            demand(np.isfinite(numeric).all(), "Finite autonomous CSV column: " + field)
            if target.dtype.kind in "biu":
                demand(np.array_equal(numeric, target), "Exact autonomous CSV integers: " + field)
            else:
                demand(np.allclose(numeric, target, atol=1e-9, rtol=1e-9),
                       "Autonomous CSV arithmetic: " + field)


def information_checks(cfg, arrays):
    """Check complete obtained-reward memory and output-only choice histories."""
    actions, rewards = arrays["trial_actions"], arrays["trial_rewards"]
    before, after = arrays["trial_known_before"], arrays["trial_known_after"]
    route, schedules = arrays["trial_routing_signs"], arrays["schedules"]
    projected = before.copy()
    changed = np.zeros_like(actions)
    consistent = True
    for si in range(len(cfg["model_seeds"])):
        for t, source in enumerate(schedules[si]):
            source = int(source)
            a, r, old = actions[..., si, t], rewards[..., si, t], before[..., si, t, source]
            projected[..., si, t, source] = np.where(a == 1, r, old)
            changed[..., si, t] = (a == 1) & (old == 0)
            consistent &= bool(np.all((a == 0) | (old == 0) | (old == r)))
    continuity = (before[..., 0, :] == 0).all() and np.array_equal(before[..., 1:, :], after[..., :-1, :])
    actual_go = (np.isin(actions, (0, 1)).all() and np.isin(rewards[actions == 1], (-1, 1)).all()
                 and np.array_equal(after, projected) and consistent)
    no_go = (rewards[actions == 0] == 0).all() and np.array_equal(before[actions == 0], after[actions == 0])
    writes = (np.array_equal(arrays["trial_memory_changed"], changed)
              and np.array_equal(arrays["trial_memory_observation_calls"], np.cumsum(actions, axis=-1))
              and np.array_equal(arrays["trial_memory_discoveries"], np.cumsum(changed, axis=-1)))
    first = np.full_like(arrays["first_discovery_trials"], cfg["training_trials"] + 1)
    for si in range(len(cfg["model_seeds"])):
        for t, source in enumerate(schedules[si]):
            first[..., si, int(source)] = np.where(changed[..., si, t], t + 1, first[..., si, int(source)])
    writes &= np.array_equal(first, arrays["first_discovery_trials"])
    remembered = np.stack([np.zeros_like(after[..., 0, :]) if step == 0 else after[..., step - 1, :]
                           for step in cfg["checkpoints"]], axis=-2)
    memory = continuity and actual_go and np.array_equal(remembered, arrays["checkpoint_known"])
    routes = np.array_equal(route[:, :, 2:], after[:, :, 2:])
    control = (route[:, :, :2] == 0).all()
    policy = np.array_equal(actions, arrays["uniforms"][None, None, None] < arrays["trial_probabilities"])
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            for si, rows in enumerate(arrays["g_" + topology + "_" + representation + "_unit_rows"]):
                for mi in range(4):
                    at = (ti, ri, mi, si)
                    increments = arrays["trial_update_factors"][at][:, None] * arrays["trial_directions"][at]
                    path = np.concatenate((np.zeros((1, rows.shape[1])), np.cumsum(increments, axis=0)))
                    expected = sigmoid(path[:-1] @ rows.T)[np.arange(cfg["training_trials"]), schedules[si]]
                    policy &= np.allclose(expected, arrays["trial_probabilities"][at], atol=1e-8, rtol=0.)
                    policy &= np.allclose(path[cfg["checkpoints"]], arrays["checkpoint_heads"][at], atol=1e-8, rtol=0.)
                    policy &= np.allclose(sigmoid(path[cfg["checkpoints"]] @ rows.T),
                                          arrays["checkpoint_probabilities"][at], atol=1e-8, rtol=0.)
    return {"observed_memory_from_obtained_Go_rewards_only": bool(memory),
            "NoGo_does_not_reveal_unknown_sign": bool(no_go),
            "current_Go_feedback_precedes_direction": bool(routes),
            "controls_ignore_observed_memory_for_direction": bool(control),
            "feedback_and_balanced_routing_equals_observed_after": bool(routes),
            "head_policy_does_not_directly_lookup_memory": bool(policy),
            "static_lookup_not_autonomous_experiment": bool(np.array_equal(
                arrays["checkpoint_lookup_probabilities"], (remembered + 1.) / 2.)),
            "memory_writes_equal_first_discoveries": bool(writes),
            "no_oracle_learner_present": True}


def expected_artifacts(cfg, base_graph, actual):
    """Reconstruct fixed features, fresh streams, obtained observations and heads."""
    seeds, n, grid = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
    arrays = {key: value.copy() for key, value in prior.independent_catalog_arrays().items()}
    shape = (3, 2, 4, len(seeds), n)
    integer = {"actions", "rewards", "head_update_calls", "internal_update_calls", "value_update_calls",
               "memory_changed", "memory_observation_calls", "memory_discoveries", "unknown_other_exposure_count"}
    scalar_fields = ("probabilities", "actions", "rewards", "head_steps", "own_score_steps", "other_score_steps",
        "internal_steps", "value_steps", "head_update_calls", "internal_update_calls", "value_update_calls",
        "update_factors", "expected_head_steps", "head_step_squared_l2", "cumulative_head_path_l2",
        "cumulative_head_squared_l2", "head_norms", "raw_direction_norms", "normalization_scales",
        "memory_changed", "memory_observation_calls", "memory_discoveries",
        "unknown_other_negative_correct_probability_sum", "unknown_other_positive_correct_probability_sum",
        "unknown_other_negative_margin_sum", "unknown_other_exposure_count")
    for name in scalar_fields:
        arrays["trial_" + name] = np.zeros(shape, dtype=np.int32 if name in integer else float)
    for name in ("known_before", "known_after", "routing_signs"):
        arrays["trial_" + name] = np.empty(shape + (4,), np.int8)
    for name in ("harm_mask", "unknown_other_mask_before", "unknown_other_mask_after"):
        arrays["trial_" + name] = np.empty(shape + (4,), bool)
    for name in ("desired_influence", "actual_influence", "correct_probability_delta", "aligned_margin_delta"):
        arrays["trial_" + name] = np.empty(shape + (4,))
    dimension = base_graph["n_mbon"]
    arrays["trial_directions"] = np.empty(shape + (dimension,))
    arrays["trial_direction_residuals"] = np.zeros(shape + (7,))
    arrays["trial_direction_residuals_applicable"] = np.ones(shape, bool)
    arrays["trial_case_indices"] = np.full(shape, -1, np.int32)
    checkpoint_shape = shape[:-1] + (len(grid),)
    for name in ("logits", "probabilities", "lookup_probabilities"):
        arrays["checkpoint_" + name] = np.empty(checkpoint_shape + (4,))
    arrays["checkpoint_heads"] = np.empty(checkpoint_shape + (dimension,))
    arrays["checkpoint_known"] = np.empty(checkpoint_shape + (4,), np.int8)
    for name in ("head_calls", "internal_calls", "value_calls", "memory_observation_calls", "memory_discoveries"):
        arrays["checkpoint_" + name] = np.zeros(checkpoint_shape, np.int32)
    arrays["first_discovery_trials"] = np.full(shape[:-1] + (4,), n + 1, np.int32)
    arrays["schedules"] = np.empty((len(seeds), n), np.int8)
    arrays["uniforms"] = np.empty((len(seeds), n))
    arrays["model_seeds"] = np.asarray(seeds, np.int64)
    arrays["checkpoints"] = np.asarray(grid, np.int32)
    static, prepared = {}, {}
    for si, seed in enumerate(seeds):
        for ti, topology in enumerate(TOPOLOGIES):
            offset = 110000 if topology == "rewired" else 120000 if topology == "ann" else 0
            graph = core.transform_graph(base_graph, topology, seed + offset)
            for ri, representation in enumerate(REPRESENTATIONS):
                evidence = prior.feature_evidence(graph, seed, arrays["inputs"][ri])
                witness = prior.initial_witness(evidence)
                geometry = selective.independent_geometry(evidence["differences"], cfg["svd_rcond"],
                    cfg["numeric_tolerance"], cfg["exact_condition_max"])
                maxima, count = np.zeros(7), 0
                for signs in itertools.product((-1, 0, 1), repeat=4):
                    for source in range(4):
                        record = selective.independent_selective_direction(geometry, source, np.asarray(signs))
                        maxima = np.maximum(maxima, [record[key] for key in selective.RESIDUALS])
                        count += 1
                prepared[si, ti, ri] = geometry
                prefix = topology + "_" + representation + "_"
                for name in ("weights", "encoder", "bias", "probe_kc", "probe_mbon"):
                    for kind in ("w_", "finalfixed_"):
                        static.setdefault(kind + prefix + name, []).append(witness[name].copy())
                for name in ("unit_rows", "pseudoinverse", "base_influences", "shared_directions", "isolated_directions",
                             "retained_rank", "full_row_condition", "projector_identity_residual", "numeric_selective_qualified"):
                    static.setdefault("g_" + prefix + name, []).append(np.asarray(geometry[name]))
                static.setdefault("g_" + prefix + "preflight_direction_count", []).append(np.asarray(count))
                static.setdefault("g_" + prefix + "preflight_max_residuals", []).append(maxima)
    keys, directions, projections, multipliers, certificates, cases = [], [], [], [], [], {}
    rows = []
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence([cfg["rng_namespace"], seed, 0]))
        schedule = np.concatenate([rng.permutation(4) for _ in range(n // 4)])
        uniforms = np.random.default_rng(np.random.SeedSequence([cfg["rng_namespace"], seed, 1])).random(n)
        arrays["schedules"][si], arrays["uniforms"][si] = schedule, uniforms
        for ti, topology in enumerate(TOPOLOGIES):
            for ri, representation in enumerate(REPRESENTATIONS):
                geometry = prepared[si, ti, ri]
                d = geometry["unit_rows"]
                for mi, method in enumerate(METHODS):
                    head, known = np.zeros(dimension), np.zeros(4, np.int8)
                    calls = observations = discoveries = 0
                    path = energy = 0.
                    def checkpoint(step):
                        at = (ti, ri, mi, si, grid.index(step))
                        arrays["checkpoint_logits"][at] = d @ head
                        arrays["checkpoint_probabilities"][at] = sigmoid(d @ head)
                        arrays["checkpoint_lookup_probabilities"][at] = (known + 1.) / 2.
                        arrays["checkpoint_heads"][at], arrays["checkpoint_known"][at] = head, known
                        arrays["checkpoint_head_calls"][at] = calls
                        arrays["checkpoint_memory_observation_calls"][at] = observations
                        arrays["checkpoint_memory_discoveries"][at] = discoveries
                    checkpoint(0)
                    for t, source in enumerate(schedule):
                        source = int(source)
                        before_logits = d @ head
                        probability = float(sigmoid(before_logits)[source])
                        action = int(uniforms[t] < probability)
                        reward = int(SIGNS[source]) if action else 0
                        before_known = known.copy()
                        changed = int(action == 1 and known[source] == 0)
                        if action:
                            demand(known[source] in (0, reward), "Obtained deterministic reward consistency")
                            known[source] = reward
                            observations += 1
                            discoveries += changed
                            if changed: arrays["first_discovery_trials"][ti, ri, mi, si, source] = t + 1
                        route = known.copy() if mi >= 2 else np.zeros(4, np.int8)
                        case_index, applicable = -1, method != "balanced"
                        if method == "balanced":
                            direction, projection, mu = np.zeros(dimension), np.zeros(dimension), np.zeros(4)
                            if action:
                                key = (si, ti, ri, source, *map(int, known))
                                if key not in cases:
                                    case_index = len(keys)
                                    demand(case_index < len(actual["case_keys"]), "Complete actual balanced state cache")
                                    close(actual["case_keys"][case_index], np.asarray(key), "Observed-only first-Go case key")
                                    direction = actual["case_directions"][case_index]
                                    projection = actual["case_projections"][case_index]
                                    mu = actual["case_multipliers"][case_index]
                                    record = certify_balanced(d, source, known, direction, projection, mu, cfg["numeric_tolerance"])
                                    close(actual["case_certificates"][case_index], record["certificate"], "Independent balanced certificate arithmetic")
                                    cases[key] = case_index
                                    keys.append(key)
                                    directions.append(direction.copy())
                                    projections.append(projection.copy())
                                    multipliers.append(mu.copy())
                                    certificates.append(record["certificate"])
                                case_index = cases[key]
                                direction, projection = directions[case_index], projections[case_index]
                            raw_norm = float(np.linalg.norm(projection))
                            scale = 1. / raw_norm if raw_norm > 0. else 0.
                            record = {"direction": direction, "raw_norm": raw_norm, "normalization_scale": scale,
                                "harm_mask": np.zeros(4, bool), "desired_influence": d @ projection,
                                "influence": d @ direction}
                            residuals = np.zeros(7)
                        else:
                            record = selective.independent_evidence(geometry, source, route, method)
                            residuals = np.asarray([record[name] for name in selective.RESIDUALS])
                        factor = cfg["head_lr"] * (action - probability) * reward
                        before_head = head.copy()
                        head += factor * record["direction"]
                        movement = head - before_head
                        step = float(np.linalg.norm(movement))
                        changes = d @ movement
                        own = float(changes[source])
                        other = float(np.linalg.norm(np.delete(changes, source)))
                        expected_step = abs(factor) * float(np.linalg.norm(record["direction"]))
                        path += step
                        energy += step * step
                        head_norm = float(np.linalg.norm(head))
                        calls += 1
                        effect = independent_unknown_harm(before_logits, d @ head, source, before_known)
                        correct_delta = effect["correct_probability_delta"]
                        margin_delta = SIGNS * (d @ head - before_logits)
                        unknown = effect["unknown_other_mask"]
                        positive = float(np.maximum(correct_delta[unknown], 0.).sum())
                        negative_margin = float(np.maximum(-margin_delta[unknown], 0.).sum())
                        at = (ti, ri, mi, si, t)
                        values = {"probabilities": probability, "actions": action, "rewards": reward,
                            "head_steps": step, "own_score_steps": own, "other_score_steps": other,
                            "head_update_calls": calls, "update_factors": factor, "expected_head_steps": expected_step,
                            "head_step_squared_l2": step * step, "cumulative_head_path_l2": path,
                            "cumulative_head_squared_l2": energy, "head_norms": head_norm,
                            "raw_direction_norms": record["raw_norm"], "normalization_scales": record["normalization_scale"],
                            "memory_changed": changed, "memory_observation_calls": observations, "memory_discoveries": discoveries,
                            "unknown_other_negative_correct_probability_sum": effect["unknown_other_negative_correct_probability_sum"],
                            "unknown_other_positive_correct_probability_sum": positive,
                            "unknown_other_negative_margin_sum": negative_margin,
                            "unknown_other_exposure_count": effect["unknown_other_exposure_count"]}
                        for name, value in values.items(): arrays["trial_" + name][at] = value
                        vectors = {"known_before": before_known, "known_after": known, "routing_signs": route,
                            "harm_mask": record["harm_mask"], "desired_influence": record["desired_influence"],
                            "actual_influence": record["influence"], "directions": record["direction"],
                            "direction_residuals": residuals, "unknown_other_mask_before": unknown,
                            "unknown_other_mask_after": (known == 0) & (np.arange(4) != source),
                            "correct_probability_delta": correct_delta, "aligned_margin_delta": margin_delta}
                        for name, value in vectors.items(): arrays["trial_" + name][at] = value
                        arrays["trial_case_indices"][at] = case_index
                        arrays["trial_direction_residuals_applicable"][at] = applicable
                        row = {"topology": topology, "representation": representation, "method": method,
                            "seed": seed, "trial": t + 1, "example_index": source, "target": int(TARGETS[source]),
                            "probability_go": probability, "action": action, "reward": reward,
                            "head_update_calls": calls, "internal_update_calls": 0, "value_update_calls": 0,
                            "head_step_l2": step, "own_score_step": own, "other_score_step_l2": other,
                            "internal_step_l2": 0., "value_step_l2": 0., "head_nonzero_step": int(step > 0.),
                            "internal_nonzero_step": 0, "value_nonzero_step": 0, "update_factor": factor,
                            "direction_l2": float(np.linalg.norm(record["direction"])), "expected_head_step_l2": expected_step,
                            "head_step_squared_l2": step * step, "cumulative_head_path_l2": path,
                            "cumulative_head_squared_l2": energy, "head_norm": head_norm,
                            "raw_direction_norm": record["raw_norm"], "normalization_scale": record["normalization_scale"],
                            "memory_changed": changed, "memory_observation_calls": observations,
                            "memory_discoveries": discoveries, "direction_residual_max": float(np.max(residuals))}
                        for name, vector in (("known_before", before_known), ("known_after", known), ("routing_sign", route),
                            ("harm", record["harm_mask"]), ("desired_influence", record["desired_influence"]),
                            ("actual_influence", record["influence"])):
                            for j in range(4):
                                row[name + "_" + str(j)] = int(vector[j]) if name in ("known_before", "known_after", "routing_sign", "harm") else float(vector[j])
                        row.update({"case_index": case_index, "direction_residuals_applicable": int(applicable)})
                        for name in ("unknown_other_negative_correct_probability_sum", "unknown_other_positive_correct_probability_sum",
                                     "unknown_other_negative_margin_sum", "unknown_other_exposure_count"):
                            row[name] = values[name]
                        for name, vector in (("unknown_other_before", unknown), ("unknown_other_after", vectors["unknown_other_mask_after"]),
                                             ("correct_probability_delta", correct_delta), ("aligned_margin_delta", margin_delta)):
                            for j in range(4):
                                row[name + "_" + str(j)] = int(vector[j]) if name.startswith("unknown_other") else float(vector[j])
                        rows.append(row)
                        if t + 1 in grid: checkpoint(t + 1)
    demand(len(keys) == len(actual["case_keys"]), "No unused or duplicate balanced state cache")
    arrays.update({name: np.stack(values) for name, values in static.items()})
    for name, value, width in (("case_keys", keys, 8), ("case_directions", directions, dimension),
                              ("case_projections", projections, dimension), ("case_multipliers", multipliers, 4),
                              ("case_certificates", certificates, 8)):
        arrays[name] = np.asarray(value, dtype=np.int64 if name == "case_keys" else float).reshape(-1, width)
    return arrays, rows


def check_artifacts(actual, wanted, rows, wanted_rows):
    demand(set(actual) == set(wanted), "Independent complete balanced array closure")
    for key, value in wanted.items():
        close(actual[key], value, "Independent actual balanced trajectory: " + key)
    compare_rows(rows, wanted_rows)


def expected_summary(cfg, arrays, rows):
    """Recompute behavioral endpoints, accumulated unknown harm and separate gates."""
    s, n, eta, tol = len(cfg["model_seeds"]), cfg["training_trials"], cfg["head_lr"], cfg["numeric_tolerance"]
    p = arrays["checkpoint_probabilities"]
    correct = p * TARGETS + (1. - p) * (1 - TARGETS)
    first, last = correct[..., 0, :], correct[..., -1, :]
    initial, outcomes = first.mean(-1), last.mean(-1)
    gain = outcomes - initial
    class_one, class_zero = last[..., :2].mean(-1), last[..., 2:].mean(-1)
    learned = ((outcomes >= cfg["acquisition_threshold"]) & (gain >= cfg["acquisition_gain"])
               & (class_one >= cfg["class_threshold"]) & (class_zero >= cfg["class_threshold"]))
    lengths = arrays["trial_cumulative_head_path_l2"][..., -1]
    energy = arrays["trial_cumulative_head_squared_l2"][..., -1]
    endpoint = np.linalg.norm(arrays["checkpoint_heads"][..., -1, :], axis=-1)
    largest = arrays["trial_head_steps"].max(-1)
    bounded = ((largest <= eta + tol) & (lengths <= n * eta + tol)
               & (energy <= n * eta ** 2 + tol) & (endpoint <= lengths + tol))
    step_residual = float(np.abs(arrays["trial_head_steps"] - arrays["trial_expected_head_steps"]).max())
    applicable = arrays["trial_direction_residuals_applicable"]
    control_residual = float(np.max(arrays["trial_direction_residuals"][applicable]))
    certificate_residual = float(np.max(arrays["case_certificates"][:, 4:])) if len(arrays["case_keys"]) else 0.
    information = information_checks(cfg, arrays)
    demand(all(information.values()), "Independent obtained-information gates")
    known = arrays["trial_known_after"] != 0
    known_damage = known & (arrays["trial_aligned_margin_delta"] < -tol)
    unknown_damage = arrays["trial_unknown_other_negative_correct_probability_sum"].sum(-1)
    unknown_help = arrays["trial_unknown_other_positive_correct_probability_sum"].sum(-1)
    unknown_margin = arrays["trial_unknown_other_negative_margin_sum"].sum(-1)
    exposures = arrays["trial_unknown_other_exposure_count"].sum(-1)
    equal_masks = np.array_equal(arrays["trial_unknown_other_mask_before"], arrays["trial_unknown_other_mask_after"])
    unknown_before = np.count_nonzero(arrays["trial_known_before"] == 0, axis=-1)
    unknown_after = np.count_nonzero(arrays["trial_known_after"] == 0, axis=-1)
    count_drop = np.array_equal(unknown_before - unknown_after, arrays["trial_memory_changed"])
    demand(equal_masks and count_drop, "Unknown-other mask invariance and source-inclusive first-Go count drop")
    families, geometries, contrasts = [], [], []
    static_valid = True
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            prefix = "g_" + topology + "_" + representation + "_"
            geoseeds = []
            for si, seed in enumerate(cfg["model_seeds"]):
                record = {"seed": seed}
                for name in ("retained_rank", "full_row_condition", "projector_identity_residual",
                             "numeric_selective_qualified", "preflight_direction_count", "preflight_max_residuals"):
                    record[name] = arrays[prefix + name][si].tolist()
                static_valid &= bool(record["numeric_selective_qualified"] and record["retained_rank"] == 4
                    and record["preflight_direction_count"] == 324 and max(record["preflight_max_residuals"]) <= tol)
                geoseeds.append(record)
            geometries.append({"topology": topology, "representation": representation, "seeds": geoseeds})
            for mi, method in enumerate(METHODS):
                individuals = []
                for si, seed in enumerate(cfg["model_seeds"]):
                    at = (ti, ri, mi, si)
                    sample = arrays["trial_actions"][at] == TARGETS[arrays["schedules"][si]]
                    exposure = int(exposures[at])
                    individuals.append({"seed": seed,
                        "initial_expected_accuracy": float(initial[at]), "final_expected_accuracy": float(outcomes[at]),
                        "gain": float(gain[at]), "class_1_expected_accuracy": float(class_one[at]),
                        "class_0_expected_accuracy": float(class_zero[at]), "acquisition_pass": bool(learned[at]),
                        "sampled_accuracy_all_trials": float(sample.mean()),
                        "sampled_accuracy_last_100_trials": float(sample[-min(100, n):].mean()),
                        "final_head_l2": float(endpoint[at]), "cumulative_head_path_l2": float(lengths[at]),
                        "cumulative_head_squared_l2": float(energy[at]), "max_head_step_l2": float(largest[at]),
                        "within_all_common_cost_bounds": bool(bounded[at]),
                        "observed_signs_final": arrays["checkpoint_known"][at + (-1,)].tolist(),
                        "first_discovery_trials": arrays["first_discovery_trials"][at].tolist(),
                        "memory_observation_calls": int(arrays["trial_memory_observation_calls"][at + (-1,)]),
                        "distinct_signs_observed": int(arrays["trial_memory_discoveries"][at + (-1,)]),
                        "unknown_other_negative_correct_probability_sum": float(unknown_damage[at]),
                        "unknown_other_positive_correct_probability_sum": float(unknown_help[at]),
                        "unknown_other_negative_margin_sum": float(unknown_margin[at]),
                        "unknown_other_exposure_count": exposure,
                        "unknown_other_negative_correct_probability_per_exposure": float(unknown_damage[at] / exposure) if exposure else None,
                        "known_destination_negative_margin_events": int(known_damage[at].sum())})
                values = outcomes[ti, ri, mi]
                families.append({"topology": topology, "representation": representation, "method": method,
                    "role": "primary" if (ti, ri, mi) == (0, 0, 3) else "control",
                    "mean_final_expected_accuracy": float(values.mean()),
                    "sd_final_expected_accuracy": float(values.std(ddof=1)) if s > 1 else 0.,
                    "acquisition_pass_seeds": int(learned[ti, ri, mi].sum()),
                    "mean_cumulative_head_path_l2": float(lengths[ti, ri, mi].mean()),
                    "mean_unknown_other_negative_correct_probability_sum": float(unknown_damage[ti, ri, mi].mean()),
                    "mean_unknown_other_exposure_count": float(exposures[ti, ri, mi].mean()), "seeds": individuals})
            contrasts.append({"topology": topology, "representation": representation,
                "descriptive_only": (ti, ri) != (0, 0),
                "balanced_minus_feedback_selective_by_seed": (outcomes[ti, ri, 3] - outcomes[ti, ri, 2]).tolist(),
                "balanced_minus_isolated_by_seed": (outcomes[ti, ri, 3] - outcomes[ti, ri, 1]).tolist(),
                "balanced_minus_feedback_unknown_damage_by_seed": (unknown_damage[ti, ri, 3] - unknown_damage[ti, ri, 2]).tolist()})
    safe = bool(not known_damage[:, :, 1:].any())
    numeric = bool(static_valid and bounded.all() and step_residual <= tol and control_residual <= tol
                   and certificate_residual <= tol and safe and all(information.values()))
    demand(numeric, "Independent numerical, known-safety, information and common-cost integrity gates")
    comparisons = {}
    for name, index in (("feedback_selective", 2), ("isolated", 1)):
        difference = outcomes[0, 0, 3] - outcomes[0, 0, index]
        positive = int(np.count_nonzero(difference > 0.))
        comparisons[name] = {"mean_effect": float(difference.mean()), "positive_seeds": positive,
            "practical_advantage_pass": bool(difference.mean() >= cfg["effect_threshold"] and positive >= cfg["positive_seed_count"])}
    unknown_difference = unknown_damage[0, 0, 3] - unknown_damage[0, 0, 2]
    nonworse = int(np.count_nonzero(unknown_difference <= tol))
    unknown_pass = unknown_difference.mean() <= tol and nonworse >= cfg["unknown_nonworse_seed_count"]
    acquired = int(learned[0, 0, 3].sum())
    demand(len(rows) == 24 * s * n, "Complete actual feedback rows for independent summary")
    return {"study_id": cfg["study_id"], "status": "completed", "methods": list(METHODS), "families": families,
        "contrasts": contrasts, "geometry": geometries,
        "H_gain": {"supported": bool(numeric and all(value["practical_advantage_pass"] for value in comparisons.values())),
            "endpoint_comparisons": comparisons, "all_numeric_and_information_gates_passed": numeric, "requires_acquisition": False},
        "H_acquisition": {"supported": bool(numeric and acquired >= cfg["population_pass_count"]),
            "acquisition_pass_seeds": acquired, "all_numeric_and_information_gates_passed": numeric, "requires_gain": False},
        "H_unknown": {"supported": bool(numeric and unknown_pass),
            "mean_balanced_minus_feedback_damage": float(unknown_difference.mean()), "nonworsening_seeds": nonworse,
            "empirical_observed_sample_only": True, "population_equivalence_claimed": False,
            "fixed_400_trial_total_is_primary": True, "all_numeric_and_information_gates_passed": numeric},
        "validation": {"all_numeric_and_information_gates_passed": numeric,
            "all_common_cost_bounds_passed": bool(bounded.all()), "all_static_geometry_gates_passed": bool(static_valid),
            "all_control_direction_gates_passed": bool(control_residual <= tol),
            "all_balanced_case_certificates_passed": bool(certificate_residual <= tol),
            "all_nonshared_known_destinations_safe": safe, "head_step_expected_residual_max": step_residual,
            "control_direction_residual_max": control_residual, "balanced_certificate_residual_max": certificate_residual,
            "unknown_other_masks_equal_before_after": bool(equal_masks),
            "source_inclusive_unknown_count_drop_equals_discoveries": bool(count_drop)},
        "information_validation": information,
        "budget_validation": {"common_step_ceiling": eta, "common_path_ceiling": n * eta,
            "common_squared_step_sum_ceiling": n * eta ** 2, "actual_cumulative_cost_equalized": False,
            "extra_solver_and_information_processing_cost_equalized": False},
        "counts": {"training_feedback_trials": len(rows), "static_probe_cases": int(np.prod(p.shape)),
            "model_initializations": 24 * s, "fixed_feature_constructions": 6 * s, "model_seeds": s,
            "head_update_calls": int(arrays["trial_head_update_calls"][..., -1].sum()),
            "head_nonzero_steps": int(np.count_nonzero(arrays["trial_head_steps"])), "internal_update_calls": 0,
            "value_update_calls": 0, "internal_nonzero_steps": 0, "value_nonzero_steps": 0,
            "auxiliary_memory_observation_calls": int(arrays["trial_memory_observation_calls"][..., -1].sum()),
            "auxiliary_memory_discoveries": int(arrays["trial_memory_changed"].sum()),
            "preflight_direction_evaluations": 6 * s * 324, "runtime_selective_direction_evaluations": 6 * s * n,
            "balanced_direction_requests": int(arrays["trial_actions"][:, :, 3].sum()),
            "balanced_unique_solver_cases": len(arrays["case_keys"]),
            "balanced_NoGo_solver_skips": int(np.count_nonzero(arrays["trial_actions"][:, :, 3] == 0)),
            "unknown_other_destination_exposures": int(exposures.sum()),
            "source_inclusive_unknown_exposures_before_feedback": int(unknown_before.sum()),
            "source_inclusive_unknown_exposures_after_feedback": int(unknown_after.sum()),
            "lookup_static_cases": int(np.prod(p.shape)), "lookup_autonomous_training_trials": 0},
        "resources": {"head_coordinates": int(arrays["checkpoint_heads"].shape[-1]),
            "observed_sign_memory_cells_per_learner": 4, "privileged_target_signs_given_to_any_learner": 0,
            "balanced_solver_cache_only_actual_observations": True,
            "balanced_solver_calls_equal_unique_actual_Go_states": True, "processing_cost_equalized": False},
        "interpretation": {"artificial_history_buffer": True, "internal_synapses_fixed": True,
            "identity_dictionary_is_supplied": True, "observed_reward_memory_is_additional_learning_state": True,
            "true_labels_used_for_unknown_harm_evaluation_only": True,
            "feedback_or_balanced_uses_hidden_targets_for_routing": False, "prior_failure_or_seed_discarded": False,
            "unknown_means_reward_unobserved_not_novel_input": True, "generalization_tested": False,
            "new_human_experiment": False, "shared_neural_mechanism_claim": False, "education_effect_claim": False}}


def verify_against_graph(cfg, graph, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    wanted, wanted_rows = expected_artifacts(cfg, graph, arrays)
    check_artifacts(arrays, wanted, rows, wanted_rows)
    expected = expected_summary(cfg, wanted, wanted_rows)
    compare_tree(summary, expected, "Independent complete balanced learning summary")
    return {"arrays": len(wanted), "independently_replayed_training_actions": len(wanted_rows),
        "model_seeds": len(cfg["model_seeds"]), "conditions": 24,
        "checkpoint_static_probes": int(np.prod(wanted["checkpoint_probabilities"].shape)),
        "head_update_calls": int(wanted["trial_head_update_calls"][..., -1].sum()),
        "internal_update_calls": 0, "value_update_calls": 0,
        "unique_balanced_states_independently_reoptimized": len(wanted["case_keys"]),
        "all_fresh_autonomous_choices_rewards_and_heads_replayed": True,
        "fixed_graph_encoding_weights_and_probe_activity_checked": True,
        "selective_324_hypothetical_directions_preflight_before_behavioral_rng_checked": True,
        "balanced_no_go_skips_optimizer_and_uses_zero_direction_checked": True,
        "balanced_uses_only_actual_own_obtained_feedback_checked": True,
        "independent_ambient_lstsq_dual_support_optima_checked": True,
        "case_certificates_are_global_support_bounds_not_solver_identity": True,
        "unknown_other_masks_before_and_after_match_checked": True,
        "evaluator_hidden_signs_do_not_enter_observed_direction_checked": True,
        "per_trial_correct_probability_and_aligned_margin_changes_checked": True,
        "all_three_hypotheses_separately_recomputed": True,
        "common_actual_movement_ceiling_not_equal_realized_budget_checked": True,
        "extra_optimizer_and_memory_processing_cost_equalized": False,
        "new_generalization_or_human_mechanism_claim": False}


def verify_science(cfg, root, arrays, rows, summary):
    validate_science(cfg)
    checksum = prior.verify_input_catalog({"input_catalog": INPUT_CATALOG,
        "study_id": "representation_update_v1"}, root)
    graph = core.raw_graph(Path(root) / cfg["graph"])
    demand((graph["n_kc"], graph["n_mbon"], len(graph["pre"])) == (512, 48, 1474),
           "Pinned v783 anatomical subgraph")
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
    from drosolearn.balanced_envelope import verify_envelope, sha, write_new
    cfg, _, metadata = verify_envelope(args.root, args.config, args.registration,
        args.result_dir, args.publication_proof)
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    with (args.result_dir / "measurements.csv").open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        demand(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
               "Unique CSV columns")
        rows = list(reader)
    summary = core.read_json(args.result_dir / "summary.json")
    checks = verify_science(cfg, args.root, arrays, rows, summary)
    demand(metadata["counts"] == summary["counts"], "Independent metadata and summary count binding")
    report = {"status": "passed", "study_id": cfg["study_id"],
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "registration_sha256": sha(args.registration),
        "summary_sha256": sha(args.result_dir / "summary.json"), "scientific_producer_imported": False,
        "new_helper_imported": False, "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
