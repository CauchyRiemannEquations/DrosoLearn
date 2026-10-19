"""Independent saved-path normalization audit; no learner, graph, or RNG calls.

The scientific producer and its helpers are never imported. Stored decisions,
obtained rewards, unit input features, and output directions are sufficient to
reconstruct the original trajectory and its descriptive matched-scalar algebra.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


TOPOLOGIES = ("real", "rewired", "ann")
REPRESENTATIONS = ("raw", "identity")
METHODS = ("shared", "isolated", "oracle_selective", "feedback_selective")
SIGNS = np.asarray([1., 1., -1., -1.])
CATEGORIES = ("self", "cross_helpful", "cross_harmful", "cross_neutral", "cross_unknown",
              "cross_same", "cross_opposite", "cross_all", "all")
PHASES = ("all", "partial", "full")
RESIDUALS = ("gram", "actual_influence", "desired_influence", "scaled_desired", "reciprocal_scale",
             "own_score", "other_score", "head_step", "scalar", "unit_direction", "checkpoint_logits",
             "stage_accounting")
METRICS = ("trial_count", "updated_trials", "cells", "updated_cells", "scale_mean_all", "scale_mean_updated",
           "raw_norm_mean", "scale_min", "scale_max", "baseline_helpful", "baseline_harmful", "baseline_net",
           "desired_helpful", "desired_harmful", "desired_net", "actual_helpful", "actual_harmful", "actual_net",
           "suppression_net", "normalization_net", "helpful_loss", "harmful_reduction", "helpful_retention",
           "harmful_retention")
ADDITIVE = METRICS[:4] + METRICS[9:22]


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, atol=1e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    demand(actual.shape == expected.shape, label + " shape")
    demand(actual.dtype.kind in "biuf" and np.isfinite(actual).all(), label + " finite")
    if expected.dtype.kind in "biu":
        demand(actual.dtype.kind in "biu" and np.array_equal(actual, expected), label + " exact")
    else:
        demand(np.allclose(actual, expected, atol=atol, rtol=1e-9), label + " arithmetic")


def compare_tree(actual, expected, label="summary"):
    if isinstance(expected, dict):
        demand(isinstance(actual, dict) and actual.keys() == expected.keys(), label + " keys")
        for key in expected:
            compare_tree(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        demand(isinstance(actual, list) and len(actual) == len(expected), label + " length")
        for i, value in enumerate(expected):
            compare_tree(actual[i], value, label + "." + str(i))
    elif isinstance(expected, bool):
        demand(type(actual) is bool and actual == expected, label + " flag")
    elif isinstance(expected, int):
        demand(type(actual) is int and actual == expected, label + " integer")
    elif isinstance(expected, float):
        demand(type(actual) in (float, int) and np.isfinite(actual)
               and np.isclose(actual, expected, atol=1e-9, rtol=1e-9), label + " numeric")
    else:
        demand(actual == expected, label + " value")


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -40., 40.)))


def independent_direction(d, inverse, source, routing, method, tolerance):
    """Raw target cancellation and its unit-length scale, computed afresh."""
    gram = d @ d.T
    harm = np.zeros(4, bool)
    if method == "shared":
        desired, raw, scale = gram[source].copy(), d[source].copy(), 1.
    elif method == "isolated":
        desired = np.eye(4)[source]
        projector = d @ inverse
        raw = inverse[:, source] / projector[source, source]
        scale = 1. / float(np.linalg.norm(raw))
    else:
        harm = routing[source] * routing * gram[source] < -tolerance
        harm[source] = False
        desired = gram[source].copy()
        desired[harm] = 0.
        raw = d[source] + inverse @ (desired - gram[source])
        scale = 1. / float(np.linalg.norm(raw)) if harm.any() else 1.
    direction = raw * scale
    return desired, raw, direction, harm, scale


def saved_parent_state(cfg, arrays, rows):
    """Reconstruct stored observations only; never draw actions or build a model."""
    seeds, n, grid = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
    eta, tol = cfg["head_lr"], cfg["numeric_tolerance"]
    shape = (3, 2, 4, len(seeds), n)
    numeric_scalars = ("probabilities", "actions", "rewards", "head_steps", "own_score_steps",
        "other_score_steps", "internal_steps", "value_steps", "head_update_calls", "internal_update_calls",
        "value_update_calls", "update_factors", "expected_head_steps", "head_step_squared_l2",
        "cumulative_head_path_l2", "cumulative_head_squared_l2", "head_norms", "raw_direction_norms",
        "normalization_scales", "memory_changed", "memory_observation_calls", "memory_discoveries")
    for field in numeric_scalars:
        value = arrays["trial_" + field]
        demand(value.shape == shape and value.dtype.kind in "biuf" and np.isfinite(value).all(),
               "Saved parent finite scalar axes: " + field)
    for field in ("known_before", "known_after", "routing_signs", "harm_mask", "desired_influence", "actual_influence"):
        value = arrays["trial_" + field]
        demand(value.shape == shape + (4,) and value.dtype.kind in "biuf" and np.isfinite(value).all(),
               "Saved parent finite destination axes: " + field)
    close(arrays["model_seeds"], np.asarray(seeds), "Saved parent seed identity")
    close(arrays["checkpoints"], np.asarray(grid), "Saved parent checkpoint identity")
    close(arrays["targets"], (SIGNS > 0).astype(int), "Saved parent targets")
    source = arrays["schedules"]
    demand(source.shape == (len(seeds), n) and source.dtype.kind in "iu"
           and np.isin(source, np.arange(4)).all(), "Saved source schedule identity")
    demand(all(np.array_equal(np.sort(block), np.arange(4))
               for schedule in source for block in schedule.reshape(-1, 4)), "Saved balanced four-case blocks")
    uniforms = arrays["uniforms"]
    demand(uniforms.shape == source.shape and np.isfinite(uniforms).all()
           and np.all((uniforms >= 0.) & (uniforms < 1.)), "Saved action uniform range")
    for field in ("internal_steps", "value_steps", "internal_update_calls", "value_update_calls"):
        demand(not np.any(arrays["trial_" + field]), "No saved internal/value update: " + field)
    demand(len(rows) == int(np.prod(shape)), "Complete saved parent row count")
    indexed = {}
    for row in rows:
        key = (row["topology"], row["representation"], row["method"], int(row["seed"]), int(row["trial"]))
        demand(key[0] in TOPOLOGIES and key[1] in REPRESENTATIONS and key[2] in METHODS
               and key[3] in seeds and 1 <= key[4] <= n, "Saved parent CSV identity")
        demand(key not in indexed, "Unique saved parent row")
        indexed[key] = row
    base = np.empty(shape + (4,))
    wanted = np.empty_like(base)
    influenced = np.empty_like(base)
    scales, raw_norms = np.empty(shape), np.empty(shape)
    full = np.empty(shape, bool)
    before_logits = np.empty_like(base)
    residuals = np.zeros(shape[:-1])
    detail = np.zeros(shape[:-1] + (len(RESIDUALS),))
    def check(a, b, label):
        close(a, b, label)
        if np.asarray(a).dtype.kind not in "biu":
            return float(np.max(np.abs(np.asarray(a) - b)))
        return 0.
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            prefix = "g_" + topology + "_" + representation + "_"
            d_all = arrays[prefix + "unit_rows"]
            demand(d_all.ndim == 3 and d_all.shape[:2] == (len(seeds), 4)
                   and np.isfinite(d_all).all(), "Saved fixed feature axes")
            close(np.linalg.norm(d_all, axis=-1), np.ones((len(seeds), 4)), "Saved unit input features")
            dimension = d_all.shape[-1]
            demand(arrays["trial_directions"].shape == shape + (dimension,)
                   and np.isfinite(arrays["trial_directions"]).all(), "Saved output direction axes")
            for si, seed in enumerate(seeds):
                d = d_all[si]
                inverse = np.linalg.pinv(d, rcond=1e-12)
                gram = d @ d.T
                close(d @ inverse, np.eye(4), "Full-rank saved decision features")
                close(arrays[prefix + "base_influences"][si], gram, "Saved Gram orientation")
                close(arrays[prefix + "pseudoinverse"][si], inverse, "Saved feature pseudoinverse")
                for mi, method in enumerate(METHODS):
                    at = (ti, ri, mi, si)
                    head, known = np.zeros(dimension), np.zeros(4, np.int8)
                    path = energy = 0.
                    calls = discoveries = 0
                    first = np.full(4, n + 1, int)
                    def checkpoint(step):
                        pos = at + (grid.index(step),)
                        scores = d @ head
                        check(arrays["checkpoint_heads"][pos], head, "Saved checkpoint output weights")
                        check(arrays["checkpoint_logits"][pos], scores, "Saved checkpoint logits")
                        check(arrays["checkpoint_probabilities"][pos], sigmoid(scores), "Saved checkpoint probabilities")
                        check(arrays["checkpoint_known"][pos], known, "Saved checkpoint observed memory")
                        check(arrays["checkpoint_lookup_probabilities"][pos], (known + 1.) / 2., "Saved static lookup")
                        check(arrays["checkpoint_head_calls"][pos], np.asarray(step), "Saved checkpoint head calls")
                        check(arrays["checkpoint_memory_observation_calls"][pos], np.asarray(calls), "Saved checkpoint observations")
                        check(arrays["checkpoint_memory_discoveries"][pos], np.asarray(discoveries), "Saved checkpoint discoveries")
                        check(arrays["checkpoint_internal_calls"][pos], np.asarray(0), "Saved checkpoint internal calls")
                        check(arrays["checkpoint_value_calls"][pos], np.asarray(0), "Saved checkpoint value calls")
                        detail[at + (10,)] = max(detail[at + (10,)], float(np.max(np.abs(
                            arrays["checkpoint_logits"][pos] - scores))))
                    checkpoint(0)
                    for t in range(n):
                        pos, i = at + (t,), int(source[si, t])
                        logits = d @ head
                        before_logits[pos] = logits
                        base[pos] = gram[i]
                        probability = float(sigmoid(logits[i]))
                        a, r = arrays["trial_actions"][pos], arrays["trial_rewards"][pos]
                        demand(a in (0, 1) and r == a * SIGNS[i], "Saved obtained action/reward only")
                        demand(a == int(uniforms[si, t] < probability), "Saved action from saved uniform")
                        check(arrays["trial_known_before"][pos], known, "Saved memory continuity before")
                        old_known = known.copy()
                        changed = int(a == 1 and known[i] == 0)
                        if a:
                            demand(known[i] in (0, r), "Deterministic observed reward consistency")
                            known[i] = int(r)
                            calls += 1
                            discoveries += changed
                            if changed:
                                first[i] = t + 1
                        routing = SIGNS.astype(int) if mi == 2 else known.copy() if mi == 3 else np.zeros(4, int)
                        check(arrays["trial_known_after"][pos], known, "Saved actual observed memory after")
                        check(arrays["trial_routing_signs"][pos], routing, "Saved feedback/control/oracle routing")
                        desired, raw, direction, harm, scale = independent_direction(d, inverse, i, routing, method, tol)
                        check(arrays["trial_harm_mask"][pos], harm, "Saved identified harmful mask")
                        check(arrays["trial_desired_influence"][pos], desired, "Saved raw target influence")
                        check(arrays["trial_actual_influence"][pos], d @ direction, "Saved unit direction influence")
                        check(arrays["trial_directions"][pos], direction, "Saved unit direction vector")
                        check(arrays["trial_raw_direction_norms"][pos], np.asarray(np.linalg.norm(raw)), "Saved raw direction norm")
                        check(arrays["trial_normalization_scales"][pos], np.asarray(scale), "Saved unit scale")
                        wanted[pos], influenced[pos] = desired, d @ direction
                        scales[pos], raw_norms[pos] = scale, np.linalg.norm(raw)
                        full[pos] = np.all(known != 0)
                        factor = eta * (a - probability) * r
                        head_new = head + factor * direction
                        increment = head_new - head
                        effect = d @ increment
                        step = float(np.linalg.norm(increment))
                        path += step
                        energy += step ** 2
                        observed = {"probabilities": probability, "update_factors": factor,
                            "head_steps": step, "own_score_steps": float(effect[i]),
                            "other_score_steps": float(np.linalg.norm(np.delete(effect, i))),
                            "head_update_calls": t + 1, "expected_head_steps": abs(factor) * np.linalg.norm(direction),
                            "head_step_squared_l2": step ** 2, "cumulative_head_path_l2": path,
                            "cumulative_head_squared_l2": energy, "head_norms": float(np.linalg.norm(head_new)),
                            "memory_changed": changed, "memory_observation_calls": calls, "memory_discoveries": discoveries}
                        actual_scalars = np.asarray([arrays["trial_" + key][pos] for key in observed])
                        expected_scalars = np.asarray(list(observed.values()))
                        for key, value in observed.items():
                            if type(value) is int:
                                demand(arrays["trial_" + key].dtype.kind in "biu"
                                       and arrays["trial_" + key][pos] == value,
                                       "Saved exact scalar reconstruction: " + key)
                        residuals[at] = max(residuals[at], check(actual_scalars, expected_scalars,
                                                              "Saved scalar reconstruction"))
                        errors = (
                            np.max(np.abs(arrays[prefix + "base_influences"][si] - gram)),
                            np.max(np.abs(arrays["trial_actual_influence"][pos] - d @ direction)),
                            np.max(np.abs(arrays["trial_desired_influence"][pos] - desired)),
                            np.max(np.abs(arrays["trial_actual_influence"][pos] - desired * scale)),
                            abs(arrays["trial_raw_direction_norms"][pos] * arrays["trial_normalization_scales"][pos] - 1.),
                            abs(arrays["trial_own_score_steps"][pos] - factor * influenced[pos][i]),
                            abs(arrays["trial_other_score_steps"][pos] - np.linalg.norm(np.delete(factor * influenced[pos], i))),
                            abs(arrays["trial_head_steps"][pos] - abs(factor) * np.linalg.norm(direction)),
                            abs(arrays["trial_update_factors"][pos] - factor), abs(np.linalg.norm(direction) - 1.))
                        detail[at][:10] = np.maximum(detail[at][:10], errors)
                        row = indexed[(topology, representation, method, seed, t + 1)]
                        expected_row = {"example_index": i, "target": int(SIGNS[i] > 0), "probability_go": probability,
                            "action": int(a), "reward": int(r), "head_update_calls": t + 1,
                            "internal_update_calls": 0, "value_update_calls": 0, "head_step_l2": step,
                            "own_score_step": float(effect[i]), "other_score_step_l2": float(np.linalg.norm(np.delete(effect, i))),
                            "internal_step_l2": 0., "value_step_l2": 0., "head_nonzero_step": int(step > 0),
                            "internal_nonzero_step": 0, "value_nonzero_step": 0, "update_factor": factor,
                            "direction_l2": float(np.linalg.norm(direction)), "expected_head_step_l2": abs(factor) * np.linalg.norm(direction),
                            "head_step_squared_l2": step ** 2, "cumulative_head_path_l2": path,
                            "cumulative_head_squared_l2": energy, "head_norm": float(np.linalg.norm(head_new)),
                            "raw_direction_norm": float(np.linalg.norm(raw)), "normalization_scale": scale,
                            "memory_changed": changed, "memory_observation_calls": calls, "memory_discoveries": discoveries}
                        for field, values in (("known_before", old_known), ("known_after", known), ("routing_sign", routing),
                                              ("harm", harm), ("desired_influence", desired), ("actual_influence", d @ direction)):
                            for j, value in enumerate(values):
                                expected_row[field + "_" + str(j)] = float(value)
                        demand(all(key in row for key in expected_row), "Saved parent CSV arithmetic fields")
                        actual_row = np.asarray([float(row[key]) for key in expected_row])
                        close(actual_row, np.asarray(list(expected_row.values())), "Saved parent CSV arithmetic")
                        head = head_new
                        if t + 1 in grid:
                            checkpoint(t + 1)
                    check(arrays["first_discovery_trials"][at], first, "Saved first observed reward trials")
    return {"base_influence": base, "desired_influence": wanted, "actual_influence": influenced,
            "scales": scales, "raw_norms": raw_norms, "full_known_after": full,
            "pre_logits": before_logits, "parent_reconstruction_residuals": residuals,
            "reconstruction_residuals": detail}


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "normalization_loss_v1", "scope": "retrospective_descriptive_normalization_loss",
             "module": "drosolearn.normalization_loss", "protocol": "docs/normalization-loss-protocol.md",
             "source_result": "results/selective_spillover_v1", "topologies": list(TOPOLOGIES),
             "representations": list(REPRESENTATIONS), "methods": list(METHODS),
             "model_seeds": list(range(17401, 17411)), "training_trials": 400,
             "checkpoints": [0, 50, 100, 200, 400], "head_lr": .1, "numeric_tolerance": 1e-8,
             "neutral_tolerance": 1e-8, "labels": SIGNS.astype(int).tolist(),
             "phases": list(PHASES), "categories": list(CATEGORIES), "external_manifest": None}
    fixture = allow_fixture and cfg.get("scope") == "integration_test"
    for key, value in fixed.items():
        if fixture and key in ("scope", "model_seeds", "training_trials", "checkpoints"):
            continue
        demand(cfg.get(key) == value, "Fixed normalization analysis: " + key)
    if fixture:
        seeds, n, grid = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
        demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(set(seeds)) == len(seeds)
               and all(type(s) is int and 90001 <= s <= 90100 for s in seeds), "Invented seeds only")
        demand(type(n) is int and 4 <= n <= 16 and n % 4 == 0, "Invented trial budget only")
        demand(type(grid) is list and grid == sorted(set(grid)) and grid[0] == 0 and grid[-1] == n
               and all(type(k) is int and 0 <= k <= n for k in grid), "Invented saved checkpoints")


def expected_artifacts(cfg, parent, parent_rows):
    state = saved_parent_state(cfg, parent, parent_rows)
    factor = parent["trial_update_factors"][..., None]
    baseline = factor * state["base_influence"] * SIGNS
    desired = factor * state["desired_influence"] * SIGNS
    actual = factor * state["actual_influence"] * SIGNS
    suppression, normalization = baseline - desired, desired - actual
    # Classify the geometric cross influence, independently of reward magnitude.
    source_signs = SIGNS[parent["schedules"]][None, None, None, ..., None]
    aligned = state["base_influence"] * source_signs * SIGNS
    classes = np.zeros_like(aligned, np.int8)
    classes[aligned > cfg["neutral_tolerance"]] = 1
    classes[aligned < -cfg["neutral_tolerance"]] = -1
    for si, schedule in enumerate(parent["schedules"]):
        for t, source in enumerate(schedule):
            classes[..., si, t, int(source)] = 2
    residuals = state["reconstruction_residuals"]
    residuals[..., -1] = np.max(np.abs((baseline - actual) - (suppression + normalization)), axis=(-1, -2))
    arrays = {"baseline_margin_delta": baseline, "desired_margin_delta": desired,
              "actual_margin_delta": actual, "suppression_margin_delta": suppression,
              "normalization_margin_delta": normalization, "scales": state["scales"],
              "raw_norms": state["raw_norms"], "full_known_after": state["full_known_after"],
              "nonzero_update": parent["trial_update_factors"] != 0., "cross_class": classes,
              "reconstruction_residuals": residuals}
    demand(all(np.isfinite(value).all() for value in arrays.values()), "Finite independent analysis arrays")
    demand(float(residuals.max()) <= cfg["numeric_tolerance"], "All independent normalization numeric checks")
    return arrays, expected_rows(cfg, arrays, parent)


def category_masks(classes, schedule, known):
    own = np.eye(4, dtype=bool)[schedule]
    same = SIGNS[schedule, None] == SIGNS[None, :]
    return (own, classes == 1, classes == -1, classes == 0, (known == 0) & ~own,
            same & ~own, ~same, ~own, np.ones_like(own))


def row_metrics(arrays, at, phase_mask, category_mask):
    """Destination masses use their masks; direction statistics use trials once."""
    selected = phase_mask[:, None] & category_mask
    updated = phase_mask & arrays["nonzero_update"][at]
    scales, norms = arrays["scales"][at][phase_mask], arrays["raw_norms"][at][phase_mask]
    scaled_updates = arrays["scales"][at][updated]
    result = {"trial_count": int(phase_mask.sum()), "updated_trials": int(updated.sum()),
              "cells": int(selected.sum()), "updated_cells": int((selected & updated[:, None]).sum()),
              "scale_mean_all": float(scales.mean()) if scales.size else None,
              "scale_mean_updated": float(scaled_updates.mean()) if scaled_updates.size else None,
              "raw_norm_mean": float(norms.mean()) if norms.size else None,
              "scale_min": float(scales.min()) if scales.size else None,
              "scale_max": float(scales.max()) if scales.size else None}
    for stage in ("baseline", "desired", "actual"):
        values = arrays[stage + "_margin_delta"][at][selected]
        result[stage + "_helpful"] = float(np.maximum(values, 0.).sum())
        result[stage + "_harmful"] = float(np.maximum(-values, 0.).sum())
        result[stage + "_net"] = float(values.sum())
    for stage in ("suppression", "normalization"):
        result[stage + "_net"] = float(arrays[stage + "_margin_delta"][at][selected].sum())
    result["helpful_loss"] = result["desired_helpful"] - result["actual_helpful"]
    result["harmful_reduction"] = result["desired_harmful"] - result["actual_harmful"]
    for name in ("helpful", "harmful"):
        numerator, denominator = result["actual_" + name], result["desired_" + name]
        result[name + "_retention"] = numerator / denominator if denominator != 0. else None
    demand(tuple(result) == METRICS, "Fixed independently calculated row metric order")
    return result


def expected_rows(cfg, arrays, parent):
    rows = []
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            for mi, method in enumerate(METHODS):
                for si, seed in enumerate(cfg["model_seeds"]):
                    at = (ti, ri, mi, si)
                    full = arrays["full_known_after"][at]
                    masks = category_masks(arrays["cross_class"][at], parent["schedules"][si],
                                           parent["trial_known_after"][at])
                    for phase, phase_mask in zip(PHASES, (np.ones_like(full), ~full, full)):
                        for category, mask in zip(CATEGORIES, masks):
                            row = {"topology": topology, "representation": representation, "method": method,
                                   "seed": seed, "phase": phase, "category": category}
                            row.update(row_metrics(arrays, at, phase_mask, mask))
                            rows.append(row)
    return rows


def stats(values):
    valid = np.asarray([value for value in values if value is not None], dtype=float)
    return {"mean": float(valid.mean()) if valid.size else None,
            "sd": float(valid.std(ddof=1)) if valid.size > 1 else 0. if valid.size else None,
            "valid_count": int(valid.size), "total_count": len(values)}


def expected_summary(cfg, arrays, rows):
    families = []
    indexed = {}
    for row in rows:
        key = (row["topology"], row["representation"], row["method"], row["seed"], row["phase"], row["category"])
        demand(key not in indexed, "Unique normalization CSV rows")
        indexed[key] = row
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            for mi, method in enumerate(METHODS):
                phases = {}
                for phase in PHASES:
                    categories, pooled = {}, {}
                    for category in CATEGORIES:
                        selected = [indexed[topology, representation, method, seed, phase, category]
                                    for seed in cfg["model_seeds"]]
                        categories[category] = {metric: stats([row[metric] for row in selected]) for metric in METRICS}
                        ratios = {}
                        for name in ("helpful", "harmful"):
                            numerator = sum(row["actual_" + name] for row in selected)
                            denominator = sum(row["desired_" + name] for row in selected)
                            ratios[name + "_retention"] = numerator / denominator if denominator != 0. else None
                        pooled[category] = ratios
                    phases[phase] = {"categories": categories, "pooled_categories": pooled}
                role = "primary_descriptive" if (ti, ri, mi) == (0, 0, 3) else (
                    "privileged_diagnostic" if mi == 2 else "descriptive_control")
                families.append({"topology": topology, "representation": representation, "method": method,
                                 "role": role, "phases": phases})
    balanced = []
    for key, row in indexed.items():
        topology, representation, method, seed, phase, category = key
        for stage in ("baseline", "desired", "actual"):
            balanced.append(abs(row[stage + "_helpful"] - row[stage + "_harmful"] - row[stage + "_net"])
                            <= cfg["numeric_tolerance"])
        balanced.append(abs(row["baseline_net"] - row["actual_net"] - row["suppression_net"] - row["normalization_net"])
                        <= cfg["numeric_tolerance"])
        if phase == "all":
            for metric in ADDITIVE:
                parts = [indexed[topology, representation, method, seed, part, category][metric]
                         for part in ("partial", "full")]
                balanced.append(abs(row[metric] - sum(parts)) <= cfg["numeric_tolerance"])
        if category in ("all", "cross_all"):
            partitions = (("self", "cross_all"),) if category == "all" else (
                ("cross_helpful", "cross_harmful", "cross_neutral"), ("cross_same", "cross_opposite"))
            # Trial counts describe a phase once, so they do not partition over destinations.
            for parts in partitions:
                for metric in ADDITIVE[2:]:
                    values = [indexed[topology, representation, method, seed, phase, part][metric] for part in parts]
                    balanced.append(abs(row[metric] - sum(values)) <= cfg["numeric_tolerance"])
    residuals = arrays["reconstruction_residuals"].reshape(-1, len(RESIDUALS)).max(axis=0)
    seeds, n = len(cfg["model_seeds"]), cfg["training_trials"]
    trials = 24 * seeds * n
    return {"study_id": cfg["study_id"], "status": "completed", "scope": cfg["scope"],
            "categories": list(CATEGORIES), "phases": list(PHASES), "families": families,
            "validation": {"residual_axis": list(RESIDUALS), "max_residual_by_axis": residuals.tolist(),
                           "all_reconstruction_checks_passed": bool(np.all(residuals <= cfg["numeric_tolerance"])),
                           "all_stage_and_partition_balances_passed": bool(all(balanced))},
            "counts": {"analyzed_parent_trial_records": trials, "trial_destination_cells": trials * 4,
                       "cross_destination_cells": trials * 3, "conditions": 24, "model_seeds": seeds,
                       "measurement_records": len(rows), "new_training_trials": 0, "new_head_update_calls": 0,
                       "new_internal_update_calls": 0, "new_value_update_calls": 0, "new_random_draws": 0,
                       "fresh_model_constructions": 0},
            "interpretation": {"prior_outcomes_known": True, "confirmatory_behavioral_hypothesis": False,
                               "saved_scalar_reference_not_counterfactual_policy": True,
                               "score_loss_not_accuracy_loss": True, "phase_is_descriptive_not_causal": True,
                               "true_labels_evaluator_only": True, "unknown_category_overlaps": True,
                               "neutral_counts_do_not_zero_exact_masses": True,
                               "internal_synapses_fixed_in_parent": True, "generalization_tested": False,
                               "new_human_experiment": False, "shared_neural_mechanism_claim": False,
                               "education_effect_claim": False}}


def check_artifacts(expected, expected_rows, arrays, rows):
    demand(set(arrays) == set(expected), "Independent eleven-field analysis artifact closure")
    for key, value in expected.items():
        close(arrays[key], value, "Independent normalization array: " + key)
    demand(len(rows) == len(expected_rows), "Complete normalization CSV rows")
    for actual, wanted in zip(rows, expected_rows):
        demand(tuple(actual) == tuple(wanted), "Normalization CSV field order and closure")
        for key, value in wanted.items():
            if value is None:
                demand(actual[key] is None or actual[key] == "", "Undefined ratio/stat must be empty: " + key)
            elif type(value) is str:
                demand(actual[key] == value, "Normalization CSV fixed identity: " + key)
            else:
                number = float(actual[key])
                valid = (number.is_integer() and int(number) == value) if type(value) is int else np.isclose(
                    number, value, atol=1e-9, rtol=1e-9)
                demand(np.isfinite(number) and valid, "Normalization CSV arithmetic: " + key)


def verify_science(cfg, parent, parent_rows, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    wanted, wanted_rows = expected_artifacts(cfg, parent, parent_rows)
    check_artifacts(wanted, wanted_rows, arrays, rows)
    wanted_summary = expected_summary(cfg, wanted, wanted_rows)
    demand(wanted_summary["validation"]["all_reconstruction_checks_passed"]
           and wanted_summary["validation"]["all_stage_and_partition_balances_passed"],
           "Independent stage/phase/category checks")
    compare_tree(summary, wanted_summary, "Independent complete normalization summary")
    trials = 24 * len(cfg["model_seeds"]) * cfg["training_trials"]
    return {"arrays": len(wanted), "parent_trial_records_checked": trials,
            "trial_destination_effects_checked": trials * 4, "measurement_records_checked": len(wanted_rows),
            "all_saved_actions_rewards_memory_directions_and_checkpoints_checked": True,
            "source_destination_orientation_checked": True, "raw_target_and_unit_scale_separated": True,
            "zero_scalar_trials_have_zero_realized_effect": True,
            "discovery_trial_included_in_actual_full_known_phase": True,
            "unknown_category_is_overlapping_not_partition": True,
            "geometric_helpful_not_same_target_assumption": True,
            "exact_mass_neutral_tolerance_not_zeroed": True,
            "undefined_ratios_and_stats_not_zero_filled": True,
            "trial_direction_stats_not_destination_weighted": True,
            "same_recorded_scalar_reference_not_counterfactual_policy": True,
            "no_rng_learner_or_graph_construction_calls": True,
            "new_training_trials": 0, "new_random_draws": 0}


def load_rows(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        demand(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames),
               "Unique saved CSV field names")
        return list(reader)


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
    from drosolearn.normalization_envelope import verify_envelope, sha, write_new
    cfg, _, meta = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.publication_proof)
    validate_science(cfg)
    with np.load(args.root / cfg["source_result"] / "arrays.npz", allow_pickle=False) as saved:
        parent = {key: saved[key] for key in saved.files}
    parent_rows = load_rows(args.root / cfg["source_result"] / "measurements.csv")
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    rows = load_rows(args.result_dir / "measurements.csv")
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf8"))
    checks = verify_science(cfg, parent, parent_rows, arrays, rows, summary)
    demand(meta["counts"] == summary["counts"], "Verified metadata/summary count binding")
    report = {"status": "passed", "study_id": cfg["study_id"],
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "registration_sha256": sha(args.registration), "summary_sha256": sha(args.result_dir / "summary.json"),
              "scientific_producer_imported": False, "new_helper_imported": False, "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
