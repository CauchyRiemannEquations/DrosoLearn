"""Independent autonomous replay for full-observation objective learning.

The new learner, the old SelfBlendHead and the cone solver are never imported.
Replay foundations are copied additively from verify_self_blend_learning.py
(self-blend-learning-v1, 0d1a7abaf35e564eb397a52bb9f44a4f1b723907).
Earlier frozen independent graph/encoding routines are reused. Each actual
U/P state is certified by ambient-space dual bounds, not solver identity.
"""
from __future__ import annotations

import argparse
import csv
import gzip
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


selective = _load("verify_selective_spillover.py", "objective_frozen_independent_selective")
cone = _load("verify_cone_frontier.py", "objective_frozen_independent_cone")
budget, prior, core = selective.budget, selective.prior, selective.core
TOPOLOGIES, REPRESENTATIONS = selective.TOPOLOGIES, selective.INPUTS
TARGETS, SIGNS = selective.TARGETS, selective.SIGNS
INPUT_CATALOG = selective.INPUT_CATALOG
METHODS = ("shared", "isolated", "feedback_selective", "balanced", "gated_balanced", "gated_self_blend", "gated_uniform", "gated_probability")
demand, close, compare_tree = cone.demand, cone.close, cone.compare_tree


def sigmoid(values):
    return 1. / (1. + np.exp(-np.clip(values, -40., 40.)))


def independent_gate(known_before, method):
    """A fourth discovery can open only a later trial's balanced branch."""
    observed = np.asarray(known_before)
    demand(method in METHODS and observed.shape == (4,) and observed.dtype.kind in "biu"
           and np.isin(observed, (-1, 0, 1)).all(), "Explicit observed memory BEFORE the update")
    full = bool(np.all(observed != 0))
    if method == "gated_balanced":
        mode = "balanced" if full else "isolated"
    elif method == "gated_self_blend":
        mode = "blend" if full else "isolated"
    elif method == "gated_uniform":
        mode = "uniform" if full else "isolated"
    elif method == "gated_probability":
        mode = "probability" if full else "isolated"
    else:
        mode = method
    return {"full_known_before": full, "mode": mode, "balanced_branch": mode in ("balanced", "blend", "uniform", "probability")}


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "objective_learning_v1", "scope": "prospective_observed_objective_learning",
        "module": "drosolearn.objective_learning", "protocol": "docs/objective-learning-protocol.md",
        "topologies": list(TOPOLOGIES), "representations": list(REPRESENTATIONS), "methods": list(METHODS),
        "model_seeds": list(range(17801, 17811)), "training_trials": 400, "checkpoints": [0, 50, 100, 200, 400],
        "head_lr": .1, "rng_namespace": 202610204, "svd_rcond": 1e-12, "numeric_tolerance": 1e-8,
        "exact_condition_max": 1e6, "acquisition_threshold": .8, "acquisition_gain": .25, "class_threshold": .7,
        "population_pass_count": 8, "effect_threshold": .05, "positive_seed_count": 8,
        "normalization": "per_input_initial_fixed_MBON_gap_unit_L2",
        "budget_convention": "unit_directions_common_eta_ceiling_with_observed_reward_memory",
        "balanced_own_objective_weight": .5, "balanced_help_objective_weight": .5,
        "gate_switch_rule": "known_before_four_then_own_uniform_or_current_probability_objective", "protection_nonworse_seed_count": 10,
        "probability_gradient_domain":"all_abs_own_current_logits_strictly_less_than_40",
        "gain_controls":["gated_balanced","isolated","gated_self_blend"],
        "state_effect_threshold":.05,"state_positive_seed_count":8,
        "measurement_storage":"lossless_deterministic_gzip_csv_crlf",
        "baseline_tag":"probability-alignment-v1","baseline_commit":"9159aa764d44183c21468974ba3b0f208d676731",
        "blend_isolated_weight": .75, "blend_balanced_weight": .25,
        "protection_absolute_tolerance": 1e-8,
        "unknown_harm_endpoint": "sum_negative_correct_probability_change_on_unknown_other_destinations_over_all_400_trials",
        "input_catalog": INPUT_CATALOG, "graph": "data/flywire_783_mb_left_kc512_s701", "external_manifest": None}
    demand(isinstance(cfg, dict), "Independent balanced configuration object")
    fixture = cfg.get("scope") == "integration_test"
    demand(not fixture or allow_fixture, "Invented fixtures cannot enter the actual experiment")
    exempt = {"scope", "model_seeds", "training_trials", "checkpoints", "population_pass_count",
              "positive_seed_count", "protection_nonworse_seed_count", "state_positive_seed_count"} if fixture else set()
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
        demand(cfg["population_pass_count"] == cfg["positive_seed_count"] == cfg["protection_nonworse_seed_count"] == cfg['state_positive_seed_count'] == len(seeds),
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


def independent_mix(rows, source, observed, isolated, balanced, tolerance=1e-8):
    """Normalize a fixed convex mix and bound feature influences, not rewards."""
    d, memory, i, b = (np.asarray(x) for x in (rows, observed, isolated, balanced))
    demand(d.ndim == 2 and d.shape[0] == 4 and source in range(4)
        and memory.shape == (4,) and memory.dtype.kind in "biu" and np.isin(memory, (-1, 1)).all()
        and i.shape == b.shape == (d.shape[1],) and all(np.isfinite(x).all() for x in (d, i, b)),
        "Fully observed finite self-blend geometry")
    demand(abs(np.linalg.norm(i)-1.) <= tolerance and abs(np.linalg.norm(b)-1.) <= tolerance,
        "The two blend ingredients must each have unit direction norm")
    fi, fb = d @ i, d @ b
    other = np.arange(4) != source
    align = memory[source] * memory
    demand(fi[source] > 0. and np.max(np.abs(fi[other])) <= tolerance
        and np.min(align * fb) >= -tolerance, "Isolated ingredient and safe balanced ingredient")
    raw = .75 * i + .25 * b
    length = float(np.linalg.norm(raw))
    demand(.5-tolerance <= length <= 1. + tolerance, "Convex mixture norm stays between one half and one")
    direction = raw / length
    influence = d @ direction
    own_floor = .75 * fi[source]
    help_floor = .25 * align[other] * fb[other]
    safety = max(-float(np.min(align * influence)), 0.)
    own_residual = max(own_floor - float(influence[source]), 0.)
    help_residual = max(float(np.max(help_floor - align[other] * influence[other])), 0.)
    demand(max(safety, own_residual, help_residual) <= tolerance,
        "Normalized mixture preserves known safety and unit-influence floors")
    residuals = np.asarray([np.linalg.norm(direction - raw / length), abs(np.linalg.norm(direction)-1.),
        safety, own_residual, help_residual])
    return {"direction": direction, "raw_norm": length, "normalization_scale": 1. / length,
        "influence": influence, "isolated_influence": fi, "balanced_influence": fb,
        "safety_residual": safety, "own_floor_residual": own_residual, "help_floor_residual": help_residual,
        "residuals": residuals}


def independent_objective_problem(rows,source,observed,logits,kind):
    """The U/P selector sees own obtained signs and its own current state only."""
    d,k,z=map(np.asarray,(rows,observed,logits))
    demand(d.ndim==2 and d.shape[0]==4 and d.dtype.kind in 'fiu' and np.isfinite(d).all()
        and type(source) is int and source in range(4) and k.shape==(4,) and k.dtype.kind in 'iu'
        and np.isin(k,(-1,1)).all() and z.shape==(4,) and z.dtype.kind in 'fiu' and np.isfinite(z).all()
        and kind in ('uniform','probability'),'Finite full own observations and explicit objective kind')
    constraints=k[source]*k[:,None]*d
    if kind=='probability':
        demand(np.all(np.abs(z)<40.),'State probability gradient strictly inside clipping domain')
        p=sigmoid(z);objective=(p*(1-p))@constraints/4.
    else:objective=constraints.sum(0)/4.
    return {'constraints':constraints,'objective':objective}


def certify_objective(rows,source,observed,logits,kind,direction,projection,multipliers,tolerance=1e-8):
    """Independent ambient active-face dual optimum for this actual U/P state."""
    problem=independent_objective_problem(rows,source,observed,logits,kind)
    a,c=problem['constraints'],problem['objective']
    w,v,mu=map(np.asarray,(direction,projection,multipliers))
    demand(w.shape==v.shape==(rows.shape[1],) and mu.shape==(4,)
        and all(value.dtype.kind in 'fiu' and np.isfinite(value).all() for value in (w,v,mu)),
        'Actual objective witness finite axes')
    witness=cone.validate_dual_witness(c,a,w,mu,tolerance)
    upper,_=cone.independent_support_bound(c,a,tolerance)
    demand(abs(float(c@w)-upper)<=tolerance and abs(witness['dual_upper_bound']-upper)<=tolerance,
        'Actual U/P independently achieves global support bound')
    certificate=np.asarray([float(c@w),np.linalg.norm(v),witness['dual_upper_bound'],np.linalg.norm(w),
        max(-float(np.min(a@w)),0.),np.max(np.abs(v-c-a.T@mu)),np.max(np.abs(mu*(a@v))),
        abs(np.linalg.norm(v)-float(c@w))])
    demand(abs(np.linalg.norm(w)-1.)<=tolerance and np.max(certificate[4:])<=tolerance,
        'Actual U/P nonzero unit direction and complete support certificate')
    return {**problem,'certificate':certificate,'optimum':upper}


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
    gated_full_before = np.all(before[:, :, 4:8] != 0, axis=-1)
    routes = (np.array_equal(route[:, :, 2:4], after[:, :, 2:4])
              and np.array_equal(route[:, :, 4:8][gated_full_before], after[:, :, 4:8][gated_full_before]))
    control = (route[:, :, :2] == 0).all() and (route[:, :, 4:8][~gated_full_before] == 0).all()
    policy = np.array_equal(actions, arrays["uniforms"][None, None, None] < arrays["trial_probabilities"])
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            for si, rows in enumerate(arrays["g_" + topology + "_" + representation + "_unit_rows"]):
                for mi in range(len(METHODS)):
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
            "controls_and_gated_isolation_ignore_memory_for_direction": bool(control),
            "feedback_balanced_and_gate_balanced_routing_equals_observed_after": bool(routes),
            "head_policy_does_not_directly_lookup_memory": bool(policy),
            "static_lookup_not_autonomous_experiment": bool(np.array_equal(
                arrays["checkpoint_lookup_probabilities"], (remembered + 1.) / 2.)),
            "memory_writes_equal_first_discoveries": bool(writes),
            "no_oracle_learner_present": True}


def gate_checks(cfg, arrays):
    before, after = arrays["trial_known_before"], arrays["trial_known_after"]
    n = cfg["training_trials"]
    counts = np.count_nonzero(before, axis=-1)
    full = counts == 4
    modes = np.broadcast_to(np.arange(8)[None, None, :, None, None], counts.shape).copy()
    modes[:, :, 4] = np.where(full[:, :, 4], 3, 1)
    modes[:, :, 5] = np.where(full[:, :, 5], 4, 1)
    modes[:, :, 6] = np.where(full[:, :, 6], 5, 1)
    modes[:, :, 7] = np.where(full[:, :, 7], 6, 1)
    first_full = np.where(np.any(np.all(after != 0, axis=-1), axis=-1),
                          np.argmax(np.all(after != 0, axis=-1), axis=-1) + 1, n + 1)
    first_mode = np.full_like(first_full, n + 1)
    first_mode[:, :, 4] = np.minimum(first_full[:, :, 4] + 1, n + 1)
    first_blend = np.full_like(first_full, n + 1)
    first_blend[:, :, 5] = np.minimum(first_full[:, :, 5] + 1, n + 1)
    first_uniform=np.full_like(first_full,n+1);first_uniform[:,:,6]=np.minimum(first_full[:,:,6]+1,n+1)
    first_probability=np.full_like(first_full,n+1);first_probability[:,:,7]=np.minimum(first_full[:,:,7]+1,n+1)
    events = np.zeros_like(counts)
    for ti in range(3):
        for ri in range(2):
            for si in range(len(cfg["model_seeds"])):
                for mi, first in ((4, first_mode), (5, first_blend),(6,first_uniform),(7,first_probability)):
                    step = int(first[ti, ri, mi, si])
                    if step <= n: events[ti, ri, mi, si, step - 1] = 1
    modes_valid = np.array_equal(arrays["trial_direction_modes"], modes)
    before_valid = (np.array_equal(arrays["trial_gate_known_counts_before"], counts)
                    and np.array_equal(arrays["trial_gate_full_before"], full))
    first_mode_valid = np.array_equal(arrays["first_balanced_mode_trials"], first_mode)
    first_blend_valid = np.array_equal(arrays["first_blend_mode_trials"], first_blend)
    switches = np.array_equal(arrays["trial_gate_switch_events"], events)
    same_discovery = all(np.array_equal(first_full[:, :, mi], first_full[:, :, 1])
        and np.array_equal(arrays["first_discovery_trials"][:, :, mi], arrays["first_discovery_trials"][:, :, 1])
        for mi in (4, 5, 6, 7))
    prefix = True
    pair_prefix = True
    factual_names = ("probabilities", "actions", "rewards", "head_steps", "own_score_steps", "other_score_steps",
        "internal_steps", "value_steps", "head_update_calls", "internal_update_calls", "value_update_calls",
        "update_factors", "expected_head_steps", "head_step_squared_l2", "cumulative_head_path_l2",
        "cumulative_head_squared_l2", "head_norms", "raw_direction_norms", "normalization_scales",
        "memory_changed", "memory_observation_calls", "memory_discoveries", "known_before", "known_after",
        "directions", "actual_influence", "desired_influence", "harm_mask", "direction_residuals",
        "direction_residuals_applicable", "unknown_other_mask_before", "unknown_other_mask_after",
        "correct_probability_delta", "aligned_margin_delta", "unknown_other_negative_correct_probability_sum",
        "unknown_other_positive_correct_probability_sum", "unknown_other_negative_margin_sum", "unknown_other_exposure_count",
        "routing_signs", "case_indices", "direction_modes", "gate_known_counts_before", "gate_full_before", "gate_switch_events",
        "blend_applicable", "blend_isolated_directions", "blend_balanced_directions", "blend_raw_norm", "blend_residuals",
        "objective_applicable","objective_logits_before","objective_vectors","objective_kind","objective_case_indices")
    discovery_isolated = True
    for ti in range(3):
        for ri in range(2):
            for si in range(len(cfg["model_seeds"])):
                for mi in (4, 5, 6, 7):
                    end = min(int(first_full[ti, ri, mi, si]), n)
                    for name in factual_names:
                        value = arrays["trial_" + name]
                        left, right = value[ti, ri, mi, si, :end], value[ti, ri, 1, si, :end]
                        prefix &= left.dtype == right.dtype and left.tobytes() == right.tobytes()
                        if mi >= 5:
                            previous = value[ti, ri, 4, si, :end]
                            pair_prefix &= left.dtype == previous.dtype and left.tobytes() == previous.tobytes()
                    if first_full[ti, ri, mi, si] <= n:
                        discovery_isolated &= arrays["trial_direction_modes"][ti, ri, mi, si, end - 1] == 1
                    for ki, step in enumerate(cfg["checkpoints"]):
                        if step <= end:
                            for name in ("heads", "logits", "probabilities", "known", "lookup_probabilities", "head_calls",
                                         "internal_calls", "value_calls", "memory_observation_calls", "memory_discoveries"):
                                left, right = arrays["checkpoint_" + name][ti, ri, mi, si, ki], arrays["checkpoint_" + name][ti, ri, 1, si, ki]
                                prefix &= left.dtype == right.dtype and left.tobytes() == right.tobytes()
                                if mi >= 5:
                                    previous = arrays["checkpoint_" + name][ti, ri, 4, si, ki]
                                    pair_prefix &= left.dtype == previous.dtype and left.tobytes() == previous.tobytes()
    partial_cases = all(np.all(arrays["trial_case_indices"][:, :, mi][~full[:, :, mi]] == -1) for mi in (4, 5, 6, 7))
    case_keys = arrays["case_keys"]
    keys_valid = (case_keys.ndim == 2 and case_keys.shape[1] == 9
                  and np.isin(case_keys[:, 3], (3, 4, 5)).all()
                  and np.all(case_keys[np.isin(case_keys[:, 3], (4, 5)), 5:] != 0))
    objective_keys=arrays['objective_case_keys']
    keys_valid &= (objective_keys.ndim==2 and objective_keys.shape[1]==10
        and np.isin(objective_keys[:,3],(6,7)).all() and np.isin(objective_keys[:,6:],(-1,1)).all())
    partial_cases &= all(np.all(arrays['trial_objective_case_indices'][:,:,mi][~full[:,:,mi]]==-1) for mi in (6,7))
    no_go = True
    for mi, late_mode in ((4, 3), (5, 4), (6, 5), (7, 6)):
        early_nogo = (~full[:, :, mi]) & (arrays["trial_actions"][:, :, mi] == 0)
        late_nogo = full[:, :, mi] & (arrays["trial_actions"][:, :, mi] == 0)
        no_go &= (np.all(arrays["trial_direction_modes"][:, :, mi][early_nogo] == 1)
             and np.allclose(np.linalg.norm(arrays["trial_directions"][:, :, mi][early_nogo], axis=-1), 1., atol=1e-8)
             and np.all(arrays["trial_direction_modes"][:, :, mi][late_nogo] == late_mode)
             and np.all(arrays["trial_directions"][:, :, mi][late_nogo] == 0.)
             and np.all(arrays["trial_case_indices"][:, :, mi][late_nogo] == -1))
        no_go &= np.all(arrays['trial_objective_case_indices'][:,:,mi][late_nogo]==-1)
    return {"mode_uses_known_before_only": bool(modes_valid),
        "fourth_discovery_update_is_isolated": bool(discovery_isolated),
        "first_balanced_mode_follows_full_discovery": bool(first_mode_valid),
        "first_blend_mode_follows_full_discovery": bool(first_blend_valid),
        "first_uniform_mode_follows_full_discovery":bool(np.array_equal(arrays['first_uniform_mode_trials'],first_uniform)),
        "first_probability_mode_follows_full_discovery":bool(np.array_equal(arrays['first_probability_mode_trials'],first_probability)),
        "exact_isolated_prefix_through_full_discovery": bool(prefix),
        "exact_gated_balanced_prefix_through_full_discovery": bool(pair_prefix),
        "gated_and_isolated_discovery_trials_equal": bool(same_discovery),
        "partial_gated_has_no_balanced_case": bool(partial_cases),
        "only_full_observation_states_in_gated_case_cache": bool(keys_valid),
        "full_NoGo_has_zero_direction_and_no_case": bool(no_go),
        "gate_switch_once_at_first_full_mode": bool(switches),
        "all_methods_record_memory_before_gate": bool(before_valid)}


def expected_artifacts(cfg, base_graph, actual):
    """Reconstruct fixed features, fresh streams, obtained observations and heads."""
    seeds, n, grid = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
    arrays = {key: value.copy() for key, value in prior.independent_catalog_arrays().items()}
    shape = (3, 2, len(METHODS), len(seeds), n)
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
    for name in ("direction_modes", "gate_known_counts_before", "gate_switch_events"):
        arrays["trial_" + name] = np.zeros(shape, np.int8)
    arrays["trial_gate_full_before"] = np.zeros(shape, bool)
    arrays["first_balanced_mode_trials"] = np.full(shape[:-1], n + 1, np.int32)
    arrays["first_blend_mode_trials"] = np.full(shape[:-1], n + 1, np.int32)
    for name in ('first_uniform_mode_trials','first_probability_mode_trials'):
        arrays[name]=np.full(shape[:-1],n+1,np.int32)
    arrays['trial_objective_applicable']=np.zeros(shape,bool)
    arrays['trial_objective_logits_before']=np.zeros(shape+(4,))
    arrays['trial_objective_vectors']=np.zeros(shape+(dimension,))
    arrays['trial_objective_kind']=np.full(shape,-1,np.int8)
    arrays['trial_objective_case_indices']=np.full(shape,-1,np.int32)
    arrays["trial_blend_applicable"] = np.zeros(shape, bool)
    arrays["trial_blend_isolated_directions"] = np.zeros(shape + (dimension,))
    arrays["trial_blend_balanced_directions"] = np.zeros(shape + (dimension,))
    arrays["trial_blend_raw_norm"] = np.zeros(shape)
    arrays["trial_blend_residuals"] = np.zeros(shape + (5,))
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
    objective_keys=[];objective_cases={}
    objective_values={key:[] for key in ('objectives','logits','directions','projections','multipliers','certificates')}
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
                        gate = independent_gate(before_known, method)
                        mode = gate["mode"]
                        mode_index = ("shared", "isolated", "feedback_selective", "balanced", "blend", "uniform", "probability").index(mode)
                        switch = int(method == "gated_balanced" and mode == "balanced"
                                     and arrays["first_balanced_mode_trials"][ti, ri, mi, si] == n + 1)
                        if switch: arrays["first_balanced_mode_trials"][ti, ri, mi, si] = t + 1
                        blend_switch = int(method == "gated_self_blend" and mode == "blend"
                            and arrays["first_blend_mode_trials"][ti, ri, mi, si] == n + 1)
                        if blend_switch: arrays["first_blend_mode_trials"][ti, ri, mi, si] = t + 1
                        switch |= blend_switch
                        for gated_method,late_mode,first_field in (
                            ('gated_uniform','uniform','first_uniform_mode_trials'),
                            ('gated_probability','probability','first_probability_mode_trials')):
                            objective_switch=int(method==gated_method and mode==late_mode and arrays[first_field][ti,ri,mi,si]==n+1)
                            if objective_switch:arrays[first_field][ti,ri,mi,si]=t+1
                            switch|=objective_switch
                        route = known.copy() if mode in ("feedback_selective", "balanced", "blend", "uniform", "probability") else np.zeros(4, np.int8)
                        case_index, applicable = -1, mode not in ("balanced", "blend", "uniform", "probability")
                        objective_index=-1;objective_kind=-1;objective_cert_residual=0.
                        if mode in ('uniform','probability'):
                            direction=np.zeros(dimension);projection=np.zeros(dimension)
                            if action:
                                objective_kind=0 if mode=='uniform' else 1
                                address=source if mode=='uniform' else t
                                key=(si,ti,ri,mi,address,source,*map(int,known))
                                if key not in objective_cases:
                                    objective_index=len(objective_keys)
                                    demand(objective_index<len(actual['objective_case_keys']),'Complete actual objective-state evidence')
                                    close(actual['objective_case_keys'][objective_index],np.asarray(key),'First encounter objective key includes method and P trial')
                                    direction=actual['objective_case_directions'][objective_index]
                                    projection=actual['objective_case_projections'][objective_index]
                                    mu=actual['objective_case_multipliers'][objective_index]
                                    certified=certify_objective(d,source,known,before_logits,mode,direction,projection,mu,cfg['numeric_tolerance'])
                                    close(actual['objective_case_objectives'][objective_index],certified['objective'],'Own observed/current-state objective')
                                    close(actual['objective_case_logits'][objective_index],before_logits,'First encounter own objective state')
                                    close(actual['objective_case_certificates'][objective_index],certified['certificate'],'Independent U/P certificate arithmetic')
                                    objective_cases[key]=objective_index;objective_keys.append(key)
                                    for field,value in [('objectives',certified['objective']),('logits',before_logits),('directions',direction),
                                        ('projections',projection),('multipliers',mu),('certificates',certified['certificate'])]:
                                        objective_values[field].append(value.copy())
                                objective_index=objective_cases[key]
                                direction=objective_values['directions'][objective_index];projection=objective_values['projections'][objective_index]
                                oat=(ti,ri,mi,si,t)
                                arrays['trial_objective_applicable'][oat]=True
                                arrays['trial_objective_logits_before'][oat]=before_logits
                                arrays['trial_objective_vectors'][oat]=independent_objective_problem(d,source,known,before_logits,mode)['objective']
                                arrays['trial_objective_kind'][oat]=objective_kind
                                arrays['trial_objective_case_indices'][oat]=objective_index
                                objective_cert_residual=float(np.max(objective_values['certificates'][objective_index][4:]))
                            raw_norm=float(np.linalg.norm(projection))
                            record={'direction':direction,'raw_norm':raw_norm,'normalization_scale':1./raw_norm if raw_norm else 0.,
                                'harm_mask':np.zeros(4,bool),'desired_influence':d@projection,'influence':d@direction}
                            residuals=np.zeros(7)
                        elif mode in ("balanced", "blend"):
                            direction, projection, mu = np.zeros(dimension), np.zeros(dimension), np.zeros(4)
                            if action:
                                key = (si, ti, ri, mi, source, *map(int, known))
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
                            if mode == "blend" and action:
                                isolated = selective.independent_evidence(geometry, source, np.zeros(4, np.int8), "isolated")["direction"]
                                mixed = independent_mix(d, source, known, isolated, direction, cfg["numeric_tolerance"])
                                blend_at = (ti, ri, mi, si, t)
                                arrays["trial_blend_applicable"][blend_at] = True
                                arrays["trial_blend_isolated_directions"][blend_at] = isolated
                                arrays["trial_blend_balanced_directions"][blend_at] = direction
                                arrays["trial_blend_raw_norm"][blend_at] = mixed["raw_norm"]
                                arrays["trial_blend_residuals"][blend_at] = mixed["residuals"]
                                record = {**mixed, "harm_mask": np.zeros(4, bool),
                                    "desired_influence": d @ (.75 * isolated + .25 * direction)}
                            else:
                                raw_norm = float(np.linalg.norm(projection))
                                scale = 1. / raw_norm if raw_norm > 0. else 0.
                                record = {"direction": direction, "raw_norm": raw_norm, "normalization_scale": scale,
                                    "harm_mask": np.zeros(4, bool), "desired_influence": d @ projection,
                                    "influence": d @ direction}
                            residuals = np.zeros(7)
                        else:
                            record = selective.independent_evidence(geometry, source, route, mode)
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
                        arrays["trial_direction_modes"][at] = mode_index
                        arrays["trial_gate_known_counts_before"][at] = np.count_nonzero(before_known)
                        arrays["trial_gate_full_before"][at] = gate["full_known_before"]
                        arrays["trial_gate_switch_events"][at] = switch
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
                        row.update({"direction_mode": mode_index, "gate_known_count_before": int(np.count_nonzero(before_known)),
                                    "gate_full_before": int(gate["full_known_before"]), "gate_switch_event": switch})
                        row.update({"blend_applicable": int(arrays["trial_blend_applicable"][at]),
                            "blend_raw_norm": float(arrays["trial_blend_raw_norm"][at]),
                            "blend_residual_max": float(np.max(arrays["trial_blend_residuals"][at]))})
                        row.update({'objective_applicable':int(arrays['trial_objective_applicable'][at]),
                            'objective_kind':int(arrays['trial_objective_kind'][at]),'objective_case_index':objective_index})
                        row['objective_vector_l2']=float(np.linalg.norm(arrays['trial_objective_vectors'][at]))
                        row['objective_certificate_residual_max']=objective_cert_residual
                        for j in range(4):row['objective_logit_before_'+str(j)]=float(arrays['trial_objective_logits_before'][at+(j,)])
                        rows.append(row)
                        if t + 1 in grid: checkpoint(t + 1)
    demand(len(keys) == len(actual["case_keys"]), "No unused or duplicate balanced state cache")
    arrays.update({name: np.stack(values) for name, values in static.items()})
    for name, value, width in (("case_keys", keys, 9), ("case_directions", directions, dimension),
                              ("case_projections", projections, dimension), ("case_multipliers", multipliers, 4),
                              ("case_certificates", certificates, 8)):
        arrays[name] = np.asarray(value, dtype=np.int64 if name == "case_keys" else float).reshape(-1, width)
    demand(len(objective_keys)==len(actual['objective_case_keys']),'No unused or duplicate U/P case evidence')
    arrays['objective_case_keys']=np.asarray(objective_keys,np.int64).reshape(-1,10)
    for field,width in [('objectives',dimension),('logits',4),('directions',dimension),('projections',dimension),('multipliers',4),('certificates',8)]:
        arrays['objective_case_'+field]=np.asarray(objective_values[field],float).reshape(-1,width)
    return arrays, rows


def check_artifacts(actual, wanted, rows, wanted_rows):
    demand(set(actual) == set(wanted), "Independent complete balanced array closure")
    for key, value in wanted.items():
        close(actual[key], value, "Independent actual balanced trajectory: " + key)
    compare_rows(rows, wanted_rows)


def blend_checks(cfg, arrays):
    """Rebuild all blend diagnostics from fixed features and certified B cases."""
    active = np.zeros_like(arrays["trial_actions"], bool)
    active[:, :, 5] = arrays["trial_gate_full_before"][:, :, 5] & (arrays["trial_actions"][:, :, 5] == 1)
    demand(np.array_equal(active, (arrays["trial_direction_modes"] == 4) & (arrays["trial_actions"] == 1)),
        "Blend active only on the new method's full-before actual Go trials")
    residuals = np.zeros(active.shape + (5,))
    norms = np.zeros(active.shape)
    tol = cfg["numeric_tolerance"]
    for ti, top in enumerate(TOPOLOGIES):
        for ri, rep in enumerate(REPRESENTATIONS):
            prefix = "g_" + top + "_" + rep + "_"
            for si in range(len(cfg["model_seeds"])):
                d = arrays[prefix + "unit_rows"][si]
                for t in np.flatnonzero(active[ti, ri, 5, si]):
                    source = int(arrays["schedules"][si, t])
                    at = (ti, ri, 5, si, int(t))
                    index = int(arrays["trial_case_indices"][at])
                    demand(0 <= index < len(arrays["case_keys"]), "An active blend has an underlying B case")
                    close(arrays["case_keys"][index], np.asarray((si,ti,ri,5,source,*map(int,arrays["trial_known_after"][at]))),
                        "Underlying B key belongs to this learner's actual observation state")
                    isolated = arrays[prefix + "isolated_directions"][si, source]
                    balanced = arrays["case_directions"][index]
                    record = independent_mix(d,source,arrays["trial_known_after"][at],isolated,balanced,tol)
                    close(arrays["trial_blend_isolated_directions"][at], isolated, "Fixed isolated blend ingredient")
                    close(arrays["trial_blend_balanced_directions"][at], balanced, "Certified B blend ingredient")
                    close(arrays["trial_directions"][at], record["direction"], "Actual normalized self-blend direction")
                    close(arrays["trial_actual_influence"][at], record["influence"], "Blend's actual fixed-feature influence")
                    close(arrays["trial_desired_influence"][at], d @ (.75 * isolated + .25 * balanced),
                        "Desired influence is the raw direction mixture, not a new optimization objective")
                    residuals[at], norms[at] = record["residuals"], record["raw_norm"]
    close(arrays["trial_blend_residuals"], residuals, "Independent five blend residuals and inactive zero padding")
    close(arrays["trial_blend_raw_norm"], norms, "Independent mixture norm and inactive zero padding")
    residual = float(np.max(residuals))
    flags = {"only_actual_post_Go_has_blend_diagnostics": bool(np.array_equal(active,arrays["trial_blend_applicable"])),
        "all_unit_mix_and_unit_score_retention_residuals_pass": bool(residual <= tol),
        "inactive_blend_padding_zero": all(not np.any(arrays["trial_blend_"+name][~active]) for name in
            ("isolated_directions","balanced_directions","raw_norm","residuals")),
        "raw_mix_norm_in_half_to_one": bool(np.all((norms[active] >= .5-tol) & (norms[active] <= 1.+tol)))}
    return {**flags,"max_residual":residual,"residual_names":["exact_mix_l2","unit_norm_abs","known_safety_deficit",
        "own_floor_deficit","help_floor_deficit"]}


def objective_checks(cfg,a):
    """Rebuild current own-state objectives, exact cache addresses and padding."""
    tol=cfg['numeric_tolerance'];active=np.zeros_like(a['trial_actions'],bool)
    active[:,:,6:]=np.all(a['trial_known_before'][:,:,6:]!=0,axis=-1)&(a['trial_actions'][:,:,6:]==1)
    flags={'only_actual_full_Go_has_objective_diagnostics':bool(np.array_equal(active,a['trial_objective_applicable'])),
        'inactive_objective_padding_zero_and_sentinel':bool(not np.any(a['trial_objective_logits_before'][~active])
            and not np.any(a['trial_objective_vectors'][~active]) and np.all(a['trial_objective_kind'][~active]==-1)
            and np.all(a['trial_objective_case_indices'][~active]==-1)),
        'current_own_preupdate_state_and_observed_objective_checked':True,
        'uniform_source_cache_probability_each_actual_trial':True,
        'all_probability_logits_strictly_unclipped':True,'all_objective_global_support_certificates_passed':True}
    used=set()
    for ti,top in enumerate(TOPOLOGIES):
        for ri,rep in enumerate(REPRESENTATIONS):
            for si in range(len(cfg['model_seeds'])):
                d=a['g_'+top+'_'+rep+'_unit_rows'][si]
                for mi,kind in ((6,'uniform'),(7,'probability')):
                    at=(ti,ri,mi,si);increments=a['trial_update_factors'][at][:,None]*a['trial_directions'][at]
                    heads=np.vstack((np.zeros((1,d.shape[1])),np.cumsum(increments,axis=0)))
                    for t0 in np.flatnonzero(active[at]):
                        t=int(t0);ix=at+(t,);source=int(a['schedules'][si,t]);z=d@heads[t]
                        known=a['trial_known_before'][ix];problem=independent_objective_problem(d,source,known,z,kind)
                        index=int(a['trial_objective_case_indices'][ix]);demand(0<=index<len(a['objective_case_keys']),'An active objective has an actual certificate')
                        used.add(index);address=source if mi==6 else t
                        flags['uniform_source_cache_probability_each_actual_trial'] &= np.array_equal(a['objective_case_keys'][index],
                            np.asarray((si,ti,ri,mi,address,source,*map(int,known))))
                        flags['current_own_preupdate_state_and_observed_objective_checked'] &= bool(a['trial_objective_kind'][ix]==mi-6
                            and np.max(np.abs(a['trial_objective_logits_before'][ix]-z))<=tol
                            and np.max(np.abs(a['trial_objective_vectors'][ix]-problem['objective']))<=tol
                            and np.max(np.abs(a['objective_case_objectives'][index]-problem['objective']))<=tol
                            and np.max(np.abs(a['trial_directions'][ix]-a['objective_case_directions'][index]))<=tol)
                        if mi==7:flags['all_probability_logits_strictly_unclipped'] &= bool(np.all(np.abs(z)<40.))
    keys=a['objective_case_keys'];ucount=int(np.count_nonzero(keys[:,3]==6));pcount=int(np.count_nonzero(keys[:,3]==7))
    flags['uniform_source_cache_probability_each_actual_trial'] &= bool(pcount==np.count_nonzero(active[:,:,7])
        and ucount<=24*len(cfg['model_seeds']) and used==set(range(len(keys))) and len({tuple(key) for key in keys})==len(keys))
    certmax=float(np.max(a['objective_case_certificates'][:,4:],initial=0.))
    flags['all_objective_global_support_certificates_passed']=bool(certmax<=tol)
    return {**{key:bool(value) for key,value in flags.items()},'max_certificate_residual':certmax}


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
                        "first_balanced_mode_trial": int(arrays["first_balanced_mode_trials"][at]),
                        "first_blend_mode_trial": int(arrays["first_blend_mode_trials"][at]),
                        "first_uniform_mode_trial":int(arrays['first_uniform_mode_trials'][at]),
                        "first_probability_mode_trial":int(arrays['first_probability_mode_trials'][at]),
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
                    "role": "co_primary" if (ti, ri)==(0,0) and mi>=6 else "control",
                    "mean_final_expected_accuracy": float(values.mean()),
                    "sd_final_expected_accuracy": float(values.std(ddof=1)) if s > 1 else 0.,
                    "acquisition_pass_seeds": int(learned[ti, ri, mi].sum()),
                    "mean_cumulative_head_path_l2": float(lengths[ti, ri, mi].mean()),
                    "mean_unknown_other_negative_correct_probability_sum": float(unknown_damage[ti, ri, mi].mean()),
                    "mean_unknown_other_exposure_count": float(exposures[ti, ri, mi].mean()), "seeds": individuals})
            contrasts.append({"topology": topology, "representation": representation,
                "descriptive_only": (ti, ri) != (0, 0),
                'endpoint_differences':{METHODS[mi]:{name:(outcomes[ti,ri,mi]-outcomes[ti,ri,METHODS.index(name)]).tolist()
                    for name in cfg['gain_controls']} for mi in (6,7)},
                'probability_minus_uniform_by_seed':(outcomes[ti,ri,7]-outcomes[ti,ri,6]).tolist()})
    gates = gate_checks(cfg, arrays)
    demand(all(gates.values()), "Independent before-memory gate and isolated-prefix integrity")
    blend = blend_checks(cfg, arrays)
    demand(all(value for value in blend.values() if isinstance(value,bool)), "Independent unit mixture and retention gates")
    objective=objective_checks(cfg,arrays)
    demand(all(value for value in objective.values() if type(value) is bool),'Independent U/P diagnostics')
    safe = bool(not known_damage[:, :, 1:].any())
    numeric = bool(static_valid and bounded.all() and step_residual <= tol and control_residual <= tol
                   and certificate_residual <= tol and safe and all(information.values()) and objective['max_certificate_residual']<=tol)
    demand(numeric, "Independent numerical, known-safety, information and common-cost integrity gates")
    gains={};acquisitions={};protections={}
    for mi in (6,7):
        method=METHODS[mi];comparisons={}
        for name in cfg['gain_controls']:
            difference=outcomes[0,0,mi]-outcomes[0,0,METHODS.index(name)]
            positives=int(np.count_nonzero(difference>0.))
            comparisons[name]={'mean_effect':float(difference.mean()),'positive_seeds':positives,
                'practical_advantage_pass':bool(difference.mean()>=cfg['effect_threshold'] and positives>=cfg['positive_seed_count'])}
        gains[method]={'supported':bool(numeric and all(record['practical_advantage_pass'] for record in comparisons.values())),
            'endpoint_comparisons':comparisons,'all_numeric_and_information_gates_passed':numeric,'requires_acquisition':False}
        acquired_count=int(learned[0,0,mi].sum())
        acquisitions[method]={'supported':bool(numeric and acquired_count>=cfg['population_pass_count']),
            'acquisition_pass_seeds':acquired_count,'all_numeric_and_information_gates_passed':numeric,'requires_gain':False}
        comparisons={}
        for name,index in (('gated_balanced',4),('isolated',1)):
            difference=unknown_damage[0,0,mi]-unknown_damage[0,0,index]
            nonworse=int(np.count_nonzero(difference<=tol))
            comparisons[name]={'mean_effect':float(difference.mean()),'nonworsening_seeds':nonworse,
                'all_seeds_nonworsening':bool(nonworse>=cfg['protection_nonworse_seed_count'])}
        maximum=float(unknown_damage[0,0,mi].max())
        protections[method]={'supported':bool(numeric and maximum<=cfg['protection_absolute_tolerance']
                and all(record['all_seeds_nonworsening'] for record in comparisons.values())),
            'maximum_primary_seed_unknown_damage':maximum,'absolute_tolerance':cfg['protection_absolute_tolerance'],
            'comparisons':comparisons,'operational_protection_expected_by_design':True,
            'novel_learning_mechanism_demonstrated':False,'all_numeric_and_information_gates_passed':numeric}
    state_difference=outcomes[0,0,7]-outcomes[0,0,6]
    active=arrays['trial_objective_applicable'];kind=arrays['trial_objective_kind'];keys=arrays['objective_case_keys']
    ucount=int(np.count_nonzero(keys[:,3]==6));pcount=int(np.count_nonzero(keys[:,3]==7))
    demand(len(rows)==48*s*n and all(len(row)==96 for row in rows),'Complete independent 96-column autonomous trial inventory')
    return {'study_id':cfg['study_id'],'status':'completed','methods':list(METHODS),'families':families,
        'contrasts':contrasts,'geometry':geometries,'H_gain':gains,'H_acquisition':acquisitions,'H_protection':protections,
        'H_state':{'supported':bool(numeric and state_difference.mean()>=cfg['state_effect_threshold']
                and np.count_nonzero(state_difference>0.)>=cfg['state_positive_seed_count']),
            'comparison':'gated_probability_minus_gated_uniform','mean_effect':float(state_difference.mean()),
            'positive_seeds':int(np.count_nonzero(state_difference>0.)),'all_numeric_and_information_gates_passed':numeric,
            'requires_acquisition':False},
        'gate_validation':gates,'blend_validation':blend,'objective_validation':objective,
        'validation':{'all_numeric_and_information_gates_passed':numeric,'all_common_cost_bounds_passed':bool(bounded.all()),
            'all_static_geometry_gates_passed':bool(static_valid),'all_control_direction_gates_passed':bool(control_residual<=tol),
            'all_balanced_case_certificates_passed':bool(certificate_residual<=tol),'all_nonshared_known_destinations_safe':safe,
            'head_step_expected_residual_max':step_residual,'control_direction_residual_max':control_residual,
            'balanced_certificate_residual_max':certificate_residual,'unknown_other_masks_equal_before_after':bool(equal_masks),
            'source_inclusive_unknown_count_drop_equals_discoveries':bool(count_drop)},
        'information_validation':information,
        'budget_validation':{'common_step_ceiling':eta,'common_path_ceiling':n*eta,'common_squared_step_sum_ceiling':n*eta**2,
            'actual_cumulative_cost_equalized':False,'extra_solver_and_information_processing_cost_equalized':False},
        'counts':{'training_feedback_trials':len(rows),'static_probe_cases':int(np.prod(p.shape)),
            'model_initializations':48*s,'fixed_feature_constructions':6*s,'model_seeds':s,
            'head_update_calls':int(arrays['trial_head_update_calls'][...,-1].sum()),
            'head_nonzero_steps':int(np.count_nonzero(arrays['trial_head_steps'])),
            'internal_update_calls':0,'value_update_calls':0,'internal_nonzero_steps':0,'value_nonzero_steps':0,
            'auxiliary_memory_observation_calls':int(arrays['trial_memory_observation_calls'][...,-1].sum()),
            'auxiliary_memory_discoveries':int(arrays['trial_memory_changed'].sum()),
            'preflight_direction_evaluations':6*s*324,'runtime_selective_direction_evaluations':6*s*n,
            'balanced_direction_requests':int(np.count_nonzero(np.isin(arrays['trial_direction_modes'],(3,4))&(arrays['trial_actions']==1))),
            'always_balanced_direction_requests':int(arrays['trial_actions'][:,:,3].sum()),
            'gated_balanced_direction_requests':int(np.count_nonzero((arrays['trial_direction_modes'][:,:,4]==3)&(arrays['trial_actions'][:,:,4]==1))),
            'gated_isolated_direction_evaluations':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,4]==1)),
            'gated_balanced_mode_trials':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,4]==3)),
            'gated_gate_activations':int(arrays['trial_gate_switch_events'][:,:,4].sum()),
            'self_blend_gate_activations':int(arrays['trial_gate_switch_events'][:,:,5].sum()),
            'self_blend_direction_requests':int(arrays['trial_blend_applicable'].sum()),
            'self_blend_mode_trials':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,5]==4)),
            'self_blend_isolated_direction_evaluations':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,5]==1)),
            'balanced_unique_solver_cases':len(arrays['case_keys']),
            'balanced_NoGo_solver_skips':int(np.count_nonzero(np.isin(arrays['trial_direction_modes'],(3,4))&(arrays['trial_actions']==0))),
            'uniform_direction_requests':int(np.count_nonzero(active&(kind==0))),
            'probability_direction_requests':int(np.count_nonzero(active&(kind==1))),
            'uniform_unique_solver_cases':ucount,'probability_unique_solver_cases':pcount,
            'objective_unique_solver_cases':ucount+pcount,'total_unique_solver_cases':len(arrays['case_keys'])+ucount+pcount,
            'uniform_gate_activations':int(arrays['trial_gate_switch_events'][:,:,6].sum()),
            'probability_gate_activations':int(arrays['trial_gate_switch_events'][:,:,7].sum()),
            'uniform_mode_trials':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,6]==5)),
            'probability_mode_trials':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,7]==6)),
            'uniform_isolated_direction_evaluations':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,6]==1)),
            'probability_isolated_direction_evaluations':int(np.count_nonzero(arrays['trial_direction_modes'][:,:,7]==1)),
            'objective_NoGo_solver_skips':int(np.count_nonzero(np.isin(arrays['trial_direction_modes'],(5,6))&(arrays['trial_actions']==0))),
            'unknown_other_destination_exposures':int(exposures.sum()),
            'source_inclusive_unknown_exposures_before_feedback':int(unknown_before.sum()),
            'source_inclusive_unknown_exposures_after_feedback':int(unknown_after.sum()),
            'lookup_static_cases':int(np.prod(p.shape)),'lookup_autonomous_training_trials':0,
            'numeric_arrays':len(arrays),'measurement_columns':96},
        'resources':{'head_coordinates':int(arrays['checkpoint_heads'].shape[-1]),'observed_sign_memory_cells_per_learner':4,
            'privileged_target_signs_given_to_any_learner':0,'balanced_solver_cache_only_actual_observations':True,
            'balanced_solver_calls_equal_unique_actual_Go_states':True,'processing_cost_equalized':False,
            'uniform_cache_by_actual_source_and_full_observed_signs':True,'probability_fresh_case_per_actual_full_Go':True},
        'interpretation':{'artificial_history_buffer':True,'internal_synapses_fixed':True,'identity_dictionary_is_supplied':True,
            'observed_reward_memory_is_additional_learning_state':True,'true_labels_used_for_unknown_harm_evaluation_only':True,
            'any_learner_uses_hidden_targets_for_routing':False,'prior_failure_or_seed_discarded':False,
            'unknown_means_reward_unobserved_not_novel_input':True,'generalization_tested':False,
            'probability_direction_tangent_not_finite_global_optimum':True,'own_actual_current_state_used_for_each_policy':True,
            'new_human_experiment':False,'shared_neural_mechanism_claim':False,'education_effect_claim':False}
        }



def verify_against_graph(cfg, graph, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    wanted, wanted_rows = expected_artifacts(cfg, graph, arrays)
    check_artifacts(arrays, wanted, rows, wanted_rows)
    expected = expected_summary(cfg, wanted, wanted_rows)
    compare_tree(summary, expected, "Independent complete balanced learning summary")
    return {"arrays": len(wanted), "independently_replayed_training_actions": len(wanted_rows),
        "model_seeds": len(cfg["model_seeds"]), "conditions": 48,
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
        "all_candidate_gain_acquisition_protection_and_P_minus_U_state_hypotheses_recomputed":True,
        "unique_objective_states_independently_reoptimized":len(wanted["objective_case_keys"]),
        "unique_uniform_states_independently_reoptimized":int(np.count_nonzero(wanted["objective_case_keys"][:,3]==6)),
        "unique_probability_states_independently_reoptimized":int(np.count_nonzero(wanted["objective_case_keys"][:,3]==7)),
        "each_P_case_uses_own_actual_preupdate_head_and_new_actual_Go_trial":True,
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


def load_rows(path):
    """Lossless compressed CSV keeps all trials, columns, signs and precision."""
    with gzip.open(path,'rt',encoding='utf8',newline='') as handle:
        reader=csv.DictReader(handle)
        demand(reader.fieldnames is not None and len(reader.fieldnames)==len(set(reader.fieldnames)),
            'Unique compressed CSV columns')
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
    from drosolearn.objective_envelope import verify_envelope, sha, write_new
    cfg, _, metadata = verify_envelope(args.root, args.config, args.registration,
        args.result_dir, args.publication_proof)
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    rows=load_rows(args.result_dir/'measurements.csv.gz')
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
