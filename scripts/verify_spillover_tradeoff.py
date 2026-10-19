"""Independent finite replay of the preregistered spillover compromise.

The current scientific producer and new direction helper are never imported.
Frozen independent raw graph, input and SVD arithmetic is reused explicitly.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "spillover_frozen_independent_budget", Path(__file__).with_name("verify_update_budget.py"))
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)
prior = budget.prior
core = budget.core
TOPOLOGIES, INPUTS, TARGETS = budget.TOPOLOGIES, budget.INPUTS, budget.TARGETS
INPUT_CATALOG = budget.INPUT_CATALOG
ALPHAS = (0., .25, .5, .75, 1.)


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "spillover_tradeoff_v1", "scope": "prospective_spillover_tradeoff_diagnosis",
             "module": "drosolearn.spillover_tradeoff", "protocol": "docs/spillover-tradeoff-protocol.md",
             "topologies": list(TOPOLOGIES), "representations": list(INPUTS), "alphas": list(ALPHAS),
             "primary_alpha": .5, "model_seeds": list(range(17301, 17311)), "training_trials": 400,
             "checkpoints": [0, 50, 100, 200, 400], "head_lr": .1, "rng_namespace": 202610195,
             "svd_rcond": 1e-12, "numeric_tolerance": 1e-8, "exact_condition_max": 1e6,
             "acquisition_threshold": .8, "acquisition_gain": .25, "class_threshold": .7,
             "population_pass_count": 8, "effect_threshold": .05, "positive_seed_count": 8,
             "normalization": "per_input_initial_fixed_MBON_gap_unit_L2", "input_catalog": INPUT_CATALOG,
             "budget_convention": "all_mixture_directions_unit_L2_common_eta_step_ceiling",
             "graph": "data/flywire_783_mb_left_kc512_s701", "external_manifest": None}
    fixture = allow_fixture and cfg.get("scope") == "integration_test"
    exempt = {"scope", "model_seeds", "training_trials", "checkpoints", "population_pass_count", "positive_seed_count"} if fixture else set()
    for key, value in fixed.items():
        if key not in exempt:
            core.demand(cfg.get(key) == value, "Fixed independent spillover diagnosis: " + key)
    if fixture:
        seeds, n, grid = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
        core.demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(set(seeds)) == len(seeds)
                    and all(type(s) is int and 90001 <= s <= 90100 for s in seeds), "Invented diagnostic seeds only")
        core.demand(type(n) is int and 4 <= n <= 16 and n % 4 == 0, "Invented diagnostic budget")
        core.demand(grid == sorted(set(grid)) and grid[0] == 0 and grid[-1] == n
                    and all(type(step) is int and 0 <= step <= n and step % 4 == 0 for step in grid), "Invented diagnostic checkpoints")
        core.demand(cfg["population_pass_count"] == cfg["positive_seed_count"] == len(seeds), "Invented pass counts")


def independent_tradeoff_report(feature_rows, rcond=1e-12, tolerance=1e-8,
                               exact_condition_max=1e6):
    """Independently normalize fixed mixtures and reproduce all gate residuals."""
    base = budget.independent_budget_report(feature_rows, rcond, tolerance, exact_condition_max)
    d, v = base["directions"][0], base["directions"][2]
    alpha = np.asarray(ALPHAS)
    cosine = np.sum(d * v, axis=-1)
    mixture = (1. - alpha[:, None, None]) * d[None] + alpha[:, None, None] * v[None]
    mixture_norms = np.linalg.norm(mixture, axis=-1)
    core.demand(np.isfinite(mixture_norms).all() and np.all(mixture_norms > 0),
                "Finite nonzero fixed mixtures required; no fallback")
    directions = mixture / mixture_norms[..., None]
    directions[0], directions[-1] = d, v
    influence = np.einsum("aic,jc->aij", directions, d)
    lengths = np.linalg.norm(directions, axis=-1)
    own = np.diagonal(influence, axis1=1, axis2=2).copy()
    off = influence.copy()
    off[:, np.arange(4), np.arange(4)] = 0.
    a = alpha[:, None]
    analytic_norms = np.sqrt((1. - a) ** 2 + a ** 2 + 2 * a * (1. - a) * cosine)
    analytic_own = ((1. - a) + a * cosine) / analytic_norms
    unit = float(np.max(np.abs(lengths - 1.)))
    conditional = float(np.max(np.abs(lengths - lengths[:1])))
    reconstruct = float(np.max(np.abs(directions * mixture_norms[..., None] - mixture)))
    norm_formula = float(np.max(np.abs(mixture_norms - analytic_norms)))
    own_formula = float(np.max(np.abs(own - analytic_own)))
    cosine_formula = float(np.max(np.abs(cosine - 1. / base["direction_norms"][1])))
    lower = float(np.max(np.maximum(.5 - mixture_norms ** 2, 0.)))
    monotonic = float(np.max(np.maximum(np.diff(own, axis=0), 0.)))
    cosine_cap = float(np.max(np.maximum(cosine - 1., 0.)))
    residuals = (unit, conditional, reconstruct, norm_formula, own_formula,
                 cosine_formula, lower, monotonic, cosine_cap)
    finite = all(np.isfinite(item).all() for item in (directions, influence, lengths, own, analytic_norms, analytic_own))
    valid = bool(base["numeric_budget_qualified"] and finite and np.all(cosine > 0.)
                 and max(residuals) <= tolerance)
    return {"base_budget": base, "alphas": alpha,
            "directions": directions, "influences": influence,
            "direction_norms": lengths, "own_influence": own,
            "spillover_squared": np.sum(off ** 2, axis=-1), "own_response_per_unit_l2": own / lengths,
            "mixture_norms": mixture_norms, "endpoint_cosines": cosine,
            "analytic_mixture_norms": analytic_norms, "analytic_own_influence": analytic_own,
            "unit_norm_residual": unit, "conditional_l2_match_residual": conditional,
            "normalization_reconstruction_residual": reconstruct,
            "mixture_norm_formula_residual": norm_formula, "own_formula_residual": own_formula,
            "endpoint_cosine_formula_residual": cosine_formula,
            "mixture_norm_squared_lower_bound_residual": lower, "own_monotonic_residual": monotonic,
            "endpoint_cosine_upper_bound_residual": cosine_cap, "numeric_spillover_qualified": valid,
            "projected_l2_exact_isolation_qualified": bool(base["projected_l2_exact_isolation_qualified"]),
            "alpha_is_fraction_of_spillover_removed": False, "all_unit_directions_optimized": False,
            "equal_conditional_scalar_l2_only": True, "equal_realized_cumulative_l2_claimed": False,
            "tolerance": float(tolerance)}


class IndependentHead:
    """Observed-action and obtained-reward learner with actual step costs."""
    def __init__(self, feature_rows, alpha, rate=.1):
        core.demand(alpha in ALPHAS, "Registered fixed mixing coefficient required")
        self.geometry = independent_tradeoff_report(feature_rows)
        self.d = self.geometry["base_budget"]["base_geometry"]["unit_rows"]
        self.direction = self.geometry["directions"][ALPHAS.index(alpha)]
        self.head = np.zeros(self.d.shape[1])
        self.rate, self.calls = rate, 0
        self.path_l2 = self.squared_l2 = 0.

    def probabilities(self):
        return 1. / (1. + np.exp(-np.clip(self.d @ self.head, -40., 40.)))

    def observe(self, index, action, reward):
        core.demand(type(index) is int and 0 <= index < 4 and type(action) is int and action in (0, 1)
                    and type(reward) is int and reward in (-1, 0, 1) and (action == 1 or reward == 0),
                    "Current identity, chosen action and obtained reward only")
        p = float(self.probabilities()[index])
        factor = self.rate * (action - p) * reward
        before = self.head.copy()
        self.head += factor * self.direction[index]
        difference = self.head - before
        norm = float(np.linalg.norm(difference))
        self.calls += 1
        self.path_l2 += norm
        self.squared_l2 += norm * norm
        scores = self.d @ difference
        return {"factor": factor, "head_step_l2": norm, "head_step_squared_l2": norm * norm,
                "own_score_step": float(scores[index]),
                "other_score_step_l2": float(np.linalg.norm(np.delete(scores, index)))}
def expected_artifacts(cfg, base_graph):
    """Rebuild all fixed features, autonomous feedback and actual step costs."""
    seeds, n, checkpoints = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
    arrays = {key: value.copy() for key, value in prior.independent_catalog_arrays().items()}
    shape = (3, 2, 5, len(seeds), n)
    integer_fields = {"actions", "rewards", "head_update_calls", "internal_update_calls", "value_update_calls"}
    fields = ("probabilities", "actions", "rewards", "head_steps", "own_score_steps", "other_score_steps",
              "internal_steps", "value_steps", "head_update_calls", "internal_update_calls", "value_update_calls",
              "update_factors", "expected_head_steps", "head_step_squared_l2", "cumulative_head_path_l2",
              "cumulative_head_squared_l2", "head_norms")
    for key in fields:
        arrays["trial_" + key] = np.zeros(shape, dtype=int if key in integer_fields else float)
    checkpoint_shape = shape[:-1] + (len(checkpoints),)
    arrays["checkpoint_logits"] = np.empty(checkpoint_shape + (4,))
    arrays["checkpoint_probabilities"] = np.empty(checkpoint_shape + (4,))
    arrays["checkpoint_heads"] = np.empty(checkpoint_shape + (base_graph["n_mbon"],))
    for field in ("head", "internal", "value"):
        arrays["checkpoint_" + field + "_calls"] = np.zeros(checkpoint_shape, int)
    arrays["schedules"] = np.empty((len(seeds), n), int)
    arrays["uniforms"] = np.empty((len(seeds), n))
    arrays["model_seeds"], arrays["checkpoints"] = np.asarray(seeds), np.asarray(checkpoints)
    arrays["alphas"] = np.asarray(ALPHAS)
    static, rows = {}, []
    for si, seed in enumerate(seeds):
        schedule_rng = np.random.default_rng(np.random.SeedSequence([202610195, seed, 0]))
        schedule = np.concatenate([schedule_rng.permutation(4) for _ in range(n // 4)])
        uniforms = np.random.default_rng(np.random.SeedSequence([202610195, seed, 1])).random(n)
        arrays["schedules"][si], arrays["uniforms"][si] = schedule, uniforms
        for ti, topology in enumerate(TOPOLOGIES):
            offset = 110000 if topology == "rewired" else 120000 if topology == "ann" else 0
            graph = core.transform_graph(base_graph, topology, seed + offset)
            for ri, representation in enumerate(INPUTS):
                evidence = prior.feature_evidence(graph, seed, arrays["inputs"][ri])
                witness = prior.initial_witness(evidence)
                tradeoff = independent_tradeoff_report(evidence["differences"])
                core.demand(tradeoff["numeric_spillover_qualified"], "Registered fixed spillover geometry failed")
                prefix = topology + "_" + representation + "_"
                numeric_tradeoff = prior.numeric_fields({key: value for key, value in tradeoff.items() if key != "base_budget"})
                numeric_tradeoff.update(prior.numeric_fields({key: value for key, value in tradeoff["base_budget"].items() if key != "updates"}, "base_budget_"))
                for kind, record in (("w_", witness), ("t_", numeric_tradeoff)):
                    for key, value in record.items():
                        static.setdefault(kind + prefix + key, []).append(np.asarray(value))
                absolute_limit = np.empty((5, 4))
                for ui in range(5):
                    for tested in range(4):
                        absolute_limit[ui, tested] = (n // 4) * cfg["head_lr"] * sum(
                            abs(tradeoff["influences"][ui, source, tested]) for source in range(4))
                probability_limit = 1. / (1. + np.exp(-np.clip(absolute_limit, -40., 40.)))
                static.setdefault("c_" + prefix + "absolute_logit_ceiling", []).append(absolute_limit)
                static.setdefault("c_" + prefix + "correct_probability_ceiling", []).append(probability_limit)
                d = tradeoff["base_budget"]["base_geometry"]["unit_rows"]
                for ui, alpha in enumerate(ALPHAS):
                    directions = tradeoff["directions"][ui]
                    direction_norms = tradeoff["direction_norms"][ui]
                    head = np.zeros(graph["n_mbon"])
                    calls, path_l2, squared_l2 = 0, 0., 0.
                    def checkpoint(step):
                        at = (ti, ri, ui, si, checkpoints.index(step))
                        logits = d @ head
                        arrays["checkpoint_logits"][at] = logits
                        arrays["checkpoint_probabilities"][at] = 1. / (1. + np.exp(-np.clip(logits, -40., 40.)))
                        arrays["checkpoint_heads"][at] = head
                        arrays["checkpoint_head_calls"][at] = calls
                    checkpoint(0)
                    for trial, example in enumerate(schedule):
                        index = int(example)
                        target = int(TARGETS[index])
                        probability = float(1. / (1. + np.exp(-np.clip((d @ head)[index], -40., 40.))))
                        action = int(uniforms[trial] < probability)
                        reward = (1 if target else -1) if action else 0
                        factor = .1 * (action - probability) * reward
                        before = head.copy()
                        head += factor * directions[index]
                        calls += 1
                        difference = head - before
                        step = float(np.linalg.norm(difference))
                        score_step = d @ difference
                        own_score = float(score_step[index])
                        other_score = float(np.linalg.norm(np.delete(score_step, index)))
                        square = step * step
                        path_l2 += step
                        squared_l2 += square
                        head_norm = float(np.linalg.norm(head))
                        predicted_step = float(abs(factor) * direction_norms[index])
                        at = (ti, ri, ui, si, trial)
                        for key, value in (("probabilities", probability), ("actions", action), ("rewards", reward),
                                           ("head_steps", step), ("own_score_steps", own_score),
                                           ("other_score_steps", other_score), ("head_update_calls", calls),
                                           ("update_factors", factor), ("expected_head_steps", predicted_step),
                                           ("head_step_squared_l2", square), ("cumulative_head_path_l2", path_l2),
                                           ("cumulative_head_squared_l2", squared_l2), ("head_norms", head_norm)):
                            arrays["trial_" + key][at] = value
                        rows.append({"topology": topology, "representation": representation, "alpha": alpha,
                                     "seed": seed, "trial": trial + 1, "example_index": index, "target": target,
                                     "probability_go": probability, "action": action, "reward": reward,
                                     "head_update_calls": calls, "internal_update_calls": 0, "value_update_calls": 0,
                                     "head_step_l2": step, "own_score_step": own_score,
                                     "other_score_step_l2": other_score, "internal_step_l2": 0., "value_step_l2": 0.,
                                     "head_nonzero_step": int(step > 0), "internal_nonzero_step": 0, "value_nonzero_step": 0,
                                     "update_factor": factor, "direction_l2": float(direction_norms[index]),
                                     "expected_head_step_l2": predicted_step, "head_step_squared_l2": square,
                                     "cumulative_head_path_l2": path_l2, "cumulative_head_squared_l2": squared_l2,
                                     "head_norm": head_norm})
                        if trial + 1 in checkpoints: checkpoint(trial + 1)
                for key in ("weights", "encoder", "bias", "probe_kc", "probe_mbon"):
                    static.setdefault("finalfixed_" + prefix + key, []).append(witness[key].copy())
    arrays.update({key: np.stack(value) for key, value in static.items()})
    return arrays, rows


def check_artifacts(expected, expected_rows, arrays, rows):
    core.demand(set(arrays) == set(expected), "Independent spillover numeric field closure differs")
    for key, value in expected.items():
        actual = np.asarray(arrays[key])
        core.demand(actual.dtype.kind in "biuf" and np.isfinite(actual).all(), "Finite spillover artifact: " + key)
        if value.dtype.kind in "biu":
            core.demand(actual.dtype.kind in "biu" and np.array_equal(actual, value), "Independent exact artifact: " + key)
        else:
            core.array_close(actual, value, "Independent full spillover replay: " + key, atol=1e-9)
    core.demand(len(rows) == len(expected_rows), "Complete independent spillover row count")
    for index, (actual, expected_row) in enumerate(zip(rows, expected_rows)):
        if all(type(value) is str for value in actual.values()):
            core.check_csv_row(actual, expected_row, "Update-budget trial " + str(index))
        else:
            core.compare_tree(actual, expected_row, "Update-budget trial " + str(index))


def expected_summary(cfg, arrays, rows):
    probability = arrays["checkpoint_probabilities"]
    correct = probability * TARGETS + (1. - probability) * (1 - TARGETS)
    first, last = correct[..., 0, :], correct[..., -1, :]
    score = last.mean(axis=-1)
    gain = score - first.mean(axis=-1)
    positive_class, negative_class = last[..., [0, 1]].mean(-1), last[..., [2, 3]].mean(-1)
    qualifies = ((score >= cfg["acquisition_threshold"]) & (gain >= cfg["acquisition_gain"])
                 & (positive_class >= cfg["class_threshold"]) & (negative_class >= cfg["class_threshold"]))
    max_step = arrays["trial_head_steps"].max(-1)
    paths = arrays["trial_cumulative_head_path_l2"][..., -1]
    energies = arrays["trial_cumulative_head_squared_l2"][..., -1]
    eta, n, tolerance = cfg["head_lr"], cfg["training_trials"], cfg["numeric_tolerance"]
    endpoint_norms = np.sqrt(np.sum(arrays["checkpoint_heads"][..., -1, :] ** 2, axis=-1))
    energy_valid = energies <= n * eta ** 2 + tolerance
    endpoint_valid = endpoint_norms <= paths + tolerance
    bounded = ((max_step <= eta + tolerance) & (paths <= eta * n + tolerance)
               & energy_valid & endpoint_valid)
    all_bounded = bool(bounded.all())
    expected_residual = float(np.max(np.abs(arrays["trial_head_steps"] - arrays["trial_expected_head_steps"])))
    primary = ALPHAS.index(cfg["primary_alpha"])
    families, geometry, contrasts, ceiling_checks = [], [], [], []
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(INPUTS):
            prefix = "t_" + topology + "_" + representation + "_"
            bound_prefix = "c_" + topology + "_" + representation + "_"
            ceiling = arrays[bound_prefix + "correct_probability_ceiling"]
            ceiling_checks.append(bool(np.all(last[ti, ri] <= np.transpose(ceiling, (1, 0, 2)) + tolerance)))
            geometry_rows = []
            for si, seed in enumerate(cfg["model_seeds"]):
                geometry_rows.append({
                    "seed": seed,
                    "retained_rank": int(arrays[prefix + "base_budget_base_geometry_unit_diagnostics_retained_rank"][si]),
                    "endpoint_exact_isolation_qualified": bool(arrays[prefix + "projected_l2_exact_isolation_qualified"][si]),
                    "numeric_spillover_qualified": bool(arrays[prefix + "numeric_spillover_qualified"][si]),
                    "direction_norms": arrays[prefix + "direction_norms"][si].tolist(),
                    "own_influence": arrays[prefix + "own_influence"][si].tolist(),
                    "spillover_squared": arrays[prefix + "spillover_squared"][si].tolist(),
                    "mixture_norms": arrays[prefix + "mixture_norms"][si].tolist(),
                    "endpoint_cosines": arrays[prefix + "endpoint_cosines"][si].tolist(),
                    "conditional_l2_match_residual": float(arrays[prefix + "conditional_l2_match_residual"][si]),
                    "absolute_logit_ceiling_by_alpha": arrays[bound_prefix + "absolute_logit_ceiling"][si].tolist(),
                    "correct_probability_ceiling_by_alpha": ceiling[si].tolist()})
            geometry.append({"topology": topology, "representation": representation, "seeds": geometry_rows})
            for ai, alpha in enumerate(ALPHAS):
                individuals = []
                for si, seed in enumerate(cfg["model_seeds"]):
                    individuals.append({
                        "seed": seed, "initial_expected_accuracy": float(first[ti, ri, ai, si].mean()),
                        "final_expected_accuracy": float(score[ti, ri, ai, si]), "gain": float(gain[ti, ri, ai, si]),
                        "class_1_expected_accuracy": float(positive_class[ti, ri, ai, si]),
                        "class_0_expected_accuracy": float(negative_class[ti, ri, ai, si]),
                        "acquisition_pass": bool(qualifies[ti, ri, ai, si]),
                        "final_head_l2": float(np.linalg.norm(arrays["checkpoint_heads"][ti, ri, ai, si, -1])),
                        "cumulative_head_path_l2": float(paths[ti, ri, ai, si]),
                        "cumulative_head_squared_l2": float(energies[ti, ri, ai, si]),
                        "max_head_step_l2": float(max_step[ti, ri, ai, si]),
                        "within_all_common_cost_bounds": bool(bounded[ti, ri, ai, si])})
                outcomes = score[ti, ri, ai]
                families.append({
                    "topology": topology, "representation": representation, "alpha": alpha,
                    "role": "primary" if (ti, ri, ai) == (0, 0, primary) else "predeclared_control",
                    "mean_final_expected_accuracy": float(outcomes.mean()),
                    "sd_final_expected_accuracy": float(outcomes.std(ddof=1)) if len(outcomes) > 1 else 0.,
                    "acquisition_pass_seeds": int(np.count_nonzero(qualifies[ti, ri, ai])),
                    "mean_cumulative_head_path_l2": float(paths[ti, ri, ai].mean()),
                    "mean_cumulative_head_squared_l2": float(energies[ti, ri, ai].mean()), "seeds": individuals})
            contrasts.append({
                "topology": topology, "representation": representation, "primary_alpha": cfg["primary_alpha"],
                "primary_minus_shared_by_seed": (score[ti, ri, primary] - score[ti, ri, 0]).tolist(),
                "primary_minus_isolated_by_seed": (score[ti, ri, primary] - score[ti, ri, -1]).tolist(),
                "primary_minus_shared_actual_path_cost_by_seed": (paths[ti, ri, primary] - paths[ti, ri, 0]).tolist(),
                "primary_minus_isolated_actual_path_cost_by_seed": (paths[ti, ri, primary] - paths[ti, ri, -1]).tolist(),
                "descriptive_only": (ti, ri) != (0, 0)})
    static_valid = all(item["numeric_spillover_qualified"] for group in geometry for item in group["seeds"])
    ceiling_valid = bool(all(ceiling_checks))
    numeric_valid = bool(static_valid and all_bounded and expected_residual <= tolerance and ceiling_valid)
    endpoint_effects = {}
    for name, endpoint in (("shared", 0), ("isolated", len(ALPHAS) - 1)):
        effect = score[0, 0, primary] - score[0, 0, endpoint]
        positive_count = int(np.count_nonzero(effect > 0))
        endpoint_effects[name] = {"mean_effect": float(effect.mean()), "positive_seeds": positive_count,
            "practical_advantage_pass": bool(effect.mean() >= cfg["effect_threshold"]
                                               and positive_count >= cfg["positive_seed_count"])}
    tradeoff = bool(numeric_valid and all(item["practical_advantage_pass"] for item in endpoint_effects.values()))
    passes = int(np.count_nonzero(qualifies[0, 0, primary]))
    seed_count = len(cfg["model_seeds"])
    coordinates = int(arrays["checkpoint_heads"].shape[-1])
    return {
        "study_id": cfg["study_id"], "status": "completed", "alphas": list(ALPHAS), "primary_alpha": cfg["primary_alpha"],
        "families": families, "geometry": geometry, "contrasts": contrasts,
        "H_tradeoff": {"supported": tradeoff, "primary_alpha": cfg["primary_alpha"],
                      "endpoint_comparisons": endpoint_effects, "numeric_spillover_qualified_all": numeric_valid,
                      "requires_acquisition": False},
        "H_acquisition": {"supported": bool(numeric_valid and passes >= cfg["population_pass_count"]),
                          "primary_alpha": cfg["primary_alpha"], "acquisition_pass_seeds": passes,
                          "numeric_spillover_qualified_all": numeric_valid, "requires_tradeoff_advantage": False},
        "budget_validation": {"conditional_scalar_matching_only": True, "actual_cumulative_cost_equalized": False,
                              "common_unit_direction_step_ceiling": eta, "common_unit_direction_path_ceiling": n * eta,
                              "common_unit_direction_squared_step_sum_ceiling": n * eta ** 2,
                              "squared_step_sum_valid_all": bool(energy_valid.all()),
                              "head_endpoint_within_path_all": bool(endpoint_valid.all()),
                              "common_bounds_valid_all": all_bounded, "head_step_expected_residual_max": expected_residual,
                              "all_static_numeric_gates_passed": bool(static_valid),
                              "analytic_probability_ceiling_valid_all": ceiling_valid},
        "counts": {"training_feedback_trials": len(rows), "static_probe_cases": int(np.prod(probability.shape)),
                   "model_initializations": 30 * seed_count, "fixed_feature_constructions": 6 * seed_count,
                   "model_seeds": seed_count, "head_update_calls": int(arrays["trial_head_update_calls"][..., -1].sum()),
                   "head_nonzero_steps": int(np.count_nonzero(arrays["trial_head_steps"])),
                   "internal_update_calls": 0, "value_update_calls": 0, "internal_nonzero_steps": 0, "value_nonzero_steps": 0},
        "resources": {"head_coordinates": coordinates, "saved_direction_coefficients_per_group": 5 * 4 * coordinates,
                      "direction_solver_SVD_per_group": 1, "diagnostic_SVD_per_group": 2,
                      "total_producer_SVD_calls": 18 * seed_count,
                      "exact_actual_cumulative_budget_matching_claimed": False, "extra_solver_cost_equalized": False},
        "interpretation": {"alpha_is_fraction_of_spillover_removed": False,
                           "posthoc_best_alpha_is_confirmatory": False, "all_unit_directions_optimized": False,
                           "artificial_history_buffer": True, "internal_synapses_fixed": True,
                           "identity_dictionary_is_supplied": True, "prior_failure_or_seed_discarded": False,
                           "generalization_tested": False, "new_human_experiment": False,
                           "shared_neural_mechanism_claim": False, "education_effect_claim": False}}


def verify_against_graph(cfg, graph, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    expected, expected_rows = expected_artifacts(cfg, graph)
    check_artifacts(expected, expected_rows, arrays, rows)
    core.compare_tree(summary, expected_summary(cfg, arrays, rows), "Independent complete spillover claims")
    return {"arrays": len(expected), "independently_replayed_training_actions": len(expected_rows),
            "model_seeds": len(cfg["model_seeds"]), "conditions": 30,
            "checkpoint_static_probes": int(np.prod(arrays["checkpoint_probabilities"].shape)),
            "head_update_calls": int(arrays["trial_head_update_calls"][..., -1].sum()),
            "internal_update_calls": 0, "value_update_calls": 0,
            "all_checkpoints_and_trial_head_states_replayed": True,
            "fixed_internal_weights_encoder_and_probe_activity_checked": True,
            "five_fixed_alpha_directions_and_analytic_own_influences_checked": True,
            "exact_frozen_shared_and_projected_l2_endpoints_checked": True,
            "all_actual_step_path_energy_and_head_norms_replayed": True,
            "all_unit_direction_actual_step_and_path_ceilings_checked": True,
            "balanced_presentation_absolute_logit_and_probability_ceilings_checked": True,
            "primary_midpoint_contrasted_against_both_endpoints_checked": True,
            "acquisition_gate_separate_from_tradeoff_gate_checked": True,
            "actual_autonomous_costs_equalized": False, "extra_solver_cost_equalized": False,
            "no_best_alpha_selected_after_observation": True,
            "no_hidden_nogo_outcome_updates": True,
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
    from drosolearn.spillover_envelope import verify_envelope, sha, write_new
    cfg, _, meta = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.publication_proof)
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    with (args.result_dir / "measurements.csv").open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        core.demand(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
                    "Unique independent trial CSV columns required")
        rows = list(reader)
    summary = core.read_json(args.result_dir / "summary.json")
    checks = verify_science(cfg, args.root, arrays, rows, summary)
    core.demand(meta["counts"] == summary["counts"], "Independent result and metadata count binding")
    report = {"status": "passed", "study_id": cfg["study_id"],
              "verified_at_utc": datetime.now(timezone.utc).isoformat(), "registration_sha256": sha(args.registration),
              "summary_sha256": sha(args.result_dir / "summary.json"), "scientific_producer_imported": False,
              "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
