"""Independent arithmetic for the predeclared output-update cost control.

The new scientific producer and its current geometry helpers are never imported.
Earlier frozen independent input, graph and SVD arithmetic is reused explicitly.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "update_budget_frozen_independent_representation",
    Path(__file__).with_name("verify_representation_update.py"))
prior = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prior)
core = prior.core
TOPOLOGIES = prior.TOPOLOGIES
INPUTS = prior.INPUTS
TARGETS = prior.TARGETS
UPDATES = ("shared", "projected_own", "projected_l2")
INPUT_CATALOG = prior.INPUT_CATALOG


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "update_budget_v1", "scope": "prospective_update_budget_diagnosis",
             "module": "drosolearn.update_budget", "protocol": "docs/update-budget-protocol.md",
             "topologies": list(TOPOLOGIES), "representations": list(INPUTS), "updates": list(UPDATES),
             "model_seeds": list(range(17201, 17211)), "training_trials": 400,
             "checkpoints": [0, 50, 100, 200, 400], "head_lr": .1, "rng_namespace": 202610194,
             "svd_rcond": 1e-12, "numeric_tolerance": 1e-8, "exact_condition_max": 1e6,
             "acquisition_threshold": .8, "acquisition_gain": .25, "class_threshold": .7,
             "population_pass_count": 8, "effect_threshold": .1, "positive_seed_count": 8,
             "normalization": "per_input_initial_fixed_MBON_gap_unit_L2", "input_catalog": INPUT_CATALOG,
             "budget_convention": "unit_direction_equal_conditional_L2_and_common_eta_step_ceiling",
             "graph": "data/flywire_783_mb_left_kc512_s701", "external_manifest": None}
    fixture = allow_fixture and cfg.get("scope") == "integration_test"
    changes = {"scope", "model_seeds", "training_trials", "checkpoints", "population_pass_count", "positive_seed_count"} if fixture else set()
    for key, value in fixed.items():
        if key not in changes:
            core.demand(cfg.get(key) == value, "Fixed independent update-budget diagnosis: " + key)
    if fixture:
        seeds, n, grid = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
        core.demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(set(seeds)) == len(seeds)
                    and all(type(s) is int and 90001 <= s <= 90100 for s in seeds), "Invented diagnostic seed axis")
        core.demand(type(n) is int and 4 <= n <= 16 and n % 4 == 0, "Invented diagnostic feedback budget")
        core.demand(grid == sorted(set(grid)) and grid[0] == 0 and grid[-1] == n
                    and all(type(step) is int and 0 <= step <= n and step % 4 == 0 for step in grid), "Invented checkpoint grid")
        core.demand(cfg["population_pass_count"] == cfg["positive_seed_count"] == len(seeds), "Invented seed criteria")


def independent_directions(feature_rows):
    """Rebuild unit-size and own-score controls before reward observations.

    Every row of D and normalized U has norm one. This equality is conditional
    on the same scalar update factor, not an equality of autonomous path costs.
    """
    evidence = prior.geometry_report(feature_rows)
    d = evidence["unit_rows"]
    own = evidence["directions"]
    norms = np.linalg.norm(own, axis=1)
    core.demand(np.isfinite(norms).all() and np.all(norms > 0),
                "Finite positive projected direction norms are required")
    unit = own / norms[:, None]
    return {"shared": d.copy(), "projected_own": own.copy(),
            "projected_l2": unit, "geometry": evidence,
            "projected_own_norms": norms,
            "projected_l2_own_effect": np.sum(unit * d, axis=1),
            "conditional_l2_residual": float(np.max(np.abs(np.linalg.norm(unit, axis=1) - np.linalg.norm(d, axis=1))))}


def independent_budget_report(feature_rows, rcond=1e-12, tolerance=1e-8, exact_condition_max=1e6):
    """Reproduce the complete numeric cost geometry from frozen SVD arithmetic."""
    base = prior.geometry_report(feature_rows, rcond, tolerance, exact_condition_max)
    d, projected = base["unit_rows"], base["directions"]
    magnitudes = np.sqrt(np.sum(projected ** 2, axis=1))
    core.demand(np.isfinite(magnitudes).all() and np.all(magnitudes > 0),
                "Positive finite output-budget direction norms")
    normalized = projected / magnitudes[:, None]
    directions = np.array([d, projected, normalized])
    influence = np.einsum("uic,jc->uij", directions, d)
    lengths = np.sqrt(np.sum(directions ** 2, axis=-1))
    own = np.array([[influence[u, i, i] for i in range(4)] for u in range(3)])
    off = influence.copy()
    for i in range(4): off[:, i, i] = 0.
    unit = float(np.max(np.abs(lengths[[0, 2]] - 1.)))
    match = float(np.max(np.abs(lengths[0] - lengths[2])))
    reconstruct = float(np.max(np.abs(normalized * magnitudes[:, None] - projected)))
    scaled = float(np.max(np.abs(own[2] * magnitudes - own[1])))
    lower = float(np.max(np.maximum(1. - magnitudes, 0.)))
    cap = float(np.max(np.maximum(own[2] - own[0], 0.)))
    valid = bool(base["numeric_update_qualified"] and
                 max(unit, match, reconstruct, scaled, lower, cap) <= tolerance)
    return {"base_geometry": base, "updates": UPDATES, "directions": directions,
            "influences": influence, "direction_norms": lengths, "own_influence": own,
            "spillover_squared": np.sum(off ** 2, axis=-1),
            "own_response_per_unit_l2": own / lengths,
            "unit_norm_residual": unit, "conditional_l2_match_residual": match,
            "l2_normalization_reconstruction_residual": reconstruct,
            "own_scaled_match_residual": scaled,
            "projected_own_norm_lower_bound_residual": lower,
            "projected_l2_own_cap_residual": cap,
            "projected_l2_exact_isolation_qualified": bool(base["exact_isolation_qualified"]),
            "numeric_budget_qualified": valid, "equal_conditional_scalar_l2_only": True,
            "equal_realized_cumulative_l2_claimed": False, "tolerance": float(tolerance)}


class IndependentHead:
    """Observed-reward actor with separately recorded actual update costs."""
    def __init__(self, feature_rows, update, rate=.1):
        core.demand(update in UPDATES, "Predeclared update control required")
        self.directions = independent_directions(feature_rows)
        self.d = self.directions["shared"]
        self.direction = self.directions[update]
        self.update, self.rate = update, rate
        self.head = np.zeros(self.d.shape[1])
        self.calls = 0
        self.path_l2 = self.squared_l2 = 0.

    def probabilities(self):
        return 1. / (1. + np.exp(-np.clip(self.d @ self.head, -40., 40.)))

    def observe(self, index, action, reward):
        core.demand(type(index) is int and 0 <= index < 4 and type(action) is int and action in (0, 1)
                    and type(reward) is int and reward in (-1, 0, 1) and (action == 1 or reward == 0),
                    "Current identity, observed action and obtained reward only")
        p = float(self.probabilities()[index])
        factor = self.rate * (action - p) * reward
        before = self.head.copy()
        self.head += factor * self.direction[index]
        difference = self.head - before
        norm = float(np.linalg.norm(difference))
        self.calls += 1
        self.path_l2 += norm
        self.squared_l2 += norm * norm
        score_step = self.d @ difference
        return {"factor": factor, "head_step_l2": norm,
                "head_step_squared_l2": norm * norm,
                "own_score_step": float(score_step[index]),
                "other_score_step_l2": float(np.linalg.norm(np.delete(score_step, index)))}


def expected_artifacts(cfg, base_graph):
    """Rebuild all fixed features, autonomous feedback and actual step costs."""
    seeds, n, checkpoints = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
    arrays = {key: value.copy() for key, value in prior.independent_catalog_arrays().items()}
    shape = (3, 2, 3, len(seeds), n)
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
    static, rows = {}, []
    for si, seed in enumerate(seeds):
        schedule_rng = np.random.default_rng(np.random.SeedSequence([202610194, seed, 0]))
        schedule = np.concatenate([schedule_rng.permutation(4) for _ in range(n // 4)])
        uniforms = np.random.default_rng(np.random.SeedSequence([202610194, seed, 1])).random(n)
        arrays["schedules"][si], arrays["uniforms"][si] = schedule, uniforms
        for ti, topology in enumerate(TOPOLOGIES):
            offset = 110000 if topology == "rewired" else 120000 if topology == "ann" else 0
            graph = core.transform_graph(base_graph, topology, seed + offset)
            for ri, representation in enumerate(INPUTS):
                evidence = prior.feature_evidence(graph, seed, arrays["inputs"][ri])
                witness = prior.initial_witness(evidence)
                budget = independent_budget_report(evidence["differences"])
                core.demand(budget["numeric_budget_qualified"], "Registered conditional-cost geometry failed")
                prefix = topology + "_" + representation + "_"
                numeric_budget = prior.numeric_fields({key: value for key, value in budget.items() if key != "updates"})
                for kind, record in (("w_", witness), ("b_", numeric_budget)):
                    for key, value in record.items():
                        static.setdefault(kind + prefix + key, []).append(np.asarray(value))
                absolute_limit = np.empty((3, 4))
                for ui in range(3):
                    for tested in range(4):
                        absolute_limit[ui, tested] = (n // 4) * cfg["head_lr"] * sum(
                            abs(budget["influences"][ui, source, tested]) for source in range(4))
                probability_limit = 1. / (1. + np.exp(-np.clip(absolute_limit, -40., 40.)))
                static.setdefault("c_" + prefix + "absolute_logit_ceiling", []).append(absolute_limit)
                static.setdefault("c_" + prefix + "correct_probability_ceiling", []).append(probability_limit)
                d = budget["base_geometry"]["unit_rows"]
                for ui, update in enumerate(UPDATES):
                    directions = budget["directions"][ui]
                    direction_norms = budget["direction_norms"][ui]
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
                        rows.append({"topology": topology, "representation": representation, "update": update,
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
    core.demand(set(arrays) == set(expected), "Independent cost-control numeric field closure differs")
    for key, value in expected.items():
        actual = np.asarray(arrays[key])
        core.demand(actual.dtype.kind in "biuf" and np.isfinite(actual).all(), "Finite cost-control artifact: " + key)
        if value.dtype.kind in "biu":
            core.demand(actual.dtype.kind in "biu" and np.array_equal(actual, value), "Independent exact artifact: " + key)
        else:
            core.array_close(actual, value, "Independent full cost-control replay: " + key, atol=1e-9)
    core.demand(len(rows) == len(expected_rows), "Complete independent cost-control row count")
    for index, (actual, expected_row) in enumerate(zip(rows, expected_rows)):
        if all(type(value) is str for value in actual.values()):
            core.check_csv_row(actual, expected_row, "Update-budget trial " + str(index))
        else:
            core.compare_tree(actual, expected_row, "Update-budget trial " + str(index))


def expected_summary(cfg, arrays, rows):
    probability = arrays["checkpoint_probabilities"]
    accuracy = probability * TARGETS + (1. - probability) * (1 - TARGETS)
    first, last = accuracy[..., 0, :], accuracy[..., -1, :]
    score = np.average(last, axis=-1)
    gain = score - first.mean(axis=-1)
    class_positive, class_negative = last[..., [0, 1]].mean(-1), last[..., [2, 3]].mean(-1)
    qualifies = (score >= cfg["acquisition_threshold"]) & (gain >= cfg["acquisition_gain"])
    qualifies &= (class_positive >= cfg["class_threshold"]) & (class_negative >= cfg["class_threshold"])
    step_max = arrays["trial_head_steps"].max(axis=-1)
    # Derive resource totals from actual new-minus-old movements, then check the
    # independently replayed stored accumulators separately in check_artifacts.
    paths = arrays["trial_cumulative_head_path_l2"][..., -1]
    energies = arrays["trial_cumulative_head_squared_l2"][..., -1]
    tolerance, eta, n = cfg["numeric_tolerance"], cfg["head_lr"], cfg["training_trials"]
    bounded = (step_max <= eta + tolerance) & (paths <= n * eta + tolerance)
    all_bounded = bool(np.take(bounded, [0, 2], axis=2).all())
    expected_residual = float(np.max(np.abs(arrays["trial_expected_head_steps"] - arrays["trial_head_steps"])))
    families, geometry, contrasts, ceiling_checks = [], [], [], []
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(INPUTS):
            prefix = "b_" + topology + "_" + representation + "_"
            ceiling_prefix = "c_" + topology + "_" + representation + "_"
            ceiling = arrays[ceiling_prefix + "correct_probability_ceiling"]
            ceiling_checks.append(bool(np.all(last[ti, ri] <= np.transpose(ceiling, (1, 0, 2)) + tolerance)))
            geometry_rows = []
            for si, seed in enumerate(cfg["model_seeds"]):
                geometry_rows.append({
                    "seed": seed, "retained_rank": int(arrays[prefix + "base_geometry_unit_diagnostics_retained_rank"][si]),
                    "exact_isolation_qualified": bool(arrays[prefix + "projected_l2_exact_isolation_qualified"][si]),
                    "numeric_budget_qualified": bool(arrays[prefix + "numeric_budget_qualified"][si]),
                    "direction_norms": arrays[prefix + "direction_norms"][si].tolist(),
                    "own_influence": arrays[prefix + "own_influence"][si].tolist(),
                    "spillover_squared": arrays[prefix + "spillover_squared"][si].tolist(),
                    "conditional_l2_match_residual": float(arrays[prefix + "conditional_l2_match_residual"][si]),
                    "absolute_logit_ceiling_by_update": arrays[ceiling_prefix + "absolute_logit_ceiling"][si].tolist(),
                    "correct_probability_ceiling_by_update": ceiling[si].tolist()})
            geometry.append({"topology": topology, "representation": representation, "seeds": geometry_rows})
            for ui, update in enumerate(UPDATES):
                individuals = []
                for si, seed in enumerate(cfg["model_seeds"]):
                    individuals.append({
                        "seed": seed, "initial_expected_accuracy": float(first[ti, ri, ui, si].mean()),
                        "final_expected_accuracy": float(score[ti, ri, ui, si]), "gain": float(gain[ti, ri, ui, si]),
                        "class_1_expected_accuracy": float(class_positive[ti, ri, ui, si]),
                        "class_0_expected_accuracy": float(class_negative[ti, ri, ui, si]),
                        "acquisition_pass": bool(qualifies[ti, ri, ui, si]),
                        "final_head_l2": float(np.linalg.norm(arrays["checkpoint_heads"][ti, ri, ui, si, -1])),
                        "cumulative_head_path_l2": float(paths[ti, ri, ui, si]),
                        "cumulative_head_squared_l2": float(energies[ti, ri, ui, si]),
                        "max_head_step_l2": float(step_max[ti, ri, ui, si]), "unit_budget_applies": ui in (0, 2),
                        "within_common_step_and_path_bound": bool(bounded[ti, ri, ui, si]) if ui in (0, 2) else None})
                outcomes = score[ti, ri, ui]
                families.append({
                    "topology": topology, "representation": representation, "update": update,
                    "mean_final_expected_accuracy": float(np.mean(outcomes)),
                    "sd_final_expected_accuracy": float(np.std(outcomes, ddof=1)) if len(outcomes) > 1 else 0.,
                    "acquisition_pass_seeds": int(np.count_nonzero(qualifies[ti, ri, ui])),
                    "mean_cumulative_head_path_l2": float(np.mean(paths[ti, ri, ui])),
                    "mean_cumulative_head_squared_l2": float(np.mean(energies[ti, ri, ui])), "seeds": individuals})
            contrasts.append({
                "topology": topology, "representation": representation,
                "l2_minus_shared_by_seed": (score[ti, ri, 2] - score[ti, ri, 0]).tolist(),
                "own_minus_l2_by_seed": (score[ti, ri, 1] - score[ti, ri, 2]).tolist(),
                "own_minus_shared_by_seed": (score[ti, ri, 1] - score[ti, ri, 0]).tolist(),
                "l2_minus_shared_actual_path_cost_by_seed": (paths[ti, ri, 2] - paths[ti, ri, 0]).tolist(),
                "descriptive_only": ti != 0 or ri != 0})
    all_numeric = all(item["numeric_budget_qualified"] for group in geometry for item in group["seeds"])
    ceiling_valid = bool(all(ceiling_checks))
    numeric_valid = bool(all_numeric and all_bounded and expected_residual <= tolerance and ceiling_valid)
    effect = score[0, 0, 2] - score[0, 0, 0]
    positive = int(np.count_nonzero(effect > 0))
    acquired = int(np.count_nonzero(qualifies[0, 0, 2]))
    supported = bool(np.mean(effect) >= cfg["effect_threshold"] and positive >= cfg["positive_seed_count"]
                     and acquired >= cfg["population_pass_count"] and numeric_valid)
    coordinates = int(arrays["checkpoint_heads"].shape[-1])
    seed_count = len(cfg["model_seeds"])
    return {
        "study_id": cfg["study_id"], "status": "completed", "families": families, "geometry": geometry,
        "contrasts": contrasts,
        "H_budget": {"supported": supported, "mean_effect": float(np.mean(effect)), "positive_seeds": positive,
                     "treated_acquisition_pass_seeds": acquired, "numeric_budget_qualified_all": numeric_valid},
        "budget_validation": {"conditional_scalar_matching_only": True, "actual_cumulative_cost_equalized": False,
                              "common_unit_direction_step_ceiling": eta, "common_unit_direction_path_ceiling": n * eta,
                              "common_bounds_valid_all": all_bounded, "head_step_expected_residual_max": expected_residual,
                              "all_static_numeric_gates_passed": bool(all_numeric),
                              "analytic_probability_ceiling_valid_all": ceiling_valid},
        "counts": {"training_feedback_trials": len(rows), "static_probe_cases": int(np.prod(probability.shape)),
                   "model_initializations": 18 * seed_count, "fixed_feature_constructions": 6 * seed_count,
                   "model_seeds": seed_count, "head_update_calls": int(arrays["trial_head_update_calls"][..., -1].sum()),
                   "head_nonzero_steps": int(np.count_nonzero(arrays["trial_head_steps"])),
                   "internal_update_calls": 0, "value_update_calls": 0, "internal_nonzero_steps": 0, "value_nonzero_steps": 0},
        "resources": {"head_coordinates": coordinates, "extra_projected_direction_coefficients_per_rule": 4 * coordinates,
                      "direction_solver_SVD_per_group": 1, "diagnostic_SVD_per_group": 2,
                      "total_producer_SVD_calls": 18 * seed_count, "exact_actual_cumulative_budget_matching_claimed": False,
                      "extra_solver_cost_equalized": False, "own_score_and_unit_L2_simultaneously_matched": False},
        "interpretation": {"artificial_history_buffer": True, "internal_synapses_fixed": True,
                           "identity_dictionary_is_supplied": True, "prior_failure_or_seed_discarded": False,
                           "generalization_tested": False, "new_human_experiment": False,
                           "shared_neural_mechanism_claim": False, "education_effect_claim": False}}


def verify_against_graph(cfg, graph, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    expected, expected_rows = expected_artifacts(cfg, graph)
    check_artifacts(expected, expected_rows, arrays, rows)
    core.compare_tree(summary, expected_summary(cfg, arrays, rows), "Independent complete budget-control scientific claims")
    return {"arrays": len(expected), "independently_replayed_training_actions": len(expected_rows),
            "model_seeds": len(cfg["model_seeds"]), "conditions": 18,
            "checkpoint_static_probes": int(np.prod(arrays["checkpoint_probabilities"].shape)),
            "head_update_calls": int(arrays["trial_head_update_calls"][..., -1].sum()),
            "internal_update_calls": 0, "value_update_calls": 0,
            "all_checkpoints_and_trial_head_states_replayed": True,
            "fixed_internal_weights_encoder_and_probe_activity_checked": True,
            "conditional_scalar_unit_L2_and_own_effect_attenuation_checked": True,
            "all_actual_step_path_energy_and_head_norms_replayed": True,
            "unit_direction_common_actual_step_and_path_ceiling_checked": True,
            "balanced_presentation_absolute_logit_and_probability_ceiling_checked": True,
            "actual_autonomous_costs_equalized": False, "extra_solver_cost_equalized": False,
            "no_hidden_nogo_outcome_updates": True, "no_generalization_or_human_mechanism_claim": True}


def verify_science(cfg, root, arrays, rows, summary):
    validate_science(cfg)
    # Reused input retains its own historical study identity and literal bytes.
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
    from drosolearn.update_budget_envelope import verify_envelope, sha, write_new
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
    core.demand(meta["counts"] == summary["counts"], "Independent result/metadata count binding")
    report = {"status": "passed", "study_id": cfg["study_id"],
              "verified_at_utc": datetime.now(timezone.utc).isoformat(), "registration_sha256": sha(args.registration),
              "summary_sha256": sha(args.result_dir / "summary.json"), "scientific_producer_imported": False,
              "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
