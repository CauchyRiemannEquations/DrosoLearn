"""Finite reward-head learning with separate own-score and weight-L2 budgets."""
from __future__ import annotations

import json
import numpy as np
from .graph import load_projection
from .human_shell_models import build_graphs
from .pattern_rule import VectorActor
from .representation_update import (RewardHead, numeric_fields, load_catalog as parent_catalog,
                                    SCIENCE as PARENT_SCIENCE)
from .update_budget_geometry import UPDATES, update_budget_geometry

TOPOLOGIES = ("real", "rewired", "ann")
REPRESENTATIONS = ("raw", "identity")
SCIENCE = {
    "study_id": "update_budget_v1", "scope": "prospective_update_budget_diagnosis",
    "module": "drosolearn.update_budget", "protocol": "docs/update-budget-protocol.md",
    "topologies": list(TOPOLOGIES), "representations": list(REPRESENTATIONS), "updates": list(UPDATES),
    "model_seeds": list(range(17201, 17211)), "training_trials": 400,
    "checkpoints": [0, 50, 100, 200, 400], "head_lr": .1, "rng_namespace": 202610194,
    "svd_rcond": 1e-12, "numeric_tolerance": 1e-8, "exact_condition_max": 1e6,
    "acquisition_threshold": .8, "acquisition_gain": .25, "class_threshold": .7,
    "population_pass_count": 8, "effect_threshold": .1, "positive_seed_count": 8,
    "normalization": "per_input_initial_fixed_MBON_gap_unit_L2",
    "budget_convention": "unit_direction_equal_conditional_L2_and_common_eta_step_ceiling",
    "input_catalog": "data/synthetic/representation_update_v1/input_catalog.json",
    "graph": "data/flywire_783_mb_left_kc512_s701", "external_manifest": None,
}


def require(ok, message):
    if not ok: raise ValueError(message)


def validate_config(cfg):
    fixture = cfg.get("scope") == "integration_test"
    exempt = {"scope", "model_seeds", "training_trials", "checkpoints", "population_pass_count", "positive_seed_count"} if fixture else set()
    for key, value in SCIENCE.items():
        if key not in exempt: require(cfg.get(key) == value, "Fixed update budget design differs: " + key)
    if fixture:
        seeds, n, steps = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
        require(type(seeds) is list and 1 <= len(seeds) <= 2 and len(seeds) == len(set(seeds))
                and all(type(s) is int and 90001 <= s <= 90100 for s in seeds), "Invented fixture seeds only")
        require(type(n) is int and 4 <= n <= 16 and n % 4 == 0, "Bounded fixture budget")
        require(steps == sorted(set(steps)) and steps[0] == 0 and steps[-1] == n
                and all(type(t) is int and 0 <= t <= n and t % 4 == 0 for t in steps), "Fixture checkpoints")
        require(cfg["population_pass_count"] == cfg["positive_seed_count"] == len(seeds), "Fixture pass counts")


def load_catalog(cfg, root):
    require(cfg["input_catalog"] == PARENT_SCIENCE["input_catalog"], "Literal parent input path")
    return parent_catalog(PARENT_SCIENCE, root)


def produce(cfg, root):
    validate_config(cfg)
    data = load_catalog(cfg, root)
    seeds, n, checkpoints = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
    graph = load_projection(root / cfg["graph"])
    trial_shape = (3, 2, 3, len(seeds), n)
    arrays = dict(data)
    for field in ("probabilities", "actions", "rewards", "head_steps", "own_score_steps", "other_score_steps",
                  "internal_steps", "value_steps", "head_update_calls", "internal_update_calls", "value_update_calls",
                  "update_factors", "expected_head_steps", "head_step_squared_l2", "cumulative_head_path_l2",
                  "cumulative_head_squared_l2", "head_norms"):
        integer = field in ("actions", "rewards") or field.endswith("calls")
        arrays["trial_" + field] = np.zeros(trial_shape, dtype=np.int32 if integer else float)
    checkpoint_shape = (3, 2, 3, len(seeds), len(checkpoints))
    for field in ("logits", "probabilities"):
        arrays["checkpoint_" + field] = np.empty(checkpoint_shape + (4,))
    arrays["checkpoint_heads"] = np.empty(checkpoint_shape + (graph.n_mbon,))
    for field in ("head", "internal", "value"):
        arrays["checkpoint_" + field + "_calls"] = np.zeros(checkpoint_shape, dtype=np.int32)
    arrays["schedules"] = np.empty((len(seeds), n), dtype=np.int8)
    arrays["uniforms"] = np.empty((len(seeds), n))
    arrays["model_seeds"] = np.asarray(seeds, dtype=np.int64)
    arrays["checkpoints"] = np.asarray(checkpoints, dtype=np.int32)
    static, prepared, rows = {}, [], []
    for si, seed in enumerate(seeds):
        graphs = build_graphs(graph, seed)
        for ti, topology in enumerate(TOPOLOGIES):
            for ri, representation in enumerate(REPRESENTATIONS):
                inputs = data["inputs"][ri]
                base = VectorActor(graphs[topology], seed, cfg["head_lr"], False, inputs, inputs)
                initial = base.witness()
                gap = initial["probe_mbon"][:, 1] - initial["probe_mbon"][:, 0]
                budget = update_budget_geometry(gap, cfg["svd_rcond"], cfg["numeric_tolerance"], cfg["exact_condition_max"])
                require(budget["numeric_budget_qualified"], "Registered numerical budget gate failed; no fallback")
                group = topology + "_" + representation + "_"
                numeric = numeric_fields({key: value for key, value in budget.items() if key != "updates"})
                for prefix, values in (("w_", initial), ("b_", numeric)):
                    for key, value in values.items():
                        static.setdefault(prefix + group + key, []).append(np.asarray(value))
                absolute_ceiling = (n // 4) * cfg["head_lr"] * np.abs(budget["influences"]).sum(axis=1)
                probability_ceiling = 1. / (1. + np.exp(-np.clip(absolute_ceiling, -40., 40.)))
                static.setdefault("c_" + group + "absolute_logit_ceiling", []).append(absolute_ceiling)
                static.setdefault("c_" + group + "correct_probability_ceiling", []).append(probability_ceiling)
                prepared.append((ti, ri, si, seed, topology, representation, base, initial, budget, group))
    # All graph/geometry gates close before any behavioral draw or feedback.
    for si, seed in enumerate(seeds):
        order_rng = np.random.default_rng(np.random.SeedSequence([cfg["rng_namespace"], seed, 0]))
        arrays["schedules"][si] = np.concatenate([order_rng.permutation(4) for _ in range(n // 4)])
        arrays["uniforms"][si] = np.random.default_rng(np.random.SeedSequence([cfg["rng_namespace"], seed, 1])).random(n)
    lookup = {step: index for index, step in enumerate(checkpoints)}
    for ti, ri, si, seed, topology, representation, base, initial, budget, group in prepared:
        features = budget["base_geometry"]["unit_rows"]
        for ui, update in enumerate(UPDATES):
            actor = RewardHead(features, budget["directions"][ui], cfg["head_lr"])
            path_l2 = squared_l2 = 0.

            def checkpoint(step):
                at = (ti, ri, ui, si, lookup[step])
                arrays["checkpoint_logits"][at] = actor.logits()
                arrays["checkpoint_probabilities"][at] = actor.probabilities()
                arrays["checkpoint_heads"][at] = actor.head
                arrays["checkpoint_head_calls"][at] = actor.calls

            checkpoint(0)
            for trial, example in enumerate(arrays["schedules"][si]):
                example = int(example)
                p = float(actor.probabilities()[example])
                target = int(data["targets"][example])
                action = int(arrays["uniforms"][si, trial] < p)
                reward = (1 if target else -1) if action else 0
                factor = cfg["head_lr"] * (action - p) * reward
                direction_l2 = float(budget["direction_norms"][ui, example])
                expected_step = abs(factor) * direction_l2
                head_step, own_step, other_step = actor.observe(example, action, reward)
                path_l2 += head_step
                squared_l2 += head_step ** 2
                head_norm = float(np.linalg.norm(actor.head))
                at = (ti, ri, ui, si, trial)
                for key, value in (("probabilities", p), ("actions", action), ("rewards", reward),
                                   ("head_steps", head_step), ("own_score_steps", own_step), ("other_score_steps", other_step),
                                   ("head_update_calls", actor.calls), ("update_factors", factor),
                                   ("expected_head_steps", expected_step), ("head_step_squared_l2", head_step ** 2),
                                   ("cumulative_head_path_l2", path_l2), ("cumulative_head_squared_l2", squared_l2),
                                   ("head_norms", head_norm)):
                    arrays["trial_" + key][at] = value
                rows.append({"topology": topology, "representation": representation, "update": update,
                             "seed": seed, "trial": trial + 1, "example_index": example, "target": target,
                             "probability_go": p, "action": action, "reward": reward,
                             "head_update_calls": actor.calls, "internal_update_calls": 0, "value_update_calls": 0,
                             "head_step_l2": head_step, "own_score_step": own_step, "other_score_step_l2": other_step,
                             "internal_step_l2": 0., "value_step_l2": 0., "head_nonzero_step": int(head_step > 0),
                             "internal_nonzero_step": 0, "value_nonzero_step": 0,
                             "update_factor": factor, "direction_l2": direction_l2, "expected_head_step_l2": expected_step,
                             "head_step_squared_l2": head_step ** 2, "cumulative_head_path_l2": path_l2,
                             "cumulative_head_squared_l2": squared_l2, "head_norm": head_norm})
                if trial + 1 in lookup: checkpoint(trial + 1)
        final = base.witness()
        for key in ("weights", "encoder", "bias", "probe_kc", "probe_mbon"):
            require(np.array_equal(initial[key], final[key]), "Fixed internal/feature witness changed")
            static.setdefault("finalfixed_" + group + key, []).append(final[key])
    arrays.update({key: np.stack(value) for key, value in static.items()})
    return arrays, rows


def summarize(cfg, arrays, rows):
    validate_config(cfg)
    p = arrays["checkpoint_probabilities"]
    shape = (3, 2, 3, len(cfg["model_seeds"]), len(cfg["checkpoints"]), 4)
    require(p.shape == shape and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "Finite aligned probabilities")
    correct = np.where(arrays["targets"] == 1, p, 1 - p)
    initial, final = correct[..., 0, :], correct[..., -1, :]
    means = final.mean(-1)
    acquired = ((means >= cfg["acquisition_threshold"])
                & (means - initial.mean(-1) >= cfg["acquisition_gain"])
                & (final[..., :2].mean(-1) >= cfg["class_threshold"])
                & (final[..., 2:].mean(-1) >= cfg["class_threshold"]))
    eta, n, tolerance = cfg["head_lr"], cfg["training_trials"], cfg["numeric_tolerance"]
    max_step = arrays["trial_head_steps"].max(-1)
    path = arrays["trial_cumulative_head_path_l2"][..., -1]
    energy = arrays["trial_cumulative_head_squared_l2"][..., -1]
    within = ((max_step <= eta + tolerance) & (path <= n * eta + tolerance))
    all_bound = bool(within[:, :, [0, 2]].all())
    expected_residual = float(np.max(np.abs(arrays["trial_head_steps"] - arrays["trial_expected_head_steps"])))
    geometry, families, contrasts, ceiling_checks = [], [], [], []
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            prefix = "b_" + topology + "_" + representation + "_"
            ceiling = arrays["c_" + topology + "_" + representation + "_correct_probability_ceiling"]
            ceiling_checks.append(bool(np.all(final[ti, ri] <= np.swapaxes(ceiling, 0, 1) + tolerance)))
            geometry.append({"topology": topology, "representation": representation, "seeds":[{
                "seed": seed, "retained_rank": int(arrays[prefix + "base_geometry_unit_diagnostics_retained_rank"][si]),
                "exact_isolation_qualified": bool(arrays[prefix + "projected_l2_exact_isolation_qualified"][si]),
                "numeric_budget_qualified": bool(arrays[prefix + "numeric_budget_qualified"][si]),
                "direction_norms": arrays[prefix + "direction_norms"][si].tolist(),
                "own_influence": arrays[prefix + "own_influence"][si].tolist(),
                "spillover_squared": arrays[prefix + "spillover_squared"][si].tolist(),
                "conditional_l2_match_residual": float(arrays[prefix + "conditional_l2_match_residual"][si]),
                "absolute_logit_ceiling_by_update": arrays["c_" + topology + "_" + representation + "_absolute_logit_ceiling"][si].tolist(),
                "correct_probability_ceiling_by_update": ceiling[si].tolist()
            } for si, seed in enumerate(cfg["model_seeds"])]})
            for ui, update in enumerate(UPDATES):
                seed_rows = []
                for si, seed in enumerate(cfg["model_seeds"]):
                    seed_rows.append({"seed": seed, "initial_expected_accuracy": float(initial[ti, ri, ui, si].mean()),
                        "final_expected_accuracy": float(means[ti, ri, ui, si]),
                        "gain": float(means[ti, ri, ui, si] - initial[ti, ri, ui, si].mean()),
                        "class_1_expected_accuracy": float(final[ti, ri, ui, si, :2].mean()),
                        "class_0_expected_accuracy": float(final[ti, ri, ui, si, 2:].mean()),
                        "acquisition_pass": bool(acquired[ti, ri, ui, si]),
                        "final_head_l2": float(np.linalg.norm(arrays["checkpoint_heads"][ti, ri, ui, si, -1])),
                        "cumulative_head_path_l2": float(path[ti, ri, ui, si]),
                        "cumulative_head_squared_l2": float(energy[ti, ri, ui, si]),
                        "max_head_step_l2": float(max_step[ti, ri, ui, si]),
                        "unit_budget_applies": ui in (0, 2),
                        "within_common_step_and_path_bound": bool(within[ti, ri, ui, si]) if ui in (0, 2) else None})
                families.append({"topology": topology, "representation": representation, "update": update,
                    "mean_final_expected_accuracy": float(means[ti, ri, ui].mean()),
                    "sd_final_expected_accuracy": float(means[ti, ri, ui].std(ddof=1)) if len(seed_rows) > 1 else 0.,
                    "acquisition_pass_seeds": int(acquired[ti, ri, ui].sum()),
                    "mean_cumulative_head_path_l2": float(path[ti, ri, ui].mean()),
                    "mean_cumulative_head_squared_l2": float(energy[ti, ri, ui].mean()),
                    "seeds": seed_rows})
            contrasts.append({"topology": topology, "representation": representation,
                "l2_minus_shared_by_seed": (means[ti, ri, 2] - means[ti, ri, 0]).tolist(),
                "own_minus_l2_by_seed": (means[ti, ri, 1] - means[ti, ri, 2]).tolist(),
                "own_minus_shared_by_seed": (means[ti, ri, 1] - means[ti, ri, 0]).tolist(),
                "l2_minus_shared_actual_path_cost_by_seed": (path[ti, ri, 2] - path[ti, ri, 0]).tolist(),
                "descriptive_only": (ti, ri) != (0, 0)})
    all_numeric = all(row["numeric_budget_qualified"] for group in geometry for row in group["seeds"])
    ceiling_valid = all(ceiling_checks)
    numeric_valid = bool(all_numeric and all_bound and expected_residual <= tolerance and ceiling_valid)
    effect = np.asarray(contrasts[0]["l2_minus_shared_by_seed"])
    treated_passes = int(acquired[0, 0, 2].sum())
    supported = bool(effect.mean() >= cfg["effect_threshold"]
                     and np.count_nonzero(effect > 0) >= cfg["positive_seed_count"]
                     and treated_passes >= cfg["population_pass_count"] and numeric_valid)
    require(len(rows) == 18 * len(cfg["model_seeds"]) * n, "Complete actual trial records")
    return {"study_id": cfg["study_id"], "status": "completed", "families": families,
        "geometry": geometry, "contrasts": contrasts,
        "H_budget": {"supported": supported, "mean_effect": float(effect.mean()),
                     "positive_seeds": int(np.count_nonzero(effect > 0)),
                     "treated_acquisition_pass_seeds": treated_passes, "numeric_budget_qualified_all": numeric_valid},
        "budget_validation": {"conditional_scalar_matching_only": True, "actual_cumulative_cost_equalized": False,
            "common_unit_direction_step_ceiling": eta, "common_unit_direction_path_ceiling": n * eta,
            "common_bounds_valid_all": all_bound, "head_step_expected_residual_max": expected_residual,
            "all_static_numeric_gates_passed": bool(all_numeric), "analytic_probability_ceiling_valid_all": ceiling_valid},
        "counts": {"training_feedback_trials": len(rows), "static_probe_cases": int(np.prod(p.shape)),
            "model_initializations": 18 * len(cfg["model_seeds"]), "fixed_feature_constructions": 6 * len(cfg["model_seeds"]),
            "model_seeds": len(cfg["model_seeds"]), "head_update_calls": int(arrays["trial_head_update_calls"][..., -1].sum()),
            "head_nonzero_steps": int(np.count_nonzero(arrays["trial_head_steps"])),
            "internal_update_calls": 0, "value_update_calls": 0, "internal_nonzero_steps": 0, "value_nonzero_steps": 0},
        "resources": {"head_coordinates": int(arrays["checkpoint_heads"].shape[-1]),
            "extra_projected_direction_coefficients_per_rule": int(4 * arrays["checkpoint_heads"].shape[-1]),
            "direction_solver_SVD_per_group": 1, "diagnostic_SVD_per_group": 2,
            "total_producer_SVD_calls": 3 * 6 * len(cfg["model_seeds"]),
            "exact_actual_cumulative_budget_matching_claimed": False,
            "extra_solver_cost_equalized": False, "own_score_and_unit_L2_simultaneously_matched": False},
        "interpretation": {"artificial_history_buffer": True, "internal_synapses_fixed": True,
            "identity_dictionary_is_supplied": True, "prior_failure_or_seed_discarded": False,
            "generalization_tested": False, "new_human_experiment": False,
            "shared_neural_mechanism_claim": False, "education_effect_claim": False}}


def render(result, output):
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result, output = Path(result), Path(output)
    require(json.loads((result / "verification.json").read_text())["status"] == "passed", "Independent verification before plotting")
    output.mkdir(exist_ok=False)
    with np.load(result / "arrays.npz", allow_pickle=False) as a:
        correct = np.where(a["targets"] == 1, a["checkpoint_probabilities"], 1 - a["checkpoint_probabilities"]).mean(-1)
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, constrained_layout=True)
        for ti, topology in enumerate(TOPOLOGIES):
            for ri, rep in enumerate(REPRESENTATIONS):
                for ui, update in enumerate(UPDATES):
                    axes[ri, ti].plot(a["checkpoints"], correct[ti, ri, ui].mean(0), label=update)
                axes[ri, ti].axhline(.8, color="gray", linestyle=":", linewidth=.8)
                axes[ri, ti].set(title=topology + " / " + rep, xlabel="Feedback trials", ylim=(.45, 1.01))
        axes[0, 0].set_ylabel("Expected correct choice probability")
        axes[1, 0].set_ylabel("Expected correct choice probability")
        axes[0, 2].legend(fontsize=8)
        fig.suptitle("Output update budget: 10 computational seeds; dotted line is mean acquisition threshold")
        fig.savefig(output / "learning-curves.png", dpi=160); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ri, rep in enumerate(REPRESENTATIONS):
            for ui, update in enumerate(UPDATES):
                axes[ri].plot(np.arange(1, a["trial_head_steps"].shape[-1] + 1),
                              a["trial_cumulative_head_path_l2"][0, ri, ui].mean(0), label=update)
            axes[ri].set(title="real / " + rep, xlabel="Feedback trials", ylabel="Cumulative head path length (L2 steps)")
        axes[1].legend(fontsize=8)
        fig.suptitle("Actual accumulated movement is recorded, not forced equal")
        fig.savefig(output / "actual-movement-costs.png", dpi=160); plt.close(fig)
