"""Predeclared partial-spillover family under a unit-L2 output update ceiling.

The recording loop reuses the immutable update-budget implementation into this
separate study source. No historical code, configuration or result is changed.
"""
from __future__ import annotations
import json
import numpy as np
from .graph import load_projection
from .human_shell_models import build_graphs
from .pattern_rule import VectorActor
from .representation_update import RewardHead, numeric_fields
from .update_budget import load_catalog, require
from .spillover_geometry import spillover_geometry
TOPOLOGIES = ("real", "rewired", "ann")
REPRESENTATIONS = ("raw", "identity")
SCIENCE = {'study_id': 'spillover_tradeoff_v1',
 'scope': 'prospective_spillover_tradeoff_diagnosis',
 'module': 'drosolearn.spillover_tradeoff',
 'protocol': 'docs/spillover-tradeoff-protocol.md',
 'topologies': ['real', 'rewired', 'ann'],
 'representations': ['raw', 'identity'],
 'model_seeds': [17301, 17302, 17303, 17304, 17305, 17306, 17307, 17308, 17309, 17310],
 'training_trials': 400,
 'checkpoints': [0, 50, 100, 200, 400],
 'head_lr': 0.1,
 'rng_namespace': 202610195,
 'svd_rcond': 1e-12,
 'numeric_tolerance': 1e-08,
 'exact_condition_max': 1000000.0,
 'acquisition_threshold': 0.8,
 'acquisition_gain': 0.25,
 'class_threshold': 0.7,
 'population_pass_count': 8,
 'effect_threshold': 0.05,
 'positive_seed_count': 8,
 'normalization': 'per_input_initial_fixed_MBON_gap_unit_L2',
 'budget_convention': 'all_mixture_directions_unit_L2_common_eta_step_ceiling',
 'input_catalog': 'data/synthetic/representation_update_v1/input_catalog.json',
 'graph': 'data/flywire_783_mb_left_kc512_s701',
 'external_manifest': None,
 'alphas': [0.0, 0.25, 0.5, 0.75, 1.0],
 'primary_alpha': 0.5}


def validate_config(cfg):
    fixture = cfg.get("scope") == "integration_test"
    exempt = {"scope", "model_seeds", "training_trials", "checkpoints", "population_pass_count", "positive_seed_count"} if fixture else set()
    for key, value in SCIENCE.items():
        if key not in exempt: require(cfg.get(key) == value, "Fixed spillover design differs: " + key)
    if fixture:
        seeds, n, steps = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
        require(type(seeds) is list and 1 <= len(seeds) <= 2 and len(seeds) == len(set(seeds))
                and all(type(s) is int and 90001 <= s <= 90100 for s in seeds), "Invented fixture seeds only")
        require(type(n) is int and 4 <= n <= 16 and n % 4 == 0, "Bounded fixture budget")
        require(steps == sorted(set(steps)) and steps[0] == 0 and steps[-1] == n
                and all(type(t) is int and 0 <= t <= n and t % 4 == 0 for t in steps), "Fixture checkpoints")
        require(cfg["population_pass_count"] == cfg["positive_seed_count"] == len(seeds), "Fixture pass counts")


def numeric_geometry(value):
    def without_strings(item):
        if isinstance(item, dict):
            return {key: without_strings(child) for key, child in item.items() if key != "updates"}
        return item
    return numeric_fields(without_strings(value))


def produce(cfg, root):
    validate_config(cfg)
    data = load_catalog(cfg, root)
    seeds, n, checkpoints = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
    graph = load_projection(root / cfg["graph"])
    trial_shape = (3, 2, len(cfg["alphas"]), len(seeds), n)
    arrays = dict(data)
    for field in ("probabilities", "actions", "rewards", "head_steps", "own_score_steps", "other_score_steps",
                  "internal_steps", "value_steps", "head_update_calls", "internal_update_calls", "value_update_calls",
                  "update_factors", "expected_head_steps", "head_step_squared_l2", "cumulative_head_path_l2",
                  "cumulative_head_squared_l2", "head_norms"):
        integer = field in ("actions", "rewards") or field.endswith("calls")
        arrays["trial_" + field] = np.zeros(trial_shape, dtype=np.int32 if integer else float)
    checkpoint_shape = (3, 2, len(cfg["alphas"]), len(seeds), len(checkpoints))
    for field in ("logits", "probabilities"):
        arrays["checkpoint_" + field] = np.empty(checkpoint_shape + (4,))
    arrays["checkpoint_heads"] = np.empty(checkpoint_shape + (graph.n_mbon,))
    for field in ("head", "internal", "value"):
        arrays["checkpoint_" + field + "_calls"] = np.zeros(checkpoint_shape, dtype=np.int32)
    arrays["schedules"] = np.empty((len(seeds), n), dtype=np.int8)
    arrays["uniforms"] = np.empty((len(seeds), n))
    arrays["model_seeds"] = np.asarray(seeds, dtype=np.int64)
    arrays["checkpoints"] = np.asarray(checkpoints, dtype=np.int32)
    arrays["alphas"] = np.asarray(cfg["alphas"], dtype=float)
    static, prepared, rows = {}, [], []
    for si, seed in enumerate(seeds):
        graphs = build_graphs(graph, seed)
        for ti, topology in enumerate(TOPOLOGIES):
            for ri, representation in enumerate(REPRESENTATIONS):
                inputs = data["inputs"][ri]
                base = VectorActor(graphs[topology], seed, cfg["head_lr"], False, inputs, inputs)
                initial = base.witness()
                gap = initial["probe_mbon"][:, 1] - initial["probe_mbon"][:, 0]
                budget = spillover_geometry(gap, cfg["alphas"], cfg["svd_rcond"], cfg["numeric_tolerance"], cfg["exact_condition_max"])
                require(budget["numeric_spillover_qualified"], "Registered numerical mixture gate failed; no fallback")
                group = topology + "_" + representation + "_"
                numeric = numeric_geometry(budget)
                for prefix, values in (("w_", initial), ("t_", numeric)):
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
        features = budget["base_budget"]["base_geometry"]["unit_rows"]
        for ui, alpha in enumerate(cfg["alphas"]):
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
                rows.append({"topology": topology, "representation": representation, "alpha": alpha,
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
    alphas = cfg["alphas"]
    primary = alphas.index(cfg["primary_alpha"])
    p = arrays["checkpoint_probabilities"]
    shape = (3, 2, len(alphas), len(cfg["model_seeds"]), len(cfg["checkpoints"]), 4)
    require(p.shape == shape and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "Finite aligned probabilities")
    require(np.array_equal(arrays["alphas"], alphas), "Registered alpha axis")
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
    head_endpoint = np.linalg.norm(arrays["checkpoint_heads"][..., -1, :], axis=-1)
    energy_within = energy <= n * eta ** 2 + tolerance
    endpoint_within = head_endpoint <= path + tolerance
    within = ((max_step <= eta + tolerance) & (path <= n * eta + tolerance) & energy_within & endpoint_within)
    all_bound = bool(within.all())
    expected_residual = float(np.max(np.abs(arrays["trial_head_steps"] - arrays["trial_expected_head_steps"])))
    geometry, families, contrasts, ceiling_checks = [], [], [], []
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            prefix = "t_" + topology + "_" + representation + "_"
            ceiling = arrays["c_" + topology + "_" + representation + "_correct_probability_ceiling"]
            ceiling_checks.append(bool(np.all(final[ti, ri] <= np.swapaxes(ceiling, 0, 1) + tolerance)))
            geometry.append({"topology": topology, "representation": representation, "seeds": [{
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
                "absolute_logit_ceiling_by_alpha": arrays["c_" + topology + "_" + representation + "_absolute_logit_ceiling"][si].tolist(),
                "correct_probability_ceiling_by_alpha": ceiling[si].tolist(),
            } for si, seed in enumerate(cfg["model_seeds"])]})
            for ai, alpha in enumerate(alphas):
                seed_rows = []
                for si, seed in enumerate(cfg["model_seeds"]):
                    seed_rows.append({"seed": seed, "initial_expected_accuracy": float(initial[ti, ri, ai, si].mean()),
                        "final_expected_accuracy": float(means[ti, ri, ai, si]),
                        "gain": float(means[ti, ri, ai, si] - initial[ti, ri, ai, si].mean()),
                        "class_1_expected_accuracy": float(final[ti, ri, ai, si, :2].mean()),
                        "class_0_expected_accuracy": float(final[ti, ri, ai, si, 2:].mean()),
                        "acquisition_pass": bool(acquired[ti, ri, ai, si]),
                        "final_head_l2": float(np.linalg.norm(arrays["checkpoint_heads"][ti, ri, ai, si, -1])),
                        "cumulative_head_path_l2": float(path[ti, ri, ai, si]),
                        "cumulative_head_squared_l2": float(energy[ti, ri, ai, si]),
                        "max_head_step_l2": float(max_step[ti, ri, ai, si]),
                        "within_all_common_cost_bounds": bool(within[ti, ri, ai, si])})
                families.append({"topology": topology, "representation": representation, "alpha": alpha,
                    "role": "primary" if (ti, ri, ai) == (0, 0, primary) else "predeclared_control",
                    "mean_final_expected_accuracy": float(means[ti, ri, ai].mean()),
                    "sd_final_expected_accuracy": float(means[ti, ri, ai].std(ddof=1)) if len(seed_rows) > 1 else 0.,
                    "acquisition_pass_seeds": int(acquired[ti, ri, ai].sum()),
                    "mean_cumulative_head_path_l2": float(path[ti, ri, ai].mean()),
                    "mean_cumulative_head_squared_l2": float(energy[ti, ri, ai].mean()), "seeds": seed_rows})
            contrasts.append({"topology": topology, "representation": representation, "primary_alpha": cfg["primary_alpha"],
                "primary_minus_shared_by_seed": (means[ti, ri, primary] - means[ti, ri, 0]).tolist(),
                "primary_minus_isolated_by_seed": (means[ti, ri, primary] - means[ti, ri, -1]).tolist(),
                "primary_minus_shared_actual_path_cost_by_seed": (path[ti, ri, primary] - path[ti, ri, 0]).tolist(),
                "primary_minus_isolated_actual_path_cost_by_seed": (path[ti, ri, primary] - path[ti, ri, -1]).tolist(),
                "descriptive_only": (ti, ri) != (0, 0)})
    all_numeric = all(row["numeric_spillover_qualified"] for group in geometry for row in group["seeds"])
    ceiling_valid = all(ceiling_checks)
    numeric_valid = bool(all_numeric and all_bound and expected_residual <= tolerance and ceiling_valid)
    endpoint_effects = {}
    for endpoint, key in (("shared", "primary_minus_shared_by_seed"), ("isolated", "primary_minus_isolated_by_seed")):
        effect = np.asarray(contrasts[0][key])
        endpoint_effects[endpoint] = {"mean_effect": float(effect.mean()),
            "positive_seeds": int(np.count_nonzero(effect > 0)),
            "practical_advantage_pass": bool(effect.mean() >= cfg["effect_threshold"]
                and np.count_nonzero(effect > 0) >= cfg["positive_seed_count"])}
    tradeoff_supported = bool(numeric_valid and all(row["practical_advantage_pass"] for row in endpoint_effects.values()))
    passes = int(acquired[0, 0, primary].sum())
    count = len(TOPOLOGIES) * len(REPRESENTATIONS) * len(alphas)
    require(len(rows) == count * len(cfg["model_seeds"]) * n, "Complete actual trial records")
    return {"study_id": cfg["study_id"], "status": "completed", "alphas": list(alphas), "primary_alpha": cfg["primary_alpha"],
        "families": families, "geometry": geometry, "contrasts": contrasts,
        "H_tradeoff": {"supported": tradeoff_supported, "primary_alpha": cfg["primary_alpha"],
            "endpoint_comparisons": endpoint_effects, "numeric_spillover_qualified_all": numeric_valid,
            "requires_acquisition": False},
        "H_acquisition": {"supported": bool(numeric_valid and passes >= cfg["population_pass_count"]),
            "primary_alpha": cfg["primary_alpha"], "acquisition_pass_seeds": passes,
            "numeric_spillover_qualified_all": numeric_valid, "requires_tradeoff_advantage": False},
        "budget_validation": {"conditional_scalar_matching_only": True, "actual_cumulative_cost_equalized": False,
            "common_unit_direction_step_ceiling": eta, "common_unit_direction_path_ceiling": n * eta,
            "common_unit_direction_squared_step_sum_ceiling": n * eta ** 2,
            "squared_step_sum_valid_all": bool(energy_within.all()),
            "head_endpoint_within_path_all": bool(endpoint_within.all()),
            "common_bounds_valid_all": all_bound, "head_step_expected_residual_max": expected_residual,
            "all_static_numeric_gates_passed": bool(all_numeric), "analytic_probability_ceiling_valid_all": ceiling_valid},
        "counts": {"training_feedback_trials": len(rows), "static_probe_cases": int(np.prod(p.shape)),
            "model_initializations": count * len(cfg["model_seeds"]), "fixed_feature_constructions": 6 * len(cfg["model_seeds"]),
            "model_seeds": len(cfg["model_seeds"]), "head_update_calls": int(arrays["trial_head_update_calls"][..., -1].sum()),
            "head_nonzero_steps": int(np.count_nonzero(arrays["trial_head_steps"])),
            "internal_update_calls": 0, "value_update_calls": 0, "internal_nonzero_steps": 0, "value_nonzero_steps": 0},
        "resources": {"head_coordinates": int(arrays["checkpoint_heads"].shape[-1]),
            "saved_direction_coefficients_per_group": int(len(alphas) * 4 * arrays["checkpoint_heads"].shape[-1]),
            "direction_solver_SVD_per_group": 1, "diagnostic_SVD_per_group": 2,
            "total_producer_SVD_calls": 3 * 6 * len(cfg["model_seeds"]),
            "exact_actual_cumulative_budget_matching_claimed": False, "extra_solver_cost_equalized": False},
        "interpretation": {"alpha_is_fraction_of_spillover_removed": False,
            "posthoc_best_alpha_is_confirmatory": False, "all_unit_directions_optimized": False,
            "artificial_history_buffer": True, "internal_synapses_fixed": True,
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
        final = correct[..., -1]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, constrained_layout=True)
        for ti, topology in enumerate(TOPOLOGIES):
            for ri, rep in enumerate(REPRESENTATIONS):
                axes[ti].errorbar(a["alphas"], final[ti, ri].mean(-1), yerr=final[ti, ri].std(-1, ddof=1),
                                 marker="o", capsize=3, label=rep)
            axes[ti].axhline(.8, color="gray", linestyle=":", linewidth=.8)
            axes[ti].axvline(.5, color="gray", linestyle="--", linewidth=.8)
            axes[ti].set(title=topology, xlabel="Direction mixture alpha", ylim=(.45, 1.01), xticks=a["alphas"])
        axes[0].set_ylabel("Final expected correct choice probability")
        axes[2].legend(fontsize=9)
        fig.suptitle("Fixed five-alpha comparison: mean +/- SD of 10 computational seeds; primary alpha=0.5")
        fig.savefig(output / "performance-by-alpha.png", dpi=160); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for rep in REPRESENTATIONS:
            prefix = "t_real_" + rep + "_"
            own = a[prefix + "own_influence"].mean(axis=(0, 2))
            spill = np.sqrt(a[prefix + "spillover_squared"].mean(axis=(0, 2)))
            axes[0].plot(a["alphas"], own, marker="o", label=rep)
            axes[1].plot(a["alphas"], spill, marker="o", label=rep)
        axes[0].set(title="Own-input response per unit movement", ylabel="Own score influence")
        axes[1].set(title="Effects on other three inputs", ylabel="RMS off-input influence")
        for ax in axes:
            ax.set_xlabel("Direction mixture alpha"); ax.set_xticks(a["alphas"])
            ax.axvline(.5, color="gray", linestyle="--", linewidth=.8); ax.legend(fontsize=9)
        fig.suptitle("Real topology: geometric tradeoff, without reward training")
        fig.savefig(output / "static-tradeoff.png", dpi=160); plt.close(fig)
