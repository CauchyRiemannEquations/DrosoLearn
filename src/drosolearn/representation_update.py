"""Finite representation x reward-head-update diagnosis, not a generalization test."""
from __future__ import annotations

import json
import numpy as np

from .graph import load_projection
from .human_shell_models import build_graphs, validate_observation
from .pattern_rule import VectorActor, catalog as previous_catalog
from .representation_geometry import geometry, hard_margin_witness, raw_parallelogram_certificate

TOPOLOGIES = ("real", "rewired", "ann")
REPRESENTATIONS = ("raw", "identity")
UPDATES = ("shared", "projected")
SCIENCE = {
    "study_id": "representation_update_v1", "scope": "prospective_representation_update_diagnosis",
    "module": "drosolearn.representation_update", "protocol": "docs/representation-update-protocol.md",
    "topologies": list(TOPOLOGIES), "representations": list(REPRESENTATIONS), "updates": list(UPDATES),
    "model_seeds": list(range(17101, 17111)), "training_trials": 400,
    "checkpoints": [0, 50, 100, 200, 400], "head_lr": .1, "rng_namespace": 202610193,
    "svd_rcond": 1e-12, "numeric_tolerance": 1e-8, "exact_condition_max": 1e6,
    "acquisition_threshold": .8, "acquisition_gain": .25, "class_threshold": .7,
    "population_pass_count": 8, "effect_threshold": .1, "positive_seed_count": 8,
    "normalization": "per_input_initial_fixed_MBON_gap_unit_L2",
    "input_catalog": "data/synthetic/representation_update_v1/input_catalog.json",
    "graph": "data/flywire_783_mb_left_kc512_s701", "external_manifest": None,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_config(cfg):
    fixture = cfg.get("scope") == "integration_test"
    exempt = {"scope", "model_seeds", "training_trials", "checkpoints", "population_pass_count", "positive_seed_count"} if fixture else set()
    for key, value in SCIENCE.items():
        if key not in exempt:
            require(cfg.get(key) == value, "Fixed diagnostic design differs: " + key)
    if fixture:
        seeds, n, steps = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
        require(type(seeds) is list and 1 <= len(seeds) <= 2 and len(seeds) == len(set(seeds))
                and all(type(s) is int and 90001 <= s <= 90100 for s in seeds), "Invented fixture seeds only")
        require(type(n) is int and 4 <= n <= 16 and n % 4 == 0, "Bounded invented fixture budget")
        require(steps == sorted(set(steps)) and steps[0] == 0 and steps[-1] == n
                and all(type(t) is int and 0 <= t <= n and t % 4 == 0 for t in steps), "Fixture checkpoints")
        require(cfg["population_pass_count"] == cfg["positive_seed_count"] == len(seeds), "Fixture seed counts")


def catalog():
    """Four exact strings, label-free lexicographic IDs, equal input norms."""
    old = previous_catalog()
    raw = old["train_catalog"].copy()
    strings = old["train_strings"][:, :4].copy()
    dictionary = sorted(map(tuple, strings.tolist()))
    ids = np.array([dictionary.index(tuple(row)) for row in strings.tolist()], dtype=np.int8)
    identity = np.zeros_like(raw)
    identity[np.arange(4), ids] = np.sqrt(2.) / 3.
    identity[:, 6] = 2. / 3.
    return {"inputs": np.stack((raw, identity)), "targets": np.array([1, 1, 0, 0], dtype=np.int8),
            "strings": strings, "lexicographic_ids": ids}


def load_catalog(cfg, root):
    from .representation_envelope import read
    value = read(root / cfg["input_catalog"])
    expected = catalog()
    require(set(value) == {"study_id", "representations", "catalog"}
            and value["study_id"] == cfg["study_id"] and value["representations"] == list(REPRESENTATIONS), "Input identity")
    require(set(value["catalog"]) == set(expected), "Exact input inventory")
    for key, wanted in expected.items():
        actual = np.asarray(value["catalog"][key])
        require(actual.shape == wanted.shape and actual.dtype.kind == wanted.dtype.kind
                and np.array_equal(actual, wanted), "Fixed finite input differs: " + key)
    return expected


def numeric_fields(value, prefix=""):
    """Stable finite NPZ schema; optional scalars always carry defined flags."""
    result = {}
    optional = {"full_row_condition", "retained_condition", "head_l2", "active_subset_mask"}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(numeric_fields(item, name + "_"))
        elif isinstance(item, str):
            continue
        else:
            if key in optional:
                result[name + "_defined"] = np.asarray(item is not None)
            result[name] = np.asarray(0. if item is None else item)
    return result


class RewardHead:
    """Only index, chosen action and obtained reward enter the update API."""
    def __init__(self, features, directions, rate):
        self.features, self.directions = np.asarray(features), np.asarray(directions)
        require(self.features.shape == self.directions.shape and self.features.ndim == 2
                and self.features.shape[0] == 4 and np.isfinite(self.features).all()
                and np.isfinite(self.directions).all(), "Four finite aligned feature/direction rows")
        require(np.isfinite(rate) and rate > 0, "Positive learning rate")
        self.rate, self.calls = rate, 0
        self.head = np.zeros(self.features.shape[1])

    def logits(self):
        return self.features @ self.head

    def probabilities(self):
        return 1. / (1. + np.exp(-np.clip(self.logits(), -40., 40.)))

    def observe(self, index, action, reward):
        require(type(index) is int and 0 <= index < 4, "Observed input index")
        validate_observation(action, reward)
        p = float(self.probabilities()[index])
        old = self.head.copy()
        self.head += self.rate * (action - p) * reward * self.directions[index]
        self.calls += 1
        step = self.head - old
        scores = self.features @ step
        return float(np.linalg.norm(step)), float(scores[index]), float(np.linalg.norm(np.delete(scores, index)))


def produce(cfg, root):
    validate_config(cfg)
    data = load_catalog(cfg, root)
    seeds, n, checkpoints = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
    graph = load_projection(root / cfg["graph"])
    t_shape = (3, 2, 2, len(seeds), n)
    arrays = dict(data)
    for key in ("probabilities", "actions", "rewards", "head_steps", "own_score_steps", "other_score_steps",
                "internal_steps", "value_steps", "head_update_calls", "internal_update_calls", "value_update_calls"):
        integer = key in ("actions", "rewards") or key.endswith("calls")
        arrays["trial_" + key] = np.zeros(t_shape, dtype=np.int32 if integer else float)
    c_shape = (3, 2, 2, len(seeds), len(checkpoints))
    arrays["checkpoint_logits"] = np.empty(c_shape + (4,))
    arrays["checkpoint_probabilities"] = np.empty(c_shape + (4,))
    arrays["checkpoint_heads"] = np.empty(c_shape + (graph.n_mbon,))
    for key in ("head", "internal", "value"):
        arrays["checkpoint_" + key + "_calls"] = np.zeros(c_shape, dtype=np.int32)
    arrays["schedules"] = np.empty((len(seeds), n), dtype=np.int8)
    arrays["uniforms"] = np.empty((len(seeds), n))
    arrays["model_seeds"] = np.asarray(seeds, dtype=np.int64)
    arrays["checkpoints"] = np.asarray(checkpoints, dtype=np.int32)
    for ri, representation in enumerate(REPRESENTATIONS):
        arrays.update(numeric_fields(hard_margin_witness(data["inputs"][ri], 2 * data["targets"] - 1), "a_" + representation + "_"))
    static, rows, prepared = {}, [], []
    checkpoint_lookup = {t: i for i, t in enumerate(checkpoints)}
    for si, seed in enumerate(seeds):
        graphs = build_graphs(graph, seed)
        for ti, topology in enumerate(TOPOLOGIES):
            for ri, representation in enumerate(REPRESENTATIONS):
                inputs = data["inputs"][ri]
                base = VectorActor(graphs[topology], seed, cfg["head_lr"], False, inputs, inputs)
                initial = base.witness()
                gap = initial["probe_mbon"][:, 1] - initial["probe_mbon"][:, 0]
                g = geometry(gap, cfg["svd_rcond"], cfg["numeric_tolerance"], cfg["exact_condition_max"])
                require(g["numeric_update_qualified"], "Registered own-step numerical tolerance failed; no fallback")
                margin = hard_margin_witness(g["unit_rows"], 2 * data["targets"] - 1,
                                             cfg["svd_rcond"], cfg["numeric_tolerance"])
                group = topology + "_" + representation + "_"
                for prefix, record in (("w_", initial), ("g_", numeric_fields(g)), ("m_", numeric_fields(margin))):
                    for key, value in record.items():
                        static.setdefault(prefix + group + key, []).append(np.asarray(value))
                prepared.append((ti, ri, si, seed, topology, representation, base, initial, g, group))
    # Every geometry must pass before the first behavioral choice, not only its own group.
    for si, seed in enumerate(seeds):
        order_rng = np.random.default_rng(np.random.SeedSequence([cfg["rng_namespace"], seed, 0]))
        arrays["schedules"][si] = np.concatenate([order_rng.permutation(4) for _ in range(n // 4)])
        arrays["uniforms"][si] = np.random.default_rng(np.random.SeedSequence([cfg["rng_namespace"], seed, 1])).random(n)
    for ti, ri, si, seed, topology, representation, base, initial, g, group in prepared:
        schedule, uniforms = arrays["schedules"][si], arrays["uniforms"][si]
        for ui, update in enumerate(UPDATES):
            directions = g["unit_rows"] if update == "shared" else g["directions"]
            actor = RewardHead(g["unit_rows"], directions, cfg["head_lr"])

            def checkpoint(step):
                at = (ti, ri, ui, si, checkpoint_lookup[step])
                arrays["checkpoint_logits"][at] = actor.logits()
                arrays["checkpoint_probabilities"][at] = actor.probabilities()
                arrays["checkpoint_heads"][at] = actor.head
                arrays["checkpoint_head_calls"][at] = actor.calls

            checkpoint(0)
            for trial, index in enumerate(schedule):
                index = int(index)
                target = int(data["targets"][index])
                p = float(actor.probabilities()[index])
                action = int(uniforms[trial] < p)
                reward = (1 if target else -1) if action else 0
                head_step, own_step, other_step = actor.observe(index, action, reward)
                at = (ti, ri, ui, si, trial)
                for key, value in (("probabilities", p), ("actions", action), ("rewards", reward),
                                   ("head_steps", head_step), ("own_score_steps", own_step),
                                   ("other_score_steps", other_step), ("head_update_calls", actor.calls)):
                    arrays["trial_" + key][at] = value
                rows.append({"topology": topology, "representation": representation, "update": update,
                             "seed": seed, "trial": trial + 1, "example_index": index, "target": target,
                             "probability_go": p, "action": action, "reward": reward,
                             "head_update_calls": actor.calls, "internal_update_calls": 0, "value_update_calls": 0,
                             "head_step_l2": head_step, "own_score_step": own_step,
                             "other_score_step_l2": other_step, "internal_step_l2": 0., "value_step_l2": 0.,
                             "head_nonzero_step": int(head_step > 0), "internal_nonzero_step": 0, "value_nonzero_step": 0})
                if trial + 1 in checkpoint_lookup:
                    checkpoint(trial + 1)
        final = base.witness()
        for key in ("weights", "encoder", "bias", "probe_kc", "probe_mbon"):
            require(np.array_equal(initial[key], final[key]), "Fixed feature/internal witness changed")
            static.setdefault("finalfixed_" + group + key, []).append(final[key])
    arrays.update({key: np.stack(value) for key, value in static.items()})
    return arrays, rows


def summarize(cfg, arrays, rows):
    validate_config(cfg)
    p = arrays["checkpoint_probabilities"]
    require(p.shape == (3, 2, 2, len(cfg["model_seeds"]), len(cfg["checkpoints"]), 4)
            and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "Aligned finite probabilities")
    correct = np.where(arrays["targets"] == 1, p, 1 - p)
    initial, final = correct[..., 0, :], correct[..., -1, :]
    scores = final.mean(-1)
    acquired = ((scores >= cfg["acquisition_threshold"])
                & (scores - initial.mean(-1) >= cfg["acquisition_gain"])
                & (final[..., :2].mean(-1) >= cfg["class_threshold"])
                & (final[..., 2:].mean(-1) >= cfg["class_threshold"]))
    families, geometries, contrasts = [], [], []
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(REPRESENTATIONS):
            prefix = "g_" + topology + "_" + representation + "_"
            influence = arrays[prefix + "influence_shared"].copy()
            influence[:, np.arange(4), np.arange(4)] = 0.
            shared_spill = np.sum(influence ** 2, axis=(1, 2)) / 4
            projected_spill = arrays[prefix + "projected_spillover_squared"].mean(-1)
            geometry_seeds = []
            for si, seed in enumerate(cfg["model_seeds"]):
                geometry_seeds.append({"seed": seed,
                    "raw_feature_rank": int(arrays[prefix + "raw_diagnostics_retained_rank"][si]),
                    "unit_feature_rank": int(arrays[prefix + "unit_diagnostics_retained_rank"][si]),
                    "unit_singular_values": arrays[prefix + "unit_diagnostics_singular_values"][si].tolist(),
                    "unit_full_row_condition": float(arrays[prefix + "unit_diagnostics_full_row_condition"][si])
                        if arrays[prefix + "unit_diagnostics_full_row_condition_defined"][si] else None,
                    "own_step_residual": float(arrays[prefix + "own_step_residual"][si]),
                    "off_diagonal_max": float(arrays[prefix + "off_diagonal_max"][si]),
                    "numeric_update_qualified": bool(arrays[prefix + "numeric_update_qualified"][si]),
                    "exact_isolation_qualified": bool(arrays[prefix + "exact_isolation_qualified"][si]),
                    "shared_mean_spillover_squared": float(shared_spill[si]),
                    "projected_mean_spillover_squared": float(projected_spill[si]),
                    "max_projected_direction_l2": float(arrays[prefix + "direction_norms"][si].max()),
                    "analytic_margin_feasible": bool(arrays["m_" + topology + "_" + representation + "_feasible"][si]),
                    "analytic_margin_head_l2": float(arrays["m_" + topology + "_" + representation + "_head_l2"][si])
                        if arrays["m_" + topology + "_" + representation + "_head_l2_defined"][si] else None})
            geometries.append({"topology": topology, "representation": representation, "seeds": geometry_seeds})
            for ui, update in enumerate(UPDATES):
                seed_rows = []
                for si, seed in enumerate(cfg["model_seeds"]):
                    seed_rows.append({"seed": seed, "initial_expected_accuracy": float(initial[ti, ri, ui, si].mean()),
                                      "final_expected_accuracy": float(scores[ti, ri, ui, si]),
                                      "gain": float(scores[ti, ri, ui, si] - initial[ti, ri, ui, si].mean()),
                                      "class_1_expected_accuracy": float(final[ti, ri, ui, si, :2].mean()),
                                      "class_0_expected_accuracy": float(final[ti, ri, ui, si, 2:].mean()),
                                      "acquisition_pass": bool(acquired[ti, ri, ui, si]),
                                      "final_head_l2": float(np.linalg.norm(arrays["checkpoint_heads"][ti, ri, ui, si, -1]))})
                families.append({"topology": topology, "representation": representation, "update": update,
                                 "mean_final_expected_accuracy": float(scores[ti, ri, ui].mean()),
                                 "sd_final_expected_accuracy": float(scores[ti, ri, ui].std(ddof=1)) if len(seed_rows) > 1 else 0.,
                                 "acquisition_pass_seeds": int(acquired[ti, ri, ui].sum()), "seeds": seed_rows})
        contrasts.append({"topology": topology,
                          "update_effect_raw_by_seed": (scores[ti, 0, 1] - scores[ti, 0, 0]).tolist(),
                          "representation_effect_shared_by_seed": (scores[ti, 1, 0] - scores[ti, 0, 0]).tolist(),
                          "interaction_by_seed": ((scores[ti, 1, 1] - scores[ti, 1, 0]) - (scores[ti, 0, 1] - scores[ti, 0, 0])).tolist(),
                          "projected_identity_minus_raw_by_seed": (scores[ti, 1, 1] - scores[ti, 0, 1]).tolist(),
                          "descriptive_only": ti != 0})
    def hypothesis(effect, passes, qualified, spill=None):
        effect = np.asarray(effect)
        supported = (effect.mean() >= cfg["effect_threshold"] and np.count_nonzero(effect > 0) >= cfg["positive_seed_count"]
                     and passes >= cfg["population_pass_count"] and qualified and (spill is None or spill > 0))
        return {"supported": bool(supported), "mean_effect": float(effect.mean()),
                "positive_seeds": int(np.count_nonzero(effect > 0)), "treated_acquisition_pass_seeds": int(passes),
                "numeric_update_qualified_all": bool(qualified), "mean_spillover_squared_reduction": spill}
    raw_g, id_g = geometries[0]["seeds"], geometries[1]["seeds"]
    all_numeric_qualified = all(g["numeric_update_qualified"] for group in geometries for g in group["seeds"])
    h_update = hypothesis(contrasts[0]["update_effect_raw_by_seed"], acquired[0, 0, 1].sum(),
                          all_numeric_qualified,
                          float(np.mean([g["shared_mean_spillover_squared"] - g["projected_mean_spillover_squared"] for g in raw_g])))
    h_representation = hypothesis(contrasts[0]["representation_effect_shared_by_seed"], acquired[0, 1, 0].sum(),
                                  all_numeric_qualified)
    require(len(rows) == 3 * 2 * 2 * len(cfg["model_seeds"]) * cfg["training_trials"], "Complete trial rows")
    return {"study_id": cfg["study_id"], "status": "completed", "families": families, "geometry": geometries,
            "contrasts": contrasts, "H_update": h_update, "H_representation": h_representation,
            "raw_input_certificate": {key: value.tolist() if isinstance(value, np.ndarray) else value
                                      for key, value in raw_parallelogram_certificate().items()},
            "analytic_input_margin_feasible": {name: bool(arrays["a_" + name + "_feasible"]) for name in REPRESENTATIONS},
            "counts": {"training_feedback_trials": len(rows), "static_probe_cases": int(np.prod(p.shape)),
                       "model_initializations": 12 * len(cfg["model_seeds"]), "model_seeds": len(cfg["model_seeds"]),
                       "fixed_feature_constructions": 6 * len(cfg["model_seeds"]),
                       "head_update_calls": int(arrays["trial_head_update_calls"][..., -1].sum()),
                       "internal_update_calls": 0, "value_update_calls": 0,
                       "head_nonzero_steps": int(np.count_nonzero(arrays["trial_head_steps"])),
                       "internal_nonzero_steps": 0, "value_nonzero_steps": 0},
            "resources": {"head_coordinates": int(arrays["checkpoint_heads"].shape[-1]),
                          "extra_projected_direction_coefficients": int(4 * arrays["checkpoint_heads"].shape[-1]),
                          "direction_solver_SVD_per_topology_representation_seed": 1,
                          "diagnostic_SVD_per_topology_representation_seed": 2,
                          "analytic_margin_SVD_per_topology_representation_seed": 15,
                          "input_analytic_margin_SVD_calls": 30,
                          "total_producer_SVD_calls": int(18 * 6 * len(cfg["model_seeds"]) + 30),
                          "head_L2_steps_matched": False,
                          "own_input_score_step_per_factor_matched": True, "same_input_L2_norm": True},
            "interpretation": {"artificial_history_buffer": True, "identity_dictionary_is_supplied_memorization_control": True,
                               "internal_synapses_fixed": True, "per_input_normalization_differs_from_prior_study": True,
                               "exact_historical_failure_cause_established": False, "generalization_tested": False,
                               "new_human_experiment": False, "shared_neural_mechanism_claim": False,
                               "education_effect_claim": False, "prior_failure_or_seed_discarded": False}}


def render(result, output):
    """Static scientific figures; call only after independent verification."""
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result, output = Path(result), Path(output)
    require(json.loads((result / "verification.json").read_text())["status"] == "passed", "Independent verification before figures")
    output.mkdir(exist_ok=False)
    with np.load(result / "arrays.npz", allow_pickle=False) as data:
        p = data["checkpoint_probabilities"]
        accuracy = np.where(data["targets"] == 1, p, 1 - p).mean(-1)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, constrained_layout=True)
        for ti, ax in enumerate(axes):
            for ri, rep in enumerate(REPRESENTATIONS):
                for ui, update in enumerate(UPDATES):
                    ax.plot(data["checkpoints"], accuracy[ti, ri, ui].mean(0),
                            label=rep + " / " + update, linestyle="-" if ui else "--")
            ax.axhline(.5, color="gray", linewidth=.7)
            ax.set(title=TOPOLOGIES[ti], xlabel="Feedback trials", ylim=(.45, 1.02))
        axes[0].set_ylabel("Expected correct choice probability")
        axes[2].legend(fontsize=8)
        fig.suptitle("Representation x head update: fixed features, 10 computational seeds")
        fig.savefig(output / "learning-curves.png", dpi=160); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
        for ri, rep in enumerate(REPRESENTATIONS):
            for ui, update in enumerate(UPDATES):
                mat = data["g_real_" + rep + "_influence_" + update].mean(0)
                im = axes[ri, ui].imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
                axes[ri, ui].set(title=rep + " / " + update, xlabel="Probed input", ylabel="Updated input",
                                 xticks=range(4), yticks=range(4))
                for i in range(4):
                    for j in range(4):
                        axes[ri, ui].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=axes, label="Score change per unit update factor")
        fig.suptitle("Real topology: mean update spillover (diagonal = own input)")
        fig.savefig(output / "update-spillover.png", dpi=160); plt.close(fig)
