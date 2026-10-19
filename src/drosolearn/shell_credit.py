"""New stationary credit-assignment diagnostic; all earlier studies immutable."""
from pathlib import Path
import numpy as np

from .graph import load_projection
from .human_shell_models import ShellActor

CONDITIONS = ("shared", "isolated", "uniform")
SIGNS = (-1, 1)


def validate_config(cfg):
    if cfg.get("scope") not in ("preregistered_stationary_credit_diagnostic", "integration_test"):
        raise ValueError("Unsupported diagnostic scope")
    fixed = {"study_id": "shell_credit_v1", "graph": "data/flywire_783_mb_left_kc512_s701",
             "conditions": list(CONDITIONS), "n_shells": 6, "rewarded_shell_count": 3,
             "reward_probability": .8, "head_lr": .1, "stream_namespace": 202610191,
             "gram_condition_max": 1e8, "geometry_tolerance": 1e-9,
             "q_reference": {"alpha": .1, "beta": 4.},
             "criteria": {"acquisition_accuracy": .75, "acquisition_gain": .15,
                          "seed_pass_fraction": .8, "endpoint_effect": .1, "positive_seed_fraction": .8}}
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError("Fixed scientific identity, geometry and criteria required")
    seeds = cfg["seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Unique nonnegative seed axis required")
    n, every = cfg["exposures_per_shell"], cfg["checkpoint_every_exposures"]
    if type(n) is not int or type(every) is not int or n < 6 or every <= 0 or n % every:
        raise ValueError("Complete positive exposure/checkpoint budget required")
    if cfg["scope"] == "preregistered_stationary_credit_diagnostic":
        if seeds != list(range(15001, 15006)) or (n, every) != (120, 10):
            raise ValueError("Five new seeds and 120 stationary exposures required")
    elif any(s in range(15001, 15006) for s in seeds):
        raise ValueError("Scientific seeds cannot be development fixtures")


def geometry(gaps, condition_max=1e8, tolerance=1e-9):
    d = np.asarray(gaps, dtype=float)
    if d.ndim != 2 or d.shape[0] != 6 or not np.isfinite(d).all():
        raise ValueError("Finite six-shell feature matrix required")
    gram = d @ d.T
    rank, condition = int(np.linalg.matrix_rank(d)), float(np.linalg.cond(gram))
    if rank != 6 or not np.isfinite(condition) or condition > condition_max:
        raise ValueError(f"Geometry gate failed: rank={rank}, gram_condition={condition!r}")
    dual = np.linalg.solve(gram, d)
    residual = float(np.max(np.abs(d @ dual.T - np.eye(6))))
    if not np.isfinite(dual).all() or residual > tolerance:
        raise ValueError(f"Dual numerical gate failed: residual={residual!r}")
    return {"feature_gaps": d, "gram": gram, "dual_rows": dual, "rank": np.asarray(rank),
            "gram_condition": np.asarray(condition), "dual_residual_max": np.asarray(residual)}


class HeadPolicy:
    """The learner receives shell/action/obtained reward, never hidden targets."""
    def __init__(self, condition, gaps, dual_rows, rate):
        if condition not in CONDITIONS:
            raise ValueError("Unknown condition")
        self.condition, self.gaps, self.dual, self.rate = condition, gaps, dual_rows, rate
        self.head = np.zeros(gaps.shape[1])
        self.calls = 0

    def probabilities(self):
        if self.condition == "uniform":
            return np.full(6, .5)
        return 1. / (1. + np.exp(-np.clip(self.gaps @ self.head, -40., 40.)))

    def observe(self, shell, action, reward):
        if (type(shell) is not int or not 1 <= shell <= 6 or type(action) is not int or action not in (0, 1)
                or type(reward) is not int or reward not in (-1, 0, 1)):
            raise ValueError("Valid observed shell/action/signed reward required")
        if action == 0 and reward != 0:
            raise ValueError("NoGo cannot observe hidden Go content")
        if self.condition == "uniform":
            return 0.
        index = shell - 1
        factor = (action - self.probabilities()[index]) * reward
        direction = self.gaps[index] if self.condition == "shared" else (
            self.gaps[index] @ self.gaps[index]) * self.dual[index]
        before = self.head.copy()
        self.head += self.rate * factor * direction
        self.calls += 1
        return float(np.linalg.norm(self.head - before))


def _rng(cfg, seed, component):
    return np.random.default_rng(np.random.SeedSequence([cfg["stream_namespace"], seed, component]))


def _produce(cfg, graph):
    """Input graph injection supports synthetic tests without scientific files."""
    validate_config(cfg)
    seeds, n_exp = cfg["seeds"], cfg["exposures_per_shell"]
    n, m, kcs, edges = 6 * n_exp, graph.n_mbon, graph.n_kc, len(graph.pre)
    checkpoints = np.arange(0, n + 1, 6 * cfg["checkpoint_every_exposures"], dtype=int)
    bases, geometries = [], []
    # Every source-geometry gate is checked before generating task/action streams.
    for seed in seeds:
        base = ShellActor(graph, seed, cfg["head_lr"], False)
        witness = base.witness()
        z = witness["probe_decision"]
        try:
            geom = geometry(z[:, 1] - z[:, 0], cfg["gram_condition_max"], cfg["geometry_tolerance"])
        except ValueError as error:
            raise ValueError(f"seed {seed}: {error}") from error
        d = geom["feature_gaps"]
        shared_impulse = (.5 * cfg["head_lr"] * d) @ d.T
        isolated_impulse = (.5 * cfg["head_lr"] * np.diag(geom["gram"])[:, None] * geom["dual_rows"]) @ d.T
        own_error = float(np.max(np.abs(np.diag(shared_impulse) - np.diag(isolated_impulse))))
        off_error = float(np.max(np.abs(isolated_impulse[~np.eye(6, dtype=bool)])))
        if own_error > cfg["geometry_tolerance"] or off_error > cfg["geometry_tolerance"]:
            raise ValueError(f"seed {seed}: Static own-logit match/isolation gate failed")
        bases.append(witness); geometries.append(geom)
    shape = (len(seeds), 3, n)
    arrays = {key: np.zeros(shape, dtype=np.int8 if key in ("actions", "rewards", "head_update_calls", "internal_update_calls") else float)
              for key in ("probabilities", "actions", "rewards", "head_steps", "internal_steps", "head_update_calls", "internal_update_calls")}
    arrays.update({"seeds": np.asarray(seeds, dtype=np.int64), "checkpoints": checkpoints,
                   "shells": np.empty((len(seeds), n), dtype=np.int8),
                   "target_go": np.empty((len(seeds), 6), dtype=np.int8),
                   "potential_rewards": np.empty((len(seeds), n), dtype=np.int8),
                   "action_uniforms": np.empty((len(seeds), n)), "outcome_uniforms": np.empty((len(seeds), n)),
                   "heads": np.zeros((len(seeds), 3, len(checkpoints), m)),
                   "probe_probabilities": np.empty((len(seeds), 3, len(checkpoints), 6)),
                   "pre": graph.pre.copy(), "post": graph.post.copy()})
    for key in ("feature_gaps", "gram", "dual_rows", "rank", "gram_condition", "dual_residual_max"):
        arrays[key] = np.stack([g[key] for g in geometries])
    for out_key, source_key in (("encoder", "encoder"), ("bias", "bias"), ("feature_scale", "feature_scale"),
                                ("weights_initial", "weights"), ("kc_initial", "probe_kc"),
                                ("mbon_initial", "probe_mbon"), ("decision_initial", "probe_decision")):
        arrays[out_key] = np.stack([w[source_key] for w in bases])
    arrays["weights_final"] = np.empty((len(seeds), 3, edges))
    arrays["kc_final"] = np.empty((len(seeds), 3, 6, 2, kcs))
    arrays["mbon_final"] = np.empty((len(seeds), 3, 6, 2, m))
    arrays["decision_final"] = np.empty((len(seeds), 3, 6, 2, m))
    for key in ("influence_shared", "influence_isolated", "influence_q_reference"):
        arrays[key] = np.empty((len(seeds), 2, 6, 6))
    rows = []
    for si, seed in enumerate(seeds):
        d, dual = arrays["feature_gaps"][si], arrays["dual_rows"][si]
        target = np.zeros(6, dtype=np.int8)
        target[_rng(cfg, seed, 0).permutation(6)[:3]] = 1
        schedule_rng = _rng(cfg, seed, 1)
        shells = np.concatenate([schedule_rng.permutation(np.arange(1, 7)) for _ in range(n_exp)]).astype(np.int8)
        outcome_u, action_u = _rng(cfg, seed, 2).random(n), _rng(cfg, seed, 3).random(n)
        potential = np.where(outcome_u < cfg["reward_probability"], 2 * target[shells - 1] - 1, 0).astype(np.int8)
        arrays["target_go"][si], arrays["shells"][si], arrays["potential_rewards"][si] = target, shells, potential
        arrays["outcome_uniforms"][si], arrays["action_uniforms"][si] = outcome_u, action_u
        for ri, reward in enumerate(SIGNS):
            shared_updates = .5 * cfg["head_lr"] * reward * d
            isolated_updates = .5 * cfg["head_lr"] * reward * np.diag(arrays["gram"][si])[:, None] * dual
            arrays["influence_shared"][si, ri] = shared_updates @ d.T
            arrays["influence_isolated"][si, ri] = isolated_updates @ d.T
            arrays["influence_q_reference"][si, ri] = np.eye(6) * reward * cfg["q_reference"]["alpha"] * cfg["q_reference"]["beta"]
            err = np.max(np.abs(np.diag(arrays["influence_shared"][si, ri]) - np.diag(arrays["influence_isolated"][si, ri])))
            if err > cfg["geometry_tolerance"]:
                raise ValueError(f"Static own-logit matching failed for seed {seed}")
        for ci, condition in enumerate(CONDITIONS):
            model = HeadPolicy(condition, d, dual, cfg["head_lr"])
            arrays["probe_probabilities"][si, ci, 0] = model.probabilities()
            seen = np.zeros(6, dtype=int)
            checkpoint_index = 1
            for ti, shell_value in enumerate(shells):
                shell = int(shell_value)
                seen[shell - 1] += 1
                p = float(model.probabilities()[shell - 1])
                action = int(action_u[ti] < p)
                reward = int(potential[ti]) if action else 0
                step = model.observe(shell, action, reward)
                at = (si, ci, ti)
                arrays["probabilities"][at], arrays["actions"][at], arrays["rewards"][at] = p, action, reward
                arrays["head_steps"][at], arrays["head_update_calls"][at] = step, int(condition != "uniform")
                rows.append({"seed": seed, "condition": condition, "trial": ti + 1, "shell": shell,
                             "exposure": int(seen[shell - 1]), "target_go": int(target[shell - 1]),
                             "action": action, "reward": reward, "p_go": p,
                             "p_correct": p if target[shell - 1] else 1 - p,
                             "head_step_l2": step, "internal_step_l2": 0.,
                             "head_update_called": int(condition != "uniform"), "internal_update_called": 0})
                if checkpoint_index < len(checkpoints) and ti + 1 == checkpoints[checkpoint_index]:
                    arrays["heads"][si, ci, checkpoint_index] = model.head
                    arrays["probe_probabilities"][si, ci, checkpoint_index] = model.probabilities()
                    checkpoint_index += 1
            arrays["weights_final"][si, ci] = arrays["weights_initial"][si]
            arrays["kc_final"][si, ci] = arrays["kc_initial"][si]
            arrays["mbon_final"][si, ci] = arrays["mbon_initial"][si]
            arrays["decision_final"][si, ci] = arrays["decision_initial"][si]
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite recorded artifact")
    return arrays, rows


def produce(cfg, root, cache=None):
    """Read the pinned graph; return artifacts without writing output files."""
    validate_config(cfg)
    return _produce(cfg, load_projection(Path(root) / cfg["graph"]))


def summarize(cfg, arrays, rows):
    validate_config(cfg)
    expected_trials = len(cfg["seeds"]) * 3 * cfg["exposures_per_shell"] * 6
    if len(rows) != expected_trials:
        raise ValueError("Incomplete trial ledger")
    outcomes, acquisition, effects = [], [], []
    crit = cfg["criteria"]
    for si, seed in enumerate(cfg["seeds"]):
        target = arrays["target_go"][si]
        values = []
        for ci, condition in enumerate(CONDITIONS):
            p = arrays["probe_probabilities"][si, ci]
            correct = np.where(target[None, :] == 1, p, 1 - p).mean(axis=1)
            before, after = float(correct[0]), float(correct[-1])
            values.append(after)
            outcomes.append({"seed": seed, "condition": condition, "before_accuracy": before,
                             "after_accuracy": after, "gain": after - before,
                             "acquired": bool(after >= crit["acquisition_accuracy"] and after - before >= crit["acquisition_gain"]),
                             "head_update_calls": int(arrays["head_update_calls"][si, ci].sum()),
                             "internal_update_calls": int(arrays["internal_update_calls"][si, ci].sum()),
                             "head_nonzero_steps": int(np.count_nonzero(arrays["head_steps"][si, ci])),
                             "internal_nonzero_steps": int(np.count_nonzero(arrays["internal_steps"][si, ci])),
                             "head_change_l2": float(np.linalg.norm(arrays["heads"][si, ci, -1] - arrays["heads"][si, ci, 0])),
                             "internal_change_l2": float(np.linalg.norm(arrays["weights_final"][si, ci] - arrays["weights_initial"][si])),
                             "kc_change_l2": float(np.linalg.norm(arrays["kc_final"][si, ci] - arrays["kc_initial"][si])),
                             "mbon_change_l2": float(np.linalg.norm(arrays["mbon_final"][si, ci] - arrays["mbon_initial"][si])),
                             "decision_feature_change_l2": float(np.linalg.norm(arrays["decision_final"][si, ci] - arrays["decision_initial"][si]))})
        effects.append({"seed": seed, "isolated_minus_shared_endpoint": values[1] - values[0]})
    for condition in CONDITIONS:
        chosen = [r for r in outcomes if r["condition"] == condition]
        count = sum(r["acquired"] for r in chosen)
        acquisition.append({"condition": condition, "acquired_seeds": count, "n_seeds": len(chosen),
                            "criterion_met": count / len(chosen) >= crit["seed_pass_fraction"],
                            "mean_endpoint_accuracy": float(np.mean([r["after_accuracy"] for r in chosen]))})
    differences = [r["isolated_minus_shared_endpoint"] for r in effects]
    mean_effect, positive = float(np.mean(differences)), sum(x > 0 for x in differences)
    effect_pass = mean_effect >= crit["endpoint_effect"] and positive / len(differences) >= crit["positive_seed_fraction"]
    isolated_pass = next(r["criterion_met"] for r in acquisition if r["condition"] == "isolated")
    static = []
    for si, seed in enumerate(cfg["seeds"]):
        shared, isolated, q = (arrays[k][si, 1] for k in ("influence_shared", "influence_isolated", "influence_q_reference"))
        off = ~np.eye(6, dtype=bool)
        static.append({"seed": seed, "rank": int(arrays["rank"][si]),
                       "gram_condition": float(arrays["gram_condition"][si]),
                       "dual_residual_max": float(arrays["dual_residual_max"][si]),
                       "matched_own_logit_error_max": float(np.max(np.abs(np.diag(shared) - np.diag(isolated)))),
                       "shared_mean_abs_offdiagonal": float(np.mean(np.abs(shared[off]))),
                       "isolated_max_abs_offdiagonal": float(np.max(np.abs(isolated[off]))),
                       "actor_mean_initial_own_logit_step": float(np.diag(shared).mean()),
                       "q_reference_initial_own_logit_step": float(np.diag(q).mean()),
                       "q_reference_over_actor_mean_step": float(np.diag(q).mean() / np.diag(shared).mean())})
    m = arrays["heads"].shape[-1]
    return {"status": "completed", "study_id": cfg["study_id"], "outcomes": outcomes,
            "acquisition": acquisition, "paired_effects": effects, "static_geometry": static,
            "counts": {"actual_choice_feedback_trials": expected_trials,
                       "head_update_calls": int(arrays["head_update_calls"].sum()),
                       "internal_update_calls": int(arrays["internal_update_calls"].sum()),
                       "new_human_observations": 0, "static_signed_source_cases": len(cfg["seeds"]) * 2 * 6,
                       "computational_seed_units": len(cfg["seeds"])},
            "H_credit": {"effect_criterion_met": bool(effect_pass), "mean_endpoint_effect": mean_effect,
                         "positive_seeds": positive, "n_seeds": len(differences),
                         "isolated_acquisition_gate": bool(isolated_pass),
                         "supported": bool(effect_pass and isolated_pass),
                         "interpretation": "stationary computational cross-cue update intervention; not the cause of the historical human-task failure"},
            "resources": {"actual_choice_feedback_trials": expected_trials,
                          "head_update_calls": int(arrays["head_update_calls"].sum()),
                          "internal_update_calls": int(arrays["internal_update_calls"].sum()),
                          "head_coordinates_per_model": m, "active_feature_coordinates": int(np.any(arrays["feature_gaps"] != 0, axis=(0, 1)).sum()),
                          "trainable_head_coordinates": {"shared": m, "isolated": m, "uniform": 0},
                          "isolated_extra_fixed_coefficients": 6 * m,
                          "isolated_precomputation": "six-by-six Gram solve; not an anatomical connection or trainable biological parameter",
                          "equal_update_l2_not_claimed": True, "same_local_own_logit_step": True,
                          "diverged_autonomous_paths_have_equal_steps_not_claimed": True},
            "acquisition_scope": "six stationary identity cues; not reversal, held-out human validation, or mathematical generalization",
            "historical_results_replaced": False, "human_similarity_tested": False,
            "biological_mechanism_tested": False, "calibration_objective_causal_effect_tested": False}
