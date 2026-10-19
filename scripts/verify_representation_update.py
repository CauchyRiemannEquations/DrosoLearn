"""Independent replay of representation and head-update diagnostics.

The current scientific producer and its algebra helpers are never imported.
Only the earlier immutable independent graph-transform arithmetic is reused.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "representation_independent_frozen_graph", Path(__file__).with_name("verify_human_shell.py"))
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)

RAW_STRINGS = ((0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 1, 0), (1, 0, 0, 1))
TARGETS = np.array([1, 1, 0, 0], dtype=int)
INPUTS = ("raw", "identity")
TOPOLOGIES = ("real", "rewired", "ann")
UPDATES = ("shared", "projected")
INPUT_CATALOG = "data/synthetic/representation_update_v1/input_catalog.json"


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "representation_update_v1", "scope": "prospective_representation_update_diagnosis",
             "module": "drosolearn.representation_update", "protocol": "docs/representation-update-protocol.md",
             "topologies": list(TOPOLOGIES), "representations": list(INPUTS), "updates": list(UPDATES),
             "model_seeds": list(range(17101, 17111)), "training_trials": 400,
             "checkpoints": [0, 50, 100, 200, 400], "head_lr": .1, "rng_namespace": 202610193,
             "svd_rcond": 1e-12, "numeric_tolerance": 1e-8, "exact_condition_max": 1e6,
             "acquisition_threshold": .8, "acquisition_gain": .25, "class_threshold": .7,
             "population_pass_count": 8, "effect_threshold": .1, "positive_seed_count": 8,
             "normalization": "per_input_initial_fixed_MBON_gap_unit_L2", "input_catalog": INPUT_CATALOG,
             "graph": "data/flywire_783_mb_left_kc512_s701", "external_manifest": None}
    fixture = allow_fixture and cfg.get("scope") == "integration_test"
    changes = {"scope", "model_seeds", "training_trials", "checkpoints", "population_pass_count", "positive_seed_count"} if fixture else set()
    for key, value in fixed.items():
        if key not in changes: core.demand(cfg.get(key) == value, "Fixed independent diagnosis: " + key)
    if fixture:
        seeds, n, grid = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
        core.demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(set(seeds)) == len(seeds)
                    and all(type(s) is int and 90001 <= s <= 90100 for s in seeds), "Invented diagnostic seed axis")
        core.demand(type(n) is int and 4 <= n <= 16 and n % 4 == 0, "Invented diagnostic feedback budget")
        core.demand(grid == sorted(set(grid)) and grid[0] == 0 and grid[-1] == n
                    and all(type(step) is int and 0 <= step <= n and step % 4 == 0 for step in grid), "Invented checkpoint grid")
        core.demand(cfg["population_pass_count"] == cfg["positive_seed_count"] == len(seeds), "Invented seed criteria")


def finite_catalog():
    """Literal four identities; lexicographic ID map never consults class labels."""
    raw = np.zeros((4, 7))
    raw[:, :4] = np.asarray(RAW_STRINGS) / 3.
    raw[:, 6] = 2. / 3.
    identity = np.zeros((4, 7))
    dictionary = sorted(RAW_STRINGS)
    for i, sequence in enumerate(RAW_STRINGS):
        identity[i, dictionary.index(sequence)] = np.sqrt(2.) / 3.
    identity[:, 6] = 2. / 3.
    return raw, identity


def independent_geometry(differences, rcond=1e-12):
    """SVD cutoff and Moore-Penrose directions independently from raw features."""
    differences = np.asarray(differences, dtype=float)
    core.demand(differences.ndim == 2 and differences.shape[0] == 4
                and differences.shape[1] >= 1 and np.isfinite(differences).all(),
                "Four finite nonempty feature rows required")
    row_norms = np.linalg.norm(differences, axis=1)
    core.demand(np.all(row_norms > 0), "Zero feature rows cannot silently use a fallback")
    d = differences / row_norms[:, None]
    left, singular, right = np.linalg.svd(d, full_matrices=False)
    retained = singular > rcond * singular[0]
    inverse = (right[retained].T / singular[retained]) @ left[:, retained].T
    projector = d @ inverse
    diagonal = np.diag(projector)
    core.demand(np.all(diagonal > 0), "Projected own-coordinate score is positive")
    projected = (inverse / diagonal[None, :]).T
    return {"norms": row_norms, "D": d, "singular_values": singular,
            "rank": int(retained.sum()), "pinv": inverse, "projector": projector,
            "shared_directions": d.copy(), "projected_directions": projected,
            "shared_influence": d @ d.T, "projected_influence": projected @ d.T,
            "shared_direction_norms": np.linalg.norm(d, axis=1),
            "projected_direction_norms": np.linalg.norm(projected, axis=1)}


def minimum_norm_margin(features, signed_labels):
    """Enumerate every nonempty subset of four candidate active constraints.

    Candidate multipliers solve the active Gram system, then the primal and
    nonnegative dual constraints are tested. This is an analytical label-given
    witness, never feedback training.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(signed_labels, dtype=float)
    core.demand(x.ndim == 2 and x.shape[0] == 4 and y.shape == (4,)
                and np.isfinite(x).all() and np.isin(y, (-1, 1)).all(),
                "Four finite label-signed margin constraints required")
    signed = y[:, None] * x
    candidates = []
    for mask in range(1, 16):
        active = np.array([i for i in range(4) if mask & (1 << i)])
        block = signed[active]
        multipliers = np.linalg.pinv(block @ block.T, rcond=1e-12) @ np.ones(len(active))
        witness = block.T @ multipliers
        margin = signed @ witness
        feasible = bool(np.min(multipliers) >= -1e-9 and np.max(np.abs(margin[active] - 1.)) <= 1e-9
                        and np.min(margin) >= 1. - 1e-9)
        candidates.append({"mask": mask, "active": active, "multipliers": multipliers,
                           "head": witness, "margins": margin, "feasible": feasible,
                           "norm": float(np.linalg.norm(witness))})
    feasible = [item for item in candidates if item["feasible"]]
    best = min(feasible, key=lambda item: (item["norm"], item["mask"])) if feasible else None
    return candidates, best


def feature_evidence(graph, seed, catalog):
    """Original random seven-input encoder and fixed normalized edge weights."""
    rng = np.random.default_rng(seed + 100000)
    encoder = rng.normal(0., .8, (graph["n_kc"], 7))
    bias = rng.uniform(-.25, .25, graph["n_kc"])
    pre, post = graph["pre"], graph["post"]
    incoming = np.maximum(np.bincount(post, weights=graph["counts"], minlength=graph["n_mbon"]), 1.)
    weights = graph["counts"] / incoming[post]
    pairs = np.stack((np.zeros_like(catalog), catalog), axis=1)
    kc = np.maximum(pairs @ encoder.T + bias, 0.)
    mb = np.empty((4, 2, graph["n_mbon"]))
    for i in range(4):
        for action in range(2):
            mb[i, action] = np.bincount(post, weights=weights * kc[i, action, pre], minlength=graph["n_mbon"])
    differences = mb[:, 1] - mb[:, 0]
    return {"encoder": encoder, "bias": bias, "pre": pre.copy(), "post": post.copy(),
            "weights": weights, "probe_kc": kc, "probe_mbon": mb,
            "differences": differences}


def independent_inverse(matrix, rcond=1e-12):
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    use = s > rcond * s[0]
    return (vh[use].T / s[use]) @ u[:, use].T


def matrix_report(matrix, rcond=1e-12):
    matrix = np.asarray(matrix, float)
    singular = np.linalg.svd(matrix, compute_uv=False)
    cutoff = max(matrix.shape) * np.finfo(float).eps * singular[0]
    retained = singular > rcond * singular[0]
    count = int(retained.sum())
    full = count == matrix.shape[0]
    return {"shape": list(matrix.shape), "singular_values": singular,
            "strict_rank": int(np.count_nonzero(singular > cutoff)), "strict_rank_cutoff": float(cutoff),
            "retained_rank": count, "retained_rank_cutoff": float(rcond * singular[0]),
            "rcond": rcond, "full_row_rank": full,
            "full_row_condition": float(singular[0] / singular[-1]) if full else None,
            "retained_condition": float(singular[0] / singular[retained][-1]) if count else None}


def geometry_report(rows, rcond=1e-12, tolerance=1e-8, exact_condition_max=1e6):
    g = independent_geometry(rows, rcond)
    diag = np.diag(g["projector"]).copy()
    off = g["projected_influence"].copy()
    np.fill_diagonal(off, 0.)
    own = float(np.max(np.abs(np.diag(g["projected_influence"]) - np.diag(g["shared_influence"]))))
    largest_off = float(np.max(np.abs(off)))
    u, s, _ = np.linalg.svd(g["D"], full_matrices=False)
    retained = u[:, s > rcond * s[0]]
    diagnostic = matrix_report(g["D"], rcond)
    qualified = own <= tolerance
    exact = bool(qualified and diagnostic["full_row_rank"]
                 and diagnostic["full_row_condition"] <= exact_condition_max and largest_off <= tolerance)
    return {"raw_rows": np.asarray(rows).copy(), "row_norms": g["norms"], "unit_rows": g["D"],
            "raw_diagnostics": matrix_report(rows, rcond), "unit_diagnostics": diagnostic,
            "pseudoinverse": g["pinv"], "projector": g["projector"], "projector_diagonal": diag,
            "directions": g["projected_directions"], "influence_shared": g["shared_influence"],
            "influence_projected": g["projected_influence"], "own_step_residual": own,
            "off_diagonal_max": largest_off,
            "projector_symmetry_residual": float(np.max(np.abs(g["projector"] - g["projector"].T))),
            "projector_idempotence_residual": float(np.max(np.abs(g["projector"] @ g["projector"] - g["projector"]))),
            "projector_svd_residual": float(np.max(np.abs(g["projector"] - retained @ retained.T))),
            "direction_norms": g["projected_direction_norms"], "projected_spillover_squared": np.sum(off ** 2, axis=1),
            "minimum_spillover_squared_analytic": 1. / diag - 1., "numeric_update_qualified": bool(qualified),
            "exact_isolation_qualified": exact, "retained_subspace_only": True,
            "tolerance": tolerance, "exact_condition_max": exact_condition_max}


def margin_report(rows, labels, rcond=1e-12, tolerance=1e-8):
    x = np.asarray(rows, float)
    signed = np.asarray(labels)[:, None] * x
    heads, residuals, minimums, flags = [], [], [], []
    for mask in range(1, 16):
        selected = [i for i in range(4) if mask & (1 << i)]
        block = signed[selected]
        head = independent_inverse(block, rcond) @ np.ones(len(selected))
        margins = signed @ head
        residual = float(np.max(np.abs(block @ head - 1.)))
        heads.append(head); residuals.append(residual); minimums.append(float(np.min(margins)))
        flags.append(bool(residual <= tolerance and np.min(margins) >= 1. - tolerance))
    heads, flags = np.asarray(heads), np.asarray(flags, bool)
    norms = np.linalg.norm(heads, axis=1)
    selected = np.flatnonzero(flags)
    best = int(selected[np.argmin(norms[selected])]) if len(selected) else None
    head = heads[best].copy() if best is not None else np.zeros(x.shape[1])
    return {"status": "feasible_under_tolerance" if best is not None else "no_feasible_candidate_under_tolerance",
            "feasible": best is not None, "head": head, "signed_margins": signed @ head,
            "head_l2": float(norms[best]) if best is not None else None,
            "active_subset_mask": best + 1 if best is not None else None, "examined_subset_count": 15,
            "feasible_candidate_count": int(flags.sum()), "candidate_heads": heads, "candidate_l2": norms,
            "candidate_equality_residuals": np.asarray(residuals), "candidate_min_margins": np.asarray(minimums),
            "candidate_feasible": flags, "rcond": rcond, "tolerance": tolerance,
            "failed_search_is_impossibility_proof": False, "analytic_probe_not_reward_learning": True}


class ReplayCell:
    """Reward-only head learner; input geometry fixed before feedback."""
    def __init__(self, graph, seed, inputs, rule, rate=.1):
        core.demand(rule in UPDATES, "Declared independent update rule required")
        self.evidence = feature_evidence(graph, seed, inputs)
        self.geometry = geometry_report(self.evidence["differences"])
        self.rule, self.rate = rule, rate
        self.head = np.zeros(graph["n_mbon"])
        self.calls = 0

    def logits(self):
        return self.geometry["unit_rows"] @ self.head

    def probabilities(self):
        return 1. / (1. + np.exp(-np.clip(self.logits(), -40., 40.)))

    def observe(self, index, action, reward):
        core.demand(type(index) is int and 0 <= index < 4 and type(action) is int and action in (0, 1)
                    and type(reward) is int and reward in (-1, 0, 1)
                    and (action == 1 or reward == 0), "Observed action and obtained reward only")
        p = float(self.probabilities()[index])
        direction = self.geometry["unit_rows"][index] if self.rule == "shared" else self.geometry["directions"][index]
        before = self.head.copy()
        self.head += self.rate * (action - p) * reward * direction
        self.calls += 1
        return float(np.linalg.norm(self.head - before))


def independent_catalog_arrays():
    raw, identity = finite_catalog()
    return {"inputs": np.stack((raw, identity)), "targets": TARGETS.copy(),
            "strings": np.asarray(RAW_STRINGS), "lexicographic_ids": np.array([0, 3, 1, 2])}


def numeric_fields(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(numeric_fields(item, name + "_"))
        elif not isinstance(item, str):
            if key in {"full_row_condition", "retained_condition", "head_l2", "active_subset_mask"}:
                result[name + "_defined"] = np.asarray(item is not None)
            result[name] = np.asarray(0. if item is None else item)
    return result


def initial_witness(evidence):
    gap = evidence["differences"]
    scale = float(np.sqrt(np.mean(np.sum(gap ** 2, axis=-1))))
    weights = evidence["weights"]
    return {"head": np.zeros(evidence["probe_mbon"].shape[-1]), "weights": weights.copy(),
            "encoder": evidence["encoder"].copy(), "bias": evidence["bias"].copy(),
            "pre": evidence["pre"].copy(), "post": evidence["post"].copy(),
            "initial_weights": weights.copy(), "upper": 2 * weights, "feature_scale": np.asarray(scale),
            "probe_kc": evidence["probe_kc"].copy(), "probe_mbon": evidence["probe_mbon"].copy(),
            "probe_decision": evidence["probe_mbon"] / scale,
            "value_table": np.empty(0), "value_catalog": np.empty((0, 7))}


def expected_artifacts(cfg, base_graph):
    """Rebuild every numeric state and every feedback row from fixed raw inputs."""
    seeds, n, checkpoints = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
    data = independent_catalog_arrays()
    arrays = {key: value.copy() for key, value in data.items()}
    shape = (3, 2, 2, len(seeds), n)
    integers = {"actions", "rewards", "head_update_calls", "internal_update_calls", "value_update_calls"}
    for key in ("probabilities", "actions", "rewards", "head_steps", "own_score_steps", "other_score_steps",
                "internal_steps", "value_steps", "head_update_calls", "internal_update_calls", "value_update_calls"):
        arrays["trial_" + key] = np.zeros(shape, dtype=int if key in integers else float)
    cshape = shape[:-1] + (len(checkpoints),)
    arrays["checkpoint_logits"] = np.empty(cshape + (4,))
    arrays["checkpoint_probabilities"] = np.empty(cshape + (4,))
    arrays["checkpoint_heads"] = np.empty(cshape + (base_graph["n_mbon"],))
    for field in ("head", "internal", "value"):
        arrays["checkpoint_" + field + "_calls"] = np.zeros(cshape, dtype=int)
    arrays["schedules"] = np.empty((len(seeds), n), dtype=int)
    arrays["uniforms"] = np.empty((len(seeds), n))
    arrays["model_seeds"], arrays["checkpoints"] = np.asarray(seeds), np.asarray(checkpoints)
    for ri, name in enumerate(INPUTS):
        arrays.update(numeric_fields(margin_report(data["inputs"][ri], 2 * TARGETS - 1), "a_" + name + "_"))
    static, rows = {}, []
    for si, seed in enumerate(seeds):
        schedule_rng = np.random.default_rng(np.random.SeedSequence([202610193, seed, 0]))
        schedule = np.concatenate([schedule_rng.permutation(4) for _ in range(n // 4)])
        uniforms = np.random.default_rng(np.random.SeedSequence([202610193, seed, 1])).random(n)
        arrays["schedules"][si], arrays["uniforms"][si] = schedule, uniforms
        for ti, topology in enumerate(TOPOLOGIES):
            offset = 110000 if topology == "rewired" else 120000 if topology == "ann" else 0
            graph = core.transform_graph(base_graph, topology, seed + offset)
            for ri, name in enumerate(INPUTS):
                evidence = feature_evidence(graph, seed, data["inputs"][ri])
                witness = initial_witness(evidence)
                geometry = geometry_report(evidence["differences"])
                core.demand(geometry["numeric_update_qualified"], "Own-score tolerance violated; no independent fallback")
                margin = margin_report(geometry["unit_rows"], 2 * TARGETS - 1)
                group = topology + "_" + name + "_"
                for prefix, record in (("w_", witness), ("g_", numeric_fields(geometry)), ("m_", numeric_fields(margin))):
                    for key, value in record.items(): static.setdefault(prefix + group + key, []).append(np.asarray(value))
                for ui, update in enumerate(UPDATES):
                    d = geometry["unit_rows"]
                    directions = d if update == "shared" else geometry["directions"]
                    head = np.zeros(graph["n_mbon"])
                    calls = 0
                    def save(step):
                        at = (ti, ri, ui, si, checkpoints.index(step))
                        logits = d @ head
                        arrays["checkpoint_logits"][at] = logits
                        arrays["checkpoint_probabilities"][at] = 1. / (1. + np.exp(-np.clip(logits, -40., 40.)))
                        arrays["checkpoint_heads"][at] = head
                        arrays["checkpoint_head_calls"][at] = calls
                    save(0)
                    for trial, item in enumerate(schedule):
                        index = int(item)
                        target = int(TARGETS[index])
                        logits = d @ head
                        probability = float(1. / (1. + np.exp(-np.clip(logits[index], -40., 40.))))
                        action = int(uniforms[trial] < probability)
                        reward = (1 if target else -1) if action else 0
                        old_head = head.copy()
                        head += .1 * (action - probability) * reward * directions[index]
                        calls += 1
                        step = head - old_head
                        score_step = d @ step
                        head_step = float(np.linalg.norm(step))
                        own_step = float(score_step[index])
                        other_step = float(np.linalg.norm(score_step[np.arange(4) != index]))
                        at = (ti, ri, ui, si, trial)
                        for key, value in (("probabilities", probability), ("actions", action), ("rewards", reward),
                                           ("head_steps", head_step), ("own_score_steps", own_step),
                                           ("other_score_steps", other_step), ("head_update_calls", calls)):
                            arrays["trial_" + key][at] = value
                        rows.append({"topology": topology, "representation": name, "update": update,
                                     "seed": seed, "trial": trial + 1, "example_index": index, "target": target,
                                     "probability_go": probability, "action": action, "reward": reward,
                                     "head_update_calls": calls, "internal_update_calls": 0, "value_update_calls": 0,
                                     "head_step_l2": head_step, "own_score_step": own_step,
                                     "other_score_step_l2": other_step, "internal_step_l2": 0., "value_step_l2": 0.,
                                     "head_nonzero_step": int(head_step > 0), "internal_nonzero_step": 0, "value_nonzero_step": 0})
                        if trial + 1 in checkpoints: save(trial + 1)
                for key in ("weights", "encoder", "bias", "probe_kc", "probe_mbon"):
                    static.setdefault("finalfixed_" + group + key, []).append(witness[key].copy())
    arrays.update({key: np.stack(value) for key, value in static.items()})
    return arrays, rows


def raw_certificate():
    rows = np.array([[0, 1, 0, 1, 0, 0, 2], [1, 0, 1, 0, 0, 0, 2],
                     [0, 1, 1, 0, 0, 0, 2], [1, 0, 0, 1, 0, 0, 2]], dtype=np.int64)
    relation = np.array([1, 1, -1, -1])
    core.demand(np.array_equal(relation @ rows, np.zeros(7, int)), "Exact literal parallelogram certificate")
    minor = rows[:3, :3]
    determinant = int(sum((1 if permutation in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1)
                          * np.prod([minor[i, permutation[i]] for i in range(3)])
                          for permutation in itertools.permutations(range(3))))
    differences = rows[[1, 2]] - rows[0]
    affine_determinant = int(differences[0, 0] * differences[1, 2] - differences[0, 2] * differences[1, 0])
    example = np.array([1, -1, 0, 2, 0, 0, 0])
    return {"scaled_integer_rows": rows.tolist(), "scale_denominator": 3,
            "row_relation_coefficients": relation.tolist(), "integer_relation_residual": (relation @ rows).tolist(),
            "rank_minor_rows": [0, 1, 2], "rank_minor_columns": [0, 1, 2], "rank_minor_determinant": determinant,
            "affine_difference_rows": [1, 2], "affine_minor_columns": [0, 2],
            "affine_minor_determinant": affine_determinant, "exact_row_rank": 3, "exact_affine_dimension": 2,
            "distinct_inputs_preserve_identity": True, "positive_negative_midpoint_identical": True,
            "strict_affine_separation_impossible_for_labels": relation.tolist(),
            "three_correct_example_head": example.tolist(), "three_correct_example_scaled_logits": (rows @ example).tolist(),
            "maximum_strict_correct_decisions": 3, "four_training_examples_only": True,
            "nonlinear_feature_separation_not_ruled_out": True}


def expected_summary(cfg, arrays):
    p = arrays["checkpoint_probabilities"]
    correct = np.where(TARGETS == 1, p, 1 - p)
    before = correct[..., 0, :]
    final = correct[..., -1, :]
    scores = final.mean(-1)
    acquired = ((scores >= cfg["acquisition_threshold"])
                & (scores - before.mean(-1) >= cfg["acquisition_gain"])
                & (final[..., :2].mean(-1) >= cfg["class_threshold"])
                & (final[..., 2:].mean(-1) >= cfg["class_threshold"]))
    families, geometries, contrasts = [], [], []
    for ti, topology in enumerate(TOPOLOGIES):
        for ri, representation in enumerate(INPUTS):
            key = "g_" + topology + "_" + representation + "_"
            shared_off = arrays[key + "influence_shared"].copy()
            shared_off[:, np.arange(4), np.arange(4)] = 0.
            shared_spill = np.mean(np.sum(shared_off ** 2, axis=-1), axis=-1)
            projected_spill = arrays[key + "projected_spillover_squared"].mean(-1)
            gs = []
            for si, seed in enumerate(cfg["model_seeds"]):
                margin_key = "m_" + topology + "_" + representation + "_"
                gs.append({"seed": seed,
                           "raw_feature_rank": int(arrays[key + "raw_diagnostics_retained_rank"][si]),
                           "unit_feature_rank": int(arrays[key + "unit_diagnostics_retained_rank"][si]),
                           "unit_singular_values": arrays[key + "unit_diagnostics_singular_values"][si].tolist(),
                           "unit_full_row_condition": float(arrays[key + "unit_diagnostics_full_row_condition"][si])
                                if arrays[key + "unit_diagnostics_full_row_condition_defined"][si] else None,
                           "own_step_residual": float(arrays[key + "own_step_residual"][si]),
                           "off_diagonal_max": float(arrays[key + "off_diagonal_max"][si]),
                           "numeric_update_qualified": bool(arrays[key + "numeric_update_qualified"][si]),
                           "exact_isolation_qualified": bool(arrays[key + "exact_isolation_qualified"][si]),
                           "shared_mean_spillover_squared": float(shared_spill[si]),
                           "projected_mean_spillover_squared": float(projected_spill[si]),
                           "max_projected_direction_l2": float(arrays[key + "direction_norms"][si].max()),
                           "analytic_margin_feasible": bool(arrays[margin_key + "feasible"][si]),
                           "analytic_margin_head_l2": float(arrays[margin_key + "head_l2"][si])
                                if arrays[margin_key + "head_l2_defined"][si] else None})
            geometries.append({"topology": topology, "representation": representation, "seeds": gs})
            for ui, update in enumerate(UPDATES):
                individuals = []
                for si, seed in enumerate(cfg["model_seeds"]):
                    initial_score = float(before[ti, ri, ui, si].mean())
                    final_score = float(scores[ti, ri, ui, si])
                    individuals.append({"seed": seed, "initial_expected_accuracy": initial_score,
                                        "final_expected_accuracy": final_score, "gain": final_score - initial_score,
                                        "class_1_expected_accuracy": float(final[ti, ri, ui, si, :2].mean()),
                                        "class_0_expected_accuracy": float(final[ti, ri, ui, si, 2:].mean()),
                                        "acquisition_pass": bool(acquired[ti, ri, ui, si]),
                                        "final_head_l2": float(np.linalg.norm(arrays["checkpoint_heads"][ti, ri, ui, si, -1]))})
                families.append({"topology": topology, "representation": representation, "update": update,
                                 "mean_final_expected_accuracy": float(scores[ti, ri, ui].mean()),
                                 "sd_final_expected_accuracy": float(scores[ti, ri, ui].std(ddof=1)) if len(individuals) > 1 else 0.,
                                 "acquisition_pass_seeds": int(acquired[ti, ri, ui].sum()), "seeds": individuals})
        contrasts.append({"topology": topology,
                          "update_effect_raw_by_seed": (scores[ti, 0, 1] - scores[ti, 0, 0]).tolist(),
                          "representation_effect_shared_by_seed": (scores[ti, 1, 0] - scores[ti, 0, 0]).tolist(),
                          "interaction_by_seed": ((scores[ti, 1, 1] - scores[ti, 1, 0]) - (scores[ti, 0, 1] - scores[ti, 0, 0])).tolist(),
                          "projected_identity_minus_raw_by_seed": (scores[ti, 1, 1] - scores[ti, 0, 1]).tolist(),
                          "descriptive_only": ti != 0})
    raw_g, id_g = geometries[0]["seeds"], geometries[1]["seeds"]
    def decide(effect, pass_count, valid, reduction=None):
        effect = np.asarray(effect)
        positive = int(np.count_nonzero(effect > 0))
        supported = (effect.mean() >= cfg["effect_threshold"] and positive >= cfg["positive_seed_count"]
                     and pass_count >= cfg["population_pass_count"] and valid and (reduction is None or reduction > 0))
        return {"supported": bool(supported), "mean_effect": float(effect.mean()), "positive_seeds": positive,
                "treated_acquisition_pass_seeds": int(pass_count), "numeric_update_qualified_all": bool(valid),
                "mean_spillover_squared_reduction": reduction}
    reduction = float(np.mean([item["shared_mean_spillover_squared"] - item["projected_mean_spillover_squared"] for item in raw_g]))
    all_qualified = all(item["numeric_update_qualified"] for group in geometries for item in group["seeds"])
    h_update = decide(contrasts[0]["update_effect_raw_by_seed"], acquired[0, 0, 1].sum(),
                      all_qualified, reduction)
    h_representation = decide(contrasts[0]["representation_effect_shared_by_seed"], acquired[0, 1, 0].sum(),
                              all_qualified)
    return {"study_id": cfg["study_id"], "status": "completed", "families": families, "geometry": geometries,
            "contrasts": contrasts, "H_update": h_update, "H_representation": h_representation,
            "raw_input_certificate": raw_certificate(),
            "analytic_input_margin_feasible": {name: bool(arrays["a_" + name + "_feasible"]) for name in INPUTS},
            "counts": {"training_feedback_trials": 12 * len(cfg["model_seeds"]) * cfg["training_trials"],
                       "static_probe_cases": int(np.prod(p.shape)), "model_initializations": 12 * len(cfg["model_seeds"]),
                       "model_seeds": len(cfg["model_seeds"]), "fixed_feature_constructions": 6 * len(cfg["model_seeds"]),
                       "head_update_calls": int(arrays["trial_head_update_calls"][..., -1].sum()),
                       "internal_update_calls": 0, "value_update_calls": 0,
                       "head_nonzero_steps": int(np.count_nonzero(arrays["trial_head_steps"])),
                       "internal_nonzero_steps": 0, "value_nonzero_steps": 0},
            "resources": {"head_coordinates": int(arrays["checkpoint_heads"].shape[-1]),
                          "extra_projected_direction_coefficients": int(arrays["checkpoint_heads"].shape[-1]) * 4,
                          "direction_solver_SVD_per_topology_representation_seed": 1,
                          "diagnostic_SVD_per_topology_representation_seed": 2,
                          "analytic_margin_SVD_per_topology_representation_seed": 15,
                          "input_analytic_margin_SVD_calls": 30,
                          "total_producer_SVD_calls": 18 * 6 * len(cfg["model_seeds"]) + 30,
                          "head_L2_steps_matched": False,
                          "own_input_score_step_per_factor_matched": True, "same_input_L2_norm": True},
            "interpretation": {"artificial_history_buffer": True, "identity_dictionary_is_supplied_memorization_control": True,
                               "internal_synapses_fixed": True, "per_input_normalization_differs_from_prior_study": True,
                               "exact_historical_failure_cause_established": False, "generalization_tested": False,
                               "new_human_experiment": False, "shared_neural_mechanism_claim": False,
                               "education_effect_claim": False, "prior_failure_or_seed_discarded": False}}


def verify_input_catalog(cfg, root):
    path = core.contained(Path(root), INPUT_CATALOG)
    core.demand(cfg["input_catalog"] == INPUT_CATALOG, "Fixed artificial input catalog path")
    document = core.read_json(path)
    core.demand(set(document) == {"study_id", "representations", "catalog"}
                and document["study_id"] == cfg["study_id"] and document["representations"] == list(INPUTS),
                "Literal artificial input catalog identity")
    expected = independent_catalog_arrays()
    core.demand(type(document["catalog"]) is dict and set(document["catalog"]) == set(expected), "Literal input field closure")
    for key, value in expected.items():
        actual = np.asarray(document["catalog"][key])
        core.demand(actual.shape == value.shape and actual.dtype.kind == value.dtype.kind
                    and np.array_equal(actual, value), "Literal independently decoded catalog: " + key)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_against_graph(cfg, graph, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    expected, expected_rows = expected_artifacts(cfg, graph)
    core.demand(set(arrays) == set(expected), "Independent numeric artifact field closure differs")
    for key, value in expected.items():
        actual = np.asarray(arrays[key])
        core.demand(actual.dtype.kind in "biuf" and np.isfinite(actual).all(), "Finite numeric artifact: " + key)
        if value.dtype.kind in "biu":
            core.demand(actual.dtype.kind in "biu" and np.array_equal(actual, value), "Independent exact integer/bool artifact: " + key)
        else:
            core.array_close(actual, value, "Independent representation replay: " + key, atol=1e-9)
    core.demand(len(rows) == len(expected_rows), "Independent complete trial row count")
    for index, (actual, expected_row) in enumerate(zip(rows, expected_rows)):
        if all(type(value) is str for value in actual.values()): core.check_csv_row(actual, expected_row, "Representation trial " + str(index))
        else: core.compare_tree(actual, expected_row, "Representation trial " + str(index))
    core.compare_tree(summary, expected_summary(cfg, arrays), "Independent full scientific summary and claim closure")
    core.demand(np.array_equal(TARGETS, arrays["targets"]), "Original finite labels preserved")
    return {"arrays": len(expected), "independently_replayed_training_actions": len(expected_rows),
            "model_seeds": len(cfg["model_seeds"]), "conditions": 12,
            "checkpoint_static_probes": int(np.prod(arrays["checkpoint_probabilities"].shape)),
            "head_update_calls": int(arrays["trial_head_update_calls"][..., -1].sum()),
            "internal_update_calls": 0, "value_update_calls": 0,
            "all_checkpoints_and_trial_head_states_replayed": True,
            "fixed_internal_weights_encoder_and_probe_activity_checked": True,
            "full_static_geometry_and_15_margin_candidates_checked": True,
            "raw_input_integer_certificate_independently_checked": True,
            "same_input_L2_norm_and_label_free_lexicographic_ID_checked": True,
            "retained_subspace_and_own_step_numeric_qualification_checked": True,
            "head_L2_budget_not_equalized": True, "no_hidden_nogo_outcome_updates": True,
            "no_generalization_or_human_mechanism_claim": True}


def verify_science(cfg, root, arrays, rows, summary):
    validate_science(cfg)
    checksum = verify_input_catalog(cfg, root)
    graph = core.raw_graph(Path(root) / cfg["graph"])
    core.demand((graph["n_kc"], graph["n_mbon"], len(graph["pre"])) == (512, 48, 1474), "Pinned anatomical subgraph")
    result = verify_against_graph(cfg, graph, arrays, rows, summary)
    result["frozen_input_catalog_sha256"] = checksum
    return result


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
    from drosolearn.representation_envelope import verify_envelope, sha, write_new
    cfg, reg, meta = verify_envelope(args.root, args.config, args.registration,
                                    args.result_dir, args.publication_proof)
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
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "registration_sha256": sha(args.registration),
              "summary_sha256": sha(args.result_dir / "summary.json"),
              "scientific_producer_imported": False, "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
