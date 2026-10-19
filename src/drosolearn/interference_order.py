"""Finite contextual Go/NoGo curricula with paired prefixes and explicit idle copies."""
from __future__ import annotations

import copy
import json
import numpy as np

from .graph import load_projection
from .human_shell_models import build_graphs
from .pattern_rule import make_actor


CONDITIONS = ("fixed", "plastic", "rewired_fixed", "rewired_plastic",
              "ann_fixed", "ann_plastic", "linear", "uniform", "context_q")
SET_ORDER = ("A", "B")
STREAMS = ("prefix_A", "prefix_B", "AB", "AA", "BA", "BB", "joint")
SCIENCE = {"study_id": "interference_order_v1", "scope": "prospectively_registered_interference_order",
           "conditions": list(CONDITIONS), "model_seeds": list(range(15201, 15211)),
           "stage_trials": 360, "joint_trials": 720, "checkpoint_every": 60, "joint_checkpoint_every": 120,
           "head_lr": .1, "internal_lr_multiplier": .1, "context_q_alpha": .1, "context_q_beta": 4.,
           "rng_namespace": 202610192, "bootstrap_repeats": 2000,
           "bootstrap_entropy": [202610192, 15301, 2], "interval_quantiles": [.05, .95],
           "primary_condition": "plastic", "acquisition_threshold": .8, "acquisition_gain": .25,
           "class_threshold": .7, "joint_threshold": .8, "population_pass_count": 8,
           "conflict_threshold": .75,
           "interference_margin": .1, "order_margin": .05,
           "input_catalog": "data/synthetic/interference_order_v1/input_catalog.json",
           "normalization": "initial_twelve_context_input_feature_gap_RMS",
           "context_values": [-1, 1], "streams": list(STREAMS), "joint_order": "deterministic_A_B_alternation",
           "idle": "literal_prefix_state_copy_without_time_or_update"}
SCIENCE["graph"] = "data/flywire_783_mb_left_kc512_s701"
SCIENCE["graph_files"] = [SCIENCE["graph"] + "/" + name for name in
                          ("edges.csv", "neurons.json", "annotations.csv", "provenance.json")]
SCIENCE["input_files"] = [SCIENCE["input_catalog"]]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_config(cfg):
    fixture = cfg.get("scope") == "integration_test"
    mutable = {"scope", "model_seeds", "stage_trials", "joint_trials", "checkpoint_every",
               "joint_checkpoint_every", "population_pass_count", "bootstrap_repeats", "bootstrap_entropy", "graph"} if fixture else set()
    for key, value in SCIENCE.items():
        if key not in mutable:
            require(cfg.get(key) == value, "Fixed order science differs: " + key)
    if fixture:
        seeds, n = cfg["model_seeds"], cfg["stage_trials"]
        require(isinstance(seeds, list) and 1 <= len(seeds) <= 2 and len(seeds) == len(set(seeds))
                and all(type(seed) is int and 91001 <= seed <= 91100 for seed in seeds), "Synthetic order fixture seeds only")
        require(type(n) is int and n in (6, 12) and cfg["joint_trials"] == 2 * n
                and cfg["checkpoint_every"] == 6 and cfg["joint_checkpoint_every"] == 12,
                "Small balanced synthetic order budget only")
        require(cfg["population_pass_count"] == len(seeds) and cfg["bootstrap_repeats"] == 16
                and cfg["bootstrap_entropy"] == [202610192, 99001, 2], "Synthetic qualification and bootstrap only")
        require(cfg["graph"] == "synthetic-unused", "Only a synthetic injected fixture graph is allowed")


def catalog():
    inputs = np.zeros((2, 6, 7))
    for task in range(2):
        inputs[task, np.arange(6), np.arange(6)] = 1.
        inputs[task, :, 6] = (-1., 1.)[task]
    return {"probe_inputs": inputs,
            "probe_targets": np.array([[1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0]], dtype=np.int8),
            "stimulus_ids": np.arange(1, 7, dtype=np.int8),
            "conflicting_ids": np.array([True, True, False, False, False, False])}


def load_catalog(cfg, root):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "Duplicate order input JSON key")
            out[key] = value
        return out
    raw = json.loads((root / cfg["input_catalog"]).read_text(encoding="utf8"), object_pairs_hook=pairs,
                     parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite order input")))
    expected = catalog()
    require(set(raw) == {"study_id", "set_order", "catalog"} and raw["study_id"] == "interference_order_v1"
            and raw["set_order"] == list(SET_ORDER) and set(raw["catalog"]) == set(expected), "Exact contextual input identity required")
    result = {}
    for key, value in expected.items():
        actual = np.asarray(raw["catalog"][key])
        require(actual.shape == value.shape and actual.dtype.kind == value.dtype.kind
                and np.array_equal(actual, value), "Contextual input differs: " + key)
        result[key] = np.asarray(actual, dtype=value.dtype)
    return result


def checkpoint_metadata(cfg):
    n = cfg["stage_trials"]
    stream, local, total = [-1], [0], [0]
    for index in range(7):
        budget = cfg["joint_trials"] if index == 6 else n
        every = cfg["joint_checkpoint_every"] if index == 6 else cfg["checkpoint_every"]
        for trial in range(every, budget + 1, every):
            stream.append(index); local.append(trial)
            total.append(trial + n if index in (2, 3, 4, 5) else trial)
    m = n // cfg["checkpoint_every"]
    return {"checkpoint_stream": np.asarray(stream, dtype=np.int8),
            "checkpoint_local_trial": np.asarray(local, dtype=np.int32),
            "checkpoint_total_trial": np.asarray(total, dtype=np.int32),
            "stream_start_checkpoint": np.array([0, 0, m, m, 2 * m, 2 * m, 0], dtype=np.int32),
            "stream_end_checkpoint": np.array([m, 2*m, 3*m, 4*m, 5*m, 6*m, len(stream)-1], dtype=np.int32)}


def produce(cfg, root, cache=None):
    validate_config(cfg)
    data = load_catalog(cfg, root)
    seeds, n, maximum = cfg["model_seeds"], cfg["stage_trials"], cfg["joint_trials"]
    graph = load_projection(root / cfg["graph"])
    arrays = dict(data)
    arrays.update(checkpoint_metadata(cfg))
    k = len(arrays["checkpoint_stream"])
    arrays["model_seeds"] = np.asarray(seeds, dtype=np.int64)
    valid = np.zeros((7, maximum), dtype=bool)
    valid[:6, :n] = True
    valid[6, :] = True
    arrays["trial_valid"] = valid
    shape = (len(CONDITIONS), len(seeds), 7, maximum)
    arrays["trial_probabilities"] = np.full(shape, .5)
    arrays["trial_actions"] = np.full(shape, -1, dtype=np.int8)
    arrays["trial_rewards"] = np.zeros(shape, dtype=np.int8)
    for kind in ("head", "internal", "value"):
        arrays[f"trial_{kind}_steps"] = np.zeros(shape)
        arrays[f"trial_{kind}_update_calls"] = np.zeros(shape, dtype=np.int32)
        arrays[f"trial_{kind}_call_increments"] = np.zeros(shape, dtype=np.int8)
    arrays["schedules"] = np.empty((len(seeds), 2, 2, n), dtype=np.int8)
    arrays["uniforms"] = np.empty((len(seeds), 2, 2, n))
    arrays["stream_tasks"] = np.full((7, maximum), -1, dtype=np.int8)
    arrays["stream_repetitions"] = np.full((7, maximum), -1, dtype=np.int8)
    for stream_i, (task, repetition) in enumerate(((0,0), (1,0), (1,0), (0,1), (0,0), (1,1))):
        arrays["stream_tasks"][stream_i, :n] = task
        arrays["stream_repetitions"][stream_i, :n] = repetition
    arrays["stream_tasks"][6] = np.arange(maximum) % 2
    arrays["stream_repetitions"][6] = 0
    arrays["stream_stimulus_indices"] = np.full((len(seeds), 7, maximum), -1, dtype=np.int8)
    arrays["stream_uniforms"] = np.zeros((len(seeds), 7, maximum))
    arrays["checkpoint_probabilities"] = np.empty((len(CONDITIONS), len(seeds), k, 2, 6))
    arrays["checkpoint_logits"] = np.empty_like(arrays["checkpoint_probabilities"])
    for kind in ("head", "internal", "value"):
        arrays[f"checkpoint_{kind}_calls"] = np.empty((len(CONDITIONS), len(seeds), k), dtype=np.int32)
        arrays[f"stream_start_{kind}_calls"] = np.empty((len(CONDITIONS), len(seeds), 7), dtype=np.int32)
    rows, snapshots_by_name = [], {}
    probes = data["probe_inputs"].reshape(12, 7)
    reference = probes.copy()
    for si, seed in enumerate(seeds):
        for task in range(2):
            for repetition in range(2):
                order = np.random.default_rng(np.random.SeedSequence([cfg["rng_namespace"], seed, task, repetition, 0]))
                action = np.random.default_rng(np.random.SeedSequence([cfg["rng_namespace"], seed, task, repetition, 1]))
                arrays["schedules"][si, task, repetition] = np.concatenate([order.permutation(6) for _ in range(n // 6)])
                arrays["uniforms"][si, task, repetition] = action.random(n)
        for stream_i in range(7):
            local_by_task = [0, 0]
            for ti in np.flatnonzero(valid[stream_i]):
                task, repetition = int(arrays["stream_tasks"][stream_i, ti]), int(arrays["stream_repetitions"][stream_i, ti])
                position = local_by_task[task]; local_by_task[task] += 1
                arrays["stream_stimulus_indices"][si, stream_i, ti] = arrays["schedules"][si, task, repetition, position]
                arrays["stream_uniforms"][si, stream_i, ti] = arrays["uniforms"][si, task, repetition, position]
        graphs = build_graphs(graph, seed)
        for ci, condition in enumerate(CONDITIONS):
            actor = make_actor("lookup_q" if condition == "context_q" else condition,
                               graphs, seed, cfg["head_lr"], reference, probes)
            states = [None] * k

            def checkpoint(model, cp):
                arrays["checkpoint_probabilities"][ci, si, cp] = np.array([model.probability(x) for x in probes]).reshape(2, 6)
                arrays["checkpoint_logits"][ci, si, cp] = np.array([model.raw_gap(x) for x in probes]).reshape(2, 6)
                for kind in ("head", "internal", "value"):
                    arrays[f"checkpoint_{kind}_calls"][ci, si, cp] = getattr(model, f"{kind}_updates")
                states[cp] = model.witness()

            checkpoint(actor, 0)

            def train_stream(model, stream_i):
                for kind in ("head", "internal", "value"):
                    arrays[f"stream_start_{kind}_calls"][ci, si, stream_i] = getattr(model, f"{kind}_updates")
                checkpoints = {int(arrays["checkpoint_local_trial"][cp]): cp
                               for cp in np.flatnonzero(arrays["checkpoint_stream"] == stream_i)}
                for ti in np.flatnonzero(valid[stream_i]):
                    task = int(arrays["stream_tasks"][stream_i, ti])
                    stimulus = int(arrays["stream_stimulus_indices"][si, stream_i, ti])
                    vector = data["probe_inputs"][task, stimulus]
                    p = model.probability(vector)
                    action = int(arrays["stream_uniforms"][si, stream_i, ti] < p)
                    target = int(data["probe_targets"][task, stimulus])
                    reward = (1 if target else -1) if action else 0
                    before = [getattr(model, f"{kind}_updates") for kind in ("head", "internal", "value")]
                    steps = model.observe(vector, action, reward)
                    at = ci, si, stream_i, ti
                    arrays["trial_probabilities"][at], arrays["trial_actions"][at], arrays["trial_rewards"][at] = p, action, reward
                    row = {"condition": condition, "seed": seed, "stream": STREAMS[stream_i], "local_trial": int(ti)+1,
                           "total_history_trial": int(ti)+1+(n if stream_i in (2,3,4,5) else 0),
                           "task": SET_ORDER[task], "stimulus_index": stimulus, "probability_go": p,
                           "action": action, "reward": reward}
                    for ki, kind in enumerate(("head", "internal", "value")):
                        calls = getattr(model, f"{kind}_updates")
                        increment = calls - before[ki]
                        arrays[f"trial_{kind}_steps"][at] = steps[ki]
                        arrays[f"trial_{kind}_update_calls"][at] = calls
                        arrays[f"trial_{kind}_call_increments"][at] = increment
                        row[f"{kind}_update_calls"] = calls
                        row[f"{kind}_call_increment"] = increment
                        row[f"{kind}_step_l2"] = steps[ki]
                        row[f"{kind}_nonzero_step"] = int(steps[ki] > 0)
                    rows.append(row)
                    if int(ti)+1 in checkpoints:
                        checkpoint(model, checkpoints[int(ti)+1])
                return model

            prefix_a = train_stream(copy.deepcopy(actor), 0)
            prefix_b = train_stream(copy.deepcopy(actor), 1)
            for stream_i, source in ((2,prefix_a), (3,prefix_a), (4,prefix_b), (5,prefix_b), (6,actor)):
                train_stream(copy.deepcopy(source), stream_i)
            require(all(state is not None for state in states), "All original checkpoints required")
            for field in states[0]:
                snapshots_by_name.setdefault(f"w_{condition}_{field}", []).append(np.stack([state[field] for state in states]))
    arrays.update({key: np.stack(values) for key, values in snapshots_by_name.items()})
    prefix_ends = arrays["stream_end_checkpoint"][:2]
    arrays["idle_probabilities"] = arrays["checkpoint_probabilities"][:, :, prefix_ends].copy()
    arrays["idle_logits"] = arrays["checkpoint_logits"][:, :, prefix_ends].copy()
    for condition in CONDITIONS:
        for field in ("head", "weights", "encoder", "bias", "pre", "post", "initial_weights", "upper",
                      "feature_scale", "probe_kc", "probe_mbon", "probe_decision", "value_table", "value_catalog"):
            arrays[f"idle_w_{condition}_{field}"] = arrays[f"w_{condition}_{field}"][:, prefix_ends].copy()
    return arrays, rows


def summarize(cfg, arrays, rows):
    validate_config(cfg)
    p, target = arrays["checkpoint_probabilities"], arrays["probe_targets"]
    k = len(arrays["checkpoint_stream"])
    require(p.shape == (len(CONDITIONS), len(cfg["model_seeds"]), k, 2, 6)
            and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "Aligned finite contextual probabilities required")
    correct = np.where(target[None,None,None] == 1, p, 1-p)
    scores = correct.mean(axis=-1)
    end = arrays["stream_end_checkpoint"]
    starts = arrays["stream_start_checkpoint"]
    prefix_pass = np.empty((len(CONDITIONS), len(cfg["model_seeds"]), 2), dtype=bool)
    joint_pass = np.empty_like(prefix_pass)
    for task in range(2):
        pos, neg = target[task] == 1, target[task] == 0
        before = scores[:,:,0,task]
        prefix = correct[:,:,end[task],task]
        joint = correct[:,:,end[6],task]
        prefix_pass[:,:,task] = ((prefix.mean(axis=-1) >= cfg["acquisition_threshold"])
                                & (prefix.mean(axis=-1)-before >= cfg["acquisition_gain"])
                                & (prefix[...,pos].mean(axis=-1) >= cfg["class_threshold"])
                                & (prefix[...,neg].mean(axis=-1) >= cfg["class_threshold"])
                                & (prefix[...,arrays["conflicting_ids"]].mean(axis=-1) >= cfg["conflict_threshold"]))
        joint_pass[:,:,task] = ((joint.mean(axis=-1) >= cfg["joint_threshold"])
                               & (joint[...,pos].mean(axis=-1) >= cfg["class_threshold"])
                               & (joint[...,neg].mean(axis=-1) >= cfg["class_threshold"])
                               & (joint[...,arrays["conflicting_ids"]].mean(axis=-1) >= cfg["conflict_threshold"]))
    qualification = prefix_pass.all(axis=-1) & joint_pass.all(axis=-1)
    loss = np.stack((scores[:,:,end[0],0]-scores[:,:,end[2],0], scores[:,:,end[1],1]-scores[:,:,end[4],1]), axis=-1)
    practice = np.stack((scores[:,:,end[3],0]-scores[:,:,end[0],0], scores[:,:,end[5],1]-scores[:,:,end[1],1]), axis=-1)
    cost = np.stack((scores[:,:,end[3],0]-scores[:,:,end[2],0], scores[:,:,end[5],1]-scores[:,:,end[4],1]), axis=-1)
    residual = cost-loss-practice
    require(np.max(np.abs(residual)) <= 1e-12, "Interference and continued-practice decomposition failed")
    competence_ab, competence_ba = scores[:,:,end[2]].mean(axis=-1), scores[:,:,end[4]].mean(axis=-1)
    order = competence_ab-competence_ba
    repeats = cfg["bootstrap_repeats"]
    rng = np.random.default_rng(np.random.SeedSequence(cfg["bootstrap_entropy"]))
    indices = rng.integers(0, len(cfg["model_seeds"]), (repeats, len(cfg["model_seeds"])))
    families = []
    for ci, condition in enumerate(CONDITIONS):
        qualification_count = int(qualification[ci].sum())
        eligible = qualification_count >= cfg["population_pass_count"]
        mean_loss = loss[ci].mean(axis=-1)
        loss_interval = [float(x) for x in np.quantile(mean_loss[indices].mean(axis=-1), cfg["interval_quantiles"], method="linear")]
        order_interval = [float(x) for x in np.quantile(order[ci][indices].mean(axis=-1), cfg["interval_quantiles"], method="linear")]
        hi = bool(eligible and mean_loss.mean() >= cfg["interference_margin"] and loss_interval[0] > 0)
        ho = bool(eligible and abs(order[ci].mean()) >= cfg["order_margin"] and (order_interval[0] > 0 or order_interval[1] < 0))
        seed_reports = []
        for si, seed in enumerate(cfg["model_seeds"]):
            tasks = []
            for task, branch, repeat_branch in ((0,2,3), (1,4,5)):
                own = correct[ci,si,end[task],task]
                afterward = correct[ci,si,end[branch],task]
                other_task = 1-task
                branch_cps = np.flatnonzero(arrays["checkpoint_stream"] == branch)
                curve_cp = np.r_[starts[branch], branch_cps]
                times = np.r_[0, arrays["checkpoint_local_trial"][branch_cps]]
                new_curve = scores[ci,si,curve_cp,other_task]
                tasks.append({"task": SET_ORDER[task], "prefix_score": float(own.mean()),
                              "prefix_acquisition_pass": bool(prefix_pass[ci,si,task]),
                              "prefix_conflicting_score": float(own[arrays["conflicting_ids"]].mean()),
                              "joint_score": float(scores[ci,si,end[6],task]), "joint_pass": bool(joint_pass[ci,si,task]),
                              "joint_conflicting_score": float(correct[ci,si,end[6],task,arrays["conflicting_ids"]].mean()),
                              "after_other_score": float(afterward.mean()),
                              "after_repetition_score": float(scores[ci,si,end[repeat_branch],task]),
                              "retention_loss": float(loss[ci,si,task]), "continued_practice_gain": float(practice[ci,si,task]),
                              "matched_budget_terminal_cost": float(cost[ci,si,task]),
                              "conflicting_id_retention_loss": float((own-afterward)[arrays["conflicting_ids"]].mean()),
                              "consistent_id_retention_loss": float((own-afterward)[~arrays["conflicting_ids"]].mean()),
                              "new_task": SET_ORDER[other_task], "new_task_checkpoint_trials": times.tolist(),
                              "new_task_curve": new_curve.tolist(),
                              "new_task_checkpoint_auc": float(np.trapezoid(new_curve, times)/cfg["stage_trials"]),
                              "new_task_prior_history_delta": float(scores[ci,si,end[branch],other_task]-scores[ci,si,end[other_task],other_task])})
            seed_reports.append({"seed": seed, "qualified": bool(qualification[ci,si]), "tasks": tasks,
                                 "mean_retention_loss": float(mean_loss[si]), "competence_AB": float(competence_ab[ci,si]),
                                 "competence_BA": float(competence_ba[ci,si]), "order_delta": float(order[ci,si])})
        def status(supported):
            return "not_assessable" if not eligible else "supported" if supported else "not_supported"
        families.append({"condition": condition, "role": "primary" if condition == cfg["primary_condition"] else "descriptive_control",
                         "qualification_pass_seeds": qualification_count, "qualification_supported": eligible,
                         "resources": {"head_parameters": int(np.prod(arrays[f"w_{condition}_head"].shape[2:])),
                                       "internal_edge_parameters": int(np.prod(arrays[f"w_{condition}_weights"].shape[2:])),
                                       "value_parameters": int(np.prod(arrays[f"w_{condition}_value_table"].shape[2:])),
                                       "fixed_encoder_parameters": int(np.prod(arrays[f"w_{condition}_encoder"].shape[2:])),
                                       "internal_parameters_trainable": condition.endswith("plastic"),
                                       "value_parameters_are_not_neural_weights": True},
                         "H_interference": {"supported": hi, "status": status(hi),
                                            "mean_retention_loss": float(mean_loss.mean()), "ci90": loss_interval},
                         "H_order": {"supported": ho, "status": status(ho), "mean_order_delta": float(order[ci].mean()), "ci90": order_interval},
                         "seeds": seed_reports})
    expected = len(CONDITIONS)*len(cfg["model_seeds"])*(6*cfg["stage_trials"]+cfg["joint_trials"])
    require(len(rows) == expected, "All original prefix/branch/joint feedback rows required")
    primary = families[CONDITIONS.index(cfg["primary_condition"])]
    return {"study_id": cfg["study_id"], "status": "completed", "families": families,
            "H_interference": copy.deepcopy(primary["H_interference"]), "H_order": copy.deepcopy(primary["H_order"]),
            "qualification_supported": primary["qualification_supported"], "max_decomposition_residual": float(np.max(np.abs(residual))),
            "analytic_context_blind_witness": {"consistent_id_accuracy": 1., "conflicting_id_accuracy": .5,
                                                "overall_accuracy": 5/6, "each_class_accuracy": 5/6,
                                                "passes_old_mean_and_class_gate": True,
                                                "passes_conflict_qualification": False},
            "counts": {"feedback_trials": expected, "static_probe_cases": int(np.prod(p.shape)),
                       "shared_prefix_feedback_trials": 2*cfg["stage_trials"]*len(CONDITIONS)*len(cfg["model_seeds"]),
                       "active_suffix_feedback_trials": 4*cfg["stage_trials"]*len(CONDITIONS)*len(cfg["model_seeds"]),
                       "joint_feedback_trials": cfg["joint_trials"]*len(CONDITIONS)*len(cfg["model_seeds"]),
                       "logical_active_history_trials": (8*cfg["stage_trials"]+cfg["joint_trials"])*len(CONDITIONS)*len(cfg["model_seeds"]),
                       "prefix_reuse_saved_feedback_trials": 2*cfg["stage_trials"]*len(CONDITIONS)*len(cfg["model_seeds"]),
                       "idle_feedback_trials": 0, "idle_new_probability_evaluations": 0,
                       "model_initializations": len(CONDITIONS)*len(cfg["model_seeds"]),
                       "trained_trajectory_clones": 7*len(CONDITIONS)*len(cfg["model_seeds"]),
                       **{f"{kind}_update_calls": int(arrays[f"trial_{kind}_call_increments"].sum()) for kind in ("head","internal","value")},
                       **{f"{kind}_nonzero_steps": int(np.count_nonzero(arrays[f"trial_{kind}_steps"])) for kind in ("head","internal","value")}},
            "bootstrap": {"repeats": repeats, "entropy": cfg["bootstrap_entropy"], "quantiles": cfg["interval_quantiles"],
                          "unit": "paired_computational_seed", "all_seeds_in_effect_estimates": True},
            "interpretation": {"new_human_experiment": False, "shared_neural_mechanism_claim": False,
                               "education_effect_claim": False, "natural_time_forgetting_tested": False,
                               "joint_solution_exists_with_supplied_context": True, "idle_is_state_copy_invariant": True,
                               "discarded_learning_failures": 0, "all_checkpoint_states_recorded": True}}
