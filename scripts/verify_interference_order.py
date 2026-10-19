"""Independent arithmetic for the bounded contextual interference/order study.

Only frozen independent pattern/graph arithmetic is reused. Scientific producer
code, its curriculum generator and its scientific summaries are not imported.
"""
from __future__ import annotations
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

_SPEC = importlib.util.spec_from_file_location("frozen_order_independent_pattern", Path(__file__).with_name("verify_pattern_rule.py"))
pattern = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pattern)
core = pattern.core
CONDITIONS = (*pattern.CONDITIONS[:8], "context_q")
STREAMS = ("prefix_A", "prefix_B", "AB", "AA", "BA", "BB", "joint")
TASKS = ("A", "B")
INPUT_CATALOG = "data/synthetic/interference_order_v1/input_catalog.json"
NAMESPACE = 202610192


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "interference_order_v1", "conditions": list(CONDITIONS),
             "head_lr": .1, "internal_lr_multiplier": .1, "context_q_alpha": .1, "context_q_beta": 4.,
             "rng_namespace": NAMESPACE, "interval_quantiles": [.05, .95], "primary_condition": "plastic",
             "acquisition_threshold": .8, "acquisition_gain": .25, "class_threshold": .7,
             "joint_threshold": .8, "conflict_threshold": .75, "interference_margin": .1, "order_margin": .05,
             "input_catalog": INPUT_CATALOG, "normalization": "initial_twelve_context_input_feature_gap_RMS",
             "context_values": [-1, 1], "streams": list(STREAMS), "joint_order": "deterministic_A_B_alternation",
             "idle": "literal_prefix_state_copy_without_time_or_update"}
    for key, value in fixed.items(): core.demand(cfg.get(key) == value, "Fixed independent contextual design: " + key)
    if allow_fixture and cfg.get("scope") == "integration_test":
        seeds, n = cfg["model_seeds"], cfg["stage_trials"]
        core.demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(seeds) == len(set(seeds))
                    and all(type(s) is int and 91001 <= s <= 91100 for s in seeds), "Invented independent fixture seeds only")
        core.demand(type(n) is int and n in (6, 12) and cfg["joint_trials"] == 2*n and cfg["checkpoint_every"] == 6
                    and cfg["joint_checkpoint_every"] == 12, "Small independent balanced fixture budget")
        core.demand(cfg["population_pass_count"] == len(seeds) and cfg["bootstrap_repeats"] == 16
                    and cfg["bootstrap_entropy"] == [NAMESPACE, 99001, 2], "Fixture inference scope")
    else:
        core.demand(cfg.get("scope") == "prospectively_registered_interference_order"
                    and cfg["model_seeds"] == list(range(15201, 15211))
                    and (cfg["stage_trials"], cfg["joint_trials"], cfg["checkpoint_every"], cfg["joint_checkpoint_every"]) == (360, 720, 60, 120)
                    and cfg["population_pass_count"] == 8 and cfg["bootstrap_repeats"] == 2000
                    and cfg["bootstrap_entropy"] == [NAMESPACE, 15301, 2]
                    and cfg["graph"] == "data/flywire_783_mb_left_kc512_s701"
                    and cfg["input_files"] == [INPUT_CATALOG], "Prospective contextual budget/source/qualification")


def catalog():
    vectors = np.empty((2, 6, 7)); vectors[:, :, :6] = np.eye(6)[None]
    vectors[0, :, 6], vectors[1, :, 6] = -1., 1.
    target = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0]], dtype=int)
    return {"probe_inputs": vectors, "probe_targets": target, "stimulus_ids": np.arange(1, 7),
            "conflicting_ids": target[0] != target[1]}


def verify_input_catalog(cfg, root):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            core.demand(key not in result, "Duplicate contextual input JSON key"); result[key] = value
        return result
    def bad_constant(value): raise ValueError("Nonfinite contextual JSON")
    path = core.contained(Path(root), INPUT_CATALOG)
    raw = json.loads(path.read_text(encoding="utf8"), object_pairs_hook=unique, parse_constant=bad_constant)
    core.demand(set(raw) == {"study_id", "set_order", "catalog"} and raw["study_id"] == "interference_order_v1"
                and raw["set_order"] == list(TASKS) and set(raw["catalog"]) == set(catalog()), "Exact contextual catalog schema")
    for key, wanted in catalog().items():
        actual = np.asarray(raw["catalog"][key])
        core.demand(actual.dtype.kind == wanted.dtype.kind and actual.shape == wanted.shape and np.array_equal(actual, wanted),
                    "Literal contextual catalog differs: " + key)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_metadata(cfg):
    n = cfg["stage_trials"]; cp_stream, local, total = [-1], [0], [0]
    for stream in range(7):
        length, every = (cfg["joint_trials"], cfg["joint_checkpoint_every"]) if stream == 6 else (n, cfg["checkpoint_every"])
        for time in range(every, length+1, every):
            cp_stream.append(stream); local.append(time); total.append(time + (n if 2 <= stream <= 5 else 0))
    ends = [max(i for i, s in enumerate(cp_stream) if s == stream) for stream in range(7)]
    return {"checkpoint_stream": np.array(cp_stream), "checkpoint_local_trial": np.array(local),
            "checkpoint_total_trial": np.array(total), "stream_end_checkpoint": np.array(ends),
            "stream_start_checkpoint": np.array([0, 0, ends[0], ends[0], ends[1], ends[1], 0])}


def address(seed, task, repetition, slot):
    core.demand(type(seed) is int and seed >= 0 and type(task) is int and task in (0, 1)
                and type(repetition) is int and repetition in (0, 1) and type(slot) is int and slot in (0, 1), "Canonical task/repetition stream address")
    return np.random.default_rng(np.random.SeedSequence([NAMESPACE, seed, task, repetition, slot]))


def curriculum(cfg, seed):
    n, maximum = cfg["stage_trials"], cfg["joint_trials"]
    schedules = np.empty((2, 2, n), int); uniforms = np.empty((2, 2, n))
    for task in range(2):
        for repetition in range(2):
            rng = address(seed, task, repetition, 0)
            schedules[task, repetition] = np.concatenate([rng.permutation(6) for _ in range(n//6)])
            uniforms[task, repetition] = address(seed, task, repetition, 1).random(n)
    valid = np.zeros((7, maximum), bool); valid[:6, :n] = True; valid[6] = True
    tasks, repetitions = np.full((7, maximum), -1, int), np.full((7, maximum), -1, int)
    orders = np.full((7, maximum), -1, int); choices = np.zeros((7, maximum))
    for stream, (task, rep) in enumerate(((0,0),(1,0),(1,0),(0,1),(0,0),(1,1))):
        tasks[stream, :n], repetitions[stream, :n] = task, rep
        orders[stream, :n], choices[stream, :n] = schedules[task, rep], uniforms[task, rep]
    tasks[6] = np.arange(maximum) % 2; repetitions[6] = 0
    orders[6, ::2], orders[6, 1::2] = schedules[0, 0], schedules[1, 0]
    choices[6, ::2], choices[6, 1::2] = uniforms[0, 0], uniforms[1, 0]
    return {"schedules": schedules, "uniforms": uniforms, "trial_valid": valid, "stream_tasks": tasks,
            "stream_repetitions": repetitions, "stream_stimulus_indices": orders, "stream_uniforms": choices}


def make_independent_actor(condition, seed, graph, probes, graph_cache):
    inner = "lookup_q" if condition == "context_q" else condition
    actor = pattern.AuditActor(inner, seed, core.graph_for(inner, seed, graph, graph_cache), probes, .1)
    if condition == "context_q": actor.values = np.zeros((12, 2))
    return actor


def expected_artifacts(cfg, graph):
    data = catalog(); probes = data["probe_inputs"].reshape(12, 7)
    seeds, maximum = cfg["model_seeds"], cfg["joint_trials"]
    a = {**data, **checkpoint_metadata(cfg), "model_seeds": np.asarray(seeds)}
    plans = [curriculum(cfg, seed) for seed in seeds]
    for field in ("trial_valid", "stream_tasks", "stream_repetitions"): a[field] = plans[0][field]
    for field in ("schedules", "uniforms", "stream_stimulus_indices", "stream_uniforms"): a[field] = np.stack([p[field] for p in plans])
    shape = (len(CONDITIONS), len(seeds), 7, maximum); k = len(a["checkpoint_stream"])
    a["trial_probabilities"], a["trial_actions"], a["trial_rewards"] = np.full(shape, .5), np.full(shape, -1, int), np.zeros(shape, int)
    for kind in ("head", "internal", "value"):
        a[f"trial_{kind}_steps"] = np.zeros(shape)
        a[f"trial_{kind}_update_calls"] = np.zeros(shape, int)
        a[f"trial_{kind}_call_increments"] = np.zeros(shape, int)
        a[f"checkpoint_{kind}_calls"] = np.empty((len(CONDITIONS), len(seeds), k), int)
        a[f"stream_start_{kind}_calls"] = np.empty((len(CONDITIONS), len(seeds), 7), int)
    a["checkpoint_probabilities"] = np.empty((len(CONDITIONS), len(seeds), k, 2, 6))
    a["checkpoint_logits"] = np.empty_like(a["checkpoint_probabilities"])
    witnesses, ledger, graph_cache = {}, [], {}
    for si, seed in enumerate(seeds):
        for ci, condition in enumerate(CONDITIONS):
            fresh = make_independent_actor(condition, seed, graph, probes, graph_cache)
            snapshots = [None] * k
            def save(actor, cp):
                a["checkpoint_probabilities"][ci, si, cp] = actor.probability(probes).reshape(2, 6)
                a["checkpoint_logits"][ci, si, cp] = actor.raw_gap(probes).reshape(2, 6)
                snapshots[cp] = actor.witness(probes)
                for kind in ("head", "internal", "value"): a[f"checkpoint_{kind}_calls"][ci, si, cp] = getattr(actor, kind + "_updates")
            save(fresh, 0)
            def run(actor, stream):
                for kind in ("head", "internal", "value"): a[f"stream_start_{kind}_calls"][ci, si, stream] = getattr(actor, kind + "_updates")
                lookup = {int(a["checkpoint_local_trial"][cp]): int(cp) for cp in np.flatnonzero(a["checkpoint_stream"] == stream)}
                for ti in np.flatnonzero(a["trial_valid"][stream]):
                    task, index = int(a["stream_tasks"][stream, ti]), int(a["stream_stimulus_indices"][si, stream, ti])
                    vector = data["probe_inputs"][task, index]; p = float(actor.probability(vector))
                    action = int(a["stream_uniforms"][si, stream, ti] < p)
                    reward = (1 if data["probe_targets"][task, index] else -1) if action else 0
                    prior = [getattr(actor, kind + "_updates") for kind in ("head", "internal", "value")]
                    steps = actor.update(vector, action, int(reward)); at = (ci, si, stream, ti)
                    a["trial_probabilities"][at], a["trial_actions"][at], a["trial_rewards"][at] = p, action, reward
                    row = {"condition": condition, "seed": seed, "stream": STREAMS[stream], "local_trial": int(ti)+1,
                           "total_history_trial": int(ti)+1+(cfg["stage_trials"] if 2 <= stream <= 5 else 0),
                           "task": TASKS[task], "stimulus_index": index, "probability_go": p, "action": action, "reward": reward}
                    for ki, kind in enumerate(("head", "internal", "value")):
                        calls = getattr(actor, kind + "_updates"); increment = calls-prior[ki]
                        a[f"trial_{kind}_steps"][at] = steps[ki]; a[f"trial_{kind}_update_calls"][at] = calls
                        a[f"trial_{kind}_call_increments"][at] = increment
                        row.update({f"{kind}_update_calls": calls, f"{kind}_call_increment": increment,
                                    f"{kind}_step_l2": steps[ki], f"{kind}_nonzero_step": int(steps[ki] > 0)})
                    ledger.append(row)
                    if int(ti)+1 in lookup: save(actor, lookup[int(ti)+1])
                return actor
            prefix_a, prefix_b = run(copy.deepcopy(fresh), 0), run(copy.deepcopy(fresh), 1)
            for stream, origin in ((2,prefix_a),(3,prefix_a),(4,prefix_b),(5,prefix_b),(6,fresh)): run(copy.deepcopy(origin), stream)
            core.demand(all(s is not None for s in snapshots), "Complete contextual checkpoint ledger")
            for field in snapshots[0]: witnesses.setdefault(f"w_{condition}_{field}", []).append(np.stack([s[field] for s in snapshots]))
    a.update({key: np.stack(value) for key, value in witnesses.items()})
    prefix = a["stream_end_checkpoint"][:2]
    a["idle_probabilities"], a["idle_logits"] = a["checkpoint_probabilities"][:, :, prefix].copy(), a["checkpoint_logits"][:, :, prefix].copy()
    for key, value in list(a.items()):
        if key.startswith("w_"): a["idle_" + key] = value[:, prefix].copy()
    return a, ledger


def expected_summary(cfg, a):
    correct = np.where(a["probe_targets"][None,None,None], a["checkpoint_probabilities"], 1-a["checkpoint_probabilities"])
    score = correct.mean(axis=-1); end, start = a["stream_end_checkpoint"], a["stream_start_checkpoint"]
    conflict = a["conflicting_ids"]; s = len(cfg["model_seeds"])
    prefix_pass = np.zeros((len(CONDITIONS), s, 2), bool); joint_pass = np.zeros_like(prefix_pass)
    for ci in range(len(CONDITIONS)):
        for si in range(s):
            for task in range(2):
                target = a["probe_targets"][task]; own = correct[ci,si,end[task],task]; joint = correct[ci,si,end[6],task]
                def classes(values): return all(values[target == category].mean() >= cfg["class_threshold"] for category in (0,1))
                prefix_pass[ci,si,task] = (own.mean() >= cfg["acquisition_threshold"] and own.mean()-score[ci,si,0,task] >= cfg["acquisition_gain"]
                                         and classes(own) and own[conflict].mean() >= cfg["conflict_threshold"])
                joint_pass[ci,si,task] = (joint.mean() >= cfg["joint_threshold"] and classes(joint) and joint[conflict].mean() >= cfg["conflict_threshold"])
    qualified = np.all(prefix_pass & joint_pass, axis=-1)
    loss = np.stack([score[:,:,end[t],t]-score[:,:,end[b],t] for t,b in ((0,2),(1,4))], axis=-1)
    practice = np.stack([score[:,:,end[b],t]-score[:,:,end[t],t] for t,b in ((0,3),(1,5))], axis=-1)
    cost = np.stack([score[:,:,end[r],t]-score[:,:,end[b],t] for t,r,b in ((0,3,2),(1,5,4))], axis=-1)
    residual = cost-loss-practice
    core.demand(np.abs(residual).max() <= 1e-12, "Loss/practice/terminal-cost identity")
    ab, ba = score[:,:,end[2]].mean(axis=-1), score[:,:,end[4]].mean(axis=-1); order = ab-ba
    rng = np.random.default_rng(np.random.SeedSequence(cfg["bootstrap_entropy"]))
    samples = rng.integers(0, s, (cfg["bootstrap_repeats"], s))
    families = []
    for ci, condition in enumerate(CONDITIONS):
        count = int(qualified[ci].sum()); eligible = count >= cfg["population_pass_count"]
        average_loss = loss[ci].mean(axis=-1)
        loss_ci = np.quantile(average_loss[samples].mean(axis=-1), cfg["interval_quantiles"], method="linear").tolist()
        order_ci = np.quantile(order[ci][samples].mean(axis=-1), cfg["interval_quantiles"], method="linear").tolist()
        hi = bool(eligible and average_loss.mean() >= cfg["interference_margin"] and loss_ci[0] > 0)
        ho = bool(eligible and abs(order[ci].mean()) >= cfg["order_margin"] and (order_ci[0] > 0 or order_ci[1] < 0))
        reports = []
        for si, seed in enumerate(cfg["model_seeds"]):
            tasks = []
            for task, branch, repeated in ((0,2,3),(1,4,5)):
                own, after = correct[ci,si,end[task],task], correct[ci,si,end[branch],task]
                cp = np.flatnonzero(a["checkpoint_stream"] == branch)
                checkpoints = np.r_[start[branch], cp]; time = np.r_[0, a["checkpoint_local_trial"][cp]]
                curve = score[ci,si,checkpoints,1-task]
                # Explicit trapezoid sum, separate from the producer's np.trapezoid.
                auc = float(np.sum(np.diff(time)*(curve[:-1]+curve[1:])*.5)/cfg["stage_trials"])
                tasks.append({"task": TASKS[task], "prefix_score": float(own.mean()), "prefix_acquisition_pass": bool(prefix_pass[ci,si,task]),
                              "prefix_conflicting_score": float(own[conflict].mean()), "joint_score": float(score[ci,si,end[6],task]),
                              "joint_pass": bool(joint_pass[ci,si,task]), "joint_conflicting_score": float(correct[ci,si,end[6],task,conflict].mean()),
                              "after_other_score": float(after.mean()), "after_repetition_score": float(score[ci,si,end[repeated],task]),
                              "retention_loss": float(loss[ci,si,task]), "continued_practice_gain": float(practice[ci,si,task]),
                              "matched_budget_terminal_cost": float(cost[ci,si,task]),
                              "conflicting_id_retention_loss": float((own-after)[conflict].mean()),
                              "consistent_id_retention_loss": float((own-after)[~conflict].mean()), "new_task": TASKS[1-task],
                              "new_task_checkpoint_trials": time.tolist(), "new_task_curve": curve.tolist(),
                              "new_task_checkpoint_auc": auc,
                              "new_task_prior_history_delta": float(score[ci,si,end[branch],1-task]-score[ci,si,end[1-task],1-task])})
            reports.append({"seed": seed, "qualified": bool(qualified[ci,si]), "tasks": tasks,
                            "mean_retention_loss": float(average_loss[si]), "competence_AB": float(ab[ci,si]),
                            "competence_BA": float(ba[ci,si]), "order_delta": float(order[ci,si])})
        def status(passed): return "not_assessable" if not eligible else "supported" if passed else "not_supported"
        families.append({"condition": condition, "role": "primary" if condition == cfg["primary_condition"] else "descriptive_control",
                         "qualification_pass_seeds": count, "qualification_supported": eligible,
                         "resources": {"head_parameters": int(np.prod(a[f"w_{condition}_head"].shape[2:])),
                                       "internal_edge_parameters": int(np.prod(a[f"w_{condition}_weights"].shape[2:])),
                                       "value_parameters": int(np.prod(a[f"w_{condition}_value_table"].shape[2:])),
                                       "fixed_encoder_parameters": int(np.prod(a[f"w_{condition}_encoder"].shape[2:])),
                                       "internal_parameters_trainable": condition.endswith("plastic"), "value_parameters_are_not_neural_weights": True},
                         "H_interference": {"supported": hi, "status": status(hi), "mean_retention_loss": float(average_loss.mean()), "ci90": loss_ci},
                         "H_order": {"supported": ho, "status": status(ho), "mean_order_delta": float(order[ci].mean()), "ci90": order_ci}, "seeds": reports})
    primary = families[CONDITIONS.index(cfg["primary_condition"])]
    feedback = len(CONDITIONS)*s*(6*cfg["stage_trials"]+cfg["joint_trials"])
    return {"study_id": cfg["study_id"], "status": "completed", "families": families,
            "H_interference": copy.deepcopy(primary["H_interference"]), "H_order": copy.deepcopy(primary["H_order"]),
            "qualification_supported": primary["qualification_supported"], "max_decomposition_residual": float(np.abs(residual).max()),
            "analytic_context_blind_witness": {"consistent_id_accuracy": 1., "conflicting_id_accuracy": .5,
                                               "overall_accuracy": 5/6, "each_class_accuracy": 5/6,
                                               "passes_old_mean_and_class_gate": True, "passes_conflict_qualification": False},
            "counts": {"feedback_trials": feedback, "static_probe_cases": int(np.prod(a["checkpoint_probabilities"].shape)),
                       "shared_prefix_feedback_trials": 2*cfg["stage_trials"]*len(CONDITIONS)*s,
                       "active_suffix_feedback_trials": 4*cfg["stage_trials"]*len(CONDITIONS)*s,
                       "joint_feedback_trials": cfg["joint_trials"]*len(CONDITIONS)*s,
                       "logical_active_history_trials": (8*cfg["stage_trials"]+cfg["joint_trials"])*len(CONDITIONS)*s,
                       "prefix_reuse_saved_feedback_trials": 2*cfg["stage_trials"]*len(CONDITIONS)*s,
                       "idle_feedback_trials": 0, "idle_new_probability_evaluations": 0,
                       "model_initializations": len(CONDITIONS)*s, "trained_trajectory_clones": 7*len(CONDITIONS)*s,
                       **{f"{kind}_update_calls": int(a[f"trial_{kind}_call_increments"].sum()) for kind in ("head","internal","value")},
                       **{f"{kind}_nonzero_steps": int(np.count_nonzero(a[f"trial_{kind}_steps"])) for kind in ("head","internal","value")}},
            "bootstrap": {"repeats": cfg["bootstrap_repeats"], "entropy": cfg["bootstrap_entropy"], "quantiles": cfg["interval_quantiles"],
                          "unit": "paired_computational_seed", "all_seeds_in_effect_estimates": True},
            "interpretation": {"new_human_experiment": False, "shared_neural_mechanism_claim": False, "education_effect_claim": False,
                               "natural_time_forgetting_tested": False, "joint_solution_exists_with_supplied_context": True,
                               "idle_is_state_copy_invariant": True, "discarded_learning_failures": 0, "all_checkpoint_states_recorded": True}}


def verify_against_graph(cfg, graph, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    expected, ledger = expected_artifacts(cfg, graph)
    core.demand(set(arrays) == set(expected), "Exact contextual numeric inventory")
    for field, value in expected.items():
        observed = np.asarray(arrays[field]); value = np.asarray(value)
        core.demand(observed.dtype.kind in "biuf" and np.isfinite(observed).all(), "Finite contextual numeric array: " + field)
        if value.dtype.kind in "biu":
            core.demand(observed.dtype.kind in "biu" and np.array_equal(observed, value), "Exact contextual integer ledger: " + field)
        else: core.array_close(observed, value, "Independent contextual replay: " + field, atol=1e-9)
    core.demand(len(rows) == len(ledger), "Complete unique prefix/suffix/joint feedback ledger")
    for i, (actual, wanted) in enumerate(zip(rows, ledger)):
        if all(type(v) is str for v in actual.values()): core.check_csv_row(actual, wanted, f"Contextual trial {i}")
        else: core.compare_tree(actual, wanted, f"Contextual trial {i}")
    core.compare_tree(summary, expected_summary(cfg, arrays), "All contextual gates, decomposition, bootstrap and resource claims")
    return {"arrays": len(expected), "independently_replayed_feedback_trials": len(ledger),
            "checkpoint_static_probes": int(np.prod(arrays["checkpoint_probabilities"].shape)),
            "model_seeds": len(cfg["model_seeds"]), "conditions": len(CONDITIONS),
            "prefix_cloning_and_idle_copies_checked": True, "actual_call_increments_not_cumulative_totals_counted": True,
            "all_neural_output_activity_and_q_states_checked": True, "conflict_specific_qualification_checked": True,
            "paired_all_seed_bootstrap_and_loss_practice_identity_checked": True, "new_human_similarity_inference": False}


def verify_science(cfg, root, arrays, rows, summary, cache=None):
    validate_science(cfg); catalog_sha = verify_input_catalog(cfg, root)
    graph = core.raw_graph(Path(root)/cfg["graph"])
    core.demand((graph["n_kc"], graph["n_mbon"], len(graph["pre"])) == (512, 48, 1474), "Pinned partial anatomy")
    checks = verify_against_graph(cfg, graph, arrays, rows, summary)
    checks["frozen_input_catalog_sha256"] = catalog_sha
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    for name in ("config", "registration", "result-dir", "out"): parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--cache", type=Path); args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve()/"src"))
    from drosolearn.prospective import verify_envelope, sha, write_new
    cfg, reg, meta = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.cache)
    core.demand(meta["study_id"] == cfg["study_id"] and meta["stage"] == "completed", "Completed contextual metadata")
    core.demand(all(meta.get(k) is False for k in ("new_human_experiment", "shared_neural_mechanism_tested", "education_effect_tested")), "Interpretation boundary")
    stamps = [datetime.fromisoformat(v) for v in (reg["registered_at_utc"],meta["started_at_utc"],meta["completed_at_utc"])]
    core.demand(all(v.tzinfo is not None and v.utcoffset().total_seconds() == 0 for v in stamps)
                and stamps[0] < stamps[1] <= stamps[2] <= datetime.now(timezone.utc), "Prospective contextual UTC chronology")
    with np.load(args.result_dir/"arrays.npz", allow_pickle=False) as z: arrays = {k:z[k] for k in z.files}
    with (args.result_dir/"measurements.csv").open(encoding="utf8", newline="") as f:
        reader = csv.DictReader(f); core.demand(reader.fieldnames is not None and len(reader.fieldnames)==len(set(reader.fieldnames)), "Unique CSV header")
        rows = list(reader)
    summary = json.loads((args.result_dir/"summary.json").read_text(encoding="utf8"))
    checks = verify_science(cfg,args.root,arrays,rows,summary,args.cache)
    core.demand(meta.get("counts")==summary["counts"], "Metadata resource count binding")
    report = {"status":"passed","study_id":cfg["study_id"],"verified_at_utc":datetime.now(timezone.utc).isoformat(),
              "registration_sha256":sha(args.registration),"summary_sha256":sha(args.result_dir/"summary.json"),
              "scientific_producer_imported":False,"counts":checks}
    write_new(args.out,report); print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__": main()
