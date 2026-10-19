"""Independent saved-trace reduction and synthetic-policy arithmetic.

This verifier never imports the scientific producer or its source parser.
Synthetic action tapes are an explicit non-identification counterexample.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

_SPEC = importlib.util.spec_from_file_location("frozen_behavior_independent_core", Path(__file__).with_name("verify_human_shell.py"))
core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(core)
CONDITIONS = core.CONDITIONS
POLICIES = ("uniform", "persistence", "q")
EVENTS = ("nogo_hidden", "go_empty", "go_gain", "go_loss")
METRICS = ("acquisition_gain", "reversal_early", "reversal_recovery")
NAMESPACE = 2026101902
FIELDS = ("kind", "condition", "seed", "participant_id", "shell", "phase", "regime", "event", "statistic",
          "numerator", "denominator", "people_with_denominator", "value", *METRICS,
          "head_updates", "internal_updates", "value_updates")


def validate_science(cfg, original):
    required = {"study_id": "shell_behavior_v1", "scope": "exploratory_saved_trace_description_and_planned_metric_counterexample",
                "synthetic_seeds": list(range(16001, 16009)),
                "synthetic_policies": list(POLICIES), "trials_per_person": 216, "rng_namespace": NAMESPACE,
                "q_alpha": .1, "q_beta": 4., "conditions": list(CONDITIONS),
                "trajectory_parent": "results/human_shell_rng_v2/trajectories.npz",
                "source_manifest": "data/human/tnu_shell_2024/source-manifest.json",
                "external_manifest": "data/human/tnu_shell_2024/source-manifest.json"}
    for key, value in required.items():
        core.demand(cfg.get(key) == value, f"Fixed descriptive setting changed: {key}")
    core.demand(cfg["test_ids"] == original["test_ids"] and cfg["test_seeds"] == original["test_seeds"]
                == list(range(13101, 13106)), "Original source partition/seed axis must remain unchanged")
    core.demand(len(cfg["test_ids"]) == 29 and len(set(cfg["test_ids"])) == 29, "Fixed 29-person axis")
    for pid in cfg["test_ids"]:
        core.demand(type(pid) is str and re.fullmatch(r"A[0-9]{4}", pid) is not None, "Canonical source ID")


def generator(seed, pid):
    core.demand(type(seed) is int and seed >= 0 and type(pid) is str and re.fullmatch(r"A[0-9]{4}", pid) is not None,
                "Canonical synthetic stream address required")
    return np.random.default_rng(np.random.SeedSequence([NAMESPACE, seed, int(pid[1:])]))


def phase_groups(record):
    """Partition each cue into contiguous state runs before making pairs."""
    result = []
    shell, states = np.asarray(record["shell"]), np.asarray(record["state"])
    core.demand(shell.ndim == states.ndim == 1 and len(shell) == len(states), "Aligned one-dimensional source inputs")
    for cue in range(1, 7):
        locations = np.flatnonzero(shell == cue)
        core.demand(len(locations) > 0, "All six original cue identities required")
        sequence = states[locations]
        starts = np.r_[0, np.flatnonzero(np.diff(sequence)) + 1]
        ends = np.r_[starts[1:], len(locations)]
        core.demand(np.array_equal(sequence[starts], [1, 2, 1, 2]), "Four original cue phases required")
        for phase, (start, end) in enumerate(zip(starts, ends)):
            result.append((cue - 1, phase, locations[start:end]))
    return result


def transition_counts(records, ids, actions):
    actions = np.asarray(actions)
    core.demand(actions.ndim == 3 and actions.shape[1] == len(ids) and np.isin(actions, [0, 1]).all(),
                "Binary aligned saved action streams required")
    streams, people, trials = actions.shape
    counts = np.zeros((streams, people, 6, 4, 4, 3), dtype=np.int64)
    for pi, pid in enumerate(ids):
        record = records[pid]
        core.demand(len(record["shell"]) == trials, "Saved trace length mismatch")
        for cue, phase, locations in phase_groups(record):
            left, right = locations[:-1], locations[1:]
            for ki in range(streams):
                previous, following = actions[ki, pi, left], actions[ki, pi, right]
                # Potential content is used only to label observed Go feedback.
                event = np.where(previous == 0, 0, np.asarray(record["content"])[left] + 1)
                for ei in range(4):
                    chosen = event == ei
                    counts[ki, pi, cue, phase, ei] = [int(chosen.sum()), int(following[chosen].sum()),
                                                     int((previous[chosen] == following[chosen]).sum())]
    return counts


def replay_policy(record, seed, pid, policy, flip=False):
    core.demand(policy in POLICIES and type(flip) is bool, "Known synthetic policy and intervention")
    n = len(record["shell"]); uniforms = generator(seed, pid).random(n)
    probability = np.empty(n); action = np.empty(n, dtype=np.int8); reward = np.empty(n, dtype=np.int8)
    steps = np.zeros(n); table = np.zeros((6, 2)); last = np.full(6, -1, dtype=int)
    for ti, cue_value in enumerate(record["shell"]):
        cue = int(cue_value) - 1
        core.demand(0 <= cue < 6, "Synthetic cue domain")
        if policy == "uniform": p = .5
        elif policy == "persistence": p = .5 if last[cue] < 0 else (.9 if last[cue] == 1 else .1)
        else: p = core.logistic(4. * (table[cue, 1] - table[cue, 0]))
        a = int(uniforms[ti] < p)
        r = (0, 1, -1)[int(record["content"][ti])] if a else 0
        if flip: r = -r
        probability[ti], action[ti], reward[ti] = p, a, r
        if policy == "q":
            old = float(table[cue, a]); table[cue, a] = old + .1 * (r - old)
            steps[ti] = abs(float(table[cue, a]) - old)
        last[cue] = a
    return probability, action, reward, steps, table


def expected_arrays(cfg, records, old_autonomous):
    ids, seeds = cfg["test_ids"], cfg["synthetic_seeds"]
    old = np.asarray(old_autonomous)
    n = cfg["trials_per_person"]
    core.demand(old.shape == (len(CONDITIONS), len(cfg["test_seeds"]), len(ids), n), "Original autonomous action axes")
    human = np.stack([records[pid]["action"] for pid in ids])
    all_actions = np.concatenate([human[None], old.reshape(-1, len(ids), n)])
    result = {"transition_counts": transition_counts(records, ids, all_actions)}
    shape = (3, len(seeds), len(ids), n)
    for field, dtype in (("probabilities", float), ("actions", np.int8), ("rewards", np.int8), ("value_steps", float)):
        result["synthetic_" + field] = np.empty(shape, dtype=dtype)
        result["counterfactual_" + field] = np.empty(shape[1:], dtype=dtype)
    result["final_value_tables"] = np.zeros((2, len(seeds), len(ids), 6, 2))
    result["synthetic_metrics"] = np.empty((3, len(seeds), len(ids), 3))
    for si, seed in enumerate(seeds):
        for pi, pid in enumerate(ids):
            record = records[pid]
            for fi, policy in enumerate(POLICIES):
                tape = replay_policy(record, seed, pid, policy)
                for field, value in zip(("probabilities", "actions", "rewards", "value_steps"), tape[:4]):
                    result["synthetic_" + field][fi, si, pi] = value
                result["synthetic_metrics"][fi, si, pi] = core.behavioral_metrics(record, tape[1])
                if policy == "q": result["final_value_tables"][0, si, pi] = tape[4]
            tape = replay_policy(record, seed, pid, "q", flip=True)
            for field, value in zip(("probabilities", "actions", "rewards", "value_steps"), tape[:4]):
                result["counterfactual_" + field][si, pi] = value
            result["final_value_tables"][1, si, pi] = tape[4]
    result["scripted_actions"] = result["synthetic_actions"][2].copy()
    result["scripted_metrics"] = np.stack([np.stack([core.behavioral_metrics(records[pid], result["scripted_actions"][si, pi])
                                                   for pi, pid in enumerate(ids)]) for si in range(len(seeds))])
    return result


def table_row(**items):
    result = dict.fromkeys(FIELDS); result.update(items)
    return result


def expected_rows(cfg, a):
    rows = []; streams = [("human", 0)] + [(c, s) for c in CONDITIONS for s in cfg["test_seeds"]]
    for ki, (condition, seed) in enumerate(streams):
        for cue in range(6):
            for phase in range(4):
                regime = core.SEASONS[cue + 1][phase % 2]
                for event_index, event in enumerate(EVENTS):
                    cells = a["transition_counts"][ki, :, cue, phase, event_index]
                    denominator = int(cells[:, 0].sum()); people = int(np.count_nonzero(cells[:, 0]))
                    for column, label in ((1, "next_go"), (2, "stay")):
                        numerator = int(cells[:, column].sum())
                        rows.append(table_row(kind="saved_transition", condition=condition, seed=seed, shell=cue + 1,
                                              phase=phase + 1, regime=regime, event=event, statistic=label,
                                              numerator=numerator, denominator=denominator, people_with_denominator=people,
                                              value=numerator / denominator if denominator else None))
    for fi, policy in enumerate((*POLICIES, "scripted_q_tape")):
        values = a["synthetic_metrics"][fi] if fi < 3 else a["scripted_metrics"]
        for si, seed in enumerate(cfg["synthetic_seeds"]):
            for pi, pid in enumerate(cfg["test_ids"]):
                rows.append(table_row(kind="synthetic_metric", condition=policy, seed=seed, participant_id=pid,
                                      **dict(zip(METRICS, map(float, values[si, pi]))), head_updates=0, internal_updates=0,
                                      value_updates=cfg["trials_per_person"] if policy == "q" else 0))
    for si, seed in enumerate(cfg["synthetic_seeds"]):
        for pi, pid in enumerate(cfg["test_ids"]):
            count = int(np.count_nonzero(a["synthetic_actions"][2, si, pi] != a["counterfactual_actions"][si, pi]))
            rows.append(table_row(kind="reward_intervention", condition="q", seed=seed, participant_id=pid,
                                  statistic="choice_disagreement", numerator=count, denominator=cfg["trials_per_person"],
                                  value=count / cfg["trials_per_person"], head_updates=0, internal_updates=0,
                                  value_updates=cfg["trials_per_person"]))
    return rows


def expected_summary(cfg, a):
    totals = a["transition_counts"][..., 0].sum(axis=(1, 2, 3, 4))
    people, trial_count, seeds = len(cfg["test_ids"]), cfg["trials_per_person"], len(cfg["synthetic_seeds"])
    n = seeds * people * trial_count
    stream_cells = a["transition_counts"].shape[0] * people
    exact = np.array_equal(a["synthetic_metrics"][2], a["scripted_metrics"])
    groups = [{"policy": policy, "metric_means": dict(zip(METRICS, map(float, a["synthetic_metrics"][fi].mean(axis=(0, 1))))),
               "online_value_update_calls": n if policy == "q" else 0} for fi, policy in enumerate(POLICIES)]
    different = int(np.count_nonzero(a["counterfactual_actions"] != a["scripted_actions"]))
    return {"status": "completed", "scope": "exploratory_saved_trace_description_and_planned_metric_counterexample",
            "counts": {"human_records_reanalyzed": people, "old_model_streams": stream_cells - people,
                       "old_total_streams": stream_cells, "old_recorded_trials_reduced": stream_cells * trial_count,
                       "next_same_shell_candidates": stream_cells * (trial_count - 6),
                       "excluded_phase_boundary_pairs": stream_cells * 18, "eligible_within_phase_pairs": int(totals.sum()),
                       "new_synthetic_policy_choices": 4 * n, "scripted_tape_reads": n, "new_human_experiments": 0,
                       "new_neural_parameter_updates": 0, "new_value_update_calls": 2 * n,
                       "nonzero_value_update_steps": int(np.count_nonzero(a["synthetic_value_steps"]) + np.count_nonzero(a["counterfactual_value_steps"]))},
            "event_order": list(EVENTS), "synthetic_policy_results": groups,
            "saved_transition_summary": {"stream_order": ["human"] + list(CONDITIONS),
                                         "per_stream_eligible_pairs": list(map(int, totals)),
                                         "conditional_rates_are_descriptive_not_causal_reward_effects": True,
                                         "missing_denominators_are_null": True},
            "metric_identification": {"planned_counterexample_verified": bool(exact),
                                      "three_summaries_uniquely_identify_reward_learning": False if exact else None,
                                      "scripted_policy_online_updates": 0, "scripted_policy_uses_precomputed_action_tape": True,
                                      "scripted_policy_is_eligible_fair_learning_control": False,
                                      "notice": "Exact equality was planned by construction; it limits mechanism identification and is not a new human learning effect."},
            "reward_intervention": {"q_vs_fixed_tape_disagreeing_choices": different, "evaluated_choices": n,
                                    "disagreement_fraction": different / n, "fixed_tape_choice_change": 0,
                                    "notice": "Reward signs reversed in a synthetic intervention; original human rewards and original comparison are unchanged."},
            "original_similarity_verdict_changed": False, "new_human_similarity_confirmation": False,
            "shared_neural_mechanism_claim": False, "education_effect_claim": False,
            "notice": "Known old outcomes informed this separate exploratory plan. Do not interpret pooled transitions, shared schedules, computational seeds or tape copies as independent people."}


def verify_against_records(cfg, records, old_autonomous, arrays, rows, summary):
    expected = expected_arrays(cfg, records, old_autonomous)
    core.demand(set(arrays) == set(expected), "Complete descriptive/synthetic numeric inventory")
    integers = {"transition_counts", "synthetic_actions", "synthetic_rewards", "counterfactual_actions", "counterfactual_rewards", "scripted_actions"}
    for field, value in expected.items():
        observed = np.asarray(arrays[field])
        if field in integers:
            core.demand(observed.dtype.kind in "biu" and np.array_equal(observed, value), f"Exact count/action/reward {field}")
        else: core.array_close(observed, value, f"Independent synthetic replay {field}")
    ledger = expected_rows(cfg, expected)
    core.demand(len(rows) == len(ledger), "All strata, empty cells and synthetic measurement rows required")
    for index, (actual, wanted) in enumerate(zip(rows, ledger)):
        if all(isinstance(v, str) for v in actual.values()): core.check_csv_row(actual, wanted, f"measurement {index}")
        else: core.compare_tree(actual, wanted, f"measurement {index}")
    core.compare_tree(summary, expected_summary(cfg, expected), "Independent complete descriptive summary")
    count = len(cfg["synthetic_seeds"]) * len(cfg["test_ids"]) * cfg["trials_per_person"]
    return {"arrays": len(expected), "measurement_records": len(ledger), "independently_replayed_synthetic_choices": 4 * count,
            "saved_action_streams": int(expected["transition_counts"].shape[0]),
            "complete_strata_including_empty_cells": int(np.prod(expected["transition_counts"].shape[0:1] + expected["transition_counts"].shape[2:5])),
            "within_phase_pairs": int(expected["transition_counts"][..., 0].sum()),
            "tape_equal_metrics_and_zero_online_updates_checked": True, "new_human_inference": False}


def verify_science(cfg, root, arrays, rows, summary, cache=None):
    root = Path(root)
    original = json.loads((root / "configs/human_shell_rng_v2.json").read_text(encoding="utf8"))
    validate_science(cfg, original)
    core.demand(cache is not None, "Frozen external source cache required")
    records, calibration, heldout, exclusions = core.parse_cohort(Path(cache) / "Dataset_TNU.zip")
    core.demand(cfg["test_ids"] == heldout and len(calibration) == 20 and set(exclusions) == {"A0016"}, "Original cohort remains unchanged")
    with np.load(root / cfg["trajectory_parent"], allow_pickle=False) as z:
        core.demand(z["actions"].shape == (2, 11, 5, 29, 216), "Original saved trajectory axes")
        original_actions = z["actions"][1]
    return verify_against_records(cfg, records, original_actions, arrays, rows, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    for name in ("config", "registration", "result-dir", "out"): parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(args.root.resolve() / "src"))
    from drosolearn.prospective import verify_envelope, sha, write_new
    cfg, reg, meta = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.cache)
    core.demand(meta["study_id"] == cfg["study_id"] and meta["stage"] == "completed", "Metadata study identity")
    core.demand(all(meta.get(key) is False for key in ("new_human_experiment", "shared_neural_mechanism_tested", "education_effect_tested")), "Interpretation flags")
    stamps = [datetime.fromisoformat(v) for v in (reg["registered_at_utc"], meta["started_at_utc"], meta["completed_at_utc"])]
    core.demand(all(v.tzinfo is not None and v.utcoffset().total_seconds() == 0 for v in stamps)
                and stamps[0] < stamps[1] <= stamps[2] <= datetime.now(timezone.utc), "Prospective UTC timeline")
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as z: arrays = {k: z[k] for k in z.files}
    with (args.result_dir / "measurements.csv").open(encoding="utf8", newline="") as f:
        reader = csv.DictReader(f)
        core.demand(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), "Unique CSV headers")
        rows = list(reader)
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf8"))
    checks = verify_science(cfg, args.root, arrays, rows, summary, args.cache)
    core.demand(meta.get("counts") == summary["counts"], "Summary counts bound to metadata")
    report = {"status": "passed", "study_id": cfg["study_id"], "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "registration_sha256": sha(args.registration), "summary_sha256": sha(args.result_dir / "summary.json"),
              "scientific_producer_imported": False, "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__": main()
