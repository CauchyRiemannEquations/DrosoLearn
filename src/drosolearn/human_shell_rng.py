"""Separately registered repair of autonomous RNG addressing only.

The original human_shell source and outputs are immutable. This explicit new
function reuses its frozen policy/measurement API; it neither calls fit nor
recomputes conditional predictions. No AST rewriting or global RNG patching.
"""
import copy
import re
import numpy as np

from .human_shell import _copies, _trajectory, behavior_metrics, check_records
from .human_shell_models import CONDITIONS, build_graphs, make_model


def action_entropy(seed, participant_id):
    """A compositional address, not the sum of two independent factor IDs."""
    if type(seed) is not int or seed < 0:
        raise ValueError("Nonnegative integer computational seed required")
    if not isinstance(participant_id, str) or re.fullmatch(r"A[0-9]{4}", participant_id) is None:
        raise ValueError("Canonical A-prefixed participant ID required")
    return (20261019, seed, int(participant_id[1:]))


def validate_action_addresses(cfg):
    ids, seeds = cfg["test_ids"], cfg["test_seeds"]
    if not ids or not seeds or len(ids) != len(set(ids)) or len(seeds) != len(set(seeds)):
        raise ValueError("Unique nonempty participant and seed axes required")
    addresses = [action_entropy(seed, pid) for seed in cfg["test_seeds"] for pid in cfg["test_ids"]]
    if len(addresses) != len(set(addresses)):
        raise ValueError("Duplicate seed/person action-stream address")
    return addresses


def produce_corrected(originalarrays, originalrows, originalwitnesses, testrecords, cfg, graph, selection):
    """Copy conditional artifacts exactly; execute only autonomous trajectories.

    Returns the same combined-array/row/witness schema as the original producer.
    Parent file hashes and literal CSV conditional-line copying are enforced by
    the outer runner; this function preserves arrays' conditional byte content
    and conditional row values without touching the caller's objects.
    """
    ids, seeds, n = cfg["test_ids"], cfg["test_seeds"], cfg["trials_per_person"]
    check_records(testrecords, ids, n)
    validate_action_addresses(cfg)
    expected = (2, len(CONDITIONS), len(seeds), len(ids), n)
    keys = ("probabilities", "actions", "rewards", "head_steps", "internal_steps", "value_steps")
    if set(originalarrays) != set(keys) or any(np.asarray(originalarrays[k]).shape != expected for k in keys):
        raise ValueError("Parent trajectory array schema differs")
    arrays = {key: np.array(originalarrays[key], copy=True, order="K") for key in keys}
    conditional = [copy.deepcopy(row) for row in originalrows if row["mode"] == "conditional"]
    if len(conditional) != len(CONDITIONS) * len(seeds) * len(ids):
        raise ValueError("Parent conditional row count differs")
    trajectory_rows = list(conditional)
    witnesses = {key: np.array(value, copy=True, order="K") for key, value in originalwitnesses.items()
                 if key.startswith("conditional_")}
    if not witnesses:
        raise ValueError("Missing original conditional witnesses")
    graphs = {seed: build_graphs(graph, seed) for seed in seeds}
    new_witnesses = {}
    for si, seed in enumerate(seeds):
        for pi, pid in enumerate(ids):
            sequence = np.random.SeedSequence(action_entropy(seed, pid))
            uniforms = np.random.default_rng(sequence).random(n)
            for ci, condition in enumerate(CONDITIONS):
                rate = selection["rates"].get(condition, cfg["rates"][0])
                model = make_model(condition, seed, rate, selection["calibration_bias"], graphs[seed])
                before, initial = _copies(model), model.witness()
                p, a, r, steps = _trajectory(model, testrecords[pid], True, uniforms)
                at = (1, ci, si, pi)
                arrays["probabilities"][at], arrays["actions"][at], arrays["rewards"][at] = p, a, r
                for step_i, key in enumerate(("head_steps", "internal_steps", "value_steps")):
                    arrays[key][at] = steps[step_i]
                final = model.witness()
                if pi == 0:
                    for stage, witness in (("initial", initial), ("final", final)):
                        for key, value in witness.items():
                            name = f"autonomous_{condition}_{stage}_{key}"
                            new_witnesses.setdefault(name, []).append(np.asarray(value))
                after = _copies(model)
                row = {"mode": "autonomous", "condition": condition, "seed": seed, "participant_id": pid,
                       "mean_nll": None, "brier": None, **behavior_metrics(testrecords[pid], a),
                       "head_update_calls": model.head_updates, "internal_update_calls": model.internal_updates,
                       "value_update_calls": model.value_updates,
                       "head_nonzero_steps": int(np.count_nonzero(steps[0])),
                       "internal_nonzero_steps": int(np.count_nonzero(steps[1])),
                       "value_nonzero_steps": int(np.count_nonzero(steps[2])),
                       "head_change_l2": float(np.linalg.norm(after[0] - before[0])),
                       "internal_change_l2": float(np.linalg.norm(after[1] - before[1])),
                       "value_change_l2": float(np.linalg.norm(after[2] - before[2]))}
                for key in ("probe_kc", "probe_mbon", "probe_decision"):
                    row[f"{key}_change_l2"] = float(np.linalg.norm(final[key] - initial[key]))
                trajectory_rows.append(row)
    witnesses.update({key: np.stack(values) for key, values in new_witnesses.items()})
    for key in keys:
        if arrays[key][0].tobytes() != np.asarray(originalarrays[key])[0].tobytes():
            raise ValueError("Conditional trajectory changed during RNG repair")
    for key in originalwitnesses:
        if key.startswith("conditional_") and witnesses[key].tobytes() != np.asarray(originalwitnesses[key]).tobytes():
            raise ValueError("Conditional witness changed during RNG repair")
    if trajectory_rows[:len(conditional)] != conditional:
        raise ValueError("Conditional metrics changed during RNG repair")
    return arrays, trajectory_rows, witnesses
