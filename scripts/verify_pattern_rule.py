"""Independent arithmetic for the finite artificial-history pattern experiment.

No scientific producer, learner, catalog generator or metric is imported.
The immutable earlier independent graph/rewiring code is reused as evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

CORE_PATH = Path(__file__).resolve().with_name("verify_human_shell.py")
SPEC = importlib.util.spec_from_file_location("pattern_independent_frozen_graph", CORE_PATH)
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)

CONDITIONS = ("fixed", "plastic", "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic", "linear", "uniform", "lookup_q")
SETS = ("acquisition", "phase_diagnostic", "new_alphabet", "new_alphabet_phase", "new_length")
INPUT_CATALOG = "data/synthetic/pattern_rule_v1/input_catalog.json"
SEQUENCES = (
    ((0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 1, 0), (1, 0, 0, 1)),
    ((0, 1, 0, 1), (1, 0, 1, 0), (0, 0, 1, 1), (1, 1, 0, 0)),
    ((2, 3, 2, 3), (3, 2, 3, 2), (2, 3, 3, 2), (3, 2, 2, 3)),
    ((2, 3, 2, 3), (3, 2, 3, 2), (2, 2, 3, 3), (3, 3, 2, 2)),
    ((0, 1, 0, 1, 0, 1), (1, 0, 1, 0, 1, 0), (0, 1, 1, 0, 0, 1), (1, 0, 0, 1, 1, 0)),
)


def encode(sequence):
    core.demand(len(sequence) in (4, 6) and all(type(x) is int and 0 <= x <= 3 for x in sequence), "Finite literal history required")
    result = np.zeros(7)
    for index, symbol in enumerate(sequence): result[index] = symbol / 3.
    result[6] = len(sequence) / 6.
    return result


def oracle(sequence):
    return int(all(a != b for a, b in zip(sequence, sequence[1:])))


def catalog():
    inputs = np.asarray([[encode(s) for s in group] for group in SEQUENCES])
    targets = np.asarray([[oracle(s) for s in group] for group in SEQUENCES], dtype=int)
    core.demand(np.array_equal(targets, np.tile([1, 1, 0, 0], (5, 1))), "Independent rule oracle disagrees with declared class")
    return inputs, targets


def analytical_catalog():
    """Exact finite shortcut counts; no stochastic or trainable estimator."""
    train = {s: oracle(s) for s in SEQUENCES[0]}
    reports = {}
    feature_functions = {"first": lambda s: s[0], "last": lambda s: s[-1],
                         "length": lambda s: len(s),
                         "frequency": lambda s: tuple(sorted(Counter(s).items())),
                         "symbol_sum": lambda s: sum(s)}
    for name, group in zip(SETS, SEQUENCES):
        targets = [oracle(s) for s in group]
        best = {}
        for key, feature in feature_functions.items():
            counts = defaultdict(Counter)
            for sequence, target in zip(group, targets): counts[feature(sequence)][target] += 1
            best[key] = {"correct_count": sum(max(c.values()) for c in counts.values()), "total": len(group),
                         "accuracy": sum(max(c.values()) for c in counts.values()) / len(group)}
        shortcut = [int(s[0] != s[-1]) for s in group]
        lookup_p = [float(train[s]) if s in train else .5 for s in group]
        lookup_accuracy = sum(p if t else 1 - p for p, t in zip(lookup_p, targets)) / len(group)
        reports[name] = {"oracle_correct": sum(oracle(s) == t for s, t in zip(group, targets)),
                         "first_last_correct": sum(p == t for p, t in zip(shortcut, targets)),
                         "lookup_expected_accuracy": lookup_accuracy,
                         "single_cue_optimal": best,
                         "class_counts": {str(c): targets.count(c) for c in (0, 1)},
                         "train_overlap": sum(s in train for s in group)}
    unique = {s for group in SEQUENCES for s in group}
    return {"slots": 20, "unique_strings": len(unique), "by_set": reports,
            "alphabet_combined_overlap": len(set(SEQUENCES[2]) & set(SEQUENCES[3])),
            "phase_novel_only_class_counts": {"0": 2, "1": 0},
            "phase_novel_only_always_nogo_accuracy": 1.}


class AuditActor:
    """Observed input/action/reward API; all raw labels remain in environment."""
    def __init__(self, condition, seed, graph, train_inputs, rate=.1):
        core.demand(condition in CONDITIONS, "Unknown finite-pattern condition")
        self.condition, self.rate = condition, rate
        self.plastic = condition.endswith("plastic")
        self.head_updates = self.internal_updates = self.value_updates = 0
        self.values = np.empty(0); self.value_catalog = np.empty((0, 7))
        self.weights = np.empty(0); self.initial_weights = np.empty(0); self.upper = np.empty(0)
        self.pre = self.post = np.empty(0, dtype=int)
        self.encoder = np.empty((0, 7)); self.bias = np.empty(0)
        self.scale = 1.
        if condition in ("uniform", "lookup_q"):
            self.head = np.empty(0)
            if condition == "lookup_q":
                self.value_catalog = np.array(train_inputs, copy=True)
                self.values = np.zeros((4, 2))
            return
        if condition in CONDITIONS[:6]:
            self.graph = graph
            self.pre, self.post = graph["pre"], graph["post"]
            rng = np.random.default_rng(seed + 100000)
            self.encoder = rng.normal(0, .8, (graph["n_kc"], 7))
            self.bias = rng.uniform(-.25, .25, graph["n_kc"])
            normalizer = np.maximum(np.bincount(self.post, weights=graph["counts"], minlength=graph["n_mbon"]), 1.)
            self.weights = graph["counts"] / normalizer[self.post]
            self.initial_weights, self.upper = self.weights.copy(), 2 * self.weights
            self.head = np.zeros(graph["n_mbon"])
        else:
            self.head = np.zeros(7)
        _, _, features = self.features(train_inputs)
        self.scale = float(np.sqrt(np.mean(np.sum((features[:, 1] - features[:, 0]) ** 2, axis=-1))))
        core.demand(np.isfinite(self.scale) and self.scale > 0, "Positive training-only feature scale")

    def features(self, inputs):
        x = np.asarray(inputs, float)
        if x.shape[-1] != 7: raise ValueError("Explicit seven-dimensional history input required")
        paired = np.stack((np.zeros_like(x), x), axis=-2)
        if self.condition in ("uniform", "lookup_q"):
            empty = np.empty((*paired.shape[:-1], 0))
            return empty, empty.copy(), empty.copy()
        if self.condition not in CONDITIONS[:6]:
            empty = np.empty((*paired.shape[:-1], 0))
            return empty, empty.copy(), paired / self.scale
        kc = np.maximum(paired @ self.encoder.T + self.bias, 0)
        mbon = np.empty((*kc.shape[:-1], self.graph["n_mbon"]))
        flat_kc, flat_mb = kc.reshape(-1, kc.shape[-1]), mbon.reshape(-1, mbon.shape[-1])
        for index, row in enumerate(flat_kc):
            flat_mb[index] = np.bincount(self.post, weights=self.weights * row[self.pre], minlength=self.graph["n_mbon"])
        return kc, mbon, mbon / self.scale

    def probability(self, inputs):
        return 1. / (1. + np.exp(-np.clip(self.raw_gap(inputs), -40., 40.)))

    def raw_gap(self, inputs):
        if self.condition == "lookup_q":
            x = np.asarray(inputs, float)
            raw = np.zeros(x.shape[:-1])
            for index, vector in enumerate(x.reshape(-1, 7)):
                matches = np.flatnonzero(np.all(self.value_catalog == vector, axis=1))
                if len(matches): raw.reshape(-1)[index] = 4. * (self.values[matches[0], 1] - self.values[matches[0], 0])
            return raw
        _, _, z = self.features(inputs)
        return (z[..., 1, :] - z[..., 0, :]) @ self.head

    def update(self, inputs, action, reward):
        core.demand(type(action) is int and action in (0, 1) and type(reward) is int and reward in (-1, 0, 1), "Observed binary action/signed reward required")
        core.demand(action == 1 or reward == 0, "Hidden NoGo result forbidden")
        before_head, before_weight, before_value = self.head.copy(), self.weights.copy(), self.values.copy()
        if self.condition == "lookup_q":
            matches = np.flatnonzero(np.all(self.value_catalog == np.asarray(inputs), axis=1))
            core.demand(len(matches) == 1, "Q updates only declared training identities")
            index = int(matches[0])
            self.values[index, action] += .1 * (reward - self.values[index, action])
            self.value_updates += 1
        elif self.condition != "uniform":
            kc, _, z = self.features(inputs)
            p = float(self.probability(inputs))
            factor = (action - p) * reward
            self.head += self.rate * factor * (z[1] - z[0])
            self.head_updates += 1
            if self.plastic:
                eligibility = before_head[self.post] * (kc[1, self.pre] - kc[0, self.pre]) / self.scale
                self.weights = np.clip(self.weights + .1 * self.rate * factor * eligibility, 0, self.upper)
                self.internal_updates += 1
        return (float(np.linalg.norm(self.head - before_head)), float(np.linalg.norm(self.weights - before_weight)),
                float(np.linalg.norm(self.values - before_value)))

    def witness(self, probes):
        kc, mbon, decision = self.features(probes)
        return {"head": self.head.copy(), "weights": self.weights.copy(), "probe_kc": kc, "probe_mbon": mbon,
                "probe_decision": decision, "encoder": self.encoder.copy(), "bias": self.bias.copy(),
                "pre": self.pre.copy(), "post": self.post.copy(), "initial_weights": self.initial_weights.copy(),
                "upper": self.upper.copy(), "feature_scale": np.asarray(self.scale),
                "value_table": self.values.copy(), "value_catalog": self.value_catalog.copy()}


def validate_science(cfg, allow_fixture=False):
    fixed = {"study_id": "pattern_rule_v1", "conditions": list(CONDITIONS),
             "head_lr": .1, "internal_lr_multiplier": .1, "rng_namespace": 202610191,
             "lookup_alpha": .1, "lookup_beta": 4.,
             "primary_condition": "plastic", "set_order": list(SETS),
             "acquisition_threshold": .8, "acquisition_gain": .25, "class_threshold": .7,
             "ood_threshold": .75, "ood_max_drop": .1,
             "normalization": "initial_training_four_input_feature_gap_RMS",
             "input_catalog": INPUT_CATALOG, "input_files": [INPUT_CATALOG],
             "representation": "six_raw_symbol_slots_divided_by_3_then_length_divided_by_6",
             "phase_diagnostic_is_generalization_test": False}
    for key, expected in fixed.items(): core.demand(cfg.get(key) == expected, "Fixed independent pattern design: " + key)
    if allow_fixture and cfg.get("scope") == "integration_test":
        seeds, n, checkpoints = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
        core.demand(type(seeds) is list and 1 <= len(seeds) <= 2 and len(seeds) == len(set(seeds))
                    and all(type(s) is int and 90001 <= s <= 90100 for s in seeds), "Synthetic independent seed axis")
        core.demand(type(n) is int and 4 <= n <= 16 and n % 4 == 0, "Small synthetic training budget")
        core.demand(checkpoints == sorted(set(checkpoints)) and checkpoints[0] == 0 and checkpoints[-1] == n
                    and all(type(t) is int and t % 4 == 0 for t in checkpoints), "Synthetic checkpoint grid")
        core.demand(cfg["population_pass_count"] == len(seeds), "Synthetic population criterion")
    else:
        core.demand(cfg.get("scope") == "prospectively_registered_artificial_pattern"
                    and cfg["model_seeds"] == list(range(15101, 15111)) and cfg["training_trials"] == 400
                    and cfg["checkpoints"] == [0, 50, 100, 200, 400] and cfg["population_pass_count"] == 8,
                    "Prospective pattern seed/budget/checkpoint design")


def independent_catalog_arrays():
    inputs, targets = catalog()
    lengths = np.asarray([[len(s) for s in group] for group in SEQUENCES], dtype=int)
    strings = np.full((5, 4, 6), -1, dtype=int)
    for gi, group in enumerate(SEQUENCES):
        for si, sequence in enumerate(group): strings[gi, si, :len(sequence)] = sequence
    return {"probe_strings": strings, "probe_lengths": lengths, "probe_inputs": inputs, "probe_targets": targets,
            "probe_seen_in_training": np.asarray([[s in SEQUENCES[0] for s in group] for group in SEQUENCES]),
            "train_strings": strings[0].copy(), "train_lengths": lengths[0].copy(), "train_catalog": inputs[0].copy()}


def verify_input_catalog(cfg, root):
    core.demand(cfg.get("input_catalog") == INPUT_CATALOG and cfg.get("input_files") == [INPUT_CATALOG],
                "Exact prospectively frozen pattern catalog path")
    path = core.contained(Path(root), INPUT_CATALOG)
    def unique_pairs(items):
        result = {}
        for key, value in items:
            core.demand(key not in result, "Duplicate pattern input JSON key")
            result[key] = value
        return result
    def invalid_constant(value): raise ValueError("Nonfinite pattern input JSON")
    document = json.loads(path.read_text(encoding="utf8"), object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
    core.demand(set(document) == {"study_id", "set_order", "catalog"} and document["study_id"] == "pattern_rule_v1"
                and document["set_order"] == list(SETS), "Frozen pattern input schema or ordering")
    expected = independent_catalog_arrays()
    core.demand(isinstance(document["catalog"], dict) and set(document["catalog"]) == set(expected), "Pattern input field closure")
    for key, value in expected.items():
        actual = np.asarray(document["catalog"][key])
        core.demand(actual.dtype.kind == value.dtype.kind and actual.shape == value.shape
                    and np.array_equal(actual, value), "Literal frozen catalog differs: " + key)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_artifacts(cfg, graph):
    data = independent_catalog_arrays()
    seeds, n, checkpoints = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
    arrays = {key: value.copy() for key, value in data.items()}
    arrays.update({"checkpoints": np.asarray(checkpoints), "model_seeds": np.asarray(seeds),
                   "schedules": np.empty((len(seeds), n), dtype=int), "uniforms": np.empty((len(seeds), n))})
    shape = (len(CONDITIONS), len(seeds), n)
    for key in ("trial_probabilities", "trial_actions", "trial_rewards", "trial_head_steps", "trial_internal_steps", "trial_value_steps",
                "trial_head_update_calls", "trial_internal_update_calls", "trial_value_update_calls"):
        arrays[key] = np.empty(shape, dtype=int if key in ("trial_actions", "trial_rewards", "trial_head_update_calls", "trial_internal_update_calls", "trial_value_update_calls") else float)
    for key in ("checkpoint_probabilities", "checkpoint_logits"):
        arrays[key] = np.empty((len(CONDITIONS), len(seeds), len(checkpoints), 5, 4))
    for key in ("checkpoint_head_calls", "checkpoint_internal_calls", "checkpoint_value_calls"):
        arrays[key] = np.empty((len(CONDITIONS), len(seeds), len(checkpoints)), dtype=int)
    rows, witnesses, graph_cache = [], {}, {}
    probes = data["probe_inputs"].reshape(20, 7)
    # The source writes trial rows by seed, then condition, then trial.
    for si, seed in enumerate(seeds):
        order_rng = np.random.default_rng(np.random.SeedSequence([202610191, seed, 0]))
        schedule = np.concatenate([order_rng.permutation(4) for _ in range(n // 4)])
        uniforms = np.random.default_rng(np.random.SeedSequence([202610191, seed, 1])).random(n)
        arrays["schedules"][si], arrays["uniforms"][si] = schedule, uniforms
        for ci, condition in enumerate(CONDITIONS):
            actor = AuditActor(condition, seed, core.graph_for(condition, seed, graph, graph_cache), data["train_catalog"])
            snapshots = []
            def save(trial):
                at = checkpoints.index(trial)
                probabilities = actor.probability(probes)
                witness = actor.witness(probes)
                raw = actor.raw_gap(probes)
                arrays["checkpoint_probabilities"][ci, si, at] = probabilities.reshape(5, 4)
                arrays["checkpoint_logits"][ci, si, at] = raw.reshape(5, 4)
                arrays["checkpoint_head_calls"][ci, si, at] = actor.head_updates
                arrays["checkpoint_internal_calls"][ci, si, at] = actor.internal_updates
                arrays["checkpoint_value_calls"][ci, si, at] = actor.value_updates
                snapshots.append(witness)
            save(0)
            for ti, example in enumerate(schedule):
                vector = data["train_catalog"][example]
                target = int(data["probe_targets"][0, example])
                p = float(actor.probability(vector))
                action = int(uniforms[ti] < p)
                reward = (1 if target == 1 else -1) if action == 1 else 0
                head_step, internal_step, value_step = actor.update(vector, action, reward)
                at = (ci, si, ti)
                for key, value in (("trial_probabilities", p), ("trial_actions", action), ("trial_rewards", reward),
                                   ("trial_head_steps", head_step), ("trial_internal_steps", internal_step), ("trial_value_steps", value_step),
                                   ("trial_head_update_calls", actor.head_updates), ("trial_internal_update_calls", actor.internal_updates),
                                   ("trial_value_update_calls", actor.value_updates)):
                    arrays[key][at] = value
                rows.append({"condition": condition, "seed": seed, "trial": ti + 1, "example_index": int(example),
                             "target": target, "probability_go": p, "action": action, "reward": reward,
                             "head_update_calls": actor.head_updates, "internal_update_calls": actor.internal_updates,
                             "value_update_calls": actor.value_updates,
                             "head_step_l2": head_step, "internal_step_l2": internal_step, "value_step_l2": value_step,
                             "head_nonzero_step": int(head_step > 0), "internal_nonzero_step": int(internal_step > 0),
                             "value_nonzero_step": int(value_step > 0)})
                if ti + 1 in checkpoints: save(ti + 1)
            for field in snapshots[0]:
                witnesses.setdefault(f"w_{condition}_{field}", []).append(np.stack([w[field] for w in snapshots]))
    arrays.update({key: np.stack(values) for key, values in witnesses.items()})
    return arrays, rows


def shortcut_summary():
    tables = []
    training = {s: oracle(s) for s in SEQUENCES[0]}
    for name, group in zip(SETS, SEQUENCES):
        target = np.asarray([oracle(s) for s in group])
        cues = {"last_only": [s[-1] for s in group], "first_only": [s[0] for s in group],
                "frequency_only": [tuple(sorted(Counter(s).items())) for s in group]}
        cue_audits = {}
        for label, values in cues.items():
            grouped = {}
            for cue, y in zip(values, target): grouped.setdefault(repr(cue), [0, 0])[int(y)] += 1
            cue_audits[label] = {"label_counts": grouped, "best_possible_accuracy": sum(max(c) for c in grouped.values()) / 4}
        first_last = [int(s[0] != s[-1]) for s in group]
        copy_last = [s[-1] % 2 for s in group]
        lookup = [float(training[s]) if s in training else .5 for s in group]
        tables.append({"set": name, "oracle_accuracy": 1.,
                       "copy_last_accuracy": sum(p == t for p, t in zip(copy_last, target)) / 4,
                       "first_last_accuracy": sum(p == t for p, t in zip(first_last, target)) / 4,
                       "exact_lookup_expected_accuracy": sum(p if t else 1 - p for p, t in zip(lookup, target)) / 4,
                       "seen_training_inputs": sum(s in training for s in group), "cue_audits": cue_audits})
    return {"unique_strings": 16, "evaluation_slots": 20, "phase_novel_labels": [0, 0],
            "phase_is_balanced_generalization_test": False, "new_alphabet_combined_overlap": 2, "sets": tables}


def expected_summary(cfg, arrays):
    p, target = arrays["checkpoint_probabilities"], arrays["probe_targets"]
    correct = np.where(target[None, None, None] == 1, p, 1 - p)
    families = []
    for ci, condition in enumerate(CONDITIONS):
        seed_records = []
        for si, seed in enumerate(cfg["model_seeds"]):
            before = float(np.mean(correct[ci, si, 0, 0]))
            final = correct[ci, si, -1]
            train_mean = float(np.mean(final[0]))
            passed = (train_mean >= cfg["acquisition_threshold"] and train_mean - before >= cfg["acquisition_gain"]
                      and np.mean(final[0, :2]) >= cfg["class_threshold"] and np.mean(final[0, 2:]) >= cfg["class_threshold"])
            groups = []
            for gi, name in enumerate(SETS):
                mean, class_one, class_zero = float(np.mean(final[gi])), float(np.mean(final[gi, :2])), float(np.mean(final[gi, 2:]))
                raw_pass = (mean >= cfg["ood_threshold"] and train_mean - mean <= cfg["ood_max_drop"]
                            and class_one >= cfg["class_threshold"] and class_zero >= cfg["class_threshold"])
                groups.append({"set": name, "expected_accuracy": mean, "class_1_expected_accuracy": class_one,
                               "class_0_expected_accuracy": class_zero,
                               "greedy_accuracy": float(np.mean((p[ci, si, -1, gi] >= .5) == target[gi])),
                               "train_minus_set": train_mean - mean,
                               "ood_criterion_before_acquisition_gate": bool(raw_pass) if gi in (2, 3, 4) else None,
                               "qualified_ood_pass": bool(passed and raw_pass) if gi in (2, 3, 4) else None})
            seed_records.append({"seed": seed, "acquisition_pass": bool(passed), "initial_train_expected_accuracy": before,
                                 "train_gain": train_mean - before, "sets": groups,
                                 "phase_anchors_expected_accuracy": float(np.mean(final[1, :2])),
                                 "phase_novel_single_class_expected_accuracy": float(np.mean(final[1, 2:])),
                                 "phase_novel_only_is_not_generalization_test": True})
        acquired = sum(r["acquisition_pass"] for r in seed_records)
        acquisition_ok = acquired >= cfg["population_pass_count"]
        reports = []
        for gi in (2, 3, 4):
            joint = sum(r["sets"][gi]["qualified_ood_pass"] for r in seed_records)
            reports.append({"set": SETS[gi], "joint_acquisition_and_ood_pass_seeds": joint,
                            "supported": bool(acquisition_ok and joint >= cfg["population_pass_count"]),
                            "status": "not_assessable_without_acquisition" if not acquisition_ok else
                            "supported" if joint >= cfg["population_pass_count"] else "not_supported"})
        families.append({"condition": condition, "role": "primary" if condition == cfg["primary_condition"] else "descriptive_control",
                         "acquisition_pass_seeds": acquired, "acquisition_supported": bool(acquisition_ok),
                         "ood": reports, "seeds": seed_records})
    primary = families[CONDITIONS.index(cfg["primary_condition"])]
    trials = len(CONDITIONS) * len(cfg["model_seeds"]) * cfg["training_trials"]
    return {"status": "completed", "study_id": cfg["study_id"], "families": families,
            "H_acquisition": {"condition": cfg["primary_condition"], "supported": primary["acquisition_supported"]},
            "H_generalization": primary["ood"], "shortcuts": shortcut_summary(),
            "counts": {"training_feedback_trials": trials, "static_probe_cases": int(np.prod(p.shape)),
                       "model_initializations": len(CONDITIONS) * len(cfg["model_seeds"]),
                       "model_seeds": len(cfg["model_seeds"]), "evaluation_slots_per_checkpoint": 20,
                       "unique_probe_strings": 16, "head_update_calls": int(arrays["trial_head_update_calls"][:, :, -1].sum()),
                       "internal_update_calls": int(arrays["trial_internal_update_calls"][:, :, -1].sum()),
                       "value_update_calls": int(arrays["trial_value_update_calls"][:, :, -1].sum()),
                       "head_nonzero_steps": int(np.count_nonzero(arrays["trial_head_steps"])),
                       "internal_nonzero_steps": int(np.count_nonzero(arrays["trial_internal_steps"])),
                       "value_nonzero_steps": int(np.count_nonzero(arrays["trial_value_steps"]))},
            "recording": {"all_initial_final_checkpoints_recorded": True, "probe_activity_is_not_persistent_memory": True,
                          "full_trial_rows_recorded": True, "head_internal_activity_separate": True},
            "interpretation": {"artificial_history_buffer": True, "symbolic_numeric_geometry_supplied": True,
                               "new_human_experiment": False, "shared_neural_mechanism_claim": False,
                               "education_effect_claim": False, "general_rule_understanding_claim": False,
                               "phase_only_balanced_generalization_claim": False, "prior_failure_or_seed_discarded": False}}


def verify_against_graph(cfg, graph, arrays, rows, summary, allow_fixture=False):
    validate_science(cfg, allow_fixture)
    expected, ledger = expected_artifacts(cfg, graph)
    core.demand(set(arrays) == set(expected), "Pattern numeric artifact inventory differs")
    integers = {"probe_strings", "probe_lengths", "probe_targets", "probe_seen_in_training", "train_strings", "train_lengths",
                "model_seeds", "checkpoints", "schedules", "trial_actions", "trial_rewards", "trial_head_update_calls",
                "trial_internal_update_calls", "trial_value_update_calls", "checkpoint_head_calls", "checkpoint_internal_calls", "checkpoint_value_calls"}
    for key, value in expected.items():
        observed = np.asarray(arrays[key])
        core.demand(observed.dtype.kind in "biuf" and np.isfinite(observed).all(), "Finite numeric pattern artifact: " + key)
        if key in integers or key.endswith("_pre") or key.endswith("_post"):
            core.demand(observed.dtype.kind in "biu" and np.array_equal(observed, value), "Exact integer pattern artifact: " + key)
        else:
            core.array_close(observed, value, "Independent pattern replay: " + key, atol=1e-9)
    core.demand(len(rows) == len(ledger), "Complete pattern training ledger required")
    for i, (actual, expected_row) in enumerate(zip(rows, ledger)):
        if all(type(value) is str for value in actual.values()): core.check_csv_row(actual, expected_row, f"Pattern trial {i}")
        else: core.compare_tree(actual, expected_row, f"Pattern trial {i}")
    # Every probability and state has already been independently replayed. Use
    # submitted verified probabilities for exact threshold/tie conventions.
    core.compare_tree(summary, expected_summary(cfg, arrays), "All pattern scientific claims")
    exact = analytical_catalog()
    core.demand(exact["unique_strings"] == 16 and exact["alphabet_combined_overlap"] == 2,
                "Finite catalog overlap semantics")
    core.demand(all(cell["accuracy"] == .5 for report in exact["by_set"].values()
                    for cell in report["single_cue_optimal"].values()), "Single-cue label balance failed")
    return {"arrays": len(expected), "independently_replayed_training_actions": len(ledger),
            "model_seeds": len(cfg["model_seeds"]), "conditions": len(CONDITIONS),
            "checkpoint_static_probes": int(np.prod(arrays["checkpoint_probabilities"].shape)),
            "head_update_calls": int(arrays["trial_head_update_calls"][:, :, -1].sum()),
            "internal_update_calls": int(arrays["trial_internal_update_calls"][:, :, -1].sum()),
            "value_update_calls": int(arrays["trial_value_update_calls"][:, :, -1].sum()),
            "unique_probe_strings": 16, "evaluation_slots_per_checkpoint": 20,
            "independent_exact_catalog_and_shortcuts": True,
            "all_checkpoints_head_internal_and_probe_states_checked": True,
            "training_only_feature_normalization_checked": True,
            "no_hidden_nogo_outcome_updates": True,
            "acquisition_and_joint_ood_gates_separate": True,
            "phase_novel_single_class_not_generalization": True}


def verify_science(cfg, root, arrays, rows, summary, cache=None):
    validate_science(cfg)
    input_sha = verify_input_catalog(cfg, root)
    graph = core.raw_graph(Path(root) / cfg["graph"])
    core.demand((graph["n_kc"], graph["n_mbon"], len(graph["pre"])) == (512, 48, 1474), "Pinned anatomical scope")
    result = verify_against_graph(cfg, graph, arrays, rows, summary)
    result["frozen_input_catalog_sha256"] = input_sha
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve() / "src"))
    from drosolearn.prospective import verify_envelope, sha, write_new
    cfg, reg, meta = verify_envelope(args.root, args.config, args.registration, args.result_dir, args.cache)
    core.demand(meta["study_id"] == cfg["study_id"] and meta["stage"] == "completed", "Completed pattern metadata identity")
    core.demand(all(meta.get(key) is False for key in ("new_human_experiment", "shared_neural_mechanism_tested", "education_effect_tested")),
                "Pattern metadata interpretation boundary")
    stamps = [datetime.fromisoformat(value) for value in (reg["registered_at_utc"], meta["started_at_utc"], meta["completed_at_utc"])]
    core.demand(all(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0 for stamp in stamps)
                and stamps[0] < stamps[1] <= stamps[2] <= datetime.now(timezone.utc), "Prospective pattern UTC chronology")
    with np.load(args.result_dir / "arrays.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    with (args.result_dir / "measurements.csv").open(encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        core.demand(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), "Pattern CSV unique columns")
        rows = list(reader)
    summary = core.read_json(args.result_dir / "summary.json")
    checks = verify_science(cfg, args.root, arrays, rows, summary, args.cache)
    core.demand(meta.get("counts", {}) == summary.get("counts", {}), "Pattern metadata counts binding")
    report = {"status": "passed", "study_id": cfg["study_id"], "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "registration_sha256": sha(args.registration), "summary_sha256": sha(args.result_dir / "summary.json"),
              "scientific_producer_imported": False, "counts": checks}
    write_new(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__": main()
