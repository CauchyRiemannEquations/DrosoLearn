"""Bounded artificial-history pattern discrimination, not biological sequence memory."""
from __future__ import annotations

import copy
import json
import numpy as np

from .graph import load_projection
from .human_shell_models import build_graphs, logistic, validate_observation
from .model import Learner


CONDITIONS = ("fixed", "plastic", "rewired_fixed", "rewired_plastic",
              "ann_fixed", "ann_plastic", "linear", "uniform", "lookup_q")
SET_ORDER = ("acquisition", "phase_diagnostic", "new_alphabet", "new_alphabet_phase", "new_length")
OOD = (2, 3, 4)
SCIENCE = {"study_id": "pattern_rule_v1", "scope": "prospectively_registered_artificial_pattern",
           "conditions": list(CONDITIONS), "model_seeds": list(range(15101, 15111)),
           "training_trials": 400, "checkpoints": [0, 50, 100, 200, 400],
           "head_lr": .1, "internal_lr_multiplier": .1, "rng_namespace": 202610191,
           "lookup_alpha": .1, "lookup_beta": 4.,
           "primary_condition": "plastic", "set_order": list(SET_ORDER),
           "acquisition_threshold": .8, "acquisition_gain": .25, "class_threshold": .7,
           "ood_threshold": .75, "ood_max_drop": .1, "population_pass_count": 8,
           "normalization": "initial_training_four_input_feature_gap_RMS",
           "input_catalog": "data/synthetic/pattern_rule_v1/input_catalog.json",
           "representation": "six_raw_symbol_slots_divided_by_3_then_length_divided_by_6",
           "phase_diagnostic_is_generalization_test": False}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_config(cfg):
    fixture = cfg.get("scope") == "integration_test"
    overrides = {"scope", "model_seeds", "training_trials", "checkpoints", "population_pass_count"} if fixture else set()
    for key, value in SCIENCE.items():
        if key not in overrides:
            require(cfg.get(key) == value, "Fixed pattern science differs: " + key)
    if fixture:
        seeds, n, checkpoints = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
        require(isinstance(seeds, list) and 1 <= len(seeds) <= 2
                and len(seeds) == len(set(seeds)) and all(type(s) is int and 90001 <= s <= 90100 for s in seeds),
                "Synthetic fixture seeds only")
        require(type(n) is int and 4 <= n <= 16 and n % 4 == 0, "Small synthetic fixture budget only")
        require(checkpoints == sorted(set(checkpoints)) and checkpoints[0] == 0 and checkpoints[-1] == n
                and all(type(t) is int and t % 4 == 0 and 0 <= t <= n for t in checkpoints),
                "Synthetic fixture checkpoints only")
        require(cfg["population_pass_count"] == len(seeds), "Fixture pass count follows synthetic seed count")


def encode(symbols):
    values = np.asarray(symbols)
    require(values.ndim == 1 and len(values) in (4, 6)
            and np.issubdtype(values.dtype, np.integer) and np.isin(values, (0, 1, 2, 3)).all(),
            "Four or six artificial symbols 0..3 required")
    result = np.zeros(7)
    result[:len(values)] = values / 3.
    result[6] = len(values) / 6.
    return result


def catalog():
    """Pure finite specification: no learner, RNG, human data or performance fit."""
    raw = [
        [(0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 1, 0), (1, 0, 0, 1)],
        [(0, 1, 0, 1), (1, 0, 1, 0), (0, 0, 1, 1), (1, 1, 0, 0)],
        [(2, 3, 2, 3), (3, 2, 3, 2), (2, 3, 3, 2), (3, 2, 2, 3)],
        [(2, 3, 2, 3), (3, 2, 3, 2), (2, 2, 3, 3), (3, 3, 2, 2)],
        [(0, 1, 0, 1, 0, 1), (1, 0, 1, 0, 1, 0), (0, 1, 1, 0, 0, 1), (1, 0, 0, 1, 1, 0)],
    ]
    lengths = np.asarray([[len(x) for x in group] for group in raw], dtype=np.int8)
    strings = np.full((5, 4, 6), -1, dtype=np.int8)
    for si, group in enumerate(raw):
        for ei, symbols in enumerate(group):
            strings[si, ei, :len(symbols)] = symbols
    return {"probe_strings": strings, "probe_lengths": lengths,
            "probe_inputs": np.asarray([[encode(x) for x in group] for group in raw]),
            "probe_targets": np.tile(np.array([1, 1, 0, 0], dtype=np.int8), (5, 1)),
            "probe_seen_in_training": np.asarray([[x in raw[0] for x in group] for group in raw]),
            "train_strings": strings[0].copy(), "train_lengths": lengths[0].copy(),
            "train_catalog": np.asarray([encode(x) for x in raw[0]])}


def load_catalog(cfg, root):
    """Read the prospectively hashed input file and validate its exact finite contract."""
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "Duplicate input catalog JSON key")
            out[key] = value
        return out
    path = root / cfg["input_catalog"]
    raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                     parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite input catalog")))
    expected = catalog()
    require(set(raw) == {"study_id", "set_order", "catalog"}
            and raw["study_id"] == "pattern_rule_v1" and raw["set_order"] == list(SET_ORDER),
            "Fixed input catalog identity required")
    require(set(raw["catalog"]) == set(expected), "Input catalog array inventory differs")
    result = {}
    for key, wanted in expected.items():
        value = np.asarray(raw["catalog"][key])
        require(value.shape == wanted.shape and value.dtype.kind == wanted.dtype.kind
                and np.array_equal(value, wanted), "Input catalog differs: " + key)
        result[key] = np.asarray(value, dtype=wanted.dtype)
    return result


def action_inputs(vectors):
    vectors = np.asarray(vectors, dtype=float)
    require(vectors.ndim == 2 and vectors.shape[1] == 7 and np.isfinite(vectors).all(), "Finite seven-feature catalog required")
    paired = np.zeros((len(vectors), 2, 7))
    paired[:, 1] = vectors
    return paired


class VectorActor(Learner):
    """The frozen shell reward rule with a new explicit seven-feature interface."""
    def __init__(self, graph, seed, rate, plastic, reference_inputs, probe_inputs):
        super().__init__(graph, seed + 100000, plastic=plastic, learning=True,
                         head_lr=rate, internal_lr=.1 * rate)
        self.head[:] = 0.
        self.pre, self.post = graph.pre, graph.post
        self.probe_inputs = action_inputs(probe_inputs)
        initial = Learner.mbon(self, Learner.kc(self, action_inputs(reference_inputs)))
        self.feature_scale = float(np.sqrt(np.mean(np.sum((initial[:, 1] - initial[:, 0]) ** 2, axis=-1))))
        require(np.isfinite(self.feature_scale) and self.feature_scale > 0, "Positive training-only feature RMS required")
        self._kc_cache, self._fixed_cache = {}, {}
        self.value_updates = 0

    def _parts(self, vector):
        vector = np.asarray(vector, float)
        require(vector.shape == (7,) and np.isfinite(vector).all(), "Finite seven-feature observation required")
        key = tuple(vector)
        if key not in self._kc_cache:
            self._kc_cache[key] = Learner.kc(self, action_inputs(vector[None])[0])
        kc = self._kc_cache[key]
        if self.plastic:
            mb = Learner.mbon(self, kc)
        else:
            if key not in self._fixed_cache:
                self._fixed_cache[key] = Learner.mbon(self, kc)
            mb = self._fixed_cache[key]
        return kc, mb / self.feature_scale

    def raw_gap(self, vector):
        _, features = self._parts(vector)
        return float((features[1] - features[0]) @ self.head)

    def probability(self, vector):
        return logistic(self.raw_gap(vector))

    def observe(self, vector, action, reward):
        validate_observation(action, reward)
        kc, features = self._parts(vector)
        p = logistic((features[1] - features[0]) @ self.head)
        factor = (action - p) * reward
        old_head = self.head.copy()
        delta = self.head_lr * factor * (features[1] - features[0])
        self.head += delta
        self.head_updates += 1
        internal_step = 0.
        if self.plastic:
            old = self.weights.copy()
            eligibility = old_head[self.post] * (kc[1, self.pre] - kc[0, self.pre]) / self.feature_scale
            self.weights = np.clip(self.weights + self.internal_lr * factor * eligibility, 0, self.upper)
            internal_step = float(np.linalg.norm(self.weights - old))
            self.internal_updates += 1
        return float(np.linalg.norm(self.head - old_head)), internal_step, 0.

    def witness(self):
        kc = Learner.kc(self, self.probe_inputs)
        mb = Learner.mbon(self, kc)
        return {"head": self.head.copy(), "weights": self.weights.copy(), "encoder": self.encoder.copy(),
                "bias": self.bias.copy(), "pre": self.pre.copy(), "post": self.post.copy(),
                "initial_weights": self.initial_weights.copy(), "upper": self.upper.copy(),
                "feature_scale": np.asarray(self.feature_scale), "probe_kc": kc,
                "probe_mbon": mb, "probe_decision": mb / self.feature_scale,
                "value_table": np.empty(0), "value_catalog": np.empty((0, 7))}


class VectorSimple:
    def __init__(self, condition, rate, reference_inputs, probe_inputs):
        require(condition in ("linear", "uniform", "lookup_q"), "Simple pattern condition required")
        self.condition, self.rate = condition, rate
        self.head = np.zeros(7) if condition == "linear" else np.empty(0)
        self.weights, self.encoder, self.bias = np.empty(0), np.empty((0, 7)), np.empty(0)
        self.probe_inputs = action_inputs(probe_inputs)
        self.feature_scale = float(np.sqrt(np.mean(np.sum(np.asarray(reference_inputs) ** 2, axis=-1)))) if condition == "linear" else 1.
        require(self.feature_scale > 0, "Positive training-only linear RMS required")
        self.head_updates = self.internal_updates = self.value_updates = 0
        self.value_catalog = np.array(reference_inputs, copy=True) if condition == "lookup_q" else np.empty((0, 7))
        self.values = np.zeros((len(self.value_catalog), 2)) if condition == "lookup_q" else np.empty(0)
        self.lookup = {tuple(row): index for index, row in enumerate(self.value_catalog)}
        require(len(self.lookup) == len(self.value_catalog), "Unique exact-input lookup catalog required")

    def raw_gap(self, vector):
        vector = np.asarray(vector, float)
        require(vector.shape == (7,) and np.isfinite(vector).all(), "Finite seven-feature observation required")
        if self.condition == "uniform":
            return 0.
        if self.condition == "lookup_q":
            index = self.lookup.get(tuple(vector))
            return 0. if index is None else float(4. * (self.values[index, 1] - self.values[index, 0]))
        return float((vector / self.feature_scale) @ self.head)

    def probability(self, vector):
        return logistic(self.raw_gap(vector))

    def observe(self, vector, action, reward):
        validate_observation(action, reward)
        if self.condition == "uniform":
            return 0., 0., 0.
        if self.condition == "lookup_q":
            index = self.lookup.get(tuple(np.asarray(vector, float)))
            require(index is not None, "Cannot learn an unregistered exact-input lookup identity")
            old = float(self.values[index, action])
            self.values[index, action] += self.rate * (reward - old)
            self.value_updates += 1
            return 0., 0., abs(float(self.values[index, action]) - old)
        p = self.probability(vector)
        old = self.head.copy()
        delta = self.rate * (action - p) * reward * np.asarray(vector) / self.feature_scale
        self.head += delta
        self.head_updates += 1
        return float(np.linalg.norm(self.head - old)), 0., 0.

    def witness(self):
        empty = np.empty((len(self.probe_inputs), 2, 0))
        return {"head": self.head.copy(), "weights": self.weights.copy(), "encoder": self.encoder.copy(),
                "bias": self.bias.copy(), "pre": np.empty(0, dtype=int), "post": np.empty(0, dtype=int),
                "initial_weights": np.empty(0), "upper": np.empty(0), "feature_scale": np.asarray(self.feature_scale),
                "probe_kc": empty.copy(), "probe_mbon": empty.copy(),
                "probe_decision": self.probe_inputs / self.feature_scale if self.condition == "linear" else empty.copy(),
                "value_table": self.values.copy(), "value_catalog": self.value_catalog.copy()}


def make_actor(condition, graphs, seed, rate, reference_inputs, probe_inputs):
    require(condition in CONDITIONS, "Known pattern condition required")
    if condition in CONDITIONS[:6]:
        kind = "real" if condition in ("fixed", "plastic") else condition.split("_")[0]
        return VectorActor(graphs[kind], seed, rate, condition.endswith("plastic"), reference_inputs, probe_inputs)
    return VectorSimple(condition, rate, reference_inputs, probe_inputs)


def shortcut_report(data):
    """Exact finite-family diagnostics, not reward-trained models."""
    raw = [[tuple(int(v) for v in data["probe_strings"][si, ei, :int(data["probe_lengths"][si, ei])])
            for ei in range(4)] for si in range(5)]
    training = dict(zip(raw[0], [1, 1, 0, 0]))
    reports = []
    for si, name in enumerate(SET_ORDER):
        target = data["probe_targets"][si]
        oracle = np.array([int(all(a != b for a, b in zip(x, x[1:]))) for x in raw[si]])
        first_last = np.array([int(x[0] != x[-1]) for x in raw[si]])
        copy_last = np.array([x[-1] % 2 for x in raw[si]])
        lookup = np.array([float(training[x]) if x in training else .5 for x in raw[si]])
        cues = {"last_only": [x[-1] for x in raw[si]], "first_only": [x[0] for x in raw[si]],
                "frequency_only": [tuple((v, x.count(v)) for v in sorted(set(x))) for x in raw[si]]}
        cue_reports = {}
        for cue_name, values in cues.items():
            groups = {}
            for cue, y in zip(values, target):
                groups.setdefault(repr(cue), [0, 0])[int(y)] += 1
            cue_reports[cue_name] = {"label_counts": groups,
                                     "best_possible_accuracy": sum(max(c) for c in groups.values()) / 4}
        reports.append({"set": name, "oracle_accuracy": float(np.mean(oracle == target)),
                        "copy_last_accuracy": float(np.mean(copy_last == target)),
                        "first_last_accuracy": float(np.mean(first_last == target)),
                        "exact_lookup_expected_accuracy": float(np.mean(np.where(target == 1, lookup, 1 - lookup))),
                        "seen_training_inputs": int(data["probe_seen_in_training"][si].sum()),
                        "cue_audits": cue_reports})
    return {"unique_strings": len({x for group in raw for x in group}), "evaluation_slots": 20,
            "phase_novel_labels": [0, 0], "phase_is_balanced_generalization_test": False,
            "new_alphabet_combined_overlap": len(set(raw[2]) & set(raw[3])), "sets": reports}


def produce(cfg, root, cache=None):
    validate_config(cfg)
    data = load_catalog(cfg, root)
    seeds, n, checkpoints = cfg["model_seeds"], cfg["training_trials"], cfg["checkpoints"]
    graph = load_projection(root / cfg["graph"])
    shape = (len(CONDITIONS), len(seeds), n)
    arrays = {key: np.empty(shape, dtype=np.int8 if key in ("trial_actions", "trial_rewards") else float)
              for key in ("trial_probabilities", "trial_actions", "trial_rewards", "trial_head_steps", "trial_internal_steps", "trial_value_steps")}
    for key in ("trial_head_update_calls", "trial_internal_update_calls", "trial_value_update_calls"):
        arrays[key] = np.zeros(shape, dtype=np.int32)
    arrays.update(data)
    arrays["checkpoints"] = np.asarray(checkpoints, dtype=np.int32)
    arrays["model_seeds"] = np.asarray(seeds, dtype=np.int64)
    arrays["schedules"] = np.empty((len(seeds), n), dtype=np.int8)
    arrays["uniforms"] = np.empty((len(seeds), n))
    arrays["checkpoint_probabilities"] = np.empty((len(CONDITIONS), len(seeds), len(checkpoints), 5, 4))
    arrays["checkpoint_logits"] = np.empty_like(arrays["checkpoint_probabilities"])
    for key in ("checkpoint_head_calls", "checkpoint_internal_calls", "checkpoint_value_calls"):
        arrays[key] = np.empty((len(CONDITIONS), len(seeds), len(checkpoints)), dtype=np.int32)
    rows, witnesses = [], {}
    checkpoint_lookup = {t: i for i, t in enumerate(checkpoints)}
    probes = data["probe_inputs"].reshape(20, 7)
    for si, seed in enumerate(seeds):
        order_rng = np.random.default_rng(np.random.SeedSequence([cfg["rng_namespace"], seed, 0]))
        schedule = np.concatenate([order_rng.permutation(4) for _ in range(n // 4)])
        uniforms = np.random.default_rng(np.random.SeedSequence([cfg["rng_namespace"], seed, 1])).random(n)
        arrays["schedules"][si], arrays["uniforms"][si] = schedule, uniforms
        graphs = build_graphs(graph, seed)
        for ci, condition in enumerate(CONDITIONS):
            actor = make_actor(condition, graphs, seed, cfg["head_lr"], data["train_catalog"], probes)
            snapshots = []

            def checkpoint():
                at = checkpoint_lookup[t]
                arrays["checkpoint_probabilities"][ci, si, at] = np.array([actor.probability(x) for x in probes]).reshape(5, 4)
                arrays["checkpoint_logits"][ci, si, at] = np.array([actor.raw_gap(x) for x in probes]).reshape(5, 4)
                arrays["checkpoint_head_calls"][ci, si, at] = actor.head_updates
                arrays["checkpoint_internal_calls"][ci, si, at] = actor.internal_updates
                arrays["checkpoint_value_calls"][ci, si, at] = actor.value_updates
                snapshots.append(actor.witness())

            t = 0
            checkpoint()
            for ti, example in enumerate(schedule):
                vector, target = data["train_catalog"][example], int(data["probe_targets"][0, example])
                p = actor.probability(vector)
                action = int(uniforms[ti] < p)
                reward = (1 if target else -1) if action else 0
                head_step, internal_step, value_step = actor.observe(vector, action, reward)
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
                t = ti + 1
                if t in checkpoint_lookup:
                    checkpoint()
            for field in snapshots[0]:
                witnesses.setdefault(f"w_{condition}_{field}", []).append(np.stack([snapshot[field] for snapshot in snapshots]))
    arrays.update({name: np.stack(values) for name, values in witnesses.items()})
    return arrays, rows


def summarize(cfg, arrays, rows):
    validate_config(cfg)
    p = arrays["checkpoint_probabilities"]
    target = arrays["probe_targets"]
    correct = np.where(target[None, None, None] == 1, p, 1 - p)
    expected_shape = (len(CONDITIONS), len(cfg["model_seeds"]), len(cfg["checkpoints"]), 5, 4)
    require(p.shape == expected_shape and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "Aligned finite checkpoint probabilities required")
    final, initial = correct[:, :, -1], correct[:, :, 0]
    acquisition = ((final[:, :, 0].mean(axis=-1) >= cfg["acquisition_threshold"])
                   & (final[:, :, 0].mean(axis=-1) - initial[:, :, 0].mean(axis=-1) >= cfg["acquisition_gain"])
                   & (final[:, :, 0, :2].mean(axis=-1) >= cfg["class_threshold"])
                   & (final[:, :, 0, 2:].mean(axis=-1) >= cfg["class_threshold"]))
    families = []
    for ci, condition in enumerate(CONDITIONS):
        seed_rows = []
        for si, seed in enumerate(cfg["model_seeds"]):
            cells = []
            train_mean = float(final[ci, si, 0].mean())
            for set_i, name in enumerate(SET_ORDER):
                mean = float(final[ci, si, set_i].mean())
                raw_criterion = (mean >= cfg["ood_threshold"] and train_mean - mean <= cfg["ood_max_drop"]
                                 and float(final[ci, si, set_i, :2].mean()) >= cfg["class_threshold"]
                                 and float(final[ci, si, set_i, 2:].mean()) >= cfg["class_threshold"])
                cells.append({"set": name, "expected_accuracy": mean,
                              "class_1_expected_accuracy": float(final[ci, si, set_i, :2].mean()),
                              "class_0_expected_accuracy": float(final[ci, si, set_i, 2:].mean()),
                              "greedy_accuracy": float(np.mean((p[ci, si, -1, set_i] >= .5) == target[set_i])),
                              "train_minus_set": train_mean - mean,
                              "ood_criterion_before_acquisition_gate": bool(raw_criterion) if set_i in OOD else None,
                              "qualified_ood_pass": bool(acquisition[ci, si] and raw_criterion) if set_i in OOD else None})
            seed_rows.append({"seed": seed, "acquisition_pass": bool(acquisition[ci, si]),
                              "initial_train_expected_accuracy": float(initial[ci, si, 0].mean()),
                              "train_gain": train_mean - float(initial[ci, si, 0].mean()), "sets": cells,
                              "phase_anchors_expected_accuracy": float(final[ci, si, 1, :2].mean()),
                              "phase_novel_single_class_expected_accuracy": float(final[ci, si, 1, 2:].mean()),
                              "phase_novel_only_is_not_generalization_test": True})
        acquired = int(acquisition[ci].sum())
        acquired_supported = acquired >= cfg["population_pass_count"]
        ood_reports = []
        for set_i in OOD:
            passed = sum(row["sets"][set_i]["qualified_ood_pass"] for row in seed_rows)
            ood_reports.append({"set": SET_ORDER[set_i], "joint_acquisition_and_ood_pass_seeds": passed,
                                "supported": bool(acquired_supported and passed >= cfg["population_pass_count"]),
                                "status": "not_assessable_without_acquisition" if not acquired_supported else
                                "supported" if passed >= cfg["population_pass_count"] else "not_supported"})
        families.append({"condition": condition, "role": "primary" if condition == cfg["primary_condition"] else "descriptive_control",
                         "acquisition_pass_seeds": acquired, "acquisition_supported": acquired_supported,
                         "ood": ood_reports, "seeds": seed_rows})
    require(len(rows) == len(CONDITIONS) * len(cfg["model_seeds"]) * cfg["training_trials"], "Full trial rows required")
    primary = families[CONDITIONS.index(cfg["primary_condition"])]
    return {"status": "completed", "study_id": cfg["study_id"], "families": families,
            "H_acquisition": {"condition": cfg["primary_condition"], "supported": primary["acquisition_supported"]},
            "H_generalization": copy.deepcopy(primary["ood"]), "shortcuts": shortcut_report(arrays),
            "counts": {"training_feedback_trials": len(rows), "static_probe_cases": int(np.prod(p.shape)),
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
                               "phase_only_balanced_generalization_claim": False,
                               "prior_failure_or_seed_discarded": False}}
