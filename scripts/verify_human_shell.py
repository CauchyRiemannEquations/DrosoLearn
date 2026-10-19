"""Independent arithmetic for the registered shell-game comparison.

Only stdlib and NumPy are imported.  No producer, learner, or source-parser
module is used by this verifier.  Scientific execution requires a frozen
registration and an already completed producer result.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np


SEASONS = {1: ("R", "N"), 2: ("N", "R"), 3: ("R", "P"),
           4: ("P", "R"), 5: ("P", "N"), 6: ("N", "P")}
CONDITIONS = ("fixed", "plastic", "rewired_fixed", "rewired_plastic",
              "ann_fixed", "ann_plastic", "linear", "q", "uniform", "bias", "persistence")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def scalar_close(actual, expected, description, atol=1e-11):
    demand(np.isfinite(actual) and np.isfinite(expected), f"Nonfinite {description}")
    demand(abs(float(actual) - float(expected)) <= atol,
           f"{description}: {actual!r} != {expected!r}")


def array_close(actual, expected, description, atol=1e-11):
    actual, expected = np.asarray(actual), np.asarray(expected)
    demand(actual.shape == expected.shape, f"Shape mismatch: {description}")
    demand(np.isfinite(actual).all() and np.isfinite(expected).all(), f"Nonfinite {description}")
    demand(np.allclose(actual, expected, rtol=0, atol=atol), f"Numeric mismatch: {description}")


def parse_cohort(archive):
    """Read original ZIP independently; do not calculate behavioral performance."""
    with zipfile.ZipFile(archive) as outer:
        demand(len(outer.namelist()) == len(set(outer.namelist())), "Duplicate outer member")
        metadata = list(csv.DictReader(io.StringIO(outer.read("A_Data.csv").decode("utf-8-sig"))))
        ids = [row["ID"].strip() for row in metadata if row.get("ID", "").strip()]
        demand(ids and len(ids) == len(set(ids)), "Missing/duplicate metadata IDs")
        demand(all(re.fullmatch(r"A\d{4}", pid) for pid in ids), "Invalid source ID")
        nested = outer.read("A_Behav.zip")
    records, excluded = {}, {}
    with zipfile.ZipFile(io.BytesIO(nested)) as inner:
        names = inner.namelist()
        demand(len(names) == len(set(names)), "Duplicate nested member")
        for pid in sorted(ids):
            trace_name = f"behavior/{pid}/GONOGO_trace_{pid}.csv"
            log_name = f"behavior/{pid}/GONOGO_shortlog_{pid}.txt"
            demand(trace_name in names and log_name in names, f"Missing standard pair {pid}")
            raw_trace, raw_log = inner.read(trace_name), inner.read(log_name)
            trace_rows = [re.split(r"[\s,;]+", line.strip()) for line in raw_trace.decode("utf-8-sig").splitlines() if line.strip()]
            log_rows = [line.replace(":", " ").split() for line in raw_log.decode("utf-8-sig").splitlines() if line.strip()]
            if pid == "A0016":
                demand(len(trace_rows) == 216 and len(log_rows) == 120 and len(log_rows[-1]) == 3,
                       "Predeclared A0016 structural exclusion changed")
                excluded[pid] = {"trace_rows": len(trace_rows), "shortlog_rows": len(log_rows),
                                 "last_shortlog_columns": len(log_rows[-1])}
                continue
            demand(all(len(row) == 3 for row in trace_rows), f"Trace columns {pid}")
            demand(all(len(row) == 5 for row in log_rows), f"Shortlog columns {pid}")
            trace, log = np.asarray(trace_rows, dtype=int), np.asarray(log_rows, dtype=int)
            demand(trace.shape == (216, 3) and log.shape == (216, 5), f"Pair length {pid}")
            demand(np.array_equal(log[:, 0], np.arange(1, 217)), f"Trial sequence {pid}")
            demand(np.array_equal(trace[:, 0], log[:, 1]) and np.array_equal(trace[:, 2], log[:, 2]),
                   f"Trace/log pairing {pid}")
            demand(np.isin(trace[:, 0], np.arange(1, 7)).all(), f"Shell range {pid}")
            demand(np.isin(trace[:, 1], (1, 2)).all(), f"State range {pid}")
            demand(np.isin(trace[:, 2], (0, 1, 2)).all(), f"Content range {pid}")
            demand(np.isin(log[:, 3], (0, 1)).all(), f"Action range {pid}")
            demand((log[log[:, 3] == 0, 4] == 0).all() and (log[log[:, 3] == 1, 4] > 0).all(),
                   f"RT/action role {pid}")
            for shell, state, content in trace:
                season = SEASONS[int(shell)][int(state) - 1]
                demand(content in {"R": (0, 1), "P": (0, 2), "N": (0,)}[season],
                       f"Impossible state/content {pid}")
            records[pid] = {"trial": log[:, 0], "shell": trace[:, 0], "state": trace[:, 1],
                            "content": trace[:, 2], "action": log[:, 3], "rt": log[:, 4],
                            "trace_sha256": hashlib.sha256(raw_trace).hexdigest(),
                            "shortlog_sha256": hashlib.sha256(raw_log).hexdigest()}
    demand(len(ids) == 50 and len(records) == 49 and set(excluded) == {"A0016"},
           "Registered source cohort changed")
    order = sorted(records, key=lambda pid: hashlib.sha256(("drosolearn-shell-v1:" + pid).encode()).hexdigest())
    return records, order[:20], order[20:], excluded


def raw_graph(directory):
    """Independent annotations/contacts loader; preserve the original edge order."""
    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    for name, key in (("edges.csv", "edges_sha256"), ("neurons.json", "neurons_sha256"),
                      ("annotations.csv", "roles_sha256")):
        demand(digest(directory / name) == provenance[key], f"Anatomical checksum {name}")
    with (directory / "annotations.csv").open(encoding="utf-8", newline="") as stream:
        annotations = list(csv.DictReader(stream))
    kc = sorted(row["root_id"] for row in annotations if row["role"] == "KC")
    mb = sorted(row["root_id"] for row in annotations if row["role"] == "MBON")
    demand(kc and mb and len(kc) == len(set(kc)) and len(mb) == len(set(mb)), "Anatomical IDs")
    ki, mi = dict(zip(kc, range(len(kc)))), dict(zip(mb, range(len(mb))))
    pre, post, weight = [], [], []
    with (directory / "edges.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["pre"] in ki and row["post"] in mi:
                demand(int(row["sign"]) == 1 and int(row["count"]) > 0, "Nonpositive contact")
                pre.append(ki[row["pre"]]); post.append(mi[row["post"]]); weight.append(float(row["count"]))
    demand(len(set(zip(pre, post))) == len(pre), "Duplicate anatomical edge")
    return {"pre": np.array(pre, dtype=int), "post": np.array(post, dtype=int),
            "counts": np.array(weight), "n_kc": len(kc), "n_mbon": len(mb)}


def transform_graph(graph, kind, seed):
    result = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in graph.items()}
    if kind == "real":
        return result
    rng = np.random.default_rng(seed)
    pre, post = result["pre"], result["post"]
    if kind == "ann":
        for target in range(result["n_mbon"]):
            locations = np.flatnonzero(post == target)
            pre[locations] = rng.choice(result["n_kc"], len(locations), replace=False)
        return result
    demand(kind == "rewired", "Unknown graph control")
    occupied = set(zip(pre, post)); accepted = attempts = 0
    goal = len(pre) * 10
    while accepted < goal and attempts < goal * 100:
        attempts += 1
        a, b = rng.choice(len(pre), 2, replace=False)
        u, v, p, q = int(pre[a]), int(pre[b]), int(post[a]), int(post[b])
        if u == v or p == q or (v, p) in occupied or (u, q) in occupied:
            continue
        occupied.remove((u, p)); occupied.remove((v, q))
        occupied.add((v, p)); occupied.add((u, q)); pre[a], pre[b] = v, u
        accepted += 1
    demand(accepted == goal, "Independent rewiring budget failed")
    demand(len(set(zip(pre, post))) == len(pre), "Rewired duplicate")
    demand(np.array_equal(np.bincount(pre, minlength=result["n_kc"]),
                          np.bincount(graph["pre"], minlength=graph["n_kc"])), "Rewired outdegree")
    demand(np.array_equal(post, graph["post"]), "Rewired incoming contact positions changed")
    return result


def logistic(gap):
    return float(1.0 / (1.0 + np.exp(-np.clip(gap, -40, 40))))


def paired_catalog():
    inputs = np.zeros((6, 2, 7))
    for index in range(6):
        inputs[index, 1, index] = 1.0
    return inputs


class AuditPolicy:
    """Independent numeric implementation; the public API never accepts content/state."""
    def __init__(self, condition, seed, rate, graph=None, bias_probability=.5):
        demand(condition in CONDITIONS, "Unknown policy")
        self.condition, self.rate = condition, rate
        self.internal_rate = rate * .1
        self.head_updates = self.internal_updates = self.value_updates = 0
        self.last = np.full(6, -1, dtype=int)
        self.bias_probability = float(bias_probability)
        self.plastic = condition in ("plastic", "rewired_plastic", "ann_plastic")
        self.scale = 1.0
        self.weights = np.empty(0); self.head = np.empty(0); self.values = np.empty(0)
        self.encoder = np.empty((0, 7)); self.bias = np.empty(0)
        if condition in CONDITIONS[:6]:
            demand(graph is not None, "Network graph required")
            self.graph = graph
            self.pre, self.post = graph["pre"], graph["post"]
            generator = np.random.default_rng(seed + 100000)
            self.encoder = generator.normal(0, .8, (graph["n_kc"], 7))
            self.bias = generator.uniform(-.25, .25, graph["n_kc"])
            self.head = np.zeros(graph["n_mbon"])
            norm = np.maximum(np.bincount(self.post, weights=graph["counts"], minlength=graph["n_mbon"]), 1.)
            self.weights = graph["counts"] / norm[self.post]
            self.initial_weights = self.weights.copy(); self.upper = 2 * self.weights
            features = np.array([self.network_features(s)[1] for s in range(1, 7)])
            self.scale = float(np.sqrt(np.mean(np.sum((features[:, 1] - features[:, 0]) ** 2, axis=-1))))
            demand(self.scale > 0, "Zero network feature scale")
        elif condition == "linear":
            self.head = np.zeros(7)
        elif condition == "q":
            self.values = np.zeros((6, 2))
        self.initial_head = self.head.copy()

    def network_features(self, shell):
        inputs = paired_catalog()[shell - 1]
        kc = np.maximum(inputs @ self.encoder.T + self.bias, 0)
        features = np.array([np.bincount(self.post, weights=self.weights * row[self.pre],
                                        minlength=self.graph["n_mbon"]) for row in kc])
        return kc, features / self.scale

    def probability(self, shell):
        demand(type(shell) is int and 1 <= shell <= 6, "Invalid shell")
        if self.condition == "uniform":
            return .5
        if self.condition == "bias":
            return self.bias_probability
        if self.condition == "persistence":
            return .5 if self.last[shell - 1] == -1 else .9 if self.last[shell - 1] else .1
        if self.condition == "q":
            return logistic(4 * (self.values[shell - 1, 1] - self.values[shell - 1, 0]))
        if self.condition == "linear":
            return logistic(self.head[shell - 1])
        _, features = self.network_features(shell)
        return logistic((features[1] - features[0]) @ self.head)

    def update(self, shell, action, reward):
        demand(type(action) is int and action in (0, 1), "Invalid observed action")
        demand(type(reward) is int and reward in (-1, 0, 1), "Invalid observed reward")
        demand(action == 1 or reward == 0, "NoGo cannot receive the hidden Go outcome")
        before_head, before_weight, before_values = self.head.copy(), self.weights.copy(), self.values.copy()
        probability = self.probability(shell)
        factor = (action - probability) * reward
        if self.condition == "q":
            self.values[shell - 1, action] += self.rate * (reward - self.values[shell - 1, action])
            self.value_updates += 1
        elif self.condition == "linear":
            self.head[shell - 1] += self.rate * factor
            self.head_updates += 1
        elif self.condition in CONDITIONS[:6]:
            kc, features = self.network_features(shell)
            self.head += self.rate * factor * (features[1] - features[0])
            self.head_updates += 1
            if self.plastic:
                eligibility = before_head[self.post] * (kc[1, self.pre] - kc[0, self.pre]) / self.scale
                self.weights = np.clip(self.weights + self.internal_rate * factor * eligibility, 0, self.upper)
                self.internal_updates += 1
        elif self.condition == "persistence":
            self.last[shell - 1] = action
        return (float(np.linalg.norm(self.head - before_head)), float(np.linalg.norm(self.weights - before_weight)),
                float(np.linalg.norm(self.values - before_values)))

    def witness(self):
        empty = np.empty((6, 2, 0))
        if self.condition in CONDITIONS[:6]:
            pieces = [self.network_features(s) for s in range(1, 7)]
            kc = np.array([part[0] for part in pieces])
            decision = np.array([part[1] for part in pieces])
            return {"head": self.head.copy(), "weights": self.weights.copy(), "probe_kc": kc,
                    "probe_mbon": decision * self.scale, "probe_decision": decision,
                    "encoder": self.encoder.copy(), "bias": self.bias.copy(),
                    "pre": self.pre.copy(), "post": self.post.copy(),
                    "initial_weights": self.initial_weights.copy(), "upper": self.upper.copy(),
                    "feature_scale": np.asarray(self.scale), "value_table": self.values.copy()}
        return {"head": self.head.copy(), "weights": self.weights.copy(),
                "probe_kc": empty.copy(), "probe_mbon": empty.copy(),
                "probe_decision": paired_catalog() if self.condition == "linear" else empty.copy(),
                "encoder": self.encoder.copy(), "bias": self.bias.copy(),
                "pre": np.empty(0, int), "post": np.empty(0, int),
                "initial_weights": np.empty(0), "upper": np.empty(0),
                "feature_scale": np.asarray(1.), "value_table": self.values.copy(),
                "last_action_state": self.last.copy()}


def replay(policy, record, mode, seed):
    """Replay past observations in order; only the environment reads hidden content."""
    demand(mode in ("conditional", "autonomous"), "Unknown replay mode")
    total = len(record["shell"])
    arrays = {key: np.empty(total, dtype=float) for key in ("p_go", "action", "reward", "head_step", "internal_step", "value_step")}
    if mode == "autonomous":
        uniforms = np.random.default_rng(seed + 300000 + int(record["id"][1:])).random(total)
    for index in range(total):
        shell = int(record["shell"][index])
        p = policy.probability(shell)
        action = int(record["action"][index]) if mode == "conditional" else int(uniforms[index] < p)
        reward = {0: 0, 1: 1, 2: -1}[int(record["content"][index])] if action else 0
        head_step, internal_step, value_step = policy.update(shell, action, reward)
        arrays["p_go"][index], arrays["action"][index], arrays["reward"][index] = p, action, reward
        arrays["head_step"][index], arrays["internal_step"][index] = head_step, internal_step
        arrays["value_step"][index] = value_step
    return arrays


def score_choices(probabilities, actions):
    p, a = np.asarray(probabilities, float), np.asarray(actions, float)
    demand(p.shape == a.shape and p.size > 0 and np.isfinite(p).all() and np.isin(a, (0, 1)).all(), "Invalid scoring input")
    demand(((p >= 0) & (p <= 1)).all(), "Probability outside unit interval")
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.mean(-a * np.log(clipped) - (1 - a) * np.log1p(-clipped))), float(np.mean((p - a) ** 2))


def behavioral_metrics(record, actions):
    """Equal shell/event weighting on R/P shells, with nonoverlapping windows."""
    actions = np.asarray(actions, dtype=float)
    demand(actions.shape == np.asarray(record["shell"]).shape, "Behavior shape")
    acquisition, early, recovery = [], [], []
    for shell in (3, 4):
        locations = np.flatnonzero(np.asarray(record["shell"]) == shell)
        states = np.asarray(record["state"])[locations]
        boundaries = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
        demand(len(boundaries) == 5, "Exactly three registered shell reversals required")
        observed_states = [int(states[boundaries[i]]) for i in range(4)]
        demand(observed_states == [1, 2, 1, 2], "Registered state sequence changed")
        for phase in range(4):
            chosen = locations[boundaries[phase]:boundaries[phase + 1]]
            demand(len(chosen) >= 6, "Overlapping behavior windows")
            season = SEASONS[shell][observed_states[phase] - 1]
            optimum = int(season == "R")
            value = actions[chosen] if optimum else 1 - actions[chosen]
            first, last = float(np.mean(value[:3])), float(np.mean(value[-3:]))
            if phase == 0:
                acquisition.append(last - first)
            else:
                early.append(first); recovery.append(last - first)
    return np.array([np.mean(acquisition), np.mean(early), np.mean(recovery)], dtype=float)


def resampling_indices(n_people, n_seeds, samples=2000, seed=14001):
    generator = np.random.default_rng(seed)
    people = generator.integers(0, n_people, size=(samples, n_people))
    seeds = generator.integers(0, n_seeds, size=(samples, n_seeds))
    return people, seeds


def crossed_draws(human, model, people_indices, seed_indices):
    human, model = np.asarray(human, dtype=float), np.asarray(model, dtype=float)
    demand(model.ndim == human.ndim + 1 and model.shape[0] == human.shape[0] and model.shape[2:] == human.shape[1:],
           "People/seed tensor mismatch")
    demand(len(people_indices) == len(seed_indices), "Bootstrap axes mismatch")
    differences, human_means, model_means = [], [], []
    for pp, ss in zip(people_indices, seed_indices):
        h = human[pp].mean(axis=0)
        m = model[pp][:, ss].mean(axis=(0, 1))
        differences.append(m - h); human_means.append(h); model_means.append(m)
    return np.array(differences), np.array(human_means), np.array(model_means)


def interval90(draws):
    return np.quantile(np.asarray(draws), (.05, .95), axis=0)


def equivalence_label(interval, margin=.10):
    bounds = np.asarray(interval, dtype=float)
    demand(bounds.ndim == 2 and bounds.shape[0] == 2 and np.isfinite(bounds).all(), "Invalid interval")
    demand((bounds[0] <= bounds[1]).all(), "Reversed interval")
    if ((bounds[0] >= -margin) & (bounds[1] <= margin)).all():
        return "supported"
    if ((bounds[1] < -margin) | (bounds[0] > margin)).any():
        return "different"
    return "inconclusive"


def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            demand(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def contained(root, relative):
    demand(type(relative) is str and relative and "\\" not in relative and ":" not in relative
           and not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Contained POSIX input required")
    path = (Path(root) / relative).resolve()
    demand(path.is_relative_to(Path(root).resolve()) and path.is_file(), "Input outside expected root")
    return path


def compare_tree(actual, expected, description, subset=False):
    if isinstance(expected, dict):
        demand(isinstance(actual, dict), f"Dictionary expected: {description}")
        demand(set(expected) <= set(actual) if subset else set(actual) == set(expected), f"Keys differ: {description}")
        for key, value in expected.items():
            compare_tree(actual[key], value, f"{description}.{key}", subset)
    elif isinstance(expected, (list, tuple)):
        demand(isinstance(actual, list) and len(actual) == len(expected), f"List length: {description}")
        for index, value in enumerate(expected):
            compare_tree(actual[index], value, f"{description}[{index}]", subset)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        demand(type(actual) is type(expected) and actual == expected, f"Value differs: {description}")
    elif isinstance(expected, (int, np.integer)):
        demand(type(actual) is int and actual == expected, f"Integer differs: {description}")
    else:
        scalar_close(actual, expected, description)


def read_measurements(path, fields):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        demand(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate CSV columns")
        result = {}
        for row in reader:
            key = tuple(row[field] for field in fields)
            demand(key not in result, "Duplicate measurement row")
            result[key] = row
    return result


def check_source_audit(archive, manifest, actual):
    """Reconstruct every member hash and the source-only episode summaries."""
    participants, members = {}, {}
    with zipfile.ZipFile(archive) as outer:
        demand(set(outer.namelist()) == {"A_Data.csv", "A_OCIR.csv", "A_Behav.zip"}, "Outer source inventory")
        members["outer"] = {name: {"bytes": len(outer.read(name)), "sha256": hashlib.sha256(outer.read(name)).hexdigest()}
                            for name in sorted(outer.namelist())}
        with zipfile.ZipFile(io.BytesIO(outer.read("A_Behav.zip"))) as inner:
            names = {item.filename for item in inner.infolist() if not item.is_dir()}
            standard = {f"behavior/{pid}/GONOGO_{kind}_{pid}.{suffix}" for pid in manifest["raw_metadata_ids"]
                        for kind, suffix in (("trace", "csv"), ("shortlog", "txt"))}
            demand(names == standard | set(manifest["additional_behavior_members"]), "Nested source inventory")
            members["behavior"] = {name: {"bytes": len(inner.read(name)), "sha256": hashlib.sha256(inner.read(name)).hexdigest()}
                                   for name in sorted(names)}
            for name, expected in manifest["additional_behavior_members"].items():
                demand(members["behavior"][name] == expected, "Additional source member changed")
            for pid in manifest["raw_metadata_ids"]:
                text = inner.read(f"behavior/{pid}/GONOGO_trace_{pid}.csv").decode("utf-8-sig")
                trace = np.array([[int(value) for value in re.split(r"[,\s]+", line.strip())]
                                  for line in text.splitlines() if line.strip()])
                text = inner.read(f"behavior/{pid}/GONOGO_shortlog_{pid}.txt").decode("utf-8-sig")
                log = [[int(value) for value in line.replace(":", " ").split()] for line in text.splitlines() if line.strip()]
                structures = {}
                for shell in range(1, 7):
                    states = trace[trace[:, 0] == shell, 1]
                    bounds = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
                    structures[str(shell)] = {"input_occurrences": len(states),
                                               "state_run_sequence": states[bounds[:-1]].tolist(),
                                               "phase_lengths": np.diff(bounds).tolist()}
                valid_rt = all(len(row) == 5 and row[3] in (0, 1)
                               and (row[4] == 0 if row[3] == 0 else row[4] > 0) for row in log)
                allowed = all(int(content) in {"R": (0, 1), "P": (0, 2), "N": (0,)}[SEASONS[int(shell)][int(state) - 1]]
                              for shell, state, content in trace)
                issues = manifest["expected_exclusions"].get(pid, [])
                participants[pid] = {"trace_trials": len(trace), "shortlog_trials": len(log),
                                     "shortlog_column_counts": sorted({len(row) for row in log}),
                                     "input_output_match_on_available_pairs": all(len(row) >= 3 and int(t[0]) == row[1] and int(t[2]) == row[2]
                                                                                   for t, row in zip(trace, log)),
                                     "original_trial_sequence_contiguous": [row[0] for row in log] == list(range(1, len(log) + 1)),
                                     "response_rt_domain_valid": valid_rt, "state_content_mapping_valid": allowed,
                                     "shell_input_structure": structures, "structural_exclusion_reasons": issues,
                                     "complete_pair_eligible": not bool(issues)}
    expected = {"schema": "drosolearn.tnu-shell-source-audit.v1", "record_id": manifest["record_id"],
                "raw_metadata_participants": 50, "complete_participants": 49, "complete_trials": 49 * 216,
                "eligible_ids": manifest["allowed_complete_ids"], "excluded": manifest["expected_exclusions"],
                "files": {name: {"bytes": spec["bytes"], "sha256": spec["sha256"]} for name, spec in manifest["files"].items()},
                "member_hashes": members, "participants": participants,
                "potential_content_is_environment_only": True, "raw_files_modified": False, "raw_files_extracted": False,
                "behavioral_performance_aggregated": False, "medication_groups_compared": False,
                "license_conflict_preserved": True}
    compare_tree(actual, expected, "Independent source audit")


def check_csv_row(actual, expected, description):
    demand(set(actual) == set(expected), f"CSV fields differ {description}")
    for key, value in expected.items():
        if value is None:
            demand(actual[key] == "", f"Missing measurement {description}.{key}")
        elif isinstance(value, str):
            demand(actual[key] == value, f"Text measurement {description}.{key}")
        elif isinstance(value, (int, np.integer)):
            demand(int(actual[key]) == value and str(int(actual[key])) == actual[key], f"Integer measurement {description}.{key}")
        else:
            scalar_close(float(actual[key]), value, f"{description}.{key}")


def graph_for(condition, seed, base, cache):
    if condition not in CONDITIONS[:6]:
        return None
    kind = "rewired" if condition.startswith("rewired") else "ann" if condition.startswith("ann") else "real"
    key = (kind, seed)
    if key not in cache:
        cache[key] = transform_graph(base, kind, seed + (110000 if kind == "rewired" else 120000 if kind == "ann" else 0))
    return cache[key]


def expected_measurement(policy, before, after, tape, record, condition, seed, pid, mode=None, rate=None):
    nll, brier = score_choices(tape["p_go"], tape["action"])
    counts = {"head_update_calls": policy.head_updates, "internal_update_calls": policy.internal_updates,
              "value_update_calls": policy.value_updates,
              "head_nonzero_steps": int(np.count_nonzero(tape["head_step"])),
              "internal_nonzero_steps": int(np.count_nonzero(tape["internal_step"])),
              "value_nonzero_steps": int(np.count_nonzero(tape["value_step"]))}
    if mode is None:
        return {"condition": condition, "rate": rate, "seed": seed, "participant_id": pid,
                "mean_nll": nll, "brier": brier, **counts}
    behavior = behavioral_metrics(record, tape["action"])
    row = {"mode": mode, "condition": condition, "seed": seed, "participant_id": pid,
           "mean_nll": nll if mode == "conditional" else None,
           "brier": brier if mode == "conditional" else None,
           **dict(zip(("acquisition_gain", "reversal_early", "reversal_recovery"), behavior.tolist())), **counts}
    for label, field in (("head", "head"), ("internal", "weights"), ("value", "value_table")):
        row[label + "_change_l2"] = float(np.linalg.norm(after[field] - before[field]))
    for field in ("probe_kc", "probe_mbon", "probe_decision"):
        row[field + "_change_l2"] = float(np.linalg.norm(after[field] - before[field]))
    return row


def expected_summary(cfg, records, model_behavior, conditional_nll, conditional_brier):
    ids, seeds = cfg["test_ids"], cfg["test_seeds"]
    names = ("acquisition_gain", "reversal_early", "reversal_recovery")
    human = np.array([behavioral_metrics(records[pid], records[pid]["action"]) for pid in ids])
    pp, ss = resampling_indices(len(ids), len(seeds), cfg["bootstrap_repeats"], cfg["bootstrap_seed"])
    families = []
    primary = CONDITIONS.index(cfg["primary_condition"])
    primary_boot = None
    for ci, condition in enumerate(CONDITIONS):
        differences, hb, mb = crossed_draws(human, model_behavior[ci], pp, ss)
        ci90 = interval90(differences)
        cells = []
        for mi, name in enumerate(names):
            bounds = ci90[:, mi]
            label = equivalence_label(bounds[:, None], cfg["equivalence_margin"])
            cells.append({"metric": name, "human_mean": float(human[:, mi].mean()),
                          "model_mean": float(model_behavior[ci, :, :, mi].mean()),
                          "difference": float(model_behavior[ci, :, :, mi].mean() - human[:, mi].mean()),
                          "ci90": bounds.tolist(), "equivalent": label == "supported",
                          "status": {"supported": "within_margin", "different": "outside_margin",
                                     "inconclusive": "inconclusive"}[label]})
        families.append({"condition": condition, "metrics": cells,
                         "all_three_within_margin": all(cell["equivalent"] for cell in cells),
                         "inference_role": "primary" if ci == primary else "descriptive_control"})
        if ci == primary:
            primary_boot = (hb, mb)
    prediction = []
    for baseline in ("uniform", "bias", "persistence"):
        gain = conditional_nll[CONDITIONS.index(baseline)] - conditional_nll[primary]
        draws = np.array([gain[pi][:, si].mean() for pi, si in zip(pp, ss)])
        bounds = interval90(draws).tolist()
        prediction.append({"baseline": baseline, "mean_improvement_nats": float(gain.mean()),
                           "ci90": bounds, "passed": bounds[0] > cfg["prediction_margin"]})
    hgain = interval90(primary_boot[0][:, 0]).tolist()
    mgain = interval90(primary_boot[1][:, 0]).tolist()
    learned_h, learned_m = hgain[0] > cfg["learning_gain_margin"], mgain[0] > cfg["learning_gain_margin"]
    similar = families[primary]["all_three_within_margin"]
    return {"status": "completed", "human_test_people": len(ids), "model_seeds": len(seeds),
            "mode_order": ["conditional", "autonomous"], "condition_order": list(CONDITIONS),
            "behavior_metric_order": list(names),
            "human_metrics": [{"participant_id": pid, **dict(zip(names, human[i].tolist()))} for i, pid in enumerate(ids)],
            "behavior_family_comparisons": families,
            "conditional_family_mean_nll": {condition: float(conditional_nll[ci].mean()) for ci, condition in enumerate(CONDITIONS)},
            "conditional_family_mean_brier": {condition: float(conditional_brier[ci].mean()) for ci, condition in enumerate(CONDITIONS)},
            "H_similarity": {"primary_condition": cfg["primary_condition"], "supported": similar,
                             "claim": "behavioral_equivalence_only", "margin": cfg["equivalence_margin"]},
            "H_prediction": {"primary_condition": cfg["primary_condition"],
                             "supported": all(item["passed"] for item in prediction),
                             "minimum_improvement_nats": cfg["prediction_margin"], "contrasts": prediction},
            "learning_qualification": {"human_initial_gain_ci90": hgain, "model_initial_gain_ci90": mgain,
                                       "human_gain_observed": bool(learned_h), "model_gain_observed": bool(learned_m),
                                       "human_gain_status": "observed" if learned_h else "not_observed",
                                       "gated_learning_pattern_supported": bool(similar and learned_h and learned_m)},
            "bootstrap": {"repeats": cfg["bootstrap_repeats"], "seed": cfg["bootstrap_seed"],
                          "quantiles": [.05, .95],
                          "method": "crossed participant/seed resampling; same people and shared seeds retained across conditions"},
            "test_modeled_trials": 2 * len(CONDITIONS) * len(seeds) * len(ids) * cfg["trials_per_person"],
            "conditional_trials": len(CONDITIONS) * len(seeds) * len(ids) * cfg["trials_per_person"],
            "autonomous_trials": len(CONDITIONS) * len(seeds) * len(ids) * cfg["trials_per_person"],
            "autonomous_nll_interpreted": False, "mechanism_claim": False, "educational_claim": False}


def verify(root, config, registration, result_dir, cache):
    """Execute only after the independent source/registration identity checks."""
    root, result_dir, cache = Path(root).resolve(), Path(result_dir).resolve(), Path(cache).resolve()
    cfg, reg = read_json(config), read_json(registration)
    demand(cfg["study_id"] == "human_shell_v1" and cfg["conditions"] == list(CONDITIONS), "Study/condition identity")
    demand(cfg["adaptive_conditions"] == list(CONDITIONS[:8]), "Adaptive condition inventory")
    demand(cfg["rates"] == [.03, .1, .3] and cfg["calibration_seeds"] == [13001, 13002, 13003]
           and cfg["test_seeds"] == list(range(13101, 13106)), "Fixed candidates/seeds")
    demand((cfg["trials_per_person"], cfg["bootstrap_repeats"], cfg["bootstrap_seed"], cfg["equivalence_margin"],
            cfg["prediction_margin"], cfg["learning_gain_margin"], cfg["primary_condition"])
           == (216, 2000, 14001, .1, .01, .05, "plastic"), "Measurement criteria changed")
    demand(reg["config"] == cfg and reg["config_sha256"] == digest(config), "Registered config differs")
    demand((result_dir / "registration.json").read_bytes() == Path(registration).read_bytes(), "Registration copy differs")
    demand(reg["registration_type"] == "exploratory_human_trace_comparison", "Study type changed")
    for group, names in (("source_files", cfg["source_files"]), ("parent_files", cfg["parent_files"]),
                         ("graph_files", cfg["graph_files"])):
        demand(len(names) == len(set(names)) and set(reg[group]) == set(names), f"Frozen inventory {group}")
        for name in names:
            demand(digest(contained(root, name)) == reg[group][name], f"Changed frozen bytes {name}")
    for name, expected in reg["source_files"].items():
        blob = subprocess.check_output(["git", "show", reg["source_git_commit"] + ":" + name], cwd=root)
        demand(hashlib.sha256(blob).hexdigest() == expected, f"Source commit mismatch {name}")
    demand(digest(__file__) == reg["source_files"]["scripts/verify_human_shell.py"], "Verifier source not frozen")
    demand(read_json(root / "results/human_task_structure_v1/audit.json")["gate"]["direct_human_model_similarity_allowed"] is False,
           "Historical comparison gate must remain closed")
    manifest = read_json(contained(root, cfg["manifest"]))
    demand(set(reg["external_files"]) == set(manifest["files"]), "External inventory differs")
    demand(not cache.is_relative_to(root), "Raw data remain outside repository")
    for name, specification in manifest["files"].items():
        path = contained(cache, name)
        demand(path.stat().st_size == specification["bytes"] and digest(path) == specification["sha256"], f"External source changed {name}")
        demand(reg["external_files"][name] == {"bytes": specification["bytes"], "sha256": specification["sha256"]}, "Registered external changed")
    metadata = read_json(result_dir / "metadata.json")
    demand(metadata["status"] == "completed" and metadata["source_git_commit"] == reg["source_git_commit"], "Completed source identity")
    demand(metadata["registration_sha256"] == digest(registration) and metadata["config_sha256"] == digest(config), "Metadata identity")
    stamps = [datetime.fromisoformat(reg["registered_at_utc"]), datetime.fromisoformat(metadata["started_at_utc"]),
              datetime.fromisoformat(metadata["completed_at_utc"])]
    demand(all(stamp.tzinfo is not None and stamp.utcoffset().total_seconds() == 0 for stamp in stamps), "UTC timestamps required")
    demand(stamps[0] < stamps[1] <= stamps[2] <= datetime.now(timezone.utc), "Prospective execution timeline")
    demand(metadata["human_model_comparison_tested"] is True and metadata["shared_neural_mechanism_tested"] is False
           and metadata["drug_effect_tested"] is False, "Interpretation scope changed")
    compare_tree(metadata, {"study_id": "human_shell_v1", "original_human_people": 49,
                            "original_human_trials": 49 * 216, "calibration_people": 20, "heldout_people": 29,
                            "calibration_conditional_observations": 8 * 3 * 3 * 20 * 216,
                            "test_conditional_predictions": 11 * 5 * 29 * 216,
                            "autonomous_model_choices": 11 * 5 * 29 * 216}, "Execution counters", subset=True)
    demand(reg["gate"]["previous_human_data_gate_unchanged"] is True
           and reg["gate"]["neural_mechanism_or_drug_effect_tested"] is False, "New comparison gate scope")
    primary_names = {"registration.json", "metadata.json", "cohort_audit.json", "selection.json", "calibration.npz",
                     "calibration.csv", "trajectories.npz", "trajectory_metrics.csv", "witnesses.npz", "summary.json"}
    artifact_map = read_json(result_dir / "artifact_hashes.json")
    demand(set(artifact_map) == primary_names, "Primary artifact inventory differs")
    for name, expected in artifact_map.items():
        demand(digest(contained(result_dir, name)) == expected, f"Primary artifact changed {name}")
    records, calibration_ids, test_ids, excluded = parse_cohort(cache / "Dataset_TNU.zip")
    check_source_audit(cache / "Dataset_TNU.zip", manifest, read_json(result_dir / "cohort_audit.json"))
    demand(calibration_ids == cfg["calibration_ids"] and test_ids == cfg["test_ids"], "Source-derived fixed split differs")
    demand(set(records) == set(manifest["allowed_complete_ids"]), "Complete cohort differs from source manifest")
    for pid, record in records.items():
        record["id"] = pid
    base, graph_cache = raw_graph(root / cfg["graph"]), {}
    go_count = sum(int(records[pid]["action"].sum()) for pid in calibration_ids)
    bias_probability = (go_count + 1) / (216 * len(calibration_ids) + 2)
    selection = read_json(result_dir / "selection.json")
    compare_tree(selection, {"calibration_bias": bias_probability, "calibration_go_count": go_count,
                            "calibration_trial_count": 216 * len(calibration_ids), "calibration_ids": calibration_ids,
                            "calibration_seeds": cfg["calibration_seeds"], "candidate_rates": cfg["rates"]}, "selection", subset=True)
    calibration_rows = read_measurements(result_dir / "calibration.csv", ("condition", "rate", "seed", "participant_id"))
    calibration_scores = np.empty((8, 3, 3, 20))
    calibration_trajectories = 0
    with np.load(result_dir / "calibration.npz", allow_pickle=False) as source:
        demand(set(source.files) == {"probabilities"}, "Calibration array inventory")
        probabilities = source["probabilities"]
        demand(probabilities.shape == (8, 3, 3, 20, 216), "Calibration tensor axes")
        for si, seed in enumerate(cfg["calibration_seeds"]):
            for ci, condition in enumerate(CONDITIONS[:8]):
                graph = graph_for(condition, seed, base, graph_cache)
                for ri, rate in enumerate(cfg["rates"]):
                    for pi, pid in enumerate(calibration_ids):
                        policy = AuditPolicy(condition, seed, rate, graph, bias_probability)
                        tape = replay(policy, records[pid], "conditional", seed)
                        array_close(probabilities[ci, ri, si, pi], tape["p_go"], "Calibration probabilities")
                        calibration_scores[ci, ri, si, pi] = score_choices(tape["p_go"], records[pid]["action"])[0]
                        expected = expected_measurement(policy, None, None, tape, records[pid], condition, seed, pid, rate=rate)
                        key = (condition, str(rate), str(seed), pid)
                        demand(key in calibration_rows, "Missing calibration row")
                        check_csv_row(calibration_rows.pop(key), expected, "Calibration row")
                        calibration_trajectories += 1
            print(f"independent calibration seed {seed} complete", flush=True)
    demand(not calibration_rows, "Extra calibration measurement rows")
    means = calibration_scores.mean(axis=(2, 3))
    selected = {condition: float(min(zip(means[ci], cfg["rates"]), key=lambda pair: (float(pair[0]), pair[1]))[1])
                for ci, condition in enumerate(CONDITIONS[:8])}
    demand(selection["rates"] == selected, "Learning-rate choice differs")
    compare_tree(selection["candidate_mean_nll"], {condition: means[ci].tolist() for ci, condition in enumerate(CONDITIONS[:8])},
                 "Candidate mean NLL")
    rows = read_measurements(result_dir / "trajectory_metrics.csv", ("mode", "condition", "seed", "participant_id"))
    model_behavior = np.empty((11, 29, 5, 3)); conditional_nll = np.empty((11, 29, 5)); conditional_brier = np.empty((11, 29, 5))
    trajectory_count = witness_arrays = 0
    expected_witness_keys = set()
    key_map = {"probabilities": "p_go", "actions": "action", "rewards": "reward", "head_steps": "head_step",
               "internal_steps": "internal_step", "value_steps": "value_step"}
    with np.load(result_dir / "trajectories.npz", allow_pickle=False) as source, np.load(result_dir / "witnesses.npz", allow_pickle=False) as witness:
        demand(set(source.files) == set(key_map), "Trajectory tensor inventory")
        trajectories = {key: source[key] for key in key_map}
        demand(all(value.shape == (2, 11, 5, 29, 216) for value in trajectories.values()), "Trajectory tensor axes")
        for mode_index, mode in enumerate(("conditional", "autonomous")):
            for si, seed in enumerate(cfg["test_seeds"]):
                for ci, condition in enumerate(CONDITIONS):
                    graph = graph_for(condition, seed, base, graph_cache)
                    for pi, pid in enumerate(test_ids):
                        policy = AuditPolicy(condition, seed, selected.get(condition, cfg["rates"][0]), graph, bias_probability)
                        before = policy.witness()
                        tape = replay(policy, records[pid], mode, seed)
                        after = policy.witness()
                        for key, tape_key in key_map.items():
                            array_close(trajectories[key][mode_index, ci, si, pi], tape[tape_key], f"{mode}/{condition}/{seed}/{pid}/{key}")
                        expected = expected_measurement(policy, before, after, tape, records[pid], condition, seed, pid, mode=mode)
                        key = (mode, condition, str(seed), pid)
                        demand(key in rows, "Missing trajectory measurement")
                        check_csv_row(rows.pop(key), expected, "Trajectory measurement")
                        if mode == "conditional":
                            conditional_nll[ci, pi, si] = expected["mean_nll"]
                            conditional_brier[ci, pi, si] = expected["brier"]
                        else:
                            model_behavior[ci, pi, si] = behavioral_metrics(records[pid], tape["action"])
                        if pi == 0:
                            for stage, values in (("initial", before), ("final", after)):
                                for field, value in values.items():
                                    name = f"{mode}_{condition}_{stage}_{field}"
                                    demand(name in witness.files, "Missing witness array")
                                    demand(witness[name].shape == (5, *np.asarray(value).shape), "Witness axes")
                                    array_close(witness[name][si], value, name)
                                    witness_arrays += 1; expected_witness_keys.add(name)
                        trajectory_count += 1
                print(f"independent {mode} seed {seed} complete", flush=True)
        demand(set(witness.files) == expected_witness_keys, "Extra witness fields")
    demand(not rows, "Extra trajectory measurement rows")
    summary = read_json(result_dir / "summary.json")
    expected = expected_summary(cfg, records, model_behavior, conditional_nll, conditional_brier)
    compare_tree(summary, expected, "Independent summary", subset=True)
    demand(set(summary) == set(expected), "Unexpected summary claim fields")
    return {"status": "passed", "study_id": cfg["study_id"], "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "registration_sha256": digest(registration), "summary_sha256": digest(result_dir / "summary.json"),
            "source_files": len(reg["source_files"]), "parent_files": len(reg["parent_files"]),
            "graph_files": len(reg["graph_files"]), "external_files": len(reg["external_files"]),
            "primary_artifacts": len(primary_names), "human_people": len(records), "human_original_trials": len(records) * 216,
            "calibration_people": len(calibration_ids), "test_people": len(test_ids), "structural_exclusions": excluded,
            "independent_calibration_trajectories": calibration_trajectories,
            "independent_calibration_observations": calibration_trajectories * 216,
            "independent_test_trajectories": trajectory_count, "independent_test_observations": trajectory_count * 216,
            "independent_witness_array_checks": witness_arrays,
            "verifier_imports_producer": False, "all_frozen_inputs_match": True,
            "verifier_modifies_original_results": False, "registered_parent_files_match": True,
            "interpretation": "independent numerical replay of existing recorded traces and measurements"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    demand(not args.out.exists(), "New independent verification output required")
    report = verify(args.root, args.config, args.registration, args.result_dir, args.cache)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": report["status"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
