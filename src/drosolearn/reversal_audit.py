"""Independent arithmetic audit for the registered history diagnostic.

Imports neither the learner nor experiment/data/summary helpers.  Saved raw
stimuli, random streams and initial parameters are replayed independently.
This verifies computation, not biological perception or preregistration credit.
"""
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json

import numpy as np


_HISTORIES = ("high_then_low", "low_then_low", "high_then_high", "fresh_low", "reset_head_after_high")
_FLAGS = {"fixed": ("real", False), "plastic": ("real", True),
          "rewired_fixed": ("rewired", False), "rewired_plastic": ("rewired", True),
          "ann_fixed": ("ann", False), "ann_plastic": ("ann", True)}
_TRIAL_FIELDS = {"seed", "condition", "history", "phase", "trial", "family_id", "left_count", "right_count",
                 "action", "reward", "p_right", "optimizer_factor", "internal_step_l2", "readout_step_l2",
                 "kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2"}
_CURVE_FIELDS = {"seed", "condition", "history", "phase", "trial", "split", "p_larger", "p_smaller",
                 "p_target", "greedy_accuracy", "mean_policy_sensitivity", "internal_change_l2",
                 "readout_change_l2", "probe_kc_change_l2", "probe_mbon_change_l2", "lower_bound_fraction", "upper_bound_fraction"}
_CASE_FIELDS = {"seed", "condition", "history", "phase", "trial", "split", "case", "family_id",
                "left_count", "right_count", "target_action", "p_right", "correct"}
_SAVED_FIELDS = {"internal_weights", "readout_weights", "kc_activity", "mbon_activity", "choice_probability",
                 "encoder", "bias", "pre", "post"}
_BOUNDARY_FIELDS = {"seed", "condition", "history", "head_reset_l2", "internal_reset_l2", "phase1_training_trials",
                    "head_updates_before_phase2", "internal_updates_before_phase2", "final_head_updates", "final_internal_updates"}
_INT_FIELDS = {"seed", "trial", "case"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _near(actual, expected, label, atol=1e-10):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    _require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
             and np.allclose(a, b, rtol=1e-9, atol=atol), f"Mismatch: {label}")


def _number(row, field):
    result = float(row[field])
    _require(np.isfinite(result), f"Nonfinite CSV {field}")
    return result


def _integer(row, field):
    result = int(row[field])
    _require(str(result) == str(row[field]), f"Noninteger CSV {field}")
    return result


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv(path, fields, key_fields):
    result = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(set(reader.fieldnames or ()) == fields, f"CSV schema: {path}")
        for row in reader:
            _require(None not in row and all(v is not None for v in row.values()), "Malformed CSV row")
            key = tuple(_integer(row, f) if f in _INT_FIELDS else row[f] for f in key_fields)
            _require(key not in result, f"Duplicate CSV key: {key}")
            result[key] = row
    return result


def _npz(path, fields):
    with np.load(path, allow_pickle=False) as archive:
        _require(set(archive.files) == set(fields), f"NPZ schema: {path}")
        return {k: archive[k].copy() for k in archive.files}


def _larger(history, phase):
    return history != "low_then_low" if phase == "phase1" else history == "high_then_high"


def _forward(inputs, encoder, bias, pre, post, weights, head):
    kc = np.maximum(np.matmul(inputs, encoder.T) + bias, 0.0)
    # Independent sparse accumulation, rather than Learner.features/bincount.
    flat = kc.reshape(-1, encoder.shape[0])
    mb = np.zeros((len(flat), head.size))
    for i, activity in enumerate(flat):
        np.add.at(mb[i], post, activity[pre] * weights)
    mb = mb.reshape(*kc.shape[:-1], head.size)
    logits = np.matmul(mb, head)
    probability = 1 / (1 + np.exp(-np.clip(logits[..., 1] - logits[..., 0], -40, 40)))
    return kc, mb, probability


def _metrics(probability, counts, larger):
    high = (counts[:, 1] > counts[:, 0]).astype(int)
    target = high if larger else 1 - high
    correct = np.where(np.abs(probability - 0.5) < 1e-12, 0.5,
                       ((probability > 0.5).astype(int) == target).astype(float))
    p_large = float(np.mean(np.where(high == 1, probability, 1 - probability)))
    return {"p_larger": p_large, "p_smaller": 1 - p_large,
            "p_target": p_large if larger else 1 - p_large,
            "greedy_accuracy": float(np.mean(correct)),
            "mean_policy_sensitivity": float(np.mean(probability * (1 - probability)))}, target, correct


def _auc(values, checkpoints):
    # Deliberately do not call the experiment's normalized_auc or np.trapezoid.
    return float(sum((a + b) * (right - left) / 2
                     for a, b, left, right in zip(values[:-1], values[1:], checkpoints[:-1], checkpoints[1:]))
                 / checkpoints[-1])


def _regenerate_dataset(seed, name, n, pair):
    """Independent generation of the explicitly registered abstract inputs."""
    rng = np.random.default_rng(seed)
    counts, nuisance, families = [], [], []
    for group in range(n // 4):
        uv = rng.uniform(0.25, 0.75, (2, 6))
        if name == "ood":
            uv = rng.uniform(0, 0.25, (2, 6)) + rng.integers(0, 2, (2, 6)) * 0.75
        low, high = pair
        for cs, ns in (((low, high), uv), ((high, low), uv[::-1]),
                       ((low, high), uv[::-1]), ((high, low), uv)):
            counts.append(cs); nuisance.append(ns.copy()); families.append(f"{name}:{group}")
    order = rng.permutation(n)
    return {"counts": np.asarray(counts, dtype=np.int64)[order], "nuisance": np.asarray(nuisance)[order],
            "family_ids": np.asarray(families)[order]}


def _dataset(stimuli, name, n, pair, fingerprints):
    counts, nuisance, families = (stimuli[f"{name}_{key}"] for key in ("counts", "nuisance", "family_ids"))
    _require(counts.shape == (n, 2) and np.issubdtype(counts.dtype, np.integer), f"Counts schema: {name}")
    _require(nuisance.shape == (n, 2, 6) and np.isfinite(nuisance).all(), f"Nuisance schema: {name}")
    _require(families.shape == (n,) and np.all(np.sort(counts, axis=1) == pair), f"Family/pair schema: {name}")
    domain = ((nuisance <= 0.25) | (nuisance >= 0.75)) if name == "ood" else ((nuisance >= 0.25) & (nuisance <= 0.75))
    _require(np.all((nuisance >= 0) & (nuisance <= 1)) and np.all(domain), f"Descriptor domain: {name}")
    unique, multiplicity = np.unique(families, return_counts=True)
    _require(set(map(str, unique)) == {f"{name}:{i}" for i in range(n // 4)} and np.all(multiplicity == 4), f"Crossed family coverage: {name}")
    # Verify entire descriptor vectors appear at both quantities and both sides.
    # Marginal single-cue chance alone cannot rule out joint cue leakage.
    for family in unique:
        assignments = {}
        for i in np.flatnonzero(families == family):
            for side in (0, 1):
                vector = np.asarray(nuisance[i, side], dtype=np.float64).tobytes()
                assignments.setdefault(vector, []).append((int(counts[i, side]), side))
        expected = sorted((int(count), side) for count in pair for side in (0, 1))
        _require(len(assignments) == 2 and all(sorted(a) == expected for a in assignments.values()), f"Joint nuisance crossing: {name}/{family}")
    target = (counts[:, 1] > counts[:, 0]).astype(int)
    _near(float(np.mean(np.argmax(counts, axis=1) == target)), 1, "Quantity oracle", atol=0)
    for j in range(6):
        diff = nuisance[:, 1, j] - nuisance[:, 0, j]
        accuracy = np.mean(np.where(diff == 0, 0.5, ((diff > 0).astype(int) == target).astype(float)))
        _near(accuracy, 0.5, f"Single nuisance chance: {name}/{j}", atol=0)
    current = set()
    for row_c, row_n in zip(counts, nuisance):
        order = np.argsort(row_c)
        canonical = np.asarray(row_n[order], dtype=np.float64).tobytes() + np.asarray(row_c[order], dtype=np.int64).tobytes()
        current.add(hashlib.sha256(canonical).hexdigest())
    _require(not current.intersection(fingerprints), f"Canonical stimulus leakage: {name}")
    fingerprints.update(current)
    inputs = np.concatenate((counts[..., None] / 8.0, (nuisance - 0.5) / 10.0), axis=-1)
    return counts, families, inputs


def _topologies(seed_path, records, n_kc, n_mbon):
    graphs = {kind: _npz(seed_path / f"{kind}_topology.npz", {"pre", "post", "counts"})
              for kind in ("real", "rewired", "ann")}
    real = graphs["real"]
    for kind, graph in graphs.items():
        pre, post, counts = (graph[k] for k in ("pre", "post", "counts"))
        _require(pre.ndim == post.ndim == counts.ndim == 1 and pre.shape == post.shape == counts.shape == real["pre"].shape and len(pre) > 0, "Topology dimensions/edge count")
        _require(np.issubdtype(pre.dtype, np.integer) and np.issubdtype(post.dtype, np.integer), "Topology integer indices")
        _require(np.all((pre >= 0) & (pre < n_kc)) and np.all((post >= 0) & (post < n_mbon)), "Topology index bounds")
        _require(np.isfinite(counts).all() and np.all(counts > 0) and np.all(counts == np.floor(counts)), "Positive integral anatomical counts")
        _require(len(set(zip(pre.tolist(), post.tolist()))) == len(pre), "Duplicate topology pairs")
        if kind == "real":
            continue
        actual = {
            "same_edge_count": len(pre) == len(real["pre"]),
            "same_kc_outdegree": np.array_equal(np.bincount(pre, minlength=n_kc), np.bincount(real["pre"], minlength=n_kc)),
            "same_mbon_indegree": np.array_equal(np.bincount(post, minlength=n_mbon), np.bincount(real["post"], minlength=n_mbon)),
            "same_incoming_weight_multisets": all(np.array_equal(np.sort(counts[post == p]), np.sort(real["counts"][real["post"] == p])) for p in range(n_mbon)),
            "duplicate_free": True,
            "changed_edge_fraction": len(set(zip(real["pre"].tolist(), real["post"].tolist())) - set(zip(pre.tolist(), post.tolist()))) / len(pre),
        }
        recorded = records[kind]["metadata"]["checks"]
        _require(set(recorded) == set(actual), "Topology metadata check schema")
        for key, value in actual.items():
            _near(recorded[key], value, f"Topology metadata {kind}/{key}", atol=0)
        invariants = ("same_edge_count", "same_mbon_indegree", "same_incoming_weight_multisets", "duplicate_free")
        _require(all(actual[k] for k in invariants), f"Matched incoming resources: {kind}")
        if kind == "rewired":
            _require(actual["same_kc_outdegree"], "Rewired KC outdegree")
    return graphs


def _summary(cfg, measurements, boundaries, summary, checkpoints):
    n, crit = cfg["trials_per_phase"], cfg["criteria"]
    outcomes = {(r["seed"], r["condition"], r["history"]): r for r in summary["outcomes"]}
    expected_keys = {(s, c, h) for s in cfg["seeds"] for c in cfg["conditions"] for h in _HISTORIES}
    _require(len(outcomes) == len(summary["outcomes"]) and set(outcomes) == expected_keys, "Summary outcome coverage")
    expected_rows, expected_pairs = {}, {}
    for seed, condition, history in sorted(expected_keys):
        key = seed, condition, history
        before, after = (None, None) if history == "fresh_low" else tuple(measurements[*key, "phase1", t, "iid"]["p_target"] for t in (0, n))
        acquired = None if before is None else bool(after >= crit["acquisition_accuracy"] and after - before >= crit["acquisition_gain"])
        seq = [measurements[*key, "phase2", t, "iid"]["p_target"] for t in checkpoints]
        expected = {"phase1_before": before, "phase1_after": after, "phase1_acquired": acquired,
                    "phase2_before": seq[0], "phase2_after": seq[-1], "phase2_iid_auc": _auc(seq, checkpoints),
                    "phase2_acquired": seq[-1] >= crit["acquisition_accuracy"] and seq[-1] - seq[0] >= crit["acquisition_gain"],
                    "phase2_ood_after": measurements[*key, "phase2", n, "ood"]["p_target"],
                    "endpoint_criterion_met": seq[-1] >= crit["adaptation_accuracy"],
                    "qualified_reversal_criterion_met": bool(acquired and seq[-1] >= crit["adaptation_accuracy"]) if history in ("high_then_low", "reset_head_after_high") else None,
                    "first_threshold_checkpoint": next((t for t, p in zip(checkpoints, seq) if p >= crit["adaptation_accuracy"]), None),
                    "phase1_training_trials": 0 if history == "fresh_low" else n, "phase2_training_trials": n,
                    "head_reset_l2": boundaries[key]["head_reset_l2"],
                    "head_updates": boundaries[key]["final_head_updates"], "internal_updates": boundaries[key]["final_internal_updates"]}
        _require(set(outcomes[key]) == set(expected) | {"seed", "condition", "history"}, "Summary outcome schema")
        for field, value in expected.items():
            if value is None or type(value) is bool:
                _require(outcomes[key][field] is value, f"Summary gate/null: {field}")
            else:
                _near(outcomes[key][field], value, f"Summary outcome {key}/{field}")
        expected_rows[key] = expected
    paired = {(r["seed"], r["condition"]): r for r in summary["paired_contrasts"]}
    pair_keys = {(s, c) for s in cfg["seeds"] for c in cfg["conditions"]}
    _require(len(paired) == len(summary["paired_contrasts"]) and set(paired) == pair_keys, "Paired summary coverage")
    for seed, condition in sorted(pair_keys):
        fresh, reversal, reset, continued = (expected_rows[seed, condition, h] for h in ("fresh_low", "high_then_low", "reset_head_after_high", "low_then_low"))
        expected = {"history_auc_penalty": fresh["phase2_iid_auc"] - reversal["phase2_iid_auc"],
                    "history_endpoint_penalty": fresh["phase2_after"] - reversal["phase2_after"],
                    "reset_auc_gain": reset["phase2_iid_auc"] - reversal["phase2_iid_auc"],
                    "reset_minus_fresh_auc": reset["phase2_iid_auc"] - fresh["phase2_iid_auc"],
                    "continued_low_minus_reversal_auc": continued["phase2_iid_auc"] - reversal["phase2_iid_auc"],
                    "old_rule_acquired": reversal["phase1_acquired"]}
        _require(set(paired[seed, condition]) == set(expected) | {"seed", "condition"}, "Paired summary schema")
        for field, value in expected.items():
            if type(value) is bool:
                _require(paired[seed, condition][field] is value, f"Paired acquisition gate: {field}")
            else:
                _near(paired[seed, condition][field], value, f"Paired contrast {seed}/{condition}/{field}")
        expected_pairs[seed, condition] = expected
    verdicts = {r["condition"]: r for r in summary["family_verdicts"]}
    _require(len(verdicts) == len(summary["family_verdicts"]) and set(verdicts) == set(cfg["conditions"]), "Family verdict coverage")
    for condition in cfg["conditions"]:
        selected = [expected_pairs[s, condition] for s in cfg["seeds"]]
        penalties = [r["history_auc_penalty"] for r in selected]
        mean, positive = float(np.mean(penalties)), sum(p > 0 for p in penalties) / len(penalties)
        qualified = [r["history_auc_penalty"] for r in selected if r["old_rule_acquired"]]
        expected = {"n_seeds": len(selected), "mean_history_auc_penalty": mean, "positive_seed_fraction": positive,
                    "sd_history_auc_penalty": float(np.std(penalties, ddof=1)) if len(penalties) > 1 else 0.0,
                    "positive_seed_count": sum(p > 0 for p in penalties),
                    "history_cost_criterion_met": mean >= crit["history_auc_penalty"] and positive >= crit["positive_seed_fraction"],
                    "qualified_seed_count": len(qualified), "qualified_mean_auc_penalty": float(np.mean(qualified)) if qualified else None}
        _require(set(verdicts[condition]) == set(expected) | {"condition", "note"}, "Family verdict schema")
        for field, value in expected.items():
            if value is None or type(value) is bool:
                _require(verdicts[condition][field] is value, f"Family gate/null: {field}")
            else:
                _near(verdicts[condition][field], value, f"Family verdict {condition}/{field}")


def _audit(output, report):
    metadata, registration, summary = (_json(output / f) for f in ("metadata.json", "registration.json", "summary.json"))
    cfg = metadata["config"]
    _require(metadata.get("status") == summary.get("status") == "completed", "Study did not complete")
    _require(cfg == registration["config"] and summary["scope"] == cfg["scope"], "Config/scope consistency")
    _require(_hash(output / "registration.json") == metadata["registration_sha256"], "Registration hash")
    times = [datetime.fromisoformat(record[field]) for record, field in ((registration, "registered_at_utc"), (metadata, "started_at_utc"), (metadata, "completed_at_utc"))]
    _require(all(t.tzinfo is not None for t in times) and times == sorted(times), "Local registration/run chronology")
    _require(cfg["scope"] in ("preregistered_diagnostic", "integration_test"), "Study scope")
    seeds, conditions = cfg["seeds"], cfg["conditions"]
    _require(seeds and all(type(s) is int and s >= 0 for s in seeds) and len(seeds) == len(set(seeds)), "Unique nonnegative integer seeds")
    _require(conditions and all(c in _FLAGS for c in conditions) and len(conditions) == len(set(conditions)), "Unique supported conditions")
    _require(tuple(cfg["histories"]) == _HISTORIES, "Five registered histories")
    _require(metadata["rng"] == {"phase1_actions": "seed+180000", "phase2_actions": "seed+190000"}, "Recorded action RNG offsets")
    n, every, ev, probe_n = (cfg[k] for k in ("trials_per_phase", "checkpoint_every", "eval_trials", "probe_rows"))
    _require(all(type(v) is int and v > 0 for v in (n, every, ev, probe_n)) and n % 4 == ev % 4 == n % every == 0 and probe_n <= ev, "Budget/crossing validity")
    if cfg["scope"] == "preregistered_diagnostic":
        _require(len(seeds) == 10, "Ten diagnostic seeds")
    pair = cfg["pair"]
    _require(len(pair) == 2 and all(type(v) is int for v in pair) and 1 <= pair[0] < pair[1] <= 8, "Quantity pair")
    _require(all(np.isfinite(cfg[k]) and cfg[k] >= 0 for k in ("head_lr", "internal_lr")), "Learning rate validity")
    _require(set(cfg["criteria"]) == {"acquisition_accuracy", "acquisition_gain", "adaptation_accuracy", "history_auc_penalty", "positive_seed_fraction"}
             and all(type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1 for v in cfg["criteria"].values())
             and cfg["criteria"]["positive_seed_fraction"] > 0, "Criteria schema/ranges")
    checkpoints = list(range(0, n + 1, every))
    hashes = _json(output / "artifact_hashes.json")
    required = {"metadata.json", "registration.json", "summary.json", "controls.json", "learning_curves.csv"}
    for seed in seeds:
        base = f"seed_{seed}"
        required.update(f"{base}/{f}" for f in ("stimuli.npz", "trials.csv", "evaluation_cases.csv", "real_topology.npz", "rewired_topology.npz", "ann_topology.npz"))
        for condition in conditions:
            for history in _HISTORIES:
                path = f"{base}/{condition}/{history}"
                required.add(path + "/boundary.json")
                required.update(f"{path}/phase2_{t:04d}.npz" for t in checkpoints)
                required.update(f"{path}/phase1_{t:04d}.npz" for t in ([0] if history == "fresh_low" else checkpoints))
    _require(required <= set(hashes), "Required primary artifact hash coverage")
    for relative, digest in hashes.items():
        path = (output / relative).resolve()
        _require(path.is_relative_to(output) and path.is_file(), f"Unsafe/missing artifact: {relative}")
        _require(_hash(path) == digest, f"Artifact hash mismatch: {relative}")
    curves = _csv(output / "learning_curves.csv", _CURVE_FIELDS, ("seed", "condition", "history", "phase", "trial", "split"))
    expected_curves = {(s, c, h, p, t, split) for s in seeds for c in conditions for h in _HISTORIES
                       for p in (("phase2",) if h == "fresh_low" else ("phase1", "phase2")) for t in checkpoints for split in ("iid", "ood")}
    _require(set(curves) == expected_curves, "Learning curve exact design coverage")
    records = _json(output / "controls.json")
    controls = {(r["seed"], r["kind"]): r for r in records}
    _require(len(records) == len(controls) and set(controls) == {(s, k) for s in seeds for k in ("real", "rewired", "ann")}, "Topology control coverage")
    fingerprints, measurements, boundaries = set(), {}, {}
    trial_count = case_count = snapshot_count = 0
    for seed in seeds:
        seed_path = output / f"seed_{seed}"
        stimuli = _npz(seed_path / "stimuli.npz", {f"{name}_{key}" for name in ("train_phase1", "train_phase2", "iid", "ood") for key in ("counts", "nuisance", "family_ids")} | {"phase1_uniforms", "phase2_uniforms"})
        datasets = {name: _dataset(stimuli, name, n if name.startswith("train_") else ev, pair, fingerprints) for name in ("train_phase1", "train_phase2", "iid", "ood")}
        for name, offset in (("train_phase1", 21), ("train_phase2", 22), ("iid", 23), ("ood", 24)):
            regenerated = _regenerate_dataset(seed * 100 + offset, name, n if name.startswith("train_") else ev, pair)
            for field, expected in regenerated.items():
                _require(np.array_equal(stimuli[f"{name}_{field}"], expected), f"Registered input RNG stream: {name}/{field}")
        for phase, offset in (("phase1", 180000), ("phase2", 190000)):
            _near(stimuli[phase + "_uniforms"], np.random.default_rng(seed + offset).random(n), f"Shared {phase} uniform RNG", atol=0)
        trials = _csv(seed_path / "trials.csv", _TRIAL_FIELDS, ("seed", "condition", "history", "phase", "trial"))
        cases = _csv(seed_path / "evaluation_cases.csv", _CASE_FIELDS, ("seed", "condition", "history", "phase", "trial", "split", "case"))
        expected_trials = {(seed, c, h, p, t) for c in conditions for h in _HISTORIES for p in (("phase2",) if h == "fresh_low" else ("phase1", "phase2")) for t in range(1, n + 1)}
        expected_cases = {(seed, c, h, p, t, split, i) for c in conditions for h in _HISTORIES for p in (("phase2",) if h == "fresh_low" else ("phase1", "phase2")) for t in (0, n) for split in ("iid", "ood") for i in range(ev)}
        _require(set(trials) == expected_trials and set(cases) == expected_cases, "Exact trial/evaluation coverage; no fresh phase1 learning")
        trial_count += len(trials); case_count += len(cases)
        first = _npz(seed_path / conditions[0] / _HISTORIES[0] / "phase1_0000.npz", _SAVED_FIELDS)
        n_kc, n_mbon = first["encoder"].shape[0], len(first["readout_weights"])
        graphs = _topologies(seed_path, {k: controls[seed, k] for k in ("real", "rewired", "ann")}, n_kc, n_mbon)
        rng = np.random.default_rng(seed + 70000)
        encoder, bias, initial_head = rng.normal(0, 0.8, (n_kc, 7)), rng.uniform(-0.25, 0.25, n_kc), rng.normal(0, 0.02, n_mbon)
        _require(n_kc > 0 and n_mbon > 0, "Positive neuron dimensions")
        probe = datasets["iid"][2][:probe_n]
        for condition in conditions:
            kind, plastic = _FLAGS[condition]
            graph = graphs[kind]; pre, post = graph["pre"], graph["post"]
            w0 = graph["counts"] / np.maximum(np.bincount(post, weights=graph["counts"], minlength=n_mbon), 1)[post]
            initial_kc, initial_mb, _ = _forward(probe, encoder, bias, pre, post, w0, initial_head)
            history_snapshots = {}
            for history in _HISTORIES:
                key, path = (seed, condition, history), seed_path / condition / history
                expected_snapshots = {("phase2", t) for t in checkpoints} | {("phase1", t) for t in ([0] if history == "fresh_low" else checkpoints)}
                _require({p.name for p in path.glob("*.npz")} == {f"{p}_{t:04d}.npz" for p, t in expected_snapshots}, "Snapshot exact design coverage")
                snapshots = {(p, t): _npz(path / f"{p}_{t:04d}.npz", _SAVED_FIELDS) for p, t in expected_snapshots}
                history_snapshots[history] = snapshots; snapshot_count += len(snapshots)
                for (phase, t), snap in snapshots.items():
                    _require(snap["encoder"].shape == (n_kc, 7) and snap["bias"].shape == (n_kc,), "Encoder/bias dimensions")
                    for field, expected in (("encoder", encoder), ("bias", bias), ("pre", pre), ("post", post)):
                        _require(np.array_equal(snap[field], expected), f"Fixed matched initialization/topology: {field}")
                    weights, head = snap["internal_weights"], snap["readout_weights"]
                    _require(weights.shape == w0.shape and head.shape == initial_head.shape and all(np.isfinite(snap[f]).all() for f in _SAVED_FIELDS), "Finite snapshot parameter schema")
                    _require(np.all((weights >= 0) & (weights <= 2 * w0 + 1e-12)), "Internal clipping bounds")
                    if not plastic: _require(np.array_equal(weights, w0), "Fixed internal weights changed")
                    kc, mb, probability = _forward(probe, encoder, bias, pre, post, weights, head)
                    for field, expected in (("kc_activity", kc), ("mbon_activity", mb), ("choice_probability", probability)):
                        _near(snap[field], expected, f"Saved probe {field}")
                    if history == "fresh_low" and phase == "phase1": continue
                    changes = {"internal_change_l2": float(np.linalg.norm(weights - w0)), "readout_change_l2": float(np.linalg.norm(head - initial_head)),
                               "probe_kc_change_l2": float(np.linalg.norm(kc - initial_kc)), "probe_mbon_change_l2": float(np.linalg.norm(mb - initial_mb)),
                               "lower_bound_fraction": float(np.mean(weights == 0)), "upper_bound_fraction": float(np.mean(weights == 2 * w0))}
                    for split in ("iid", "ood"):
                        counts, families, inputs = datasets[split]
                        _, _, probability = _forward(inputs, encoder, bias, pre, post, weights, head)
                        metrics, target, correct = _metrics(probability, counts, _larger(history, phase))
                        measurements[*key, phase, t, split] = metrics
                        row = curves[*key, phase, t, split]
                        for field, expected in {**metrics, **changes}.items():
                            _near(_number(row, field), expected, f"Curve {key}/{phase}/{t}/{split}/{field}")
                        if t in (0, n):
                            for i in range(ev):
                                case = cases[*key, phase, t, split, i]
                                _require(case["family_id"] == str(families[i]) and _integer(case, "left_count") == counts[i, 0] and _integer(case, "right_count") == counts[i, 1] and _integer(case, "target_action") == target[i], "Evaluation stimulus/target identity")
                                _near(_number(case, "p_right"), probability[i], "Evaluation probability")
                                _near(_number(case, "correct"), correct[i], "Evaluation greedy score")
                weights, head = w0.copy(), initial_head.copy()
                _near(snapshots["phase1", 0]["internal_weights"], weights, "Initial internal weights")
                _near(snapshots["phase1", 0]["readout_weights"], head, "Initial head weights")
                boundary = _json(path / "boundary.json")
                _require(set(boundary) == _BOUNDARY_FIELDS and all(boundary[f] == value for f, value in zip(("seed", "condition", "history"), key)), "Boundary schema/identity")
                pretrain = 0 if history == "fresh_low" else n
                for phase in (("phase2",) if history == "fresh_low" else ("phase1", "phase2")):
                    if phase == "phase2":
                        old_head = head.copy()
                        if history == "reset_head_after_high": head = initial_head.copy()
                        expected_boundary = {"head_reset_l2": float(np.linalg.norm(head - old_head)), "internal_reset_l2": 0,
                                             "phase1_training_trials": pretrain, "head_updates_before_phase2": pretrain,
                                             "internal_updates_before_phase2": pretrain if plastic else 0,
                                             "final_head_updates": pretrain + n, "final_internal_updates": pretrain + n if plastic else 0}
                        for field, expected in expected_boundary.items(): _near(boundary[field], expected, f"Boundary intervention/counter {field}")
                    _near(snapshots[phase, 0]["internal_weights"], weights, "Phase boundary internal continuity")
                    _near(snapshots[phase, 0]["readout_weights"], head, "Phase boundary head/intervention continuity")
                    counts, families, inputs = datasets[f"train_{phase}"]
                    for i, stimulus in enumerate(inputs):
                        row = trials[*key, phase, i + 1]
                        _require(row["family_id"] == str(families[i]) and _integer(row, "left_count") == counts[i, 0] and _integer(row, "right_count") == counts[i, 1], "Shared training stimulus identity")
                        action, reward = _integer(row, "action"), _integer(row, "reward")
                        _require(action in (0, 1) and reward in (0, 1), "Binary training action/reward")
                        kc, mb, probability = _forward(stimulus, encoder, bias, pre, post, weights, head)
                        probability = float(probability)
                        _near(_number(row, "p_right"), probability, "Pre-update trial policy probability")
                        _require(action == int(stimuli[phase + "_uniforms"][i] < probability), "Shared action RNG choice")
                        target = int(counts[i, 1] > counts[i, 0])
                        if not _larger(history, phase): target = 1 - target
                        _require(reward == int(action == target), "Phase/history reward rule")
                        factor = (action - probability) * (reward - 0.5)
                        _near(_number(row, "optimizer_factor"), factor, "Trial optimizer factor")
                        next_head = head + cfg["head_lr"] * factor * (mb[1] - mb[0])
                        eligibility = head[post] * (kc[1, pre] - kc[0, pre])
                        next_weights = np.minimum(np.maximum(weights + cfg["internal_lr"] * factor * eligibility, 0), 2 * w0) if plastic else weights.copy()
                        expected_trial = {"internal_step_l2": np.linalg.norm(next_weights - weights), "readout_step_l2": np.linalg.norm(next_head - head),
                                          "kc_left_l2": np.linalg.norm(kc[0]), "kc_right_l2": np.linalg.norm(kc[1]), "mbon_left_l2": np.linalg.norm(mb[0]), "mbon_right_l2": np.linalg.norm(mb[1])}
                        for field, expected in expected_trial.items(): _near(_number(row, field), expected, f"Trial update/activity {field}")
                        if not plastic: _require(_number(row, "internal_step_l2") == 0, "Fixed internal trial change")
                        weights, head = next_weights, next_head
                        if (i + 1) % every == 0:
                            _near(snapshots[phase, i + 1]["internal_weights"], weights, "Replayed internal checkpoint")
                            _near(snapshots[phase, i + 1]["readout_weights"], head, "Replayed head checkpoint")
                boundaries[key] = boundary
            # Counterfactual and matched-prefix identities are exact, not merely
            # similar scores. The independently replayed weights are checked above.
            for t in checkpoints:
                high = history_snapshots["high_then_low"]["phase1", t]
                for history in ("high_then_high", "reset_head_after_high"):
                    for field in _SAVED_FIELDS:
                        _require(np.array_equal(high[field], history_snapshots[history]["phase1", t][field]), "Identical high-rule pretraining prefix")
                if not plastic:
                    for field in _SAVED_FIELDS:
                        _require(np.array_equal(history_snapshots["fresh_low"]["phase2", t][field], history_snapshots["reset_head_after_high"]["phase2", t][field]), "Fixed reset-head must equal fresh counterfactual")
            for t in range(1, n + 1):
                reference = trials[seed, condition, "high_then_low", "phase1", t]
                for history in ("high_then_high", "reset_head_after_high"):
                    actual = trials[seed, condition, history, "phase1", t]
                    _require(all(actual[field] == reference[field] for field in _TRIAL_FIELDS - {"history"}), "Identical high-rule trial prefix")
                if not plastic:
                    fresh = trials[seed, condition, "fresh_low", "phase2", t]
                    reset = trials[seed, condition, "reset_head_after_high", "phase2", t]
                    _require(all(fresh[field] == reset[field] for field in _TRIAL_FIELDS - {"history"}), "Fixed reset/fresh trial identity")
        print(f"audited seed {seed} ({seeds.index(seed) + 1}/{len(seeds)})", flush=True)
    _summary(cfg, measurements, boundaries, summary, checkpoints)
    report["counts"].update(artifact_hashes=len(hashes), seeds=len(seeds), conditions=len(conditions), histories=len(_HISTORIES),
                            trials=trial_count, curve_rows=len(curves), evaluation_cases=case_count, checkpoints=snapshot_count,
                            canonical_fingerprints=len(fingerprints), boundaries=len(boundaries))
    report["checks"].extend([
        "All required primary artifact hashes, exact design coverage and local registration/run chronology",
        "Full joint descriptor crossing, single-cue chance and canonical stimulus disjointness across seeds/splits",
        "All input arrays and shared phase1/phase2 uniform streams regenerated independently from registered seed offsets",
        "All three raw topologies: duplicate-free, matched incoming count multisets, edge/indegree resources and rewired outdegree",
        "Independent encoder/bias/head initialization, n/8 oracle encoding, sparse forward activity and evaluation arithmetic",
        "Every recorded action/reward and policy update replayed without Learner.update or experiment helpers",
        "Bounded internal changes, fixed zero internal steps, phase-boundary reset and counters verified",
        "Identical high-rule pretraining prefixes and fixed reset-head/fresh counterfactual at every trial/checkpoint",
        "Independent checkpoint trapezoid AUC, endpoints, acquisition/reversal gates, paired penalties and all-seed family verdicts",
    ])


def verify(output: Path) -> dict:
    """Write an auditable pass/failure report; primary artifacts remain unchanged."""
    output = Path(output).resolve()
    report = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "method": "independent numpy/stdlib replay; no learner/data/experiment/summary helper imports",
              "input_scope": "oracle quantity n/8 and six crossed artificial descriptors; algorithmic history diagnostic",
              "counts": {}, "checks": [], "errors": [],
              "not_verified": [
                  "Biological LC11 perception, DAN dynamics, physiological timescales or shared human/fly mechanisms",
                  "Remote private Git publication authenticity and source identity outside this output; local chronology is self-recorded",
                  "Confirmatory statistical significance, broad mathematical rules or educational implications",
                  "Seeds as independent animals: matched histories are dependent computational repetitions",
              ]}
    try:
        _require(output.is_dir(), f"Output directory missing: {output}")
        _audit(output, report)
        report["passed"] = True
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
