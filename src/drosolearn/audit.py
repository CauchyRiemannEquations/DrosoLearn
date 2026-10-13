"""Independent arithmetic and artifact audit; imports no experiment/model helpers.

The input is the documented oracle n/8 plus six artificial descriptors.  An
audit pass checks saved computation, not visual numerosity or biological validity.
"""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone

import numpy as np


_NUISANCE = ("object_size", "total_area", "density", "spacing", "layout", "visual_extent")
_FLAGS = {
    "fixed": ("real", False, True), "plastic": ("real", True, True),
    "rewired_fixed": ("rewired", False, True), "rewired_plastic": ("rewired", True, True),
    "ann_fixed": ("ann", False, True), "ann_plastic": ("ann", True, True),
    "frozen": ("real", False, False), "random_reward": ("real", True, True),
    "internal_only": ("real", True, False),
}
_TRIAL_FIELDS = {
    "seed", "condition", "phase", "trial", "family_id", "left_count", "right_count",
    "action", "reward", "correct", "p_right", "internal_step_l2", "readout_step_l2",
    "kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2",
}
_CURVE_FIELDS = {
    "seed", "condition", "phase", "trial", "split", "low", "high", "ratio", "accuracy",
    "expected_reward", "internal_change_l2", "readout_change_l2", "probe_kc_change_l2",
    "probe_mbon_change_l2",
}
_CASE_FIELDS = {
    "seed", "condition", "phase", "trial", "split", "case", "family_id",
    "left_count", "right_count", "target_action", "p_right", "correct",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _near(actual, expected, label, atol=1e-10):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    _require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
             and np.allclose(a, b, rtol=1e-9, atol=atol), f"Mismatch: {label}")


def _number(row, field):
    value = float(row[field])
    _require(np.isfinite(value), f"Nonfinite CSV {field}")
    return value


def _integer(row, field):
    value = int(row[field])
    _require(str(value) == str(row[field]), f"Noninteger CSV {field}")
    return value


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _indexed_csv(path, fields, key_fields):
    result = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(set(reader.fieldnames or ()) == fields, f"CSV schema: {Path(path).name}")
        for row in reader:
            _require(None not in row and all(v is not None for v in row.values()), "Malformed CSV row")
            key = tuple(_integer(row, f) if f in ("seed", "trial", "case") else row[f]
                        for f in key_fields)
            _require(key not in result, f"Duplicate row: {Path(path).name} {key}")
            result[key] = row
    return result


def _npz(path, fields):
    with np.load(path, allow_pickle=False) as archive:
        _require(set(archive.files) == set(fields), f"NPZ schema: {path}")
        return {key: archive[key].copy() for key in archive.files}


def _input(counts, nuisance):
    # The supplied quantity is an oracle abstraction, not an extracted dot count.
    return np.concatenate((counts[..., None] / 8.0, (nuisance - 0.5) / 10.0), axis=-1)


def _forward(inputs, encoder, bias, pre, post, weights, head):
    kc = np.maximum(np.matmul(inputs, encoder.T) + bias, 0.0)
    matrix = np.zeros((encoder.shape[0], head.size), dtype=float)
    # Scatter saved sparse anatomical pairs into an independent dense operator.
    np.add.at(matrix, (pre, post), weights)
    mbon = np.matmul(kc, matrix)
    logits = np.matmul(mbon, head)
    p_right = 1.0 / (1.0 + np.exp(-np.clip(logits[..., 1] - logits[..., 0], -40.0, 40.0)))
    return kc, mbon, p_right


def _metrics(probability, counts, larger):
    target = (counts[:, 1] > counts[:, 0]).astype(int)
    if not larger:
        target = 1 - target
    correct = np.where(np.abs(probability - 0.5) < 1e-12, 0.5,
                       ((probability > 0.5).astype(int) == target).astype(float))
    return {"accuracy": float(np.mean(correct)),
            "expected_reward": float(np.mean(np.where(target == 1, probability, 1 - probability)))}, target, correct


def _audit(output, report):
    metadata, registration = _json(output / "metadata.json"), _json(output / "registration.json")
    cfg, summary = metadata["config"], _json(output / "summary.json")
    _require(metadata.get("status") == summary.get("status") == "completed", "Run did not complete")
    _require(cfg == registration["config"], "Registered config differs from executed config")
    _require(_hash(output / "registration.json") == metadata["registration_sha256"], "Registration hash")
    _require(summary["scope"] == cfg["scope"], "Summary scope")
    seeds, conditions = cfg["seeds"], cfg["conditions"]
    _require(len(seeds) == len(set(seeds)) and len(conditions) == len(set(conditions)), "Duplicate design cells")
    _require(all(c in _FLAGS for c in conditions), "Unsupported condition in independent auditor")
    tmax, every, eval_n = cfg["trials_per_phase"], cfg["checkpoint_every"], cfg["eval_trials"]
    _require(tmax > 0 and every > 0 and tmax % every == 0 and eval_n > 0, "Invalid trial budget")
    checkpoints = list(range(0, tmax + 1, every))
    pairs = {"train_acquisition": cfg["train_pair"], "train_reversal": cfg["train_pair"],
             "iid": cfg["train_pair"], "ood": cfg["train_pair"]}
    pairs.update({"transfer_" + "_".join(map(str, p)): p for p in cfg["transfer_pairs"]})
    eval_splits = [s for s in pairs if not s.startswith("train_")]
    phases = ("acquisition", "reversal")

    hashes = _json(output / "artifact_hashes.json")
    required = {"metadata.json", "registration.json", "summary.json", "controls.json", "trials.csv",
                "learning_curves.csv", "evaluation_cases.csv"}
    for seed in seeds:
        base = f"seed_{seed}"
        required.update(f"{base}/{f}" for f in ("stimuli.npz", "cue_audit.json", "rewired_topology.npz", "ann_topology.npz"))
        required.update(f"{base}/{c}/{phase}_{t:04d}.npz"
                        for c in conditions for phase in phases for t in checkpoints)
    _require(required <= set(hashes), "Primary artifacts omitted from hash manifest")
    for relative, digest in hashes.items():
        path = (output / relative).resolve()
        _require(path.is_relative_to(output) and path.is_file(), f"Unsafe/missing artifact: {relative}")
        _require(_hash(path) == digest, f"Artifact hash mismatch: {relative}")
    report["counts"]["artifact_hashes"] = len(hashes)
    report["checks"].append("All manifest hashes and required primary artifact coverage")

    trials = _indexed_csv(output / "trials.csv", _TRIAL_FIELDS, ("seed", "condition", "phase", "trial"))
    curves = _indexed_csv(output / "learning_curves.csv", _CURVE_FIELDS,
                          ("seed", "condition", "phase", "trial", "split"))
    cases = _indexed_csv(output / "evaluation_cases.csv", _CASE_FIELDS,
                         ("seed", "condition", "phase", "trial", "split", "case"))
    expected_trials = {(s, c, p, t) for s in seeds for c in conditions for p in phases for t in range(1, tmax + 1)}
    expected_curves = {(s, c, p, t, split) for s in seeds for c in conditions for p in phases
                       for t in checkpoints for split in eval_splits}
    expected_cases = {(s, c, p, t, split, i) for s in seeds for c in conditions for p in phases
                      for t in (0, tmax) for split in eval_splits for i in range(eval_n)}
    _require(set(trials) == expected_trials, "Trial/seed/condition/phase coverage")
    _require(set(curves) == expected_curves, "Curve checkpoint/split coverage")
    _require(set(cases) == expected_cases, "Evaluation case coverage")
    outcomes = {(o["seed"], o["condition"]): o for o in summary["outcomes"]}
    _require(len(outcomes) == len(summary["outcomes"]) and set(outcomes) == {(s, c) for s in seeds for c in conditions},
             "Summary seed/condition coverage")
    control_records = {(r["seed"], r["kind"]): r for r in _json(output / "controls.json")}
    _require(set(control_records) == {(s, k) for s in seeds for k in ("rewired", "ann")}, "Control coverage")
    report["counts"].update(trials=len(trials), curve_rows=len(curves), evaluation_cases=len(cases),
                             seeds=len(seeds), conditions=len(conditions))
    report["checks"].append("Exact design coverage and binary trial/evaluation schemas")
    seen_fingerprints = set()
    saved_fields = {"internal_weights", "readout_weights", "kc_activity", "mbon_activity",
                    "choice_probability", "encoder", "bias", "pre", "post"}

    for seed in seeds:
        seed_path = output / f"seed_{seed}"
        stimuli = _npz(seed_path / "stimuli.npz", {f"{name}_{key}" for name in pairs
                       for key in ("counts", "nuisance", "family_ids")})
        cue_audit = _json(seed_path / "cue_audit.json")
        _require(set(cue_audit) == set(pairs), "Cue audit split coverage")
        datasets = {}
        for name, pair in pairs.items():
            count, nuisance, families = (stimuli[f"{name}_{k}"] for k in ("counts", "nuisance", "family_ids"))
            n = tmax if name.startswith("train_") else eval_n
            _require(count.shape == (n, 2) and np.issubdtype(count.dtype, np.integer), f"Counts schema: {name}")
            _require(nuisance.shape == (n, 2, 6) and np.isfinite(nuisance).all(), f"Nuisance schema: {name}")
            _require(families.shape == (n,) and n % 4 == 0, f"Family schema: {name}")
            _require(np.all(np.sort(count, axis=1) == pair) and 1 <= min(pair) < max(pair) <= 8, f"Quantity pair: {name}")
            _require(np.all((nuisance >= 0) & (nuisance <= 1)), f"Descriptor range: {name}")
            domain_ok = ((nuisance <= 0.25) | (nuisance >= 0.75)) if name == "ood" else ((nuisance >= 0.25) & (nuisance <= 0.75))
            _require(np.all(domain_ok), f"Descriptor domain: {name}")
            unique_families, multiplicities = np.unique(families, return_counts=True)
            _require(set(map(str, unique_families)) == {f"{name}:{i}" for i in range(n // 4)}
                     and np.all(multiplicities == 4), f"Crossed family coverage: {name}")
            fingerprints = set()
            for row_c, row_n in zip(count, nuisance):
                order = np.argsort(row_c)
                canonical = np.asarray(row_n[order], dtype=np.float64).tobytes() + np.asarray(row_c[order], dtype=np.int64).tobytes()
                fingerprints.add(hashlib.sha256(canonical).hexdigest())
            _require(not seen_fingerprints.intersection(fingerprints), f"Canonical leakage: seed {seed}, {name}")
            seen_fingerprints.update(fingerprints)
            oracle = float(np.mean(np.argmax(count, axis=1) == (count[:, 1] > count[:, 0]).astype(int)))
            _near(oracle, 1.0, "quantity oracle", atol=0)
            _near(cue_audit[name]["oracle_accuracy"], oracle, "recorded quantity oracle", atol=0)
            _require(set(cue_audit[name]["single_nuisance_accuracy"]) == set(_NUISANCE), "Cue audit nuisance coverage")
            for j, nuisance_name in enumerate(_NUISANCE):
                difference = nuisance[:, 1, j] - nuisance[:, 0, j]
                target = (count[:, 1] > count[:, 0]).astype(int)
                accuracy = float(np.mean(np.where(difference == 0, 0.5, ((difference > 0).astype(int) == target).astype(float))))
                _near(accuracy, 0.5, f"Single nuisance {name}/{nuisance_name}", atol=0)
                _near(cue_audit[name]["single_nuisance_accuracy"][nuisance_name], accuracy, "Recorded nuisance oracle", atol=0)
            datasets[name] = (count, nuisance, families, _input(count, nuisance))

        controls = {kind: _npz(seed_path / f"{kind}_topology.npz", {"pre", "post", "counts"})
                    for kind in ("rewired", "ann")}
        initial_by_kind = {}
        seed_initial = None
        for condition in conditions:
            kind, internal_enabled, head_enabled = _FLAGS[condition]
            condition_path = seed_path / condition
            expected_names = {f"{phase}_{t:04d}.npz" for phase in phases for t in checkpoints}
            _require({p.name for p in condition_path.glob("*.npz")} == expected_names, "Extra/missing checkpoints")
            snapshots = {(p, t): _npz(condition_path / f"{p}_{t:04d}.npz", saved_fields)
                         for p in phases for t in checkpoints}
            initial = snapshots["acquisition", 0]
            encoder, bias, pre, post = (initial[k] for k in ("encoder", "bias", "pre", "post"))
            w0, h0 = initial["internal_weights"], initial["readout_weights"]
            _require(encoder.ndim == 2 and encoder.shape[1] == 7 and bias.shape == (encoder.shape[0],), "Encoder schema")
            _require(w0.ndim == h0.ndim == pre.ndim == post.ndim == 1 and pre.shape == post.shape == w0.shape, "Sparse projection schema")
            _require(np.issubdtype(pre.dtype, np.integer) and np.issubdtype(post.dtype, np.integer), "Projection index dtype")
            _require(np.all((pre >= 0) & (pre < encoder.shape[0])) and np.all((post >= 0) & (post < h0.size)), "Projection index bounds")
            _require(len(set(zip(pre.tolist(), post.tolist()))) == len(pre) and len(pre) > 0, "Duplicate/empty projection")
            _require(all(np.isfinite(initial[k]).all() for k in saved_fields), "Nonfinite snapshot")
            _require(np.all(w0 > 0), "Initial internal weights must be positive")
            if seed_initial is None:
                seed_initial = initial
            for key in ("encoder", "bias", "readout_weights"):
                _near(initial[key], seed_initial[key], f"Matched initialization {key}")
            if kind in initial_by_kind:
                for key in ("pre", "post", "internal_weights"):
                    _near(initial[key], initial_by_kind[kind][key], f"Matched topology initialization {kind}")
            initial_by_kind[kind] = initial
            if kind in controls:
                ctrl = controls[kind]
                _require(np.array_equal(pre, ctrl["pre"]) and np.array_equal(post, ctrl["post"]), "Saved control topology mismatch")
                norm = np.maximum(np.bincount(post, weights=ctrl["counts"], minlength=h0.size), 1.0)
                _near(w0, ctrl["counts"] / norm[post], "Control incoming normalization")
            probe = datasets["iid"][3][:16]
            initial_kc, initial_mb, _ = _forward(probe, encoder, bias, pre, post, w0, h0)
            measures = {}
            for phase in phases:
                for t in checkpoints:
                    snap = snapshots[phase, t]
                    for key in ("encoder", "bias", "pre", "post"):
                        _require(np.array_equal(snap[key], initial[key]), f"Fixed encoder/mask changed: {key}")
                    weights, head = snap["internal_weights"], snap["readout_weights"]
                    _require(weights.shape == w0.shape and head.shape == h0.shape and np.isfinite(weights).all()
                             and np.isfinite(head).all(), "Learned parameter schema")
                    _require(np.all((weights >= 0) & (weights <= 2 * w0 + 1e-12)), "Internal weight bounds")
                    if not internal_enabled:
                        _require(np.array_equal(weights, w0), "Fixed internal weights changed")
                    if not head_enabled:
                        _require(np.array_equal(head, h0), "Disabled readout weights changed")
                    kc, mb, probe_probability = _forward(probe, encoder, bias, pre, post, weights, head)
                    _near(snap["kc_activity"], kc, "Saved KC activity")
                    _near(snap["mbon_activity"], mb, "Saved MBON activity")
                    _near(snap["choice_probability"], probe_probability, "Saved probe probability")
                    changes = {"internal_change_l2": float(np.linalg.norm(weights - w0)),
                               "readout_change_l2": float(np.linalg.norm(head - h0)),
                               "probe_kc_change_l2": float(np.linalg.norm(kc - initial_kc)),
                               "probe_mbon_change_l2": float(np.linalg.norm(mb - initial_mb))}
                    for split in eval_splits:
                        count, _, families, inputs = datasets[split]
                        _, _, probability = _forward(inputs, encoder, bias, pre, post, weights, head)
                        metrics, target, correct = _metrics(probability, count, phase == "acquisition")
                        measures[phase, t, split] = metrics
                        curve = curves[seed, condition, phase, t, split]
                        for key, value in {**metrics, **changes, "low": min(pairs[split]),
                                           "high": max(pairs[split]), "ratio": min(pairs[split]) / max(pairs[split])}.items():
                            _near(_number(curve, key), value, f"Curve {seed}/{condition}/{phase}/{t}/{split}/{key}")
                        if t in (0, tmax):
                            for i in range(eval_n):
                                case = cases[seed, condition, phase, t, split, i]
                                _require(case["family_id"] == str(families[i]) and _integer(case, "left_count") == count[i, 0]
                                         and _integer(case, "right_count") == count[i, 1] and _integer(case, "target_action") == target[i], "Evaluation stimulus/target identity")
                                _near(_number(case, "p_right"), probability[i], "Evaluation case probability")
                                _near(_number(case, "correct"), correct[i], "Evaluation case greedy score")

            # Replay policy arithmetic directly, using saved initial parameters and
            # the recorded choices/rewards, then verify each update and checkpoint.
            weights, head = w0.copy(), h0.copy()
            action_rng, reward_rng = np.random.default_rng(seed + 80000), np.random.default_rng(seed + 90000)
            for phase in phases:
                _near(snapshots[phase, 0]["internal_weights"], weights, "Phase boundary internal continuity")
                _near(snapshots[phase, 0]["readout_weights"], head, "Phase boundary head continuity")
                count, _, families, inputs = datasets[f"train_{phase}"]
                for i in range(tmax):
                    row = trials[seed, condition, phase, i + 1]
                    _require(row["family_id"] == str(families[i]) and _integer(row, "left_count") == count[i, 0]
                             and _integer(row, "right_count") == count[i, 1], "Training stimulus identity")
                    action, reward, recorded_correct = (_integer(row, k) for k in ("action", "reward", "correct"))
                    _require(action in (0, 1) and reward in (0, 1) and recorded_correct in (0, 1), "Nonbinary action/reward/correct")
                    kc, mb, probability = _forward(inputs[i], encoder, bias, pre, post, weights, head)
                    probability = float(probability)
                    _near(_number(row, "p_right"), probability, "Pre-update trial probability")
                    _require(action == int(action_rng.random() < _number(row, "p_right")), "Action RNG stream")
                    correct_action = int(count[i, 1] > count[i, 0]) if phase == "acquisition" else int(count[i, 1] < count[i, 0])
                    _require(recorded_correct == int(action == correct_action), "Trial reward target correctness")
                    expected_reward = int(reward_rng.integers(0, 2)) if condition == "random_reward" else recorded_correct
                    _require(reward == expected_reward, "Reward rule / independent Bernoulli stream")
                    for label, value in {"kc_left_l2": np.linalg.norm(kc[0]), "kc_right_l2": np.linalg.norm(kc[1]),
                                         "mbon_left_l2": np.linalg.norm(mb[0]), "mbon_right_l2": np.linalg.norm(mb[1])}.items():
                        _near(_number(row, label), value, label)
                    factor = (action - probability) * (reward - 0.5)
                    next_head = head + cfg["head_lr"] * factor * (mb[1] - mb[0]) if head_enabled else head.copy()
                    eligibility = head[post] * (kc[1, pre] - kc[0, pre])
                    next_weights = np.minimum(np.maximum(weights + cfg["internal_lr"] * factor * eligibility, 0), 2 * w0) if internal_enabled else weights.copy()
                    _near(_number(row, "internal_step_l2"), np.linalg.norm(next_weights - weights), "Internal trial step")
                    _near(_number(row, "readout_step_l2"), np.linalg.norm(next_head - head), "Readout trial step")
                    if not internal_enabled:
                        _require(_number(row, "internal_step_l2") == 0, "Fixed internal step must equal zero")
                    if not head_enabled:
                        _require(_number(row, "readout_step_l2") == 0, "Disabled head step must equal zero")
                    weights, head = next_weights, next_head
                    if (i + 1) in checkpoints:
                        _near(snapshots[phase, i + 1]["internal_weights"], weights, "Replayed internal checkpoint")
                        _near(snapshots[phase, i + 1]["readout_weights"], head, "Replayed head checkpoint")

            outcome = outcomes[seed, condition]
            before = measures["acquisition", 0, "iid"]
            after = measures["acquisition", tmax, "iid"]
            gain = after["expected_reward"] - before["expected_reward"]
            criteria = cfg["criteria"]
            acquired = after["expected_reward"] >= criteria["acquisition_accuracy"] and gain >= criteria["acquisition_gain"]
            first = next((t for t in checkpoints if t > 0 and measures["acquisition", t, "iid"]["expected_reward"] >= criteria["acquisition_accuracy"]), None)
            expected = {
                "before_accuracy": before["expected_reward"], "after_accuracy": after["expected_reward"],
                "before_greedy_accuracy": before["accuracy"], "after_greedy_accuracy": after["accuracy"],
                "gain": gain, "ood_accuracy": measures["acquisition", tmax, "ood"]["expected_reward"],
                "reversal_before_accuracy": measures["reversal", 0, "iid"]["expected_reward"],
                "reversal_after_accuracy": measures["reversal", tmax, "iid"]["expected_reward"],
                "trainable_internal_parameters": len(w0) if internal_enabled else 0,
                "trainable_readout_parameters": len(h0) if head_enabled else 0,
                "effective_readout_inputs": len(set(post.tolist())),
                "internal_updates": 2 * tmax if internal_enabled else 0,
                "head_updates": 2 * tmax if head_enabled else 0,
            }
            for key, value in expected.items():
                _near(outcome[key], value, f"Summary {seed}/{condition}/{key}")
            _require(outcome["primary_metric"] == "exact stochastic choice accuracy (target probability)", "Primary metric labeling")
            booleans = {"acquisition_criterion_met": acquired,
                        "generalization_criterion_met": acquired and expected["ood_accuracy"] >= criteria["generalization_accuracy"],
                        "reversal_criterion_met": acquired and expected["reversal_after_accuracy"] >= criteria["reversal_accuracy"]}
            for key, value in booleans.items():
                _require(isinstance(outcome[key], bool) and outcome[key] == bool(value), f"Summary gate: {key}")
            _require(outcome["first_threshold_checkpoint"] == first, "First threshold checkpoint")
            transfer_names = {s for s in eval_splits if s.startswith("transfer_")}
            _require(set(outcome["transfer"]) == transfer_names, "Transfer summary coverage")
            for split in transfer_names:
                for key, value in measures["acquisition", tmax, split].items():
                    _near(outcome["transfer"][split][key], value, f"Transfer metric {split}/{key}")
                _require(outcome["transfer"][split]["interpretable_after_acquisition"] == bool(acquired), "Transfer acquisition gate")

        # Independently verify topological control invariants when a real condition
        # is part of the configured design (all default designs include one).
        if "real" in initial_by_kind:
            real = initial_by_kind["real"]
            base_pre, base_post = real["pre"], real["post"]
            for kind, ctrl in controls.items():
                cp, cq = ctrl["pre"], ctrl["post"]
                _require(cp.shape == cq.shape == base_pre.shape and len(set(zip(cp.tolist(), cq.tolist()))) == len(cp), "Control edge count/duplicates")
                check = control_records[seed, kind]["metadata"]["checks"]
                actual = {
                    "same_edge_count": len(cp) == len(base_pre),
                    "same_kc_outdegree": np.array_equal(np.bincount(base_pre, minlength=real["encoder"].shape[0]), np.bincount(cp, minlength=real["encoder"].shape[0])),
                    "same_mbon_indegree": np.array_equal(np.bincount(base_post, minlength=real["readout_weights"].size), np.bincount(cq, minlength=real["readout_weights"].size)),
                    "duplicate_free": True,
                    "changed_edge_fraction": len(set(zip(base_pre.tolist(), base_post.tolist())) - set(zip(cp.tolist(), cq.tolist()))) / len(cp),
                }
                for key, value in actual.items():
                    _near(check[key], value, f"Control metadata {kind}/{key}")
                if kind == "rewired":
                    _require(all(actual[k] for k in ("same_edge_count", "same_kc_outdegree", "same_mbon_indegree", "duplicate_free")), "Rewiring invariant")
        else:
            report["not_verified"].append(f"Seed {seed}: original topology invariants unavailable without a real condition checkpoint")

    report["counts"]["canonical_fingerprints"] = len(seen_fingerprints)
    report["counts"]["checkpoints"] = len(seeds) * len(conditions) * len(phases) * len(checkpoints)
    report["checks"].extend([
        "Canonical stimulus disjointness across all seeds/splits; exact quantity oracle 1 and single-descriptor baseline 0.5",
        "Manual n/8 input, ReLU KC, sparse-weight scatter, MBON logits and probabilities for every checkpoint/split",
        "Recomputed greedy diagnostics and exact stochastic choice accuracy, evaluation cases, activity and parameter changes",
        "Independent arithmetic replay of all trial updates, action RNG and separate random-reward Bernoulli RNG",
        "Frozen/internal-fixed zero changes, internal-only zero head updates, bounds and phase continuity",
        "Summary acquisition gain, generalization/reversal gates, transfer interpretation and update counts",
    ])


def verify(output: Path) -> dict:
    """Audit an experiment directory and always write verification.json.

    A failed assertion produces passed=false and an error; callers should treat it
    as a failed check.  The report itself is a derived artifact and is not included
    in the experiment's original artifact hash manifest.
    """
    output = Path(output).resolve()
    report = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
        "method": "independent numpy/stdlib arithmetic; no Learner/evaluate/data helper calls",
        "input_scope": "oracle quantity n/8 plus six crossed artificial descriptors; abstract reward association",
        "counts": {}, "checks": [], "errors": [],
        "not_verified": [
            "Biological LC11 perception, DAN dynamics, physiological time constants or shared human/fly learning mechanisms",
            "Source data identity outside this output, original preregistration authenticity or hashes of external current source files",
            "Claims of confirmatory significance, broad mathematical rule extraction or educational transfer",
            "Independence of biological animals: seeds are computational repetitions of one supplied topology",
        ],
    }
    try:
        _require(output.is_dir(), f"Output directory absent: {output}")
        _audit(output, report)
        report["passed"] = True
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
