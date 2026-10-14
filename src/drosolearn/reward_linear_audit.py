"""Independent replay of reward-only linear and normalized network learners.

Only immutable helpers from earlier independent auditors are reused. This file
does not import a learner, data producer, runner, or producer summary function.
The training-derived scale is reconstructed from initial, unlabeled features.
"""
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import json

import numpy as np

from .encoding_audit import (
    _require, _near, _integer, _number, _json, _hash, _csv, _npz,
    _regenerate_dataset, _dataset, _inputs, _mbon, _metrics, _topologies, _cache,
)
from .linearity_audit import _source_projection


_FREE = {1, 3, 5, 6, 7, 8}
_PAIRS = tuple(pair for pair in combinations(range(1, 9), 2) if pair != (2, 4))
_UNSEEN = np.asarray([i for i, pair in enumerate(_PAIRS) if set(pair) <= _FREE])
_ANCHORED = np.asarray([i for i, pair in enumerate(_PAIRS) if not set(pair) <= _FREE])
_FOCAL = _PAIRS.index((5, 7))
_FLAGS = {"linear_raw": ("linear", False), "linear_normalized": ("linear", False),
          "linear_random_reward": ("linear", False),
          "fixed": ("real", False), "plastic": ("real", True),
          "rewired_fixed": ("rewired", False), "rewired_plastic": ("rewired", True),
          "ann_fixed": ("ann", False), "ann_plastic": ("ann", True)}
_ALPHAS = (.25, .5, 1., 1.5)
_TRIAL_FIELDS = {"seed", "condition", "trial", "family_id", "left_count", "right_count", "action", "reward",
                 "p_right", "optimizer_factor", "internal_step_l2", "readout_step_l2", "kc_left_l2", "kc_right_l2",
                 "mbon_left_l2", "mbon_right_l2", "decision_left_l2", "decision_right_l2"}
_CURVE_FIELDS = {"seed", "condition", "trial", "split", "p_target", "greedy_accuracy", "internal_change_l2",
                 "readout_change_l2", "probe_kc_change_l2", "probe_mbon_change_l2", "probe_decision_change_l2"}
_SAVED_FIELDS = {"internal_weights", "readout_weights", "kc_activity", "mbon_activity", "choice_probability",
                 "encoder", "bias", "pre", "post", "decision_activity", "feature_scale", "initial_feature_gap_rms"}
_TRANSFER_FIELDS = {"code_values", "score_cache", "map_p_target", "map_greedy_accuracy", "map_p_right"}


def _panel():
    return np.asarray([[1, 2, 3, 4, 6 - alpha, 6, 6 + alpha, 8] for alpha in _ALPHAS], dtype=float)


def _raw_features(inputs, encoder, bias, pre, post, weights, head_size, linear):
    if linear:
        empty = np.empty((*inputs.shape[:-1], 0), dtype=float)
        return empty, empty.copy(), np.asarray(inputs, dtype=float)
    kc = np.maximum(inputs @ encoder.T + bias, 0)
    mb = _mbon(kc, pre, post, weights, head_size)
    return kc, mb, mb


def _state(inputs, encoder, bias, pre, post, weights, head, scale, linear):
    kc, mb, raw = _raw_features(inputs, encoder, bias, pre, post, weights, head.size, linear)
    decision = raw / scale
    scores = decision @ head
    p = 1 / (1 + np.exp(-np.clip(scores[..., 1] - scores[..., 0], -40, 40)))
    return kc, mb, decision, p


def _transfer(saved, datasets, panel, encoder, bias, pre, post, weights, head, scale, linear):
    levels = np.unique(panel)
    _near(saved["code_values"], levels, "Unique mapped scalar values", atol=0)
    if linear:
        cache = []
        for a, b in _PAIRS:
            data = datasets[f"pair_{a}_{b}"]
            values = []
            for code in levels:
                inputs = np.concatenate((np.full((*data["counts"].shape, 1), float(code) / 8),
                                         .1 * (data["nuisance"] - .5)), axis=-1)
                values.append((inputs / scale) @ head)
            cache.append(np.stack(values, axis=-1))
        cache = np.asarray(cache)
    else:
        # Earlier audit helper independently accumulates the sparse readout
        # gains with add.at and evaluates each mapped scalar code separately.
        cache = _cache(datasets, levels, encoder, bias, pre, post, weights, head / scale)
    _near(saved["score_cache"], cache, "Every independent frozen candidate score")
    means, greedy, cases = [], [], []
    for j, pair in enumerate(_PAIRS):
        data = datasets[f"pair_{pair[0]}_{pair[1]}"]
        counts, size = data["counts"], len(data["counts"])
        indices = np.searchsorted(levels, panel[:, counts - 1])
        offsets = np.arange(size)[None, :, None] * (2 * len(levels)) + np.arange(2)[None, None, :] * len(levels)
        scores = cache[j].reshape(-1)[offsets + indices]
        p = 1 / (1 + np.exp(-np.clip(scores[..., 1] - scores[..., 0], -40, 40)))
        target = (counts[:, 1] > counts[:, 0]).astype(int)[None, :]
        correct = np.where(np.abs(p - .5) < 1e-12, .5, ((p > .5).astype(int) == target).astype(float))
        means.append(np.mean(np.where(target == 1, p, 1 - p), axis=1))
        greedy.append(correct.mean(axis=1)); cases.append(p)
    expected = {"map_p_target": np.asarray(means).T, "map_greedy_accuracy": np.asarray(greedy).T,
                "map_p_right": np.transpose(np.asarray(cases), (1, 0, 2))}
    for field, value in expected.items():
        _near(saved[field], value, f"Every mapped transfer entry: {field}")
    return expected


def _equal(actual, expected, label):
    if type(expected) is bool or expected is None:
        _require(actual is expected, f"Independent summary gate: {label}")
    elif isinstance(expected, str):
        _require(type(actual) is str and actual == expected, f"Independent summary text: {label}")
    elif isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), f"Independent summary schema: {label}")
        for key, value in expected.items():
            _equal(actual[key], value, f"{label}/{key}")
    elif isinstance(expected, list) and expected and type(expected[0]) is bool:
        _require(isinstance(actual, list) and len(actual) == len(expected) and
                 all(type(a) is bool and a is b for a, b in zip(actual, expected)), f"Independent summary flags: {label}")
    else:
        _near(actual, expected, f"Independent summary value: {label}")


def _summary(cfg, measurements, transfer, summary):
    _require(set(summary) == {"status", "scope", "outcomes", "family_verdicts", "paired_reward_contrasts",
                             "hypotheses", "actual_training_trials", "frozen_evaluation_maps", "interpretation"},
             "Summary top-level schema")
    criteria, n = cfg["criteria"], cfg["trials"]
    rows = {(row["seed"], row["condition"]): row for row in summary["outcomes"]}
    keys = {(seed, condition) for seed in cfg["seeds"] for condition in cfg["conditions"]}
    _require(len(rows) == len(summary["outcomes"]) and set(rows) == keys, "One summary row per actual fit")
    expected_rows = {}
    for seed, condition in sorted(keys):
        before, after = (measurements[seed, condition, t, "iid"]["p_target"] for t in (0, n))
        acquired = bool(after >= criteria["acquisition_accuracy"] and after - before >= criteria["acquisition_gain"])
        baseline, final = (transfer[seed, condition, stage] for stage in ("baseline", "final"))
        unseen = final["map_p_target"][:, _UNSEEN].mean(axis=1)
        expected = {"before_accuracy": before, "after_accuracy": after, "gain": after - before, "acquired": acquired,
                    "ood_accuracy": measurements[seed, condition, n, "ood"]["p_target"],
                    "head_updates": n, "internal_updates": n if _FLAGS[condition][1] else 0,
                    "focal_before": baseline["map_p_target"][:, _FOCAL].tolist(),
                    "focal_after": final["map_p_target"][:, _FOCAL].tolist(),
                    "focal_greedy_after": final["map_greedy_accuracy"][:, _FOCAL].tolist(),
                    "unseen_before": baseline["map_p_target"][:, _UNSEEN].mean(axis=1).tolist(),
                    "unseen_after": unseen.tolist(),
                    "anchored_after": final["map_p_target"][:, _ANCHORED].mean(axis=1).tolist(),
                    "all_after": final["map_p_target"].mean(axis=1).tolist(),
                    "qualified_unseen_transfer": [bool(acquired and value >= criteria["transfer_accuracy"]) for value in unseen],
                    "qualified_focal_transfer": [bool(acquired and value >= criteria["transfer_accuracy"])
                                                 for value in final["map_p_target"][:, _FOCAL]]}
        _require(set(rows[seed, condition]) == set(expected) | {"seed", "condition"}, "Summary outcome schema")
        for field, value in expected.items():
            _equal(rows[seed, condition][field], value, f"{seed}/{condition}/{field}")
        expected_rows[seed, condition] = expected
    verdicts = {row["condition"]: row for row in summary["family_verdicts"]}
    _require(len(verdicts) == len(summary["family_verdicts"]) and set(verdicts) == set(cfg["conditions"]),
             "Family verdict coverage")
    expected_verdicts = {}
    for condition in cfg["conditions"]:
        selected = [expected_rows[seed, condition] for seed in cfg["seeds"]]
        acquired = sum(row["acquired"] for row in selected)
        joint = sum(row["qualified_unseen_transfer"][2] for row in selected)
        expected = {"n_seeds": len(selected), "acquisition_count": acquired,
                    "acquisition_criterion_met": bool(acquired / len(selected) >= criteria["seed_pass_fraction"]),
                    "joint_identity_transfer_count": joint,
                    "joint_transfer_criterion_met": bool(joint / len(selected) >= criteria["seed_pass_fraction"]),
                    "mean_after_accuracy": float(np.mean([row["after_accuracy"] for row in selected])),
                    "sd_after_accuracy": float(np.std([row["after_accuracy"] for row in selected], ddof=1)) if len(selected) > 1 else 0.,
                    "mean_unseen_after": np.mean([row["unseen_after"] for row in selected], axis=0).tolist()}
        _require(set(verdicts[condition]) == set(expected) | {"condition"}, "Family verdict schema")
        for field, value in expected.items():
            _equal(verdicts[condition][field], value, f"{condition}/{field}")
        expected_verdicts[condition] = expected
    paired = {row["seed"]: row for row in summary["paired_reward_contrasts"]}
    _require(len(paired) == len(summary["paired_reward_contrasts"]) and set(paired) == set(cfg["seeds"]),
             "One paired reward contrast per seed")
    gaps = []
    for seed in cfg["seeds"]:
        gap = expected_rows[seed, "linear_normalized"]["after_accuracy"] - expected_rows[seed, "linear_random_reward"]["after_accuracy"]
        _require(set(paired[seed]) == {"seed", "iid_reward_gap"}, "Paired reward contrast schema")
        _near(paired[seed]["iid_reward_gap"], gap, "Independent all-seed paired contingent-minus-random reward contrast")
        gaps.append(gap)
    mean_gap, positive = float(np.mean(gaps)), sum(gap > 0 for gap in gaps)
    primary = expected_verdicts["linear_normalized"]
    hypotheses = {"H_R": {"criterion_met": bool(primary["acquisition_criterion_met"] and
                                               mean_gap >= criteria["reward_gap"] and
                                               positive / len(gaps) >= criteria["seed_pass_fraction"]),
                           "mean_iid_reward_gap": mean_gap, "positive_seed_count": positive,
                           "acquisition_count": primary["acquisition_count"]},
                  "H_G": {"criterion_met": bool(primary["joint_transfer_criterion_met"]),
                           "joint_identity_transfer_count": primary["joint_identity_transfer_count"]}}
    _equal(summary["hypotheses"], hypotheses, "hypotheses")
    _require(summary["actual_training_trials"] == len(keys) * n and summary["frozen_evaluation_maps"] == 4,
             "One actual fit per condition; four maps are frozen evaluations")
    _equal(summary["interpretation"], "independent immediate binary-reward learning on supplied scalar order; initial TRAIN gradient RMS normalized, not equal capacity or biological mechanisms",
           "interpretation")


def _source_freeze(output, metadata, registration, cfg):
    root = (output / metadata["source_root"]).resolve()
    _require(root.is_dir(), "Registered source root missing; use the registered code checkout")
    config = Path(metadata["config_path"])
    config = (root / config).resolve() if not config.is_absolute() else config.resolve()
    _require(config.is_file() and _hash(config) == registration["config_sha256"] and _json(config) == cfg,
             "Registered config bytes/content")
    expected = {path.relative_to(root).as_posix() for path in (root / "src").rglob("*.py") if path.is_file()}
    expected.update(path.relative_to(root).as_posix() for path in (root / cfg["graph"]).glob("*") if path.is_file())
    expected.update(("docs/reward-linear-protocol.md", "docs/reward-linear-background.md", "pyproject.toml",
                     "requirements-lock.txt", "data/source-manifest.json", "data/LICENSE.md"))
    _require(set(registration["source_hashes"]) == expected, "Registered source file set")
    for relative, digest in registration["source_hashes"].items():
        source = (root / relative).resolve()
        _require(source.is_relative_to(root) and source.is_file() and _hash(source) == digest,
                 f"Registered source byte hash: {relative}")
    return root


def _audit(output, report):
    metadata, registration, summary = (_json(output / name) for name in ("metadata.json", "registration.json", "summary.json"))
    cfg = metadata["config"]
    _require(metadata.get("status") == summary.get("status") == "completed", "Study did not complete")
    _require(cfg == registration["config"] and summary["scope"] == cfg["scope"], "Config/scope consistency")
    _require(_hash(output / "registration.json") == metadata["registration_sha256"], "Registration hash")
    times = [datetime.fromisoformat(record[field]) for record, field in
             ((registration, "registered_at_utc"), (metadata, "started_at_utc"), (metadata, "completed_at_utc"))]
    _require(all(value.tzinfo is not None for value in times) and times == sorted(times), "Local registration/run chronology")
    _require(cfg["scope"] in ("preregistered_diagnostic", "integration_test"), "Study scope")
    seeds, conditions = cfg["seeds"], cfg["conditions"]
    _require(seeds and all(type(seed) is int and seed >= 0 for seed in seeds) and len(set(seeds)) == len(seeds),
             "Unique nonnegative seeds")
    if cfg["scope"] == "integration_test":
        _require(not any(seed in range(6001, 6011) for seed in seeds), "Scientific seeds cannot alias engineering fixtures")
    _require(tuple(conditions) == tuple(_FLAGS), "Exact nine registered learner conditions")
    fixed = {"experiment_id": "independent_reward_linear_diagnostic_v1", "graph": "data/flywire_783_mb_left_kc512_s701",
             "head_initialization": "zeros_all_conditions",
             "feature_normalization": "fixed_initial_train_feature_gap_rms_except_linear_raw",
             "random_reward": "independent_seed_plus290000_uniform_below0.5"}
    _require(all(cfg.get(key) == value for key, value in fixed.items()),
             "Fixed study identity, topology, initialization, normalization and reward rules")
    _require(cfg["train_pair"] == [2, 4] and cfg["primary_pairs"] == "fully_unseen15" and tuple(cfg["code_alphas"]) == _ALPHAS,
             "Anchored training, fully unseen primary pairs and four alphas")
    _require(metadata["rng"] == {"actions": "seed+280000", "rewards": "seed+290000", "encoder": "seed+70000",
                                "rewired": "seed+50000", "ann": "seed+60000",
                                "data": "seed*100+31,+32,+33,+40+pair_index"}, "Registered RNG offsets")
    _require(metadata["initial_head"] == "zero" and metadata["primary_pairs"] == "fully_unseen15",
             "Metadata zero-head and primary-pair scope")
    n, every, ev, probe_n = (cfg[key] for key in ("trials", "checkpoint_every", "eval_trials", "probe_rows"))
    _require(all(type(value) is int and value > 0 for value in (n, every, ev, probe_n)) and
             n % 4 == ev % 4 == n % every == 0 and probe_n <= ev, "Budget/crossing validity")
    _require(all(type(cfg[key]) in (int, float) and np.isfinite(cfg[key]) and cfg[key] >= 0
                 for key in ("head_lr", "internal_lr")), "Learning rates")
    criteria = cfg["criteria"]
    _require(set(criteria) == {"acquisition_accuracy", "acquisition_gain", "transfer_accuracy", "seed_pass_fraction", "reward_gap"}
             and all(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1 for value in criteria.values())
             and criteria["seed_pass_fraction"] > 0, "Criteria schema/ranges")
    if cfg["scope"] == "preregistered_diagnostic":
        _require(seeds == list(range(6001, 6011)) and tuple(conditions) == tuple(_FLAGS),
                 "Registered ten new seeds and nine model conditions")
        _require((n, every, ev, probe_n, cfg["head_lr"], cfg["internal_lr"]) == (400, 100, 128, 8, .1, .01),
                 "Fixed registered experiment resources")
        _require(criteria == {"acquisition_accuracy": .75, "acquisition_gain": .15, "transfer_accuracy": .75,
                              "seed_pass_fraction": .8, "reward_gap": .15}, "Fixed registered criteria")
    root = _source_freeze(output, metadata, registration, cfg)
    source_graph, n_kc, n_mbon, source_metadata = _source_projection(root, cfg["graph"])
    _require(metadata["graph"] == source_metadata, "Independent source graph metadata")
    checkpoints = list(range(0, n + 1, every))
    hashes = _json(output / "artifact_hashes.json")
    required = {"metadata.json", "registration.json", "summary.json", "controls.json", "model_resources.json", "learning_curves.csv",
                "spacing_codes.npy", "code_geometry.npz"}
    for seed in seeds:
        base = f"seed_{seed}"
        required.update(f"{base}/{name}" for name in
                        ("stimuli.npz", "trials.csv", "real_topology.npz", "rewired_topology.npz", "ann_topology.npz"))
        for condition in conditions:
            path = f"{base}/{condition}"
            required.update(f"{path}/checkpoint_{t:04d}.npz" for t in checkpoints)
            required.update(f"{path}/transfer_{stage}.npz" for stage in ("baseline", "final"))
    _require(required <= set(hashes), "Required primary artifact hash coverage")
    for relative, digest in hashes.items():
        path = (output / relative).resolve()
        _require(path.is_relative_to(output) and path.is_file(), f"Unsafe/missing artifact: {relative}")
        _require(_hash(path) == digest, f"Artifact hash mismatch: {relative}")
    panel = np.load(output / "spacing_codes.npy", allow_pickle=False)
    _require(np.issubdtype(panel.dtype, np.floating) and np.array_equal(panel, _panel()), "Exact four-map spacing panel")
    _require(np.isfinite(panel).all() and np.all(np.diff(panel, axis=1) > 0) and
             np.all(panel[:, 0] == 1) and np.all(panel[:, -1] == 8), "Strict rank and global range preservation")
    _require(np.array_equal(panel[:, [1, 3]], np.tile([2, 4], (4, 1))) and
             np.array_equal(panel[2], np.arange(1, 9)), "Anchors and identity map")
    geometry = _npz(output / "code_geometry.npz", {"pairs", "quantity_ratios", "quantity_differences", "code_gaps",
                    "normalized_code_gaps", "code_midpoints", "normalized_code_midpoints", "code_order_agreement"})
    _require(np.issubdtype(geometry["pairs"].dtype, np.integer) and np.array_equal(geometry["pairs"], np.asarray(_PAIRS)),
             "Code geometry pair order")
    gaps = np.asarray([[mapping[b - 1] - mapping[a - 1] for a, b in _PAIRS] for mapping in panel])
    midpoints = np.asarray([[(mapping[b - 1] + mapping[a - 1]) / 2 for a, b in _PAIRS] for mapping in panel])
    for field, value in (("quantity_ratios", [a / b for a, b in _PAIRS]), ("quantity_differences", [b - a for a, b in _PAIRS]),
                         ("code_gaps", gaps), ("normalized_code_gaps", gaps / 8), ("code_midpoints", midpoints),
                         ("normalized_code_midpoints", midpoints / 8), ("code_order_agreement", np.ones((4, 27)))):
        _near(geometry[field], value, f"Independent code geometry: {field}", atol=0)
    curves = _csv(output / "learning_curves.csv", _CURVE_FIELDS, ("seed", "condition", "trial", "split"))
    _require(set(curves) == {(seed, condition, t, split) for seed in seeds for condition in conditions
                            for t in checkpoints for split in ("iid", "ood")}, "Learning curve exact design coverage")
    records = _json(output / "controls.json")
    controls = {(record["seed"], record["kind"]): record for record in records}
    _require(len(controls) == len(records) and set(controls) == {(seed, kind) for seed in seeds
             for kind in ("real", "rewired", "ann")}, "Topology control coverage")
    resource_rows = _json(output / "model_resources.json")
    resources = {(row["seed"], row["condition"]): row for row in resource_rows}
    _require(len(resources) == len(resource_rows) and set(resources) == {(seed, condition) for seed in seeds for condition in conditions},
             "Resource metadata exact condition coverage")
    fingerprints, measurements, transfer = set(), {}, {}
    trial_count = snapshot_count = transfer_count = matrix_entries = case_entries = cache_entries = 0
    split_specs = [("train", (2, 4), n, 31), ("iid", (2, 4), ev, 32), ("ood", (2, 4), ev, 33)]
    split_specs += [(f"pair_{a}_{b}", (a, b), ev, 40 + i) for i, (a, b) in enumerate(_PAIRS)]
    for seed in seeds:
        seed_path = output / f"seed_{seed}"
        stimuli = _npz(seed_path / "stimuli.npz", {f"{name}_{field}" for name, _, _, _ in split_specs
                       for field in ("counts", "nuisance", "family_ids")} | {"action_uniforms", "reward_uniforms"})
        datasets = {}
        for name, pair, size, offset in split_specs:
            datasets[name] = _dataset(stimuli, name, size, pair, fingerprints)
            for field, expected in _regenerate_dataset(seed * 100 + offset, name, size, pair).items():
                _require(np.array_equal(datasets[name][field], expected), f"Registered data RNG: {name}/{field}")
        uniforms = np.random.default_rng(seed + 280000).random(n)
        reward_uniforms = np.random.default_rng(seed + 290000).random(n)
        _require(np.array_equal(stimuli["action_uniforms"], uniforms), "Shared action uniform RNG")
        _require(np.array_equal(stimuli["reward_uniforms"], reward_uniforms), "Independent random reward uniform RNG")
        for name in ("train", "iid", "ood"):
            original_input = _inputs(datasets[name], np.arange(1, 9))
            _require(all(np.array_equal(original_input, _inputs(datasets[name], mapping)) for mapping in panel),
                     "All four anchored training/IID/OOD inputs bitwise identical")
        trials = _csv(seed_path / "trials.csv", _TRIAL_FIELDS, ("seed", "condition", "trial"))
        _require(set(trials) == {(seed, condition, t) for condition in conditions for t in range(1, n + 1)},
                 "One actual fit per condition; exact trial design")
        trial_count += len(trials)
        graphs = _topologies(seed_path, {kind: controls[seed, kind] for kind in ("real", "rewired", "ann")},
                             seed, n_kc, n_mbon, source_metadata)
        _require(all(np.array_equal(graphs["real"][field], source_graph[field]) for field in source_graph),
                 "Stored real topology equals independently read source")
        rng = np.random.default_rng(seed + 70000)
        network_encoder = rng.normal(0, .8, (n_kc, 7))
        network_bias = rng.uniform(-.25, .25, n_kc)
        probe = _inputs(datasets["iid"], np.arange(1, 9))[:probe_n]
        training_inputs = _inputs(datasets["train"], np.arange(1, 9))
        for condition in conditions:
            kind, plastic = _FLAGS[condition]
            linear = kind == "linear"
            if linear:
                encoder, bias = np.empty((0, 7)), np.empty(0)
                pre, post = np.empty(0, dtype=int), np.empty(0, dtype=int)
                w0, initial_head = np.empty(0), np.zeros(7)
            else:
                encoder, bias = network_encoder, network_bias
                graph = graphs[kind]
                pre, post = graph["pre"], graph["post"]
                normalizer = np.maximum(np.bincount(post, weights=graph["counts"], minlength=n_mbon), 1)
                w0 = graph["counts"] / normalizer[post]
                initial_head = np.zeros(n_mbon)
            train_kc, _, train_raw = _raw_features(training_inputs, encoder, bias, pre, post, w0, initial_head.size, linear)
            initial_rms = float(np.sqrt(np.mean(np.sum((train_raw[:, 1] - train_raw[:, 0]) ** 2, axis=-1))))
            _require(np.isfinite(initial_rms) and initial_rms > 0, "Positive train-only initial feature-gap RMS")
            scale = 1. if condition == "linear_raw" else initial_rms
            expected_resource = {"seed": seed, "condition": condition, "feature_scale": scale,
                                 "initial_feature_gap_rms": initial_rms, "readout_parameters": initial_head.size,
                                 "internal_parameters": len(w0) if plastic else 0,
                                 "reward_rule": "action-independent Bernoulli(.5)" if condition == "linear_random_reward"
                                                else "chosen larger quantity=1;other=0"}
            _equal(resources[seed, condition], expected_resource, f"Independent resource metadata: {seed}/{condition}")
            path = seed_path / condition
            _require({p.name for p in path.glob("*.npz")} == {f"checkpoint_{t:04d}.npz" for t in checkpoints}
                     | {"transfer_baseline.npz", "transfer_final.npz"}, "Exact frozen-state file coverage")
            initial_kc, initial_mb, initial_decision, _ = _state(probe, encoder, bias, pre, post, w0,
                                                                 initial_head, scale, linear)
            snapshots = {}
            for t in checkpoints:
                snap = _npz(path / f"checkpoint_{t:04d}.npz", _SAVED_FIELDS)
                snapshots[t] = snap; snapshot_count += 1
                for field, expected in (("encoder", encoder), ("bias", bias), ("pre", pre), ("post", post)):
                    _require(np.array_equal(snap[field], expected), f"Frozen encoder/bias/topology: {field}")
                for field, expected in (("feature_scale", scale), ("initial_feature_gap_rms", initial_rms)):
                    _require(snap[field].shape == (), f"Scalar normalization schema: {field}")
                    _near(snap[field], expected, f"Train-only fixed normalization: {field}")
                weights, head = snap["internal_weights"], snap["readout_weights"]
                _require(weights.shape == w0.shape and head.shape == initial_head.shape and
                         all(np.isfinite(snap[field]).all() for field in _SAVED_FIELDS), "Finite snapshot parameter schema")
                _require(np.all((weights >= 0) & (weights <= 2 * w0 + 1e-12)), "Internal clipping bounds")
                if not plastic:
                    _require(np.array_equal(weights, w0), "Fixed internal weights changed")
                kc, mb, decision, probability = _state(probe, encoder, bias, pre, post, weights, head, scale, linear)
                for field, expected in (("kc_activity", kc), ("mbon_activity", mb), ("decision_activity", decision),
                                        ("choice_probability", probability)):
                    _near(snap[field], expected, f"Saved probe state: {field}")
                changes = {"internal_change_l2": float(np.linalg.norm(weights - w0)),
                           "readout_change_l2": float(np.linalg.norm(head - initial_head)),
                           "probe_kc_change_l2": float(np.linalg.norm(kc - initial_kc)),
                           "probe_mbon_change_l2": float(np.linalg.norm(mb - initial_mb)),
                           "probe_decision_change_l2": float(np.linalg.norm(decision - initial_decision))}
                for split in ("iid", "ood"):
                    data = datasets[split]
                    _, _, _, p = _state(_inputs(data, np.arange(1, 9)), encoder, bias, pre, post, weights,
                                        head, scale, linear)
                    metrics = _metrics(p, data["counts"])
                    measurements[seed, condition, t, split] = metrics
                    for field, expected in {**metrics, **changes}.items():
                        _near(_number(curves[seed, condition, t, split], field), expected, f"Curve: {t}/{split}/{field}")
                if t in (0, n):
                    stage = "baseline" if t == 0 else "final"
                    saved = _npz(path / f"transfer_{stage}.npz", _TRANSFER_FIELDS)
                    transfer[seed, condition, stage] = _transfer(saved, datasets, panel, encoder, bias, pre, post,
                                                               weights, head, scale, linear)
                    transfer_count += 1; matrix_entries += 4 * 27
                    case_entries += 4 * 27 * ev; cache_entries += 27 * ev * 2 * len(np.unique(panel))
            weights, head = w0.copy(), initial_head.copy()
            _near(snapshots[0]["internal_weights"], weights, "Initialized internal weights")
            _require(np.array_equal(snapshots[0]["readout_weights"], head), "Exact zero initial readout weights")
            _require(np.array_equal(snapshots[0]["choice_probability"], np.full(probe_n, .5)), "Exact unbiased baseline policy")
            data = datasets["train"]
            for i, x in enumerate(training_inputs):
                row = trials[seed, condition, i + 1]
                _require(row["family_id"] == str(data["family_ids"][i]) and
                         _integer(row, "left_count") == data["counts"][i, 0] and
                         _integer(row, "right_count") == data["counts"][i, 1], "Shared training stimulus identity")
                action, reward = _integer(row, "action"), _integer(row, "reward")
                _require(action in (0, 1) and reward in (0, 1), "Binary action/reward")
                kc, mb, raw = _raw_features(x, encoder, bias, pre, post, weights, head.size, linear)
                decision = raw / scale
                probability = float(1 / (1 + np.exp(-np.clip((decision[1] - decision[0]) @ head, -40, 40))))
                _near(_number(row, "p_right"), probability, "Pre-update trial probability")
                _require(action == int(uniforms[i] < probability), "Regenerated shared action choice")
                target = int(data["counts"][i, 1] > data["counts"][i, 0])
                expected_reward = int(reward_uniforms[i] < .5) if condition == "linear_random_reward" else int(action == target)
                _require(reward == expected_reward, "Registered contingent or independent random reward")
                factor = (action - probability) * (reward - .5)
                _near(_number(row, "optimizer_factor"), factor, "Trial optimizer factor")
                next_head = head + cfg["head_lr"] * factor * (decision[1] - decision[0])
                if plastic:
                    eligibility = head[post] * (kc[1, pre] - kc[0, pre]) / scale
                    next_weights = np.minimum(np.maximum(weights + cfg["internal_lr"] * factor * eligibility, 0), 2 * w0)
                else:
                    next_weights = weights.copy()
                expected_trial = {"internal_step_l2": np.linalg.norm(next_weights - weights),
                                  "readout_step_l2": np.linalg.norm(next_head - head),
                                  "kc_left_l2": np.linalg.norm(kc[0]), "kc_right_l2": np.linalg.norm(kc[1]),
                                  "mbon_left_l2": np.linalg.norm(mb[0]), "mbon_right_l2": np.linalg.norm(mb[1]),
                                  "decision_left_l2": np.linalg.norm(decision[0]), "decision_right_l2": np.linalg.norm(decision[1])}
                for field, expected in expected_trial.items():
                    _near(_number(row, field), expected, f"Manual trial update/activity: {field}")
                if not plastic or i == 0:
                    _require(_number(row, "internal_step_l2") == 0, "Fixed or zero-old-head first internal trial change")
                weights, head = next_weights, next_head
                if (i + 1) % every == 0:
                    _near(snapshots[i + 1]["internal_weights"], weights, "Replayed internal checkpoint; evaluation frozen")
                    _near(snapshots[i + 1]["readout_weights"], head, "Replayed readout checkpoint; evaluation frozen")
        print(f"audited seed {seed} ({seeds.index(seed) + 1}/{len(seeds)})", flush=True)
    _summary(cfg, measurements, transfer, summary)
    report["counts"].update(artifact_hashes=len(hashes), source_hashes=len(registration["source_hashes"]),
                            seeds=len(seeds), conditions=len(conditions), actual_fits=len(seeds) * len(conditions),
                            trials=trial_count, curve_rows=len(curves), checkpoints=snapshot_count, transfer_files=transfer_count,
                            frozen_evaluation_maps=4, transfer_probability_entries=matrix_entries,
                            case_probability_entries=case_entries, candidate_score_entries=cache_entries,
                            primary_unseen_pairs=15, anchored_pairs=12, canonical_fingerprints=len(fingerprints))
    report["checks"].extend([
        "Required primary hashes, config/source byte freeze, source file set, and local registration/run chronology",
        "All30 raw splits independently regenerated, complete joint nuisance crossing, canonical disjointness and anchored map identity",
        "Anatomical source independently parsed; real/rewired/ANN topologies and control RNG independently reconstructed",
        "Every initial feature-gap RMS reconstructed solely from training inputs and initial features; scale fixed at all checkpoints",
        "All readouts initialized exactly to zero; linear models preserve empty biological-state arrays",
        "Every contingent/random reward, shared action choice, trial probability and reward-only update independently replayed",
        "Internal gradients use old readout weights and fixed-scale chain rule; clipping and first zero-head internal step independently checked",
        "Every checkpoint's internal/readout parameters, KC/MBON/decision activations and IID/OOD responses independently reconstructed",
        "All baseline/final candidate scores, four-map probabilities, pair means and greedy responses independently reconstructed",
        "Acquisition, acquired-only transfer gates, all-seed aggregates and paired contingent-minus-random reward contrast independently checked",
    ])


def verify(output: Path) -> dict:
    """Write the audit report separately without modifying primary artifacts."""
    output = Path(output).resolve()
    report = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "method": "independent numpy/stdlib arithmetic replay; only prior independent audit helpers reused",
              "input_scope": "reward-only learners; artificial scalar input; fixed train-derived initial feature-gap scale",
              "counts": {}, "checks": [], "errors": [], "not_verified": [
                  "Visual extraction of numerosity, LC11/DAN dynamics, or shared human/fly neural mechanisms",
                  "Remote private Git publication authenticity; local chronology is self-recorded",
                  "Statistical significance, mathematical rule discovery, or educational implications",
                  "Equal feature-gap scale or equal trial budgets as complete equivalence of model capacity or optimization",
                  "Frozen spacing maps as independent learners or computational seeds as biological individuals"]}
    try:
        _require(output.is_dir(), f"Output directory missing: {output}")
        _audit(output, report)
        report["passed"] = True
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
