"""Independent arithmetic audit of calibrated score-distillation diagnostics.

The only project imports are immutable arithmetic/data-validation helpers from
the earlier independent auditor. No learner, update, producer, or summary code
is invoked. Code maps are frozen evaluations of one trained policy per family.
"""
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import csv
import json

import numpy as np

from .encoding_audit import (
    _require, _near, _integer, _number, _json, _hash, _csv, _npz,
    _regenerate_dataset, _dataset, _inputs, _mbon, _forward, _metrics,
    _topologies, _cache,
)


_FREE = {1, 3, 5, 6, 7, 8}
_PAIRS = tuple(p for p in combinations(range(1, 9), 2) if p != (2, 4))
_UNSEEN = np.asarray([i for i, pair in enumerate(_PAIRS) if set(pair) <= _FREE])
_ANCHORED = np.asarray([i for i, pair in enumerate(_PAIRS) if not set(pair) <= _FREE])
_FOCAL = _PAIRS.index((5, 7))
_FLAGS = {"fixed": ("real", False), "plastic": ("real", True),
          "rewired_fixed": ("rewired", False), "rewired_plastic": ("rewired", True),
          "ann_fixed": ("ann", False), "ann_plastic": ("ann", True)}
_ALPHAS = (.25, .5, 1., 1.5)
_TRIAL_FIELDS = {"seed", "condition", "trial", "family_id", "left_count", "right_count", "action", "reward",
                 "p_right", "optimizer_factor", "internal_step_l2", "readout_step_l2", "kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2"}
_CURVE_FIELDS = {"seed", "condition", "trial", "split", "p_target", "greedy_accuracy", "internal_change_l2",
                 "readout_change_l2", "probe_kc_change_l2", "probe_mbon_change_l2"}
_SAVED_FIELDS = {"internal_weights", "readout_weights", "kc_activity", "mbon_activity", "choice_probability", "encoder", "bias", "pre", "post"}
_TRANSFER_FIELDS = {"code_values", "score_cache", "map_p_target", "map_greedy_accuracy", "map_p_right"}
_TRANSFER_FIELDS |= {"teacher_map_scores", "teacher_map_margin"}
for _reference in ("affine", "scalar"):
    _TRANSFER_FIELDS.update(f"{_reference}_{field}" for field in (
        "map_scores", "map_margin", "map_p_right", "map_p_target", "map_greedy_accuracy", "probability_mae"))
_REFERENCE_FIELDS = {"affine_coefficients", "affine_rank", "affine_singular_values", "affine_calibration_score_rmse",
                     "scalar_coefficients", "scalar_anchor_means", "scalar_calibration_score_rmse", "calibration_inputs", "calibration_scores"}


def _panel():
    return np.asarray([[1, 2, 3, 4, 6 - alpha, 6, 6 + alpha, 8] for alpha in _ALPHAS], dtype=float)


def _teacher_scores(inputs, encoder, bias, pre, post, weights, head):
    """Full anatomical scatter, retaining unclipped absolute calibration scores."""
    kc = np.maximum(np.matmul(inputs, encoder.T) + bias, 0)
    return np.matmul(_mbon(kc, pre, post, weights, len(head)), head)


def _independent_affine_fit(design, scores):
    """QR fit and independent SVD rank; do not invoke producer's least squares."""
    singular = np.linalg.svd(design, compute_uv=False)
    tolerance = singular.max() * max(design.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > tolerance))
    _require(design.ndim == 2 and design.shape[1] == 8 and rank == 8,
             "Independent calibration design must have rank8")
    orthogonal, triangular = np.linalg.qr(design, mode="reduced")
    coefficients = np.linalg.solve(triangular, orthogonal.T @ scores)
    return coefficients, rank, singular


def _source_projection(root, graph):
    """Read anatomical CSV directly, without graph.load_projection."""
    directory = root / graph
    provenance = _json(directory / "provenance.json")
    for filename, field in (("edges.csv", "edges_sha256"), ("neurons.json", "neurons_sha256"), ("annotations.csv", "roles_sha256")):
        _require(_hash(directory / filename) == provenance[field], f"Pinned source provenance: {filename}")
    ids = _json(directory / "neurons.json")
    with (directory / "annotations.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(set(ids)) == len(ids) == len(rows) and {row["root_id"] for row in rows} == set(ids), "Anatomical neuron IDs")
    for row in rows:
        if row["role"] in ("KC", "MBON"):
            _require(row["cell_class"] == {"KC": "Kenyon_Cell", "MBON": "MBON"}[row["role"]], "Anatomical class/role")
    kc = sorted(row["root_id"] for row in rows if row["role"] == "KC")
    mb = sorted(row["root_id"] for row in rows if row["role"] == "MBON")
    ki, mi = {identifier: i for i, identifier in enumerate(kc)}, {identifier: i for i, identifier in enumerate(mb)}
    pre, post, counts = [], [], []
    with (directory / "edges.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["pre"] in ki and row["post"] in mi:
                _require(int(row["sign"]) == 1 and int(row["count"]) > 0, "Positive anatomical KC contacts")
                pre.append(ki[row["pre"]]); post.append(mi[row["post"]]); counts.append(float(row["count"]))
    _require(pre and len(set(zip(pre, post))) == len(pre), "Nonempty unique anatomical pairs")
    metadata = {"source": provenance, "restricted_pairs": len(pre), "contacts": sum(counts),
                "active_kcs": len(set(pre)), "active_mbons": len(set(post)),
                "scope": "one-hop KC->MBON; DAN/APL/recurrent paths excluded"}
    return {"pre": np.asarray(pre), "post": np.asarray(post), "counts": np.asarray(counts)}, len(kc), len(mb), metadata


def _reference(saved, calibration, encoder, bias, pre, post, weights, head):
    inputs = _inputs(calibration, np.arange(1, 9))
    scores = _teacher_scores(inputs, encoder, bias, pre, post, weights, head)
    _near(saved["calibration_inputs"], inputs, "Independent calibration inputs", atol=0)
    _near(saved["calibration_scores"], scores, "Independent privileged calibration teacher scores")
    design = np.column_stack((np.ones(inputs.size // 7), inputs.reshape(-1, 7)))
    coefficients, rank, singular = _independent_affine_fit(design, scores.reshape(-1))
    _require(saved["affine_rank"].shape == () and np.issubdtype(saved["affine_rank"].dtype, np.integer)
             and int(saved["affine_rank"]) == rank == 8, "Independent affine calibration rank8")
    anchor_means = np.asarray([scores[calibration["counts"] == count].mean() for count in (2, 4)])
    slope = (anchor_means[1] - anchor_means[0]) / .25
    scalar = np.asarray([anchor_means[0] - .25 * slope, slope])
    expected = {"affine_coefficients": coefficients, "affine_rank": np.asarray(rank), "affine_singular_values": singular,
                "affine_calibration_score_rmse": np.asarray(np.sqrt(np.mean((design @ coefficients - scores.reshape(-1)) ** 2))),
                "scalar_coefficients": scalar, "scalar_anchor_means": anchor_means,
                "scalar_calibration_score_rmse": np.asarray(np.sqrt(np.mean((scalar[0] + scalar[1] * inputs[..., 0] - scores) ** 2))),
                "calibration_inputs": inputs, "calibration_scores": scores}
    for field, value in expected.items():
        _near(saved[field], value, f"Independent calibration-only reference fit: {field}")
    return expected


def _transfer(saved, reference, datasets, panel, encoder, bias, pre, post, weights, head):
    levels = np.unique(panel)
    _near(saved["code_values"], levels, "Unique mapped scalar values", atol=0)
    cache = _cache(datasets, levels, encoder, bias, pre, post, weights, head)
    _near(saved["score_cache"], cache, "Every independent frozen candidate score")
    means, greedy, cases, teacher_scores = [], [], [], []
    for j, pair in enumerate(_PAIRS):
        data = datasets[f"pair_{pair[0]}_{pair[1]}"]
        counts, n = data["counts"], len(data["counts"])
        indices = np.searchsorted(levels, panel[:, counts - 1])
        offsets = np.arange(n)[None, :, None] * (2 * len(levels)) + np.arange(2)[None, None, :] * len(levels)
        scores = cache[j].reshape(-1)[offsets + indices]
        probability = 1 / (1 + np.exp(-np.clip(scores[..., 1] - scores[..., 0], -40, 40)))
        target = (counts[:, 1] > counts[:, 0]).astype(int)[None, :]
        correct = np.where(np.abs(probability - .5) < 1e-12, .5, ((probability > .5).astype(int) == target).astype(float))
        means.append(np.mean(np.where(target == 1, probability, 1 - probability), axis=1))
        greedy.append(correct.mean(axis=1)); cases.append(probability); teacher_scores.append(scores)
    expected = {"map_p_target": np.asarray(means).T, "map_greedy_accuracy": np.asarray(greedy).T,
                "map_p_right": np.transpose(np.asarray(cases), (1, 0, 2)),
                "teacher_map_scores": np.transpose(np.asarray(teacher_scores), (1, 0, 2, 3))}
    expected["teacher_map_margin"] = expected["teacher_map_scores"][..., 1] - expected["teacher_map_scores"][..., 0]
    for kind in ("affine", "scalar"):
        coefficients = reference[f"{kind}_coefficients"]
        outputs = []
        for mapping in panel:
            candidate_scores = []
            for a, b in _PAIRS:
                inputs = _inputs(datasets[f"pair_{a}_{b}"], mapping)
                score = coefficients[0] + (inputs @ coefficients[1:] if kind == "affine" else inputs[..., 0] * coefficients[1])
                candidate_scores.append(score)
            outputs.append(np.asarray(candidate_scores))
        outputs = np.asarray(outputs)
        margin = outputs[..., 1] - outputs[..., 0]
        probability = 1 / (1 + np.exp(-np.clip(margin, -40, 40)))
        target = np.asarray([(datasets[f"pair_{a}_{b}"]["counts"][:, 1] > datasets[f"pair_{a}_{b}"]["counts"][:, 0]).astype(int)
                             for a, b in _PAIRS])[None, ...]
        decisions = np.where(np.abs(probability - .5) < 1e-12, .5,
                             ((probability > .5).astype(int) == target).astype(float))
        expected.update({f"{kind}_map_scores": outputs, f"{kind}_map_margin": margin, f"{kind}_map_p_right": probability,
                         f"{kind}_map_p_target": np.where(target == 1, probability, 1 - probability).mean(axis=-1),
                         f"{kind}_map_greedy_accuracy": decisions.mean(axis=-1),
                         f"{kind}_probability_mae": np.abs(probability - expected["map_p_right"]).mean(axis=-1)})
    for field, value in expected.items():
        _near(saved[field], value, f"Every mapped transfer entry: {field}")
    return expected


def _reference_metrics(transfer, kind):
    probability = transfer[f"{kind}_map_p_right"]
    teacher = transfer["map_p_right"]
    difference = transfer[f"{kind}_map_margin"] - transfer["teacher_map_margin"]
    score_difference = transfer[f"{kind}_map_scores"] - transfer["teacher_map_scores"]
    def decision(p):
        return np.where(np.abs(p - .5) < 1e-12, .5, (p > .5).astype(float))
    return {"unseen_probability_mae": float(np.abs(probability[:, _UNSEEN] - teacher[:, _UNSEEN]).mean()),
            "unseen_margin_mae": float(np.abs(difference[:, _UNSEEN]).mean()),
            "unseen_margin_rmse": float(np.sqrt(np.mean(difference[:, _UNSEEN] ** 2))),
            "unseen_score_rmse": float(np.sqrt(np.mean(score_difference[:, _UNSEEN] ** 2))),
            "unseen_greedy_disagreement": float(np.abs(decision(probability[:, _UNSEEN]) - decision(teacher[:, _UNSEEN])).mean()),
            "anchored_probability_mae": float(np.abs(probability[:, _ANCHORED] - teacher[:, _ANCHORED]).mean()),
            "all_probability_mae": float(np.abs(probability - teacher).mean()),
            "focal_probability_mae": float(np.abs(probability[:, _FOCAL] - teacher[:, _FOCAL]).mean()),
            "focal_reference_p_correct": transfer[f"{kind}_map_p_target"][:, _FOCAL].tolist(),
            "focal_reference_greedy_accuracy": transfer[f"{kind}_map_greedy_accuracy"][:, _FOCAL].tolist(),
            "unseen_reference_p_correct": transfer[f"{kind}_map_p_target"][:, _UNSEEN].mean(axis=1).tolist()}


def _equal_claim(actual, expected, label):
    if type(expected) is bool or expected is None:
        _require(actual is expected, f"Independent summary gate: {label}")
    elif isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), f"Independent summary schema: {label}")
        for key, value in expected.items():
            _equal_claim(actual[key], value, f"{label}/{key}")
    else:
        _near(actual, expected, f"Independent summary value: {label}")


def _summary(cfg, measurements, transfer, summary):
    _require(set(summary) == {"status", "scope", "outcomes", "family_verdicts", "actual_training_trials",
                             "frozen_evaluation_maps", "reference_models", "interpretation"}, "Summary top-level schema")
    criteria, n = cfg["criteria"], cfg["trials"]
    rows = {(row["seed"], row["condition"]): row for row in summary["outcomes"]}
    keys = {(seed, condition) for seed in cfg["seeds"] for condition in cfg["conditions"]}
    _require(len(rows) == len(summary["outcomes"]) and set(rows) == keys, "One summary row per actual fit")
    expected_rows = {}
    for seed, condition in sorted(keys):
        before, after = (measurements[seed, condition, t, "iid"]["p_target"] for t in (0, n))
        acquired = bool(after >= criteria["acquisition_accuracy"] and after - before >= criteria["acquisition_gain"])
        baseline, final = (transfer[seed, condition, stage] for stage in ("baseline", "final"))
        metrics = {stage: {kind: _reference_metrics(transfer[seed, condition, stage], kind) for kind in ("affine", "scalar")}
                   for stage in ("baseline", "final")}
        fidelity = bool(metrics["final"]["affine"]["unseen_probability_mae"] <= criteria["probability_mae"])
        expected = {"before_accuracy": before, "after_accuracy": after, "gain": after - before, "acquired": acquired,
                    "ood_accuracy": measurements[seed, condition, n, "ood"]["p_target"],
                    "head_updates": n, "internal_updates": n if _FLAGS[condition][1] else 0,
                    **metrics, "focal_model_before": baseline["map_p_target"][:, _FOCAL].tolist(),
                    "focal_model_after": final["map_p_target"][:, _FOCAL].tolist(),
                    "focal_model_greedy_after": final["map_greedy_accuracy"][:, _FOCAL].tolist(),
                    "fidelity_passed": fidelity, "qualified_fidelity": bool(acquired and fidelity)}
        _require(set(rows[seed, condition]) == set(expected) | {"seed", "condition"}, "Summary outcome schema")
        for field, value in expected.items():
            _equal_claim(rows[seed, condition][field], value, field)
        expected_rows[seed, condition] = expected
    verdicts = {row["condition"]: row for row in summary["family_verdicts"]}
    _require(len(verdicts) == len(summary["family_verdicts"]) and set(verdicts) == set(cfg["conditions"]), "Family verdict coverage")
    for condition in cfg["conditions"]:
        selected = [expected_rows[seed, condition] for seed in cfg["seeds"]]
        errors = [row["final"]["affine"]["unseen_probability_mae"] for row in selected]
        qualified = [row["final"]["affine"]["unseen_probability_mae"] for row in selected if row["acquired"]]
        passing = sum(row["fidelity_passed"] for row in selected)
        def mean(kind, field):
            return float(np.mean([row["final"][kind][field] for row in selected]))
        expected = {"n_seeds": len(selected), "mean_affine_probability_mae": float(np.mean(errors)),
                    "sd_affine_probability_mae": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.,
                    "passing_seed_count": passing,
                    "criterion_met": bool(np.mean(errors) <= criteria["probability_mae"] and passing / len(selected) >= criteria["seed_pass_fraction"]),
                    "acquisition_count": sum(row["acquired"] for row in selected),
                    "qualified_count": sum(row["qualified_fidelity"] for row in selected),
                    "qualified_mean_affine_probability_mae": float(np.mean(qualified)) if qualified else None,
                    "mean_scalar_probability_mae": mean("scalar", "unseen_probability_mae"),
                    "mean_affine_margin_rmse": mean("affine", "unseen_margin_rmse"),
                    "mean_scalar_margin_rmse": mean("scalar", "unseen_margin_rmse"),
                    "mean_affine_greedy_disagreement": mean("affine", "unseen_greedy_disagreement"),
                    "mean_scalar_greedy_disagreement": mean("scalar", "unseen_greedy_disagreement")}
        _require(set(verdicts[condition]) == set(expected) | {"condition"}, "Family verdict schema")
        for field, value in expected.items():
            _equal_claim(verdicts[condition][field], value, field)
    _require(summary["actual_training_trials"] == len(cfg["seeds"]) * len(cfg["conditions"]) * n
             and summary["frozen_evaluation_maps"] == 4 and summary["reference_models"] == 2,
             "One actual fit per seed/family; maps are evaluations and references are privileged score-distillation models")


def _source_freeze(output, metadata, registration, cfg):
    root = (output / metadata["source_root"]).resolve()
    _require(root.is_dir(), "Registered source root missing; use the registered code checkout")
    config = Path(metadata["config_path"])
    config = (root / config).resolve() if not config.is_absolute() else config.resolve()
    _require(config.is_file() and _hash(config) == registration["config_sha256"] and _json(config) == cfg, "Registered config bytes/content")
    expected = {path.relative_to(root).as_posix() for path in (root / "src").rglob("*.py") if path.is_file()}
    expected.update(path.relative_to(root).as_posix() for path in (root / cfg["graph"]).glob("*") if path.is_file())
    expected.update(("docs/linearity-protocol.md", "docs/linearity-background.md", "pyproject.toml", "requirements-lock.txt", "data/source-manifest.json", "data/LICENSE.md"))
    _require(set(registration["source_hashes"]) == expected, "Registered source file set")
    for relative, digest in registration["source_hashes"].items():
        source = (root / relative).resolve()
        _require(source.is_relative_to(root) and source.is_file() and _hash(source) == digest, f"Registered source byte hash: {relative}")
    return root


def _audit(output, report):
    metadata, registration, summary = (_json(output / name) for name in ("metadata.json", "registration.json", "summary.json"))
    cfg = metadata["config"]
    _require(metadata.get("status") == summary.get("status") == "completed", "Study did not complete")
    _require(cfg == registration["config"] and summary["scope"] == cfg["scope"], "Config/scope consistency")
    _require(_hash(output / "registration.json") == metadata["registration_sha256"], "Registration hash")
    times = [datetime.fromisoformat(record[field]) for record, field in ((registration, "registered_at_utc"), (metadata, "started_at_utc"), (metadata, "completed_at_utc"))]
    _require(all(value.tzinfo is not None for value in times) and times == sorted(times), "Local registration/run chronology")
    _require(cfg["scope"] in ("preregistered_diagnostic", "integration_test"), "Study scope")
    seeds, conditions = cfg["seeds"], cfg["conditions"]
    _require(seeds and all(type(seed) is int and seed >= 0 for seed in seeds) and len(set(seeds)) == len(seeds), "Unique nonnegative seeds")
    _require(conditions and all(condition in _FLAGS for condition in conditions) and len(set(conditions)) == len(conditions), "Unique supported model families")
    _require(cfg["train_pair"] == [2, 4] and cfg["primary_pairs"] == "fully_unseen15" and tuple(cfg["code_alphas"]) == _ALPHAS
             and cfg["references"] == ["affine", "scalar"], "Anchored training, fully unseen primary pairs, two references and four alphas")
    _require(metadata["rng"] == {"actions": "seed+280000", "encoder": "seed+70000", "rewired": "seed+50000", "ann": "seed+60000", "data": "seed*100+31,+32,+33,+34 calibration,+40+pair_index"}, "Registered RNG offsets")
    _require(metadata["reference_scope"] == "privileged calibration on teacher scores at2/4; no independent reward learning", "Privileged reference calibration scope")
    n, every, ev, cal, probe_n = (cfg[key] for key in ("trials", "checkpoint_every", "eval_trials", "calibration_trials", "probe_rows"))
    _require(all(type(value) is int and value > 0 for value in (n, every, ev, cal, probe_n))
             and n % 4 == ev % 4 == cal % 4 == n % every == 0 and cal >= 16 and probe_n <= ev, "Budget/crossing validity")
    _require(all(type(cfg[key]) in (int, float) and np.isfinite(cfg[key]) and cfg[key] >= 0 for key in ("head_lr", "internal_lr")), "Learning rates")
    criterion_keys = {"acquisition_accuracy", "acquisition_gain", "probability_mae", "seed_pass_fraction"}
    _require(set(cfg["criteria"]) == criterion_keys and all(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1 for value in cfg["criteria"].values()) and cfg["criteria"]["seed_pass_fraction"] > 0, "Criteria schema/ranges")
    if cfg["scope"] == "preregistered_diagnostic":
        _require(seeds == list(range(5001, 5011)) and tuple(conditions) == tuple(_FLAGS), "Registered ten new seeds and six model families")
        _require((n, every, ev, cal, probe_n, cfg["head_lr"], cfg["internal_lr"]) == (400, 100, 128, 128, 8, .1, .01), "Fixed registered experiment resources")
        _require(cfg["criteria"] == {"acquisition_accuracy": .75, "acquisition_gain": .15, "probability_mae": .02, "seed_pass_fraction": .8}, "Fixed registered criteria")
    root = _source_freeze(output, metadata, registration, cfg)
    source_graph, n_kc, n_mbon, source_metadata = _source_projection(root, cfg["graph"])
    _require(metadata["graph"] == source_metadata, "Independent source graph metadata")
    checkpoints = list(range(0, n + 1, every))
    hashes = _json(output / "artifact_hashes.json")
    required = {"metadata.json", "registration.json", "summary.json", "controls.json", "learning_curves.csv", "spacing_codes.npy", "code_geometry.npz"}
    for seed in seeds:
        base = f"seed_{seed}"
        required.update(f"{base}/{name}" for name in ("stimuli.npz", "trials.csv", "real_topology.npz", "rewired_topology.npz", "ann_topology.npz"))
        for condition in conditions:
            path = f"{base}/{condition}"
            required.update(f"{path}/checkpoint_{t:04d}.npz" for t in checkpoints)
            required.update(f"{path}/transfer_{stage}.npz" for stage in ("baseline", "final"))
            required.update(f"{path}/reference_{stage}.npz" for stage in ("baseline", "final"))
    _require(required <= set(hashes), "Required primary artifact hash coverage")
    for relative, digest in hashes.items():
        path = (output / relative).resolve()
        _require(path.is_relative_to(output) and path.is_file(), f"Unsafe/missing artifact: {relative}")
        _require(_hash(path) == digest, f"Artifact hash mismatch: {relative}")
    panel = np.load(output / "spacing_codes.npy", allow_pickle=False)
    _require(np.issubdtype(panel.dtype, np.floating) and np.array_equal(panel, _panel()), "Exact four-map spacing panel")
    _require(np.isfinite(panel).all() and np.all(np.diff(panel, axis=1) > 0) and np.all(panel[:, 0] == 1) and np.all(panel[:, -1] == 8), "Strict rank and global range preservation")
    _require(np.array_equal(panel[:, [1, 3]], np.tile([2, 4], (4, 1))) and np.array_equal(panel[2], np.arange(1, 9)), "Anchors and identity map")
    geometry = _npz(output / "code_geometry.npz", {"pairs", "quantity_ratios", "quantity_differences", "code_gaps", "normalized_code_gaps", "code_midpoints", "normalized_code_midpoints", "code_order_agreement"})
    _require(np.issubdtype(geometry["pairs"].dtype, np.integer) and np.array_equal(geometry["pairs"], np.asarray(_PAIRS)), "Code geometry pair order")
    gaps = np.asarray([[mapping[b - 1] - mapping[a - 1] for a, b in _PAIRS] for mapping in panel])
    midpoints = np.asarray([[(mapping[b - 1] + mapping[a - 1]) / 2 for a, b in _PAIRS] for mapping in panel])
    for field, value in (("quantity_ratios", [a / b for a, b in _PAIRS]), ("quantity_differences", [b - a for a, b in _PAIRS]),
                         ("code_gaps", gaps), ("normalized_code_gaps", gaps / 8), ("code_midpoints", midpoints), ("normalized_code_midpoints", midpoints / 8),
                         ("code_order_agreement", np.ones((4, 27)))):
        _near(geometry[field], value, f"Independent code geometry: {field}", atol=0)
    _near(geometry["code_midpoints"][:, _FOCAL], [6, 6, 6, 6], "Focal code midpoint constant", atol=0)
    _near(geometry["code_gaps"][:, _FOCAL], [.5, 1, 2, 3], "Focal code gaps", atol=0)
    curves = _csv(output / "learning_curves.csv", _CURVE_FIELDS, ("seed", "condition", "trial", "split"))
    _require(set(curves) == {(seed, condition, t, split) for seed in seeds for condition in conditions for t in checkpoints for split in ("iid", "ood")}, "Learning curve exact design coverage")
    records = _json(output / "controls.json")
    controls = {(record["seed"], record["kind"]): record for record in records}
    _require(len(controls) == len(records) and set(controls) == {(seed, kind) for seed in seeds for kind in ("real", "rewired", "ann")}, "Topology control coverage")
    fingerprints, measurements, transfer = set(), {}, {}
    trial_count = snapshot_count = transfer_count = reference_count = matrix_entries = case_entries = cache_entries = 0
    split_specs = [("train", (2, 4), n, 31), ("iid", (2, 4), ev, 32), ("ood", (2, 4), ev, 33), ("calibration", (2, 4), cfg["calibration_trials"], 34)]
    split_specs += [(f"pair_{a}_{b}", (a, b), ev, 40 + i) for i, (a, b) in enumerate(_PAIRS)]
    for seed in seeds:
        seed_path = output / f"seed_{seed}"
        stimuli = _npz(seed_path / "stimuli.npz", {f"{name}_{field}" for name, _, _, _ in split_specs for field in ("counts", "nuisance", "family_ids")} | {"action_uniforms"})
        datasets = {}
        for name, pair, size, offset in split_specs:
            datasets[name] = _dataset(stimuli, name, size, pair, fingerprints)
            for field, expected in _regenerate_dataset(seed * 100 + offset, name, size, pair).items():
                _require(np.array_equal(datasets[name][field], expected), f"Registered data RNG: {name}/{field}")
        uniforms = np.random.default_rng(seed + 280000).random(n)
        _require(np.array_equal(stimuli["action_uniforms"], uniforms), "Shared action uniform RNG")
        for name in ("train", "iid", "ood", "calibration"):
            original_input = _inputs(datasets[name], np.arange(1, 9))
            _require(all(np.array_equal(original_input, _inputs(datasets[name], mapping)) for mapping in panel), "All four anchored training/IID/OOD inputs bitwise identical")
        trials = _csv(seed_path / "trials.csv", _TRIAL_FIELDS, ("seed", "condition", "trial"))
        _require(set(trials) == {(seed, condition, t) for condition in conditions for t in range(1, n + 1)}, "One actual fit per family; exact trial design")
        trial_count += len(trials)
        graphs = _topologies(seed_path, {kind: controls[seed, kind] for kind in ("real", "rewired", "ann")}, seed, n_kc, n_mbon, source_metadata)
        _require(all(np.array_equal(graphs["real"][field], source_graph[field]) for field in source_graph), "Stored real topology equals independently read source")
        rng = np.random.default_rng(seed + 70000)
        encoder, bias, initial_head = rng.normal(0, .8, (n_kc, 7)), rng.uniform(-.25, .25, n_kc), rng.normal(0, .02, n_mbon)
        probe = _inputs(datasets["iid"], np.arange(1, 9))[:probe_n]
        train_kc = np.maximum(_inputs(datasets["train"], np.arange(1, 9)) @ encoder.T + bias, 0)
        for condition in conditions:
            kind, plastic = _FLAGS[condition]
            graph = graphs[kind]; pre, post = graph["pre"], graph["post"]
            normalizer = np.maximum(np.bincount(post, weights=graph["counts"], minlength=n_mbon), 1)
            w0 = graph["counts"] / normalizer[post]
            path = seed_path / condition
            _require({p.name for p in path.glob("*.npz")} == {f"checkpoint_{t:04d}.npz" for t in checkpoints}
                     | {"transfer_baseline.npz", "transfer_final.npz", "reference_baseline.npz", "reference_final.npz"}, "Exact frozen-state file coverage")
            initial_kc, initial_mb, _ = _forward(probe, encoder, bias, pre, post, w0, initial_head)
            snapshots = {}
            for t in checkpoints:
                snap = _npz(path / f"checkpoint_{t:04d}.npz", _SAVED_FIELDS); snapshots[t] = snap; snapshot_count += 1
                for field, expected in (("encoder", encoder), ("bias", bias), ("pre", pre), ("post", post)):
                    _require(np.array_equal(snap[field], expected), f"Frozen encoder/bias/topology: {field}")
                weights, head = snap["internal_weights"], snap["readout_weights"]
                _require(weights.shape == w0.shape and head.shape == initial_head.shape and all(np.isfinite(snap[field]).all() for field in _SAVED_FIELDS), "Finite snapshot parameter schema")
                _require(np.all((weights >= 0) & (weights <= 2 * w0 + 1e-12)), "Internal clipping bounds")
                if not plastic:
                    _require(np.array_equal(weights, w0), "Fixed internal weights changed")
                kc, mb, probability = _forward(probe, encoder, bias, pre, post, weights, head)
                for field, expected in (("kc_activity", kc), ("mbon_activity", mb), ("choice_probability", probability)):
                    _near(snap[field], expected, f"Saved probe state: {field}")
                changes = {"internal_change_l2": float(np.linalg.norm(weights - w0)), "readout_change_l2": float(np.linalg.norm(head - initial_head)),
                           "probe_kc_change_l2": float(np.linalg.norm(kc - initial_kc)), "probe_mbon_change_l2": float(np.linalg.norm(mb - initial_mb))}
                for split in ("iid", "ood"):
                    data = datasets[split]
                    _, _, p = _forward(_inputs(data, np.arange(1, 9)), encoder, bias, pre, post, weights, head)
                    metrics = _metrics(p, data["counts"]); measurements[seed, condition, t, split] = metrics
                    for field, expected in {**metrics, **changes}.items():
                        _near(_number(curves[seed, condition, t, split], field), expected, f"Curve: {t}/{split}/{field}")
                if t in (0, n):
                    stage = "baseline" if t == 0 else "final"
                    reference = _reference(_npz(path / f"reference_{stage}.npz", _REFERENCE_FIELDS), datasets["calibration"],
                                           encoder, bias, pre, post, weights, head)
                    saved = _npz(path / f"transfer_{stage}.npz", _TRANSFER_FIELDS)
                    transfer[seed, condition, stage] = _transfer(saved, reference, datasets, panel, encoder, bias, pre, post, weights, head)
                    reference_count += 1
                    transfer_count += 1; matrix_entries += 4 * 27; case_entries += 4 * 27 * ev; cache_entries += 27 * ev * 2 * len(np.unique(panel))
            weights, head = w0.copy(), initial_head.copy()
            _near(snapshots[0]["internal_weights"], weights, "Initialized internal weights")
            _near(snapshots[0]["readout_weights"], head, "Initialized readout weights")
            data = datasets["train"]
            for i, kc in enumerate(train_kc):
                row = trials[seed, condition, i + 1]
                _require(row["family_id"] == str(data["family_ids"][i]) and _integer(row, "left_count") == data["counts"][i, 0] and _integer(row, "right_count") == data["counts"][i, 1], "Shared training stimulus identity")
                action, reward = _integer(row, "action"), _integer(row, "reward")
                _require(action in (0, 1) and reward in (0, 1), "Binary action/reward")
                mb = _mbon(kc, pre, post, weights, n_mbon)
                probability = float(1 / (1 + np.exp(-np.clip((mb[1] - mb[0]) @ head, -40, 40))))
                _near(_number(row, "p_right"), probability, "Pre-update trial probability")
                _require(action == int(uniforms[i] < probability), "Regenerated shared action choice")
                target = int(data["counts"][i, 1] > data["counts"][i, 0])
                _require(reward == int(action == target), "Reward for larger true quantity")
                factor = (action - probability) * (reward - .5)
                _near(_number(row, "optimizer_factor"), factor, "Trial optimizer factor")
                next_head = head + cfg["head_lr"] * factor * (mb[1] - mb[0])
                eligibility = head[post] * (kc[1, pre] - kc[0, pre])
                next_weights = np.minimum(np.maximum(weights + cfg["internal_lr"] * factor * eligibility, 0), 2 * w0) if plastic else weights.copy()
                expected_trial = {"internal_step_l2": np.linalg.norm(next_weights - weights), "readout_step_l2": np.linalg.norm(next_head - head),
                                  "kc_left_l2": np.linalg.norm(kc[0]), "kc_right_l2": np.linalg.norm(kc[1]), "mbon_left_l2": np.linalg.norm(mb[0]), "mbon_right_l2": np.linalg.norm(mb[1])}
                for field, expected in expected_trial.items():
                    _near(_number(row, field), expected, f"Manual trial update/activity: {field}")
                if not plastic:
                    _require(_number(row, "internal_step_l2") == 0, "Fixed internal trial change")
                weights, head = next_weights, next_head
                if (i + 1) % every == 0:
                    _near(snapshots[i + 1]["internal_weights"], weights, "Replayed internal checkpoint; evaluation frozen")
                    _near(snapshots[i + 1]["readout_weights"], head, "Replayed readout checkpoint; evaluation frozen")
        print(f"audited seed {seed} ({seeds.index(seed) + 1}/{len(seeds)})", flush=True)
    _summary(cfg, measurements, transfer, summary)
    report["counts"].update(artifact_hashes=len(hashes), source_hashes=len(registration["source_hashes"]), seeds=len(seeds), conditions=len(conditions),
                            actual_fits=len(seeds) * len(conditions), trials=trial_count, curve_rows=len(curves), checkpoints=snapshot_count, transfer_files=transfer_count,
                            reference_files=reference_count, reference_models=2,
                            calibration_candidate_score_entries=reference_count * cal * 2,
                            frozen_evaluation_maps=4, transfer_probability_entries=matrix_entries, case_probability_entries=case_entries, candidate_score_entries=cache_entries,
                            reference_case_probability_entries=case_entries * 2, raw_margin_entries=case_entries * 3,
                            raw_selected_candidate_score_entries=case_entries * 3 * 2,
                            focal_pairs=1, secondary_unseen_pairs=15, anchored_pairs=12, canonical_fingerprints=len(fingerprints))
    report["checks"].extend([
        "Required primary byte hashes, config/source file-set and byte freeze, and local registration/run chronology",
        "Four exact order-preserving maps; constant anchors/range and focal midpoint; true quantity ratio and mapped code gap kept separate",
        "All31 raw splits regenerated; calibration/train/IID/OOD/27-pair canonical disjointness, complete descriptor crossing and registered RNG",
        "Anatomical source projection independently parsed; real/rewired/ANN raw topologies and control RNG regenerated",
        "Every checkpoint's fixed encoder, topology, internal weights, readout weights and probe activations independently recomputed",
        "Every actual reward-only policy-gradient update replayed using old-head eligibility and clipping; one actual fit per family",
        "Full anatomical calibration scores reconstructed; affine8-parameter reference independently fit by QR with SVD rank8; scalar reference by two anchor means",
        "Every baseline/final teacher/reference absolute score, unclipped margin, clipped probability, four-map mean and greedy response independently recomputed",
        "Reference fitting uses only disjoint2/4 calibration scores; fully unseen15-pair errors aggregate all four maps and all held-out cases",
        "Acquisition and fidelity are distinct gates; primary mean probability MAE and passing seed fraction use all seeds; baseline and qualified summaries kept separate",
        "Raw margin/absolute-score residuals and greedy disagreement are recomputed independently of sigmoid probability agreement",
    ])


def verify(output: Path) -> dict:
    """Write verification.json separately, preserving every primary artifact."""
    output = Path(output).resolve()
    report = {"audited_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "method": "independent numpy/stdlib arithmetic replay; only previous independent audit helpers reused",
              "input_scope": "four order-preserving artificial scalar maps; one reward learner and two privileged score-distillation references per family",
              "counts": {}, "checks": [], "errors": [], "not_verified": [
                  "Visual numerosity extraction, ratio-based quantity perception, LC11/DAN dynamics or shared human/fly mechanisms",
                  "Remote private Git publication authenticity; local chronology is self-recorded",
                  "Statistical significance, mathematical rule discovery or educational implications",
                  "A probability-MAE threshold as proof of globally affine scores, equivalent learning mechanisms or independent reference reward learning",
                  "Four maps as independent learners or computational seeds as biological animals"]}
    try:
        _require(output.is_dir(), f"Output directory missing: {output}")
        _audit(output, report)
        report["passed"] = True
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
