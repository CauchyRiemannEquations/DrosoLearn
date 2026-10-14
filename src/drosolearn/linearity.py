"""Training-anchor score compression; references are not reward learners."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import importlib.metadata
import json
import os
import platform

import numpy as np

from .spacing import (ALPHAS, ANCHORED, FAMILIES, FOCAL_INDEX, IDENTITY, PAIRS, PRIMARY,
                      CURVE_FIELDS, TRIAL_FIELDS, encoded_inputs, policy_metrics, spacing_panel)
from .spacing import make_splits as base_splits, transfer_evaluate as teacher_evaluate
from .data import make_dataset, assert_disjoint
from .experiment import CONDITIONS, git_state, write_json
from .graph import load_projection, random_ann, rewire, sha256
from .model import Learner

REFERENCES = ("affine", "scalar")


def validate_config(cfg):
    if cfg["scope"] not in ("preregistered_diagnostic", "integration_test"):
        raise ValueError("Unsupported study scope")
    if (not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"])
            or any(type(s) is not int or s < 0 for s in cfg["seeds"])):
        raise ValueError("Invalid seeds")
    if (cfg["train_pair"] != [2, 4] or tuple(cfg["code_alphas"]) != ALPHAS
            or tuple(cfg["references"]) != REFERENCES or cfg["primary_pairs"] != "fully_unseen15"):
        raise ValueError("Invalid fixed reference/anchor/pair design")
    conditions = cfg["conditions"]
    if not conditions or len(set(conditions)) != len(conditions) or any(c not in FAMILIES for c in conditions):
        raise ValueError("Invalid model families")
    for key in ("trials", "checkpoint_every", "eval_trials", "calibration_trials", "probe_rows"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError("Invalid trial or probe budget")
    if (cfg["trials"] % 4 or cfg["eval_trials"] % 4 or cfg["calibration_trials"] % 4
            or cfg["trials"] % cfg["checkpoint_every"] or cfg["probe_rows"] > cfg["eval_trials"]
            or cfg["calibration_trials"] < 32):
        raise ValueError("Crossing, checkpoints and full-rank calibration require valid budgets")
    for key in ("head_lr", "internal_lr"):
        if type(cfg[key]) not in (int, float) or not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError("Invalid learning rate")
    criteria = cfg["criteria"]
    if set(criteria) != {"acquisition_accuracy", "acquisition_gain", "probability_mae", "seed_pass_fraction"}:
        raise ValueError("Invalid reference criteria set")
    if any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in criteria.values()):
        raise ValueError("Invalid criterion")
    if criteria["seed_pass_fraction"] <= 0:
        raise ValueError("Positive seed pass fraction required")
    if cfg["scope"] == "preregistered_diagnostic":
        if cfg["seeds"] != list(range(5001, 5011)) or tuple(conditions) != FAMILIES:
            raise ValueError("This protocol fixes ten new seeds5001–5010 and six families")
        resources = tuple(cfg[k] for k in ("trials", "checkpoint_every", "eval_trials", "calibration_trials", "probe_rows", "head_lr", "internal_lr"))
        if resources != (400, 100, 128, 128, 8, .1, .01):
            raise ValueError("Registered resources and learning rates are fixed")
        if criteria != {"acquisition_accuracy": .75, "acquisition_gain": .15, "probability_mae": .02, "seed_pass_fraction": .8}:
            raise ValueError("Registered criteria are fixed")


def study_hashes(root, cfg):
    root = Path(root)
    files = list((root / "src").rglob("*.py")) + list((root / cfg["graph"]).glob("*"))
    files += [root / name for name in ("docs/linearity-protocol.md", "docs/linearity-background.md",
                                      "pyproject.toml", "requirements-lock.txt", "data/source-manifest.json", "data/LICENSE.md")]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files) if p.is_file()}


def register(root, config, destination):
    root, config, destination = map(Path, (root, config, destination))
    cfg = json.loads(config.read_text(encoding="utf-8"))
    validate_config(cfg)
    load_projection(root / cfg["graph"])
    record = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg, "config_sha256": sha256(config),
              "source_hashes": study_hashes(root, cfg), "git": git_state(root),
              "notice": "prospective private Git/local freeze; not accredited public preregistration",
              "previous_studies": ["v0.1.0-smoke", "study-reversal-v1", "study-encoding-v1", "study-spacing-v1"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return record


def make_splits(cfg, seed):
    splits = base_splits(cfg, seed)
    splits["calibration"] = make_dataset(seed * 100 + 34, "calibration", cfg["calibration_trials"])
    assert_disjoint(list(splits.values()))
    return splits


def fit_references(inputs, teacher_scores):
    """Only two trained quantity coordinates; no evaluation data or labels."""
    inputs, scores = np.asarray(inputs), np.asarray(teacher_scores)
    if inputs.ndim != 3 or inputs.shape[1:] != (2, 7) or scores.shape != inputs.shape[:2]:
        raise ValueError("Invalid calibration input/score shape")
    if not np.isfinite(inputs).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite calibration")
    x, y = inputs.reshape(-1, 7), scores.reshape(-1)
    if not np.array_equal(np.unique(x[:, 0]), [.25, .5]):
        raise ValueError("References may only fit trained2/4 coordinates")
    design = np.column_stack((np.ones(len(x)), x))
    coefficients, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    if rank != 8:
        raise ValueError("Calibration affine design must have full rank8")
    means = np.asarray([y[x[:, 0] == q].mean() for q in (.25, .5)])
    slope = (means[1] - means[0]) / .25
    scalar = np.asarray([means[0] - slope * .25, slope])
    return {"affine_coefficients": coefficients, "affine_rank": np.asarray(rank),
            "affine_singular_values": singular,
            "affine_calibration_score_rmse": np.asarray(np.sqrt(np.mean((design @ coefficients - y) ** 2))),
            "scalar_coefficients": scalar, "scalar_anchor_means": means,
            "scalar_calibration_score_rmse": np.asarray(np.sqrt(np.mean((scalar[0] + x[:, 0] * scalar[1] - y) ** 2)))}


def calibrate(model, dataset):
    inputs = encoded_inputs(dataset, spacing_panel()[IDENTITY])
    _, activity = model.features(inputs)
    scores = activity @ model.head
    return {"calibration_inputs": inputs, "calibration_scores": scores, **fit_references(inputs, scores)}


def selected_scores(cache, datasets, panel, values):
    locations = np.searchsorted(values, panel)
    result = []
    for j, (low, high) in enumerate(PAIRS):
        ds = datasets[f"pair_{low}_{high}"]
        index = locations[:, ds.counts - 1]
        rows = np.arange(len(ds.counts))[None, :, None]
        sides = np.arange(2)[None, None, :]
        result.append(cache[j, rows, sides, index])
    return np.stack(result, axis=1)


def greedy_side(probabilities):
    p = np.asarray(probabilities)
    return np.where(np.abs(p - .5) < 1e-12, .5, (p > .5).astype(float))


def transfer_evaluate(model, datasets, panel, reference):
    teacher = teacher_evaluate(model, datasets, panel)
    scores = selected_scores(teacher["score_cache"], datasets, panel, teacher["code_values"])
    teacher.update(teacher_map_scores=scores, teacher_map_margin=scores[..., 1] - scores[..., 0])
    inputs = np.asarray([[encoded_inputs(datasets[f"pair_{a}_{b}"], codes) for a, b in PAIRS] for codes in panel])
    target = np.asarray([datasets[f"pair_{a}_{b}"].target(True) for a, b in PAIRS])[None, :]
    for name in REFERENCES:
        coefficients = reference[f"{name}_coefficients"]
        prediction = (coefficients[0] + inputs @ coefficients[1:] if name == "affine"
                      else coefficients[0] + inputs[..., 0] * coefficients[1])
        margin = prediction[..., 1] - prediction[..., 0]
        p = 1 / (1 + np.exp(-np.clip(margin, -40, 40)))
        correct = np.where(np.abs(p - .5) < 1e-12, .5, ((p > .5).astype(int) == target).astype(float))
        teacher.update({f"{name}_map_scores": prediction, f"{name}_map_margin": margin,
                        f"{name}_map_p_right": p, f"{name}_map_p_target": np.where(target == 1, p, 1 - p).mean(axis=2),
                        f"{name}_map_greedy_accuracy": correct.mean(axis=2),
                        f"{name}_probability_mae": np.abs(p - teacher["map_p_right"]).mean(axis=2)})
    return teacher


def metrics(transfer, reference):
    error = np.abs(transfer[f"{reference}_map_p_right"] - transfer["map_p_right"])
    margin = transfer[f"{reference}_map_margin"] - transfer["teacher_map_margin"]
    scores = transfer[f"{reference}_map_scores"] - transfer["teacher_map_scores"]
    disagreement = np.abs(greedy_side(transfer[f"{reference}_map_p_right"]) - greedy_side(transfer["map_p_right"]))
    return {"unseen_probability_mae": float(error[:, PRIMARY].mean()),
            "unseen_margin_mae": float(np.abs(margin[:, PRIMARY]).mean()),
            "unseen_margin_rmse": float(np.sqrt(np.mean(margin[:, PRIMARY] ** 2))),
            "unseen_score_rmse": float(np.sqrt(np.mean(scores[:, PRIMARY] ** 2))),
            "unseen_greedy_disagreement": float(disagreement[:, PRIMARY].mean()),
            "anchored_probability_mae": float(error[:, ANCHORED].mean()), "all_probability_mae": float(error.mean()),
            "focal_probability_mae": float(error[:, FOCAL_INDEX].mean()),
            "focal_reference_p_correct": transfer[f"{reference}_map_p_target"][:, FOCAL_INDEX].tolist(),
            "focal_reference_greedy_accuracy": transfer[f"{reference}_map_greedy_accuracy"][:, FOCAL_INDEX].tolist(),
            "unseen_reference_p_correct": transfer[f"{reference}_map_p_target"][:, PRIMARY].mean(axis=1).tolist()}


def summarize(cfg, curves, transfer):
    index = {(r["seed"], r["condition"], r["trial"], r["split"]): r for r in curves}
    rows = []
    for seed in cfg["seeds"]:
        for condition in cfg["conditions"]:
            before, after = (index[seed, condition, t, "iid"]["p_target"] for t in (0, cfg["trials"]))
            acquired = bool(after >= cfg["criteria"]["acquisition_accuracy"] and after - before >= cfg["criteria"]["acquisition_gain"])
            stages = {stage: {name: metrics(transfer[seed, condition, stage], name) for name in REFERENCES} for stage in ("baseline", "final")}
            passed = stages["final"]["affine"]["unseen_probability_mae"] <= cfg["criteria"]["probability_mae"]
            rows.append({"seed": seed, "condition": condition, "before_accuracy": before, "after_accuracy": after,
                         "gain": after - before, "acquired": acquired,
                         "ood_accuracy": index[seed, condition, cfg["trials"], "ood"]["p_target"],
                         "head_updates": cfg["trials"], "internal_updates": cfg["trials"] if CONDITIONS[condition][1] else 0,
                         **stages, "focal_model_before": transfer[seed, condition, "baseline"]["map_p_target"][:, FOCAL_INDEX].tolist(),
                         "focal_model_after": transfer[seed, condition, "final"]["map_p_target"][:, FOCAL_INDEX].tolist(),
                         "focal_model_greedy_after": transfer[seed, condition, "final"]["map_greedy_accuracy"][:, FOCAL_INDEX].tolist(),
                         "fidelity_passed": bool(passed), "qualified_fidelity": bool(acquired and passed)})
    verdicts = []
    for condition in cfg["conditions"]:
        selected = [r for r in rows if r["condition"] == condition]
        errors = [r["final"]["affine"]["unseen_probability_mae"] for r in selected]
        qualified = [r["final"]["affine"]["unseen_probability_mae"] for r in selected if r["acquired"]]
        passing = sum(r["fidelity_passed"] for r in selected)
        verdicts.append({"condition": condition, "n_seeds": len(selected),
                         "mean_affine_probability_mae": float(np.mean(errors)),
                         "sd_affine_probability_mae": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.,
                         "passing_seed_count": passing,
                         "criterion_met": bool(np.mean(errors) <= cfg["criteria"]["probability_mae"] and passing / len(selected) >= cfg["criteria"]["seed_pass_fraction"]),
                         "acquisition_count": sum(r["acquired"] for r in selected),
                         "qualified_count": sum(r["qualified_fidelity"] for r in selected),
                         "qualified_mean_affine_probability_mae": float(np.mean(qualified)) if qualified else None,
                         **{f"mean_{name}_{metric}": float(np.mean([r["final"][name][f"unseen_{metric}"] for r in selected]))
                            for name, metric in (("scalar", "probability_mae"), ("affine", "margin_rmse"), ("scalar", "margin_rmse"),
                                                 ("affine", "greedy_disagreement"), ("scalar", "greedy_disagreement"))}})
    return {"status": "completed", "scope": cfg["scope"], "outcomes": rows, "family_verdicts": verdicts,
            "actual_training_trials": len(cfg["seeds"]) * len(cfg["conditions"]) * cfg["trials"],
            "frozen_evaluation_maps": 4, "reference_models": 2,
            "interpretation": "teacher-score surrogate fidelity, not independent reward learning or exact global linearity"}


def run(root, config, registration, output):
    root, config, registration, output = [Path(p).resolve() for p in (root, config, registration, output)]
    cfg = json.loads(config.read_text(encoding="utf-8"))
    validate_config(cfg)
    reg = json.loads(registration.read_text(encoding="utf-8"))
    if sha256(config) != reg["config_sha256"] or cfg != reg["config"] or study_hashes(root, cfg) != reg["source_hashes"]:
        raise ValueError("Source, configuration or protocol changed after registration")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "registration.json", reg)
    graph = load_projection(root / cfg["graph"])
    panel = spacing_panel()
    np.save(output / "spacing_codes.npy", panel)
    gaps = np.asarray([[row[b - 1] - row[a - 1] for a, b in PAIRS] for row in panel])
    midpoints = np.asarray([[(row[b - 1] + row[a - 1]) / 2 for a, b in PAIRS] for row in panel])
    np.savez_compressed(output / "code_geometry.npz", pairs=np.asarray(PAIRS),
                        quantity_ratios=np.asarray([a / b for a, b in PAIRS]),
                        quantity_differences=np.asarray([b - a for a, b in PAIRS]),
                        code_gaps=gaps, normalized_code_gaps=gaps / 8,
                        code_midpoints=midpoints, normalized_code_midpoints=midpoints / 8,
                        code_order_agreement=(gaps > 0).astype(float))
    def recorded_path(path):
        return (path.relative_to(root) if path.is_relative_to(root) else path).as_posix()
    try:
        source_root = Path(os.path.relpath(root, output)).as_posix()
    except ValueError:
        source_root = root.as_posix()
    metadata = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg, "git": git_state(root),
                "source_root": source_root, "config_path": recorded_path(config), "registration_sha256": sha256(registration),
                "graph": graph.metadata, "python": platform.python_version(), "platform": platform.platform(),
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "matplotlib")},
                "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "rng": {"actions": "seed+280000", "encoder": "seed+70000", "rewired": "seed+50000", "ann": "seed+60000",
                        "data": "seed*100+31,+32,+33,+34 calibration,+40+pair_index"},
                "training_reuse": "one trajectory per seed/family; four frozen code evaluations, no retraining per map",
                "primary_pairs": "fully_unseen15", "example_pair": [5, 7], "anchor_evaluation_identity": True,
                "reference_scope": "privileged calibration on teacher scores at2/4; no independent reward learning",
                "reproduce": f"python -m drosolearn.linearity run --config {recorded_path(config)} --registration {recorded_path(registration)} --out results/NEW_LINEARITY",
                "reproduce_argv": ["python", "-m", "drosolearn.linearity", "run", "--config", recorded_path(config),
                                   "--registration", recorded_path(registration), "--out", "results/NEW_LINEARITY"]}
    write_json(output / "metadata.json", metadata)
    curves, transfers, controls = [], {}, []
    try:
        for seed in cfg["seeds"]:
            datasets = make_splits(cfg, seed)
            for name in ("train", "iid", "ood", "calibration"):
                reference = encoded_inputs(datasets[name], panel[IDENTITY])
                if not all(np.array_equal(reference, encoded_inputs(datasets[name], codes)) for codes in panel):
                    raise ValueError("Training or anchor evaluation changed across maps")
            seed_path = output / f"seed_{seed}"
            seed_path.mkdir()
            uniforms = np.random.default_rng(seed + 280000).random(cfg["trials"])
            np.savez_compressed(seed_path / "stimuli.npz", **{f"{name}_{key}": value for name, ds in datasets.items()
                                for key, value in (("counts", ds.counts), ("nuisance", ds.nuisance), ("family_ids", np.asarray(ds.family_ids)))},
                                action_uniforms=uniforms)
            graphs = {"real": graph, "rewired": rewire(graph, seed + 50000), "ann": random_ann(graph, seed + 60000)}
            for kind, g in graphs.items():
                np.savez_compressed(seed_path / f"{kind}_topology.npz", pre=g.pre, post=g.post, counts=g.weights)
                controls.append({"seed": seed, "kind": kind, "metadata": g.metadata})
            with (seed_path / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=TRIAL_FIELDS)
                writer.writeheader()
                for condition in cfg["conditions"]:
                    kind, plastic, _, _ = CONDITIONS[condition]
                    g = graphs[kind]
                    model = Learner(g, seed + 70000, plastic, True, cfg["head_lr"], cfg["internal_lr"])
                    path = seed_path / condition
                    path.mkdir()
                    probe = encoded_inputs(datasets["iid"], panel[IDENTITY])[:cfg["probe_rows"]]
                    initial = model.snapshot(probe)
                    def checkpoint(t):
                        snap = model.snapshot(probe)
                        np.savez_compressed(path / f"checkpoint_{t:04d}.npz", **snap, encoder=model.encoder,
                                            bias=model.bias, pre=g.pre, post=g.post)
                        for split in ("iid", "ood"):
                            reference = encoded_inputs(datasets[split], panel[IDENTITY])
                            reference_probability = model.probabilities(reference)
                            for codes in panel:
                                mapped_inputs = encoded_inputs(datasets[split], codes)
                                if (not np.array_equal(reference, mapped_inputs)
                                        or not np.array_equal(reference_probability, model.probabilities(mapped_inputs))):
                                    raise ValueError("Checkpoint anchor input/output differs across maps")
                            curves.append({"seed": seed, "condition": condition, "trial": t, "split": split,
                                           **policy_metrics(model, datasets[split], panel[IDENTITY]),
                                           "internal_change_l2": float(np.linalg.norm(model.weights - initial["internal_weights"])),
                                           "readout_change_l2": float(np.linalg.norm(model.head - initial["readout_weights"])),
                                           "probe_kc_change_l2": float(np.linalg.norm(snap["kc_activity"] - initial["kc_activity"])),
                                           "probe_mbon_change_l2": float(np.linalg.norm(snap["mbon_activity"] - initial["mbon_activity"]))})
                        if t in (0, cfg["trials"]):
                            stage = "baseline" if t == 0 else "final"
                            reference = calibrate(model, datasets["calibration"])
                            np.savez_compressed(path / f"reference_{stage}.npz", **reference)
                            metrics = transfer_evaluate(model, datasets, panel, reference)
                            np.savez_compressed(path / f"transfer_{stage}.npz", **metrics)
                            transfers[seed, condition, stage] = metrics
                    checkpoint(0)
                    ds = datasets["train"]
                    inputs, targets = encoded_inputs(ds, panel[IDENTITY]), ds.target(True)
                    for i, x in enumerate(inputs):
                        kc, mb = model.features(x)
                        p = float(1 / (1 + np.exp(-np.clip((mb[1] - mb[0]) @ model.head, -40, 40))))
                        action = int(uniforms[i] < p)
                        reward = int(action == targets[i])
                        old_w, old_h = model.weights.copy(), model.head.copy()
                        model.update(x, action, reward)
                        writer.writerow({"seed": seed, "condition": condition, "trial": i + 1,
                                         "family_id": ds.family_ids[i], "left_count": int(ds.counts[i, 0]), "right_count": int(ds.counts[i, 1]),
                                         "action": action, "reward": reward, "p_right": p, "optimizer_factor": (action - p) * (reward - 0.5),
                                         "internal_step_l2": float(np.linalg.norm(model.weights - old_w)), "readout_step_l2": float(np.linalg.norm(model.head - old_h)),
                                         "kc_left_l2": float(np.linalg.norm(kc[0])), "kc_right_l2": float(np.linalg.norm(kc[1])),
                                         "mbon_left_l2": float(np.linalg.norm(mb[0])), "mbon_right_l2": float(np.linalg.norm(mb[1]))})
                        if (i + 1) % cfg["checkpoint_every"] == 0:
                            checkpoint(i + 1)
                    if not np.isfinite(model.head).all() or not np.isfinite(model.weights).all():
                        raise ValueError("Nonfinite learned parameters")
            print(f"completed seed {seed} ({cfg['seeds'].index(seed) + 1}/{len(cfg['seeds'])})", flush=True)
        with (output / "learning_curves.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CURVE_FIELDS)
            writer.writeheader()
            writer.writerows(curves)
        write_json(output / "controls.json", controls)
        summary = summarize(cfg, curves, transfers)
        write_json(output / "summary.json", summary)
        metadata.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "metadata.json", metadata)
        write_json(output / "artifact_hashes.json", {p.relative_to(output).as_posix(): sha256(p)
                                                   for p in sorted(output.rglob("*")) if p.is_file()})
        return summary
    except Exception as error:
        metadata.update(status="failed", failure=repr(error), failed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "metadata.json", metadata)
        if curves:
            with (output / "partial_learning_curves.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CURVE_FIELDS)
                writer.writeheader()
                writer.writerows(curves)
        raise



def main():
    parser = argparse.ArgumentParser(description="Training-anchor linear reference diagnostic")
    parser.add_argument("command", choices=("register", "run", "analyze", "verify"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    for name in ("config", "registration", "out"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, (path if path.is_absolute() else args.root / path).resolve())
    if args.command == "register":
        if args.config is None:
            parser.error("--config required")
        result = register(args.root, args.config, args.out)
        print(json.dumps({"registered_at_utc": result["registered_at_utc"], "config_sha256": result["config_sha256"]}))
    elif args.command == "run":
        if args.config is None or args.registration is None:
            parser.error("--config and --registration required")
        run(args.root, args.config, args.registration, args.out)
        from .linearity_analysis import analyze
        analyze(args.out)
    elif args.command == "analyze":
        from .linearity_analysis import analyze
        analyze(args.out)
    else:
        from .linearity_audit import verify
        result = verify(args.out)
        print(json.dumps(result))
        if not result["passed"]:
            raise SystemExit(1)



if __name__ == "__main__":
    main()
