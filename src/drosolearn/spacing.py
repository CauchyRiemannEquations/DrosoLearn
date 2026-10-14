"""Anchored, order-preserving scalar spacing: one fit, four frozen evaluations."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import importlib.metadata
import json
import os
import platform

import numpy as np

from .encoding import (ANCHORED, FAMILIES, PAIRS, PRIMARY, candidate_cache,
                       encoded_inputs, make_splits, policy_metrics)
from .experiment import CONDITIONS, git_state, write_json
from .graph import load_projection, random_ann, rewire, sha256
from .model import Learner

ALPHAS = (0.25, 0.5, 1.0, 1.5)
FOCAL_INDEX = PAIRS.index((5, 7))
TIGHT, IDENTITY = 0, 2
TRIAL_FIELDS = ("seed", "condition", "trial", "family_id", "left_count", "right_count",
                "action", "reward", "p_right", "optimizer_factor", "internal_step_l2",
                "readout_step_l2", "kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2")
CURVE_FIELDS = ("seed", "condition", "trial", "split", "p_target", "greedy_accuracy",
                "internal_change_l2", "readout_change_l2", "probe_kc_change_l2", "probe_mbon_change_l2")


def spacing_panel():
    """Raw code units; the shared encoder divides them by eight."""
    return np.asarray([[1, 2, 3, 4, 6 - a, 6, 6 + a, 8] for a in ALPHAS], dtype=float)


def validate_config(cfg):
    if cfg["scope"] not in ("preregistered_diagnostic", "integration_test"):
        raise ValueError("Unsupported study scope")
    seeds = cfg["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Invalid seeds")
    if cfg["scope"] == "preregistered_diagnostic" and len(seeds) != 10:
        raise ValueError("Ten prospectively registered seeds required")
    if cfg["train_pair"] != [2, 4] or cfg["primary_pair"] != [5, 7] or tuple(cfg["code_alphas"]) != ALPHAS:
        raise ValueError("Spacing design requires fixed anchors, focal pair and four alphas")
    conditions = cfg["conditions"]
    if not conditions or len(set(conditions)) != len(conditions) or any(c not in FAMILIES for c in conditions):
        raise ValueError("Invalid model families")
    if cfg["scope"] == "preregistered_diagnostic" and tuple(conditions) != FAMILIES:
        raise ValueError("All six registered model families required")
    for key in ("trials", "checkpoint_every", "eval_trials", "probe_rows"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError("Invalid trial, checkpoint or probe budget")
    if (cfg["trials"] % 4 or cfg["eval_trials"] % 4 or cfg["trials"] % cfg["checkpoint_every"]
            or cfg["probe_rows"] > cfg["eval_trials"]):
        raise ValueError("Budgets must preserve crossing and checkpoint coverage")
    for key in ("head_lr", "internal_lr"):
        if type(cfg[key]) not in (int, float) or not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError("Invalid learning rate")
    criteria = cfg["criteria"]
    required = {"acquisition_accuracy", "acquisition_gain", "transfer_accuracy",
                "sensitivity_increase", "positive_seed_fraction", "rank_accuracy"}
    if set(criteria) != required:
        raise ValueError("Missing or extra evaluation criteria")
    if any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in criteria.values()):
        raise ValueError("Invalid evaluation criterion")
    if criteria["positive_seed_fraction"] <= 0:
        raise ValueError("Positive seed fraction required")
    if cfg["scope"] == "preregistered_diagnostic":
        if seeds != list(range(4001, 4011)):
            raise ValueError("This registered protocol fixes seeds4001–4010")
        resources = tuple(cfg[k] for k in ("trials", "checkpoint_every", "eval_trials", "probe_rows", "head_lr", "internal_lr"))
        if resources != (400, 100, 128, 8, .1, .01):
            raise ValueError("This registered protocol fixes the trial budget and learning rates")
        if criteria != {"acquisition_accuracy": .75, "acquisition_gain": .15, "transfer_accuracy": .75,
                        "sensitivity_increase": .1, "positive_seed_fraction": .8, "rank_accuracy": .9}:
            raise ValueError("This registered protocol fixes its evaluation criteria")


def study_hashes(root, cfg):
    root = Path(root)
    files = list((root / "src").rglob("*.py")) + list((root / cfg["graph"]).glob("*"))
    files += [root / name for name in ("docs/spacing-protocol.md", "docs/spacing-background.md",
                                      "pyproject.toml", "requirements-lock.txt",
                                      "data/source-manifest.json", "data/LICENSE.md")]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files) if p.is_file()}


def register(root, config, destination):
    root, config, destination = map(Path, (root, config, destination))
    cfg = json.loads(config.read_text(encoding="utf-8"))
    validate_config(cfg)
    load_projection(root / cfg["graph"])
    record = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
              "config_sha256": sha256(config), "source_hashes": study_hashes(root, cfg),
              "git": git_state(root), "notice": "prospective private Git/local freeze; not accredited public preregistration",
              "previous_studies": ["v0.1.0-smoke", "study-reversal-v1", "study-encoding-v1"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return record


def panel_metrics(cache, datasets, panel, code_values):
    values = np.asarray(code_values)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.all(np.diff(values) > 0):
        raise ValueError("Cache code values must be finite, unique and sorted")
    locations = np.searchsorted(values, panel)
    if np.any(locations >= len(values)) or not np.array_equal(values[locations], panel):
        raise ValueError("Code panel is not represented in cache")
    means = np.empty((len(panel), len(PAIRS)))
    greedy = np.empty_like(means)
    cases = []
    for j, pair in enumerate(PAIRS):
        ds = datasets[f"pair_{pair[0]}_{pair[1]}"]
        indices = locations[:, ds.counts - 1]
        rows = np.arange(len(ds.counts))[None, :]
        left = cache[j, rows, 0, indices[:, :, 0]]
        right = cache[j, rows, 1, indices[:, :, 1]]
        p = 1 / (1 + np.exp(-np.clip(right - left, -40, 40)))
        target = ds.target(True)[None, :]
        means[:, j] = np.where(target == 1, p, 1 - p).mean(axis=1)
        decisions = np.where(np.abs(p - 0.5) < 1e-12, 0.5, ((p > 0.5).astype(int) == target).astype(float))
        greedy[:, j] = decisions.mean(axis=1)
        cases.append(p)
    return means, greedy, np.stack(cases, axis=1)


def transfer_evaluate(model, datasets, panel):
    state = [x.copy() for x in (model.weights, model.head, model.encoder, model.bias)]
    counters = (model.head_updates, model.internal_updates)
    values = np.unique(panel)
    cache = candidate_cache(model, datasets, values)
    means, greedy, cases = panel_metrics(cache, datasets, panel, values)
    if not all(np.array_equal(old, new) for old, new in zip(state, (model.weights, model.head, model.encoder, model.bias))):
        raise ValueError("Evaluation changed model state")
    if counters != (model.head_updates, model.internal_updates):
        raise ValueError("Evaluation changed update counters")
    return {"code_values": values, "score_cache": cache, "map_p_target": means,
            "map_greedy_accuracy": greedy, "map_p_right": cases}


def summarize(cfg, curves, transfer):
    index = {(r["seed"], r["condition"], r["trial"], r["split"]): r for r in curves}
    criteria, n = cfg["criteria"], cfg["trials"]
    outcomes = []
    for seed in cfg["seeds"]:
        for condition in cfg["conditions"]:
            before = index[seed, condition, 0, "iid"]["p_target"]
            after = index[seed, condition, n, "iid"]["p_target"]
            acquired = after >= criteria["acquisition_accuracy"] and after - before >= criteria["acquisition_gain"]
            baseline = transfer[seed, condition, "baseline"]["map_p_target"]
            final = transfer[seed, condition, "final"]["map_p_target"]
            greedy = transfer[seed, condition, "final"]["map_greedy_accuracy"][:, FOCAL_INDEX]
            focal_before, focal_after = baseline[:, FOCAL_INDEX], final[:, FOCAL_INDEX]
            gap_before = float(focal_before[IDENTITY] - focal_before[TIGHT])
            gap_after = float(focal_after[IDENTITY] - focal_after[TIGHT])
            unseen_before, unseen_after = baseline[:, PRIMARY].mean(axis=1), final[:, PRIMARY].mean(axis=1)
            outcomes.append({"seed": seed, "condition": condition, "before_accuracy": before,
                             "after_accuracy": after, "gain": after - before, "acquired": bool(acquired),
                             "ood_accuracy": index[seed, condition, n, "ood"]["p_target"],
                             "head_updates": n, "internal_updates": n if CONDITIONS[condition][1] else 0,
                             "focal_before": focal_before.tolist(), "focal_after": focal_after.tolist(),
                             "focal_gain": (focal_after - focal_before).tolist(), "focal_greedy_after": greedy.tolist(),
                             "unseen_before": unseen_before.tolist(), "unseen_after": unseen_after.tolist(),
                             "anchored_after": final[:, ANCHORED].mean(axis=1).tolist(), "all_after": final.mean(axis=1).tolist(),
                             "raw_focal_gap_before": gap_before, "raw_focal_gap_after": gap_after,
                             "sensitivity_increase": gap_after - gap_before,
                             "qualified_focal_transfer": (acquired & (focal_after >= criteria["transfer_accuracy"])).tolist(),
                             "qualified_unseen_transfer": (acquired & (unseen_after >= criteria["transfer_accuracy"])).tolist(),
                             "rank_robust": bool(acquired and np.all(greedy >= criteria["rank_accuracy"]))})
    verdicts = []
    for condition in cfg["conditions"]:
        rows = [r for r in outcomes if r["condition"] == condition]
        effects = [r["sensitivity_increase"] for r in rows]
        qualified = [r["sensitivity_increase"] for r in rows if r["acquired"]]
        mean, positive = float(np.mean(effects)), sum(e > 0 for e in effects)
        verdicts.append({"condition": condition, "n_seeds": len(rows), "mean_sensitivity_increase": mean,
                         "sd_sensitivity_increase": float(np.std(effects, ddof=1)) if len(effects) > 1 else 0.0,
                         "positive_seed_count": positive,
                         "criterion_met": bool(mean >= criteria["sensitivity_increase"] and positive / len(rows) >= criteria["positive_seed_fraction"]),
                         "qualified_seed_count": len(qualified),
                         "qualified_mean_sensitivity_increase": float(np.mean(qualified)) if qualified else None,
                         "rank_robust_count": sum(r["rank_robust"] for r in rows),
                         "focal_transfer_counts": np.sum([r["qualified_focal_transfer"] for r in rows], axis=0).tolist(),
                         "unseen_transfer_counts": np.sum([r["qualified_unseen_transfer"] for r in rows], axis=0).tolist()})
    return {"status": "completed", "scope": cfg["scope"], "outcomes": outcomes, "family_verdicts": verdicts,
            "actual_training_trials": len(cfg["seeds"]) * len(cfg["conditions"]) * n, "frozen_evaluation_maps": 4,
            "interpretation": "same learned function under order-preserving focal spacing; probabilities and greedy decisions distinguished"}


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
                        "data": "seed*100+31,+32,+33,+40+pair_index"},
                "training_reuse": "one trajectory per seed/family; four frozen code evaluations, no retraining per map",
                "primary_pair": [5, 7], "primary_input_midpoint": 0.75, "anchor_evaluation_identity": True,
                "reproduce": f"python -m drosolearn.spacing run --config {recorded_path(config)} --registration {recorded_path(registration)} --out results/NEW_SPACING",
                "reproduce_argv": ["python", "-m", "drosolearn.spacing", "run", "--config", recorded_path(config),
                                   "--registration", recorded_path(registration), "--out", "results/NEW_SPACING"]}
    write_json(output / "metadata.json", metadata)
    curves, transfers, controls = [], {}, []
    try:
        for seed in cfg["seeds"]:
            datasets = make_splits(cfg, seed)
            for name in ("train", "iid", "ood"):
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
                            metrics = transfer_evaluate(model, datasets, panel)
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
    parser = argparse.ArgumentParser(description="Order-preserving focal input-spacing diagnostic")
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
        from .spacing_analysis import analyze
        analyze(args.out)
    elif args.command == "analyze":
        from .spacing_analysis import analyze
        analyze(args.out)
    else:
        from .spacing_audit import verify
        result = verify(args.out)
        print(json.dumps(result))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
