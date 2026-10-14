"""Independent reward learners with prospectively fixed initial feature scaling."""
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
                      CURVE_FIELDS as BASE_CURVE_FIELDS, TRIAL_FIELDS as BASE_TRIAL_FIELDS,
                      encoded_inputs, make_splits, policy_metrics, spacing_panel, panel_metrics)
from .experiment import CONDITIONS, git_state, write_json
from .graph import load_projection, random_ann, rewire, sha256
from .model import Learner

CONDITIONS_ALL = ("linear_raw", "linear_normalized", "linear_random_reward", *FAMILIES)
TRIAL_FIELDS = (*BASE_TRIAL_FIELDS, "decision_left_l2", "decision_right_l2")
CURVE_FIELDS = (*BASE_CURVE_FIELDS, "probe_decision_change_l2")

def gap_rms(features):
    features = np.asarray(features, dtype=float)
    if features.ndim != 3 or features.shape[1] != 2 or not np.isfinite(features).all():
        raise ValueError("Finite paired TRAIN feature array required")
    value = float(np.sqrt(np.mean(np.sum((features[:, 1] - features[:, 0]) ** 2, axis=-1))))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Positive initial feature-gap RMS required")
    return value


class LinearLearner:
    """Seven input coefficients; no labels or teacher scores in the update API."""
    def __init__(self, train_inputs, normalize=True, head_lr=.1):
        train_inputs = np.asarray(train_inputs, dtype=float)
        if train_inputs.ndim != 3 or train_inputs.shape[1:] != (2, 7):
            raise ValueError("Expected paired7D TRAIN inputs")
        self.initial_feature_gap_rms = gap_rms(train_inputs)
        self.feature_scale = self.initial_feature_gap_rms if normalize else 1.
        self.head = np.zeros(7)
        self.weights = np.empty(0)
        self.encoder, self.bias = np.empty((0, 7)), np.empty(0)
        self.pre = self.post = np.empty(0, dtype=int)
        self.head_lr = head_lr
        self.head_updates = self.internal_updates = 0

    def raw_activities(self, inputs):
        shape = np.asarray(inputs).shape[:-1] + (0,)
        return np.empty(shape), np.empty(shape)

    def features(self, inputs):
        kc, _ = self.raw_activities(inputs)
        return kc, np.asarray(inputs) / self.feature_scale

    def probabilities(self, inputs):
        _, features = self.features(inputs)
        scores = features @ self.head
        return 1 / (1 + np.exp(-np.clip(scores[..., 1] - scores[..., 0], -40, 40)))

    def update(self, inputs, action, reward):
        if action not in (0, 1) or reward not in (0, 1):
            raise ValueError("Expected binary action/reward")
        p = float(self.probabilities(inputs))
        _, features = self.features(inputs)
        self.head += self.head_lr * (action - p) * (reward - .5) * (features[1] - features[0])
        self.head_updates += 1
        return p

    def snapshot(self, inputs):
        return snapshot(self, inputs)


class NormalizedLearner(Learner):
    """Initial TRAIN feature-gap normalization held fixed throughout learning."""
    def __init__(self, graph, seed, train_inputs, plastic=False, head_lr=.1, internal_lr=.01):
        super().__init__(graph, seed, plastic, True, head_lr, internal_lr)
        self.head[:] = 0.
        self.pre, self.post = graph.pre, graph.post
        self.initial_feature_gap_rms = gap_rms(Learner.features(self, train_inputs)[1])
        self.feature_scale = self.initial_feature_gap_rms

    def raw_activities(self, inputs):
        return Learner.features(self, inputs)

    def features(self, inputs):
        kc, raw = self.raw_activities(inputs)
        return kc, raw / self.feature_scale

    def update(self, inputs, action, reward):
        if action not in (0, 1) or reward not in (0, 1):
            raise ValueError("Expected binary action/reward")
        kc, features = self.features(inputs)
        p = float(1 / (1 + np.exp(-np.clip((features[1] - features[0]) @ self.head, -40, 40))))
        factor, old_head = (action - p) * (reward - .5), self.head.copy()
        self.head += self.head_lr * factor * (features[1] - features[0])
        self.head_updates += 1
        if self.plastic:
            eligibility = old_head[self.post] * (kc[1, self.pre] - kc[0, self.pre]) / self.feature_scale
            self.weights = np.clip(self.weights + self.internal_lr * factor * eligibility, 0, self.upper)
            self.internal_updates += 1
        return p

    def snapshot(self, inputs):
        return snapshot(self, inputs)


def snapshot(model, inputs):
    kc, mb = model.raw_activities(inputs)
    _, decision = model.features(inputs)
    return {"internal_weights": model.weights.copy(), "readout_weights": model.head.copy(),
            "kc_activity": kc, "mbon_activity": mb, "decision_activity": decision,
            "choice_probability": model.probabilities(inputs),
            "feature_scale": np.asarray(model.feature_scale),
            "initial_feature_gap_rms": np.asarray(model.initial_feature_gap_rms)}


def validate_config(cfg):
    if cfg["scope"] not in ("preregistered_diagnostic", "integration_test"):
        raise ValueError("Unsupported scope")
    seeds = cfg["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Invalid seeds")
    if cfg["scope"] == "integration_test" and any(s in range(6001, 6011) for s in seeds):
        raise ValueError("Scientific seeds must not be used for development fixtures")
    if cfg["train_pair"] != [2, 4] or tuple(cfg["code_alphas"]) != ALPHAS or cfg["primary_pairs"] != "fully_unseen15":
        raise ValueError("Fixed quantity and code design required")
    fixed = {"experiment_id": "independent_reward_linear_diagnostic_v1",
             "graph": "data/flywire_783_mb_left_kc512_s701",
             "head_initialization": "zeros_all_conditions",
             "feature_normalization": "fixed_initial_train_feature_gap_rms_except_linear_raw",
             "random_reward": "independent_seed_plus290000_uniform_below0.5"}
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError("Study identity, topology, initialization and reward rules are fixed")
    if tuple(cfg["conditions"]) != CONDITIONS_ALL:
        raise ValueError("Nine fixed learner conditions required")
    for k in ("trials", "eval_trials", "checkpoint_every", "probe_rows"):
        if type(cfg[k]) is not int or cfg[k] <= 0:
            raise ValueError("Invalid budget")
    if cfg["trials"] % 4 or cfg["eval_trials"] % 4 or cfg["trials"] % cfg["checkpoint_every"] or cfg["probe_rows"] > cfg["eval_trials"]:
        raise ValueError("Crossed trials and checkpoints required")
    if any(type(cfg[k]) not in (int, float) or not np.isfinite(cfg[k]) or cfg[k] < 0 for k in ("head_lr", "internal_lr")):
        raise ValueError("Invalid learning rate")
    required = {"acquisition_accuracy", "acquisition_gain", "transfer_accuracy", "seed_pass_fraction", "reward_gap"}
    crit = cfg["criteria"]
    if set(crit) != required or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in crit.values()) or crit["seed_pass_fraction"] <= 0:
        raise ValueError("Invalid fixed criteria")
    if cfg["scope"] == "preregistered_diagnostic":
        if seeds != list(range(6001, 6011)):
            raise ValueError("New seeds6001–6010 required")
        if tuple(cfg[k] for k in ("trials", "checkpoint_every", "eval_trials", "probe_rows", "head_lr", "internal_lr")) != (400, 100, 128, 8, .1, .01):
            raise ValueError("Fixed resources required")
        if crit != {"acquisition_accuracy": .75, "acquisition_gain": .15, "transfer_accuracy": .75, "seed_pass_fraction": .8, "reward_gap": .15}:
            raise ValueError("Fixed criteria required")


def study_hashes(root, cfg):
    root = Path(root)
    paths = list((root / "src").rglob("*.py")) + list((root / cfg["graph"]).glob("*"))
    paths += [root / p for p in ("docs/reward-linear-protocol.md", "docs/reward-linear-background.md", "pyproject.toml", "requirements-lock.txt", "data/source-manifest.json", "data/LICENSE.md")]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths) if p.is_file()}


def transfer_evaluate(model, datasets, panel):
    before = [a.copy() for a in (model.weights, model.head, model.encoder, model.bias)]
    counters = model.head_updates, model.internal_updates
    scale = model.feature_scale
    values, caches = np.unique(panel), []
    if isinstance(model, NormalizedLearner):
        gain = np.bincount(model.pre, weights=model.weights * model.head[model.post], minlength=model.graph.n_kc) / scale
    for a, b in PAIRS:
        ds = datasets[f"pair_{a}_{b}"]
        q = np.broadcast_to(values / 8, (*ds.counts.shape, len(values)))[..., None]
        ns = np.broadcast_to(.1 * (ds.nuisance[..., None, :] - .5), (*ds.counts.shape, len(values), 6))
        x = np.concatenate((q, ns), axis=-1)
        caches.append(model.kc(x) @ gain if isinstance(model, NormalizedLearner) else x @ (model.head / scale))
    cache = np.asarray(caches)
    means, greedy, cases = panel_metrics(cache, datasets, panel, values)
    if not all(np.array_equal(a, b) for a, b in zip(before, (model.weights, model.head, model.encoder, model.bias))) or counters != (model.head_updates, model.internal_updates) or scale != model.feature_scale:
        raise ValueError("Evaluation changed learner state")
    return {"code_values": values, "score_cache": cache, "map_p_target": means, "map_greedy_accuracy": greedy, "map_p_right": cases}


def summarize(cfg, curves, transfers):
    index = {(r["seed"], r["condition"], r["trial"], r["split"]): r for r in curves}
    crit, rows, verdicts = cfg["criteria"], [], []
    for seed in cfg["seeds"]:
        for condition in cfg["conditions"]:
            before = index[seed, condition, 0, "iid"]["p_target"]
            after = index[seed, condition, cfg["trials"], "iid"]["p_target"]
            acquired = bool(after >= crit["acquisition_accuracy"] and after - before >= crit["acquisition_gain"])
            baseline = transfers[seed, condition, "baseline"]
            final = transfers[seed, condition, "final"]
            unseen = final["map_p_target"][:, PRIMARY].mean(axis=1)
            rows.append({"seed": seed, "condition": condition, "before_accuracy": before, "after_accuracy": after,
                         "gain": after - before, "acquired": acquired,
                         "ood_accuracy": index[seed, condition, cfg["trials"], "ood"]["p_target"],
                         "head_updates": cfg["trials"], "internal_updates": cfg["trials"] if condition in FAMILIES and CONDITIONS[condition][1] else 0,
                         "focal_before": baseline["map_p_target"][:, FOCAL_INDEX].tolist(),
                         "focal_after": final["map_p_target"][:, FOCAL_INDEX].tolist(),
                         "focal_greedy_after": final["map_greedy_accuracy"][:, FOCAL_INDEX].tolist(),
                         "unseen_before": baseline["map_p_target"][:, PRIMARY].mean(axis=1).tolist(), "unseen_after": unseen.tolist(),
                         "anchored_after": final["map_p_target"][:, ANCHORED].mean(axis=1).tolist(),
                         "all_after": final["map_p_target"].mean(axis=1).tolist(),
                         "qualified_unseen_transfer": (acquired & (unseen >= crit["transfer_accuracy"])).tolist(),
                         "qualified_focal_transfer": (acquired & (final["map_p_target"][:, FOCAL_INDEX] >= crit["transfer_accuracy"])).tolist()})
    for condition in cfg["conditions"]:
        selected = [r for r in rows if r["condition"] == condition]
        acquired = sum(r["acquired"] for r in selected)
        joint = sum(r["qualified_unseen_transfer"][IDENTITY] for r in selected)
        verdicts.append({"condition": condition, "n_seeds": len(selected), "acquisition_count": acquired,
                         "acquisition_criterion_met": acquired / len(selected) >= crit["seed_pass_fraction"],
                         "joint_identity_transfer_count": joint, "joint_transfer_criterion_met": joint / len(selected) >= crit["seed_pass_fraction"],
                         "mean_after_accuracy": float(np.mean([r["after_accuracy"] for r in selected])),
                         "sd_after_accuracy": float(np.std([r["after_accuracy"] for r in selected], ddof=1)) if len(selected) > 1 else 0.,
                         "mean_unseen_after": np.mean([r["unseen_after"] for r in selected], axis=0).tolist()})
    paired = []
    for seed in cfg["seeds"]:
        correct = next(r for r in rows if r["seed"] == seed and r["condition"] == "linear_normalized")
        random = next(r for r in rows if r["seed"] == seed and r["condition"] == "linear_random_reward")
        paired.append({"seed": seed, "iid_reward_gap": correct["after_accuracy"] - random["after_accuracy"]})
    mean_gap = float(np.mean([r["iid_reward_gap"] for r in paired]))
    positive = sum(r["iid_reward_gap"] > 0 for r in paired)
    primary = next(v for v in verdicts if v["condition"] == "linear_normalized")
    return {"status": "completed", "scope": cfg["scope"], "outcomes": rows, "family_verdicts": verdicts, "paired_reward_contrasts": paired,
            "hypotheses": {"H_R": {"criterion_met": bool(primary["acquisition_criterion_met"] and mean_gap >= crit["reward_gap"] and positive / len(paired) >= crit["seed_pass_fraction"]),
                                      "mean_iid_reward_gap": mean_gap, "positive_seed_count": positive, "acquisition_count": primary["acquisition_count"]},
                           "H_G": {"criterion_met": bool(primary["joint_transfer_criterion_met"]), "joint_identity_transfer_count": primary["joint_identity_transfer_count"]}},
            "actual_training_trials": len(cfg["seeds"]) * len(cfg["conditions"]) * cfg["trials"], "frozen_evaluation_maps": 4,
            "interpretation": "independent immediate binary-reward learning on supplied scalar order; initial TRAIN gradient RMS normalized, not equal capacity or biological mechanisms"}


def register(root, config, destination):
    root, config, destination = map(Path, (root, config, destination))
    cfg = json.loads(config.read_text(encoding="utf-8"))
    validate_config(cfg)
    load_projection(root / cfg["graph"])
    record = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
              "config_sha256": sha256(config), "source_hashes": study_hashes(root, cfg),
              "git": git_state(root), "notice": "prospective private Git/local freeze; not accredited public preregistration",
              "previous_studies": ["v0.1.0-smoke", "study-reversal-v1", "study-encoding-v1", "study-spacing-v1", "study-linearity-v1"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return record

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
                "rng": {"actions": "seed+280000", "rewards": "seed+290000", "encoder": "seed+70000", "rewired": "seed+50000", "ann": "seed+60000",
                        "data": "seed*100+31,+32,+33,+40+pair_index"},
                "training_reuse": "one trajectory per seed/family; four frozen code evaluations, no retraining per map",
                "primary_pairs": "fully_unseen15", "example_pair": [5, 7], "initial_head": "zero", "anchor_evaluation_identity": True,
                "reproduce": f"python -m drosolearn.reward_linear run --config {recorded_path(config)} --registration {recorded_path(registration)} --out results/NEW_REWARD_LINEAR",
                "reproduce_argv": ["python", "-m", "drosolearn.reward_linear", "run", "--config", recorded_path(config),
                                   "--registration", recorded_path(registration), "--out", "results/NEW_REWARD_LINEAR"]}
    write_json(output / "metadata.json", metadata)
    curves, transfers, controls, resources = [], {}, [], []
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
            reward_uniforms = np.random.default_rng(seed + 290000).random(cfg["trials"])
            np.savez_compressed(seed_path / "stimuli.npz", **{f"{name}_{key}": value for name, ds in datasets.items()
                                for key, value in (("counts", ds.counts), ("nuisance", ds.nuisance), ("family_ids", np.asarray(ds.family_ids)))},
                                action_uniforms=uniforms, reward_uniforms=reward_uniforms)
            graphs = {"real": graph, "rewired": rewire(graph, seed + 50000), "ann": random_ann(graph, seed + 60000)}
            for kind, g in graphs.items():
                np.savez_compressed(seed_path / f"{kind}_topology.npz", pre=g.pre, post=g.post, counts=g.weights)
                controls.append({"seed": seed, "kind": kind, "metadata": g.metadata})
            with (seed_path / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=TRIAL_FIELDS)
                writer.writeheader()
                for condition in cfg["conditions"]:
                    train_inputs = encoded_inputs(datasets["train"], panel[IDENTITY])
                    if condition.startswith("linear_"):
                        model = LinearLearner(train_inputs, normalize=condition != "linear_raw", head_lr=cfg["head_lr"])
                    else:
                        kind, plastic, _, _ = CONDITIONS[condition]
                        model = NormalizedLearner(graphs[kind], seed + 70000, train_inputs, plastic, cfg["head_lr"], cfg["internal_lr"])
                    resources.append({"seed": seed, "condition": condition, "feature_scale": model.feature_scale,
                                      "initial_feature_gap_rms": model.initial_feature_gap_rms, "readout_parameters": len(model.head),
                                      "internal_parameters": len(model.weights) if condition in FAMILIES and CONDITIONS[condition][1] else 0,
                                      "reward_rule": "action-independent Bernoulli(.5)" if condition == "linear_random_reward" else "chosen larger quantity=1;other=0"})
                    path = seed_path / condition
                    path.mkdir()
                    probe = encoded_inputs(datasets["iid"], panel[IDENTITY])[:cfg["probe_rows"]]
                    initial = model.snapshot(probe)
                    def checkpoint(t):
                        snap = model.snapshot(probe)
                        np.savez_compressed(path / f"checkpoint_{t:04d}.npz", **snap, encoder=model.encoder,
                                            bias=model.bias, pre=model.pre, post=model.post)
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
                                           "probe_mbon_change_l2": float(np.linalg.norm(snap["mbon_activity"] - initial["mbon_activity"])),
                                           "probe_decision_change_l2": float(np.linalg.norm(snap["decision_activity"] - initial["decision_activity"]))})
                        if t in (0, cfg["trials"]):
                            stage = "baseline" if t == 0 else "final"
                            metrics = transfer_evaluate(model, datasets, panel)
                            np.savez_compressed(path / f"transfer_{stage}.npz", **metrics)
                            transfers[seed, condition, stage] = metrics
                    checkpoint(0)
                    ds = datasets["train"]
                    inputs, targets = encoded_inputs(ds, panel[IDENTITY]), ds.target(True)
                    for i, x in enumerate(inputs):
                        kc, decision = model.features(x)
                        _, mb = model.raw_activities(x)
                        p = float(1 / (1 + np.exp(-np.clip((decision[1] - decision[0]) @ model.head, -40, 40))))
                        action = int(uniforms[i] < p)
                        reward = int(reward_uniforms[i] < .5) if condition == "linear_random_reward" else int(action == targets[i])
                        old_w, old_h = model.weights.copy(), model.head.copy()
                        model.update(x, action, reward)
                        writer.writerow({"seed": seed, "condition": condition, "trial": i + 1,
                                         "family_id": ds.family_ids[i], "left_count": int(ds.counts[i, 0]), "right_count": int(ds.counts[i, 1]),
                                         "action": action, "reward": reward, "p_right": p, "optimizer_factor": (action - p) * (reward - 0.5),
                                         "internal_step_l2": float(np.linalg.norm(model.weights - old_w)), "readout_step_l2": float(np.linalg.norm(model.head - old_h)),
                                         "kc_left_l2": float(np.linalg.norm(kc[0])), "kc_right_l2": float(np.linalg.norm(kc[1])),
                                         "mbon_left_l2": float(np.linalg.norm(mb[0])), "mbon_right_l2": float(np.linalg.norm(mb[1])),
                                         "decision_left_l2": float(np.linalg.norm(decision[0])), "decision_right_l2": float(np.linalg.norm(decision[1]))})
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
        write_json(output / "model_resources.json", resources)
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
    parser = argparse.ArgumentParser(description="Independent reward linear learner diagnostic")
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
        from .reward_linear_analysis import analyze
        analyze(args.out)
    elif args.command == "analyze":
        from .reward_linear_analysis import analyze
        analyze(args.out)
    else:
        from .reward_linear_audit import verify
        result = verify(args.out)
        print(json.dumps(result))
        if not result["passed"]:
            raise SystemExit(1)

if __name__ == "__main__":
    main()
