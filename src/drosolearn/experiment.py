"""Prospectively frozen experiments with all seeds/conditions and raw records."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
import numpy as np
from .data import make_dataset, assert_disjoint, analytical_accuracy, NUISANCE_NAMES
from .graph import load_projection, rewire, random_ann, sha256
from .model import Learner

CONDITIONS = {
    "fixed": ("real", False, True, True), "plastic": ("real", True, True, True),
    "rewired_fixed": ("rewired", False, True, True), "rewired_plastic": ("rewired", True, True, True),
    "ann_fixed": ("ann", False, True, True), "ann_plastic": ("ann", True, True, True),
    "frozen": ("real", False, False, False), "random_reward": ("real", True, True, True),
    "internal_only": ("real", True, True, False)
}

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

def git_state(root):
    def command(args):
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unavailable"
    return {"commit": command(["rev-parse", "HEAD"]), "status": command(["status", "--porcelain"])}

def source_hashes(root):
    files = sorted((root / "src").rglob("*.py")) + sorted((root / "data").rglob("*"))
    files += [root / "docs/experimental-protocol.md", root / "pyproject.toml"]
    return {p.relative_to(root).as_posix(): sha256(p) for p in files if p.is_file()}

def validate_config(cfg):
    if cfg["scope"] not in ("smoke", "confirmatory"):
        raise ValueError("Unknown scope")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Duplicate or missing seeds")
    if any(type(s) is not int or s < 0 for s in cfg["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if cfg["scope"] == "confirmatory" and len(cfg["seeds"]) < 10:
        raise ValueError("Confirmatory run needs at least ten registered seeds")
    if len(set(cfg["conditions"])) != len(cfg["conditions"]) or not cfg["conditions"]:
        raise ValueError("Duplicate or empty condition list")
    if any(c not in CONDITIONS for c in cfg["conditions"]):
        raise ValueError("Unknown condition")
    for k in ("checkpoint_every", "trials_per_phase", "eval_trials"):
        if type(cfg[k]) is not int or cfg[k] <= 0:
            raise ValueError("Budgets and checkpoint interval must be positive integers")
    if cfg["trials_per_phase"] % cfg["checkpoint_every"]:
        raise ValueError("Checkpoint must divide training budget")
    if min(cfg["head_lr"], cfg["internal_lr"]) < 0 or not np.isfinite([cfg["head_lr"], cfg["internal_lr"]]).all():
        raise ValueError("Invalid learning rates")
    for k in ("trials_per_phase", "eval_trials"):
        if cfg[k] <= 0 or cfg[k] % 4:
            raise ValueError("Crossed datasets need positive multiples of four")
    pairs = [cfg["train_pair"], *cfg["transfer_pairs"]]
    if any(len(p) != 2 or any(type(n) is not int for n in p) or not 1 <= p[0] < p[1] <= 8 for p in pairs):
        raise ValueError("Invalid quantity pair")
    if len(set(map(tuple, pairs))) != len(pairs):
        raise ValueError("Duplicate training/transfer pairs")
    for v in cfg["criteria"].values():
        if not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 1:
            raise ValueError("Criteria must be finite probabilities in [0,1]")
    if cfg["criteria"]["confirmatory_seed_fraction"] <= 0:
        raise ValueError("Confirmatory seed fraction must be positive")

def register(root, config_path, destination):
    root, config_path, destination = Path(root), Path(config_path), Path(destination)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(cfg)
    load_projection(root / cfg["graph"])
    record = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
              "config_sha256": sha256(config_path), "source_hashes": source_hashes(root),
              "git": git_state(root), "notice": "local prospective freeze; not an externally registered preregistration"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    return record

def evaluate(model, dataset, larger):
    probability = model.probabilities(dataset.inputs())
    target = dataset.target(larger)
    prediction = (probability > 0.5).astype(int)
    correct = np.where(np.abs(probability - 0.5) < 1e-12, 0.5, (prediction == target).astype(float))
    return {"accuracy": float(correct.mean()),
            "expected_reward": float(np.where(target == 1, probability, 1 - probability).mean())}, probability, correct

def make_splits(cfg, seed):
    base = seed * 100
    n, ev = cfg["trials_per_phase"], cfg["eval_trials"]
    pair = tuple(cfg["train_pair"])
    datasets = {"train_acquisition": make_dataset(base + 1, "train_acquisition", n, pair),
                "train_reversal": make_dataset(base + 2, "train_reversal", n, pair),
                "iid": make_dataset(base + 3, "iid", ev, pair),
                "ood": make_dataset(base + 4, "ood", ev, pair, "ood")}
    for i, p in enumerate(cfg["transfer_pairs"]):
        key = "transfer_" + "_".join(map(str, p))
        datasets[key] = make_dataset(base + 10 + i, key, ev, tuple(p))
    assert_disjoint(list(datasets.values()))
    for dataset in datasets.values():
        if analytical_accuracy(dataset) != 1.0 or any(analytical_accuracy(dataset, nuisance_index=i) != 0.5 for i in range(6)):
            raise ValueError("Oracle/cue solvability gate failed")
    return datasets

def csv_writer(path, rows):
    if not rows:
        raise ValueError("No records to preserve")
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def run(root, config_path, registration_path, output):
    root, output = Path(root), Path(output)
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_config(cfg)
    reg = json.loads(Path(registration_path).read_text(encoding="utf-8"))
    if sha256(config_path) != reg["config_sha256"] or source_hashes(root) != reg["source_hashes"]:
        raise ValueError("Config/source/protocol changed after registration; make a new registration")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "registration.json", reg)
    anatomical = load_projection(root / cfg["graph"])
    metadata = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
                "registration_sha256": sha256(registration_path), "git": git_state(root),
                "python": platform.python_version(), "platform": platform.platform(),
                "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "matplotlib")},
                "graph": anatomical.metadata, "seed_unit": "computational repeat, not independent flies",
                "reproduce": f"python -m drosolearn.cli run --config {Path(config_path).relative_to(root).as_posix()} --registration {Path(registration_path).relative_to(root).as_posix()} --out results/NEW_DIRECTORY"}
    write_json(output / "metadata.json", metadata)
    trials, curves, errors, outcomes, controls = [], [], [], [], []
    try:
        for seed in cfg["seeds"]:
            datasets = make_splits(cfg, seed)
            seed_path = output / f"seed_{seed}"
            seed_path.mkdir()
            np.savez_compressed(seed_path / "stimuli.npz", **{
                f"{name}_{key}": value for name, ds in datasets.items()
                for key, value in [("counts", ds.counts), ("nuisance", ds.nuisance),
                                   ("family_ids", np.asarray(ds.family_ids))]})
            graphs = {"real": anatomical, "rewired": rewire(anatomical, seed + 50000),
                      "ann": random_ann(anatomical, seed + 60000)}
            for kind in ("rewired", "ann"):
                controls.append({"seed": seed, "kind": kind, "metadata": graphs[kind].metadata})
                np.savez_compressed(seed_path / f"{kind}_topology.npz", pre=graphs[kind].pre,
                                    post=graphs[kind].post, counts=graphs[kind].weights)
            for condition in cfg["conditions"]:
                kind, plastic, learning, head_learning = CONDITIONS[condition]
                graph = graphs[kind]
                model = Learner(graph, seed + 70000, plastic, learning, cfg["head_lr"], cfg["internal_lr"], head_learning)
                action_rng = np.random.default_rng(seed + 80000)
                reward_rng = np.random.default_rng(seed + 90000)
                path = seed_path / condition
                path.mkdir()
                probe = datasets["iid"].inputs()[:16]
                initial = model.snapshot(probe)
                checkpoints = {}

                def checkpoint(phase, t, larger):
                    snap = model.snapshot(probe)
                    np.savez_compressed(path / f"{phase}_{t:04d}.npz", **snap,
                                        encoder=model.encoder, bias=model.bias, pre=graph.pre, post=graph.post)
                    measures = {}
                    for name, ds in datasets.items():
                        if name.startswith("train_"):
                            continue
                        metrics, prob, correct = evaluate(model, ds, larger)
                        measures[name] = metrics
                        curves.append({"seed": seed, "condition": condition, "phase": phase, "trial": t,
                                       "split": name, "low": int(ds.counts.min()), "high": int(ds.counts.max()),
                                       "ratio": float(ds.counts.min() / ds.counts.max()), **metrics,
                                       "internal_change_l2": float(np.linalg.norm(model.weights - initial["internal_weights"])),
                                       "readout_change_l2": float(np.linalg.norm(model.head - initial["readout_weights"])),
                                       "probe_kc_change_l2": float(np.linalg.norm(snap["kc_activity"] - initial["kc_activity"])),
                                       "probe_mbon_change_l2": float(np.linalg.norm(snap["mbon_activity"] - initial["mbon_activity"]))})
                        if t in (0, cfg["trials_per_phase"]):
                            for i in range(len(ds.counts)):
                                errors.append({"seed": seed, "condition": condition, "phase": phase, "trial": t,
                                               "split": name, "case": i, "family_id": ds.family_ids[i],
                                               "left_count": int(ds.counts[i, 0]), "right_count": int(ds.counts[i, 1]),
                                               "target_action": int(ds.target(larger)[i]), "p_right": float(prob[i]),
                                               "correct": float(correct[i])})
                    checkpoints[phase, t] = measures

                for phase, larger in (("acquisition", True), ("reversal", False)):
                    ds = datasets[f"train_{phase}"]
                    checkpoint(phase, 0, larger)
                    inputs = ds.inputs()
                    for i, stimulus in enumerate(inputs):
                        kc, mb = model.features(stimulus)
                        p = float(model.probabilities(stimulus))
                        action = int(action_rng.random() < p)
                        correct_action = int(ds.target(larger)[i])
                        reward = int(action == correct_action)
                        if condition == "random_reward":
                            reward = int(reward_rng.integers(0, 2))
                        before_w, before_h = model.weights.copy(), model.head.copy()
                        model.update(stimulus, action, reward)
                        trials.append({"seed": seed, "condition": condition, "phase": phase, "trial": i + 1,
                                       "family_id": ds.family_ids[i], "left_count": int(ds.counts[i, 0]),
                                       "right_count": int(ds.counts[i, 1]), "action": action, "reward": reward,
                                       "correct": int(action == correct_action), "p_right": p,
                                       "internal_step_l2": float(np.linalg.norm(model.weights - before_w)),
                                       "readout_step_l2": float(np.linalg.norm(model.head - before_h)),
                                       "kc_left_l2": float(np.linalg.norm(kc[0])), "kc_right_l2": float(np.linalg.norm(kc[1])),
                                       "mbon_left_l2": float(np.linalg.norm(mb[0])), "mbon_right_l2": float(np.linalg.norm(mb[1]))})
                        if (i + 1) % cfg["checkpoint_every"] == 0:
                            checkpoint(phase, i + 1, larger)
                    if not (np.isfinite(model.weights).all() and np.isfinite(model.head).all()):
                        raise ValueError("Nonfinite learned parameters")
                t = cfg["trials_per_phase"]
                before, after = checkpoints["acquisition", 0]["iid"], checkpoints["acquisition", t]["iid"]
                criteria = cfg["criteria"]
                acquired = after["expected_reward"] >= criteria["acquisition_accuracy"] and after["expected_reward"] - before["expected_reward"] >= criteria["acquisition_gain"]
                first = next((i for i in range(cfg["checkpoint_every"], t + 1, cfg["checkpoint_every"])
                              if checkpoints["acquisition", i]["iid"]["expected_reward"] >= criteria["acquisition_accuracy"]), None)
                outcomes.append({"seed": seed, "condition": condition, "primary_metric": "exact stochastic choice accuracy (target probability)",
                                 "before_accuracy": before["expected_reward"], "after_accuracy": after["expected_reward"],
                                 "before_greedy_accuracy": before["accuracy"], "after_greedy_accuracy": after["accuracy"],
                                 "gain": after["expected_reward"] - before["expected_reward"],
                                 "acquisition_criterion_met": acquired, "first_threshold_checkpoint": first,
                                 "ood_accuracy": checkpoints["acquisition", t]["ood"]["expected_reward"],
                                 "generalization_criterion_met": acquired and checkpoints["acquisition", t]["ood"]["expected_reward"] >= criteria["generalization_accuracy"],
                                 "transfer": {name: {**m, "interpretable_after_acquisition": acquired}
                                              for name, m in checkpoints["acquisition", t].items() if name.startswith("transfer_")},
                                 "reversal_before_accuracy": checkpoints["reversal", 0]["iid"]["expected_reward"],
                                 "reversal_after_accuracy": checkpoints["reversal", t]["iid"]["expected_reward"],
                                 "reversal_criterion_met": acquired and checkpoints["reversal", t]["iid"]["expected_reward"] >= criteria["reversal_accuracy"],
                                 "trainable_internal_parameters": len(model.weights) if plastic and learning else 0,
                                 "trainable_readout_parameters": len(model.head) if head_learning and learning else 0,
                                 "effective_readout_inputs": len(set(graph.post)), "head_updates": model.head_updates,
                                 "internal_updates": model.internal_updates})
            write_json(seed_path / "cue_audit.json", {name: {"oracle_accuracy": analytical_accuracy(ds),
                        "single_nuisance_accuracy": {n: analytical_accuracy(ds, nuisance_index=i) for i, n in enumerate(NUISANCE_NAMES)}}
                         for name, ds in datasets.items()})
        csv_writer(output / "trials.csv", trials)
        csv_writer(output / "learning_curves.csv", curves)
        csv_writer(output / "evaluation_cases.csv", errors)
        write_json(output / "controls.json", controls)
        write_json(output / "summary.json", {"scope": cfg["scope"], "outcomes": outcomes,
                   "status": "completed", "interpretation": "abstract reward association only; no evidence for visual numerosity, mathematical rule extraction, or shared biological mechanism"})
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["status"] = "completed"
        write_json(output / "metadata.json", metadata)
        # Hash all primary artifacts; plotting/report files are subsequently added by analysis.
        write_json(output / "artifact_hashes.json", {p.relative_to(output).as_posix(): sha256(p)
                    for p in sorted(output.rglob("*")) if p.is_file()})
    except Exception as e:
        metadata["status"] = "failed"
        metadata["failure"] = repr(e)
        write_json(output / "metadata.json", metadata)
        if trials: csv_writer(output / "partial_trials.csv", trials)
        if curves: csv_writer(output / "partial_learning_curves.csv", curves)
        raise
    return outcomes
