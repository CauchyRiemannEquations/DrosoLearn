"""Additive diagnostic of prior learning, head reset, and reversal cost.

Keeps the v0.1.0 learner/update/protocol unchanged; uses separate registrations.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import importlib.metadata
import json
import os
import platform
import numpy as np
from .data import make_dataset, assert_disjoint, analytical_accuracy
from .experiment import CONDITIONS, git_state, write_json
from .graph import load_projection, rewire, random_ann, sha256
from .model import Learner

HISTORIES = ("high_then_low", "low_then_low", "high_then_high", "fresh_low", "reset_head_after_high")
FAMILIES = ("fixed", "plastic", "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic")
TRIAL_FIELDS = ("seed", "condition", "history", "phase", "trial", "family_id", "left_count", "right_count",
                "action", "reward", "p_right", "optimizer_factor", "internal_step_l2", "readout_step_l2",
                "kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2")
CURVE_FIELDS = ("seed", "condition", "history", "phase", "trial", "split", "p_larger", "p_smaller",
                "p_target", "greedy_accuracy", "mean_policy_sensitivity", "internal_change_l2",
                "readout_change_l2", "probe_kc_change_l2", "probe_mbon_change_l2", "lower_bound_fraction", "upper_bound_fraction")
CASE_FIELDS = ("seed", "condition", "history", "phase", "trial", "split", "case", "family_id",
               "left_count", "right_count", "target_action", "p_right", "correct")

def validate_config(cfg):
    if cfg["scope"] not in ("preregistered_diagnostic", "integration_test"):
        raise ValueError("Unsupported study scope")
    if not cfg["seeds"] or any(type(s) is not int or s < 0 for s in cfg["seeds"]) or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Invalid or duplicate seeds")
    if cfg["scope"] == "preregistered_diagnostic" and len(cfg["seeds"]) != 10:
        raise ValueError("Diagnostic requires ten preregistered seeds")
    if not cfg["conditions"] or len(set(cfg["conditions"])) != len(cfg["conditions"]) or any(c not in FAMILIES for c in cfg["conditions"]):
        raise ValueError("Invalid model families")
    if tuple(cfg["histories"]) != HISTORIES:
        raise ValueError("All five histories, in registered order, are required")
    for k in ("trials_per_phase", "checkpoint_every", "eval_trials", "probe_rows"):
        if type(cfg[k]) is not int or cfg[k] <= 0:
            raise ValueError("Budgets must be positive integers")
    if cfg["trials_per_phase"] % 4 or cfg["eval_trials"] % 4 or cfg["trials_per_phase"] % cfg["checkpoint_every"] or cfg["probe_rows"] > cfg["eval_trials"]:
        raise ValueError("Invalid crossing/checkpoint/probe budget")
    pair = cfg["pair"]
    if len(pair) != 2 or any(type(x) is not int for x in pair) or not 1 <= pair[0] < pair[1] <= 8:
        raise ValueError("Invalid quantity pair")
    for k in ("head_lr", "internal_lr"):
        if not np.isfinite(cfg[k]) or cfg[k] < 0:
            raise ValueError("Invalid learning rate")
    for v in cfg["criteria"].values():
        if type(v) not in (float, int) or not np.isfinite(v) or not 0 <= v <= 1:
            raise ValueError("Invalid criterion")
    if cfg["criteria"]["positive_seed_fraction"] <= 0:
        raise ValueError("Positive seed fraction must be positive")

def study_hashes(root, cfg):
    root = Path(root)
    files = list((root / "src").rglob("*.py")) + list((root / cfg["graph"]).glob("*"))
    files += [root / "docs/reversal-protocol.md", root / "pyproject.toml", root / "requirements-lock.txt",
              root / "data/source-manifest.json", root / "data/LICENSE.md"]
    # Independently acquired human files are not inputs to this experiment.
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files) if p.is_file()}

def register(root, config, destination):
    root, config, destination = Path(root), Path(config), Path(destination)
    cfg = json.loads(config.read_text(encoding="utf-8"))
    validate_config(cfg)
    load_projection(root / cfg["graph"])
    record = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
              "config_sha256": sha256(config), "source_hashes": study_hashes(root, cfg), "git": git_state(root),
              "notice": "prospective private Git/local freeze; not an accredited public preregistration",
              "previous_study": "v0.1.0-smoke; immutable prior results; this is a separate diagnostic"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    return record

def make_splits(cfg, seed):
    n, ev, pair, base = cfg["trials_per_phase"], cfg["eval_trials"], tuple(cfg["pair"]), seed * 100
    datasets = {"train_phase1": make_dataset(base + 21, "train_phase1", n, pair),
                "train_phase2": make_dataset(base + 22, "train_phase2", n, pair),
                "iid": make_dataset(base + 23, "iid", ev, pair),
                "ood": make_dataset(base + 24, "ood", ev, pair, "ood")}
    assert_disjoint(list(datasets.values()))
    for ds in datasets.values():
        if analytical_accuracy(ds) != 1 or any(analytical_accuracy(ds, nuisance_index=i) != 0.5 for i in range(6)):
            raise ValueError("Cue/solvability gate failed")
    return datasets

def probe_metrics(model, ds, larger):
    probability = model.probabilities(ds.inputs())
    target = ds.target(larger)
    high = ds.target(True)
    p_large = float(np.where(high == 1, probability, 1 - probability).mean())
    correct = np.where(np.abs(probability - 0.5) < 1e-12, 0.5, ((probability > 0.5).astype(int) == target).astype(float))
    return {"p_larger": p_large, "p_smaller": 1 - p_large, "p_target": p_large if larger else 1 - p_large,
            "greedy_accuracy": float(correct.mean()), "mean_policy_sensitivity": float((probability * (1 - probability)).mean())}, probability, correct

def normalized_auc(values, every, duration):
    values = np.asarray(values, dtype=float)
    if len(values) != duration // every + 1:
        raise ValueError("AUC requires baseline and every checkpoint")
    return float(np.trapezoid(values, dx=every) / duration)

def summarize(cfg, curves, boundaries):
    index = {(r["seed"], r["condition"], r["history"], r["phase"], r["trial"], r["split"]): r for r in curves}
    n, every, crit = cfg["trials_per_phase"], cfg["checkpoint_every"], cfg["criteria"]
    rows, pairs, family_verdicts = [], [], []
    for seed in cfg["seeds"]:
        for family in cfg["conditions"]:
            for history in cfg["histories"]:
                seq = [index[seed, family, history, "phase2", t, "iid"]["p_target"] for t in range(0, n + 1, every)]
                if history == "fresh_low":
                    first_phase_before = first_phase_after = None
                    acquired = None
                else:
                    first_phase_before = index[seed, family, history, "phase1", 0, "iid"]["p_target"]
                    first_phase_after = index[seed, family, history, "phase1", n, "iid"]["p_target"]
                    acquired = first_phase_after >= crit["acquisition_accuracy"] and first_phase_after - first_phase_before >= crit["acquisition_gain"]
                boundary = next(b for b in boundaries if (b["seed"], b["condition"], b["history"]) == (seed, family, history))
                rows.append({"seed": seed, "condition": family, "history": history,
                             "phase1_before": first_phase_before, "phase1_after": first_phase_after,
                             "phase1_acquired": acquired, "phase2_before": seq[0], "phase2_after": seq[-1],
                             "phase2_iid_auc": normalized_auc(seq, every, n),
                             "phase2_ood_after": index[seed, family, history, "phase2", n, "ood"]["p_target"],
                             "endpoint_criterion_met": seq[-1] >= crit["adaptation_accuracy"],
                             "phase2_acquired": seq[-1] >= crit["acquisition_accuracy"] and seq[-1] - seq[0] >= crit["acquisition_gain"],
                             "qualified_reversal_criterion_met": bool(acquired and seq[-1] >= crit["adaptation_accuracy"]) if history in ("high_then_low", "reset_head_after_high") else None,
                             "first_threshold_checkpoint": next((t for t, p in zip(range(0, n + 1, every), seq) if p >= crit["adaptation_accuracy"]), None),
                             "phase1_training_trials": 0 if history == "fresh_low" else n,
                             "phase2_training_trials": n, "head_reset_l2": boundary["head_reset_l2"],
                             "head_updates": boundary["final_head_updates"], "internal_updates": boundary["final_internal_updates"]})
            indexed = {r["history"]: r for r in rows if r["seed"] == seed and r["condition"] == family}
            fresh, reversal, reset, continued_low = [indexed[h] for h in ("fresh_low", "high_then_low", "reset_head_after_high", "low_then_low")]
            pairs.append({"seed": seed, "condition": family,
                          "history_auc_penalty": fresh["phase2_iid_auc"] - reversal["phase2_iid_auc"],
                          "history_endpoint_penalty": fresh["phase2_after"] - reversal["phase2_after"],
                          "reset_auc_gain": reset["phase2_iid_auc"] - reversal["phase2_iid_auc"],
                          "reset_minus_fresh_auc": reset["phase2_iid_auc"] - fresh["phase2_iid_auc"],
                          "continued_low_minus_reversal_auc": continued_low["phase2_iid_auc"] - reversal["phase2_iid_auc"],
                          "old_rule_acquired": reversal["phase1_acquired"]})
    for family in cfg["conditions"]:
        selected = [r for r in pairs if r["condition"] == family]
        penalties = [r["history_auc_penalty"] for r in selected]
        qualified = [r for r in selected if r["old_rule_acquired"]]
        mean = float(np.mean(penalties)); fraction = sum(p > 0 for p in penalties) / len(penalties)
        family_verdicts.append({"condition": family, "n_seeds": len(selected), "mean_history_auc_penalty": mean,
                               "sd_history_auc_penalty": float(np.std(penalties, ddof=1)) if len(penalties) > 1 else 0.0,
                               "positive_seed_count": sum(p > 0 for p in penalties),
                               "positive_seed_fraction": fraction, "history_cost_criterion_met": mean >= crit["history_auc_penalty"] and fraction >= crit["positive_seed_fraction"],
                               "qualified_seed_count": len(qualified),
                               "qualified_mean_auc_penalty": float(np.mean([r["history_auc_penalty"] for r in qualified])) if qualified else None,
                               "note": "all-seed primary; qualified subset secondary; seed is not an animal"})
    return {"status": "completed", "scope": cfg["scope"], "outcomes": rows, "paired_contrasts": pairs,
            "family_verdicts": family_verdicts, "interpretation": "algorithmic history/intervention diagnostic on oracle input; no human/biological mechanism established"}

def run(root, config, registration, output):
    root, config, registration, output = map(Path, (root, config, registration, output))
    cfg = json.loads(config.read_text(encoding="utf-8")); validate_config(cfg)
    reg = json.loads(registration.read_text(encoding="utf-8"))
    if sha256(config) != reg["config_sha256"] or study_hashes(root, cfg) != reg["source_hashes"]:
        raise ValueError("Registered config/source/protocol changed; create a new registration")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "registration.json", reg)
    original = load_projection(root / cfg["graph"])
    def recorded_path(path):
        resolved = path.resolve()
        return (resolved.relative_to(root.resolve()) if resolved.is_relative_to(root.resolve()) else resolved).as_posix()
    metadata = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg, "git": git_state(root),
                "registration_sha256": sha256(registration), "graph": original.metadata,
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "matplotlib")},
                "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "seed_unit": "computational repeat of one supplied topology; matched histories are dependent",
                "rng": {"phase1_actions": "seed+180000", "phase2_actions": "seed+190000"},
                "reproduce": f"python -m drosolearn.reversal run --config {recorded_path(config)} --registration {recorded_path(registration)} --out results/NEW_REVERSAL",
                "reproduce_argv": ["python", "-m", "drosolearn.reversal", "run", "--config", recorded_path(config),
                                   "--registration", recorded_path(registration), "--out", "results/NEW_REVERSAL"]}
    write_json(output / "metadata.json", metadata)
    curves, boundaries, controls = [], [], []
    try:
        for seed in cfg["seeds"]:
            ds = make_splits(cfg, seed); seed_path = output / f"seed_{seed}"; seed_path.mkdir()
            uniforms = {phase: np.random.default_rng(seed + offset).random(cfg["trials_per_phase"])
                        for phase, offset in (("phase1", 180000), ("phase2", 190000))}
            np.savez_compressed(seed_path / "stimuli.npz", **{f"{name}_{key}": value for name, data in ds.items()
                                 for key, value in (("counts", data.counts), ("nuisance", data.nuisance), ("family_ids", np.asarray(data.family_ids)))},
                                 phase1_uniforms=uniforms["phase1"], phase2_uniforms=uniforms["phase2"])
            graphs = {"real": original, "rewired": rewire(original, seed + 50000), "ann": random_ann(original, seed + 60000)}
            for kind, graph in graphs.items():
                np.savez_compressed(seed_path / f"{kind}_topology.npz", pre=graph.pre, post=graph.post, counts=graph.weights)
                controls.append({"seed": seed, "kind": kind, "metadata": graph.metadata})
            with (seed_path / "trials.csv").open("w", encoding="utf-8", newline="") as trial_file, (seed_path / "evaluation_cases.csv").open("w", encoding="utf-8", newline="") as case_file:
                trial_writer = csv.DictWriter(trial_file, fieldnames=TRIAL_FIELDS); trial_writer.writeheader()
                case_writer = csv.DictWriter(case_file, fieldnames=CASE_FIELDS); case_writer.writeheader()
                for condition in cfg["conditions"]:
                    kind, plastic, _, _ = CONDITIONS[condition]; graph = graphs[kind]
                    for history in cfg["histories"]:
                        model = Learner(graph, seed + 70000, plastic, True, cfg["head_lr"], cfg["internal_lr"])
                        path = seed_path / condition / history; path.mkdir(parents=True)
                        probe = ds["iid"].inputs()[:cfg["probe_rows"]]; initial = model.snapshot(probe)

                        def checkpoint(phase, t, larger, record=True):
                            snap = model.snapshot(probe)
                            np.savez_compressed(path / f"{phase}_{t:04d}.npz", **snap, encoder=model.encoder, bias=model.bias,
                                                pre=graph.pre, post=graph.post)
                            if not record: return
                            for split in ("iid", "ood"):
                                measures, p, correct = probe_metrics(model, ds[split], larger)
                                curves.append({"seed": seed, "condition": condition, "history": history, "phase": phase,
                                               "trial": t, "split": split, **measures,
                                               "internal_change_l2": float(np.linalg.norm(model.weights - initial["internal_weights"])),
                                               "readout_change_l2": float(np.linalg.norm(model.head - initial["readout_weights"])),
                                               "probe_kc_change_l2": float(np.linalg.norm(snap["kc_activity"] - initial["kc_activity"])),
                                               "probe_mbon_change_l2": float(np.linalg.norm(snap["mbon_activity"] - initial["mbon_activity"])),
                                               "lower_bound_fraction": float(np.mean(model.weights == 0)),
                                               "upper_bound_fraction": float(np.mean(model.weights == model.upper))})
                                if t in (0, cfg["trials_per_phase"]):
                                    data = ds[split]; targets = data.target(larger)
                                    for i in range(len(data.counts)):
                                        case_writer.writerow({"seed": seed, "condition": condition, "history": history, "phase": phase,
                                                              "trial": t, "split": split, "case": i, "family_id": data.family_ids[i],
                                                              "left_count": int(data.counts[i,0]), "right_count": int(data.counts[i,1]),
                                                              "target_action": int(targets[i]), "p_right": float(p[i]), "correct": float(correct[i])})

                        def train_phase(phase, larger):
                            data = ds[f"train_{phase}"]; inputs, targets = data.inputs(), data.target(larger)
                            checkpoint(phase, 0, larger)
                            for i, stimulus in enumerate(inputs):
                                kc, mb = model.features(stimulus)
                                p = float(1 / (1 + np.exp(-np.clip((mb[1] - mb[0]) @ model.head, -40, 40))))
                                action = int(uniforms[phase][i] < p); reward = int(action == targets[i])
                                previous_w, previous_h = model.weights.copy(), model.head.copy()
                                model.update(stimulus, action, reward)
                                trial_writer.writerow({"seed": seed, "condition": condition, "history": history, "phase": phase,
                                                       "trial": i + 1, "family_id": data.family_ids[i],
                                                       "left_count": int(data.counts[i,0]), "right_count": int(data.counts[i,1]),
                                                       "action": action, "reward": reward, "p_right": p,
                                                       "optimizer_factor": float((action - p) * (reward - 0.5)),
                                                       "internal_step_l2": float(np.linalg.norm(model.weights - previous_w)),
                                                       "readout_step_l2": float(np.linalg.norm(model.head - previous_h)),
                                                       "kc_left_l2": float(np.linalg.norm(kc[0])), "kc_right_l2": float(np.linalg.norm(kc[1])),
                                                       "mbon_left_l2": float(np.linalg.norm(mb[0])), "mbon_right_l2": float(np.linalg.norm(mb[1]))})
                                if (i + 1) % cfg["checkpoint_every"] == 0: checkpoint(phase, i + 1, larger)

                        if history != "fresh_low": train_phase("phase1", history != "low_then_low")
                        else: checkpoint("phase1", 0, False, record=False)
                        head_before = model.head.copy(); weights_before = model.weights.copy()
                        if history == "reset_head_after_high": model.head = initial["readout_weights"].copy()
                        boundary = {"seed": seed, "condition": condition, "history": history,
                                    "head_reset_l2": float(np.linalg.norm(model.head - head_before)),
                                    "internal_reset_l2": float(np.linalg.norm(model.weights - weights_before)),
                                    "phase1_training_trials": 0 if history == "fresh_low" else cfg["trials_per_phase"],
                                    "head_updates_before_phase2": model.head_updates, "internal_updates_before_phase2": model.internal_updates}
                        train_phase("phase2", history == "high_then_high")
                        if not (np.isfinite(model.head).all() and np.isfinite(model.weights).all()): raise ValueError("Nonfinite parameters")
                        boundary.update(final_head_updates=model.head_updates, final_internal_updates=model.internal_updates)
                        boundaries.append(boundary); write_json(path / "boundary.json", boundary)
            print(f"completed seed {seed} ({cfg['seeds'].index(seed)+1}/{len(cfg['seeds'])})", flush=True)
        with (output / "learning_curves.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CURVE_FIELDS); writer.writeheader(); writer.writerows(curves)
        write_json(output / "controls.json", controls)
        summary = summarize(cfg, curves, boundaries); write_json(output / "summary.json", summary)
        metadata.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "metadata.json", metadata)
        write_json(output / "artifact_hashes.json", {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
        return summary
    except Exception as error:
        metadata.update(status="failed", failure=repr(error), failed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "metadata.json", metadata)
        if curves:
            from .experiment import csv_writer
            csv_writer(output / "partial_learning_curves.csv", curves)
        raise

def main():
    parser = argparse.ArgumentParser(description="Reversal history diagnostic; prior study preserved")
    parser.add_argument("command", choices=("register", "run", "analyze", "verify"))
    parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path)
    parser.add_argument("--registration", type=Path); parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(); args.root = args.root.resolve()
    for name in ("config", "registration", "out"):
        value = getattr(args, name)
        if value is not None: setattr(args, name, (value if value.is_absolute() else args.root / value).resolve())
    if args.command == "register":
        if args.config is None: parser.error("--config required")
        record = register(args.root, args.config, args.out)
        print(json.dumps({"registered_at_utc": record["registered_at_utc"], "config_sha256": record["config_sha256"]}))
    elif args.command == "run":
        if args.config is None or args.registration is None: parser.error("--config and --registration required")
        run(args.root, args.config, args.registration, args.out)
        from .reversal_analysis import analyze
        analyze(args.out)
    elif args.command == "analyze":
        from .reversal_analysis import analyze
        analyze(args.out)
    else:
        from .reversal_audit import verify
        result = verify(args.out); print(json.dumps(result))
        if not result["passed"]: raise SystemExit(1)

if __name__ == "__main__": main()
