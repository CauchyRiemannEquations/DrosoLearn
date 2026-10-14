"""Shared-prefix readout magnitude intervention; prior studies remain immutable."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import csv
import importlib.metadata
import json
import os
import platform
import numpy as np

from .data import make_dataset, assert_disjoint, analytical_accuracy
from .experiment import CONDITIONS, git_state, write_json
from .graph import load_projection, random_ann, rewire, sha256
from .reward_linear import LinearLearner, NormalizedLearner
from .bandit_reversal import ExpectedLinear, expected_factor
from .reversal import probe_metrics, normalized_auc
from .reversal import TRIAL_FIELDS as OLD_TRIAL_FIELDS, CURVE_FIELDS as OLD_CURVE_FIELDS
from .spacing import FAMILIES as NETWORK_FAMILIES

FAMILIES = ("linear_normalized", *NETWORK_FAMILIES)
HISTORIES = ("fresh_low", "gamma_0", "gamma_025", "gamma_05", "gamma_1")
GAMMAS = {"gamma_0": 0., "gamma_025": .25, "gamma_05": .5, "gamma_1": 1.}
SHARED_PREFIX = "shared_prefix"
TRIAL_FIELDS = (*OLD_TRIAL_FIELDS, "decision_left_l2", "decision_right_l2")
CURVE_FIELDS = (*OLD_CURVE_FIELDS, "probe_decision_change_l2", "larger")
EXPECTED_FIELDS = ("seed", "history", "phase", "trial", "family_id", "left_count", "right_count",
                   "target", "p_right", "expected_factor", "readout_step_l2")


def clone_scaled(model, gamma):
    """Independent branch: scale readout only; retain all other ancestor state."""
    if type(gamma) not in (int, float) or not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Finite attenuation in [0,1] required")
    branch = copy.deepcopy(model)
    branch.head *= gamma
    return branch


def raw_margin(model, inputs):
    _, features = model.features(inputs)
    scores = features @ model.head
    return scores[..., 1] - scores[..., 0]


def validate_config(cfg):
    if cfg["scope"] not in ("preregistered_diagnostic", "integration_test"):
        raise ValueError("Unsupported study scope")
    seeds = cfg["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Invalid seeds")
    if cfg["scope"] == "integration_test" and any(s in range(8001, 8011) for s in seeds):
        raise ValueError("Scientific seeds cannot be development fixtures")
    fixed = {"experiment_id": "head_attenuation_diagnostic_v1",
             "graph": "data/flywire_783_mb_left_kc512_s701", "pair": [2, 4],
             "head_initialization": "zeros_all_conditions",
             "feature_normalization": "fixed_initial_phase1_train_feature_gap_rms",
             "expected_reference": "conditional_expected_action_linear",
             "attenuation_mode": "scale_readout_only_preserve_all_other_state",
             "gammas": [0., .25, .5, 1.], "primary_gamma": .25}
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError("Fixed study identity, initialization and reference design required")
    if any(type(gamma) not in (int, float) or not np.isfinite(gamma) for gamma in cfg["gammas"]) or type(cfg["primary_gamma"]) not in (int, float):
        raise ValueError("Numeric attenuation declarations required; booleans are not scales")
    if tuple(cfg["conditions"]) != FAMILIES or tuple(cfg["histories"]) != HISTORIES:
        raise ValueError("Seven families and five fixed histories required")
    for k in ("trials_per_phase", "checkpoint_every", "eval_trials", "probe_rows"):
        if type(cfg[k]) is not int or cfg[k] <= 0:
            raise ValueError("Invalid budget")
    if cfg["trials_per_phase"] % 4 or cfg["eval_trials"] % 4 or cfg["trials_per_phase"] % cfg["checkpoint_every"] or cfg["probe_rows"] > cfg["eval_trials"]:
        raise ValueError("Crossed trials, complete checkpoints and valid probes required")
    if any(type(cfg[k]) not in (int, float) or not np.isfinite(cfg[k]) or cfg[k] < 0 for k in ("head_lr", "internal_lr")):
        raise ValueError("Invalid learning rate")
    c = cfg["criteria"]
    if set(c) != {"acquisition_accuracy", "acquisition_gain", "adaptation_accuracy", "endpoint_benefit", "seed_pass_fraction"} or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in c.values()) or c["seed_pass_fraction"] <= 0:
        raise ValueError("Invalid criteria")
    if cfg["scope"] == "preregistered_diagnostic":
        if seeds != list(range(8001, 8011)) or tuple(cfg[k] for k in ("trials_per_phase", "checkpoint_every", "eval_trials", "probe_rows", "head_lr", "internal_lr")) != (400, 100, 128, 8, .1, .01):
            raise ValueError("New seeds and fixed resources required")
        if c != {"acquisition_accuracy": .75, "acquisition_gain": .15, "adaptation_accuracy": .75, "endpoint_benefit": .15, "seed_pass_fraction": .8}:
            raise ValueError("Fixed prospective criteria required")


def study_hashes(root, cfg):
    root = Path(root)
    paths = list((root / "src").rglob("*.py")) + list((root / cfg["graph"]).glob("*"))
    paths += [root / p for p in ("docs/head-attenuation-protocol.md", "docs/head-attenuation-background.md", "pyproject.toml", "requirements-lock.txt", "data/source-manifest.json", "data/LICENSE.md")]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths) if p.is_file()}


def register(root, config, destination):
    root, config, destination = map(Path, (root, config, destination))
    cfg = json.loads(config.read_text(encoding="utf-8"))
    validate_config(cfg)
    load_projection(root / cfg["graph"])
    record = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
              "config_sha256": sha256(config), "source_hashes": study_hashes(root, cfg), "git": git_state(root),
              "notice": "prospective private Git freeze, not accredited public preregistration",
              "previous_studies": ["v0.1.0-smoke", "study-reversal-v1", "study-encoding-v1", "study-spacing-v1", "study-linearity-v1", "study-reward-linear-v1", "study-bandit-reversal-v1"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as out:
        json.dump(record, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    return record


def make_splits(cfg, seed):
    n, ev, base = cfg["trials_per_phase"], cfg["eval_trials"], seed * 100
    data = {"train_phase1": make_dataset(base + 31, "train_phase1", n),
            "train_phase2": make_dataset(base + 34, "train_phase2", n),
            "iid": make_dataset(base + 32, "iid", ev), "ood": make_dataset(base + 33, "ood", ev, domain="ood")}
    assert_disjoint(list(data.values()))
    for ds in data.values():
        if analytical_accuracy(ds) != 1 or any(analytical_accuracy(ds, nuisance_index=i) != .5 for i in range(6)):
            raise ValueError("Solvability or nuisance crossing failed")
    return data


def state_pack(model, probe, datasets):
    snap = {**model.snapshot(probe), "encoder": model.encoder.copy(), "bias": model.bias.copy(),
            "pre": model.pre.copy(), "post": model.post.copy(),
            "head_updates": np.asarray(model.head_updates), "internal_updates": np.asarray(model.internal_updates),
            "initial_internal_weights": getattr(model, "initial_weights", np.empty(0)).copy(),
            "internal_upper_bounds": getattr(model, "upper", np.empty(0)).copy(),
            "incoming_normalizer": getattr(model, "normalizer", np.empty(0)).copy()}
    for split in ("iid", "ood"):
        inputs = datasets[split].inputs()
        snap[f"{split}_p_right"] = model.probabilities(inputs)
        snap[f"{split}_raw_margin"] = raw_margin(model, inputs)
    return snap


def summarize(cfg, curves):
    index = {(r["seed"], r["condition"], r["history"], r["phase"], r["trial"], r["split"]): r for r in curves}
    n, every, c = cfg["trials_per_phase"], cfg["checkpoint_every"], cfg["criteria"]
    outcomes, prefixes, contrasts, verdicts = [], [], [], []
    for seed in cfg["seeds"]:
        for condition in (*cfg["conditions"], "expected_linear"):
            reference = condition == "expected_linear"
            p0, p1 = (index[seed, condition, SHARED_PREFIX, "phase1", t, "iid"]["p_target"] for t in (0,n))
            acquired = None if reference else bool(p1 >= c["acquisition_accuracy"] and p1-p0 >= c["acquisition_gain"])
            prefixes.append({"seed": seed, "condition": condition, "before": p0, "after": p1, "gain": p1-p0,
                             "acquired": acquired, "physical_training_updates": n})
            for history in HISTORIES:
                seq = [index[seed, condition, history, "phase2", t, "iid"]["p_target"] for t in range(0,n+1,every)]
                fresh = history == "fresh_low"
                fresh_gate = bool(seq[-1] >= c["acquisition_accuracy"] and seq[-1]-seq[0] >= c["acquisition_gain"])
                passed = bool(seq[-1] >= c["adaptation_accuracy"])
                auc = normalized_auc(seq,every,n)
                outcomes.append({"seed": seed, "condition": condition, "history": history,
                    "gamma": None if fresh else GAMMAS[history],
                    "phase1_before": None if fresh else p0, "phase1_after": None if fresh else p1,
                    "phase1_acquired": None if fresh else acquired,
                    "phase2_before": seq[0], "phase2_after": seq[-1], "phase2_gain": seq[-1]-seq[0],
                    "phase2_auc": auc, "phase2_baseline_adjusted_auc": auc-seq[0],
                    "frozen_no_learning_endpoint": seq[0], "frozen_no_learning_auc": seq[0],
                    "phase2_target_passed": None if reference else passed,
                    "qualified_adaptation": None if reference else bool(fresh_gate if fresh else acquired and passed),
                    "fresh_acquired": fresh_gate if fresh and not reference else None,
                    "first_observed_75_checkpoint": None if reference else next((t for t,v in zip(range(0,n+1,every),seq) if v>=c["adaptation_accuracy"]),None),
                    "phase2_ood_after": index[seed,condition,history,"phase2",n,"ood"]["p_target"],
                    "phase2_greedy_after": index[seed,condition,history,"phase2",n,"iid"]["greedy_accuracy"],
                    "head_updates": n if fresh else 2*n,
                    "internal_updates": (n if fresh else 2*n) if condition in NETWORK_FAMILIES and CONDITIONS[condition][1] else 0,
                    "physical_phase2_updates": n,
                    "algorithm": "conditional_expected_action_reference" if reference else "stochastic_reward"})
    keyed = {(r["seed"],r["condition"],r["history"]):r for r in outcomes}
    for condition in cfg["conditions"]:
        selected=[]
        for seed in cfg["seeds"]:
            part,full,fresh,reset,half=(keyed[seed,condition,h] for h in ("gamma_025","gamma_1","fresh_low","gamma_0","gamma_05"))
            record={"seed":seed,"condition":condition,
                    "endpoint_benefit":part["phase2_after"]-full["phase2_after"],
                    "initial_probability_difference":part["phase2_before"]-full["phase2_before"],
                    "gain_difference":part["phase2_gain"]-full["phase2_gain"],
                    "raw_auc_difference":part["phase2_auc"]-full["phase2_auc"],
                    "baseline_adjusted_auc_difference":part["phase2_baseline_adjusted_auc"]-full["phase2_baseline_adjusted_auc"],
                    "reset_minus_full_endpoint":reset["phase2_after"]-full["phase2_after"],
                    "reset_minus_fresh_endpoint":reset["phase2_after"]-fresh["phase2_after"],
                    "half_minus_full_endpoint":half["phase2_after"]-full["phase2_after"],
                    "old_rule_acquired":full["phase1_acquired"],"fresh_acquired":fresh["fresh_acquired"]}
            contrasts.append(record); selected.append(record)
        benefits=[r["endpoint_benefit"] for r in selected]
        positive=sum(v>0 for v in benefits); old_count=sum(r["old_rule_acquired"] for r in selected)
        verdicts.append({"condition":condition,"n_seeds":len(selected),"mean_endpoint_benefit":float(np.mean(benefits)),
            "sd_endpoint_benefit":float(np.std(benefits,ddof=1)) if len(benefits)>1 else 0.,
            "positive_seed_count":positive,"high_prefix_acquisition_count":old_count,
            "fresh_acquisition_count":sum(r["fresh_acquired"] for r in selected),
            "criterion_met":bool(np.mean(benefits)>=c["endpoint_benefit"] and positive/len(selected)>=c["seed_pass_fraction"] and old_count/len(selected)>=c["seed_pass_fraction"])})
    comparisons=[]
    for seed in cfg["seeds"]:
        for history in HISTORIES:
            for t in range(0,n+1,every):
                actual=index[seed,"linear_normalized",history,"phase2",t,"iid"]["p_target"]
                expected=index[seed,"expected_linear",history,"phase2",t,"iid"]["p_target"]
                comparisons.append({"seed":seed,"history":history,"trial":t,"stochastic_p_target":actual,"reference_p_target":expected,"difference":actual-expected})
    return {"status":"completed","scope":cfg["scope"],"outcomes":outcomes,"prefix_outcomes":prefixes,
            "paired_contrasts":contrasts,"family_verdicts":verdicts,
            "primary_hypothesis":{"name":"H_A",**next(v for v in verdicts if v["condition"]=="linear_normalized")},
            "reference_comparisons":comparisons,"actual_training_trials":len(cfg["seeds"])*len(cfg["conditions"])*6*n,
            "expected_reference_updates":len(cfg["seeds"])*6*n,
            "interpretation":"readout magnitude intervention; endpoint mixes immediate baseline change and subsequent path; plastic eligibility also scales; reference is not an ensemble mean"}


def run(root, config, registration, output):
    root, config, registration, output = [Path(p).resolve() for p in (root, config, registration, output)]
    cfg, reg = (json.loads(p.read_text(encoding="utf-8")) for p in (config, registration))
    validate_config(cfg)
    if sha256(config) != reg["config_sha256"] or cfg != reg["config"] or study_hashes(root, cfg) != reg["source_hashes"]:
        raise ValueError("Source, configuration or protocol changed after freeze")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "registration.json", reg)
    graph = load_projection(root / cfg["graph"])
    def recorded(path):
        return (path.relative_to(root) if path.is_relative_to(root) else path).as_posix()
    try:
        source_root = Path(os.path.relpath(root, output)).as_posix()
    except ValueError:
        source_root = root.as_posix()
    metadata = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg, "git": git_state(root),
                "source_root": source_root, "config_path": recorded(config), "registration_sha256": sha256(registration),
                "graph": graph.metadata, "python": platform.python_version(), "platform": platform.platform(),
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "matplotlib")},
                "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "rng": {"phase1_actions": "seed+280000", "phase2_actions": "seed+300000", "encoder": "seed+70000", "rewired": "seed+50000", "ann": "seed+60000", "data": "seed*100+31 phase1,+34 phase2,+32 IID,+33 OOD"},
                "reference_scope": "label-privileged conditional expected-action drift; no rewards sampled; not ensemble expectation",
                "training_reuse": "one actual high prefix per seed/family; four independent scaled-head clones; separate fresh low; six physically trained phases",
                "reproduce_argv": ["python", "-m", "drosolearn.head_attenuation", "run", "--config", recorded(config), "--registration", recorded(registration), "--out", "results/NEW_HEAD_ATTENUATION"]}
    write_json(output / "metadata.json", metadata)
    curves, boundaries, resources, controls = [], [], [], []
    n = cfg["trials_per_phase"]
    try:
        for seed in cfg["seeds"]:
            datasets = make_splits(cfg, seed)
            base = output / f"seed_{seed}"
            base.mkdir()
            uniforms = {"phase1": np.random.default_rng(seed + 280000).random(n), "phase2": np.random.default_rng(seed + 300000).random(n)}
            np.savez_compressed(base / "stimuli.npz", **{f"{name}_{field}": value for name, ds in datasets.items() for field, value in (("counts", ds.counts), ("nuisance", ds.nuisance), ("family_ids", np.asarray(ds.family_ids)))}, action_phase1_uniforms=uniforms["phase1"], action_phase2_uniforms=uniforms["phase2"])
            graphs = {"real": graph, "rewired": rewire(graph, seed + 50000), "ann": random_ann(graph, seed + 60000)}
            for kind, g in graphs.items():
                np.savez_compressed(base / f"{kind}_topology.npz", pre=g.pre, post=g.post, counts=g.weights)
                controls.append({"seed": seed, "kind": kind, "metadata": g.metadata})
            with (base / "trials.csv").open("w", encoding="utf-8", newline="") as actual_file, (base / "expected_updates.csv").open("w", encoding="utf-8", newline="") as expected_file:
                actual_writer = csv.DictWriter(actual_file, fieldnames=TRIAL_FIELDS)
                expected_writer = csv.DictWriter(expected_file, fieldnames=EXPECTED_FIELDS)
                actual_writer.writeheader()
                expected_writer.writeheader()
                for condition in (*cfg["conditions"], "expected_linear"):
                    train = datasets["train_phase1"].inputs()
                    if condition == "expected_linear":
                        ancestor = ExpectedLinear(train,cfg["head_lr"])
                    elif condition == "linear_normalized":
                        ancestor = LinearLearner(train,head_lr=cfg["head_lr"])
                    else:
                        kind,plastic,_,_ = CONDITIONS[condition]
                        ancestor = NormalizedLearner(graphs[kind],seed+70000,train,plastic,cfg["head_lr"],cfg["internal_lr"])
                    pristine = copy.deepcopy(ancestor)
                    probe = datasets["iid"].inputs()[:cfg["probe_rows"]]
                    initial = ancestor.snapshot(probe)
                    def checkpoint(model,path,history,phase,t,larger):
                        snap = state_pack(model,probe,datasets)
                        for split in ("iid","ood"):
                            metrics,_,_ = probe_metrics(model,datasets[split],larger)
                            w=model.weights
                            curves.append({"seed":seed,"condition":condition,"history":history,"phase":phase,"trial":t,"split":split,**metrics,
                                "internal_change_l2":float(np.linalg.norm(w-initial["internal_weights"])),
                                "readout_change_l2":float(np.linalg.norm(model.head-initial["readout_weights"])),
                                "probe_kc_change_l2":float(np.linalg.norm(snap["kc_activity"]-initial["kc_activity"])),
                                "probe_mbon_change_l2":float(np.linalg.norm(snap["mbon_activity"]-initial["mbon_activity"])),
                                "probe_decision_change_l2":float(np.linalg.norm(snap["decision_activity"]-initial["decision_activity"])),
                                "lower_bound_fraction":float(np.mean(w<=0)) if w.size else 0.,
                                "upper_bound_fraction":float(np.mean(w>=model.upper)) if w.size else 0.,"larger":int(larger)})
                        after = state_pack(model,probe,datasets)
                        if any(not np.array_equal(snap[k],after[k]) for k in snap):
                            raise ValueError("Evaluation changed learner state")
                        np.savez_compressed(path/f"{phase}_{t:04d}.npz",**snap)
                    def train_phase(model,path,history,phase,larger):
                        ds=datasets[f"train_{phase}"]; targets=ds.target(larger)
                        checkpoint(model,path,history,phase,0,larger)
                        for i,x in enumerate(ds.inputs()):
                            kc,decision=model.features(x); _,mb=model.raw_activities(x)
                            p=float(model.probabilities(x)); old_w,old_h=model.weights.copy(),model.head.copy()
                            if condition=="expected_linear":
                                factor=expected_factor(p,int(targets[i])); model.update_expected(x,int(targets[i]))
                                expected_writer.writerow({"seed":seed,"history":history,"phase":phase,"trial":i+1,
                                    "family_id":ds.family_ids[i],"left_count":int(ds.counts[i,0]),"right_count":int(ds.counts[i,1]),
                                    "target":int(targets[i]),"p_right":p,"expected_factor":factor,"readout_step_l2":float(np.linalg.norm(model.head-old_h))})
                            else:
                                action=int(uniforms[phase][i]<p); reward=int(action==targets[i]); model.update(x,action,reward)
                                actual_writer.writerow({"seed":seed,"condition":condition,"history":history,"phase":phase,"trial":i+1,
                                    "family_id":ds.family_ids[i],"left_count":int(ds.counts[i,0]),"right_count":int(ds.counts[i,1]),
                                    "action":action,"reward":reward,"p_right":p,"optimizer_factor":(action-p)*(reward-.5),
                                    "internal_step_l2":float(np.linalg.norm(model.weights-old_w)),"readout_step_l2":float(np.linalg.norm(model.head-old_h)),
                                    "kc_left_l2":float(np.linalg.norm(kc[0])),"kc_right_l2":float(np.linalg.norm(kc[1])),
                                    "mbon_left_l2":float(np.linalg.norm(mb[0])),"mbon_right_l2":float(np.linalg.norm(mb[1])),
                                    "decision_left_l2":float(np.linalg.norm(decision[0])),"decision_right_l2":float(np.linalg.norm(decision[1]))})
                            if (i+1)%cfg["checkpoint_every"]==0: checkpoint(model,path,history,phase,i+1,larger)
                        if not np.isfinite(model.head).all() or not np.isfinite(model.weights).all(): raise ValueError("Nonfinite learned parameters")
                    prefix_path=base/condition/SHARED_PREFIX; prefix_path.mkdir(parents=True)
                    train_phase(ancestor,prefix_path,SHARED_PREFIX,"phase1",True)
                    for history in HISTORIES:
                        gamma=GAMMAS.get(history)
                        before_model=pristine if gamma is None else ancestor
                        model=copy.deepcopy(pristine) if gamma is None else clone_scaled(ancestor,gamma)
                        path=base/condition/history; path.mkdir()
                        before=state_pack(before_model,probe,datasets); after=state_pack(model,probe,datasets)
                        np.savez_compressed(path/"boundary_before.npz",**before); np.savez_compressed(path/"boundary_after.npz",**after)
                        boundary={"seed":seed,"condition":condition,"history":history,"gamma":gamma,
                            "shared_prefix_path":None if gamma is None else f"seed_{seed}/{condition}/{SHARED_PREFIX}/phase1_{n:04d}.npz",
                            "shared_prefix_sha256":None if gamma is None else sha256(prefix_path/f"phase1_{n:04d}.npz"),
                            "head_change_l2":float(np.linalg.norm(after["readout_weights"]-before["readout_weights"])),
                            "internal_change_l2":float(np.linalg.norm(after["internal_weights"]-before["internal_weights"])),
                            "feature_scale_before":float(before["feature_scale"]),"feature_scale_after":float(after["feature_scale"]),
                            "head_updates_before":int(before["head_updates"]),"head_updates_after":int(after["head_updates"]),
                            "internal_updates_before":int(before["internal_updates"]),"internal_updates_after":int(after["internal_updates"])}
                        write_json(path/"boundary.json",boundary); boundaries.append(boundary)
                        resources.append({"seed":seed,"condition":condition,"history":history,"feature_scale":model.feature_scale,
                            "initial_feature_gap_rms":model.initial_feature_gap_rms,"readout_parameters":len(model.head),
                            "internal_parameters":len(model.weights) if condition in NETWORK_FAMILIES and CONDITIONS[condition][1] else 0,
                            "algorithm":"conditional_expected_action_reference" if condition=="expected_linear" else "stochastic_reward"})
                        train_phase(model,path,history,"phase2",False)
            print(f"completed seed {seed} ({cfg['seeds'].index(seed) + 1}/{len(cfg['seeds'])})", flush=True)
        with (output / "learning_curves.csv").open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=CURVE_FIELDS)
            writer.writeheader()
            writer.writerows(curves)
        write_json(output / "controls.json", controls)
        write_json(output / "model_resources.json", resources)
        write_json(output / "boundaries.json", boundaries)
        summary = summarize(cfg, curves)
        write_json(output / "summary.json", summary)
        metadata.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "metadata.json", metadata)
        write_json(output / "artifact_hashes.json", {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
        return summary
    except Exception as error:
        metadata.update(status="failed", failure=repr(error), failed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "metadata.json", metadata)
        if curves:
            with (output / "partial_learning_curves.csv").open("w", encoding="utf-8", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=CURVE_FIELDS)
                writer.writeheader()
                writer.writerows(curves)
        raise


def main():
    parser = argparse.ArgumentParser(description="Shared-prefix head attenuation diagnostic")
    parser.add_argument("command", choices=("register", "run", "analyze", "verify"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    for key in ("config", "registration", "out"):
        path = getattr(args, key)
        if path is not None:
            setattr(args, key, (path if path.is_absolute() else args.root / path).resolve())
    if args.command == "register":
        if args.config is None:
            parser.error("--config required")
        result = register(args.root, args.config, args.out)
        print(json.dumps({"registered_at_utc": result["registered_at_utc"], "config_sha256": result["config_sha256"]}))
    elif args.command == "run":
        if args.config is None or args.registration is None:
            parser.error("--config and --registration required")
        run(args.root, args.config, args.registration, args.out)
        from .head_attenuation_analysis import analyze
        analyze(args.out)
    elif args.command == "analyze":
        from .head_attenuation_analysis import analyze
        analyze(args.out)
    else:
        from .head_attenuation_audit import verify
        result = verify(args.out)
        print(json.dumps(result))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
