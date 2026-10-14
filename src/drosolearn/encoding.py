"""Frozen-policy input-order ablation with an exhaustive anchored code panel."""
from pathlib import Path
from itertools import combinations, permutations
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

FREE = (1, 3, 5, 6, 7, 8)
PAIRS = tuple(p for p in combinations(range(1, 9), 2) if p != (2, 4))
PRIMARY = np.array([i for i, p in enumerate(PAIRS) if all(n in FREE for n in p)], dtype=int)
ANCHORED = np.array([i for i, p in enumerate(PAIRS) if i not in set(PRIMARY)], dtype=int)
FAMILIES = ("fixed", "plastic", "rewired_fixed", "rewired_plastic", "ann_fixed", "ann_plastic")
FITS = ("quantity_present", "quantity_removed")
TRIAL_FIELDS = ("seed", "condition", "fit", "trial", "family_id", "left_count", "right_count", "action", "reward",
                "p_right", "optimizer_factor", "internal_step_l2", "readout_step_l2", "kc_left_l2", "kc_right_l2", "mbon_left_l2", "mbon_right_l2")
CURVE_FIELDS = ("seed", "condition", "fit", "trial", "split", "p_target", "greedy_accuracy", "internal_change_l2", "readout_change_l2", "probe_kc_change_l2", "probe_mbon_change_l2")

def anchored_panel():
    panel = np.tile(np.arange(1, 9), (720, 1))
    panel[:, np.asarray(FREE) - 1] = np.asarray(list(permutations(FREE)))
    return panel

def encoded_inputs(dataset, codes):
    codes = np.asarray(codes)
    if codes.shape != (8,): raise ValueError("Need eight artificial identity codes")
    quantity = codes[dataset.counts - 1, None] / 8.0
    return np.concatenate((quantity, 0.1 * (dataset.nuisance - 0.5)), axis=-1)

def validate_config(cfg):
    if cfg["scope"] not in ("preregistered_diagnostic", "integration_test"): raise ValueError("Unsupported scope")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]) or any(type(s) is not int or s < 0 for s in cfg["seeds"]): raise ValueError("Invalid seeds")
    if cfg["scope"] == "preregistered_diagnostic" and len(cfg["seeds"]) != 10: raise ValueError("Need ten registered seeds")
    if tuple(cfg["fits"]) != FITS or cfg["train_pair"] != [2,4]: raise ValueError("This diagnostic requires anchored 2-vs-4 and both fits")
    if not cfg["conditions"] or len(set(cfg["conditions"])) != len(cfg["conditions"]) or any(c not in FAMILIES for c in cfg["conditions"]): raise ValueError("Invalid model families")
    for key in ("trials", "checkpoint_every", "eval_trials", "probe_rows"):
        if type(cfg[key]) is not int or cfg[key] <= 0: raise ValueError("Invalid budget")
    if cfg["trials"] % 4 or cfg["eval_trials"] % 4 or cfg["trials"] % cfg["checkpoint_every"] or cfg["probe_rows"] > cfg["eval_trials"]: raise ValueError("Invalid crossing/checkpoint/probe budget")
    for key in ("head_lr", "internal_lr"):
        if type(cfg[key]) not in (int,float) or not np.isfinite(cfg[key]) or cfg[key] < 0: raise ValueError("Invalid learning rate")
    for value in cfg["criteria"].values():
        if type(value) not in (int,float) or not np.isfinite(value) or not 0 <= value <= 1: raise ValueError("Invalid criterion")
    if cfg["criteria"]["positive_seed_fraction"] <= 0: raise ValueError("Positive fraction required")

def study_hashes(root, cfg):
    root = Path(root)
    files = list((root / "src").rglob("*.py")) + list((root / cfg["graph"]).glob("*"))
    files += [root / "docs/encoding-protocol.md", root / "docs/encoding-background.md", root / "pyproject.toml",
              root / "requirements-lock.txt", root / "data/source-manifest.json", root / "data/LICENSE.md"]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files) if p.is_file()}

def register(root, config, destination):
    root, config, destination = map(Path, (root, config, destination))
    cfg = json.loads(config.read_text(encoding="utf-8")); validate_config(cfg); load_projection(root / cfg["graph"])
    record = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg, "config_sha256": sha256(config),
              "source_hashes": study_hashes(root,cfg), "git": git_state(root),
              "notice": "prospective private Git/local freeze; not accredited public preregistration",
              "previous_studies": ["v0.1.0-smoke", "study-reversal-v1"]}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x",encoding="utf-8") as stream:
        json.dump(record,stream,indent=2,ensure_ascii=False,allow_nan=False); stream.write("\n")
    return record

def make_splits(cfg, seed):
    base, n, ev = seed*100, cfg["trials"], cfg["eval_trials"]
    data = {"train": make_dataset(base+31,"train",n), "iid": make_dataset(base+32,"iid",ev),
            "ood": make_dataset(base+33,"ood",ev,domain="ood")}
    for i,pair in enumerate(PAIRS): data[f"pair_{pair[0]}_{pair[1]}"] = make_dataset(base+40+i,f"pair_{pair[0]}_{pair[1]}",ev,pair)
    assert_disjoint(list(data.values()))
    for ds in data.values():
        if analytical_accuracy(ds) != 1.0 or any(analytical_accuracy(ds,nuisance_index=i) != .5 for i in range(6)): raise ValueError("Cue/solvability gate failed")
    return data

def policy_metrics(model, ds, codes):
    p = model.probabilities(encoded_inputs(ds,codes)); target = ds.target(True)
    correct = np.where(np.abs(p-.5)<1e-12,.5,((p>.5).astype(int)==target).astype(float))
    return {"p_target": float(np.where(target==1,p,1-p).mean()), "greedy_accuracy": float(correct.mean())}

def candidate_cache(model, datasets, code_values):
    # MBON and head are linear here: collapse their product for frozen evaluation.
    # This changes neither parameters nor inputs, and is checked against full forward calls.
    gain = np.bincount(model.graph.pre,weights=model.weights*model.head[model.graph.post],minlength=model.graph.n_kc)
    output=[]
    for pair in PAIRS:
        ds=datasets[f"pair_{pair[0]}_{pair[1]}"]
        nuis=0.1*(ds.nuisance-.5)
        q=np.broadcast_to(np.asarray(code_values)/8.0, (*ds.counts.shape,len(code_values)))[...,None]
        x=np.concatenate((q,np.broadcast_to(nuis[...,None,:],(*ds.counts.shape,len(code_values),6))),axis=-1)
        output.append(model.kc(x)@gain)
    return np.asarray(output)

def panel_metrics(cache, datasets, panel):
    means=np.empty((len(panel),len(PAIRS))); greedy=np.empty_like(means)
    ordered_cases=np.empty((len(PAIRS),len(datasets[f"pair_{PAIRS[0][0]}_{PAIRS[0][1]}"].counts)))
    for j,pair in enumerate(PAIRS):
        ds=datasets[f"pair_{pair[0]}_{pair[1]}"]; n=len(ds.counts)
        indices=panel[:,ds.counts-1]-1
        cases=np.arange(n)[None,:]
        left=cache[j,cases,0,indices[:,:,0]]; right=cache[j,cases,1,indices[:,:,1]]
        p=1/(1+np.exp(-np.clip(right-left,-40,40)))
        target=ds.target(True)[None,:]
        means[:,j]=np.where(target==1,p,1-p).mean(axis=1)
        correct=np.where(np.abs(p-.5)<1e-12,.5,((p>.5).astype(int)==target).astype(float))
        greedy[:,j]=correct.mean(axis=1)
        ordered_cases[j]=p[0]
    return means,greedy,ordered_cases

def transfer_evaluate(model,datasets,fit,panel):
    levels=np.arange(1,9) if fit=="quantity_present" else np.array([0])
    cache=candidate_cache(model,datasets,levels)
    if fit=="quantity_present":
        means,greedy,cases=panel_metrics(cache,datasets,panel)
        if not np.allclose(means[:,PRIMARY].mean(axis=0),.5,rtol=0,atol=1e-12): raise ValueError("Full unseen permutation symmetry failed")
        return {"code_values":levels,"score_cache":cache,"map_p_target":means,"map_greedy_accuracy":greedy,"ordered_p_right":cases}
    means=[]; greedies=[]; cases=[]
    for j,pair in enumerate(PAIRS):
        ds=datasets[f"pair_{pair[0]}_{pair[1]}"]
        p=1/(1+np.exp(-np.clip(cache[j,:,1,0]-cache[j,:,0,0],-40,40)))
        target=ds.target(True)
        means.append(float(np.where(target==1,p,1-p).mean()))
        greedies.append(float(np.where(np.abs(p-.5)<1e-12,.5,((p>.5).astype(int)==target).astype(float)).mean())); cases.append(p)
    if not np.allclose(means,.5,rtol=0,atol=1e-12): raise ValueError("Quantity-removed symmetry failed")
    return {"code_values":levels,"score_cache":cache,"p_target":np.array(means),"greedy_accuracy":np.array(greedies),"p_right":np.array(cases)}

def summarize(cfg,curves,transfer):
    idx={(r["seed"],r["condition"],r["fit"],r["trial"],r["split"]):r for r in curves}
    rows=[]; contrasts=[]; verdicts=[]; crit=cfg["criteria"]; n=cfg["trials"]
    for seed in cfg["seeds"]:
        for condition in cfg["conditions"]:
            for fit in FITS:
                before=idx[seed,condition,fit,0,"iid"]["p_target"]; after=idx[seed,condition,fit,n,"iid"]["p_target"]
                acquired=after>=crit["acquisition_accuracy"] and after-before>=crit["acquisition_gain"]
                t=transfer[seed,condition,fit,"final"]
                row={"seed":seed,"condition":condition,"fit":fit,"before_accuracy":before,"after_accuracy":after,"gain":after-before,"acquired":acquired,
                     "ood_accuracy":idx[seed,condition,fit,n,"ood"]["p_target"],"head_updates":n,"internal_updates":n if CONDITIONS[condition][1] else 0}
                if fit=="quantity_present":
                    means=t["map_p_target"]
                    baseline=transfer[seed,condition,fit,"baseline"]["map_p_target"]
                    row.update(ordered_unseen=float(means[0,PRIMARY].mean()),permuted_unseen=float(means[:,PRIMARY].mean()),
                               ordered_unseen_before=float(baseline[0,PRIMARY].mean()),ordered_unseen_gain=float((means[0,PRIMARY]-baseline[0,PRIMARY]).mean()),
                               permuted_unseen_before=float(baseline[:,PRIMARY].mean()),
                               ordered_anchored=float(means[0,ANCHORED].mean()),permuted_anchored=float(means[:,ANCHORED].mean()),
                               ordered_all=float(means[0].mean()),permuted_all=float(means.mean()),
                               qualified_ordered_transfer=bool(acquired and means[0,PRIMARY].mean()>=crit["transfer_accuracy"]),
                               qualified_permuted_transfer=bool(acquired and means[:,PRIMARY].mean()>=crit["transfer_accuracy"]))
                    contrasts.append({"seed":seed,"condition":condition,"representation_gap":row["ordered_unseen"]-row["permuted_unseen"],"acquired":acquired})
                else: row.update(removed_unseen=float(t["p_target"][PRIMARY].mean()),removed_all=float(t["p_target"].mean()),
                                 removed_unseen_before=float(transfer[seed,condition,fit,"baseline"]["p_target"][PRIMARY].mean()))
                rows.append(row)
    for condition in cfg["conditions"]:
        selected=[r for r in contrasts if r["condition"]==condition]; gaps=[r["representation_gap"] for r in selected]; qualified=[r["representation_gap"] for r in selected if r["acquired"]]
        mean=float(np.mean(gaps)); positive=sum(g>0 for g in gaps)
        verdicts.append({"condition":condition,"n_seeds":len(gaps),"mean_gap":mean,"sd_gap":float(np.std(gaps,ddof=1)) if len(gaps)>1 else 0.0,
                         "positive_seed_count":positive,"criterion_met":mean>=crit["representation_gap"] and positive/len(gaps)>=crit["positive_seed_fraction"],
                         "qualified_seed_count":len(qualified),"qualified_mean_gap":float(np.mean(qualified)) if qualified else None})
    return {"status":"completed","scope":cfg["scope"],"outcomes":rows,"contrasts":contrasts,"family_verdicts":verdicts,
            "actual_training_trials":len(cfg["seeds"])*len(cfg["conditions"])*2*n,"permutations":720,
            "interpretation":"frozen learned function under different supplied scalar identities; no mathematical rule discovery or biological mechanism"}

def run(root,config,registration,output):
    root,config,registration,output=map(Path,(root,config,registration,output))
    cfg=json.loads(config.read_text(encoding="utf-8")); validate_config(cfg); reg=json.loads(registration.read_text(encoding="utf-8"))
    if sha256(config)!=reg["config_sha256"] or study_hashes(root,cfg)!=reg["source_hashes"]: raise ValueError("Source/config/protocol changed after registration")
    output.mkdir(parents=True,exist_ok=False); write_json(output/"registration.json",reg)
    original=load_projection(root/cfg["graph"]); panel=anchored_panel(); np.save(output/"permutations.npy",panel)
    concordance=np.array([[float(row[b-1]>row[a-1]) for a,b in PAIRS] for row in panel])
    code_gaps=np.array([[row[b-1]-row[a-1] for a,b in PAIRS] for row in panel])
    np.savez_compressed(output/"code_geometry.npz",pairs=np.array(PAIRS),quantity_ratios=np.array([a/b for a,b in PAIRS]),
                        quantity_differences=np.array([b-a for a,b in PAIRS]),code_gaps=code_gaps,code_order_agreement=concordance)
    write_json(output/"code_audit.json",{"anchors":{"2":2,"4":4},"fully_unseen_pairs":[list(PAIRS[i]) for i in PRIMARY],"anchored_pairs":[list(PAIRS[i]) for i in ANCHORED],
               "code_order_oracle_unseen":float(concordance[:,PRIMARY].mean()),"code_order_oracle_anchored":float(concordance[:,ANCHORED].mean()),"code_order_oracle_all":float(concordance.mean()),
               "identity_aware_inverse_oracle":1.0,"inverse_oracle_scope":"unique scalar mappings only; quantity_removed is noninvertible",
               "code_geometry_gap_units":"integer code units1..8; divide by8 for actual scalar input gap",
               "oracle_is_privileged":True,"full_unseen_policy_chance":"structural symmetry fixed before execution, not discovered learning failure"})
    def recorded_path(p):
        p=p.resolve(); return (p.relative_to(root.resolve()) if p.is_relative_to(root.resolve()) else p).as_posix()
    metadata={"started_at_utc":datetime.now(timezone.utc).isoformat(),"config":cfg,"git":git_state(root),"registration_sha256":sha256(registration),"graph":original.metadata,
              "python":platform.python_version(),"platform":platform.platform(),"packages":{p:importlib.metadata.version(p) for p in ("numpy","matplotlib")},
              "thread_environment":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
              "rng":{"actions":"seed+280000","encoder":"seed+70000","rewired":"seed+50000","ann":"seed+60000","data":"seed*100+31,+32,+33,+40+pair_index"},
              "training_reuse":"one quantity-present trajectory reused under all720 frozen input-code evaluations; a separate quantity-removed fit",
              "reproduce":f"python -m drosolearn.encoding run --config {recorded_path(config)} --registration {recorded_path(registration)} --out results/NEW_ENCODING"}
    write_json(output/"metadata.json",metadata); curves=[]; transfers={}; controls=[]
    try:
        for seed in cfg["seeds"]:
            datasets=make_splits(cfg,seed); seed_path=output/f"seed_{seed}"; seed_path.mkdir()
            uniforms=np.random.default_rng(seed+280000).random(cfg["trials"])
            np.savez_compressed(seed_path/"stimuli.npz",**{f"{name}_{key}":value for name,ds in datasets.items() for key,value in (("counts",ds.counts),("nuisance",ds.nuisance),("family_ids",np.array(ds.family_ids)))},action_uniforms=uniforms)
            graphs={"real":original,"rewired":rewire(original,seed+50000),"ann":random_ann(original,seed+60000)}
            for kind,g in graphs.items():
                np.savez_compressed(seed_path/f"{kind}_topology.npz",pre=g.pre,post=g.post,counts=g.weights); controls.append({"seed":seed,"kind":kind,"metadata":g.metadata})
            with (seed_path/"trials.csv").open("w",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=TRIAL_FIELDS); writer.writeheader()
                for condition in cfg["conditions"]:
                    kind,plastic,_,_=CONDITIONS[condition]; graph=graphs[kind]
                    for fit in FITS:
                        codes=np.arange(1,9) if fit=="quantity_present" else np.zeros(8,dtype=int)
                        model=Learner(graph,seed+70000,plastic,True,cfg["head_lr"],cfg["internal_lr"])
                        path=seed_path/condition/fit; path.mkdir(parents=True)
                        probe=encoded_inputs(datasets["iid"],codes)[:cfg["probe_rows"]]; initial=model.snapshot(probe)
                        def checkpoint(t):
                            snap=model.snapshot(probe)
                            np.savez_compressed(path/f"checkpoint_{t:04d}.npz",**snap,encoder=model.encoder,bias=model.bias,pre=graph.pre,post=graph.post)
                            for split in ("iid","ood"):
                                metrics=policy_metrics(model,datasets[split],codes)
                                curves.append({"seed":seed,"condition":condition,"fit":fit,"trial":t,"split":split,**metrics,
                                               "internal_change_l2":float(np.linalg.norm(model.weights-initial["internal_weights"])),"readout_change_l2":float(np.linalg.norm(model.head-initial["readout_weights"])),
                                               "probe_kc_change_l2":float(np.linalg.norm(snap["kc_activity"]-initial["kc_activity"])),"probe_mbon_change_l2":float(np.linalg.norm(snap["mbon_activity"]-initial["mbon_activity"]))})
                            if t in (0,cfg["trials"]):
                                stage="baseline" if t==0 else "final"; metrics=transfer_evaluate(model,datasets,fit,panel)
                                np.savez_compressed(path/f"transfer_{stage}.npz",**metrics); transfers[seed,condition,fit,stage]=metrics
                        checkpoint(0); ds=datasets["train"]; inputs=encoded_inputs(ds,codes); targets=ds.target(True)
                        # All anchored maps give exactly the same training inputs.
                        if fit=="quantity_present" and not all(np.array_equal(inputs,encoded_inputs(ds,p)) for p in panel): raise ValueError("Anchored train inputs differ")
                        for i,x in enumerate(inputs):
                            kc,mb=model.features(x); p=float(1/(1+np.exp(-np.clip((mb[1]-mb[0])@model.head,-40,40))))
                            action=int(uniforms[i]<p); reward=int(action==targets[i]); old_w,old_h=model.weights.copy(),model.head.copy(); model.update(x,action,reward)
                            writer.writerow({"seed":seed,"condition":condition,"fit":fit,"trial":i+1,"family_id":ds.family_ids[i],"left_count":int(ds.counts[i,0]),"right_count":int(ds.counts[i,1]),
                                             "action":action,"reward":reward,"p_right":p,"optimizer_factor":float((action-p)*(reward-.5)),"internal_step_l2":float(np.linalg.norm(model.weights-old_w)),"readout_step_l2":float(np.linalg.norm(model.head-old_h)),
                                             "kc_left_l2":float(np.linalg.norm(kc[0])),"kc_right_l2":float(np.linalg.norm(kc[1])),"mbon_left_l2":float(np.linalg.norm(mb[0])),"mbon_right_l2":float(np.linalg.norm(mb[1]))})
                            if (i+1)%cfg["checkpoint_every"]==0: checkpoint(i+1)
                        if not np.isfinite(model.head).all() or not np.isfinite(model.weights).all(): raise ValueError("Nonfinite learned parameters")
            print(f"completed seed {seed} ({cfg['seeds'].index(seed)+1}/{len(cfg['seeds'])})",flush=True)
        with (output/"learning_curves.csv").open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=CURVE_FIELDS); writer.writeheader(); writer.writerows(curves)
        write_json(output/"controls.json",controls); summary=summarize(cfg,curves,transfers); write_json(output/"summary.json",summary)
        metadata.update(status="completed",completed_at_utc=datetime.now(timezone.utc).isoformat()); write_json(output/"metadata.json",metadata)
        write_json(output/"artifact_hashes.json",{p.relative_to(output).as_posix():sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
        return summary
    except Exception as error:
        metadata.update(status="failed",failure=repr(error),failed_at_utc=datetime.now(timezone.utc).isoformat()); write_json(output/"metadata.json",metadata)
        if curves:
            from .experiment import csv_writer
            csv_writer(output/"partial_learning_curves.csv",curves)
        raise

def main():
    parser=argparse.ArgumentParser(description="Frozen-policy input-order diagnostic")
    parser.add_argument("command",choices=("register","run","analyze","verify")); parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--config",type=Path); parser.add_argument("--registration",type=Path); parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); args.root=args.root.resolve()
    for name in ("config","registration","out"):
        p=getattr(args,name)
        if p is not None:setattr(args,name,(p if p.is_absolute() else args.root/p).resolve())
    if args.command=="register":
        if args.config is None:parser.error("--config required")
        r=register(args.root,args.config,args.out); print(json.dumps({"registered_at_utc":r["registered_at_utc"],"config_sha256":r["config_sha256"]}))
    elif args.command=="run":
        if args.config is None or args.registration is None:parser.error("--config and --registration required")
        run(args.root,args.config,args.registration,args.out)
        from .encoding_analysis import analyze
        analyze(args.out)
    elif args.command=="analyze":
        from .encoding_analysis import analyze
        analyze(args.out)
    else:
        from .encoding_audit import verify
        r=verify(args.out); print(json.dumps(r))
        if not r["passed"]:raise SystemExit(1)

if __name__=="__main__":main()
